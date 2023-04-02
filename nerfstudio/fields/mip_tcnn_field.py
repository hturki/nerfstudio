# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
"""
import math
from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, List, Optional, Any

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfacc import ContractionType, contract
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.dataparsers.adop_dataparser import TRAIN_INDICES
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

CONSOLE = Console(width=120)

EXPLICIT_LEVEL = "explicit_level"
THIRD_PI = math.pi / 3.


class InterpolationModel(Enum):
    MLP_RGB = auto()
    MLP_DENSITY = auto()
    FEATURE = auto()


class MipTCNNField(Field):
    """TCNN implementation of the Instant-NGP field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        num_images: number of images, requried if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
        num_levels: number of levels of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
    """

    def __init__(
            self,
            aabb,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            spatial_distortion: SpatialDistortion = None,
            contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
            appearance_embedding_dim: int = 32,
            use_train_appearance_embedding: bool = True,
            use_average_appearance_embedding: bool = False,
            base_resolution: int = 16,
            max_resolution: int = 4096,
            features_per_level: int = 2,
            num_levels: int = 16,
            log2_hashmap_size: int = 19,
            interpolation_model: InterpolationModel = InterpolationModel.MLP_RGB,
            use_frustum_area: bool = False,
            same_color_mlp: bool = False,
            level_window: Optional[int] = None,
            training_level_jitter: float = 0,
            level_anneal: Optional[int] = None,
            level_anneal_cosine: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)

        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type
        self.spatial_distortion = spatial_distortion

        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)
            self.use_train_appearance_embedding = use_train_appearance_embedding
            self.use_average_appearance_embedding = use_average_appearance_embedding

        self.features_per_level = features_per_level
        self.use_frustum_area = use_frustum_area
        self.same_color_mlp = same_color_mlp
        self.interpolation_model = interpolation_model

        if level_window is not None:
            assert level_window > 0

        self.level_window = level_window

        assert 0 <= training_level_jitter <= 1
        self.training_level_jitter = training_level_jitter

        if interpolation_model != InterpolationModel.MLP_RGB:
            assert same_color_mlp

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        area_of_interest = (aabb[1] - aabb[0]).max()
        if self.contraction_type != ContractionType.AABB:
            area_of_interest *= 2  # Unbounded sphere maps the world from [-1, 1] to [-2, 2]

        self.areas = []

        per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        )

        if not self.same_color_mlp:
            mlp_heads = []

        if self.interpolation_model != InterpolationModel.FEATURE:
            mlp_bases = []
        else:
            assert level_window is not None

        for i in range(level_window if level_window is not None else 0, num_levels):
            self.areas.append((area_of_interest / (base_resolution * (per_level_scale ** i))).square())

            if self.interpolation_model != InterpolationModel.FEATURE:
                mlp_bases.append(tcnn.Network(
                    n_input_dims=features_per_level * (level_window if level_window is not None else (i + 1)),
                    n_output_dims=1 + self.geo_feat_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ))

            if not same_color_mlp:
                mlp_heads.append(tcnn.Network(
                    n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                    n_output_dims=3,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": hidden_dim_color,
                        "n_hidden_layers": num_layers_color - 1,
                    },
                ))

        if same_color_mlp:
            self.mlp_head = tcnn.Network(
                n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )
        else:
            self.mlp_heads = nn.ModuleList(mlp_heads)

        if self.interpolation_model == InterpolationModel.FEATURE:
            self.mlp_base = tcnn.Network(
                n_input_dims=features_per_level * level_window,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )
        else:
            self.mlp_bases = nn.ModuleList(mlp_bases)

        self.weight_anneal = None
        if level_anneal is not None:
            self.level_anneal = level_anneal
            self.level_anneal_cosine = level_anneal_cosine
            self.all_areas = self.areas
            self.areas = self.areas[:1]
            self.weight_anneal = 1

    def anneal_weights(self, step: int) -> None:
        if step > self.level_anneal:
            self.areas = self.all_areas
            self.weight_anneal = None
            return

        anneal_frac = step / self.level_anneal
        cur_level = anneal_frac * len(self.all_areas)

        if cur_level < 1:
            return

        if cur_level >= len(self.areas):
            self.areas = self.all_areas[:int(cur_level) + 1]
            CONSOLE.log('Step: {}, new level count: {}'.format(step, len(self.areas)))

        self.weight_anneal = cur_level % 1
        if self.level_anneal_cosine:
            self.weight_anneal = (1 - math.cos(self.weight_anneal * math.pi)) / 2

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions_flat = positions.view(-1, 3)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

        encoding = self.encoding(positions_flat)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]

            if self.level_window:
                level_features = encoding[:,
                                 level * self.features_per_level:(level + self.level_window) * self.features_per_level]
            else:
                level_features = encoding[:, :(level + 1) * self.features_per_level]

            if self.interpolation_model == InterpolationModel.FEATURE:
                h = self.mlp_base(level_features)
            else:
                h = self.mlp_bases[level](level_features)

            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = F.softplus(density_before_activation.to(positions) - 1).view(ray_samples.frustums.starts.shape)
            return density, base_mlp_out

        pixel_areas = get_pixel_areas(ray_samples.frustums.starts, ray_samples.frustums.ends,
                                      ray_samples.frustums.pixel_area, self.use_frustum_area)
        if self.training and self.training_level_jitter > 0:
            to_jitter = torch.rand_like(pixel_areas) <= self.training_level_jitter
            jitter_factor = torch.rand_like(pixel_areas[to_jitter])
            pixel_areas[to_jitter] = jitter_factor * self.areas[0] + (1 - jitter_factor) * pixel_areas[to_jitter]

        level_indices, level_weights = get_weights(pixel_areas, self.areas, self.weight_anneal)

        if self.interpolation_model == InterpolationModel.MLP_RGB:
            density = None
            level_embeddings = []
        elif self.interpolation_model == InterpolationModel.MLP_DENSITY:
            interpolated_h = None
        elif self.interpolation_model == InterpolationModel.FEATURE:
            interpolated_features = None
        else:
            raise Exception(self.interpolation_model)

        for level, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
            if cur_level_indices.shape[0] > 0:
                if self.level_window:
                    level_features = \
                        encoding[cur_level_indices,
                        level * self.features_per_level:(level + self.level_window) * self.features_per_level]
                else:
                    level_features = encoding[cur_level_indices, :(level + 1) * self.features_per_level]

                if self.interpolation_model == InterpolationModel.MLP_RGB:
                    level_h = self.mlp_bases[level](level_features)
                    density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                    level_embeddings.append(level_mlp_out)
                    level_density = F.softplus(density_before_activation.to(positions) - 1)
                    if density is None:
                        density = torch.zeros(positions_flat.shape[0], *level_density.shape[1:],
                                              dtype=level_density.dtype, device=level_density.device)
                    density[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_density
                elif self.interpolation_model == InterpolationModel.MLP_DENSITY:
                    level_h = self.mlp_bases[level](level_features)
                    if interpolated_h is None:
                        interpolated_h = torch.zeros(positions_flat.shape[0], *level_h.shape[1:],
                                                     dtype=level_h.dtype, device=level_h.device)
                    interpolated_h[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_h
                elif self.interpolation_model == InterpolationModel.FEATURE:
                    if interpolated_features is None:
                        interpolated_features = torch.zeros(positions_flat.shape[0], *level_features.shape[1:],
                                                            dtype=level_features.dtype, device=level_features.device)
                    interpolated_features[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_features
                else:
                    raise Exception(self.interpolation_model)
            else:
                if self.interpolation_model == InterpolationModel.MLP_RGB:
                    level_embeddings.append([])

        if self.interpolation_model == InterpolationModel.MLP_RGB:
            additional_info = (level_indices, level_weights, level_embeddings)
        elif self.interpolation_model == InterpolationModel.MLP_DENSITY:
            density_before_activation, mlp_out = torch.split(interpolated_h, [1, self.geo_feat_dim], dim=-1)
            density = F.softplus(density_before_activation.to(positions) - 1)
            additional_info = mlp_out
        elif self.interpolation_model == InterpolationModel.FEATURE:
            h = self.mlp_base(interpolated_features)
            density_before_activation, mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = F.softplus(density_before_activation.to(positions) - 1)
            additional_info = mlp_out
        else:
            raise Exception(self.interpolation_model)

        if self.training:
            level_counts = defaultdict(int)
            for i, cur_level_indices in enumerate(level_indices):
                level_counts[i] = cur_level_indices.shape[0] / positions_flat.shape[0]

            return density.view(*ray_samples.frustums.shape, -1), (additional_info, level_counts, pixel_areas)
        else:
            levels = torch.zeros_like(ray_samples.frustums.starts)

            for i, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
                levels.view(-1, 1)[cur_level_indices] += i * cur_level_weights.unsqueeze(-1)

            return density.view(*ray_samples.frustums.shape, -1), (additional_info, levels)

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[Any] = None):
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if self.appearance_embedding_dim > 0:
            if ray_samples.metadata is not None and TRAIN_INDICES in ray_samples.metadata:
                embedded_appearance = self.embedding_appearance(ray_samples.metadata[TRAIN_INDICES].squeeze())
            elif self.training:
                embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

            embedded_appearance = embedded_appearance.view(-1, self.appearance_embedding_dim)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            h = torch.cat([d, density_embedding] + ([embedded_appearance] if self.appearance_embedding_dim > 0 else []),
                          dim=-1)
            mlp_head = self.mlp_head if self.same_color_mlp else self.mlp_heads[level]
            return {FieldHeadNames.RGB: mlp_head(h).view(directions.shape).to(directions)}

        outputs = {}

        if self.training:
            density_embedding, level_counts, pixel_areas = density_embedding
            outputs[FieldHeadNames.LEVEL_COUNTS] = level_counts
            outputs[FieldHeadNames.PIXEL_AREAS] = pixel_areas.unsqueeze(-1)
        else:
            density_embedding, levels = density_embedding
            outputs[FieldHeadNames.LEVELS] = levels

        if self.interpolation_model != InterpolationModel.MLP_RGB:
            h = torch.cat([d, density_embedding] + ([embedded_appearance] if self.appearance_embedding_dim > 0 else []),
                          dim=-1)
            rgbs = self.mlp_head(h).view(directions.shape).to(directions)
            outputs[FieldHeadNames.RGB] = rgbs
            return outputs

        level_indices, level_weights, level_embeddings = density_embedding

        rgbs = None
        for i, (cur_level_indices, cur_level_weights, cur_level_embeddings) in enumerate(zip(level_indices,
                                                                                             level_weights,
                                                                                             level_embeddings)):
            if cur_level_indices.shape[0] > 0:
                h = torch.cat([d[cur_level_indices], cur_level_embeddings] +
                              ([embedded_appearance[cur_level_indices]]
                               if self.appearance_embedding_dim > 0 else []), dim=-1)

                level_rgbs = self.mlp_head(h) if self.same_color_mlp else self.mlp_heads[i](h)
                if rgbs is None:
                    rgbs = torch.zeros(*directions.shape, dtype=level_rgbs.dtype,
                                       device=level_rgbs.device)
                rgbs.view(-1, 3)[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_rgbs

        outputs[FieldHeadNames.RGB] = rgbs
        return outputs

    def get_opacity(self, positions: TensorType["bs":..., 3], step_size) -> TensorType["bs":..., 1]:
        """Returns the opacity for a position. Used primarily by the occupancy grid.

        Args:
            positions: the positions to evaluate the opacity at.
            step_size: the step size to use for the opacity evaluation.
        """
        density = self.density_fn(positions, None)

        opacity = density * step_size
        return opacity

    def density_fn(self, positions: TensorType["bs":..., 3], times: TensorType["bs":..., 1]) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        # Need to figure out a better way to describe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times
        )
        ray_samples.metadata = {EXPLICIT_LEVEL: len(self.areas) - (1 if self.weight_anneal is not None else 2)}
        density, _ = self.get_density(ray_samples)
        return density


@torch.jit.script
def get_weights(pixel_areas: torch.Tensor, areas: list[float], weight_anneal: Optional[float] = None) -> Tuple[
    List[torch.Tensor], List[torch.Tensor]]:
    sorted_pixel_areas, ordering = pixel_areas.sort(descending=True)
    level_indices = []
    level_weights = []

    last_weights = torch.empty(0)

    final_level_annealed_first_weights = torch.empty(0)
    mid = 0
    end = 0
    num_levels = len(areas)
    for i in range(num_levels):
        if i == 0:
            mid = (sorted_pixel_areas > areas[i]).sum()
            cur_level_indices = [ordering[:mid]]
            cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=pixel_areas.dtype)]
        else:
            start = mid
            mid = end
            cur_level_indices = [ordering[start:mid]]
            if weight_anneal is not None and i == num_levels - 1:
                cur_level_weights = [final_level_annealed_first_weights]
            else:
                cur_level_weights = [1 - last_weights]

        if i < num_levels - 1:
            end = mid + (sorted_pixel_areas[mid:] > areas[i + 1]).sum()
            cur_level_indices.append(ordering[mid:end])
            last_weights = (sorted_pixel_areas[mid:end] - areas[i + 1]) / (areas[i] - areas[i + 1])

            if weight_anneal is not None and i == num_levels - 2:
                final_level_annealed_first_weights = (1 - last_weights) * weight_anneal
                last_weights = last_weights / (last_weights + final_level_annealed_first_weights)

            cur_level_weights.append(last_weights)

            if weight_anneal is not None and i == num_levels - 2:
                cur_level_indices.append(ordering[end:])
                cur_level_weights.append(
                    torch.full_like(cur_level_indices[-1], 1 - weight_anneal, dtype=pixel_areas.dtype))
        else:
            cur_level_indices.append(ordering[mid:])

            if weight_anneal is not None:
                cur_level_weights.append(torch.full_like(cur_level_indices[-1], weight_anneal, dtype=pixel_areas.dtype))
            else:
                cur_level_weights.append(torch.ones_like(cur_level_indices[-1], dtype=pixel_areas.dtype))

        level_indices.append(torch.cat(cur_level_indices))
        level_weights.append(torch.cat(cur_level_weights))

    return level_indices, level_weights


@torch.jit.script
def get_pixel_areas(starts: torch.Tensor, ends: torch.Tensor, pixel_area: torch.Tensor, use_frustum_area: bool,
                    third_pi: float = THIRD_PI) -> torch.Tensor:
    if use_frustum_area:
        cone_radius = torch.sqrt(pixel_area) / 1.7724538509055159
        start_cone_radius = starts * cone_radius
        end_cone_radius = ends * cone_radius
        pixel_areas = (third_pi * (ends - starts) * (
                start_cone_radius.square() + end_cone_radius.square() + start_cone_radius * end_cone_radius)).view(-1)
    else:
        pixel_areas = (pixel_area * (((starts + ends) / 2) ** 2)).view(-1)
    return pixel_areas


def interpolation_model(model: str) -> InterpolationModel:
    if model.casefold() == 'mlp_rgb':
        return InterpolationModel.MLP_RGB
    if model.casefold() == 'mlp_density':
        return InterpolationModel.MLP_DENSITY
    if model.casefold() == 'feature':
        return InterpolationModel.FEATURE
    else:
        raise Exception(model)
