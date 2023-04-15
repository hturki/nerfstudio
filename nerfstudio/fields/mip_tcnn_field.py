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

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfacc import ContractionType, contract
from nerfstudio.cameras import camera_utils

from nerfstudio.cameras.cameras import Cameras

from nerfstudio.utils.math import expected_sin

from nerfstudio.field_components.encodings import NeRFEncoding
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

P = torch.FloatTensor([[0.8506508, 0, 0.5257311],
                       [0.809017, 0.5, 0.309017],
                       [0.5257311, 0.8506508, 0],
                       [1, 0, 0],
                       [0.809017, 0.5, -0.309017],
                       [0.8506508, 0, -0.5257311],
                       [0.309017, 0.809017, -0.5],
                       [0, 0.5257311, -0.8506508],
                       [0.5, 0.309017, -0.809017],
                       [0, 1, 0],
                       [-0.5257311, 0.8506508, 0],
                       [-0.309017, 0.809017, -0.5],
                       [0, 0.5257311, 0.8506508],
                       [-0.309017, 0.809017, 0.5],
                       [0.309017, 0.809017, 0.5],
                       [0.5, 0.309017, 0.809017],
                       [0.5, -0.309017, 0.809017],
                       [0, 0, 1],
                       [-0.5, 0.309017, 0.809017],
                       [-0.809017, 0.5, 0.309017],
                       [-0.809017, 0.5, -0.309017]]).T


class InterpolationModel(Enum):
    MLP_RGB = auto()
    MLP_DENSITY = auto()
    SINGLE = auto()


class LevelFeatures(Enum):
    TRUNCATE = auto()
    ANNEAL = auto()
    FULL = auto()


class AuxiliaryInfo(Enum):
    NONE = auto()
    SCALE = auto()
    ALL_SCALES = auto()
    COV = auto()
    IPE = auto()
    IPE_GRID = auto()


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
            contraction_type: ContractionType = ContractionType.AABB,
            appearance_embedding_dim: int = 32,
            use_train_appearance_embedding: bool = True,
            use_average_appearance_embedding: bool = False,
            base_resolution: int = 16,
            max_resolution: int = 4096,
            features_per_level: int = 2,
            num_levels: int = 16,
            log2_hashmap_size: int = 19,
            interpolation_model: InterpolationModel = InterpolationModel.MLP_RGB,
            level_features: LevelFeatures = LevelFeatures.TRUNCATE,
            auxiliary_info: AuxiliaryInfo = AuxiliaryInfo.NONE,
            interpolate_levels: bool = True,
            training_level_jitter: float = 0,
            level_anneal: Optional[int] = None,
            level_anneal_cosine: bool = True,
            train_lod_bias: bool = False,
            finest_occ_grid: bool = True,
            freq_dim: int = 4,
            freq_resolution: int = 8192,
            do_residual: bool = False,
            cameras: Cameras = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.cameras = cameras

        assert level_anneal is None
        assert training_level_jitter == 0

        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type
        self.spatial_distortion = spatial_distortion

        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)
            self.use_train_appearance_embedding = use_train_appearance_embedding
            self.use_average_appearance_embedding = use_average_appearance_embedding

        self.features_per_level = features_per_level
        self.interpolation_model = interpolation_model
        self.level_features = level_features
        self.auxiliary_info = auxiliary_info
        self.interpolate_levels = interpolate_levels
        self.finest_occ_grid = finest_occ_grid
        self.do_residual = do_residual

        if train_lod_bias:
            self.register_parameter('lod_bias', nn.Parameter(torch.zeros(1), requires_grad=True))
        else:
            self.register_buffer('lod_bias', torch.zeros(1), persistent=False)

        assert 0 <= training_level_jitter <= 1
        self.training_level_jitter = training_level_jitter

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))
        self.log_per_level_scale = math.log(per_level_scale)

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

        self.grid_offsets = [0]
        total_count = 0
        for i in range(num_levels):
            grid_scale = np.exp2(i * math.log2(per_level_scale)) * base_resolution - 1
            resolution = int(math.ceil(grid_scale) + 1)

            params_in_level = resolution ** 3
            params_in_level = int(math.ceil(params_in_level / 8)) * 8
            params_in_level = min(params_in_level, 1 << log2_hashmap_size)
            total_count += params_in_level
            self.grid_offsets.append(total_count * features_per_level)

        assert total_count * features_per_level == self.encoding.params.shape[0]

        area_of_interest = (aabb[1] - aabb[0]).max()
        if self.spatial_distortion is not None:
            area_of_interest *= 2  # Unbounded sphere maps the world from [-1, 1] to [-2, 2]

        self.base_log = math.log(area_of_interest, per_level_scale) - math.log(base_resolution, per_level_scale)
        self.level_resolutions = []
        for i in range(num_levels):
            self.level_resolutions.append(area_of_interest / (base_resolution * (per_level_scale ** i)))
        self.register_buffer('scales', torch.FloatTensor(self.level_resolutions), persistent=False)

        if auxiliary_info == AuxiliaryInfo.NONE:
            auxiliary_dims = 0
        elif auxiliary_info == AuxiliaryInfo.SCALE:
            auxiliary_dims = 1
        elif auxiliary_info == AuxiliaryInfo.ALL_SCALES:
            auxiliary_dims = len(self.level_resolutions)
        elif auxiliary_info == AuxiliaryInfo.COV:
            auxiliary_dims = 9
        elif auxiliary_info == AuxiliaryInfo.IPE:
            if self.spatial_distortion is not None:
                self.register_buffer("P", P, persistent=False)
                auxiliary_dims = 2 * P.shape[1]
            else:
                self.pos_enc = NeRFEncoding(in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0,
                                            include_input=False)
                auxiliary_dims = self.pos_enc.num_frequencies * 3 * 2
        elif auxiliary_info == AuxiliaryInfo.IPE_GRID:
            if self.spatial_distortion is not None:
                self.register_buffer("P", P, persistent=False)
                freq_tables = 2 * P.shape[1] // 3
            else:
                self.pos_enc = NeRFEncoding(in_dim=3, num_frequencies=16, min_freq_exp=0.0,
                                            max_freq_exp=16, include_input=False)
                freq_tables = 16 * 2

            freq_encodings = []
            for _ in range(freq_tables):
                freq_encodings.append(tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 1,
                        "n_features_per_level": freq_dim,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": freq_resolution,
                        "max_resolution": freq_resolution,
                        "per_level_scale": 1,
                    }
                ))
            self.freq_encodings = nn.ModuleList(freq_encodings)
            auxiliary_dims = freq_tables * freq_dim
        else:
            raise Exception(auxiliary_info)

        if interpolation_model != InterpolationModel.SINGLE:
            mlp_bases = []

            for i in range(num_levels):
                if level_features == LevelFeatures.TRUNCATE:
                    encoding_dims = (i + 1) * features_per_level
                else:
                    encoding_dims = num_levels * features_per_level

                mlp_bases.append(tcnn.Network(
                    n_input_dims=encoding_dims + auxiliary_dims,
                    n_output_dims=1 + self.geo_feat_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": hidden_dim,
                        "n_hidden_layers": num_layers - 1,
                    },
                ))
            self.mlp_bases = nn.ModuleList(mlp_bases)
        else:
            self.mlp_base = tcnn.Network(
                n_input_dims=num_levels * features_per_level + auxiliary_dims,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )

        if interpolation_model == InterpolationModel.MLP_RGB:
            mlp_heads = []
            for i in range(num_levels):
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
            self.mlp_heads = nn.ModuleList(mlp_heads)
        else:
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

        self.weight_anneal = None
        if level_anneal is not None:
            self.level_anneal = level_anneal
            self.level_anneal_cosine = level_anneal_cosine
            self.all_resolutions = self.level_resolutions
            self.level_resolutions = self.level_resolutions[:1]
            self.weight_anneal = 1

    def anneal_weights(self, step: int) -> None:
        if step > self.level_anneal:
            self.level_resolutions = self.all_resolutions
            self.weight_anneal = None
            return

        anneal_frac = step / self.level_anneal
        cur_level = anneal_frac * len(self.all_areas)

        if cur_level < 1:
            return

        if cur_level >= len(self.areas):
            self.level_resolutions = self.all_resolutions[:int(cur_level) + 1]
            CONSOLE.log('Step: {}, new level count: {}'.format(step, len(self.areas)))

        self.weight_anneal = cur_level % 1
        if self.level_anneal_cosine:
            self.weight_anneal = (1 - math.cos(self.weight_anneal * math.pi)) / 2

    def get_density(self, ray_samples: RaySamples):
        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            raise Exception()

        # if self.training and self.training_level_jitter > 0:
        #     to_jitter = torch.rand_like(pixel_areas) <= self.training_level_jitter
        #     jitter_factor = torch.rand_like(pixel_areas[to_jitter])
        #     pixel_areas[to_jitter] = jitter_factor * self.areas[0] + (1 - jitter_factor) * pixel_areas[to_jitter]

        positions = ray_samples.frustums.get_positions()

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions_flat = positions.view(-1, 3)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

        # Assuming pixels are square
        pixel_widths = (ray_samples.frustums.pixel_area * (
                ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2) ** 2)).sqrt().view(-1)

        encoding = self.encoding(positions_flat)
        if self.level_features == LevelFeatures.ANNEAL:
            assert self.weight_anneal is None
            scale_factors = (pixel_widths.unsqueeze(-1) / self.scales).clamp_min(1).unsqueeze(-1)
            scale_factors = scale_factors.repeat(1, 1, self.features_per_level) \
                .view(-1, len(self.scales) * self.features_per_level)
            encoding = encoding / scale_factors

        pixel_levels = self.lod_bias + self.base_log - torch.log(pixel_widths) / self.log_per_level_scale

        auxiliary_info = self._get_auxiliary_info(ray_samples, pixel_widths)
        if self.interpolation_model == InterpolationModel.SINGLE:
            if self.level_features == LevelFeatures.TRUNCATE:
                for i, resolution in enumerate(self.level_resolutions):
                    encoding[pixel_widths > resolution,
                    i * self.features_per_level:(i + 1) * self.features_per_level] = 0

            if self.auxiliary_info == AuxiliaryInfo.SCALE:
                auxiliary_info = (pixel_levels / (len(self.scales) - 1) - 0.5).unsqueeze(-1)

            h = self.mlp_base(torch.cat([encoding, auxiliary_info], -1) if auxiliary_info is not None else encoding)
            density_before_activation, additional_info = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = F.softplus(density_before_activation.to(positions) - 1).view(ray_samples.frustums.starts.shape)
            if self.training:
                return density, additional_info
            else:
                return density, (additional_info, pixel_levels)

        if self.do_residual:
            level_indices, level_weights = get_weights_residual(pixel_levels, len(self.level_resolutions),
                                                                self.interpolate_levels)
        elif self.interpolate_levels:
            level_indices, level_weights = get_weights(pixel_levels, len(self.level_resolutions))
        else:
            level_indices = get_levels(pixel_levels, len(self.level_resolutions))
            level_weights = []
            for li in level_indices:
                level_weights.append(torch.ones_like(li, dtype=pixel_levels.dtype))

        if self.interpolation_model == InterpolationModel.MLP_RGB:
            density = None
            level_embeddings = []
        else:
            interpolated_h = None

        for level, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
            if cur_level_indices.shape[0] > 0:
                level_features = encoding[cur_level_indices]
                if self.level_features == LevelFeatures.TRUNCATE:
                    level_features = level_features[:, :(level + 1) * self.features_per_level]

                if self.auxiliary_info == AuxiliaryInfo.SCALE:
                    level_auxiliary_info = (pixel_levels[cur_level_indices] - level).unsqueeze(-1)
                else:
                    level_auxiliary_info = auxiliary_info[cur_level_indices] if auxiliary_info is not None else None

                base_input = torch.cat([level_features, level_auxiliary_info],
                                       -1) if level_auxiliary_info is not None else level_features

                level_h = self.mlp_bases[level](base_input)

                if self.interpolation_model == InterpolationModel.MLP_RGB:
                    density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                    level_embeddings.append(level_mlp_out)
                    level_density = F.softplus(density_before_activation.to(positions) - 1)
                    if density is None:
                        density = torch.zeros(positions_flat.shape[0], *level_density.shape[1:],
                                              dtype=level_density.dtype, device=level_density.device)
                    density[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_density
                elif self.interpolation_model == InterpolationModel.MLP_DENSITY:
                    if interpolated_h is None:
                        interpolated_h = torch.zeros(positions_flat.shape[0], *level_h.shape[1:],
                                                     dtype=level_h.dtype, device=level_h.device)
                    interpolated_h[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_h
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
        else:
            raise Exception(self.interpolation_model)

        if self.training:
            level_counts = defaultdict(int)
            for i, cur_level_indices in enumerate(level_indices):
                level_counts[i] = cur_level_indices.shape[0] / positions_flat.shape[0]

            return density.view(*ray_samples.frustums.shape, -1), (additional_info, level_counts)
        else:
            return density.view(*ray_samples.frustums.shape, -1), (additional_info, pixel_levels)

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
            return {FieldHeadNames.RGB: self.mlp_heads[level](h).view(directions.shape).to(directions)}

        outputs = {}

        if self.training:
            if self.interpolation_model != InterpolationModel.SINGLE:
                density_embedding, level_counts = density_embedding
                outputs[FieldHeadNames.LEVEL_COUNTS] = level_counts
        else:
            density_embedding, levels = density_embedding
            outputs[FieldHeadNames.LEVELS] = levels.view(ray_samples.frustums.starts.shape)

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

                level_rgbs = self.mlp_heads[i](h)
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
        density = self.density_fn(positions, None, step_size=step_size)

        opacity = density * step_size
        return opacity

    def density_fn(self, positions: TensorType["bs":..., 3], times: TensorType["bs":..., 1],
                   step_size: int = None, origins: Optional[torch.Tensor] = None,
                   directions: Optional[torch.Tensor] = None, starts: Optional[torch.Tensor] = None,
                   ends: Optional[torch.Tensor] = None, pixel_area: Optional[torch.Tensor] = None) \
            -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        if origins is None:
            camera_ids = torch.randint(0, len(self.cameras), (positions.shape[0],), device=positions.device)
            cameras = self.cameras.to(camera_ids.device)[camera_ids]
            origins = cameras.camera_to_worlds[:, :, 3]
            directions = positions - origins
            directions, _ = camera_utils.normalize_with_norm(directions, -1)
            coords = torch.cat(
                [torch.rand_like(origins[..., :1]) * cameras.height, torch.rand_like(origins[..., :1]) * cameras.width],
                -1).floor().long()

            pixel_area = cameras.generate_rays(torch.arange(len(cameras)).unsqueeze(-1), coords=coords).pixel_area
            starts = (origins - positions).norm(dim=-1, keepdim=True) - step_size / 2
            ends = starts + step_size

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
            ),
            times=times
        )

        if self.finest_occ_grid:
            ray_samples.metadata = {EXPLICIT_LEVEL: len(self.areas) - (1 if self.weight_anneal is not None else 2)}

        density, _ = self.get_density(ray_samples)
        return density

    def _get_auxiliary_info(self, ray_samples: RaySamples, pixel_widths: torch.Tensor) -> Optional[torch.Tensor]:
        #  Scale is calculated per level
        if self.auxiliary_info == AuxiliaryInfo.NONE or self.auxiliary_info == AuxiliaryInfo.SCALE:
            return None
        elif self.auxiliary_info == AuxiliaryInfo.ALL_SCALES:
            # Scale to [0, 1]
            return (torch.sigmoid(self.scales / pixel_widths.unsqueeze(-1)) - 0.5) * 2
        elif self.auxiliary_info in {AuxiliaryInfo.COV, AuxiliaryInfo.IPE, AuxiliaryInfo.IPE_GRID}:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            if self.auxiliary_info == AuxiliaryInfo.COV:
                return gaussian_samples.cov.view(-1, 9)
            else:
                if self.spatial_distortion is not None:
                    lifted_mean = torch.matmul(gaussian_samples.mean, self.P)
                    lifted_diag = torch.sum(self.P * torch.matmul(gaussian_samples.cov, self.P), dim=-2)
                    encoded_xyz = expected_sin(
                        torch.cat([lifted_mean, lifted_mean + torch.pi / 2.0], dim=-1),
                        torch.cat(2 * [lifted_diag], dim=-1)
                    )
                else:
                    encoded_xyz = self.pos_enc(gaussian_samples.mean, covs=gaussian_samples.cov)

                encoded_xyz = encoded_xyz.view(-1, encoded_xyz.shape[-1])
                if self.auxiliary_info == AuxiliaryInfo.IPE:
                    return encoded_xyz
                elif self.auxiliary_info == AuxiliaryInfo.IPE_GRID:
                    encoded_xyz = encoded_xyz / 2 + 0.5

                    auxiliary_info = None
                    for i, freq_encoding in enumerate(self.freq_encodings):
                        level_features = freq_encoding(encoded_xyz[:, 3 * i:3 * (i + 1)])
                        if auxiliary_info is None:
                            auxiliary_info = torch.empty(*level_features.shape[:-1],
                                                         self.features_per_level * len(self.freq_encodings) // 2,
                                                         dtype=level_features.dtype, device=level_features.device)
                        auxiliary_info[:, i * (self.features_per_level // 2):(i + 1) * (self.features_per_level // 2)] \
                            = level_features

                    return auxiliary_info
                else:
                    raise Exception(self.auxiliary_info)
        else:
            raise Exception(self.auxiliary_info)


@torch.jit.script
def get_levels(pixel_levels: torch.Tensor, num_levels: int) -> List[torch.Tensor]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices = []

    start = 0
    for i in range(num_levels - 1):
        end = start + (sorted_pixel_levels[start:] < i).sum()
        level_indices.append(ordering[start:end])
        start = end

    level_indices.append(ordering[start:])

    return level_indices


# @torch.jit.script
def get_weights(pixel_levels: torch.Tensor, num_levels: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices = []
    level_weights = []

    mid = 0
    end = 0
    for i in range(num_levels):
        if i == 0:
            mid = (sorted_pixel_levels < i).sum()
            cur_level_indices = [ordering[:mid]]
            cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=pixel_levels.dtype)]
        else:
            start = mid
            mid = end
            cur_level_indices = [ordering[start:mid]]
            cur_level_weights = [sorted_pixel_levels[start:mid] - (i - 1)]

        if i < num_levels - 1:
            end = mid + (sorted_pixel_levels[mid:] < i + 1).sum()
            cur_level_indices.append(ordering[mid:end])
            cur_level_weights.append(1 - (sorted_pixel_levels[mid:end] - i))
        else:
            cur_level_indices.append(ordering[mid:])
            cur_level_weights.append(torch.ones_like(cur_level_indices[-1], dtype=pixel_levels.dtype))

        level_indices.append(torch.cat(cur_level_indices))
        level_weights.append(torch.cat(cur_level_weights))

    return level_indices, level_weights


@torch.jit.script
def get_weights_residual(pixel_levels: torch.Tensor, num_levels: int, interpolate: bool) -> Tuple[
    List[torch.Tensor], List[torch.Tensor]]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=True)
    level_indices = []
    level_weights = []

    end = 0
    for i in range(num_levels):
        if i == 0:
            cur_level_indices = [ordering]
        else:
            end = (sorted_pixel_levels > i).sum()
            cur_level_indices = [ordering[:end]]

        cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=pixel_levels.dtype)]

        if i > 0 and interpolate:
            next_level_end = end + (sorted_pixel_levels[end:] > i - 1).sum()
            cur_level_indices.append(ordering[end:next_level_end])
            cur_level_weights.append(sorted_pixel_levels[end:next_level_end] - (i - 1))

        level_indices.append(torch.cat(cur_level_indices))
        level_weights.append(torch.cat(cur_level_weights))

    return level_indices, level_weights


def interpolation_model(model: str) -> InterpolationModel:
    if model.casefold() == 'mlp_rgb':
        return InterpolationModel.MLP_RGB
    if model.casefold() == 'mlp_density':
        return InterpolationModel.MLP_DENSITY
    if model.casefold() == 'single':
        return InterpolationModel.SINGLE
    else:
        raise Exception(model)


def level_features(level_features: str) -> LevelFeatures:
    if level_features.casefold() == 'truncate':
        return LevelFeatures.TRUNCATE
    if level_features.casefold() == 'anneal':
        return LevelFeatures.ANNEAL
    if level_features.casefold() == 'full':
        return LevelFeatures.FULL
    else:
        raise Exception(level_features)


def auxiliary_info(info: str) -> AuxiliaryInfo:
    if info.casefold() == 'none':
        return AuxiliaryInfo.NONE
    if info.casefold() == 'scale':
        return AuxiliaryInfo.SCALE
    if info.casefold() == 'all_scales':
        return AuxiliaryInfo.ALL_SCALES
    if info.casefold() == 'cov':
        return AuxiliaryInfo.COV
    if info.casefold() == 'ipe':
        return AuxiliaryInfo.IPE
    if info.casefold() == 'ipe_grid':
        return AuxiliaryInfo.IPE_GRID
    else:
        raise Exception(info)
