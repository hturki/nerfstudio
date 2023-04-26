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
from nerfstudio.field_components.activations import trunc_exp

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
        num_images: number of images, required if use_appearance_embedding is True
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
            num_scales: Optional[int] = None,
            scale_factor: Optional[float] = None,
            separate_encoding: bool = True,
            separate_res: bool = False,
            background_level: bool = False,
            interpolate_levels: bool = True,
            training_level_jitter: float = 0,
            lod_bias: float = 0,
            train_lod_bias: bool = False,
            finest_occ_grid: bool = True,
            freq_dim: int = 4,
            freq_resolution: int = 8192,
            do_residual: bool = False,
            cameras: Cameras = None,
            debug: bool = False
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.cameras = cameras
        self.debug = debug

        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type
        self.spatial_distortion = spatial_distortion

        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)
            self.use_train_appearance_embedding = use_train_appearance_embedding
            self.use_average_appearance_embedding = use_average_appearance_embedding

        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        self.interpolation_model = interpolation_model
        self.level_features = level_features
        self.auxiliary_info = auxiliary_info
        self.interpolate_levels = interpolate_levels
        self.finest_occ_grid = finest_occ_grid
        self.do_residual = do_residual

        if train_lod_bias:
            self.register_parameter('lod_bias', nn.Parameter(torch.FloatTensor([lod_bias]), requires_grad=True))
        else:
            self.register_buffer('lod_bias', torch.FloatTensor([lod_bias]), persistent=False)

        assert 0 <= training_level_jitter <= 1
        self.training_level_jitter = training_level_jitter

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        area_of_interest = (aabb[1] - aabb[0]).max()
        if self.spatial_distortion is not None:
            area_of_interest *= 2  # Unbounded sphere maps the world from [-1, 1] to [-2, 2]

        per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))
        if scale_factor is None:
            scale_factor = per_level_scale

        self.log_scale_factor = math.log(scale_factor)

        if num_scales is None:
            num_scales = num_levels

        self.num_scales = num_scales

        # Get base log of lowest mip level
        self.base_log = math.log(area_of_interest, scale_factor) - (
                math.log(max_resolution, scale_factor) - (num_scales - 1))

        feature_levels = []
        for cur_level in range(num_levels):
            feature_levels.append(self.base_log - (math.log(area_of_interest, scale_factor) - math.log(
                base_resolution * (per_level_scale ** cur_level), scale_factor)))

        CONSOLE.log("Feature levels: {}".format(feature_levels))
        self.register_buffer('feature_levels', torch.FloatTensor(feature_levels), persistent=False)

        self.background_level = background_level
        self.separate_encoding = separate_encoding
        if separate_encoding:
            encodings = []
            self.table_offsets = []

            for scale in range(num_scales):
                if separate_res:
                    assert level_features == LevelFeatures.FULL, level_features
                    cur_max_res = max_resolution / (scale_factor ** (num_scales - 1 - scale))
                    cur_level_scale = math.exp(math.log(cur_max_res / base_resolution) / (num_levels - 1))
                else:
                    cur_level_scale = per_level_scale

                if level_features == LevelFeatures.TRUNCATE:
                    cur_levels = (self.feature_levels - 1e-8 <= scale).sum().item()
                else:
                    cur_levels = num_levels

                encoding = tcnn.Encoding(n_input_dims=3,
                                         encoding_config={
                                             "otype": "HashGrid",
                                             "n_levels": cur_levels,
                                             "n_features_per_level": features_per_level,
                                             "log2_hashmap_size": log2_hashmap_size,
                                             "base_resolution": base_resolution,
                                             "per_level_scale": cur_level_scale,
                                         })

                try:
                    self.table_offsets.append(self._get_encoding_offsets(encoding, cur_level_scale, cur_levels))
                except:
                    # Sometimes this changes very slightly
                    cur_level_scale = encoding.native_tcnn_module.hyperparams()['per_level_scale']
                    self.table_offsets.append(self._get_encoding_offsets(encoding, cur_level_scale, cur_levels))

                encodings.append(encoding)

            if background_level:
                encodings.append(tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": num_levels,
                        "n_features_per_level": features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale,
                    }
                ))

            self.encodings = nn.ModuleList(encodings)
        else:
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

            self.table_offsets = [self._get_encoding_offsets(self.encoding, per_level_scale, num_levels)]

        if auxiliary_info == AuxiliaryInfo.NONE:
            auxiliary_dims = 0
        elif auxiliary_info == AuxiliaryInfo.SCALE:
            auxiliary_dims = 1
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

            for i in range(num_scales):
                if level_features == LevelFeatures.TRUNCATE:
                    encoding_dims = max((self.feature_levels - 1e-8 <= i).sum() * features_per_level, 0)
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

            if background_level:
                mlp_bases.append(tcnn.Network(
                    n_input_dims=num_levels * features_per_level + auxiliary_dims,
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
            for i in range(num_scales):
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

            if background_level:
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

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()
        explicit_level = ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            if self.background_level and (not explicit_level):
                in_background = positions.view(-1, 3).abs().max(dim=-1)[0] > 1

            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions_flat = positions.view(-1, 3)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

        # Assuming pixels are square
        sample_distances = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2)
        pixel_widths = (ray_samples.frustums.pixel_area.sqrt() * sample_distances).view(-1)

        if explicit_level:
            level = ray_samples.metadata[EXPLICIT_LEVEL] - (0 if self.interpolate_levels else 1e-5)
            pixel_levels = torch.full_like(pixel_widths, level)
        else:
            pixel_levels = self.lod_bias + self.base_log - torch.log(pixel_widths) / self.log_scale_factor
        if self.training and self.training_level_jitter > 0:
            # Randomly assign some training pixels to coarser levels
            to_jitter = torch.rand_like(pixel_levels) <= self.training_level_jitter
            jitter_factor = torch.rand_like(pixel_levels[to_jitter])
            pixel_levels[to_jitter] *= jitter_factor

        if self.background_level and (not explicit_level):
            all_pixel_levels = pixel_levels
            pixel_levels = pixel_levels[in_background <= 0]
            original_indices = \
                torch.arange(positions_flat.shape[0], device=in_background.device, dtype=torch.long)[in_background <= 0]

        if self.do_residual:
            level_indices, level_weights = get_weights_residual(pixel_levels, self.num_scales, self.interpolate_levels)
        elif self.interpolate_levels:
            level_indices, level_weights = get_weights(pixel_levels, self.num_scales)
        else:
            level_indices = get_levels(pixel_levels, self.num_scales)
            level_weights = []
            for li in level_indices:
                level_weights.append(torch.ones_like(li, dtype=pixel_levels.dtype))

        if self.background_level and (not explicit_level):
            pixel_levels = all_pixel_levels
            level_indices = [original_indices[x] for x in level_indices]
            level_indices.append(
                torch.arange(positions_flat.shape[0], device=in_background.device, dtype=torch.long)[in_background > 0])
            level_weights.append(torch.ones_like(level_indices[-1], dtype=pixel_levels.dtype))

        if self.background_level and explicit_level and level == self.num_scales:
            level_indices.insert(0, torch.LongTensor([]))
            level_weights.insert(0, torch.FloatTensor([]))

        auxiliary_info = self._get_auxiliary_info(ray_samples)

        if not self.separate_encoding:
            encoding = self.encoding(positions_flat)
            encoding = self._anneal_features(encoding, self.feature_levels, pixel_levels)

        if self.interpolation_model == InterpolationModel.SINGLE:
            if self.separate_encoding:
                encoding = None
                for level, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
                    if cur_level_indices.shape[0] > 0:
                        cur_pixel_levels = pixel_levels[cur_level_indices]

                        level_encoding = self.encodings[level](positions_flat[cur_level_indices])
                        if self.level_features == LevelFeatures.TRUNCATE:
                            truncated_level_encoding = torch.zeros_like(level_encoding)
                            # Have to do it this way to avoid gradient issues
                            for i, feature_level in enumerate(self.feature_levels):
                                truncated_level_encoding[cur_pixel_levels >= feature_level,
                                :(i + 1) * self.features_per_level] = level_encoding[cur_pixel_levels >= feature_level,
                                                                      :(i + 1) * self.features_per_level]
                            level_encoding = truncated_level_encoding

                        level_encoding = self._anneal_features(level_encoding, self.feature_levels, cur_pixel_levels)
                        if encoding is None:
                            encoding = torch.zeros(positions_flat.shape[0], *level_encoding.shape[1:],
                                                   dtype=level_encoding.dtype, device=level_encoding.device)
                        encoding[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_encoding
            elif self.level_features == LevelFeatures.TRUNCATE:
                for i, feature_level in enumerate(self.feature_levels):
                    encoding[pixel_levels < feature_level,
                    i * self.features_per_level:(i + 1) * self.features_per_level] = 0

            if self.auxiliary_info == AuxiliaryInfo.SCALE:
                auxiliary_info = (pixel_levels.unsqueeze(-1) / (self.num_scales - 1) - 0.5).detach()

            h = self.mlp_base(torch.cat([encoding, auxiliary_info], -1) if auxiliary_info is not None else encoding)
            density_before_activation, additional_info = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation.to(positions) - 1).view(ray_samples.frustums.starts.shape)
            return self._wrap_density(density, pixel_levels, level_indices, additional_info)

        if self.interpolation_model == InterpolationModel.MLP_RGB:
            density = None
            level_embeddings = []
        else:
            interpolated_h = None

        for level, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
            if cur_level_indices.shape[0] > 0:
                cur_pixel_levels = pixel_levels[cur_level_indices]
                if not self.separate_encoding:
                    level_encoding = encoding[cur_level_indices]
                else:
                    level_encoding = self.encodings[level](positions_flat[cur_level_indices])

                level_encoding = self._anneal_features(level_encoding, self.feature_levels, cur_pixel_levels)
                if self.auxiliary_info == AuxiliaryInfo.SCALE:
                    level_auxiliary_info = (cur_pixel_levels.unsqueeze(-1) - level).detach()
                else:
                    level_auxiliary_info = auxiliary_info[cur_level_indices] if auxiliary_info is not None else None

                if self.level_features == LevelFeatures.TRUNCATE:
                    level_encoding = level_encoding[:, :self.mlp_bases[level].n_input_dims
                                                        - (level_auxiliary_info.shape[
                                                               -1] if level_auxiliary_info is not None else 0)]

                base_input = torch.cat([level_encoding, level_auxiliary_info],
                                       -1) if level_auxiliary_info is not None else level_encoding

                level_h = self.mlp_bases[level](base_input)
                if self.interpolation_model == InterpolationModel.MLP_RGB:
                    density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                    level_embeddings.append(level_mlp_out)
                    level_density = trunc_exp(density_before_activation.to(positions) - 1)
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
            density = trunc_exp(density_before_activation.to(positions) - 1)
            additional_info = mlp_out
        else:
            raise Exception(self.interpolation_model)

        return self._wrap_density(density.view(ray_samples.frustums.starts.shape), pixel_levels, level_indices,
                                  additional_info)

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

        outputs = {}

        if self.training:
            density_embedding, level_counts = density_embedding
            outputs[FieldHeadNames.LEVEL_COUNTS] = level_counts
        else:
            density_embedding, levels = density_embedding
            outputs[FieldHeadNames.LEVELS] = levels.view(ray_samples.frustums.starts.shape)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            if self.interpolation_model == InterpolationModel.MLP_RGB:
                _, _, level_embeddings = density_embedding
                density_embedding = level_embeddings[level]

                mlp_head = self.mlp_heads[level]
            else:
                mlp_head = self.mlp_head

            h = torch.cat(
                [d, density_embedding] + ([embedded_appearance] if self.appearance_embedding_dim > 0 else []),
                dim=-1)
            outputs[FieldHeadNames.RGB] = mlp_head(h).view(directions.shape).to(directions)
            return outputs

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
            ray_samples.metadata = {EXPLICIT_LEVEL: self.num_scales - 1}

        density, _ = self.get_density(ray_samples)
        return density

    def _get_auxiliary_info(self, ray_samples: RaySamples) -> Optional[torch.Tensor]:
        #  Scale is calculated per level
        if self.auxiliary_info == AuxiliaryInfo.NONE or self.auxiliary_info == AuxiliaryInfo.SCALE:
            return None
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

    def _anneal_features(self, encoding: torch.Tensor, feature_levels: torch.Tensor,
                         pixel_levels: torch.Tensor) -> torch.Tensor:
        if self.level_features != LevelFeatures.ANNEAL:
            return encoding

        scale_factors = (feature_levels.unsqueeze(-1) - pixel_levels + 1).clamp_min(1).unsqueeze(-1).transpose(0, 1)
        scale_factors = scale_factors.repeat(1, 1, self.features_per_level) \
            .view(-1, len(feature_levels) * self.features_per_level)
        return encoding / scale_factors

    def _wrap_density(self, density: torch.Tensor, pixel_levels: torch.Tensor, level_indices: List[torch.Tensor],
                      additional_info: Any) -> Tuple[torch.Tensor, Tuple[Any, Any]]:
        if self.training:
            level_counts = defaultdict(int)
            for i, cur_level_indices in enumerate(level_indices):
                level_counts[i] = cur_level_indices.shape[0] / pixel_levels.shape[0]

            return density, (additional_info, level_counts)
        else:
            return density, (additional_info, pixel_levels.view(density.shape))

    def _get_encoding_offsets(self, encoding: nn.Module, per_level_scale: float, num_levels: int) -> List[int]:
        grid_offsets = [0]
        total_count = 0
        for i in range(num_levels):
            grid_scale = np.exp2(i * math.log2(per_level_scale)) * self.base_resolution - 1
            resolution = int(math.ceil(grid_scale) + 1)

            params_in_level = resolution ** encoding.n_input_dims
            params_in_level = int(math.ceil(params_in_level / 8)) * 8
            params_in_level = min(params_in_level, 1 << self.log2_hashmap_size)

            total_count += params_in_level
            grid_offsets.append(total_count * self.features_per_level)

        estimated = total_count * self.features_per_level
        expected = encoding.params.shape[0]
        assert estimated == expected, f"{estimated} {expected}"
        return grid_offsets

    # def _get_encoding_offsets(self, encoding: nn.Module, per_level_scale: float, num_levels: int) -> List[int]:
    #     grid_offsets = [0]
    #     total_count = 0
    #     for i in range(num_levels):
    #         grid_scale = torch.exp2(i * torch.log2(torch.FloatTensor([per_level_scale]).squeeze())) * self.base_resolution - 1
    #         resolution = (torch.ceil(grid_scale) + 1).int()
    #
    #         params_in_level = resolution ** 3
    #         params_in_level = (torch.ceil(params_in_level / 8)).int() * 8
    #         params_in_level = min(params_in_level.item(), 1 << self.log2_hashmap_size)
    #         total_count += params_in_level
    #         grid_offsets.append(total_count * self.features_per_level)
    #
    #     import pdb; pdb.set_trace()
    #     assert total_count * self.features_per_level == encoding.params.shape[0]
    #     return grid_offsets


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


@torch.jit.script
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
    if info.casefold() == 'cov':
        return AuxiliaryInfo.COV
    if info.casefold() == 'ipe':
        return AuxiliaryInfo.IPE
    if info.casefold() == 'ipe_grid':
        return AuxiliaryInfo.IPE_GRID
    else:
        raise Exception(info)
