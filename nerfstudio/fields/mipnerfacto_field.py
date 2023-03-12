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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import tinycudann as tcnn
import torch
from torch import nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)
from nerfstudio.field_components.spatial_distortions import (
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.nerfacto_field import get_normalized_directions

EXPLICIT_LEVEL = "explicit_level"


class MipNerfactoField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
            self,
            aabb,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_levels: int = 8,
            features_per_level: int = 4,
            base_res: int = 16,
            max_res: int = 2048,
            log2_hashmap_size: int = 19,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.num_levels = num_levels
        self.features_per_level = features_per_level

        self.spatial_distortion = spatial_distortion

        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            }
        )

        self.areas = []
        mlp_bases = []
        mlp_heads = []

        for i in range(num_levels):
            self.areas.append(1 / (base_res * (2 ** i)))
            mlp_bases.append(tcnn.Network(
                n_input_dims=features_per_level * (i + 1),
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            ))

            mlp_heads.append(tcnn.Network(
                n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            ))

        self.mlp_bases = nn.ModuleList(mlp_bases)
        self.mlp_heads = nn.ModuleList(mlp_heads)

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        encoding = self.encoding(positions.view(-1, 3))

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            h = self.mlp_bases[level](encoding[:, :(level + 1) * self.features_per_level])
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation.to(positions)).view(ray_samples.frustums.starts.shape)
            return density, base_mlp_out

        level_indices, level_weights = _get_weights(ray_samples.frustums.starts, ray_samples.frustums.ends,
                                                    ray_samples.frustums.pixel_area, self.num_levels, self.areas)
        density = torch.zeros_like(ray_samples.frustums.starts)
        level_embeddings = []

        for i, (level_mlp_base, cur_level_indices, cur_level_weights) in enumerate(
                zip(self.mlp_bases, level_indices, level_weights)):
            if cur_level_indices.shape[0] > 0:
                h = level_mlp_base(encoding[cur_level_indices, :(i + 1) * self.features_per_level])
                density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
                density.view(-1, 1)[cur_level_indices] += \
                    cur_level_weights.unsqueeze(-1) * trunc_exp(density_before_activation.to(positions))
                level_embeddings.append(base_mlp_out)
            else:
                level_embeddings.append([])

        return density, (level_indices, level_weights, level_embeddings)

    def get_outputs(self, ray_samples: RaySamples,
                    density_embedding: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            h = torch.cat([d, density_embedding], dim=-1)
            return {FieldHeadNames.RGB: self.mlp_heads[level](h).view(directions.shape).to(directions)}

        level_indices, level_weights, level_embeddings = density_embedding

        rgbs = torch.zeros_like(directions)
        outputs = {}

        if self.training:
            level_counts = defaultdict(int)

        for i, (level_mlp_head, cur_level_indices, cur_level_weights, cur_level_embeddings) in enumerate(
                zip(self.mlp_heads,
                    level_indices,
                    level_weights,
                    level_embeddings)):
            if cur_level_indices.shape[0] > 0:
                h = torch.cat([d[cur_level_indices], cur_level_embeddings], dim=-1)
                rgbs.view(-1, 3)[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_mlp_head(h)

            if self.training:
                level_counts[i] = cur_level_indices.shape[0] / directions_flat.shape[0]

        outputs[FieldHeadNames.RGB] = rgbs

        if self.training:
            outputs[FieldHeadNames.LEVEL_COUNTS] = level_counts

        return outputs


@torch.jit.script
def _get_weights(starts: torch.Tensor, ends: torch.Tensor, pixel_area: torch.Tensor, num_levels: int,
                 areas: list[float]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    pixel_areas = (pixel_area * (((starts + ends) / 2) ** 2)).view(-1)

    sorted_pixel_areas, ordering = pixel_areas.sort(descending=True)
    level_indices = []
    level_weights = []

    last_weights = torch.empty(0)

    start = 0
    mid = 0
    end = 0
    for i in range(num_levels):
        if i == 0:
            mid = (sorted_pixel_areas > areas[i]).sum()
            cur_level_indices = [ordering[:mid]]
            cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=starts.dtype)]
        else:
            start = mid
            mid = end
            cur_level_indices = [ordering[start:mid]]
            cur_level_weights = [1 - last_weights]

        if i < num_levels - 1:
            end = mid + (sorted_pixel_areas[mid:] > areas[i + 1]).sum()
            cur_level_indices.append(ordering[mid:end])
            last_weights = (sorted_pixel_areas[mid:end] - areas[i + 1]) / (areas[i] - areas[i + 1])
            cur_level_weights.append(last_weights)
        else:
            cur_level_indices.append(ordering[mid:])
            cur_level_weights.append(torch.ones_like(cur_level_indices[-1], dtype=starts.dtype))

        level_indices.append(torch.cat(cur_level_indices))
        level_weights.append(torch.cat(cur_level_weights))

    return level_indices, level_weights
