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
from typing import Tuple, Optional, Any

import tinycudann as tcnn
import torch
from rich.console import Console
from torch.distributions import MultivariateNormal
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.dataparsers.adop_dataparser import TRAIN_INDICES
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

CONSOLE = Console(width=120)


class SSTCNNField(Field):
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
            disable_scene_contraction: bool = False,
            appearance_embedding_dim: int = 32,
            use_train_appearance_embedding: bool = True,
            use_average_appearance_embedding: bool = False,
            base_resolution: int = 16,
            max_resolution: int = 4096,
            features_per_level: int = 2,
            num_levels: int = 16,
            log2_hashmap_size: int = 19,
            samples: int = 8,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb, persistent=False)

        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion
        self.disable_scene_contraction = disable_scene_contraction

        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)
            self.use_train_appearance_embedding = use_train_appearance_embedding
            self.use_average_appearance_embedding = use_average_appearance_embedding

        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.features_per_level = features_per_level
        self.samples = samples

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

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

        self.mlp_base = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

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

    def get_density(self, ray_samples: RaySamples, super_sample: bool = True):
        positions_flat = ray_samples.frustums.get_positions().view(-1, 3)

        if super_sample:
            frust = ray_samples.frustums
            frust = Frustums(
                origins=frust.origins.double(),
                directions=frust.directions.double(),
                starts=frust.starts.double(),
                ends=frust.ends.double(),
                pixel_area=frust.pixel_area.double(),
            )
            gaussian = frust.get_gaussian_blob()
            samples = MultivariateNormal(gaussian.mean.view(-1, 3), gaussian.cov.view(-1, 3, 3))
            positions_flat = samples.sample((self.samples,)).view(-1, 3).to(positions_flat)

        if self.spatial_distortion is not None:
            positions_flat = self.spatial_distortion(positions_flat)
            positions_flat = (positions_flat + 2.0) / 4.0
        else:
            positions_flat = SceneBox.get_normalized_positions(positions_flat, self.aabb)

        encoding = self.encoding(positions_flat)
        h = self.mlp_base(encoding)
        density_before_activation, additional_info = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        density = trunc_exp(density_before_activation.to(positions_flat) - 1)

        if not self.training:
            density = torch.nan_to_num(density)
            additional_info = torch.nan_to_num(additional_info)

        if super_sample:
            density = density.view(self.samples, -1).mean(dim=0)

        return density.view(ray_samples.frustums.starts.shape), additional_info

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

        h = torch.cat([d.repeat(1, self.samples).view(-1, d.shape[-1]), density_embedding] +
                      ([embedded_appearance.repeat(1, self.samples).view(-1, self.appearance_embedding_dim)]
                       if self.appearance_embedding_dim > 0 else []), dim=-1)

        rgbs = self.mlp_head(h).reshape(self.samples, -1).mean(dim=0)
        rgbs = rgbs.view(directions.shape).to(directions)

        if not self.training:
            rgbs = torch.nan_to_num(rgbs)

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

    def density_fn(
            self, positions: TensorType["bs":..., 3], times: TensorType["bs":..., 1],
            step_size: int = None, origins: Optional[torch.Tensor] = None,
            directions: Optional[torch.Tensor] = None, starts: Optional[torch.Tensor] = None,
            ends: Optional[torch.Tensor] = None, pixel_area: Optional[torch.Tensor] = None) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
            times: the times of the samples
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
            times=times,
        )
        density, _ = self.get_density(ray_samples, super_sample=False)
        return density
