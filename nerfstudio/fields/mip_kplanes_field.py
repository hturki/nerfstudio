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
Fields for K-Planes (https://sarafridov.github.io/K-Planes/).
"""

import itertools
from dataclasses import field
from typing import Dict, List, Optional, Tuple

import torch
from nerfacc import contract, ContractionType
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.dataparsers.adop_dataparser import TRAIN_INDICES
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from nerfstudio.fields.kplanes_field import interpolate_ms_features
from nerfstudio.fields.mip_tcnn_field import get_weights

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=120)


def interpolate_ms_features_fl(
        pts: torch.Tensor,
        ms_grids: nn.ModuleList,
        frustum_lengths: List[torch.Tensor],
        grid_voxel_sizes: List[List[float]],
        use_residuals: bool,
        grid_dimensions: int,
        use_tcnn: bool,
        is_training: bool
) -> Tuple[torch.Tensor, Dict[str, float]]:
    coo_combs = list([list(x) for x in itertools.combinations(range(pts.shape[-1]), grid_dimensions)])
    level_map = {}
    interp_space = 1.0
    for ci, (coo_comb, fl) in enumerate(zip(coo_combs, frustum_lengths)):
        level_indices, level_weights = get_weights(fl.view(-1), len(ms_grids), grid_voxel_sizes[ci])
        if not is_training:
            level_map[ci] = torch.zeros_like(fl.unsqueeze(-1))
        interp_out_plane = None
        for i, (cur_level_indices, cur_level_weights) in enumerate(zip(level_indices, level_weights)):
            for level in (range(i) if use_residuals else [i]):
                if cur_level_indices.shape[0] == 0:
                    continue

                grid = ms_grids[level][ci]

                if use_tcnn:
                    l_interp_out_plane = grid(pts[cur_level_indices][..., coo_comb])
                else:
                    l_interp_out_plane = grid_sample_wrapper(grid, pts[cur_level_indices][..., coo_comb])

                if interp_out_plane is None:
                    interp_out_plane = torch.zeros(pts.shape[0], l_interp_out_plane.shape[-1],
                                                   dtype=l_interp_out_plane.dtype, device=l_interp_out_plane.device)

                interp_out_plane[cur_level_indices] += cur_level_weights.unsqueeze(-1) * l_interp_out_plane

            if is_training:
                level_map['{}-{}'.format(i, ci)] = cur_level_indices.shape[0] / pts.shape[0]
            else:
                level_map[ci].view(-1, 1)[cur_level_indices] += i * cur_level_weights.unsqueeze(-1)

        interp_space = interp_space * interp_out_plane

    return interp_space, level_map


class MipKPlanesField(Field):
    """K-Planes field."""

    def __init__(
            self,
            aabb: TensorType,
            num_images: int,
            geo_feat_dim: int = 15,
            grid_config: List[Dict] = field(default_factory=lambda: []),
            concat_features_across_scales: bool = True,
            multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4, 8]),
            spatial_distortion: Optional[SpatialDistortion] = None,
            appearance_embedding_dim: int = 0,
            use_train_appearance_embedding: bool = True,
            use_average_appearance_embedding: bool = False,
            linear_decoder: bool = False,
            linear_decoder_layers: Optional[int] = None,
            use_tcnn: bool = False,
            tcnn_type: str = "DenseGrid",
            use_frustum_lengths: bool = True,
            use_residuals: bool = False
    ) -> None:

        super().__init__()

        self.register_buffer("aabb", aabb)
        self.num_images = num_images
        self.geo_feat_dim = geo_feat_dim
        self.grid_config = grid_config
        self.concat_features_across_scales = concat_features_across_scales
        self.spatial_distortion = spatial_distortion
        self.use_tcnn = use_tcnn

        self.use_frustum_lengths = use_frustum_lengths
        self.use_residuals = use_residuals
        if use_frustum_lengths:
            assert not concat_features_across_scales

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0

        input_dims = grid_config[0]["input_coordinate_dim"]
        self.coo_combs = list(
            [list(x) for x in itertools.combinations(range(input_dims), grid_config[0]["grid_dimensions"])])
        self.grid_voxel_sizes = [[] for _ in self.coo_combs]
        area_of_interest = aabb[1] - aabb[0]
        if input_dims > 3:
            area_of_interest = torch.cat([area_of_interest, torch.ones_like(area_of_interest[..., :1])], -1)

        for res in multiscale_res:
            # initialize coordinate grid
            config = grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]

            for i, (cur_res, coo_comb) in enumerate(zip(config["resolution"], self.coo_combs)):
                self.grid_voxel_sizes[i].append((area_of_interest[coo_comb] / cur_res).max())

            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
                use_tcnn=use_tcnn,
                tcnn_type=tcnn_type
            )

            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features_across_scales:
                self.feature_dim += gp[-1].n_output_dims if use_tcnn else gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].n_output_dims if use_tcnn else gp[-1].shape[1]

            self.grids.append(gp)

        # 2. Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_train_appearance_embedding = use_train_appearance_embedding
            self.use_average_appearance_embedding = use_average_appearance_embedding  # for test-time

        # 3. Init decoder params
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.linear_decoder = linear_decoder
        # 4. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,  # self.direction_encoder.n_output_dims,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                    self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        grid_input = ray_samples.frustums.get_positions().view(-1, 3)

        if self.spatial_distortion is not None:
            grid_input = self.spatial_distortion(grid_input)
            if self.use_tcnn:
                grid_input = (grid_input + 2.0) / 4.0
            else:
                grid_input = grid_input / 2  # from [-2, 2] to [-1, 1]
        else:
            if self.use_tcnn:
                grid_input = contract(x=grid_input, roi=self.aabb, type=ContractionType.AABB)
            else:
                # Input should be in [-1, 1]
                grid_input = (grid_input - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1.0

        if self.grid_config[0]["input_coordinate_dim"] == 4:
            grid_input = torch.cat([grid_input, ray_samples.times.reshape(-1, 1)], -1)

        if self.use_frustum_lengths:
            distances = ray_samples.frustums.directions * (ray_samples.frustums.ends - ray_samples.frustums.starts)
            frustum_lengths = []
            for coo_comb in self.coo_combs:
                frustum_lengths.append(distances[..., torch.LongTensor(coo_comb)].norm(dim=-1))

            features, level_map = interpolate_ms_features_fl(
                grid_input,
                ms_grids=self.grids,
                frustum_lengths=frustum_lengths,
                grid_voxel_sizes=self.grid_voxel_sizes,
                use_residuals=self.use_residuals,
                grid_dimensions=self.grid_config[0]["grid_dimensions"],
                use_tcnn=self.use_tcnn,
                is_training=self.training
            )
        else:
            features = interpolate_ms_features(
                grid_input,
                ms_grids=self.grids,
                grid_dimensions=self.grid_config[0]["grid_dimensions"],
                concat_features=self.concat_features_across_scales,
                num_levels=None,
                use_tcnn=self.use_tcnn
            )
            level_map = {}

        if self.linear_decoder:
            density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        else:
            features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(grid_input) - 1)
        return density, (features, level_map)

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Tuple[List[torch.Tensor], List[torch.Tensor]] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}
        density_embedding, level_map = density_embedding
        if self.training:
            outputs[FieldHeadNames.LEVEL_COUNTS] = level_map
        else:
            outputs[FieldHeadNames.LEVELS] = level_map

        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if self.linear_decoder:
            color_features = [density_embedding]
        else:
            directions = shift_directions_for_tcnn(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training or self.use_train_appearance_embedding:
                if TRAIN_INDICES in ray_samples.metadata:
                    embedded_appearance = self.embedding_appearance(ray_samples.metadata[TRAIN_INDICES].squeeze())
                else:
                    embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (ray_samples.frustums.directions.shape[0], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (ray_samples.frustums.directions.shape[0], self.appearance_embedding_dim),
                    device=directions.device,
                )

            if not self.linear_decoder:
                color_features.append(embedded_appearance.view(-1, self.appearance_embedding_dim))

        color_features = torch.cat(color_features, dim=-1)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if self.linear_decoder:
            if self.appearance_embedding_dim > 0:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]

            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*outputs_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*outputs_shape, -1)

        outputs[FieldHeadNames.RGB] = rgb
        return outputs
