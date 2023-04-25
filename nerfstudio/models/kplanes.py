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
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.kplanes_field import KPlanesDensityField, KPlanesField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.mip_instant_ngp import ssim
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.colors import get_color


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)

    near_plane: float = 0.05
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    # proposal sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""

    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    """Arguments for the proposal density fields."""

    num_proposal_samples: Optional[Tuple[int]] = (256, 128)
    """Number of samples per ray for each proposal network."""

    num_samples: Optional[int] = 48
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""

    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""

    use_train_appearance_embedding: bool = True

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    train_with_random_bg: bool = True
    """Use a randomized background during training"""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "img": 1.0,
            "interlevel": 1.0,
            "distortion": 0.001,
            "plane_tv": 0.0001,
            "plane_tv_proposal_net": 0.0001,
            "l1_time_planes": 0.0001,
            "l1_time_planes_proposal_net": 0.0001,
            "time_smoothness_weight": 0.1,
            "time_smoothness_weight_proposal_net": 0.001,
        }
    )
    """Loss coefficients."""


class KPlanesModel(Model):
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """

    config: KPlanesModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # Fields
        self.field = KPlanesField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            concat_features_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_train_appearance_embedding=self.config.use_train_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                linear_decoder=self.config.linear_decoder,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    linear_decoder=self.config.linear_decoder,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        if self.config.is_contracted:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=self.config.single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_samples,
            num_proposal_samples_per_ray=self.config.num_proposal_samples,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim  # structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        outputs = {}
        if self.training and self.config.train_with_random_bg:
            background = torch.rand_like(ray_bundle.origins)
            outputs["bg_color"] = background
        else:
            background = None

        outputs["rgb"] = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB], weights=weights, background=background
        )
        outputs["depth"] = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        outputs["accumulation"] = self.renderer_accumulation(weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = self._get_image(batch, outputs)

        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

            field_grids = [x.plane_coefs for x in self.field.grids]
            metrics_dict["plane_tv"] = space_tv_loss(field_grids)

            prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
            metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)

            if len(self.config.grid_base_resolution) == 4:
                metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
                metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
                metrics_dict["time_smoothness_weight"] = time_smoothness(field_grids)
                metrics_dict["time_smoothness_weight_proposal_net"] = time_smoothness(prop_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = self._get_image(batch, outputs)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        if "weights" in batch:
            weights = batch["weights"].to(self.device).view(-1, 1)
            rgb_loss *= weights

        loss_dict = {"rgb_loss": rgb_loss.mean()}

        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = self._get_image(batch, outputs)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        if "mask" in batch:
            mask = batch["mask"]
            assert torch.all(mask[:, mask.sum(dim=0) > 0])
            image = image[:, mask.sum(dim=0).squeeze() > 0]
            rgb = rgb[:, mask.sum(dim=0).squeeze() > 0]

        ssim = self.ssim(image, rgb)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = np.exp(-0.1 * np.log(10.) * float(psnr.item()))
        dssim = np.sqrt((1 - float(ssim)) / 2)
        avg_error = np.exp(np.mean(np.log(np.array([mse, dssim, float(lpips)]))))

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips),
            "avg_error": avg_error
        }  # type: ignore

        if "weights" in batch:
            weight = torch.unique(batch["weights"]).item()
            for key, val in set(metrics_dict.items()):
                metrics_dict[f"{key}_{weight}"] = val
            for key, val in set(images_dict.items()):
                if 'level' not in key:
                    images_dict[f"{key}_{weight}"] = val

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def _get_image(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["image"].to(self.device)

        if self.config.train_with_random_bg:
            alpha = batch["alpha"].to(self.device)
            bg = outputs["bg_color"] if self.training else get_color(self.config.background_color).to(self.device)
            image = image * alpha + bg * (1.0 - alpha)

        return image


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.
    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension
    Returns:
        Total variance
    """

    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each plane in the grids.
    Args:
        multi_res_grids: Grids to compute total variance over
    Returns:
        Total variance
    """

    total = 0.0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = [0, 1, 2]
        else:
            spatial_planes = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
    return total


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.
    Args:
        multi_res_grids: Grids to compute L1 distance over
    Returns:
         L1 distance from the multiplicative identity (1)
    """

    total = 0.0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()

    return total


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes time smoothness across a plane.
    Args:
        t: Plane tensor
    Returns:
        Time smoothness
    """

    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes time smoothness across each plane in the grids.
    Args:
        multi_res_grids: Grids to compute time smoothness over
    Returns:
        Time smoothness
    """

    total = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])

    return total
