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
Implementation of Instant NGP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import numpy as np
import torch
import torch.nn.functional as F
from nerfacc import ContractionType
from torch.nn import Parameter, SmoothL1Loss, HuberLoss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.mip_instant_ngp import ssim
from nerfstudio.utils import colormaps, colors
from nerfstudio.utils.colors import get_color


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 24
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE
    """Contraction type used for spatial deformation of the field."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    alpha_thre: float = 1e-2
    """Alpha threshold for skipping empty space. Should be set to 0 for blender scenes."""
    render_step_size: float = 0.01
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: Optional[float] = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""

    hidden_dim: int = 64
    hidden_dim_color: int = 64

    depth_lambda: float = 0
    num_levels: int = 16
    features_per_level: int = 8

    train_with_random_bg: bool = False


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.config.render_step_size = (
            (self.scene_box.aabb[1] - self.scene_box.aabb[0]).max()
            * math.sqrt(3)
            / self.config.max_num_samples_per_ray
        ).item()

        print('Render step size', self.config.render_step_size)

        self.field = TCNNInstantNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.background_color = "random"
        if self.config.background_color in ["white", "black"]:
            self.background_color = colors.COLORS_DICT[self.config.background_color]

        self.renderer_rgb = RGBRenderer(background_color=self.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = HuberLoss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            'fields': list(self.field.parameters()),
        }

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
                alpha_thre=self.config.alpha_thre,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        outputs = {}
        if self.training and self.config.train_with_random_bg:
            background = torch.rand_like(ray_bundle.origins)
            outputs["bg_color"] = background
        else:
            background = None

        outputs["rgb"] = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
            background=background
        )

        outputs["depth"] = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )

        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        outputs["accumulation"]  = accumulation
        outputs["alive_ray_mask"]  = accumulation.squeeze(-1) > 0
        outputs["num_samples_per_ray"]  = packed_info[:, 1]
        outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = self._get_image(batch, outputs)
        metrics_dict = {}
        rgb = outputs["rgb"]
        assert torch.isfinite(rgb).all(), rgb
        mask = outputs["alive_ray_mask"]
        metrics_dict["alive_ray_mask"] = mask.sum()

        metrics_dict["mse"] = F.mse_loss(rgb[mask], image[mask])
        metrics_dict["psnr"] = self.psnr(rgb[mask], image[mask])
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = self._get_image(batch, outputs)
        mask = outputs["alive_ray_mask"]

        rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        if "weights" in batch:
            weights = batch["weights"].to(self.device).view(-1, 1)
            rgb_loss *= weights[mask]

        loss_dict = {"rgb_loss": rgb_loss.mean()}

        if "depth_image" in batch and self.config.depth_lambda > 0:
            euclidian_depth = batch["depth_image"] * outputs["directions_norm"]
            depth_loss = F.mse_loss(outputs["depth"][mask], euclidian_depth[mask], reduction="none")
            if "weights" in batch:
                depth_loss *= weights[mask]
            loss_dict["depth"] = self.config.depth_lambda * depth_loss.mean()

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = self._get_image(batch, outputs)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        alive_ray_mask = colormaps.apply_colormap(outputs["alive_ray_mask"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)

        depth_vis = []
        if "depth_image" in batch:
            depth_vis.append(colormaps.apply_depth_colormap(
                batch["depth_image"] * outputs["directions_norm"],
            ))

        depth_vis.append(depth)
        combined_depth = torch.cat(depth_vis, dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
        }

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
                images_dict[f"{key}_{weight}"] = val

        return metrics_dict, images_dict

    def _get_image(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["image"].to(self.device)

        if self.config.train_with_random_bg:
            alpha = batch["alpha"].to(self.device)
            bg = outputs["bg_color"] if self.training else get_color(self.config.background_color).to(self.device)
            image = image * alpha + bg * (1.0 - alpha)

        return image
