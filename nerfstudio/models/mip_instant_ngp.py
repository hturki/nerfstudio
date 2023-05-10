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
from typing import Dict, List, Optional, Tuple, Type, Any

import nerfacc
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.nn import Parameter, SmoothL1Loss
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
from nerfstudio.fields.mip_tcnn_field import MipTCNNField, EXPLICIT_LEVEL, interpolation_model, level_features, \
    auxiliary_info
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, SemanticRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors
from nerfstudio.utils.colors import get_color

CONSOLE = Console(width=120)


def ssim(
        target_rgbs: torch.Tensor,
        rgbs: torch.Tensor,
        max_val: float = 1,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
) -> float:
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.
    Args:
      rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      target_rgbs: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
    Returns:
      Each image's mean SSIM.
    """
    device = rgbs.device
    ori_shape = rgbs.size()
    width, height, num_channels = ori_shape[-3:]
    rgbs = rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    target_rgbs = target_rgbs.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(rgbs)
    mu1 = filt_fn(target_rgbs)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgbs ** 2) - mu00
    sigma11 = filt_fn(target_rgbs ** 2) - mu11
    sigma01 = filt_fn(rgbs * target_rgbs) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    return torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1).item()


@dataclass
class MipInstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: MipNGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 1024
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Number of grid levels"""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    alpha_thre: float = 1e-2
    """Alpha threshold for skipping empty space. Should be set to 0 for blender scenes."""
    near_plane: float = 0.02
    """How far along ray to start sampling."""
    far_plane: Optional[float] = None
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    features_per_level: int = 8
    """Feature dimension at each level."""
    base_resolution: int = 16
    """Base resolution of the hashmap for the base mlp."""
    max_resolution: int = 65536
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19

    interpolation_model: Literal["mlp_rgb", "mlp_density", "single"] = "mlp_rgb"
    level_features: Literal["truncate", "anneal", "full"] = "truncate"
    auxiliary_info: Literal["none", "scale", "cov", "ipe", "ipe_grid"] = "none"
    num_scales: Optional[int] = None
    scale_factor: Optional[float] = None
    separate_encoding: bool = True
    interpolate_levels: bool = True
    lod_bias: int = 0
    train_lod_bias: bool = False
    freq_dim: int = 4
    freq_resolution: int = 8192
    do_residual: bool = False
    training_level_jitter: float = 0

    grid_loss_mult: float = 0

    train_with_random_bg: bool = False
    use_sigma_fn: bool = True

    appearance_embedding_dim: int = 32
    use_train_appearance_embedding: bool = True
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    depth_lambda: float = 0
    force_weight_1: bool = False
    finest_occ_grid: bool = False
    debug: bool = False
    background_level: bool = False
    separate_res: bool = True

    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""


class MipNGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: MipInstantNGPModelConfig
    field: MipTCNNField

    def __init__(self, config: MipInstantNGPModelConfig, metadata: Dict[str, Any], **kwargs) -> None:
        self.cameras = metadata["cameras"]
        self.near = metadata.get("near", None)
        self.far = metadata.get("far", None)

        super().__init__(config=config, **kwargs)

        self.render_step_size = (
                (self.scene_box.aabb[1] - self.scene_box.aabb[0]).max()
                * math.sqrt(3)
                / self.config.max_num_samples_per_ray
        ).item()

        print('Render step size', self.render_step_size)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scale_factor == 0:
            self.config.scale_factor = None

        if self.config.num_scales == 0:
            self.config.num_scales = None

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_box.aabb.flatten(),
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        self.field = MipTCNNField(
            self.occupancy_grid.aabbs[-1].view(2, 3),
            num_images=self.num_train_data,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            disable_scene_contraction=True,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_train_appearance_embedding=self.config.use_train_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            base_resolution=self.config.base_resolution,
            max_resolution=self.config.max_resolution,
            features_per_level=self.config.features_per_level,
            num_levels=self.config.num_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            interpolation_model=interpolation_model(self.config.interpolation_model),
            level_features=level_features(self.config.level_features),
            auxiliary_info=auxiliary_info(self.config.auxiliary_info),
            num_scales=self.config.num_scales,
            scale_factor=self.config.scale_factor,
            separate_encoding=self.config.separate_encoding,
            interpolate_levels=self.config.interpolate_levels,
            training_level_jitter=self.config.training_level_jitter,
            lod_bias=self.config.lod_bias,
            train_lod_bias=self.config.train_lod_bias,
            freq_dim=self.config.freq_dim,
            freq_resolution=self.config.freq_resolution,
            do_residual=self.config.do_residual,
            cameras=self.cameras,
            finest_occ_grid=self.config.finest_occ_grid,
            debug=self.config.debug,
            background_level=self.config.background_level,
            separate_res=self.config.separate_res
        )

        # self.collider = AABBBoxCollider(self.occupancy_grid.aabbs[-1],
        #                                 near_plane=self.near if self.near is not None else self.config.near_plane)

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.background_color = "random"
        if self.config.background_color in ["white", "black"]:
            self.background_color = colors.COLORS_DICT[self.config.background_color]

        background_color = self.config.background_color
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_level = SemanticRenderer()

        # losses
        self.rgb_loss = SmoothL1Loss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.trained_levels = set()

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x, None, self.render_step_size) * self.render_step_size,
            )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        mlps = []
        fields = []
        for name, child in self.field.named_children():
            if 'mlp' in name:
                mlps += child.parameters()
            else:
                fields += child.parameters()

        return {
            'mlps': mlps,
            'fields': fields,
        }

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = self.get_outputs_inner(ray_bundle)
        if not self.training and (not ray_bundle.metadata.get('ignore_levels', False)):
            for i in range(self.field.num_scales):
                if i in self.trained_levels:
                    level_outputs = self.get_outputs_inner(ray_bundle, i)
                    outputs[f"rgb_level_{i}"] = level_outputs["rgb"]
                    outputs[f"depth_level_{i}"] = level_outputs["depth"]

        outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        ends = ray_samples.frustums.ends
        if self.config.force_weight_1:
            ends = ends.clone()
            ends[packed_info[..., 0].long() - 1] = 1e10
            ends[-1] = 1e10

        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ends[..., 0],
        )[0][..., None]

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

        if self.training:
            outputs["level_counts"] = field_outputs[FieldHeadNames.LEVEL_COUNTS]
        elif explicit_level is None:
            levels = field_outputs[FieldHeadNames.LEVELS]
            outputs["levels"] = self.renderer_level(weights=weights,
                                                    semantics=levels.clamp(0, self.field.num_scales - 1),
                                                    ray_indices=ray_indices, num_rays=num_rays)

        if explicit_level is None:
            accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
            alive_ray_mask = accumulation.squeeze(-1) > 0

            outputs["accumulation"] = accumulation
            outputs["alive_ray_mask"] = alive_ray_mask  # the rays we kept from sampler
            outputs["num_samples_per_ray"] = packed_info[:, 1]
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

        if self.training:
            for key, val in outputs["level_counts"].items():
                metrics_dict[f"level_counts_{key}"] = val
                if val > 0:
                    self.trained_levels.add(key)

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

        if self.training:
            if self.config.grid_loss_mult > 0:
                grid_weight_decay = 0
                if self.config.separate_encoding:
                    encodings = self.field.encodings
                else:
                    encodings = [self.field.encoding]

                assert len(encodings) == len(self.field.table_offsets)

                for encoding, offsets in zip(encodings, self.field.table_offsets):
                    for i in range(len(offsets) - 1):
                        grid_weight_decay += encoding.params[offsets[i]:offsets[i + 1]].square().mean()

                loss_dict["grid_weight_loss"] = self.config.grid_loss_mult * grid_weight_decay

            for key, val in loss_dict.items():
                if not math.isfinite(val):
                    raise Exception('Loss is not finite: {}'.format(loss_dict))

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

        if not self.training:
            images_dict["levels"] = colormaps.apply_colormap(outputs["levels"] / self.config.num_levels, cmap="turbo")

        for i in range(self.config.num_levels):
            if f"rgb_level_{i}" in outputs:
                images_dict[f"rgb_level_{i}"] = torch.cat([image, outputs[f"rgb_level_{i}"]], dim=1)
                images_dict[f"depth_level_{i}"] = colormaps.apply_depth_colormap(
                    outputs[f"depth_level_{i}"],
                    accumulation=outputs["accumulation"],
                )

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

        return metrics_dict, images_dict

    def _get_image(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["image"].to(self.device)

        if self.config.train_with_random_bg:
            alpha = batch["alpha"].to(self.device)
            bg = outputs["bg_color"] if self.training else get_color(self.config.background_color).to(self.device)
            image = image * alpha + bg * (1.0 - alpha)

        return image
