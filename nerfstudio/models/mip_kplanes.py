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
from matplotlib.colors import ListedColormap
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
from nerfstudio.fields.kplanes_field import KPlanesDensityField
from nerfstudio.fields.mip_kplanes_field import MipKPlanesField
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
    RGBRenderer, SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.kplanes import space_tv_loss
from nerfstudio.models.mip_instant_ngp import ssim
from nerfstudio.models.mipnerfacto import turbo_colormap_data
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.colors import get_color


@dataclass
class MipKPlanesModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: MipKPlanesModel)

    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""

    grid_config: List[Dict] = field(
        default_factory=lambda: [
            {"grid_dimensions": 2, "input_coordinate_dim": 3, "output_coordinate_dim": 32, "resolution": [64, 64, 64]}
        ]
    )

    is_ndc: bool = False
    is_contracted: bool = True
    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    concat_features_across_scales: bool = False
    linear_decoder: bool = False
    linear_decoder_layers: Optional[int] = 4

    # proposal sampling arguments
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_input_coords": 3, "num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_input_coords": 3, "num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    num_proposal_samples: Optional[Tuple[int]] = (256, 128)
    num_samples: Optional[int] = 48
    single_jitter: bool = False
    proposal_warmup: int = 5000
    proposal_update_every: int = 5
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0

    # appearance embedding (set to 0 to disable)
    appearance_embedding_dim: int = 32
    use_train_appearance_embedding: bool = True
    use_average_appearance_embedding: bool = False

    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    train_with_random_bg: bool = False

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"img": 1.0, "plane_tv": 0.01, "plane_tv_proposal_net": 0.0001, "distortion": 0.001, "interlevel": 1.0}
    )

    use_tcnn: bool = False
    tcnn_type: str = "DenseGrid"

    use_frustum_lengths: bool = True
    use_residuals: bool = False


class MipKPlanesModel(Model):
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """

    config: MipKPlanesModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # Fields
        self.field = MipKPlanesField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_config=self.config.grid_config,
            concat_features_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_train_appearance_embedding=self.config.use_train_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
            use_tcnn=self.config.use_tcnn,
            tcnn_type=self.config.tcnn_type,
            use_frustum_lengths=self.config.use_frustum_lengths,
            use_residuals=self.config.use_residuals,
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
                use_tcnn=self.config.use_tcnn,
                tcnn_type=self.config.tcnn_type,
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
                    use_tcnn=self.config.use_tcnn,
                    tcnn_type=self.config.tcnn_type,
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

        if self.config.is_contracted or self.config.is_ndc:
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
        background_color = self.config.background_color
        self.renderer_rgb = RGBRenderer(
            background_color=background_color, return_background=self.config.train_with_random_bg
        )
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_level = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss(reduction="none")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim  # structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

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

        if self.training and self.config.train_with_random_bg:
            standard_bg_color = self.renderer_rgb.background_color
            self.renderer_rgb.background_color = "random"

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        if self.config.train_with_random_bg:
            rgb, bg_color = rgb

        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.training and self.config.train_with_random_bg:
            self.renderer_rgb.background_color = standard_bg_color
            outputs["bg_color"] = bg_color

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["level_counts"] = field_outputs[FieldHeadNames.LEVEL_COUNTS]
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        else:
            for key, val in field_outputs[FieldHeadNames.LEVELS].items():
                outputs[f"levels_{key}"] = self.renderer_level(weights=weights, semantics=val)

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = self._get_image(batch, outputs)

        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["plane_tv"] = space_tv_loss(self.field.grids, self.config.use_tcnn)
            metrics_dict["plane_tv_proposal_net"] = space_tv_loss([p.grids for p in self.proposal_networks],
                                                                  self.config.use_tcnn)
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])

            for key, val in outputs["level_counts"].items():
                metrics_dict[f"level_counts_{key}"] = val

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = self._get_image(batch, outputs)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        if "weights" in batch:
            weights = batch["weights"].to(self.device).unsqueeze(-1)
            rgb_loss *= weights

        loss_dict = {"rgb_loss": rgb_loss.mean()}

        if self.training:
            loss_dict["plane_tv"] = metrics_dict["plane_tv"].clone()
            loss_dict["plane_tv_proposal_net"] = metrics_dict["plane_tv_proposal_net"].clone()
            loss_dict["distortion"] = metrics_dict["distortion"].clone()
            loss_dict["interlevel"] = metrics_dict["interlevel"].clone()
            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

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

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        if not self.training:
            for i in range(len(self.field.coo_combs)):
                images_dict[f"levels_{i}"] = colormaps.apply_colormap(
                    outputs[f"levels_{i}"] / len(self.config.multiscale_res), cmap=ListedColormap(turbo_colormap_data))

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
