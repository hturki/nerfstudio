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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional, Any

import numpy as np
import torch
from nerfstudio.utils.colors import get_color
from torch.nn import Parameter
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
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.mip_tcnn_field import MipTCNNField, EXPLICIT_LEVEL, interpolation_model
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.mip_instant_ngp import ssim
from nerfstudio.utils import colormaps


@dataclass
class MipNerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: MipNerfactoModel)
    near_plane: float = 0.001
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_resolution: int = 16
    """Base resolution of the hashmap for the base mlp."""
    max_resolution: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 512)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    use_train_appearance_embedding: bool = True
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""

    interpolation_model: Literal["mlp_rgb", "mlp_density", "feature", "ipe", "anneal"] = "ipe"
    use_frustum_area: bool = False
    same_color_mlp: bool = True
    level_window: Optional[int] = None
    training_level_jitter: float = 0

    appearance_embedding_dim: int = 32
    features_per_level: int = 8

    train_with_random_bg: bool = False

    use_all_features: bool = False
    anneal_features: bool = False


class MipNerfactoModel(Model):
    config: MipNerfactoModelConfig

    def __init__(self, config: MipNerfactoModelConfig, metadata: Dict[str, Any], **kwargs) -> None:
        self.near = metadata.get("near", None)
        self.far = metadata.get("far", None)
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        near = self.near if self.near is not None else self.config.near_plane
        far = self.far if self.far is not None else self.config.far_plane

        if self.config.disable_scene_contraction:
            scene_contraction = None
            self.collider = AABBBoxCollider(self.scene_box, near_plane=near)
        else:
            scene_contraction = SceneContraction(order=float("inf"))
            # Collider
            self.collider = NearFarCollider(near_plane=near, far_plane=far)

        # Fields
        self.field = MipTCNNField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_train_appearance_embedding=self.config.use_train_appearance_embedding,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            base_resolution=self.config.base_resolution,
            max_resolution=self.config.max_resolution,
            features_per_level=self.config.features_per_level,
            num_levels=self.config.num_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            interpolation_model=interpolation_model(self.config.interpolation_model),
            use_frustum_area=self.config.use_frustum_area,
            same_color_mlp=self.config.same_color_mlp,
            level_window=self.config.level_window,
            training_level_jitter=self.config.training_level_jitter,
            use_all_features=self.config.use_all_features,
            anneal_features=self.config.anneal_features
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()

        proposal_net_args_list = []
        for i in range(num_prop_nets, 0, -1):
            proposal_net_args_list.append({
                "hidden_dim": 16,
                "log2_hashmap_size": self.config.log2_hashmap_size,
                "num_levels": self.config.num_levels,
                "base_res": self.config.base_resolution,
                "max_res": self.config.max_resolution // (2 ** (i - 1)),
                "use_linear": False
            })

        if self.config.use_same_proposal_network:
            assert len(proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = proposal_net_args_list[min(i, len(proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
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

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_level = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss(reduction="none")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim  # structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        mlps = []
        fields = []
        for children in [self.field.named_children(), self.proposal_networks.named_children()]:
            for name, child in children:
                if 'mlp' in name:
                    mlps += child.parameters()
                else:
                    fields += child.parameters()

        return {
            'mlps': mlps,
            'fields': fields,
        }

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
        outputs = self.get_outputs_inner(ray_bundle)
        # if not self.training and (not ray_bundle.metadata.get('ignore_levels', False)):
        #     num_levels = self.config.grid_feature_scales if self.config.grid_feature_scales is not None else self.config.num_levels
        #     for i in range(num_levels):
        #         level_outputs = self.get_outputs_inner(ray_bundle, i)
        #         outputs[f"rgb_level_{i}"] = level_outputs["rgb"]
        #         outputs[f"depth_level_{i}"] = level_outputs["depth"]

        outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

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

        outputs["rgb"] = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights,
                                           background=background)
        outputs["depth"] = self.renderer_depth(weights=weights, ray_samples=ray_samples)

        if explicit_level is None:
            outputs["accumulation"] = self.renderer_accumulation(weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["level_counts"] = field_outputs[FieldHeadNames.LEVEL_COUNTS]
        else:
            outputs["levels"] = self.renderer_level(weights=weights, semantics=field_outputs[FieldHeadNames.LEVELS])

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = self._get_image(batch, outputs)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
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
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * metrics_dict["interlevel"]
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

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
            images_dict["levels"] = colormaps.apply_colormap(outputs["levels"] / self.config.num_levels, cmap="turbo")

        for i in range(self.config.num_levels):
            if f"rgb_level_{i}" in outputs:
                images_dict[f"rgb_level_{i}"] = torch.cat([image, outputs[f"rgb_level_{i}"]], dim=1)
                images_dict[f"depth_level_{i}"] = colormaps.apply_depth_colormap(
                    outputs[f"depth_level_{i}"],
                    accumulation=outputs["accumulation"],
                )

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
