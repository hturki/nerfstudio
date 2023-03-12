import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Literal

import numpy as np
import torch
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.image_metadata import ImageMetadata
from nerfstudio.data.datamanagers.random_subset_dataset import RandomSubsetDataset, RAY_INDEX
from nerfstudio.data.dataparsers.adop_dataparser import AdopDataParserConfig, TRAIN_INDICES
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, DataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.model_components.ray_generators import RayGenerator

CONSOLE = Console(width=120)


@dataclass
class RandomSubsetDataManagerConfig(DataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: RandomSubsetDataManager)
    """Target class to instantiate."""
    dataparser: DataParserConfig = AdopDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 4096
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 8192
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval; if None, uses all val images."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(
        optimizer=AdamOptimizerConfig(lr=6e-6, eps=1e-15),
        scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=125000))
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    items_per_chunk: int = 12800000
    """Number of entries to load into memory at a time"""
    max_viewer_images: Optional[int] = 1000
    """Maximum number of images to show in viewer"""
    local_cache_path: Optional[str] = "/scratch/hturki/mipnerfacto-cache"
    """Caches images and metadata in specific path if set."""


class RandomSubsetDataManager(DataManager):
    config: RandomSubsetDataManagerConfig

    train_dataset: InputDataset  # Used by the viewer and other things in trainer
    eval_batch_dataset: RandomSubsetDataset

    def __init__(
            self,
            config: RandomSubsetDataManagerConfig,
            device: Union[torch.device, str] = 'cpu',
            test_mode: Literal['test', 'val', 'inference'] = 'test',
            world_size: int = 1,
            local_rank: int = 0
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = 'test' if test_mode in ['test', 'inference'] else 'val'
        self.dataparser = self.config.dataparser.setup()

        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='train')

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataparser_outputs.cameras.size, device=self.device)
        self.train_ray_generator = RayGenerator(self.train_dataparser_outputs.cameras.to(self.device),
                                                self.train_camera_optimizer)

        self.train_batch_dataset = RandomSubsetDataset(
            items=self._get_image_metadata(self.train_dataparser_outputs),
            items_per_chunk=self.config.items_per_chunk,
        )

        self.iter_train_image_dataloader = iter([])

        if len(self.train_dataparser_outputs.image_filenames) > self.config.max_viewer_images:
            indices = set(
                np.linspace(0, len(self.train_dataparser_outputs.image_filenames), self.config.max_viewer_images,
                            endpoint=False, dtype=np.int32))
            viewer_outputs = self.dataparser.get_dataparser_outputs(split='train', indices=indices)
        else:
            viewer_outputs = self.train_dataparser_outputs

        self.train_dataset = InputDataset(viewer_outputs)

        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='test')

        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataparser_outputs.cameras.size, device=self.device)
        self.eval_ray_generator = RayGenerator(self.eval_dataparser_outputs.cameras.to(self.device),
                                               self.eval_camera_optimizer)

        self.eval_dataset = True
        self.eval_batch_dataset = RandomSubsetDataset(
            items=self._get_image_metadata(self.eval_dataparser_outputs),
            items_per_chunk=(self.config.eval_num_rays_per_batch * 10)
        )

        self.iter_eval_batch_dataloader = iter([])

    @cached_property
    def fixed_indices_eval_dataloader(self):
        return FixedIndicesEvalDataloader(
            input_dataset=InputDataset(self.eval_dataparser_outputs),
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def _set_train_loader(self):
        batch_size = self.config.train_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_batch_dataset, self.world_size, self.local_rank)
            assert self.config.train_num_rays_per_batch % self.world_size == 0
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size,
                                                     sampler=self.train_sampler, num_workers=0, pin_memory=True)
        else:
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=0, pin_memory=True)

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def _set_eval_batch_loader(self):
        batch_size = self.config.eval_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_batch_dataset, self.world_size, self.local_rank)
            assert self.config.eval_num_rays_per_batch % self.world_size == 0
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size, sampler=self.eval_sampler,
                                                    num_workers=0, pin_memory=True)
        else:
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True)

        self.iter_eval_batch_dataloader = iter(self.eval_batch_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = next(self.iter_train_image_dataloader, None)
        if batch is None:
            self.train_batch_dataset.load_chunk()
            self._set_train_loader()
            batch = next(self.iter_train_image_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        batch = next(self.iter_eval_batch_dataloader, None)
        if batch is None:
            self.eval_batch_dataset.load_chunk()
            self._set_eval_batch_loader()
            batch = next(self.iter_eval_batch_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])

        if TRAIN_INDICES in self.eval_dataparser_outputs.metadata:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}

            train_indices = self.eval_dataparser_outputs.metadata[TRAIN_INDICES].to(ray_bundle.camera_indices.device)
            ray_bundle.metadata[TRAIN_INDICES] = train_indices[ray_bundle.camera_indices]

        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index = random.choice(self.fixed_indices_eval_dataloader.image_indices)
        ray_bundle, batch = self.fixed_indices_eval_dataloader.get_data_from_image_idx(image_index)

        if TRAIN_INDICES in self.eval_dataparser_outputs.metadata:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}

            train_indices = self.eval_dataparser_outputs.metadata[TRAIN_INDICES].to(ray_bundle.camera_indices.device)
            ray_bundle.metadata[TRAIN_INDICES] = train_indices[ray_bundle.camera_indices]

        return image_index, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != 'off':
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups

    def _get_image_metadata(self, outputs: DataparserOutputs) -> List[ImageMetadata]:
        local_cache_path = Path(self.config.local_cache_path) if self.config.local_cache_path is not None else None

        items = []
        for i in range(len(outputs.image_filenames)):
            items.append(
                ImageMetadata(str(outputs.image_filenames[i]), outputs.cameras.width[i], outputs.cameras.height[i],
                              str(outputs.mask_filenames[i]) if outputs.mask_filenames is not None else None,
                              local_cache_path))

        return items
