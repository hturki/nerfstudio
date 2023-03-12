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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class MulticamDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Multicam)
    """target class to instantiate"""
    data: Path = Path("data/multiscale/mic")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = "white"
    """alpha color of background"""


@dataclass
class Multicam(DataParser):
    config: MulticamDataParserConfig

    def __init__(self, config: MulticamDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None and self.alpha_color.casefold() != "none":
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / "metadata.json")[split]
        image_filenames = []
        poses = []
        width = []
        height = []
        focal_length = []
        cx = []
        cy = []
        weights = []

        for i in range(len(meta["file_path"])):
            image_filenames.append(self.data / meta["file_path"][i])
            poses.append(np.array(meta["cam2world"][i])[:3])
            width.append(meta["width"][i])
            height.append(meta["height"][i])
            focal_length.append(meta["focal"][i])
            cx.append(meta["width"][i] / 2.0)
            cy.append(meta["height"][i] / 2.0)
            weights.append(meta["lossmult"][i])

        poses = np.array(poses).astype(np.float32)
        camera_to_world = torch.from_numpy(poses[:, :3])

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        radius = 1.3 if "ship" not in str(self.data) else 1.5
        scene_box = SceneBox(
            aabb=torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]], dtype=torch.float32)
        )

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.FloatTensor(focal_length),
            fy=torch.FloatTensor(focal_length),
            cx=torch.FloatTensor(cx),
            cy=torch.FloatTensor(cy),
            width=torch.IntTensor(width),
            height=torch.IntTensor(height),
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata={'weights': weights}
        )

        return dataparser_outputs
