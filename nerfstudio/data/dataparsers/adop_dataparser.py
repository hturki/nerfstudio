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
from typing import Type, List, Optional

import numpy as np
import torch
from pyquaternion import Quaternion

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

OPENCV_TO_OPENGL = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])

DOWN_TO_FORWARD = torch.DoubleTensor([[1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 1]])

TRAIN_INDICES = "train_indices"
WEIGHTS = "weights"


@dataclass
class AdopDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Adop)
    """target class to instantiate"""
    data: Path = Path("data/adop/boat")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    train_with_val_images: bool = False

    train_split: float = 0.9

    train_with_adop_data: bool = True
    downsamples: List[int] = field(default_factory=lambda: [1])


@dataclass
class Adop(DataParser):
    config: AdopDataParserConfig

    def __init__(self, config: AdopDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor

    def get_dataparser_outputs(self, split="train", downsamples:Optional[List[int]] = None):
        if downsamples is None:
            downsamples = self.config.downsamples

        with (self.config.data / "images.txt").open() as f:
            image_paths = f.readlines()

        with (self.config.data / ("adop-poses.txt" if self.config.train_with_adop_data else "poses.txt")).open() as f:
            poses = f.readlines()

        with (self.config.data / (
                "undistorted_intrinsics_adop.txt" if self.config.train_with_adop_data else "undistorted_intrinsics.txt")).open() as f:
            intrinsics = f.readlines()

        assert len(image_paths) == len(poses) == len(intrinsics)

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        skew = []

        for image_path, c2w, K in zip(image_paths, poses, intrinsics):
            image_filenames.append(self.data / 'undistorted_images' / image_path.strip())

            pose_line = [float(x) for x in c2w.strip().split()]
            c2w = torch.DoubleTensor(
                Quaternion(w=pose_line[3], x=pose_line[0], y=pose_line[1], z=pose_line[2]).transformation_matrix)
            c2w[:3, 3] = torch.DoubleTensor(pose_line[4:])

            if self.config.train_with_adop_data:
                c2w = torch.inverse(c2w)

            c2ws.append((DOWN_TO_FORWARD @ (c2w @ OPENCV_TO_OPENGL))[:3].unsqueeze(0))

            K_line = [float(x) for x in K.strip().split()]
            width.append(int(K_line[0]))
            height.append(int(K_line[1]))
            fx.append(K_line[2])
            fy.append(K_line[6])
            cx.append(K_line[4])
            cy.append(K_line[7])

            if K_line[3] > 0:
                skew.append(K_line[3])

        c2ws = torch.cat(c2ws)

        # if self.config.train_with_adop_data:
        #     with (self.data / 'adop-scene-bounds.txt').open() as f:
        #         lines = f.readlines()
        #         assert len(lines) == 2, len(lines)
        #         min_bounds = torch.DoubleTensor([float(x) for x in lines[0].strip().split()])
        #         max_bounds = torch.DoubleTensor([float(x) for x in lines[1].strip().split()])
        # else:
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        print('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        print('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        c2ws[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = set(np.linspace(0, len(image_paths), int(len(image_paths) * self.config.train_split),
                                        endpoint=False, dtype=np.int32))

        if split.casefold() == 'train':
            cameras_width = []
            cameras_height = []
            cameras_fx = []
            cameras_fy = []
            cameras_cx = []
            cameras_cy = []
            cameras_skew = []
            train_indices = []
            weights = []
            if self.config.train_with_val_images:
                mask_filenames = None
            else:
                mask_filenames = []
                for i in range(len(image_paths)):
                    for j in downsamples:
                        train_indices.append(i)
                        cameras_width.append(width[i] // j)
                        cameras_height.append(height[i] // j)
                        cameras_fx.append(fx[i] / j)
                        cameras_fy.append(fy[i] / j)
                        cameras_cx.append(cx[i] / j)
                        cameras_cy.append(cy[i] / j)
                        if len(skew) > 0:
                            cameras_skew.append(skew[i])
                        weights.append(j ** 2)

                        if i in train_indices:
                            mask_filenames.append(self.data / 'image_full.png')
                        else:
                            mask_filenames.append(self.data / 'image_left.png')

            indices = torch.LongTensor(train_indices)
            weights = torch.FloatTensor(weights)

            cameras = Cameras(
                camera_to_worlds=c2ws[indices].float(),
                fx=torch.FloatTensor(cameras_fx),
                fy=torch.FloatTensor(cameras_fy),
                cx=torch.FloatTensor(cameras_cx),
                cy=torch.FloatTensor(cameras_cy),
                width=torch.LongTensor(cameras_width),
                height=torch.LongTensor(cameras_height),
                skew=torch.FloatTensor(cameras_skew).unsqueeze(-1) if len(cameras_skew) > 0 else None,
                camera_type=CameraType.PERSPECTIVE,
            )
        else:
            val_indices = []
            mask_filenames = []
            for i in range(len(image_paths)):
                if i not in train_indices:
                    val_indices.append(i)
                    mask_filenames.append(self.data / 'image_right.png')

            indices = torch.LongTensor(val_indices)
            weights = None

            cameras = Cameras(
                camera_to_worlds=c2ws[indices].float(),
                fx=torch.FloatTensor(fx)[indices],
                fy=torch.FloatTensor(fy)[indices],
                cx=torch.FloatTensor(cx)[indices],
                cy=torch.FloatTensor(cy)[indices],
                width=torch.IntTensor(width)[indices],
                height=torch.IntTensor(height)[indices],
                skew=torch.FloatTensor(skew)[indices].unsqueeze(-1) if len(skew) > 0 else None,
                camera_type=CameraType.PERSPECTIVE,
            )

        print('Num images in split {}: {}'.format(split, len(indices)))

        metadata = {TRAIN_INDICES: indices, "cameras": cameras}
        if weights is not None:
            metadata[WEIGHTS] = weights

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            mask_filenames=mask_filenames,
            metadata=metadata
        )

        return dataparser_outputs
