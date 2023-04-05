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
Weighted dataset.
"""

from typing import Dict

import numpy as np
import torch
from PIL import Image

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class WeightedDataset(InputDataset):
    """Dataset that returns images and loss weights.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "weights" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["weights"], list)
        self.weights = self.metadata["weights"]
        self.depth_images = self.metadata.get("depth_image", None)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {"weights": self.weights[data["image_idx"]] * torch.ones(*data["image"].shape[:2])}

        if self.depth_images is not None:
            filepath = self.depth_images[data["image_idx"]]
            depth_image = torch.FloatTensor(np.load(filepath)).unsqueeze(-1) / self.metadata["pose_scale_factor"]
            metadata["depth_image"] = depth_image

        return metadata
