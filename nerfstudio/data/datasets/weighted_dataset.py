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

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path


class WeightedDataset(InputDataset):
    """Dataset that returns images and loss weights.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "weights" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["weights"], list)
        self.weights = self.metadata["weights"]

    def get_metadata(self, data: Dict) -> Dict:
        return {"weights": self.weights[data["image_idx"]] * torch.ones(*data["image"].shape[:2])}
