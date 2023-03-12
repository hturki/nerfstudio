from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set, List

import torch
from rich.console import Console
from torch.utils.data import Dataset

from nerfstudio.data.datamanagers.image_metadata import ImageMetadata

CONSOLE = Console(width=120)

RGB = 'image'
MASK = 'mask'
IMAGE_INDEX = 'image_index'
PIXEL_INDEX = 'pixel_index'
RAY_INDEX = 'ray_index'


class RandomSubsetDataset(Dataset):

    def __init__(self,
                 items: List[ImageMetadata],
                 items_per_chunk: int):
        super(RandomSubsetDataset, self).__init__()

        self.items = items
        self.items_per_chunk = items_per_chunk

        self.chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self.on_demand_executor = ThreadPoolExecutor(max_workers=16)

        pixel_indices_to_sample = []
        for item in items:
            pixel_indices_to_sample.append(item.W * item.H)

        self.pixel_indices_to_sample = torch.LongTensor(pixel_indices_to_sample)
        assert len(self.pixel_indices_to_sample) > 0

        self.loaded_fields = None
        self.loaded_field_offset = 0
        self.chunk_future = None
        self.loaded_chunk = None

    def load_chunk(self) -> None:
        if self.chunk_future is None:
            self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

        self.loaded_chunk = self.chunk_future.result()
        self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

    def __len__(self) -> int:
        return self.loaded_chunk[RGB].shape[0] if self.loaded_chunk is not None else 0

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {}
        for key, value in self.loaded_chunk.items():
            if key != PIXEL_INDEX:
                item[key] = value[idx]

        metadata_item = self.items[item[IMAGE_INDEX]]
        width = metadata_item.W

        # image index, row, col
        item[RAY_INDEX] = torch.LongTensor([
            item[IMAGE_INDEX],
            self.loaded_chunk[PIXEL_INDEX][idx] // width,
            self.loaded_chunk[PIXEL_INDEX][idx] % width])

        return item

    def _load_chunk_inner(self) -> Dict[str, torch.Tensor]:
        loaded_chunk = defaultdict(list)
        loaded = 0

        while loaded < self.items_per_chunk:
            if self.loaded_fields is None or self.loaded_field_offset >= len(self.loaded_fields[IMAGE_INDEX]):
                self.loaded_fields = {}
                self.loaded_field_offset = 0

                to_shuffle = self._load_random_subset()

                shuffled_indices = torch.randperm(len(to_shuffle[IMAGE_INDEX]))
                for key, val in to_shuffle.items():
                    self.loaded_fields[key] = val[shuffled_indices]

            to_add = self.items_per_chunk - loaded
            for key, val in self.loaded_fields.items():
                loaded_chunk[key].append(val[self.loaded_field_offset:self.loaded_field_offset + to_add])

            added = len(self.loaded_fields[IMAGE_INDEX][self.loaded_field_offset:self.loaded_field_offset + to_add])
            loaded += added
            self.loaded_field_offset += added

        loaded_chunk = {k: torch.cat(v) for k, v in loaded_chunk.items()}

        loaded_fields = self._load_fields(loaded_chunk[IMAGE_INDEX], loaded_chunk[PIXEL_INDEX], {RGB}, True)
        for key, val in loaded_fields.items():
            loaded_chunk[key] = val

        return loaded_chunk

    def _load_random_subset(self) -> Dict[str, torch.Tensor]:
        image_indices = torch.randint(0, len(self.items), (self.items_per_chunk,))
        pixel_indices = (
                torch.rand((self.items_per_chunk,)) * self.pixel_indices_to_sample[image_indices]).floor().long()

        mask = self._load_fields(image_indices, pixel_indices, {MASK})[MASK]

        return {
            IMAGE_INDEX: image_indices[mask > 0],
            PIXEL_INDEX: pixel_indices[mask > 0]
        }

    def _load_fields(self, image_indices: torch.Tensor, pixel_indices: torch.Tensor, fields_to_load: Set[str],
                     verbose: bool = False) -> Dict[str, torch.Tensor]:
        assert image_indices.shape == pixel_indices.shape

        sorted_image_indices, ordering = image_indices.sort()
        unique_image_indices, counts = torch.unique_consecutive(sorted_image_indices, return_counts=True)
        load_futures = {}

        offset = 0
        for image_index, image_count in zip(unique_image_indices, counts):
            load_futures[int(image_index)] = self.on_demand_executor.submit(
                self._load_image_fields, image_index, pixel_indices[ordering[offset:offset + image_count]],
                fields_to_load)
            offset += image_count

        loaded = {}
        offset = 0
        for i, (image_index, image_count) in enumerate(zip(unique_image_indices, counts)):
            if i % 1000 == 0 and verbose:
                CONSOLE.log('Loading image {} of {}'.format(i, len(unique_image_indices)))
            loaded_features = load_futures[int(image_index)].result()
            to_put = ordering[offset:offset + image_count]

            for key, value in loaded_features.items():
                if i == 0:
                    loaded[key] = torch.zeros(image_indices.shape[0:1] + value.shape[1:], dtype=value.dtype)
                loaded[key][to_put] = value

            offset += image_count
            del load_futures[int(image_index)]

        return loaded

    def _load_image_fields(self, image_index: int, pixel_indices: torch.Tensor, fields_to_load: Set[str]) -> \
            Dict[str, torch.Tensor]:
        fields = {}

        item = self.items[image_index]
        for field in fields_to_load:
            if field == RGB:
                fields[RGB] = item.load_image().view(-1, 3)[pixel_indices].float() / 255.
            elif field == MASK:
                fields[MASK] = item.load_mask().view(-1)[pixel_indices]
            else:
                raise Exception('Unrecognized field: {}'.format(field))

        return fields
