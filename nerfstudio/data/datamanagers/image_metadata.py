import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: str, W: int, H: int, mask_path: Optional[str], local_cache: Optional[Path]):
        self.image_path = image_path
        self.W = W
        self.H = H
        self.mask_path = mask_path
        self._local_cache = local_cache

    def load_image(self) -> torch.Tensor:
        if self._local_cache is not None and not self.image_path.startswith(str(self._local_cache)):
            self.image_path = self._load_from_cache(self.image_path)

        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> torch.Tensor:
        if self.mask_path is None:
            return torch.ones(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not self.mask_path.startswith(str(self._local_cache)):
            self.mask_path = self._load_from_cache(self.mask_path)

        mask = Image.open(self.mask_path)
        size = mask.size

        if size[0] != self.W or size[1] != self.H:
            mask = mask.resize((self.W, self.H), Image.NEAREST)

        return torch.BoolTensor(np.asarray(mask))

    def _load_from_cache(self, remote_path: str) -> str:
        sha_hash = hashlib.sha256()
        sha_hash.update(remote_path.encode('utf-8'))
        hashed = sha_hash.hexdigest()
        cache_path = self._local_cache / hashed[:2] / hashed[2:4] / '{}{}'.format(hashed, Path(remote_path).suffix)

        if cache_path.exists():
            return str(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = '{}.{}'.format(cache_path, uuid.uuid4())
        shutil.copy(remote_path, tmp_path)

        os.rename(tmp_path, cache_path)
        return str(cache_path)
