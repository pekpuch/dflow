"""Small, dependency-light reader for DeepSpec target-cache v2."""

from __future__ import annotations

import json
import mmap
import os
import struct
from collections import OrderedDict

import numpy as np
import torch


INDEX = struct.Struct("<QIIQQQQQ")


class DeepSpecCache(torch.utils.data.Dataset):
    """Read DeepSpec's mmap cache without importing the DeepSpec repository."""

    def __init__(self, path: str, max_open_shards: int = 4):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        if int(self.manifest.get("version", -1)) != 2:
            raise ValueError("Only DeepSpec target-cache version 2 is supported")
        required = {
            "num_samples", "num_shards", "target_layer_ids", "hidden_dtype",
            "token_dtype", "mask_dtype", "index_record_size", "hidden_size",
            "shards",
        }
        missing = sorted(required - set(self.manifest))
        if missing:
            raise ValueError(f"Cache manifest is missing fields: {missing}")
        if self.manifest["hidden_dtype"] != "bfloat16":
            raise ValueError("Only bfloat16 target hidden states are supported")
        if self.manifest["token_dtype"] != "int32":
            raise ValueError("Only int32 cache token ids are supported")
        if self.manifest["mask_dtype"] != "uint8":
            raise ValueError("Only uint8 cache loss masks are supported")
        if int(self.manifest["index_record_size"]) != INDEX.size:
            raise ValueError("Cache index record size does not match DeepSpec v2")
        if int(self.manifest["num_shards"]) != len(self.manifest["shards"]):
            raise ValueError("Cache num_shards does not match shard metadata")
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.target_layer_ids = [int(x) for x in self.manifest["target_layer_ids"]]
        self.index_file = None
        self.index_map = None
        self.max_open_shards = max_open_shards
        self.shards: OrderedDict[int, tuple[object, mmap.mmap]] = OrderedDict()
        self.shard_paths = {
            int(x["shard_id"]): os.path.join(self.path, x["file_name"])
            for x in self.manifest["shards"]
        }
        expected_shards = list(range(len(self.shard_paths)))
        if sorted(self.shard_paths) != expected_shards:
            raise ValueError("Cache shard ids must be contiguous starting at zero")

    def __len__(self):
        return self.num_samples

    def validate_target_path(self, target_path: str):
        recorded = self.manifest.get("target_model_name_or_path")
        if recorded is None:
            return
        recorded_path = os.path.normcase(os.path.abspath(str(recorded)))
        actual_path = os.path.normcase(os.path.abspath(str(target_path)))
        if recorded_path != actual_path:
            raise ValueError(
                "DeepSpec cache was created for a different target path: "
                f"cache={recorded!r}, target={target_path!r}"
            )

    def close(self):
        for handle, mapping in self.shards.values():
            mapping.close()
            handle.close()
        self.shards.clear()
        if self.index_map is not None:
            self.index_map.close()
            self.index_map = None
        if self.index_file is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _index(self):
        if self.index_map is None:
            self.index_file = open(os.path.join(self.path, "samples.idx"), "rb")
            self.index_map = mmap.mmap(self.index_file.fileno(), 0, access=mmap.ACCESS_READ)
        return self.index_map

    def _shard(self, shard_id: int):
        if shard_id in self.shards:
            self.shards.move_to_end(shard_id)
            return self.shards[shard_id][1]
        handle = open(self.shard_paths[shard_id], "rb")
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.shards[shard_id] = (handle, mapping)
        while len(self.shards) > self.max_open_shards:
            _, (old_handle, old_mapping) = self.shards.popitem(last=False)
            old_mapping.close()
            old_handle.close()
        return mapping

    @staticmethod
    def _read(mapping, offset, count, dtype):
        arr = np.frombuffer(mapping, dtype=dtype, count=count, offset=int(offset)).copy()
        return torch.from_numpy(arr)

    def __getitem__(self, index: int):
        if not 0 <= int(index) < self.num_samples:
            raise IndexError(index)
        record = INDEX.unpack_from(self._index(), int(index) * INDEX.size)
        sample_id, shard_id, seq_len, ids_off, _attn_off, mask_off, hidden_off, _last_off = record
        if int(sample_id) != int(index):
            raise ValueError(f"Cache index is not dense at {index}: {sample_id}")
        seq_len = int(seq_len)
        mapping = self._shard(int(shard_id))
        input_ids = self._read(mapping, ids_off, seq_len, np.int32).to(torch.long)
        loss_mask = self._read(mapping, mask_off, seq_len, np.uint8).to(torch.bool)
        hidden_count = seq_len * len(self.target_layer_ids) * self.hidden_size
        raw = self._read(mapping, hidden_off, hidden_count, np.uint16)
        target_hidden = raw.view(torch.bfloat16).view(seq_len, -1)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden,
        }
