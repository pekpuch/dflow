#!/usr/bin/env python3
"""Train DFlow from local Qwen target, DFlash draft and DeepSpec cache.

The default ``target`` verifier mode follows DFlow (arXiv:2609.06498): the
draft is trained for an initial masked round and two reject-conditioned rounds;
the target verifies proposals, and the first mismatch determines the next
anchor. Target hidden states for the draft come from the local DeepSpec cache.

This is a single-process PyTorch reference trainer. It intentionally does not
download data or models; run it on the server/NPU only.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "vllm-workspace" / "dflash"))

from dflash import DFlowDraftModel  # noqa: E402
from dflash.model import extract_context_feature  # noqa: E402
from deepspec_cache import DeepSpecCache  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    target_path: str
    draft_path: str
    cache_path: str
    output_path: str
    device: str = "npu"
    dtype: str = "bfloat16"
    samples: int = 80_000
    epochs: int = 6
    learning_rate: float = 6e-4
    block_size: int = 16
    reject_rounds: int = 2
    loss_decay: float = 0.9
    mask_embedding_scale: float = 1.0
    max_open_shards: int = 4
    seed: int = 0


# Edit these local paths before starting the server run.  There are no command
# line parameters: all training defaults below match the DFlow paper.
TRAIN_CONFIG = TrainConfig(
    target_path="/srv/models/Qwen3.5-9B",
    draft_path="/srv/models/qwen-dflow-5-layer",
    cache_path="/srv/cache/qwen3.5-9b-regeneration",
    output_path="/srv/models/qwen-dflow-5-layer-trained",
)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def _device(name: str) -> torch.device:
    if name == "npu":
        if not hasattr(torch, "npu"):
            raise RuntimeError("This PyTorch build has no torch.npu; run on the NPU server")
        return torch.device("npu")
    return torch.device(name)


def _selected_hidden(outputs, layer_ids: list[int]) -> torch.Tensor:
    # Transformers exposes the embedding output at hidden_states[0].
    return extract_context_feature(list(outputs.hidden_states), layer_ids)


def _weighted_ce(logits, labels, loss_mask, decay):
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1), reduction="none",
    ).view_as(labels)
    weights = torch.pow(
        torch.tensor(decay, device=logits.device, dtype=token_loss.dtype),
        torch.arange(labels.shape[1], device=logits.device, dtype=token_loss.dtype),
    ).unsqueeze(0) * loss_mask.to(token_loss.dtype)
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def _target_hidden(target, ids, layer_ids):
    with torch.no_grad():
        outputs = target(input_ids=ids, output_hidden_states=True, use_cache=False)
    return _selected_hidden(outputs, layer_ids).detach()


def _output_head(target):
    head = target.get_output_embeddings()
    if head is None:
        raise ValueError("The target model has no output embedding/head")
    return head


def _mask_id(draft, tokenizer):
    value = getattr(draft, "mask_token_id", None)
    if value is None:
        value = getattr(tokenizer, "mask_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("Cannot determine mask token id from draft or tokenizer")
    return int(value)


def _draft_value(draft, name, default):
    return getattr(draft.config, "dflash_config", {}).get(
        name, getattr(draft.config, name, default)
    )


def _load_local(cfg, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.target_path, local_files_only=True, trust_remote_code=True,
    )
    target_kwargs = dict(
        local_files_only=True, trust_remote_code=True, torch_dtype=dtype,
    )
    try:
        target = AutoModelForCausalLM.from_pretrained(cfg.target_path, **target_kwargs)
    except ValueError:
        # Some Qwen3.5 local snapshots register as multimodal even when the
        # training cache contains text-only token sequences.
        from transformers import AutoModelForImageTextToText
        target = AutoModelForImageTextToText.from_pretrained(
            cfg.target_path, **target_kwargs,
        )
    target = target.to(device=device, dtype=dtype).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    draft = DFlowDraftModel.from_pretrained(
        cfg.draft_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype).train()
    return tokenizer, target, draft


def _validate_layout(draft, cache, block_size):
    if len(draft.layers) != 5:
        raise ValueError(
            "The DFlow paper uses a five-layer drafter; "
            f"the supplied draft has {len(draft.layers)} layers"
        )
    draft_layers = list(map(int, draft.target_layer_ids))
    if draft_layers != cache.target_layer_ids:
        raise ValueError(
            "Draft/cache target_layer_ids differ: "
            f"draft={draft_layers}, cache={cache.target_layer_ids}"
        )
    expected = len(cache.target_layer_ids) * cache.hidden_size
    if draft.fc.in_features != expected:
        raise ValueError(
            f"Draft fc expects {draft.fc.in_features} features, cache provides {expected}"
        )
    actual_block = int(draft.block_size if block_size is None else block_size)
    if actual_block <= 0:
        raise ValueError("block_size must be positive")
    return actual_block


def _choose_anchor(input_ids, loss_mask, block_size, rng):
    max_start = input_ids.numel() - block_size
    if max_start < 1:
        return None
    candidates = [
        start for start in range(1, max_start + 1)
        if bool(loss_mask[start:start + block_size].any())
    ]
    return rng.choice(candidates) if candidates else None


def _make_relay(draft, target, hidden_source, proposals, target_tokens,
                start, mismatch, next_anchor, block_size, device):
    # The hidden state at verifier position p predicts p+1.  If p is the
    # first rejected proposal, it is relayed to position 1 of the next block.
    source_start = next_anchor
    count = min(block_size, hidden_source.shape[0] - source_start)
    if count <= 0:
        return None
    verifier = hidden_source[source_start:source_start + count].unsqueeze(0)
    verifier = draft.hidden_norm(draft.fc(verifier)).detach()
    positions = torch.arange(1, count + 1, device=device)
    embedding = target.get_input_embeddings()
    correction = embedding(target_tokens[:, mismatch:mismatch + 1])
    rejected = embedding(proposals[:, mismatch:mismatch + 1])
    return {
        "positions": positions,
        "verifier_hidden": verifier[:, :count],
        "target_correction": correction.expand(-1, count, -1).detach(),
        "rejected_token": rejected.expand(-1, count, -1).detach(),
    }


def train() -> None:
    cfg = TRAIN_CONFIG
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = _device(cfg.device)
    dtype = _dtype(cfg.dtype)

    tokenizer, target, draft = _load_local(cfg, dtype, device)
    draft.mask_embedding_scale = float(cfg.mask_embedding_scale)
    cache = DeepSpecCache(cfg.cache_path, max_open_shards=cfg.max_open_shards)
    cache.validate_target_path(cfg.target_path)
    block_size = _validate_layout(draft, cache, cfg.block_size)
    mask_id = _mask_id(draft, tokenizer)
    layer_ids = cache.target_layer_ids
    output_head = _output_head(target)
    target_embedding = target.get_input_embeddings()
    input_embedding_scale = float(_draft_value(draft, "input_embedding_scale", 1.0))
    optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.learning_rate)
    rng = random.Random(cfg.seed)

    seen = 0
    step = 0
    try:
        for epoch in range(cfg.epochs):
            order = list(range(len(cache)))
            rng.shuffle(order)
            samples_this_epoch = order[:min(cfg.samples, len(order))]
            for sample_id in samples_this_epoch:
                sample = cache[sample_id]
                input_ids = sample["input_ids"].to(device)
                loss_mask = sample["loss_mask"].to(device)
                cached_hidden = sample["target_hidden_states"].to(device=device, dtype=dtype)
                anchor = _choose_anchor(input_ids, loss_mask, block_size, rng)
                if anchor is None:
                    continue

                losses = []
                relay = None
                for round_id in range(cfg.reject_rounds + 1):
                    # DFlash consumes the committed anchor token followed by
                    # B-1 masks and predicts only the B-1 following tokens.
                    end = min(anchor + block_size, input_ids.numel())
                    labels = input_ids[anchor + 1:end].unsqueeze(0)
                    round_mask = loss_mask[anchor + 1:end].unsqueeze(0)
                    if labels.shape[1] == 0:
                        break
                    context_hidden = cached_hidden[:anchor].unsqueeze(0)
                    query_ids = torch.cat(
                        (input_ids[anchor:anchor + 1],
                         torch.full_like(labels, mask_id)), dim=0,
                    ).unsqueeze(0)
                    noise = (target_embedding(query_ids) * input_embedding_scale).detach()
                    positions = torch.arange(0, end, device=device).unsqueeze(0)
                    hidden = draft.forward_dflow(
                        position_ids=positions,
                        target_hidden=context_hidden,
                        mask_embedding=noise,
                        relay=relay,
                    )
                    # Position zero is the committed anchor; it is an input,
                    # not a prediction target.
                    logits = draft.compute_logits(hidden[:, 1:], output_head)
                    losses.append(_weighted_ce(logits, labels, round_mask, cfg.loss_decay))
                    proposals = logits.detach().argmax(dim=-1)
                    if round_id == cfg.reject_rounds:
                        break

                    matches = proposals[0].eq(labels[0])
                    mismatch_indices = (~matches).nonzero(as_tuple=False)
                    # An all-matching block has no rejected suffix to relay.
                    if not mismatch_indices.numel():
                        break
                    mismatch = int(mismatch_indices[0].item())
                    next_anchor = anchor + 1 + mismatch
                    verify_ids = torch.cat(
                        (input_ids[:anchor + 1], proposals[0]), dim=0,
                    ).unsqueeze(0)
                    hidden_source = _target_hidden(target, verify_ids, layer_ids)[0]
                    relay = _make_relay(
                        draft, target, hidden_source, proposals, labels,
                        anchor, mismatch, next_anchor, block_size, device,
                    )
                    if relay is None:
                        break
                    anchor = next_anchor

                total_loss = torch.stack(losses).sum()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                seen += 1
                step += 1
                if step % 100 == 0:
                    print(
                        f"epoch={epoch} step={step} samples={seen} "
                        f"loss={float(total_loss.detach()):.5f}", flush=True,
                    )
    finally:
        cache.close()

    dflow_config = dict(getattr(draft.config, "dflow_config", {}) or {})
    dflow_config.update(
        block_size=block_size,
        reject_rounds=cfg.reject_rounds,
        loss_decay=cfg.loss_decay,
        mask_token_id=mask_id,
        mask_embedding_scale=cfg.mask_embedding_scale,
    )
    draft.config.dflow_config = dflow_config
    draft.config.architectures = ["DFlowDraftModel"]
    draft.config.auto_map = {"AutoModel": "dflash.DFlowDraftModel"}
    os.makedirs(cfg.output_path, exist_ok=True)
    draft.save_pretrained(cfg.output_path)
    tokenizer.save_pretrained(cfg.output_path)
    print(f"saved DFlow checkpoint to {cfg.output_path}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
