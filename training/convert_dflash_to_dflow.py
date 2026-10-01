"""Add DFlow relay parameters to an existing local DFlash checkpoint.

This does not run inference or training.  The relay's final projection is
zero-initialized, so the converted checkpoint is behaviorally identical to
DFlash until the new parameters are trained.
"""

from __future__ import annotations

import os

import torch

from dflash import DFlowDraftModel


# Edit these two local paths before running this conversion script.  It has no
# command line parameters.
DRAFT_PATH = "/srv/models/qwen-dflash-5-layer"
OUTPUT_PATH = "/srv/models/qwen-dflow-5-layer"
REJECT_ROUNDS = 2
MASK_EMBEDDING_SCALE = 1.0


def main() -> None:
    if os.path.exists(OUTPUT_PATH) and os.listdir(OUTPUT_PATH):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT_PATH}")

    model = DFlowDraftModel.from_pretrained(
        DRAFT_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    if len(model.layers) != 5:
        raise ValueError(
            "The DFlow paper uses a five-layer drafter; "
            f"the supplied draft has {len(model.layers)} layers"
        )
    with torch.no_grad():
        torch.nn.init.zeros_(model.relay_mlp[-1].weight)
        torch.nn.init.zeros_(model.relay_mlp[-1].bias)
    dflow_config = dict(getattr(model.config, "dflow_config", {}) or {})
    dflow_config.update(
        reject_rounds=REJECT_ROUNDS,
        mask_embedding_scale=MASK_EMBEDDING_SCALE,
    )
    model.config.dflow_config = dflow_config
    model.config.architectures = ["DFlowDraftModel"]
    model.config.auto_map = {"AutoModel": "dflash.DFlowDraftModel"}
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    print(f"saved DFlow checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
