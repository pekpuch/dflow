# DFlow workspace

This workspace contains the DFlow implementation described in
[arXiv:2609.06498](https://arxiv.org/abs/2609.06498), based on the DFlash
implementation from `z-lab/dflash`.

`vllm-workspace/vllm` is pinned to `v0.25.0`.  There is no upstream
`v0.25.0` tag in `vllm-ascend`; the closest matching development branch is
`releases/v0.25.1rc`, which is checked out in `vllm-workspace/vllm-ascend`.
The two repositories must be installed from the same Python environment on a
Linux Ascend host with a compatible CANN/torch-npu stack.

The Ascend checkout contains compatibility gates for its `v0.25.1rc` API
surface, so the exact vLLM 0.25.0 + Ascend combination still needs validation
on the target NPU image.  The DFlow changes are isolated to the speculative
decoding path; if the image exposes only the 0.25.1rc API, pin vLLM to the
matching upstream revision before serving.

The training reference follows the paper's reported setup: a five-layer draft,
block size 16, 80K samples, six epochs, learning rate `6e-4`, and two
reject-conditioned rounds.  The trainer is local-only: it expects a local
target model, an existing local DFlash checkpoint, and a local DeepSpec target
cache.  It does not download a dataset or model.  On the Ascend host, install the three local
packages into one environment:

```bash
pip install -e vllm-workspace/vllm
pip install -e vllm-workspace/vllm-ascend
pip install -e vllm-workspace/dflash
```

If the existing draft is plain DFlash, edit `DRAFT_PATH` and `OUTPUT_PATH` at
the top of `training/convert_dflash_to_dflow.py`, then run the script without
arguments.  The last relay projection is zero-initialized, so the converted
checkpoint starts with the original DFlash behavior:

```bash
python training/convert_dflash_to_dflow.py
```

Next edit the four paths in `TRAIN_CONFIG` at the top of
`training/train_dflow.py` and run it without arguments:

```bash
python training/train_dflow.py
```

The trainer always uses the paper's target-verification rollout: ordinary
target context states come from the DeepSpec cache, while each proposal block
is verified by the frozen local target to build the rejected-suffix relay.
There is no cache-only approximation mode.

The resulting checkpoint declares `DFlowDraftModel`.  With the modified vLLM
and Ascend plugin it can be selected explicitly:

```bash
vllm serve Qwen/Qwen3-4B \
  --speculative-config '{"method":"dflow","model":"./checkpoints/Qwen3-4B-DFlow","num_speculative_tokens":16}'
```

This command follows the DFlow paper's rollout and loss structure while using
the supplied local regeneration cache.  The paper itself used 80K
`ShareGPT_Vicuna_unfiltered` samples and trained a five-layer DFlash drafter;
the Qwen3.5 target and DeepSpec cache are a local adaptation, not the exact
paper experiment.

The runtime relay keeps verifier hidden states from the rejected suffix,
projects them with the trained DFlow relay module, shifts them to the matching
prediction positions, and applies the target-correction/rejected-token
boundary conditioning from equations (4)-(5) of the paper.
