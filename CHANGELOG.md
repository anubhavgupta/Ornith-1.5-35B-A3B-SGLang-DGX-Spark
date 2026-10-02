# Changelog

All notable changes to this project are documented here. Dates are commit dates.

## 2026-10-02 — initial setup

Bootstrapped from
[`Qwen3.6-35b-SGLang-DGX-Spark`](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark)
(same base recipe and GB10 tuning), retargeted to serve
[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
with the
[`DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4`](https://huggingface.co/DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4)
DFlash2 draft.

**Changed from the Qwen3.6 source repo** (both models share the same
40-layer hybrid-GDN MoE text backbone shape, confirmed from each
checkpoint's `config.json`, so most pool-sizing constants carry over
unchanged):

- `MODEL_ID` default → `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`.
- `DRAFT_MODEL` / `DRAFT_REVISION` → `DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4`
  @ `ec736c35cde2b4021f00ee31ec218195a1ff5937`. This draft has a
  different shape than Qwen3.6's (3 self-attn layers / 4 KV heads /
  MoE draft FFN, vs 6 layers / 8 KV heads / dense FFN), so
  `DF_BLOCK_SIZE` (8 → 7, matching the draft's `block_size`) and
  `DRAFT_KV_BYTES_PER_TOKEN` were recomputed from its `config.json`.
- `WEIGHTS_GIB` (27.4 → 24.2) and `DRAFT_WEIGHTS_GIB` (1.0 → 1.7) are
  estimated from each repo's safetensors shard sizes on disk, **not yet
  measured on-device** for this checkpoint — re-measure on first boot
  (see `start.sh` / `start-dflash.sh` comments and `numbers.md`).
- `SERVED_MODEL_NAME` / `CONTAINER_NAME` → `ornith-1.5-35b-a3b-sglang`.
- Removed the Qwen3.6 repo's `numbers.md` benchmark sweep and
  `CHANGELOG.md` history (not applicable to this checkpoint) and
  replaced `numbers.md` with a placeholder describing the sweep still
  to be run.

**Unchanged:** `start.sh`'s GDN state-pool math, `MAMBA_STATE_BYTES_PER_SLOT`
/ `KV_BYTES_PER_TOKEN` (target-side shape is identical to Qwen3.6-35B-A3B),
default sampling profile, `--reasoning-parser qwen3` /
`--tool-call-parser qwen3_coder` (Ornith's chat template uses the same
`<think>` / `<tool_call>` format).
