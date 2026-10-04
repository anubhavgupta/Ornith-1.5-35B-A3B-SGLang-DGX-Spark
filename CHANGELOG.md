# Changelog

All notable changes to this project are documented here. Dates are commit dates.

## 2026-10-04 — `.env.mtp`, MTP benchmarks, fill-mode guard

- New tracked `.env.mtp`: 26 × 262K full-context streams at 0.92 in `fill`
  mode with the checkpoint's own MTP head (3 steps, 4 draft tokens).
  `start-mtp.sh` loads it after `.env`. Boot: 6,815,848-token KV pool,
  171 GDN slots, ~5.2 GB `MemAvailable`.
- README: MTP columns in both benchmark tables. MTP at 26 streams gives the
  highest total throughput (613 tok/s code, 497 tok/s mixed).
- `start.sh` `fill` mode now refuses to start when N full contexts don't fit
  the budget, and prints the max that does. It used to clamp GDN slots to
  the pin floor and over-allocate (MTP at 28 left 1.8 GB `MemAvailable`).

## 2026-10-04 — Concurrency benchmarks, env comments

- README: concurrency benchmark table. No-spec (1–30 streams) peaks at
  470 tok/s total at 30; DFlash (1–12 streams) gives 93 tok/s single-stream
  and 346 tok/s total at 12.
- `.env.dflash` and `.env.no-spec`: every flag now has a comment explaining
  what it does. Values unchanged.
- README: benchmark section now states the mixed prose/code/math prompt set,
  and adds a code-only table. Code-only, DFlash reaches 132 tok/s
  single-stream (accept length ~3.3) and 456 tok/s at 12; no-spec 525 tok/s
  at 30.

## 2026-10-04 — `.env.no-spec`

- New tracked `.env.no-spec`: 30 × 262K full-context streams at 0.92 in
  `fill` mode without speculative decoding. `start.sh` loads it after `.env`
  only when run directly (no spec wrapper). Boot: 7,864,320-token KV pool,
  165 GDN slots, ~9 GB `MemAvailable`.

## 2026-10-04 — `.env.dflash`

- New tracked `.env.dflash` with the tuned DFlash config (12 × 262K,
  0.92, `fill`, 4 draft tokens). `start-dflash.sh` loads it automatically
  after `.env`; shell env and `.env` still win.

## 2026-10-04 — 12 full-context streams at 0.92, `fill` pool mode

- `MEM_FRACTION_STATIC` 0.85 → 0.92, `MAX_CONCURRENT_REQUESTS` 24 → 12,
  `MAMBA_POOL_MODE` ratio → new `fill`: KV capped at 12 × 262144 and the
  rest of the budget becomes GDN prefix-cache slots (`POOL_OVERHEAD_GIB`
  7.1, measured; `FILL_MARGIN_GIB` 0.5). Boot: 3,145,776-token KV pool,
  374 GDN slots, ~7–8 GB `MemAvailable` (auto/pin at 0.90: 48 slots,
  ~18 GB free). Measured alternatives at 0.93: auto 13.6 full contexts
  (~4 GB free), ratio 3.51 8.5 full contexts with 1,032 GDN slots.
- The DFlash draft window stays off by default: with
  `--speculative-draft-window-size 4096` this image still allocates the
  full draft KV pool (25.5 GB for 2.23M tokens), so it saved no memory.
- Pool estimate fix: start.sh no longer counts draft KV as 0 when
  `DF_DRAFT_WINDOW` is set; the printed KV pool now matches SGLang.

## 2026-10-03 — adopt sparkrun recipe defaults

- Image → `lmsysorg/sglang:dev-cu13` @ `sha256:035f29e9…` (main `65f759144`).
- `MAX_CONCURRENT_REQUESTS` 2 → 24, `MEM_FRACTION_STATIC` 0.5 → 0.85,
  `CHUNKED_PREFILL` 8192 → 4096, `MAMBA_POOL_MODE` auto → ratio with
  `MAMBA_FULL_MEMORY_RATIO=3.51`; `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
  always set; `DROP_CACHES=1` drops the page cache before launch.
  Boot: 1.94M-token KV pool, 907 GDN slots, ~13 GB `MemAvailable`.
- New `LOAD_FORMAT` (default `auto`). The recipe's `fastsafetensors`
  (pip-installed at launch) loads in 18 s vs 113 s but measured 45 GB vs
  24.4 GB for the target, shrinking the KV pool to 1.27M tokens, so it is
  opt-in.
- Concurrency-1 decode unchanged on the new image (112.7 tok/s, DFlash 4).

## 2026-10-03 — switch DFlash2 draft

- `DRAFT_MODEL` / `DRAFT_REVISION` →
  [`jzinno/Ornith-1.5-35B-A3B-DFlash2`](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2)
  @ `9b4852c05fd00b672b7434b1bb105bc03c8682b0` (BF16, 6 layers / 8 KV
  heads / dense FFN, 4096 sliding window).
- `DF_BLOCK_SIZE` 7 → 4 (sparkrun recipe; benchmarked vs 10 at concurrency 1),
  `DRAFT_KV_BYTES_PER_TOKEN` 3 KB → 12 KB (fp8), `DRAFT_WEIGHTS_GIB`
  1.7 → 1.0 (measured: 0.97 GB draft, 24.27 GB target at load).
- `start.sh` now passes `--moe-runner-backend ${MOE_RUNNER_BACKEND}`
  (default `flashinfer_cutlass`): `auto` picked `flashinfer_trtllm` on
  SM121, which crashes on NVFP4 MoE.

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
