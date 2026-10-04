# Ornith-1.5-35B-A3B on SGLang for DGX Spark

Ready-to-run scripts to serve **[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)** (ornith-ai ModelOpt NVFP4) with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA DGX Spark (GB10, 128 GB unified memory). Three decode modes are available:

| Script | Decode mode | Best for |
|---|---|---|
| `./start-dflash.sh` | **DFlash2** speculative decoding ([`jzinno/Ornith-1.5-35B-A3B-DFlash2`](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2), 4 draft tokens) | **Recommended to try first.** Fastest per stream at low concurrency, pending validation on this checkpoint |
| `./start-mtp.sh` | MTP speculative decoding (the checkpoint's built-in head, 3 steps) | Context above 262K (YaRN), which DFlash doesn't support |
| `./start.sh` | Plain decoding (no speculation) | Many concurrent streams (dozens to hundreds) |

All three serve an OpenAI-compatible API on port **8888** with the model name **`ornith-1.5-35b-a3b-sglang`**. They share one container name, so only one can run at a time.

> This repo was bootstrapped from [`Qwen3.6-35b-SGLang-DGX-Spark`](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark), retargeted to Ornith-1.5-35B-A3B (same 40-layer hybrid-GDN MoE text backbone shape as Qwen3.6-35B-A3B, confirmed from both checkpoints' `config.json`s). The pool-sizing constants that depend only on that shared shape carry over; the ones that depend on actual weight size or the DFlash2 draft's shape were recomputed from each HF repo's `config.json` / file sizes but **not yet measured on this box** — see `numbers.md` and the comments next to each default in `start.sh` / `start-dflash.sh`.

## Requirements

- DGX Spark / GB10 (aarch64, SM121) with Docker and the NVIDIA container runtime
- About 24 GB of disk for the weights (target ≈21.8 GiB plus the DFlash draft at ≈1.0 GiB), downloaded into `./.cache/huggingface` on first start
- Optional: `export HF_TOKEN=...` in `~/.bashrc` for faster Hub downloads

## Quick start

```bash
cp .env.sample .env          # optional; defaults work without it
./start-dflash.sh            # or ./start.sh / ./start-mtp.sh
./stop.sh                    # stops whichever is running
```

The start script streams the server log (also saved to `.sglang.log`) and returns once the API responds. Test it with:

```bash
curl http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ornith-1.5-35b-a3b-sglang",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

- **Thinking** is on by default; reasoning comes back in `reasoning_content`. `preserve_thinking` is on by default too: the reasoning of earlier assistant turns is kept in the prompt, as the model card recommends for agents (`CHAT_TEMPLATE_KWARGS` in `.env`). Disable it per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- **Tool calling** works out of the box (`qwen3_coder` parser, matching Ornith's `<tool_call><function=...>` template format); just send `tools`.
- **Default sampling** is temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0 (the model card's general-tasks profile). It applies only when a request omits a value; values your client sends always win. Change it with `SAMPLING_*` in `.env`.

## Configuration (`.env` or shell env)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_REQUESTS` | `12` | Concurrent streams, each able to reach the full 262K at once (KV pool capped at 12 × 262144 tokens). The sparkrun recipe uses 24 sharing a smaller pool. |
| `CONTEXT_LENGTH` | `262144` | Max context per stream (`1024..1000000`). Above 262144 needs `YARN=1` (auto at exactly 1M). Not available with DFlash. |
| `MEM_FRACTION_STATIC` | `0.92` | Ceiling on the memory SGLang may reserve (sparkrun recipe uses 0.85; 0.92 leaves ~7–8 GB `MemAvailable`, 0.93 left ~4 GB). See [Memory safety](#memory-safety). |
| `CHUNKED_PREFILL` | `4096` | Prefill chunk size (sparkrun recipe; 8192 measured ~5% faster TTFT on a 33K prompt) |
| `MOE_RUNNER_BACKEND` | `flashinfer_cutlass` | `auto` crashes on NVFP4 MoE on SM121 |
| `LOAD_FORMAT` | `auto` | `fastsafetensors` (recipe) loads in ~18 s vs ~113 s but costs ~21 GB of KV/GDN pool on GB10 |
| `DROP_CACHES` | `1` | Drop the host page cache before launch (recipe's `drop-caches`) |
| `MODEL_ID` | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | Target checkpoint |
| `MAX_TOTAL_TOKENS` | auto | KV cap. Pin mode: `N × (CONTEXT_LENGTH + draft tokens)`; ratio mode (default): uncapped. |
| `MAMBA_POOL_MODE` / `MAMBA_FULL_MEMORY_RATIO` | `fill` / `3.51` | GDN state pool sizing. `fill` caps KV at N × context and turns the rest of the budget into GDN prefix-cache slots (374 at the defaults). `auto` pins N × 4 slots at 262K. `ratio` with 3.51 (sparkrun recipe) keeps ~1000 slots for prefix caching but fits only ~8 full contexts at 0.93. |
| `SAMPLING_TEMPERATURE` / `_TOP_P` / `_TOP_K` / `_MIN_P` / `_REPETITION_PENALTY` | `0.6` / `0.95` / `20` / `0.0` / `1.0` | Server default sampling. Applied by mounting a patched `generation_config.json` into the container; the host cache isn't changed. |
| `CHAT_TEMPLATE_KWARGS` | `{"preserve_thinking": true}` | Server default chat-template kwargs; keys the client sends win |
| `DF_BLOCK_SIZE` | `4` | DFlash draft tokens per step (4 vs 10 at concurrency 1: 10 is ~4% faster on average but ~15% slower on chat/prose) |
| `MTP_STEPS` / `MTP_DRAFT` | `3` / `4` | MTP chain length (carried over from the Qwen3.6 setup; not yet swept for this checkpoint) |
| `EXTRA_ARGS` | — | Extra SGLang flags, appended last |
| `DOCKER_ENV` | — | Extra container env, e.g. `SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824` (needed for spec modes above ~150 streams) |

On every start, the script prints the derived state pool and KV cache sizes, and warns if `MAX_CONCURRENT_REQUESTS × CONTEXT_LENGTH` won't fit in the memory budget.

## Image

All scripts use `lmsysorg/sglang:dev-cu13` (the sparkrun recipe's image), pinned by digest: `lmsysorg/sglang@sha256:035f29e9…` (main `65f759144`, 2026-10-02). With `LOAD_FORMAT=fastsafetensors`, the loader is pip-installed into the container at launch (the image doesn't ship it). It's pulled automatically on first run.

> ⚠️ Do **not** use the older `lmsysorg/sglang:qwen38-27b` image with this checkpoint. Ornith's `lm_head` is also quantized (`W4A16_NVFP4`, per `hf_quant_config.json`); the older image is known to drop quantized `lm_head` scales on load for a related checkpoint, producing garbage output. Validate on first boot.

## Benchmark results (GB10)

Not yet run for this checkpoint — see [`numbers.md`](numbers.md) for the sweep methodology carried over from the Qwen3.6 setup this repo is based on, and what needs re-measuring (`WEIGHTS_GIB`, `DRAFT_WEIGHTS_GIB`, GDN pool constants, and throughput/capacity numbers are all checkpoint-specific).

## Memory safety

GB10's GPU and OS share one memory pool, and **GPU allocations can't be swapped**. The Qwen3.6 setup this repo is based on hard-froze the box twice at `MEM_FRACTION_STATIC=0.95` with spec modes at their maximum concurrency — assume the same risk applies here until this checkpoint has its own capacity sweep.

- The defaults (`MEM_FRACTION_STATIC=0.92`, 12 full-context streams) are above the 0.80 ceiling the Qwen3.6 setup used. Boot leaves ~7–8 GB `MemAvailable` (0.93 left only ~4 GB); drop to 0.80 if the box gets tight.
- **Before high-concurrency experiments**, run a watchdog that does `docker kill` when `MemAvailable` drops below ~3 GB.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Main launcher (no spec); holds all the pool-sizing logic |
| `start-dflash.sh` / `start-mtp.sh` | Thin wrappers that add the speculative-decoding flags, then call `start.sh` |
| `stop.sh` | Stops the container (idempotent) |
| `.env.sample` | All settings, documented |
| `.env.dflash` | Tuned DFlash config, auto-loaded by `start-dflash.sh` after `.env` (12 × 262K, 0.92, `fill`) |
| `numbers.md` | Measurement methodology and what's been carried over vs. still needs re-measuring for this checkpoint |

## Links

- [ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) · [DFlash2 draft](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2) · [SGLang docs](https://docs.sglang.io)

## Credits

- **[anubhavgupta / Qwen3.6-35b-SGLang-DGX-Spark](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark):** the base recipe this repo is built on (launch scripts, GDN state-pool sizing, DFlash/MTP wrappers, GB10 tuning).
- **[MiaAI-Lab / Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark):** the original base recipe that setup built on.
- **[SGLang](https://github.com/sgl-project/sglang):** the serving engine, including DFlash2, MTP/EAGLE, the hybrid GDN radix cache and the quantized `lm_head` support this setup depends on.
- **Model owners:**
  - [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) for Ornith-1.5-35B-A3B and its NVFP4 quantization
  - [jzinno](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2) for the DFlash2 draft model
