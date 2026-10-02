# Ornith-1.5-35B-A3B on SGLang for DGX Spark

Ready-to-run scripts to serve **[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)** (ornith-ai ModelOpt NVFP4) with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA DGX Spark (GB10, 128 GB unified memory). Three decode modes are available:

| Script | Decode mode | Best for |
|---|---|---|
| `./start-dflash.sh` | **DFlash2** speculative decoding ([`DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4`](https://huggingface.co/DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4), block 7) | **Recommended to try first.** Fastest per stream at low concurrency, pending validation on this checkpoint |
| `./start-mtp.sh` | MTP speculative decoding (the checkpoint's built-in head, 3 steps) | Context above 262K (YaRN), which DFlash doesn't support |
| `./start.sh` | Plain decoding (no speculation) | Many concurrent streams (dozens to hundreds) |

All three serve an OpenAI-compatible API on port **8888** with the model name **`ornith-1.5-35b-a3b-sglang`**. They share one container name, so only one can run at a time.

> This repo was bootstrapped from [`Qwen3.6-35b-SGLang-DGX-Spark`](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark), retargeted to Ornith-1.5-35B-A3B (same 40-layer hybrid-GDN MoE text backbone shape as Qwen3.6-35B-A3B, confirmed from both checkpoints' `config.json`s). The pool-sizing constants that depend only on that shared shape carry over; the ones that depend on actual weight size or the DFlash2 draft's shape were recomputed from each HF repo's `config.json` / file sizes but **not yet measured on this box** — see `numbers.md` and the comments next to each default in `start.sh` / `start-dflash.sh`.

## Requirements

- DGX Spark / GB10 (aarch64, SM121) with Docker and the NVIDIA container runtime
- About 24 GB of disk for the weights (target ≈21.8 GiB plus the DFlash draft at ≈1.7 GiB), downloaded into `./.cache/huggingface` on first start
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
| `MAX_CONCURRENT_REQUESTS` | `2` | Concurrent streams. The script sizes the GDN state pool and the KV cache cap from this. |
| `CONTEXT_LENGTH` | `262144` | Max context per stream (`1024..1000000`). Above 262144 needs `YARN=1` (auto at exactly 1M). Not available with DFlash. |
| `MEM_FRACTION_STATIC` | `0.5` | Ceiling on the memory SGLang may reserve. **Keep ≤ 0.80** (see [Memory safety](#memory-safety)). |
| `MODEL_ID` | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | Target checkpoint |
| `MAX_TOTAL_TOKENS` | auto | KV cap. Default `N × (CONTEXT_LENGTH + draft tokens)`; `0` = use everything the fraction allows. |
| `MAMBA_POOL_MODE` | `auto` | GDN state pool sizing: `pin` (N × 4 slots) or `ratio` (computed `--mamba-full-memory-ratio`) |
| `SAMPLING_TEMPERATURE` / `_TOP_P` / `_TOP_K` / `_MIN_P` / `_REPETITION_PENALTY` | `0.6` / `0.95` / `20` / `0.0` / `1.0` | Server default sampling. Applied by mounting a patched `generation_config.json` into the container; the host cache isn't changed. |
| `CHAT_TEMPLATE_KWARGS` | `{"preserve_thinking": true}` | Server default chat-template kwargs; keys the client sends win |
| `DF_BLOCK_SIZE` | `7` | DFlash draft tokens per step (matches the draft's trained `block_size`; not yet swept on this box) |
| `MTP_STEPS` / `MTP_DRAFT` | `3` / `4` | MTP chain length (carried over from the Qwen3.6 setup; not yet swept for this checkpoint) |
| `EXTRA_ARGS` | — | Extra SGLang flags, appended last |
| `DOCKER_ENV` | — | Extra container env, e.g. `SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824` (needed for spec modes above ~150 streams) |

On every start, the script prints the derived state pool and KV cache sizes, and warns if `MAX_CONCURRENT_REQUESTS × CONTEXT_LENGTH` won't fit in the memory budget.

## Image

All scripts use the official SGLang nightly, pinned by digest: `lmsysorg/sglang@sha256:00205b89…` (= `nightly-cu134-20260909-708f51e`). It's pulled automatically on first run.

> ⚠️ Do **not** use the older `lmsysorg/sglang:qwen38-27b` image with this checkpoint. Ornith's `lm_head` is also quantized (`W4A16_NVFP4`, per `hf_quant_config.json`); the older image is known to drop quantized `lm_head` scales on load for a related checkpoint, producing garbage output. Validate on first boot.

## Benchmark results (GB10)

Not yet run for this checkpoint — see [`numbers.md`](numbers.md) for the sweep methodology carried over from the Qwen3.6 setup this repo is based on, and what needs re-measuring (`WEIGHTS_GIB`, `DRAFT_WEIGHTS_GIB`, GDN pool constants, and throughput/capacity numbers are all checkpoint-specific).

## Memory safety

GB10's GPU and OS share one memory pool, and **GPU allocations can't be swapped**. The Qwen3.6 setup this repo is based on hard-froze the box twice at `MEM_FRACTION_STATIC=0.95` with spec modes at their maximum concurrency — assume the same risk applies here until this checkpoint has its own capacity sweep.

- **Keep `MEM_FRACTION_STATIC` ≤ 0.80**, and set `MAX_CONCURRENT_REQUESTS` conservatively until you've swept max safe concurrency for this checkpoint.
- **Before high-concurrency experiments**, run a watchdog that does `docker kill` when `MemAvailable` drops below ~3 GB.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Main launcher (no spec); holds all the pool-sizing logic |
| `start-dflash.sh` / `start-mtp.sh` | Thin wrappers that add the speculative-decoding flags, then call `start.sh` |
| `stop.sh` | Stops the container (idempotent) |
| `.env.sample` | All settings, documented |
| `numbers.md` | Measurement methodology and what's been carried over vs. still needs re-measuring for this checkpoint |

## Links

- [ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) · [DFlash2 draft](https://huggingface.co/DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4) · [SGLang docs](https://docs.sglang.io)

## Credits

- **[anubhavgupta / Qwen3.6-35b-SGLang-DGX-Spark](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark):** the base recipe this repo is built on (launch scripts, GDN state-pool sizing, DFlash/MTP wrappers, GB10 tuning).
- **[MiaAI-Lab / Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark):** the original base recipe that setup built on.
- **[SGLang](https://github.com/sgl-project/sglang):** the serving engine, including DFlash2, MTP/EAGLE, the hybrid GDN radix cache and the quantized `lm_head` support this setup depends on.
- **Model owners:**
  - [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) for Ornith-1.5-35B-A3B and its NVFP4 quantization
  - [DaoCloud](https://huggingface.co/DaoCloud/Ornith-1.5-35B-A3B-DFlash2-2.6B-A0.3B-NVFP4) for the DFlash2 draft model
