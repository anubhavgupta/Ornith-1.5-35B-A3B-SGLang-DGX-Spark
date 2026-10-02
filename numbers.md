# Numbers: Ornith-1.5-35B-A3B NVFP4 on SGLang, DGX Spark (GB10)

No throughput/capacity sweep has been run on this box for this checkpoint
yet. This repo was bootstrapped from
[`Qwen3.6-35b-SGLang-DGX-Spark`](https://github.com/anubhavgupta/Qwen3.6-35b-SGLang-DGX-Spark)
(same text-backbone shape: 40 layers = 30 linear-attn/GDN + 10
full-attention, hidden_size 2048), so the GDN state-pool and target KV
byte constants (`MAMBA_STATE_BYTES_PER_SLOT`, `KV_BYTES_PER_TOKEN` in
`start.sh`) are carried over and should hold. Everything that depends on
*this* checkpoint's actual weight size or the DFlash2 draft's actual
shape (`WEIGHTS_GIB`, `DRAFT_WEIGHTS_GIB`, `DRAFT_KV_BYTES_PER_TOKEN`,
`DF_BLOCK_SIZE`) was computed from the HF repos' `config.json` /
safetensors file sizes, not measured on-device — see the comments next
to each default in `start.sh` / `start-dflash.sh`.

To fill this file in, run the same sweep methodology as the Qwen3.6
repo this is based on:

1. Boot each mode (`start.sh`, `start-mtp.sh`, `start-dflash.sh`) and
   capture the "Load weight end ... mem usage=" and "Mamba Cache is
   allocated" / "KV Cache is allocated" lines from `.sglang.log` to
   confirm/replace `WEIGHTS_GIB`, `MAMBA_STATE_BYTES_PER_SLOT`,
   `KV_BYTES_PER_TOKEN`.
2. Benchmark single-stream and low-concurrency output tok/s per prompt
   category (reasoning / chat / code / essay) for each mode.
3. Sweep `DF_BLOCK_SIZE` (and `DF_DRAFT_ATTN=fa4` if available) to find
   the best DFlash block size for this draft.
4. Sweep `MAX_CONCURRENT_REQUESTS` at `MEM_FRACTION_STATIC=0.95` per
   mode to find max safe concurrency (expect ~90% of the boot-tested
   maximum, per the GB10 memory-safety notes below).
5. Record the best launch command in `best-config.txt` once found.

## Memory safety (carried over from the Qwen3.6 setup — unverified for this checkpoint)

GB10's GPU and OS share one memory pool, and GPU allocations can't be
swapped. The Qwen3.6 setup this repo is based on hard-froze the box twice
at `MEM_FRACTION_STATIC=0.95` with spec modes at their maximum
concurrency. Until this checkpoint has its own capacity sweep:

- Keep `MEM_FRACTION_STATIC` ≤ 0.80.
- Run a watchdog that `docker kill`s the container when `MemAvailable`
  drops below ~3 GB before any high-concurrency experiment.
