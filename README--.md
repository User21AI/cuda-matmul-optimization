# CUDA SGEMM from scratch

Step-by-step optimization of single-precision matrix multiplication (`C = A * B`, row-major, FP32) in CUDA, from a naive kernel to one that reaches roughly 97-103% of cuBLAS on large matrices. Every stage is checked for correctness against cuBLAS and timed in the same harness.

**Hardware:** Google Colab, GPU: `<paste the GPU name from the "GPU:" line>`
**Timing method:** 5 warm-up launches, then the median of 20 launches (`cudaEvent`), FP32, GFLOPS = 2n³ / time. Each size was run 3 times; the middle run is reported.

## Results

| Stage | Technique | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 1 | Naive (one thread per output) | TBD | TBD | TBD |
| 2 | Shared-memory tiling | TBD | TBD | TBD |
| 3 | Register blocking (4x4 per thread) | TBD | TBD | TBD |
| 4 | `float4` loads, transposed A tile, padded shared memory (64x64 tile) | 1479 | 1802 | 2610 |
| 5 | 128x128 tile, 8x8 per thread | 2171 | 4351 | 4292 |
| - | cuBLAS `cublasSgemm` | TBD | TBD | TBD |

All numbers in GFLOPS. Stage 5 vs cuBLAS: 76.8% (1024), 102.7% (2048), 96.9% (4096). Differences of a few percent are within run-to-run noise on a shared Colab GPU.

## What each stage changes

1. **Naive:** each thread reads a row of A and a column of B from global memory. Memory bound.
2. **Tiling:** blocks stage tiles of A and B in shared memory, so each global load is reused across the block.
3. **Register blocking:** each thread computes several outputs, so values loaded from shared memory are reused from registers.
4. **Vectorized access:** `float4` global loads, A stored transposed in shared memory so each thread reads its values with one 128-bit load, and 4 floats of padding to avoid bank conflicts on the transposed stores.
5. **Larger tiles:** 128x128 block tile and 8x8 outputs per thread, which raises FMAs per shared-memory load from 16:8 to 64:16.

## Notes and limitations

- Stage 5 is slower relative to cuBLAS at 1024: 128x128 tiles give only 64 blocks, which underfills the GPU.
- Sizes must be multiples of 128 (no edge handling).
- Not implemented: double buffering, warp tiling, tensor cores.

## Build and run

```
nvcc -O3 -arch=native stage5.cu -o stage5 -lcublas
./stage5 4096
```

Each run prints a PASS/FAIL check against cuBLAS before the timings.
