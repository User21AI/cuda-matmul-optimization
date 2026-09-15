# CUDA Matrix Multiplication Optimization

Progressive optimization of single-precision matrix multiplication (SGEMM) on CUDA, starting from a naive implementation and improving it stage by stage — measuring and explaining the speedup at each step rather than just reporting a final number.

Each stage is correctness-checked against a CPU reference implementation before being benchmarked, so every reported number reflects a kernel that actually produces the right answer.

## Results (1024 x 1024 x 1024, single precision)

|Stage|Technique|GFLOPS|Speedup vs Naive|
|-|-|-|-|
|1|Naive (global memory only)|385.93|1.0x|
|2|Shared memory tiling|614.28|1.59x|
|3|2D register blocking (4x4 per thread)|2395.65|6.21x|

Benchmarked on \[NVIDIA T4 via Google Colab], averaged over 10 runs after a warm-up launch.

## Stage-by-stage explanation

### Stage 1: Naive kernel

Each thread computes exactly one output element, reading directly from global memory for every multiply-add. Simple and correct, but extremely wasteful: neighboring threads redundantly re-read the same rows and columns of A and B from slow global memory.

### Stage 2: Shared memory tiling

Threads in a block cooperatively load a tile of A and a tile of B into fast on-chip shared memory once, then all threads in the block reuse that tile instead of each going back to global memory independently. This cuts redundant global memory traffic and gives a \~1.6x speedup.

### Stage 3: 2D register blocking

Instead of one thread computing one output element, each thread now computes a 4x4 block (16 elements). This lets the thread reuse values it already holds in fast registers across multiple output elements per shared-memory read, raising arithmetic intensity (more math done per memory access). This gave the largest single jump: \~3.9x over Stage 2, \~6.2x over the naive baseline.

## What's next

* **Stage 4:** Profile with Nsight Compute to identify remaining bottlenecks, then tune memory access patterns for full warp coalescing
* **Stage 5:** Compare final kernel against cuBLAS's `cublasSgemm` and explain the remaining gap (double buffering, tensor core usage, hand-tuned assembly)

## Build \& run

Each stage is a self-contained `.cu` file that includes a CPU correctness check followed by a GFLOPS benchmark.

```bash
nvcc -O3 matmul\_naive.cu -o matmul\_naive \&\& ./matmul\_naive
nvcc -O3 matmul\_tiled.cu -o matmul\_tiled \&\& ./matmul\_tiled
nvcc -O3 matmul\_regblock.cu -o matmul\_regblock \&\& ./matmul\_regblock
```

Developed and tested on Google Colab (T4 GPU).

