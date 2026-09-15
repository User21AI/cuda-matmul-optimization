// Stage 3: CUDA matrix multiplication with 2D register blocking
// C = A * B, where A is (M x K), B is (K x N), C is (M x N)
//
// Building on Stage 2's shared memory tiling: now each THREAD computes a
// small TM x TN block of output elements (4x4 = 16 elements) instead of
// just one. This means each thread reuses values it already holds in
// registers across multiple output elements, instead of re-reading shared
// memory for each one — raising arithmetic intensity (more math per byte
// read).
//
// Tile sizes:
//   TILE_M x TILE_N = 64 x 64   -> output tile each thread BLOCK computes
//   TILE_K          = 16        -> depth of each tile along the shared dim
//   TM x TN         = 4 x 4     -> output block each single THREAD computes
//   blockDim        = 16 x 16   -> (TILE_M/TM) x (TILE_N/TN) = 256 threads
//
// Build:   nvcc -O3 matmul_regblock.cu -o matmul_regblock
// Run:     ./matmul_regblock

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define TM 4
#define TN 4
// threads per block = (TILE_M/TM) * (TILE_N/TN) = 16 * 16 = 256
#define THREADS_X (TILE_N / TN)   // 16
#define THREADS_Y (TILE_M / TM)   // 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Register-blocked kernel: each thread computes a TM x TN block of C.
// ---------------------------------------------------------------------------
__global__ void matmul_regblock_kernel(const float *A, const float *B,
                                        float *C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int blockRow = blockIdx.y * TILE_M;
    int blockCol = blockIdx.x * TILE_N;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
    int threadRow = threadIdx.y;                        // 0..15
    int threadCol = threadIdx.x;                         // 0..15

    // Per-thread accumulator, held in registers.
    float sum[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            sum[i][j] = 0.0f;

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // ---- Cooperative load of As (TILE_M x TILE_K = 1024 elements) ----
        // 256 threads, 4 elements each.
        for (int i = 0; i < (TILE_M * TILE_K) / (THREADS_X * THREADS_Y); ++i) {
            int idx = tid + i * (THREADS_X * THREADS_Y);
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int g_row = blockRow + r;
            int g_col = t * TILE_K + c;
            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // ---- Cooperative load of Bs (TILE_K x TILE_N = 1024 elements) ----
        for (int i = 0; i < (TILE_K * TILE_N) / (THREADS_X * THREADS_Y); ++i) {
            int idx = tid + i * (THREADS_X * THREADS_Y);
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int g_row = t * TILE_K + r;
            int g_col = blockCol + c;
            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // ---- Compute phase: reuse register values across TM x TN block ----
        for (int kk = 0; kk < TILE_K; ++kk) {
            float aReg[TM];
            float bReg[TN];

            for (int i = 0; i < TM; ++i)
                aReg[i] = As[threadRow * TM + i][kk];
            for (int j = 0; j < TN; ++j)
                bReg[j] = Bs[kk][threadCol * TN + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    sum[i][j] += aReg[i] * bReg[j];
        }

        __syncthreads();
    }

    // ---- Write results back to global memory ----
    for (int i = 0; i < TM; ++i) {
        int g_row = blockRow + threadRow * TM + i;
        if (g_row >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int g_col = blockCol + threadCol * TN + j;
            if (g_col < N) {
                C[g_row * N + g_col] = sum[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2 tiled kernel, kept for benchmark comparison.
// ---------------------------------------------------------------------------
#define SIMPLE_TILE 16
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    __shared__ float As[SIMPLE_TILE][SIMPLE_TILE];
    __shared__ float Bs[SIMPLE_TILE][SIMPLE_TILE];

    int row = blockIdx.y * SIMPLE_TILE + threadIdx.y;
    int col = blockIdx.x * SIMPLE_TILE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + SIMPLE_TILE - 1) / SIMPLE_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * SIMPLE_TILE + threadIdx.x;
        int b_row = t * SIMPLE_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < SIMPLE_TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Stage 1 naive kernel, kept for benchmark comparison.
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementation, used to check correctness.
// ---------------------------------------------------------------------------
void matmul_cpu_reference(const float *A, const float *B, float *C, int M,
                           int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void fill_random(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool check_correctness(const float *ref, const float *test, int size,
                        float tol = 1e-2f) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max absolute difference vs CPU reference: %f\n", max_diff);
    return max_diff < tol;
}

int main() {
    // ---- Step 1: correctness check on a small matrix ----
    {
        int M = 64, N = 64, K = 64;
        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        float *h_A = (float *)malloc(bytesA);
        float *h_B = (float *)malloc(bytesB);
        float *h_C = (float *)malloc(bytesC);
        float *h_C_ref = (float *)malloc(bytesC);

        srand(42);
        fill_random(h_A, M * K);
        fill_random(h_B, K * N);

        matmul_cpu_reference(h_A, h_B, h_C_ref, M, N, K);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytesA));
        CUDA_CHECK(cudaMalloc(&d_B, bytesB));
        CUDA_CHECK(cudaMalloc(&d_C, bytesC));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

        dim3 blockDim(THREADS_X, THREADS_Y);
        dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        matmul_regblock_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        bool ok = check_correctness(h_C_ref, h_C, M * N);
        printf("Correctness check (64x64), register-blocked kernel: %s\n\n",
               ok ? "PASSED" : "FAILED");

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        if (!ok) {
            fprintf(stderr, "Correctness check failed — fix the kernel before benchmarking.\n");
            return EXIT_FAILURE;
        }
    }

    // ---- Step 2: benchmark naive vs tiled vs register-blocked ----
    {
        int M = 1024, N = 1024, K = 1024;
        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        float *h_A = (float *)malloc(bytesA);
        float *h_B = (float *)malloc(bytesB);

        srand(123);
        fill_random(h_A, M * K);
        fill_random(h_B, K * N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytesA));
        CUDA_CHECK(cudaMalloc(&d_B, bytesB));
        CUDA_CHECK(cudaMalloc(&d_C, bytesC));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

        double flops = 2.0 * M * N * K;
        const int num_runs = 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float ms;

        printf("Matrix size: %dx%dx%d\n\n", M, N, K);

        // --- Naive ---
        {
            dim3 blockDim(16, 16);
            dim3 gridDim((N + 15) / 16, (M + 15) / 16);
            matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < num_runs; ++i)
                matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            double gflops = (flops / ((ms / num_runs) / 1000.0)) / 1e9;
            printf("Naive (Stage 1):          %.2f GFLOPS\n", gflops);
        }

        // --- Tiled ---
        {
            dim3 blockDim(SIMPLE_TILE, SIMPLE_TILE);
            dim3 gridDim((N + SIMPLE_TILE - 1) / SIMPLE_TILE,
                         (M + SIMPLE_TILE - 1) / SIMPLE_TILE);
            matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < num_runs; ++i)
                matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            double gflops = (flops / ((ms / num_runs) / 1000.0)) / 1e9;
            printf("Shared memory tiled (Stage 2): %.2f GFLOPS\n", gflops);
        }

        // --- Register blocked ---
        {
            dim3 blockDim(THREADS_X, THREADS_Y);
            dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
            matmul_regblock_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < num_runs; ++i)
                matmul_regblock_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            double gflops = (flops / ((ms / num_runs) / 1000.0)) / 1e9;
            printf("Register blocked (Stage 3): %.2f GFLOPS\n", gflops);
        }

        printf("\n(Record all three numbers — this is your Stage 3 README row.)\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_A); free(h_B);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    return 0;
}
