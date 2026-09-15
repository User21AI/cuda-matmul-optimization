

// C = A * B, where A is (M x K), B is (K x N), C is (M x N)
// One thread computes one output element, reads directly from global memory.
//
// Build:   nvcc -O3 matmul_naive.cu -o matmul_naive
// Run:     ./matmul_naive

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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
// Naive kernel: each thread computes one element of C.
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
// Simple triple loop — slow, but this is our source of truth.
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

// Returns true if max absolute difference is within tolerance.
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

        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);

        matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        bool ok = check_correctness(h_C_ref, h_C, M * N);
        printf("Correctness check (64x64): %s\n\n", ok ? "PASSED" : "FAILED");

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        if (!ok) {
            fprintf(stderr, "Correctness check failed — fix the kernel before benchmarking.\n");
            return EXIT_FAILURE;
        }
    }

    // ---- Step 2: benchmark on a larger matrix ----
    {
        int M = 1024, N = 1024, K = 1024;
        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        float *h_A = (float *)malloc(bytesA);
        float *h_B = (float *)malloc(bytesB);
        float *h_C = (float *)malloc(bytesC);

        srand(123);
        fill_random(h_A, M * K);
        fill_random(h_B, K * N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytesA));
        CUDA_CHECK(cudaMalloc(&d_B, bytesB));
        CUDA_CHECK(cudaMalloc(&d_C, bytesC));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);

        // Warm-up run (first launch pays kernel load / clock ramp-up cost)
        matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const int num_runs = 10;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_runs; ++i) {
            matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_runs;

        // FLOPs for one MxNxK matmul = 2*M*N*K (one multiply + one add per term)
        double flops = 2.0 * M * N * K;
        double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

        printf("Matrix size: %dx%dx%d\n", M, N, K);
        printf("Average kernel time: %.4f ms\n", avg_ms);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("(Record this number — it's your Stage 1 baseline for the README.)\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    return 0;
}

