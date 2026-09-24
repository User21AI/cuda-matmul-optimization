// Stage 5: bigger tiles (128x128 per block, 8x8 outputs per thread)
// Contains Stage 4 too, so both are timed in the same run against cuBLAS.
// Compile (Colab):  nvcc -O3 -arch=native stage5.cu -o stage5 -lcublas
// Run:              ./stage5 1024   (then 2048, 4096; n must be a multiple of 128)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------- Stage 4 (64x64 tile, 4x4 per thread) ----------------
#define V4_BM 64
#define V4_BN 64
#define V4_BK 8
#define V4_PAD 4

__global__ void __launch_bounds__(256)
sgemm_v4(const float* __restrict__ A, const float* __restrict__ B,
         float* __restrict__ C, int M, int N, int K)
{
    __shared__ __align__(16) float As[V4_BK][V4_BM + V4_PAD];
    __shared__ __align__(16) float Bs[V4_BK][V4_BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    A += (size_t)blockIdx.y * V4_BM * K;
    B += blockIdx.x * V4_BN;
    C += (size_t)blockIdx.y * V4_BM * N + blockIdx.x * V4_BN;

    const int a_row = tid / 2, a_col = (tid % 2) * 4;
    const int b_row = (tid - 128) / 16, b_col = ((tid - 128) % 16) * 4;
    float acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += V4_BK) {
        if (tid < 128) {
            float4 t = *reinterpret_cast<const float4*>(&A[(size_t)a_row * K + k0 + a_col]);
            As[a_col + 0][a_row] = t.x; As[a_col + 1][a_row] = t.y;
            As[a_col + 2][a_row] = t.z; As[a_col + 3][a_row] = t.w;
        } else {
            *reinterpret_cast<float4*>(&Bs[b_row][b_col]) =
                *reinterpret_cast<const float4*>(&B[(size_t)(k0 + b_row) * N + b_col]);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V4_BK; ++k) {
            float4 a4 = *reinterpret_cast<float4*>(&As[k][ty * 4]);
            float4 b4 = *reinterpret_cast<float4*>(&Bs[k][tx * 4]);
            float a[4] = {a4.x, a4.y, a4.z, a4.w};
            float b[4] = {b4.x, b4.y, b4.z, b4.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        *reinterpret_cast<float4*>(&C[(size_t)(ty * 4 + i) * N + tx * 4]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
}

// ---------------- Stage 5 (128x128 tile, 8x8 per thread) ----------------
#define V5_BM 128
#define V5_BN 128
#define V5_BK 8
#define V5_PAD 4

// Block = 16x16 = 256 threads. Each thread owns an 8x8 output made of four 4x4 pieces:
// rows {ty*4..+3, 64+ty*4..+3} x cols {tx*4..+3, 64+tx*4..+3}.
// Splitting the 8 into two groups of 4 spaced 64 apart keeps the shared-memory float4 reads conflict-free.
__global__ void __launch_bounds__(256)
sgemm_v5(const float* __restrict__ A, const float* __restrict__ B,
         float* __restrict__ C, int M, int N, int K)
{
    __shared__ __align__(16) float As[V5_BK][V5_BM + V5_PAD];   // transposed: As[k][m]
    __shared__ __align__(16) float Bs[V5_BK][V5_BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;                        // 0..255

    A += (size_t)blockIdx.y * V5_BM * K;
    B += blockIdx.x * V5_BN;
    C += (size_t)blockIdx.y * V5_BM * N + blockIdx.x * V5_BN;

    // Every thread loads exactly one float4 of A (128x8 = 256 float4) and one of B (8x128 = 256 float4)
    const int a_row = tid / 2;            // 0..127
    const int a_col = (tid % 2) * 4;      // 0 or 4
    const int b_row = tid / 32;           // 0..7
    const int b_col = (tid % 32) * 4;     // 0..124

    float acc[8][8] = {};

    for (int k0 = 0; k0 < K; k0 += V5_BK) {
        float4 t = *reinterpret_cast<const float4*>(&A[(size_t)a_row * K + k0 + a_col]);
        As[a_col + 0][a_row] = t.x; As[a_col + 1][a_row] = t.y;
        As[a_col + 2][a_row] = t.z; As[a_col + 3][a_row] = t.w;
        *reinterpret_cast<float4*>(&Bs[b_row][b_col]) =
            *reinterpret_cast<const float4*>(&B[(size_t)(k0 + b_row) * N + b_col]);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V5_BK; ++k) {
            float4 a0 = *reinterpret_cast<float4*>(&As[k][ty * 4]);
            float4 a1 = *reinterpret_cast<float4*>(&As[k][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<float4*>(&Bs[k][tx * 4]);
            float4 b1 = *reinterpret_cast<float4*>(&Bs[k][tx * 4 + 64]);
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row = (i / 4) * 64 + ty * 4 + (i % 4);
        *reinterpret_cast<float4*>(&C[(size_t)row * N + tx * 4]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(&C[(size_t)row * N + tx * 4 + 64]) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    if (n % V5_BM != 0) { printf("n must be a multiple of %d\n", V5_BM); return 1; }

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | matrix %dx%d\n", prop.name, n, n);

    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> hA(n * (size_t)n), hB(n * (size_t)n), hC(n * (size_t)n), hRef(n * (size_t)n);
    for (auto& x : hA) x = (rand() / (float)RAND_MAX) - 0.5f;
    for (auto& x : hB) x = (rand() / (float)RAND_MAX) - 0.5f;

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes)); CUDA_CHECK(cudaMalloc(&dRef, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid4(n / V4_BN, n / V4_BM);
    dim3 grid5(n / V5_BN, n / V5_BM);

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    auto run_cublas = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dRef, n);
    };
    auto run_v4 = [&]() { sgemm_v4<<<grid4, block>>>(dA, dB, dC, n, n, n); };
    auto run_v5 = [&]() { sgemm_v5<<<grid5, block>>>(dA, dB, dC, n, n, n); };

    // ---- correctness (both stages vs cuBLAS) ----
    run_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
    auto check = [&](const char* name, auto&& fn) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        fn();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        double max_err = 0;
        for (size_t i = 0; i < hC.size(); ++i) max_err = std::max(max_err, (double)fabsf(hC[i] - hRef[i]));
        printf("%s max abs error vs cuBLAS: %.3e  %s\n", name, max_err, max_err < 1e-2 ? "(PASS)" : "(FAIL)");
    };
    check("Stage 4", run_v4);
    check("Stage 5", run_v5);

    // ---- timing: 5 warmup, median of 20 ----
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int WARMUP = 5, RUNS = 20;
    double flops = 2.0 * n * (double)n * n;
    auto bench = [&](auto&& fn) {
        for (int i = 0; i < WARMUP; ++i) fn();
        std::vector<float> t;
        for (int i = 0; i < RUNS; ++i) {
            cudaEventRecord(s); fn(); cudaEventRecord(e);
            cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms, s, e); t.push_back(ms);
        }
        return median(t);
    };

    float ms4 = bench(run_v4), ms5 = bench(run_v5), msc = bench(run_cublas);
    printf("Stage 4 : %8.3f ms  %8.2f GFLOPS  (%.1f%% of cuBLAS)\n", ms4, flops / (ms4 * 1e6), 100.0 * msc / ms4);
    printf("Stage 5 : %8.3f ms  %8.2f GFLOPS  (%.1f%% of cuBLAS)\n", ms5, flops / (ms5 * 1e6), 100.0 * msc / ms5);
    printf("cuBLAS  : %8.3f ms  %8.2f GFLOPS\n", msc, flops / (msc * 1e6));

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
