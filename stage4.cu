
// Stage 4: vectorized loads + transposed A tile + float4 shared reads + padding
// Compile (Colab):  nvcc -O3 -arch=native stage4.cu -o stage4 -lcublas
// Run:              ./stage4 1024
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#define BM 64      // block tile rows (of C)
#define BN 64      // block tile cols (of C)
#define BK 8       // K slice per iteration
#define TM 4       // per-thread rows
#define TN 4       // per-thread cols
#define PAD 4      // keeps float4 alignment, removes store bank conflicts on As

// Block = 16x16 = 256 threads. Requires M,N multiples of 64 and K multiple of 8.
__global__ void sgemm_v4(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K)
{
    __shared__ __align__(16) float As[BK][BM + PAD];   // A tile stored TRANSPOSED: As[k][m]
    __shared__ __align__(16) float Bs[BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;              // 0..255

    A += (size_t)blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (size_t)blockIdx.y * BM * N + blockIdx.x * BN;

    // Threads 0..127 load the A tile (64x8 = 128 float4), threads 128..255 load the B tile (8x64 = 128 float4)
    const int a_row = tid / 2;                 // 0..63
    const int a_col = (tid % 2) * 4;           // 0 or 4
    const int b_row = (tid - 128) / 16;        // 0..7
    const int b_col = ((tid - 128) % 16) * 4;  // 0..60

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (tid < 128) {
            float4 t = *reinterpret_cast<const float4*>(&A[(size_t)a_row * K + k0 + a_col]);
            As[a_col + 0][a_row] = t.x;
            As[a_col + 1][a_row] = t.y;
            As[a_col + 2][a_row] = t.z;
            As[a_col + 3][a_row] = t.w;
        } else {
            *reinterpret_cast<float4*>(&Bs[b_row][b_col]) =
                *reinterpret_cast<const float4*>(&B[(size_t)(k0 + b_row) * N + b_col]);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a4 = *reinterpret_cast<float4*>(&As[k][ty * TM]);
            float4 b4 = *reinterpret_cast<float4*>(&Bs[k][tx * TN]);
            float a[TM] = {a4.x, a4.y, a4.z, a4.w};
            float b[TN] = {b4.x, b4.y, b4.z, b4.w};
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(&C[(size_t)(ty * TM + i) * N + tx * TN]) = out;
    }
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    if (n % BM != 0 || n % BK != 0) { printf("n must be a multiple of %d\n", BM); return 1; }

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
    dim3 grid(n / BN, n / BM);

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    // cuBLAS is column-major: computing B*A in col-major gives A*B in row-major
    auto run_cublas = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dRef, n);
    };

    // ---- correctness ----
    sgemm_v4<<<grid, block>>>(dA, dB, dC, n, n, n);
    run_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (size_t i = 0; i < hC.size(); ++i) max_err = std::max(max_err, (double)fabsf(hC[i] - hRef[i]));
    printf("max abs error vs cuBLAS: %.3e  %s\n", max_err, max_err < 1e-2 ? "(PASS)" : "(FAIL)");

    // ---- timing: warmup, then median of 20 runs ----
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

    float ms_v4 = bench([&]() { sgemm_v4<<<grid, block>>>(dA, dB, dC, n, n, n); });
    float ms_cb = bench(run_cublas);
    printf("Stage 4 : %8.3f ms  %8.2f GFLOPS\n", ms_v4, flops / (ms_v4 * 1e6));
    printf("cuBLAS  : %8.3f ms  %8.2f GFLOPS\n", ms_cb, flops / (ms_cb * 1e6));
    printf("Stage 4 reaches %.1f%% of cuBLAS\n", 100.0 * ms_cb / ms_v4);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
