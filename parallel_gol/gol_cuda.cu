#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

__device__ int wrap(int i, int N) {
    if (i < 0) return i + N;
    if (i >= N) return i - N;
    return i;
}

__global__ void gol_step_kernel(const int *curr, int *next, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    int idx = y * N + x;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            int ny = wrap(y + dy, N);
            int nx = wrap(x + dx, N);
            int nidx = ny * N + nx;
            count += curr[nidx];
        }
    }

    int alive = curr[idx];
    int new_state = 0;
    if (alive) {
        if (count == 2 || count == 3) new_state = 1;
    } else {
        if (count == 3) new_state = 1;
    }
    next[idx] = new_state;
}

void init_random(int *h_grid, int N, double alive_prob, unsigned int seed) {
    srand(seed);
    int total = N * N;
    for (int i = 0; i < total; ++i) {
        double r = rand() / (double)RAND_MAX;
        h_grid[i] = (r < alive_prob) ? 1 : 0;
    }
}

void print_grid(const int *h_grid, int N) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int val = h_grid[y * N + x];
            putchar(val ? 'X' : '.');
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s N steps [print]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    int do_print = 0;

    if (argc >= 4) {
        do_print = atoi(argv[3]) != 0;
    }

    if (N <= 0 || steps < 0) {
        fprintf(stderr, "N must be > 0 and steps >= 0\n");
        return 1;
    }

    int total = N * N;
    int bytes = total * sizeof(int);

    int *h_grid = (int *)malloc(bytes);
    if (h_grid == NULL) {
        fprintf(stderr, "malloc failed for host grid\n");
        return 1;
    }

    init_random(h_grid, N, 0.30, 42U);

    if (do_print && N <= 30) {
        printf("Initial state:\n");
        print_grid(h_grid, N);
    }

    int *d_curr = NULL;
    int *d_next = NULL;
    cudaError_t err;

    err = cudaMalloc((void **)&d_curr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_curr failed: %s\n", cudaGetErrorString(err));
        free(h_grid);
        return 1;
    }

    err = cudaMalloc((void **)&d_next, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_next failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_curr);
        free(h_grid);
        return 1;
    }

    err = cudaMemcpy(d_curr, h_grid, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_curr);
        cudaFree(d_next);
        free(h_grid);
        return 1;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double t0 = now_seconds();

    cudaEventRecord(start);
    for (int s = 0; s < steps; ++s) {
        gol_step_kernel<<<gridDim, blockDim>>>(d_curr, d_next, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel launch failed at step %d: %s\n", s, cudaGetErrorString(err));
            cudaFree(d_curr);
            cudaFree(d_next);
            free(h_grid);
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed at step %d: %s\n", s, cudaGetErrorString(err));
            cudaFree(d_curr);
            cudaFree(d_next);
            free(h_grid);
            return 1;
        }
        int *tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    double t1 = now_seconds();
    double elapsed_wall = t1 - t0;

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    err = cudaMemcpy(h_grid, d_curr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_curr);
        cudaFree(d_next);
        free(h_grid);
        return 1;
    }

    printf("N=%d steps=%d gpu_time_ms=%f wall_time_s=%f\n",
           N, steps, elapsed_ms, elapsed_wall);

    if (do_print && N <= 30) {
        printf("Final state:\n");
        print_grid(h_grid, N);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_curr);
    cudaFree(d_next);
    free(h_grid);

    return 0;
}
