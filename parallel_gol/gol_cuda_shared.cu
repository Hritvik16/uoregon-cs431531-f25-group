#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

typedef struct {
    int N;
    int *data;
} Grid;

static Grid grid_create(int N) {
    Grid g;
    g.N = N;
    g.data = (int *)malloc((size_t)N * (size_t)N * sizeof(int));
    if (g.data == NULL) {
        fprintf(stderr, "malloc failed for grid size %d\n", N);
        exit(1);
    }
    return g;
}

static void grid_free(Grid *g) {
    if (g->data != NULL) {
        free(g->data);
        g->data = NULL;
    }
    g->N = 0;
}

static inline int *grid_at(Grid *g, int y, int x) {
    return &g->data[(size_t)y * (size_t)g->N + (size_t)x];
}

static inline const int *grid_at_const(const Grid *g, int y, int x) {
    return &g->data[(size_t)y * (size_t)g->N + (size_t)x];
}

static void init_random(Grid *g, unsigned int seed) {
    int N = g->N;
    srand(seed);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int r = rand() % 10;
            *grid_at(g, y, x) = (r < 3) ? 1 : 0;
        }
    }
}

static double elapsed_seconds(const struct timeval *start, const struct timeval *end) {
    double s = (double)start->tv_sec + (double)start->tv_usec * 1.0e-6;
    double e = (double)end->tv_sec   + (double)end->tv_usec   * 1.0e-6;
    return e - s;
}

__global__ void gol_step_kernel_shared(const int *curr, int *next, int N) {
    int blockOriginX = blockIdx.x * blockDim.x;
    int blockOriginY = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tileWidth = blockDim.x + 2;
    int tileHeight = blockDim.y + 2;
    int numTileElems = tileWidth * tileHeight;

    extern __shared__ int tile[];

    int threadIndex = ty * blockDim.x + tx;

    for (int idx = threadIndex; idx < numTileElems; idx += blockDim.x * blockDim.y) {
        int sy = idx / tileWidth;
        int sx = idx % tileWidth;
        int gx = blockOriginX + sx - 1;
        int gy = blockOriginY + sy - 1;
        if (gx < 0) gx += N;
        else if (gx >= N) gx -= N;
        if (gy < 0) gy += N;
        else if (gy >= N) gy -= N;
        tile[sy * tileWidth + sx] = curr[gy * N + gx];
    }

    __syncthreads();

    int x = blockOriginX + tx;
    int y = blockOriginY + ty;

    if (x >= N || y >= N) {
        return;
    }

    int sx = tx + 1;
    int sy = ty + 1;
    int centerIndex = sy * tileWidth + sx;

    int neighbors = 0;
    neighbors += tile[(sy - 1) * tileWidth + (sx - 1)];
    neighbors += tile[(sy - 1) * tileWidth + sx];
    neighbors += tile[(sy - 1) * tileWidth + (sx + 1)];
    neighbors += tile[sy * tileWidth + (sx - 1)];
    neighbors += tile[sy * tileWidth + (sx + 1)];
    neighbors += tile[(sy + 1) * tileWidth + (sx - 1)];
    neighbors += tile[(sy + 1) * tileWidth + sx];
    neighbors += tile[(sy + 1) * tileWidth + (sx + 1)];

    int state = tile[centerIndex];
    int newState;
    if (state) {
        if (neighbors == 2 || neighbors == 3) newState = 1;
        else newState = 0;
    } else {
        if (neighbors == 3) newState = 1;
        else newState = 0;
    }

    next[y * N + x] = newState;
}

int main(int argc, char **argv) {
    int N = 1024;
    int steps = 100;
    unsigned int seed = 0;
    int blockDimX = 16;
    int blockDimY = 16;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (argc > 3) seed = (unsigned int)atoi(argv[3]);
    if (argc > 4) blockDimX = atoi(argv[4]);
    if (argc > 5) blockDimY = atoi(argv[5]);
    if (blockDimX <= 0) blockDimX = 16;
    if (blockDimY <= 0) blockDimY = blockDimX;

    Grid h_grid = grid_create(N);
    init_random(&h_grid, seed);

    int *d_curr = NULL;
    int *d_next = NULL;
    size_t bytes = (size_t)N * (size_t)N * sizeof(int);

    cudaError_t err;
    err = cudaMalloc((void **)&d_curr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_curr failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void **)&d_next, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_next failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_curr, h_grid.data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    dim3 block(blockDimX, blockDimY);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    size_t sharedBytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(int);

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, 0);

    for (int s = 0; s < steps; s++) {
        gol_step_kernel_shared<<<grid, block, sharedBytes>>>(d_curr, d_next, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        int *tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;
    }

    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);

    gettimeofday(&t_end, NULL);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, ev_start, ev_stop);

    double wall_time_s = elapsed_seconds(&t_start, &t_end);

    err = cudaMemcpy(h_grid.data, d_curr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_curr);
    cudaFree(d_next);
    grid_free(&h_grid);

    printf("N %d steps %d seed %u block %dx%d\n", N, steps, seed, blockDimX, blockDimY);
    printf("gpu_time_ms_shared %f\n", gpu_time_ms);
    printf("wall_time_s %f\n", wall_time_s);

    return 0;
}
