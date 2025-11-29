#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

typedef struct {
    int N;
    int *data;
} Grid;

Grid grid_create(int N) {
    Grid g;
    g.N = N;
    g.data = (int *)malloc((size_t)N * (size_t)N * sizeof(int));
    if (g.data == NULL) {
        fprintf(stderr, "malloc failed for grid size %d\n", N);
        exit(1);
    }
    return g;
}

void grid_free(Grid *g) {
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

static inline int wrap(int i, int N) {
    if (i < 0) return i + N;
    if (i >= N) return i - N;
    return i;
}

int count_neighbors(const Grid *g, int y, int x) {
    int N = g->N;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            int ny = wrap(y + dy, N);
            int nx = wrap(x + dx, N);
            const int *cell = grid_at_const(g, ny, nx);
            count += *cell;
        }
    }
    return count;
}

void step_grid(const Grid *curr, Grid *next) {
    int N = curr->N;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int alive = *grid_at_const(curr, y, x);
            int neighbors = count_neighbors(curr, y, x);
            int new_state = 0;
            if (alive) {
                if (neighbors == 2 || neighbors == 3) new_state = 1;
            } else {
                if (neighbors == 3) new_state = 1;
            }
            *grid_at(next, y, x) = new_state;
        }
    }
}

void init_random(Grid *g, double alive_prob, unsigned int seed) {
    int N = g->N;
    srand(seed);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            double r = rand() / (double)RAND_MAX;
            *grid_at(g, y, x) = (r < alive_prob) ? 1 : 0;
        }
    }
}

void print_grid(const Grid *g) {
    int N = g->N;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int val = *grid_at_const(g, y, x);
            putchar(val ? 'X' : '.');
        }
        putchar('\n');
    }
}

double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
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

    Grid curr = grid_create(N);
    Grid next = grid_create(N);

    init_random(&curr, 0.30, 42U);

    if (do_print && N <= 30) {
        printf("Initial state:\n");
        print_grid(&curr);
    }

    double t0 = now_seconds();

    for (int s = 0; s < steps; ++s) {
        step_grid(&curr, &next);
        int *tmp = curr.data;
        curr.data = next.data;
        next.data = tmp;
    }

    double t1 = now_seconds();
    double elapsed = t1 - t0;

    printf("N=%d steps=%d elapsed=%f s\n", N, steps, elapsed);

    if (do_print && N <= 30) {
        printf("Final state:\n");
        print_grid(&curr);
    }

    grid_free(&curr);
    grid_free(&next);

    return 0;
}
