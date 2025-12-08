#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

typedef struct {
    int N;
    int *data;
    int *ghost_cells; // {{top}, {right}, {bottom}, {left}}
} Grid;

Grid grid_create(int N) {
    Grid g;
    g.N = N;
    g.data = (int *)calloc((size_t)N * (size_t)N, sizeof(int));
    g.ghost_cells = (int *)calloc((size_t)N * (size_t)4, sizeof(int));
    if (g.data == NULL || g.ghost_cells == NULL) {
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
    if (g->ghost_cells != NULL) {
        free(g->ghost_cells);
        g->ghost_cells = NULL;
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

static inline int height(int i, int N) {
    return floor(i / N);
}

static inline int mathematical_modulo(int a, int b) {
    int r = a % b;
    if (r < 0) {
        r += b;
    }
    return r;
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

// Extract regular grid from large grid
void extractor(Grid* big, Grid* small) {
    for (int i = 0; i < small->N; i++) {
        int first_pointer_indx = (i * small->N);
        int second_pointer_indx = ((i + 1) * big->N) + 1;
        memcpy(small->data + first_pointer_indx, big->data + second_pointer_indx, small->N * sizeof(int));
    }
}

// Insert small grid into large grid.
// Load Ghost Data
void load_big_grid(Grid* big, Grid* small, int* ghost_cells) {
    for (int i = 0; i < big->N; i++) {
        if (i == 0) {
            memcpy(&big->data[0], ghost_cells, big->N * sizeof(int));
        } else if (i == big->N - 1) {
            memcpy(&big->data[(big->N)*(big->N - 1)], &ghost_cells[big->N * 2], big->N * sizeof(int));
        } else {
            big->data[i * big->N]           = ghost_cells[(big->N * 3)+ i];
            big->data[(i + 1) * big->N - 1] = ghost_cells[big->N + i];
            memcpy(big->data + (i * big->N) + 1, small->data + ((i - 1) * small->N), small->N * sizeof(int));
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

void print_grid_subd(Grid* g, int num_sub_divisions, int sub_divisions_per_layer) {
    for (int i = 0; i < sub_divisions_per_layer; i++) {
        for (int j = 0; j < sub_divisions_per_layer * g[i].N; j++) {
            for (int k = 0; k < g[i].N; k++) {
                int val = *grid_at(&g[(i * sub_divisions_per_layer) + (j % sub_divisions_per_layer)], floor(j / sub_divisions_per_layer), k);
                putchar(val ? 'X' : '.');
            }
            if (j % sub_divisions_per_layer == sub_divisions_per_layer - 1) {
                putchar('\n');
            }
        }
    }
}

void step_grid_omp(const Grid *curr, Grid *next) {
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

void step_grid(const Grid *curr, Grid *next) {
    int N = curr->N;
    
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

// Step all sub grids when subdivisions are set
void step_grid_subd(Grid *curr, Grid *next, int num_sub_grids, int sub_grid_per_layer, int rank, int useomp) {

    int* ghost_cells = (int *)calloc((curr->N + 2) * 4, sizeof(int));

    // top
    int top_grid_idx;
    if (height(rank, sub_grid_per_layer) != 0) {
        top_grid_idx = rank - sub_grid_per_layer;
    } else {
        top_grid_idx = rank + (sub_grid_per_layer * (sub_grid_per_layer - 1));
    }
    int top_left_idx  = mathematical_modulo((top_grid_idx - 1),  (sub_grid_per_layer)) + sub_grid_per_layer * height(top_grid_idx, sub_grid_per_layer);
    int top_right_idx = mathematical_modulo((top_grid_idx + 1),  (sub_grid_per_layer)) + sub_grid_per_layer * height(top_grid_idx, sub_grid_per_layer);

    // bottom
    int bottom_grid_idx;
    if (height(rank, sub_grid_per_layer) != (sub_grid_per_layer - 1)) {
        bottom_grid_idx = rank + sub_grid_per_layer;
    } else {
        bottom_grid_idx = rank - (sub_grid_per_layer * (sub_grid_per_layer - 1));
    }
    int bottom_left_idx  = mathematical_modulo((bottom_grid_idx - 1),  (sub_grid_per_layer)) + sub_grid_per_layer * height(bottom_grid_idx, sub_grid_per_layer);
    int bottom_right_idx = mathematical_modulo((bottom_grid_idx + 1),  (sub_grid_per_layer)) + sub_grid_per_layer * height(bottom_grid_idx, sub_grid_per_layer);
   
    // right
    int right_grid_idx;
    if (rank % sub_grid_per_layer != (sub_grid_per_layer - 1)) {
        right_grid_idx = rank + 1;
    } else {
        right_grid_idx = rank - (sub_grid_per_layer - 1);
    }
    
    // left
    int left_grid_idx;
    if (rank % sub_grid_per_layer != 0) {
        left_grid_idx = rank - 1;
    } else {
        left_grid_idx = rank + (sub_grid_per_layer - 1);
    }
    MPI_Send(curr->ghost_cells, curr->N, MPI_INT, top_grid_idx, 0, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells, 1, MPI_INT, top_left_idx, 1, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + curr->N - 1, 1, MPI_INT, top_right_idx, 2, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + (curr->N * 2), curr->N, MPI_INT, bottom_grid_idx, 3, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + (curr->N * 2), 1, MPI_INT, bottom_left_idx, 4, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + (curr->N * 3) - 1, 1, MPI_INT, bottom_right_idx, 5, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + (curr->N * 3), curr->N, MPI_INT, left_grid_idx, 6, MPI_COMM_WORLD);
    MPI_Send(curr->ghost_cells + (curr->N), curr->N, MPI_INT, right_grid_idx, 7, MPI_COMM_WORLD);

    MPI_Recv(ghost_cells + 1, curr->N, MPI_INT, top_grid_idx, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells, 1, MPI_INT, top_left_idx, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + curr->N + 1, 1, MPI_INT, top_right_idx, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + ((curr->N + 2) * 2) + 1, curr->N, MPI_INT, bottom_grid_idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + ((curr->N + 2) * 2), 1, MPI_INT, bottom_left_idx, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + ((curr->N + 2) * 3) - 1, 1, MPI_INT, bottom_right_idx, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + ((curr->N + 2) * 3) + 1, curr->N, MPI_INT, left_grid_idx, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ghost_cells + ((curr->N + 2)) + 1, curr->N, MPI_INT, right_grid_idx, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Grid ghost_filled_sub_grid_current = grid_create(curr->N + 2);
    Grid ghost_filled_sub_grid_next    = grid_create(curr->N + 2);
    load_big_grid(&ghost_filled_sub_grid_current, curr, ghost_cells);
    if (useomp) {
        step_grid_omp(&ghost_filled_sub_grid_current, &ghost_filled_sub_grid_next);
    }  
    else {
        step_grid(&ghost_filled_sub_grid_current, &ghost_filled_sub_grid_next);
    }
    extractor(&ghost_filled_sub_grid_next, next);
    free(ghost_cells);
    grid_free(&ghost_filled_sub_grid_current);
    grid_free(&ghost_filled_sub_grid_next);
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

void init_ghost_cells(Grid *g) {
    int i = 0;
    int j = 0;
    // top
    for (j = 0; j < g->N; j++) {
        g->ghost_cells[j] = *grid_at(g, i, j);
    }
    // right
    j = g->N - 1;
    for (i = 0; i < g->N; i++) {
        g->ghost_cells[g->N + i] = *grid_at(g, i, j);
    }
    // bottom
    i = g->N - 1;
    for (j = 0; j < g->N; j++) {
        g->ghost_cells[(2 * (g->N)) + j] = *grid_at(g, i, j);
    }
    // left
    j = 0;
    for (i = 0; i < g->N; i++) {
        g->ghost_cells[(3 * (g->N)) + i] = *grid_at(g, i, j); 
    }
}

double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

int init_and_time(int steps, int N, int do_print) {

    Grid curr = grid_create(N);
    Grid next = grid_create(N);
    init_random(&curr, 0.30, 42U);
    // init_ghost_cells(&curr); Leaving out ghost cell generation here. Is not needed.
    if (do_print && N <= 30) {
        printf("Initial state:\n");
        print_grid(&curr);
    }
    double t0 = now_seconds();

    for (int s = 0; s < steps; ++s) {
        step_grid_omp(&curr, &next);
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

void init_and_time_subd(Grid* curr, Grid* next, int steps, int N, int do_print, int sub_grid_size, int rank, int size, int useomp) {
    int num_sub_grids = size;
    srand(time(NULL) * rank + rank);
    init_random(curr, 0.30, rand());
    init_ghost_cells(curr);

    if (do_print && N <= 30) {
        //printf("Initial state:\n");
        //print_grid_subd(curr, num_sub_grids, N/sub_grid_size);
    }
    
    for (int s = 0; s < steps; ++s) {
        step_grid_subd(curr, next, num_sub_grids, N/sub_grid_size, rank, useomp);
        int *tmp = curr->data;
        curr->data = next->data;
        next->data = tmp;
        init_ghost_cells(curr);
    }
}

int test_step(void (*step_func)(Grid*, Grid*)) {
    // Tests a specific step function on a grid and checks it correctly iterates a single step of conways game of life. 
    Grid grid_first  = grid_create(6);
    Grid grid_second = grid_create(6);
    for (int i = 12; i < 14; i++) {
        grid_first.data[i] = 1;
    }
    for (int i = 16; i < 20; i++) {
        grid_first.data[i] = 1;
    }
    for (int i = 22; i < 24; i++) {
        grid_first.data[i] = 1;
    }
    step_func(&grid_first, &grid_second);
    
    int correct = (grid_second.data[6] == 1) && (grid_second.data[11] == 1) && (grid_second.data[13] == 1) && (grid_second.data[16] == 1) && (grid_second.data[19] == 1) && (grid_second.data[22] == 1) && (grid_second.data[24] == 1) && (grid_second.data[29] == 1) && (grid_second.data[12] == 0) && grid_second.data[23] == 0;
    for (int i = 0; i < 6; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }

    for (int i = 7; i < 11; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }
    for (int i = 17; i < 19; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }
    for (int i = 14; i < 16; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }
    for (int i = 20; i < 22; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }
    for (int i = 25; i < 29; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }
    for (int i = 30; i < 36; i++) {
        correct = correct && (grid_second.data[i] == 0);
    }

    grid_free(&grid_first);
    grid_free(&grid_second);
    return correct;
}

int test_two() {

    int data[] = {1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0};
    Grid test = grid_create(6);
    Grid out  = grid_create(6);
    memcpy(test.data, data, 36 * sizeof(int));
    step_grid(&test, &out);
    print_grid(&out);
    return 1;
}
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s N steps [print] [number_of_subgrids] [useomp]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    int do_print = 0;
    int num_sub_divisions = -1;
    int sub_grid_size;
    int useomp = 0;

    // Adds ability to specify subdivisions
    if (argc >= 5) {
        num_sub_divisions = atoi(argv[4]);
        if ((N % (int)sqrt(num_sub_divisions) != 0 && sqrt(num_sub_divisions) == (int)sqrt(num_sub_divisions))) {
            fprintf(stderr, "number of subdivisions must be a number which divides the total grid into subgrids of whole number size");
            return 1;
        }
        sub_grid_size = N / sqrt(num_sub_divisions);
        if (argc >= 6) {
            useomp = atoi(argv[5]);
        }
    }

    if (argc >= 4) {
        do_print = atoi(argv[3]) != 0;
    }

    if (N <= 0 || steps < 0) {
        fprintf(stderr, "N must be > 0 and steps >= 0\n");
        return 1;
    }

    Grid *results = (Grid*)malloc(sizeof(Grid) * num_sub_divisions);
    for (int i = 0; i < num_sub_divisions; i++) {
        results[i] = grid_create(sub_grid_size);
    }
    int size, rank;
    if (argc >= 5) {
        double elapsed;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        //printf("%d\n", rank);
        
        Grid curr = grid_create(sub_grid_size);
        Grid next = grid_create(sub_grid_size);
        init_and_time_subd(&curr, &next, steps, N, do_print, sub_grid_size, rank, size, useomp);
        if (rank == 0) {
            for (int i = 1; i < num_sub_divisions; i++) {
                MPI_Recv(results[i].data, sub_grid_size * sub_grid_size, MPI_INT, i, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //MPI_Recv(results[i].ghost_cells, sub_grid_size * 4, MPI_INT, i, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            memcpy(results[0].data, curr.data, sub_grid_size * sub_grid_size * sizeof(int));
            //memcpy(results)
        } else {
            MPI_Send(curr.data, sub_grid_size * sub_grid_size, MPI_INT, 0, 9, MPI_COMM_WORLD);
            //MPI_Send(curr.ghost_cells, subgrid_size * 4, MPI_INT, i, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //print_grid(&results[rank]);
        double t1 = MPI_Wtime();
        
        double local_elapsed = t1 - t0;
        MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (!rank) {
            printf("N=%d steps=%d elapsed=%f s\n", N, steps, elapsed);
        }
        MPI_Finalize();
    } else {
        init_and_time(steps, N, do_print);
    }

    if (do_print && N <= 30 && argc >= 5 && !rank) {
        printf("Final state:\n");
        //print_grid_subd(results, num_sub_divisions, N/sub_grid_size);
        print_grid_subd(results, num_sub_divisions, N/sub_grid_size);
        for (int i = 0; i < num_sub_divisions; i++) {
            grid_free(&results[i]);
        }
        free(results);
    }
    
    return 0;
}
