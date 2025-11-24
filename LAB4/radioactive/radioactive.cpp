#include "radioactive.h"


void read_file(const char* filename, double* m, unsigned int n) {
    FILE* fd = fopen(filename, "r");
    if (!fd) {
        perror("Cannot open file to read");
        exit(1);
    }

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int idx = i * n + j; 
            int ret = fscanf(fd, "%lf,", &m[idx]);
            if (ret == EOF) {
                printf("Reached EOF at element [%u][%u]\n", i, j);
                break;
            } else if (ret != 1) {
                printf("Error reading value at [%u][%u]\n", i, j);
                break;
            }
        }
    }

    fclose(fd);
}

void check_matrix_equal(double* A, double* B, unsigned int n){
    for (unsigned int idx = 0; idx < n; ++idx){
        if (A[idx] != B[idx]) {
            unsigned int i = idx / 4000;
            unsigned int j = idx % 4000;
            printf("Difference found at position (%u, %u): A=%.6f, B=%.6f\n",
                   i, j, A[idx], B[idx]);
            printf("Two matrices are not equal.\n");
            return;
        }
    }
    
    printf("Check matrix equal successfully\n");
}


void write_log(const char *filename, double** m, unsigned int n){
    FILE* fd = fopen(filename, "w");

    if (!fd){
        perror("Can not open file to write");
        exit(1);
    }

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            fprintf(fd, "%.2f ", m[i][j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}


void sequential_PDE(double* A, double* Cnew,
                    unsigned int n, int T,
                    double D, double k, double lambda,
                    double ux, double uy,
                    double dt, double dx, double dy) {

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                int idx = i * n + j; 
                double current = A[idx];

                // Advection (upwind)
                double adv_x = (i > 0) ? ux * (current - A[(i - 1) * n + j]) / dx
                                       : ux * current / dx;
                double adv_y = (j > 0) ? uy * (current - A[i * n + (j - 1)]) / dy
                                       : uy * current / dy;

                // Diffusion (discrete Laplacian)
                double up    = (i > 0)     ? A[(i - 1) * n + j] : 0.0; // C[i-1,j]
                double down  = (i < n - 1) ? A[(i + 1) * n + j] : 0.0; // C[i+1,j]
                double left  = (j > 0)     ? A[i * n + (j - 1)] : 0.0; // C[i,j-1]
                double right = (j < n - 1) ? A[i * n + (j + 1)] : 0.0; // C[i,j+1]

                double lap_x = (down - 2.0 * current + up) / (dx * dx);   
                double lap_y = (right - 2.0 * current + left) / (dy * dy);
                double diffusion = D * (lap_x + lap_y);
                double decay = (lambda + k) * current;

                // PDE update
                Cnew[idx] = current + dt * (-adv_x - adv_y + diffusion - decay);
            }
        }

        // Swap Cnew → A (copy arrays)
        for (int i = 0; i < n * n; i++) {
            A[i] = Cnew[i];
        }
    }
}



void parallel_PDE(double* local_A, double* local_C, unsigned int rows, int N, int T,
                  double D, double k, double lambda,
                  double ux, double uy,
                  double dt, double dx, double dy,
                  int world_rank, int world_size)
{
    double* lowest_row_above = (double*)malloc(N * sizeof(double));
    double* highest_row_below = (double*)malloc(N * sizeof(double));

    for (int t = 0; t < T; t++) {

        int up_rank = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
        int down_rank = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

        MPI_Request reqs[4];

        // ===== EXCHANGE BOUNDARY ROWS (halo swap) =====
        MPI_Irecv(lowest_row_above, N, MPI_DOUBLE, up_rank,   0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(highest_row_below, N, MPI_DOUBLE, down_rank, 1, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(local_A, N, MPI_DOUBLE, up_rank,   1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&local_A[(rows - 1) * N], N, MPI_DOUBLE, down_rank, 0, MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // ===== PDE UPDATE =====
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < N; j++) {

                int idx = i * N + j;
                double current = local_A[idx];

                // --- Advection (upwind) ---
                double adv_x = (i > 0) ? ux * (current - local_A[(i - 1) * N + j]) / dx
                            : (world_rank == 0 ? (ux * current / dx) : (ux * (current - lowest_row_above[j]) / dx));

                double adv_y = (j > 0) ? uy * (current - local_A[i * N + (j - 1)]) / dy : uy * current / dy;

                // --- Diffusion (Laplace) ---
                double up    = (i > 0) ? local_A[(i - 1) * N + j] 
                        : (world_rank == 0 ? 0.0 : lowest_row_above[j]);
                double down  = (i < rows - 1) ? local_A[(i + 1) * N + j] 
                                            : (world_rank == world_size - 1 ? 0.0 : highest_row_below[j]);
                double left  = (j > 0) ? local_A[i * N + (j - 1)] : 0.0;
                double right = (j < N - 1) ? local_A[i * N + (j + 1)] : 0.0;

                double laplace = (down - 2.0 * current + up) / (dx * dx)  
                            + (right - 2.0 * current + left) / (dy * dy); 

                // --- Update ---
                local_C[idx] = current + dt * (-adv_x - adv_y + D * laplace - (lambda + k) * current);
            }
        }

        // copy C → A
        for (int i = 0; i < rows * N; i++){
            local_A[i] = local_C[i];
        }  

        // BARRIER 
        MPI_Barrier(MPI_COMM_WORLD);

        // REDUCE
        int local_clean = 0;
        for (int i = 0; i < rows * N; i++)
            if (local_A[i] <= 0.0) local_clean++;

        int global_clean = 0;
        MPI_Reduce(&local_clean, &global_clean, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // printf("Iteration %d: Clean cells = %d\n", t, global_clean);
        }
    }

    // stop signal
    int stop = 1;
    MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(lowest_row_above);
    free(highest_row_below);
}

  
int main(int argc, char** argv) {
    unsigned int N = 4000;
    double* A = NULL;
    double* A_mpi = NULL;
    double* Cseq = NULL;              // Only rank 0 holds the full matrix
    double* CnewPar = NULL;
    double* local_A = NULL;
    double* local_C = NULL;

    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int chunk = (N + world_size - 1) / world_size; // ceil(N / world_size)
    int padded = chunk * world_size;               // padded total rows

    // Simulation parameters
    int T = 100;
    double D = 1000.0;
    double k = 1.0e-4;
    double lambda = 3.0e-5;
    double dt = 1.0;
    double ux = 3.3;
    double uy = 1.4;
    double dx = 10.0;
    double dy = 10.0;
    // Allocate full matrix only on rank 0
   if (world_rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        A_mpi = (double*)calloc(padded * N, sizeof(double)); // padding = 0
        Cseq = (double*)malloc(N * N * sizeof(double));
        CnewPar = (double*)malloc(padded * N * sizeof(double));
        printf("Reading file radioactive_matrix.csv ...\n");
        double t0 = MPI_Wtime();
        read_file("radioactive_matrix.csv", A, N);
        double t1 = MPI_Wtime();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_mpi[i * N + j] = A[i * N + j];
            }
        }
        printf("Reading finished in %.6f s\n", t1 - t0);
        t0 = MPI_Wtime();
        sequential_PDE(A, Cseq, N, T, D, k, lambda, ux, uy, dt, dx, dy);
        t1 = MPI_Wtime();
        printf("Sequential PDE time: %.6f seconds\n", t1 - t0);
        free(A);
    }

    // Allocate local memory for each process
    local_A = (double*)malloc(chunk * N * sizeof(double));
    local_C = (double*)malloc(chunk * N * sizeof(double));


    // Scatter padded matrix rows
    MPI_Scatter((world_rank == 0 ? A_mpi : nullptr), chunk * N, MPI_DOUBLE, local_A, chunk * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (world_rank == 0) printf("Starting parallel simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double s0 = MPI_Wtime();

    // Perform local PDE computation
    parallel_PDE(local_A, local_C, chunk, N, T, D, k, lambda, ux, uy, dt, dx, dy,
                 world_rank, world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double s1 = MPI_Wtime();

    if (world_rank == 0) printf("Parallel PDE time: %.6f seconds\n", s1 - s0);

    MPI_Gather(local_C, chunk * N, MPI_DOUBLE, CnewPar, chunk * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        s0 = MPI_Wtime();
        check_matrix_equal(Cseq, CnewPar, N * N);
        s1 = MPI_Wtime();
        printf("Check is valid seq and parallel in .%6f seconds\n", s1 - s0);
        free(A_mpi);
        free(Cseq);
        free(CnewPar);
    }

    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}

