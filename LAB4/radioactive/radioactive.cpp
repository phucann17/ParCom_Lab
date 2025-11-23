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

void convolution_seq(double *A, double K[3][3], double *B, int Asize, int ksize) {
    int pad = ksize / 2;

    for (int iteration = 0; iteration < 100; ++iteration) {
        for (int i = 0; i < Asize; ++i) {
            for (int j = 0; j < Asize; ++j) {
                double sum = 0.0;
                for (int ki = 0; ki < ksize; ++ki) {
                    for (int kj = 0; kj < ksize; ++kj) {
                        int ii = i + ki - pad;
                        int jj = j + kj - pad;

                        if (ii < 0 || jj < 0 || ii >= Asize || jj >= Asize)
                            sum += 30.0 * K[ki][kj];
                        else
                            sum += A[ii * Asize + jj] * K[ki][kj];
                    }
                }
                B[i * Asize + j] = sum;
            }
        }
        for (int i = 0; i < Asize * Asize; ++i) {
            A[i] = B[i];
        }
    }
}


void convolution_MPI_OpenMP_sync(double* local_A, double K[][3], double* local_C,
                                 int chunk, int ksize, int n, int world_rank, int world_size) {

    int pad = ksize / 2;
    int rows_local = chunk;
    double* halo_top    = (double*)malloc(n * pad * sizeof(double));
    double* halo_bottom = (double*)malloc(n * pad * sizeof(double));
    int up   = world_rank - 1;
    int down = world_rank + 1;

    for (int iter = 0; iter < 100; ++iter) {
        // --- Synchronous halo exchange ---
        if (up >= 0) {
            MPI_Send(local_A, n * pad, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
            MPI_Recv(halo_top, n * pad, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (down < world_size) {
            MPI_Send(&local_A[(rows_local - pad) * n], n * pad, MPI_DOUBLE, down, 1, MPI_COMM_WORLD);
            MPI_Recv(halo_bottom, n * pad, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // --- Convolution with OpenMP ---
        #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(6)
        for (int i = 0; i < rows_local; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int ki = 0; ki < ksize; ++ki) {
                    for (int kj = 0; kj < ksize; ++kj) {
                        int ii = i + ki - pad;
                        int jj = j + kj - pad;
                        double val = 30.0; // default padding
                        if (jj >= 0 && jj < n) {
                            if (ii >= 0 && ii < rows_local) {
                                val = local_A[ii * n + jj];
                            } else if (ii < 0 && up >= 0) {
                                val = halo_top[ii + pad * n + jj]; // top halo
                            } else if (ii >= rows_local && down < world_size) {
                                val = halo_bottom[(ii - rows_local) * n + jj]; // bottom halo
                            }
                        }
                        sum += val * K[ki][kj];
                    }
                }
                local_C[i * n + j] = sum;
            }
        }
        // Swap: local_C → local_A
        #pragma omp parallel for
        for (int idx = 0; idx < rows_local * n; ++idx)
            local_A[idx] = local_C[idx];
    }

    free(halo_top);
    free(halo_bottom);
}

int main(int argc, char** argv){
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
    double kernel[3][3] = {
        {0.05, 0.1, 0.05},
        {0.1,  0.4, 0.1},
        {0.05, 0.1, 0.05}
    };
    int chunk = (N + world_size - 1) / world_size; // ceil(N / world_size)
    int padded = chunk * world_size;               // padded total rows

    if (world_rank == 0) {
        printf("Start simulation for radioactive convolution!\n");
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
        printf("Start simulation sequential convolution!\n");
        t0 = MPI_Wtime();
        // sequential_PDE(A, Cseq, N, T, D, k, lambda, ux, uy, dt, dx, dy);
        convolution_seq(A, kernel, Cseq, N, 3);
        t1 = MPI_Wtime();
        printf("Finish Sequential PDE with time: %.6f seconds\n", t1 - t0);
        free(A);
    }
    local_A = (double*)malloc(chunk * N * sizeof(double));
    local_C = (double*)malloc(chunk * N * sizeof(double));
    MPI_Scatter((world_rank == 0 ? A_mpi : nullptr), chunk * N, MPI_DOUBLE, local_A, chunk * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (world_rank == 0) printf("Starting parallel simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double s0 = MPI_Wtime();

    convolution_MPI_OpenMP_sync(local_A, kernel, local_C, chunk, 3, N, world_rank, world_size);

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
}


// 10.1.8.138  slots=8
// 10.1.8.32   slots=4
// 10.1.8.33   slots=4
// 10.1.8.34   slots=4
// 10.1.8.41   slots=4
// 10.1.8.42   slots=4
// 10.1.8.71   slots=4
// 10.1.8.72   slots=4
// 10.1.8.73   slots=4
// 10.1.8.74   slots=4
// 10.1.8.75   slots=4
// 10.1.8.76   slots=4
// 10.1.8.77   slots=4
// 10.1.8.78   slots=4
// 10.1.8.79   slots=4
// 10.1.8.80   slots=4
// 10.1.8.81   slots=4
// 10.1.8.82   slots=4