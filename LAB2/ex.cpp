#include <mpi.h>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <algorithm>


void read_file(const char* filename, double** m, unsigned int n) {
    FILE* fd = fopen(filename, "r");
    if (!fd) {
        perror("Cannot open file to read");
        exit(1);
    }

    // while (!feof(fd)){
    //     if (ferror(fd)){
    //         printf("Error reading file.\n");
    //         exit(1);
    //     }
    // }
for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
        int ret = fscanf(fd, "%lf,", &m[i][j]);
        if (ret == EOF) {
            printf("Reached EOF at element [%u][%u]\n", i, j);
            break;
        } else if (ret != 1) {
            printf("Error reading value at [%u][%u]\n", i, j);
            break;
        } else {
            //printf("Read m[%u][%u] = %.6lf\n", i, j, m[i][j]);
        }
    }
}

    fclose(fd);
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

double** alloc_matrix(unsigned int n){
    double** m = (double**)malloc(n * sizeof(double*));

    if (!m){
        perror("Can not allocate memory for m");
        exit(1);
    }

    for (int i = 0; i < n; ++i){
        m[i] = (double*)malloc(n * sizeof(double));
        if (!m[i]){
            perror("Can not allocate memory for row of m");
            exit(1);
        }
    }

    return m;
}

void free_matrix(double** m, unsigned int n){
    for (int i = 0; i < n; ++i){
        free(m[i]);
    }
    free(m);
}

void sequential_PDE(double** A, double** Cnew,
                         unsigned int n, int T,
                         double D, double k, double lambda,
                         double ux, double uy,
                         double dt, double dx, double dy) {
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                // Advection (upwind)
                double adv_x = 0.0, adv_y = 0.0;
                if (i > 0) adv_x = ux * (A[i][j] - A[i - 1][j]) / dx;
                else       adv_x = ux * A[i][j] / dx;

                if (j > 0) adv_y = uy * (A[i][j] - A[i][j - 1]) / dy;
                else       adv_y = uy * A[i][j] / dy;

                // Diffusion (discrete Laplacian)
                double diff = 0.0;
                diff += ((i > 0 ? A[i - 1][j] : 0) - 2 * A[i][j] + (i < n - 1 ? A[i + 1][j] : 0)) / (dx * dx);
                diff += ((j > 0 ? A[i][j - 1] : 0) - 2 * A[i][j] + (j < n - 1 ? A[i][j + 1] : 0)) / (dy * dy);

                // PDE update (Advection + Diffusion + Decay)
                Cnew[i][j] = A[i][j] + dt * (-adv_x - adv_y + D * diff - (lambda + k) * A[i][j]);
            }
        }

        // Copy Cnew → A
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i][j] = Cnew[i][j];
    }
}

void parallel_PDE(double** A, unsigned int n, int T,
                  double D, double k, double lambda,
                  double ux, double uy,
                  double dt, double dx, double dy)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = n / size;

    // Cấp phát bộ nhớ cục bộ

    // Chỉ rank 0 có mảng 2D đầy đủ -> chuyển sang dạng 1D để Scatter
    double* flat_A = NULL;
    if (rank == 0) {
        flat_A = malloc(n * n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                flat_A[i*n + j] = A[i][j];
    }

    // 1️⃣ Phát các tham số PDE cho mọi tiến trình
    MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ux, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&uy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2️⃣ Chia ma trận cho các tiến trình
    MPI_Scatter(flat_A, local_rows * n, MPI_DOUBLE,
                local_A, local_rows * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 3️⃣ Mỗi tiến trình xử lý PDE trên phần mình
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < local_rows - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                double adv_x = ux * (local_A[i*n + j] - local_A[(i-1)*n + j]) / dx;
                double adv_y = uy * (local_A[i*n + j] - local_A[i*n + j-1]) / dy;

                double diff = ((local_A[(i-1)*n + j] - 2*local_A[i*n + j] + local_A[(i+1)*n + j]) / (dx*dx))
                            + ((local_A[i*n + j-1] - 2*local_A[i*n + j] + local_A[i*n + j+1]) / (dy*dy));

                local_C[i*n + j] = local_A[i*n + j] +
                                   dt * (-adv_x - adv_y + D * diff - (lambda + k) * local_A[i*n + j]);
            }
        }

        // Copy ngược C → A
        for (int i = 1; i < local_rows - 1; i++)
            for (int j = 1; j < n - 1; j++)
                local_A[i*n + j] = local_C[i*n + j];

        // Đồng bộ giữa các tiến trình mỗi bước thời gian
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 4️⃣ Thu kết quả về rank 0
    MPI_Gather(local_A, local_rows * n, MPI_DOUBLE,
               flat_A, local_rows * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // 5️⃣ Giảm tổng (tùy chọn)
    double local_sum = 0.0, total_sum = 0.0;
    for (int i = 0; i < local_rows * n; i++) local_sum += local_A[i];
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total sum = %.2f\n", total_sum);

        // Gán lại vào A 2D
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i][j] = flat_A[i*n + j];

        free(flat_A);
    }

    free(local_A);
    free(local_C);
}

int main(int argc, char** argv) {
    unsigned int N = 4000;
    int T = 100;
    double D = 1000.0;
    double k = 1.0e-3;        // 10e-4 = 1e-3
    double lambda = 3.0e-5;
    double dt = 1.0;
    double ux = 3.3;
    double uy = 1.4;
    double dx = 3.16227;
    double dy = 3.16227;

    double** A = alloc_matrix(N);
    double** CnewSeq = alloc_matrix(N);
    double** CnewPar = alloc_matrix(N);

    printf("Reading file radioactive_matrix.csv ...........\n");
    read_file("radioactive_matrix.csv", A, N);
    // printf("A[0][0] = %f, A[0][1] = %f\n", A[0][0], A[0][1]);

    printf("Writing matrix A to file ...........\n");
    write_log("A.txt", A, N);

    printf("Starting simulation...\n");
    double s0 = MPI_Wtime();
    sequential_PDE(A, Cnew, N, T, D, k, lambda, ux, uy, dt, dx, dy);
    double s1 = MPI_Wtime();

    // s0 = MPI_Wtime();
    // parallel_PDE(A, CnewPar, N, T, D, k, lambda, ux, uy, dt, dx, dy);
    // s1 = MPI_Wtime();

    printf("Time to sequential dispersion: %.6lf seconds\n", s1 - s0);
    printf("Writing result to result_matrix.txt ...........\n");
    write_log("result_sequential_matrix.txt", CnewSeq, N);
    write_log("result_parallel_matrix.txt", CnewSeq, N);

    free_matrix(CnewSeq, N);
    free_matrix(A, N);
    free_matrix(CnewPar, N);

    printf("Simulation finished successfully.\n");
    return 0;
}

