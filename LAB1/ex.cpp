#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 4000
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
            fprintf(fd, "%2.f ", m[i][j]);
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


void convolution_seq(double **A, double K[3][3], double **B, int Asize, int ksize) {
    int pad = ksize / 2;

    for (int i = 0; i < Asize; ++i) {
        for (int j = 0; j < Asize; ++j) {
            for (int ki = 0; ki < ksize; ++ki) {
                for (int kj = 0; kj < ksize; ++kj) {
                    int ii = i + ki - pad;
                    int jj = j + kj - pad;

                    //Replicate padding
                    if (ii < 0) ii = 0;                  // first row
                    if (jj < 0) jj = 0;                  // first column
                    if (ii >= Asize) ii = Asize - 1;     // final row
                    if (jj >= Asize) jj = Asize - 1;     // final column

                    B[i][j] += A[ii][jj] * K[ki][kj];
                }
            }
        }
    }
}

void convolution_parallel_1(double **A, double K[3][3], double **B, int Asize, int ksize) {
    int pad = ksize / 2;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (int i = 0; i < Asize; ++i) {
        for (int j = 0; j < Asize; ++j) {
            for (int ki = 0; ki < ksize; ++ki) {
                for (int kj = 0; kj < ksize; ++kj) {
                    int ii = i + ki - pad;
                    int jj = j + kj - pad;

                    //Replicate padding
                    if (ii < 0) ii = 0;                  // first row
                    if (jj < 0) jj = 0;                  // first column
                    if (ii >= Asize) ii = Asize - 1;     // final row
                    if (jj >= Asize) jj = Asize - 1;     // final column

                    B[i][j] += A[ii][jj] * K[ki][kj];
                }
            }
        }
    }
}

void convolution_parallel_2(double **A, double K[3][3], double **B, int Asize, int ksize) {
    int pad = ksize / 2;
    int block = 128; 

    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(10)
    for (int bi = 0; bi < Asize; bi += block) {
        for (int bj = 0; bj < Asize; bj += block) {
            for (int i = bi; i < bi + block && i < Asize; ++i) {
                for (int j = bj; j < bj + block && j < Asize; ++j) {
                    double sum = 0.0;
                    for (int ki = 0; ki < ksize; ++ki) {
                        for (int kj = 0; kj < ksize; ++kj) {
                            int ii = i + ki - pad;
                            int jj = j + kj - pad;

                            // replicate padding
                            if (ii < 0) ii = 0;
                            if (jj < 0) jj = 0;
                            if (ii >= Asize) ii = Asize - 1;
                            if (jj >= Asize) jj = Asize - 1;

                            sum += A[ii][jj] * K[ki][kj];
                        }
                    }
                    B[i][j] = sum;
                }
            }
        }
    }
}

void parallel_matrix_subtraction(double **a, double **b, double **res, int n) {
    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = a[i][j] - b[i][j];
        }
    }
}



int main(){
    double** A = alloc_matrix(N);
    double** res = alloc_matrix(N);
    double** res1 = alloc_matrix(N);
    double** res2 = alloc_matrix(N);
    double** res_sub1 = alloc_matrix(N);
    double** res_sub2 = alloc_matrix(N);
    double kernel[3][3] = {
        {0.05, 0.1, 0.05},
        {0.1,  0.4, 0.1},
        {0.05, 0.1, 0.05}
    };

    // double kernel1[3][3] = {
    //     {1, 0, 1},
    //     {0,  1, 0},
    //     {1, 0, 1}
    // };

    // double input[7][7] = {
    //      {0,1,1,1,0,0,0},
    //     {0,0,1,1,1,0,0},
    //     {0,0,0,1,1,1,0},
    //     {0,0,0,1,1,0,0},
    //     {0,0,1,1,0,0,0},
    //     {0,1,1,0,0,0,0},
    //     {1,1,0,0,0,0,0}
    // };

    double s0 = omp_get_wtime();
    read_file("heat_matrix.csv", A, N);
    double s1 = omp_get_wtime();
    printf("Time to read file: %.6lf seconds\n", s1 - s0);

    s0 = omp_get_wtime();
    convolution_seq(A, kernel, res, N, 3);
    s1 = omp_get_wtime();
    printf("Time to sequence convolution: %.6lf seconds\n", s1 - s0);

    s0 = omp_get_wtime();
    convolution_parallel_1(A, kernel, res1, N, 3);
    s1 = omp_get_wtime();
    printf("Time to parallel convolution 1: %.6lf seconds\n", s1 - s0);

    s0 = omp_get_wtime();
    convolution_parallel_2(A, kernel, res2, N, 3);
    s1 = omp_get_wtime();
    printf("Time to parallel convolution 2: %.6lf seconds\n", s1 - s0);

    parallel_matrix_subtraction(res, res1, res_sub1, N);
    parallel_matrix_subtraction(res, res2, res_sub2, N);

    s0 = omp_get_wtime();
    write_log("A.txt", res_sub1, N);
    write_log("B.txt", res_sub2, N);
    s1 = omp_get_wtime();
    printf("Time to write file: %.6lf seconds\n", s1 - s0);


    free_matrix(A, N);
    free_matrix(res, N);
    free_matrix(res1, N);
    free_matrix(res2, N);
    free_matrix(res_sub1, N);
    free_matrix(res_sub2, N);
    return 0;
}