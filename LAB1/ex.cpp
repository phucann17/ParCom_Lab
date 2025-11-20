#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 4000
#define NUM_THREADS 10
void read_file(const char* filename, double* m, unsigned int n) {
    FILE* fd = fopen(filename, "r");
    if (!fd) {
        perror("Cannot open file to read");
        exit(1);
    }

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int ret = fscanf(fd, "%lf,", &m[i * n + j]);
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


void convolution_parallel_1(double *A, double K[3][3], double *B, int Asize, int ksize) {
    int pad = ksize / 2;  

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        for (int iteration = 0; iteration < 100; ++iteration) {
            #pragma omp for schedule(static, N * N / NUM_THREADS) collapse(2) 
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

            #pragma omp for schedule(static, N * N / NUM_THREADS)
            for (int i = 0; i < Asize * Asize; i++)
                A[i] = B[i];
        }
    }
}

void convolution_parallel_2(double *A, double K[3][3], double *B, int Asize, int ksize) {
    int pad = ksize / 2;
    int block = 128;

    #pragma omp parallel num_threads(NUM_THREADS)
    for (int iteration = 0; iteration < 100; ++iteration) {
        #pragma omp for collapse(2) schedule(dynamic)
        for (int bi = 0; bi < Asize; bi += block) {
            for (int bj = 0; bj < Asize; bj += block) {
                for (int i = bi; i < bi + block && i < Asize; ++i) {
                    for (int j = bj; j < bj + block && j < Asize; ++j) {

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
            }
        }
        #pragma omp for schedule(dynamic, N * N / NUM_THREADS)
        for (int i = 0; i < Asize * Asize; ++i) {
            A[i] = B[i];
        }
    }
}



int main(){
    double* A = (double*)malloc(N*N*sizeof(double));
    double* res = (double*)malloc(N*N*sizeof(double));
    double* res1 = (double*)malloc(N*N*sizeof(double));
    double* res2 = (double*)malloc(N*N*sizeof(double));
    double kernel[3][3] = {
        {0.05, 0.1, 0.05},
        {0.1,  0.4, 0.1},
        {0.05, 0.1, 0.05}
    };

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


    s0 = omp_get_wtime();
    check_matrix_equal(res, res1, N);
    check_matrix_equal(res, res2, N);
    s1 = omp_get_wtime();
    printf("Time to check: %.6lf seconds\n", s1 - s0);

    free(A);
    free(res);
    free(res1);
    free(res2);
    return 0;
}