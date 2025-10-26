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

void convolution_seq(double** A, double** K, double** B, int Asize, int ksize) {
    for (unsigned int i = 0; i < Asize; ++i){
        for (unsigned int j = 0; j < Asize; ++j){
            for (unsigned int k = 0; k < ksize; ++k){
                int ii = i + k;
                for (unsigned int m = 0; m < ksize; ++m){
                    int jj = j + m;
                    B[i][j] += A[ii][jj] * K[k][m];                }
            }
        }
    }
}


int main(){
    double** A = alloc_matrix(N);
    double kernel[3][3] = {
        {0.05, 0.1, 0.05},
        {0.1,  0.4, 0.1},
        {0.05, 0.1, 0.05}
    };

    double kernel1[3][3] = {
        {1, 0, 1},
        {0,  1, 0},
        {1, 0, 1}
    };

    double input[3][3] = {
        {0, 1, 1, 1, 0, 0, 0},
        {0,  1, 0},
        {1, 0, 1}
    };

    double s0 = omp_get_wtime();
    read_file("heat_matrix.csv", A, N);
    double s1 = omp_get_wtime();
    printf("Time to read file: %.6lf seconds\n", s1 - s0);

    s0 = omp_get_wtime();
    write_log("A.txt", A, N);
    s1 = omp_get_wtime();
    printf("Time to write file: %.6lf seconds\n", s1 - s0);



    free_matrix(A, N);
    return 0;
}