#ifndef RADIOACTIVE_H
#define RADIOACTIVE_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>


#define NUM_THREADS 10
#define KSIZE 3
void convolution_seq(double *A, double K[3][3], double *B, int Asize, int ksize);
void convolution_MPI_OpenMP_sync(double* A, double K[KSIZE][KSIZE], int n);
void read_file(const char* filename, double* m, unsigned int n);
#endif