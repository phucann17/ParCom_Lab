#ifndef PDE_H
#define PDE_H
#include "mpi.h"
#include "omp.h"
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <algorithm>
void sequential_PDE(double* A, double* Cnew,
                    unsigned int n, int T,
                    double D, double k, double lambda,
                    double ux, double uy,
                    double dt, double dx, double dy);
void parallel_PDE_sync(double* local_A, double* local_C, unsigned int rows, int N, int T,
                  double D, double k, double lambda,
                  double ux, double uy,
                  double dt, double dx, double dy,
                  int world_rank, int world_size);
void parallel_PDE_async(double* local_A, double* local_C, unsigned int rows, int N, int T,
                  double D, double k, double lambda,
                  double ux, double uy,
                  double dt, double dx, double dy,
                  int world_rank, int world_size);

#endif