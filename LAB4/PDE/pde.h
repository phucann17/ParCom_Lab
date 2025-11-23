#ifndef PDE_H
#define PDE_H
#include "mpi.h"
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <algorithm>
void sequential_PDE(double* A, double* Cnew,
                    unsigned int n, int T,
                    double D, double k, double lambda,
                    double ux, double uy,
                    double dt, double dx, double dy);

#endif