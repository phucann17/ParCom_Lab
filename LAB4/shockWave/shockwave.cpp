#include "shockwave.h"

// Sequential version for comparison
void sequential_shock_wave_blast(double *A, double *res, double W, double *C) {
    int center_x = N / 2;
    int center_y = N / 2;

    for (int t = 1; t <= SECONDS; ++t) {

        double max_R = t * 343.0;
        double max_cells = max_R / 10.0;
        if (max_cells > N/2) break;

        int min_i = std::max(0, center_x - (int)max_cells);
        int max_i = std::min(N-1, center_x + (int)max_cells);
        int min_j = std::max(0, center_y - (int)max_cells);
        int max_j = std::min(N-1, center_y + (int)max_cells);

        for (int i = min_i; i <= max_i; ++i) {
            for (int j = min_j; j <= max_j; ++j) {
                if (res[i*N + j] > 0) continue;

                double dx = (i - center_x) * 10.0;
                double dy = (j - center_y) * 10.0;
                double R = std::sqrt(dx*dx + dy*dy);
                if (R < 1e-9) continue;
                if (R <= max_R) {
                    double Z = R * std::pow(W, -1.0/3.0);
                    double U = -0.21436 + 1.35034 * std::log10(Z);

                    double logPso = 0, Ui = 1.0;
                    for (int k = 0; k < 9; ++k) {
                        logPso += C[k] * Ui;
                        Ui *= U;
                    }

                    double Pso = std::pow(10.0, logPso);
                    res[i*N + j] = Pso;
                    A[i*N + j] = Pso;
                }
            }
        }
    }
}