#include "shockwave.h"

// Configuration setups
const int MAP_SIZE = 4000; // 4000x4000
const double CELL_SIZE = 10.0; // 10 meters
const double CENTER = 2000.0; // Center coordinate
const double YIELD_W = 5000000000; // 5000 kilotons
const double SPEED_SOUND = 343.0; // Speed of sound in m/s
const int SIM_DURATION = 100; // 100 seconds

// Kingery-Bulmash Coefficients
const double C[] = {
    2.611369, -1.690128, 0.00805, 0.336743,
    -0.005162, -0.080923, -0.004785, 0.007930, 0.000768
};

double calculate_pressure_at_cell(int r, int c) {
    double dy = (r - CENTER) * CELL_SIZE;
    double dx = (c - CENTER) * CELL_SIZE;
    // Distance from the blast (meters)
    double R = sqrt(dx * dx + dy * dy);

    if (R < 1.0) R = 1.0; // Prevent divide 0 issues

    double t_arrival = R / SPEED_SOUND;
    // if (t_arrival > SIM_DURATION) return 0.0;

    // Calculate Scaled Distance Z = R * W^(-1/3)
    double Z = R * pow(YIELD_W, -1.0 / 3.0); 
    // Calculate intermediate value U
    double U = -0.21436 + 1.35034 * std::log10(Z);
    // Calculate log10(Pso)
    double log10Pso = 0.0;
    for (int i = 0; i <= 8; ++i) {
        log10Pso += C[i] * pow(U, i);
    }

    return std::pow(10.0, log10Pso); // Pressure in kPa
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Choosing mode 0 = Synchronous (Default), 1 = Asynchronous
    int mode = 0; 
    if (argc > 1) mode = atoi(argv[1]);

    // domain decomposition
    // Split rows among world_size processes
    int rows_per_proc = MAP_SIZE / world_size;
    int start_row = world_rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    
    // handle redundant if MAP_SIZE not divide nodes
    if (world_rank == world_size - 1) {
        end_row = MAP_SIZE;
        rows_per_proc = end_row - start_row;
    }

    // Add 2 rows (Ghost rows) for demo Halo Exchange: [0] is ghost above [rows+1] là ghost down
    // Real data start from index 1 to rows_per_proc
    std::vector<std::vector<double>> local_map(rows_per_proc + 2, std::vector<double>(MAP_SIZE, 0.0));

    std::vector<double> send_top(MAP_SIZE), send_bot(MAP_SIZE);
    std::vector<double> recv_top(MAP_SIZE), recv_bot(MAP_SIZE);

    if (world_rank == 0) {
        std::cout << "=== DISTRIBUTED MPI SIMULATION ===" << std::endl;
        std::cout << "Nodes: " << world_size << " | Grid: " << MAP_SIZE << "x" << MAP_SIZE << std::endl;
        std::cout << "Mode: " << (mode == 0 ? "Synchronous" : "Asynchronous") << std::endl;
        std::cout << "Hybrid: MPI + OpenMP" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    double start_time = MPI_Wtime();

    for (int t = 1; t <= SIM_DURATION; ++t) {
        // Trao đổi biên (Halo Exchange)
        // Idea
        // Mục đích: Demo kỹ thuật phân tán. 
        // Rank i gửi dòng đầu của mình cho Rank i-1, và dòng cuối cho Rank i+1
        
        // Copy real data to send buffers
        #pragma omp parallel for
        for(int c=0; c<MAP_SIZE; ++c) {
            send_top[c] = local_map[1][c];                 
            send_bot[c] = local_map[rows_per_proc][c];   
        }

        int top_neighbor = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
        int bot_neighbor = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

        if (mode == 0) { 
            // blocking
            MPI_Sendrecv(send_top.data(), MAP_SIZE, MPI_DOUBLE, top_neighbor, 0,
                         local_map[0].data(), MAP_SIZE, MPI_DOUBLE, top_neighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            MPI_Sendrecv(send_bot.data(), MAP_SIZE, MPI_DOUBLE, bot_neighbor, 1,
                         local_map[rows_per_proc + 1].data(), MAP_SIZE, MPI_DOUBLE, bot_neighbor, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } 
        else {
            // non-blocking for decrease compute in CPU
            MPI_Request requests[4];
            int req_count = 0;

            if (top_neighbor != MPI_PROC_NULL) {
                MPI_Isend(send_top.data(), MAP_SIZE, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &requests[req_count++]);
                MPI_Irecv(local_map[0].data(), MAP_SIZE, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD, &requests[req_count++]);
            }
            if (bot_neighbor != MPI_PROC_NULL) {
                MPI_Isend(send_bot.data(), MAP_SIZE, MPI_DOUBLE, bot_neighbor, 1, MPI_COMM_WORLD, &requests[req_count++]);
                MPI_Irecv(local_map[rows_per_proc + 1].data(), MAP_SIZE, MPI_DOUBLE, bot_neighbor, 0, MPI_COMM_WORLD, &requests[req_count++]);
            }

            // Computation Overlapse (Tính toán chồng lấp)
            // Trong khi mạng đang gửi biên, ta tính toán phần "RUỘT" (Inner part)
            // Phần ruột không phụ thuộc vào ghost cell (index 2 đến rows_per_proc-1)
            #pragma omp parallel for collapse(2)
            for (int r = 2; r < rows_per_proc; ++r) {
                for (int c = 0; c < MAP_SIZE; ++c) {
                    int global_r = start_row + (r - 1);
                    double dist = std::sqrt(pow((global_r-CENTER)*CELL_SIZE,2) + pow((c-CENTER)*CELL_SIZE,2));
                    if (dist/SPEED_SOUND <= t) { 
                        local_map[r][c] = calculate_pressure_at_cell(global_r, c);
                    }
                }
            }

            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        // Idea
        // Nếu là Async thì chỉ cần tính phần Biên (Rows 1 & rows_per_proc).
        // Nếu là Sync thì tính toàn bộ (Rows 1 đến rows_per_proc).
        
        int r_start = (mode == 1) ? 1 : 1;
        int r_end   = (mode == 1) ? rows_per_proc : rows_per_proc;
        
        #pragma omp parallel for collapse(2)
        for (int r = r_start; r <= r_end; ++r) {
            for (int c = 0; c < MAP_SIZE; ++c) {
                if (mode == 1 && r > 1 && r < rows_per_proc) continue;
                // Map from local index to global index
                int global_r = start_row + (r - 1); 

                double dy = (global_r - CENTER) * CELL_SIZE;
                double dx = (c - CENTER) * CELL_SIZE;
                double dist = std::sqrt(dx*dx + dy*dy);
                
                if (dist / SPEED_SOUND <= t) {
                     local_map[r][c] = calculate_pressure_at_cell(global_r, c);
                }
            }
        }
    }

    double end_time = MPI_Wtime();

    // Gather result
    // Idea
    // Chỉ Rank 0 mới cần bộ nhớ cho toàn bản đồ để verify
    std::vector<double> global_map_flat;
    std::vector<double> local_map_flat;
    std::vector<int> recv_rows;
    std::vector<int> recv_counts;
    std::vector<int> displs;

    // Flatten local map (remove 2 rows ghost) and send to Master
    local_map_flat.resize(rows_per_proc * MAP_SIZE);
    #pragma omp parallel for collapse(2)
    for (int r = 1; r <= rows_per_proc; ++r) {
        for (int c = 0; c < MAP_SIZE; ++c) {
            local_map_flat[(r-1)*MAP_SIZE + c] = local_map[r][c];
        }
    }

    if (world_rank == 0) {
        global_map_flat.resize(MAP_SIZE * MAP_SIZE);
        recv_rows.resize(world_size);
    }

    // Gather number of rows from each process
    MPI_Gather(&rows_per_proc, 1, MPI_INT,
               recv_rows.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; ++i) {
            recv_counts[i] = recv_rows[i] * MAP_SIZE;
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }

    MPI_Gatherv(local_map_flat.data(), rows_per_proc * MAP_SIZE, MPI_DOUBLE,
                global_map_flat.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Simulation Time: " << (end_time - start_time) << " seconds." << std::endl;
        
        // Checking (2050, 2050) near center
        double val = global_map_flat[2050 * MAP_SIZE + 2050];
        std::cout << "Value at (2050, 2050): " << val << std::endl;
        
        if (val > 0) std::cout << "Verification: Data seems valid (Non-zero)." << std::endl;
        else std::cout << "Verification: WARNING (Value is 0)." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
