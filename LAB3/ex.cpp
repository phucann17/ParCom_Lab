#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <stdio.h>
#include <omp.h>

#define N 4000
#define NUM_THREAD 10
#define SECONDS 100

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
struct Task {
    int id;
    int start_row, end_row; int t;
    double* A; double* res;        
    double W; double* C;          
    Task(int id, int s, int e, double* A, double* res, double W, double* C, int t)
        : id(id), start_row(s), end_row(e), A(A), res(res), W(W), C(C), t(t) {}
};


class TaskQueue {
private:
    std::mutex mtx;
    std::queue<Task*> queue;
    static TaskQueue* instance;
    bool finished = false;

public:

    void enqueue(Task* task) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(task);
    }

    Task* dequeue() {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) {
            // printf("Dequeue empty\n");
            return nullptr;
        }
        Task* task = queue.front();
        queue.pop();
        return task;
    }

    static TaskQueue* get() {
        if (instance == nullptr)
            instance = new TaskQueue();
        return instance;
    }

    bool isEmpty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
    }
    bool isFinished() {
        std::lock_guard<std::mutex> lock(mtx);
        return finished;
    }
};
TaskQueue* TaskQueue::instance = nullptr;

class Worker {
private:
    bool stop;
    std::thread t;
    int id;
    void run() {
        while (true) {
            Task* task = TaskQueue::get()->dequeue();
            if (task) {
                printf("Worker %d executes task %d\n", this->id, task->id);
                process(task);
                delete task;
            } else if (TaskQueue::get()->isFinished()) {
                break;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

public:
    
    void process(Task* task) {
        int center = N / 2;
        double max_R = task->t * 343.0;

        for (int i = task->start_row; i <= task->end_row; ++i) {

            for (int j = 0; j < N; ++j) {
                if (task->res[i*N + j] > 0) continue;

                double dx = (i - center)*10.0;
                double dy = (j - center)*10.0;
                double R = std::sqrt(dx*dx + dy*dy);
                if (R < 1e-9 || R > max_R) continue;

                double Z = R * std::pow(task->W, -1.0/3.0);
                double U = -0.21436 + 1.35034 * std::log10(Z);

                double logPso = 0, Ui = 1;
                for (int k = 0; k < 9; k++) {
                    logPso += task->C[k] * Ui;
                    Ui *= U;
                }

                double Pso = std::pow(10.0, logPso);
                task->res[i*N + j] = Pso;
                task->A[i*N + j] = Pso;
            }
        }
    }
    Worker(int id) : id(id), stop(false) { // khởi tạo vector với N mutex
        t = std::thread(&Worker::run, this);
    }

    void exit() {
        stop = true;
        t.join();
    }
};

// Sequential version for comparison
void sequential_shock_wave_blast(double *A, double *res, double W, double *C) {
    int center_x = N / 2;
    int center_y = N / 2;

    for (int t = 1; t <= SECONDS; ++t) {
        double max_R = t * 343.0;
        int min_i = std::max(0, center_x - (int)(max_R/10));
        int max_i = std::min(N-1, center_x + (int)(max_R/10));
        int min_j = std::max(0, center_y - (int)(max_R/10));
        int max_j = std::min(N-1, center_y + (int)(max_R/10));
        // printf("I = %d\n", t);
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

int main() {
    double *A_seq = (double*)malloc(N*N*sizeof(double));
    double *A_par = (double*)malloc(N*N*sizeof(double));
    double *res_seq = (double*)malloc(N*N*sizeof(double));
    double *res_parallel = (double*)malloc(N*N*sizeof(double));
    double W = 5e9;

    double C[9] = {2.611369, -1.690128, 0.00805, 0.336743,
                    -0.005162, -0.080923, -0.004785, 0.007930, 0.000768};

    for (int i = 0; i < N*N; ++i) {
        A_seq[i] = 0;
        res_seq[i] = 0;
        res_parallel[i] = 0;
    }

    // ---------------- Sequential execution ----------------
    double s0 = omp_get_wtime();
    sequential_shock_wave_blast(A_seq, res_seq, W, C);
    double s1 = omp_get_wtime();
    // printf("Center Pso (sequential): %f kPa\n", res_seq[(N/2)*N + (N/2)]);
    printf("Time to sequential: %.6lf s\n", s1 - s0);
    // ---------------- Parallel execution ----------------
    double p0 = omp_get_wtime();
    std::vector<Worker*> workers;
    for (int i = 0; i < NUM_THREAD; ++i)
        workers.push_back(new Worker(i));

    int taskid = 0;
    int center = N / 2;
    
    for (int t = 1; t <= SECONDS; ++t) {
        int max_R = t * 343;
        int min_i = std::max(0, center - max_R/10);
        int max_i = std::min(N-1, center + max_R/10);

        bool has_cell = false;
        for (int i = min_i; i <= max_i; ++i) {
            int dy_max = (int)std::sqrt(max_R*max_R - (i-center)*(i-center)*100.0)/10;
            int min_j = std::max(0, center - dy_max);
            int max_j = std::min(N-1, center + dy_max);
            if (min_j <= max_j) { has_cell = true; break; }
        }

        if (has_cell) {
            // printf("HAS CELL at t=%d \n", t);
            TaskQueue::get()->enqueue(new Task(taskid++, min_i, max_i, A_par, res_parallel, W, C, t));
        }
    }
    TaskQueue::get()->setFinished();
    
    // while (!TaskQueue::get()->isEmpty()) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // }
    for (Worker* w : workers) {
        w->exit();
        delete w;
    }
    double p1 = omp_get_wtime();
    // printf("Center Pso (parallel): %f kPa\n", res_parallel[(N/2)*N + (N/2)]);
    printf("Time to parallel: %.6lf s\n", p1 - p0);
    check_matrix_equal(res_seq, res_parallel, N);


    free(A_seq);
    free(A_par);
    free(res_seq);
    free(res_parallel);
    return 0;
}
