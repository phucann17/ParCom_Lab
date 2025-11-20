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
#define NUM_THREAD 4
#define SECONDS 100

struct Task {
    int start_row;      
    int end_row;        
    double* A;          
    double* res;        
    double W;           
    double* C;          
    int t;              

    Task(int s, int e, double* A, double* res, double W, double* C, int t)
        : start_row(s), end_row(e), A(A), res(res), W(W), C(C), t(t) {}
};

class TaskQueue {
private:
    std::mutex mtx;
    std::queue<Task*> queue;
    std::condition_variable cv;
    int remaining_tasks = 0;
    static TaskQueue* instance;

public:
    void enqueue(Task* task) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(task);
        remaining_tasks++;
        cv.notify_one();
    }

    Task* dequeue() {
        std::unique_lock<std::mutex> lock(mtx);
        while (queue.empty()) {
            cv.wait(lock);
        }
        Task* task = queue.front();
        queue.pop();
        return task;
    }

    void task_done() {
        std::lock_guard<std::mutex> lock(mtx);
        remaining_tasks--;
        cv.notify_one();
    }

    bool all_done() {
        std::lock_guard<std::mutex> lock(mtx);
        return remaining_tasks == 0;
    }

    static TaskQueue* get() {
        if (!instance) instance = new TaskQueue();
        return instance;
    }
};
TaskQueue* TaskQueue::instance = nullptr;

class Worker {
private:
    bool stop;
    std::thread t;
    int id;

    void run() {
        int center_x = N / 2;
        int center_y = N / 2;

        while (!stop) {
            Task* task = TaskQueue::get()->dequeue();  // <-- dùng hàm public
            if (task != nullptr) {
                // Xử lý task như cũ
                for (int i = task->start_row; i < task->end_row; ++i) {
                    for (int j = 0; j < N; ++j) {
                        double dx = (i - center_x) * 10.0;
                        double dy = (j - center_y) * 10.0;
                        double R = sqrt(dx*dx + dy*dy);
                        double max_R = task->t * 343.0;

                        if (R <= max_R && task->res[i*N + j] == 0) {
                            double Z = R * pow(task->W, -1.0/3.0);
                            double U = -0.21436 + 1.35034 * log10(Z);

                            double logPso = 0, Ui = 1.0;
                            for (int k = 0; k < 9; ++k) {
                                logPso += task->C[k] * Ui;
                                Ui *= U;
                            }

                            double Pso = pow(10.0, logPso);
                            task->res[i*N + j] = Pso;
                            task->A[i*N + j] = Pso;
                        }
                    }
                }

                delete task; // free memory
                TaskQueue::get()->task_done(); // thông báo task đã xong
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

public:
    Worker(int id_) : id(id_), stop(false) {
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
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (res[i*N + j] > 0) continue;

                double dx = (i - center_x) * 10.0;
                double dy = (j - center_y) * 10.0;
                double R = std::sqrt(dx*dx + dy*dy);

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
    printf("Center Pso (sequential): %f kPa\n", res_seq[(N/2)*N + (N/2)]);
    printf("Time to sequential: %.6lf s\n", s1 - s0);
    // ---------------- Parallel execution ----------------
    std::vector<Worker*> workers;
    for (int i = 0; i < NUM_THREAD; ++i)
        workers.push_back(new Worker(i));

    int rows_per_task = N / NUM_THREAD;

    double p0 = omp_get_wtime();
    for (int t = 1; t <= SECONDS; ++t) {
        for (int start = 0; start < N; start += rows_per_task) {
            int end = std::min(start + rows_per_task, N);
            TaskQueue::get()->enqueue(new Task(start, end, res_parallel, res_parallel, W, C, t));
        }
    }

    // Chờ đến khi tất cả task xong
    while (!TaskQueue::get()->all_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    double p1 = omp_get_wtime();

    for (Worker* w : workers) {
        w->exit();
        delete w;
    }

    printf("Center Pso (parallel): %f kPa\n", res_parallel[(N/2)*N + (N/2)]);
    printf("Time to parallel: %.6lf s\n", p1 - p0);

   


    free(A_seq);
    free(res_seq);
    free(res_parallel);
    return 0;
}
