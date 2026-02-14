#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define OBSTACLE -1
#define UNVISITED -2

const int WIDTH = 512;
const int HEIGHT = 512;

__global__
void wave_step(int* grid, int level, int* changed) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    int index = y * WIDTH + x;

    if (grid[index] == level) {

        int dr[4] = {-1, 1, 0, 0};
        int dc[4] = {0, 0, -1, 1};

        for (int k = 0; k < 4; k++) {

            int nx = x + dc[k];
            int ny = y + dr[k];

            if (nx >= 0 && nx < WIDTH &&
                ny >= 0 && ny < HEIGHT) {

                int nidx = ny * WIDTH + nx;

                if (grid[nidx] == UNVISITED) {
                    grid[nidx] = level + 1;
                    *changed = 1;
                }
            }
        }
    }
}

int main() {

    std::vector<int> h_grid(WIDTH * HEIGHT, UNVISITED);

    srand((unsigned)time(0));

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (rand() % 100 < 30)
            h_grid[i] = OBSTACLE;
    }

    int sr = 0, sc = 0;
    int tr = HEIGHT - 1, tc = WIDTH - 1;

    h_grid[sr*WIDTH + sc] = 0;
    h_grid[tr*WIDTH + tc] = UNVISITED;

    int* d_grid;
    int* d_changed;

    cudaMalloc(&d_grid, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_grid, h_grid.data(),
               WIDTH * HEIGHT * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 gridDim((WIDTH+15)/16, (HEIGHT+15)/16);

    int level = 0;
    int changed;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    do {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int),
                   cudaMemcpyHostToDevice);

        wave_step<<<gridDim, block>>>(d_grid, level, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&changed, d_changed, sizeof(int),
                   cudaMemcpyDeviceToHost);

        level++;

    } while (changed);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_grid.data(), d_grid,
               WIDTH * HEIGHT * sizeof(int),
               cudaMemcpyDeviceToHost);

    if (h_grid[tr*WIDTH + tc] != UNVISITED)
        std::cout << "Target reached at level "
                  << h_grid[tr*WIDTH + tc] << "\n";
    else
        std::cout << "No path found\n";

    std::cout << "GPU Execution Time: "
              << milliseconds / 1000.0 << " seconds\n";

    cudaFree(d_grid);
    cudaFree(d_changed);

    return 0;
}
