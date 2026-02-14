#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define OBSTACLE -1
#define UNVISITED -2

const int WIDTH = 8;
const int HEIGHT = 8;

__global__
void wave_step(int* grid, int level, int* changed) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    int idx = y * WIDTH + x;

    if (grid[idx] == level) {

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

void print_grid(const std::vector<int>& grid) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            int val = grid[r*WIDTH + c];
            if (val == OBSTACLE)
                std::cout << " X ";
            else if (val == UNVISITED)
                std::cout << " . ";
            else
                std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {

    std::vector<int> h_grid(WIDTH * HEIGHT, UNVISITED);

    h_grid[1*WIDTH + 1] = OBSTACLE;
    h_grid[1*WIDTH + 2] = OBSTACLE;
    h_grid[2*WIDTH + 4] = OBSTACLE;
    h_grid[3*WIDTH + 3] = OBSTACLE;
    h_grid[4*WIDTH + 5] = OBSTACLE;

    int sr = 0, sc = 0;
    int tr = 7, tc = 7;

    h_grid[sr*WIDTH + sc] = 0;

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

    cudaMemcpy(h_grid.data(), d_grid,
               WIDTH * HEIGHT * sizeof(int),
               cudaMemcpyDeviceToHost);

    print_grid(h_grid);

    if (h_grid[tr*WIDTH + tc] != UNVISITED)
        std::cout << "Target reached at level "
                  << h_grid[tr*WIDTH + tc] << "\n";
    else
        std::cout << "No path found\n";

    cudaFree(d_grid);
    cudaFree(d_changed);

    return 0;
}
