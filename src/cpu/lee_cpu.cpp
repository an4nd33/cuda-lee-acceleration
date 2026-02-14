#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define OBSTACLE -1
#define UNVISITED -2

const int WIDTH = 512;
const int HEIGHT = 512;

inline int idx(int r, int c) {
    return r * WIDTH + c;
}

int main() {

    std::vector<int> grid(WIDTH * HEIGHT, UNVISITED);

    srand((unsigned)time(0));

    // 30% random obstacles
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (rand() % 100 < 30)
            grid[i] = OBSTACLE;
    }

    int sr = 0, sc = 0;
    int tr = HEIGHT - 1, tc = WIDTH - 1;

    grid[idx(sr, sc)] = 0;
    grid[idx(tr, tc)] = UNVISITED;

    int level = 0;
    bool changed;

    auto start = std::chrono::high_resolution_clock::now();

    do {
        changed = false;

        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {

                if (grid[idx(r,c)] == level) {

                    int dr[4] = {-1, 1, 0, 0};
                    int dc[4] = {0, 0, -1, 1};

                    for (int k = 0; k < 4; k++) {

                        int nr = r + dr[k];
                        int nc = c + dc[k];

                        if (nr >= 0 && nr < HEIGHT &&
                            nc >= 0 && nc < WIDTH) {

                            if (grid[idx(nr,nc)] == UNVISITED) {
                                grid[idx(nr,nc)] = level + 1;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        level++;

    } while (changed);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (grid[idx(tr,tc)] != UNVISITED)
        std::cout << "Target reached at level "
                  << grid[idx(tr,tc)] << "\n";
    else
        std::cout << "No path found\n";

    std::cout << "CPU Execution Time: "
              << duration.count() << " seconds\n";

    return 0;
}
