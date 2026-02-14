#include <iostream>
#include <vector>

#define OBSTACLE -1
#define UNVISITED -2

const int WIDTH = 8;
const int HEIGHT = 8;

inline int idx(int r, int c) {
    return r * WIDTH + c;
}

void print_grid(const std::vector<int>& grid) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            int val = grid[idx(r,c)];
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

    std::vector<int> grid(WIDTH * HEIGHT, UNVISITED);

    // Obstacles
    grid[idx(1,1)] = OBSTACLE;
    grid[idx(1,2)] = OBSTACLE;
    grid[idx(2,4)] = OBSTACLE;
    grid[idx(3,3)] = OBSTACLE;
    grid[idx(4,5)] = OBSTACLE;

    int sr = 0, sc = 0;
    int tr = 7, tc = 7;

    grid[idx(sr, sc)] = 0;

    int level = 0;
    bool changed;

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

    } while (changed && grid[idx(tr,tc)] == UNVISITED);

    print_grid(grid);

    if (grid[idx(tr,tc)] != UNVISITED)
        std::cout << "Target reached at level "
                  << grid[idx(tr,tc)] << "\n";
    else
        std::cout << "No path found\n";

    return 0;
}
