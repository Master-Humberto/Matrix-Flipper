#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define WINDOW_WIDTH 25
#define WINDOW_HEIGHT 25
#define NUM_PIXELS (WINDOW_WIDTH * WINDOW_HEIGHT)

#define SET_BIT(arr, idx) (arr[(idx) / 8] |= (1 << ((idx) % 8)))
#define GET_BIT(arr, idx) ((arr[(idx) / 8] >> ((idx) % 8)) & 1)

typedef struct {
    double weight;
    int x;
    int y;
    int idx;
} WeightInfo;

typedef struct {
    double time;
    int x;
    int y;
} TimeInfo;

void save_matrix_as_pgm_sequence(WeightInfo* weight_info_array, const char* filename, int width, int height);
double run_painting_iteration(double gamma, int minkowski_p, double learning_rate,
                              double epsilon, WeightInfo* weight_info_array, unsigned int* seed_ptr,
                              unsigned char* painted, TimeInfo* times_info,
                              int* painted_order, int* spiral_indices);
int* generate_spiral_indices();

int main(int argc, char* argv[]) {
    WeightInfo* weight_info_array = (WeightInfo*)malloc(NUM_PIXELS * sizeof(WeightInfo));
    if (!weight_info_array) {
        fprintf(stderr, "Failed to allocate memory for weight_info_array.\n");
        return 1;
    }

    int* spiral_indices = generate_spiral_indices();
    if (!spiral_indices) {
        fprintf(stderr, "Failed to generate spiral indices.\n");
        free(weight_info_array);
        return 1;
    }

    srand((unsigned int)time(NULL));
    for (int i = 0; i < NUM_PIXELS; i++) {
        weight_info_array[i].weight = (double)rand() / RAND_MAX;
        weight_info_array[i].idx = i;
        weight_info_array[i].x = i % WINDOW_WIDTH;
        weight_info_array[i].y = i / WINDOW_WIDTH;
    }

    size_t painted_array_size = (NUM_PIXELS + 7) / 8;
    unsigned char* painted = (unsigned char*)calloc(painted_array_size, sizeof(unsigned char));
    if (!painted) {
        fprintf(stderr, "Failed to allocate memory for painted bitset.\n");
        free(weight_info_array);
        free(spiral_indices);
        return 1;
    }

    TimeInfo* times_info = (TimeInfo*)malloc(NUM_PIXELS * sizeof(TimeInfo));
    if (!times_info) {
        fprintf(stderr, "Failed to allocate memory for times_info.\n");
        free(weight_info_array);
        free(painted);
        free(spiral_indices);
        return 1;
    }

    int* painted_order = (int*)malloc(NUM_PIXELS * sizeof(int));
    if (!painted_order) {
        fprintf(stderr, "Failed to allocate memory for painted_order.\n");
        free(weight_info_array);
        free(painted);
        free(times_info);
        free(spiral_indices);
        return 1;
    }

    double gamma_values[] = {0.5, 0.7, 0.9, 0.95, 0.99};
    int gamma_count = sizeof(gamma_values) / sizeof(gamma_values[0]);

    int minkowski_p_values[] = {1, 2, 3};
    int minkowski_p_count = sizeof(minkowski_p_values) / sizeof(minkowski_p_values[0]);

    double learning_rate_values[] = {0.01, 0.05, 0.1, 0.2};
    int learning_rate_count = sizeof(learning_rate_values) / sizeof(learning_rate_values[0]);

    double epsilon_values[] = {0.05, 0.1, 0.2, 0.3};
    int epsilon_count = sizeof(epsilon_values) / sizeof(epsilon_values[0]);

    double best_total_loss = INFINITY;
    int best_iteration = -1;
    double best_gamma = 0.0;
    int best_minkowski_p = 1;
    double best_learning_rate = 0.0;
    double best_epsilon = 0.0;
    WeightInfo* best_weight_info_array = (WeightInfo*)malloc(NUM_PIXELS * sizeof(WeightInfo));
    if (!best_weight_info_array) {
        fprintf(stderr, "Failed to allocate memory for best_weight_info_array.\n");
        free(weight_info_array);
        free(painted);
        free(times_info);
        free(painted_order);
        free(spiral_indices);
        return 1;
    }

    unsigned int seed = (unsigned int)time(NULL);

    save_matrix_as_pgm_sequence(weight_info_array, "initialized_weights", WINDOW_WIDTH, WINDOW_HEIGHT);

    int iteration = 0;
    for (int gi = 0; gi < gamma_count; gi++) {
        for (int mi = 0; mi < minkowski_p_count; mi++) {
            for (int li = 0; li < learning_rate_count; li++) {
                for (int ei = 0; ei < epsilon_count; ei++) {
                    double gamma = gamma_values[gi];
                    int minkowski_p = minkowski_p_values[mi];
                    double learning_rate = learning_rate_values[li];
                    double epsilon = epsilon_values[ei];

                    srand((unsigned int)time(NULL));
                    for (int i = 0; i < NUM_PIXELS; i++) {
                        weight_info_array[i].weight = (double)rand() / RAND_MAX;
                    }

                    double total_loss = 0.0;
                    for (int iter = 0; iter < 100; iter++) {
                        memset(painted, 0, painted_array_size);
                        memset(times_info, 0, NUM_PIXELS * sizeof(TimeInfo));
                        memset(painted_order, 0, NUM_PIXELS * sizeof(int));

                        seed = (unsigned int)time(NULL) + iter;

                        total_loss += run_painting_iteration(
                            gamma, minkowski_p, learning_rate, epsilon, weight_info_array, &seed,
                            painted, times_info, painted_order, spiral_indices);
                    }

                    total_loss /= 100.0;

                    iteration++;
                    printf("Hyperparameter Set %d completed. Average Total loss: %f\n", iteration, total_loss);
                    printf("Hyperparameters: GAMMA=%.2f, MINKOWSKI_P=%d, LEARNING_RATE=%.2f, EPSILON=%.2f\n",
                           gamma, minkowski_p, learning_rate, epsilon);

                    if (total_loss < best_total_loss) {
                        best_total_loss = total_loss;
                        best_iteration = iteration;
                        best_gamma = gamma;
                        best_minkowski_p = minkowski_p;
                        best_learning_rate = learning_rate;
                        best_epsilon = epsilon;
                        memcpy(best_weight_info_array, weight_info_array, NUM_PIXELS * sizeof(WeightInfo));
                    }
                }
            }
        }
    }

    printf("Training with best hyperparameters for 1000 iterations...\n");
    memcpy(weight_info_array, best_weight_info_array, NUM_PIXELS * sizeof(WeightInfo));
    for (int iter = 0; iter < 1000; iter++) {
        memset(painted, 0, painted_array_size);
        memset(times_info, 0, NUM_PIXELS * sizeof(TimeInfo));
        memset(painted_order, 0, NUM_PIXELS * sizeof(int));

        seed = (unsigned int)time(NULL) + iter;

        double total_loss = run_painting_iteration(
            best_gamma, best_minkowski_p, best_learning_rate, best_epsilon, weight_info_array, &seed,
            painted, times_info, painted_order, spiral_indices);
        printf("Iteration %d, Loss: %f\n", iter + 1, total_loss);
    }

    printf("\nBest Iteration: %d\n", best_iteration);
    printf("Best Total Loss: %f\n", best_total_loss);
    printf("Best Hyperparameters:\n");
    printf("GAMMA: %f\n", best_gamma);
    printf("MINKOWSKI_P: %d\n", best_minkowski_p);
    printf("LEARNING_RATE: %f\n", best_learning_rate);
    printf("EPSILON (Exploration Rate): %f\n", best_epsilon);

    save_matrix_as_pgm_sequence(best_weight_info_array, "best_weights", WINDOW_WIDTH, WINDOW_HEIGHT);

    free(weight_info_array);
    free(best_weight_info_array);
    free(painted);
    free(times_info);
    free(painted_order);
    free(spiral_indices);

    printf("Training completed. Exiting.\n");

    return 0;
}

double run_painting_iteration(double gamma, int minkowski_p, double learning_rate,
                              double epsilon, WeightInfo* weight_info_array, unsigned int* seed_ptr,
                              unsigned char* painted, TimeInfo* times_info,
                              int* painted_order, int* spiral_indices) {
    unsigned int seed = *seed_ptr;

    seed = seed * 1103515245 + 12345;
    *seed_ptr = seed;
    srand(seed);

    int painted_count = 0;

    for (int i = 0; i < NUM_PIXELS; i++) {
        times_info[i].time = 0.0;
        times_info[i].x = i % WINDOW_WIDTH;
        times_info[i].y = i / WINDOW_WIDTH;
    }

    double total_loss = 0.0;

    int current_idx = 0;
    int current_x = current_idx % WINDOW_WIDTH;
    int current_y = current_idx / WINDOW_WIDTH;

    while (painted_count < NUM_PIXELS) {
        SET_BIT(painted, current_idx);
        painted_order[painted_count++] = current_idx;

        clock_t start_time = clock();

        for (volatile int i = 0; i < 1000; i++);

        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        times_info[current_idx].time = elapsed_time;

        double loss = 0.0;

        for (int i = 0; i < painted_count - 1; i++) {
            int idx = painted_order[i];
            int step = painted_count - i - 1;
            loss += pow(gamma, step) * times_info[idx].time;
        }

        if (painted_count > 1) {
            int prev_idx = painted_order[painted_count - 2];
            int prev_x = prev_idx % WINDOW_WIDTH;
            int prev_y = prev_idx / WINDOW_WIDTH;

            double distance = pow(pow(abs(current_x - prev_x), minkowski_p) +
                                  pow(abs(current_y - prev_y), minkowski_p),
                                  1.0 / minkowski_p);
            loss += distance;
        }

        total_loss += loss;

        weight_info_array[current_idx].weight -= learning_rate * loss;
        weight_info_array[current_idx].weight = fmax(0.0, fmin(1.0, weight_info_array[current_idx].weight));

        int next_idx = -1;
        int attempts = 0;
        if ((double)rand() / RAND_MAX < epsilon) {
            do {
                next_idx = rand() % NUM_PIXELS;
                attempts++;
                if (attempts > NUM_PIXELS) {
                    break;
                }
            } while (GET_BIT(painted, next_idx));
        } else {
            next_idx = current_idx + 1;
            while (next_idx < NUM_PIXELS && GET_BIT(painted, next_idx)) {
                next_idx++;
            }
        }

        if (next_idx >= NUM_PIXELS || GET_BIT(painted, next_idx)) {
            break;
        }

        current_idx = next_idx;
        current_x = current_idx % WINDOW_WIDTH;
        current_y = current_idx / WINDOW_WIDTH;
    }

    return total_loss;
}

void save_matrix_as_pgm_sequence(WeightInfo* weight_info_array, const char* filename, int width, int height) {
    int num_pixels = width * height;

    int* indices = (int*)malloc(num_pixels * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Failed to allocate memory for indices.\n");
        return;
    }

    for (int i = 0; i < num_pixels; i++) {
        indices[i] = i;
    }

    for (int i = 0; i < num_pixels - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < num_pixels; j++) {
            if (weight_info_array[indices[j]].weight > weight_info_array[indices[max_idx]].weight) {
                max_idx = j;
            }
        }
        int temp_idx = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = temp_idx;
    }

    unsigned char* image = (unsigned char*)calloc(num_pixels, sizeof(unsigned char));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image.\n");
        free(indices);
        return;
    }

    for (int step = 0; step < num_pixels; step++) {
        int idx = indices[step];
        image[idx] = 255;

        char full_filename[256];
        sprintf(full_filename, "%s_%04d.pgm", filename, step);
        FILE* fp = fopen(full_filename, "wb");
        if (!fp) {
            fprintf(stderr, "Cannot open file %s for writing\n", full_filename);
            continue;
        }
        fprintf(fp, "P5\n%d %d\n255\n", width, height);
        fwrite(image, sizeof(unsigned char), num_pixels, fp);
        fclose(fp);
    }

    free(indices);
    free(image);
}

int* generate_spiral_indices() {
    int* spiral_indices = (int*)malloc(NUM_PIXELS * sizeof(int));
    if (!spiral_indices) {
        fprintf(stderr, "Failed to allocate memory for spiral_indices.\n");
        return NULL;
    }

    int x_start = 0, y_start = 0;
    int x_end = WINDOW_WIDTH - 1, y_end = WINDOW_HEIGHT - 1;
    int index = 0;

    while (x_start <= x_end && y_start <= y_end) {
        for (int x = x_start; x <= x_end; x++) {
            if (index >= NUM_PIXELS) break;
            spiral_indices[index++] = y_start * WINDOW_WIDTH + x;
        }
        y_start++;

        for (int y = y_start; y <= y_end; y++) {
            if (index >= NUM_PIXELS) break;
            spiral_indices[index++] = y * WINDOW_WIDTH + x_end;
        }
        x_end--;

        if (y_start <= y_end) {
            for (int x = x_end; x >= x_start; x--) {
                if (index >= NUM_PIXELS) break;
                spiral_indices[index++] = y_end * WINDOW_WIDTH + x;
            }
            y_end--;
        }

        if (x_start <= x_end) {
            for (int y = y_end; y >= y_start; y--) {
                if (index >= NUM_PIXELS) break;
                spiral_indices[index++] = y * WINDOW_WIDTH + x_start;
            }
            x_start++;
        }
    }

    return spiral_indices;
}