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

// Function prototype
double run_painting_iteration(double gamma, int minkowski_p, double learning_rate,
                              double epsilon, WeightInfo* weight_info_array,
                              unsigned char* painted, TimeInfo* times_info,
                              int* painted_order);

// Comparator function for qsort
int compare_weights(const void* a, const void* b) {
    const WeightInfo* w1 = *(const WeightInfo**)a;
    const WeightInfo* w2 = *(const WeightInfo**)b;

    if (w1->weight < w2->weight) return 1;  // Descending order
    if (w1->weight > w2->weight) return -1;
    return 0;
}

void save_weights_as_pgm_sequence(WeightInfo* weight_info_array, int iteration, int width, int height) {
    int num_pixels = width * height;

    // Create an array of pointers to WeightInfo
    WeightInfo** weight_pointers = (WeightInfo**)malloc(num_pixels * sizeof(WeightInfo*));
    if (!weight_pointers) {
        fprintf(stderr, "Failed to allocate memory for weight_pointers.\n");
        return;
    }
    for (int i = 0; i < num_pixels; i++) {
        weight_pointers[i] = &weight_info_array[i];
    }

    // Sort the pointers based on weight in descending order
    qsort(weight_pointers, num_pixels, sizeof(WeightInfo*), compare_weights);

    // Create images for each step
    unsigned char* image = (unsigned char*)calloc(num_pixels, sizeof(unsigned char));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image.\n");
        free(weight_pointers);
        return;
    }

    memset(image, 0, num_pixels); // Initialize all pixels to black

    for (int step = 0; step < num_pixels; step++) {
        int x = weight_pointers[step]->x;
        int y = weight_pointers[step]->y;
        int image_idx = y * width + x;
        image[image_idx] = 255;  // Set the pixel at (x, y) to white

        char full_filename[256];
        sprintf(full_filename, "weights_%04d_%04d.pgm", iteration, step);
        FILE* fp = fopen(full_filename, "wb");
        if (!fp) {
            fprintf(stderr, "Cannot open file %s for writing\n", full_filename);
            continue;
        }
        fprintf(fp, "P5\n%d %d\n255\n", width, height);
        fwrite(image, sizeof(unsigned char), num_pixels, fp);
        fclose(fp);
    }

    free(weight_pointers);
    free(image);
}

void create_gif_from_pgm_images(int iteration) {
    char command[512];
    sprintf(command, "convert -delay 10 -loop 0 weights_%04d_*.pgm weights_%04d.gif", iteration, iteration);
    system(command);
}

void delete_pgm_files(int iteration) {
    char command[256];
    sprintf(command, "rm weights_%04d_*.pgm", iteration);
    system(command);
}

void run_iterations(int total_iterations, int width, int height, WeightInfo* weight_info_array,
                    double gamma, int minkowski_p, double learning_rate, double epsilon) {
    // Initialize necessary variables
    size_t painted_array_size = (NUM_PIXELS + 7) / 8;
    unsigned char* painted = (unsigned char*)calloc(painted_array_size, sizeof(unsigned char));
    TimeInfo* times_info = (TimeInfo*)malloc(NUM_PIXELS * sizeof(TimeInfo));
    int* painted_order = (int*)malloc(NUM_PIXELS * sizeof(int));

    if (!painted || !times_info || !painted_order) {
        fprintf(stderr, "Failed to allocate memory for iteration variables.\n");
        free(painted);
        free(times_info);
        free(painted_order);
        return;
    }

    for (int iter = 1; iter <= total_iterations; iter++) {
        printf("Debug: Starting iteration %d\n", iter);
        memset(painted, 0, painted_array_size);
        memset(times_info, 0, NUM_PIXELS * sizeof(TimeInfo));
        memset(painted_order, 0, NUM_PIXELS * sizeof(int));

        double total_loss = run_painting_iteration(
            gamma, minkowski_p, learning_rate, epsilon, weight_info_array,
            painted, times_info, painted_order);
        printf("Iteration %d, Loss: %f\n", iter, total_loss);

        // Save images and create GIFs at specified iterations
        if (iter % 100 == 0) {
            printf("Debug: Saving images for iteration %d\n", iter);
            save_weights_as_pgm_sequence(weight_info_array, iter, width, height);
            create_gif_from_pgm_images(iter);
            delete_pgm_files(iter);
            printf("Debug: Finished saving images for iteration %d\n", iter);
        }
    }

    free(painted);
    free(times_info);
    free(painted_order);
}

int main(int argc, char* argv[]) {
    WeightInfo* weight_info_array = (WeightInfo*)malloc(NUM_PIXELS * sizeof(WeightInfo));
    if (!weight_info_array) {
        fprintf(stderr, "Failed to allocate memory for weight_info_array.\n");
        return 1;
    }

    // Seed the random number generator once
    srand((unsigned int)time(NULL));

    for (int i = 0; i < NUM_PIXELS; i++) {
        weight_info_array[i].weight = (double)rand() / RAND_MAX;
        weight_info_array[i].idx = i;
        weight_info_array[i].x = i % WINDOW_WIDTH;
        weight_info_array[i].y = i / WINDOW_WIDTH;
    }

    // Save the initial weights as GIF
    save_weights_as_pgm_sequence(weight_info_array, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    create_gif_from_pgm_images(0);
    delete_pgm_files(0);

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
        return 1;
    }

    int iteration = 0;
    for (int gi = 0; gi < gamma_count; gi++) {
        for (int mi = 0; mi < minkowski_p_count; mi++) {
            for (int li = 0; li < learning_rate_count; li++) {
                for (int ei = 0; ei < epsilon_count; ei++) {
                    double gamma = gamma_values[gi];
                    int minkowski_p = minkowski_p_values[mi];
                    double learning_rate = learning_rate_values[li];
                    double epsilon = epsilon_values[ei];

                    // Re-randomize weights before each hyperparameter set
                    for (int i = 0; i < NUM_PIXELS; i++) {
                        weight_info_array[i].weight = (double)rand() / RAND_MAX;
                    }

                    double total_loss = 0.0;
                    for (int iter = 0; iter < 100; iter++) {
                        // Initialize variables for each iteration
                        size_t painted_array_size = (NUM_PIXELS + 7) / 8;
                        unsigned char* painted = (unsigned char*)calloc(painted_array_size, sizeof(unsigned char));
                        TimeInfo* times_info = (TimeInfo*)malloc(NUM_PIXELS * sizeof(TimeInfo));
                        int* painted_order = (int*)malloc(NUM_PIXELS * sizeof(int));

                        if (!painted || !times_info || !painted_order) {
                            fprintf(stderr, "Failed to allocate memory for iteration variables.\n");
                            free(painted);
                            free(times_info);
                            free(painted_order);
                            continue;
                        }

                        total_loss += run_painting_iteration(
                            gamma, minkowski_p, learning_rate, epsilon, weight_info_array,
                            painted, times_info, painted_order);

                        // Free allocated memory
                        free(painted);
                        free(times_info);
                        free(painted_order);
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

    run_iterations(1000, WINDOW_WIDTH, WINDOW_HEIGHT, weight_info_array,
                   best_gamma, best_minkowski_p, best_learning_rate, best_epsilon);

    printf("\nBest Iteration: %d\n", best_iteration);
    printf("Best Total Loss: %f\n", best_total_loss);
    printf("Best Hyperparameters:\n");
    printf("GAMMA: %f\n", best_gamma);
    printf("MINKOWSKI_P: %d\n", best_minkowski_p);
    printf("LEARNING_RATE: %f\n", best_learning_rate);
    printf("EPSILON (Exploration Rate): %f\n", best_epsilon);

    // Save the final weights
    save_weights_as_pgm_sequence(weight_info_array, 1000, WINDOW_WIDTH, WINDOW_HEIGHT);
    create_gif_from_pgm_images(1000);
    delete_pgm_files(1000);

    free(weight_info_array);
    free(best_weight_info_array);

    printf("Training completed. Exiting.\n");

    return 0;
}

double run_painting_iteration(double gamma, int minkowski_p, double learning_rate,
                              double epsilon, WeightInfo* weight_info_array,
                              unsigned char* painted, TimeInfo* times_info,
                              int* painted_order) {
    int painted_count = 0;

    for (int i = 0; i < NUM_PIXELS; i++) {
        times_info[i].time = 0.0;
        times_info[i].x = i % WINDOW_WIDTH;
        times_info[i].y = i / WINDOW_WIDTH;
    }

    double total_loss = 0.0;

    // Start from the unpainted pixel with the highest weight
    int current_idx = -1;
    double max_weight = -INFINITY;
    for (int i = 0; i < NUM_PIXELS; i++) {
        if (!GET_BIT(painted, i) && weight_info_array[i].weight > max_weight) {
            max_weight = weight_info_array[i].weight;
            current_idx = i;
        }
    }
    if (current_idx == -1) {
        // All pixels are painted
        return total_loss;
    }
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

        if ((double)rand() / RAND_MAX < epsilon) {
            // Exploration: pick a random unpainted pixel
            int attempts = 0;
            do {
                next_idx = rand() % NUM_PIXELS;
                attempts++;
                if (attempts > NUM_PIXELS) {
                    break;
                }
            } while (GET_BIT(painted, next_idx));
        } else {
            // Exploitation: pick the unpainted pixel with the highest weight
            double max_weight = -INFINITY;
            for (int i = 0; i < NUM_PIXELS; i++) {
                if (!GET_BIT(painted, i) && weight_info_array[i].weight > max_weight) {
                    max_weight = weight_info_array[i].weight;
                    next_idx = i;
                }
            }
            if (next_idx == -1) {
                // All pixels are painted
                break;
            }
        }

        if (next_idx == -1 || GET_BIT(painted, next_idx)) {
            // No unpainted pixels left
            break;
        }

        current_idx = next_idx;
        current_x = current_idx % WINDOW_WIDTH;
        current_y = current_idx / WINDOW_WIDTH;
    }

    return total_loss;
}