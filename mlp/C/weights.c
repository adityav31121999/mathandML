
#include "include/mlp.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Generate a random number from a normal distribution
 * @param[in] mean The mean of the distribution
 * @param[in] stddev The standard deviation of the distribution
 * @return A random number from the specified normal distribution
 */
double normal_random(double mean, double stddev) {
    // Generate two uniform random numbers between 0 and 1
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    
    // Apply the Box-Muller transform to get a standard normal random number
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    
    // Scale and shift the number to match the desired mean and standard deviation
    return mean + stddev * z;
}


/**
 * @brief Initialize the weights of a neural network with random values
 * @param[in] net The MLP to initialize
 */
void initializeWeights(struct MLP* net) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize input to hidden weights
    for (int i = 0; i < net->neurons; i++) {
        for (int j = 0; j < net->in; j++) {
            // Initialize each weight with a random value between -1 and 1
            net->iweights[i][j] = normal_random(0.0, 1.0) * (j + 1);
        }
    }

    // Initialize hidden to hidden weights
    for (int i = 0; i < net->layers - 1; i++) {
        for (int j = 0; j < net->neurons; j++) {
            for (int k = 0; k < net->neurons; k++) {
                // Initialize each weight with a random value between -1 and 1
                net->weights[i][j][k] = (i + j + normal_random(0.0, 1.0)) / (k + 1);
            }
        }
    }

    // Initialize hidden to output weights
    for (int i = 0; i < net->out; i++) {
        for (int j = 0; j < net->neurons; j++) {
            // Initialize each weight with a random value between -1 and 1
            net->oweights[i][j] = normal_random(0.0, 1.0) * (j + 1);
        }
    }
}
