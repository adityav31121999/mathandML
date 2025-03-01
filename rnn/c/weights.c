
#include "include/rnn.h"
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
 * @brief Initialize the weights of a recurrent neural network with random values
 * @param[in] net The RNN to initialize
 */
void initializeWeights(struct rnn* net) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Xavier/Glorot initialization scaling factor for input weights
    double input_scale = sqrt(2.0 / (net->in + net->hidden));
    
    // Xavier/Glorot initialization scaling factor for recurrent weights
    double recurrent_scale = sqrt(2.0 / (net->hidden * 2));
    
    // Xavier/Glorot initialization scaling factor for output weights
    double output_scale = sqrt(2.0 / (net->hidden + net->out));

    // Initialize input to hidden weights
    for (int i = 0; i < net->hidden; i++) {
        for (int j = 0; j < net->in; j++) {
            net->inputs[i][j] = normal_random(0.0, input_scale);
        }
    }

    // Initialize hidden to hidden (recurrent) weights
    for (int i = 0; i < net->hidden; i++) {
        for (int j = 0; j < net->hidden; j++) {
            net->hidden_states[i][j] = normal_random(0.0, recurrent_scale);
        }
    }

    // Initialize hidden to output weights
    for (int i = 0; i < net->out; i++) {
        for (int j = 0; j < net->hidden; j++) {
            net->outputs[i][j] = normal_random(0.0, output_scale);
        }
    }
    
    // Initialize biases to zero
    for (int i = 0; i < net->hidden; i++) {
        net->bh[i] = 0.0;
    }
    
    for (int i = 0; i < net->out; i++) {
        net->by[i] = 0.0;
    }
}
