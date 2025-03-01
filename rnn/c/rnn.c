
#include "include/rnn.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/**
 * @brief Initialize the RNN structure.
 * @param rnn Pointer to the RNN structure.
 * @param input_size Size of the input layer.
 * @param hidden_size Size of the hidden layer.
 * @param output_size Size of the output layer.
 * @return 1 if successful, 0 otherwise.
 */
int rnn_init(rnn_t *rnn, int input_size, int hidden_size, int output_size) {
    if (!rnn) 
        return 0;
    
    // Initialize basic parameters
    rnn->in = input_size;
    rnn->hidden = hidden_size;
    rnn->out = output_size;
    rnn->time_steps = 1;  // Default value, can be changed later
    rnn->epochs = 100;    // Default value, can be changed later
    rnn->learning = 0.01; // Default learning rate
    rnn->mse = 0.0;
    rnn->status = false;
    
    // Allocate memory for weights
    rnn->Wxh = (double**)malloc(hidden_size * sizeof(double*));
    rnn->Whh = (double**)malloc(hidden_size * sizeof(double*));
    rnn->Why = (double**)malloc(output_size * sizeof(double*));
    
    if (!rnn->Wxh || !rnn->Whh || !rnn->Why) {
        return 0; // Memory allocation failed
    }
    
    // Allocate memory for each row of weights
    for (int i = 0; i < hidden_size; i++) {
        rnn->Wxh[i] = (double*)malloc(input_size * sizeof(double));
        rnn->Whh[i] = (double*)malloc(hidden_size * sizeof(double));
        if (!rnn->Wxh[i] || !rnn->Whh[i]) return 0;
    }
    
    for (int i = 0; i < output_size; i++) {
        rnn->Why[i] = (double*)malloc(hidden_size * sizeof(double));
        if (!rnn->Why[i]) return 0;
    }
    
    // Allocate memory for biases
    rnn->bh = (double*)malloc(hidden_size * sizeof(double));
    rnn->by = (double*)malloc(output_size * sizeof(double));
    
    if (!rnn->bh || !rnn->by) return 0;
    
    // Allocate memory for gradients
    rnn->dWxh = (double**)malloc(hidden_size * sizeof(double*));
    rnn->dWhh = (double**)malloc(hidden_size * sizeof(double*));
    rnn->dWhy = (double**)malloc(output_size * sizeof(double*));
    
    if (!rnn->dWxh || !rnn->dWhh || !rnn->dWhy) return 0;
    
    for (int i = 0; i < hidden_size; i++) {
        rnn->dWxh[i] = (double*)malloc(input_size * sizeof(double));
        rnn->dWhh[i] = (double*)malloc(hidden_size * sizeof(double));
        if (!rnn->dWxh[i] || !rnn->dWhh[i]) return 0;
    }
    
    for (int i = 0; i < output_size; i++) {
        rnn->dWhy[i] = (double*)malloc(hidden_size * sizeof(double));
        if (!rnn->dWhy[i]) return 0;
    }
    
    rnn->dbh = (double*)malloc(hidden_size * sizeof(double));
    rnn->dby = (double*)malloc(output_size * sizeof(double));
    
    if (!rnn->dbh || !rnn->dby) return 0;
    
    // Initialize other arrays to NULL (they'll be allocated when needed)
    rnn->inputs = NULL;
    rnn->outputs = NULL;
    rnn->expected = NULL;
    rnn->hidden_states = NULL;
    
    // Initialize weights with small random values
    srand(time(NULL));
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            rnn->Wxh[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        }
        for (int j = 0; j < hidden_size; j++) {
            rnn->Whh[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        }
        rnn->bh[i] = 0.0;
    }
    
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            rnn->Why[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        }
        rnn->by[i] = 0.0;
    }
    
    return 1; // Success
}