/**
 * @file rnn.hpp
 * Header file for the Recurrent Neural Network (RNN) class and its related functions.
 * This file contains the declaration of the RNN class, which is used to create
 * and manage a recurrent neural network. The file also includes
 * necessary headers and dependencies required for the RNN class.
 *
 * Dependencies:
 * - <activations.hpp>: For activation functions used in the neural network.
 *
 * The RNN class provides methods to initialize the network, perform forward
 * propagation through time, and apply backpropagation through time (BPTT).
 */
#ifndef RNN_H
#define RNN_H 1

#include <stdbool.h>
#include "activations.h"

/**
 * @brief Recurrent Neural Network structure
 */
typedef struct rnn {
    // member variables
    unsigned int in;            // number of inputs
    unsigned int out;           // number of outputs
    unsigned int hidden;        // number of hidden units
    unsigned int time_steps;    // number of time steps for unfolding
    unsigned int epochs;        // number of epochs
    double mse;                 // mean square error
    double learning;            // learning rate
    bool status;                // 1 if completely trained
    
    // member containers (dynamically allocated arrays)
    double** inputs;            // sequence of input vectors [time_steps][in]
    double** outputs;           // sequence of output vectors [time_steps][out]
    double** expected;          // expected output sequences [time_steps][out]
    double** hidden_states;     // hidden states at each time step [time_steps][hidden]
    
    double** Wxh;               // input to hidden weights [hidden][in]
    double** Whh;               // hidden to hidden weights (recurrent) [hidden][hidden]
    double** Why;               // hidden to output weights [out][hidden]
    
    double* bh;                 // hidden bias [hidden]
    double* by;                 // output bias [out]
    
    // Gradients
    double** dWxh;              // gradients for input to hidden weights [hidden][in]
    double** dWhh;              // gradients for hidden to hidden weights [hidden][hidden]
    double** dWhy;              // gradients for hidden to output weights [out][hidden]
    double* dbh;                // gradients for hidden bias [hidden]
    double* dby;                // gradients for output bias [out]
} rnn_t;

// Constructor functions
rnn_t* rnn_create(unsigned int in, unsigned int hidden, unsigned int out, unsigned int time_steps, unsigned int epochs, 
                            double learning);
rnn_t* rnn_create_with_data(double*** inputs, double*** expected,unsigned int in, unsigned int out, unsigned int hidden, 
                           unsigned int time_steps, unsigned int epochs, double learning);

// Member functions
double rnn_get_l1_penalty(rnn_t* rnn);
double rnn_get_l2_penalty(rnn_t* rnn);

void rnn_forward(rnn_t* rnn);                      // forward pass through time
void rnn_backward(rnn_t* rnn);                     // backward pass through time (BPTT)
void rnn_update_weights(rnn_t* rnn);               // update weights after backprop
void rnn_clip_gradients(rnn_t* rnn, double threshold); // clip gradients to prevent explosion

void rnn_train(rnn_t* rnn);
void rnn_train_sequences(rnn_t* rnn, double**** sequences, unsigned int num_sequences);
void rnn_validate(rnn_t* rnn);
void rnn_test(rnn_t* rnn);
void rnn_initialize_weights(rnn_t* rnn);

double** rnn_predict(rnn_t* rnn, double*** input_sequence);

// Destructor
void rnn_free(rnn_t* rnn);

// rnn-related functions
double compute_loss_with_l1(double*** outputs, double*** expected, rnn_t* rnn, double lambda);
double compute_loss_with_l2(double*** outputs, double*** expected, rnn_t* rnn, double lambda);
double compute_perplexity(double*** outputs, double*** expected, rnn_t* rnn);
double dropout_generalisation(double*** outputs, double*** expected, rnn_t* rnn, double dropout_rate);

#endif
