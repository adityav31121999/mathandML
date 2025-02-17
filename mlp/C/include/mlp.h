
#ifndef MLP_H
#define MLP_H 1

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "activations.h"

/**
 * @brief Multi-layer Perceptron struct (No Biases)
 */
struct MLP {
    // Member variables
    unsigned int in;            // Number of inputs
    unsigned int out;           // Number of outputs
    unsigned int layers;        // Number of layers
    unsigned int neurons;       // Number of neurons per layer
    unsigned int epochs;        // Number of epochs
    double mse;                 // Mean square error
    double learning;            // Learning rate
    bool status;                // Training completion flag

    // Member containers
    double* input;          // Input vector
    double* output;         // Output vector
    double* expected;       // Expected output
    double*** weights;      // Weight matrices for layers
    double** iweights;      // Input to hidden weights
    double** oweights;      // Hidden to output weights
    double** hlayers;       // Hidden layers
    double** activations;   // Activations for each layer
    double*** gweights;     // Gradient of weights for layers
    double** giweights;     // Gradient of input to hidden weights
    double** goweights;     // Gradient of hidden to output weights
};

// Function declarations

struct MLP* createMLP(unsigned int in, unsigned int out, unsigned int epochs, double learning);
void initializeWeights(struct MLP* net);
void forward(struct MLP* net);
void backward(struct MLP* net);
void backprop(struct MLP* net);
void backwithL1(struct MLP* net);
void backwithL2(struct MLP* net);
void rprop(struct MLP* net, double** gradients);
void train(struct MLP* net);
void validate(struct MLP* net);
void test(struct MLP* net);
void freeMLP(struct MLP* net);

// Loss functions

double computeLossWithL1(double* outputs, double* targets, int size, struct MLP* net, double lambda);
double computeLossWithL2(double* outputs, double* targets, int size, struct MLP* net, double lambda);
double dropoutGeneralisation(double* outputs, double* targets, int size, double dropout_rate);

#endif // MLP_H

