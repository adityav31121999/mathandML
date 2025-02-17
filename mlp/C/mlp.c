
#include "include/mlp.h"

/**
 * @brief Creates a new Multi-Layer Perceptron (MLP) instance
 * @param[in] in The number of inputs
 * @param[in] out The number of outputs
 * @param[in] epochs The number of epochs to train
 * @param[in] learning The learning rate
 * @return A pointer to a newly allocated MLP instance, or NULL if the allocation fails
 */
struct MLP* createMLP(unsigned int in, unsigned int out, unsigned int epochs, double learning) {
    struct MLP* mlp = (struct MLP*)malloc(sizeof(struct MLP));
    if (!mlp) return NULL;

    // Set the number of inputs, outputs, layers, neurons, epochs, and learning rate
    mlp->in = in;
    mlp->out = out;
    mlp->layers = in + out;
    mlp->neurons = in * out;
    mlp->epochs = epochs;
    mlp->learning = learning;

    // Allocate memory for the input, output, and expected arrays
    mlp->input = (double*)calloc(in, sizeof(double));
    mlp->output = (double*)calloc(out, sizeof(double));
    mlp->expected = (double*)calloc(out, sizeof(double));

    // Allocate memory for the weights, gradients, and hidden layers
    mlp->iweights = (double**)malloc(mlp->neurons * sizeof(double*));
    mlp->oweights = (double**)malloc(out * sizeof(double*));
    mlp->weights = (double***)malloc((mlp->layers - 1) * sizeof(double**));
    mlp->hlayers = (double**)malloc(mlp->layers * sizeof(double*));
    mlp->activations = (double**)malloc(mlp->layers * sizeof(double*));
    mlp->giweights = (double**)malloc(mlp->neurons * sizeof(double*));
    mlp->goweights = (double**)malloc(out * sizeof(double*));
    mlp->gweights = (double***)malloc((mlp->layers - 1) * sizeof(double**));

    // Initialize the weights and gradients
    for (unsigned int i = 0; i < mlp->neurons; ++i) {
        mlp->iweights[i] = (double*)calloc(in, sizeof(double));
        mlp->giweights[i] = (double*)calloc(in, sizeof(double));
    }
    for (unsigned int i = 0; i < out; ++i) {
        mlp->oweights[i] = (double*)calloc(mlp->neurons, sizeof(double));
        mlp->goweights[i] = (double*)calloc(mlp->neurons, sizeof(double));
    }
    for (unsigned int i = 0; i < mlp->layers - 1; ++i) {
        mlp->weights[i] = (double**)malloc(mlp->neurons * sizeof(double*));
        mlp->gweights[i] = (double**)malloc(mlp->neurons * sizeof(double*));
        for (unsigned int j = 0; j < mlp->neurons; ++j) {
            mlp->weights[i][j] = (double*)calloc(mlp->neurons, sizeof(double));
            mlp->gweights[i][j] = (double*)calloc(mlp->neurons, sizeof(double));
        }
    }
    for (unsigned int i = 0; i < mlp->layers; ++i) {
        mlp->hlayers[i] = (double*)calloc(mlp->neurons, sizeof(double));
        mlp->activations[i] = (double*)calloc(mlp->neurons, sizeof(double));
    }

    // Initialize the weights
    initializeWeights(mlp);

    return mlp;
}

/**
 * @brief Frees the memory allocated for a MLP instance
 * @param[in] net The MLP instance to free
 */
void freeMLP(struct MLP *net) {
    if (net == NULL) {
        return;
    }

    // Free the weights
    for (unsigned int i = 0; i < net->neurons; ++i) {
        free(net->iweights[i]);
        free(net->giweights[i]);
    }
    for (unsigned int i = 0; i < net->out; ++i) {
        free(net->oweights[i]);
        free(net->goweights[i]);
    }
    for (unsigned int i = 0; i < net->layers - 1; ++i) {
        for (unsigned int j = 0; j < net->neurons; ++j) {
            free(net->weights[i][j]);
            free(net->gweights[i][j]);
        }
        free(net->weights[i]);
        free(net->gweights[i]);
    }
    free(net->weights);
    free(net->gweights);

    // Free the hidden layers
    for (unsigned int i = 0; i < net->layers; ++i) {
        free(net->hlayers[i]);
        free(net->activations[i]);
    }
    free(net->hlayers);
    free(net->activations);

    // Free the input and output
    free(net->input);
    free(net->output);
    free(net->expected);

    // Free the MLP struct
    free(net);
}
