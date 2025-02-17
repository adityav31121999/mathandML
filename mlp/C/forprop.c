
#include "include/mlp.h"

/**
 * @brief Perform forward propagation through the MLP network. This function 
 * performs a single forward pass through the MLP, computing the activations 
 * for each layer and the output of the network.
 * @param net the MLP on which forward propagation is to be performed
 */
void forward(struct MLP* net) {
    // Compute activations for first hidden layer
    // Compute activations for first hidden layer
    for (unsigned int i = 0; i < net->neurons; i++) {
        double sum = 0.0;
        // Sum the weighted inputs for each neuron in the first hidden layer
        for (unsigned int j = 0; j < net->in; j++) {
            sum += net->input[j] * net->iweights[i][j];
        }
        net->hlayers[0][i] = sum;
        net->activations[0][i] = sigmoid(sum);
    }

    // Compute activations for hidden layers
    for (unsigned int i = 1; i < net->layers - 1; i++) {
        for (unsigned int j = 0; j < net->neurons; j++) {
            double sum = 0.0;
            for (unsigned int k = 0; k < net->neurons; k++) {
                sum += net->activations[i - 1][k] * net->weights[i - 1][j][k];
            }
            net->hlayers[i][j] = sum;
            net->activations[i][j] = sigmoid(sum);
        }
    }

    // Compute output layer activations
    for (unsigned int i = 0; i < net->out; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < net->neurons; j++) {
            sum += net->activations[net->layers - 2][j] * net->oweights[i][j];
        }
        net->output[i] = sum;
    }
}
