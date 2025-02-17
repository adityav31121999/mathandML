
#include "include/mlp.h"


/**
 * @brief The backward propagation function. This function performs the
 * backward propagation and calculates the error.
 */
void backward(struct MLP* net) {
    int i, j, k;
    double* error = (double*)malloc(net->out * sizeof(double));
    double** ow = (double**)malloc(net->out * sizeof(double*));
    for (i = 0; i < net->out; i++) {
        ow[i] = (double*)calloc(net->neurons, sizeof(double));
    }
    double* dh1 = (double*)calloc(net->neurons, sizeof(double));
    
    // Calculate output layer error
    for (i = 0; i < net->out; i++) {
        error[i] = net->expected[i] - net->output[i];
    }

    // Calculate gradients for output weights
    for (i = 0; i < net->out; i++) {
        for (j = 0; j < net->neurons; j++) {
            ow[i][j] = error[i] * net->activations[net->layers - 2][j];
        }
    }

    // Calculate hidden layer deltas
    for (i = 0; i < net->neurons; i++) {
        double sum = 0.0;
        for (j = 0; j < net->out; j++) {
            sum += error[j] * net->oweights[j][i];
        }
        dh1[i] = sum * sigmoidder(net->hlayers[net->layers - 2][i]);
    }

    // Update output weights
    for (i = 0; i < net->out; i++) {
        for (j = 0; j < net->neurons; j++) {
            net->oweights[i][j] += net->learning * ow[i][j];
        }
    }

    // Handle hidden layers
    double*** dweights = (double***)malloc((net->layers - 1) * sizeof(double**));
    for (i = 0; i < net->layers - 1; i++) {
        dweights[i] = (double**)malloc(net->neurons * sizeof(double*));
        for (j = 0; j < net->neurons; j++) {
            dweights[i][j] = (double*)calloc(net->neurons, sizeof(double));
        }
    }
    double* layer_error = dh1;

    for (i = net->layers - 2; i >= 0; i--) {
        double* next_error = (double*)calloc(net->neurons, sizeof(double));
        
        for (j = 0; j < net->neurons; j++) {
            for (k = 0; k < net->neurons; k++) {
                dweights[i][j][k] = layer_error[k] * net->activations[i][j];
            }
        }

        for (j = 0; j < net->neurons; j++) {
            double sum = 0.0;
            for (k = 0; k < net->neurons; k++) {
                sum += layer_error[k] * net->weights[i][k][j];
            }
            next_error[j] = sum * sigmoidder(net->hlayers[i][j]);
        }
        
        free(layer_error);
        layer_error = next_error;
    }

    // Update hidden layer weights
    for (i = net->layers - 2; i >= 0; i--) {
        for (j = 0; j < net->neurons; j++) {
            for (k = 0; k < net->neurons; k++) {
                net->weights[i][j][k] += net->learning * dweights[i][j][k];
            }
        }
    }

    // Update input weights
    for (i = 0; i < net->neurons; i++) {
        for (j = 0; j < net->in; j++) {
            net->iweights[i][j] += net->learning * layer_error[i];
        }
    }

    free(error);
    free(dh1);
    for (i = 0; i < net->out; i++) {
        free(ow[i]);
    }
    free(ow);
    for (i = 0; i < net->layers - 1; i++) {
        for (j = 0; j < net->neurons; j++) {
            free(dweights[i][j]);
        }
        free(dweights[i]);
    }
    free(dweights);
    free(layer_error);
}

/**
 * @brief Backpropagation with gradients
 */
void backprop(struct MLP* net) {
    // Compute output layer error
    double output_error[net->out];
    for (unsigned int i = 0; i < net->out; ++i) {
        output_error[i] = net->output[i] - net->expected[i];
    }
    // Compute output layer gradients
    for (unsigned int i = 0; i < net->out; ++i) {
        for (unsigned int j = 0; j < net->neurons; ++j) {
            net->goweights[i][j] = output_error[i] * net->activations[net->layers - 1][j];
        }
    }

    // Compute hidden layer gradients
    double hidden_error[net->neurons];
    for (unsigned int i = 0; i < net->neurons; ++i) {
        double error_sum = 0.0;
        for (unsigned int j = 0; j < net->out; ++j) {
            error_sum += net->oweights[j][i] * output_error[j];
        }
        hidden_error[i] = error_sum * net->activations[net->layers - 1][i] * (1 - net->activations[net->layers - 1][i]);
    }
    
    // Compute gradients for the hidden layer
    for (unsigned int i = 0; i < net->neurons; ++i) {
        for (unsigned int j = 0; j < net->in; ++j) {
            net->giweights[i][j] = hidden_error[i] * net->input[j];
        }
    }
    
    // Propagate error backward through the network
    for (int l = net->layers - 2; l >= 0; --l) {
        double layer_error[net->neurons];
        for (unsigned int i = 0; i < net->neurons; ++i) {
            double error_sum = 0.0;
            for (unsigned int j = 0; j < net->neurons; ++j) {
                error_sum += net->weights[l][j][i] * net->gweights[l][j][i];
            }
            layer_error[i] = error_sum * net->activations[l][i] * (1 - net->activations[l][i]);
        }
        
        // Compute gradients for the current layer
        for (unsigned int i = 0; i < net->neurons; ++i) {
            for (unsigned int j = 0; j < net->neurons; ++j) {
                net->gweights[l][i][j] = layer_error[i] * net->activations[l][j];
            }
        }
    }
}


/**
 * @brief Perform backpropagation with L1 regularization
 */
void backwithL1(struct MLP* net) {
    double lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop(net);

    // Update weights with L1 regularization
    for (unsigned int l = 0; l < net->layers - 1; ++l) {
        for (unsigned int i = 0; i < net->neurons; ++i) {
            for (unsigned int j = 0; j < net->neurons; ++j) {
                double gradient = net->gweights[l][i][j];
                if (net->weights[l][i][j] > 0) {
                    net->weights[l][i][j] -= net->learning * (lambda + gradient);
                } else {
                    net->weights[l][i][j] -= net->learning * (-lambda + gradient);
                }
            }
        }
    }

    for (unsigned int i = 0; i < net->neurons; ++i) {
        for (unsigned int j = 0; j < net->in; ++j) {
            double gradient = net->giweights[i][j];
            if (net->iweights[i][j] > 0) {
                net->iweights[i][j] -= net->learning * (lambda + gradient);
            } else {
                net->iweights[i][j] -= net->learning * (-lambda + gradient);
            }
        }
    }

    for (unsigned int i = 0; i < net->out; ++i) {
        for (unsigned int j = 0; j < net->neurons; ++j) {
            double gradient = net->goweights[i][j];
            if (net->oweights[i][j] > 0) {
                net->oweights[i][j] -= net->learning * (lambda + gradient);
            } else {
                net->oweights[i][j] -= net->learning * (-lambda + gradient);
            }
        }
    }
}

void backwithL2(struct MLP* net) {
    double lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop(net);

    // Compute L2 penalty
    double l2_penalty = 0.0;
    for (unsigned int l = 0; l < net->layers - 1; ++l) {
        for (unsigned int i = 0; i < net->neurons; ++i) {
            for (unsigned int j = 0; j < net->neurons; ++j) {
                l2_penalty += net->weights[l][i][j] * net->weights[l][i][j];
            }
        }
    }
    for (unsigned int i = 0; i < net->neurons; ++i) {
        for (unsigned int j = 0; j < net->in; ++j) {
            l2_penalty += net->iweights[i][j] * net->iweights[i][j];
        }
    }
    for (unsigned int i = 0; i < net->out; ++i) {
        for (unsigned int j = 0; j < net->neurons; ++j) {
            l2_penalty += net->oweights[i][j] * net->oweights[i][j];
        }
    }

    // Update weights with L2 regularization
    for (unsigned int l = 0; l < net->layers - 1; ++l) {
        for (unsigned int i = 0; i < net->neurons; ++i) {
            for (unsigned int j = 0; j < net->neurons; ++j) {
                double gradient = net->gweights[l][i][j];
                net->weights[l][i][j] -= net->learning * (lambda * net->weights[l][i][j] + gradient);
            }
        }
    }
    for (unsigned int i = 0; i < net->neurons; ++i) {
        for (unsigned int j = 0; j < net->in; ++j) {
            double gradient = net->giweights[i][j];
            net->iweights[i][j] -= net->learning * (lambda * net->iweights[i][j] + gradient);
        }
    }
    for (unsigned int i = 0; i < net->out; ++i) {
        for (unsigned int j = 0; j < net->neurons; ++j) {
            double gradient = net->goweights[i][j];
            net->oweights[i][j] -= net->learning * (lambda * net->oweights[i][j] + gradient);
        }
    }

    // Compute loss with L2 penalty
    double loss = computeLossWithL2(net->output, net->expected, net->in, net, lambda);
    printf("Loss with L2 penalty: %f\n", loss);
}


/**
 * @brief Rprop algorithm for MLP
 * @param net The MLP network
 * @param gradients The gradient matrix
 * @note This version only updates the weights and doesn't update the bias
 */
void rprop(struct MLP* net, double** gradients) {
    const double etaPlus = 1.2;     // Increase factor
    const double etaMinus = 0.5;    // Decrease factor
    const double deltaMax = 50.0;   // Maximum update value
    const double deltaMin = 1e-6;   // Minimum update value

    double** deltaWeights = (double**)malloc(net->out * sizeof(double*));
    for (unsigned int i = 0; i < net->out; ++i) {
        deltaWeights[i] = (double*)malloc(net->neurons * sizeof(double));
        for (unsigned int j = 0; j < net->neurons; ++j) {
            deltaWeights[i][j] = deltaMin;
        }
    }

    for (unsigned int epoch = 0; epoch < net->epochs; ++epoch) {
        double totalError = 0.0;
        forward(net);
        backward(net);

        // Compute mean square error
        double error = 0.0;
        for (unsigned int i = 0; i < net->out; ++i) {
            double outputError = (net->expected[i] - net->output[i]);
            error += pow(outputError, 2);
        }
        error /= net->out;
        totalError += error;

        // Update weights using Rprop
        for (unsigned int i = 0; i < net->out; ++i) {
            for (unsigned int j = 0; j < net->neurons; ++j) {
                double grad = gradients[i][j];
                if (grad * net->goweights[i][j] > 0) {
                    deltaWeights[i][j] = fmin(deltaWeights[i][j] * etaPlus, deltaMax);
                    net->oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                    net->goweights[i][j] = grad;
                } else if (grad * net->goweights[i][j] < 0) {
                    deltaWeights[i][j] = fmax(deltaWeights[i][j] * etaMinus, deltaMin);
                    net->oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                    net->goweights[i][j] = 0;
                } else {
                    net->oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                    net->goweights[i][j] = grad;
                }
            }
        }

        totalError /= net->epochs;
        printf("Epoch %u/%u - Mean Squared Error: %f\n", epoch + 1, net->epochs, totalError);
        if (totalError < 0.01) {
            net->status = true;
            break;
        }
    }

    for (unsigned int i = 0; i < net->out; ++i) {
        free(deltaWeights[i]);
    }
    free(deltaWeights);
}
