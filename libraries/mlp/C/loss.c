
#include "include/mlp.h"

/**
 * @brief Computes the L1 regularization penalty for a given MLP network
 * @param net MLP
 */
double getL1Penalty(struct MLP* net) {
    double penalty = 0.0;
    for (int i = 0; i < net->layers - 1; i++) {
        for (int j = 0; j < net->neurons; j++) {
            for (int k = 0; k < net->neurons; k++) {
                penalty += fabs(net->weights[i][j][k]);
            }
        }
    }
    return penalty;
}

/**
 * @brief Computes the L2 regularization penalty for a given MLP network
 * @param net MLP
 */
double getL2Penalty(struct MLP* net) {
    double penalty = 0.0;
    for (int i = 0; i < net->layers - 1; i++) {
        for (int j = 0; j < net->neurons; j++) {
            for (int k = 0; k < net->neurons; k++) {
                penalty += net->weights[i][j][k] * net->weights[i][j][k];
            }
        }
    }
    return penalty;
}

/**
 * @brief Computes the loss function with L1 regularization for a given MLP network
 * @param outputs The output of the network
 * @param targets The target values
 * @param size The size of the inputs
 * @param net The MLP network
 * @param lambda The regularization strength
 * @return The computed loss
 */
double computeLossWithL1(double* outputs, double* targets, int size, struct MLP* net, double lambda) {
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        loss += fabs(outputs[i] - targets[i]);
    }
    return loss + 0.5 * lambda * getL1Penalty(net);
}

/**
 * @brief Computes the loss function with L2 regularization for a given MLP network
 * @param outputs The output of the network
 * @param targets The target values
 * @param size The size of the inputs
 * @param net The MLP network
 * @param lambda The regularization strength
 * @return The computed loss
 */
double computeLossWithL2(double* outputs, double* targets, int size, struct MLP* net, double lambda) {
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        loss += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
    }
    return 0.5 * loss + 0.5 * lambda * getL2Penalty(net);
}

/**
 * @brief Computes the loss function with dropout generalization for a given MLP network
 * @param outputs The output of the network
 * @param targets The target values
 * @param size The size of the inputs
 * @param dropout_rate The dropout rate
 * @return The computed loss
 */
double dropoutGeneralisation(double* outputs, double* targets, int size, double dropout_rate) {
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        loss += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
    }
    return loss / (1.0 - dropout_rate);
}
