
#include "include/mlp.h"


/**
 * @brief Training function for MLP (error threshold: 10^-6)
 */
void train(struct MLP* net) {
    unsigned int e = 0;
    while (1) {
        forward(net);
        net->mse = MSE(net->expected, net->output, net->out);
        if (net->mse < 1e-6)
            break;
        printf("Rep. NO.:%u Errors: %f\n", e, net->mse);
        backward(net);
        e++;
    }
    net->epochs = e;
    forward(net);
}

/**
 * @brief Validation function for MLP
 */
void validate(struct MLP* net) {
    // Assuming validation data is available in some form
    double* validation_input = (double*)calloc(net->in, sizeof(double)); // Replace with actual validation input
    double* validation_expected = (double*)calloc(net->out, sizeof(double)); // Replace with actual expected output

    net->input = validation_input;
    net->expected = validation_expected;

    forward(net);

    double mse = 0.0;
    for (size_t i = 0; i < net->out; ++i) {
        mse += pow(net->expected[i] - net->output[i], 2);
    }
    mse /= net->out;

    printf("Validation MSE: %f\n", mse);

    free(validation_input);
    free(validation_expected);
}

/**
 * @brief Testing function for MLP
 */
void test(struct MLP* net) {
    // Assuming test data is available in some form
    double* test_input = (double*)calloc(net->in, sizeof(double)); // Replace with actual test input
    double* test_expected = (double*)calloc(net->out, sizeof(double)); // Replace with actual expected output

    net->input = test_input;
    net->expected = test_expected;

    forward(net);

    printf("Expected <-> Output\n");
    printf("Test Results:\n");
    for (size_t i = 0; i < net->out; ++i) {
        printf("%f <-> %f\n", net->expected[i], net->output[i]);
    }

    free(test_input);
    free(test_expected);
}
