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
#ifndef RNN_HPP
#define RNN_HPP 1

#include <vector>
#include "activations.hpp"

/**
 * @brief Recurrent Neural Network class
 */
class rnn {
public:
// member variables
    unsigned int in;            // number of inputs
    unsigned int out;           // number of outputs
    unsigned int hidden;        // number of hidden units
    unsigned int time_steps;    // number of time steps for unfolding
    unsigned int epochs;        // number of epochs
    double mse;                 // mean square error
    double learning;            // learning rate
    bool status;                // 1 if completely trained
// member containers
    std::vector<std::vector<double>> inputs;       // sequence of input vectors
    std::vector<std::vector<double>> outputs;      // sequence of output vectors
    std::vector<std::vector<double>> expected;     // expected output sequences
    std::vector<std::vector<double>> hidden_states; // hidden states at each time step
    
    std::vector<std::vector<double>> Wxh;          // input to hidden weights
    std::vector<std::vector<double>> Whh;          // hidden to hidden weights (recurrent)
    std::vector<std::vector<double>> Why;          // hidden to output weights
    
    std::vector<double> bh;                        // hidden bias
    std::vector<double> by;                        // output bias
    
    // Gradients
    std::vector<std::vector<double>> dWxh;         // gradients for input to hidden weights
    std::vector<std::vector<double>> dWhh;         // gradients for hidden to hidden weights
    std::vector<std::vector<double>> dWhy;         // gradients for hidden to output weights
    std::vector<double> dbh;                       // gradients for hidden bias
    std::vector<double> dby;                       // gradients for output bias

// member functions
    // default constructor
    rnn() = default;
    rnn(unsigned int in, unsigned int hidden, unsigned int out, unsigned int time_steps, 
        unsigned int epochs, double learning);
    rnn(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected,
        unsigned int hidden, unsigned int time_steps, unsigned int epochs, double learning);

    double getL1Penalty();
    double getL2Penalty();

    void forward();                                // forward pass through time
    void backward();                               // backward pass through time (BPTT)
    void update_weights();                         // update weights after backprop
    void clip_gradients(double threshold);         // clip gradients to prevent explosion
    
    void train();
    void train(std::vector<std::vector<std::vector<double>>> sequences);
    void validate();
    void test();
    void initializeWeights();
    
    std::vector<double> predict(std::vector<std::vector<double>> input_sequence);
    
    // default destructor
    ~rnn() = default;
};

// rnn-related functions
double computeLossWithL1(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, rnn&, double);
double computeLossWithL2(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, rnn&, double);
double computePerplexity(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, rnn&);
double dropoutGeneralisation(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, rnn&, double);

#endif
