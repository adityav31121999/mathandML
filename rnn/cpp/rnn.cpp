
#include "rnn.hpp"

/**
 * @brief Default constructor for the rnn class. This constructor initializes the
 * recurrent neural network with the given parameters.
 * @param in number of inputs
 * @param hidden number of hidden neurons
 * @param out number of outputs
 * @param time_steps number of time steps for the recurrent network
 * @param epochs number of epochs for training
 * @param learning learning rate for the network
 */
rnn::rnn(unsigned int in, unsigned int hidden, unsigned int out, unsigned int time_steps, unsigned int epochs, double learning) {
    this->in = in;
    this->hidden = hidden;
    this->out = out;
    this->time_steps = time_steps;
    this->epochs = epochs;
    this->learning = learning;
    this->status = false;
    this->mse = 0.0;

    inputs.resize(time_steps, std::vector<double>(in, 0.0));
    expected.resize(time_steps, std::vector<double>(out, 0.0));
    
    // Initialize weight matrices using vectors
    Wxh.resize(hidden, std::vector<double>(in, 0.0));   // Input to hidden weights
    Whh.resize(hidden, std::vector<double>(hidden, 0.0)); // Hidden to hidden weights (recurrent)
    Why.resize(out, std::vector<double>(hidden, 0.0));  // Hidden to output weights
    
    // Initialize bias vectors
    bh.resize(hidden, 0.0);     // Hidden layer bias
    by.resize(out, 0.0);        // Output layer bias
    
    // Initialize gradient containers
    dWxh.resize(hidden, std::vector<double>(in, 0.0));
    dWhh.resize(hidden, std::vector<double>(hidden, 0.0));
    dWhy.resize(out, std::vector<double>(hidden, 0.0));
    dbh.resize(hidden, 0.0);
    dby.resize(out, 0.0);
    
    // Initialize containers for forward pass
    hidden_states.resize(time_steps + 1, std::vector<double>(hidden, 0.0)); // +1 for initial state
    outputs.resize(time_steps, std::vector<double>(out, 0.0));
    
    // Initialize weights with random values
    initializeWeights();
}

/**
 * @brief Constructor for the rnn class. This constructor initializes the
 * recurrent neural network with the given parameters and sets the inputs
 * and expected outputs.
 * @param inputs vector of input vectors for different time steps
 * @param expected vector of expected output vectors
 * @param hidden number of hidden neurons
 * @param time_steps number of time steps for the recurrent network
 * @param epochs number of epochs for training
 * @param learning learning rate for the network
 */
rnn::rnn(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected, unsigned int hidden, unsigned int time_steps, unsigned int epochs, double learning) {
    this->in = inputs[0].size();
    this->hidden = hidden;
    this->out = expected[0].size();
    this->time_steps = time_steps;
    this->epochs = epochs;
    this->learning = learning;
    this->status = false;
    this->mse = 0.0;
    
    // Store input and expected output data
    this->inputs = inputs;
    this->expected = expected;
    
    // Initialize weight matrices using vectors
    Wxh.resize(hidden, std::vector<double>(in, 0.0));   // Input to hidden weights
    Whh.resize(hidden, std::vector<double>(hidden, 0.0)); // Hidden to hidden weights (recurrent)
    Why.resize(out, std::vector<double>(hidden, 0.0));  // Hidden to output weights
    
    // Initialize bias vectors
    bh.resize(hidden, 0.0);     // Hidden layer bias
    by.resize(out, 0.0);        // Output layer bias
    
    // Initialize gradient containers
    dWxh.resize(hidden, std::vector<double>(in, 0.0));
    dWhh.resize(hidden, std::vector<double>(hidden, 0.0));
    dWhy.resize(out, std::vector<double>(hidden, 0.0));
    dbh.resize(hidden, 0.0);
    dby.resize(out, 0.0);
    
    // Initialize containers for forward pass
    hidden_states.resize(time_steps + 1, std::vector<double>(hidden, 0.0)); // +1 for initial state
    outputs.resize(time_steps, std::vector<double>(out, 0.0));
    
    // Initialize weights with random values
    initializeWeights();
}