
# CMakeLists.txt for Multilayer Perceptron
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(RNN C)

# "include" folder
include_directories(include)

add_library(rnn STATIC
    # activation functions
    activations.c

    # mlp files
    rnn.c
    # forprop.cpp
    # backprop.cpp
    # train.cpp
    weights.cpp
    # loss.cpp
)
