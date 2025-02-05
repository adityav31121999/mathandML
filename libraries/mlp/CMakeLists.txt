
# CMakeLists.txt for Multilayer Perceptron
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(MLP C CXX)

# "include" folder
include_directories(include)

add_library(mlp STATIC
    # activation functions
    activations.cpp

    # mlp files
    mlp.cpp
    forprop.cpp
    backprop.cpp
    train.cpp
    weights.cpp
    loss.cpp
)
