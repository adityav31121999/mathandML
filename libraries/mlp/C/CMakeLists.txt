
# CMakeLists.txt for Multilayer Perceptron
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(MLP C)

# "include" folder
include_directories(include)

add_library(mlp SHARED
    # activation functions
    activations.c

    # mlp files
    mlp.c
    forprop.c
    backprop.c
    train.c
    weights.c
    loss.c
)
