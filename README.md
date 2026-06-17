# ML

A stb-style (for now) ML library built while learning. The Git commit history shows my learning path. Future updates depend on what I want to learn and what I want to change.

## About

ML is a small machine learning library written in C, inspired by tsoding's videos and project.

- https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw
- https://github.com/tsoding/nn.h

This project is not intended to be a production-ready machine learning framework. It is a learning-oriented project where I implement the basic parts of neural networks from scratch.

## Current Features

The current version includes:

1. Basic matrix structure and operations
2. Multi-layer neural network structure
3. Forward propagation
4. Finite difference gradient estimation
5. Backpropagation
6. Gradient storage and parameter updates
7. Mini-batch training
8. Multiple activation functions
9. Model saving and loading
10. Training data loading
11. Simple model verification
12. Image fitting and visualization experiments

## Logging

Define `ML_LOG_LEVEL` before including `ml.h` to control log output:

| Level | Output | Usage |
|---|---|---|
| `0` | none (zero overhead) | Release builds |
| `1` | `LOG_ERROR` | Production (default) |
| `2` | + `LOG_WARN` | Tuning |
| `3` | + `LOG_INFO` | Training progress |
| `4` | + `LOG_DEBUG` | Debugging |

example:
```sh
gcc -DML_LOG_LEVEL=3 ml.c -o ml -lm
```

Without `-D`, defaults to level 1.

## Build

A simple build script is provided:

```sh
./build.sh
```

You can also compile manually:

```sh
gcc -Wall -Wextra -O3 ml.c -o ml -lm
```

Some experiments may require external libraries such as `raylib`.

For example, if you want to build the visualization program, you may need something like:

```sh
gcc -Wall -Wextra -O3 train_visualizer.c -o train_visualizer -lraylib -lm
```

The exact command may depend on your system and installed libraries.

## Project Style

This project is currently written in a stb-style direction. This may change in the future. The current structure is not final, and future changes will depend on what I want to learn next.

## Notes

This repository is experimental. Code may be rewritten, renamed, removed, or reorganized at any time. The main purpose of the project is learning, not API stability. That said, you are welcome to fork this project and experiment with it yourself.

## Possible Future Directions

Some possible future directions include:

1. Cleaner stb-style organization
2. More flexible layer abstraction
3. Separate loss and optimizer modules
4. Better model serialization
5. More examples
6. GPU backend experiments
7. Transformer-related components
8. A more general ML library architecture

(None of these are guaranteed)

## License

This project is licensed under the MIT License.
