# ML

A stb-style (for now) ML library built while learning. The Git commit history shows my learning path. Future updates depend on what I want to learn and what I want to change.

## About

ML is a small machine learning library written in C.

This project is not intended to be a production-ready machine learning framework. It is a learning-oriented project where I implement the basic parts of neural networks from scratch.

The repository is also meant to be read through its Git history. Each commit represents a different stage of my learning process, from finite difference training to backpropagation, mini-batch training, model serialization, visualization experiments, and later architectural changes.

## Learning Through Commits

The commit history is part of this project.

You can read the commits in order to follow how the library evolved:

```sh
git log --oneline --reverse
```

Current learning stages include:

```text
feat: add finite difference training
feat: implement backpropagation
feat: add model serialization and training data import
feat: add mini-batch training and image training visualization
refactor: add layer abstraction
docs: add README.md and LICENSE
```

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

## Repository Structure

```text
.
├── ml.h                # Main stb-style ML library header
├── ml.c                # Main training / experiment program
├── train_visualizer.c  # Raylib-based training visualization experiment
├── build.sh            # Simple build script
└── README.md
```

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

## Usage

The project is still changing, so the API is not stable yet.

A typical usage pattern looks like this:

```c
#define ML_IMPLEMENTATION
#include "ml.h"

int main(void) {
  size_t layers[] = {2, 16, 16, 1};

  Act acts[] = {
      ML_LRELU,
      ML_LRELU,
      ML_SIGMOID,
  };

  Mod m = ml_model_alloc(layers, sizeof(layers) / sizeof(layers[0]), 1, acts);
  Grad g = ml_grad_alloc(m);

  ml_model_rand(m, -1.0f, 1.0f);

  // Train or run the model here.

  ml_grad_free(&g);
  ml_model_free(&m);

  return 0;
}
```

The exact API may change as the project evolves.

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
