# Graphzy

A spring-physics based graph visualization tool built with Raylib.

## Requirements

- CMake 3.10 or higher
- C compiler (clang or gcc)
- Raylib library

## Building

### Install Raylib (macOS)

```bash
brew install raylib
```

### Build with CMake

```bash
# Create build directory
mkdir build && cd build

# Generate build files
cmake ..

# Build the project
cmake --build .
```

## Running

From the build directory:

```bash
# Run directly
./graphzy

# Or use the custom target
cmake --build . --target run

# Run with a custom graph file
./graphzy my_graph.txt
```

## Graph File Format

The first line must contain the number of nodes.
Each subsequent line defines a link between nodes in the format `node1-node2`.

Example:

```
6
0-1
0-2
0-5
1-2
1-3
2-4
3-4
3-5
4-5
```
