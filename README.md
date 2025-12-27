# Mandelbrot Set

GPU-accelerated Mandelbrot set fractal visualization in C++11 with CUDA.

![Language Distribution](https://img.shields.io/badge/CUDA-49.6%25-blue) ![C++](https://img.shields.io/badge/C%2B%2B-24.0%25-orange) ![Python](https://img.shields.io/badge/Python-19.9%25-yellow)

## Overview

This project implements the Mandelbrot set fractal visualization with CUDA acceleration for high-performance rendering. The Mandelbrot set is a famous fractal defined by the complex function:

\[ z_{n+1} = z_n^2 + c \]

where \( z \) and \( c \) are complex numbers. The set consists of all complex numbers \( c \) for which the sequence does not diverge.

## Features

- 🚀 **CUDA Acceleration** - GPU-accelerated computation for fast rendering
- 🎨 **High-Quality Output** - FreeImage integration for image export
- 📊 **Performance Analysis** - Python scripts for speedup analysis
- 🔧 **Cross-Platform** - Supports Linux and macOS

## Requirements

- **CUDA Toolkit** (for GPU acceleration)
- **FreeImage** library
  - Ubuntu/Debian: `sudo apt-get install libfreeimage-dev`
  - macOS: `brew install freeimage`
- **C++11 Compatible Compiler**
  - GCC 4.7.2+ (Linux)
  - Clang with Xcode 4.6+ (macOS)

## Building

### Clone the Repository

```bash
git clone https://github.com/AakashR13/Mandelbrot_set.git
cd Mandelbrot_set
```

### Using Makefile (Recommended)

Simply run:

```bash
make
```

The Makefile handles all compilation flags and dependencies automatically.

### Manual Compilation

If you prefer to compile manually:

**Ubuntu (GCC 4.7.2+):**
```bash
g++ -std=c++11 -O3 save_image.cpp utils.cpp mandel.cpp -lfreeimage
```

**Ubuntu 12.04 (GCC 4.6.x):**
Comment any `<chrono>` header usage in `mandel.cpp`, then:
```bash
g++ -std=c++0x save_image.cpp utils.cpp mandel.cpp -lfreeimage
```

**macOS (Xcode 4.6+):**
```bash
clang++ -std=c++11 -stdlib=libc++ save_image.cpp utils.cpp mandel.cpp -lfreeimage
```

## Usage

After compilation, run the executable to generate Mandelbrot set images:

```bash
./a.out
```

The program will generate fractal images using the configured parameters.

### Performance Analysis

Use the included Python script to analyze CUDA speedup:

```bash
python find_speedup.py
```

## Project Structure

```
Mandelbrot_set/
├── src/              # Source files
│   ├── mandel.cpp   # Main Mandelbrot calculation
│   ├── save_image.cpp
│   └── utils.cpp
├── bin/              # Compiled binaries
├── res/              # Resources
├── reports/          # Performance reports
└── find_speedup.py   # Performance analysis script
```

## Language Distribution

- **CUDA**: 49.6% - GPU kernels for acceleration
- **C++**: 24.0% - Core computation logic
- **Python**: 19.9% - Analysis and utility scripts
- **Makefile**: 5.7% - Build configuration
- **C**: 0.8% - Low-level utilities

## References

- [Original Implementation](https://github.com/sol-prog/Mandelbrot_set) by sol-prog
- [Project Article](http://solarianprogrammer.com/2013/02/28/mandelbroot-set-cpp-11/)
- [Mandelbrot Set - Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)

## License

This project is based on [sol-prog/Mandelbrot_set](https://github.com/sol-prog/Mandelbrot_set), licensed under **GPL v3**.

CUDA extensions and benchmarking were added by **Aakash Gorla**.

---

**Original Copyright:** 2013 Sol from www.solarianprogrammer.com

See the [LICENSE](http://www.gnu.org/copyleft/gpl.html) file for details.

## Acknowledgments

- Original implementation by [sol-prog](https://github.com/sol-prog)
- [FreeImage](http://freeimage.sourceforge.net/) for image I/O
