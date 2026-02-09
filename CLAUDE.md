# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ image processing project using OpenCV. The project uses CMake as its build system and requires C++17 standard.

## Build and Run Commands

### Initial Setup
```bash
mkdir -p build
cd build
cmake ..
```

### Build
From the `build/` directory:
```bash
cmake --build .
```

Or rebuild from project root:
```bash
cd build && cmake .. && cmake --build .
```

### Run
From the `build/` directory:
```bash
./OpenCV_Project      # Linux/Mac
OpenCV_Project.exe    # Windows
```

### Clean Build
```bash
rm -rf build/*
cd build && cmake .. && cmake --build .
```

## Project Structure

- `src/` - C++ source files (.cpp). All .cpp files in this directory are automatically included in the build via `file(GLOB SOURCES "src/*.cpp")`
- `include/` - Header files (.h, .hpp). This directory is included in the include path but may not exist yet
- `build/` - Build artifacts (gitignored, must be created manually)
- `CMakeLists.txt` - CMake configuration

## CMake Configuration

- Minimum CMake version: 3.10
- C++ standard: C++17 (required)
- Main executable name: `OpenCV_Project`
- All source files in `src/*.cpp` are automatically globbed and included in the build
- OpenCV is found via `find_package(OpenCV REQUIRED)` and must be installed system-wide

## Requirements

- CMake 3.10+
- OpenCV (installed in the system)
- C++17-compatible compiler

## Architecture Notes

The CMakeLists.txt automatically includes all .cpp files from the src/ directory using `file(GLOB)`. When adding new source files, simply place them in src/ and rebuild - no need to modify CMakeLists.txt.

The include/ directory is configured in the include path, so headers can be included with `#include "header.hpp"` syntax.