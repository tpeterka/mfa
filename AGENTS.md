# AGENTS.md

This file contains guidelines for agentic coding agents working in the MFA (Multivariate Functional Approximation) repository.

## Build Commands

### Basic Build
```bash
# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install
make install
```

### Build Options
- `-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo` (default: Release)
- `-Dmfa_thread=tbb|kokkos|sycl|serial` (default: serial)
- `-Dmfa_build_examples=ON|OFF` (default: ON)
- `-Dmfa_build_tests=ON|OFF` (default: ON)
- `-Deigen_thread=ON|OFF` (enable OpenMP for Eigen)
- `-Dmfa_python=ON|OFF` (build Python bindings)

### Testing Commands
```bash
# Run all tests
cd build && ctest

# Run specific test
ctest -R fixed-sinc-2d-test

# Run tests with verbose output
ctest --verbose

# Run tests in parallel
ctest -j$(nproc)

# Run single test executable directly
./tests/fixed-test -h  # see options
./tests/fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0
```

## Code Style Guidelines

### Naming Conventions
- **Classes**: PascalCase (e.g., `MFA`, `Encoder`, `Decoder`)
- **Functions**: PascalCase for public methods (e.g., `Encode()`, `Decode()`)
- **Variables/Parameters**: snake_case (e.g., `dom_dim`, `pt_dim`, `nctrl_pts`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MFA_NAW`, `MFA_LOW_D`)
- **Files**: snake_case with `.hpp`/`.cpp` extensions (e.g., `mfa_data.hpp`)

### Includes Organization
```cpp
// Standard C++ library (alphabetical)
#include <iostream>
#include <fstream>
#include <vector>

// External dependencies (Eigen, MPI, etc.)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>

// MFA internal includes (alphabetical)
#include <mfa/types.hpp>
#include <mfa/mfa_data.hpp>
#include <mfa/encoder.hpp>
```

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Brace style**: K&R (opening brace on same line for functions)
- **Line length**: ~120 characters max
- **Alignment**: Align parameters for readability

```cpp
void long_function_name(
        Type1      param1,            // description
        Type2      param2,            // description  
        Type3      param3)            // description
{
    // implementation
}

class ClassName
{
public:
    // public members

private:
    // private members
};
```

### Template Usage
- Use `T` for primary numeric type template parameter
- Support both `float` and `double` precision
- Heavy use of Eigen's template system
- Conditional compilation for different backends:

```cpp
template <typename T>                           // float or double
class MFA {
    // Implementation
};

#ifdef MFA_TBB
    // TBB implementation
#elif defined(MFA_KOKKOS) 
    // Kokkos implementation
#else
    // Serial implementation
#endif
```

### Error Handling
```cpp
// Error reporting pattern
if (error_condition) {
    fmt::print(stderr, "ERROR: Descriptive message\n");
    fmt::print(stderr, "       Additional details: value1={}\n", var1);
    exit(1);
}

// Exception pattern
if (invalid_condition) {
    throw MFAError("Descriptive error message");
}
```

### Memory Management
- Use `unique_ptr` for exclusive ownership
- Use `shared_ptr` for shared resources
- Follow Rule of Five with `= delete` for copy/move operations
- Prefer Eigen types for automatic memory management

### Documentation Style
```cpp
//--------------------------------------------------------------
// Brief description of class/module
//
// Author Name
// Institution  
// email@institution.gov
//--------------------------------------------------------------
```

- Inline parameter descriptions aligned to the right
- Mathematical notation follows standard conventions
- Reference algorithms/papers when applicable

### Eigen Integration
- Use predefined typedefs from `types.hpp`: `MatrixX<T>`, `VectorX<T>`, etc.
- Matrices stored in column-major order by default
- Leverage Eigen's expression templates for performance

### Threading Patterns
- All performance-critical code should support multiple backends
- Use conditional compilation for threading-specific code
- Serial version should always be available as fallback

## Common Patterns

### File Headers
Every source file should have the standard header block with author information and brief description.

### Mathematical Code
- Follow notation from NURBS/B-spline literature
- Single letters for mathematical concepts (i, j, k for indices)
- Document reference algorithms and papers

### Testing
- Use Catch2 framework included in tests/
- Test names follow descriptive pattern: `component-scenario-test`
- Include both positive and negative test cases
- Test with different input sizes and data types

### Performance Considerations
- Profile before optimizing
- Consider cache efficiency for large matrices
- Use Eigen's optimized operations where possible
- Test with both float and double precision

## Important Files
- `include/mfa/types.hpp`: Core type definitions and Eigen typedefs
- `include/mfa/mfa.hpp`: Main MFA class interface
- `CMakeLists.txt`: Build configuration
- `tests/CMakeLists.txt`: Test definitions and examples of test commands

## Dependencies
- **Required**: Eigen3, MPI, DIY, fmt
- **Optional**: TBB, Kokkos, OpenMP, CLP (for weights)
- **Testing**: Catch2 (header-only, included)
- **Python**: pybind11 (optional)

This codebase is research-oriented with emphasis on mathematical correctness and performance. Follow existing patterns and maintain compatibility with multiple threading backends.