#!/bin/bash
# Build and test s390x GEMM kernels

set -e

echo "Building s390x GEMM test..."

# Detect PyTorch paths
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INCLUDE="${TORCH_PATH}/include"
TORCH_LIB="${TORCH_PATH}/lib"

# Compiler flags for s390x
CXX_FLAGS="-std=c++17 -O3 -march=z14 -mvx -mzvector"
CXX_FLAGS="${CXX_FLAGS} -Wall -Wextra"
CXX_FLAGS="${CXX_FLAGS} -I${TORCH_INCLUDE}"
CXX_FLAGS="${CXX_FLAGS} -I${TORCH_INCLUDE}/torch/csrc/api/include"
CXX_FLAGS="${CXX_FLAGS} -I./csrc/cpu"

LINKER_FLAGS="-L${TORCH_LIB} -ltorch -ltorch_cpu -lc10"
LINKER_FLAGS="${LINKER_FLAGS} -Wl,-rpath,${TORCH_LIB}"

# Build
echo "Compiling..."
g++ ${CXX_FLAGS} \
    test_s390x_gemm.cpp \
    csrc/cpu/sgl-kernels/gemm_s390x.cpp \
    -o test_s390x_gemm \
    ${LINKER_FLAGS}

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
    echo "Running tests..."
    ./test_s390x_gemm
else
    echo "✗ Compilation failed!"
    exit 1
fi
