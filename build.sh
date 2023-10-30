#!/bin/bash

#module load CUDA/9.2.88
#OR
module load intel/2020b

module load CMake/3.12.1

cmake \
    -Dcaliper_DIR=/scratch/group/csce435-f23/Caliper-MPI/caliper/share/cmake/caliper \
    -Dadiak_DIR=/scratch/group/csce435-f23/Adiak/adiak/lib/cmake/adiak \
    .

make