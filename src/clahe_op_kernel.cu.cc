#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cstdint>
#include <iostream>
#include <random>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
// TODO auto-include via Makefile
#include "../include/cub/cub/cub.cuh"
#include "../include/cudahelpers/cuda_helpers.cuh"


__global__ void ClaheKernel(const int* in, const int N, float* out) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + 1;
  }
}

namespace kernel
{

/*! 
*\brief compute sliding window clahe for a fixed line in 3D Tensor along the x axis, where the 
    starting point (y, z)
* \tparam In base type of input
* \tparam Out base type of output
* \tparam Index integer type used for indexing
* \tparam MaxColorDepth maximum color depth value of input
* \param[in] in input tensor
* \param[out] out output tensor
* \param[in] data_dim dimensions of input/output
* \param[in] window_dim dimensions of sliding window
* \param[in] relative_clip_limit relative clip limit
*/
template<
    class In,
    class Out,
    class Count,
    class Index,
    Index MaxColorDepth>    
__global__ void clahe(const int* in, 
                         float* out, 
                         int* histogram, 
                         const dim3 data_dim, 
                         const dim3 window_dim, 
                         float relative_clip_limit,
                         bool multiplicative_redistribution)
{   
    const Index grid_dim_x = gridDim.x*blockDim.x;
    const Index grid_dim_y = gridDim.y*blockDim.y;
    const Index grid_dim_z = gridDim.z*blockDim.z;

    const Index gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const Index gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    const Index gid_z = blockIdx.z * blockDim.z + threadIdx.z;
    const Index gid   = (grid_dim_y*grid_dim_x)*gid_z+grid_dim_x*gid_y+gid_x;

    for(Index z = gid_z; z < data_dim.z; z += gridDim.z*blockDim.z)
        for(Index y = gid_y; y < data_dim.y; y += gridDim.y*blockDim.y)
            for(Index x = gid_x; x < data_dim.x; x += gridDim.x*blockDim.x)
    {
        for(Index i = gid; i < MaxColorDepth; i += gridDim.x*blockDim.x)
        {
            //hist[gid*MaxColorDepth+i] = 0; // TODO
        }
    }
}

} // namespace kernel

void ClaheKernelLauncher(const int* in, 
                         float* out, 
                         int* histogram, 
                         const dim3 data_dim, 
                         const dim3 window_dim, 
                         float relative_clip_limit,
                         bool multiplicative_redistribution) {
    // Datatypes
    using In = int;
    using Out = float;
    using Count = std::uint8_t;
    using Index = std::size_t;
    static constexpr In MaxColorDepth = 4096;
    
    // Dimensions
    const dim3  grid_dim(1, 1, 1);
    const dim3  block_dim(32, 2, 2);
    const Index block_dim_flat = block_dim.x*block_dim.y*block_dim.z;
    const Index grid_dim_flat = grid_dim.x*grid_dim.y*grid_dim.z*block_dim_flat;
    const Index smem_bytes = 49152; // maximum available smem

    // Kernel Call
    kernel::clahe<In, Out, Count, Index, MaxColorDepth>
    <<<grid_dim, block_dim, smem_bytes>>>
    (in, out, histogram, data_dim, window_dim, relative_clip_limit, multiplicative_redistribution);
}




#endif
