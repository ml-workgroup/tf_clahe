#include <cstdint>
#include <iostream>
#include <random>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
// TODO auto-include via Makefile
#include "../include/cub/cub/cub.cuh"
#include "../include/cudahelpers/cuda_helpers.cuh"

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
    Index MaxColorDepth, 
    bool  MultiplicativeRedistribution>    
__global__ void clahe(
    const In  * const   in, 
          Out * const   out,
          Count * const hist,
    const dim3          data_dim,
    const dim3          window_dim,
    const float         relative_clip_limit)
{   
    const Index grid_dim_x = gridDim.x*blockDim.x;
    const Index grid_dim_y = gridDim.y*blockDim.y;
    const Index grid_dim_z = gridDim.z*blockDim.z;

    const Index gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const Index gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    const Index gid_z = blockIdx.z * blockDim.z + threadIdx.z;
    const Index gid   = (grid_dim_y*grid_dim_x)*gid_z+grid_dim_x*gid_y+gid_x;

    printf("(%llu %llu %llu), %llu\n", gid_x, gid_y, gid_z, gid);

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

template<class In>
__host__ void generate_input(In * in, const dim3 dim, const In max_color_depth)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<In> dis(0, max_color_depth-1);
 
    for (In i = 0; i < dim.x*dim.y*dim.z; ++i)
    {
        in[i] = dis(gen);
    }
}

template<class In, class Out>
__host__ void validate_output(const In &in, const Out &out, const dim3 dim)
{
    // TODO
}

int main (int argc, char *argv[])
{
    // define data types and globals
    using In = int;
    using Out = float;
    using Count = std::uint8_t;
    using Index = std::size_t;
    static constexpr In MaxColorDepth = 4096;
    static constexpr bool MultiplicativeRedistribution = true;
    static constexpr Index DeviceId = 0;
    const float relative_clip_limit = 0.01;

    // define dimensions
    const dim3 data_dim = {500, 500, 50};
    const Index data_dim_flat = data_dim.x*data_dim.y*data_dim.z;
    const dim3 window_dim = {5, 5, 5};
    const Index window_dim_flat = window_dim.x*window_dim.y*window_dim.z;

    // set kernel config
    const dim3  grid_dim(1, 1, 1);
    const dim3  block_dim(32, 2, 2);
    const Index block_dim_flat = block_dim.x*block_dim.y*block_dim.z;
    const Index grid_dim_flat = grid_dim.x*grid_dim.y*grid_dim.z*block_dim_flat;
    const Index smem_bytes = 49152; // maximum available smem

    std::cout << data_dim_flat << " " << window_dim_flat << " " << grid_dim_flat << std::endl;
    // set CUDA device 
    cudaSetDevice(DeviceId); CUERR
    
    // allocate memory
    In * in_h = nullptr;
    cudaMallocHost(&in_h, sizeof(In)*data_dim_flat); CUERR
    In * in_d = nullptr; 
    cudaMalloc(&in_d, sizeof(In)*data_dim_flat); CUERR

    Count * hist_d = nullptr;
    cudaMalloc(&hist_d, sizeof(Count)*grid_dim_flat*MaxColorDepth); CUERR
    cudaMemset(hist_d, 0, sizeof(Count)*grid_dim_flat*MaxColorDepth); CUERR
    
    Out * out_h = nullptr;
    cudaMallocHost(&out_h, sizeof(Out)*data_dim_flat); CUERR
    Out * out_d = nullptr; 
    cudaMalloc(&out_d, sizeof(Out)*data_dim_flat); CUERR
    
    // generate input
    TIMERSTART(generate_input)
    generate_input(in_h, data_dim, MaxColorDepth);
    TIMERSTOP(generate_input)
    cudaMemcpy(in_d, in_h, sizeof(In)*data_dim_flat, H2D); CUERR

    // zero output array (just to sooth my paranoia)
    cudaMemset(out_d, 0, sizeof(Out)*data_dim_flat); CUERR
   
    // execute kernel
    TIMERSTART(clahe)
    cudaProfilerStart(); CUERR
    kernel::clahe<In, Out, Count, Index, MaxColorDepth, MultiplicativeRedistribution>
    <<<grid_dim, block_dim, smem_bytes>>>
    (in_d, out_d, hist_d, data_dim, window_dim, relative_clip_limit);
    cudaDeviceSynchronize(); CUERR
    cudaProfilerStop(); CUERR
    TIMERSTOP(clahe)
    const float tp = ((sizeof(In)*data_dim_flat)/1073741824)/(timeclahe/1000.0);
    std::cout << "THROUGHPUT: " << tp << " GB/s" << std::endl;

    // copy output
    cudaMemcpy(out_h, out_d, sizeof(Out)*data_dim_flat, D2H); CUERR

    //validate output
    validate_output(in_h, out_h, data_dim);

    // free memory
    cudaFreeHost(in_h); CUERR
    cudaFree(in_d); CUERR
    cudaFreeHost(out_h); CUERR
    cudaFree(out_d); CUERR
    cudaFree(hist_d); CUERR
}