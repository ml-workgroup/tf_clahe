#include <cstdint>
#include <iostream>
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
    class Index,
    Index MaxColorDepth, 
    bool  MultiplicativeRedistribution>    
__global__ void clahe(
    const In  * const in, 
          Out * const out,
    const dim3        data_dim,
    const dim3        window_dim,
    const float       relative_clip_limit) 
{   
   // TODO 
}

} // namespace kernel

template<class In>
__host__ void generate_input(In &in, const dim3 dim)
{
    // TODO
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
    using Index = std::size_t;
    static constexpr Index MaxColorDepth = 4096;
    static constexpr bool MultiplicativeRedistribution = true;
    static constexpr Index DeviceId = 0;
    const float relative_clip_limit = 0.5; // TODO realistic clip limit

    // define dimensions
    const dim3 data_dim = {50, 50, 50};
    const Index data_dim_flat = data_dim.x*data_dim.y*data_dim.z;
    const dim3 window_dim = {5, 5, 5};
    const Index window_dim_flat = window_dim.x*window_dim.y*window_dim.z;

    // set CUDA device
    cudaSetDevice(DeviceId); CUERR
    
    // allocate memory
    In * in_h = nullptr;
    cudaMallocHost(&in_h, sizeof(In)*data_dim_flat); CUERR
    In * in_d = nullptr; 
    cudaMalloc(&in_d, sizeof(In)*data_dim_flat); CUERR

    Out * out_h = nullptr;
    cudaMallocHost(&out_h, sizeof(Out)*data_dim_flat); CUERR
    Out * out_d = nullptr; 
    cudaMalloc(&out_d, sizeof(Out)*data_dim_flat); CUERR
    
    // generate input
    generate_input(in_h, data_dim);
    cudaMemcpy(in_d, in_h, sizeof(In)*data_dim_flat, H2D); CUERR

    // zero output array (just to sooth my paranoia)
    cudaMemset(out_d, 0, sizeof(Out)*data_dim_flat); CUERR

    // set kernel config
    const dim3  grid_dim;
    const dim3  block_dim;
    const Index smem_bytes = 0;

    // execute kernel
    TIMERSTART(clahe)
    kernel::clahe<In, Out, Index, MaxColorDepth, MultiplicativeRedistribution>
    <<<grid_dim, block_dim, smem_bytes>>>
    (in_d, out_d, data_dim, window_dim, relative_clip_limit);
    cudaDeviceSynchronize(); CUERR
    TIMERSTOP(clahe)
    const float tp = ((sizeof(In)*data_dim_flat)/1073741824)/(timeclahe/1000.0);
    std::cout << "THROUGHPUT: " << tp << "GB/s" << std::endl;

    // copy output
    cudaMemcpy(out_h, out_d, sizeof(Out)*data_dim_flat, D2H); CUERR

    //validate output
    validate_output(in_h, out_h, data_dim);

    // free memory
    cudaFreeHost(in_h); CUERR
    cudaFree(in_d); CUERR
    cudaFreeHost(out_h); CUERR
    cudaFree(out_d); CUERR
}