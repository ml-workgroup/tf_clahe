/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "vector_types.h"

using namespace tensorflow;

REGISTER_OP("Clahe")
    .Input("input: int32")
    .Attr("window_size: int")
    .Attr("relative_clip_limit: float")
    .Attr("multiplicative_redistribution: bool")
    .Output("output: float")
    .Doc(R"doc(
      ToDo: Write doc!
)doc");

void ClaheKernelLauncher(const int* in, 
                         float* out, 
                         int* histogram, 
                         const dim3 data_dim, 
                         const dim3 window_dim, 
                         float relative_clip_limit,
                         bool multiplicative_redistribution);

class ClaheOp : public OpKernel {
    public:
        explicit ClaheOp(OpKernelConstruction* context) : OpKernel(context) {
            // Parse attributes from function call
            context->GetAttr("window_size", &window_size);
            context->GetAttr("relative_clip_limit", &relative_clip_limit);
            context->GetAttr("multiplicative_redistribution", &multiplicative_redistribution);
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            auto input_shape = input_tensor.shape();
            auto input = input_tensor.flat<int32>();

            // Define dimensions
            const dim3 data_dim = {input_shape.dim_size(0), input_shape.dim_size(1),
                input_shape.dim_size(2)};
            const dim3 window_dim = {window_size, window_size, window_size};

            // Allocate memory for histograms
            Tensor histogram_tensor(DT_INT32, TensorShape({}));    
            int64 histogram_size = 32 * 2 * 2 * 4096 * sizeof(int);  // ToDo: exact size
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, 
                TensorShape({histogram_size}), 
                &histogram_tensor));
            auto histogram = histogram_tensor.template flat<int>();
            
            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                                &output_tensor));
            auto output = output_tensor->template flat<float>();

            // Call the cuda kernel launcher
            ClaheKernelLauncher(input.data(), output.data(), histogram.data(), 
                                data_dim, window_dim, relative_clip_limit, multiplicative_redistribution);
        }
    private:
        int window_size;
        float relative_clip_limit;
        bool multiplicative_redistribution;
};

REGISTER_KERNEL_BUILDER(Name("Clahe").Device(DEVICE_GPU), ClaheOp);

