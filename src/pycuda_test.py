import pycuda.autoinit

import pycuda.driver as drv
import nibabel as nib
import numpy as np

from pycuda.compiler import SourceModule

nifti_path = "dyn5.nii.gz"
cuda_code_path = "clahe_kernel.cu"
nvcc_path = "/usr/local/cuda/bin/nvcc"

# Load nifti
nifti = nib.load(nifti_path)
nifti_array = nifti.get_fdata().astype(np.int32)

# Load and compile cuda code
with open(cuda_code_path, "r") as f:
    cuda_code = f.read()
module = SourceModule(cuda_code, nvcc_path)
clahe_fn = module.get_function("ClaheKernel")

# Compute CLAHE
window_size = 4
clip_limit = 0.01
multiplicative_redistribution = True

unique_values = np.unique(nifti_array).astype(np.int32)
lookup_table = np.searchsorted(unique_values, nifti_array).astype(np.int32)

result = np.zeros_like(nifti_array, dtype=np.float32)

a = np.array
grid_size = (nifti_array.shape[0] - window_size, 1, 1)
block_size = (nifti_array.shape[1] - window_size, 1, 1)
clahe_fn(drv.In(nifti_array), drv.Out(result), drv.In(lookup_table), drv.In(unique_values), a(unique_values.size),
         a(nifti_array.shape[0]), a(nifti_array.shape[1]), a(nifti_array.shape[2]),
         a(window_size), a(window_size), a(window_size),
         a(clip_limit), a(multiplicative_redistribution),
         grid=grid_size, block=block_size)


# Store result
affine = nifti.affinelsls
result = np.array(result, dtype=np.int32)
print(np.min(result), np.max(result), np.average(result))

nib.Nifti1Image(result, affine).to_filename("result.nii.gz")
