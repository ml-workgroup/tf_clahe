#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <limits>
#include <time.h>

const int MAX_COLOR_DEPTH = 4096;  // Maximum depth of grayscales

/**
    Compute sliding window clahe for a fixed line in 3D Tensor along the x axis, where the 
    starting point (y, z) is determined by the block and thread ID.
*/
__global__  void ClaheKernel(const int *data, float *out,
    int *lookUpTable, int* uniqueValues, int nUniqueValues,
    int width, int height, int depth,
    int windowWidth, int windowHeight, int windowDepth,
    float relativeClipLimit,
    bool multiplicative_redistribution = true) 
{   
    int z0 = blockIdx.x;
    int y0 = threadIdx.x;

    // Compute histogram of first block in sliding window
    int* histogram = new int[nUniqueValues];
    for(int i = 0; i < nUniqueValues; ++i){
        histogram[i] = 0;
    }
    for(int z = z0; z < z0 + windowDepth; ++z){
        for(int y = y0; y < y0 + windowHeight; ++y){
            int index = z * height * width + y * width;
            for(int xPtr = index; xPtr < index + windowWidth - 1; ++xPtr){
                histogram[lookUpTable[xPtr]] += 1;
            }
        }
    }
    /*
    printf("Initial Histogram: ");
    for(int i = 0; i < nUniqueValues; ++i){
        printf("%d ", histogram[i]);
    }
    printf("\n");
    */
    // Slide block along X axis
    double windowSize = windowDepth * windowHeight * windowWidth;
    double absoluteClipLimit = relativeClipLimit * windowSize / nUniqueValues;

    for(int x0 = 0; x0 < width - windowWidth +1; ++x0){

        // Update histogram by appending the slice of the (y/z plane)
        for(int z = z0; z < z0 + windowDepth; ++z){
            for(int y = y0; y < y0 + windowHeight; ++y){
                int index = z * height * width + y * width + x0 + windowWidth - 1;
                histogram[lookUpTable[index]] += 1;
            }
        }
        
        /*
        printf("Histogram for x=%d: ", x0);
        for(int i = 0; i < nUniqueValues; ++i){
            printf("%d ", histogram[i]);
        }
        printf("\n");
        */

        // Calculate (limited) CDF
        int currentIndex = int((z0 + windowDepth) / 2) * height * width + int((y0 + windowHeight) / 2) * width + 
                            int((x0 + windowWidth) / 2);

        int clipPos = lookUpTable[currentIndex];
        double cdfSum = 0;
        for(int i = 0; i < clipPos; ++i){
            cdfSum += min(double(histogram[i]), absoluteClipLimit);
        }
        double cdfAtPosition = cdfSum + min(double(histogram[clipPos]), absoluteClipLimit) / 2;
        for(int i = clipPos; i < nUniqueValues; ++i){
            cdfSum += min(double(histogram[i]), absoluteClipLimit);
        }
        
        // Redistribute clipped region of cdf
        if(multiplicative_redistribution)
            out[currentIndex] = cdfAtPosition / cdfSum;
        else
            out[currentIndex] = (cdfAtPosition + (windowSize - cdfSum) * 
                                 (uniqueValues[clipPos] + 0.5) / (uniqueValues[nUniqueValues - 1] + 1)) /
                                 windowSize;

        // Update histogram by "cutting" of the front slice (x/y plane) of the block
        for(int z = z0; z < z0 + windowDepth; ++z){
            for(int y = y0; y < y0 + windowHeight; ++y){
                int index = z * height * width + y * width + x0;
                histogram[lookUpTable[index]] -= 1;
            }
        }
    }

    free(histogram);
}



/**
    Find all unique values in the array of size nElements and a create lookup table that contains
    the index of the grayscale value in an ordered array of unique values.

    ToDo: Maybe this could also be done directly in CUDA
*/
void compute_lookup_table(const int* data, int*& lookUpTable, int nElements, int*& uniqueValues, int& nUniqueValues)
{
    // Create array of size of possible color values and create marker if value exists in data
    int referenceArray[MAX_COLOR_DEPTH] = {0};
    for(int i = 0; i < nElements; ++i){
        if(referenceArray[data[i]] == 0){
            nUniqueValues++;
        }
        referenceArray[data[i]]++;
    }

    // Create array of unique items and reference new index in referenceArray
    cudaMallocManaged(&uniqueValues, nUniqueValues * sizeof(int));
    int currentIndex = 0;
    for(int i = 0; i < MAX_COLOR_DEPTH; ++i){
        if(referenceArray[i] > 0){
            referenceArray[i] = currentIndex;
            uniqueValues[currentIndex] = i;
            currentIndex++;
        }
    }

    // Create lookup table
    cudaMallocManaged(&lookUpTable, nElements * sizeof(int));
    for(int i = 0; i < nElements; ++i){
        lookUpTable[i] = referenceArray[data[i]];
    }
}

void ClaheKernelLauncher(const int *data, float *out,
    int width, int height, int depth,
    int windowWidth, int windowHeight, int windowDepth,
    float relativeClipLimit,
    bool multiplicative_redistribution = true) 
{
    // Initialize Lookup Table for grey values
    int* lookUpTable = NULL;
    int* uniqueValues = NULL;
    int nUniqueValues = 0;
    compute_lookup_table(data, lookUpTable, width * height * depth, uniqueValues, nUniqueValues);


    // Compute CLAHE
    ClaheKernel<<<depth, height>>>(
        data, out,
        lookUpTable, uniqueValues, nUniqueValues,
        width, height, depth, windowWidth, windowHeight, windowDepth, 
        relativeClipLimit, multiplicative_redistribution);

    cudaFree(uniqueValues);
    cudaFree(lookUpTable);
}


int main()
{
    // Initialize Data
    int size = 10;
    int windowSize = 2;
    int* data = NULL;
    float* out = NULL;
    cudaMallocManaged(&data, size * size * size * sizeof(int));
    cudaMallocManaged(&out,  size * size * size * sizeof(float));
    

    // Generate random entries
    srand(time(NULL));
    for(int i = 0; i < size * size * size; ++i){
        data[i] = rand() % 10;
    }

    // Print data
    int fixedX = 3;
    printf("\nData for x = %d:\n", fixedX);
    for(int z = 0; z < size; ++z){
        for(int y = 0; y < size; ++y){
            int index = z * size * size + y * size + fixedX;
            printf("%d ", data[index]);
        }
        printf("\n");
    }

    // Initialize Lookup Table for grey values
    int* lookUpTable = NULL;
    int* uniqueValues = NULL;
    int nUniqueValues = 0;
    compute_lookup_table(data, lookUpTable, size * size * size, uniqueValues, nUniqueValues);

    // Compute CLAHE
    int height = size;
    int depth = size;
    ClaheKernel<<<size, size>>>(
        data, out,
        lookUpTable, uniqueValues, nUniqueValues,
        size, size, size, windowSize, windowSize, windowSize, 
        0.1, true);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Print result
    printf("\nResult for x = %d:\n", fixedX);
    for(int z = 0; z < size; ++z){
        for(int y = 0; y < size; ++y){
            int index = z * size * size + y * size + fixedX;
            printf("%0.2f ", out[index]);
        }
        printf("\n");
    }

    cudaFree(data);
    cudaFree(out);
    cudaFree(uniqueValues);
    cudaFree(lookUpTable);
    return 0;
}