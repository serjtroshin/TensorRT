/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


 #include "mySeluPlugin.h"
 #include <cuda_fp16.h>

// SELU constants
constexpr float alpha = 1.6732632423543772848170429916717f;
constexpr float scale = 1.0507009873554804934193349852946f;

__global__ void mySeluKernel(
    const int N,
    const float* input,
    float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
         output[i] = input[i] >= 0 ? scale * input[i] : scale * alpha * (exp(input[i]) - 1);
    }
 }

 int mySeluInference(
     const int n,
     float* input,
     float* output,
     cudaStream_t stream) {
    // NCHW
    const int nThreads = 512;

    int nBlocks = (n + nThreads - 1) / nThreads;

    mySeluKernel<<<nBlocks, nThreads, 0, stream>>>(n, input, output);

     cudaError_t err = cudaGetLastError();
     if ( cudaSuccess != err )
     {
         fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
         return 1;
     }
     return 0;
 }

 int mySeluPlugin::enqueue(
     int batchSize,
     const void* const* inputs,
     void** outputs,
     void* workspace,
     cudaStream_t stream) {
    return mySeluInference(batchSize * mBatchDim, (float*)inputs[0], (float*)outputs[0], stream);
 }
