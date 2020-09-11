/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "DCNv2Plugin.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define KERNEL_POSITION \
	int position = (blockDim.x * blockIdx.x + threadIdx.x); \
 	if (position >= (edge)) return;

#define cublasCheck(op) \
 	do {	\
 		auto ret = (op);	\
 		if (ret != CUBLAS_STATUS_SUCCESS) {	\
 			INFO("%s fail, %d != %d", #op, ret, CUBLAS_STATUS_SUCCESS); \
 			abort();	\
 		}	\
 	} while (0);

template <typename T>
static __global__ void sigmoidKernel(const T* input, T* output, int edge);

template <>
__global__ void sigmoidKernel(const float* input, float* output, int edge)
{
	KERNEL_POSITION;
	output[position] = 1 / (1 + exp(-input[position]));
}

template<>
__global__ void sigmoidKernel(const half* input, half* output, int edge)
{
	KERNEL_POSITION;
	half one = 1.0f;
	output[position] = one / (one + hexp(-input[position]));
}

static __device__ float dmcnIm2colBilinearFP32(const float *bottom_data, const int data_width,
	const int height, const int width, float h, float w)
{
	int h_low = floor(h);
	int w_low = floor(w);
	int h_high = h_low + 1;
	int w_high = w_low + 1;

	float lh = h - h_low;
	float lw = w - w_low;
	float hh = 1 - lh, hw = 1 - lw;

	float v1 = 0;
	float v2 = 0;
	float v3 = 0;
	float v4 = 0;
	if (h_low >= 0 && w_low >= 0)
    	v1 = bottom_data[h_low * data_width + w_low];
	if (h_low >= 0 && w_high <= width - 1)
	    v2 = bottom_data[h_low * data_width + w_high];
	if (h_high <= height - 1 && w_low >= 0)
	    v3 = bottom_data[h_high * data_width + w_low];
	if (h_high <= height - 1 && w_high <= width - 1)
    	v4 = bottom_data[h_high * data_width + w_high];

	float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

	float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
	return val;
}

static __device__ half dmcnIm2colBilinearFP16(const half *bottom_data, const int data_width,
	const int height, const int width, const half& h, const half& w)
{
	int h_low = hfloor(h);
	int w_low = hfloor(w);
	int h_high = h_low + 1;
	int w_high = w_low + 1;

	half one = 1.0f;
	half h_low_hf = h_low;
	half w_low_hf = w_low;
	half lh = h - h_low_hf;
	half lw = w - w_low_hf;
	half hh = one - lh, hw = one - lw;

	half zero = 0.0f;
	half v1 = zero;
	half v2 = zero;
	half v3 = zero;
	half v4 = zero;
	if (h_low >= 0 && w_low >= 0)
	    v1 = bottom_data[h_low * data_width + w_low];
	if (h_low >= 0 && w_high <= width - 1)
	    v2 = bottom_data[h_low * data_width + w_high];
	if (h_high <= height - 1 && w_low >= 0)
	    v3 = bottom_data[h_high * data_width + w_low];
	if (h_high <= height - 1 && w_high <= width - 1)
	    v4 = bottom_data[h_high * data_width + w_high];

	half w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
	return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

template <typename T>
static __global__ void DCNIm2colKernel(
	const T *data_input, const T *data_offset, const T *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	T *data_output, int edge);

template <>
__global__ void DCNIm2colKernel(
	const float *data_input, const float *data_offset, const float *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	float *data_output, int edge)
{
	KERNEL_POSITION;

	const int f_area_input = width_input * height_input;
	const int f_area_output = width_output * height_output;

	// index index of output matrix
	const int w_output = position % width_output;
	const int h_output = (position / width_output) % height_output;
	const int c_input = (position / width_output / height_output) % num_channels;

	const int c_output = c_input * kernel_h * kernel_w;
	const int deformable_group_index = c_input / channel_per_deformable_group;
	const int h_input = h_output * stride_h - pad_h;
	const int w_input = w_output * stride_w - pad_w;

	int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
	float *data_output_ptr = data_output + data_output_offset;
	const float *data_input_ptr = data_input + c_input * f_area_input;
	const float *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
	const float *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

	for (int i = 0; i < kernel_h; ++i)
	{
		for (int j = 0; j < kernel_w; ++j)
		{
			const int row = i + h_input;
			const int col = j + w_input;
			const int kernel_index = i * kernel_w + j;

			const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
			const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
			const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

			const float offset_h = data_offset_ptr[offset_h_offset];
			const float offset_w = data_offset_ptr[offset_w_offset];
			const float mask = data_mask_ptr[mask_offset];

			float val = 0;
			const float h_im = h_input + i * dilation_h + offset_h;
			const float w_im = w_input + j * dilation_w + offset_w;

			if (h_im > -1 && w_im > -1 && h_im < height_input && w_im < width_input)
			{
				val = dmcnIm2colBilinearFP32(data_input_ptr, width_input, height_input, width_input, h_im, w_im);
			}
			*data_output_ptr = val * mask;
			data_output_ptr += f_area_output;
		}
	}
}

template <>
__global__ void DCNIm2colKernel(
	const half *data_input, const half *data_offset, const half *data_mask,
	const int height_input, const int width_input, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int channel_per_deformable_group,
	const int batch_size, const int num_channels, const int deformable_group,
	const int height_output, const int width_output,
	half *data_output, int edge)
{
	KERNEL_POSITION;

	const int f_area_input = width_input * height_input;
	const int f_area_output = width_output * height_output;

	// index index of output matrix
	const int w_output = position % width_output;
	const int h_output = (position / width_output) % height_output;
	const int c_input = (position / width_output / height_output) % num_channels;

	const int c_output = c_input * kernel_h * kernel_w;
	const int deformable_group_index = c_input / channel_per_deformable_group;
	const int h_input = h_output * stride_h - pad_h;
	const int w_input = w_output * stride_w - pad_w;

	half width_input_hf = __float2half(width_input);
	half height_input_hf = __float2half(height_input);

	half h_input_hf = __float2half(h_input);
	half w_input_hf = __float2half(w_input);
	half dilation_h_hf = __float2half(dilation_h);
	half dilation_w_hf = __float2half(dilation_w);

	int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
	half *data_output_ptr = data_output + data_output_offset;
	const half *data_input_ptr = data_input + c_input * f_area_input;
	const half *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
	const half *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

	half n_one = -1.0f;
	half zero = 0.0f;

	for (int i = 0; i < kernel_h; ++i)
	{
		for (int j = 0; j < kernel_w; ++j)
		{
			half i_hf = __float2half(i);
			half j_hf = __float2half(j);
			const int row = i + h_input;
			const int col = j + w_input;
			const int kernel_index = i * kernel_w + j;

			const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
			const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
			const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

			const half offset_h = data_offset_ptr[offset_h_offset];
			const half offset_w = data_offset_ptr[offset_w_offset];
			const half mask = data_mask_ptr[mask_offset];

			half val = zero;
			half h_im = h_input_hf + i_hf * dilation_h_hf + offset_h;
			half w_im = w_input_hf + j_hf * dilation_w_hf + offset_w;

			if (h_im > n_one && w_im > n_one && h_im < height_input_hf && w_im < width_input_hf)
			{
				val = dmcnIm2colBilinearFP16(data_input_ptr, width_input_hf, height_input_hf, width_input_hf, h_im, w_im);
			}
			*data_output_ptr = val * mask;
			data_output_ptr += f_area_output;
		}
	}
}

template <typename T>
static __global__ void biasKernel(T* data_input, const T* bias, const int f_area, int edge)
{
	KERNEL_POSITION;
	int bias_index = position / f_area;
	data_input[position] += bias[bias_index];
}

template <typename T>
inline void segemm_native(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha, /* host or device pointer */
	const T *A,
	int lda,
	const T *B,
	int ldb,
	float beta, /* host or device pointer */
	T *C,
	int ldc);

template <>
inline void segemm_native<float>(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha, /* host or device pointer */
	const float *A,
	int lda,
	const float *B,
	int ldb,
	float beta, /* host or device pointer */
	float *C,
	int ldc)
{
	cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, 
		ldb, &beta, C, ldc);
}

template <>
inline void segemm_native<half>(cublasHandle_t handle,
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	float alpha,
	const half *A,
	int lda,
	const half *B,
	int ldb,
	float beta,
	half *C,
	int ldc)
{
	auto halpha = half(alpha);
	auto hbeta = half(beta);
	cublasGemmEx(handle, transa, transb, m, n, k, &halpha, A, CUDA_R_16F, lda, B, 
		CUDA_R_16F, ldb, &hbeta, C, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DFALT);
}


namespace nvinfer1
{
namespace dcnv2
{

template <typename T>
void enqueue_native(cublasHandle_t handle,
	const T* input1,
	const T* input2,
	const T* weights,
	const T* bias,
	T* output,
    const Dims& input1_shape,
    const Dims& input2_shape,
    const Dims& weights_shape,
    const Dims& output_shape,
    const int deformable_groups,
    void* workspace,
    cudaStream_t stream)
{
	int kernel_size = weights_shape.d[W_DIM];
	size_t maskSize = static_cast<size_t>(input1_shape.d[H_DIM] *
		input1_shape.d[W_DIM] * kernel_size * kernel_size * deformable_groups);
	size_t im2colSize = static_cast<size_t>(input1_shape.d[C_DIM] *
		kernel_size * kernel_size * output_shape.d[H_DIM] * output_shape.d[W_DIM]);

	const int m = output_shape.d[C_DIM];
	const int n = output_shape.d[H_DIM] * output_shape.d[W_DIM];
	const int k = input1_shape.d[C_DIM] * kernel_size * kernel_size;
	float alpha = 1.0;
	float beta = 0.0;

	for (int ibatch = 0 ; ibatch < input2_shape.d[N_DIM]; ++ibatch)
	{
		T* maskWorkspacePtr = static_cast<T*>(workspace) + (maskSize + im2colSize) * ibatch;
		T* im2colWorkspacePtr = static_cast<T*>(workspace) + (maskSize + im2colSize) * 
			ibatch + maskSize;

		const T* inputMask = static_cast<const T*>(input2) + (ibatch, input2_shape.d[C_DIM] / 3 * 2);
		dim3 sigmoidGrid(maskSize);
		dim3 sigmoidBlock(maskSize);
		sigmoidKernel<<<sigmoidGrid, sigmoidBlock, 0, stream>>> (
			inputMask, maskWorkspacePtr, maskSize);

		const int INPUT1_ELEMS_PER_BATCH = input1_shape.d[C_DIM] * input1_shape.d[H_DIM] * 
			input1_shape.d[W_DIM];
		const T* datainput = static_cast<const T*>(input1) + (ibatch * INPUT1_ELEMS_PER_BATCH);

		const int INPUT2_ELEMS_PER_BATCH = input2_shape.d[C_DIM] * input2_shape.d[H_DIM] * 
			input2_shape.d[W_DIM];
		const T* offset = static_cast<const T*>(input2) + (ibatch * INPUT2_ELEMS_PER_BATCH);

		dim3 im2colGrid(im2colSize);
		dim3 im2colBlock(im2colSize);
		DCNIm2colKernel<<<im2colGrid, im2colBlock, 0, stream>>>(
			datainput, offset, maskWorkspacePtr, input1_shape.d[H_DIM], input1_shape.d[W_DIM], 
			kernel_size, kernel_size, 1, 1, 1, 1, 1, 1,
			input1_shape.d[C_DIM], input1_shape.d[N_DIM], input1_shape.d[C_DIM], deformable_groups,
			output_shape.d[H_DIM], output_shape.d[W_DIM], im2colWorkspacePtr, im2colSize);

		const T* weightKernel = static_cast<const T*>(weights);

		const int OUTPUT_ELEMS_PER_BATCH = output_shape.d[C_DIM] * output_shape.d[H_DIM] * 
			output_shape.d[W_DIM];
		T* batch_output = static_cast<T*>(output) + (ibatch * OUTPUT_ELEMS_PER_BATCH);
		segemm_native(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, im2colWorkspacePtr, 
			n, weightKernel, k, beta, batch_output, n);

		if (bias)
		{
			const T* weightBias = static_cast<const T*>(bias);
			size_t edge = output_shape.d[C_DIM] * output_shape.d[H_DIM] * output_shape.d[W_DIM];
			size_t area = output_shape.d[H_DIM] * output_shape.d[W_DIM];

			dim3 biasGrid(edge);
			dim3 biasBlock(edge);
			biasKernel<<<biasGrid, biasBlock, 0, stream>>>(batch_output, weightBias, area, edge);
		}
	}
}

void enqueue_call(const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
	const DataType& iType, const Dims& input_shape, const Dims& input_shape1,
	const Dims& weights_shape, const Dims& output_shape, DataType mType,
	cublasHandle_t cublasHandle_, const DCNv2Parameters& mParam)
{
	if (mType == nvinfer1::DataType::kFLOAT)
	{
		enqueue_native<float>(cublasHandle_, static_cast<const float*>(inputs[0]),
			static_cast<const float*>(inputs[1]), static_cast<const float*>(inputs[2]),
			static_cast<const float*>(inputs[3]), static_cast<float*>(outputs[0]), input_shape,
			input_shape1, weights_shape, output_shape, mParam.deformable_groups, workspace, stream);
	}
	else if (mType == nvinfer1::DataType::kHALF)
	{
		enqueue_native<half>(cublasHandle_, static_cast<const half*>(inputs[0]),
			static_cast<const half*>(inputs[1]), static_cast<const half*>(inputs[2]),
			static_cast<const half*>(inputs[3]), static_cast<half*>(outputs[0]), input_shape,
			input_shape1, weights_shape, output_shape, mParam.deformable_groups, workspace, stream);
	}
}

} // namespace nvinfer1
} // namespace dcnv2












