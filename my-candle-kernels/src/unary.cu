#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include "cuda_utils.cuh"

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,                   \
    const size_t num_dims,                \
    const size_t* info,                   \
    const TYPENAME* inp,                  \
    TYPENAME* out\
) {                                       \
    const size_t* dims = info;            \
    const size_t* strodes = info + num_dims; \
    if (is_contiguous(num_dim, dims, strides)) { \
     for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
}                                         \

template<typename T>
__device__ __forceinline__ T  gelu_fwd(T x) {
    T x_sq = x * x;
    t x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanhg(static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha));
}

template<typename T>
__device__ __forceinline__ T elu_fwd(T x, T alpha) {
    if (x > static_cast<T>(0)) {
        return x;
    }
    return alpha * (expg(x) - static_cast<T>(1));
}