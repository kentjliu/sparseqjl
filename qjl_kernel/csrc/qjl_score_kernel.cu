#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define EMB_DIM 128
#define FULL_MASK 0xffffffff


template <typename T>
__device__ float convert_to_float(T value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return 0.0f;
}

template <>
__device__ float convert_to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
__device__ float convert_to_float<float>(float value) {
    return value; 
}

template <>
__device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) {
    return static_cast<float>(value); 
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

template<typename T, typename Tproj>
__global__ void calc_score_kernel(
    T* query_states,
    const uint8_t* key_quant,
    const uint8_t* key_outlier_quant,
    T* key_norm,
    T* key_outlier_norm,
    const uint8_t* outlier_indices,
    const float* query_sketch,
    const Tproj* rand_prj,
    float* scores,
    int batch_size, int head_size, int n_size, int group_size, int sketch_dim, int outlier_sketch_dim, int emb_dim,
    int outlier_counts) {

    size_t bh = blockIdx.x;
    size_t n = blockIdx.y;
    size_t threadLane = threadIdx.x;
    size_t wIdx = threadIdx.y;
    size_t gIdx = blockIdx.z * WARP_SIZE;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_outlier_indices = (bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (bh * sketch_dim);
    const float* q_sketch = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm = key_norm + base_index_key_norm;
    const T* outlier_norm = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (bh * emb_dim);
    const T* query = query_states + base_index_query_states;

    // load query states into shared memory
    __shared__ float shared_query[EMB_DIM];
    size_t tIdx = wIdx * WARP_SIZE + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < emb_dim; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_query[tile_idx] = convert_to_float<T>(query[tile_idx]);
    }
    // load outlier indices into shared buffer
    __shared__ uint8_t shared_outlier_ind[WARP_SIZE];
    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }
    // allocate shared memory to inner products of quantized keys or outliers with query_sketch
    __shared__ float shared_innprod[WARP_SIZE];
    __shared__ float shared_outlier_innprod[WARP_SIZE];
    if (wIdx == 0) {
        shared_innprod[threadLane] = 0.0;
        shared_outlier_innprod[threadLane] = 0.0;
    }
    __syncthreads();

    // reserve shared memory for a block of query sketch and query outlier sketch
    __shared__ float shared_q_sketch[WARP_SIZE][8];
    __shared__ float shared_q_outliers_sketch[WARP_SIZE][8];
    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8*WARP_SIZE)){
        // load a block of query sketch and compute query outlier sketch
        for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
            shared_q_sketch[q_idx/8][q_idx%8] = 0.0;
            shared_q_outliers_sketch[q_idx/8][q_idx%8] = 0.0;
            if (chnl_tile+q_idx < sketch_dim){
                shared_q_sketch[q_idx/8][q_idx%8] = q_sketch[chnl_tile+q_idx];
                for (size_t i{0}; i < outlier_counts; i++){
                    int otlr_idx = shared_outlier_ind[i];
                    shared_q_outliers_sketch[q_idx/8][q_idx%8] += shared_query[otlr_idx] * convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile+q_idx]); // convert_to_float(const_query[bh][otlr_idx])
                }
            }
        }

        for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
            // load key quant and outlier quant
            uint8_t key_quant_buffer = k_quant[grp_tile*hash_dim + chnl_tile/8 + threadLane];
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile + 8*threadLane < outlier_sketch_dim){
                outlier_quant_buffer = outlier_quant[grp_tile*outlier_hash_dim + chnl_tile/8 + threadLane];
            }
            __syncthreads();

            float k_inner_prod = 0.0;
            float outlier_inner_prod = 0.0;
            for (int shift = 0; shift < 8; shift++) {
                float q_sketch_val = shared_q_sketch[threadLane][shift] - shared_q_outliers_sketch[threadLane][shift];
                k_inner_prod += (((key_quant_buffer >> shift)&1) ? q_sketch_val :-q_sketch_val);
                if (chnl_tile + 8*threadLane < outlier_sketch_dim) {
                    float q_otlr_sketch_val = shared_q_outliers_sketch[threadLane][shift];
                    outlier_inner_prod += (((outlier_quant_buffer >> shift)&1) ? q_otlr_sketch_val :-q_otlr_sketch_val);
                }
            }
            __syncthreads();

            k_inner_prod = warpReduceSum(k_inner_prod);
            outlier_inner_prod = warpReduceSum(outlier_inner_prod);
            __syncthreads();
            if (threadLane == 0) {
                shared_innprod[grp_tile] += k_inner_prod;
                shared_outlier_innprod[grp_tile] += outlier_inner_prod;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    if (gIdx+threadLane >= group_size) return;
    if (wIdx == 0) {
        float scl = sqrtf(M_PI_2) / static_cast<float>(sketch_dim);
        float scl_otlr = sqrtf(M_PI_2) / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float norm_k = sqrtf(pow(convert_to_float<T>(k_norm[threadLane]), 2) - pow(norm_otlr, 2));
        float score = scl * norm_k * shared_innprod[threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[threadLane];
        scores[(bh * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}


template <typename T, typename Tproj>
torch::Tensor QJLScoreCudaTemplate(
    torch::Tensor key_quant,
    torch::Tensor key_outlier_quant,
    torch::Tensor key_norm,
    torch::Tensor key_outlier_norm,
    torch::Tensor outlier_indices,
    torch::Tensor query_sketch,
    torch::Tensor query_states,
    torch::Tensor rand_prj) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);

    int batch = key_quant.size(0);
    int head = key_quant.size(1);
    int n = key_quant.size(2);
    int group_size = key_quant.size(3);
    int emb_dim = query_states.size(3);
    int sketch_dim = rand_prj.size(1);
    int outlier_sketch_dim = 8*key_outlier_quant.size(4);
    int outlier_counts = outlier_indices.size(3);

    auto scores = torch::zeros({batch, head, n * group_size, 1}, options).contiguous();
    
    auto query_states_ptr = query_states.data_ptr<T>();
    auto key_norm_ptr = key_norm.data_ptr<T>();
    auto key_outlier_norm_ptr = key_outlier_norm.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();

    int blocksPerGroup = (group_size + WARP_SIZE - 1) / WARP_SIZE;
    dim3 numBlocks(batch * head, n, blocksPerGroup);
    dim3 threadsPerBlockDim(WARP_SIZE, WARPS_PER_BLOCK, 1);

    calc_score_kernel<<<numBlocks, threadsPerBlockDim>>>(
        query_states_ptr,
        key_quant.data_ptr<uint8_t>(),
        key_outlier_quant.data_ptr<uint8_t>(),
        key_norm_ptr,
        key_outlier_norm_ptr,
        outlier_indices.data_ptr<uint8_t>(),
        query_sketch.data_ptr<float>(),
        rand_prj_ptr,
        scores.data_ptr<float>(),
        batch, head, n, group_size, sketch_dim, outlier_sketch_dim, emb_dim, outlier_counts);

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_score_cuda_half_half", &QJLScoreCudaTemplate<c10::Half, c10::Half>, "Cuda kernel to calculate scores fully parallel using Half precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_half_float", &QJLScoreCudaTemplate<c10::Half, float>, "Cuda kernel to calculate scores fully parallel using Half to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_float_float", &QJLScoreCudaTemplate<float, float>, "Cuda kernel to calculate scores fully parallel using Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_bf16_bf16", &QJLScoreCudaTemplate<at::BFloat16, at::BFloat16>, "Cuda kernel to calculate scores fully parallel using BF16 precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_bf16_float", &QJLScoreCudaTemplate<at::BFloat16, float>, "Cuda kernel to calculate scores fully parallel using BF16 to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));
}
