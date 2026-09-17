#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Chonk Buffer pool statistics (GGML_HIP_VVM_POOL builds): JSON array of
// per-device pool states, same schema as ggml_vulkan_vvm_stats_json.
GGML_BACKEND_API const char * ggml_hip_vvm_stats_json(void);
// Planner path: compute the full auto-placement plan ONCE from the model
// file's tensor inventory, then resolve per tensor by NAME. Falls back to
// NULL (default placement) for tensors outside the plan - note this differs
// from the Vulkan pick_named, which falls back to budget-based auto_pick.
// kv_bytes: expected KV cache size (reserves dense-device headroom).
// Returned buffer-type pointers are valid until backend shutdown.
GGML_BACKEND_API void ggml_hip_vvm_auto_plan(const char * model_path, uint64_t kv_bytes);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_hip_vvm_auto_pick_named(const char * tensor_name, size_t nbytes);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
