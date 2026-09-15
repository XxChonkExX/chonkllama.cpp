#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// VVM (Chonk Buffer) integration - active when built with GGML_VK_VVM_POOL
// and runtime env GGML_VK_VVM_POOL=1; otherwise stats returns "[]" and
// auto-pick returns NULL.
// Snapshot per-device free-VRAM budgets before a model load, then call
// ggml_vulkan_vvm_auto_pick() per tensor to distribute weights across
// Vulkan devices proportionally to their remaining budget.
GGML_BACKEND_API void ggml_vulkan_vvm_auto_begin(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_vulkan_vvm_auto_pick(size_t nbytes);
// Planner path: compute the full auto-placement plan ONCE from the model
// file's tensor inventory (names+sizes read from the GGUF itself), then
// resolve per tensor by NAME. The plan rediscovers the measured champion
// (dense on the fastest GPU, ~14/48 expert layers on GPU, rest on CPU)
// from first principles - no hand-tuned --n-cpu-moe needed.
// kv_bytes: expected KV cache size (reserves dense-device headroom).
// Falls back to budget-based auto_pick() for tensors outside the plan.
GGML_BACKEND_API void ggml_vulkan_vvm_auto_plan(const char * model_path, uint64_t kv_bytes);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_vulkan_vvm_auto_pick_named(const char * tensor_name, size_t nbytes);
// Returns a JSON array of live Chonk Buffer pools: device, block size, block
// count, allocations, used/free/capacity bytes, fragmentation. The returned
// pointer is valid until the next call.
GGML_BACKEND_API const char * ggml_vulkan_vvm_stats_json(void);

#ifdef  __cplusplus
}
#endif
