// ggml-vvm-common.h — shared Chonk Buffer hook logic for the Vulkan and HIP
// backends (ggml-vulkan.cpp, ggml-cuda.cu).
//
// The two hooks were ~500 lines of near-identical code (env parsing, pool
// config, stats JSON, auto-plan, per-tensor pick) that had already drifted:
// different HEAP_FRACTION env names, warn-vs-silent block-size handling,
// asymmetric pick fallbacks. This header is the single choke point; each
// backend keeps only its device plumbing (pool storage, locking, buffer
// allocation, device-specific flags) plus thin wrappers.
//
// Backend differences that remain DELIBERATE (do not "unify" without
// measuring): Vulkan sets enableDeviceAddress/memoryPriority from its
// device caps; Vulkan's alloc hook adds SHADER_DEVICE_ADDRESS usage;
// Vulkan's pick falls back to budget-based auto_pick while HIP returns
// nullptr (no budget machinery on the HIP side).

#pragma once

#include "vulkan_vm/vulkan_vm.hpp"

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Runtime enable switch (per-backend memoized wrapper calls this).
// ---------------------------------------------------------------------------

inline bool ggml_vvm_env_on(const char * env_name) {
    const char * env = getenv(env_name);
    return env != nullptr && env[0] == '1';
}

// ---------------------------------------------------------------------------
// PoolConfig construction: identical policy on both backends.
//
// Rationale preserved from the original hook comments:
// - 1 GiB blocks match ggml's suballocation size; unlimited count lets the
//   pool grow to the full heap.
// - VRAM budget: legit splits commit 92-98% of heap, so the hard fraction
//   cap stays OPT-IN (unified name GGML_VVM_HEAP_FRACTION; the legacy
//   GGML_VK_VVM_HEAP_FRACTION is still honored).
// - Chonk Chunks: 2 MiB bases + tiered small-alloc blocks cover the observed
//   serve-path small range (5/7 MB scratch, 41/58 MB context buffers).
// - PURE DEVICE_LOCAL default: ReBAR-mapped VRAM lost ~3x decode on Arc.
// ---------------------------------------------------------------------------

inline void ggml_vvm_default_pool_config(vvm::PoolConfig & pcfg,
                                         bool enable_device_address,
                                         float memory_priority,
                                         const char * log_tag) {
    pcfg.blockSize = 1ull * 1024ull * 1024ull * 1024ull;
    pcfg.maxBlocks = 0;
    pcfg.enableHostVisible = false;
    pcfg.enableExternal = false;
    pcfg.enableDeviceAddress = enable_device_address;
    pcfg.memoryPriority = memory_priority;

    if (const char * hf = getenv("GGML_VVM_HEAP_FRACTION")) {
        float v = (float)atof(hf);
        if (v > 0.0f && v <= 1.0f) pcfg.maxHeapFraction = v;
    } else if (const char * hf_old = getenv("GGML_VK_VVM_HEAP_FRACTION")) {
        float v = (float)atof(hf_old);
        if (v > 0.0f && v <= 1.0f) pcfg.maxHeapFraction = v;
    }

    if (const char * bs = getenv("GGML_VVM_BLOCK_SIZE")) {
        unsigned long long v = strtoull(bs, nullptr, 0);
        if (v >= 256ull * 1024ull && v <= 8ull * 1024ull * 1024ull * 1024ull) {
            pcfg.blockSize = v;
        } else if (v != 0) {
            GGML_LOG_WARN("%s: ignoring out-of-range GGML_VVM_BLOCK_SIZE=%llu (256 KiB..8 GiB)\n",
                          log_tag, v);
        }
    }
    pcfg.preferPureDeviceLocal = true;
    if (const char * pl = getenv("GGML_VVM_PURE_LOCAL")) {
        if (pl[0] == '0') {
            pcfg.preferPureDeviceLocal = false;
        }
    }
    if (const char * nd = getenv("GGML_VVM_NO_DEDICATED")) {
        if (nd[0] == '1') {
            pcfg.dedicatedAllocateInfo = false;
        }
    }
    pcfg.allocationAlignment = 2ull * 1024ull * 1024ull;
    pcfg.chunkTiers = {
        {  1ull * 1024ull * 1024ull,   8ull * 1024ull * 1024ull},
        {  4ull * 1024ull * 1024ull,  32ull * 1024ull * 1024ull},
        { 16ull * 1024ull * 1024ull,  64ull * 1024ull * 1024ull},
        { 64ull * 1024ull * 1024ull, 256ull * 1024ull * 1024ull},
    };
    if (const char * ba = getenv("GGML_VVM_BASE_ALIGN")) {
        unsigned long long v = strtoull(ba, nullptr, 0);
        pcfg.allocationAlignment = v;   // 0 disables (falls back to minAlignment)
    }
    if (const char * cm = getenv("GGML_VVM_CHUNK_MB")) {
        unsigned long long v = strtoull(cm, nullptr, 0);
        if (v == 0) {
            pcfg.smallAllocThreshold = 0;   // disable chunk routing
            pcfg.chunkBlockSize = 0;
            pcfg.chunkTiers.clear();
        } else {
            pcfg.chunkBlockSize = v * 1024ull * 1024ull;
        }
    }
}

// ---------------------------------------------------------------------------
// Stats JSON: one pool entry, shared schema for /vvm/stats on both backends.
// ---------------------------------------------------------------------------

inline void ggml_vvm_append_pool_json(std::string & json, bool & first,
                                      const char * devname,
                                      const vvm::PoolConfig & cfg,
                                      const vvm::PoolStats & s) {
    char buf[512];
    if (!first) json += ",";
    first = false;
    snprintf(buf, sizeof(buf),
        "{\"device\":\"%s\",\"blockSize\":%llu,\"blocks\":%u,"
        "\"allocations\":%u,\"dedicated\":%u,"
        "\"capacityBytes\":%llu,\"usedBytes\":%llu,\"freeBytes\":%llu,"
        "\"largestFreeBytes\":%llu,\"fragmentation\":%.3f}",
        devname,
        (unsigned long long)cfg.blockSize,
        s.blockCount, s.allocationCount, s.dedicatedCount,
        (unsigned long long)s.totalCapacity, (unsigned long long)s.totalUsed,
        (unsigned long long)s.totalFree, (unsigned long long)s.largestFreeBlock,
        (double)s.fragmentationRatio);
    json += buf;
}

// ---------------------------------------------------------------------------
// Auto-plan: compute the full PlacementPlan once from the model file's
// tensor inventory. Caller holds its own lock; this function is pure
// compute + logging. `source` filters to devices this binary can serve
// (Vulkan / Hip); the other runtime's entries describe the same physical
// GPUs the local listing also reports, so dropping them loses nothing.
// ---------------------------------------------------------------------------

// Planned KV hold: the planner accounts for the KV cache, but pools are
// created lazily AFTER the plan and the KV allocates even later - without a
// hold, expert tensors can consume the budget the KV needs (the 262K OOM
// sequence). Stash the planned KV bytes at plan time; get_pool() calls
// pool->reserve() right after create so the hold is enforced from the first
// allocation. Each backend stores one hold per device.
struct ggml_vvm_kv_hold {
    uint64_t bytes = 0;
    bool reserved = false;
};

// Reserve the planned KV once per pool (idempotent). Returns true when the
// hold is in place; false = refused (plan over-committed: callers keep the
// hold unreserved and the pool budget still protects late KV).
inline bool ggml_vvm_reserve_kv(vvm::UnifiedMemoryPool * pool, ggml_vvm_kv_hold & hold,
                                const char * log_tag, int device) {
    if (pool == nullptr || hold.reserved || hold.bytes == 0) {
        return hold.reserved;
    }
    if (!pool->reserve(hold.bytes)) {
        GGML_LOG_WARN("%s: VVM KV hold of %llu MiB refused (plan over-commits); "
                      "KV allocates best-effort\n",
                      log_tag, (unsigned long long)(hold.bytes / 1024 / 1024));
        return false;
    }
    hold.reserved = true;
    GGML_LOG_INFO("%s: VVM KV hold %llu MiB reserved (device %d)\n",
                  log_tag, (unsigned long long)(hold.bytes / 1024 / 1024), device);
    return true;
}

inline bool ggml_vvm_compute_plan(vvm::DeviceSource source,
                                  const char * model_path, uint64_t kv_bytes,
                                  const char * log_tag, vvm::PlacementPlan & out) {
    if (model_path == nullptr || model_path[0] == 0) {
        return false;
    }
    auto specs = vvm::read_gguf_inventory(model_path);
    if (specs.empty()) {
        GGML_LOG_WARN("%s: VVM auto-plan: no inventory for %s, budget fallback\n", log_tag, model_path);
        return false;
    }
    auto devices = vvm::enumerate_all_devices();
    std::vector<vvm::BackendDeviceInfo> mine;
    for (auto & d : devices) {
        if (d.source == source) {
            mine.push_back(d);
        }
    }
    // Heap fraction shared with the pool cap: the plan must budget against
    // the same ceiling the pool enforces.
    float planFrac = 0.90f;
    if (const char * hf = getenv("GGML_VVM_HEAP_FRACTION")) {
        float v = (float)atof(hf);
        if (v > 0.0f && v <= 1.0f) planFrac = v;
    }
    out = vvm::auto_place_experts(mine, specs, kv_bytes, planFrac);
    GGML_LOG_INFO("%s: VVM auto-plan: %s\n", log_tag, out.summary);
    return true;
}

// ---------------------------------------------------------------------------
// Per-tensor pick: resolve a placement decision from a stored plan.
// Returns {have_plan, want_cpu, device_index}; the caller maps
// device_index through its own backend buffer-type function. want_cpu covers
// LookupTable, out-of-range expert layers, and missing dense devices.
// A tensor outside any plan entry (non-expert, non-LUT on a plan without
// dense info) resolves to the plan's dense device.
// ---------------------------------------------------------------------------

struct ggml_vvm_pick {
    bool have_plan = false;
    bool want_cpu = false;
    int32_t device_index = -1;
};

inline ggml_vvm_pick ggml_vvm_pick_from_plan(const vvm::PlacementPlan & plan,
                                             const char * tensor_name) {
    ggml_vvm_pick out;
    if (tensor_name == nullptr) {
        return out;
    }
    out.have_plan = true;
    int32_t layer = -1;
    const vvm::TensorClass cls = vvm::classify_tensor(tensor_name, &layer);
    if (cls == vvm::TensorClass::LookupTable) {
        out.want_cpu = true;
        return out;
    }
    if (cls == vvm::TensorClass::Expert) {
        if (layer < 0 || static_cast<size_t>(layer) >= plan.experts.size()) {
            out.want_cpu = true;   // outside the plan: safe direction is CPU
            return out;
        }
        const vvm::ExpertPlacement & ep = plan.experts[static_cast<size_t>(layer)];
        out.want_cpu = ep.onCpu || ep.deviceIndex < 0;
        out.device_index = ep.deviceIndex;
        return out;
    }
    out.want_cpu = plan.denseDeviceIndex < 0;
    out.device_index = plan.denseDeviceIndex;
    return out;
}
