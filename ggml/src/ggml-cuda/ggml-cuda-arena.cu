#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // for ggml_backend_buffer_init()
#include "ggml-impl.h"
#include "ggml-cuda-arena.h"

#include "ggml-cuda/common.cuh"

#include <cstddef>
#include <cuda_runtime.h>
#include <new>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

struct cuda_arena
{
    int    device;
    void * base;
    size_t size;
    // optional: cudaStream_t stream;
};

static void * cuda_arena_get_base(ggml_backend_buffer_t buffer)
{
    return ((cuda_arena*) buffer->context)->base;
}

static void cuda_arena_free_buffer(ggml_backend_buffer_t b)
{
    cuda_arena *ctx = (cuda_arena*) b->context;
    cudaSetDevice(ctx->device);
    cudaFree(ctx->base);               // or VMM unmap if you used cuMem* APIs
    delete ctx;
}

static void cuda_arena_clear(ggml_backend_buffer_t b, uint8_t v)
{
    cuda_arena *ctx = (cuda_arena*) b->context;
    cudaSetDevice(ctx->device);
    cudaMemset(ctx->base, v, ctx->size);
}

static void cuda_arena_set_tensor(ggml_backend_buffer_t b, ggml_tensor *t, const void *data, size_t off, size_t sz)
{
    cuda_arena *ctx = (cuda_arena*) b->context;
    cudaSetDevice(ctx->device);
    cudaMemcpy((char*) t->data + off, data, sz, cudaMemcpyHostToDevice);
}

static void cuda_arena_get_tensor(ggml_backend_buffer_t b, const ggml_tensor *t, void *data, size_t off, size_t sz)
{
    cuda_arena *ctx = (cuda_arena*) b->context;
    cudaSetDevice(ctx->device);
    cudaMemcpy(data, (const char*) t->data + off, sz, cudaMemcpyDeviceToHost);
}

static bool cuda_arena_cpy_tensor(ggml_backend_buffer_t b, const ggml_tensor *src, ggml_tensor *dst)
{
    if (!ggml_backend_buffer_is_host(src->buffer))
    {
        cuda_arena *ctx = (cuda_arena*) b->context;
        cudaSetDevice(ctx->device);

        // Use the *device allocation size*, and assert src/dst agree:
        ggml_backend_buffer_type_t src_buft = ggml_backend_buffer_get_type(src->buffer);
        ggml_backend_buffer_type_t dst_buft = ggml_backend_buffer_get_type(dst->buffer);
        size_t src_sz = ggml_backend_buft_get_alloc_size(src_buft, const_cast<ggml_tensor *>(src));
        size_t dst_sz = ggml_backend_buft_get_alloc_size(dst_buft, dst);
        size_t sz = src_sz;
        if (src_sz != dst_sz) {
            // be strict; mismatched layouts is a bug for weights
            // you can also choose: sz = std::min(src_sz, dst_sz);
            GGML_ASSERT(src_sz == dst_sz);
        }

        cudaMemcpy(dst->data, src->data, sz, cudaMemcpyDeviceToDevice);
        return true;
    }
    return false;
}

static enum ggml_status cuda_arena_init_tensor(ggml_backend_buffer_t b, ggml_tensor *t)
{
    cuda_arena *ctx = (cuda_arena*) b->context;
    cudaSetDevice(ctx->device);

    size_t logical = ggml_nbytes(t);
    size_t device  = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(b), t);
    if (device > logical)
        cudaMemset((char*) t->data + logical, 0, device - logical);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_buffer_i cuda_arena_iface = {
    /*free_buffer*/   cuda_arena_free_buffer,
    /*get_base*/      cuda_arena_get_base,
    /*init_tensor*/   cuda_arena_init_tensor,
    /*memset_tensor*/ nullptr,
    /*set_tensor*/    cuda_arena_set_tensor,
    /*get_tensor*/    cuda_arena_get_tensor,
    /*set_tensor_2d*/ nullptr,
    /*get_tensor_2d*/ nullptr,
    /*cpy_tensor*/    cuda_arena_cpy_tensor,
    /*clear*/         cuda_arena_clear,
    /*reset*/         nullptr,
};

ggml_backend_buffer_t ggml_cuda_arena_create_on(ggml_backend_dev_t dev, size_t bytes, int device_ordinal)
{
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev); // keep CUDA buft
    cuda_arena *ctx = new(std::nothrow) cuda_arena{};
    if (!ctx)
        return nullptr;

    // pick device ordinal from dev if you track it; otherwise pass it in
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    ctx->device = device_ordinal;
    ctx->size   = bytes;

    cudaSetDevice(ctx->device);
    if (cudaMalloc(&ctx->base, bytes) != cudaSuccess)
    {
        delete ctx;
        return nullptr;
    }

    ggml_backend_buffer_t arena = ggml_backend_buffer_init(buft, cuda_arena_iface, ctx, bytes);
    ggml_backend_buffer_set_usage(arena, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return arena;
}

size_t ggml_cuda_arena_alignment(ggml_backend_buffer_t arena)
{
    return ggml_backend_buffer_get_alignment(arena);
}

bool ggml_cuda_arena_place(ggml_backend_buffer_t arena, ggml_tensor *t, size_t off)
{
    void *base = ggml_backend_buffer_get_base(arena);
    return ggml_backend_tensor_alloc(arena, t, (char*) base + off) == GGML_STATUS_SUCCESS;
}

//This checks the free_buffer function signature to see if equals the cuda_arena one
static bool ggml_backend_buffer_is_cuda_arena(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == cuda_arena_free_buffer;
}

extern "C"
bool ggml_backend_buffer_is_cuda_arena_public(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_is_cuda_arena(buffer);   // call the static one
}

struct ggml_cuda_copy_event
{
    int device;
    cudaEvent_t event;
};

ggml_cuda_copy_event * ggml_cuda_copy_event_create(ggml_backend_buffer_t b)
{
    cuda_arena * ctx = (cuda_arena *) b->context;
    cudaSetDevice(ctx->device);

    ggml_cuda_copy_event * ev = new ggml_cuda_copy_event;
    ev->device = ctx->device;
    ev->event = nullptr;

    cudaEventCreateWithFlags(&ev->event, cudaEventDisableTiming);

    return ev;
}

void ggml_cuda_copy_event_destroy(ggml_cuda_copy_event * ev)
{
    if (!ev)
        return;

    cudaSetDevice(ev->device);

    if (ev->event)
        cudaEventDestroy(ev->event);

    delete ev;
}

void ggml_cuda_copy_event_wait(ggml_cuda_copy_event * ev)
{
    GGML_ASSERT(ev);

    cudaSetDevice(ev->device);
    cudaEventSynchronize(ev->event);
}

void ggml_cuda_arena_tensor_write_raw_async(ggml_backend_buffer_t b,
                                            ggml_tensor * t,
                                            const void * src,
                                            size_t nbytes,
                                            ggml_cuda_copy_event * ev) {
    cuda_arena * ctx = (cuda_arena *) b->context;
    cudaSetDevice(ctx->device);

    // Safety: destination must belong to this arena
    GGML_ASSERT(t->buffer == b);

    // And we must not write past the device allocation for this tensor
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(b);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(buft, t);
    GGML_ASSERT(nbytes <= dev_bytes);

    cudaStream_t stream = 0;

    // Raw H2D copy of the entire packed region (padded tail included)
    cudaMemcpyAsync(t->data, src, nbytes, cudaMemcpyHostToDevice, stream);

    if (ev)
        cudaEventRecord(ev->event, stream);
}

void ggml_cuda_arena_tensor_write_raw(ggml_backend_buffer_t b,
                                      ggml_tensor * t,
                                      const void * src,
                                      size_t nbytes) {
    cuda_arena * ctx = (cuda_arena *) b->context;
    cudaSetDevice(ctx->device);

    // Safety: destination must belong to this arena
    GGML_ASSERT(t->buffer == b);

    // And we must not write past the device allocation for this tensor
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(b);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(buft, t);
    GGML_ASSERT(nbytes <= dev_bytes);

    // Raw H2D copy of the entire packed region (padded tail included)
    cudaMemcpy(t->data, src, nbytes, cudaMemcpyHostToDevice);

    cudaStreamSynchronize(0); // force default-stream copy completion before returning
}

////////////////////////////////////////////////////////////////////////////////
// backend
////////////////////////////////////////////////////////////////////////////////

struct ggml_backend_cuda_arena_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    //int op_offload_min_batch_size;
};

struct ggml_cuda_arena_cpu_fb_key {
    const ggml_tensor * cpu_src0;

    ggml_type src1_type;
    ggml_type ids_type;
    ggml_type dst_type;

    int64_t src1_ne[4];
    int64_t ids_ne[4];
    int64_t dst_ne[4];

    size_t src1_nb[4];
    size_t ids_nb[4];
    size_t dst_nb[4];

    bool operator==(const ggml_cuda_arena_cpu_fb_key & other) const {
        if (cpu_src0  != other.cpu_src0)  return false;
        if (src1_type != other.src1_type) return false;
        if (ids_type  != other.ids_type)  return false;
        if (dst_type  != other.dst_type)  return false;

        for (int i = 0; i < 4; ++i) {
            if (src1_ne[i] != other.src1_ne[i]) return false;
            if (ids_ne [i] != other.ids_ne [i]) return false;
            if (dst_ne [i] != other.dst_ne [i]) return false;

            if (src1_nb[i] != other.src1_nb[i]) return false;
            if (ids_nb [i] != other.ids_nb [i]) return false;
            if (dst_nb [i] != other.dst_nb [i]) return false;
        }

        return true;
    }
};

struct ggml_cuda_arena_cpu_fb_key_hash {
    size_t operator()(const ggml_cuda_arena_cpu_fb_key & key) const {
        size_t h = (size_t) (uintptr_t) key.cpu_src0;

        auto mix = [&](uint64_t v) {
            h ^= (size_t) v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };

        mix((uint64_t) key.src1_type);
        mix((uint64_t) key.ids_type);
        mix((uint64_t) key.dst_type);

        for (int i = 0; i < 4; ++i) {
            mix((uint64_t) key.src1_ne[i]);
            mix((uint64_t) key.ids_ne [i]);
            mix((uint64_t) key.dst_ne [i]);

            mix((uint64_t) key.src1_nb[i]);
            mix((uint64_t) key.ids_nb [i]);
            mix((uint64_t) key.dst_nb [i]);
        }

        return h;
    }
};

struct ggml_cuda_arena_cpu_fb_plan {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * cpu_src1 = nullptr;
    ggml_tensor * cpu_ids  = nullptr;
    ggml_tensor * cpu_dst  = nullptr;

    ggml_cgraph * graph = nullptr;
};

struct ggml_cuda_arena_cpu_fallback_scratch {
    std::mutex mutex;

    std::unordered_map<
        ggml_cuda_arena_cpu_fb_key,
        ggml_cuda_arena_cpu_fb_plan,
        ggml_cuda_arena_cpu_fb_key_hash
    > plans;
};

struct ggml_backend_cuda_arena_context {
    int device;

    // Borrowed pointers. Do not free these from arena.
    ggml_backend_t backend_cuda = nullptr;
    ggml_backend_t backend_cpu  = nullptr;

    ggml_cuda_arena_offloader_i offloader = {};
    ggml_cuda_arena_cpu_fallback_scratch cpu_fb = {};

    ggml_backend_cuda_arena_context(int device_) : device(device_) {};
};

static const char * ggml_backend_cuda_arena_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "CUDA_ARENA";
}

static void ggml_backend_cuda_arena_free(ggml_backend_t backend) {
    ggml_backend_cuda_arena_context * ctx = (ggml_backend_cuda_arena_context *) backend->context;

    for (auto & kv : ctx->cpu_fb.plans) {
        ggml_cuda_arena_cpu_fb_plan & plan = kv.second;

        if (plan.buffer != nullptr) {
            ggml_backend_buffer_free(plan.buffer);
            plan.buffer = nullptr;
        }

        if (plan.ctx != nullptr) {
            ggml_free(plan.ctx);
            plan.ctx = nullptr;
        }
    }

    ctx->cpu_fb.plans.clear();

    delete ctx;
    delete backend;
}

static ggml_cuda_arena_cpu_fb_key ggml_cuda_arena_make_cpu_fb_key(
    const ggml_tensor * cpu_src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    const ggml_tensor * dst
) {
    ggml_cuda_arena_cpu_fb_key key = {};

    key.cpu_src0  = cpu_src0;
    key.src1_type = src1->type;
    key.ids_type  = ids->type;
    key.dst_type  = dst->type;

    for (int i = 0; i < 4; ++i) {
        key.src1_ne[i] = src1->ne[i];
        key.ids_ne [i] = ids ->ne[i];
        key.dst_ne [i] = dst ->ne[i];

        key.src1_nb[i] = src1->nb[i];
        key.ids_nb [i] = ids ->nb[i];
        key.dst_nb [i] = dst ->nb[i];
    }

    return key;
}

static ggml_cuda_arena_cpu_fb_plan * ggml_cuda_arena_get_or_create_cpu_fb_plan(
    ggml_backend_cuda_arena_context * ctx,
    const ggml_cuda_arena_cpu_fb_key & key,
    ggml_tensor * cpu_src0,
    ggml_tensor * src1,
    ggml_tensor * ids,
    ggml_tensor * node
) {
    auto it = ctx->cpu_fb.plans.find(key);
    if (it != ctx->cpu_fb.plans.end())
        return &it->second;

    ggml_init_params params = {
        /* .mem_size   = */ 256 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    // Create one persistent metadata context for this CPU fallback shape.
    ggml_context * plan_ctx = ggml_init(params);

    if (plan_ctx == nullptr) {
        GGML_LOG_ERROR("%s: ggml_init failed\n", __func__);
        return nullptr;
    }

    // Create persistent CPU input metadata matching the live activation layout.
    ggml_tensor * cpu_src1 = ggml_dup_tensor_layout_public(plan_ctx, src1);

    // Create persistent CPU input metadata matching the live expert-id layout.
    ggml_tensor * cpu_ids = ggml_dup_tensor_layout_public(plan_ctx, ids);

    if (cpu_src1 == nullptr || cpu_ids == nullptr) {
        ggml_free(plan_ctx);
        return nullptr;
    }

    ggml_set_name(cpu_src1, "cuda_arena_fb_src1");
    ggml_set_name(cpu_ids,  "cuda_arena_fb_ids");

    // Build the persistent CPU MUL_MAT_ID op using the permanent CPU expert weight.
    ggml_tensor * cpu_dst = ggml_mul_mat_id(plan_ctx, cpu_src0, cpu_src1, cpu_ids);

    if (cpu_dst == nullptr) {
        ggml_free(plan_ctx);
        return nullptr;
    }

    ggml_set_name(cpu_dst, "cuda_arena_fb_dst");

    // Build the persistent tiny CPU graph for this fallback op.
    ggml_cgraph * cpu_graph = ggml_new_graph(plan_ctx);

    if (cpu_graph == nullptr) {
        ggml_free(plan_ctx);
        return nullptr;
    }

    // Attach the CPU fallback output to the persistent tiny graph.
    ggml_build_forward_expand(cpu_graph, cpu_dst);

    // Allocate persistent CPU backend storage for this fallback plan.
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(plan_ctx, ctx->backend_cpu);

    if (cpu_buffer == nullptr) {
        ggml_free(plan_ctx);
        return nullptr;
    }

    ggml_cuda_arena_cpu_fb_plan plan = {};
    plan.ctx      = plan_ctx;
    plan.buffer   = cpu_buffer;
    plan.cpu_src1 = cpu_src1;
    plan.cpu_ids  = cpu_ids;
    plan.cpu_dst  = cpu_dst;
    plan.graph    = cpu_graph;

    // Cache the completed CPU fallback plan for reuse during graph_compute.
    auto inserted = ctx->cpu_fb.plans.emplace(key, plan);

    GGML_LOG_INFO("%s: created CPU fallback plan for %s\n", __func__, ggml_get_name(node) ? ggml_get_name(node) : "(null)");

    return &inserted.first->second;
}

enum ggml_status ggml_backend_cuda_arena_prebuild_cpu_fallbacks(ggml_backend_t backend_arena, ggml_cgraph * graph)
{
    auto * ctx = (ggml_backend_cuda_arena_context *) backend_arena->context;

    GGML_ASSERT(ctx->backend_cpu != nullptr);
    GGML_ASSERT(ctx->offloader.get_cpu_mirror != nullptr);

    int prepared = 0;

    std::lock_guard<std::mutex> lock(ctx->cpu_fb.mutex);

    for (int i = 0; i < graph->n_nodes; ++i)
    {
        ggml_tensor * node = graph->nodes[i];

        if (node->op != GGML_OP_MUL_MAT_ID)
            continue;

        ggml_tensor * src0 = node->src[0];
        ggml_tensor * src1 = node->src[1];
        ggml_tensor * ids  = node->src[2];

        if (!src0 || !src1 || !ids)
            continue;

        // Resolve the graph's CUDA_ARENA/proxy expert weight to its permanent CPU mirror.
        ggml_tensor * cpu_src0 = ctx->offloader.get_cpu_mirror(ctx->offloader.user_data, src0);

        if (cpu_src0 == nullptr)
            continue;

        // Create the stable cache key for this CPU fallback shape.
        ggml_cuda_arena_cpu_fb_key key = ggml_cuda_arena_make_cpu_fb_key(cpu_src0, src1, ids, node);

        // Build or reuse the persistent CPU fallback graph for this MoE matmul.
        ggml_cuda_arena_cpu_fb_plan * plan = ggml_cuda_arena_get_or_create_cpu_fb_plan(ctx, key, cpu_src0, src1, ids, node);

        if (plan == nullptr) {
            GGML_LOG_ERROR("%s: failed to create CPU fallback plan for node=%s\n", __func__, ggml_get_name(node) ? ggml_get_name(node) : "(null)");
            return GGML_STATUS_ALLOC_FAILED;
        }

        ++prepared;
    }

    GGML_LOG_INFO("%s: prepared %d CPU fallback plan(s), cached total=%zu\n", __func__, prepared, ctx->cpu_fb.plans.size());

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_cuda_arena_cpu_fallback_mul_mat_id(ggml_backend_t backend, ggml_tensor * node)
{
    auto * ctx = (ggml_backend_cuda_arena_context *) backend->context;

    GGML_ASSERT(ctx->backend_cpu != nullptr);
    GGML_ASSERT(ctx->offloader.get_cpu_mirror != nullptr);

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * ids  = node->src[2];

    // Resolve the CUDA_ARENA/proxy expert weight tensor back to its permanent CPU mirror.
    ggml_tensor * cpu_src0 = ctx->offloader.get_cpu_mirror(ctx->offloader.user_data, src0);

    if (cpu_src0 == nullptr) {
        GGML_LOG_ERROR("%s: no CPU mirror for src0=%s node=%s\n", __func__, ggml_get_name(src0) ? ggml_get_name(src0) : "(null)", ggml_get_name(node) ? ggml_get_name(node) : "(null)");
        return GGML_STATUS_FAILED;
    }

    // Recreate the same key used during graph-time CPU fallback prebuild.
    ggml_cuda_arena_cpu_fb_key key = ggml_cuda_arena_make_cpu_fb_key(cpu_src0, src1, ids, node);

    std::lock_guard<std::mutex> lock(ctx->cpu_fb.mutex);

    auto it = ctx->cpu_fb.plans.find(key);

    if (it == ctx->cpu_fb.plans.end()) {
        GGML_LOG_ERROR("%s: missing prebuilt CPU fallback plan for node=%s\n", __func__, ggml_get_name(node) ? ggml_get_name(node) : "(null)");
        return GGML_STATUS_FAILED;
    }

    ggml_cuda_arena_cpu_fb_plan & plan = it->second;

    // Copy the live activation/work tensor into the prebuilt CPU fallback input.
    ggml_backend_tensor_copy(src1, plan.cpu_src1);

    // Copy the live selected expert IDs into the prebuilt CPU fallback input.
    ggml_backend_tensor_copy(ids, plan.cpu_ids);

    // Execute the prebuilt tiny CPU fallback graph.
    enum ggml_status status = ggml_backend_graph_compute(ctx->backend_cpu, plan.graph);

    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: CPU fallback graph_compute failed for node=%s status=%d\n", __func__, ggml_get_name(node) ? ggml_get_name(node) : "(null)", (int) status);
        return status;
    }

    // Copy the CPU fallback output back into the original CUDA_ARENA node output.
    ggml_backend_tensor_copy(plan.cpu_dst, node);

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cuda_arena_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * graph
) {
    auto * ctx = (ggml_backend_cuda_arena_context *) backend->context;

    GGML_ASSERT(ctx->backend_cpu != nullptr);
    GGML_ASSERT(ctx->backend_cuda != nullptr);
    GGML_ASSERT(ctx->offloader.get_cpu_mirror != nullptr);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];

        if (node->op != GGML_OP_MUL_MAT_ID) {
            GGML_LOG_ERROR("%s: unexpected op %s node=%s\n", __func__, ggml_op_name(node->op), ggml_get_name(node) ? ggml_get_name(node) : "(null)");
            return GGML_STATUS_FAILED;
        }

        // Execute this CUDA_ARENA MoE matmul through its prebuilt CPU fallback plan.
        enum ggml_status status = ggml_cuda_arena_cpu_fallback_mul_mat_id(backend, node);

        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: CPU fallback failed for node[%d] %s status=%d\n", __func__, i, ggml_get_name(node) ? ggml_get_name(node) : "(null)", (int) status);
            return status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_cuda_arena_interface = {
    /* .get_name                = */ ggml_backend_cuda_arena_get_name,
    /* .free                    = */ ggml_backend_cuda_arena_free,
    /* .set_tensor_async        = */ NULL, //ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ NULL, //ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL, //ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ NULL, //ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL, //ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ NULL, //ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_arena_graph_compute,
    /* .event_record            = */ NULL, //ggml_backend_cuda_event_record,
    /* .event_wait              = */ NULL, //ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ NULL, //ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_arena_guid() {
    static ggml_guid guid = { 0xdb, 0xcc, 0x56, 0xd9, 0xc4, 0x52, 0x48, 0x32, 0xac, 0xef, 0xa8, 0x26, 0x4f, 0xe4, 0x6e, 0x33 };
    return &guid;
}

bool ggml_backend_is_cuda_arena(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_arena_guid());
}

static const char * ggml_backend_cuda_arena_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_arena_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_arena_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_arena_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemGetInfo(free, total));

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_arena_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__)

}

static enum ggml_backend_dev_type ggml_backend_cuda_arena_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);

    // CUDA_ARENA is not a model placement GPU.
    // It is a scheduler-side accelerator backend for selected MoE ops.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_cuda_arena_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;

    props->name        = ggml_backend_cuda_arena_device_get_name(dev);
    props->description = ggml_backend_cuda_arena_device_get_description(dev);
    props->type        = ggml_backend_cuda_arena_device_get_type(dev);

    // CUDA_ARENA must not claim the physical CUDA PCI id.
    // Otherwise llama_prepare_model_devices deduplicates it against CUDA0.
    props->device_id   = nullptr;

    ggml_backend_cuda_arena_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    //TODO: What are these for? We need to be careful here and research these
    //props->caps = {
    //    /* .async                 = */ true,
    //    /* .host_buffer           = */ host_buffer,
    //    /* .buffer_from_host_ptr  = */ false,
    //    /* .events                = */ events,
    //};

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,       //as long as get_host_buffer_type is NULL, this must be false
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_cuda_arena_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;
    return ggml_backend_cuda_arena_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_arena_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static bool ggml_backend_cuda_arena_device_supports_op(
    ggml_backend_dev_t dev,
    const ggml_tensor * op
) {
    GGML_UNUSED(dev);

    if (op->op != GGML_OP_MUL_MAT_ID)
        return false;

    const char * name = ggml_get_name(op);

    // Start narrow. Adjust names to whatever your graph actually emits.
    if (name
         && (strstr(name, "ffn_moe")  ||
        strstr(name, "ffn_gate") ||
        strstr(name, "ffn_up")   ||
        strstr(name, "ffn_down"))
    )
    {
        GGML_LOG_INFO("%s: returning true for %s\n", __func__, name);
        return true;
    }

    return false;
}

//TODO: Should determine which buffer llama actually thinks the MoE layers are on, I think only one type is actually valid
static bool ggml_backend_cuda_arena_device_supports_buft(
    ggml_backend_dev_t dev,
    ggml_backend_buffer_type_t buft
) {
    auto * ctx = (ggml_backend_cuda_arena_device_context *) dev->context;

    if (buft == ggml_backend_cpu_buffer_type())
        return true;

    if (buft == ggml_backend_cuda_buffer_type(ctx->device))
        return true;

    //if (buft == ggml_backend_cuda_arena_buffer_type(ctx->device))
    //    return true;

    return false;
}

static bool ggml_backend_cuda_arena_device_offload_op(
    ggml_backend_dev_t dev,
    const ggml_tensor * op
) {
    GGML_UNUSED(dev);

    if (op->op != GGML_OP_MUL_MAT_ID)
        return false;

    const char * name = ggml_get_name(op);

    if (name 
         && (strstr(name, "ffn_moe_gate") ||
        strstr(name, "ffn_moe_up")   ||
        strstr(name, "ffn_moe_down"))
    )
    {
        GGML_LOG_INFO("%s: returning true for %s\n", __func__, name);
        return true;
    }

    return false;
}

static const ggml_backend_device_i ggml_backend_cuda_arena_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_arena_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_arena_device_get_description,
    /* .get_memory              = */ NULL,
    /* .get_type                = */ ggml_backend_cuda_arena_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_arena_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_arena_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_arena_device_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL, //ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_arena_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_arena_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_arena_device_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static const char * ggml_backend_cuda_arena_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_ARENA_NAME;
}

struct ggml_backend_cuda_arena_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};


static size_t ggml_backend_cuda_arena_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_arena_reg_context * ctx = (ggml_backend_cuda_arena_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_arena_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_arena_reg_context * ctx = (ggml_backend_cuda_arena_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

//TODO: determine if any of these are needed. Possibly none are needed
static void * ggml_backend_cuda_arena_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0)
        return nullptr;
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0)
        return nullptr;
    if (strcmp(name, "ggml_backend_get_features") == 0)
        return nullptr;
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_arena_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_arena_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_arena_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_arena_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_arena_reg_get_proc_address,
};
// backend registry
ggml_backend_reg_t ggml_backend_cuda_arena_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_arena_reg_context * ctx = new ggml_backend_cuda_arena_reg_context;
            //const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            //TODO: are we going to eventually need ggml_cuda_info() or some equivalent? cudaGetDeviceCount is a temporary solution
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            //for (int i = 0; i < ggml_cuda_info().device_count; i++) {
            for (int i = 0; i < device_count; i++) {
                ggml_backend_cuda_arena_device_context * dev_ctx = new ggml_backend_cuda_arena_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_ARENA_NAME + std::to_string(i);

                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                //dev_ctx->description = prop.name;
                dev_ctx->description = std::string("CUDA arena on ") + prop.name;

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i));
                dev_ctx->pci_bus_id = pci_bus_id;
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                //dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* ggml_backend_device_i  .iface   = */ ggml_backend_cuda_arena_device_interface,
                    /* ggml_backend_reg_t     .reg     = */ &reg,
                    /* void *                 .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_arena_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

void ggml_backend_cuda_arena_set_delegates(
    ggml_backend_t backend_arena,
    ggml_backend_t backend_cuda,
    ggml_backend_t backend_cpu
) {
    GGML_ASSERT(ggml_backend_is_cuda_arena(backend_arena));

    ggml_backend_cuda_arena_context * ctx = (ggml_backend_cuda_arena_context *) backend_arena->context;

    ctx->backend_cuda = backend_cuda;
    ctx->backend_cpu  = backend_cpu;
}

void ggml_backend_cuda_arena_set_offloader(
    ggml_backend_t backend_arena,
    const ggml_cuda_arena_offloader_i * offloader
) {
    GGML_ASSERT(ggml_backend_is_cuda_arena(backend_arena));
    GGML_ASSERT(offloader);

    ggml_backend_cuda_arena_context * ctx = (ggml_backend_cuda_arena_context *) backend_arena->context;
    ctx->offloader = *offloader;
}

ggml_backend_t ggml_backend_cuda_arena_init(int device)
{
    ggml_backend_reg_t reg = ggml_backend_cuda_arena_reg();
    const size_t ndev = ggml_backend_reg_dev_count(reg);

    if (device < 0 || (size_t) device >= ndev) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_arena_context * ctx = new ggml_backend_cuda_arena_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_arena_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_arena_guid(),
        /* .iface   = */ ggml_backend_cuda_arena_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_arena_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_arena_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_arena_reg)