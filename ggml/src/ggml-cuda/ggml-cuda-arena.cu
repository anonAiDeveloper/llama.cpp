#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // for ggml_backend_buffer_init()
#include "ggml-impl.h"
#include "ggml-cuda-arena.h"

#include "ggml-cuda/common.cuh"

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

struct ggml_backend_cuda_arena_context {
    int device;

    // Borrowed pointers. Do not free these from arena.
    ggml_backend_t backend_cuda = nullptr;
    ggml_backend_t backend_cpu  = nullptr;
    ggml_backend_cuda_arena_context(int device_) : device(device_) {};
};

static const char * ggml_backend_cuda_arena_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "CUDA_ARENA";
}

static void ggml_backend_cuda_arena_free(ggml_backend_t backend) {
    ggml_backend_cuda_arena_context * cuda_ctx = (ggml_backend_cuda_arena_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static enum ggml_status ggml_backend_cuda_arena_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * graph
) {
    GGML_UNUSED(backend);
    GGML_UNUSED(graph);

    GGML_LOG_ERROR("%s: CUDA_ARENA received graph_compute while supports_op=false\n", __func__);
    return GGML_STATUS_FAILED;
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
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_arena_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_arena_device_context * ctx = (ggml_backend_cuda_arena_device_context *)dev->context;

    props->name        = ggml_backend_cuda_arena_device_get_name(dev);
    props->description = ggml_backend_cuda_arena_device_get_description(dev);
    props->type        = ggml_backend_cuda_arena_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
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

//TODO: support GGML_OP_MUL_MAT_ID for MoE
static bool ggml_backend_cuda_arena_device_supports_op(
    ggml_backend_dev_t dev,
    const ggml_tensor * op
) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
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
    /* .offload_op              = */ NULL,
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

    auto * ctx = (ggml_backend_cuda_arena_context *) backend_arena->context;

    ctx->backend_cuda = backend_cuda;
    ctx->backend_cpu  = backend_cpu;
}

//TODO: incomplete, this is the cuda version. make it the cuda_arena version
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