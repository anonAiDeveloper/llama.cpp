#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // for ggml_backend_buffer_init()
#include "ggml-impl.h"
#include "ggml-cuda-arena.h"

#include <cuda_runtime.h>
#include <new>

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
    // Fast D2D if src is device-backed
    if (!ggml_backend_buffer_is_host(src->buffer))
    {
        cuda_arena *ctx = (cuda_arena*) b->context;
        cudaSetDevice(ctx->device);
        cudaMemcpy(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice);
        return true;
    }
    return false;
}

static enum ggml_status cuda_arena_init_tensor(ggml_backend_buffer_t b, ggml_tensor *t)
{
    // Optional: zero padded tail for quantized tensors, mirroring ggml-cuda.
    // size_t orig = ggml_nbytes(t);
    // size_t pad  = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(b), t);
    // if (pad > orig) cudaMemset((char*) t->data + orig, 0, pad - orig);
    (void)b; (void)t;    //to avoid not referenced warnings
    return ggml_status::GGML_STATUS_SUCCESS;
}

static const ggml_backend_buffer_i cuda_arena_iface = {
    /*free_buffer*/   cuda_arena_free_buffer,
    /*get_base*/      cuda_arena_get_base,
    /*init_tensor*/   cuda_arena_init_tensor,
    /*memset_tensor*/ nullptr,
    /*set_tensor*/    cuda_arena_set_tensor,
    /*get_tensor*/    cuda_arena_get_tensor,
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