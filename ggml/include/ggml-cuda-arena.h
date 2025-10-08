#pragma once
#include "ggml-cuda.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Debug additions:

GGML_BACKEND_API bool ggml_backend_buffer_is_cuda_arena_public(ggml_backend_buffer_t buffer);

// Copy arbitrary number of bytes (up to alloc size) from host -> device tensor memory.
// Bypasses the ggml_nbytes() logical limit.
void ggml_cuda_arena_tensor_write_raw(ggml_backend_buffer_t arena,
                                      ggml_tensor * t,
                                      const void * src,
                                      size_t nbytes);

// Create one big CUDA arena buffer on a given device.
// Returns a ggml_backend_buffer_t whose buft == ggml_backend_dev_buffer_type(dev),
// but whose iface is your custom one (so you control addressing/copies).
GGML_BACKEND_API ggml_backend_buffer_t ggml_cuda_arena_create_on(ggml_backend_dev_t dev, size_t bytes, int device_ordinal);

// Optional helpers:
GGML_BACKEND_API size_t ggml_cuda_arena_alignment(ggml_backend_buffer_t arena);     // delegate to buft alignment
GGML_BACKEND_API bool   ggml_cuda_arena_place(ggml_backend_buffer_t arena,
                             struct ggml_tensor * t,
                             size_t offset); // calls ggml_backend_tensor_alloc



#ifdef  __cplusplus
}
#endif