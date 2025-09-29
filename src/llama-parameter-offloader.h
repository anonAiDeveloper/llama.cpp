#include "llama-impl.h"

#include "llama-model.h"

#include "ggml-cuda.h"
#include "ggml-cuda-arena.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <unordered_set>

// signature must match ggml_backend_sched_eval_callback
bool llama_offloader_eval_cb(ggml_tensor * t, bool ask, void * ud);

struct parameter_offloader
{
public:
    bool ready = false;

    std::vector<ggml_tensor*> collected_order;      // CPU weights in first-use order
    std::unordered_set<ggml_tensor*> collect_seen;  // dedupe during collection

    // Arena + twin-context
    llama_model*                model;
    ggml_backend_buffer_t       arena         = nullptr;  // your CUDA arena buffer
    ggml_context*               ctx_gpu_twins = nullptr;  // no-alloc ctx for duplicated GPU tensors

    // Cached placement info
    ggml_backend_buffer_type_t  buft  = nullptr;          // ggml_backend_buffer_get_type(arena)
    char*                       base  = nullptr;          // ggml_backend_buffer_get_base(arena)
    size_t                      cap   = 0;               // ggml_backend_buffer_get_size(arena)
    size_t                      align = 0;               // ggml_backend_buffer_get_alignment(arena)
    size_t                      cur_off = 0;             // next free offset (bytes) inside arena
    size_t placed_bytes = 0;
    size_t placed       = 0;

    // Scheduling
    std::vector<int>                      ready_after;       // barrier per feed-order index
    std::vector<ggml_tensor*>             cpu_tensors_in_order;  // feed-order list of CPU twins
    std::vector<ggml_tensor*>             gpu_tensors_in_order;  // feed-order list of GPU twins
    std::unordered_map<ggml_tensor*, int> gpu2index;         // GPU twin -> feed-order index

    std::atomic<int> tensor_idx_loaded{ -1 };  // highest index copied into arena

    std::atomic<int> tensor_idx_used_mod{-1};         // last seen feed index this pass
    std::atomic<int> tensor_idx_used_epoch{0};        // increments when index decreases (wrap)
    std::atomic<long long> tensor_idx_used_seq{ -1 }; // tensor_idx_used_epoch*tensor_count + tensor_idx_used_mod


    // Fast lookups
    // GPU->CPU: answer “what CPU weight backs this GPU twin?”
    std::unordered_map<ggml_tensor*, ggml_tensor*> gpu2cpu;
    // CPU->GPU: answer “do we already have a GPU twin for this CPU weight?”
    std::unordered_map<ggml_tensor*, ggml_tensor*> cpu2gpu;

    //map the gpu tensors to hashes recorded at init, to ensure data integrity
    std::unordered_map<ggml_tensor*, uint64_t> gpu_hashes;

    std::unordered_set<ggml_tensor*> weight_set; // CPU weight ptrs

    // Runtime trackers (set by your streaming/callback code during inference)
    ggml_tensor* last_streamed_gpu   = nullptr;  // last tensor copied into VRAM (GPU twin)
    ggml_tensor* current_in_use_gpu  = nullptr;  // tensor currently being used on GPU

    void init(ggml_backend_buffer_t arena,     llama_context_params params,
              ggml_context        * ctx_twins, llama_context      * lctx);
    parameter_offloader(llama_model * model);
    ~parameter_offloader();

    void copy_host_to_arena_with_transform(ggml_tensor * src_host, ggml_tensor * dst_arena);

    ggml_tensor * cpu_tensor_to_arena(ggml_tensor * w_cpu);

    // runtime
    std::thread        copy_thread;
    std::atomic<bool>  stop_stream{false};

    bool node_reads_tracked_weight(ggml_tensor * t, int * out_idx);
    bool wants_observe(ggml_tensor * node);
    void on_eval_tensor(ggml_tensor * node);

    // start/stop the streaming worker
    void start_streamer();
    void stop_streamer_join();

    void print_snapshot();
private:
    std::mutex              node_mu_;
    std::condition_variable node_cv_;

    void stream_worker();
};