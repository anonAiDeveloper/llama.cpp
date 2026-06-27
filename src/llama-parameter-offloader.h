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
bool llama_offloader_graph_cb(ggml_backend_sched_t sched, struct ggml_cgraph * graph, void * ud);

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

    struct offloader_schedule
    {
        // Scheduling
        std::vector<int>                      ready_after;           // per index: last safe copy index before arena overlap risk
        std::vector<ggml_tensor*>             cpu_tensors_in_order;  // feed-order list of CPU twins
        std::vector<ggml_tensor*>             gpu_tensors_in_order;  // feed-order list of GPU twins
        std::unordered_map<ggml_tensor*, int> gpu2index;             // GPU twin -> feed-order index

        std::vector<size_t> start; // arena start offset for each scheduled GPU tensor
        std::vector<size_t> end;   // arena end offset for each scheduled GPU tensor

        uint64_t generation = 0; // monotonic id for debugging schedule swaps
    };
    std::mutex schedule_mu; // protects schedule_current / schedule_next swaps and schedule reads
    offloader_schedule schedule_current;          // active schedule used by reader and streamer
    offloader_schedule schedule_next;             // candidate schedule built from latest graph callback
    std::atomic<uint64_t> schedule_generation{0}; // latest published schedule generation
    size_t schedule_next_prefix  = 0;       // common prefix length between active and candidate schedules
    bool schedule_next_identical = false;   // candidate schedule exactly matches active schedule
    bool schedule_next_valid     = false;   // candidate comparison stats are valid

    void retarget_schedule_tensors(offloader_schedule & schedule);

    bool swap_next_schedule(); // swaps after retargeting and gate build

    void build_schedule_gates(offloader_schedule & schedule); // compute copy barriers

    // Fast lookups
    // GPU->CPU: answer “what CPU weight backs this GPU twin?”
    std::unordered_map<ggml_tensor*, ggml_tensor*> gpu2cpu;
    // CPU->GPU: answer “do we already have a GPU twin for this CPU weight?”
    std::unordered_map<ggml_tensor*, ggml_tensor*> cpu2gpu;

    //map the gpu tensors to hashes recorded at init, to ensure data integrity
    std::unordered_map<ggml_tensor*, uint64_t> gpu_hashes;

    std::unordered_set<ggml_tensor*> cpu_weight_set; // CPU weight ptrs
    std::unordered_set<ggml_tensor*> gpu_weight_set; // GPU weight ptrs

    void init(ggml_backend_buffer_t arena,     llama_context_params params,
              ggml_context        * ctx_twins, llama_context      * lctx);
    parameter_offloader(llama_model * model);
    ~parameter_offloader();

    void copy_host_to_arena_with_transform(ggml_tensor * src_host, ggml_tensor * dst_arena);


    // runtime
    std::thread        copy_thread;
    std::atomic<bool>  stop_stream{false};

    bool node_reads_tracked_weight(ggml_tensor * t, int * out_idx);
    bool wants_observe(ggml_tensor * node);
    bool on_eval_tensor(ggml_tensor * node);

    // start/stop the streaming worker
    void start_streamer();
    void stop_streamer_join();

    void print_snapshot(offloader_schedule & schedule);
private:
    void seed_all_weights_from_model();

    ggml_tensor * init_cpu_tensor_to_arena(ggml_tensor * w_cpu);

    std::atomic<long long> tensor_idx_copied_ordinal{-1}; // last copied ordinal in current schedule stream
    std::atomic<long long> tensor_idx_used_ordinal{-1};   // last read ordinal in current schedule stream

    std::mutex              node_mu_;
    std::condition_variable node_cv_;

    struct PackedHostBytes {
        ggml_backend_buffer_t buf = nullptr;  // owns the RAM block
        void * base = nullptr;                // host pointer to the packed bytes
        size_t bytes = 0;                     // exact device-sized byte count
    };

    // Track which host weights have been permanently device-packed
    std::unordered_map<ggml_tensor*, PackedHostBytes> host_packed_;

    // We own these new host buffers; free them in ~parameter_offloader()
    std::vector<ggml_backend_buffer_t> owned_host_buffers_;

    // Permanently transform a host tensor to the device-native layout (stored on host).
    // Returns false if already transformed or if preconditions are not met.
    bool transform_cpu_tensor_to_device_layout(ggml_tensor * w_cpu);

    // (optional) helper: transform all collected host weights
    size_t transform_all_cpu_weights_to_device_layout();

    void stream_worker();
    
    inline bool no_transform_needed_for_backend_(const ggml_tensor *t) const;
public:
    inline void upload_weight_auto(ggml_tensor *w_cpu, ggml_tensor *w_gpu);
};