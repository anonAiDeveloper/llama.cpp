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
int32_t llama_offloader_moe_residency_cb(int block_id, int32_t expert_id, void * ud);

struct parameter_offloader
{
public:
    static constexpr int32_t MOE_CACHE_SLOT_COUNT = 16;     //temporary hardcoded slot count. In the future we will make this configurable, perhaps with per-model recommended defaults

    struct parameter_offloader_model_i {
        bool (*weight_supported)(const std::string & name);
        bool (*node_may_read_dense_weight)(const ggml_tensor * node);
    };

    bool ready = false;

    std::vector<ggml_tensor*> collected_order;      // CPU weights in first-use order
    std::unordered_set<ggml_tensor*> collect_seen;  // dedupe during collection

    // Arena + twin-context
    llama_model*                model;
    const parameter_offloader_model_i * model_i = nullptr; // selected once from model->arch
    ggml_backend_buffer_t       arena           = nullptr; // offloader CUDA arena buffer
    ggml_context*               ctx_gpu_twins   = nullptr; // no-alloc ctx for duplicated GPU tensors
    ggml_context*               ctx_moe_cache   = nullptr;
    bool                        owns_arena      = false;
    int32_t                     moe_cache_n_slots = 0;

    // Cached placement info
    ggml_backend_buffer_type_t  arena_buffer_type  = nullptr;   // backend type for arena allocation sizing/layout
    char*                       arena_base  = nullptr;          // byte 0 of arena; offsets are relative to this
    size_t                      arena_size  = 0;                // full arena size
    size_t                      arena_dense_size   = 0;         // dense region size; static storage ends here
    size_t                      arena_stream_size  = 0;         // dense streaming region size
    size_t                      arena_alignment = 0;            // required byte alignment for arena placement

    struct offloader_schedule
    {
        // Scheduling
        std::vector<int>                      ready_after;           // per index: last safe copy index before arena overlap risk
        std::vector<ggml_tensor*>             cpu_tensors_in_order;  // feed-order list of CPU twins
        std::vector<ggml_tensor*>             gpu_tensors_in_order;  // feed-order list of GPU twins
        std::unordered_map<ggml_tensor*, int> gpu2index;             // GPU twin -> feed-order index

        std::vector<size_t> start_offset; // arena start offset for each scheduled GPU tensor
        std::vector<size_t> end_offset;   // arena end offset for each scheduled GPU tensor
        std::vector<ggml_tensor*> read_last_node;             // per schedule index: node at which this schedule position may be released. Usually, not always, the last instance of it being read
        std::unordered_map<ggml_tensor*, int> read_next;      // node -> next schedule index when the immediately following read is a first read
    };
    std::atomic<bool> schedule_swap_requested { false };
    std::mutex schedule_mutex;                    // protects schedule_current / schedule_next swaps and schedule reads
    offloader_schedule schedule_current;          // active schedule used by reader and streamer
    offloader_schedule schedule_next;             // candidate schedule built from latest graph callback
    std::atomic<uint64_t> schedule_generation{0}; // latest published schedule generation

    // Cached no-halt fit and exact offsets for a streamed tensor set + managed read pattern.
    struct streaming_fit_cache_entry
    {
        std::vector<ggml_tensor*> gpu_tensors_in_order;
        std::vector<int> read_signature;
        size_t streaming_size = 0;
        std::vector<size_t> offsets;
    };
    std::unordered_map<uint64_t, std::vector<streaming_fit_cache_entry>> streaming_fit_cache;
    uint64_t streaming_fit_cache_current_hash = 0;
    int streaming_fit_cache_current = -1;

    size_t generate_streaming_fit(const offloader_schedule & schedule, const ggml_cgraph * graph);
    void retarget_schedule_tensors(offloader_schedule & schedule);

    bool swap_next_schedule(size_t streaming_fit); // swaps after retargeting and gate build

    void build_schedule_gates(offloader_schedule & schedule); // compute copy barriers

    // Fast lookups
    // GPU->CPU: answer "what CPU weight backs this GPU twin?"
    std::unordered_map<ggml_tensor*, ggml_tensor*> gpu2cpu;
    // CPU->GPU: answer "do we already have a GPU twin for this CPU weight?"
    std::unordered_map<ggml_tensor*, ggml_tensor*> cpu2gpu;

    // Must preserve original CPU tensors by name even after patch_model_refs_for()
    // changes model->tensors_by_name to point at GPU/placeholder twins.
    std::unordered_map<std::string, ggml_tensor *> cpu_weight_by_name;

    //map the gpu tensors to hashes recorded at init, to ensure data integrity
    std::unordered_map<ggml_tensor*, uint64_t> gpu_hashes;

    std::unordered_set<ggml_tensor*> cpu_weight_set; // CPU weight ptrs
    std::unordered_set<ggml_tensor*> gpu_weight_set; // GPU weight ptrs

    void init_moe_cache(ggml_backend_buffer_t arena, int32_t n_slots);
    int32_t debug_cache_moe_expert(int block_id, int32_t expert_id);

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

    ggml_tensor * init_cpu_tensor_to_arena(ggml_tensor * w_cpu, size_t current_offset);

    void attach_arena(ggml_backend_buffer_t arena);
    void clear_moe_cache_refs();
    std::mutex moe_cache_mu;
    int32_t moe_cache_next_slot = 0;

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

    std::atomic<int> copy_publishers_in_flight{0};
    void publish_copy_when_ready(long long ordinal, uint64_t generation, ggml_cuda_copy_event * ev);
    void publish_copy_now(long long ordinal, uint64_t generation);
public:
    inline ggml_cuda_copy_event * upload_weight_auto(ggml_tensor *w_cpu, ggml_tensor *w_gpu);       //TODO: Why is this public?
};