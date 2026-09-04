#include "llama-impl.h"

#include "llama-model.h"

#include "ggml-cuda.h"
#include "ggml-cuda-arena.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
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
    // Misc helper functions
    struct parameter_offloader_model_i {
        bool (*weight_supported)(const std::string & name);
        bool (*node_may_read_dense_weight)(const ggml_tensor * node);
    };

    size_t get_gpu_aligned_size(ggml_tensor * tensor, size_t alignment);

    // Fast lookups
    // GPU->CPU: answer "what CPU weight backs this GPU twin?"
    std::unordered_map<ggml_tensor*, ggml_tensor*> gpu2cpu;
    // CPU->GPU: answer "do we already have a GPU twin for this CPU weight?"
    std::unordered_map<ggml_tensor*, ggml_tensor*> cpu2gpu;

    // Must preserve original CPU tensors by name even after patch_model_refs_for()
    // changes model->tensors_by_name to point at GPU/placeholder twins.
    std::unordered_map<std::string, ggml_tensor *> cpu_weight_by_name;


    std::unordered_set<ggml_tensor*> cpu_weight_set; // CPU weight ptrs
    std::unordered_set<ggml_tensor*> gpu_weight_set; // GPU weight ptrs

    void copy_host_to_arena_with_transform(ggml_tensor * src_host, ggml_tensor * dst_arena);

    // Init and destructor functions
    bool ready = false;

    std::vector<ggml_tensor*> collected_order;      // CPU weights in first-use order

    // Arena + twin-context
    llama_model*                model;
    const parameter_offloader_model_i * model_i = nullptr; // selected once from model->arch
    ggml_backend_buffer_t       arena           = nullptr; // offloader CUDA arena buffer
    ggml_context*               ctx_gpu_twins   = nullptr; // no-alloc ctx for duplicated GPU tensors
    bool                        owns_arena      = false;

    // Cached placement info
    ggml_backend_buffer_type_t  arena_buffer_type  = nullptr;   // backend type for arena allocation sizing/layout
    char*                       arena_base  = nullptr;          // byte 0 of arena; offsets are relative to this
    size_t                      arena_size  = 0;                // full arena size
    size_t                      arena_dense_size   = 0;         // dense region size; static storage ends here
    size_t                      arena_stream_size  = 0;         // dense streaming region size
    size_t                      arena_alignment = 0;            // required byte alignment for arena placement

    void init(ggml_backend_buffer_t arena, llama_context_params params, ggml_context * ctx_twins);
    parameter_offloader(llama_model * model);
    ~parameter_offloader();

    // Streaming thread functions
    struct offloader_schedule
    {
        // Scheduling
        std::vector<int>                      ready_after;           // per index: last safe copy index before arena overlap risk
        std::vector<ggml_tensor*>             cpu_tensors_in_order;  // feed-order list of CPU twins
        std::vector<ggml_tensor*>             gpu_tensors_in_order;  // feed-order list of GPU twins
        std::unordered_map<ggml_tensor*, int> gpu2index;             // GPU twin -> feed-order index

        std::vector<size_t> start_offset; // arena start offset for each scheduled GPU tensor
        std::vector<size_t> end_offset;   // arena end offset for each scheduled GPU tensor
    };

    std::atomic<bool> schedule_swap_requested { false };
    std::mutex schedule_mutex;                    // protects schedule_current / schedule_next swaps and schedule reads
    offloader_schedule schedule_current;          // active schedule used by reader and streamer
    offloader_schedule schedule_next;             // candidate schedule built from latest graph callback
    std::atomic<uint64_t> schedule_generation{0}; // latest published schedule generation

    // runtime
    std::thread        copy_thread;
    std::atomic<bool>  stop_stream{false};

    void start();

    // Eval callback functions
    bool node_reads_tracked_weight(ggml_tensor * t, int * out_idx);
    bool wants_observe(ggml_tensor * node);
    bool on_eval_tensor(ggml_tensor * node);

    struct node_group {
        std::vector<ggml_tensor *> nodes;
        std::vector<ggml_tensor *> tensors;       // read order must be kept intact
        size_t bytes;
    };

    // Graph callback functions
    struct dense_graph_analysis
    {
        uint64_t hash = 0;                                              // Hash of the managed dense-read graph used for graph-cache lookup and comparison.
        std::vector<ggml_tensor *> gpu_tensors_in_order;                // Managed dense GPU tensors in first-read order for this graph.
        std::unordered_map<ggml_tensor *, int> gpu2index;               // Reverse lookup from managed GPU tensor to gpu_tensors_in_order index.
        std::vector<ggml_tensor *> graph_nodes;                         // Graph nodes that read one or more managed dense tensors, in graph order.
        std::vector<std::vector<ggml_tensor *>> graph_nodes_tensors;    // Managed dense tensors read by each corresponding entry in read_nodes. Read order must be kept intact
        std::vector<ggml_tensor *> release_node_by_tensor;              // For each streamed schedule index, the graph node whose completion releases that index and every preceding unreleased index.
        std::unordered_map<ggml_tensor *, int> next_required_tensor_idx;// Graph node -> first newly-read streamed tensor index that COPY must reach before compute advances.
        std::vector<node_group> node_pairs;                             // Adjacent pairs of nodes (duplicates are collapsed), filtering away static tensors
        bool dense_fits_arena = false;                                  // True when all managed dense tensors for this graph fit in the dense arena simultaneously.
    };

    struct dense_graph_cache_entry
    {
        std::vector<ggml_tensor *> static_dense_order;                  // Static dense tensor membership/order selected for this cached graph.
        offloader_schedule schedule;                                    // Final streamed tensor order, arena placement, and copy gates for this cached graph.
    };

    struct streaming_fit_lifetime_analysis
    {
        std::vector<std::vector<int>> managed_node_reads;               // Streamed schedule indices read at each managed streamed read position.
        std::vector<size_t> tensor_bytes;                               // Device allocation size of each streamed tensor, indexed by streamed schedule index.
        std::vector<std::vector<int>> resident_tensor_indices;          // Streamed tensors that must coexist at each read position, including prefetch and monotonic-release lifetimes.
    };

    dense_graph_analysis graph_analysis_current;  // active graph read/release metadata used by runtime
    dense_graph_analysis graph_analysis_next;     // analysis for the graph currently being prepared

    std::unordered_map<uint64_t, dense_graph_cache_entry> dense_graph_cache;

    size_t streaming_fit_lower_bound = 0;
    size_t streaming_fit_upper_bound = 0;
    size_t static_tensor_bytes = 0;

    std::vector<ggml_tensor *> static_dense_order;
    std::vector<ggml_tensor *> static_dense_order_current;
    std::unordered_set<ggml_tensor *> static_dense_set;
    std::unordered_set<ggml_tensor *> deprioritized_dense_set;

    // Analyze managed dense-weight reads for this graph and build the graph-specific read structure/hash.
    uint64_t analyze_dense_graph(ggml_backend_sched_t sched, const ggml_cgraph * graph, dense_graph_analysis & analysis);

    // Calculate the current streamed lower/upper arena-size bounds from graph reads and static membership.
    void streaming_fit_calculate_bounds(dense_graph_analysis & analysis);

    // Add or eject static dense tensors to fit the current upper-bound split while preserving the spacing heuristic.
    //TODO: Eventually we want to get this to fit as close to LOWER bound as we can, but that can get complicated.
    bool select_static_dense_tensors(const dense_graph_analysis & analysis);

    // Build the finalized streamed tensor schedule from the current graph analysis and static selection.
    void build_next_schedule(offloader_schedule & schedule, dense_graph_analysis & analysis);

    // Build current-graph runtime release and next-copy metadata for the finalized streamed schedule.
    void build_graph_runtime_metadata(dense_graph_analysis & analysis, const offloader_schedule & schedule);

    // Build fitter-only streamed lifetimes and resident sets used to derive physical coexistence constraints.
    void build_streaming_fit_lifetimes(
        const std::vector<ggml_tensor *> & gpu_tensors_in_order,
        const std::unordered_map<ggml_tensor *, int> & gpu2index,
        const dense_graph_analysis & analysis,
        streaming_fit_lifetime_analysis & fit_analysis) const;

    // Find and store a valid no-halt physical placement for every tensor in the finalized streamed schedule.
    size_t generate_streaming_fit(offloader_schedule & schedule, const dense_graph_analysis & analysis);

    // Apply the solved streamed offsets and pack/upload the current static dense tensors into the arena.
    void seat_dense_tensors(offloader_schedule & schedule);

    bool swap_next_schedule(size_t streaming_fit); // swaps after applying the solved schedule

    void build_schedule_gates(offloader_schedule & schedule); // compute copy barriers

    // MoE cache
    static constexpr int32_t MOE_CACHE_SLOT_COUNT = 16;     //temporary hardcoded slot count. In the future we will make this configurable, perhaps with per-model recommended defaults

    ggml_context*               ctx_moe_cache   = nullptr;
    int32_t                     moe_cache_n_slots = 0;

    void init_moe_cache(ggml_backend_buffer_t arena, int32_t n_slots);
    int32_t debug_cache_moe_expert(int block_id, int32_t expert_id);

    // Diagnostics
    //map the gpu tensors to hashes recorded at init, to ensure data integrity
    std::unordered_map<ggml_tensor*, uint64_t> gpu_hashes;

    void print_snapshot(offloader_schedule & schedule, ggml_log_level level = GGML_LOG_LEVEL_INFO);
    void print_tensor_order(const std::vector<ggml_tensor *> & tensors, const std::vector<size_t> & offsets, ggml_log_level level = GGML_LOG_LEVEL_INFO);
private:
    // Misc helper functions
    inline bool no_transform_needed_for_backend_(const ggml_tensor *t) const;

    inline ggml_cuda_copy_event * upload_weight_auto(ggml_tensor *w_cpu, ggml_tensor *w_gpu);

    // Init and destructor functions
    void seed_all_weights_from_model();

    // Index every model tensor pointer slot once so CPU->GPU patching can use direct lookup.
    void build_model_ref_lookup();

    // Patch every indexed model tensor pointer slot associated with one CPU tensor.
    void patch_model_refs_for(ggml_tensor * w_cpu, ggml_tensor * w_gpu);

    // CPU weight -> every llama_model/llama_layer/name-map pointer slot that must follow its GPU twin.
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor **>> model_ref_slots;

    ggml_tensor * init_cpu_tensor_to_arena(ggml_tensor * w_cpu, size_t & current_offset);

    // start/stop the streaming worker
    void start_streamer();
    void stop_streamer_join();

    void attach_arena(ggml_backend_buffer_t arena);

    struct PackedHostBytes {
        ggml_backend_buffer_t buf = nullptr;  // owns the RAM block
        void * base = nullptr;                // host pointer to the packed bytes
        size_t bytes = 0;                     // exact device-sized byte count
    };

    // Track which host weights have been permanently device-packed
    std::unordered_map<ggml_tensor*, PackedHostBytes> host_packed_;

    // Permanently transform a host tensor to the device-native layout (stored on host).
    // Returns false if already transformed or if preconditions are not met.
    bool transform_cpu_tensor_to_device_layout(ggml_tensor * w_cpu);

    // (optional) helper: transform all collected host weights
    size_t transform_all_cpu_weights_to_device_layout();

    // Streaming thread functions
    std::atomic<long long> tensor_idx_copied_ordinal{-1}; // last copied ordinal in current schedule stream
    std::atomic<long long> tensor_idx_used_ordinal{-1};   // last read ordinal in current schedule stream

    std::mutex              node_mu_;
    std::condition_variable node_cv_;

    void stream_worker();

    std::atomic<int> copy_publishers_in_flight{0};
    void publish_copy_when_ready(long long ordinal, uint64_t generation, ggml_cuda_copy_event * ev);
    void publish_copy_now(long long ordinal, uint64_t generation);

    // Eval callback functions

    // Graph callback functions

    // MoE cache
    void clear_moe_cache_refs();
    std::mutex moe_cache_mu;
    int32_t moe_cache_next_slot = 0;

    // Diagnostics
};