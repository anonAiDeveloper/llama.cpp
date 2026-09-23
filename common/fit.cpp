#include "fit.h"

#include "log.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Maximum number of redistributed GPU placement windows. 0 leaves stock placement unchanged.
#ifndef COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS
#define COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS 8
#endif

static_assert(COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS >= 0, "COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS must be non-negative");

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

using common_fit_graph_walk_callback = void (*)(ggml_cgraph * graph, uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, void * user_data);

// Internal helper implemented in llama-context.cpp. It rebuilds the same representative PP/TG graphs used by stock scheduler reservation without allocating compute buffers.
LLAMA_API bool llama_context_walk_reserve_graphs(llama_context * ctx, common_fit_graph_walk_callback callback, void * user_data);

struct common_fit_placement_tensor {
    std::string name;
    ggml_backend_buffer_type_t device_buft = nullptr;
    bool device_eligible = false;
    bool graph_seen = false;
};

struct common_fit_placement_region {
    bool device_eligible = false;
    std::vector<std::string> tensors;
};

struct common_fit_placement_graph {
    uint32_t n_tokens = 0;
    uint32_t n_seqs = 0;
    uint32_t n_outputs = 0;
    std::vector<common_fit_placement_region> regions;
    std::vector<std::vector<size_t>> weight_nodes;
};

struct common_fit_placement_probe_result {
    size_t n_devices = 0;
    size_t n_blocks = 0;
    std::vector<common_fit_placement_tensor> tensors;
    std::vector<common_fit_placement_graph> graphs;
};

// MoE expert prefetch does not choose CPU/GPU placement while the graph is built. Each MoE block is built with K + 1 lanes covering every possible CPU/GPU split of the K selected experts.
// The scheduler selects one lane at execution time after querying the residency callback.
//
// Building those lanes still requires both expert tensor sets to exist in the model: the ordinary CPU expert banks and the GPU cache banks. At runtime parameter_offloader::init_moe_cache()
// creates the real arena-backed cache tensors before llama_init_from_model(). Fit runs with no_alloc and, for automatic arena sizing, before that arena exists.
// Without this shim, the fit model would have no GPU cache tensors from which to build the GPU sides of the lanes.
//
// common_fit_moe_topology supplies only that missing tensor topology. It creates metadata-only cache tensors with the runtime shapes and sharing pattern, then attaches them to a zero-byte buffer
// of the target GPU BUFT and installs the model's *_cache pointers before context construction. The zero-byte buffer consumes no VRAM; it only preserves logical GPU buffer-type identity.
// The ordinary graph builder and scheduler then reserve the same K + 1 lane topology as runtime.
//
// This helper deliberately does NOT predict expert residency, select lanes, enumerate CPU/GPU combinations, or count cache storage as ordinary resident model memory.
// The graph already contains every lane, and the real cache storage belongs to the parameter-offloader arena.
struct common_fit_moe_cache_field {
    ggml_tensor * llama_layer::* cpu;
    ggml_tensor * llama_layer::* cache;
};

static const common_fit_moe_cache_field common_fit_moe_cache_fields[] = {
    { &llama_layer::ffn_gate_exps,      &llama_layer::ffn_gate_exps_cache },     { &llama_layer::ffn_down_exps,      &llama_layer::ffn_down_exps_cache },
    { &llama_layer::ffn_up_exps,        &llama_layer::ffn_up_exps_cache },       { &llama_layer::ffn_gate_up_exps,   &llama_layer::ffn_gate_up_exps_cache },
    { &llama_layer::ffn_gate_chexps,    &llama_layer::ffn_gate_chexps_cache },   { &llama_layer::ffn_down_chexps,    &llama_layer::ffn_down_chexps_cache },
    { &llama_layer::ffn_up_chexps,      &llama_layer::ffn_up_chexps_cache },     { &llama_layer::ffn_gate_exps_b,    &llama_layer::ffn_gate_exps_b_cache },
    { &llama_layer::ffn_down_exps_b,    &llama_layer::ffn_down_exps_b_cache },   { &llama_layer::ffn_up_exps_b,      &llama_layer::ffn_up_exps_b_cache },
    { &llama_layer::ffn_gate_up_exps_b, &llama_layer::ffn_gate_up_exps_b_cache },{ &llama_layer::ffn_gate_exps_s,    &llama_layer::ffn_gate_exps_s_cache },
    { &llama_layer::ffn_down_exps_s,    &llama_layer::ffn_down_exps_s_cache },   { &llama_layer::ffn_up_exps_s,      &llama_layer::ffn_up_exps_s_cache },
    { &llama_layer::ffn_gate_exps_in_s, &llama_layer::ffn_gate_exps_in_s_cache },{ &llama_layer::ffn_down_exps_in_s, &llama_layer::ffn_down_exps_in_s_cache },
    { &llama_layer::ffn_up_exps_in_s,   &llama_layer::ffn_up_exps_in_s_cache },
};

struct common_fit_moe_cache_bank {
    int field_id;
    ggml_type type;
    int n_dims;
    int64_t ne[GGML_MAX_DIMS];
    ggml_tensor * tensor;
};

class common_fit_moe_topology {
public:
    ~common_fit_moe_topology() {
        reset();
    }

    void init(llama_model * model, ggml_backend_buffer_type_t buft, int32_t n_slots) {
        reset();

        if (n_slots <= 0)
            return;

        model_ = model;

        const int n_layers = (int) model_->hparams.n_layer();
        const int n_fields = (int) (sizeof(common_fit_moe_cache_fields) / sizeof(common_fit_moe_cache_fields[0]));
        size_t n_source_banks = 0;

        for (int il = 0; il < n_layers; ++il) {
            llama_layer & layer = model_->layers[il];

            for (int field_id = 0; field_id < n_fields; ++field_id) {
                const common_fit_moe_cache_field & field = common_fit_moe_cache_fields[field_id];
                layer.*(field.cache) = nullptr;

                if (layer.*(field.cpu) != nullptr)
                    ++n_source_banks;
            }
        }

        if (n_source_banks == 0)
            return;

        GGML_ASSERT(n_source_banks <= SIZE_MAX / ggml_tensor_overhead());

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * n_source_banks,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };

        ctx_ = ggml_init(params);
        if (!ctx_)
            throw std::runtime_error("failed to create fit-only MoE cache metadata context");

        // Preserve the target GPU BUFT without allocating cache bytes; graph construction only needs tensor placement identity here.
        buffer_ = ggml_backend_buft_alloc_buffer(buft, 0);
        if (!buffer_)
            throw std::runtime_error("failed to create fit-only MoE cache buffer");

        std::vector<common_fit_moe_cache_bank> banks;
        banks.reserve(n_source_banks);

        for (int il = 0; il < n_layers; ++il) {
            llama_layer & layer = model_->layers[il];

            for (int field_id = 0; field_id < n_fields; ++field_id) {
                const common_fit_moe_cache_field & field = common_fit_moe_cache_fields[field_id];
                ggml_tensor * cpu = layer.*(field.cpu);

                if (!cpu)
                    continue;

                GGML_ASSERT(cpu->view_src == nullptr);
                GGML_ASSERT(!ggml_is_transposed(cpu));
                GGML_ASSERT(ggml_is_contiguous(cpu));

                const int n_dims = ggml_n_dims(cpu);
                GGML_ASSERT(n_dims >= 1);
                GGML_ASSERT(n_dims <= GGML_MAX_DIMS);

                const int expert_dim = n_dims - 1;
                int64_t cache_ne[GGML_MAX_DIMS];

                for (int d = 0; d < GGML_MAX_DIMS; ++d)
                    cache_ne[d] = 1;

                for (int d = 0; d < n_dims; ++d)
                    cache_ne[d] = cpu->ne[d];

                cache_ne[expert_dim] = n_slots;

                // Runtime shares compatible cache banks across layers. Mirror that pointer sharing so the fit graph sees the same tensor topology.
                size_t bank_id = banks.size();

                for (size_t i = 0; i < banks.size(); ++i) {
                    const common_fit_moe_cache_bank & bank = banks[i];

                    if (bank.field_id != field_id || bank.type != cpu->type || bank.n_dims != n_dims)
                        continue;

                    bool same_shape = true;

                    for (int d = 0; d < n_dims; ++d) {
                        if (bank.ne[d] != cache_ne[d]) {
                            same_shape = false;
                            break;
                        }
                    }

                    if (same_shape) {
                        bank_id = i;
                        break;
                    }
                }

                if (bank_id == banks.size()) {
                    ggml_tensor * cache = ggml_new_tensor(ctx_, cpu->type, n_dims, cache_ne);
                    GGML_ASSERT(cache);
                    cache->buffer = buffer_;

                    common_fit_moe_cache_bank bank = {};
                    bank.field_id = field_id;
                    bank.type = cpu->type;
                    bank.n_dims = n_dims;
                    bank.tensor = cache;

                    for (int d = 0; d < GGML_MAX_DIMS; ++d)
                        bank.ne[d] = cache_ne[d];

                    banks.push_back(bank);
                }

                layer.*(field.cache) = banks[bank_id].tensor;
            }
        }
    }

    void reset() {
        if (model_) {
            const int n_layers = (int) model_->hparams.n_layer();
            const int n_fields = (int) (sizeof(common_fit_moe_cache_fields) / sizeof(common_fit_moe_cache_fields[0]));

            for (int il = 0; il < n_layers; ++il) {
                llama_layer & layer = model_->layers[il];

                for (int field_id = 0; field_id < n_fields; ++field_id)
                    layer.*(common_fit_moe_cache_fields[field_id].cache) = nullptr;
            }
        }

        if (buffer_)
            ggml_backend_buffer_free(buffer_);

        if (ctx_)
            ggml_free(ctx_);

        model_ = nullptr;
        ctx_ = nullptr;
        buffer_ = nullptr;
    }

private:
    llama_model * model_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

static void common_fit_moe_topology_init(common_fit_moe_topology & topology, llama_model * model, const llama_context_params & cparams) {
    if (!cparams.moe_expert_prefetch)
        return;

    if (llama_model_n_devices(model) != 1)
        throw std::runtime_error("MoE expert prefetch fit currently supports exactly one accelerator device");

    ggml_backend_dev_t dev = llama_model_get_device(model, 0);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft)
        throw std::runtime_error("MoE expert prefetch fit device has no default buffer type");

    topology.init(model, buft, (int32_t) model->hparams.n_expert_used);
}

static ggml_tensor * common_fit_placement_root(ggml_tensor * tensor) {
    while (tensor && tensor->view_src)
        tensor = tensor->view_src;

    return tensor;
}

static ggml_backend_buffer_type_t common_fit_placement_device_buft(const llama_model * model, ggml_tensor * tensor) {
    tensor = common_fit_placement_root(tensor);
    if (!tensor || !tensor->buffer)
        return nullptr;

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    if (!buft || ggml_backend_buft_is_host(buft))
        return nullptr;

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev)
        return nullptr;

    for (int i = 0; i < llama_model_n_devices(model); ++i) {
        if (llama_model_get_device(model, i) == dev)
            return buft;
    }

    return nullptr;
}

struct common_fit_placement_capture {
    common_fit_placement_probe_result * result = nullptr;
    std::unordered_map<ggml_tensor *, size_t> tensor_index;
};

static void common_fit_placement_capture_graph(ggml_cgraph * graph, uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, void * user_data) {
    common_fit_placement_capture * capture = static_cast<common_fit_placement_capture *>(user_data);
    GGML_ASSERT(capture);
    GGML_ASSERT(capture->result);
    GGML_ASSERT(graph);

    common_fit_placement_graph graph_result;
    graph_result.n_tokens = n_tokens;
    graph_result.n_seqs = n_seqs;
    graph_result.n_outputs = n_outputs;

    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        std::vector<size_t> node_weights;
        bool node_device_eligible = true;

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * weight = common_fit_placement_root(node->src[j]);
            if (!weight)
                continue;

            auto it = capture->tensor_index.find(weight);
            if (it == capture->tensor_index.end())
                continue;

            if (std::find(node_weights.begin(), node_weights.end(), it->second) != node_weights.end())
                continue;

            node_weights.push_back(it->second);
            node_device_eligible = node_device_eligible && capture->result->tensors[it->second].device_eligible;
        }

        if (node_weights.empty())
            continue;

        graph_result.weight_nodes.push_back(node_weights);

        for (size_t tensor_index : node_weights)
            capture->result->tensors[tensor_index].graph_seen = true;

        if (graph_result.regions.empty() || graph_result.regions.back().device_eligible != node_device_eligible) {
            graph_result.regions.emplace_back();
            graph_result.regions.back().device_eligible = node_device_eligible;
        }

        common_fit_placement_region & region = graph_result.regions.back();
        for (size_t tensor_index : node_weights) {
            const common_fit_placement_tensor & tensor = capture->result->tensors[tensor_index];
            if (std::find(region.tensors.begin(), region.tensors.end(), tensor.name) == region.tensors.end())
                region.tensors.push_back(tensor.name);
        }
    }

    capture->result->graphs.push_back(std::move(graph_result));
}

static const char * common_fit_placement_policy_pattern(common_layer_fraction_t fraction) {
    switch (fraction) {
        case LAYER_FRACTION_NONE:
            return nullptr;
        case LAYER_FRACTION_ATTN:
            return "blk\\.\\d+\\.ffn_(gate|up|gate_up|down).*";
        case LAYER_FRACTION_UP:
            return "blk\\.\\d+\\.ffn_(gate|gate_up|down).*";
        case LAYER_FRACTION_GATE:
            return "blk\\.\\d+\\.ffn_down.*";
        case LAYER_FRACTION_MOE:
            return "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
    }

    GGML_ABORT("fatal error");
}

static common_layer_fraction_t common_fit_placement_policy_from_fit_overrides(const llama_model_params & mparams, size_t & n_policy_overrides) {
    n_policy_overrides = 0;
    if (!mparams.tensor_buft_overrides)
        return LAYER_FRACTION_NONE;

    common_layer_fraction_t fraction = LAYER_FRACTION_NONE;

    for (const llama_model_tensor_buft_override * override = mparams.tensor_buft_overrides; override->pattern; ++override) {
        if (!override->buft || !ggml_backend_buft_is_host(override->buft))
            continue;

        const std::string pattern = override->pattern;
        const size_t suffix_pos = pattern.find("\\.ffn_");
        if (suffix_pos == std::string::npos)
            continue;

        const std::string suffix = pattern.substr(suffix_pos);
        if (suffix == "\\.ffn_(up|down|gate_up|gate)_(ch|)exps") {
            ++n_policy_overrides;
            fraction = LAYER_FRACTION_MOE;
        } else if (suffix == "\\.ffn_(gate|up|gate_up|down).*") {
            ++n_policy_overrides;
            if (fraction == LAYER_FRACTION_NONE)
                fraction = LAYER_FRACTION_ATTN;
        } else if (suffix == "\\.ffn_(gate|gate_up|down).*") {
            ++n_policy_overrides;
            if (fraction == LAYER_FRACTION_NONE)
                fraction = LAYER_FRACTION_UP;
        } else if (suffix == "\\.ffn_down.*") {
            ++n_policy_overrides;
            if (fraction == LAYER_FRACTION_NONE)
                fraction = LAYER_FRACTION_GATE;
        }
    }

    return fraction;
}

static common_fit_placement_probe_result common_fit_probe_placement(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        common_layer_fraction_t policy_fraction,
        bool fit_generated_overrides) {
    common_fit_placement_probe_result result;

    llama_model_params probe_mparams = *mparams;
    probe_mparams.n_gpu_layers = -1;
    probe_mparams.no_alloc = true;
    probe_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    probe_mparams.defer_non_host_weights = false;

    llama_model_tensor_buft_override policy_overrides[2] = {};
    if (fit_generated_overrides) {
        // Stock fit's overflow overrides are positional. Remove only that layer location and
        // extrapolate the placement class it selected across the repeating layers.
        if (const char * pattern = common_fit_placement_policy_pattern(policy_fraction)) {
            policy_overrides[0] = { pattern, ggml_backend_cpu_buffer_type() };
            policy_overrides[1] = { nullptr, nullptr };
            probe_mparams.tensor_buft_overrides = policy_overrides;
        } else {
            probe_mparams.tensor_buft_overrides = nullptr;
        }
    }

    llama_model * model = llama_model_load_from_file(path_model, probe_mparams);
    if (!model)
        throw std::runtime_error("failed to load placement probe model");

    result.n_devices = llama_model_n_devices(model);
    result.n_blocks = llama_model_n_layer(model);

    llama_context * ctx = nullptr;
    common_fit_moe_topology moe_topology;

    try {
        common_fit_placement_capture capture;
        capture.result = &result;

        const auto & tensor_map = llama_internal_get_tensor_map(model);
        result.tensors.reserve(tensor_map.size());
        capture.tensor_index.reserve(tensor_map.size());

        for (const auto & [name, tensor_raw] : tensor_map) {
            ggml_tensor * tensor = common_fit_placement_root(tensor_raw);
            if (!tensor || capture.tensor_index.find(tensor) != capture.tensor_index.end())
                continue;

            const size_t tensor_index = result.tensors.size();
            ggml_backend_buffer_type_t device_buft = common_fit_placement_device_buft(model, tensor);
            capture.tensor_index.emplace(tensor, tensor_index);
            result.tensors.push_back({ name, device_buft, device_buft != nullptr, false });
        }

        llama_context_params probe_cparams = *cparams;
        probe_cparams.cb_eval = nullptr;
        probe_cparams.cb_eval_user_data = nullptr;
        probe_cparams.cb_graph = nullptr;
        probe_cparams.cb_graph_user_data = nullptr;
        probe_cparams.cb_moe_residency = nullptr;
        probe_cparams.cb_moe_residency_user_data = nullptr;

        common_fit_moe_topology_init(moe_topology, model, probe_cparams);

        ctx = llama_init_from_model(model, probe_cparams);
        if (!ctx)
            throw std::runtime_error("failed to create placement probe context");

        if (!llama_context_walk_reserve_graphs(ctx, common_fit_placement_capture_graph, &capture))
            throw std::runtime_error("failed to build placement probe graphs");

        llama_free(ctx);
        ctx = nullptr;
        moe_topology.reset();
        llama_model_free(model);
        model = nullptr;
    } catch (...) {
        if (ctx)
            llama_free(ctx);
        moe_topology.reset();
        if (model)
            llama_model_free(model);
        throw;
    }

    return result;
}

static void common_fit_placement_probe_print(const common_fit_placement_probe_result & result) {
    size_t n_device_eligible = 0;
    size_t n_seen = 0;

    for (const common_fit_placement_tensor & tensor : result.tensors) {
        n_device_eligible += tensor.device_eligible ? 1 : 0;
        n_seen += tensor.graph_seen ? 1 : 0;
        LOG_TRC("common_fit: placement tensor: %-64s device_eligible=%d graph_seen=%d\n", tensor.name.c_str(), tensor.device_eligible ? 1 : 0, tensor.graph_seen ? 1 : 0);
    }

    LOG_INF("common_fit: placement probe: %zu graphs, %zu/%zu tensors seen in graphs, %zu device eligible, %zu CPU required\n",
        result.graphs.size(), n_seen, result.tensors.size(), n_device_eligible, result.tensors.size() - n_device_eligible);

    for (size_t graph_id = 0; graph_id < result.graphs.size(); ++graph_id) {
        const common_fit_placement_graph & graph = result.graphs[graph_id];
        LOG_TRC("common_fit: placement graph %zu: n_tokens=%u n_seqs=%u n_outputs=%u regions=%zu\n",
            graph_id, graph.n_tokens, graph.n_seqs, graph.n_outputs, graph.regions.size());

        for (size_t region_id = 0; region_id < graph.regions.size(); ++region_id) {
            const common_fit_placement_region & region = graph.regions[region_id];
            LOG_TRC("common_fit:   region %zu: %s, %zu tensors\n", region_id, region.device_eligible ? "DEVICE_ELIGIBLE" : "CPU_REQUIRED", region.tensors.size());

            for (const std::string & name : region.tensors)
                LOG_TRC("common_fit:     %s\n", name.c_str());
        }
    }

    if (n_seen != result.tensors.size()) {
        // TODO: Add another stock-representative graph shape if a model needs one for placement coverage.
        for (const common_fit_placement_tensor & tensor : result.tensors) {
            if (!tensor.graph_seen)
                LOG_TRC("common_fit: placement tensor not present in PP/TG reserve graphs: %s\n", tensor.name.c_str());
        }
    }
}

static int common_fit_placement_block_id(const std::string & name) {
    if (name.size() < 6 || name.compare(0, 4, "blk.") != 0)
        return -1;

    size_t pos = 4;
    int block_id = 0;
    bool have_digit = false;

    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
        have_digit = true;
        block_id = block_id * 10 + (name[pos] - '0');
        ++pos;
    }

    return have_digit && pos < name.size() && name[pos] == '.' ? block_id : -1;
}

static std::vector<std::vector<size_t>> common_fit_placement_border_groups(const common_fit_placement_probe_result & result) {
    std::vector<std::vector<size_t>> border_groups(result.n_blocks);
    if (result.graphs.empty() || result.n_blocks < 2)
        return border_groups;

    // Each internal block border is one placement unit. For a CPU-required section inside block i,
    // eligible reads before it belong to border i and eligible reads after it belong to border i + 1.
    // A tensor is usable only when every representative graph that reads it assigns it to the same border.
    std::vector<int> tensor_border(result.tensors.size(), -2); // -2 = not seen yet, -1 = not a safe internal-border tensor

    for (const common_fit_placement_graph & graph : result.graphs) {
        std::vector<int> first_read(result.tensors.size(), -1);
        std::vector<int> last_read(result.tensors.size(), -1);
        std::vector<int> cpu_first(result.n_blocks, -1);
        std::vector<int> cpu_last(result.n_blocks, -1);

        for (size_t node_id = 0; node_id < graph.weight_nodes.size(); ++node_id) {
            for (size_t tensor_id : graph.weight_nodes[node_id]) {
                if (first_read[tensor_id] < 0)
                    first_read[tensor_id] = (int) node_id;
                last_read[tensor_id] = (int) node_id;

                const int block_id = common_fit_placement_block_id(result.tensors[tensor_id].name);
                if (block_id < 0 || block_id >= (int) result.n_blocks || result.tensors[tensor_id].device_eligible)
                    continue;

                if (cpu_first[block_id] < 0)
                    cpu_first[block_id] = (int) node_id;
                cpu_last[block_id] = (int) node_id;
            }
        }

        for (size_t tensor_id = 0; tensor_id < result.tensors.size(); ++tensor_id) {
            if (first_read[tensor_id] < 0)
                continue;

            int graph_border = -1;
            const common_fit_placement_tensor & tensor = result.tensors[tensor_id];
            const int block_id = common_fit_placement_block_id(tensor.name);

            if (tensor.device_eligible && block_id >= 0 && block_id < (int) result.n_blocks && cpu_first[block_id] >= 0) {
                if (last_read[tensor_id] < cpu_first[block_id] && block_id > 0) {
                    graph_border = block_id;
                } else if (first_read[tensor_id] > cpu_last[block_id] && block_id + 1 < (int) result.n_blocks) {
                    graph_border = block_id + 1;
                }
            }

            if (tensor_border[tensor_id] == -2) {
                tensor_border[tensor_id] = graph_border;
            } else if (tensor_border[tensor_id] != graph_border) {
                tensor_border[tensor_id] = -1;
            }
        }
    }

    for (size_t tensor_id = 0; tensor_id < result.tensors.size(); ++tensor_id) {
        const int border = tensor_border[tensor_id];
        if (border > 0 && border < (int) result.n_blocks)
            border_groups[border].push_back(tensor_id);
    }

    return border_groups;
}

static std::vector<bool> common_fit_placement_probe_selected(
        const char * path_model,
        const llama_model_params & mparams,
        const common_fit_placement_probe_result & probe) {
    llama_model_params probe_mparams = mparams;
    probe_mparams.no_alloc = true;
    probe_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    probe_mparams.defer_non_host_weights = false;

    llama_model * model = llama_model_load_from_file(path_model, probe_mparams);
    if (!model)
        throw std::runtime_error("failed to load fitted placement probe model");

    std::unordered_map<std::string, size_t> tensor_index;
    tensor_index.reserve(probe.tensors.size());
    for (size_t i = 0; i < probe.tensors.size(); ++i)
        tensor_index.emplace(probe.tensors[i].name, i);

    std::vector<bool> selected(probe.tensors.size(), false);
    std::vector<bool> found(probe.tensors.size(), false);

    const auto & tensor_map = llama_internal_get_tensor_map(model);
    for (const auto & [name, tensor_raw] : tensor_map) {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end())
            continue;

        ggml_tensor * tensor = common_fit_placement_root(tensor_raw);
        selected[it->second] = common_fit_placement_device_buft(model, tensor) != nullptr;
        found[it->second] = true;
    }

    llama_model_free(model);

    for (size_t i = 0; i < found.size(); ++i) {
        if (!found[i])
            throw std::runtime_error("fitted placement probe did not recreate tensor '" + probe.tensors[i].name + "'");
    }

    return selected;
}

static std::string common_fit_placement_regex_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size() * 2);

    for (char c : value) {
        switch (c) {
            case '\\': case '.': case '^': case '$': case '|': case '(': case ')':
            case '[': case ']': case '*': case '+': case '?': case '{': case '}':
                result.push_back('\\');
                break;
            default:
                break;
        }
        result.push_back(c);
    }

    return result;
}

// Placement fitting is not thread safe. Keep generated pattern storage alive until the subsequent model load consumes the fitted params.
static std::vector<std::string> common_fit_placement_override_patterns;

static void common_fit_placement_set_overrides(
        const common_fit_placement_probe_result & probe,
        const std::vector<bool> & stock_selected,
        const std::vector<bool> & desired_selected,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t ntbo,
        llama_model_params & mparams) {
    struct override_bucket {
        ggml_backend_buffer_type_t buft = nullptr;
        std::vector<std::string> names;
    };

    std::vector<llama_model_tensor_buft_override> stock_overrides;
    if (mparams.tensor_buft_overrides) {
        for (const llama_model_tensor_buft_override * override = mparams.tensor_buft_overrides; override->pattern; ++override)
            stock_overrides.push_back(*override);
    }

    std::vector<override_bucket> buckets;
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

    for (size_t i = 0; i < probe.tensors.size(); ++i) {
        const common_fit_placement_tensor & tensor = probe.tensors[i];
        if (stock_selected[i] == desired_selected[i])
            continue;

        ggml_backend_buffer_type_t buft = desired_selected[i] ? tensor.device_buft : cpu_buft;
        GGML_ASSERT(buft);

        auto it = std::find_if(buckets.begin(), buckets.end(), [&](const override_bucket & bucket) { return bucket.buft == buft; });
        if (it == buckets.end()) {
            buckets.push_back({});
            buckets.back().buft = buft;
            it = buckets.end() - 1;
        }
        it->names.push_back(tensor.name);
    }

    struct generated_override {
        std::string pattern;
        ggml_backend_buffer_type_t buft = nullptr;
    };

    std::vector<generated_override> generated;
    constexpr size_t max_pattern_length = 4096;

    for (const override_bucket & bucket : buckets) {
        std::string pattern = "^(?:";
        bool has_name = false;

        for (const std::string & name : bucket.names) {
            const std::string escaped = common_fit_placement_regex_escape(name);
            const size_t added = escaped.size() + (has_name ? 1 : 0) + 2;

            if (has_name && pattern.size() + added > max_pattern_length) {
                pattern += ")$";
                generated.push_back({ std::move(pattern), bucket.buft });
                pattern = "^(?:";
                has_name = false;
            }

            if (has_name)
                pattern += '|';
            pattern += escaped;
            has_name = true;
        }

        if (has_name) {
            pattern += ")$";
            generated.push_back({ std::move(pattern), bucket.buft });
        }
    }

    if (generated.size() + stock_overrides.size() + 1 > ntbo)
        throw common_params_fit_exception("llama_max_tensor_buft_overrides() == " + std::to_string(ntbo) + " is insufficient for distributed placement");

    common_fit_placement_override_patterns.clear();
    common_fit_placement_override_patterns.reserve(generated.size());
    for (generated_override & entry : generated)
        common_fit_placement_override_patterns.push_back(std::move(entry.pattern));

    size_t itbo = 0;
    for (size_t i = 0; i < generated.size(); ++i) {
        tensor_buft_overrides[itbo].pattern = common_fit_placement_override_patterns[i].c_str();
        tensor_buft_overrides[itbo].buft = generated[i].buft;
        ++itbo;
    }
    for (const llama_model_tensor_buft_override & override : stock_overrides)
        tensor_buft_overrides[itbo++] = override;
    tensor_buft_overrides[itbo] = { nullptr, nullptr };
    mparams.tensor_buft_overrides = tensor_buft_overrides;
}

static void common_fit_placement_distribute(
        const char * path_model,
        llama_model_params & mparams,
        const llama_context_params * cparams,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t ntbo,
        bool fit_generated_overrides) {
    if (!mparams.defer_non_host_weights)
        return;

    common_layer_fraction_t policy_fraction = LAYER_FRACTION_NONE;
    size_t n_policy_overrides = 0;
    if (fit_generated_overrides) {
        policy_fraction = common_fit_placement_policy_from_fit_overrides(mparams, n_policy_overrides);

        if (policy_fraction != LAYER_FRACTION_NONE && mparams.n_gpu_layers > 0 &&
            size_t(mparams.n_gpu_layers) > n_policy_overrides + 1) {
            // TODO: Represent mixed full-repeating + partial-repeating fits without collapsing the two placement classes.
            LOG_TRC("common_fit: distributed placement left unchanged for mixed full/partial fitted placement\n");
            return;
        }
    }

    // Placement steps 1-2: remove only the NGL location. User placement overrides remain
    // intact; stock-fit overflow overrides are generalized from their fitted layer location.
    const common_fit_placement_probe_result probe = common_fit_probe_placement(
        path_model, &mparams, cparams, policy_fraction, fit_generated_overrides);

    if (probe.n_devices != 1) {
        // TODO: Preserve per-device placement when parameter offloading is extended to multi-device streaming.
        LOG_TRC("common_fit: distributed placement is currently left unchanged for %zu devices\n", probe.n_devices);
        return;
    }

    if (fit_generated_overrides && policy_fraction != LAYER_FRACTION_NONE)
        LOG_INF("common_fit: placement probe generalized stock fit overflow policy %d across repeating layers\n", int(policy_fraction));

    common_fit_placement_probe_print(probe);

    const std::vector<bool> stock_selected = common_fit_placement_probe_selected(path_model, mparams, probe);
    std::vector<bool> desired_selected = stock_selected;

    size_t target_blocks = 0;
    if (mparams.n_gpu_layers < 0) {
        target_blocks = probe.n_blocks;
    } else if (mparams.n_gpu_layers > 1) {
        target_blocks = std::min(probe.n_blocks, size_t(mparams.n_gpu_layers - 1));
    }

    if (target_blocks == 0 || probe.n_blocks == 0) {
        LOG_INF("common_fit: distributed placement has no repeating blocks to redistribute\n");
        return;
    }

    if (COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS == 0) {
        LOG_INF("common_fit: distributed placement disabled by COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS=0; leaving stock placement unchanged\n");
        return;
    }

    // Only repeating-block placement is redistributed. Input/output tensors and graph-unseen tensors retain stock placement.
    for (size_t tensor_id = 0; tensor_id < probe.tensors.size(); ++tensor_id) {
        const common_fit_placement_tensor & tensor = probe.tensors[tensor_id];
        if (tensor.graph_seen && common_fit_placement_block_id(tensor.name) >= 0)
            desired_selected[tensor_id] = false;
    }

    const std::vector<std::vector<size_t>> border_groups = common_fit_placement_border_groups(probe);

    std::vector<size_t> available_borders;
    for (size_t border = 1; border < border_groups.size(); ++border) {
        if (!border_groups[border].empty()) {
            available_borders.push_back(border);
            LOG_TRC("common_fit: placement border %zu: %zu eligible tensors\n", border, border_groups[border].size());
        }
    }

    LOG_INF("common_fit: placement overlay: %zu/%zu internal block borders have safe device-eligible groups\n",
        available_borders.size(), probe.n_blocks > 0 ? probe.n_blocks - 1 : 0);

    if (target_blocks >= probe.n_blocks) {
        for (size_t tensor_id = 0; tensor_id < probe.tensors.size(); ++tensor_id) {
            if (probe.tensors[tensor_id].graph_seen && probe.tensors[tensor_id].device_eligible && common_fit_placement_block_id(probe.tensors[tensor_id].name) >= 0)
                desired_selected[tensor_id] = true;
        }

        LOG_INF("common_fit: distributed placement selected all %zu repeating blocks\n", probe.n_blocks);
        common_fit_placement_set_overrides(probe, stock_selected, desired_selected, tensor_buft_overrides, ntbo, mparams);
        return;
    }

    if (available_borders.empty()) {
        LOG_INF("common_fit: distributed placement found no safe device-eligible block-border groups; leaving stock placement unchanged\n");
        return;
    }

    const size_t n_block_units = std::min(target_blocks, available_borders.size());
    constexpr size_t max_gpu_windows = COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS;
    const size_t n_windows = std::min(max_gpu_windows, n_block_units);

    // Spread at most COMMON_FIT_PARAM_OFFLOAD_GPU_GROUPS placement windows over execution order. Window lengths and gaps are measured only
    // in block-border units; tensor sizes are deliberately irrelevant to NGL accounting.
    std::vector<bool> selected_border(probe.n_blocks, false);
    const size_t skipped_units = available_borders.size() - n_block_units;
    size_t cursor = skipped_units / (n_windows + 1) + (0 < skipped_units % (n_windows + 1) ? 1 : 0);

    for (size_t window = 0; window < n_windows; ++window) {
        const size_t window_length = n_block_units / n_windows + (window < n_block_units % n_windows ? 1 : 0);
        const size_t first_rank = cursor;

        for (size_t i = 0; i < window_length; ++i)
            selected_border[available_borders[cursor++]] = true;

        const size_t last_rank = cursor - 1;
        LOG_TRC("common_fit: placement window %zu: block borders %zu..%zu (%zu block units)\n",
            window, available_borders[first_rank], available_borders[last_rank], window_length);

        const size_t gap_id = window + 1;
        const size_t gap = skipped_units / (n_windows + 1) + (gap_id < skipped_units % (n_windows + 1) ? 1 : 0);
        cursor += gap;
    }

    size_t n_selected_tensors = 0;
    for (size_t border = 1; border < selected_border.size(); ++border) {
        if (!selected_border[border])
            continue;

        for (size_t tensor_id : border_groups[border]) {
            if (!desired_selected[tensor_id]) {
                desired_selected[tensor_id] = true;
                ++n_selected_tensors;
            }
        }
    }

    LOG_INF("common_fit: distributed placement selected %zu/%zu repeating-block units across %zu GPU windows (%zu tensors from %zu/%zu safe block-border groups)\n",
        n_block_units, target_blocks, n_windows, n_selected_tensors, n_block_units, available_borders.size());

    common_fit_placement_set_overrides(probe, stock_selected, desired_selected, tensor_buft_overrides, ntbo, mparams);
}

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;
    mparams_copy.defer_non_host_weights = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    common_fit_moe_topology moe_topology;

    try {
        common_fit_moe_topology_init(moe_topology, model, *cparams);
    } catch (...) {
        moe_topology.reset();
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw;
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        moe_topology.reset();
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                if (!mparams->defer_non_host_weights)
                    ret[i].mb.model += mb.model;

                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model) + llama_model_n_layer_nextn(model);
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    moe_topology.reset();
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (hp_nct - n_ctx_min) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (hp_nct - n_ctx_min);
                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

common_params_fit_status common_fit_distribute_param_offload(
        const char * path_model,
        llama_model_params * mparams,
        const llama_context_params * cparams,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        bool fit_generated_overrides) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;

    LOG_INF("%s: analyzing parameter-offload placement\n", __func__);

    try {
        common_fit_placement_distribute(
            path_model, *mparams, cparams, tensor_buft_overrides,
            llama_max_tensor_buft_overrides(), fit_generated_overrides);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to distribute parameter-offload placement: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while distributing parameter-offload placement: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }

    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: parameter-offload placement analysis took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}

