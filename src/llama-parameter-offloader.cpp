#include "llama-parameter-offloader.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdint>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <math.h>       /* isfinite */
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <stdexcept>

// TODO: Investigate Page-Locked Host Memory for host_packed_ upload buffers.
//Explanation: CPU tensors are not currently pinned. Pinning them speeds up cpu->gpu copies but also requires some VRAM overheard

/////////////////////////////////////
//   DEBUGGING SWITCHES
/////////////////////////////////////
//Uncomment LLAMA_DIAGNOSE_COPY to run copy/read synchronously
//Set this number to the max number of copies ahead you want to allow
//#define LLAMA_DIAGNOSE_COPY 9999
//#define LLAMA_DIAGNOSE_COPY 1

//#define LLAMA_CHECK_WEIGHTS 1
//#define LLAMA_CHECK_WEIGHTS_VERBOSE 1
//#define LLAMA_CHECK_NON_FINITES 1
//#define LLAMA_CHECK_NODES 1
//#define LLAMA_PRINT_ALL_NODES 1
//#define LLAMA_LOG_READS 2
//#define LLAMA_LOG_COPIES 2

//#define LLAMA_PRINT_WEIGHT_READS
/////////////////////////////////////

#ifndef GGML_USE_CUDA
#warning "WARNING! GGML_USE_CUDA is not defined!"
#endif
//#include "llama-offloader-diagnostic.h"

/////////////////////////////////////
//   HELPER FUNCTIONS
/////////////////////////////////////
static inline size_t align_up(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

static inline size_t align_down(size_t x, size_t a)
{
    return x & ~(a - 1);
}

inline size_t parameter_offloader::get_gpu_aligned_size(ggml_tensor * tensor, size_t alignment)
{
    return align_up(ggml_backend_buft_get_alloc_size(arena_buffer_type, gpu2cpu.at(tensor)), alignment);
}

static inline bool ranges_overlap(size_t a0, size_t a1, size_t b0, size_t b1)
{
    return !(a1 <= b0 || b1 <= a0);
}

static inline void update_next_size_event(size_t arena_size, size_t required_size, size_t alignment, size_t & next_size_event)
{
    required_size = align_up(required_size, alignment);

    if (required_size > arena_size)
        next_size_event = std::min(next_size_event, required_size);
}

static inline int ordinal_mod(long long ordinal, int n)
{
    return (int)(ordinal % (long long)n);
}

static inline int ring_distance(int from, int to, int n)
{
    return (to - from + n) % n;
}

static inline long long advance_ordinal_to_idx(long long ordinal, int idx, int tensor_count)
{
    if (ordinal < 0)
        return idx;

    const int cur = ordinal_mod(ordinal, tensor_count);
    const int d = (idx - cur + tensor_count) % tensor_count;

    return ordinal + d;
}

/////////////////////////////////////
//   MODEL SPECIFIC
/////////////////////////////////////
static inline bool offloader_name_ends(const std::string & name, const char * suffix)
{
    const size_t n = std::strlen(suffix);
    return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}

//Deepseek 2 weights
static bool parameter_offloader_deepseek2_weight_supported(const std::string & name)
{
    return
        offloader_name_ends(name, ".attn_norm.weight")      || // RMSNorm scale before attention block; 1D vector applied to residual stream before Q/K/V work
        offloader_name_ends(name, ".attn_q_a.weight")       || // First low-rank Q projection: hidden -> q_lora_rank before q_a_norm/q_b
        offloader_name_ends(name, ".attn_q_a_norm.weight")  || // RMSNorm scale on low-rank Q activation between q_a and q_b; 1D vector
        offloader_name_ends(name, ".attn_q_b.weight")       || // Second low-rank Q projection: q_lora_rank -> full per-head Q
        offloader_name_ends(name, ".attn_k_b.weight")       || // MLA absorbed K projection used after KV compression in MLA path
        offloader_name_ends(name, ".attn_kv_a_mqa.weight")  || // Shared KV compression projection: hidden -> kv_lora_rank + rope K part
        offloader_name_ends(name, ".attn_kv_a_norm.weight") || // RMSNorm scale on compressed KV activation before K/V expansion; 1D vector
        offloader_name_ends(name, ".attn_v_b.weight")       || // MLA absorbed V projection used by attention output path
        offloader_name_ends(name, ".attn_kv_b.weight")      || // Legacy unsplit KV expansion tensor for older/non-MLA GGUFs; replaces separate k_b/v_b
        offloader_name_ends(name, ".attn_output.weight")    || // Attention output projection back to model hidden size
        offloader_name_ends(name, ".ffn_norm.weight")       || // RMSNorm scale before FFN/MoE block; 1D vector
        offloader_name_ends(name, ".ffn_gate.weight")       || // Dense-layer FFN gate projection for leading non-MoE layers
        offloader_name_ends(name, ".ffn_up.weight")         || // Dense-layer FFN up projection for leading non-MoE layers
        offloader_name_ends(name, ".ffn_down.weight")       || // Dense-layer FFN down projection for leading non-MoE layers
        offloader_name_ends(name, ".ffn_gate_inp.weight")   || // MoE router/gating projection: hidden -> expert scores
        offloader_name_ends(name, ".exp_probs_b.bias")      || // Optional MoE expert-score/probability bias; 1D vector over experts
        //offloader_name_ends(name, ".ffn_down_exps.weight")  || // SPARSE: Routed MoE expert-bank down matrices; packed per expert
        //offloader_name_ends(name, ".ffn_gate_exps.weight")  || // SPARSE: Routed MoE expert-bank gate matrices; packed per expert
        //offloader_name_ends(name, ".ffn_up_exps.weight")    || // SPARSE: Routed MoE expert-bank up matrices; packed per expert
        offloader_name_ends(name, ".ffn_gate_shexp.weight") || // Shared expert FFN gate projection; always used, not routed by top-k
        offloader_name_ends(name, ".ffn_up_shexp.weight")   || // Shared expert FFN up projection; always used, not routed by top-k
        offloader_name_ends(name, ".ffn_down_shexp.weight") || // Shared expert FFN down projection; always used, not routed by top-k
        name == "token_embd.weight"                         ||
        name == "output_norm.weight"                        ||   // Final RMSNorm scale before logits
        name == "output.weight";                                 // LM head / output projection from hidden state to vocabulary logits
}

//GPT-OSS weights. These all fit within 4GB, and yes thats the 120B parameter model. This leads to odd behavior where copies no longer need to wait on reads.
static bool parameter_offloader_gpt_oss_weight_supported(const std::string & name)
{
    return
        name == "token_embd.weight"                              ||   // Input embedding table
        name == "output_norm.weight"                             ||   // Final RMSNorm scale
        offloader_name_ends(name, ".attn_norm.weight")          ||   // RMSNorm scale before attention
        offloader_name_ends(name, ".post_attention_norm.weight")||   // RMSNorm scale before MoE
        offloader_name_ends(name, ".attn_qkv.weight")           ||   // Optional fused Q/K/V projection
        offloader_name_ends(name, ".attn_qkv.bias")             ||   // Optional fused Q/K/V bias
        offloader_name_ends(name, ".attn_q.weight")             ||   // Separate Q projection when fused QKV is absent
        offloader_name_ends(name, ".attn_k.weight")             ||   // Separate K projection when fused QKV is absent
        offloader_name_ends(name, ".attn_v.weight")             ||   // Separate V projection when fused QKV is absent
        offloader_name_ends(name, ".attn_q.bias")               ||   // Optional Q bias
        offloader_name_ends(name, ".attn_k.bias")               ||   // Optional K bias
        offloader_name_ends(name, ".attn_v.bias")               ||   // Optional V bias
        offloader_name_ends(name, ".attn_output.weight")        ||   // Attention output projection
        offloader_name_ends(name, ".attn_output.bias")          ||   // Attention output bias
        offloader_name_ends(name, ".attn_sinks.weight")         ||   // Attention sinks
        offloader_name_ends(name, ".ffn_gate_inp.weight")       ||   // MoE router/gating projection
        offloader_name_ends(name, ".ffn_gate_inp.bias")         ||   // MoE router/gating bias
        //offloader_name_ends(name, ".ffn_gate_exps.weight")     || // SPARSE: routed expert gate matrices
        //offloader_name_ends(name, ".ffn_down_exps.weight")     || // SPARSE: routed expert down matrices
        //offloader_name_ends(name, ".ffn_up_exps.weight")       || // SPARSE: routed expert up matrices
        //offloader_name_ends(name, ".ffn_gate_exps.bias")       || // SPARSE: routed expert gate biases
        //offloader_name_ends(name, ".ffn_down_exps.bias")       || // SPARSE: routed expert down biases
        //offloader_name_ends(name, ".ffn_up_exps.bias")         || // SPARSE: routed expert up biases
        //offloader_name_ends(name, ".ffn_gate_exps.scale")      || // SPARSE: routed expert gate scales
        //offloader_name_ends(name, ".ffn_down_exps.scale")      || // SPARSE: routed expert down scales
        //offloader_name_ends(name, ".ffn_up_exps.scale")        || // SPARSE: routed expert up scales
        //offloader_name_ends(name, ".ffn_gate_exps.input_scale")|| // SPARSE: routed expert gate input scales
        //offloader_name_ends(name, ".ffn_down_exps.input_scale")|| // SPARSE: routed expert down input scales
        //offloader_name_ends(name, ".ffn_up_exps.input_scale")  || // SPARSE: routed expert up input scales
        name == "output.weight";                                      // LM head / output projection
}

// DeepSeek V4 weights
static bool parameter_offloader_deepseek4_weight_supported(const std::string & name)
{
    return
        name == "token_embd.weight"                                ||   // Token embedding
        offloader_name_ends(name, ".hc_attn_fn.weight")            ||   // Hyperconnection projection before attention
        offloader_name_ends(name, ".hc_attn_base.weight")          ||   // HC attention base stores pre[hc], post[hc], and comb[hc*hc] affine biases.
        offloader_name_ends(name, ".hc_attn_scale.weight")         ||   // HC attention scale stores separate pre, post, and comb affine scales.
        offloader_name_ends(name, ".attn_norm.weight")             ||   // Attention RMSNorm
        offloader_name_ends(name, ".attn_sinks.weight")            ||   // Attention sinks
        offloader_name_ends(name, ".attn_q_a.weight")              ||   // First low-rank Q projection
        offloader_name_ends(name, ".attn_q_a_norm.weight")         ||   // Low-rank Q RMSNorm
        offloader_name_ends(name, ".attn_q_b.weight")              ||   // Second low-rank Q projection
        offloader_name_ends(name, ".attn_kv.weight")               ||   // Shared attention KV projection
        offloader_name_ends(name, ".attn_kv_a_norm.weight")        ||   // Compressed KV RMSNorm
        offloader_name_ends(name, ".attn_compressor_kv.weight")    ||   // Compressed-attention KV projection
        offloader_name_ends(name, ".attn_compressor_gate.weight")  ||   // Compressed-attention score projection
        offloader_name_ends(name, ".attn_compressor_ape.weight")   ||   // Compressed-attention positional table
        offloader_name_ends(name, ".attn_compressor_norm.weight")  ||   // Compressed-attention RMSNorm
        offloader_name_ends(name, ".indexer_compressor_kv.weight") ||   // Indexer-state KV projection
        offloader_name_ends(name, ".indexer_compressor_gate.weight") || // Indexer-state score projection
        offloader_name_ends(name, ".indexer_compressor_ape.weight")  || // Indexer positional table
        offloader_name_ends(name, ".indexer_compressor_norm.weight") || // Indexer RMSNorm
        offloader_name_ends(name, ".indexer.attn_q_b.weight")      ||   // LID query projection
        offloader_name_ends(name, ".indexer.proj.weight")          ||   // LID weight projection
        offloader_name_ends(name, ".attn_output_a.weight")         ||   // Attention output projection A
        offloader_name_ends(name, ".attn_output_b.weight")         ||   // Attention output projection B
        offloader_name_ends(name, ".hc_ffn_fn.weight")             ||   // Hyperconnection projection before FFN
        offloader_name_ends(name, ".hc_ffn_base.weight")           ||   // HC FFN base stores pre[hc], post[hc], and comb[hc*hc] affine biases.
        offloader_name_ends(name, ".hc_ffn_scale.weight")          ||   // HC FFN scale stores separate pre, post, and comb affine scales.
        offloader_name_ends(name, ".ffn_norm.weight")              ||   // FFN RMSNorm
        offloader_name_ends(name, ".ffn_gate_inp.weight")          ||   // Dense MoE router projection
        offloader_name_ends(name, ".ffn_gate_tid2eid.weight")      ||   // Hash-router token-to-expert table
        offloader_name_ends(name, ".exp_probs_b.bias")             ||   // MoE expert-probability bias
        //offloader_name_ends(name, ".ffn_gate_exps.weight")       ||   // SPARSE
        //offloader_name_ends(name, ".ffn_down_exps.weight")       ||   // SPARSE
        //offloader_name_ends(name, ".ffn_up_exps.weight")         ||   // SPARSE
        offloader_name_ends(name, ".ffn_gate_shexp.weight")        ||   // Shared expert gate projection
        offloader_name_ends(name, ".ffn_up_shexp.weight")          ||   // Shared expert up projection
        offloader_name_ends(name, ".ffn_down_shexp.weight")        ||   // Shared expert down projection
        name == "output_hc_fn.weight"                              ||   // Final hyperconnection projection
        name == "output_hc_scale.weight"                           ||   // Final hyperconnection scale
        name == "output_hc_base.weight"                            ||   // Final hyperconnection base
        name == "output_norm.weight"                               ||   // Final RMSNorm
        name == "output.weight";                                        // LM head
}

//Op filters to speed up graph walking. Each model only checks ops that can directly read one of its enabled dense weights.
static bool parameter_offloader_deepseek2_node_may_read_dense_weight(const ggml_tensor * node)
{
    if (!node)
        return false;

    switch (node->op) {
        case GGML_OP_GET_ROWS:       // token_embd.weight;
        case GGML_OP_MUL:            // attn/output/FFN/Q/KV norm weights; currently disabled
        case GGML_OP_MUL_MAT:        // Q/KV/attention output, dense/shared FFN, router, and output weights
        case GGML_OP_ADD:            // exp_probs_b.bias; currently disabled
        //case GGML_OP_MUL_MAT_ID:   // routed MoE expert banks; sparse and handled separately
            return true;
        default:
            return false;
    }
}

static bool parameter_offloader_gpt_oss_node_may_read_dense_weight(const ggml_tensor * node)
{
    if (!node)
        return false;

    switch (node->op) {
        case GGML_OP_GET_ROWS:       // token_embd.weight
        case GGML_OP_MUL:            // attention/post-attention/output norm weights
        case GGML_OP_MUL_MAT:        // QKV, attention output, router, and output weights
        case GGML_OP_ADD:            // QKV, attention-output, and router biases
        case GGML_OP_SOFT_MAX:       // attn_sinks.weight on the non-flash attention path
        case GGML_OP_FLASH_ATTN_EXT: // attn_sinks.weight on the flash-attention path
        //case GGML_OP_MUL_MAT_ID:   // routed expert gate/up/down weights; sparse and handled separately
        //case GGML_OP_ADD_ID:       // routed expert gate/up/down biases; sparse and handled separately
            return true;
        default:
            return false;
    }
}

static bool parameter_offloader_deepseek4_node_may_read_dense_weight(const ggml_tensor * node)
{
    if (!node)
        return false;

    switch (node->op) {
        case GGML_OP_GET_ROWS:       // token embedding, compressor/indexer APE, hash-router table
        case GGML_OP_MUL:            // norm weights and final HC scale
        case GGML_OP_MUL_MAT:        // HC, attention, compressor/indexer, FFN, router, output
        case GGML_OP_ADD:            // expert-probability bias and final HC base
        case GGML_OP_SOFT_MAX:       // attn_sinks on non-flash attention
        case GGML_OP_FLASH_ATTN_EXT: // attn_sinks on flash attention
        //case GGML_OP_MUL_MAT_ID:   // routed expert banks; SPARSE
        case GGML_OP_DSV4_HC_COMB:   // per-layer HC base/scale
            return true;
        default:
            return false;
    }
}

static const parameter_offloader::parameter_offloader_model_i parameter_offloader_deepseek2_i = {
    /*weight_supported*/            parameter_offloader_deepseek2_weight_supported,
    /*node_may_read_dense_weight*/ parameter_offloader_deepseek2_node_may_read_dense_weight,
};

static const parameter_offloader::parameter_offloader_model_i parameter_offloader_gpt_oss_i = {
    /*weight_supported*/            parameter_offloader_gpt_oss_weight_supported,
    /*node_may_read_dense_weight*/ parameter_offloader_gpt_oss_node_may_read_dense_weight,
};

static const parameter_offloader::parameter_offloader_model_i parameter_offloader_deepseek4_i = {
    /*weight_supported*/            parameter_offloader_deepseek4_weight_supported,
    /*node_may_read_dense_weight*/ parameter_offloader_deepseek4_node_may_read_dense_weight,
};

/////////////////////////////////////
//   INITIALIZATION
/////////////////////////////////////
parameter_offloader::parameter_offloader(llama_model  * model)
    : model(model)
{
    switch (model->arch)
    {
        case LLM_ARCH_DEEPSEEK2:
            model_i = &parameter_offloader_deepseek2_i;
            break;
        case LLM_ARCH_OPENAI_MOE:
            model_i = &parameter_offloader_gpt_oss_i;
            break;
        case LLM_ARCH_DEEPSEEK4:
            model_i = &parameter_offloader_deepseek4_i;
            break;
        default:
            throw std::runtime_error("parameter_offloader: unsupported model architecture");
    }

    cpu_weight_set.clear();
    cpu_weight_set.reserve(model->tensors_by_name.size());

    cpu_weight_by_name.clear();
    cpu_weight_by_name.reserve(model->tensors_by_name.size());

    for (const auto & kv : model->tensors_by_name)
    {
        if (kv.second)
        {
            cpu_weight_set.insert(kv.second);
            cpu_weight_by_name.emplace(kv.first, kv.second);
        }
    }
}

void parameter_offloader::attach_arena(ggml_backend_buffer_t arena)
{
    GGML_ASSERT(arena);

    if (this->arena) {
        GGML_ASSERT(this->arena == arena);
        return;
    }

    this->arena = arena;

    arena_buffer_type = ggml_backend_buffer_get_type(arena);
    arena_base        = (char *) ggml_backend_buffer_get_base(arena);
    arena_size        = ggml_backend_buffer_get_size(arena);
    arena_dense_size  = arena_size;
    arena_stream_size = arena_size;
    arena_alignment   = ggml_backend_buffer_get_alignment(arena);

    GGML_ASSERT(arena_buffer_type);
    GGML_ASSERT(arena_base);
    GGML_ASSERT(arena_size > 0);
    GGML_ASSERT(arena_alignment > 0);
}

// Call this *before* transform/upload, i.e. at the top of parameter_offloader::init()
// Guarantees collected_order contains *all* host-backed model weights
void parameter_offloader::seed_all_weights_from_model()
{
    collected_order.clear();

    // Gather (name,tensor) to get a deterministic ordering (lexicographic by name)
    std::vector<std::pair<std::string, ggml_tensor *>> named;
    //named.reserve(model->tensors_by_name.size());

    for (const auto & kv : model->tensors_by_name)
    {

        ggml_tensor * t = kv.second;
        if (!t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer))
            continue; // only real host weights
        // Only keep actual weights you intend to manage (you already populated cpu_weight_set)
        if (cpu_weight_set.find(t) == cpu_weight_set.end())
            continue;

        if (!model_i->weight_supported(kv.first))
        {
            LLAMA_LOG_INFO("%s: rejecting %s\n", __func__, kv.first.c_str());
            continue;
        }

        named.emplace_back(kv.first, t);
    }

    std::sort(named.begin(), named.end(), [](auto &a, auto &b){ return a.first < b.first; });

    std::unordered_set<ggml_tensor*> collect_seen;  // dedupe during collection
    for (auto & kv : named)
    {
        ggml_tensor * t = kv.second;
        if (collect_seen.insert(t).second)
            collected_order.push_back(t);
    }

    //LLAMA_LOG_INFO("%s: model->tensors_by_name.size() == %lu\n", __func__, model->tensors_by_name.size());
    //LLAMA_LOG_INFO("%s: named.size() == %lu\n", __func__, named.size());
    LLAMA_LOG_INFO("%s: found %lu host-backed weights\n", __func__, collected_order.size());
}

size_t parameter_offloader::transform_all_cpu_weights_to_device_layout()
{
    size_t packed = 0;
    // Optional: skip obvious non-weights or already-packed
    for (ggml_tensor * w_cpu : collected_order)
        if (transform_cpu_tensor_to_device_layout(w_cpu))
            ++packed;
    return packed;
}

bool parameter_offloader::transform_cpu_tensor_to_device_layout(ggml_tensor * w_cpu) {
    if (!w_cpu) return false;

    // only real host-backed weights (usually file-mapped)
    if (!(w_cpu->buffer && ggml_backend_buffer_is_host(w_cpu->buffer))) return false;

    // already packed?
    if (host_packed_.count(w_cpu)) return false;

    GGML_ASSERT(arena);
    GGML_ASSERT(arena_buffer_type);
    GGML_ASSERT(arena_base);

    const size_t logical   = ggml_nbytes(w_cpu);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);

    //TODO: Move this requirement into the arena fitter once minimum viable arena sizing is defined.
    if (dev_bytes > arena_dense_size)
        throw std::runtime_error("parameter_offloader: tensor device layout does not fit inside dense arena scratch space");

    // Allocate a host RAM buffer sized like the device allocation so we can memcpy the whole region later.
    ggml_backend_buffer_type_t host_buft = ggml_backend_cpu_buffer_type(); // use pinned-host type if you have one
    ggml_backend_buffer_t host_buf = ggml_backend_buft_alloc_buffer(host_buft, dev_bytes);
    GGML_ASSERT(host_buf);
    uint8_t * host_base = (uint8_t *) ggml_backend_buffer_get_base(host_buf);

    ggml_init_params tmp_ip{ 64*1024, nullptr, true };
    ggml_context * tmp_ctx = ggml_init(tmp_ip);
    GGML_ASSERT(tmp_ctx);

    // Reuse the beginning of the dense arena as temporary transform scratch; no persistent dense tensor has been seated yet.
    ggml_tensor * tmp_dev = ggml_dup_tensor_layout_public(tmp_ctx, w_cpu);
    GGML_ASSERT(tmp_dev);
    GGML_ASSERT(ggml_backend_tensor_alloc(arena, tmp_dev, arena_base) == GGML_STATUS_SUCCESS);

    //TODO: Verify the CUDA arena tensor_set path applies the same type-specific device packing as the standard CUDA device buffer for every managed quantized type.
    // H2D: this call should trigger CUDA-side transform/packing for the tensor layout
    ggml_backend_tensor_set(tmp_dev, w_cpu->data, 0, logical);

    // D2H: read back ONLY the logical payload (tensor_get is bounded by ggml_nbytes())
    ggml_backend_tensor_get(tmp_dev, host_base, 0, logical);

    // Zero-fill the padded tail so later H2D memcpy can copy dev_bytes safely
    if (dev_bytes > logical) {
        std::memset(host_base + logical, 0, dev_bytes - logical);
    }

    // Cleanup temporary metadata; arena scratch is intentionally left disposable.
    ggml_free(tmp_ctx);

    // Remember the packed bytes for this weight
    host_packed_.emplace(w_cpu, PackedHostBytes{ host_buf, host_base, dev_bytes });
    return true;
}

// Enumerate model tensor pointer slots once so later CPU->GPU patching is a direct lookup instead of a full model scan.
void parameter_offloader::build_model_ref_lookup()
{
    model_ref_slots.clear();
    model_ref_slots.reserve(cpu_weight_set.size());

    // Build the original CPU-tensor name lookup once so the old pointer-OR-name matching behavior is preserved exactly.
    std::unordered_map<std::string, std::vector<ggml_tensor *>> cpu_weights_by_name;
    cpu_weights_by_name.reserve(cpu_weight_set.size());

    for (ggml_tensor * w_cpu : cpu_weight_set)
    {
        const char * name = ggml_get_name(w_cpu);
        if (name)
            cpu_weights_by_name[name].push_back(w_cpu);
    }

    // Register one mutable tensor-pointer slot under one CPU tensor, without adding the same slot twice.
    auto register_slot = [&](ggml_tensor * w_cpu, ggml_tensor * & slot) {
        std::vector<ggml_tensor **> & slots = model_ref_slots[w_cpu];
        if (std::find(slots.begin(), slots.end(), &slot) == slots.end())
            slots.push_back(&slot);
    };

    // Index one model member using both pointer identity and tensor-name matching, exactly like the old patch loop.
    auto INDEX = [&](ggml_tensor * & slot) {
        if (!slot)
            return;

        if (cpu_weight_set.find(slot) != cpu_weight_set.end())
            register_slot(slot, slot);

        const char * name = ggml_get_name(slot);
        if (!name)
            return;

        auto it = cpu_weights_by_name.find(name);
        if (it == cpu_weights_by_name.end())
            return;

        for (ggml_tensor * w_cpu : it->second)
            register_slot(w_cpu, slot);
    };

    // -------------------
    // top-level (model)
    // -------------------
    INDEX(model->tok_embd);
    INDEX(model->type_embd);
    INDEX(model->pos_embd);
    INDEX(model->tok_norm);
    INDEX(model->tok_norm_b);

    INDEX(model->output_norm);
    INDEX(model->output_norm_b);
    INDEX(model->output);
    INDEX(model->output_b);
    INDEX(model->output_norm_enc);

    INDEX(model->output_s);
    INDEX(model->output_in_s);
    INDEX(model->hc_head_fn);
    INDEX(model->hc_head_base);
    INDEX(model->hc_head_scale);

    INDEX(model->nextn_proj_pre);
    INDEX(model->nextn_proj_post);

    INDEX(model->cls);
    INDEX(model->cls_b);
    INDEX(model->cls_out);
    INDEX(model->cls_out_b);
    INDEX(model->cls_norm);

    INDEX(model->conv1d);
    INDEX(model->conv1d_b);

    INDEX(model->altup_proj);
    INDEX(model->altup_unembd_proj);
    INDEX(model->per_layer_tok_embd);
    INDEX(model->per_layer_model_proj);
    INDEX(model->per_layer_proj_norm);

    INDEX(model->fc);
    INDEX(model->d2t);

    // -------------------
    // per-layer
    // -------------------
    const int nl = (int)model->hparams.n_layer();
    for (int il = 0; il < nl; ++il)
    {
        llama_layer & L = model->layers[il];

        // normalization
        INDEX(L.attn_norm);        INDEX(L.attn_norm_b);
        INDEX(L.attn_norm_2);      INDEX(L.attn_norm_2_b);
        INDEX(L.attn_q_norm);      INDEX(L.attn_q_norm_b);
        INDEX(L.attn_k_norm);      INDEX(L.attn_k_norm_b);
        INDEX(L.attn_out_norm);    INDEX(L.attn_out_norm_b);
        INDEX(L.attn_q_a_norm);    INDEX(L.attn_kv_a_norm);
        INDEX(L.attn_kv_norm);
        INDEX(L.attn_sub_norm);    INDEX(L.attn_post_norm);
        INDEX(L.ffn_sub_norm);     INDEX(L.attn_norm_cross);
        INDEX(L.attn_norm_enc);    INDEX(L.ssm_norm);
        INDEX(L.ssm_dt_norm);      INDEX(L.ssm_b_norm);
        INDEX(L.ssm_c_norm);

        // attention
        INDEX(L.wq);        INDEX(L.wk);        INDEX(L.wv);        INDEX(L.wo);
        INDEX(L.wqkv);      INDEX(L.wq_a);      INDEX(L.wq_b);      INDEX(L.wkv_a_mqa);
        INDEX(L.wkv);       INDEX(L.wkv_b);     INDEX(L.wk_b);      INDEX(L.wv_b);
        INDEX(L.wqkv_b);    INDEX(L.wo_a);      INDEX(L.wo_b);
        INDEX(L.wq_cross);  INDEX(L.wk_cross);  INDEX(L.wv_cross);  INDEX(L.wo_cross);
        INDEX(L.wq_enc);    INDEX(L.wk_enc);    INDEX(L.wv_enc);    INDEX(L.wo_enc);
        INDEX(L.wqkv_gate);

        // relative position bias
        INDEX(L.attn_rel_b);       INDEX(L.attn_rel_b_enc);
        INDEX(L.attn_rel_b_cross);

        // normalization
        INDEX(L.ffn_norm);       INDEX(L.ffn_norm_b);
        INDEX(L.ffn_post_norm);  INDEX(L.ffn_post_norm_1); INDEX(L.ffn_post_norm_2);
        INDEX(L.ffn_pre_norm_2); INDEX(L.layer_out_norm);  INDEX(L.layer_out_norm_b);
        INDEX(L.ffn_norm_exps);  INDEX(L.ffn_norm_enc);

        // ff
        INDEX(L.ffn_gate);       INDEX(L.ffn_down);
        INDEX(L.ffn_up);         INDEX(L.ffn_gate_enc);
        INDEX(L.ffn_down_enc);   INDEX(L.ffn_up_enc);

        // ff MoE
        INDEX(L.ffn_gate_inp);      INDEX(L.ffn_gate_inp_s);
        INDEX(L.ffn_gate_tid2eid);
        //INDEX(L.ffn_gate_exps);     INDEX(L.ffn_down_exps);       //sparse layers are handled separately
        //INDEX(L.ffn_up_exps);       INDEX(L.ffn_gate_up_exps);
        INDEX(L.ffn_gate_inp_b);
        //INDEX(L.ffn_gate_exps_b);    INDEX(L.ffn_down_exps_b);
        //INDEX(L.ffn_up_exps_b);      INDEX(L.ffn_gate_up_exps_b);

        // ff MoE per-expert scales (NVFP4 per-tensor scale2)
        // Routed expert tensors are handled by the sparse cache.
        //INDEX(L.ffn_gate_exps_s);     INDEX(L.ffn_down_exps_s);
        //INDEX(L.ffn_up_exps_s);

        // ff MoE latent proj
        INDEX(L.ffn_latent_down);     INDEX(L.ffn_latent_up);

        // ffn shared expert (shexp)
        INDEX(L.ffn_gate_inp_shexp);  INDEX(L.ffn_gate_shexp);
        INDEX(L.ffn_down_shexp);      INDEX(L.ffn_up_shexp);

        // ff adjugate experts (chexps)
        //INDEX(L.ffn_gate_chexps);     INDEX(L.ffn_down_chexps);       //sparse layers are handled separately
        //INDEX(L.ffn_up_chexps);

        // ffn bias
        INDEX(L.ffn_gate_b);   INDEX(L.ffn_down_b);
        INDEX(L.ffn_up_b);     INDEX(L.ffn_act);
        INDEX(L.ffn_exp_probs_b);

        // mamba proj
        INDEX(L.ssm_in);   INDEX(L.ssm_x);
        INDEX(L.ssm_dt);   INDEX(L.ssm_out);

        // mamba
        INDEX(L.ssm_conv1d);   INDEX(L.ssm_a);
        INDEX(L.ssm_d);

        // mamba bias
        INDEX(L.ssm_conv1d_b); INDEX(L.ssm_dt_b);

        // qwen3next
        INDEX(L.ssm_beta_alpha);

        // qwen3.5
        INDEX(L.ssm_alpha);

        // rwkv
        INDEX(L.time_mix_w1);     INDEX(L.time_mix_w2);
        INDEX(L.time_mix_lerp_x); INDEX(L.time_mix_lerp_w);
        INDEX(L.time_mix_lerp_k); INDEX(L.time_mix_lerp_v);
        INDEX(L.time_mix_lerp_r); INDEX(L.time_mix_lerp_g);
        INDEX(L.time_mix_lerp_fused);

        INDEX(L.time_mix_first);      INDEX(L.time_mix_decay);
        INDEX(L.time_mix_decay_w1);   INDEX(L.time_mix_decay_w2);
        INDEX(L.time_mix_key);        INDEX(L.time_mix_key_b);
        INDEX(L.time_mix_value);      INDEX(L.time_mix_value_b);
        INDEX(L.time_mix_receptance); INDEX(L.time_mix_receptance_b);
        INDEX(L.time_mix_gate);

        // rwkv7
        INDEX(L.time_mix_w0);
        INDEX(L.time_mix_a0);  INDEX(L.time_mix_a1);  INDEX(L.time_mix_a2);
        INDEX(L.time_mix_v0);  INDEX(L.time_mix_v1);  INDEX(L.time_mix_v2);
        INDEX(L.time_mix_g1);  INDEX(L.time_mix_g2);
        INDEX(L.time_mix_k_k); INDEX(L.time_mix_k_a); INDEX(L.time_mix_r_k);

        INDEX(L.time_mix_ln);  INDEX(L.time_mix_ln_b);
        INDEX(L.time_mix_output);

        INDEX(L.channel_mix_lerp_k);   INDEX(L.channel_mix_lerp_r);

        INDEX(L.channel_mix_key);      INDEX(L.channel_mix_receptance);
        INDEX(L.channel_mix_value);

        // long rope factors
        INDEX(L.rope_long); INDEX(L.rope_short); INDEX(L.rope_freqs);

        // bitnet scale
        INDEX(L.wq_s);   INDEX(L.wk_s);   INDEX(L.wv_s);   INDEX(L.wo_s);
        INDEX(L.wqkv_s); INDEX(L.wqkv_gate_s);
        INDEX(L.ffn_gate_s);       INDEX(L.ffn_up_s);       INDEX(L.ffn_down_s);
        INDEX(L.ffn_gate_shexp_s); INDEX(L.ffn_up_shexp_s); INDEX(L.ffn_down_shexp_s);
        INDEX(L.ssm_in_s);    INDEX(L.ssm_out_s);
        INDEX(L.ssm_alpha_s); INDEX(L.ssm_beta_s);

        // input scales
        INDEX(L.wq_in_s);   INDEX(L.wk_in_s);   INDEX(L.wv_in_s);   INDEX(L.wo_in_s);
        INDEX(L.wqkv_in_s); INDEX(L.wqkv_gate_in_s);
        INDEX(L.ffn_gate_in_s);       INDEX(L.ffn_up_in_s);        INDEX(L.ffn_down_in_s);
        // Routed expert tensors are handled by the sparse cache.
        //INDEX(L.ffn_gate_exps_in_s);  INDEX(L.ffn_down_exps_in_s); INDEX(L.ffn_up_exps_in_s);
        INDEX(L.ffn_gate_shexp_in_s); INDEX(L.ffn_up_shexp_in_s);  INDEX(L.ffn_down_shexp_in_s);
        INDEX(L.ssm_in_in_s);    INDEX(L.ssm_out_in_s);
        INDEX(L.ssm_alpha_in_s); INDEX(L.ssm_beta_in_s);

        // altup & laurel
        INDEX(L.per_layer_inp_gate); INDEX(L.per_layer_proj); INDEX(L.per_layer_post_norm);
        INDEX(L.altup_correct_coef);  INDEX(L.altup_correct_scale);
        INDEX(L.altup_predict_coef);  INDEX(L.altup_router);
        INDEX(L.altup_router_norm);
        INDEX(L.laurel_l);  INDEX(L.laurel_r);
        INDEX(L.laurel_post_norm);

        // openai-moe
        INDEX(L.attn_sinks);

        // cogvlm
        INDEX(L.visexp_attn_wqkv);  INDEX(L.visexp_attn_wo);
        INDEX(L.visexp_ffn_gate);   INDEX(L.visexp_ffn_down);
        INDEX(L.visexp_ffn_up);

        // xIELU activation parameters for Apertus
        INDEX(L.ffn_act_alpha_n);  INDEX(L.ffn_act_alpha_p);
        INDEX(L.ffn_act_beta);     INDEX(L.ffn_act_eps);

        // Kimi Linear KDA
        INDEX(L.ssm_q_conv); INDEX(L.ssm_k_conv); INDEX(L.ssm_v_conv);
        INDEX(L.ssm_f_a);    INDEX(L.ssm_f_b);    INDEX(L.ssm_beta);
        INDEX(L.ssm_g_a);    INDEX(L.ssm_g_b);    INDEX(L.ssm_o_norm);

        // DSA
        INDEX(L.indexer_k_norm); INDEX(L.indexer_k_norm_b); INDEX(L.indexer_proj);
        INDEX(L.indexer_attn_k); INDEX(L.indexer_attn_q_b);

        // DeepSeek V4
        INDEX(L.hc_attn_fn);       INDEX(L.hc_ffn_fn);
        INDEX(L.hc_attn_base);     INDEX(L.hc_attn_scale);
        INDEX(L.hc_ffn_base);      INDEX(L.hc_ffn_scale);
        INDEX(L.attn_comp_wkv);    INDEX(L.attn_comp_wgate);
        INDEX(L.attn_comp_ape);    INDEX(L.attn_comp_norm);
        INDEX(L.indexer_comp_wkv); INDEX(L.indexer_comp_wgate);
        INDEX(L.indexer_comp_ape); INDEX(L.indexer_comp_norm);

        // gemma4 layer output scale, reused for talkie embedding skip scale
        INDEX(L.out_scale);
    }

    // Index model->tensors_by_name using both value-pointer identity and the map key as the old name fallback.
    for (auto & kv : model->tensors_by_name)
    {
        if (kv.second && cpu_weight_set.find(kv.second) != cpu_weight_set.end())
            register_slot(kv.second, kv.second);

        auto it = cpu_weights_by_name.find(kv.first);
        if (it == cpu_weights_by_name.end())
            continue;

        for (ggml_tensor * w_cpu : it->second)
            register_slot(w_cpu, kv.second);
    }
}

// Patch every pre-indexed model slot that refers to this CPU tensor.
void parameter_offloader::patch_model_refs_for(ggml_tensor * w_cpu, ggml_tensor * w_gpu)
{
    auto it = model_ref_slots.find(w_cpu);
    if (it == model_ref_slots.end())
        return;

    for (ggml_tensor ** slot : it->second)
        *slot = w_gpu;
}

ggml_tensor * parameter_offloader::init_cpu_tensor_to_arena(ggml_tensor * w_cpu, size_t & current_offset)
{
    GGML_ASSERT(ctx_gpu_twins);
    GGML_ASSERT(arena);
    GGML_ASSERT(w_cpu);
    GGML_ASSERT(w_cpu->buffer && ggml_backend_buffer_is_host(w_cpu->buffer));     // Must be a “real” weight buffer on host
    GGML_ASSERT(w_cpu->view_src == nullptr);                                       // Views complicate placement; for weights we expect contiguous

    // If we already mirrored this weight, return the existing twin
    auto it_cpu2gpu = cpu2gpu.find(w_cpu);
    if (it_cpu2gpu != cpu2gpu.end())
    {
        LLAMA_LOG_WARN("%s: %s is already mirrored, skipping...\n", __func__, ggml_get_name(w_cpu));
        return it_cpu2gpu->second;
    }
    
    // Compute padded slot as the backend will expect it on device
    const size_t slot_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);
    size_t off              = align_up(current_offset, arena_alignment);
    
    if (off + slot_bytes > arena_dense_size)
        off = 0; // wrap

    // starting from current 'off' (possibly just wrapped to 0), bump until unused
    const size_t bump      = arena_alignment;                         // step by arena alignment
    const size_t max_tries = arena_dense_size / arena_alignment + 2;  // safety bound
    size_t tries = 0;
    while (std::any_of(gpu2cpu.begin(), gpu2cpu.end(),
                [&](const auto &kv) { return kv.first && kv.first->data == static_cast<void*>(arena_base + off); }))
    {
        off = align_up(off + bump, arena_alignment);
        if (off + slot_bytes > arena_dense_size)
            off = 0; // wrap again if we ran past the end
        if (++tries > max_tries) {
            LLAMA_LOG_WARN("arena: could not find unique pointer for '%s' "
                        "(arena_dense_size=%zu, arena_alignment=%zu, entries=%zu) — proceeding with overlap\n",
                        ggml_get_name(w_cpu), arena_dense_size, arena_alignment, gpu2cpu.size());
            break; // fall through; last 'off' may collide but we’ve warned
        }
    }

    // Duplicate tensor metadata into the GPU-twins context (no data yet)
    ggml_tensor* w_gpu = ggml_dup_tensor_layout_public(ctx_gpu_twins, w_cpu);
    GGML_ASSERT(w_gpu);
    ggml_set_name(w_gpu, ggml_get_name(w_cpu)); // keep names consistent (optional)

    // Bind GPU twin into the arena at [arena_base + off]
    GGML_ASSERT(ggml_backend_tensor_alloc(arena, w_gpu, arena_base + off) == GGML_STATUS_SUCCESS);

    // Upload
    ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu, w_gpu);
    if (ev) {
        ggml_cuda_copy_event_wait(ev);
        ggml_cuda_copy_event_destroy(ev);
    }

    // Register mappings
    gpu2cpu.emplace(w_gpu, w_cpu);
    cpu2gpu.emplace(w_cpu, w_gpu);
    gpu_weight_set.insert(w_gpu);

    int idx = (int)schedule_current.gpu_tensors_in_order.size();
    schedule_current.cpu_tensors_in_order.push_back(w_cpu);
    schedule_current.gpu_tensors_in_order.push_back(w_gpu);
    schedule_current.gpu2index.emplace(w_gpu, idx);

    // Bump arena pointer
    current_offset = off + slot_bytes;

    patch_model_refs_for(w_cpu, w_gpu);

#ifdef LLAMA_CHECK_WEIGHTS
    //record hashes right after we create the gpu tensors
    const size_t nbytes_g = ggml_nbytes(w_gpu);
    std::vector<uint8_t> tmp_(nbytes_g);
    // copy device -> host for the logical bytes
    ggml_backend_tensor_get(w_gpu, tmp_.data(), 0, nbytes_g);
    gpu_hashes[w_gpu] = fnv1a64(tmp_.data(), nbytes_g);
#endif

    return w_gpu;
}

void parameter_offloader::init(ggml_backend_buffer_t arena, llama_context_params cparams, ggml_context * ctx_twins)
{
    attach_arena(arena);

    GGML_ASSERT(ctx_gpu_twins == nullptr);
    GGML_ASSERT(!cparams.moe_expert_prefetch || ctx_moe_cache != nullptr);

    owns_arena     = true;
    ctx_gpu_twins  = ctx_twins;

    // Optional: reserve to avoid rehash during init
    gpu2cpu.reserve(4096);
    cpu2gpu.reserve(4096);

    seed_all_weights_from_model();

    const size_t packed = transform_all_cpu_weights_to_device_layout();
    LLAMA_LOG_INFO("host-packing: %zu/%zu weights packed on host\n",
                packed, collected_order.size());

    // Build the model-slot index once, then patch each mirrored tensor through direct lookup.
    build_model_ref_lookup();

    size_t current_offset = 0;
    for (ggml_tensor * w_cpu : collected_order)
        (void) init_cpu_tensor_to_arena(w_cpu, current_offset);

    model_ref_slots.clear();

    //////////////////////////////////////////////////////////////////
    //       CREATE COPY SCHEDULE
    //////////////////////////////////////////////////////////////////

    build_schedule_gates(schedule_current);

    //////////////////////////////////////////////////////////////////
    //       FINISH UP
    //////////////////////////////////////////////////////////////////

    const size_t tensor_count = schedule_current.gpu_tensors_in_order.size();
    tensor_idx_copied_ordinal.store((long long)tensor_count - 1, std::memory_order_release);
    tensor_idx_used_ordinal.store(  (long long)tensor_count - 1, std::memory_order_release);

    ready = true;
    LLAMA_LOG_INFO("%s ready\n", __func__);

    //print_snapshot(schedule_current);

    // Optional log
    size_t peak = 0;
    if (!schedule_current.end_offset.empty())
        peak = *std::max_element(schedule_current.end_offset.begin(), schedule_current.end_offset.end());
    LLAMA_LOG_INFO("%s: vram-offload: scheduled %zu tensors; peak logical occupancy ~%zu bytes\n", __func__, tensor_count, peak);
}
parameter_offloader::~parameter_offloader()
{
    stop_streamer_join();
    clear_moe_cache_refs();
    if (ctx_moe_cache) {
        ggml_free(ctx_moe_cache);
        ctx_moe_cache = nullptr;
    }
    if (ctx_gpu_twins) {
        ggml_free(ctx_gpu_twins);
        ctx_gpu_twins = nullptr;
    }
    if (arena && owns_arena) {
        ggml_backend_buffer_free(arena);
    }
    arena = nullptr;

    // host_packed_ owns the permanent host buffers allocated by transform_cpu_tensor_to_device_layout().
    for (auto & kv : host_packed_)
        if (kv.second.buf)
            ggml_backend_buffer_free(kv.second.buf);
    host_packed_.clear();
}

void parameter_offloader::start()
{
    GGML_ASSERT(ready);
#if defined(LLAMA_DIAGNOSE_COPY)
    //streamer thread disabled during debug mode, so get things started with the first copy
    ggml_tensor *w_cpu = schedule_current.cpu_tensors_in_order[0];
    ggml_tensor *w_gpu = schedule_current.gpu_tensors_in_order[0];

    ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu, w_gpu);
    if (ev) {
        ggml_cuda_copy_event_wait(ev);
        ggml_cuda_copy_event_destroy(ev);
    }

    tensor_idx_copied_ordinal.store((long long)tensor_count, std::memory_order_release);
#else
    start_streamer();                         // begin background H2D streaming
#endif
}
void parameter_offloader::start_streamer() {
    stop_stream.store(false, std::memory_order_release);
    copy_thread = std::thread(&parameter_offloader::stream_worker, this);
}
void parameter_offloader::stop_streamer_join() {
    stop_stream.store(true, std::memory_order_release);
    node_cv_.notify_all();

    if (copy_thread.joinable())
        copy_thread.join();

    std::unique_lock<std::mutex> lk(node_mu_);
    node_cv_.wait(lk, [&] {
        return copy_publishers_in_flight.load(std::memory_order_acquire) == 0;
    });
}


/////////////////////////////////////
//   READ
/////////////////////////////////////

// Return true if node 't' reads any tracked GPU twin; optionally output the
// furthest tracked weight used by the node in the active schedule.
bool parameter_offloader::node_reads_tracked_weight(ggml_tensor * t, int * out_idx = nullptr)
{
    if (!model_i->node_may_read_dense_weight(t))
        return false;

    int best_idx = -1;
    const int N  = (int)schedule_current.gpu_tensors_in_order.size();

    for (int k = 0; k < GGML_MAX_SRC; ++k)
    {
        ggml_tensor * source = t->src[k];
        if (!source)
            break;

        while (source->view_src)
            source = source->view_src;

        auto it = schedule_current.gpu2index.find(source);
        if (it == schedule_current.gpu2index.end())
            continue;

        const int idx = it->second;
        if (best_idx < 0) {
            best_idx = idx;
            continue;
        }

        // pick "furthest ahead" vs the last used index to keep monotonic progress
        const long long used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_relaxed);
        if (used_ordinal < 0)
        {
            if (idx > best_idx)
                best_idx = idx;

            continue;
        }

        const int last = ordinal_mod(used_ordinal, N);

        const int dist_best = (best_idx - last + N) % N;
        const int dist_new  = (idx      - last + N) % N;
        if (dist_new > dist_best)
            best_idx = idx;
    }

    if (best_idx < 0)
        return false;

    if (out_idx)
        *out_idx = best_idx;

    return true;
}

// Ask-phase: only opt in for nodes that read any tracked weight.
// Keeps batching intact for other nodes.
bool parameter_offloader::wants_observe(ggml_tensor * node)
{
    if (schedule_current.gpu_tensors_in_order.empty())
        return false;

#ifdef LLAMA_CHECK_NODES
    const char * name = ggml_get_name(node);
    for (int i = 0; i < GGML_MAX_SRC; ++i)
    {
        ggml_tensor * src_node = node->src[i];
        if (!src_node)
            break;

        while (src_node->view_src)
            src_node = src_node->view_src;

        const char * src_name = ggml_get_name(src_node);
        ggml_backend_buffer_t buf = src_node->buffer;

        // skip true model weights that we track (either CPU weight or its GPU twin)
        bool is_tracked_weight =
            (cpu2gpu.find(src_node) != cpu2gpu.end()) ||
            (gpu2cpu.find(src_node) != gpu2cpu.end());

        if (!is_tracked_weight)
        //if (!node_reads_tracked_weight(src_node, /*out_idx*/ nullptr))
        {
            // 1) Is this source’sp buffer literally the same buffer handle as your weight arena?
            if (src_node->buffer == arena) {
                LLAMA_LOG_WARN("[ACT-IN-WEIGHT-ARENA] node=%s tens=%s ptr=%p\n",
                            ggml_get_name(node)?: "(unnamed)",
                            ggml_get_name(src_node)?: "(unnamed)", src_node->data);
            }

            // 2) Does this pointer fall inside your arena’s [arena_base, arena_base+arena_dense_size)?
            char *q = (char*) src_node->data;
            if (q && q >= arena_base && q < arena_base + arena_dense_size) {
                LLAMA_LOG_WARN("[ACT-IN-WEIGHT-RANGE] node=%s tens=%s ptr=%p (inside [%p, %p))\n",
                            ggml_get_name(node)?: "(unnamed)",
                            ggml_get_name(src_node)?: "(unnamed)", src_node->data, arena_base, arena_base + arena_dense_size);
            }
        }

#ifdef LLAMA_PRINT_ALL_NODES
        //LLAMA_LOG_INFO("[SRC node=%s src=%d tens=%s buf=%s op=%s host=%d ptr=%p]\n",
        //        name ? name : "(unnamed)",
        //        i, n ? n : "(unnamed)", ggml_backend_buffer_name(buf), ggml_op_to_string(node->op), (int)ggml_backend_buffer_is_host(buf), src_node->data);
        // Column widths; adjust as needed
#define NAME_W 30
#define SRC_W   2
#define TENS_W 30
#define BUF_W  16
#define OP_W   24
#define PTR_W  18  // width for %p column; %p length may vary by platform
#define NN(s) ((s) ? (s) : "(unnamed)")
        LLAMA_LOG_INFO(
            "[SRC  node=%-*s src=%*d src_name=%-*s buf=%-*s op=%-*s src_op=%-*s host=%d ptr=%*p]\n",
            NAME_W, NN(name),                        // node
            SRC_W,  i,                               // src
            TENS_W, NN(src_name),                    // src_name
            BUF_W,  ggml_backend_buffer_name(buf),   // buf
            OP_W,   ggml_op_to_string(node->op),     // op
            OP_W,   ggml_op_to_string(src_node->op), // src_op
            (int)ggml_backend_buffer_is_host(buf),   // host
            PTR_W,  (void*)src_node->data            // ptr
        );
#endif /* #ifdef LLAMA_PRINT_ALL_NODES */
#ifdef LLAMA_CHECK_NON_FINITES
        if (finite_check_node(src_node, true))
            run_probe_rules_for_bad_source(src_node, true);
#endif
    }
#ifdef LLAMA_CHECK_NON_FINITES
    //checking the node itself pre-compute is meaningless
    //finite_check_node(node, true);
#endif
#endif /* #ifdef LLAMA_CHECK_NODES */
    return node_reads_tracked_weight(node, /*out_idx*/ nullptr);
}

// Called after an observed node executes; advance all schedule positions released by this node and wait for the next required streamed tensor if COPY has not reached it yet.
bool parameter_offloader::on_eval_tensor(ggml_tensor * node)
{
    if (schedule_current.gpu_tensors_in_order.empty())
        return false;

    int idx = -1;
    if (!node_reads_tracked_weight(node, &idx))
        return false;  //should this be a return true?

    // for readable logs
    const int tensor_count = (int)schedule_current.gpu_tensors_in_order.size();
    //if (idx == tensor_count - 1)
    //    LLAMA_LOG_INFO("%s got to idx %d\n", __func__, idx);

    long long used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_relaxed);
    bool gate_advanced = false;

    for (int i = 0; i < tensor_count; ++i)
    {
        const int release_idx = ordinal_mod(used_ordinal + 1, tensor_count);
        if (graph_analysis_current.release_node_by_tensor[release_idx] != node)
            break;

        ++used_ordinal;
        gate_advanced = true;
    }

    if (gate_advanced)
    {
        tensor_idx_used_ordinal.store(used_ordinal, std::memory_order_release);

        // wake streamer
        node_cv_.notify_all();
    }

#ifdef LLAMA_LOG_READS
    LLAMA_LOG_INFO("[R.%d]", idx);
#endif

    // Find the first newly-required streamed tensor that COPY must reach before compute advances past this node.
    auto read_next_it = graph_analysis_current.next_required_tensor_idx.find(node);
    
    // Convert that schedule index to its monotonic COPY ordinal for the current traversal of the ring.
    const long long needed_copy_ordinal = read_next_it != graph_analysis_current.next_required_tensor_idx.end() ? advance_ordinal_to_idx(used_ordinal, read_next_it->second, tensor_count) : -1;

#if defined(LLAMA_DIAGNOSE_COPY)
    long long copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_relaxed);

    for (;;)
    {
        // snapshot the current reader ordinal
        long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);
        const int i = ordinal_mod(copied_ordinal + 1, tensor_count);

        {
            const int r_idx = cur_used_ordinal < 0 ? tensor_count - 1 : ordinal_mod(cur_used_ordinal, tensor_count); // current reader index
            const int bar   = schedule_current.ready_after[r_idx];         // last copyable index while r is read
            const int di    = ring_distance(r_idx, i, tensor_count);       // distance from reader to copy slot
            const int dbar  = ring_distance(r_idx, bar, tensor_count);     // distance from reader to barrier

            bool allowed = (di <= dbar) && (di <= LLAMA_DIAGNOSE_COPY || copied_ordinal < needed_copy_ordinal);

            if (!allowed)
                break;
        }

        // Safe to copy slot i
        ggml_tensor *w_cpu_ = schedule_current.cpu_tensors_in_order[i];
        ggml_tensor *w_gpu_ = schedule_current.gpu_tensors_in_order[i];

        ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu_, w_gpu_);
        if (ev) {
            ggml_cuda_copy_event_wait(ev);
            ggml_cuda_copy_event_destroy(ev);
        }

    #ifdef LLAMA_LOG_COPIES
        LLAMA_LOG_INFO("[C.%d]", i);
    #endif

        ++copied_ordinal;
        tensor_idx_copied_ordinal.store(copied_ordinal, std::memory_order_release);
    }
#endif /* defined(LLAMA_DIAGNOSE_COPY) */
    
    // Wait only when this node has a next-tensor COPY requirement.
    if (read_next_it != graph_analysis_current.next_required_tensor_idx.end())
    {
        long long cur_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);

        if (cur_copied_ordinal < needed_copy_ordinal)
        {
        #if LLAMA_LOG_READS > 1
            LLAMA_LOG_INFO("[RB.%d.%lld.%lld.%lld]", idx, cur_copied_ordinal, needed_copy_ordinal, used_ordinal);
        #endif
            std::unique_lock<std::mutex> lk(node_mu_);
            node_cv_.wait(lk, [&]{
                return stop_stream.load(std::memory_order_acquire) ||
                       tensor_idx_copied_ordinal.load(std::memory_order_acquire) >= needed_copy_ordinal;
            });
            if (stop_stream.load(std::memory_order_acquire)) {
                // shutting down; don't block compute
                return false;
            }
        }
    }

#ifdef LLAMA_CHECK_WEIGHTS
    {
        // Resolve the GPU twin + its CPU source so we can print name/size/offset
        ggml_tensor * w_gpu = schedule_current.gpu_tensors_in_order[idx];
        ggml_tensor * w_cpu = gpu2cpu.at(w_gpu);
        const char * name   = ggml_get_name(w_gpu);
        const size_t off    = (size_t)((char *) w_gpu->data - arena_base);

        const size_t nbytes_g = ggml_nbytes(w_gpu);
        std::vector<uint8_t> tmp_(nbytes_g);
        // copy device -> host for the logical bytes
        ggml_backend_tensor_get(w_gpu, tmp_.data(), 0, nbytes_g);
        if (gpu_hashes[w_gpu] != fnv1a64(tmp_.data(), nbytes_g))
        {
            LLAMA_LOG_WARN(
                "[MISMATCH] idx=%d name=%s off=%zu bytes=%zu\n",
                idx, name ? name : "(unnamed)", (size_t)off, (size_t)nbytes_g
            );
        }
#ifdef LLAMA_CHECK_WEIGHTS_VERBOSE
        else
        {
            LLAMA_LOG_INFO(
                "[GOOD] idx=%d name=%s off=%zu bytes=%zu\n",
                idx, name ? name : "(unnamed)", (size_t)off, (size_t)nbytes_g
            );
        }
#endif
    }
#endif

#ifdef LLAMA_CHECK_NON_FINITES
    finite_check_node(node);

    if (const char *nm = ggml_get_name(node)) {
        if (strncmp(nm, "fattn_mla", 9) == 0) {
            // node has just been computed and flagged non-finite in your log
            // inspect its inputs *post*-compute too:

            ggml_tensor * w = node->src[0]; // V weight
            ggml_tensor * x = node->src[1]; // FA output (permuted)

            // Check the FA output that feeds the GEMM:
            if (x)
                finite_check_node(x);    // now meaningful (x exists & is computed)

            // Sanity: weight should be finite (it’s constant), but assert anyway
            if (w)
                finite_check_node(w);
        }
    }
#endif

    return true;
}


// signature must match ggml_backend_sched_eval_callback
bool llama_offloader_eval_cb(ggml_tensor * node, bool ask, void * ud)
{
    struct parameter_offloader * po = static_cast<parameter_offloader *>(ud);
    if (!po)
    {
        LLAMA_LOG_INFO("%s ask == %d\n", __func__, ask);
        return !ask;
        //return true;
    }

    // Only return true for weights you actually track; keeps batching intact
    if (ask) {
        return po->wants_observe(node);
    } else {
        // === POST-COMPUTE ===
        po->on_eval_tensor(node);  // keep your current bookkeeping first

    //LLAMA_LOG_INFO("%s after ask == %d\n", __func__, ask);

#ifdef LLAMA_CHECK_NON_FINITES
        // 3) alias/overlap guard for view/reshape/concat/permutation outputs
        auto ranges_overlap = [](char *a0, size_t an, char *b0, size_t bn) -> bool {
            char *a1 = a0 + an, *b1 = b0 + bn;
            return !(a1 <= b0 || b1 <= a0);
        };
        auto logical_nbytes = [](const ggml_tensor *t) -> size_t {
            return ggml_nbytes(t);
        };

        switch (node->op) {
            case GGML_OP_CONCAT:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_VIEW: {
                char  * outp = (char *) node->data;
                size_t on    = logical_nbytes(node);
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor * s = node->src[k];
                    if (!s) break;
                    while (s->view_src) s = s->view_src;
                    char  * inp = (char *) s->data;
                    size_t in   = logical_nbytes(s);
                    if (outp && inp && ranges_overlap(outp, on, inp, in)) {
                        LLAMA_LOG_WARN("[ALIAS-OVERLAP] op=%s node=%s out=%p in=%p",
                                       ggml_op_to_string(node->op),
                                       ggml_get_name(node) ? ggml_get_name(node) : "(unnamed)",
                                       outp, inp);
                    }
                }
            } break;
            default: break;
        }

        // Optional: when fattn_mla-* trips, check its RHS input post-compute
        if (const char * nm = ggml_get_name(node)) {
            if (strncmp(nm, "fattn_mla", 9) == 0) {
                ggml_tensor * rhs = node->src[1];
                if (rhs) (void) finite_check_node(rhs, /*log_if_bad=*/true);
            }
        }

        // your diagnose-copy/read (if any), then:
        // node_cv_.notify_all();
#endif
        return true;
    }
}

/////////////////////////////////////
//   COPY
/////////////////////////////////////
inline bool parameter_offloader::no_transform_needed_for_backend_(const ggml_tensor *t) const {
    const size_t logical   = ggml_nbytes(t);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, t);

    switch (t->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
            return dev_bytes == logical;   // safe: same bytes, no backend packing
        default:
            // If you *know* a quantized type for your CUDA kernels matches host layout,
            // add it here and keep the dev_bytes==logical guard:
            // case GGML_TYPE_Q8_0: return dev_bytes == logical;
            return false;
    }
}

void parameter_offloader::copy_host_to_arena_with_transform(ggml_tensor * src_host, ggml_tensor * dst_arena)
{
    GGML_ASSERT(src_host && dst_arena);
    GGML_ASSERT(ggml_backend_buffer_is_host(src_host->buffer));
    GGML_ASSERT(dst_arena->buffer == arena);

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(dev);

    ggml_backend_buffer_type_t dev_buft = ggml_backend_dev_buffer_type(dev);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(dev_buft, src_host);

    ggml_backend_buffer_t tmp_buf = ggml_backend_buft_alloc_buffer(dev_buft, dev_bytes);
    GGML_ASSERT(tmp_buf);

    ggml_init_params tmp_ip{ 64*1024, nullptr, true };
    ggml_context * tmp_ctx = ggml_init(tmp_ip);
    GGML_ASSERT(tmp_ctx);

    ggml_tensor * tmp_dev = ggml_dup_tensor_layout_public(tmp_ctx, src_host);
    GGML_ASSERT(tmp_dev);

    void * tmp_base = ggml_backend_buffer_get_base(tmp_buf);
    GGML_ASSERT(ggml_backend_tensor_alloc(tmp_buf, tmp_dev, tmp_base) == GGML_STATUS_SUCCESS);

    // This applies the CUDA-side packing/transform for the tensor's type:
    ggml_backend_tensor_set(tmp_dev, src_host->data, 0, ggml_nbytes(src_host));

    // Now copy the transformed device bytes into your arena twin (D2D):
    ggml_backend_tensor_copy(tmp_dev, dst_arena);

    ggml_free(tmp_ctx);
    ggml_backend_buffer_free(tmp_buf);
}

inline ggml_cuda_copy_event * parameter_offloader::upload_weight_auto(ggml_tensor *w_cpu, ggml_tensor *w_gpu) {
    GGML_ASSERT(w_cpu && w_gpu);
    GGML_ASSERT(ggml_backend_buffer_is_host(w_cpu->buffer));
    GGML_ASSERT(w_gpu->buffer == arena);

    auto it = host_packed_.find(w_cpu);
    if (it != host_packed_.end()) {
        if (ggml_backend_buffer_is_cuda_arena_public(arena)) {
            const size_t dev_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);
            ggml_cuda_copy_event * ev = ggml_cuda_copy_event_create(arena);

            ggml_cuda_arena_tensor_write_raw_async(arena, w_gpu, it->second.base, dev_bytes, ev);

            return ev;
        } else {
            const size_t logical = ggml_nbytes(w_gpu);
            ggml_backend_tensor_set(w_gpu, it->second.base, 0, logical);
            return nullptr;
        }
    }

    if (no_transform_needed_for_backend_(w_cpu)) {
        const size_t logical = ggml_nbytes(w_cpu);
        ggml_backend_tensor_set(w_gpu, w_cpu->data, 0, logical);
        return nullptr;
    }

    copy_host_to_arena_with_transform(w_cpu, w_gpu);
    return nullptr;
}

void parameter_offloader::publish_copy_when_ready(long long ordinal, uint64_t generation, ggml_cuda_copy_event * ev)
{
    ggml_cuda_copy_event_wait(ev);
    ggml_cuda_copy_event_destroy(ev);

    std::unique_lock<std::mutex> lk(node_mu_);

    node_cv_.wait(lk, [&] {
        return stop_stream.load(std::memory_order_acquire) ||
               schedule_generation.load(std::memory_order_acquire) != generation ||
               tensor_idx_copied_ordinal.load(std::memory_order_acquire) == ordinal - 1;
    });

    if (stop_stream.load(std::memory_order_acquire))
        return;

    if (schedule_generation.load(std::memory_order_acquire) != generation)
        return;

    GGML_ASSERT(tensor_idx_copied_ordinal.load(std::memory_order_acquire) == ordinal - 1);

    tensor_idx_copied_ordinal.store(ordinal, std::memory_order_release);

    #ifdef LLAMA_LOG_COPIES
        LLAMA_LOG_INFO("[C.%d]", ordinal);
    #endif

    lk.unlock();
    node_cv_.notify_all();
}

void parameter_offloader::publish_copy_now(long long ordinal, uint64_t generation)
{
    std::unique_lock<std::mutex> lk(node_mu_);

    node_cv_.wait(lk, [&] {
        return stop_stream.load(std::memory_order_acquire) ||
            schedule_generation.load(std::memory_order_acquire) != generation ||
            tensor_idx_copied_ordinal.load(std::memory_order_acquire) == ordinal - 1;
    });

    if (stop_stream.load(std::memory_order_acquire))
        return;

    if (schedule_generation.load(std::memory_order_acquire) != generation)
        return;

    tensor_idx_copied_ordinal.store(ordinal, std::memory_order_release);

    #ifdef LLAMA_LOG_COPIES
        LLAMA_LOG_INFO("[C.%d]", ordinal);
    #endif

    lk.unlock();
    node_cv_.notify_all();
}

void parameter_offloader::stream_worker()
{
    LLAMA_LOG_INFO("%s started\n", __func__);

    long long submitted_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
    uint64_t submitted_generation = schedule_generation.load(std::memory_order_acquire);
    const int max_in_flight_copies = 8;

    for (;;)
    {
        if (stop_stream.load(std::memory_order_acquire))
            return;

        // Give a pending schedule swap priority over new copy submissions.
        if (schedule_swap_requested.load(std::memory_order_acquire))
        {
            std::unique_lock<std::mutex> lk(node_mu_);

            node_cv_.wait(lk, [&] {
                return stop_stream.load(std::memory_order_acquire) ||
                    !schedule_swap_requested.load(std::memory_order_acquire);
            });

            if (stop_stream.load(std::memory_order_acquire))
                return;

            continue;
        }

        ggml_tensor * w_cpu = nullptr;        // CPU source selected for this copy
        ggml_tensor * w_gpu = nullptr;        // GPU arena tensor selected for this copy
        long long wait_used_ordinal = -1;     // reader ordinal that blocked this copy
        uint64_t wait_generation = 0;         // schedule generation observed before blocking
        bool throttled = false;               // true when copy was blocked only by in-flight copy cap
        bool allowed = false;                 // true when selected copy is inside safe window

        {
            std::unique_lock<std::mutex> schedule_lock(schedule_mutex); // blocks schedule swap during copy selection/upload

            // A swap may have been requested after the check above but before this thread acquired schedule_mutex.
            if (schedule_swap_requested.load(std::memory_order_acquire))
                continue;

            const int tensor_count = (int)schedule_current.gpu_tensors_in_order.size(); // active schedule size
            if (tensor_count == 0)
            {
                const uint64_t empty_generation = schedule_generation.load(std::memory_order_acquire);

                // A future graph may require streaming again. Release schedule_mutex so
                // swap_next_schedule() can publish it, then sleep until the schedule changes.
                schedule_lock.unlock();

                std::unique_lock<std::mutex> lk(node_mu_);
                node_cv_.wait(lk, [&] {
                    return stop_stream.load(std::memory_order_acquire) ||
                        schedule_generation.load(std::memory_order_acquire) != empty_generation;
                });

                if (stop_stream.load(std::memory_order_acquire))
                    return;

                continue;
            }

            const uint64_t cur_generation = schedule_generation.load(std::memory_order_acquire);
            if (cur_generation != submitted_generation) {
                submitted_generation = cur_generation;
                submitted_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
            }

            //We load from the atomic variable because swap_next_schedule can change this
            const long long last_submitted_ordinal = submitted_ordinal;                                   // last submitted ordinal
            const long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);   // last reader ordinal

            const bool startup = cur_used_ordinal < 0; // true before the first graph read

            const int i = ordinal_mod(last_submitted_ordinal + 1, tensor_count); // next copy index
            const int r_idx = startup ? tensor_count - 1 : ordinal_mod(cur_used_ordinal, tensor_count); // reader or virtual startup index
            const int bar   = schedule_current.ready_after[r_idx];
            const int di    = ring_distance(r_idx, i, tensor_count);   // distance from reader/startup to copy slot
            const int dbar  = ring_distance(r_idx, bar, tensor_count); // distance from reader/startup to barrier

            GGML_ASSERT(bar >= 0);

            allowed = (di <= dbar);

            if (allowed && copy_publishers_in_flight.load(std::memory_order_acquire) >= max_in_flight_copies)
            {
            #if LLAMA_LOG_COPIES > 1
                LLAMA_LOG_INFO("[CT.%d]", i);
            #endif
                allowed = false;
                throttled = true;
            }

            if (!allowed)
            {
            #if LLAMA_LOG_COPIES > 1
                LLAMA_LOG_INFO("[CB.%d.%d.%d.%d.%d]", i, r_idx, bar, di, dbar);
            #endif
                wait_used_ordinal = cur_used_ordinal;
                wait_generation = schedule_generation.load(std::memory_order_acquire);
            }
            else
            {
                w_cpu = schedule_current.cpu_tensors_in_order[i];
                w_gpu = schedule_current.gpu_tensors_in_order[i];

                const long long ordinal = submitted_ordinal + 1;
                const uint64_t generation = submitted_generation;

                ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu, w_gpu);

                submitted_ordinal = ordinal;

                copy_publishers_in_flight.fetch_add(1, std::memory_order_acq_rel);

                if (ev) {
                    std::thread([this, ordinal, generation, ev] {
                        publish_copy_when_ready(ordinal, generation, ev);
                        copy_publishers_in_flight.fetch_sub(1, std::memory_order_acq_rel);
                        node_cv_.notify_all();
                    }).detach();
                } else {
                    std::thread([this, ordinal, generation] {
                        publish_copy_now(ordinal, generation);
                        copy_publishers_in_flight.fetch_sub(1, std::memory_order_acq_rel);
                        node_cv_.notify_all();
                    }).detach();
                }
            }
        }

        if (!allowed)
        {
            std::unique_lock<std::mutex> lk(node_mu_);
            node_cv_.wait(lk, [&]{
                return stop_stream.load(std::memory_order_acquire) ||
                    tensor_idx_used_ordinal.load(std::memory_order_acquire) != wait_used_ordinal ||
                    schedule_generation.load(std::memory_order_acquire) != wait_generation ||
                    (throttled && copy_publishers_in_flight.load(std::memory_order_acquire) < max_in_flight_copies);
            });

            if (stop_stream.load(std::memory_order_acquire))
                return;

            continue;
        }
    }
}

/////////////////////////////////////
//   SCHEDULE MANAGEMENT
/////////////////////////////////////
//return the index of the first different tensor
static inline size_t common_prefix_len(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b)
{
    const size_t n = std::min(a.size(), b.size());

    size_t i = 0;
    for (; i < n; ++i)
        if (a[i] != b[i])
            break;

    return i;
}

#ifdef LLAMA_PRINT_WEIGHT_READS
static void print_all_weight_reads(parameter_offloader * po, ggml_cgraph * graph);
#endif

// Choose how strict you want the selection to be:
//
// 0 -> Walk the whole graph in order; collect weights from *all* nodes (simplest, no internal deps)
// 1 -> Walk only the splits whose backend buffer type matches your arena buffer type (better if you
//      want to limit to the GPU backend you’re streaming into)

//#define PO_GRAPH_FILTER_BY_BACKEND 1

uint64_t parameter_offloader::analyze_dense_graph(ggml_backend_sched_t sched, const ggml_cgraph * graph, dense_graph_analysis & analysis)
{
    (void)sched;
    analysis = dense_graph_analysis{};

    // 64-bit FNV-1a is non-cryptographic; assuming uniform outputs, collision probability among n cached graphs is ~n(n-1)/(2*2^64), about 2.7e-14 at n=1000; collisions are not currently verified structurally.
    const uint64_t fnv_offset_basis = 14695981039346656037ULL;
    const uint64_t fnv_prime = 1099511628211ULL;
    uint64_t hash = fnv_offset_basis;

    // Add one 64-bit value to the dense-read FNV-1a signature.
    auto hash_value = [&](uint64_t value) {
        for (int i = 0; i < 8; ++i)
        {
            hash ^= value & 0xffULL;
            hash *= fnv_prime;
            value >>= 8;
        }
    };

    std::unordered_set<ggml_tensor *> seen;
    seen.reserve(gpu_weight_set.size());

    const size_t alignment = arena_alignment ? arena_alignment : 1;
    size_t dense_tensor_bytes = 0;

    // Record one managed dense-read node without applying any static/streaming placement decision.
    auto collect_node_weights = [&](ggml_tensor * node) {
        if (!model_i->node_may_read_dense_weight(node))
            return;

        std::vector<ggml_tensor *> graph_nodes_tensors;

        for (int k = 0; k < GGML_MAX_SRC; ++k)
        {
            ggml_tensor * w_gpu = node->src[k];

            if (!w_gpu)
                break;

            while (w_gpu->view_src)
                w_gpu = w_gpu->view_src;

            if (gpu_weight_set.find(w_gpu) == gpu_weight_set.end())
                continue;

            if (std::find(graph_nodes_tensors.begin(), graph_nodes_tensors.end(), w_gpu) != graph_nodes_tensors.end())
                continue;

            //TODO: Confirm that this maintains tensor read order
            graph_nodes_tensors.push_back(w_gpu);

            if (seen.insert(w_gpu).second)
            {
                const int idx = (int)analysis.gpu_tensors_in_order.size();
                analysis.gpu_tensors_in_order.push_back(w_gpu);
                analysis.gpu2index.emplace(w_gpu, idx);

                ggml_tensor * w_cpu = gpu2cpu.at(w_gpu);
                const size_t slot_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);

                dense_tensor_bytes += align_up(slot_bytes, alignment);
            }
        }

        if (graph_nodes_tensors.empty())
            return;

        analysis.graph_nodes.push_back(node);
        analysis.graph_nodes_tensors.push_back(graph_nodes_tensors);

        hash_value(graph_nodes_tensors.size());

        for (ggml_tensor * w_gpu : graph_nodes_tensors)
            hash_value((uint64_t)(uintptr_t)w_gpu);

        hash_value(UINT64_MAX);
    };

#if PO_GRAPH_FILTER_BY_BACKEND > 0
    // Optional precise mode: analyze only scheduler splits assigned to the arena's backend buffer type.
    const ggml_backend_buffer_type_t target_buft = ggml_backend_buffer_get_type(arena);

    for (int si = 0; si < sched->n_splits; ++si)
    {
        const ggml_backend_sched_split & sp = sched->splits[si];

        if (sched->bufts[sp.backend_id] != target_buft)
            continue;

        for (int j = sp.i_start; j < sp.i_end; ++j)
            collect_node_weights(graph->nodes[j]);
    }

    // Fall back to the full graph if no managed dense reads were found in matching splits.
    if (analysis.graph_nodes.empty())
#endif /* if PO_GRAPH_FILTER_BY_BACKEND > 0 */
    {
        for (int i = 0; i < graph->n_nodes; ++i)
            collect_node_weights(graph->nodes[i]);
    }

    hash_value(analysis.graph_nodes.size());
    hash_value(analysis.gpu_tensors_in_order.size());
    analysis.hash = hash;

    analysis.dense_fits_arena = !analysis.gpu_tensors_in_order.empty() && dense_tensor_bytes <= arena_dense_size;

    if (analysis.dense_fits_arena)
    {
        LLAMA_LOG_INFO(
            "%s: all managed dense tensors for this graph fit in the arena "
            "(required=%zu MiB, available=%zu MiB, tensors=%zu)\n",
            __func__,
            dense_tensor_bytes / (1024 * 1024),
            arena_dense_size / (1024 * 1024),
            analysis.gpu_tensors_in_order.size());
    }

    return hash;
}

void parameter_offloader::build_streaming_fit_lifetimes(
    const std::vector<ggml_tensor *> & gpu_tensors_in_order,
    const std::unordered_map<ggml_tensor *, int> & gpu2index,
    const dense_graph_analysis & analysis,
    streaming_fit_lifetime_analysis & fit_analysis) const
{
    fit_analysis = streaming_fit_lifetime_analysis{};

    const size_t tensor_count = gpu_tensors_in_order.size();

    if (tensor_count == 0)
        return;

    fit_analysis.managed_node_reads.reserve(analysis.graph_nodes_tensors.size());

    // Project the graph's managed dense reads onto the CURRENT streamed tensor set.
    for (const std::vector<ggml_tensor *> & graph_nodes_tensors : analysis.graph_nodes_tensors)
    {
        std::vector<int> streamed_tensor_indices;

        for (ggml_tensor * w_gpu : graph_nodes_tensors)
        {
            auto index_it = gpu2index.find(w_gpu);

            if (index_it != gpu2index.end())
                streamed_tensor_indices.push_back(index_it->second);
        }

        if (streamed_tensor_indices.empty())
            continue;

        std::sort(streamed_tensor_indices.begin(), streamed_tensor_indices.end());
        streamed_tensor_indices.erase(std::unique(streamed_tensor_indices.begin(), streamed_tensor_indices.end()), streamed_tensor_indices.end());
        fit_analysis.managed_node_reads.push_back(std::move(streamed_tensor_indices));
    }

    const int managed_node_count = (int)fit_analysis.managed_node_reads.size();

    if (managed_node_count == 0)
        throw std::runtime_error("parameter_offloader: streamed tensor set has no managed read positions");

    fit_analysis.tensor_bytes.resize(tensor_count);

    for (size_t i = 0; i < tensor_count; ++i)
        fit_analysis.tensor_bytes[i] = ggml_backend_buft_get_alloc_size(arena_buffer_type, gpu2cpu.at(gpu_tensors_in_order[i]));

    std::vector<int> first_read_node_idx(tensor_count, -1);
    std::vector<int> last_read_node_idx(tensor_count, -1);

    // Nodes are timestamps only. Repeated reads extend the lifetime of the same tensor; they never create another allocation.
    for (int i = 0; i < managed_node_count; ++i)
    {
        for (int tensor_idx : fit_analysis.managed_node_reads[i])
        {
            if (first_read_node_idx[tensor_idx] < 0)
                first_read_node_idx[tensor_idx] = i;

            last_read_node_idx[tensor_idx] = i;
        }
    }

    std::vector<int> reuse_after_node_idx(tensor_count);
    int latest_reuse_node_idx = -1;

    // Release remains monotonic in streaming order, exactly like tensor_idx_used_ordinal at runtime.
    for (size_t i = 0; i < tensor_count; ++i)
    {
        latest_reuse_node_idx = std::max(latest_reuse_node_idx, last_read_node_idx[i]);
        reuse_after_node_idx[i] = latest_reuse_node_idx;
    }

    std::vector<int> prefetch_node_idx(tensor_count);

    // COPY must be able to prepare a tensor during the immediately preceding streamed read position.
    for (size_t i = 0; i < tensor_count; ++i)
        prefetch_node_idx[i] = first_read_node_idx[i] == 0 ? managed_node_count - 1 : first_read_node_idx[i] - 1;

    fit_analysis.resident_tensor_indices.resize(managed_node_count);

    // Build the exact set of streamed tensors that must coexist at every streamed read position.
    for (int node_idx = 0; node_idx < managed_node_count; ++node_idx)
    {
        std::vector<int> & resident = fit_analysis.resident_tensor_indices[node_idx];

        for (size_t tensor_idx = 0; tensor_idx < tensor_count; ++tensor_idx)
        {
            const int prefetch_idx = prefetch_node_idx[tensor_idx];
            const int first_read_idx = first_read_node_idx[tensor_idx];
            const int reuse_idx = reuse_after_node_idx[tensor_idx];
            const bool is_resident = prefetch_idx < first_read_idx ? prefetch_idx <= node_idx && node_idx <= reuse_idx : node_idx >= prefetch_idx || node_idx <= reuse_idx;

            if (is_resident)
                resident.push_back((int)tensor_idx);
        }
    }
}

// CONTRACT: Calculates only streaming_fit_lower_bound, streaming_fit_upper_bound, node_pairs
//TODO: We probably want to establish virtual multi-tensor nodes prior to calling this for 100% accuracy
void parameter_offloader::streaming_fit_calculate_bounds(dense_graph_analysis & analysis)
{
    analysis.node_pairs.clear();
    std::vector<node_group> node_triples;

    const size_t alignment = arena_alignment ? arena_alignment : 1;
    const int node_total = analysis.graph_nodes.size();

    for (int i = 0; i < node_total; ++i)
    {
        std::vector<ggml_tensor *> streaming_tensors;
        for (const auto & tensor : analysis.graph_nodes_tensors[i])
            if (static_dense_set.find(tensor) == static_dense_set.end())
                streaming_tensors.push_back(tensor);

        //skip nodes that are subsets of the previous node
        bool is_subset = i != 0;
        for (int j = 0; is_subset && j < streaming_tensors.size(); ++j)
            is_subset = is_subset && std::find(analysis.graph_nodes_tensors[i - 1].begin(),
                                               analysis.graph_nodes_tensors[i - 1].end(),
                                               streaming_tensors[j]) != analysis.graph_nodes_tensors[i - 1].end();
        if (is_subset)
            continue;

        auto new_node_group = [&](int target_nodes)
        {
            node_group group {
                { analysis.graph_nodes[i] },
                streaming_tensors,
                0
            };
            int n_nodes = 1;
            
            for (int j = i == node_total - 1 ? 0 : i + 1;
                j != i;
                j == node_total - 1 ? j = 0 : ++j)
            {
                bool found_unique_tensor = false;
                for (const auto &tensor : analysis.graph_nodes_tensors[j])
                {
                    if (std::find(group.tensors.begin(), group.tensors.end(), tensor) == group.tensors.end()
                         && static_dense_set.find(tensor) == static_dense_set.end())
                    {
                        if (n_nodes < target_nodes)
                        {
                            group.tensors.push_back(tensor);
                            found_unique_tensor = true;
                        }
                        else
                            return group;
                    }
                }

                if (found_unique_tensor)
                    ++n_nodes;

                group.nodes.push_back(analysis.graph_nodes[j]);
            }

            return group;
        };

        auto count_group_bytes = [&](node_group & group)
        {
            group.bytes = 0;
            for (const auto & tensor : group.tensors)
                group.bytes += align_up(ggml_backend_buft_get_alloc_size(arena_buffer_type, gpu2cpu.at(tensor)), alignment);
        };

        node_group pair = new_node_group(2);
        node_group triple = new_node_group(3);

        count_group_bytes(pair);
        count_group_bytes(triple);

        analysis.node_pairs.push_back(std::move(pair));
        node_triples.push_back(std::move(triple));
    }
    
    auto calculate_bound = [&](std::vector<node_group> & groups)
    {
        size_t bound = 0;

        for (const auto & group : groups)
        {
            size_t ties = 0;
            for (const auto & other : groups)
                ties += other.bytes == group.bytes;

            bound = std::max(bound, group.bytes + (ties - 1) * alignment);
        }

        return bound;
    };

    streaming_fit_lower_bound = calculate_bound(analysis.node_pairs);
    streaming_fit_upper_bound = std::max(streaming_fit_lower_bound, calculate_bound(node_triples));
}

// CONTRACT: May modify only static_dense_order, static_dense_set, and static_tensor_bytes; returns whether static membership changed.
bool parameter_offloader::select_static_dense_tensors(const dense_graph_analysis & analysis)
{
    const size_t tensor_count = analysis.gpu_tensors_in_order.size();
    const size_t alignment = arena_alignment ? arena_alignment : 1;
    const size_t static_capacity = streaming_fit_upper_bound >= arena_dense_size ? 0 : arena_dense_size - streaming_fit_upper_bound;

    // If the new upper bound has moved into static storage, eject newest statics in FILO order until it fits.
    if (static_tensor_bytes > static_capacity)
    {
        size_t ejected_count = 0;

        while (!static_dense_order.empty() && static_tensor_bytes > static_capacity)
        {
            ggml_tensor * w_gpu = static_dense_order.back();
            ggml_tensor * w_cpu = gpu2cpu.at(w_gpu);
            const size_t slot_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);
            const size_t old_static_count = static_dense_order.size();

            static_dense_order.pop_back();
            static_dense_set.erase(w_gpu);
            static_tensor_bytes = old_static_count == 1 ? 0 : static_tensor_bytes - align_up(slot_bytes, alignment);
            ++ejected_count;
        }

        LLAMA_LOG_INFO("%s: ejected %zu static dense tensors; static_bytes=%zu upper_bound=%zu\n", __func__, ejected_count, static_tensor_bytes, streaming_fit_upper_bound);
        return ejected_count != 0;
    }

    if (tensor_count == 0)
        return false;

    std::vector<size_t> selected_indices;
    selected_indices.reserve(tensor_count);

    // Existing static tensors participate in spacing exactly like tensors selected during this call.
    for (ggml_tensor * w_gpu : static_dense_order)
    {
        auto it = analysis.gpu2index.find(w_gpu);

        if (it != analysis.gpu2index.end())
            selected_indices.push_back((size_t)it->second);
    }

    const size_t original_static_count = static_dense_order.size();

    // Greedily choose fitting non-deprioritized tensors first, then largest tensors, then maximize cyclic distance from tensors already selected for static storage.
    while (selected_indices.size() < tensor_count)
    {
        size_t best_idx = SIZE_MAX;
        size_t best_spacing = 0;
        size_t best_bytes = 0;
        size_t best_footprint = 0;
        bool best_deprioritized = false;

        for (size_t i = 0; i < tensor_count; ++i)
        {
            ggml_tensor * w_gpu = analysis.gpu_tensors_in_order[i];

            if (static_dense_set.find(w_gpu) != static_dense_set.end())
                continue;

            ggml_tensor * w_cpu = gpu2cpu.at(w_gpu);
            const size_t bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);
            const size_t static_cursor = arena_dense_size - static_tensor_bytes;

            if (bytes > static_cursor)
                continue;

            const size_t off = align_down(static_cursor - bytes, alignment);
            const size_t footprint = static_cursor - off;

            if (static_tensor_bytes + footprint > static_capacity)
                continue;

            size_t spacing = tensor_count;

            for (size_t selected_idx : selected_indices)
            {
                const size_t linear_distance = i > selected_idx ? i - selected_idx : selected_idx - i;
                const size_t cyclic_distance = std::min(linear_distance, tensor_count - linear_distance);
                spacing = std::min(spacing, cyclic_distance);
            }

            const bool deprioritized = deprioritized_dense_set.find(w_gpu) != deprioritized_dense_set.end();

            if (best_idx == SIZE_MAX || (best_deprioritized && !deprioritized) || (best_deprioritized == deprioritized && (bytes > best_bytes || (bytes == best_bytes && spacing > best_spacing))))
            {
                best_idx = i;
                best_spacing = spacing;
                best_bytes = bytes;
                best_footprint = footprint;
                best_deprioritized = deprioritized;
            }
        }

        if (best_idx == SIZE_MAX)
            break;

        ggml_tensor * w_gpu = analysis.gpu_tensors_in_order[best_idx];
        static_dense_order.push_back(w_gpu);
        static_dense_set.insert(w_gpu);
        static_tensor_bytes += best_footprint;
        selected_indices.push_back(best_idx);
    }

    const size_t selected_count = static_dense_order.size() - original_static_count;

    if (selected_count == 0)
        return false;

    LLAMA_LOG_INFO("%s: selected %zu static dense tensors; static_bytes=%zu lower_bound=%zu upper_bound=%zu\n", __func__, selected_count, static_tensor_bytes, streaming_fit_lower_bound, streaming_fit_upper_bound);
    return true;
}

void parameter_offloader::build_graph_runtime_metadata(dense_graph_analysis & analysis, const offloader_schedule & schedule)
{
    const size_t tensor_count = schedule.gpu_tensors_in_order.size();

    analysis.release_node_by_tensor.assign(tensor_count, nullptr);
    analysis.next_required_tensor_idx.clear();

    if (tensor_count == 0)
        return;

    std::vector<int> read_last_position(tensor_count, -1);
    std::vector<uint8_t> seen(tensor_count, 0);
    ggml_tensor * previous_read_node = nullptr;
    int first_read_idx = -1;

    // Materialize the current graph's release/copy synchronization metadata from the one graph analysis pass.
    for (size_t read_pos = 0; read_pos < analysis.graph_nodes_tensors.size(); ++read_pos)
    {
        ggml_tensor * node = analysis.graph_nodes[read_pos];
        bool node_reads_streamed_weight = false;
        int node_first_read_idx = -1;

        for (ggml_tensor * w_gpu : analysis.graph_nodes_tensors[read_pos])
        {
            auto it = schedule.gpu2index.find(w_gpu);

            if (it == schedule.gpu2index.end())
                continue;

            const int idx = it->second;
            node_reads_streamed_weight = true;

            if (!seen[(size_t)idx])
            {
                seen[(size_t)idx] = 1;
                node_first_read_idx = idx;
            }

            analysis.release_node_by_tensor[(size_t)idx] = node;
            read_last_position[(size_t)idx] = (int)read_pos;
        }

        if (!node_reads_streamed_weight)
            continue;

        if (first_read_idx < 0 && node_first_read_idx >= 0)
            first_read_idx = node_first_read_idx;

        if (previous_read_node && node_first_read_idx >= 0)
            analysis.next_required_tensor_idx[previous_read_node] = node_first_read_idx;

        previous_read_node = node;
    }

    if (previous_read_node && first_read_idx >= 0)
        analysis.next_required_tensor_idx[previous_read_node] = first_read_idx;

    int release_position = -1;
    ggml_tensor * release_node = nullptr;

    // Release remains monotonic in streaming order so tensor_idx_used_ordinal can advance monotonically at runtime.
    for (size_t i = 0; i < tensor_count; ++i)
    {
        if (read_last_position[i] > release_position)
        {
            release_position = read_last_position[i];
            release_node = analysis.release_node_by_tensor[i];
        }

        analysis.release_node_by_tensor[i] = release_node;
    }
}

void parameter_offloader::build_next_schedule(offloader_schedule & schedule, dense_graph_analysis & analysis)
{
    schedule = offloader_schedule{};

    // Materialize only the finalized streamed tensor set; graph read timing remains in dense_graph_analysis.
    for (ggml_tensor * w_gpu : analysis.gpu_tensors_in_order)
    {
        if (static_dense_set.find(w_gpu) != static_dense_set.end())
            continue;

        const int idx = (int)schedule.gpu_tensors_in_order.size();
        schedule.gpu_tensors_in_order.push_back(w_gpu);
        schedule.cpu_tensors_in_order.push_back(gpu2cpu.at(w_gpu));
        schedule.gpu2index.emplace(w_gpu, idx);
    }
}

// Generate a compact no-halt streaming arena size and a valid fixed offset for every streamed tensor.
// Graph nodes are used only to derive tensor read lifetimes; placement itself is entirely tensor based.

/*
 *   generate_streaming_fit generates and caches the smallest required size for the streaming area that allows
 * tensor prefetching without halting.
 * 
 * unique pointer addresses -   Each tensor will be given a unique address, to avoid
 *                            any unforseen conflicts that may depend on pointer uninqueness.
 *
 * lower-bound -   The lower bound of the streaming area size is equal to the largest requirement
 *               across TWO consecutive streamed read positions, plus any unavoidable start-offset displacement.
 *
 * upper-bound -   The upper bound of the required area is equal to the largest requirement across
 *               THREE consecutive streamed read positions, plus any unavoidable start-offset displacement.
 *
 * fragmentation problem -   Consider the following tensors in a size 10 arena:
 *                         A [0,6)             B [6,10)
 *                         C [0,2)   D [2,4)   Z [4,8)
 *
 *                           A would be prefetched Z is being read, but this configuration would cause a COPY HALT
 *                         as Z and A overlap the same area in memory. Z must be released before A can be copied,
 *                         and because copy is usually the bottleneck this exacerbates said bottleneck. Moving
 *                         Z to [6, 10) solves this issue, but such an easy solution may not always be easy to find.
 */

//this version solves unique addresses last
size_t parameter_offloader::generate_streaming_fit(offloader_schedule & schedule, const dense_graph_analysis & analysis)
{
    const size_t alignment = arena_alignment ? arena_alignment : 1;
    size_t pos = 0;

    const int node_groups_total = analysis.node_pairs.size();

    //Identify priority node groups, they are the ones tied for largest size
    size_t largest_size = 0;
    for (int i = 0; i < node_groups_total; ++i)
        largest_size = std::max(largest_size, analysis.node_pairs[i].bytes);

    //size_t fit_size = analysis.lower_bound;
    size_t fit_size = largest_size;         //dont use lower_bound, we're going to nudge fit_size up in the final step of fitting
    
    /*
    * Fit recursively by valleys: anchor the largest constrained node-pair near the arena boundaries,
    * then immediately solve the interval between placed constraints. Within each valley, place the
    * largest remaining constraint, solve the smaller sub-valley first, and use available slack to move
    * the new wall outward on failure. Try the alternate boundary configuration before rejecting the
    * candidate arena size; increasing the size moves right-anchored constraints and opens fresh starts.
    */

    std::map<ggml_tensor *, std::pair<size_t, int>> tensor_offsets;     //key: tensor, value: size, cycle
    std::vector<ggml_tensor *> placement_log;

    //return false if it fails to fit a node
    auto fit_next_largest_node = [&](auto&& self, const std::vector<node_group> & groups, size_t left_bound, size_t right_bound) -> size_t
    {
        if (groups.empty())
            return 0;

        size_t smallest_conflict = 0;

        //Get the largest node_pair, leftmost breaks ties
        std::vector<node_group>::const_iterator target = std::max_element(
            groups.begin(), groups.end(),
            [](const node_group& a, const node_group& b) { return a.bytes < b.bytes; });

        bool is_largest = target->bytes == largest_size;

        std::vector<node_group> left(groups.begin(), target);
        std::vector<node_group> right(target + 1, groups.end());

        size_t left_bytes = 0, right_bytes = 0;
        std::set<ggml_tensor *> seen;

        //Naively walk the superset of all tensors in the left and right nodes
        pos = left_bound;
        for (const node_group & group : left)
            for (ggml_tensor * tensor : group.tensors)
                if (tensor_offsets.count(tensor) == 0 && seen.insert(tensor).second)
                {
                    size_t bytes = get_gpu_aligned_size(tensor, alignment);
                    if (pos + bytes > fit_size)
                    {
                        left_bytes += fit_size - pos;
                        pos = 0;
                    }
                    pos += bytes;
                    left_bytes += bytes;
                }

        seen.clear();
        pos = right_bound;
        for (auto group = right.rbegin(); group != right.rend(); ++group)
            for (auto tensor = group->tensors.rbegin(); tensor != group->tensors.rend(); ++tensor)
                if (tensor_offsets.count(*tensor) == 0 && seen.insert(*tensor).second)
                {
                    size_t bytes = get_gpu_aligned_size(*tensor, alignment);
                    if (pos < bytes)
                    {
                        right_bytes += pos;
                        pos = fit_size;
                    }
                    pos -= bytes;
                    right_bytes += bytes;
                }
        //TODO: bytes isn't necessarily a measure of complexity, we probably want to count nodes or tensors
        const bool left_first = left_bytes <= right_bytes;
        //const bool left_first = left.count() <= right.count();

        //some of this node-pair's tensors might already be placed, only place unplaced tensors
        std::vector<ggml_tensor *> unplaced_tensors, placed_tensors;
        for (ggml_tensor * tensor : target->tensors)
            tensor_offsets.count(tensor) == 0 ?
                unplaced_tensors.push_back(tensor)
                : placed_tensors.push_back(tensor);

        //I thought I needed this but now im not sure
        //size_t unplaced_bytes = 0;
        //for (ggml_tensor * tensor : unplaced_tensors)
        //    unplaced_bytes += get_gpu_aligned_size(tensor, alignment);

        //largest node pairs only have a few different valid places they can fit
        std::vector<size_t> lnp_offsets;
        if (is_largest)
        {
            pos = 0;
            if (placed_tensors.empty())
            {
                for (auto it = target->tensors.rbegin(); it != target->tensors.rend(); ++it)
                {
                    lnp_offsets.push_back(pos);
                    pos += get_gpu_aligned_size(*it, alignment);
                }

                pos = left_bound + left_bytes;
                while (pos > fit_size)
                    pos -= fit_size;

                //rotate it so we try the tightest fit first
                std::rotate(lnp_offsets.begin(), std::lower_bound(lnp_offsets.begin(), lnp_offsets.end(), pos), lnp_offsets.end());
            }
            else   //the end of the right-most already placed tensor
                lnp_offsets.push_back(tensor_offsets[placed_tensors.back()].first + get_gpu_aligned_size(placed_tensors.back(), alignment));
        }
        else
            lnp_offsets.push_back(0);   //TODO: just put a dummy value here for now, but later we may want to explore alternative configurations for non-largest node pairs

        std::map<ggml_tensor *, size_t> tensor_offsets_temp;

        for (size_t candidate_offset : lnp_offsets)
        {
            auto place_tensor_against_left = [&](size_t left_bound, ggml_tensor * tensor)
            {
                size_t tensor_aligned_size = get_gpu_aligned_size(tensor, alignment);
                if (left_bound + tensor_aligned_size > fit_size)
                    left_bound = 0;    //need to wrap around to the beginning
                tensor_offsets_temp[tensor] = left_bound;
                return left_bound + tensor_aligned_size;
            };
            auto place_tensor_against_right = [&](size_t right_bound, ggml_tensor * tensor)
            {
                size_t tensor_aligned_size = get_gpu_aligned_size(tensor, alignment);
                if (right_bound < tensor_aligned_size)
                    right_bound = fit_size - tensor_aligned_size;    //need to wrap around to the end
                else
                    right_bound -= tensor_aligned_size;
                tensor_offsets_temp[tensor] = right_bound;
                return right_bound;
            };

            if (unplaced_tensors.empty())
            {
                //do nothing
            }
            else if (!placed_tensors.empty())
            {
                if (placed_tensors[0] == target->tensors[0]) //leftmost tensor has been placed
                {
                    pos = left_bound;
                    for (ggml_tensor * tensor : unplaced_tensors)
                        pos = place_tensor_against_left(pos, tensor);
                }
                else if (placed_tensors.back() == target->tensors.back()) //rightmost tensor has been placed
                {
                    pos = right_bound;
                    for (auto it = unplaced_tensors.rbegin(); it != unplaced_tensors.rend(); ++it)
                        pos = place_tensor_against_right(pos, *it);
                }
                else
                {
                    size_t anchor = std::find(target->tensors.begin(), target->tensors.end(), placed_tensors[0]) - target->tensors.begin();
                    pos = tensor_offsets.at(target->tensors[anchor]).first + get_gpu_aligned_size(target->tensors[anchor], alignment);

                    for (size_t i = 1; i < target->tensors.size(); ++i)
                    {
                        ggml_tensor * tensor = target->tensors[(anchor + i) % target->tensors.size()];
                        if (tensor_offsets.count(tensor))
                            pos = tensor_offsets.at(tensor).first + get_gpu_aligned_size(tensor, alignment);
                        else
                            pos = place_tensor_against_left(pos, tensor);
                    }
                }          
            }
            else if (is_largest)
            {
                pos = candidate_offset;
                for (ggml_tensor * tensor : unplaced_tensors)
                        pos = place_tensor_against_left(pos, tensor);
            }
            else if (left_first) //fit against left side
            {
                pos = left_bound + left_bytes;
                while (pos > fit_size)
                    pos -= fit_size;

                for (ggml_tensor * tensor : unplaced_tensors)
                    pos = place_tensor_against_left(pos, tensor);
            }
            else    //fit against right side
            {
                //Can't do this, it underflows
                //pos = right_bound - right_bytes;
                //while (right_bound < right_bytes)
                //    pos += fit_size;
                pos = (right_bound + fit_size - right_bytes % fit_size) % fit_size;

                for (auto it = unplaced_tensors.rbegin(); it != unplaced_tensors.rend(); ++it)
                    pos = place_tensor_against_right(pos, *it);
            }

            //tensor_offsets.insert(tensor_offsets_temp.begin(), tensor_offsets_temp.end());
            size_t rollback_pos = placement_log.size();
            for (const auto & entry : tensor_offsets_temp)
                if (tensor_offsets.insert({ entry.first, { entry.second, -1 } }).second)
                    placement_log.push_back(entry.first);

            //All tensors are now placed, check that no COPY HALT occurs. This is easy, just check that no tensor in the node-pair conflicts with any other
            size_t conflict_size = 0;
            for (int i = 0; i < target->tensors.size(); ++i)
                for (int j = i + 1; j < target->tensors.size(); ++j)
                {
                    ggml_tensor * a = target->tensors[i], * b = target->tensors[j];
                    size_t a1 = tensor_offsets.at(a).first, a2 = a1 + get_gpu_aligned_size(a, alignment);
                    size_t b1 = tensor_offsets.at(b).first, b2 = b1 + get_gpu_aligned_size(b, alignment);
                    if (ranges_overlap(a1, a2, b1, b2))
                        conflict_size = std::max(conflict_size, std::min(a2, b2) - std::max(a1, b1));
                }

            //if (unplaced_tensors.empty())
            //    return conflict_size;

            auto handle_failure = [&](size_t error_amount)
            {
                while (placement_log.size() > rollback_pos)
                {
                    tensor_offsets.erase(placement_log.back());
                    placement_log.pop_back();
                }
                smallest_conflict = smallest_conflict == 0 ? error_amount : std::min(smallest_conflict, error_amount);
            };

            if (conflict_size)
            {
                handle_failure(conflict_size);
                continue;
            }

            size_t target_left  = tensor_offsets.at(target->tensors.front()).first;
            size_t target_right = tensor_offsets.at(target->tensors.back()).first + get_gpu_aligned_size(target->tensors.back(), alignment);

            //recursively solve the rest of the fit
            size_t extra_bytes_needed = 0;
            if (left_first)
            {
                extra_bytes_needed = self(self, left, left_bound, target_left);
                if (extra_bytes_needed)
                {
                    handle_failure(extra_bytes_needed);
                    continue;
                }
                extra_bytes_needed = self(self, right, target_right, right_bound);
                if (extra_bytes_needed)
                {
                    handle_failure(extra_bytes_needed);
                    continue;
                }
                return 0;
            }
            else
            {
                extra_bytes_needed = self(self, right, target_right, right_bound);
                if (extra_bytes_needed)
                {
                    handle_failure(extra_bytes_needed);
                    continue;
                }
                extra_bytes_needed = self(self, left, left_bound, target_left);
                if (extra_bytes_needed)
                {
                    handle_failure(extra_bytes_needed);
                    continue;
                }
                return 0;
            }
        }

        return smallest_conflict;
    };

    size_t bytes_needed;
    while ((bytes_needed = fit_next_largest_node(fit_next_largest_node, analysis.node_pairs, 0, fit_size)) > 0)
    {
        if (fit_size == streaming_fit_upper_bound)
            throw std::runtime_error("parameter_offloader:: failed to find fit within upper bound");

        tensor_offsets.clear();
        
        fit_size += bytes_needed;
        if (fit_size > streaming_fit_upper_bound)
            fit_size = streaming_fit_upper_bound;
    }

    //Dedup tensor addresses

    std::vector<std::vector<ggml_tensor *>> cycles_x_tensors_in_order;
    std::multimap<size_t, ggml_tensor *> offset_x_tensor_ascending;     //key: offset, value: tensor, order: lowest first
    std::map<ggml_tensor *, std::multimap<size_t, ggml_tensor *>::iterator> tensor_x_iterator;  //for fast updating of offset_x_tensor_ascending

    std::map<ggml_tensor *, std::vector<int>> tensor_x_node_pairs;      //lookup node-pairs by tensor
    for (int i = 0; i < analysis.node_pairs.size(); ++i)
        for (ggml_tensor * tensor : analysis.node_pairs[i].tensors)
            tensor_x_node_pairs[tensor].push_back(i);
    for (auto it = tensor_x_node_pairs.begin(); it != tensor_x_node_pairs.end(); ++it)
        std::stable_sort(it->second.begin(), it->second.end(), [&](int a, int b) {
            return analysis.node_pairs[a].bytes > analysis.node_pairs[b].bytes;
        });
        
    int cycle = 0;
    size_t prev_offset = 0;
    for (ggml_tensor * tensor : analysis.gpu_tensors_in_order)
    {
        std::pair<size_t, int> & offset_info = tensor_offsets.at(tensor);
        size_t offset = offset_info.first;

        if (!cycles_x_tensors_in_order.empty() && offset <= prev_offset)
            ++cycle;

        if (cycles_x_tensors_in_order.size() <= (size_t)cycle)
            cycles_x_tensors_in_order.emplace_back();

        offset_info.second = cycle;
        cycles_x_tensors_in_order[cycle].push_back(tensor);

        prev_offset = offset;

        tensor_x_iterator[tensor] = offset_x_tensor_ascending.insert({ tensor_offsets.at(tensor).first, tensor });
    }

    //Search addresses starting at 0 for duplicates, if duplicate is found nudge the smallest tensors up. The largest (leftmost wins ties) tensor remains still
    while (!offset_x_tensor_ascending.empty())
    {
        std::multimap<size_t, ggml_tensor *>::iterator first = offset_x_tensor_ascending.begin();
        std::multimap<size_t, ggml_tensor *>::iterator next = first;

        std::vector<ggml_tensor *> dup_tensors = { first->second }; //first tensor
        while (++next != offset_x_tensor_ascending.end() && first->first == next->first)
            dup_tensors.push_back(next->second);

        if (dup_tensors.size() > 1)
        {
            std::sort(dup_tensors.begin(), dup_tensors.end(), [&](ggml_tensor * a, ggml_tensor * b)
            {
                int ap = tensor_x_node_pairs[a][0], bp = tensor_x_node_pairs[b][0];
                return analysis.node_pairs[ap].bytes != analysis.node_pairs[bp].bytes ?
                    analysis.node_pairs[ap].bytes > analysis.node_pairs[bp].bytes : ap < bp;
            });

            for (int dup_idx = 1; dup_idx < dup_tensors.size(); ++dup_idx)
            {
                auto nudge_tensor = [&](auto&& recurse_tensor, ggml_tensor * tensor, size_t nudge) -> void
                {
                    //First nudge the tensor's own cycle

                    std::vector<ggml_tensor *> nudged_tensors;
                    int cycle = tensor_offsets[tensor].second;
                    int j = 0;
                    while (cycles_x_tensors_in_order[cycle][j] != tensor)
                        ++j;
                    for (;j < cycles_x_tensors_in_order[cycle].size(); ++j)
                    {
                        ggml_tensor * target = cycles_x_tensors_in_order[cycle][j];
                        size_t target_nudge; 
                        if (target == tensor)
                            target_nudge = nudge;
                        else
                        {
                            ggml_tensor * prev_target = cycles_x_tensors_in_order[cycle][j - 1];
                            size_t prev_end = tensor_offsets[prev_target].first + get_gpu_aligned_size(prev_target, alignment);
                            target_nudge = prev_end > tensor_offsets[target].first ? prev_end - tensor_offsets[target].first : 0;
                        }

                        if (target_nudge == 0)
                            break;

                        nudged_tensors.push_back(target);
                        tensor_offsets[target].first += target_nudge;
                        offset_x_tensor_ascending.erase(tensor_x_iterator[target]);
                        tensor_x_iterator[target] = offset_x_tensor_ascending.insert({ tensor_offsets[target].first, target });
                    }

                    //Then see if any tensors that share a node-pair with that tensor also need a nudge
                    auto nudge_node_pair_tensors = [&](auto&& self, ggml_tensor * root)
                    {
                        for (int node_idx : tensor_x_node_pairs[root])
                            for (ggml_tensor * tensor : analysis.node_pairs[node_idx].tensors)
                                if (tensor != root)
                                {
                                    //its important to do the full ranges_overlap check here
                                    size_t tensor_start = tensor_offsets.at(tensor).first, tensor_end = tensor_start + get_gpu_aligned_size(tensor, alignment);
                                    size_t prev_start = tensor_offsets.at(root).first, prev_end = prev_start + get_gpu_aligned_size(root, alignment);
                                    if (ranges_overlap(tensor_start, tensor_end, prev_start, prev_end))
                                    {
                                        //The only way it can overlap at this point is if root encroached upon tensor from the left, so nudge it
                                        recurse_tensor(recurse_tensor, tensor, align_up(prev_end - tensor_start, alignment));
                                    }
                                }
                    };

                    for (int j = 0; j < nudged_tensors.size(); ++j)
                        nudge_node_pair_tensors(nudge_node_pair_tensors, nudged_tensors[j]);
                };

                nudge_tensor(nudge_tensor, dup_tensors[dup_idx], alignment * dup_idx);
            }
        }

        //drop lowest unique address from the search, it wont be moved later
        tensor_x_iterator.erase(offset_x_tensor_ascending.begin()->second);
        offset_x_tensor_ascending.erase(offset_x_tensor_ascending.begin());
    }

    //recompute fit from final geometry
    for (const std::vector<ggml_tensor *> & tensors : cycles_x_tensors_in_order)
    {
        ggml_tensor * tensor = tensors.back();
        fit_size = std::max(fit_size,
            tensor_offsets[tensor].first + get_gpu_aligned_size(tensor, alignment));
    }

    schedule.start_offset.resize(analysis.gpu_tensors_in_order.size());
    schedule.end_offset.resize(analysis.gpu_tensors_in_order.size());

    for (size_t i = 0; i < schedule.gpu_tensors_in_order.size(); ++i)
    {
        ggml_tensor * tensor = schedule.gpu_tensors_in_order[i];
        schedule.start_offset[i] = tensor_offsets.at(tensor).first;
        schedule.end_offset[i] = schedule.start_offset[i] + ggml_backend_buft_get_alloc_size(arena_buffer_type, gpu2cpu.at(tensor));
    }

    return fit_size;
}

bool parameter_offloader::swap_next_schedule(size_t streaming_fit)
{
    // Candidate schedule has been collected; compare it once against the active schedule.
    const size_t common_prefix_len_ = common_prefix_len(schedule_current.gpu_tensors_in_order, schedule_next.gpu_tensors_in_order);
    bool schedule_next_identical =
        common_prefix_len_ == schedule_current.gpu_tensors_in_order.size() &&
        common_prefix_len_ == schedule_next.gpu_tensors_in_order.size() &&
        schedule_current.start_offset == schedule_next.start_offset;

    const bool graph_identical = graph_analysis_current.hash == graph_analysis_next.hash;
    const bool streaming_fit_identical = arena_stream_size == streaming_fit;

    long long new_copied_ordinal = -1; // copied ordinal required before graph may start reading
    int startup_copy_idx = 0; // furthest first-read weight required by the first managed node
    bool changed = false; // true when a new schedule was published

    if (graph_identical && schedule_next_identical && streaming_fit_identical)
    {
        // TODO: If a future model's first node requires the second scheduled weight, there is an unresolved startup-read bug here.
        // Fix by ensuring the new schedule's first-node copy requirement is satisfied before returning from the schedule_next_identical path.
        std::swap(graph_analysis_current, graph_analysis_next);
        graph_analysis_next = dense_graph_analysis{};
        schedule_next = offloader_schedule{};

        return false;
    }

    // Stop the streamer from beginning any more old-schedule uploads.
    schedule_swap_requested.store(true, std::memory_order_release);
    node_cv_.notify_all();

    {
        std::lock_guard<std::mutex> schedule_lock(schedule_mutex); // blocks streamer while tensor pointers move
        
        //wait for in-flight copies to complete
        {
            std::unique_lock<std::mutex> lk(node_mu_);
            node_cv_.wait(lk, [&] {
                return copy_publishers_in_flight.load(std::memory_order_acquire) == 0;
            });
        }

        std::unordered_map<ggml_tensor *, size_t> old_offsets; // old arena offsets before retarget
        old_offsets.reserve(schedule_current.gpu_tensors_in_order.size());

        for (ggml_tensor * w_gpu : schedule_current.gpu_tensors_in_order)
        {
            GGML_ASSERT(w_gpu);
            GGML_ASSERT(w_gpu->data);
            old_offsets[w_gpu] = (size_t)((char *)w_gpu->data - arena_base);
        }

        if (streaming_fit > arena_dense_size)
            throw std::runtime_error("parameter_offloader: streaming fit exceeds available dense arena");

        arena_stream_size = streaming_fit;
        seat_dense_tensors(schedule_next);

        const size_t prefix_limit = std::min(schedule_current.gpu_tensors_in_order.size(), schedule_next.gpu_tensors_in_order.size()); // max prefix to compare
        // prefix that kept same tensor identity and same arena offset
        size_t schedule_reusable_count = 0;
        for (schedule_reusable_count = 0; schedule_reusable_count < prefix_limit; ++schedule_reusable_count)
        {
            ggml_tensor * old_gpu = schedule_current.gpu_tensors_in_order[schedule_reusable_count]; // tensor in old schedule prefix
            ggml_tensor * new_gpu = schedule_next.gpu_tensors_in_order[schedule_reusable_count]; // tensor in new schedule prefix

            if (old_gpu != new_gpu)
                break;

            auto it = old_offsets.find(new_gpu); // old offset for this tensor before retarget

            if (it == old_offsets.end())
                break;

            const size_t new_offset = (size_t)((char *)new_gpu->data - arena_base); // new offset after retarget

            if (it->second != new_offset)
                break;
        }

        //TODO: shouldn't old_used_ordinal always be -1, given that we only call swap after the full schedule is read?
        const long long old_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire); // last copied ordinal in old schedule
        const long long old_used_ordinal   = tensor_idx_used_ordinal.load(std::memory_order_acquire); // last used ordinal in old schedule

        long long copied_ahead = (old_used_ordinal < 0) ? old_copied_ordinal + 1 : old_copied_ordinal - old_used_ordinal; // copied entries available ahead of reader

        if (copied_ahead < 0)
            copied_ahead = 0;

        const long long reusable_copied = std::min((long long)schedule_reusable_count, copied_ahead); // preserved copied prefix count
        new_copied_ordinal = reusable_copied - 1; // copied ordinal after schedule swap

        std::swap(schedule_current, schedule_next);
        std::swap(graph_analysis_current, graph_analysis_next);

        if (!graph_analysis_current.release_node_by_tensor.empty())
        {
            auto it = graph_analysis_current.next_required_tensor_idx.find(graph_analysis_current.release_node_by_tensor.back());
            if (it != graph_analysis_current.next_required_tensor_idx.end())
                startup_copy_idx = it->second;
        }

        schedule_generation.fetch_add(1, std::memory_order_relaxed) + 1;

        schedule_next = offloader_schedule{};
        graph_analysis_next = dense_graph_analysis{};

        tensor_idx_copied_ordinal.store(new_copied_ordinal, std::memory_order_release);
        tensor_idx_used_ordinal.store(-1, std::memory_order_release);

    #if defined(LLAMA_DIAGNOSE_COPY)
        const size_t tensor_count = schedule_current.gpu_tensors_in_order.size();

        if (tensor_count > 0)
        {
            long long copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_relaxed);

            for (;;)
            {
                // snapshot the current reader ordinal
                long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);
                const int i = ordinal_mod(copied_ordinal + 1, tensor_count);

                {
                    const int r_idx = cur_used_ordinal < 0 ? tensor_count - 1 : ordinal_mod(cur_used_ordinal, tensor_count); // current reader index
                    const int bar   = schedule_current.ready_after[r_idx];         // last copyable index while r is read
                    const int di    = ring_distance(r_idx, i, tensor_count);       // distance from reader to copy slot
                    const int dbar  = ring_distance(r_idx, bar, tensor_count);     // distance from reader to barrier

                    bool allowed = (di <= dbar) && (di <= LLAMA_DIAGNOSE_COPY || copied_ordinal < startup_copy_idx);

                    if (!allowed)
                        break;
                }

                // Safe to copy slot i
                ggml_tensor *w_cpu_ = schedule_current.cpu_tensors_in_order[i];
                ggml_tensor *w_gpu_ = schedule_current.gpu_tensors_in_order[i];

                ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu_, w_gpu_);
                if (ev) {
                    ggml_cuda_copy_event_wait(ev);
                    ggml_cuda_copy_event_destroy(ev);
                }

            #ifdef LLAMA_LOG_COPIES
                LLAMA_LOG_INFO("[C.%d]", i);
            #endif

                ++copied_ordinal;
                tensor_idx_copied_ordinal.store(copied_ordinal, std::memory_order_release);
            }

            const long long startup_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
            GGML_ASSERT(startup_copied_ordinal >= startup_copy_idx);
        }
    #endif /* defined(LLAMA_DIAGNOSE_COPY) */

        changed = true;

        print_snapshot(schedule_current);
    }

    // The new schedule is now completely published and schedule_mutex is free.
    schedule_swap_requested.store(false, std::memory_order_release);
    node_cv_.notify_all();

    //TODO: This must block if we don't have the first tensor copied into the new schedule. If the first schedule changed between schedules, this should always fire
#if !defined(LLAMA_DIAGNOSE_COPY)
    if (!schedule_current.gpu_tensors_in_order.empty() && new_copied_ordinal < startup_copy_idx)
    {
    #if LLAMA_LOG_READS > 1
        const long long cur_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire); // copied ordinal after publishing schedule
        LLAMA_LOG_INFO("[RB.%d.%lld.%lld.%lld]", startup_copy_idx, cur_copied_ordinal, (long long)startup_copy_idx, (long long)-1);
    #endif

        std::unique_lock<std::mutex> lk(node_mu_);
        node_cv_.wait(lk, [&]{
            return stop_stream.load(std::memory_order_acquire) || tensor_idx_copied_ordinal.load(std::memory_order_acquire) >= startup_copy_idx;
        });
    }
#endif

    return changed;
}

void parameter_offloader::seat_dense_tensors(offloader_schedule & schedule)
{
    const size_t tensor_count = schedule.gpu_tensors_in_order.size(); // number of streamed GPU twins to retarget
    const size_t a = arena_alignment ? arena_alignment : 1; // alignment used for arena start offsets
    const size_t reusable_static_count = common_prefix_len(static_dense_order_current, static_dense_order); // unchanged top-down static prefix already resident at the same offsets

    // Seat every streamed tensor at the exact offset stored in the solved schedule.
    if (tensor_count > 0)
    {
        if (schedule.start_offset.size() != tensor_count || schedule.end_offset.size() != tensor_count)
            throw std::runtime_error("parameter_offloader: solved schedule is missing streamed tensor offsets");

        for (size_t i = 0; i < tensor_count; ++i)
        {
            ggml_tensor * w_gpu = schedule.gpu_tensors_in_order[i]; // existing GPU twin whose data pointer will move
            ggml_tensor * w_cpu = schedule.cpu_tensors_in_order[i]; // CPU weight used only to compute padded device size

            GGML_ASSERT(w_gpu);
            GGML_ASSERT(w_cpu);
            GGML_ASSERT(w_gpu->buffer == arena);

            const size_t slot_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu); // padded device bytes for this tensor
            const size_t off = schedule.start_offset[i]; // exact arena-relative start offset generated by generate_streaming_fit()

            if (schedule.end_offset[i] != off + slot_bytes || schedule.end_offset[i] > arena_stream_size)
                throw std::runtime_error("parameter_offloader: solved streamed tensor offset is invalid");

            w_gpu->data = arena_base + off;
            ggml_backend_buffer_init_tensor(arena, w_gpu); // refresh backend tensor metadata after moving the arena pointer
        }
    }

    // Pack static dense tensors downward from the top of the dense arena and upload only the changed suffix.
    size_t static_cursor = arena_dense_size;

    for (size_t i = 0; i < static_dense_order.size(); ++i)
    {
        ggml_tensor * w_gpu = static_dense_order[i];
        ggml_tensor * w_cpu = gpu2cpu.at(w_gpu);
        const size_t slot_bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, w_cpu);

        if (slot_bytes > static_cursor)
            throw std::runtime_error("parameter_offloader: static dense tensor does not fit inside dense arena");

        const size_t off = align_down(static_cursor - slot_bytes, a);

        if (off < arena_stream_size)
            throw std::runtime_error("parameter_offloader: streaming fit overlaps static dense storage");

        w_gpu->data = arena_base + off;
        ggml_backend_buffer_init_tensor(arena, w_gpu);

        if (i >= reusable_static_count)
        {
            ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu, w_gpu);
            if (ev)
            {
                ggml_cuda_copy_event_wait(ev);
                ggml_cuda_copy_event_destroy(ev);
            }
        }

        static_cursor = off;
    }

    static_tensor_bytes = arena_dense_size - static_cursor;
    static_dense_order_current = static_dense_order;
}

void parameter_offloader::build_schedule_gates(offloader_schedule & schedule)
{
    const size_t tensor_count = schedule.gpu_tensors_in_order.size();

    GGML_ASSERT(schedule.cpu_tensors_in_order.size() == tensor_count);
    GGML_ASSERT(schedule.gpu2index.size() == tensor_count);

    schedule.ready_after.assign(tensor_count, -1); // post-read maximum copy barrier

    if (tensor_count == 0)
        return;

    const int N = (int)tensor_count;
    const bool placement_already_solved = schedule.start_offset.size() == tensor_count && schedule.end_offset.size() == tensor_count;

    if (!placement_already_solved)
    {
        schedule.start_offset.assign(tensor_count, 0);
        schedule.end_offset.assign(tensor_count, 0);

        // Initialization still derives placement from the tensors' current arena pointers.
        for (int i = 0; i < N; ++i)
        {
            ggml_tensor * t_gpu = schedule.gpu_tensors_in_order[i];
            ggml_tensor * t_cpu = schedule.cpu_tensors_in_order[i];

            GGML_ASSERT(t_gpu && t_gpu->data);
            GGML_ASSERT(t_cpu);

            const size_t off = (size_t)((char *)t_gpu->data - arena_base);
            const size_t bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, t_cpu);

            schedule.start_offset[i] = off;
            schedule.end_offset[i] = off + bytes;
        }
    }

    for (size_t i = 0; i < tensor_count; ++i)
    {
        if (schedule.start_offset[i] >= schedule.end_offset[i] || schedule.end_offset[i] > arena_dense_size)
            throw std::runtime_error("parameter_offloader: schedule placement lies outside dense arena");
    }

    for (int r = 0; r < N; ++r)
    {
        int barrier = r; // unwrapped last safe copy position after reading r

        std::vector<std::pair<size_t, size_t>> ring_byte_ranges; // conservative coalesced byte spans for copied-but-unread future tensors

        for (int j = r + 1; j < r + N; ++j)
        {
            const int copy_idx = j % N; // candidate schedule index that would be copied next
            bool copy_is_safe = true; // true if this copy does not overwrite any protected future tensor span

            for (const auto & range : ring_byte_ranges)
            {
                if (ranges_overlap(range.first, range.second, schedule.start_offset[copy_idx], schedule.end_offset[copy_idx]))
                {
                    copy_is_safe = false;
                    break;
                }
            }

            if (!copy_is_safe)
                break;

            barrier = j;

            if (!ring_byte_ranges.empty() && ring_byte_ranges.back().second <= schedule.start_offset[copy_idx])
            {
                ring_byte_ranges.back().second = schedule.end_offset[copy_idx];
            }
            else
            {
                ring_byte_ranges.emplace_back(schedule.start_offset[copy_idx], schedule.end_offset[copy_idx]);
            }
        }

        schedule.ready_after[r] = barrier % N;
    }
}

//TODO: the streaming fit can be made tighter. Currently it aims for upper_bound, but we can get very close to lower_bound in theory for additional performance gains
bool llama_offloader_graph_cb(ggml_backend_sched_t sched, struct ggml_cgraph * graph, void * ud)
{
    LLAMA_LOG_INFO("%s 1\n", __func__);

    struct parameter_offloader *po = static_cast<parameter_offloader *>(ud);
    if (!po || !sched || !graph)
        return true;

    if (!po->ready)
        return true;

#ifdef LLAMA_PRINT_WEIGHT_READS
    print_all_weight_reads(po, graph);
#endif

    const uint64_t graph_hash = po->analyze_dense_graph(sched, graph, po->graph_analysis_next);
    const bool dense_fits_arena = po->graph_analysis_next.dense_fits_arena;

    std::unordered_set<ggml_tensor *> newly_deprioritized;

    // Deprioritize tensors that are actually static now but are omitted from the incoming graph.
    for (ggml_tensor * w_gpu : po->static_dense_order_current)
    {
        if (po->graph_analysis_next.gpu2index.find(w_gpu) != po->graph_analysis_next.gpu2index.end())
            continue;

        if (po->deprioritized_dense_set.insert(w_gpu).second)
            newly_deprioritized.insert(w_gpu);
    }

    // Invalidate only cached layouts that made a newly deprioritized tensor static.
    if (!newly_deprioritized.empty())
    {
        for (auto it = po->dense_graph_cache.begin(); it != po->dense_graph_cache.end(); )
        {
            bool invalidate = false;

            for (ggml_tensor * w_gpu : it->second.static_dense_order)
            {
                if (newly_deprioritized.find(w_gpu) == newly_deprioritized.end())
                    continue;

                invalidate = true;
                break;
            }

            if (invalidate)
                it = po->dense_graph_cache.erase(it);
            else
                ++it;
        }
    }

    auto cache_it = po->dense_graph_cache.find(graph_hash);
    size_t streaming_fit = 0;

    if (cache_it == po->dense_graph_cache.end())
    {
        // A new graph gets its own static/streamed partition.
        po->static_dense_order.clear();
        po->static_dense_set.clear();
        po->static_tensor_bytes = 0;

        if (dense_fits_arena)
        {
            po->static_dense_order = po->graph_analysis_next.gpu_tensors_in_order;
            po->static_dense_set.insert(po->static_dense_order.begin(), po->static_dense_order.end());
        }
        else
        {
            po->streaming_fit_calculate_bounds(po->graph_analysis_next);

            bool ejecting = false;

            for (;;)
            {
                if (!po->select_static_dense_tensors(po->graph_analysis_next))
                    break;

                po->streaming_fit_calculate_bounds(po->graph_analysis_next);

                const size_t static_capacity = po->streaming_fit_upper_bound >= po->arena_dense_size ? 0 : po->arena_dense_size - po->streaming_fit_upper_bound;

                if (po->static_tensor_bytes > static_capacity)
                {
                    ejecting = true;
                    continue;
                }

                if (ejecting)
                    break;
            }
        }

        // Canonicalize the finalized static layout so the same tensor set always receives the same top-down placement.
        std::sort(po->static_dense_order.begin(), po->static_dense_order.end(), [](ggml_tensor * lhs, ggml_tensor * rhs) {
            const char * lhs_name = ggml_get_name(lhs);
            const char * rhs_name = ggml_get_name(rhs);
            const int name_cmp = std::strcmp(lhs_name ? lhs_name : "", rhs_name ? rhs_name : "");
            return name_cmp != 0 ? name_cmp < 0 : (uintptr_t)lhs < (uintptr_t)rhs;
        });

        if (dense_fits_arena)
        {
            po->schedule_next = parameter_offloader::offloader_schedule{};
        }
        else
        {
            po->build_next_schedule(po->schedule_next, po->graph_analysis_next);
            po->build_graph_runtime_metadata(po->graph_analysis_next, po->schedule_next);
            streaming_fit = po->generate_streaming_fit(po->schedule_next, po->graph_analysis_next);
            po->build_schedule_gates(po->schedule_next);
        }

        parameter_offloader::dense_graph_cache_entry cache_entry;
        cache_entry.static_dense_order = po->static_dense_order;
        cache_entry.schedule = po->schedule_next;
        po->dense_graph_cache.emplace(graph_hash, std::move(cache_entry));
    }
    else
    {
        po->static_dense_order = cache_it->second.static_dense_order;
        po->static_dense_set.clear();
        po->static_tensor_bytes = 0;

        const size_t alignment = po->arena_alignment ? po->arena_alignment : 1;
        size_t static_cursor = po->arena_dense_size;

        // Rebuild the non-cached lookup/counting state from the cached static order.
        for (ggml_tensor * w_gpu : po->static_dense_order)
        {
            ggml_tensor * w_cpu = po->gpu2cpu.at(w_gpu);
            const size_t slot_bytes = ggml_backend_buft_get_alloc_size(po->arena_buffer_type, w_cpu);

            po->static_dense_set.insert(w_gpu);

            if (slot_bytes > static_cursor)
                throw std::runtime_error("parameter_offloader: cached static dense layout exceeds dense arena");

            static_cursor = align_down(static_cursor - slot_bytes, alignment);
        }

        po->static_tensor_bytes = po->arena_dense_size - static_cursor;

        po->schedule_next = cache_it->second.schedule;

        if (!dense_fits_arena)
        {
            po->build_graph_runtime_metadata(po->graph_analysis_next, po->schedule_next);

            // A cached schedule already contains the solved physical placement, so recover its streaming extent directly.
            // TODO: should the streaming_fit be cached?
            for (size_t end_offset : po->schedule_next.end_offset)
                streaming_fit = std::max(streaming_fit, end_offset);
        }
    }

    LLAMA_LOG_INFO("%s 2\n", __func__);

    po->swap_next_schedule(streaming_fit);

    LLAMA_LOG_INFO("%s 3\n", __func__);

    return true;
}

/////////////////////////////////////
//   MOE EXPERT CACHE
/////////////////////////////////////
struct moe_cache_field {
    const char * name;

    ggml_tensor * llama_layer::* cpu;
    ggml_tensor * llama_layer::* cache;
};

static const moe_cache_field moe_cache_fields[] = {
    // Routed expert matrices
    {
        "gate_exps",
        &llama_layer::ffn_gate_exps,
        &llama_layer::ffn_gate_exps_cache,
    },
    {
        "down_exps",
        &llama_layer::ffn_down_exps,
        &llama_layer::ffn_down_exps_cache,
    },
    {
        "up_exps",
        &llama_layer::ffn_up_exps,
        &llama_layer::ffn_up_exps_cache,
    },
    {
        "gate_up_exps",
        &llama_layer::ffn_gate_up_exps,
        &llama_layer::ffn_gate_up_exps_cache,
    },

    // Adjugate expert matrices
    {
        "gate_chexps",
        &llama_layer::ffn_gate_chexps,
        &llama_layer::ffn_gate_chexps_cache,
    },
    {
        "down_chexps",
        &llama_layer::ffn_down_chexps,
        &llama_layer::ffn_down_chexps_cache,
    },
    {
        "up_chexps",
        &llama_layer::ffn_up_chexps,
        &llama_layer::ffn_up_chexps_cache,
    },

    // Routed expert biases
    {
        "gate_exps_b",
        &llama_layer::ffn_gate_exps_b,
        &llama_layer::ffn_gate_exps_b_cache,
    },
    {
        "down_exps_b",
        &llama_layer::ffn_down_exps_b,
        &llama_layer::ffn_down_exps_b_cache,
    },
    {
        "up_exps_b",
        &llama_layer::ffn_up_exps_b,
        &llama_layer::ffn_up_exps_b_cache,
    },
    {
        "gate_up_exps_b",
        &llama_layer::ffn_gate_up_exps_b,
        &llama_layer::ffn_gate_up_exps_b_cache,
    },

    // Routed expert NVFP4 scales
    {
        "gate_exps_s",
        &llama_layer::ffn_gate_exps_s,
        &llama_layer::ffn_gate_exps_s_cache,
    },
    {
        "down_exps_s",
        &llama_layer::ffn_down_exps_s,
        &llama_layer::ffn_down_exps_s_cache,
    },
    {
        "up_exps_s",
        &llama_layer::ffn_up_exps_s,
        &llama_layer::ffn_up_exps_s_cache,
    },

    // Routed expert NVFP4 input scales
    {
        "gate_exps_in_s",
        &llama_layer::ffn_gate_exps_in_s,
        &llama_layer::ffn_gate_exps_in_s_cache,
    },
    {
        "down_exps_in_s",
        &llama_layer::ffn_down_exps_in_s,
        &llama_layer::ffn_down_exps_in_s_cache,
    },
    {
        "up_exps_in_s",
        &llama_layer::ffn_up_exps_in_s,
        &llama_layer::ffn_up_exps_in_s_cache,
    },
};

struct moe_cache_bank_build {
    int field_id;

    enum ggml_type type;

    int n_dims;
    int64_t ne[GGML_MAX_DIMS];

    ggml_tensor * tensor;

    size_t offset;
};

void parameter_offloader::clear_moe_cache_refs()
{
    if (!model)
        return;

    const int n_layers = (int) model->hparams.n_layer();
    const int n_fields = (int) (sizeof(moe_cache_fields) / sizeof(moe_cache_fields[0]));

    for (int il = 0; il < n_layers; ++il) {
        llama_layer & layer = model->layers[il];

        for (int field_id = 0; field_id < n_fields; ++field_id) {
            const moe_cache_field & field = moe_cache_fields[field_id];
            layer.*(field.cache) = nullptr;
        }
    }
}

void parameter_offloader::init_moe_cache(
    ggml_backend_buffer_t arena,
    int32_t n_slots)
{
    GGML_ASSERT(model);
    GGML_ASSERT(arena);
    GGML_ASSERT(n_slots >= model->hparams.n_expert_used);
    GGML_ASSERT(ctx_moe_cache == nullptr);

    if (llama_model_n_devices(model) != 1)
        throw std::runtime_error("MoE expert prefetch currently supports exactly one accelerator device");

    ggml_backend_dev_t arena_device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(arena));
    ggml_backend_dev_t model_device = llama_model_get_device(model, 0);
    if (arena_device != model_device)
        throw std::runtime_error("MoE expert cache arena is not allocated on the model's accelerator device");

    attach_arena(arena);

    const int n_layers = (int) model->hparams.n_layer();
    const int n_fields = (int) (sizeof(moe_cache_fields) / sizeof(moe_cache_fields[0]));

    size_t n_source_banks = 0;

    for (int il = 0; il < n_layers; ++il) {
        llama_layer & layer = model->layers[il];

        for (int field_id = 0; field_id < n_fields; ++field_id) {
            const moe_cache_field & field = moe_cache_fields[field_id];

            layer.*(field.cache) = nullptr;

            if (layer.*(field.cpu) != nullptr)
                ++n_source_banks;
        }
    }

    if (n_source_banks == 0) {
        LLAMA_LOG_INFO("%s: model contains no routed MoE expert weight banks\n", __func__);
        return;
    }

    GGML_ASSERT(n_source_banks <= SIZE_MAX / ggml_tensor_overhead());

    const size_t metadata_size = ggml_tensor_overhead() * n_source_banks;

    ggml_init_params params = {
        /* .mem_size   = */ metadata_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ctx_moe_cache = ggml_init(params);
    GGML_ASSERT(ctx_moe_cache);

    std::vector<moe_cache_bank_build> banks;
    banks.reserve(n_source_banks);

    for (int il = 0; il < n_layers; ++il) {
        llama_layer & layer = model->layers[il];

        for (int field_id = 0; field_id < n_fields; ++field_id) {
            const moe_cache_field & field = moe_cache_fields[field_id];

            ggml_tensor * cpu = layer.*(field.cpu);

            if (cpu == nullptr)
                continue;

            GGML_ASSERT(cpu->view_src == nullptr);
            GGML_ASSERT(!ggml_is_transposed(cpu));
            GGML_ASSERT(ggml_is_contiguous(cpu));

            const int n_dims = ggml_n_dims(cpu);

            GGML_ASSERT(n_dims >= 1);
            GGML_ASSERT(n_dims <= GGML_MAX_DIMS);

            const int expert_dim = n_dims - 1;

            GGML_ASSERT(cpu->ne[expert_dim] > 0);

            // Copy the source shape and replace the expert count with the cache slot count.
            int64_t cache_ne[GGML_MAX_DIMS];

            for (int d = 0; d < GGML_MAX_DIMS; ++d)
                cache_ne[d] = 1;

            for (int d = 0; d < n_dims; ++d)
                cache_ne[d] = cpu->ne[d];

            cache_ne[expert_dim] = n_slots;

            size_t bank_id = banks.size();

            // Reuse a cache bank when the field, type, rank, and shortened shape match.
            for (size_t i = 0; i < banks.size(); ++i) {
                const moe_cache_bank_build & bank = banks[i];

                if (bank.field_id != field_id ||
                    bank.type     != cpu->type ||
                    bank.n_dims   != n_dims)
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
                ggml_tensor * cache = ggml_new_tensor(ctx_moe_cache, cpu->type, n_dims, cache_ne);

                GGML_ASSERT(cache);
                GGML_ASSERT(!ggml_is_transposed(cache));
                GGML_ASSERT(ggml_is_contiguous(cache));

                ggml_format_name(cache, "moe_cache_%s_%d", field.name, (int) banks.size());

                moe_cache_bank_build bank = {};

                bank.field_id = field_id;
                bank.type     = cpu->type;
                bank.n_dims   = n_dims;
                bank.tensor   = cache;
                bank.offset   = 0;

                for (int d = 0; d < GGML_MAX_DIMS; ++d)
                    bank.ne[d] = cache_ne[d];

                banks.push_back(bank);
            }

            layer.*(field.cache) = banks[bank_id].tensor;
        }
    }

    const size_t cache_align = ggml_backend_buft_get_alignment(arena_buffer_type);

    size_t total_size = 0;

    for (moe_cache_bank_build & bank : banks) {
        total_size = align_up(total_size, cache_align);
        bank.offset = total_size;

        total_size += ggml_backend_buft_get_alloc_size(arena_buffer_type, bank.tensor);
    }

    total_size = align_up(total_size, cache_align);

    GGML_ASSERT(total_size > 0);
    GGML_ASSERT(total_size < arena_size);

    const size_t moe_cache_offset = align_down(arena_size - total_size, cache_align);
    const size_t moe_cache_size   = total_size;
    arena_dense_size = moe_cache_offset;
    arena_stream_size = arena_dense_size;

    GGML_ASSERT(arena_dense_size > 0);
    GGML_ASSERT(moe_cache_offset + moe_cache_size <= arena_size);

    char * cache_base = arena_base + moe_cache_offset;

    for (moe_cache_bank_build & bank : banks) {
        GGML_ASSERT(ggml_backend_tensor_alloc(arena, bank.tensor, cache_base + bank.offset) == GGML_STATUS_SUCCESS);

        LLAMA_LOG_INFO("%s: %s type=%s shape=[%lld,%lld,%lld,%lld] rank=%d offset=%zu bytes=%zu\n", __func__,
            ggml_get_name(bank.tensor),
            ggml_type_name(bank.tensor->type),
            (long long) bank.tensor->ne[0],
            (long long) bank.tensor->ne[1],
            (long long) bank.tensor->ne[2],
            (long long) bank.tensor->ne[3],
            bank.n_dims,
            moe_cache_offset + bank.offset,
            ggml_backend_buffer_get_alloc_size(arena, bank.tensor));
    }

    moe_cache_n_slots = n_slots;

    LLAMA_LOG_INFO("%s: reserved %zu bytes at arena offset %zu for %zu routed MoE cache banks with %d slots each; %zu bytes remain for dense streaming\n",
        __func__, moe_cache_size, moe_cache_offset, banks.size(), n_slots, arena_dense_size);
}

// TODO: This is a proof-of-concept test function.
// It performs no intelligent MoE slice prefetching.
// Every other requested expert is copied to the GPU cache; the rest stay on CPU.
int32_t parameter_offloader::debug_cache_moe_expert(
    int block_id,
    int32_t expert_id)
{
    std::lock_guard<std::mutex> lock(moe_cache_mu);

    if (!model || !ctx_moe_cache || moe_cache_n_slots <= 0) {
        return -1;
    }

    const int n_layers = (int) model->hparams.n_layer();

    GGML_ASSERT(block_id >= 0);
    GGML_ASSERT(block_id < n_layers);

    // Debug-only: first request uses the GPU cache, second stays on CPU,
    // third uses the GPU cache, and so on.
    static uint64_t debug_request_index = 0;

    const bool fetch_this_request =
        (debug_request_index++ % 2) == 0;

    if (!fetch_this_request) {
        return -1;
    }

    llama_layer & layer = model->layers[block_id];

    const int32_t cache_slot = moe_cache_next_slot;
    bool copied_any = false;

    const int n_fields =
        (int) (sizeof(moe_cache_fields) / sizeof(moe_cache_fields[0]));

    for (int field_id = 0; field_id < n_fields; ++field_id) {
        const moe_cache_field & field = moe_cache_fields[field_id];

        ggml_tensor * cpu   = layer.*(field.cpu);
        ggml_tensor * cache = layer.*(field.cache);

        GGML_ASSERT((cpu == nullptr) == (cache == nullptr));

        if (cpu == nullptr) {
            continue;
        }

        GGML_ASSERT(cpu->buffer);
        //GGML_ASSERT(ggml_backend_buffer_is_host(cpu->buffer));    //is there an important reason for this here? I have no clue
        GGML_ASSERT(cpu->data);
        GGML_ASSERT(cpu->view_src == nullptr);
        GGML_ASSERT(!ggml_is_transposed(cpu));
        GGML_ASSERT(ggml_is_contiguous(cpu));

        GGML_ASSERT(cache->buffer == arena);
        GGML_ASSERT(cache->data);
        GGML_ASSERT(cache->view_src == nullptr);
        GGML_ASSERT(!ggml_is_transposed(cache));
        GGML_ASSERT(ggml_is_contiguous(cache));

        GGML_ASSERT(cpu->type == cache->type);

        const int n_dims = ggml_n_dims(cpu);

        GGML_ASSERT(n_dims >= 1);
        GGML_ASSERT(n_dims <= GGML_MAX_DIMS);
        GGML_ASSERT(ggml_n_dims(cache) == n_dims);

        const int expert_dim = n_dims - 1;

        // All dimensions except the final expert dimension must match.
        for (int d = 0; d < expert_dim; ++d)
            GGML_ASSERT(cpu->ne[d] == cache->ne[d]);

        GGML_ASSERT(expert_id >= 0);
        GGML_ASSERT(expert_id < cpu->ne[expert_dim]);

        GGML_ASSERT(cache_slot >= 0);
        GGML_ASSERT(cache_slot < cache->ne[expert_dim]);
        GGML_ASSERT(cache->ne[expert_dim] == moe_cache_n_slots);

        GGML_ASSERT(cpu->nb[expert_dim] == cache->nb[expert_dim]);

        // One final-dimension stride contains one complete expert slice.
        const size_t expert_bytes = cpu->nb[expert_dim];

        const void * src =
            (const char *) cpu->data +
            (size_t) expert_id * cpu->nb[expert_dim];

        const size_t dst_offset =
            (size_t) cache_slot * cache->nb[expert_dim];

        ggml_backend_tensor_set(
            cache,
            src,
            dst_offset,
            expert_bytes);

        copied_any = true;
    }

    if (!copied_any) {
        return -1;
    }

    moe_cache_next_slot =
        (cache_slot + 1) % moe_cache_n_slots;

    return cache_slot;
}

int32_t llama_offloader_moe_residency_cb(
    int block_id,
    int32_t expert_id,
    void * ud)
{
    parameter_offloader * po = static_cast<parameter_offloader *>(ud);

    if (!po) {
        return -1;
    }

    //return -1; //use this to force 100% cpu rate

    return po->debug_cache_moe_expert(block_id, expert_id);
}

/////////////////////////////////////
//   DIAGNOSTICS
/////////////////////////////////////
void parameter_offloader::print_snapshot(offloader_schedule & schedule, ggml_log_level level)
{
    size_t tensor_count = schedule.gpu_tensors_in_order.size();

    std::vector<size_t> start(tensor_count);
    std::vector<size_t> end(tensor_count);
    start.reserve(tensor_count);
    end.reserve(tensor_count);

    for (size_t i = 0; i < tensor_count; ++i)
    {
        ggml_tensor *t_gpu = schedule.gpu_tensors_in_order[i];
        GGML_ASSERT(t_gpu && t_gpu->data);
        ggml_tensor *t_cpu = gpu2cpu.at(t_gpu);

        const size_t off   = (size_t)((char*)t_gpu->data - arena_base);                  // arena-relative
        const size_t bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, t_cpu); // padded size

        start[i] = off;
        end[i]   = off + bytes;
        GGML_ASSERT(end[i] <= arena_dense_size);
    }

    GGML_ASSERT(graph_analysis_current.release_node_by_tensor.size() == tensor_count);

    std::unordered_map<ggml_tensor *, int> read_gate_by_node;
    read_gate_by_node.reserve(tensor_count);

    for (int i = 0; i < (int)tensor_count; ++i)
    {
        ggml_tensor * node = graph_analysis_current.release_node_by_tensor[i];
        if (node)
            read_gate_by_node[node] = i;
    }

    for (int i = 0; i < (int)tensor_count; ++i)
    {
        ggml_tensor * w_gpu  = schedule.gpu_tensors_in_order[i]; // GPU tensor being printed
        const char * name    = ggml_get_name(w_gpu); // mirrored model tensor name

        ggml_tensor * read_node = graph_analysis_current.release_node_by_tensor[i];
        int read_gate = read_node ? read_gate_by_node.at(read_node) : -1;

        llama_log_internal(level, "%s %4d %4d %4d %10lu %10lu %5d %s\n",
            __func__,
            i,
            read_gate,
            schedule.ready_after[i],
            start[i],
            end[i],
            schedule.ready_after[i] - i,
            name ? name : "(unnamed)");
    }
}

void parameter_offloader::print_tensor_order(const std::vector<ggml_tensor *> & tensors, const std::vector<size_t> & offsets, ggml_log_level level)
{
    const size_t tensor_count = tensors.size();

    GGML_ASSERT(offsets.size() == tensor_count);

    for (size_t i = 0; i < tensor_count; ++i)
    {
        ggml_tensor * t_gpu = tensors[i];
        GGML_ASSERT(t_gpu);

        const char * name = ggml_get_name(t_gpu);
        ggml_tensor * t_cpu = gpu2cpu.at(t_gpu);
        const size_t bytes = ggml_backend_buft_get_alloc_size(arena_buffer_type, t_cpu);
        const size_t off = offsets[i];

        if (off == SIZE_MAX)
        {
            llama_log_internal(level, "%s %4zu %10zu %10s %10s %s\n",
                __func__,
                i,
                bytes,
                "UNPLACED",
                "UNPLACED",
                name ? name : "(unnamed)");
            continue;
        }

        const size_t end = off + bytes;

        GGML_ASSERT(end <= arena_dense_size);

        llama_log_internal(level, "%s %4zu %10zu %10zu %10zu %s\n",
            __func__,
            i,
            bytes,
            off,
            end,
            name ? name : "(unnamed)");
    }
}

#ifdef LLAMA_PRINT_WEIGHT_READS
static bool offloader_weight_is_sparse(const ggml_tensor * weight)
{
    const char * name = ggml_get_name(weight);
    if (!name)
        return false;

    return strstr(name, ".ffn_gate_exps.")    ||
           strstr(name, ".ffn_down_exps.")    ||
           strstr(name, ".ffn_up_exps.")      ||
           strstr(name, ".ffn_gate_up_exps.") ||
           strstr(name, ".ffn_gate_chexps.")  ||
           strstr(name, ".ffn_down_chexps.")  ||
           strstr(name, ".ffn_up_chexps.");
}

static void print_all_weight_reads(parameter_offloader * po, ggml_cgraph * graph)
{
    std::unordered_set<std::string> seen;
    int event = 0;

    LLAMA_LOG_INFO("%-6s %-70s %-7s %-7s %-7s %-8s\n",
        "EVENT", "WEIGHT", "REPEAT", "MANAGED", "SPARSE", "RESIDENT");

    for (int i = 0; i < graph->n_nodes; ++i)
    {
        ggml_tensor * node = graph->nodes[i];

        for (int k = 0; k < GGML_MAX_SRC; ++k)
        {
            ggml_tensor * weight = node->src[k];
            if (!weight)
                break;

            while (weight->view_src)
                weight = weight->view_src;

            auto managed_it = po->gpu2cpu.find(weight);
            const bool managed = managed_it != po->gpu2cpu.end();

            if (!managed && po->cpu_weight_set.find(weight) == po->cpu_weight_set.end())
                continue;

            ggml_tensor * model_weight = managed ? managed_it->second : weight;
            const char * name = ggml_get_name(model_weight);
            const bool repeat = !seen.insert(name ? name : "").second;
            const bool sparse = offloader_weight_is_sparse(model_weight);
            const char * resident = ggml_backend_buffer_is_host(model_weight->buffer) ? "HOST" : "DEVICE";

            LLAMA_LOG_INFO("%-6d %-70s %-7s %-7s %-7s %-8s\n",
                event++, name,
                repeat ? "YES" : "",
                managed ? "YES" : "",
                sparse ? "SPARSE" : "",
                resident);
        }
    }
}
#endif
