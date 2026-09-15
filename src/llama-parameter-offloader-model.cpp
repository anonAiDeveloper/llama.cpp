#include "ggml.h"
#include "llama-parameter-offloader-model.h"

#include <cstring>
#include <stdexcept>


static inline bool offloader_name_ends(const std::string & name, const char * suffix)
{
    const size_t n = std::strlen(suffix);
    return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}

//Deepseek 2 weights
static bool parameter_offloader_deepseek2_weight_supported(const std::string & name)
{
    return
        //name == "token_embd.weight"                         || // Input token embedding table; GET_ROWS selects only rows for current token IDs
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
        name == "output_norm.weight"                        || // Final RMSNorm scale before logits
        name == "output.weight";                               // LM head / output projection from hidden state to vocabulary logits
}

static const std::vector<std::string> parameter_offloader_deepseek2_cpu_weight_patterns = {
    R"(^token_embd\.weight$)",
    R"(\.ffn_(down|gate|up)_exps\.weight$)",
};

static const std::vector<std::string> parameter_offloader_deepseek2_gpu_weight_patterns = {
    R"(\.attn_(norm|q_a|q_a_norm|q_b|k_b|kv_a_mqa|kv_a_norm|v_b|kv_b|output)\.weight$)",
    R"(\.ffn_(norm|gate|up|down|gate_inp|gate_shexp|up_shexp|down_shexp)\.weight$)",
    R"(\.exp_probs_b\.bias$)",
    R"(^output_norm\.weight$)",
    R"(^output\.weight$)",
};

//GPT-OSS weights
static bool parameter_offloader_gpt_oss_weight_supported(const std::string & name)
{
    return
        //name == "token_embd.weight"                              ||   // Input token embedding table; GET_ROWS selects only rows for current token IDs
        name == "output_norm.weight"                             ||   // Final RMSNorm scale before logits
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
        offloader_name_ends(name, ".attn_output.weight")        ||   // Attention output projection back to model hidden size
        offloader_name_ends(name, ".attn_output.bias")          ||   // Attention output bias
        offloader_name_ends(name, ".attn_sinks.weight")         ||   // Per-head attention sink parameters; 1D vector over attention heads
        offloader_name_ends(name, ".ffn_gate_inp.weight")       ||   // MoE router/gating projection: hidden -> expert scores
        offloader_name_ends(name, ".ffn_gate_inp.bias")         ||   // MoE router/gating bias; one value per expert
        //offloader_name_ends(name, ".ffn_gate_exps.weight")     || // SPARSE: Routed MoE expert-bank gate matrices; packed per expert
        //offloader_name_ends(name, ".ffn_down_exps.weight")     || // SPARSE: Routed MoE expert-bank down matrices; packed per expert
        //offloader_name_ends(name, ".ffn_up_exps.weight")       || // SPARSE: Routed MoE expert-bank up matrices; packed per expert
        //offloader_name_ends(name, ".ffn_gate_exps.bias")       || // SPARSE: routed expert gate biases
        //offloader_name_ends(name, ".ffn_down_exps.bias")       || // SPARSE: routed expert down biases
        //offloader_name_ends(name, ".ffn_up_exps.bias")         || // SPARSE: routed expert up biases
        //offloader_name_ends(name, ".ffn_gate_exps.scale")      || // SPARSE: routed expert gate scales
        //offloader_name_ends(name, ".ffn_down_exps.scale")      || // SPARSE: routed expert down scales
        //offloader_name_ends(name, ".ffn_up_exps.scale")        || // SPARSE: routed expert up scales
        //offloader_name_ends(name, ".ffn_gate_exps.input_scale")|| // SPARSE: routed expert gate input scales
        //offloader_name_ends(name, ".ffn_down_exps.input_scale")|| // SPARSE: routed expert down input scales
        //offloader_name_ends(name, ".ffn_up_exps.input_scale")  || // SPARSE: routed expert up input scales
        name == "output.weight";                                      // LM head / output projection from hidden state to vocabulary logits
}

static const std::vector<std::string> parameter_offloader_gpt_oss_cpu_weight_patterns = {
    R"(^token_embd\.weight$)",
    R"(\.ffn_(gate|down|up)_exps\.(weight|bias|scale|input_scale)$)",
};

static const std::vector<std::string> parameter_offloader_gpt_oss_gpu_weight_patterns = {
    R"(^output_norm\.weight$)",
    R"(\.(attn_norm|post_attention_norm|attn_sinks)\.weight$)",
    R"(\.(attn_qkv|attn_q|attn_k|attn_v|attn_output|ffn_gate_inp)\.(weight|bias)$)",
    R"(^output\.weight$)",
};

// DeepSeek V4 weights
static bool parameter_offloader_deepseek4_weight_supported(const std::string & name)
{
    return
        //name == "token_embd.weight"                                ||   // Input token embedding table; GET_ROWS selects only rows for current token IDs
        offloader_name_ends(name, ".hc_attn_fn.weight")            ||   // Hyperconnection projection before attention
        offloader_name_ends(name, ".hc_attn_base.weight")          ||   // HC attention base stores pre[hc], post[hc], and comb[hc*hc] affine biases.
        offloader_name_ends(name, ".hc_attn_scale.weight")         ||   // HC attention scale stores separate pre, post, and comb affine scales.
        offloader_name_ends(name, ".attn_norm.weight")             ||   // RMSNorm scale before attention; 1D vector over hidden size
        offloader_name_ends(name, ".attn_sinks.weight")            ||   // Per-head attention sink parameters; 1D vector over attention heads
        offloader_name_ends(name, ".attn_q_a.weight")              ||   // First low-rank Q projection: hidden -> q_lora_rank before q_a_norm/q_b
        offloader_name_ends(name, ".attn_q_a_norm.weight")         ||   // RMSNorm scale on low-rank Q activation between q_a and q_b; 1D vector
        offloader_name_ends(name, ".attn_q_b.weight")              ||   // Second low-rank Q projection: q_lora_rank -> full per-head Q
        offloader_name_ends(name, ".attn_kv.weight")               ||   // Shared attention KV projection
        offloader_name_ends(name, ".attn_kv_a_norm.weight")        ||   // RMSNorm scale on projected KV activation before RoPE; 1D vector over attention-head width
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
        offloader_name_ends(name, ".ffn_norm.weight")              ||   // RMSNorm scale before FFN/MoE block; 1D vector over hidden size
        offloader_name_ends(name, ".ffn_gate_inp.weight")          ||   // Dense MoE router/gating projection: hidden -> expert scores
        offloader_name_ends(name, ".ffn_gate_tid2eid.weight")      ||   // Hash-router token-to-expert table
        offloader_name_ends(name, ".exp_probs_b.bias")             ||   // MoE expert-score/probability bias; 1D vector over experts
        //offloader_name_ends(name, ".ffn_gate_exps.weight")       ||   // SPARSE: Routed MoE expert-bank gate matrices; packed per expert
        //offloader_name_ends(name, ".ffn_down_exps.weight")       ||   // SPARSE: Routed MoE expert-bank down matrices; packed per expert
        //offloader_name_ends(name, ".ffn_up_exps.weight")         ||   // SPARSE: Routed MoE expert-bank up matrices; packed per expert
        offloader_name_ends(name, ".ffn_gate_shexp.weight")        ||   // Shared expert FFN gate projection; not routed by top-k
        offloader_name_ends(name, ".ffn_up_shexp.weight")          ||   // Shared expert FFN up projection; not routed by top-k
        offloader_name_ends(name, ".ffn_down_shexp.weight")        ||   // Shared expert FFN down projection; not routed by top-k
        name == "output_hc_fn.weight"                              ||   // Final hyperconnection projection
        name == "output_hc_scale.weight"                           ||   // Final hyperconnection scale
        name == "output_hc_base.weight"                            ||   // Final hyperconnection base
        name == "output_norm.weight"                               ||   // Final RMSNorm scale before logits
        name == "output.weight";                                        // LM head / output projection from hidden state to vocabulary logits
}

static const std::vector<std::string> parameter_offloader_deepseek4_cpu_weight_patterns = {
    R"(^token_embd\.weight$)",
    R"(\.ffn_(gate|down|up)_exps\.weight$)",
};

static const std::vector<std::string> parameter_offloader_deepseek4_gpu_weight_patterns = {
    R"(\.(hc_attn_fn|hc_attn_base|hc_attn_scale|attn_norm|attn_sinks|attn_q_a|attn_q_a_norm|attn_q_b|attn_kv|attn_kv_a_norm|attn_compressor_kv|attn_compressor_gate|attn_compressor_ape|attn_compressor_norm|indexer_compressor_kv|indexer_compressor_gate|indexer_compressor_ape|indexer_compressor_norm|indexer\.attn_q_b|indexer\.proj|attn_output_a|attn_output_b|hc_ffn_fn|hc_ffn_base|hc_ffn_scale|ffn_norm|ffn_gate_inp|ffn_gate_tid2eid|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight$)",
    R"(\.exp_probs_b\.bias$)",
    R"(^output(_hc_fn|_hc_scale|_hc_base|_norm)?\.weight$)",
};

//Op filters to speed up graph walking. Each model configures the ops that can directly read one of its enabled dense weights.
static void parameter_offloader_deepseek2_node_may_read_dense_weight(bool * dense_read_ops)
{
    dense_read_ops[GGML_OP_GET_ROWS] = true;       // token_embd.weight;
    dense_read_ops[GGML_OP_MUL]      = true;       // attn/output/FFN/Q/KV norm weights; currently disabled
    dense_read_ops[GGML_OP_MUL_MAT]  = true;       // Q/KV/attention output, dense/shared FFN, router, and output weights
    dense_read_ops[GGML_OP_ADD]      = true;       // exp_probs_b.bias; currently disabled
    //dense_read_ops[GGML_OP_MUL_MAT_ID] = true;   // routed MoE expert banks; sparse and handled separately
}

static void parameter_offloader_gpt_oss_node_may_read_dense_weight(bool * dense_read_ops)
{
    dense_read_ops[GGML_OP_GET_ROWS]       = true; // token_embd.weight
    dense_read_ops[GGML_OP_MUL]            = true; // attention/post-attention/output norm weights
    dense_read_ops[GGML_OP_MUL_MAT]        = true; // QKV, attention output, router, and output weights
    dense_read_ops[GGML_OP_ADD]            = true; // QKV, attention-output, and router biases
    dense_read_ops[GGML_OP_SOFT_MAX]       = true; // attn_sinks.weight on the non-flash attention path
    dense_read_ops[GGML_OP_FLASH_ATTN_EXT] = true; // attn_sinks.weight on the flash-attention path
    //dense_read_ops[GGML_OP_MUL_MAT_ID] = true;   // routed expert gate/up/down weights; sparse and handled separately
    //dense_read_ops[GGML_OP_ADD_ID] = true;       // routed expert gate/up/down biases; sparse and handled separately
}

static void parameter_offloader_deepseek4_node_may_read_dense_weight(bool * dense_read_ops)
{
    dense_read_ops[GGML_OP_GET_ROWS]       = true; // token embedding, compressor/indexer APE, hash-router table
    dense_read_ops[GGML_OP_MUL]            = true; // norm weights and final HC scale
    dense_read_ops[GGML_OP_MUL_MAT]        = true; // HC, attention, compressor/indexer, FFN, router, output
    dense_read_ops[GGML_OP_ADD]            = true; // expert-probability bias and final HC base
    dense_read_ops[GGML_OP_SOFT_MAX]       = true; // attn_sinks on non-flash attention
    dense_read_ops[GGML_OP_FLASH_ATTN_EXT] = true; // attn_sinks on flash attention
    //dense_read_ops[GGML_OP_MUL_MAT_ID]   = true; // routed expert banks; SPARSE
    dense_read_ops[GGML_OP_DSV4_HC_COMB]   = true; // per-layer HC base/scale
}

extern parameter_offloader_model_i parameter_offloader_deepseek2_i = {
    /*cpu_weight_patterns*/         parameter_offloader_deepseek2_cpu_weight_patterns,
    /*gpu_weight_patterns*/         parameter_offloader_deepseek2_gpu_weight_patterns,
    /*configure_dense_read_ops*/    parameter_offloader_deepseek2_node_may_read_dense_weight,
};

extern parameter_offloader_model_i parameter_offloader_gpt_oss_i = {
    /*cpu_weight_patterns*/         parameter_offloader_gpt_oss_cpu_weight_patterns,
    /*gpu_weight_patterns*/         parameter_offloader_gpt_oss_gpu_weight_patterns,
    /*configure_dense_read_ops*/    parameter_offloader_gpt_oss_node_may_read_dense_weight,
};

extern parameter_offloader_model_i parameter_offloader_deepseek4_i = {
    /*cpu_weight_patterns*/         parameter_offloader_deepseek4_cpu_weight_patterns,
    /*gpu_weight_patterns*/         parameter_offloader_deepseek4_gpu_weight_patterns,
    /*configure_dense_read_ops*/    parameter_offloader_deepseek4_node_may_read_dense_weight,
};

parameter_offloader_model_i * parameter_offloader_get_model_i(llama_model  * model)
{
    switch (model->arch)
    {
        case LLM_ARCH_DEEPSEEK2:
            return &parameter_offloader_deepseek2_i;
            break;
        case LLM_ARCH_OPENAI_MOE:
            return &parameter_offloader_gpt_oss_i;
            break;
        case LLM_ARCH_DEEPSEEK4:
            return &parameter_offloader_deepseek4_i;
            break;
        default:
            throw std::runtime_error("parameter_offloader: unsupported model architecture");
    }
}