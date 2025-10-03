#pragma once
#include "ggml.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-context.h"
#include <cstring>

constexpr const char* ggml_op_to_string(ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:                 return "GGML_OP_NONE";

        case GGML_OP_DUP:                  return "GGML_OP_DUP";
        case GGML_OP_ADD:                  return "GGML_OP_ADD";
        case GGML_OP_ADD_ID:               return "GGML_OP_ADD_ID";
        case GGML_OP_ADD1:                 return "GGML_OP_ADD1";
        case GGML_OP_ACC:                  return "GGML_OP_ACC";
        case GGML_OP_SUB:                  return "GGML_OP_SUB";
        case GGML_OP_MUL:                  return "GGML_OP_MUL";
        case GGML_OP_DIV:                  return "GGML_OP_DIV";
        case GGML_OP_SQR:                  return "GGML_OP_SQR";
        case GGML_OP_SQRT:                 return "GGML_OP_SQRT";
        case GGML_OP_LOG:                  return "GGML_OP_LOG";
        case GGML_OP_SIN:                  return "GGML_OP_SIN";
        case GGML_OP_COS:                  return "GGML_OP_COS";
        case GGML_OP_SUM:                  return "GGML_OP_SUM";
        case GGML_OP_SUM_ROWS:             return "GGML_OP_SUM_ROWS";
        case GGML_OP_MEAN:                 return "GGML_OP_MEAN";
        case GGML_OP_ARGMAX:               return "GGML_OP_ARGMAX";
        case GGML_OP_COUNT_EQUAL:          return "GGML_OP_COUNT_EQUAL";
        case GGML_OP_REPEAT:               return "GGML_OP_REPEAT";
        case GGML_OP_REPEAT_BACK:          return "GGML_OP_REPEAT_BACK";
        case GGML_OP_CONCAT:               return "GGML_OP_CONCAT";
        case GGML_OP_SILU_BACK:            return "GGML_OP_SILU_BACK";
        case GGML_OP_NORM:                 return "GGML_OP_NORM";
        case GGML_OP_RMS_NORM:             return "GGML_OP_RMS_NORM";
        case GGML_OP_RMS_NORM_BACK:        return "GGML_OP_RMS_NORM_BACK";
        case GGML_OP_GROUP_NORM:           return "GGML_OP_GROUP_NORM";
        case GGML_OP_L2_NORM:              return "GGML_OP_L2_NORM";

        case GGML_OP_MUL_MAT:              return "GGML_OP_MUL_MAT";
        case GGML_OP_MUL_MAT_ID:           return "GGML_OP_MUL_MAT_ID";
        case GGML_OP_OUT_PROD:             return "GGML_OP_OUT_PROD";

        case GGML_OP_SCALE:                return "GGML_OP_SCALE";
        case GGML_OP_SET:                  return "GGML_OP_SET";
        case GGML_OP_CPY:                  return "GGML_OP_CPY";
        case GGML_OP_CONT:                 return "GGML_OP_CONT";
        case GGML_OP_RESHAPE:              return "GGML_OP_RESHAPE";
        case GGML_OP_VIEW:                 return "GGML_OP_VIEW";
        case GGML_OP_PERMUTE:              return "GGML_OP_PERMUTE";
        case GGML_OP_TRANSPOSE:            return "GGML_OP_TRANSPOSE";
        case GGML_OP_GET_ROWS:             return "GGML_OP_GET_ROWS";
        case GGML_OP_GET_ROWS_BACK:        return "GGML_OP_GET_ROWS_BACK";
        case GGML_OP_SET_ROWS:             return "GGML_OP_SET_ROWS";
        case GGML_OP_DIAG:                 return "GGML_OP_DIAG";
        case GGML_OP_DIAG_MASK_INF:        return "GGML_OP_DIAG_MASK_INF";
        case GGML_OP_DIAG_MASK_ZERO:       return "GGML_OP_DIAG_MASK_ZERO";
        case GGML_OP_SOFT_MAX:             return "GGML_OP_SOFT_MAX";
        case GGML_OP_SOFT_MAX_BACK:        return "GGML_OP_SOFT_MAX_BACK";
        case GGML_OP_ROPE:                 return "GGML_OP_ROPE";
        case GGML_OP_ROPE_BACK:            return "GGML_OP_ROPE_BACK";
        case GGML_OP_CLAMP:                return "GGML_OP_CLAMP";
        case GGML_OP_CONV_TRANSPOSE_1D:    return "GGML_OP_CONV_TRANSPOSE_1D";
        case GGML_OP_IM2COL:               return "GGML_OP_IM2COL";
        case GGML_OP_IM2COL_BACK:          return "GGML_OP_IM2COL_BACK";
        case GGML_OP_IM2COL_3D:            return "GGML_OP_IM2COL_3D";
        case GGML_OP_CONV_2D:              return "GGML_OP_CONV_2D";
        case GGML_OP_CONV_3D:              return "GGML_OP_CONV_3D";
        case GGML_OP_CONV_2D_DW:           return "GGML_OP_CONV_2D_DW";
        case GGML_OP_CONV_TRANSPOSE_2D:    return "GGML_OP_CONV_TRANSPOSE_2D";
        case GGML_OP_POOL_1D:              return "GGML_OP_POOL_1D";
        case GGML_OP_POOL_2D:              return "GGML_OP_POOL_2D";
        case GGML_OP_POOL_2D_BACK:         return "GGML_OP_POOL_2D_BACK";
        case GGML_OP_UPSCALE:              return "GGML_OP_UPSCALE";
        case GGML_OP_PAD:                  return "GGML_OP_PAD";
        case GGML_OP_PAD_REFLECT_1D:       return "GGML_OP_PAD_REFLECT_1D";
        case GGML_OP_ROLL:                 return "GGML_OP_ROLL";
        case GGML_OP_ARANGE:               return "GGML_OP_ARANGE";
        case GGML_OP_TIMESTEP_EMBEDDING:   return "GGML_OP_TIMESTEP_EMBEDDING";
        case GGML_OP_ARGSORT:              return "GGML_OP_ARGSORT";
        case GGML_OP_LEAKY_RELU:           return "GGML_OP_LEAKY_RELU";

        case GGML_OP_FLASH_ATTN_EXT:       return "GGML_OP_FLASH_ATTN_EXT";
        case GGML_OP_FLASH_ATTN_BACK:      return "GGML_OP_FLASH_ATTN_BACK";
        case GGML_OP_SSM_CONV:             return "GGML_OP_SSM_CONV";
        case GGML_OP_SSM_SCAN:             return "GGML_OP_SSM_SCAN";
        case GGML_OP_WIN_PART:             return "GGML_OP_WIN_PART";
        case GGML_OP_WIN_UNPART:           return "GGML_OP_WIN_UNPART";
        case GGML_OP_GET_REL_POS:          return "GGML_OP_GET_REL_POS";
        case GGML_OP_ADD_REL_POS:          return "GGML_OP_ADD_REL_POS";
        case GGML_OP_RWKV_WKV6:            return "GGML_OP_RWKV_WKV6";
        case GGML_OP_GATED_LINEAR_ATTN:    return "GGML_OP_GATED_LINEAR_ATTN";
        case GGML_OP_RWKV_WKV7:            return "GGML_OP_RWKV_WKV7";

        case GGML_OP_UNARY:                return "GGML_OP_UNARY";

        case GGML_OP_MAP_CUSTOM1:          return "GGML_OP_MAP_CUSTOM1";
        case GGML_OP_MAP_CUSTOM2:          return "GGML_OP_MAP_CUSTOM2";
        case GGML_OP_MAP_CUSTOM3:          return "GGML_OP_MAP_CUSTOM3";

        case GGML_OP_CUSTOM:               return "GGML_OP_CUSTOM";

        case GGML_OP_CROSS_ENTROPY_LOSS:        return "GGML_OP_CROSS_ENTROPY_LOSS";
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:   return "GGML_OP_CROSS_ENTROPY_LOSS_BACK";
        case GGML_OP_OPT_STEP_ADAMW:            return "GGML_OP_OPT_STEP_ADAMW";
        case GGML_OP_OPT_STEP_SGD:              return "GGML_OP_OPT_STEP_SGD";

        case GGML_OP_GLU:                  return "GGML_OP_GLU";

        case GGML_OP_COUNT:                return "GGML_OP_COUNT";
    }
    return "GGML_OP_UNKNOWN";
}

class slice_only_diag {

public:
    slice_only_diag()
    {
        // After CUDA is initialized (e.g., right after llama_init_from_model):
        ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        size_t arena_bytes = /* e.g. */ (size_t)2 * 1024ull * 1024ull * 1024ull; // TODO: parametrize
        ggml_backend_buffer_t arena = ggml_cuda_arena_create_on(cuda_dev, arena_bytes, /*device_ordinal=*/0);
    }

    // nothing to own; purely stateless
    // optional: toggle to avoid boundaries inside fused attention/PE while testing
    bool avoid_attn_pe = false;

    bool node_reads_tracked_weight(ggml_tensor * t)
    {
        switch (t->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_ADD:        // keep identical to your current rule
                break;
            default:
                return false;
        }
        // scan sources for any model weight (GPU or CPU; but we are not mirroring here)
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            ggml_tensor * s = t->src[k];
            if (!s) break;
            ggml_tensor * p = s;
            while (p->view_src) p = p->view_src;
            // heuristic: weights have a name and are not leaf activations
            // (this is just for slicing similarity; no ownership is changed)
            if (ggml_get_name(p) && p->buffer && !ggml_backend_buffer_is_host(p->buffer)) {
                return true;  // device-backed (likely weight) – good enough for slicing parity
            }
            // also allow host-backed weights (no offloader)
            if (ggml_get_name(p) && p->buffer && ggml_backend_buffer_is_host(p->buffer) &&
                p->op == GGML_OP_NONE) {
                return true;
            }
        }
        return false;
    }

    bool wants_observe(ggml_tensor * t)
    {
        const char * nm = ggml_get_name(t);
        if (avoid_attn_pe && nm) {
            if (strncmp(nm, "q_pe-",       5) == 0) return false;
            if (strncmp(nm, "q_nope",      6) == 0) return false;
            if (strncmp(nm, "__fattn__",   9) == 0) return false;
            if (strncmp(nm, "fattn_mla",   9) == 0) return false;
            if (strncmp(nm, "kqv_out",     7) == 0) return false;
        }
        return node_reads_tracked_weight(t);
    }
};

// Evaluate-callback: reproduces slicing only; no side effects
static bool llama_slice_only_eval_cb(ggml_tensor * t, bool ask, void * ud)
{
    slice_only_diag * diag = static_cast<slice_only_diag*>(ud);
    if (!diag) return true;

    if (ask) {
        // you can log boundaries for visibility
        bool wants = diag->wants_observe(t);
        if (wants) {
            const char * nm = ggml_get_name(t);
            //LLAMA_LOG_INFO("[BOUNDARY %s%s]",
            //               nm ? " " : "",
            //               nm ? nm : "");
        }
        return wants;
    } else {
        // post-exec: deliberately no bookkeeping, no copying
        return true;
    }
}

uint64_t fnv1a64(const void * p, size_t n)
{
    const uint8_t * b = static_cast<const uint8_t *>(p);
    uint64_t h = 1469598103934665603ull;        // offset basis
    const uint64_t prime = 1099511628211ull;
    for (size_t i = 0; i < n; ++i)
    {
        h ^= b[i];
        h *= prime;
    }
    return h;
};
void hash_compare_tensor(ggml_tensor * w_cpu, ggml_tensor * w_gpu, char * base, int idx)
{
    // FNV-1a 64-bit (simple, fast)


    const char * name   = ggml_get_name(w_gpu); // you mirrored names when duplicating
    const size_t off    = (size_t)((char *) w_gpu->data - base);
    const size_t nbytes = ggml_nbytes(w_cpu);

    // CPU view
    uint64_t h_cpu = 0;
    if (w_cpu && w_cpu->data && nbytes)
        h_cpu = fnv1a64(w_cpu->data, nbytes);

    // GPU snapshot -> host and hash
    uint64_t h_gpu = 0;
    if (nbytes)
    {
        std::vector<uint8_t> tmp(nbytes);
        // copy device -> host for the logical bytes
        ggml_backend_tensor_get(w_gpu, tmp.data(), 0, nbytes);
        h_gpu = fnv1a64(tmp.data(), nbytes);
    }

    if (h_cpu != h_gpu) {
        LLAMA_LOG_WARN(
            "[MISMATCH] idx=%d name=%s off=%zu bytes=%zu h_cpu=%016llx h_gpu=%016llx\n",
            idx, name ? name : "(unnamed)", (size_t)off, (size_t)nbytes,
            (unsigned long long)h_cpu, (unsigned long long)h_gpu
        );
    }
#ifdef LLAMA_CHECK_WEIGHTS_VERBOSE
    else
    {
        LLAMA_LOG_INFO(
            "[GOOD] idx=%d name=%s off=%zu bytes=%zu hash=%016llx\n",
            idx, name ? name : "(unnamed)", (size_t)off, (size_t)nbytes,
            (unsigned long long)h_cpu
        );
    }
#endif
}

//returns true if non-finite is detected, max_bytes defaults to 1 MiB
bool finite_check_node(ggml_tensor * node, bool ask = false, size_t max_bytes = 1ull << 20)
{
    const size_t nbytes  = ggml_nbytes(node);
    const ggml_type t    = node->type;

    // Cap how much we read back (debug-only)
    const size_t to_copy = std::min<size_t>(nbytes, max_bytes);
    std::vector<uint8_t> tmp(to_copy);
    ggml_backend_tensor_get(node, tmp.data(), 0, to_copy);

    auto has_non_finite_f32 = [&](const float *p, size_t n) -> bool {
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(p[i]))
                return true;
        return false;
    };

    bool bad = false;
    if (t == GGML_TYPE_F32)
        bad = has_non_finite_f32(reinterpret_cast<const float*>(tmp.data()), to_copy/sizeof(float));
    else if (t == GGML_TYPE_F16)
    {
        // minimal fp16 scan: expand a small sample
        const uint16_t *h = reinterpret_cast<const uint16_t*>(tmp.data());
        size_t n = to_copy/sizeof(uint16_t);
        for (size_t i = 0; i < n; i += (n > 8192 ? n/8192 : 1)) // stride-sample to keep it cheap
        {
            // fp16 NaN: exponent all ones and mantissa non-zero
            uint16_t v = h[i];
            uint16_t exp = (v >> 10) & 0x1F, mant = v & 0x3FF;
            if (exp == 0x1F && mant != 0)
            {
                bad = true;
                break;
            }
        }
    }
    if (bad)
    {
        const char * nm = ggml_get_name(node);
        LLAMA_LOG_WARN("NON-FINITE%s in node '%s' (type=%d, bytes=%zu)\n", ask ? "" : " COMPUTED", nm ? nm : "(unnamed)", (int)t, nbytes);
        // Optional: llama_synchronize(lctx_) and abort to catch stack trace cleanly
    }
    return bad;
}

// ---------- Probe rules: if <trigger> is non-finite, also check these producers ----------

struct probe_spec {
    int         depth;       // how many src-hops upward: 1=immediate sources, 2=sources-of-sources, ...
    std::string match;       // name prefix (or substring if 'substring==true')
    bool        substring;   // match as substring instead of prefix
};

// prefix/substring matchers
static inline bool name_has_prefix(const char * nm, const std::string & pref) {
    return nm && strncmp(nm, pref.c_str(), pref.size()) == 0;
}
static inline bool name_has_substring(const char * nm, const std::string & sub) {
    return nm && strstr(nm, sub.c_str()) != nullptr;
}

using probe_rules_t = std::unordered_map<std::string, std::vector<probe_spec>>;

// EDIT this function to add/adjust your rules:
static const probe_rules_t & get_probe_rules() {
    static const probe_rules_t rules = {
        // If "Qcur-*" is bad, check its producers: q_pe-*, q_nope_absorbed-*, q_nope-*, q- (depth 1)
        // and also the RoPE table "leaf_*" as a source-of-source (depth 2).
        { "Qcur-", {
            { 1, "q_pe-",             false },
            { 1, "q_nope_absorbed-",  false },
            { 1, "q_nope-",           false },
            { 1, "q-",                false },
            { 2, "leaf_",             true  },   // substring match
        }},
        { "q_nope_absorbed-", {
            { 1, "q-",               false },   // immediate producer
            { 2, "q_nope-",          false },   // immediate producer
            { 1, ".attn_q_b.weight", true },    // typical Q-branch B weight (adjust if your arch uses a different name)
            { 1, ".attn_k_b.weight", true },    // include this too since your earlier logs showed this name under q_nope_absorbed-*
        }},
        // You can add more patterns as needed, e.g.:
        // { "Kcur-", { {1, "kv_cmpr-", false}, {1, "k_pe-", false} } },
        // { "Vcur-", { {1, "kv_cmpr-", false} } },
    };
    return rules;
}

static inline ggml_tensor * peel_view(ggml_tensor * t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}

// Walk 'depth' levels upward through src[] and collect nodes at exactly that depth
static void collect_at_depth(ggml_tensor * start, int depth, std::vector<ggml_tensor*> & out) {
    std::vector<ggml_tensor*> cur{ peel_view(start) };
    for (int d = 0; d < depth; ++d) {
        std::vector<ggml_tensor*> nxt;
        for (ggml_tensor * n : cur) {
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                ggml_tensor * s = n->src[k];
                if (!s) break;
                nxt.push_back(peel_view(s));
            }
        }
        cur.swap(nxt);
        if (cur.empty()) break;
    }
    out = cur;
}

// Call when a *source* tensor 'src_node' has just tested non-finite (pre-compute)
static void run_probe_rules_for_bad_source(ggml_tensor * src_node, bool ask) {
    ggml_tensor * peeled = peel_view(src_node);
    const char  * sname  = ggml_get_name(peeled);
    if (!sname)
        return;

    const auto & rules = get_probe_rules();
    for (const auto & kv : rules) {
        const std::string & trigger = kv.first;
        if (!name_has_prefix(sname, trigger))
            continue;

        for (const probe_spec & spec : kv.second) {
            std::vector<ggml_tensor*> targets;
            collect_at_depth(peeled, spec.depth, targets);
            for (ggml_tensor * t : targets) {
                const char * tn = ggml_get_name(t);
                bool match = spec.substring ? name_has_substring(tn, spec.match)
                                            : name_has_prefix   (tn, spec.match);
                if (match) {
                    bool bad = finite_check_node(t, ask, 1ull << 30);  //Check 1 GB this time
                    if (!bad)
                        LLAMA_LOG_INFO("%s %s is finite\n", __func__, tn);
                }
            }
        }
        // Only one trigger should match; break after the first for clarity
        break;
    }
}

