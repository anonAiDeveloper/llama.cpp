#include "llama-parameter-offloader.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <climits>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <math.h>       /* isfinite */
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <stdexcept>

/////////////////////////////////////
//   DEBUGGING SWITCHES
/////////////////////////////////////
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

//#define LLAMA_NAIVE_OFFLOADER 1
/////////////////////////////////////

#ifndef GGML_USE_CUDA
#warning "WARNING! GGML_USE_CUDA is not defined!"
#endif
#include "llama-offloader-diagnostic.h"

static inline size_t align_up(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

static inline int ordinal_mod(long long ordinal, int n)
{
    return (int)(ordinal % (long long)n);
}

static inline long long advance_ordinal_to_idx(long long ordinal, int idx, int tensor_count)
{
    if (ordinal < 0)
        return idx;

    const int cur = ordinal_mod(ordinal, tensor_count);
    const int d = (idx - cur + tensor_count) % tensor_count;

    return ordinal + d;
}

inline bool parameter_offloader::no_transform_needed_for_backend_(const ggml_tensor *t) const {
    const size_t logical   = ggml_nbytes(t);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(buft, t);

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

parameter_offloader::parameter_offloader(llama_model  * model)
    : model(model)
{
    // Require DeepSeek-2
    //if (model->arch != LLM_ARCH_DEEPSEEK2) {
    //    LLAMA_LOG_WARN("%s: registry constructor: non-DeepSeek2 arch (%d) no ordering applied\n", __func__, (int)model->arch);
    //    return;
    //}

    cpu_weight_set.clear();
    cpu_weight_set.reserve(model->tensors_by_name.size());
    for (const auto & kv : model->tensors_by_name)
        if (kv.second)
            cpu_weight_set.insert(kv.second);
}

// Call this *before* transform/upload, i.e. at the top of parameter_offloader::init()
// Guarantees collected_order contains *all* host-backed model weights
void parameter_offloader::seed_all_weights_from_model()
{
    collected_order.clear();
    collect_seen.clear();

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

        auto ends = [](const std::string & s, const char * suffix) {
            const size_t n = std::strlen(suffix);
            return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
        };

        const bool supported =
            ends(kv.first, ".attn_q_a.weight")      ||
            ends(kv.first, ".attn_q_b.weight")      ||
            ends(kv.first, ".attn_k_b.weight")      ||
            ends(kv.first, ".attn_kv_a_mqa.weight") ||
            ends(kv.first, ".attn_v_b.weight")      ||
            ends(kv.first, ".attn_output.weight")   ||
            ends(kv.first, ".ffn_gate.weight")      ||
            ends(kv.first, ".ffn_up.weight")        ||
            ends(kv.first, ".ffn_down.weight")      ||
            ends(kv.first, ".ffn_gate_inp.weight")  ||
            ends(kv.first, ".ffn_gate_shexp.weight")||
            ends(kv.first, ".ffn_up_shexp.weight")  ||
            ends(kv.first, ".ffn_down_shexp.weight")||
            kv.first == "output.weight";

        if (!supported)
            continue;

        named.emplace_back(kv.first, t);
    }

    std::sort(named.begin(), named.end(), [](auto &a, auto &b){ return a.first < b.first; });

    for (auto & kv : named)
    {
        ggml_tensor * t = kv.second;
        if (collect_seen.insert(t).second)
            collected_order.push_back(t);
    }

    //LLAMA_LOG_INFO("%s: model->tensors_by_name.size() == %d\n", __func__, model->tensors_by_name.size());
    //LLAMA_LOG_INFO("%s: named.size() == %d\n", __func__, named.size());
    LLAMA_LOG_INFO("%s: found %d host-backed weights\n", __func__, collected_order.size());
}

void parameter_offloader::init(
    ggml_backend_buffer_t   arena,
    llama_context_params    cparams,
    ggml_context          * ctx_twins,
    llama_context         * lctx)
{
    // Move this to constructor?
    {
        this->arena         = arena;
        this->ctx_gpu_twins = ctx_twins;

        buft  = ggml_backend_buffer_get_type(arena);
        base  = (char*) ggml_backend_buffer_get_base(arena);
        cap   = ggml_backend_buffer_get_size(arena);
        align = ggml_backend_buffer_get_alignment(arena);
        cur_off = 0;

        // Optional: reserve to avoid rehash during init
        gpu2cpu.reserve(4096);
        cpu2gpu.reserve(4096);
    }

#ifndef LLAMA_NAIVE_OFFLOADER
    seed_all_weights_from_model();
#endif

    const size_t packed = transform_all_cpu_weights_to_device_layout();
    LLAMA_LOG_INFO("host-packing: %zu/%zu weights packed on host\n",
                packed, collected_order.size());

    for (ggml_tensor * w_cpu : collected_order)
        (void) init_cpu_tensor_to_arena(w_cpu);

    //print_model_tensor_stats(model);

    //re-target the pointers so that the tensors are right-justified in the arena
    //if (false)
    {
        const size_t tensor_count = schedule_current.gpu_tensors_in_order.size();
        if (tensor_count > 2)
        {
            // collect starts/lengths once
            std::vector<size_t> start(tensor_count);
            std::vector<size_t> len(tensor_count);
            std::vector<size_t> endv(tensor_count);
            for (size_t i = 0; i < tensor_count; ++i)
            {
                ggml_tensor *tg = schedule_current.gpu_tensors_in_order[i];
                ggml_tensor *tc = gpu2cpu.at(tg);
                start[i] = (size_t)((char*)tg->data - base);
                len  [i] = ggml_backend_buft_get_alloc_size(buft, tc);
                endv [i] = start[i] + len[i];
            }

            // locate final wrap -> last generation = [last_cut .. N-1]
            int last_cut = -1;
            for (size_t i = 1; i < tensor_count; ++i)
                if (start[i] < start[i - 1])
                    last_cut = (int)i;

            int k_begin = (last_cut == -1) ? 0 : last_cut;
            int tail_cnt = (int)tensor_count - k_begin;

            if (tail_cnt == 1)
            {
                const int k    = k_begin;
                const int prev = (k - 1 + (int)tensor_count) % (int)tensor_count;
                const int next = (k + 1) % (int)tensor_count;

                auto overlaps = [](size_t a0, size_t a1, size_t b0, size_t b1) -> bool {
                    return !(a1 <= b0 || b1 <= a0);
                };

                const size_t sz = len[k];

                // mid arena, aligned, clamped
                size_t cand = align_up((cap > sz ? (cap/2 - sz/2) : 0), align);
                if (cand + sz > cap)
                    cand = cap - sz;

                const size_t step = align ? align : 1;
                const size_t max_tries = cap / step + 2;

                for (size_t tries = 0; tries < max_tries; ++tries)
                {
                    const size_t a0 = cand;
                    const size_t a1 = cand + sz;
                    const bool clash =
                        overlaps(a0, a1, start[prev], endv[prev]) ||
                        overlaps(a0, a1, start[next], endv[next]);

                    bool unique = true;
                    if (!clash)
                        for (size_t j = 0; j < tensor_count; ++j) {
                            if ((int)j == k)
                                continue;
                            if (start[j] == a0)
                            { 
                                unique = false;
                                break;
                            }
                        }

                    if (!clash && unique)
                    {
                        ggml_tensor *w_gpu = schedule_current.gpu_tensors_in_order[k];
                        w_gpu->data = base + a0;
                        ggml_backend_buffer_init_tensor(arena, w_gpu);
                        start[k] = a0; endv[k] = a1;
                        break;
                    }

                    cand += step;
                    if (cand + sz > cap)
                        cand = 0; // wrap search
                }
            }
        }
    }

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

    // Optional log
    size_t peak = 0;
    if (!schedule_current.end.empty())
        peak = *std::max_element(schedule_current.end.begin(), schedule_current.end.end());
    LLAMA_LOG_INFO("%s: vram-offload: scheduled %zu tensors; peak logical occupancy ~%zu bytes\n", __func__, tensor_count, peak);
}
parameter_offloader::~parameter_offloader()
{
    stop_streamer_join();
    if (ctx_gpu_twins) {
        ggml_free(ctx_gpu_twins);
        ctx_gpu_twins = nullptr;
    }
    if (arena) {
        ggml_backend_buffer_free(arena);
        arena = nullptr;
    }
    for (auto b : owned_host_buffers_)
        if (b)
            ggml_backend_buffer_free(b);
    owned_host_buffers_.clear();
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

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(dev);
    ggml_backend_buffer_type_t dev_buft = ggml_backend_dev_buffer_type(dev);

    const size_t logical   = ggml_nbytes(w_cpu);
    const size_t dev_bytes = ggml_backend_buft_get_alloc_size(dev_buft, w_cpu);

    // Allocate a host RAM buffer sized like the device allocation so we can memcpy the whole region later.
    ggml_backend_buffer_type_t host_buft = ggml_backend_cpu_buffer_type(); // use pinned-host type if you have one
    ggml_backend_buffer_t host_buf = ggml_backend_buft_alloc_buffer(host_buft, dev_bytes);
    GGML_ASSERT(host_buf);
    uint8_t * host_base = (uint8_t *) ggml_backend_buffer_get_base(host_buf);

    // Create a temporary device tensor to invoke the backend’s packing path
    ggml_backend_buffer_t tmp_dev_buf = ggml_backend_buft_alloc_buffer(dev_buft, dev_bytes);
    GGML_ASSERT(tmp_dev_buf);

    ggml_init_params tmp_ip{ 64*1024, nullptr, true };
    ggml_context * tmp_ctx = ggml_init(tmp_ip);
    GGML_ASSERT(tmp_ctx);

    ggml_tensor * tmp_dev = ggml_dup_tensor_layout_public(tmp_ctx, w_cpu);
    GGML_ASSERT(tmp_dev);
    GGML_ASSERT(ggml_backend_tensor_alloc(tmp_dev_buf, tmp_dev,
                ggml_backend_buffer_get_base(tmp_dev_buf)) == GGML_STATUS_SUCCESS);

    // H2D: this call triggers CUDA-side transform/packing for the tensor layout
    ggml_backend_tensor_set(tmp_dev, w_cpu->data, 0, logical);

    // D2H: read back ONLY the logical payload (tensor_get is bounded by ggml_nbytes())
    ggml_backend_tensor_get(tmp_dev, host_base, 0, logical);

    // Zero-fill the padded tail so later H2D memcpy can copy dev_bytes safely
    if (dev_bytes > logical) {
        std::memset(host_base + logical, 0, dev_bytes - logical);
    }

    // Cleanup temps
    ggml_free(tmp_ctx);
    ggml_backend_buffer_free(tmp_dev_buf);

    // Remember the packed bytes for this weight
    host_packed_.emplace(w_cpu, PackedHostBytes{ host_buf, host_base, dev_bytes });
    return true;
}


// For the given model, replace the cpu weight w_cpu pointer with a pointer to w_gpu
// This should be called on each cpu weight that needs to be pointer to a gpu weight ONCE on init
// We do this by brute force checking each weight member in the model, until we find a match to update
void patch_model_refs_for(llama_model * model, ggml_tensor * w_cpu, ggml_tensor * w_gpu) {
    // same matching rule you used: pointer match, else name match
    const char * cname = ggml_get_name(w_cpu);

    auto equal = [&](ggml_tensor * t) -> bool {
        if (t == w_cpu) return true;
        if (!cname || !t) return false;
        const char * tname = ggml_get_name(t);
        return tname && std::strcmp(tname, cname) == 0;
    };

    auto SET = [&](ggml_tensor *& slot) {
        if (equal(slot)) slot = w_gpu;
    };

    // -------------------
    // top-level (model)
    // -------------------
    SET(model->tok_embd);
    SET(model->type_embd);
    SET(model->pos_embd);
    SET(model->tok_norm);
    SET(model->tok_norm_b);

    SET(model->output_norm);
    SET(model->output_norm_b);
    SET(model->output);
    SET(model->output_b);
    SET(model->output_norm_enc);

    SET(model->cls);
    SET(model->cls_b);
    SET(model->cls_out);
    SET(model->cls_out_b);

    SET(model->conv1d);
    SET(model->conv1d_b);

    // -------------------
    // per-layer
    // -------------------
    const int nl = (int) model->hparams.n_layer;
    for (int il = 0; il < nl; ++il) {
        auto & L = model->layers[il];

        // normalization
        SET(L.attn_norm);        SET(L.attn_norm_b);
        SET(L.attn_norm_2);      SET(L.attn_norm_2_b);
        SET(L.attn_q_norm);      SET(L.attn_q_norm_b);
        SET(L.attn_k_norm);      SET(L.attn_k_norm_b);
        SET(L.attn_out_norm);    SET(L.attn_out_norm_b);
        SET(L.attn_q_a_norm);    SET(L.attn_kv_a_norm);
        SET(L.attn_sub_norm);    SET(L.attn_post_norm);
        SET(L.ffn_sub_norm);     SET(L.attn_norm_cross);
        SET(L.attn_norm_enc);

        // attention weights
        SET(L.wq);     SET(L.wk);     SET(L.wv);     SET(L.wo);
        SET(L.wqkv);   SET(L.wq_a);   SET(L.wq_b);   SET(L.wkv_a_mqa);
        SET(L.wkv_b);  SET(L.wk_b);   SET(L.wv_b);
        SET(L.wq_cross);  SET(L.wk_cross);  SET(L.wv_cross);  SET(L.wo_cross);
        SET(L.wq_enc);    SET(L.wk_enc);    SET(L.wv_enc);    SET(L.wo_enc);

        // attention bias & relpos
        SET(L.bq); SET(L.bk); SET(L.bv); SET(L.bo); SET(L.bqkv);
        SET(L.attn_rel_b);
        SET(L.attn_rel_b_enc);
        SET(L.attn_rel_b_cross);

        // ffn core
        SET(L.ffn_gate);
        SET(L.ffn_down);
        SET(L.ffn_up);
        SET(L.ffn_gate_enc);
        SET(L.ffn_down_enc);
        SET(L.ffn_up_enc);

        // ffn MoE
        SET(L.ffn_gate_inp);
        SET(L.ffn_gate_exps);
        SET(L.ffn_down_exps);
        SET(L.ffn_up_exps);

        // ffn shared expert
        SET(L.ffn_gate_inp_shexp);
        SET(L.ffn_gate_shexp);
        SET(L.ffn_down_shexp);
        SET(L.ffn_up_shexp);

        // ffn extras / bias
        SET(L.ffn_norm);   SET(L.ffn_norm_b);
        SET(L.ffn_post_norm);
        SET(L.layer_out_norm); SET(L.layer_out_norm_b);
        SET(L.ffn_norm_exps);  SET(L.ffn_norm_enc);
        SET(L.ffn_gate_b);
        SET(L.ffn_down_b);
        SET(L.ffn_up_b);
        SET(L.ffn_act);
        SET(L.ffn_exp_probs_b);

        // mamba proj
        SET(L.ssm_in);  SET(L.ssm_x);  SET(L.ssm_dt);  SET(L.ssm_out);

        // mamba core/bias
        SET(L.ssm_conv1d);  SET(L.ssm_a);  SET(L.ssm_d);
        SET(L.ssm_conv1d_b); SET(L.ssm_dt_b);

        // RWKV / RWKV7 family
        SET(L.time_mix_w1); SET(L.time_mix_w2);
        SET(L.time_mix_lerp_x); SET(L.time_mix_lerp_w);
        SET(L.time_mix_lerp_k); SET(L.time_mix_lerp_v);
        SET(L.time_mix_lerp_r); SET(L.time_mix_lerp_g);
        SET(L.time_mix_lerp_fused);

        SET(L.time_mix_first); SET(L.time_mix_decay);
        SET(L.time_mix_decay_w1); SET(L.time_mix_decay_w2);
        SET(L.time_mix_key);  SET(L.time_mix_key_b);
        SET(L.time_mix_value); SET(L.time_mix_value_b);
        SET(L.time_mix_receptance); SET(L.time_mix_receptance_b);
        SET(L.time_mix_gate);

        SET(L.time_mix_w0);
        SET(L.time_mix_a0); SET(L.time_mix_a1); SET(L.time_mix_a2);
        SET(L.time_mix_v0); SET(L.time_mix_v1); SET(L.time_mix_v2);
        SET(L.time_mix_g1); SET(L.time_mix_g2);
        SET(L.time_mix_k_k); SET(L.time_mix_k_a); SET(L.time_mix_r_k);

        SET(L.time_mix_ln);   SET(L.time_mix_ln_b);
        SET(L.time_mix_output);

        SET(L.channel_mix_lerp_k);
        SET(L.channel_mix_lerp_r);
        SET(L.channel_mix_key);
        SET(L.channel_mix_receptance);
        SET(L.channel_mix_value);

        // rope & bitnet scales
        SET(L.rope_long); SET(L.rope_short); SET(L.rope_freqs);
        SET(L.wq_scale); SET(L.wk_scale); SET(L.wv_scale); SET(L.wo_scale);
        SET(L.ffn_gate_scale); SET(L.ffn_up_scale); SET(L.ffn_down_scale);
    }

    // keep the name map coherent too
    if (cname) {
        for (auto & kv : model->tensors_by_name) {
            if (kv.second == w_cpu || kv.first == cname) {
                kv.second = w_gpu;
                // don't break: same name can appear multiple times
            }
        }
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

ggml_tensor * parameter_offloader::init_cpu_tensor_to_arena(ggml_tensor * w_cpu)
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
    const size_t slot_bytes = ggml_backend_buft_get_alloc_size(buft, w_cpu);
    size_t off              = align_up(cur_off, align);
    
    if (off + slot_bytes > cap)
        off = 0; // wrap

    // starting from current 'off' (possibly just wrapped to 0), bump until unused
    const size_t bump      = align;                     // step by arena alignment
    const size_t max_tries = cap / align + 2;          // safety bound
    size_t tries = 0;
    while (std::any_of(gpu2cpu.begin(), gpu2cpu.end(),
                [&](const auto &kv) { return kv.first && kv.first->data == static_cast<void*>(base + off); }))
    {
        off = align_up(off + bump, align);
        if (off + slot_bytes > cap)
            off = 0; // wrap again if we ran past the end
        if (++tries > max_tries) {
            LLAMA_LOG_WARN("arena: could not find unique pointer for '%s' "
                        "(cap=%zu, align=%zu, entries=%zu) — proceeding with overlap\n",
                        ggml_get_name(w_cpu), cap, align, gpu2cpu.size());
            break; // fall through; last 'off' may collide but we’ve warned
        }
    }

    // Duplicate tensor metadata into the GPU-twins context (no data yet)
    ggml_tensor* w_gpu = ggml_dup_tensor_layout_public(ctx_gpu_twins, w_cpu);
    GGML_ASSERT(w_gpu);
    ggml_set_name(w_gpu, ggml_get_name(w_cpu)); // keep names consistent (optional)

    // Bind GPU twin into the arena at [base + off]
    GGML_ASSERT(ggml_backend_tensor_alloc(arena, w_gpu, base + off) == GGML_STATUS_SUCCESS);

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
    if ((int)schedule_current.ready_after.size() <= idx)
        schedule_current.ready_after.resize(idx + 1, INT_MAX); // fill later

    // Bump arena pointer
    cur_off = off + slot_bytes;

    patch_model_refs_for(model, w_cpu, w_gpu);

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

inline ggml_cuda_copy_event * parameter_offloader::upload_weight_auto(ggml_tensor *w_cpu, ggml_tensor *w_gpu) {
    GGML_ASSERT(w_cpu && w_gpu);
    GGML_ASSERT(ggml_backend_buffer_is_host(w_cpu->buffer));
    GGML_ASSERT(w_gpu->buffer == arena);

    auto it = host_packed_.find(w_cpu);
    if (it != host_packed_.end()) {
        if (ggml_backend_buffer_is_cuda_arena_public(arena)) {
            const size_t dev_bytes = ggml_backend_buft_get_alloc_size(buft, w_cpu);
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

void parameter_offloader::retarget_schedule_tensors(offloader_schedule & schedule)
{
    const size_t tensor_count = schedule.gpu_tensors_in_order.size(); // number of existing GPU twins to retarget

    if (tensor_count == 0)
        return;

    const size_t a = align ? align : 1; // alignment used for arena start offsets

    size_t cur = 0; // next candidate arena offset

    std::vector<size_t> used_starts; // start offsets already assigned in this retarget pass
    used_starts.reserve(tensor_count);

    std::vector<size_t> start(tensor_count); // arena-relative start offset for each scheduled tensor
    std::vector<size_t> len(tensor_count); // backend padded allocation size for each scheduled tensor
    std::vector<size_t> endv(tensor_count); // arena-relative end offset for each scheduled tensor

    for (size_t i = 0; i < tensor_count; ++i)
    {
        ggml_tensor * w_gpu = schedule.gpu_tensors_in_order[i]; // existing GPU twin whose data pointer will move
        ggml_tensor * w_cpu = schedule.cpu_tensors_in_order[i]; // CPU weight used only to compute padded device size

        GGML_ASSERT(w_gpu);
        GGML_ASSERT(w_cpu);
        GGML_ASSERT(w_gpu->buffer == arena);

        const size_t slot_bytes = ggml_backend_buft_get_alloc_size(buft, w_cpu); // padded device bytes for this tensor
        GGML_ASSERT(slot_bytes <= cap);

        size_t off = align_up(cur, a); // candidate arena-relative start offset

        if (off + slot_bytes > cap)
            off = 0;

        const size_t max_tries = cap / a + 2; // bound for unique-start search
        size_t tries = 0; // number of alternate starts tried

        while (std::find(used_starts.begin(), used_starts.end(), off) != used_starts.end())
        {
            off = align_up(off + a, a);

            if (off + slot_bytes > cap)
                off = 0;

            GGML_ASSERT(++tries <= max_tries);
        }

        w_gpu->data = base + off;
        ggml_backend_buffer_init_tensor(arena, w_gpu); // refresh backend tensor metadata after moving the arena pointer

        used_starts.push_back(off);

        start[i] = off;
        len[i] = slot_bytes;
        endv[i] = off + slot_bytes;

        cur = off + slot_bytes;
    }

    if (tensor_count <= 2)
        return;

    int last_cut = -1; // first index of the last wrapped placement generation

    for (size_t i = 1; i < tensor_count; ++i)
        if (start[i] < start[i - 1])
            last_cut = (int)i;

    const int k_begin = (last_cut == -1) ? 0 : last_cut; // first tensor in the final placement generation
    const int tail_cnt = (int)tensor_count - k_begin; // number of tensors in the final placement generation

    if (tail_cnt != 1)
        return;

    const int k = k_begin; // singleton tail tensor index to relocate
    const int prev = (k - 1 + (int)tensor_count) % (int)tensor_count; // previous schedule index around the ring
    const int next = (k + 1) % (int)tensor_count; // next schedule index around the ring

    auto overlaps = [](size_t a0, size_t a1, size_t b0, size_t b1) -> bool {
        return !(a1 <= b0 || b1 <= a0);
    };

    const size_t sz = len[k]; // padded size of singleton tail tensor

    size_t cand = align_up((cap > sz ? (cap / 2 - sz / 2) : 0), a); // middle-ish candidate arena offset

    if (cand + sz > cap)
        cand = cap - sz;

    const size_t step = a; // relocation search step
    const size_t max_tries = cap / step + 2; // relocation search bound

    for (size_t tries = 0; tries < max_tries; ++tries)
    {
        const size_t a0 = cand; // candidate relocated start offset
        const size_t a1 = cand + sz; // candidate relocated end offset

        const bool clash =
            overlaps(a0, a1, start[prev], endv[prev]) ||
            overlaps(a0, a1, start[next], endv[next]);

        bool unique = true; // true if no other tensor has this exact start offset

        if (!clash)
        {
            for (size_t j = 0; j < tensor_count; ++j)
            {
                if ((int)j == k)
                    continue;

                if (start[j] == a0)
                {
                    unique = false;
                    break;
                }
            }
        }

        if (!clash && unique)
        {
            ggml_tensor * w_gpu = schedule.gpu_tensors_in_order[k]; // singleton tail GPU tensor to relocate

            w_gpu->data = base + a0;
            ggml_backend_buffer_init_tensor(arena, w_gpu); // refresh backend tensor metadata after moving the arena pointer

            start[k] = a0;
            endv[k] = a1;

            break;
        }

        cand += step;

        if (cand + sz > cap)
            cand = 0;
    }
}

// Return true if node 't' reads any tracked GPU twin; optionally output the
// feed-order index you should advance to (choose the furthest-ahead in ring).
bool parameter_offloader::node_reads_tracked_weight(ggml_tensor * t, int * out_idx = nullptr)
{
    int best_idx = -1;
    const int N  = (int)schedule_current.gpu_tensors_in_order.size();

    // quick filter by op to reduce callback overhead
    //switch (t->op) {
    //    case GGML_OP_MUL_MAT:
    //    case GGML_OP_ADD:      // only if you mirrored bias tensors
    //        break;
    //    default:
    //        return false;
    //}
    //TODO: should we re-add something here? What was the old code?

    // scan node sources
    for (int k = 0; k < GGML_MAX_SRC; ++k)
    {
        ggml_tensor * s = t->src[k];
        if (!s)
            break;

        while (s->view_src)
            s = s->view_src;

        //if (ggml_n_dims(s) < 2)
        //    continue; // skip vectors/scalars that are not matmul-style weights

        auto it = schedule_current.gpu2index.find(s);
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

    if (best_idx >= 0 && out_idx)
        *out_idx = best_idx;
    return best_idx >= 0;
}

// Ask-phase: only opt in for nodes that read any tracked weight.
// Keeps batching intact for other nodes.
bool parameter_offloader::wants_observe(ggml_tensor * node)
{
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

            // 2) Does this pointer fall inside your arena’s [base, base+cap)?
            char *q = (char*) src_node->data;
            if (q && q >= base && q < base + cap) {
                LLAMA_LOG_WARN("[ACT-IN-WEIGHT-RANGE] node=%s tens=%s ptr=%p (inside [%p, %p))\n",
                            ggml_get_name(node)?: "(unnamed)",
                            ggml_get_name(src_node)?: "(unnamed)", src_node->data, base, base + cap);
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


// Called when a node we opted into observing is actually executed.
// Pick the “latest” managed weight used by this node and advance your
// tensor_idx_used_mod/tensor_idx_used_epoch/tensor_idx_used_seq just like you do now.
bool parameter_offloader::on_eval_tensor(ggml_tensor * node)
{
    int idx = -1;
    if (!node_reads_tracked_weight(node, &idx))
        return false;  //should this be a return true?

    // for readable logs
    const int tensor_count = (int)schedule_current.gpu_tensors_in_order.size();
    //if (idx == tensor_count - 1)
    //    LLAMA_LOG_INFO("%s got to idx %d\n", __func__, idx);

    // Resolve the GPU twin + its CPU source so we can print name/size/offset
    ggml_tensor * w_gpu = schedule_current.gpu_tensors_in_order[idx];
    ggml_tensor * w_cpu = gpu2cpu[w_gpu];

    const char * name   = ggml_get_name(w_gpu);
    const size_t off    = (size_t)((char *) w_gpu->data - base);
    const size_t bytes  = ggml_backend_buft_get_alloc_size(buft, w_cpu);
    const size_t nbytes = ggml_nbytes(w_cpu);

    const long long old_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_relaxed);
    const long long used_ordinal = advance_ordinal_to_idx(old_used_ordinal, idx, tensor_count);

    tensor_idx_used_ordinal.store(used_ordinal, std::memory_order_release);

    // wake streamer
    node_cv_.notify_all();

    // Verbose, but super handy while tuning:
#ifdef LLAMA_LOG_READS
    LLAMA_LOG_INFO("[R.%d]", idx);
#endif

#if defined(LLAMA_DIAGNOSE_COPY)
    auto ring_dist = [&](int from, int to) -> int {
        return (to - from + tensor_count) % tensor_count; // forward distance in ring
    };

    long long copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_relaxed);

    for (;;)
    {
        // snapshot the current reader ordinal
        long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);
        const int i = ordinal_mod(copied_ordinal + 1, tensor_count);

        {
            const int r_idx = ordinal_mod(cur_used_ordinal, tensor_count); // current reader index
            const int bar   = schedule_current.ready_after[r_idx];         // last copyable index while r is read
            const int di    = ring_dist(r_idx, i);                         // distance from reader to copy slot
            const int dbar  = ring_dist(r_idx, bar);                       // distance from reader to barrier

            bool allowed = (di <= dbar) && (di <= LLAMA_DIAGNOSE_COPY);

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
    
    long long cur_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
    const long long needed_copy_ordinal = used_ordinal + 1; // +1 because we're in post-compute and need the next one

    if (cur_copied_ordinal < needed_copy_ordinal)
    {
    #if LLAMA_LOG_READS > 1
        LLAMA_LOG_INFO("[RB.%d.%lld.%lld.%lld]",
                       idx, cur_copied_ordinal, needed_copy_ordinal, used_ordinal);
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

#ifdef LLAMA_CHECK_WEIGHTS
    //hash_compare_tensor(w_cpu, w_gpu, base, idx);
    //if (idx == 606)
    {
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

            // Peel views for clarity (optional)
            ggml_tensor * pw = w;
            while (pw && pw->view_src)
                pw = pw->view_src;

            ggml_tensor * px = x;
            while (px && px->view_src)
                px = px->view_src;

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

void parameter_offloader::start_streamer() {
    stop_stream.store(false, std::memory_order_release);
    copy_thread = std::thread(&parameter_offloader::stream_worker, this);
}
void parameter_offloader::stop_streamer_join() {
    stop_stream.store(true, std::memory_order_release);
    node_cv_.notify_all();
    if (copy_thread.joinable())
        copy_thread.join();
}

void parameter_offloader::stream_worker()
{
    LLAMA_LOG_INFO("%s started\n", __func__);

    long long submitted_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
    uint64_t submitted_generation = schedule_generation.load(std::memory_order_acquire);

    for (;;)
    {
        if (stop_stream.load(std::memory_order_acquire))
            return;

        ggml_tensor * w_cpu = nullptr;        // CPU source selected for this copy
        ggml_tensor * w_gpu = nullptr;        // GPU arena tensor selected for this copy
        long long wait_used_ordinal = -1;     // reader ordinal that blocked this copy
        uint64_t wait_generation = 0;         // schedule generation observed before blocking
        bool allowed = false;                 // true when selected copy is inside safe window
        int copy_idx = -1;                    // schedule index selected for copy. TODO: Why do we have this function? When its read doesn't it always equal i? Seems redundant, consider removing

        {
            std::lock_guard<std::mutex> schedule_lk(schedule_mu); // blocks schedule swap during copy selection/upload

            const int tensor_count = (int)schedule_current.gpu_tensors_in_order.size(); // active schedule size
            if (tensor_count == 0)
                return;

            const uint64_t cur_generation = schedule_generation.load(std::memory_order_acquire);
            if (cur_generation != submitted_generation) {
                submitted_generation = cur_generation;
                submitted_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire);
            }

            auto ring_dist = [&](int from, int to) -> int {
                return (to - from + tensor_count) % tensor_count; // forward distance in active ring
            };
            //We load from the atomic variable because swap_next_schedule can change this
            const long long last_submitted_ordinal = submitted_ordinal;                                   // last submitted ordinal
            const long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);   // last reader ordinal

            const bool startup = cur_used_ordinal < 0; // true before the first graph read

            const int i = ordinal_mod(last_submitted_ordinal + 1, tensor_count); // next copy index
            const int r_idx = startup ? tensor_count - 1 : ordinal_mod(cur_used_ordinal, tensor_count); // reader or virtual startup index
            const int bar   = schedule_current.ready_after[r_idx];
            const int di    = ring_dist(r_idx, i);   // distance from reader/startup to copy slot
            const int dbar  = ring_dist(r_idx, bar); // distance from reader/startup to barrier

            GGML_ASSERT(bar >= 0);

            allowed = (di <= dbar);

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
                copy_idx = i;
                w_cpu = schedule_current.cpu_tensors_in_order[i];
                w_gpu = schedule_current.gpu_tensors_in_order[i];

                const long long ordinal = submitted_ordinal + 1;
                const uint64_t generation = submitted_generation;

                ggml_cuda_copy_event * ev = upload_weight_auto(w_cpu, w_gpu);

                submitted_ordinal = ordinal;

                if (ev)
                    std::thread(&parameter_offloader::publish_copy_when_ready, this, ordinal, generation, ev).detach();
                else
                    std::thread(&parameter_offloader::publish_copy_now, this, ordinal, generation).detach();
            }
        }

        if (!allowed)
        {
            std::unique_lock<std::mutex> lk(node_mu_);
            node_cv_.wait(lk, [&]{
                return stop_stream.load(std::memory_order_acquire) ||
                    tensor_idx_used_ordinal.load(std::memory_order_acquire) != wait_used_ordinal ||
                    schedule_generation.load(std::memory_order_acquire) != wait_generation;
            });

            if (stop_stream.load(std::memory_order_acquire))
                return;

            continue;
        }
    }
}


void parameter_offloader::print_snapshot(offloader_schedule & schedule)
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

        const size_t off   = (size_t)((char*)t_gpu->data - base);              // arena-relative
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t_cpu);    // padded size

        start[i] = off;
        end[i]   = off + bytes;
        GGML_ASSERT(end[i] <= cap);
    }

    for (int i = 0; i < tensor_count; ++i)
    {
        ggml_tensor * w_gpu  = schedule.gpu_tensors_in_order[i]; // GPU tensor being printed
        const char * name    = ggml_get_name(w_gpu); // mirrored model tensor name
        const char * op_name = ggml_op_name(w_gpu->op); // ggml op name without the GGML_OP_ prefix

        LLAMA_LOG_INFO("%s %4d %4d %10lu %10lu %5d %-24s %s\n",
            __func__,
            i,
            schedule.ready_after[i],
            start[i],
            end[i],
            schedule.ready_after[i] - i,
            op_name ? std::string("GGML_OP_").append(op_name).c_str() : "GGML_OP_UNKNOWN",
            name ? name : "(unnamed)");

        //LLAMA_LOG_INFO("\"%s\",\n", name);
    }

}



// signature must match ggml_backend_sched_eval_callback
bool llama_offloader_eval_cb(ggml_tensor * node, bool ask, void * ud)
{
    struct parameter_offloader * po = static_cast<parameter_offloader *>(ud);
    if (!po)
        return true;

    // ---- PHASE A: order collection (ready == false) ----
    if (!po->ready)
    {
#ifdef LLAMA_NAIVE_OFFLOADER
        //Naive weight collection. If LLAMA_NAIVE_OFFLOADER is true then initial weight collection is done here

        // Only observe matmuls during warm-up to keep batching intact
        if (ask)
            return node->op == GGML_OP_MUL_MAT;

        // find the real model weight among sources
        ggml_tensor * w = nullptr;
        for (int k = 0; k < GGML_MAX_SRC; ++k)
        {
            ggml_tensor * s = node->src[k];
            if (!s)
                break;
            ggml_tensor * p = s;
            while (p->view_src)
                p = p->view_src; // peel views

            if (po->cpu_weight_set.find(p) == po->cpu_weight_set.end())
                continue; // must be a model weight
            if (!(p->buffer && ggml_backend_buffer_is_host(p->buffer)))
                continue; // on host
            w = p;
            break;
        }
        if (!w)
            return true; // nothing to record

        if (po->collect_seen.insert(w).second) {
            po->collected_order.push_back(w);
        }
        return true;
#else
        if (!po)
            return !ask;
#endif
    }

    //LLAMA_LOG_INFO("%s before ask == %d\n", __func__, ask);

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

// Choose how strict you want the selection to be:
//
// 0 -> Walk the whole graph in order; collect weights from *all* nodes (simplest, no internal deps)
// 1 -> Walk only the splits whose backend buffer type matches your arena buffer type (better if you
//      want to limit to the GPU backend you’re streaming into)

//#define PO_GRAPH_FILTER_BY_BACKEND 1

bool llama_offloader_graph_cb(ggml_backend_sched_t sched, struct ggml_cgraph * graph, void * ud)
{
#ifndef LLAMA_NAIVE_OFFLOADER
    struct parameter_offloader *po = static_cast<parameter_offloader *>(ud);
    if (!po || !sched || !graph)
        return true;

    //LLAMA_LOG_INFO("%s po->ready == %d\n", __func__, po->ready);

    if (!po->ready)
        return true;

    // Reset & (re)build the order from this graph
    po->schedule_next = parameter_offloader::offloader_schedule{};
    po->collect_seen.clear();

#if PO_GRAPH_FILTER_BY_BACKEND > 0
    // Optional precise mode: collect only tracked weight reads from scheduler splits
    // assigned to the arena's backend buffer type.

    // We will walk the scheduler "splits" and only consider the ones that will execute
    // on the same *buffer type* as the offloader's arena (e.g., CUDA arena).
    // NOTE: requires "ggml-backend-impl.h" so we can access sched internals.
    const ggml_backend_buffer_type_t target_buft =
        ggml_backend_buffer_get_type(po->arena);

    for (int si = 0; si < sched->n_splits; ++si) {
        const ggml_backend_sched_split & sp = sched->splits[si];

        // Skip splits that will execute on a backend whose buffer type doesn't match our arena.
        // (This keeps us from scheduling copies for CPU-only splits, etc.)
        if (sched->bufts[sp.backend_id] != target_buft)
            continue;

        // Walk the original graph nodes covered by this split
        for (int j = sp.i_start; j < sp.i_end; ++j)
        {
            ggml_tensor * node = graph->nodes[j];
            if (!node)
                continue;

            // Skip pure view ops; they don't introduce new weight reads
            if (node->view_src)
                continue;

            //switch (node->op)
            //{
            //    case GGML_OP_MUL_MAT:
            //    case GGML_OP_ADD:
            //        break;
            //    default:
            //        continue;
            //}

            // Collect unique first-use weights from the node's sources
            for (int k = 0; k < GGML_MAX_SRC; ++k)
            {
                ggml_tensor * s = node->src[k];
                if (!s)
                    break;

                // Peel views so we hit the real weight tensor
                while (s->view_src)
                    s = s->view_src; // peel views
                
                //if (ggml_n_dims(s) < 2)
                //    continue; // skip vectors/scalars that are not matmul-style weights

                if (po->gpu_weight_set.find(s) == po->gpu_weight_set.end())
                    continue; // must be a model weight

                if (!po->collect_seen.insert(s).second)
                {
                    const char * duplicate_name = ggml_get_name(s);
                    throw std::runtime_error(std::string("duplicate weight in graph schedule: ") +
                                            (duplicate_name ? duplicate_name : "(unnamed)"));
                }

                const int idx = (int)po->schedule_next.gpu_tensors_in_order.size(); // candidate schedule index
                ggml_tensor * w_gpu = s;                                            // GPU twin referenced by graph
                ggml_tensor * w_cpu = po->gpu2cpu.at(w_gpu);                         // CPU source for this GPU twin

                po->schedule_next.gpu_tensors_in_order.push_back(w_gpu);
                po->schedule_next.cpu_tensors_in_order.push_back(w_cpu);
                po->schedule_next.gpu2index.emplace(w_gpu, idx);
            }
        }
    }

    // If nothing matched (e.g., you compiled out impl access), fall back to walking the full graph:
    if (po->schedule_next.gpu_tensors_in_order.empty())
#endif /* if PO_GRAPH_FILTER_BY_BACKEND > 0 */
    {
        // Simple, robust fallback: walk the whole graph in topo order
        for (int i = 0; i < graph->n_nodes; ++i)
        {
            ggml_tensor * node = graph->nodes[i];
            if (!node)
                continue;
            
            if (node->view_src)
                continue;

            //switch (node->op)
            //{
            //    case GGML_OP_MUL_MAT:
            //    case GGML_OP_ADD:
            //        break;
            //    default:
            //        continue;
            //}

            for (int k = 0; k < GGML_MAX_SRC; ++k)
            {
                ggml_tensor * s = node->src[k];
                if (!s)
                    break;
                while (s->view_src)
                    s = s->view_src; // peel views

                //if (ggml_n_dims(s) < 2)
                //    continue; // skip vectors/scalars that are not matmul-style weights

                if (po->gpu_weight_set.find(s) == po->gpu_weight_set.end())
                    continue; // must be a model weight

                if (!po->collect_seen.insert(s).second)
                {
                    const char * duplicate_name = ggml_get_name(s);
                    //TODO: Should we be supporting duplicate weights in the graph schedule?
                    throw std::runtime_error(std::string("duplicate weight in graph schedule: ") + (duplicate_name ? duplicate_name : "(unnamed)"));
                }

                const int idx = (int)po->schedule_next.gpu_tensors_in_order.size(); // candidate schedule index
                ggml_tensor * w_gpu = s;                                            // GPU twin referenced by graph
                ggml_tensor * w_cpu = po->gpu2cpu.at(w_gpu);                         // CPU source for this GPU twin

                po->schedule_next.gpu_tensors_in_order.push_back(w_gpu);
                po->schedule_next.cpu_tensors_in_order.push_back(w_cpu);
                po->schedule_next.gpu2index.emplace(w_gpu, idx);
            }
        }
    }

    // Candidate schedule has been collected; compare it once against the active schedule.
    po->schedule_next_prefix = common_prefix_len(
        po->schedule_current.gpu_tensors_in_order,
        po->schedule_next.gpu_tensors_in_order);

    po->schedule_next_identical =
        po->schedule_next_prefix == po->schedule_current.gpu_tensors_in_order.size() &&
        po->schedule_next_prefix == po->schedule_next.gpu_tensors_in_order.size();

    po->schedule_next_valid = true;

    const bool schedule_changed = po->swap_next_schedule(); // true if retarget moved tensors for this graph

#endif /* ifndef LLAMA_NAIVE_OFFLOADER */
    return true;
}

bool parameter_offloader::swap_next_schedule()
{
    long long new_copied_ordinal = -1; // copied ordinal required before graph may start reading
    bool changed = false; // true when a new schedule was published

    {
        std::lock_guard<std::mutex> schedule_lk(schedule_mu); // blocks streamer while tensor pointers move

        GGML_ASSERT(schedule_next_valid);

        const bool identical = schedule_next_identical; // true if graph order did not change
        schedule_next_valid = false;

        //LLAMA_LOG_INFO("%s identical == %d\n", __func__, identical);
        if (identical)
        {
            schedule_next = offloader_schedule{};
            schedule_next_prefix = 0;
            schedule_next_identical = false;
            return false;
        }

        std::unordered_map<ggml_tensor *, size_t> old_offsets; // old arena offsets before retarget
        old_offsets.reserve(schedule_current.gpu_tensors_in_order.size());

        for (ggml_tensor * w_gpu : schedule_current.gpu_tensors_in_order)
        {
            GGML_ASSERT(w_gpu);
            GGML_ASSERT(w_gpu->data);
            old_offsets[w_gpu] = (size_t)((char *)w_gpu->data - base);
        }

        retarget_schedule_tensors(schedule_next);

        build_schedule_gates(schedule_next);

        const size_t prefix_limit = std::min(schedule_current.gpu_tensors_in_order.size(), schedule_next.gpu_tensors_in_order.size()); // max prefix to compare
        // prefix that kept same tensor identity and same arena offset
        for (schedule_next_prefix = 0; schedule_next_prefix < prefix_limit; ++schedule_next_prefix)
        {
            ggml_tensor * old_gpu = schedule_current.gpu_tensors_in_order[schedule_next_prefix]; // tensor in old schedule prefix
            ggml_tensor * new_gpu = schedule_next.gpu_tensors_in_order[schedule_next_prefix]; // tensor in new schedule prefix

            if (old_gpu != new_gpu)
                break;

            auto it = old_offsets.find(new_gpu); // old offset for this tensor before retarget

            if (it == old_offsets.end())
                break;

            const size_t new_offset = (size_t)((char *)new_gpu->data - base); // new offset after retarget

            if (it->second != new_offset)
                break;
        }

        //TODO: shouldn't old_used_ordinal always be -1, given that we only call swap after the full schedule is read?
        const long long old_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire); // last copied ordinal in old schedule
        const long long old_used_ordinal   = tensor_idx_used_ordinal.load(std::memory_order_acquire); // last used ordinal in old schedule

        long long copied_ahead = (old_used_ordinal < 0) ? old_copied_ordinal + 1 : old_copied_ordinal - old_used_ordinal; // copied entries available ahead of reader

        if (copied_ahead < 0)
            copied_ahead = 0;

        const long long reusable_copied = std::min((long long)schedule_next_prefix, copied_ahead); // preserved copied prefix count
        new_copied_ordinal = reusable_copied - 1; // copied ordinal after schedule swap

        std::swap(schedule_current, schedule_next);

        schedule_current.generation = schedule_generation.fetch_add(1, std::memory_order_relaxed) + 1;

        schedule_next = offloader_schedule{};
        schedule_next_prefix = 0;
        schedule_next_identical = false;

        tensor_idx_copied_ordinal.store(new_copied_ordinal, std::memory_order_release);
        tensor_idx_used_ordinal.store(-1, std::memory_order_release);

    #if defined(LLAMA_DIAGNOSE_COPY)
        const size_t tensor_count = schedule_current.gpu_tensors_in_order.size();

        auto ring_dist = [&](int from, int to) -> int {
            return (to - from + tensor_count) % tensor_count; // forward distance in ring
        };

        long long copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_relaxed);

        for (;;)
        {
            // snapshot the current reader ordinal
            long long cur_used_ordinal = tensor_idx_used_ordinal.load(std::memory_order_acquire);
            const int i = ordinal_mod(copied_ordinal + 1, tensor_count);

            {
                const int r_idx = ordinal_mod(cur_used_ordinal, tensor_count); // current reader index
                const int bar   = schedule_current.ready_after[r_idx];         // last copyable index while r is read
                const int di    = ring_dist(r_idx, i);                         // distance from reader to copy slot
                const int dbar  = ring_dist(r_idx, bar);                       // distance from reader to barrier

                bool allowed = (di <= dbar) && (di <= LLAMA_DIAGNOSE_COPY);

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

        changed = true;

        print_snapshot(schedule_current);
    }

    node_cv_.notify_all();

    //TODO: This must block if we don't have the first tensor copied into the new schedule. If the first schedule changed between schedules, this should always fire
#if !defined(LLAMA_DIAGNOSE_COPY)
    if (new_copied_ordinal < 0)
    {
    #if LLAMA_LOG_READS > 1
        const long long cur_copied_ordinal = tensor_idx_copied_ordinal.load(std::memory_order_acquire); // copied ordinal after publishing schedule
        LLAMA_LOG_INFO("[RB.%d.%lld.%lld.%lld]", 0, cur_copied_ordinal, new_copied_ordinal, (long long)-1);
    #endif

        std::unique_lock<std::mutex> lk(node_mu_);
        node_cv_.wait(lk, [&]{
            return stop_stream.load(std::memory_order_acquire) || tensor_idx_copied_ordinal.load(std::memory_order_acquire) >= 0;
        });
    }
#endif

    return changed;
}

void parameter_offloader::build_schedule_gates(offloader_schedule & schedule)
{
    const size_t tensor_count = schedule.gpu_tensors_in_order.size();

    GGML_ASSERT(schedule.cpu_tensors_in_order.size() == tensor_count);
    GGML_ASSERT(schedule.gpu2index.size() == tensor_count);

    schedule.ready_after.assign(tensor_count, -1); // post-read maximum copy barrier
    schedule.start.assign(tensor_count, 0);        // arena start offset for each GPU tensor
    schedule.end.assign(tensor_count, 0);          // arena end offset for each GPU tensor

    if (tensor_count == 0)
        return;

    const int N = (int)tensor_count;

    for (int i = 0; i < N; ++i)
    {
        ggml_tensor * t_gpu = schedule.gpu_tensors_in_order[i];
        ggml_tensor * t_cpu = schedule.cpu_tensors_in_order[i];

        GGML_ASSERT(t_gpu && t_gpu->data);
        GGML_ASSERT(t_cpu);

        const size_t off = (size_t)((char *)t_gpu->data - base); // arena-relative start
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t_cpu); // backend padded size

        schedule.start[i] = off;
        schedule.end[i] = off + bytes;

        GGML_ASSERT(schedule.end[i] <= cap);
    }

    auto overlaps_abs = [](size_t a0, size_t a1, size_t b0, size_t b1) -> bool {
        return !(a1 <= b0 || b1 <= a0);
    };

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
                if (overlaps_abs(range.first, range.second, schedule.start[copy_idx], schedule.end[copy_idx]))
                {
                    copy_is_safe = false;
                    break;
                }
            }

            if (!copy_is_safe)
                break;

            barrier = j;

            if (!ring_byte_ranges.empty() && ring_byte_ranges.back().second <= schedule.start[copy_idx])
            {
                ring_byte_ranges.back().second = schedule.end[copy_idx];
            }
            else
            {
                ring_byte_ranges.emplace_back(schedule.start[copy_idx], schedule.end[copy_idx]);
            }
        }

        schedule.ready_after[r] = barrier % N;
    }
}