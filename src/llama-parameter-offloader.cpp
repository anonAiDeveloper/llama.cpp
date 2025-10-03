#include "llama-parameter-offloader.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"


#include <algorithm>
#include <climits>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <math.h>       /* isfinite */

// add near top of file
#include <cstring>

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
//#define LLAMA_LOG_READS 1
//#define LLAMA_LOG_COPIES 1
/////////////////////////////////////

#include "llama-offloader-diagnostic.h"

static inline size_t align_up(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

parameter_offloader::parameter_offloader(llama_model  * model)
    : model(model)
{
    // Require DeepSeek-2
    //if (model->arch != LLM_ARCH_DEEPSEEK2) {
    //    LLAMA_LOG_WARN("%s: registry constructor: non-DeepSeek2 arch (%d) no ordering applied\n", __func__, (int)model->arch);
    //    return;
    //}

    weight_set.clear();
    weight_set.reserve(model->tensors_by_name.size());
    for (const auto & kv : model->tensors_by_name) {
        if (kv.second) weight_set.insert(kv.second);
    }
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

    // Mirror weights exactly in collected first-use order
    if (collected_order.empty())
        LLAMA_LOG_WARN("%s: no weights collected\n", __func__);
    else
        for (ggml_tensor * w_cpu : collected_order)
            (void) cpu_tensor_to_arena(w_cpu);

    //print_model_tensor_stats(model);

    //re-target the pointers so that the tensors are right-justified in the arena
    //if (false)
    {
        const size_t tensor_count = gpu_tensors_in_order.size();
        if (tensor_count > 2)
        {
            // collect starts/lengths once
            std::vector<size_t> start(tensor_count);
            std::vector<size_t> len(tensor_count);
            std::vector<size_t> endv(tensor_count);
            for (size_t i = 0; i < tensor_count; ++i)
            {
                ggml_tensor *tg = gpu_tensors_in_order[i];
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
                        ggml_tensor *w_gpu = gpu_tensors_in_order[k];
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

    const size_t tensor_count = gpu_tensors_in_order.size();
    ready_after.assign(tensor_count, -1);   // -1 => no barrier

    // Collect [start,end) in arena for each feed-order tensor
    std::vector<size_t> start(tensor_count);
    std::vector<size_t> end(tensor_count);
    start.reserve(tensor_count);
    end.reserve(tensor_count);

    for (size_t i = 0; i < tensor_count; ++i)
    {
        ggml_tensor *t_gpu = gpu_tensors_in_order[i];
        GGML_ASSERT(t_gpu && t_gpu->data);
        ggml_tensor *t_cpu = gpu2cpu.at(t_gpu);

        const size_t off   = (size_t)((char*)t_gpu->data - base);              // arena-relative
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t_cpu);    // padded size

        start[i] = off;
        end[i]   = off + bytes;
        GGML_ASSERT(end[i] <= cap);
    }
    auto overlaps_abs = [&](size_t a0, size_t a1, size_t b0, size_t b1) -> bool {
        // half-open absolute intervals [a0,a1), [b0,b1)
        return !(a1 <= b0 || b1 <= a0);
    };

    // compute “generation” per tensor (increments when placement wraps)
    std::vector<int> gen(tensor_count);
    gen[0] = 0;
    for (size_t i = 1; i < tensor_count; ++i) {
        gen[i] = gen[i - 1] + (start[i] < start[i - 1] ? 1 : 0);
    }
    const int last_gen = gen.empty() ? 0 : gen.back();
    // tail index for each generation (last occurrence)
    std::vector<int> tail_of_gen(last_gen + 1, -1);
    std::vector<int> head_of_gen(last_gen + 1, -1);
    for (int i = 0; i < (int)tensor_count; ++i)
    {
        tail_of_gen[gen[i]] = i;
        if (head_of_gen[gen[i]] == -1)
            head_of_gen[gen[i]] = i;
    }

    for (size_t i = 0; i < tensor_count; ++i)
    {
        int barrier = -1;
        int barrier_mod = -1;
        for (size_t j = i + 1; j < (tensor_count * 2); ++j)
        {
            int step = j;
            int step_mod = j % (int)tensor_count;
            if (overlaps_abs(start[i], end[i], start[step_mod], end[step_mod]))
                break;
            barrier = step;   //later, we will modulus this to within tensor_count
            barrier_mod = step_mod;
        }
        if (barrier == -1)
            barrier = barrier_mod = i % tensor_count;

        // trim-back rule: do not let barrier jump 2+ generations ahead
        {
            const int gi     = gen[(int)i];
            const int gj     = gen[barrier_mod];
            const int gnext  = (gi == last_gen) ? 0 : (gi + 1);

            // if the chosen barrier lives beyond the next generation, clamp to the tail of next gen
            if (gj != gi && gj != gnext) {
                const int tail_next = (gnext <= last_gen) ? tail_of_gen[gnext] : -1;
                if (tail_next != -1) {
                    // clamp to tail of next gen, but keep 'barrier' unwrapped relative to i
                    const int rd = (tail_next - (int)i + tensor_count) % tensor_count; // ring distance in [0..tensor_count-1]
                    barrier = (int)i + rd;
                    barrier_mod = tail_next;
                }
            }
        }

        ready_after[i] = barrier;
    }

    // Within each generation, barriers must be non-decreasing as we move forward
    int current_barrier = ready_after[tensor_count - 1];
    int current_gen = last_gen;
    for (int i = (int)tensor_count - 2; i >= 0; --i)
    {
        if (gen[i] != current_gen)
        {
            current_gen = gen[i];
        }
        else
        {
            if (ready_after[i] > current_barrier)
                ready_after[i] = current_barrier;
        }

        current_barrier = ready_after[i];
    }

    //Finally, modulus the barriers to be within tensor_count
    for (size_t i = 0; i < tensor_count; ++i)
        ready_after[i] = ready_after[i] % tensor_count;

    //////////////////////////////////////////////////////////////////
    //       FINISH UP
    //////////////////////////////////////////////////////////////////

    //tensor_idx_loaded.store(tensor_count - 1);
    tensor_idx_copied_mod.store(tensor_count - 1);
    tensor_idx_copied_seq.store(tensor_count - 1);
    
    tensor_idx_used_mod.store(tensor_count - 1);
    //tensor_idx_used_seq.store(tensor_idx_used_epoch*tensor_count + tensor_idx_used_mod);
    tensor_idx_used_seq.store(tensor_count - 1);

    ready = true;

    print_snapshot();

#if defined(LLAMA_DIAGNOSE_COPY)
    //streamer thread disabled during debug mode, so get things started with the first copy
    ggml_tensor *w_cpu = cpu_tensors_in_order[0];
    ggml_tensor *w_gpu = gpu_tensors_in_order[0];
    //ggml_backend_tensor_copy(w_cpu, w_gpu);
    copy_host_to_arena_with_transform(w_cpu, w_gpu);

    //this is not out of range, as its meant to go over to represent wrap-around
    //tensor_idx_loaded.store(tensor_count);
    tensor_idx_copied_mod.store(0);   //tensor_count % tensor_count
    tensor_idx_copied_epoch.store(1);
    tensor_idx_copied_seq.store(tensor_count);
#else
    start_streamer();                         // begin background H2D streaming
#endif

    // Optional log
    size_t peak = 0;
    if (!end.empty())
        peak = *std::max_element(end.begin(), end.end());
    LLAMA_LOG_INFO("vram-offload: scheduled %zu tensors; peak logical occupancy ~%zu bytes\n", tensor_count, peak);
}
parameter_offloader::~parameter_offloader() {
    stop_streamer_join();
    if (ctx_gpu_twins) {
        ggml_free(ctx_gpu_twins);
        ctx_gpu_twins = nullptr;
    }
    if (arena) {
        ggml_backend_buffer_free(arena);
        arena = nullptr;
    }
}

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

ggml_tensor * parameter_offloader::cpu_tensor_to_arena(ggml_tensor * w_cpu)
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

    // Always upload via transform-aware path (CPU -> stock CUDA tmp -> D2D into arena)
    copy_host_to_arena_with_transform(w_cpu, w_gpu);

    // Register mappings
    gpu2cpu.emplace(w_gpu, w_cpu);
    cpu2gpu.emplace(w_cpu, w_gpu);

    int idx = (int)gpu_tensors_in_order.size();
    cpu_tensors_in_order.push_back(w_cpu);
    gpu_tensors_in_order.push_back(w_gpu);
    gpu2index.emplace(w_gpu, idx);
    if ((int)ready_after.size() <= idx)
        ready_after.resize(idx + 1, INT_MAX); // fill later

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




// Return true if node 't' reads any tracked GPU twin; optionally output the
// feed-order index you should advance to (choose the furthest-ahead in ring).
bool parameter_offloader::node_reads_tracked_weight(ggml_tensor * t, int * out_idx = nullptr)
{
    int best_idx = -1;
    const int N  = (int)gpu_tensors_in_order.size();

    // quick filter by op to reduce callback overhead
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:      // only if you mirrored bias tensors
            break;
        default:
            return false;
    }

    // scan node sources
    for (int k = 0; k < GGML_MAX_SRC; ++k)
    {
        ggml_tensor * s = t->src[k];
        if (!s)
            break;

        auto it = gpu2index.find(s);
        if (it == gpu2index.end())
            continue;

        const int idx = it->second;
        if (best_idx < 0) {
            best_idx = idx;
            continue;
        }

        // pick “furthest ahead” vs the last used index to keep monotonic progress
        const int last = tensor_idx_used_mod.load(std::memory_order_relaxed);
        if (last < 0)
            continue;

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
    const int tensor_count = (int)gpu_tensors_in_order.size();
    //if (idx == tensor_count - 1)
    //    LLAMA_LOG_INFO("%s got to idx %d\n", __func__, idx);
    auto ring_dist = [&](int from, int to) -> int {
        return (to - from + tensor_count) % tensor_count; // forward distance in ring
    };

    // Resolve the GPU twin + its CPU source so we can print name/size/offset
    ggml_tensor * w_gpu = gpu_tensors_in_order[idx];
    ggml_tensor * w_cpu = gpu2cpu[w_gpu];

    const char * name   = ggml_get_name(w_gpu); // you mirrored names when duplicating
    const size_t off    = (size_t)((char *) w_gpu->data - base);
    const size_t bytes  = ggml_backend_buft_get_alloc_size(buft, w_cpu);
    const size_t nbytes = ggml_nbytes(w_cpu);

    int last  = tensor_idx_used_mod.load(std::memory_order_relaxed);
    int epoch = tensor_idx_used_epoch.load(std::memory_order_relaxed);

    const long long seq = (long long)epoch * (long long)tensor_count + idx;
    if (last >= 0 && ((idx - last) < 0))
        ++epoch;

    tensor_idx_used_mod.store(idx, std::memory_order_release);
    tensor_idx_used_epoch.store(epoch, std::memory_order_relaxed);
    tensor_idx_used_seq.store(seq, std::memory_order_release);

    // wake streamer
    node_cv_.notify_all();
    
    long long cur_cseq  = tensor_idx_copied_seq.load(std::memory_order_acquire);
    const long long needed_copy_seq = (long long)epoch * (long long)tensor_count + idx + 1; //+1 because we're in the post-compute and we need to ready the next one

    if (cur_cseq < needed_copy_seq)
    {
#if LLAMA_LOG_READS > 1
    LLAMA_LOG_INFO("[RB.%d.%lld.%lld.%lld.%d]", idx, cur_cseq, needed_copy_seq, seq, epoch);
#endif
        std::unique_lock<std::mutex> lk(node_mu_);
        node_cv_.wait(lk, [&]{
            return stop_stream.load(std::memory_order_acquire) ||
                   tensor_idx_copied_seq.load(std::memory_order_acquire) >= needed_copy_seq;
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

    // Verbose, but super handy while tuning:
#ifdef LLAMA_LOG_READS
    LLAMA_LOG_INFO("[R.%d]", idx);
#endif

#if defined(LLAMA_DIAGNOSE_COPY)
    //int i = (tensor_idx_loaded.load(std::memory_order_relaxed) + 1) % (int)tensor_count;
    int last_cmod = tensor_idx_copied_mod.load(std::memory_order_relaxed);
    int i = (last_cmod + 1) % (int)tensor_count;
    int copy_epoch = tensor_idx_copied_epoch.load(std::memory_order_relaxed);
    
    for (;;)
    {
        // snapshot the current reader sequence
        long long cur_seq = tensor_idx_used_seq.load(std::memory_order_acquire);

        {
            const int r_idx = (int)(cur_seq % tensor_count); // current reader index
            const int bar   = ready_after[r_idx];            // last copyable index while r is read
            const int di    = ring_dist(r_idx, i);           // distance from reader to copy slot
            const int dbar  = ring_dist(r_idx, bar);         // distance from reader to barrier

            //bool allowed = (di <= dbar);                // push copies up to and including 'bar'
            bool allowed = (di <= dbar) && (di <= LLAMA_DIAGNOSE_COPY);

            if (!allowed)
            {
                //LLAMA_LOG_INFO("[NO_COPY.%d.%d.%d.%d]", r_idx, bar, di, dbar);
                break;
            }
        }

        // Safe to copy slot i
        ggml_tensor *w_cpu_ = cpu_tensors_in_order[i];
        ggml_tensor *w_gpu_ = gpu_tensors_in_order[i];
        //ggml_backend_tensor_copy(w_cpu_, w_gpu_);
        copy_host_to_arena_with_transform(w_cpu_, w_gpu_);
#ifdef LLAMA_LOG_COPIES
        LLAMA_LOG_INFO("[C.%d]", i);
#endif
        //tensor_idx_loaded.store(i, std::memory_order_release);
        tensor_idx_copied_mod.store(i, std::memory_order_release);
        if (last_cmod >= 0 && ((i - last_cmod) < 0)) {
            ++copy_epoch;
            tensor_idx_copied_epoch.store(copy_epoch, std::memory_order_relaxed);
        }
        const long long cseq = (long long)copy_epoch * (long long)tensor_count + i;
        tensor_idx_copied_seq.store(cseq, std::memory_order_release);

        // advance ring
        i = (i + 1) % (int)tensor_count;
    }
    //LLAMA_LOG_INFO("\n", i);
#endif /* defined(LLAMA_DIAGNOSE_COPY) */

    return true;
}

void parameter_offloader::start_streamer() {
    stop_stream.store(false, std::memory_order_release);
    copy_thread = std::thread(&parameter_offloader::stream_worker, this);
}
void parameter_offloader::stop_streamer_join() {
    stop_stream.store(true, std::memory_order_release);
    if (copy_thread.joinable())
        copy_thread.join();
}

void parameter_offloader::stream_worker()
{
    const size_t tensor_count = gpu_tensors_in_order.size();
    LLAMA_LOG_INFO("%s tensor_count == %lu\n", __func__, tensor_count);
    if (tensor_count == 0)
        return;

    auto ring_dist = [&](int from, int to) -> int {
        // forward distance in [0..N-1]
        return (to - from + (int)tensor_count) % (int)tensor_count;
    };

    // next slot to copy
    //int i = (tensor_idx_loaded.load(std::memory_order_relaxed) + 1) % (int)tensor_count;
    int last_cmod = tensor_idx_copied_mod.load(std::memory_order_relaxed);
    int i = (last_cmod + 1) % (int)tensor_count;
    int copy_epoch = tensor_idx_copied_epoch.load(std::memory_order_relaxed);

    for (;;)
    {
        if (stop_stream.load(std::memory_order_acquire))
            return;

        // snapshot the current reader sequence
        long long cur_seq = tensor_idx_used_seq.load(std::memory_order_acquire);

        if (cur_seq < 0) {
            // No reader yet (startup). Prefill opportunistically without waiting.
            // (If you want to limit prefill, add a soft cap here.)
        } else {
            const int r_idx = (int)(cur_seq % tensor_count);// current reader index
            const int bar   = ready_after[r_idx];           // last copyable index while r_idx is read
            const int di    = ring_dist(r_idx, i);          // distance from reader to copy slot
            const int dbar  = ring_dist(r_idx, bar);        // distance from reader to barrier

            // Allowed window is (r_idx, ..., bar) inclusive. di == 0 means "r_idx" itself (disallowed).
            bool allowed = (di <= dbar);
            //bool allowed = (di != 0) && (di <= dbar);

            if (!allowed)
            {
#if LLAMA_LOG_COPIES > 1
                LLAMA_LOG_INFO("[CB.%d.%d.%d.%d.%d]", i, r_idx, bar, di, dbar);
#endif
                // Not safe to copy this slot yet; wait until reader advances (tensor_idx_used_seq changes)
                std::unique_lock<std::mutex> lk(node_mu_);
                node_cv_.wait(lk, [&]{
                    return stop_stream.load(std::memory_order_acquire) ||
                           tensor_idx_used_seq.load(std::memory_order_acquire) != cur_seq;
                });
                if (stop_stream.load(std::memory_order_acquire))
                    return;
                // re-evaluate with new reader position
                continue;
            }
        }

        // Safe to copy slot i
        ggml_tensor *w_cpu = cpu_tensors_in_order[i];
        ggml_tensor *w_gpu = gpu_tensors_in_order[i];
        //ggml_backend_tensor_copy(w_cpu, w_gpu);
        copy_host_to_arena_with_transform(w_cpu, w_gpu);

        //tensor_idx_loaded.store(i, std::memory_order_release);
        tensor_idx_copied_mod.store(i, std::memory_order_release);
        if (last_cmod >= 0 && ((i - last_cmod) < 0)) {
            ++copy_epoch;
            tensor_idx_copied_epoch.store(copy_epoch, std::memory_order_relaxed);
        }
        const long long cseq = (long long)copy_epoch * (long long)tensor_count + i;
        tensor_idx_copied_seq.store(cseq, std::memory_order_release);

#ifdef LLAMA_LOG_COPIES
        LLAMA_LOG_INFO("[C.%d]", i);
        //LLAMA_LOG_INFO("[C.%d.%lld.%d]", i, cseq, copy_epoch);
#endif

        // wake any pre-compute waiters blocked on copy readiness
        node_cv_.notify_all();

        // advance ring
        last_cmod = i;
        i = (i + 1) % (int)tensor_count;
    }
}


void parameter_offloader::print_snapshot()
{
    size_t tensor_count = gpu_tensors_in_order.size();

    std::vector<size_t> start(tensor_count);
    std::vector<size_t> end(tensor_count);
    start.reserve(tensor_count);
    end.reserve(tensor_count);

    for (size_t i = 0; i < tensor_count; ++i)
    {
        ggml_tensor *t_gpu = gpu_tensors_in_order[i];
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
        ggml_tensor * w_gpu = gpu_tensors_in_order[i];
        const char * name   = ggml_get_name(w_gpu); // you mirrored names when duplicating
        LLAMA_LOG_INFO("%s %4d %4d %10lu %10lu %5d %s\n", __func__, i, ready_after[i], start[i], end[i], ready_after[i] - i, name);
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

            if (po->weight_set.find(p) == po->weight_set.end())
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
    }

    // Only return true for weights you actually track; keeps batching intact
    if (ask) {
        return po->wants_observe(node);
    } else {
        // === POST-COMPUTE ===
        po->on_eval_tensor(node);  // keep your current bookkeeping first

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
