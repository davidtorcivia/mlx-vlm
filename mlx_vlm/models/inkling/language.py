from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache
from ..mlp import SwiGLUMLP
from ..switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
from .config import TextConfig as ModelConfig


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_cache_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_cache_tree(v) for k, v in value.items()}
    return value


# Sentinel marking a KVCache that was empty (keys is None) at snapshot time.
# KVCache.state raises on empty caches, and the deep-8 MTP drafter legitimately
# snapshots blocks whose caches have never been written (only block 0 is fed
# during prefill), so empties must round-trip through snapshot/restore.
_EMPTY_KV = object()


def _subcaches(c):
    return getattr(c, "caches", None) or [c]


def _snapshot_cache_state(caches):
    """Deep-copy the full state of every cache so a speculative block can be
    rolled back by replay. Inkling's short-conv slots keep only the last K-1
    inputs and cannot be trimmed, so we restore-and-replay instead."""
    snapshot = []
    for c in caches:
        if c is None:
            snapshot.append(None)
            continue
        subs = []
        for sc in _subcaches(c):
            if getattr(sc, "keys", False) is None:
                subs.append(_EMPTY_KV)
            else:
                subs.append(_clone_cache_tree(sc.state))
        snapshot.append(subs)
    arrays = [v for _, v in tree_flatten(snapshot) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
    return snapshot


def _restore_cache_state(caches, snapshot):
    for c, s in zip(caches, snapshot):
        if c is None or s is None:
            continue
        for sc, ss in zip(_subcaches(c), s):
            if ss is _EMPTY_KV:
                sc.keys = None
                sc.values = None
                sc.offset = 0
            else:
                sc.state = _clone_cache_tree(ss)


_MASK_SRC = r"""
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + int(Q_OFF)) - int(j);   // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
"""
_mask_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask",
    input_names=["rel", "proj"],
    output_names=["out"],
    source=_MASK_SRC,
)


def _rup(a, m):
    return ((a + m - 1) // m) * m


def banded_additive_mask(rel, proj, q_offset, S, sliding, rel_extent):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    # Metal template args must be Python int/bool/Dtype. Batch engines (e.g.
    # omlx) hand cache offsets over as numpy/mx scalars; coerce or the kernel
    # rejects the template. Scalar Q_OFF also means a batch must share one
    # offset — true today (per-request rows are padded to a common offset).
    q_offset = int(q_offset)
    S = int(S)
    sliding = int(sliding)
    rel_extent = int(rel_extent)
    if mx.default_device() == mx.gpu:
        return _mask_kernel(
            inputs=[rel, proj],
            template=[
                ("T", dtype),
                ("B", B),
                ("H", H),
                ("LQ", LQ),
                ("S", S),
                ("Q_OFF", q_offset),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    rl = (rel @ proj).transpose(0, 2, 1, 3)
    qp = mx.arange(LQ) + q_offset
    kp = mx.arange(S)
    dist = qp[:, None] - kp[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S))
    pb = mx.take_along_axis(rl, gidx, axis=-1)
    pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
    neg = dist < 0
    if sliding > 0:
        neg = neg | (dist >= sliding)
    return mx.where(neg[None, None], mx.array(-1e30, dtype), pb).astype(dtype)


_SCONV_SRC = r"""
    uint c = thread_position_in_grid.x;   // channel
    uint b = thread_position_in_grid.y;   // batch row
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        // Match the unfused path's rounding: the conv emits bf16 (rounded)
        // before the fp32 residual add; the layer residual is a second bf16
        // add on top (as the decoder layer's x + sconv(r) was).
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }
"""
_sconv_kernel = mx.fast.metal_kernel(
    name="inkling_sconv_decode",
    input_names=["x", "state", "w", "res"],
    output_names=["out", "nstate"],
    source=_SCONV_SRC,
)


class InklingShortConvolution(nn.Module):
    """Depthwise causal 1-D conv over the previous ``kernel_size - 1`` states, plus a
    residual add. Kept in fp32 for stability (matches the reference). ``conv_idx`` selects
    this conv's slot in the layer's shared conv cache."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        residual: Optional[mx.array] = None,
    ):
        dt = x.dtype
        K = self.kernel_size
        if (
            cache is not None
            and mask is None
            and K == 4
            and x.shape[1] <= 8
            and mx.default_device() == mx.gpu
        ):
            B, L, C = x.shape
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((B, K - 1, C), dtype=mx.float32)
            out, nstate = _sconv_kernel(
                inputs=[
                    x,
                    state,
                    self.conv.weight.reshape(-1),
                    residual if residual is not None else x,
                ],
                template=[
                    ("T", dt),
                    ("B", B),
                    ("L", L),
                    ("C", C),
                    ("K", K),
                    ("HAS_RES", residual is not None),
                ],
                grid=(_rup(C, 32), B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, L, C), (B, K - 1, C)],
                output_dtypes=[dt, mx.float32],
            )
            cache[self.conv_idx] = nstate
            return out
        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = xp[:, -(K - 1) :, :]
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        out = (out + res).astype(dt)
        return out if residual is None else residual + out


_ATTN_SRC = r"""
    // Flash-style decode attention for L=1: one threadgroup of 128 threads
    // (4 simdgroups) per (batch, q-head); simdgroups split the key range and
    // merge online-softmax partials at the end. Replaces q_norm + banded-mask
    // materialization + masked SDPA + log-tau scaling + the transposes around
    // them. K/V are read from the cache views via injected strides (S comes
    // from k_shape, so no per-step kernel respecialization); the raw fused
    // projection row is read via column offsets. Sliding layers scan only
    // their window. bf16 rounding points of the unfused path (normed q, tau
    // products, band value) are reproduced.
    uint lane = thread_index_in_simdgroup;        // 0..31
    uint sg   = simdgroup_index_in_threadgroup;   // 0..3
    uint tid  = sg * 32 + lane;                   // 0..127
    uint h    = threadgroup_position_in_grid.y;   // q head
    uint b    = threadgroup_position_in_grid.z;   // batch row
    float eps    = params[0];
    float alpha  = params[1];
    float nfloor = params[2];
    const int S = k_shape[2];
    const uint hkv = h / (HQ / HKV);
    const size_t qrow = (size_t)b * qkvr_strides[0]
        + (size_t)(qkvr_shape[1] - 1) * qkvr_strides[1];
    const size_t qc = qkvr_strides[2];
    const device T* qp = qkvr + qrow + (size_t)h * D * qc;          // q columns
    const device T* rp = qkvr + qrow + ((size_t)R_OFF + (size_t)h * DR) * qc;
    float tau = 1.0f;
    if (HAS_TAU) {
        tau = 1.0f + alpha * metal::log(metal::max((float)S / nfloor, 1.0f));
        tau = (float)((T)tau);
    }
    // Cooperative q RMSNorm in fp32 (one dim per thread); round to bf16 like
    // the unfused path, then the tau product rounds again.
    threadgroup float qs[D];
    threadgroup float red[4];
    threadgroup float tgm[4], tgl[4];
    threadgroup float tgacc[4][D];
    float x0 = (float)qp[tid * qc];
    float ps_ = simd_sum(x0 * x0);
    if (lane == 0) red[sg] = ps_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = red[0] + red[1] + red[2] + red[3];
    float inv = metal::rsqrt(ss / (float)D + eps);
    float qn = (float)((T)(x0 * inv * (float)qw[tid]));
    if (HAS_TAU) qn = (float)((T)(qn * tau));
    qs[tid] = qn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / (float)D;
    const int jstart = (SLIDING > 0 && S > SLIDING) ? S - SLIDING : 0;
    const int j0base = (jstart / 32) * 32;
    float m = -INFINITY, l = 0.0f;
    float4 acc = float4(0.0f);
    for (int j0 = j0base + (int)sg * 32; j0 < S; j0 += 128) {
        int j = j0 + (int)lane;
        float score = -INFINITY;
        if (j >= jstart && j < S) {
            const size_t krow = (size_t)b * k_strides[0]
                + (size_t)hkv * k_strides[1] + (size_t)j * k_strides[2];
            // packed 4-wide loads: cache buffers are only element-aligned,
            // so vec<T,4> (8-byte alignment) must not be used here.
            const device packed_ushort4* kp4 =
                (const device packed_ushort4*)(k + krow);
            float dot = 0.0f;
            for (uint t = 0; t < D / 4; ++t) {
                packed_ushort4 kr = kp4[t];
                float4 kv = float4((float)as_type<T>(kr.x), (float)as_type<T>(kr.y),
                                   (float)as_type<T>(kr.z), (float)as_type<T>(kr.w));
                dot += qs[t * 4 + 0] * kv.x + qs[t * 4 + 1] * kv.y
                     + qs[t * 4 + 2] * kv.z + qs[t * 4 + 3] * kv.w;
            }
            int dist = (S - 1) - j;
            float band = 0.0f;
            if (dist < REL_EXTENT) {
                float accb = 0.0f;
                for (uint d = 0; d < DR; ++d)
                    accb += (float)rp[d * qc] * (float)rproj[d * REL_EXTENT + dist];
                band = (float)((T)accb);
                if (HAS_TAU) band = (float)((T)(band * tau));
            }
            score = dot * scale + band;
        }
        float cm = simd_max(score);
        if (cm == -INFINITY) continue;
        float nm = metal::max(m, cm);
        float corr = (m > -INFINITY) ? metal::exp(m - nm) : 0.0f;
        float p = (score > -INFINITY) ? metal::exp(score - nm) : 0.0f;
        l = l * corr + simd_sum(p);
        acc *= corr;
        for (uint u = 0; u < 32; ++u) {
            float pu = simd_shuffle(p, (ushort)u);
            if (pu > 0.0f) {
                int ju = j0 + (int)u;
                const size_t vrow = (size_t)b * v_strides[0]
                    + (size_t)hkv * v_strides[1] + (size_t)ju * v_strides[2];
                const device packed_ushort4* vp4 =
                    (const device packed_ushort4*)(v + vrow);
                packed_ushort4 vr = vp4[lane];
                acc += pu * float4((float)as_type<T>(vr.x), (float)as_type<T>(vr.y),
                                   (float)as_type<T>(vr.z), (float)as_type<T>(vr.w));
            }
        }
        m = nm;
    }
    // Merge the four simdgroup partials (each lane owns dims lane*4..+3).
    if (lane == 0) { tgm[sg] = m; tgl[sg] = l; }
    tgacc[sg][lane * 4 + 0] = acc.x;
    tgacc[sg][lane * 4 + 1] = acc.y;
    tgacc[sg][lane * 4 + 2] = acc.z;
    tgacc[sg][lane * 4 + 3] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = metal::max(metal::max(tgm[0], tgm[1]), metal::max(tgm[2], tgm[3]));
    float w0 = (tgm[0] > -INFINITY) ? metal::exp(tgm[0] - M) : 0.0f;
    float w1 = (tgm[1] > -INFINITY) ? metal::exp(tgm[1] - M) : 0.0f;
    float w2 = (tgm[2] > -INFINITY) ? metal::exp(tgm[2] - M) : 0.0f;
    float w3 = (tgm[3] > -INFINITY) ? metal::exp(tgm[3] - M) : 0.0f;
    float Lt = tgl[0] * w0 + tgl[1] * w1 + tgl[2] * w2 + tgl[3] * w3;
    float od = tgacc[0][tid] * w0 + tgacc[1][tid] * w1
             + tgacc[2][tid] * w2 + tgacc[3][tid] * w3;
    device T* op = out + ((size_t)b * HQ + h) * D;
    op[tid] = (T)(od / Lt);
"""
_attn_kernel = mx.fast.metal_kernel(
    name="inkling_attn_decode",
    input_names=["qkvr", "k", "v", "qw", "rproj", "params"],
    output_names=["out"],
    source=_ATTN_SRC,
    ensure_row_contiguous=False,
)

# Escape hatch: force the unfused decode-attention path (debugging/AB tests).
_FUSED_ATTN_DECODE = True
# Below this key length the unfused mask+SDPA path is slightly faster (fewer,
# larger kernels beat the custom kernel at tiny S); above it the fused kernel
# wins outright and by 4k context is ~2.6x faster end-to-end.
_FUSED_ATTN_MIN_S = 512


class InklingAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        self.d_rel = config.d_rel
        self.scale = 1.0 / self.head_dim
        self.log_floor = None if self.is_sliding else config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha

        # q/k/v/r share the input row; their weights are stacked at load
        # (see fuse_qkvr) so decode does one matmul instead of four.
        self.qkvr_dims = (
            self.n_heads * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_heads * self.d_rel,
        )
        self.qkvr_proj = nn.Linear(
            config.hidden_size, sum(self.qkvr_dims), bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self._attn_params = None
        self.k_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=0
        )
        self.v_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=1
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_proj = mx.zeros((self.d_rel, self.rel_extent))

    def __call__(self, x, cache=None, conv_mask=None):
        B, L, _ = x.shape
        kv = cache[0] if cache is not None else None
        conv = cache[1] if cache is not None else None

        qkvr = self.qkvr_proj(x)
        dq, dk, dv, _ = self.qkvr_dims
        k = self.k_sconv(qkvr[..., dq : dq + dk], cache=conv, mask=conv_mask)
        v = self.v_sconv(
            qkvr[..., dq + dk : dq + dk + dv], cache=conv, mask=conv_mask
        )

        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
        S = k.shape[2]

        if (
            L == 1
            and _FUSED_ATTN_DECODE
            and S >= _FUSED_ATTN_MIN_S
            # the kernel's 128-thread threadgroup layout assumes head_dim 128
            and self.head_dim == 128
            and self.n_heads % self.n_kv == 0
            and mx.default_device() == mx.gpu
            # the kernel reads k/v/qkvr as one dtype; a promoted-fp32 cache
            # (e.g. fp32 norm weights) must take the unfused path
            and k.dtype == x.dtype
            and v.dtype == x.dtype
        ):
            if self._attn_params is None:
                self._attn_params = mx.array(
                    [
                        self.q_norm.eps,
                        self.log_alpha,
                        float(self.log_floor) if self.log_floor is not None else 1.0,
                    ],
                    dtype=mx.float32,
                )
                self._rel_proj_cast = (
                    self.rel_proj
                    if self.rel_proj.dtype == x.dtype
                    else self.rel_proj.astype(x.dtype)
                )
            out = _attn_kernel(
                inputs=[
                    qkvr,
                    k,
                    v,
                    self.q_norm.weight,
                    self._rel_proj_cast,
                    self._attn_params,
                ],
                template=[
                    ("T", x.dtype),
                    ("HQ", self.n_heads),
                    ("HKV", self.n_kv),
                    ("D", self.head_dim),
                    ("DR", self.d_rel),
                    ("R_OFF", dq + dk + dv),
                    ("REL_EXTENT", self.rel_extent),
                    ("SLIDING", self.sliding),
                    ("HAS_TAU", self.log_floor is not None),
                ],
                grid=(128, self.n_heads, B),
                threadgroup=(128, 1, 1),
                output_shapes=[(B, 1, self.n_heads * self.head_dim)],
                output_dtypes=[x.dtype],
            )[0]
            return self.o_proj(out)

        q = qkvr[..., :dq]
        r = qkvr[..., dq + dk + dv :].reshape(B, L, self.n_heads, self.d_rel)
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        # Query positions derive from the post-update key length: the queries
        # are always the last L of the S cached positions. Do NOT trust
        # kv.offset here — batch cache implementations (e.g. an engine's
        # BatchKVCache) disagree with plain KVCache by one on whether the
        # in-flight token is counted, which shifts the whole relative-position
        # band during decode and derails generation.
        offset = S - L

        mask = banded_additive_mask(
            r, self.rel_proj.astype(x.dtype), offset, S, self.sliding, self.rel_extent
        )
        if self.log_floor is not None:
            qpos = (mx.arange(L) + offset + 1).astype(mx.float32)
            tau = 1.0 + self.log_alpha * mx.log(mx.maximum(qpos / self.log_floor, 1.0))
            tau = tau.reshape(1, 1, L, 1).astype(x.dtype)
            q = q * tau
            mask = mx.where(mask > -1e29, mask * tau, mask)

        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class InklingDenseMLP(SwiGLUMLP):
    """Dense SwiGLU MLP (shared ``SwiGLUMLP``) with a learned output scale."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.hidden_size, config.dense_intermediate_size)
        self.global_scale = mx.ones((1,))

    def __call__(self, x):
        return super().__call__(x) * self.global_scale


class InklingSwitchGLU(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, num_experts, **kwargs):
        super().__init__(input_dims, hidden_dims, num_experts, **kwargs)
        self.gate_scale = mx.ones((num_experts,))  # s13 (fused gate/up scale2)
        self.out_scale = mx.ones((num_experts,))  # s13 * s2
        self._scales_trivial = None

    def _per_expert(self, scale, idx, like):
        s = scale[idx].astype(like.dtype)
        return s.reshape(s.shape + (1,) * (like.ndim - s.ndim))

    def __call__(self, x, indices) -> mx.array:
        # Non-NVFP4 checkpoints carry all-ones expert scales; skip the two
        # gather+mul chains entirely then (checked once, after load).
        if self._scales_trivial is None:
            self._scales_trivial = bool(
                (mx.all(self.gate_scale == 1) & mx.all(self.out_scale == 1)).item()
            )
        if self._scales_trivial:
            return super().__call__(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x_gate = x_gate * self._per_expert(self.gate_scale, idx, x_gate)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        x = x * self._per_expert(self.out_scale, idx, x)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


_ROUTE_SRC = r"""
    // One simdgroup (32 lanes) per token; each lane owns ceil(R/32) experts
    // (strided by 32 so logit reads coalesce). Top-K via K rounds of
    // simd_max; ties resolved to the lowest lane (deterministic).
    uint lane = thread_position_in_grid.x;   // 0..31, simd lane
    uint n = thread_position_in_grid.y;      // token
    if (n >= N) return;
    const device T* lg = logits + (size_t)n * (R + SH);
    float ws = wscale[0];
    constexpr int PER = (R + 31) / 32;
    float sc[PER];
    float tl_[PER];
    bool taken[PER];
    for (int t = 0; t < PER; ++t) {
        uint j = lane + (uint)t * 32u;
        taken[t] = false;
        if (j < R) {
            float l = (float)lg[j];
            tl_[t] = l;
            sc[t] = 1.0f / (1.0f + metal::exp(-l)) + (float)corr[j];
        } else {
            sc[t] = -INFINITY;
        }
    }
    uint bidx[K];
    float btl[K];
    for (uint kk = 0; kk < K; ++kk) {
        float lb = -INFINITY; int lt = -1;
        for (int t = 0; t < PER; ++t)
            if (!taken[t] && sc[t] > lb) { lb = sc[t]; lt = t; }
        float gb = simd_max(lb);
        ushort wl = (ushort)simd_min(lb == gb ? lane : 32u);
        uint wj = simd_shuffle(lt >= 0 ? lane + (uint)lt * 32u : 0u, wl);
        float wtl = simd_shuffle(lt >= 0 ? tl_[lt] : 0.0f, wl);
        if (lane == (uint)wl && lt >= 0 && sc[lt] == gb) taken[lt] = true;
        bidx[kk] = wj; btl[kk] = wtl;
    }
    // Routing weights: softmax over logsigmoid of the K routed + SH shared
    // logits, times route_scale * global_scale (folded into ws). Computed
    // redundantly on every lane (cheap; K + SH values).
    float lp[K + SH];
    float m = -INFINITY;
    for (uint t = 0; t < K + SH; ++t) {
        float tv = (t < K) ? btl[t] : (float)lg[R + (t - K)];
        float a = -tv;
        float lad = metal::max(a, 0.0f)
                  + metal::log(1.0f + metal::exp(-metal::fabs(a)));
        lp[t] = -lad;
        m = metal::max(m, lp[t]);
    }
    float se = 0.0f;
    for (uint t = 0; t < K + SH; ++t) se += metal::exp(lp[t] - m);
    float lse = m + metal::log(se);
    if (lane < K) {
        idx[(size_t)n * K + lane] = bidx[lane];
        wk[(size_t)n * K + lane] = (T)(metal::exp(lp[lane] - lse) * ws);
    }
    T gv[SH];
    for (uint s = 0; s < SH; ++s) gv[s] = (T)(metal::exp(lp[K + s] - lse) * ws);
    device T* gp = gamma + (size_t)n * SH * I;
    for (uint t = lane; t < SH * I; t += 32u) gp[t] = gv[t / I];
"""
_route_kernel = mx.fast.metal_kernel(
    name="inkling_moe_route",
    input_names=["logits", "corr", "wscale"],
    output_names=["idx", "wk", "gamma"],
    source=_ROUTE_SRC,
)


@partial(mx.compile, shapeless=True)
def _swiglu_scaled(gate, up, s):
    return nn.silu(gate) * up * s


class InklingSharedExpertsDense(nn.Module):
    """The ``n_shared`` always-on experts as one dense SwiGLU: expert weights are
    concatenated at load (see ``shared_experts_to_dense``) so the fixed-index
    gather_qmm path becomes three plain matmuls. Per-token expert weights arrive
    pre-broadcast over the expert-major intermediate (``gamma``) and are applied
    before down_proj, which distributes over the concatenated experts exactly."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.up_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.down_proj = nn.Linear(num_experts * hidden_dims, input_dims, bias=False)

    def __call__(self, x, gamma):
        return self.down_proj(_swiglu_scaled(self.gate_proj(x), self.up_proj(x), gamma))


def fuse_qkvr(weights):
    """Stack per-layer q/k/v/r projection tensors (rows, plus scales/biases for
    quantized checkpoints) into the single ``qkvr_proj``. Row-concat of
    quantized matrices is exact: each output row keeps its own groups."""
    out = dict(weights)
    prefixes = {
        k[: -len("q_proj.weight")]
        for k in weights
        if k.endswith(".self_attn.q_proj.weight")
    }
    for p in prefixes:
        for leaf in ("weight", "scales", "biases"):
            parts = [out.pop(f"{p}{n}_proj.{leaf}", None) for n in "qkvr"]
            if all(v is not None for v in parts):
                out[f"{p}qkvr_proj.{leaf}"] = mx.concatenate(parts, axis=0)
            elif any(v is not None for v in parts):
                raise ValueError(f"partial q/k/v/r {leaf} set under {p}")
    return out


def shared_experts_to_dense(weights):
    """Remap SwitchGLU-shaped shared-expert tensors ``[E, out, in]`` (bf16 or
    quantized triplets) to the dense concatenated layout of
    ``InklingSharedExpertsDense``. Expert-major on the intermediate axis, so
    gate/up stack experts along rows and down stacks along input columns."""
    out = {}
    for k, v in weights.items():
        if ".shared_experts." in k and isinstance(v, mx.array) and v.ndim == 3:
            if ".down_proj." in k:
                v = v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            else:
                v = v.reshape(-1, v.shape[2])
        out[k] = v
    return out


class InklingSparseMoE(nn.Module):
    """Sigmoid-gated fine-grained MoE: top-k routed experts (+ correction-bias selection)
    plus always-on shared experts, weighted by a logsigmoid/logsumexp softmax."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.intermediate_size = config.intermediate_size
        self.gate_weight = mx.zeros((self.n_routed + self.n_shared, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))
        self.switch_mlp = InklingSwitchGLU(
            config.hidden_size, config.intermediate_size, self.n_routed
        )
        self.shared_experts = InklingSharedExpertsDense(
            config.hidden_size, config.intermediate_size, self.n_shared
        )
        self._wscale = None

    def _route(self, logits):
        """Expert selection + routing weights. On GPU the whole post-matmul
        chain (sigmoid + bias + top-k + logsigmoid softmax + scaling) is one
        kernel; it also emits the shared-expert weights pre-broadcast over the
        expert-major dense intermediate."""
        N = logits.shape[0]
        if self._wscale is None:
            self._wscale = mx.array(
                [self.route_scale], dtype=mx.float32
            ) * self.global_scale.astype(mx.float32)
        if mx.default_device() == mx.gpu:
            return _route_kernel(
                inputs=[logits, self.e_score_correction_bias, self._wscale],
                template=[
                    ("T", logits.dtype),
                    ("N", N),
                    ("R", self.n_routed),
                    ("SH", self.n_shared),
                    ("K", self.top_k),
                    ("I", self.intermediate_size),
                ],
                grid=(32, N, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (N, self.top_k),
                    (N, self.top_k),
                    (N, self.n_shared * self.intermediate_size),
                ],
                output_dtypes=[mx.uint32, logits.dtype, logits.dtype],
            )
        scores = mx.sigmoid(logits.astype(mx.float32))
        sfc = scores[:, : self.n_routed] + self.e_score_correction_bias
        idx = mx.argpartition(-sfc, self.top_k - 1, axis=-1)[:, : self.top_k]
        routed_logits = logits[:, : self.n_routed]
        shared_logits = logits[:, -self.n_shared :]
        tl = mx.concatenate(
            [mx.take_along_axis(routed_logits, idx, axis=-1), shared_logits], axis=-1
        ).astype(mx.float32)
        lp = -mx.logaddexp(mx.zeros_like(tl), -tl)
        w = (
            mx.exp(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
            * self.route_scale
            * self.global_scale
        )
        topk_w = w[:, : self.top_k].astype(logits.dtype)
        gamma = mx.repeat(
            w[:, -self.n_shared :].astype(logits.dtype),
            self.intermediate_size,
            axis=-1,
        )
        return idx.astype(mx.uint32), topk_w, gamma

    def __call__(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        gw = self.gate_weight
        if gw.dtype != x.dtype:
            gw = gw.astype(x.dtype)
        logits = xf @ gw.T
        idx, topk_w, gamma = self._route(logits)
        yr = (self.switch_mlp(xf, idx) * topk_w[..., None]).sum(axis=-2)
        ys = self.shared_experts(xf, gamma)
        return (yr + ys).reshape(B, L, D).astype(x.dtype)


class InklingDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = InklingAttention(config, layer_idx)
        self.mlp = (
            InklingDenseMLP(config)
            if config.layer_is_dense(layer_idx)
            else InklingSparseMoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=2
        )
        self.mlp_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=3
        )

    def __call__(self, x, cache=None, conv_mask=None):
        conv = cache[1] if cache is not None else None
        r = self.self_attn(self.input_layernorm(x), cache=cache, conv_mask=conv_mask)
        h = self.attn_sconv(r, cache=conv, mask=conv_mask, residual=x)
        r = self.mlp(self.post_attention_layernorm(h))
        return self.mlp_sconv(r, cache=conv, mask=conv_mask, residual=h)


class InklingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        self.layers = [
            InklingDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids):
        h = self.embed_tokens(input_ids)
        if self.embed_norm is not None:
            h = self.embed_norm(h)
        return h

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        skip_final_norm: bool = False,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)
        return h if skip_final_norm else self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = InklingModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits_from_norm(self, h):
        h = h / self.config.logits_mup_width_multiplier
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        uv = self.config.unpadded_vocab_size
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        pre_norm = self.model(inputs, cache, inputs_embeds, skip_final_norm=True)
        logits = (
            None if skip_logits else self._logits_from_norm(self.model.norm(pre_norm))
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[pre_norm] if return_hidden else None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._logits_from_norm(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> Optional[mx.array]:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        snapshot = _snapshot_cache_state(cache)
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, (snapshot, inputs)

    def rollback_speculative_cache(
        self, caches, gdn_states, accepted, block_size
    ) -> int:
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(int(a) for a in accepted)
        snapshot, verify_inputs = gdn_states
        _restore_cache_state(caches, snapshot)
        keep = accepted + 1
        if keep > 0:
            self(verify_inputs[:, :keep], cache=caches, skip_logits=True)
        return accepted

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.model.layers]
