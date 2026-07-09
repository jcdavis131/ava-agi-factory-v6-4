"""
Ava — YaRN RoPE 10k→1M + QK-Norm + Multi-J-Space support
Solo personal project, no connection to employer, built with public/free-tier only

Fixed per specs/04_model_and_configs.md. The bugs that were here mattered:

  1. Attention had NO causal mask -- every position attended to the future, so
     the "language model" could trivially read the token it was predicting.
     Nothing trained on it could have worked. Now F.scaled_dot_product_attention
     with is_causal=True (also faster: flash/mem-efficient kernels).
  2. rotate_half() used interleaved pairing (x[..., ::2] / x[..., 1::2]) while
     get_cos_sin() builds cos/sin as cat((freqs, freqs)) -- the half-split
     (LLaMA) layout. The two disagreed, so RoPE applied a garbage rotation.
  3. _prev_workspaces was cached across forward passes without .detach(), so the
     second backward pass walked into a freed graph. Training could not reach
     step 2. Cross-step persistence is now opt-in (`use_memory`) and detached.
  4. lm_head was never tied to embed; heads/head_dim/layer counts were hardcoded
     at 16x128 regardless of d_model.
  5. cos/sin were recomputed inside every block despite all blocks sharing an
     identical RoPE. Computed once per forward now.

Also adds, config-gated, what base1b needs to fit in 12GB: grouped-query
attention (n_kv_heads) and SwiGLU, plus gradient checkpointing.
"""
import math
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # F.rms_norm requires torch>=2.4 (pinned in docker/requirements.gpu.txt)
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class YaRNScaledRoPE(nn.Module):
    """
    True YaRN/RoPE 10k→1M + QK-Norm per Peng et al. 2023
    scale <=1: standard RoPE inv_freq = 1/(base^(i/dim)), base 10k
    1<scale<=2: NTK-aware base' = base * scale^(dim/(dim-2))
    scale>2: YaRN ramp blending — low freq interpolated, high freq preserved
    """

    def __init__(self, dim=64, base=10000, max_seq=131072):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq = max_seq
        self.scale = 1.0
        self.attn_factor = 1.0
        self.mscale = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _ntk_base(self, base, scale):
        return base * (scale ** (self.dim / (self.dim - 2)))

    def update(self, base: int, scale: float):
        self.base = base
        self.scale = scale
        d = self.dim
        if scale <= 1.0:
            inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
            self.attn_factor = 1.0
            self.mscale = 1.0
        elif scale <= 2.0:
            b_prime = self._ntk_base(base, scale)
            inv = 1.0 / (b_prime ** (torch.arange(0, d, 2).float() / d))
            self.attn_factor = 1.0
            self.mscale = 1.0
        else:
            b_prime = self._ntk_base(base, scale)
            inv_ntk = 1.0 / (b_prime ** (torch.arange(0, d, 2).float() / d))
            inv_interp = inv_ntk / scale
            low = int(d // 2 * 0.3)
            high = int(d // 2 * 0.7)
            inv = inv_ntk.clone()
            if low > 0:
                inv[:low] = 1.0 / (base ** (torch.arange(0, low * 2, 2).float() / d))
            inv[high:] = inv_interp[high:]
            if high > low:
                ramp = torch.linspace(0, 1, high - low)
                inv[low:high] = inv[low:high] * (1 - ramp) + inv_interp[low:high] * ramp
            self.attn_factor = 0.1 * math.log(scale) + 1.0
            self.mscale = min(1.414, max(1.0, 0.1 * math.log(scale) + 1.0))
        self.inv_freq = inv.to(self.inv_freq.device)

    def get_cos_sin(self, seq_len: int, device=None, dtype=None):
        dev = device or self.inv_freq.device
        t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dev))
        emb = torch.cat((freqs, freqs), dim=-1)          # half-split layout
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def rotate_half(x):
    """Half-split rotation, matching get_cos_sin's cat((freqs, freqs)) layout.

    The old interleaved version (x[..., ::2] / x[..., 1::2]) paired dimension i
    with i+1, but cos/sin pair dimension i with i + dim/2. Mixing the two
    conventions rotates by an arbitrary angle per dimension.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    # q,k: [B, H, L, D]; cos,sin: [L, D] -> broadcast over batch and heads
    cos_q = cos[: q.shape[-2]].unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin_q = sin[: q.shape[-2]].unsqueeze(0).unsqueeze(0).to(q.dtype)
    cos_k = cos[: k.shape[-2]].unsqueeze(0).unsqueeze(0).to(k.dtype)
    sin_k = sin[: k.shape[-2]].unsqueeze(0).unsqueeze(0).to(k.dtype)
    return (q * cos_q) + (rotate_half(q) * sin_q), (k * cos_k) + (rotate_half(k) * sin_k)


class SwiGLU(nn.Module):
    """Gated MLP. base1b uses this at mlp_ratio=1.0 to land at ~1.17B params;
    the blueprint's dense 4x GELU at d2048x48 layers alone is 2.4B and will not
    fit 12GB with optimizer state."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock1B(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, head_dim=128, use_qk_norm=True,
                 n_kv_heads: Optional[int] = None, mlp: str = "gelu",
                 mlp_mult: int = 4, mlp_ratio: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.qk_norm_q = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.qk_norm_k = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        if mlp == "swiglu":
            self.mlp = SwiGLU(d_model, int((mlp_ratio or 4.0) * d_model))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * mlp_mult, bias=False),
                nn.GELU(),
                nn.Linear(d_model * mlp_mult, d_model, bias=False),
            )
        # Kept for backward compat with apply_rope_scaling(); the model computes
        # cos/sin once per forward from its own rope and passes them in.
        self.rope = YaRNScaledRoPE(dim=head_dim, base=10000)

    def forward(self, x, cos, sin, attn_factor=1.0):
        B, L, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm prevents logit explosion and entropy collapse at 128k
        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.n_rep > 1:  # grouped-query attention
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # CAUSAL. Was an unmasked full-softmax einsum: the model saw the future.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=attn_factor / math.sqrt(self.head_dim)
        )
        out = out.transpose(1, 2).reshape(B, L, -1)
        x = x + self.o_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.proj = nn.Linear(1024, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, images):
        if images is None:
            return None
        return self.norm(self.proj(images))


class AudioEncoder(nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.proj = nn.Linear(512, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, audio):
        if audio is None:
            return None
        return self.norm(self.proj(audio))


class AvaModel1B(nn.Module):
    """
    Three regimes explicit:
    - early sensory: Vision/Audio encoders (no RoPE)
    - middle workspace: Fusion + Multi-J-Space -> broadcast
    - final motor: Reasoning + LM head collapse to next token
    + Text encoder for RoPE long context
    """

    def __init__(self, vocab_size=128000, d_model=2048, n_text=12, n_fusion=28, n_reason=8,
                 multi_jspace_enabled=True, n_heads=16, head_dim=128, use_qk_norm=True,
                 n_kv_heads=None, mlp="gelu", mlp_mult=4, mlp_ratio=None,
                 tie_lm_head=False, tie_verbalizer=False, multimodal=True,
                 use_memory=False, jspace_slots=None, jspace_half_life=None,
                 jspace_num_heads=4, rope_base=10000, gradient_checkpointing=False,
                 jspace_causal=True, jspace_chunk_size=128):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.multi_jspace_enabled = multi_jspace_enabled
        self.multimodal = multimodal
        # Cross-step workspace persistence. OFF during training: it would carry a
        # graph (and a batch dimension) across steps. ON for eval persistence tests.
        self.use_memory = use_memory
        self.gradient_checkpointing = gradient_checkpointing

        self.embed = nn.Embedding(vocab_size, d_model)

        def _block():
            return TransformerBlock1B(d_model, n_heads=n_heads, head_dim=head_dim,
                                      use_qk_norm=use_qk_norm, n_kv_heads=n_kv_heads,
                                      mlp=mlp, mlp_mult=mlp_mult, mlp_ratio=mlp_ratio)

        self.text_layers = nn.ModuleList([_block() for _ in range(n_text)])
        self.fusion_layers = nn.ModuleList([_block() for _ in range(n_fusion)])
        self.reasoning_layers = nn.ModuleList([_block() for _ in range(n_reason)])

        self.vision_enc = VisionEncoder(d_model) if multimodal else None
        self.audio_enc = AudioEncoder(d_model) if multimodal else None
        self.fusion_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.embed.weight

        # One RoPE for the whole model: every block's schedule is identical, so
        # computing cos/sin per block was pure waste.
        self.rope = YaRNScaledRoPE(dim=head_dim, base=rope_base)

        self.jspace = None
        self.multi_jspace = None
        if multi_jspace_enabled:
            try:
                from multi_jspace_module import MultiJSpace
                self.multi_jspace = MultiJSpace(
                    d_model=d_model, vocab_size=vocab_size, slots=jspace_slots,
                    half_life=jspace_half_life, num_heads=jspace_num_heads,
                    shared_verbalizer_weight=self.lm_head.weight if tie_verbalizer else None,
                    causal=jspace_causal, chunk_size=jspace_chunk_size,
                )
            except ImportError:
                self.multi_jspace = None
        else:
            try:
                from j_space_module import JSpaceModule
                self.jspace = JSpaceModule(d_model=d_model, vocab_size=vocab_size)
            except ImportError:
                self.jspace = None

        self._prev_workspaces = None
        self.rope_base = rope_base
        self.rope_scale = 1.0
        self.init_weights()

    # -- initialization ------------------------------------------------------

    def init_weights(self, std: float = 0.02) -> None:
        """GPT-2/LLaMA-style scaled init.

        Torch's default nn.Embedding init is N(0,1). With lm_head tied to embed
        that puts initial logits at O(sqrt(d)) and cross-entropy at ~196 for a
        vocab of 8192, where ln(8192)=9.01 is the correct value for a uniform
        predictor. Training from there wastes thousands of steps just shrinking
        the output layer, if it doesn't diverge first.

        Residual output projections are additionally scaled by 1/sqrt(2*n_layers)
        so the residual stream variance does not grow with depth.

        (The blueprint's network_init_sota.py multiplies lm_head by 1/sqrt(d);
        with tied weights that silently rescales the embedding table too.)
        """
        n_layers = max(1, self.n_layers)
        resid_std = std / math.sqrt(2 * n_layers)

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
            elif isinstance(mod, nn.MultiheadAttention):
                # in_proj_weight is a raw Parameter, not an nn.Linear
                if getattr(mod, "in_proj_weight", None) is not None:
                    nn.init.normal_(mod.in_proj_weight, mean=0.0, std=std)
                if getattr(mod, "in_proj_bias", None) is not None:
                    nn.init.zeros_(mod.in_proj_bias)
            elif isinstance(mod, RMSNorm):
                nn.init.ones_(mod.weight)

        # residual-path projections: shrink with depth
        for blk in list(self.text_layers) + list(self.fusion_layers) + list(self.reasoning_layers):
            nn.init.normal_(blk.o_proj.weight, mean=0.0, std=resid_std)
            down = blk.mlp.down if isinstance(blk.mlp, SwiGLU) else blk.mlp[2]
            nn.init.normal_(down.weight, mean=0.0, std=resid_std)

        # RMSNorm gains are set last: the loop above may have hit them as Linear-free
        for mod in self.modules():
            if isinstance(mod, RMSNorm):
                nn.init.ones_(mod.weight)

        # lm_head: when tied, it IS embed -- rescaling here would corrupt the
        # embedding table. Only initialize it separately when untied.
        if self.lm_head.weight is not self.embed.weight:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    @property
    def n_layers(self) -> int:
        return len(self.text_layers) + len(self.fusion_layers) + len(self.reasoning_layers)

    # -- workspace memory ----------------------------------------------------

    def reset_memory(self) -> None:
        self._prev_workspaces = None

    def _memory_for(self, batch_size: int):
        """Detached previous workspaces, or None if unusable.

        Two ways this used to explode: (a) the tensors carried a graph into the
        next step's backward, (b) a batch-size change made the slot broadcast
        shapes disagree. Both are now non-events.
        """
        if not self.use_memory or self._prev_workspaces is None:
            return None
        prev = self._prev_workspaces
        if any(t.shape[0] != batch_size for t in prev.values()):
            self._prev_workspaces = None
            return None
        return prev

    def _run_layers(self, layers, x, cos, sin, attn_factor):
        for blk in layers:
            if self.gradient_checkpointing and self.training:
                x = _ckpt(blk, x, cos, sin, attn_factor, use_reentrant=False)
            else:
                x = blk(x, cos, sin, attn_factor)
        return x

    def forward(self, images=None, audio=None, input_ids=None, task_type="deliberate"):
        if input_ids is None:
            raise ValueError("input_ids required")
        B, L = input_ids.shape
        x = self.embed(input_ids)

        # early sensory — no RoPE for vision/audio
        if self.multimodal and images is not None and self.vision_enc is not None:
            v = self.vision_enc(images)
            if v is not None and v.dim() == 3:
                x = x + v.mean(dim=1, keepdim=True)
        if self.multimodal and audio is not None and self.audio_enc is not None:
            a = self.audio_enc(audio)
            if a is not None and a.dim() == 3:
                x = x + a.mean(dim=1, keepdim=True)

        cos, sin = self.rope.get_cos_sin(L, device=x.device)
        af = self.rope.attn_factor

        x = self._run_layers(self.text_layers, x, cos, sin, af)
        x = self._run_layers(self.fusion_layers, x, cos, sin, af)
        fused = self.fusion_norm(x)

        if self.multi_jspace_enabled and self.multi_jspace is not None:
            fused_seq, jspace_out = self.multi_jspace(
                fused, task_type=task_type, prev_workspaces=self._memory_for(B)
            )
            if self.use_memory:
                self._prev_workspaces = {k: v.detach() for k, v in jspace_out["workspaces"].items()}
            enhanced = fused_seq
        elif self.jspace is not None:
            enhanced, jspace_out = self.jspace(fused, task_type=task_type)
        else:
            enhanced = fused
            jspace_out = {"broadcast_strength": torch.norm(enhanced, dim=-1).mean(),
                          "verbalizable_mass": torch.tensor(0.06, device=fused.device)}

        x = self._run_layers(self.reasoning_layers, enhanced, cos, sin, af)
        logits = self.lm_head(x)
        return {"lm_logits": logits, "jspace": jspace_out, "fused": fused}

    # -- branch surgery ------------------------------------------------------

    def freeze_spaces(self, freeze_list: List[str]):
        """freeze_spaces(["system1"]) sets requires_grad=False.
        Supports system1, system2, critic, planner, router, arbitration."""
        if self.multi_jspace is None:
            return
        for n in freeze_list:
            mod = getattr(self.multi_jspace, n, None)
            if mod is None:
                raise ValueError(f"unknown space {n!r}")
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def apply_rope_scaling(model: AvaModel1B, base: int, scale: float):
    model.rope_base = base
    model.rope_scale = scale
    model.rope.update(base, scale)
    # keep per-block ropes in sync for anything still reading them
    for blk in list(model.text_layers) + list(model.fusion_layers) + list(model.reasoning_layers):
        blk.rope.update(base, scale)


def get_model(vocab_size=128000, d_model=2048, multi_jspace_enabled=True):
    """Blueprint-compatible factory. New code should use ava.model.build_model(cfg)."""
    return AvaModel1B(vocab_size=vocab_size, d_model=d_model,
                      multi_jspace_enabled=multi_jspace_enabled)
