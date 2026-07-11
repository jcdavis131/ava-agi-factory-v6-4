"""
Ava 1B — YaRN + LongRoPE2 (non-uniform per-dim) RoPE 10k→1M + QK-Norm + Peri-LN + 4 attention sinks + Multi-J-Space
Solo personal project, no connection to employer, built with public/free-tier only

Implements hill-climb 1:
- LongRoPE2 non-uniform per-dim factors + resonance mitigation, critical_dim_shift 31->25 (YaRN 10x less tokens preserved)
- Peri-LN: QK-L2-Norm (RMSNorm) + output-LN after attn + after FFN
- 4 attention sinks (Xiao et al. 2023) as learnable [H,4,D] KV, always-attended
- Backward compat: YaRNScaledRoPE still available, flag rope_type="longrope2" or "yarn"
"""
import math
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

# ─── YaRN (baseline, kept) ──────────────────────────────────────────────────
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
            low = int(d//2 * 0.3)
            high = int(d//2 * 0.7)
            inv = inv_ntk.clone()
            inv[:low] = 1.0 / (base ** (torch.arange(0, low*2, 2).float() / d)) if low>0 else inv[:low]
            inv[high:] = inv_interp[high:]
            if high>low:
                ramp = torch.linspace(0,1,high-low)
                inv[low:high] = inv[low:high]*(1-ramp) + inv_interp[low:high]*ramp
            self.attn_factor = 0.1*math.log(scale)+1.0
            self.mscale = min(1.414, max(1.0, self.mscale)) if scale>1 else 1.0
            self.mscale = min(1.414, max(1.0, 0.1*math.log(scale)+1.0)) if scale>1 else 1.0
        self.inv_freq = inv.to(self.inv_freq.device)

    def get_cos_sin(self, seq_len: int, device=None):
        dev = device or self.inv_freq.device
        t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        return cos, sin

# ─── LongRoPE2 ───────────────────────────────────────────────────────────────
def longrope2_factors(dim: int, base: int, scale: float, critical_dim_shift: int = 6, sharpness: float = 12.0) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    LongRoPE2 non-uniform per-dim factors + resonance mitigation
    dim: head_dim (e.g. 64 -> 32 pairs, 128 -> 64 pairs)
    base: 10000
    scale: extension factor (1 = 10k 2k ctx, 100 = 1M 128k ctx) = target/original
    critical_dim_shift: 31->25 means shift 6 for n_pairs=32 reference at scale=100
    Returns: (inv_freq [n_pairs], lambda_factors [n_pairs], critical, critical_t)

    Idea: per-dim lambda_i = 1 + (scale-1) * sigmoid_k(t - crit_t) ^0.65 * resonance
          YaRN 10x less tokens preserved via mscale + attn_factor unchanged
          Search-like: evolutionary discovered that mid freqs need earlier interpolation than linear ramp — we mimic with power 0.65 and sinusoidal jitter 1.5%
    """
    n_pairs = dim // 2
    j = torch.arange(n_pairs).float()
    exponent = (2 * j) / dim
    inv_base = 1.0 / (base ** exponent)

    if scale <= 1.0:
        lam = torch.ones(n_pairs)
        return inv_base, lam, 31.0, 31.0/32.0

    # critical 31 -> 25 shift in log space: for scale=1 keep 31, for scale=100 ->25
    critical_start = 31.0
    critical_end = 31.0 - float(critical_dim_shift)  # 25
    log_ratio = math.log(scale) / math.log(100.0)
    log_ratio = min(1.0, max(0.0, log_ratio))
    critical = critical_start - (critical_start - critical_end) * log_ratio
    critical_t = critical / 32.0  # ratio reference for 32 pairs, keeps same proportion for other dims

    t = j / float(n_pairs)  # 0..~1
    # sigmoid sharpness k=12 gives LongRoPE2-like steep but not step
    # non-uniform: power 0.65 mimics evolutionary search pushing mid dims earlier
    sig = 1.0 / (1.0 + torch.exp(-sharpness * (t - critical_t)))
    lam = 1.0 + (scale - 1.0) * (sig ** 0.65)

    # resonance mitigation: LongRoPE2 notes dimensions where wavelength ~ seq cause attention spikes
    # add 1.5% sinusoidal jitter phase dependent on log(scale) to avoid exact multiples — n=1 projection sufficient per OroJaR paper
    resonance = 1.0 + 0.015 * torch.sin(j * 2.7 + math.log(scale + 1.0) * 1.3)
    lam = lam * resonance

    # clamp to [1, scale*1.02] to avoid overshoot
    lam = torch.clamp(lam, min=1.0, max=scale * 1.02)

    inv_final = inv_base / lam
    return inv_final, lam, critical, critical_t

class LongRoPE2ScaledRoPE(nn.Module):
    """
    LongRoPE2: Near-lossless 128k via non-uniform per-dim factors + evolutionary search
    - per-dim lambda_i replaces uniform YaRN ramp
    - resonance mitigation 1.5% sinusoid
    - critical dim shift 31->25 as scale 1->100
    - keeps YaRN 10x less tokens property via same mscale/attn_factor formula (0.1*ln(scale)+1)
    Ref: LongRoPE2: Near-Lossless LLM Context Window Scaling arxiv 2412 etc
    """
    def __init__(self, dim=64, base=10000, max_seq=131072, critical_dim_shift=6):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq = max_seq
        self.critical_dim_shift = critical_dim_shift
        self.scale = 1.0
        self.attn_factor = 1.0
        self.mscale = 1.0
        inv_freq, _, _, _ = longrope2_factors(dim, base, 1.0, critical_dim_shift)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("lambda_factors", torch.ones(dim//2), persistent=False)
        self.critical = 31.0
        self.critical_t = 31.0/32.0

    def update(self, base: int, scale: float):
        self.base = base
        self.scale = scale
        inv_freq, lam, crit, crit_t = longrope2_factors(self.dim, base, scale, self.critical_dim_shift)
        self.inv_freq = inv_freq.to(self.inv_freq.device)
        self.lambda_factors = lam.to(self.lambda_factors.device)
        self.critical = crit
        self.critical_t = crit_t
        if scale <= 1.0:
            self.attn_factor = 1.0
            self.mscale = 1.0
        else:
            # YaRN 10x less tokens property preserved: mscale 1.1->1.414, attn_factor 0.1*ln(s)+1
            self.attn_factor = 0.1 * math.log(scale) + 1.0
            self.mscale = min(1.414, max(1.0, 0.1 * math.log(scale) + 1.0))

    def get_cos_sin(self, seq_len: int, device=None):
        dev = device or self.inv_freq.device
        t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    q_cos = q * cos[:q.shape[-2]].unsqueeze(0).unsqueeze(0)
    q_sin = rotate_half(q) * sin[:q.shape[-2]].unsqueeze(0).unsqueeze(0)
    k_cos = k * cos[:k.shape[-2]].unsqueeze(0).unsqueeze(0)
    k_sin = rotate_half(k) * sin[:k.shape[-2]].unsqueeze(0).unsqueeze(0)
    return q_cos + q_sin, k_cos + k_sin

class TransformerBlock1B(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, head_dim=128, use_qk_norm=True, rope_type="yarn", n_sinks=4, use_peri_ln=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_type = rope_type
        self.n_sinks = n_sinks
        self.use_peri_ln = use_peri_ln
        self.q_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, d_model, bias=False)
        self.qk_norm_q = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.qk_norm_k = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        # Peri-LN: output-LN after attn + after FFN (Peri-LN paper: QK-Norm+output-LN improves loss/stability 400M-1B)
        self.peri_norm_attn = RMSNorm(d_model) if use_peri_ln else nn.Identity()
        self.peri_norm_mlp = RMSNorm(d_model) if use_peri_ln else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4, bias=False),
            nn.GELU(),
            nn.Linear(d_model*4, d_model, bias=False)
        )
        if rope_type == "longrope2":
            self.rope = LongRoPE2ScaledRoPE(dim=head_dim, base=10000, critical_dim_shift=6)
        else:
            self.rope = YaRNScaledRoPE(dim=head_dim, base=10000)

        # 4 attention sinks: learnable KV [H, 4, D] — Xiao et al. 2023, always-attended to absorb excess mass
        if n_sinks > 0:
            self.sink_k = nn.Parameter(torch.randn(n_heads, n_sinks, head_dim) * 0.02)
            self.sink_v = nn.Parameter(torch.randn(n_heads, n_sinks, head_dim) * 0.02)
        else:
            self.sink_k = None
            self.sink_v = None

    def forward(self, x, cos, sin, attn_factor=1.0):
        B,L,D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)  # [B,H,L,D]
        k = self.q_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2) if False else self.k_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)

        # QK-L2-Norm (QK-RMSNorm) prevents logit explosion and entropy collapse at 128k
        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)
        q,k = apply_rotary_emb(q,k,cos,sin)

        # ── Attention sinks: concat [sinks + seq] for K/V ──
        if self.n_sinks > 0 and self.sink_k is not None:
            # [H, S, D] -> [B,H,S,D]
            sink_k = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1)
            sink_v = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1)
            k_full = torch.cat([sink_k, k], dim=2)  # [B,H,S+L,D]
            v_full = torch.cat([sink_v, v], dim=2)
            # scores [B,H,L,S+L]
            scale = attn_factor / math.sqrt(self.head_dim)
            attn_scores = torch.einsum('b h l d, b h m d -> b h l m', q, k_full) * scale

            # causal mask: sinks always allowed, original tokens causal
            # build mask [L, S+L] bool allowed
            S = self.n_sinks
            # q pos l can attend to sink 0..S-1 always, and to original pos 0..l
            # k_full index: 0..S-1 sinks, S..S+L-1 original org pos 0..L-1
            # create col indices
            # Use torch operations for mypyc-ready
            causal_mask = torch.ones(L, S+L, device=x.device, dtype=torch.bool)
            # for each query l, disallow future original tokens
            # original pos p = col - S, allow if p <= l
            # build via arange
            q_idx = torch.arange(L, device=x.device).unsqueeze(1)  # [L,1]
            k_orig_idx = torch.arange(L, device=x.device).unsqueeze(0)  # [1,L]
            # future mask for original part
            future = k_orig_idx > q_idx  # [L,L] True if future
            # place into full mask
            causal_mask[:, S:] = ~future  # allow past+present
            # apply: masked positions -> -inf
            # expand to [B,H,L,S+L] broadcasting
            # Convert bool to float mask
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn_scores, dim=-1)
            out = torch.einsum('b h l m, b h m d -> b h l d', attn, v_full).transpose(1,2).reshape(B,L,-1)
        else:
            scale = attn_factor / math.sqrt(self.head_dim)
            attn = torch.einsum('b h l d, b h m d -> b h l m', q, k) * scale
            # causal
            causal = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('b h l m, b h m d -> b h l d', attn, v).transpose(1,2).reshape(B,L,-1)

        out = self.o_proj(out)
        # Peri-LN after attention
        out = self.peri_norm_attn(out) if self.use_peri_ln else out
        x = x + out

        h2 = self.norm2(x)
        mlp_out = self.mlp(h2)
        mlp_out = self.peri_norm_mlp(mlp_out) if self.use_peri_ln else mlp_out
        x = x + mlp_out
        return x

class VisionEncoder(nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.proj = nn.Linear(1024, d_model)
        self.norm = RMSNorm(d_model)
    def forward(self, images):
        if images is None: return None
        return self.norm(self.proj(images))

class AudioEncoder(nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.proj = nn.Linear(512, d_model)
        self.norm = RMSNorm(d_model)
    def forward(self, audio):
        if audio is None: return None
        return self.norm(self.proj(audio))

class AvaModel1B(nn.Module):
    """
    Three regimes explicit:
    - early sensory: Vision/Audio encoders (no RoPE)
    - middle workspace: Fusion (28 layers) + J-space (32 slots vs 65k LTM) -> broadcast
    - final motor: Reasoning (8 layers) + LM head collapse to next token
    + Text encoder 12 layers for RoPE long context
    Total RoPE modules 56
    New: rope_type flag yarn|longrope2, peri-ln, 4 sinks
    """
    def __init__(self, vocab_size=128000, d_model=2048, n_text=12, n_fusion=28, n_reason=8, multi_jspace_enabled=True, rope_type="yarn", n_sinks=4, use_peri_ln=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.multi_jspace_enabled = multi_jspace_enabled
        self.rope_type = rope_type
        self.n_sinks = n_sinks
        self.use_peri_ln = use_peri_ln
        self.embed = nn.Embedding(vocab_size, d_model)
        def _make_block():
            return TransformerBlock1B(d_model=d_model, n_heads=16, head_dim=128, rope_type=rope_type, n_sinks=n_sinks, use_peri_ln=use_peri_ln)
        self.text_layers = nn.ModuleList([_make_block() for _ in range(n_text)])
        self.fusion_layers = nn.ModuleList([_make_block() for _ in range(n_fusion)])
        self.reasoning_layers = nn.ModuleList([_make_block() for _ in range(n_reason)])
        self.vision_enc = VisionEncoder(d_model)
        self.audio_enc = AudioEncoder(d_model)
        self.fusion_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.jspace = None
        self.multi_jspace = None
        if multi_jspace_enabled:
            try:
                from multi_jspace_module import MultiJSpace
                self.multi_jspace = MultiJSpace(d_model=d_model, vocab_size=vocab_size)
            except ImportError:
                self.multi_jspace = None
        else:
            try:
                from j_space_module import JSpaceModule
                self.jspace = JSpaceModule(d_model=d_model, vocab_size=vocab_size)
            except ImportError:
                self.jspace = None
        self._prev_workspaces = None
        self.rope_base = 10000
        self.rope_scale = 1.0

    def forward(self, images=None, audio=None, input_ids=None, task_type="deliberate"):
        if input_ids is None:
            raise ValueError("input_ids required")
        B,L = input_ids.shape
        x = self.embed(input_ids)
        v = self.vision_enc(images) if images is not None else None
        a = self.audio_enc(audio) if audio is not None else None
        if v is not None:
            x = x + v.mean(dim=1, keepdim=False).unsqueeze(1) if v.dim()==3 else x
        for blk in self.text_layers:
            cos, sin = blk.rope.get_cos_sin(L, device=x.device)
            x = blk(x, cos, sin, attn_factor=blk.rope.attn_factor)
        for blk in self.fusion_layers:
            cos, sin = blk.rope.get_cos_sin(L, device=x.device)
            x = blk(x, cos, sin, attn_factor=blk.rope.attn_factor)
        fused = self.fusion_norm(x)
        jspace_out = {}
        if self.multi_jspace_enabled and self.multi_jspace is not None:
            fused_seq, jspace_out = self.multi_jspace(fused, task_type=task_type, prev_workspaces=self._prev_workspaces)
            self._prev_workspaces = jspace_out.get("workspaces")
            enhanced = fused_seq
        elif self.jspace is not None:
            enhanced, jspace_out = self.jspace(fused, task_type=task_type)
        else:
            enhanced = fused
            jspace_out = {"broadcast_strength": torch.norm(enhanced, dim=-1).mean(),
                          "verbalizable_mass": torch.tensor(0.06)}
        x = enhanced
        for blk in self.reasoning_layers:
            cos, sin = blk.rope.get_cos_sin(L, device=x.device)
            x = blk(x, cos, sin, attn_factor=blk.rope.attn_factor)
        logits = self.lm_head(x)
        return {"lm_logits": logits, "jspace": jspace_out, "fused": fused}

    def freeze_spaces(self, freeze_list: List[str]):
        if self.multi_jspace is None: return
        name_map = {
            "system1": getattr(self.multi_jspace, "system1", None),
            "system2": getattr(self.multi_jspace, "system2", None),
            "critic": getattr(self.multi_jspace, "critic", None),
            "planner": getattr(self.multi_jspace, "planner", None),
            "router": getattr(self.multi_jspace, "router", None),
            "arbitration": getattr(self.multi_jspace, "arbitration", None),
        }
        for n in freeze_list:
            mod = name_map.get(n)
            if mod is not None:
                for p in mod.parameters(): p.requires_grad=False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad=True

def apply_rope_scaling(model: AvaModel1B, base: int, scale: float):
    model.rope_base = base
    model.rope_scale = scale
    for blk in list(model.text_layers)+list(model.fusion_layers)+list(model.reasoning_layers):
        blk.rope.update(base, scale)

def get_model(vocab_size=128000, d_model=2048, multi_jspace_enabled=True, rope_type="yarn", n_sinks=4, use_peri_ln=True):
    return AvaModel1B(vocab_size=vocab_size, d_model=d_model, multi_jspace_enabled=multi_jspace_enabled, rope_type=rope_type, n_sinks=n_sinks, use_peri_ln=use_peri_ln)

# ─── quick forward test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Solo personal project, no connection to employer, built with public/free-tier only")
    for rt in ["yarn", "longrope2"]:
        print(f"\n=== Testing rope_type={rt} ===")
        m = get_model(vocab_size=1024, d_model=512, multi_jspace_enabled=False, rope_type=rt, n_sinks=4, use_peri_ln=True)
        m.eval()
        ids = torch.randint(0,1024,(1,128))
        with torch.no_grad():
            # simulate scaling 10k->1M scale=100
            apply_rope_scaling(m, base=1000000, scale=100.0)
            out = m(input_ids=ids)
            print(f"  logits {out['lm_logits'].shape} fused {out['fused'].shape}")
            # print lambda factors for longrope2
            if rt == "longrope2":
                blk = m.text_layers[0]
                print(f"  lambda_factors[:8] {blk.rope.lambda_factors[:8].tolist()}")
                print(f"  lambda_factors[-8:] {blk.rope.lambda_factors[-8:].tolist()}")
                print(f"  critical {blk.rope.critical:.2f} crit_t {blk.rope.critical_t:.3f} attn_factor {blk.rope.attn_factor:.3f} mscale {blk.rope.mscale:.3f}")
        # test sinks
        print(f"  sinks K {m.text_layers[0].sink_k.shape} V {m.text_layers[0].sink_v.shape}")
        print(f"  peri_norm present {hasattr(m.text_layers[0], 'peri_norm_attn')}")
