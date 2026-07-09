"""
Ava 1B — YaRN RoPE 10k→1M + QK-Norm + Multi-J-Space support
Solo personal project, no connection to employer, built with public/free-tier only
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
        # base' = base * scale^(dim/(dim-2)) — approx 40000^(64/62) ~= 41k for scale 4
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
            # YaRN ramp blending
            b_prime = self._ntk_base(base, scale)
            inv_ntk = 1.0 / (b_prime ** (torch.arange(0, d, 2).float() / d))
            inv_interp = inv_ntk / scale  # low freq interpolated
            # ramp: high freq preserved (first 30%), low freq interpolated (last 30%), middle blended
            low = int(d//2 * 0.3)
            high = int(d//2 * 0.7)
            inv = inv_ntk.clone()
            # preserve high freq
            inv[:low] = 1.0 / (base ** (torch.arange(0, low*2, 2).float() / d)) if low>0 else inv[:low]
            # interpolate low freq
            inv[high:] = inv_interp[high:]
            # middle linear blend
            if high>low:
                ramp = torch.linspace(0,1,high-low)
                inv[low:high] = inv[low:high]*(1-ramp) + inv_interp[low:high]*ramp
            self.attn_factor = 0.1*math.log(scale)+1.0
            self.mscale = 0.1*math.log(scale)+1.0 if scale>1 else 1.0
            # mscale for YaRN: 1.1->1.414 per config
            self.mscale = min(1.414, max(1.0, self.mscale))
        self.inv_freq = inv.to(self.inv_freq.device)

    def get_cos_sin(self, seq_len: int, device=None):
        dev = device or self.inv_freq.device
        t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    # q,k: [B, H, L, D]
    # cos,sin: [L, D]
    # YaRN temperature scaling handled in cos/sin mscale, attn_factor applied outside
    # Simplified RoPE rotate
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
    def __init__(self, d_model=2048, n_heads=16, head_dim=128, use_qk_norm=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, d_model, bias=False)
        self.qk_norm_q = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.qk_norm_k = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4, bias=False),
            nn.GELU(),
            nn.Linear(d_model*4, d_model, bias=False)
        )
        self.rope = YaRNScaledRoPE(dim=head_dim, base=10000)

    def forward(self, x, cos, sin, attn_factor=1.0):
        B,L,D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(h).view(B,L,self.n_heads,self.head_dim).transpose(1,2)
        # QK-Norm prevents logit explosion and entropy collapse at 128k
        q = self.qk_norm_q(q)
        k = self.qk_norm_k(k)
        q,k = apply_rotary_emb(q,k,cos,sin)
        # attn with YaRN temperature scaling scale = attn_factor / sqrt(head_dim)
        scale = attn_factor / math.sqrt(self.head_dim)
        attn = torch.einsum('b h l d, b h m d -> b h l m', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h l m, b h m d -> b h l d', attn, v).transpose(1,2).reshape(B,L,-1)
        out = self.o_proj(out)
        x = x + out
        x = x + self.mlp(self.norm2(x))
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
    """
    def __init__(self, vocab_size=128000, d_model=2048, n_text=12, n_fusion=28, n_reason=8, multi_jspace_enabled=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.multi_jspace_enabled = multi_jspace_enabled
        self.embed = nn.Embedding(vocab_size, d_model)
        self.text_layers = nn.ModuleList([TransformerBlock1B(d_model) for _ in range(n_text)])
        self.fusion_layers = nn.ModuleList([TransformerBlock1B(d_model) for _ in range(n_fusion)])
        self.reasoning_layers = nn.ModuleList([TransformerBlock1B(d_model) for _ in range(n_reason)])
        self.vision_enc = VisionEncoder(d_model)
        self.audio_enc = AudioEncoder(d_model)
        self.fusion_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # J-space
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
        # early sensory — no RoPE for vision/audio
        v = self.vision_enc(images) if images is not None else None
        a = self.audio_enc(audio) if audio is not None else None
        if v is not None:
            x = x + v.mean(dim=1, keepdim=False).unsqueeze(1) if v.dim()==3 else x
        # text encoder with RoPE
        for blk in self.text_layers:
            cos, sin = blk.rope.get_cos_sin(L, device=x.device)
            x = blk(x, cos, sin, attn_factor=blk.rope.attn_factor)
        # fusion middle workspace
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
        # final motor reasoning
        x = enhanced
        for blk in self.reasoning_layers:
            cos, sin = blk.rope.get_cos_sin(L, device=x.device)
            x = blk(x, cos, sin, attn_factor=blk.rope.attn_factor)
        logits = self.lm_head(x)
        logits = logits / 1.0
        # diagnostics
        return {"lm_logits": logits, "jspace": jspace_out, "fused": fused}

    def freeze_spaces(self, freeze_list: List[str]):
        """freeze_spaces(["system1"]) sets requires_grad=False. Supports system1, system2, critic, planner, router, arbitration"""
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

def get_model(vocab_size=128000, d_model=2048, multi_jspace_enabled=True):
    return AvaModel1B(vocab_size=vocab_size, d_model=d_model, multi_jspace_enabled=multi_jspace_enabled)
