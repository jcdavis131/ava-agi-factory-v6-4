"""
multi_jspace_module.py — society of 4 workspaces that collaborate
Solo personal project, no connection to employer, built with public/free-tier only

S1 Fast 32 hl=8 associative
S2 Slow 64 hl=300 verifiable
Critic 16 hl=30 safety/eval-aware
Planner 32 hl=150 deadlines/env_deltas

Fixed per specs/04_model_and_configs.md:

  1. JacobianLens had no `top_concepts` method, but SingleWorkspace.forward
     guarded the call with `hasattr(...)`. The guard was always False, so
     `verbalizable_mass` was the literal constant 0.06 for every input. Every
     reportability metric derived from it was meaningless. Implemented for real.
  2. Each SingleWorkspace allocated TWO full [V, D] verbalizer matrices and
     threw one away. At the blueprint's V=128000, D=2048 that is ~2GB of dead
     parameters per workspace. Now one matrix, optionally tied to lm_head.
  3. `prev_ws` from a previous step with a different batch size raised on
     broadcast. Guarded (and the caller now detaches).
  4. slots/half_life/num_heads were hardcoded; they come from AvaConfig now.

  5. THE WORKSPACE WAS NOT CAUSAL, and this is the important one.
     `attn(slots, fused, fused)` read the ENTIRE sequence, and
     `broad_proj(ws.mean(1)).expand(-1, L, -1)` broadcast that whole-sequence
     summary back to every position -- including position 0. Under teacher
     forcing, the prediction at position t therefore saw tokens > t. Masking
     self-attention does nothing about it; measured leak was ~0.20 in logits at
     early positions while the bare transformer stack measured exactly 0.0.

     Fix: the workspace is now CHUNK-RECURRENT. The sequence is processed in
     chunks of `chunk_size`; the broadcast applied to chunk c is derived from
     the workspace state accumulated over chunks < c only, and the state is
     updated *after* the broadcast is emitted. Chunk 0 broadcasts from the
     learned slot prior, which carries no data. Cross-step `prev_workspaces`
     slots in as the chunk-0 state, so persistence still works across steps.

     This is the standard block-recurrent formulation (cf. Perceiver AR,
     Block-Recurrent Transformer) and is what Global Workspace Theory implies
     anyway: the workspace is a recurrent bottleneck over time, not an oracle
     summary of the future. Cost is ~neutral: the read attention over all
     chunks sums to S*L*D, exactly as the single full-sequence read did.

     Consequence: the broadcast a position sees is up to `chunk_size` tokens
     stale. That is inherent to chunked recurrence; shrink chunk_size to trade
     compute for freshness. `causal=False` restores whole-sequence pooling and
     is ONLY valid for inspection/eval of a complete text, never for training.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SLOTS = {"system1": 32, "system2": 64, "critic": 16, "planner": 32}
DEFAULT_HALF_LIFE = {"system1": 8, "system2": 300, "critic": 30, "planner": 150}


class JacobianLens(nn.Module):
    """Maps a workspace into vocabulary space -- what the workspace 'would say'.

    `concept_vec` takes a real tokenizer id, not a hash. The blueprint's
    eval_branch_harness.py used `sha256(concept) % vocab` against a randn
    matrix, which is a random direction with no relationship to the concept.
    """

    def __init__(self, d_model=2048, vocab_size=128000, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.verbalizer = nn.Linear(d_model, vocab_size, bias=False)
        if weight is not None:
            if tuple(weight.shape) != (vocab_size, d_model):
                raise ValueError(f"shared verbalizer weight {tuple(weight.shape)} != {(vocab_size, d_model)}")
            self.verbalizer.weight = weight  # tied to lm_head

    def concept_vec(self, token_id: int):
        """Unit vector for a concept, addressed by its real token id."""
        vec = self.verbalizer.weight[token_id]
        return F.normalize(vec, dim=0), token_id

    def top_concepts(self, ws: torch.Tensor, k: int = 8):
        """(top_idx, top_vals, mass) for a workspace [B, S, D].

        `mass` is the probability the workspace's mean slot puts on its top-k
        vocabulary items -- the 'verbalizable mass'. A workspace holding a
        crisply nameable concept concentrates mass; a diffuse one does not.
        """
        logits = self.verbalizer(ws.mean(dim=1))          # [B, V]
        probs = F.softmax(logits.float(), dim=-1)
        top_vals, top_idx = probs.topk(min(k, self.vocab_size), dim=-1)
        return top_idx, top_vals, top_vals.sum(dim=-1)     # mass: [B]


class SingleWorkspace(nn.Module):
    def __init__(self, d_model, slots, target_hl, vocab_size=128000, name="S1",
                 num_heads=8, shared_verbalizer_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.name = name
        self.slots = slots
        self.d_model = d_model
        self.target_hl = target_hl
        self.slot_emb = nn.Parameter(torch.randn(1, slots, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.mod_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.broad_proj = nn.Linear(d_model, d_model)

        self.jlens = JacobianLens(d_model, vocab_size, weight=shared_verbalizer_weight)
        self.verbalizer = self.jlens.verbalizer  # alias; single matrix

        # learnable decay: retention(t) = exp(-ln2*t/hl), driven toward target by
        # half_life_loss rather than fixed
        init_decay = math.exp(-math.log(2) / target_hl)
        self.decay_logit = nn.Parameter(
            torch.log(torch.tensor(init_decay / (1 - init_decay + 1e-9))).clamp(-5, 5)
        )

    def decay_factor(self):
        return torch.sigmoid(self.decay_logit).clamp(0.01, 0.99)

    def hl_est(self):
        d = self.decay_factor().item()
        return -math.log(2) / math.log(d) if d < 1 else 1000.0

    def init_state(self, batch_size: int, prev_ws=None):
        """Chunk-0 workspace state: the cross-step carry if usable, else the prior."""
        slots = self.slot_emb.expand(batch_size, -1, -1)
        if prev_ws is not None and prev_ws.shape[0] == batch_size:
            decay = self.decay_factor()
            slots = slots * 0.5 + prev_ws * decay + (1 - decay) * slots * 0.1
        return slots

    def broadcast_from(self, state, length: int):
        """Broadcast emitted to a chunk, derived ONLY from the prefix state."""
        return self.broad_proj(state.mean(dim=1, keepdim=True)).expand(-1, length, -1)

    def read(self, state, chunk):
        """Fold one chunk of tokens into the workspace state (called AFTER broadcast)."""
        decay = self.decay_factor()
        slots = state * decay + self.slot_emb.expand(state.shape[0], -1, -1) * (1 - decay) * 0.1

        ws, _ = self.attn(slots, chunk, chunk)
        ws2, _ = self.self_attn(ws, ws, ws)
        ws = ws + ws2
        gate = self.mod_gate(ws.mean(dim=1, keepdim=True))
        return ws * gate

    def probe(self, state):
        """Read-only diagnostics of a workspace state (no gradient, no logits path)."""
        with torch.no_grad():
            top_idx, top_vals, v_mass = self.jlens.top_concepts(state)
        return {"verbalizable_mass": v_mass.mean(), "top_concepts": top_idx,
                "top_probs": top_vals, "hl_est": self.hl_est()}

    def forward(self, fused, prev_ws=None):
        """Non-causal whole-sequence read. Inspection only -- see module docstring."""
        B, L, _ = fused.shape
        state = self.read(self.init_state(B, prev_ws), fused)
        broadcast = self.broadcast_from(state, L)
        b_str = broadcast.norm(dim=-1).mean() / (fused.norm(dim=-1).mean() + 1e-6)
        m = self.probe(state)
        m.update({"broadcast_strength": b_str, "workspace": state})
        return state, broadcast, m


class Router(nn.Module):
    _TASK_BIAS = {
        "automatic": (0.6, -0.3, -0.2, -0.2),
        "deliberate": (-0.3, 0.6, -0.2, 0.0),
        "safety": (-0.4, -0.2, 0.8, -0.2),
        "temporal": (-0.4, 0.0, -0.2, 0.6),
    }

    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 4))
        # Branch prior (log-space), set by BRANCH_CONFIGS router_bias. None = off.
        self.register_buffer("branch_bias", torch.zeros(4), persistent=True)
        self._branch_bias_active = False

    def set_branch_bias(self, probs: Optional[list]) -> None:
        if probs is None:
            self.branch_bias.zero_()
            self._branch_bias_active = False
            return
        if len(probs) != 4:
            raise ValueError(f"router_bias must have 4 entries, got {len(probs)}")
        p = torch.tensor(probs, dtype=self.branch_bias.dtype, device=self.branch_bias.device)
        self.branch_bias.copy_(torch.log(p.clamp_min(1e-6)))
        self._branch_bias_active = True

    def forward(self, pooled, task_type="deliberate"):
        logits = self.mlp(pooled)  # [B,4] -> [S1, S2, Critic, Planner]
        bias = torch.tensor(self._TASK_BIAS.get(task_type, self._TASK_BIAS["deliberate"]),
                            device=logits.device, dtype=logits.dtype) * 1.5
        logits_b = logits + bias
        if self._branch_bias_active:
            logits_b = logits_b + self.branch_bias.to(logits.dtype)
        return F.softmax(logits_b, dim=-1), logits


class Arbitration(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model * 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, s1_mean, s2_mean):
        x = torch.cat([s1_mean, s2_mean], dim=-1)
        return torch.sigmoid(self.mlp(x))  # S2 can veto S1


SPACE_NAMES = ("system1", "system2", "critic", "planner")


class MultiJSpace(nn.Module):
    def __init__(self, d_model=2048, vocab_size=128000, slots=None, half_life=None,
                 num_heads=4, shared_verbalizer_weight: Optional[torch.Tensor] = None,
                 causal: bool = True, chunk_size: int = 128):
        super().__init__()
        slots = slots or DEFAULT_SLOTS
        half_life = half_life or DEFAULT_HALF_LIFE
        ws_heads = 8 if d_model % 8 == 0 else num_heads
        self.causal = causal
        self.chunk_size = chunk_size

        def _ws(name):
            return SingleWorkspace(d_model, slots=slots[name], target_hl=half_life[name],
                                   vocab_size=vocab_size, name=name, num_heads=ws_heads,
                                   shared_verbalizer_weight=shared_verbalizer_weight)

        self.system1 = _ws("system1")
        self.system2 = _ws("system2")
        self.critic = _ws("critic")
        self.planner = _ws("planner")
        self.router = Router(d_model)
        self.arbitration = Arbitration(d_model)
        self.cross_s1_reads_s2 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_s2_reads_s1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_s2_reads_planner = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_critic_reads_all = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def spaces(self):
        return {n: getattr(self, n) for n in SPACE_NAMES}

    def _mix(self, states):
        """Cross-space attention. Operates on prefix-only states, so it is causal."""
        ws1, ws2, wsc, wsp = (states[n] for n in SPACE_NAMES)
        ws1_r, _ = self.cross_s1_reads_s2(ws1, ws2, ws2)
        ws2_r1, _ = self.cross_s2_reads_s1(ws2, ws1, ws1)
        ws2_r2, _ = self.cross_s2_reads_planner(ws2, wsp, wsp)
        ws2 = ws2 + ws2_r1 * 0.3 + ws2_r2 * 0.3
        ws1 = ws1 + ws1_r * 0.2
        all_ws = torch.cat([ws1, ws2, wsc, wsp], dim=1)
        wsc_r, _ = self.cross_critic_reads_all(wsc, all_ws, all_ws)
        wsc = wsc + wsc_r * 0.4
        return {"system1": ws1, "system2": ws2, "critic": wsc, "planner": wsp}

    def _emit(self, states, chunk_len, task_type):
        """Broadcast for one chunk, from prefix state only. Returns (combined, route, veto)."""
        B = states["system1"].shape[0]
        b = {n: self.spaces()[n].broadcast_from(states[n], chunk_len) for n in SPACE_NAMES}

        pooled = torch.stack([states[n].mean(dim=1) for n in SPACE_NAMES], 0).mean(0)
        route_probs, route_logits = self.router(pooled, task_type=task_type)
        veto = self.arbitration(states["system1"].mean(dim=1), states["system2"].mean(dim=1))

        w = [route_probs[:, i].view(B, 1, 1) for i in range(4)]
        w[1] = w[1] * (1 + veto.view(B, 1, 1) * 0.5)  # confident S2 veto raises its weight
        combined = sum(wi * b[n] for wi, n in zip(w, SPACE_NAMES))
        return combined, route_probs, route_logits, veto

    def forward(self, fused, task_type="deliberate", prev_workspaces=None):
        B, L, _ = fused.shape
        prev = prev_workspaces or {}
        spaces = self.spaces()
        states = {n: spaces[n].init_state(B, prev.get(n)) for n in SPACE_NAMES}

        if not self.causal:
            # Inspection path: read the whole sequence, THEN broadcast. Leaks the
            # future by construction -- never use for training.
            states = self._mix({n: spaces[n].read(states[n], fused) for n in SPACE_NAMES})
            combined_all, route_probs, route_logits, veto = self._emit(states, L, task_type)
            fused_out = fused + combined_all
            n_chunks = 1
        else:
            chunk = min(self.chunk_size, L)
            outs, routes, logits_acc, vetos = [], [], [], []

            for start in range(0, L, chunk):
                seg = fused[:, start:start + chunk]
                cl = seg.shape[1]

                # 1) broadcast from the PREFIX state (chunk 0 -> learned prior: no data)
                combined, rp, rl, veto = self._emit(states, cl, task_type)
                outs.append(seg + combined)
                routes.append(rp)
                logits_acc.append(rl)
                vetos.append(veto)

                # 2) only now fold this chunk into the state, for the *next* chunk
                states = self._mix({n: spaces[n].read(states[n], seg) for n in SPACE_NAMES})

            fused_out = torch.cat(outs, dim=1)
            combined_all = fused_out - fused
            route_probs = torch.stack(routes, 0).mean(0)
            route_logits = torch.stack(logits_acc, 0).mean(0)
            veto = torch.stack(vetos, 0).mean(0)
            n_chunks = len(outs)

        # Diagnostics read the FINAL state (has seen the whole sequence). Safe:
        # they never touch the logits path, only metrics and losses.
        per_space = {}
        for n in SPACE_NAMES:
            m = spaces[n].probe(states[n])
            m["workspace"] = states[n]
            m["broadcast_strength"] = (
                spaces[n].broadcast_from(states[n], 1).norm(dim=-1).mean()
                / (fused.norm(dim=-1).mean() + 1e-6)
            )
            per_space[n] = m

        metrics = {
            **per_space,
            "route_probs": route_probs, "route_logits": route_logits,
            "veto": veto.mean(), "broadcast": combined_all,
            "broadcast_strength": combined_all.norm(dim=-1).mean() / (fused.norm(dim=-1).mean() + 1e-6),
            "workspaces": states,
            "n_chunks": n_chunks,
        }
        return fused_out, metrics


# Losses
class MultiJSpaceLosses(nn.Module):
    def __init__(self):
        super().__init__()

    def half_life_loss(self, workspace: SingleWorkspace, target: float):
        decay = workspace.decay_factor()
        target_decay = math.exp(-math.log(2) / target)
        return F.mse_loss(decay, torch.tensor(target_decay, device=decay.device))

    def inter_space_mi_regularizer(self, ws1, ws2, target_cos=0.45):
        # I(S1;S2) proxy via cosine 0.3-0.6 complementary
        c1 = ws1.mean(dim=1)
        c2 = ws2.mean(dim=1)
        cos = F.cosine_similarity(c1, c2, dim=-1).mean()
        return F.mse_loss(cos, torch.tensor(target_cos, device=cos.device))

    def routing_loss(self, route_probs, task_type="deliberate"):
        target_map = {
            "automatic": torch.tensor([0.6, 0.15, 0.1, 0.15]),
            "deliberate": torch.tensor([0.15, 0.55, 0.1, 0.2]),
            "safety": torch.tensor([0.1, 0.2, 0.6, 0.1]),
            "temporal": torch.tensor([0.1, 0.3, 0.1, 0.5]),
        }
        tgt = target_map.get(task_type, target_map["deliberate"]).to(route_probs.device)
        tgt = tgt.unsqueeze(0).expand_as(route_probs)
        return F.kl_div(route_probs.clamp(1e-6, 1).log(), tgt, reduction="batchmean")

    def reportability_loss(self, workspace, target_concepts, j_lens):
        # CE(verbalizer(workspace.mean), target_concept)
        logits = j_lens.verbalizer(workspace.mean(dim=1))
        return F.cross_entropy(logits, target_concepts)

    def broadcast_loss(self, b_strength, target=0.2):
        return F.mse_loss(b_strength, torch.tensor(target, device=b_strength.device))

    def selectivity_loss(self, ws_var, task_type):
        # automatic -> low variance, deliberate -> high variance
        return ws_var if task_type == "automatic" else -ws_var

    def modulation_loss(self, sim_with, sim_without, margin=0.5):
        return F.relu(margin - (sim_with - sim_without)).mean()


def compute_half_life_curves(decay, max_tokens=200):
    hl = -math.log(2) / math.log(decay) if decay < 1 else 1000
    return [math.exp(-math.log(2) * t / hl) for t in range(max_tokens)]


def compute_capacity_law():
    ks = [2, 4, 6, 8, 10, 12, 16, 20, 25, 32]
    # Dehaene: tracking ~25 but ~6 distinct due to overlap
    s1 = [math.exp(-0.12 * max(0, k - 6)) for k in ks]
    s2 = [math.exp(-0.08 * max(0, k - 10)) for k in ks]
    combined = [0.6 * b + 0.4 * a for a, b in zip(s1, s2)]
    return ks, s1, s2, combined
