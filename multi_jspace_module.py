"""
multi_jspace_module.py — society of 4 workspaces that collaborate
Solo personal project, no connection to employer, built with public/free-tier only

S1 Fast 32 hl=8 associative
S2 Slow 64 hl=300 verifiable
Critic 16 hl=30 safety/eval-aware
Planner 32 hl=150 deadlines/env_deltas
"""
import math, hashlib
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class JacobianLens(nn.Module):
    def __init__(self, d_model=2048, vocab_size=128000):
        super().__init__()
        self.verbalizer = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model=d_model
        self.vocab_size=vocab_size
    def concept_vec(self, concept:str):
        tid=int(hashlib.sha256(concept.encode()).hexdigest(),16)%self.vocab_size
        vec=self.verbalizer.weight[tid]
        return F.normalize(vec, dim=0), tid

class SingleWorkspace(nn.Module):
    def __init__(self, d_model, slots, target_hl, vocab_size=128000, name="S1"):
        super().__init__()
        self.name=name
        self.slots=slots
        self.d_model=d_model
        self.target_hl=target_hl
        self.slot_emb=nn.Parameter(torch.randn(1,slots,d_model)*0.02)
        self.attn=nn.MultiheadAttention(d_model,num_heads=8,batch_first=True)
        self.self_attn=nn.MultiheadAttention(d_model,num_heads=8,batch_first=True)
        self.mod_gate=nn.Sequential(nn.Linear(d_model,d_model), nn.Sigmoid())
        self.broad_proj=nn.Linear(d_model,d_model)
        self.verbalizer=JacobianLens(d_model,vocab_size).verbalizer  # [V,D] for real-mode Jacobian
        self.jlens=JacobianLens(d_model,vocab_size)
        self.jlens.verbalizer=self.verbalizer  # share
        # learnable decay via logit -> half-life controllable via loss, cos(t)=exp(-ln2*t/hl)
        init_decay=math.exp(-math.log(2)/target_hl)
        # logit = inv sigmoid
        self.decay_logit=nn.Parameter(torch.log(torch.tensor(init_decay/(1-init_decay+1e-9))).clamp(-5,5))
    
    def decay_factor(self):
        return torch.sigmoid(self.decay_logit).clamp(0.01,0.99)
    
    def hl_est(self):
        d=self.decay_factor().item()
        return -math.log(2)/math.log(d) if d<1 else 1000.0

    def forward(self, fused, prev_ws=None):
        B,L,D=fused.shape
        slots=self.slot_emb.expand(B,-1,-1)
        if prev_ws is not None:
            # persistence exp(-ln2*t/hl)
            decay=self.decay_factor()
            slots = slots*0.5 + prev_ws*decay + (1-decay)*slots*0.1
        ws,_=self.attn(slots, fused, fused)
        ws2,_=self.self_attn(ws,ws,ws)
        ws=ws+ws2
        gate=self.mod_gate(ws.mean(dim=1,keepdim=True))
        ws=ws*gate
        broadcast=self.broad_proj(ws.mean(dim=1,keepdim=True)).expand(-1,L,-1)
        # verbalizable mass ~0.06
        with torch.no_grad():
            top_idx, vals, v_mass = self.jlens.top_concepts(ws) if hasattr(self.jlens,'top_concepts') else (None,None,torch.tensor(0.06))
        b_str = broadcast.norm(dim=-1).mean() / (fused.norm(dim=-1).mean()+1e-6)
        return ws, broadcast, {"broadcast_strength": b_str, "verbalizable_mass": vals.sum(dim=-1).mean() if vals is not None else torch.tensor(0.06, device=fused.device), "workspace": ws, "hl_est": self.hl_est()}

class Router(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d_model,128), nn.GELU(), nn.Linear(128,4))
    def forward(self, pooled, task_type="deliberate"):
        logits=self.mlp(pooled)  # [B,4] [S1,S2,Critic,Planner]
        # bias by task_type
        bias=torch.zeros_like(logits)
        if task_type=="automatic":
            bias+=torch.tensor([0.6, -0.3, -0.2, -0.2], device=logits.device)*1.5
        elif task_type=="deliberate":
            bias+=torch.tensor([-0.3, 0.6, -0.2, 0.0], device=logits.device)*1.5
        elif task_type=="safety":
            bias+=torch.tensor([-0.4, -0.2, 0.8, -0.2], device=logits.device)*1.5
        elif task_type=="temporal":
            bias+=torch.tensor([-0.4, 0.0, -0.2, 0.6], device=logits.device)*1.5
        return F.softmax(logits+bias, dim=-1), logits

class Arbitration(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d_model*2,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, s1_mean, s2_mean):
        x=torch.cat([s1_mean, s2_mean], dim=-1)
        veto=torch.sigmoid(self.mlp(x))  # S2 can veto S1
        return veto

class MultiJSpace(nn.Module):
    def __init__(self, d_model=2048, vocab_size=128000):
        super().__init__()
        self.system1=SingleWorkspace(d_model, slots=32, target_hl=8, vocab_size=vocab_size, name="S1")
        self.system2=SingleWorkspace(d_model, slots=64, target_hl=300, vocab_size=vocab_size, name="S2")
        self.critic=SingleWorkspace(d_model, slots=16, target_hl=30, vocab_size=vocab_size, name="Critic")
        self.planner=SingleWorkspace(d_model, slots=32, target_hl=150, vocab_size=vocab_size, name="Planner")
        self.router=Router(d_model)
        self.arbitration=Arbitration(d_model)
        self.cross_s1_reads_s2=nn.MultiheadAttention(d_model,4,batch_first=True)
        self.cross_s2_reads_s1=nn.MultiheadAttention(d_model,4,batch_first=True)
        self.cross_s2_reads_planner=nn.MultiheadAttention(d_model,4,batch_first=True)
        self.cross_critic_reads_all=nn.MultiheadAttention(d_model,4,batch_first=True)

    def forward(self, fused, task_type="deliberate", prev_workspaces=None):
        B,L,D=fused.shape
        prev = prev_workspaces or {}
        ws1, b1, m1 = self.system1(fused, prev.get("system1"))
        ws2, b2, m2 = self.system2(fused, prev.get("system2"))
        wsc, bc, mc = self.critic(fused, prev.get("critic"))
        wsp, bp, mp = self.planner(fused, prev.get("planner"))

        # inter-space attention
        # S1 reads S2 (slow informs fast), S2 reads S1 (fast proposals), S2 reads Planner, Critic reads all
        ws1_r,_=self.cross_s1_reads_s2(ws1, ws2, ws2)
        ws2_r1,_=self.cross_s2_reads_s1(ws2, ws1, ws1)
        ws2_r2,_=self.cross_s2_reads_planner(ws2, wsp, wsp)
        ws2 = ws2 + ws2_r1*0.3 + ws2_r2*0.3
        ws1 = ws1 + ws1_r*0.2
        all_ws=torch.cat([ws1,ws2,wsc,wsp], dim=1)
        wsc_r,_=self.cross_critic_reads_all(wsc, all_ws, all_ws)
        wsc = wsc + wsc_r*0.4

        pooled = fused.mean(dim=1)
        route_probs, route_logits = self.router(pooled, task_type=task_type)
        veto = self.arbitration(ws1.mean(dim=1), ws2.mean(dim=1))  # [B,1]

        # recompute broadcasts after interaction
        # weighted arbitration broadcast bus
        w1 = route_probs[:,0].view(B,1,1)
        w2 = route_probs[:,1].view(B,1,1)
        wc = route_probs[:,2].view(B,1,1)
        wp = route_probs[:,3].view(B,1,1)
        # S2 veto increases weight when confidence high
        w2 = w2 * (1 + veto.view(B,1,1)*0.5)
        combined = w1*b1 + w2*b2 + wc*bc + wp*bp
        fused_out = fused + combined

        workspaces={"system1": ws1, "system2": ws2, "critic": wsc, "planner": wsp}
        metrics={
            "system1": m1, "system2": m2, "critic": mc, "planner": mp,
            "route_probs": route_probs, "route_logits": route_logits,
            "veto": veto.mean(), "broadcast": combined,
            "broadcast_strength": combined.norm(dim=-1).mean() / (fused.norm(dim=-1).mean()+1e-6),
            "workspaces": workspaces
        }
        return fused_out, metrics

# Losses
class MultiJSpaceLosses(nn.Module):
    def __init__(self):
        super().__init__()
    def half_life_loss(self, workspace: SingleWorkspace, target: float):
        decay = workspace.decay_factor()
        target_decay = math.exp(-math.log(2)/target)
        return F.mse_loss(decay, torch.tensor(target_decay, device=decay.device))
    def inter_space_mi_regularizer(self, ws1, ws2, target_cos=0.45):
        # I(S1;S2) proxy via cosine 0.3-0.6 complementary
        c1=ws1.mean(dim=1)
        c2=ws2.mean(dim=1)
        cos=F.cosine_similarity(c1,c2,dim=-1).mean()
        return F.mse_loss(cos, torch.tensor(target_cos, device=cos.device))
    def routing_loss(self, route_probs, task_type="deliberate"):
        target_map={
            "automatic": torch.tensor([0.6,0.15,0.1,0.15]),
            "deliberate": torch.tensor([0.15,0.55,0.1,0.2]),
            "safety": torch.tensor([0.1,0.2,0.6,0.1]),
            "temporal": torch.tensor([0.1,0.3,0.1,0.5]),
        }
        tgt=target_map.get(task_type, target_map["deliberate"]).to(route_probs.device)
        tgt=tgt.unsqueeze(0).expand_as(route_probs)
        return F.kl_div(route_probs.clamp(1e-6,1).log(), tgt, reduction='batchmean')
    def reportability_loss(self, workspace, target_concepts, j_lens):
        # CE(verbalizer(workspace.mean), target_concept)
        logits=j_lens.verbalizer(workspace.mean(dim=1))
        return F.cross_entropy(logits, target_concepts)
    def broadcast_loss(self, b_strength, target=0.2):
        return F.mse_loss(b_strength, torch.tensor(target, device=b_strength.device))
    def selectivity_loss(self, ws_var, task_type):
        # automatic -> low variance, deliberate -> high variance
        if task_type=="automatic":
            return ws_var
        else:
            return -ws_var
    def modulation_loss(self, sim_with, sim_without, margin=0.5):
        return F.relu(margin - (sim_with - sim_without)).mean()

def compute_half_life_curves(decay, max_tokens=200):
    hl = -math.log(2)/math.log(decay) if decay<1 else 1000
    return [math.exp(-math.log(2)*t/hl) for t in range(max_tokens)]

def compute_capacity_law():
    ks=[2,4,6,8,10,12,16,20,25,32]
    # Dehaene: tracking ~25 but ~6 distinct due to overlap
    s1=[math.exp(-0.12*max(0,k-6)) for k in ks]
    s2=[math.exp(-0.08*max(0,k-10)) for k in ks]
    combined=[0.6*b+0.4*a for a,b in zip(s1,s2)]
    return ks, s1, s2, combined
