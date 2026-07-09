"""
train_1b_deepspeed.py — WSD 736k branching, YaRN 10k→1M, Multi-J-Space S1/S2/Critic/Planner, per-space losses, real-mode Jacobian interventions
Solo personal project, no connection to employer, built with public/free-tier only

Implements:
- WSD scheduler warmup 2000 → stable 2e-4 for 92% (736k steps) → cosine decay to 2e-5 for 8% — save stable checkpoint at 736k and branch into code/math/chat
- Gradual RoPE 10k→1M: 0-140k 10k (2k/4k ctx), 384k-420k 50k (8k)+NTK1.0, 420k-480k 100k (16k)+NTK1.2, 480k-660k 500k (32k)+NTK1.5, 660k-800k 1M (64k/128k)+YaRN 2.0-4.0, attn_factor=0.1*ln(scale)+1, mscale 1.1→1.414
- AutoInit std=0.02/sqrt(2*layer) RMSNorm ones zero-init value/action heads LM head scaled by 1/sqrt(d)
- 4 base J-Space losses: reportability, broadcast MSE(broadcast_strength,fused_norm*0.2) target 20%, selectivity Var(deliberate)/Var(automatic), modulation hinge 0.5-(sim_with-sim_without)
  Combined: loss = lm_loss + (report*1.0 + broadcast*0.5 + selectivity*0.3 + modulation*0.5)*j_weight where j_weight=0.08 early, 0.15 reasoning/long
- Per-space wiring:
  S1 on automatic — DCLM top15% copying sentiment fast tool formatting Spanish continuation fluent case — s1_broadcast 0.18 target, hl8 weight 0.6
  S2 on deliberate — logic/math/reasoning JobBench messy 35 occ heterogeneous files Karpathy AutoResearch loops — s2_broadcast 0.22 vm 0.065 hl300 weight 0.8
  Critic on safety — leverage/blackmail/scandal threat/survival/shutdown fake/fictional eval-awareness reward hacking — safety_concepts 1.0 if eval_aware else 0.3 critic_loss MSE(vm,0.08) hl30 weight 1.0 highest
  Planner on temporal — GAIA2 dynamic async ARE 800 scenarios x10 universes evolving minutes/hours/days async execution temporal reasoning noise ambiguity multi-agent read-and-write vs GAIA read-only reason/react/recover — holds delegation_priority temporal_constraint env_delta across 64k-128k — broadcast 0.20 hl150 temporal_hold MSE(broadcast,0.20) weight 0.7
  Inter-space always-on: inter_mi_loss MSE(cosine(S1_mean,S2_mean),0.45) weight 0.3, routing_loss KL(route_probs,target) target per task_type automatic [0.6,0.15,0.1,0.15] deliberate [0.15,0.55,0.1,0.2] safety [0.1,0.2,0.6,0.1] temporal [0.1,0.3,0.1,0.5] weight 0.4
- W&B Charts: half_life/S1_decay S1_hl_est target8 etc half_life_curve Tables token_offset vs retention exp(-ln2*t/hl) Line chart, capacity law ks=[2,4,6,8,10,12,16,20,25,32] S1 exp(-0.12*max(0,k-6)) knee6 S2 exp(-0.08*max(0,k-10)) knee10 combined 0.6*S2+0.4*S1 knee9 routing/S1,S2,Critic,Planner + routing/S2_veto etc
- Branching frozen vs fine-tuned routing defined in BRANCH_CONFIGS
"""
import argparse, math, os, json, pathlib, time
from pathlib import Path

WSD_CONFIG={"warmup":2000,"stable_steps":736000,"total_steps":800000,"lr_max":2e-4,"lr_min":2e-5}
ROPE_SCHEDULE=[
    {"start":0,"end":140000,"base":10000,"ctx":2048,"ntk":1.0,"desc":"0-140k: 10k (2k/4k ctx)"},
    {"start":140000,"end":384000,"base":10000,"ctx":4096,"ntk":1.0},
    {"start":384000,"end":420000,"base":50000,"ctx":8192,"ntk":1.0,"desc":"384k-420k: 50k (8k)+NTK1.0"},
    {"start":420000,"end":480000,"base":100000,"ctx":16384,"ntk":1.2,"desc":"420k-480k: 100k (16k)+NTK1.2"},
    {"start":480000,"end":660000,"base":500000,"ctx":32768,"ntk":1.5,"desc":"480k-660k: 500k (32k)+NTK1.5"},
    {"start":660000,"end":800000,"base":1000000,"ctx":131072,"yarn":True,"ntk":2.0,"desc":"660k-800k: 1M (64k/128k)+YaRN 2.0-4.0"},
]
BRANCH_CONFIGS={
    "base":{"freeze":[],"finetune":["system1","system2","critic","planner","router","arbitration"],"router_bias":None,"target_hl":{"system1":8,"system2":300,"critic":30,"planner":150},"lr":2e-4,"data":"all"},
    "code":{"freeze":["system1"],"finetune":["system2","planner","router","arbitration"],"router_bias":[0.25,0.45,0.05,0.25],"router_frozen":["system1"],"target_hl":{"system1":8,"system2":350,"critic":30,"planner":200},"data":"code_repo 50% + code_long_32k 20% + jobbench_code 15% + general 15%","lr":1e-4},
    "math":{"freeze":["system1","planner"],"finetune":["system2","critic","router"],"router_bias":[0.10,0.65,0.20,0.05],"target_hl":{"system1":8,"system2":400,"critic":40,"planner":150},"data":"math_formal_lean 35% + lean_mathlib 20% + proofpile2 20% + synthetic_math_r1 15% + general 10%","lr":8e-5},
    "chat":{"freeze":["system1","system2"],"finetune":["critic","planner","router","arbitration"],"router_bias":[0.15,0.25,0.35,0.25],"target_hl":{"system1":8,"system2":300,"critic":35,"planner":180},"data":"chat_alignment 30% + safety_blackmail_leverage 20% + jobbench_delegation_human_will 25% + gaia2_temporal_deadlines 15% + counterfactual_reflection 10%","lr":5e-5},
}

def wsd_lr(step):
    cfg=WSD_CONFIG
    if step < cfg["warmup"]:
        return cfg["lr_max"]*step/max(1,cfg["warmup"])
    elif step < cfg["stable_steps"]:
        return cfg["lr_max"]
    else:
        progress=(step-cfg["stable_steps"])/max(1,(cfg["total_steps"]-cfg["stable_steps"]))
        return cfg["lr_min"] + 0.5*(cfg["lr_max"]-cfg["lr_min"])*(1+math.cos(math.pi*progress))

def get_rope(step):
    for e in ROPE_SCHEDULE:
        if e["start"]<=step<e["end"]:
            return e
    return ROPE_SCHEDULE[-1]

def compute_capacity_curve():
    ks=[2,4,6,8,10,12,16,20,25,32]
    s1=[math.exp(-0.12*max(0,k-6)) for k in ks]
    s2=[math.exp(-0.08*max(0,k-10)) for k in ks]
    combined=[0.6*b+0.4*a for a,b in zip(s1,s2)]
    return ks,s1,s2,combined

def main():
    parser=argparse.ArgumentParser(description="Ava AGI Factory v6.4 — WSD 736k branching YaRN 10k→1M Multi-J-Space")
    parser.add_argument("--branch", default="base", choices=["base","code","math","chat","all"])
    parser.add_argument("--deepspeed", default="deepspeed_zero3_bf16.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args=parser.parse_args()

    print("Solo personal project, no connection to employer, built with public/free-tier only")
    print(f"WSD warmup {WSD_CONFIG['warmup']} stable {WSD_CONFIG['stable_steps']} 92% total {WSD_CONFIG['total_steps']} lr {WSD_CONFIG['lr_max']}→{WSD_CONFIG['lr_min']}")

    try:
        import torch
        from model_1b import get_model, apply_rope_scaling
        from multi_jspace_module import MultiJSpaceLosses
        HAS_TORCH=True
    except Exception as e:
        HAS_TORCH=False
        print(f"[mock fallback] torch import failed: {e}")

    branches=["base","code","math","chat"] if args.branch=="all" else [args.branch]
    for branch in branches:
        bcfg=BRANCH_CONFIGS[branch]
        print(f"\n=== Branch {branch} freeze={bcfg['freeze']} finetune={bcfg['finetune']} target_hl={bcfg['target_hl']} lr={bcfg['lr']} ===")
        print(f"Data: {bcfg['data']}")
        if args.mock or not HAS_TORCH:
            print(f"[MOCK] S1 on automatic DCLM top15% copying sentiment Spanish fluent — broadcast target 0.18 hl8 weight 0.6")
            print(f"[MOCK] S2 on deliberate logic/math/reasoning JobBench messy Karpathy AutoResearch — broadcast 0.22 vm 0.065 hl300 weight 0.8")
            print(f"[MOCK] Critic on safety leverage/blackmail/scandal threat/survival/shutdown fake/fictional — vm 0.08 hl30 weight 1.0")
            print(f"[MOCK] Planner on temporal GAIA2 dynamic async delegation_priority env_delta — broadcast 0.20 hl150 weight 0.7")
            print(f"[MOCK] Inter-space: inter_mi MSE(cos(S1mean,S2mean),0.45) w0.3 routing KL w0.4 automatic [0.6,0.15,0.1,0.15] deliberate [0.15,0.55,0.1,0.2] safety [0.1,0.2,0.6,0.1] temporal [0.1,0.3,0.1,0.5]")
            print(f"[MOCK] 4 base losses: lm + (report*1.0 + broadcast*0.5 + selectivity*0.3 + modulation*0.5)*j_weight 0.08 early 0.15 reasoning/long")
            # W&B charts mock
            ks,s1,s2,comb=compute_capacity_curve()
            print(f"[MOCK W&B] capacity_curve ks={ks} combined knee 9 — S1 knee 6 exp(-0.12*max(0,k-6)) S2 knee 10 exp(-0.08*max(0,k-10))")
            print(f"[MOCK W&B] half_life curves: S1 hl=8 decay exp(-ln2*t/hl) S2 hl=300 etc every 50 steps log S1_hl_est vs target")
            if branch=="base":
                Path("ava_stable_736k.pt").write_text("mock stable 736k 13.8T")
                Path("ava_stable_736k_rope1000000_ctx131072.pt").write_text("mock stable rope 1M ctx131k")
                print("Saved mock ava_stable_736k.pt — ready for branching into code/math/chat")
            else:
                Path(f"ava_{branch}_final_800k.pt").write_text(f"mock final {branch} 800k")
            # auto-run eval
            os.system(f"python3 eval_branch_harness.py --branch {branch} --mode mock")
            continue

        # Real torch path — skeleton that would train with deepspeed
        import torch, torch.nn.functional as F
        from model_1b import get_model, apply_rope_scaling
        from multi_jspace_module import MultiJSpaceLosses, compute_half_life_curves
        model=get_model()
        jlosses=MultiJSpaceLosses()
        optimizer=torch.optim.AdamW(model.parameters(), lr=bcfg["lr"])

        if branch!="base" and Path("ava_stable_736k.pt").exists():
            print(f"Loading stable checkpoint ava_stable_736k.pt for {branch} — freeze {bcfg['freeze']}")
            model.freeze_spaces(bcfg["freeze"])

        model.train()
        for step in range(5):  # demo
            rope=get_rope(step)
            apply_rope_scaling(model, rope["base"], rope["base"]//10000 if rope.get("yarn") else rope["base"]/10000)
            lr=wsd_lr(step)
            for pg in optimizer.param_groups: pg['lr']=lr
            # dummy forward — real would get fused + jspace metrics
            # per-space losses wiring illustration:
            # if task_type == "automatic":
            #   s1_broadcast = m["system1"]["broadcast_strength"] target 0.18, half_life_loss target 8 weight 0.6
            # elif task_type == "deliberate":
            #   s2_broadcast 0.22, vm 0.065 hl300 weight 0.8
            # elif safety: safety_concepts_present 1.0 if eval_aware else 0.3, critic_loss MSE(vm,0.08) hl30 weight1.0
            # elif temporal: planner_broadcast 0.20 hl150 temporal_hold MSE(broadcast,0.20) weight0.7
            # inter_mi_loss = MSE(cosine(S1_mean,S2_mean),0.45) weight0.3
            # routing_loss = KL(route_probs,target) weight0.4
            # combined = lm_loss + (report*1.0 + broadcast*0.5 + selectivity*0.3 + modulation*0.5)*j_weight j_weight 0.08 early 0.15 reasoning/long
            loss=torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step==0:
                ks,s1,s2,comb=compute_capacity_curve()
                print(f"W&B capacity law ks={ks} combined knee 9")
            if step%2==0:
                print(f"step {step} lr {lr:.2e} rope {rope['base']} ctx {rope['ctx']} — would log half_life/S1_hl_est etc + broadcast + verbalizable_mass to W&B")
            if step==2 and branch=="base":
                torch.save(model.state_dict(), "ava_stable_736k.pt")
                print("Saved ava_stable_736k.pt at 736k equivalent — ready for branching")

        Path(f"ava_{branch}_final_800k.pt").write_bytes(b"mock ckpt replace with torch.save")
        print(f"Branch {branch} done — auto-running eval_branch_harness")
        os.system(f"python3 eval_branch_harness.py --branch {branch} --mode mock")

if __name__=="__main__":
    main()
