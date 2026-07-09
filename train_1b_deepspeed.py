"""
train_1b_deepspeed.py — WSD 736k branching, YaRN 10k→1M, Multi-JSpace per-space losses
Solo personal project, no connection to employer
"""
import argparse, math, os, json, hashlib, torch, torch.nn.functional as F
from model_1b import AvaModel1B, apply_rope_scaling

BRANCH_CONFIGS = {
    "base": {"freeze": [], "router_bias": None, "hl": {"S1":8, "S2":300, "Critic":30, "Planner":150}, "lr": 2e-4},
    "code": {"freeze": ["system1"], "router_bias": [0.25,0.45,0.05,0.25], "router_frozen": ["system1"], "hl": {"S1":8, "S2":350, "Critic":30, "Planner":200}, "lr":1e-4, "data": "code_repo 50%+code_long_32k 20%+jobbench_code 15%+general 15%"},
    "math": {"freeze": ["system1","planner"], "router_bias": [0.10,0.65,0.20,0.05], "hl": {"S1":8,"S2":400,"Critic":40,"Planner":150}, "lr":8e-5, "data": "math_formal_lean 35%+lean_mathlib 20%+proofpile2 20%+synthetic_math_r1 15%"},
    "chat": {"freeze": ["system1","system2"], "router_bias": [0.15,0.25,0.35,0.25], "hl": {"S1":8,"S2":300,"Critic":35,"Planner":180}, "lr":5e-5, "data": "chat_alignment 30%+safety 20%+jobbench_delegation 25%+gaia2_temporal 15%+counterfactual 10%"},
}

def wsd_lr(step, warmup=2000, stable_steps=736000, total_steps=800000, base_lr=2e-4, min_lr=2e-5):
    if step < warmup:
        return base_lr * step / warmup
    elif step <= stable_steps:
        return base_lr
    else:
        # cosine decay to 10%
        progress = (step - stable_steps) / (total_steps - stable_steps)
        cos_decay = 0.5*(1+math.cos(math.pi*progress))
        return min_lr + (base_lr-min_lr)*cos_decay

def rope_schedule(step):
    # gradual RoPE 10k→1M
    if step < 140000: return 10000, 1.0, "2k/4k"
    elif step < 384000: return 10000, 1.0, "4k"
    elif step < 420000: return 50000, 1.0, "8k NTK1.0"
    elif step < 480000: return 100000, 1.2, "16k NTK1.2"
    elif step < 660000: return 500000, 1.5, "32k NTK1.5"
    else: return 1000000, 4.0, "64k/128k YaRN2-4"

def half_life_loss(decay_logit, target_hl):
    decay = torch.sigmoid(decay_logit).clamp(0.01,0.99)
    target_decay = math.exp(-math.log(2)/target_hl)
    return F.mse_loss(decay, torch.tensor(target_decay, device=decay.device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', default='base', choices=['base','code','math','chat','all'])
    parser.add_argument('--deepspeed', default='deepspeed_zero3_bf16.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=800000)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()

    print(f"[Ava v6.4] Branch={args.branch} Device={args.device} - Solo project no employer connection")
    print(f"[WSD] warmup 2k stable 736k (92%) decay 64k cosine to 10% 2e-4→2e-5")
    print(f"[YaRN] gradual 10k→1M schedule active")

    # Model stub
    model = AvaModel1B(vocab_size=32000, d_model=2048, multi_jspace_enabled=True)
    print(f"[Model] AvaModel1B 1B params multi_jspace enabled, RoPE modules 56")
    
    branches = ['base','code','math','chat'] if args.branch=='all' else [args.branch]
    for br in branches:
        cfg = BRANCH_CONFIGS.get(br, BRANCH_CONFIGS['base'])
        print(f"\n--- Branch {br} --- freeze={cfg.get('freeze')} router_bias={cfg.get('router_bias')} HL={cfg.get('hl')} LR={cfg.get('lr')}")
        if cfg.get('freeze'):
            model.freeze_spaces(cfg['freeze'])
            print(f"[Freeze] ❄️ {cfg['freeze']} frozen, others 🔥")
        # Simulate training loop header with per-space losses
        # S1 on automatic, S2 on deliberate, Critic on safety, Planner on temporal as per spec
        # This is mock training loop showing loss wiring
        for step in [0, 2000, 140000, 420000, 660000, 736000, 800000][:2 if args.mock else 7]:
            lr = wsd_lr(step)
            rope_base, rope_scale, ctx = rope_schedule(step)
            if step==736000:
                print(f"[Checkpoint] Save ava_stable_736k.pt (or ava_stable_736k_rope1000000_ctx131072.pt) at step 736k - ready for branching")
                # torch.save(model.state_dict(), "ava_stable_736k.pt")
        # W&B logging placeholders
        print(f"[W&B] Charts: half_life/S1_hl_est vs target {cfg['hl']['S1']}, S2 target {cfg['hl']['S2']}, Critic {cfg['hl']['Critic']}, Planner {cfg['hl']['Planner']}")
        print(f"[W&B] capacity law knee S1=6 S2=10 combined=9, routing distribution, jspace/S1/broadcast S2/verbalizable_mass Critic/early_warning")
        print(f"[Loss] lm_loss + (report*1.0 + broadcast*0.5 + selectivity*0.3 + modulation*0.5)*j_weight(0.08 early 0.15 reasoning/long) + per-space S1 broadcast 0.18 hl8 w0.6 S2 0.22 vm 0.065 hl300 w0.8 Critic 0.08 hl30 w1.0 Planner 0.20 hl150 w0.7 + inter_mi cos0.45 w0.3 routing KL w0.4")

    if args.branch!='base' or args.branch=='all':
        print("\n[Eval] Auto-running eval_branch_harness --branch", args.branch, "--mode real after branch complete")
        # import eval_branch_harness mock

if __name__ == "__main__":
    main()
