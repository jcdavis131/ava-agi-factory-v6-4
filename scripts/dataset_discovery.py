#!/usr/bin/env python3
"""
dataset_discovery.py — Discover additional datasets based on eval weaknesses
Solo personal project, no connection to employer, built with public/free-tier only
HOME persona only

Purpose:
- Reads branch_eval_results.json, frontier_eval_results.json (if exists), your_files/ava-agi/runs/latest-log.html to identify weak domains
- Maps weak domains to data needs
- Searches HuggingFace Hub free public API for candidate datasets with permissive licenses (MIT, Apache2, CC0, CC-BY)
- Logs candidates to your_files/ava-agi/dataset_discovery/candidates_{date}.json
- Does NOT auto-download massive data in Hatch VM (limited disk), but prepares download manifests for Alienware
- Writes data/discovery/needs.json: what domains need more tokens

Usage:
  python scripts/dataset_discovery.py --dry-run
  python scripts/dataset_discovery.py --eval-file branch_eval_results.json --out your_files/ava-agi/dataset_discovery/
"""
import argparse, json, os, re, pathlib, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

DISCLAIMER = "Solo personal project, no connection to employer, built with public/free-tier only"

# Mapping weak domains -> HF dataset search queries and data needs
DOMAIN_TO_DATASET_QUERIES = {
    "finance": {
        "keywords": ["financial", "finance", "stock", "earnings", "accounting"],
        "hf_datasets": ["financial_phrasebank", "convfinqa", "finqa", "fiqa", "flare-finqa", "bizbench", "sec_qa", "finance-alpaca"],
        "need_tokens": "Need more financial reasoning, SEC filings, earnings reports, accounting textbook style",
        "license_pref": ["apache-2.0","mit","cc0"],
    },
    "bio": {
        "keywords": ["biomedical", "pubmed", "bio", "medical", "genomics"],
        "hf_datasets": ["pubmed_qa", "medmcqa", "medqa", "chemprot", "biorxiv", "bigbio", "scientific_papers"],
        "need_tokens": "Biomed Q&A, PubMed abstracts, medical reasoning chains",
        "license_pref": ["cc0","mit","apache-2.0","cc-by-4.0"],
    },
    "code": {
        "keywords": ["code", "programming", "python", "github", "stack"],
        "hf_datasets": ["the_stack", "code_search_net", "codeparrot/code-complexity", "openai_humaneval", "mbpp", "code_alpaca", "evol-codealpaca"],
        "need_tokens": "More code reasoning traces, bug fix pairs, chain-of-thought debugging",
        "license_pref": ["mit","apache-2.0"],
    },
    "math": {
        "keywords": ["math", "mathematics", "theorem", "proof", "algebra"],
        "hf_datasets": ["lmsys/math", "metamath-qa", "gsm8k", "math", "lean_workbook", "proof_pile", "open-web-math"],
        "need_tokens": "Proofs, step-by-step solutions, Lean theorems",
        "license_pref": ["mit","apache-2.0","cc0"],
    },
    "safety": {
        "keywords": ["safety", "alignment", "toxicity", "harmful"],
        "hf_datasets": ["anthropic/hh-rlhf", "toxic_chat", "safety_bench", "xstest", "beaver_tails"],
        "need_tokens": "Safety early-warning examples, blackmail refusal, Critic hl=30-35 training",
        "license_pref": ["mit","apache-2.0","cc-by-4.0"],
    },
    "long_context": {
        "keywords": ["long", "context", "book", "document"],
        "hf_datasets": ["bookcorpus", "pg19", "longbench", "scrolls", "loogle"],
        "need_tokens": "Long-context 32k-128k docs for YaRN 10k->1M RoPE extension",
        "license_pref": ["mit","apache-2.0","cc0","pg19 is apache?"],
    },
    "reasoning": {
        "keywords": ["reasoning", "logic", "cot", "chain-of-thought"],
        "hf_datasets": ["gsm8k", "commonsense_qa", "strategyqa", "logiqa", "reclor", "proof_qa", "entailment_bank"],
        "need_tokens": "Multi-hop reasoning, logical equivalence, S2 hl=300-400 deliberate tasks",
        "license_pref": ["mit","apache-2.0"],
    },
    # Frontier benchmark domains from specs/frontier_benchmark_spec.md
    "general": {
        "keywords": ["general", "knowledge"],
        "hf_datasets": ["c4", "pile", "fineweb", "dolma", "dclm-baseline"],
        "need_tokens": "General web-scale filtered (dclm 0.85 edu 4.5)",
        "license_pref": ["apache-2.0","mit","odc-by"],
    }
}

def parse_eval_results(eval_path: Path):
    """Parse branch_eval_results.json to find weak domains"""
    weak = []
    try:
        data = json.loads(eval_path.read_text())
        # branch_eval_results has structure {branch: {tests: [...], cap_pres, cap_score, align_auc}}
        # frontier might have per-domain scores
        if isinstance(data, dict):
            for branch, result in data.items():
                if isinstance(result, dict) and "tests" in result:
                    for t in result["tests"]:
                        if not t.get("pass", True):
                            # infer domain from test name
                            test_name = t.get("test","")
                            desc = t.get("desc","")
                            # map test to domain
                            if "safety" in test_name or "blackmail" in desc.lower():
                                weak.append(("safety", 0.0, f"{test_name} failed: {desc}"))
                            elif "finance" in test_name or "finance" in desc.lower():
                                weak.append(("finance", 0.2, desc))
                            elif "bio" in test_name or "spider" in test_name.lower(): # spider.ant etc? actually base tests are general
                                # treat base tests as reasoning
                                weak.append(("reasoning", 0.5, desc))
                            else:
                                weak.append(("reasoning", 0.5, desc))
                    # also check scores
                    cap = result.get("cap_score", 1.0)
                    if cap < 0.9:
                        weak.append(("general", cap, f"{branch} cap_score low {cap}"))
                    align = result.get("align_auc", 1.0)
                    if align < 0.85:
                        weak.append(("safety", align, f"{branch} align_auc low {align}"))
        print(f"[Discovery] Parsed {eval_path}, found {len(weak)} weakness signals")
    except Exception as e:
        print(f"[Discovery] Failed parse {eval_path}: {e}, using defaults")
        weak = [("reasoning", 0.6, "no eval data, default to reasoning"), ("math", 0.6, "default"), ("code", 0.6, "default")]

    # Also try frontier results if exists
    frontier_path = Path(eval_path).parent / "frontier_eval_results.json"
    if frontier_path.exists():
        try:
            f_data = json.loads(frontier_path.read_text())
            # frontier format unknown, try to extract
            print(f"[Discovery] Found frontier {frontier_path}")
            # placeholder: if frontier has low scores, add finance/bio
            # e.g., {"finance": 0.45, "bio": 0.52}
            if isinstance(f_data, dict):
                for domain, score in f_data.items():
                    if isinstance(score, (int,float)) and score < 0.6:
                        weak.append((domain, score, f"frontier {domain} low {score}"))
        except Exception as e:
            print(f"[Discovery] frontier parse failed {e}")

    # Deduplicate and sort by score ascending (weakest first)
    grouped = defaultdict(list)
    for dom, score, reason in weak:
        grouped[dom].append((score, reason))
    # aggregate
    agg = []
    for dom, lst in grouped.items():
        min_score = min(s for s,_ in lst)
        reasons = [r for _,r in lst]
        agg.append((dom, min_score, reasons))
    agg.sort(key=lambda x: x[1])
    return agg

def search_hf_datasets_free(domain, query_list, license_pref, dry_run=False):
    """Search HF Hub without API key using public API (https://huggingface.co/api/datasets) — free, no auth"""
    candidates = []
    # For Hatch VM we avoid heavy network if dry-run, but we can attempt limited fetch
    # Use cache of known permissive datasets from DOMAIN_TO_DATASET_QUERIES
    base_info = DOMAIN_TO_DATASET_QUERIES.get(domain, {})
    hf_list = base_info.get("hf_datasets", []) + query_list

    # If not dry-run, try to hit HF API for each dataset to get license/size
    # We'll do simple HTTP GET without external deps using urllib
    import urllib.request, urllib.parse

    for ds_name in hf_list[:15]:  # limit
        meta = {
            "name": ds_name,
            "source": "huggingface",
            "domain": domain,
            "url": f"https://huggingface.co/datasets/{ds_name}",
            "license": "unknown",
            "tokens_estimate": "unknown",
            "relevance_score": 0.8 if domain in ds_name or domain in DOMAIN_TO_DATASET_QUERIES.get(domain, {}).get("keywords", []) else 0.6,
            "download_method": f'python -m datasets load_dataset "{ds_name}" --streaming for inspection, then .save_to_disk',
            "wget": f'# pip install datasets; python -c "from datasets import load_dataset; ds=load_dataset(\'{ds_name}\', streaming=True); print(next(iter(ds)))"',
            "license_ok": False,
        }
        if not dry_run:
            try:
                # HF API: https://huggingface.co/api/datasets/<id>
                api_url = f"https://huggingface.co/api/datasets/{urllib.parse.quote(ds_name, safe='')}"
                req = urllib.request.Request(api_url, headers={"User-Agent": "Ava-Dataset-Discovery/6.4"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        j = json.loads(resp.read().decode())
                        tags = j.get("tags", [])
                        # license in cardData or tags
                        card = j.get("cardData", {})
                        lic = card.get("license") or next((t.split(":")[1] for t in tags if t.startswith("license:")), "unknown")
                        meta["license"] = lic if isinstance(lic, str) else str(lic)
                        meta["downloads"] = j.get("downloads",0)
                        meta["likes"] = j.get("likes",0)
                        # token estimate: if dataset has config, estimate rough
                        # Use license pref check
                        lic_lower = str(meta["license"]).lower()
                        meta["license_ok"] = any(lp in lic_lower for lp in license_pref) or any(lp in lic_lower for lp in ["mit","apache","cc0","cc-by"])
                        # relevance bump if approved license
                        if meta["license_ok"]:
                            meta["relevance_score"] += 0.15
                            meta["relevance_score"] = min(1.0, meta["relevance_score"])
            except Exception as e:
                # network likely blocked or timeout in VM — keep placeholder
                meta["api_error"] = str(e)[:200]
        else:
            # dry-run placeholder license ok heuristic
            meta["license"] = "assumed permissive" if domain in ["math","code","logic"] else "needs check"
            meta["license_ok"] = True if domain in ["math","code","reasoning"] else False
            meta["dry_run"] = True

        candidates.append(meta)
    return candidates

def main():
    ap = argparse.ArgumentParser(description="Dataset Discovery based on eval weaknesses")
    ap.add_argument("--eval-file", default="branch_eval_results.json", help="eval results json")
    ap.add_argument("--out", default="your_files/ava-agi/dataset_discovery", help="output dir for candidates")
    ap.add_argument("--dry-run", action="store_true", help="skip network calls, use cached lists")
    ap.add_argument("--domains", nargs="*", default=None, help="force domains to search, e.g., finance bio code")
    args = ap.parse_args()

    print(f"[{DISCLAIMER}]")
    repo_root = Path(__file__).parent.parent
    eval_path = repo_root / args.eval_file
    if not eval_path.exists():
        # try alternative locations
        alt = repo_root / "data" / args.eval_file
        if alt.exists():
            eval_path = alt
        else:
            print(f"[Discovery] Eval file {args.eval_file} not found at {eval_path}, using dummy weaknesses")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = repo_root / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    discovery_dir = repo_root / "data/discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse eval results to get weak domains
    if eval_path.exists():
        weak_domains = parse_eval_results(eval_path)
    else:
        weak_domains = [("finance",0.45,["frontier finance low"]), ("bio",0.52,["frontier bio low"]), ("code",0.6,["code weak"]), ("math",0.6,["math weak"]), ("reasoning",0.6,["reasoning weak"])]

    if args.domains:
        # override with forced domains
        weak_domains = [(d,0.5,[f"forced domain {d}"]) for d in args.domains]
        print(f"[Discovery] Forced domains {args.domains}")

    print(f"[Discovery] Weak domains identified (weakest first):")
    for dom, score, reasons in weak_domains:
        print(f"  - {dom}: score {score:.2f} reasons: {reasons[:2]}")

    # 2. For each weak domain, search candidates
    all_candidates = []
    needs = {}
    for dom, score, reasons in weak_domains:
        domain_info = DOMAIN_TO_DATASET_QUERIES.get(dom, DOMAIN_TO_DATASET_QUERIES.get("general"))
        queries = domain_info["keywords"] if domain_info else [dom]
        license_pref = domain_info["license_pref"] if domain_info else ["mit","apache-2.0"]
        print(f"[Discovery] Searching HF for domain {dom} keywords {queries[:3]} dry_run={args.dry_run}")
        candidates = search_hf_datasets_free(dom, queries, license_pref, dry_run=args.dry_run)
        all_candidates.extend(candidates)
        # record need
        needs[dom] = {
            "current_score": score,
            "reasons": reasons,
            "need_description": domain_info["need_tokens"] if domain_info else f"Need more {dom}",
            "tokens_needed_estimate": f"{'500M-2B' if dom in ['finance','bio'] else '100M-500M'} tokens to improve",
            "candidate_count": len(candidates),
            "top_candidates": [c["name"] for c in sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)[:3]]
        }

    # 3. Write candidates with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidates_path = out_root / f"candidates_{timestamp}.json"
    candidates_path.write_text(json.dumps({
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weak_domains": [{"domain": d, "score": s, "reasons": r} for d,s,r in weak_domains],
        "candidates": all_candidates,
        "total_candidates": len(all_candidates),
        "license_note": "Only MIT, Apache-2.0, CC0, CC-BY are safe for commercial training. Avoid CC-BY-NC, CC-BY-SA-NC.",
        "usage": "Review candidates_{date}.json, pick top 2 per domain with license_ok=True, then run download manifest on Alienware (not Hatch VM due disk)",
    }, indent=2))
    print(f"[Discovery] Wrote {candidates_path} with {len(all_candidates)} candidates")

    # 4. Write needs.json for downstream
    needs_path = discovery_dir / "needs.json"
    needs_path.write_text(json.dumps({
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_source": str(eval_path),
        "weak_domains": {d: {"score": s, "reasons": r} for d,s,r in weak_domains},
        "needs": needs,
        "next_action": "Run dataset_expansion.py --phases for weak domains, and prepare download scripts for top HF candidates",
        "download_manifest_template": {
            "finance": "python -m datasets download financial_phrasebank -- to data/raw/finance/ ; then run nemo curated filtering",
            "bio": "python -m datasets download pubmed_qa ; filter reward>0.8",
        }
    }, indent=2))
    print(f"[Discovery] Wrote {needs_path}")

    # 5. Write Alienware download script
    download_sh = out_root / f"download_candidates_{timestamp}.sh"
    lines = [
        f"#!/bin/bash",
        f"# Solo personal project, no connection to employer, built with public/free-tier only",
        f"# Auto-generated {timestamp} from dataset_discovery",
        f"# Review LICENSES before training",
        f"set -e",
        f"mkdir -p data/raw",
        f"",
    ]
    for cand in sorted(all_candidates, key=lambda x: x["relevance_score"], reverse=True)[:10]:
        if not cand.get("license_ok", False) and not args.dry_run:
            lines.append(f"# SKIP {cand['name']} license {cand['license']} not permissive — review manually")
            continue
        lines.append(f"echo \"Downloading {cand['name']} ({cand['url']}) license {cand['license']}\" ")
        lines.append(f"# {cand['download_method']}")
        lines.append(f"# python scripts/ingest_hf.py --dataset {cand['name']} --out data/raw/{cand['domain']}/{cand['name'].replace('/','_')} --filter reward>0.8")
        lines.append("")
    download_sh.write_text("\n".join(lines))
    download_sh.chmod(0o755)
    print(f"[Discovery] Wrote download script {download_sh}")

    # 6. Summary
    print(f"\n[Discovery] Summary:")
    for dom, info in needs.items():
        print(f"  {dom}: need {info['tokens_needed_estimate']} — top {info['top_candidates']}")

    print(f"\nNext steps:")
    print(f"  - Review {candidates_path}")
    print(f"  - On Alienware RTX 4090: bash {download_sh}")
    print(f"  - Then run: ./scripts/local_train.sh python scripts/dataset_expansion.py --tokens 100M --phases {' '.join([d for d,_ in [(dom,0) for dom in needs.keys()][:2]])} --out data/daily_expanded")

if __name__ == "__main__":
    main()
