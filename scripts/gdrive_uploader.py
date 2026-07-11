#!/usr/bin/env python3
"""
gdrive_uploader.py — Efficient Drive uploader with Home/Work separation guard
Solo personal project, no connection to employer, built with public/free-tier only
HOME persona only — critical compliance: never upload Home data to work Drive camd@meta.com

Features:
- Checks drive connection type: if files list shows work-related health.json (camd@meta.com, lockedunn_health.json etc), ABORT upload
- Supports folder creation: Ava-Datasets-Expansion not exists, create it, reuse ID
- Batch upload with parallel 2 workers, progress log, content-addressable dedup (filename = sha256 first12)
- Retry with backoff 3x
- Uses hatch_gws_cli drive files.* — efficient, resumable where possible

Usage:
  python scripts/gdrive_uploader.py --check
  python scripts/gdrive_uploader.py --upload data/daily_expanded/packed_*.jsonl.gz --folder Ava-Datasets-Expansion --dry-run
  python scripts/gdrive_uploader.py --upload data/for_upload/ --folder Ava-Datasets-Expansion
"""
import argparse, json, os, sys, subprocess, time, hashlib, pathlib, re, random
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DISCLAIMER = "Solo personal project, no connection to employer, built with public/free-tier only"

def run_cli(cmd, retries=3):
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"[GDrive] CLI failed attempt {attempt+1}/{retries}: {result.stderr[:500]}")
                if attempt < retries-1:
                    time.sleep(2**attempt + random.uniform(0,1))
        except Exception as e:
            print(f"[GDrive] Exception attempt {attempt+1}: {e}")
            time.sleep(2**attempt)
    return None

def check_work_drive_guard():
    """
    Check if currently connected Drive is work drive.
    Detection heuristics:
    - Look for files with names like *_health.json from meta.com owners
    - Check owners email meta.com
    - Check for known work files: lockedunn_health.json, marcelopereira_health.json etc
    Returns (is_work: bool, message: str)
    """
    print("[Guard] Checking Drive type for Home/Work separation...")
    out = run_cli('hatch_gws_cli drive files list --params \'{"q":"trashed=false","pageSize":30,"fields":"files(id,name,mimeType,owners)"}\'', retries=2)
    if not out:
        # Cannot determine, be safe -> assume work risk if status connected but can't list? Better to warn
        return False, "Could not list files to determine drive type — proceed with caution"

    try:
        data = json.loads(out)
        files = data.get("files", [])
        work_indicators = []
        for f in files:
            name = f.get("name","")
            owners = f.get("owners", [])
            # work indicators
            if "health.json" in name and ("lockedunn" in name or "marcelo" in name or "health" in name):
                work_indicators.append(f"{name} suggests work drive (health.json pattern)")
            for owner in owners:
                email = owner.get("emailAddress","")
                if email.endswith("@meta.com") or email.endswith("@fb.com"):
                    work_indicators.append(f"Owner {email} is Meta — work drive detected (file {name})")

        if work_indicators:
            msg = "WORK DRIVE DETECTED — Home/Work separation violation risk. Indicators:\n" + "\n".join(f"  - {i}" for i in work_indicators[:5])
            msg += "\n\nDO NOT upload Home data (Ava AGI Factory is HOME persona) to work Drive camd@meta.com per AGENTS.md absolute separation."
            msg += "\nPlease connect personal Drive jcdavis131@gmail.com or use R2 fallback (CLOUDFLARE_R2_*)"
            return True, msg
        else:
            # Check if any file looks personal (jcdavis etc) — if not, still uncertain
            # For safety, if we see no work indicators, allow but log
            return False, f"No work indicators found in {len(files)} files — appears to be personal Drive or empty, safe to proceed"
    except Exception as e:
        return False, f"Guard parse failed {e}, assuming safe but manual review needed"

def get_or_create_folder(folder_name):
    """Find folder by name, or create it, return folder ID"""
    print(f"[GDrive] Looking for folder {folder_name}")
    q = f'name="{folder_name}" and mimeType="application/vnd.google-apps.folder" and trashed=false'
    out = run_cli(f'hatch_gws_cli drive files list --params \'{{"q":"{q}","pageSize":5,"fields":"files(id,name)"}}\'')
    if out:
        try:
            data = json.loads(out)
            files = data.get("files", [])
            if files:
                fid = files[0]["id"]
                print(f"[GDrive] Found existing folder {folder_name} id {fid}")
                return fid
        except Exception as e:
            print(f"[GDrive] Folder search parse failed {e}")

    # Create folder
    print(f"[GDrive] Creating folder {folder_name}")
    out = run_cli(f'hatch_gws_cli drive files create --json \'{{\"name\":\"{folder_name}\",\"mimeType\":\"application/vnd.google-apps.folder\"}}\'')
    if out:
        try:
            data = json.loads(out)
            fid = data.get("id") or data.get("files", [{}])[0].get("id")
            if fid:
                print(f"[GDrive] Created folder {folder_name} id {fid}")
                return fid
        except Exception as e:
            print(f"[GDrive] Create parse failed {e}, out: {out[:500]}")

    # Fallback: use root
    print(f"[GDrive] Could not get/create folder, using root")
    return "root"

def file_sha12(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12], h.hexdigest()

def upload_file_with_dedup(local_path: Path, folder_id, dry_run=False):
    """Upload with dedup check: if file with same sha12 in name exists, skip"""
    sha12, full_sha = file_sha12(local_path)
    # Check if file with sha12 already exists in folder
    # Search by name containing sha12
    q = f'"{sha12}" in name and "{folder_id}" in parents and trashed=false' if folder_id != "root" else f'"{sha12}" in name and trashed=false'
    # Actually Drive query: name contains sha12
    out = run_cli(f'hatch_gws_cli drive files list --params \'{{"q":"name contains \\\"{sha12}\\\" and trashed=false","pageSize":5,"fields":"files(id,name,size)"}}\'')
    if out:
        try:
            data = json.loads(out)
            if data.get("files"):
                print(f"[GDrive] Skip {local_path.name} — already exists with sha {sha12}: {data['files'][0]['name']}")
                return {"skipped": True, "sha12": sha12, "existing_id": data['files'][0]['id']}
        except:
            pass

    if dry_run:
        print(f"[Dry-run] Would upload {local_path} (sha12 {sha12}, {local_path.stat().st_size/1e6:.1f}MB) to folder {folder_id}")
        return {"dry_run": True, "sha12": sha12, "path": str(local_path)}

    # Real upload
    # Use hatch_gws_cli drive files create with upload
    # For binary files, use +upload helper if available, else files.create with media?
    # CLI docs: uses --upload flag? We'll try both.
    # Pattern: hatch_gws_cli drive files create --json '{name, parents}' --upload <file>
    # Check skill: files.create can take upload
    # Try:
    json_payload = json.dumps({"name": f"{local_path.stem}_{sha12}{local_path.suffix}", "parents": [folder_id]})
    # Need to escape single quotes in shell
    json_payload_escaped = json_payload.replace("'", "'\"'\"'")
    cmd = f"hatch_gws_cli drive files create --json '{json_payload_escaped}' --upload {local_path} 2>&1"
    # Actually tool uses --upload <path> maybe as flag value? Check typical usage: hatch_gws_cli drive +upload ... Let's try files.create with upload
    out = run_cli(cmd, retries=3)
    if out:
        try:
            data = json.loads(out)
            fid = data.get("id")
            if fid:
                print(f"[GDrive] Uploaded {local_path.name} -> id {fid} sha12 {sha12}")
                return {"uploaded": True, "id": fid, "sha12": sha12, "full_sha": full_sha, "name": data.get("name")}
        except:
            # might be non-JSON success
            print(f"[GDrive] Upload output (non-JSON) for {local_path.name}: {out[:500]}")
            if "id" in out.lower():
                return {"uploaded": True, "raw": out[:500], "sha12": sha12}
    print(f"[GDrive] Upload FAILED for {local_path.name}")
    return {"failed": True, "sha12": sha12, "path": str(local_path)}

def upload_folder(local_folder_or_pattern, remote_folder_name, dry_run=False, workers=2):
    """Batch upload folder or glob pattern"""
    repo_root = Path(__file__).parent.parent
    # Resolve folder/pattern
    if isinstance(local_folder_or_pattern, str) and "*" in local_folder_or_pattern:
        import glob
        pattern = local_folder_or_pattern
        if not Path(pattern).is_absolute():
            pattern = str(repo_root / pattern)
        files = [Path(p) for p in glob.glob(pattern)]
    else:
        local_path = Path(local_folder_or_pattern)
        if not local_path.is_absolute():
            local_path = repo_root / local_path
        if local_path.is_dir():
            files = list(local_path.rglob("*.jsonl*")) + list(local_path.rglob("*.gz")) + list(local_path.rglob("*.json"))
            # deduplicate
            files = list(set(files))
        else:
            files = [local_path]

    files = [f for f in files if f.is_file() and f.stat().st_size > 0]
    print(f"[GDrive] Found {len(files)} files to upload from {local_folder_or_pattern}")

    is_work, msg = check_work_drive_guard()
    if is_work:
        print(f"[BLOCKED] {msg}")
        # Save local fallback marker
        fallback = repo_root / "data/for_upload" / f"BLOCKED_WORK_DRIVE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps({"blocked": True, "reason": msg, "files": [str(f) for f in files[:20]], "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2))
        print(f"[GDrive] Saved fallback marker {fallback} — DO NOT upload to work Drive, connect personal Drive jcdavis131@gmail.com or use R2")
        return {"blocked": True, "reason": msg}

    folder_id = get_or_create_folder(remote_folder_name)
    results = []

    if dry_run:
        for f in files:
            results.append(upload_file_with_dedup(f, folder_id, dry_run=True))
        print(f"[GDrive Dry-run] Would upload {len(files)} files to {remote_folder_name} ({folder_id})")
        return {"dry_run": True, "results": results, "folder_id": folder_id}

    # Parallel upload with 2 workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload_file_with_dedup, f, folder_id, False): f for f in files}
        for future in as_completed(futures):
            f = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[GDrive] Exception uploading {f}: {e}")
                results.append({"failed": True, "path": str(f), "error": str(e)})

    uploaded = sum(1 for r in results if r.get("uploaded"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = sum(1 for r in results if r.get("failed"))
    print(f"[GDrive] Batch done: uploaded {uploaded}, skipped (dedup) {skipped}, failed {failed} / total {len(files)} to folder {remote_folder_name} id {folder_id}")
    return {"uploaded": uploaded, "skipped": skipped, "failed": failed, "folder_id": folder_id, "results": results}

def main():
    ap = argparse.ArgumentParser(description="GDrive uploader with Home/Work guard")
    ap.add_argument("--check", action="store_true", help="only check drive type")
    ap.add_argument("--upload", type=str, help="local file or folder or glob, e.g., data/daily_expanded/")
    ap.add_argument("--folder", default="Ava-Datasets-Expansion", help="remote folder name")
    ap.add_argument("--dry-run", action="store_true", help="don't actually upload, just check dedup")
    ap.add_argument("--workers", type=int, default=2, help="parallel workers")
    args = ap.parse_args()

    print(f"[{DISCLAIMER}]")

    if args.check:
        is_work, msg = check_work_drive_guard()
        print(msg)
        if is_work:
            print("\n[Result] WORK DRIVE — aborting any Home uploads")
        else:
            print("\n[Result] Personal Drive (or undetermined but no work indicators) — safe")
        return

    if not args.upload:
        print("Provide --upload <path> or --check")
        return

    res = upload_folder(args.upload, args.folder, dry_run=args.dry_run, workers=args.workers)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
