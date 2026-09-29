"""Freeze the held out test set: PCBWorld's D3 test lists mapped onto the Freerouting PCBench mirror.

Inputs
  data/splits/pcbworld_d3.json   PCBWorld configs/datasets/d3.json (commit recorded below)
  data/mirror/freerouting        partial clone of freerouting/freerouting with the PCBench fixtures

Outputs (committed to the repository)
  splits/test_d3.json            one entry per test board with tier, source repo, license, sha256 of raw.kicad_pcb
  splits/test_exclude_repos.txt  every source repository behind a test board; nothing from these may be trained on
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "data/splits/pcbworld_d3.json"
MIRROR = ROOT / "data/mirror/freerouting"
FIX = MIRROR / "scripts/benchmark/fixtures/PCBench"
OUT = ROOT / "splits"


def repo_key(url):
    """Normalize a source URL to owner/repo in lower case; anything else is kept as is."""
    m = re.match(r"https?://(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)/([^/]+)/([^/#?]+)", url or "")
    if not m:
        return (url or "").strip().lower() or "unknown"
    return f"{m.group(1)}/{m.group(2)}/{m.group(3).removesuffix('.git')}".lower()


def main():
    d3 = json.load(open(D3))
    catalog = {b["board_id"]: b for b in json.load(open(FIX / "catalog.json"))["boards"]}
    mirror_commit = subprocess.run(["git", "-C", str(MIRROR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    overrides = d3.get("_test_overrides", {})
    rows, problems = [], []
    for tier in ("easy", "medium", "hard"):
        names = [overrides.get(tier, {}).get(n, n) for n in d3[tier]["test"]]
        for name in names:
            board_id = re.sub(r"^\d{4}_", "", name)
            folder = FIX / board_id
            if board_id not in catalog or not (folder / "raw.kicad_pcb").exists():
                problems.append(f"{tier}: {name} has no mirror folder")
                continue
            cat = catalog[board_id]
            manifest = json.load(open(folder / "board-manifest.json"))
            raw = (folder / "raw.kicad_pcb").read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            # board-manifest.json was generated on Windows from CRLF files; the mirror stores LF.
            # The manifest hash must match after converting line endings, or the content changed.
            upstream = manifest["source_files"]["raw_kicad_pcb"]["sha256"]
            crlf = hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
            if upstream not in (digest, crlf):
                problems.append(f"{board_id}: raw.kicad_pcb content differs from board-manifest.json even after line ending conversion")
            ref = cat.get("reference", {})
            rows.append({
                "pcbworld_name": name,
                "pcbworld_tier": tier,
                "board_id": board_id,
                "source": cat.get("source", ""),
                "source_repo": repo_key(cat.get("source", "")),
                "license_spdx": cat.get("license_spdx") or "",
                "mirror_tier": cat.get("tier"),
                "layers": cat["board"]["layers"],
                "nets": cat["board"]["nets"],
                "components": cat["board"]["components"],
                "reference_segments": ref.get("segments"),
                "reference_vias": ref.get("vias"),
                "reference_track_length_mm": ref.get("track_length_mm"),
                "reference_unconnected": (ref.get("kicad_drc_breakdown") or {}).get("unconnected_count"),
                "raw_kicad_pcb_sha256": digest,
                "pcbench_source_sha256_crlf": upstream,
            })
    OUT.mkdir(exist_ok=True)
    json.dump({
        "description": "Held out test set. PCBWorld D3 test lists mapped onto the Freerouting PCBench mirror. Never train on these boards or on any board from the repositories in test_exclude_repos.txt.",
        "pcbworld_d3_commit": "ed411746224f",
        "pcbworld_test_overrides_applied": overrides,
        "mirror_commit": mirror_commit,
        "boards": rows,
    }, open(OUT / "test_d3.json", "w"), indent=1)
    repos = sorted({r["source_repo"] for r in rows})
    (OUT / "test_exclude_repos.txt").write_text("\n".join(repos) + "\n")
    tiers = {t: sum(1 for r in rows if r["pcbworld_tier"] == t) for t in ("easy", "medium", "hard")}
    print(f"test boards: {len(rows)} {tiers} | excluded repos: {len(repos)} | mirror commit {mirror_commit[:12]}")
    print(f"layers: { {l: sum(1 for r in rows if r['layers']==l) for l in sorted({r['layers'] for r in rows})} } | with unconnected items in the human reference: {sum(1 for r in rows if (r['reference_unconnected'] or 0) > 0)}")
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
