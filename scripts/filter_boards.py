"""Turn the DRC summary and the inventory into the board pool manifest.

    .venv/bin/python scripts/filter_boards.py

Reads  data/drc_summary_refill.csv (scripts/drc_batch.py --refill), splits/mirror_inventory.csv,
       splits/test_d3.json, splits/test_exclude_repos.txt, and the mirror catalog.
Writes splits/mirror_pool.json: every mirror board with its tier and the reasons it was rejected, if any.

Tiers
  clean    fully connected after zone refill, has an outline, has tracks, no electrical errors,
           no clearance errors under the board's own rules
  flagged  same but with 1 to 10 clearance errors (usable for training, reported separately)
  rejected everything else, with reasons
Boards in the test set, or from a test repository, are marked held_out whatever their tier.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data/mirror/freerouting/scripts/benchmark/fixtures/PCBench"
MAX_FLAGGED_CLEARANCE = 10


def repo_key(url):
    m = re.match(r"https?://(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)/([^/]+)/([^/#?]+)", url or "")
    return f"{m.group(1)}/{m.group(2)}/{m.group(3).removesuffix('.git')}".lower() if m else (url or "").strip().lower() or "unknown"


def main():
    drc = {r["board_id"]: r for r in csv.DictReader(open(ROOT / "data/drc_summary_refill.csv"))}
    inv = {r["board_id"]: r for r in csv.DictReader(open(ROOT / "splits/mirror_inventory.csv"))}
    catalog = {b["board_id"]: b for b in json.load(open(FIX / "catalog.json"))["boards"]}
    test = json.load(open(ROOT / "splits/test_d3.json"))
    test_ids = {b["board_id"] for b in test["boards"]}
    test_repos = set(l.strip() for l in open(ROOT / "splits/test_exclude_repos.txt") if l.strip())

    rows, tiers, reasons = [], Counter(), Counter()
    for bid in sorted(catalog):
        d, i, c = drc.get(bid), inv.get(bid), catalog[bid]
        rej = []
        if d is None or d.get("error"):
            rej.append("drc did not run")
        else:
            if int(d["unconnected"]) > 0:
                rej.append("unconnected after refill")
            if int(d["invalid_outline"]) > 0:
                rej.append("no valid outline")
            if int(d["electrical_errors"]) > 0:
                rej.append("shorts or crossing tracks")
        if i is None or i.get("error"):
            rej.append("inventory failed")
        else:
            if float(i["segments"] or 0) == 0:
                rej.append("no tracks")
            if float(i["area_cm2"] or 0) == 0 and "no valid outline" not in rej:
                rej.append("no valid outline")
        clearance = int(d["clearance_total"]) if d and not d.get("error") else None
        if not rej and clearance is not None and clearance > MAX_FLAGGED_CLEARANCE:
            rej.append(f"more than {MAX_FLAGGED_CLEARANCE} clearance errors")
        tier = "rejected" if rej else ("clean" if clearance == 0 else "flagged")
        repo = repo_key(c.get("source", ""))
        held_out = bid in test_ids or repo in test_repos
        rows.append({
            "board_id": bid, "tier": tier, "held_out": held_out, "in_test_set": bid in test_ids, "source_repo": repo,
            "license_spdx": c.get("license_spdx") or "", "layers": c["board"]["layers"], "clearance_errors": clearance,
            "copper_errors": int(d["copper_errors"]) if d and not d.get("error") else None,
            "unconnected": int(d["unconnected"]) if d and not d.get("error") else None,
            "segments": int(float(i["segments"])) if i and not i.get("error") else None,
            "rejected_because": rej,
        })
        tiers[(tier, held_out)] += 1
        for r in rej:
            reasons[r] += 1
    out = {
        "description": "Mirror board pool after the DRC and connectivity filter (zones refilled before checking). "
                       "Training uses tier clean (and flagged, reported separately) with held_out false.",
        "max_flagged_clearance_errors": MAX_FLAGGED_CLEARANCE,
        "boards": rows,
    }
    json.dump(out, open(ROOT / "splits/mirror_pool.json", "w"), indent=1)
    print("tier counts (tier, held out):", dict(tiers))
    train_clean = sum(1 for r in rows if r["tier"] == "clean" and not r["held_out"])
    train_flagged = sum(1 for r in rows if r["tier"] == "flagged" and not r["held_out"])
    print(f"training pool: clean {train_clean}, flagged {train_flagged}, total {train_clean + train_flagged}")
    print("rejection reasons:", reasons.most_common())
    lic = Counter(r["license_spdx"] or "none" for r in rows if r["tier"] != "rejected" and not r["held_out"])
    print("licenses in the training pool:", lic.most_common(6))


if __name__ == "__main__":
    main()
