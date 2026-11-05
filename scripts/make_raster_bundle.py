"""Pack the rasters and splits into one archive for a private cloud training dataset.

    .venv/bin/python scripts/make_raster_bundle.py --tag 2026-11-05

Writes data/bundles/learned-copper-rasters-<tag>.tar.gz with
    data/raster/<board_id>.npz      every board in splits/raster_manifest.json
    splits/raster_manifest.json
    splits/mirror_pool.json
    README.txt                      what the files are and the terms they come with
and prints the size and the sha256 (also written to <bundle>.sha256). Extracting the archive at the
root of a clone of the repository puts every file where scripts/train_unet.py expects it.

The rasters derive from PCBench boards through the Freerouting mirror. The archive is for private
research use and is never committed (data/ is ignored by git) or redistributed.
"""
import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """learned-copper rasters, bundle {tag}

Private research data for the learned-copper project (github.com/Om-singhaI/learned-copper).
Built {date} from the repository's data/raster directory.

Contents
  data/raster/<board_id>.npz   {n_boards} two layer boards, {n_samples} samples,
                               {n_held_out} boards held out
  splits/raster_manifest.json  board id, source repository, held out flag, test set flag, tier,
                               sample count and mm per pixel for every raster
  splits/mirror_pool.json      every board of the Freerouting PCBench mirror with its tier,
                               design rule counts and the reason for any rejection
  README.txt                   this file

Each npz holds x uint8 [N, 6, {size}, {size}], y uint8 [N, 2, {size}, {size}], net_ids int32 [N] and
mm_per_px float32. One sample per net with at least two pads, at most 60 nets per board. Input
channels: 0 outline, 1 this net's pads on the front layer, 2 this net's pads on the back layer,
3 other pads on the front, 4 other pads on the back, 5 copper of all other nets on either layer.
Targets: this net's copper on the front and on the back. scripts/rasterize.py in the repository
documents how they were built.

Terms
The rasters are derived from PCBench boards, obtained through the Freerouting benchmark mirror,
and the boards keep the licenses recorded per board in splits/mirror_pool.json (license_spdx).
This bundle exists so the owner can train on a cloud machine. It is for private research use
only. Do not redistribute it, do not publish it as a dataset, and do not attach it to a public
notebook. Boards with held_out true in the manifest are never used for training.

Use
Extract at the root of a clone of the repository:
  tar -xzf learned-copper-rasters-{tag}.tar.gz -C learned-copper
then run scripts/train_unet.py (see docs/cloud-training.md).
"""


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="goes into the file name, for example 2026-11-05")
    p.add_argument("--data-dir", default=str(ROOT / "data/raster"))
    p.add_argument("--manifest", default=str(ROOT / "splits/raster_manifest.json"))
    p.add_argument("--pool", default=str(ROOT / "splits/mirror_pool.json"))
    p.add_argument("--out-dir", default=str(ROOT / "data/bundles"))
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    boards = sorted(manifest["boards"], key=lambda b: b["board_id"])
    data_dir = Path(args.data_dir)
    missing = [b["board_id"] for b in boards if not (data_dir / f"{b['board_id']}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} boards of the manifest have no raster, "
                         f"for example {missing[:3]}")
    known = {b["board_id"] for b in boards}
    extra = sorted(q.stem for q in data_dir.glob("*.npz") if q.stem not in known)
    if extra:
        print(f"note: {len(extra)} npz files are not in the manifest and are left out, "
              f"for example {extra[:3]}")

    readme = README.format(
        tag=args.tag, date=time.strftime("%B %-d, %Y"), n_boards=len(boards),
        n_samples=sum(int(b["n_samples"]) for b in boards),
        n_held_out=sum(1 for b in boards if b["held_out"]), size=manifest.get("size", 256))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"learned-copper-rasters-{args.tag}.tar.gz"
    t0 = time.time()
    with tarfile.open(bundle, "w:gz", compresslevel=6) as tar:
        data = readme.encode()
        info = tarfile.TarInfo("README.txt")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
        tar.add(args.manifest, arcname="splits/raster_manifest.json")
        tar.add(args.pool, arcname="splits/mirror_pool.json")
        for b in boards:
            tar.add(data_dir / f"{b['board_id']}.npz", arcname=f"data/raster/{b['board_id']}.npz")
    digest = sha256_of(bundle)
    size = bundle.stat().st_size
    (out_dir / f"{bundle.name}.sha256").write_text(f"{digest}  {bundle.name}\n")
    print(f"wrote {bundle}")
    print(f"  {len(boards)} boards, {size:,} bytes ({size / 1e6:.1f} MB), {time.time() - t0:.0f}s")
    print(f"  sha256 {digest}")


if __name__ == "__main__":
    main()
