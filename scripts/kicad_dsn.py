"""Specctra DSN export and SES import through KiCad's bundled Python.

Run with the interpreter inside KiCad.app, never the project venv:
    $KICAD_PY scripts/kicad_dsn.py export board.kicad_pcb board.dsn
    $KICAD_PY scripts/kicad_dsn.py import board.kicad_pcb board.ses routed.kicad_pcb

The exporter needs a wx application object, and on some boards it opens an
error dialog that never closes in a headless run. Callers should give each
board its own subprocess with a timeout; see run_with_timeout below.
"""
import subprocess
import sys

KICAD_PY = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"


def run_with_timeout(args, timeout=120):
    """Run this script in a fresh KiCad Python process. Returns (returncode, output)."""
    try:
        p = subprocess.run([KICAD_PY, __file__, *args], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main():
    import pcbnew
    import wx

    wx.App(False)
    cmd = sys.argv[1]
    if cmd == "export":
        src, dst = sys.argv[2], sys.argv[3]
        board = pcbnew.LoadBoard(src)
        ok = pcbnew.ExportSpecctraDSN(board, dst)
        print("export", "ok" if ok else "failed", dst)
        sys.exit(0 if ok else 1)
    if cmd == "import":
        src, ses, dst = sys.argv[2], sys.argv[3], sys.argv[4]
        board = pcbnew.LoadBoard(src)
        ok = pcbnew.ImportSpecctraSES(board, ses)
        if ok:
            pcbnew.SaveBoard(dst, board)
        print("import", "ok" if ok else "failed", dst, "tracks+vias", sum(1 for _ in board.GetTracks()))
        sys.exit(0 if ok else 1)
    print("usage: export SRC DST | import SRC SES DST")
    sys.exit(2)


if __name__ == "__main__":
    main()
