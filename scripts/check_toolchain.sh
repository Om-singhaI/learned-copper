#!/usr/bin/env bash
# Verifies every tool the pipeline needs. Run from the repository root.
# Paths can be overridden with KICAD_APP, JAVA21 and FREEROUTING_JAR.
KICAD_APP="${KICAD_APP:-/Applications/KiCad/KiCad.app/Contents}"
KICAD_CLI="$KICAD_APP/MacOS/kicad-cli"
KICAD_PY="$KICAD_APP/Frameworks/Python.framework/Versions/Current/bin/python3"
JAVA21="${JAVA21:-/opt/homebrew/opt/openjdk@21/bin/java}"
FREEROUTING_JAR="${FREEROUTING_JAR:-tools/freerouting.jar}"
fail=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }

echo "KiCad"
if v=$("$KICAD_CLI" version 2>/dev/null); then ok "kicad-cli $v"; else bad "kicad-cli not found at $KICAD_CLI"; fi
if v=$("$KICAD_PY" -c 'import pcbnew,sys;print(pcbnew.GetBuildVersion(), "python", sys.version.split()[0])' 2>/dev/null); then ok "pcbnew import via bundled python ($v)"; else bad "pcbnew import failed with $KICAD_PY"; fi
if "$KICAD_PY" -c 'import wx' 2>/dev/null; then ok "wx available in bundled python (needed for DSN export)"; else bad "wx missing in bundled python"; fi

echo "Java and Freerouting"
if v=$("$JAVA21" -version 2>&1 | head -1); then ok "$v"; else bad "java not found at $JAVA21"; fi
if [ -f "$FREEROUTING_JAR" ]; then ok "freerouting jar at $FREEROUTING_JAR ($(du -h "$FREEROUTING_JAR" | cut -f1))"; else bad "freerouting jar missing at $FREEROUTING_JAR (download from github.com/freerouting/freerouting/releases)"; fi

echo "Python environment"
if [ -x .venv/bin/python ]; then
  ok "venv $(.venv/bin/python --version)"
  if .venv/bin/python -c 'import torch, lightgbm, datasets, kiutils, numpy, pandas' 2>/dev/null; then ok "dependencies import"; else bad "dependencies missing, run: .venv/bin/python -m pip install -e '.[dev]'"; fi
else
  bad "no .venv, run: /opt/homebrew/bin/python3.13 -m venv .venv"
fi

echo "Disk"
ok "$(df -h / | tail -1 | awk '{print $4" free"}')"
exit $fail
