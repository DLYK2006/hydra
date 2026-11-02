#!/usr/bin/env bash
# G5 checker. Hand-reviewed; do not edit. Run from repo root on WSL2, with the Mac
# worker running the G3 build in --mode block. Runs gen_hydra.py (Windows Python) into
# its own staging dir data/live/g5/ (ignored via data/live/), then, in order:
#   [stage HIDDEN] scripts/g5_compare_hidden.py: every step's full block-11 output
#                  (STAGE/hidden_f32.bin) byte-identical to the committed chained-mirror
#                  oracle data/g5_mirror_hidden.bin (hidden states, not logits);
#   [stage LOGITS] scripts/g5_compare.py: 20 token rows byte-identical to
#                  data/gen_ref_tokens.txt; per-step max|hydra - gen_ref_logits_f64| <=
#                  G5 tolerance from data/gen_ref_summary.txt; every step's f64 margins
#                  (reference and Hydra) > 2 x tol;
#   [stage LOCK]   first full pass: freeze the logits as data/g5_hydra_logits.bin and
#                  its sha256 as data/g5_hydra_logits.sha256 (BOTH tracked; commit both).
#                  Once the pin is in HEAD: pin and .bin present and equal to HEAD's,
#                  sha256sum of the .bin equal to the pin line, and fresh logits
#                  cmp-identical to the .bin. A committed pin never leads to a re-freeze.
# The oracle (data/g5_mirror_hidden.bin + _index.txt) must be committed and unmodified:
# it is generated from the HF checkpoint and G1 gold tokens, before any Hydra run.
set -u
# Windows Python: WSL2's native Python has no numpy/torch (CLAUDE.md, M1 notes).
# Relative paths work only because the repo lives on /mnt/c (interop maps cwd).
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
STAGE=data/live/g5
FROZEN=data/g5_hydra_logits.bin
PIN=data/g5_hydra_logits.sha256
ORACLE=data/g5_mirror_hidden.bin
ORACLE_IDX=data/g5_mirror_hidden_index.txt
fail=0

die() { echo "$1"; echo "G5_CHECK_FAIL"; exit 1; }

echo "[stage PRECHECK]"
[ -e "$STAGE" ] && die "ERROR: $STAGE exists; remove it to rerun (refusing to overwrite)"
for f in gen_hydra.py hydra_pb2.py data/gen_ref_tokens.txt data/gen_ref_logits_f64.bin \
         data/gen_ref_summary.txt "$ORACLE" "$ORACLE_IDX" \
         scripts/g5_compare.py scripts/g5_compare_hidden.py; do
  [ -f "$f" ] || die "MISSING $f"
done
# git must work first: otherwise "not committed" could not be told apart from "git broken"
# and both the oracle check and the lock would fail open.
git rev-parse --verify -q HEAD > /dev/null || die "ERROR: git unavailable or not in a repo with a HEAD"
for f in "$ORACLE" "$ORACLE_IDX" data/gen_ref_tokens.txt data/gen_ref_summary.txt; do
  git cat-file -e "HEAD:$f" 2>/dev/null || die "ERROR: $f is not committed"
  git diff --quiet HEAD -- "$f" || die "ERROR: $f differs from its committed version"
done

echo "[stage LOCK-STATE]"
# Lock state is decided by git: once the pin is committed, a lost/modified pin or a
# lost/changed frozen file is a hard failure, never a re-freeze. The pin is compared
# byte-for-byte to sha256sum of $FROZEN itself, so it cannot be redirected elsewhere.
if git cat-file -e "HEAD:$PIN" 2>/dev/null; then
  locked=1
  [ -f "$PIN" ] || die "ERROR: $PIN is committed but missing from the working tree"
  git diff --quiet HEAD -- "$PIN" || die "ERROR: $PIN differs from its committed version"
  git cat-file -e "HEAD:$FROZEN" 2>/dev/null || die "ERROR: $PIN is committed but $FROZEN is not"
  [ -f "$FROZEN" ] || die "ERROR: $FROZEN missing; restore it with git (refusing to re-freeze)"
  git diff --quiet HEAD -- "$FROZEN" || die "ERROR: $FROZEN differs from its committed version"
  [ "$(sha256sum "$FROZEN")" = "$(cat "$PIN")" ] || die "ERROR: sha256sum $FROZEN != committed pin $PIN"
  echo "locked: frozen logits $(cut -c1-16 "$PIN")..."
else
  locked=0
  git cat-file -e "HEAD:$FROZEN" 2>/dev/null && die "ERROR: $FROZEN is committed but $PIN is not; resolve by hand"
  [ -e "$FROZEN" ] && die "ERROR: $FROZEN exists but no pin is committed; resolve by hand"
  [ -e "$PIN" ] && die "ERROR: $PIN exists but is not committed; commit it and $FROZEN first"
  echo "unlocked: first full pass will freeze"
fi
mkdir -p "$STAGE" || die "ERROR: cannot create $STAGE"

echo "[stage GEN]"
"$PY" gen_hydra.py --out "$STAGE" > "$STAGE/gen.log" 2>&1; rc=$?
tr -d '\r' < "$STAGE/gen.log" | grep -E "^step |^Wrote |ERROR|Error|rejected|bad reply"
for f in tokens.txt logits_f64.bin hidden_f32.bin; do
  [ -f "$STAGE/$f" ] || { echo "missing $STAGE/$f"; rc=99; }
done
[ "$rc" -eq 0 ] || die "FAIL: gen_hydra.py exit $rc, see $STAGE/gen.log"

echo "[stage HIDDEN]"
"$PY" scripts/g5_compare_hidden.py "$STAGE" 2>&1 | tr -d '\r'; hrc=${PIPESTATUS[0]}
[ "$hrc" -eq 0 ] || { echo "FAIL: hidden-state gate exit $hrc"; fail=1; }

echo "[stage LOGITS]"
"$PY" scripts/g5_compare.py "$STAGE" 2>&1 | tr -d '\r'; crc=${PIPESTATUS[0]}
[ "$crc" -eq 0 ] || { echo "FAIL: logit gate exit $crc"; fail=1; }

echo "[stage LOCK]"
if [ "$fail" -eq 0 ]; then
  if [ "$locked" -eq 1 ]; then
    if cmp -s "$STAGE/logits_f64.bin" "$FROZEN"; then
      echo "PASS logits byte-identical to frozen $FROZEN ($(cut -c1-16 "$PIN")...)"
    else
      echo "FAIL logits differ from frozen $FROZEN"; fail=1
    fi
  else
    cp "$STAGE/logits_f64.bin" "$FROZEN" && sha256sum "$FROZEN" > "$PIN" \
      || die "ERROR: freeze failed"
    echo "FROZEN first full pass: $FROZEN, pin $(cut -c1-16 "$PIN")... in $PIN"
    echo "NEXT (by hand): commit $FROZEN and $PIN together; reruns refuse until both are committed"
  fi
else
  echo "skipped (an earlier gate failed; nothing frozen)"
fi

if [ "$fail" -eq 0 ]; then echo "G5_CHECK_PASS"; else echo "G5_CHECK_FAIL"; exit 1; fi
