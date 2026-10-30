#!/usr/bin/env bash
# Mode regression gate. Hand-reviewed; do not edit.
# Usage (from repo root on WSL2): bash scripts/regress_mode.sh <mode>
#   <mode> = layernorm | qkv | attention | mlp
# Precondition: the Mac worker is already running in --mode <mode>.
# Replays the 2026-10-28 live-run orchestrator commands for <mode> from a
# private staging dir data/live/regress_<mode>/ (ignored via data/live/) and
# requires every fresh response to be byte-identical (cmp) to the tracked June
# response of the same name in data/.
#   layernorm: --T 1, --T 4  -> ln_response_T1.bin, ln_response_T4.bin
#   qkv:       --T 4         -> qkv_response_T4.bin
#   attention: --T 1, --T 8  -> attn_response_T1.bin, attn_response_T8.bin
#   mlp:       --T 8         -> mlp_response_gelu_T8.bin, mlp_response_T8.bin
set -u
mode="${1:-}"
case "$mode" in
  layernorm) prefix=ln;   runs="1 4"; outs_for() { echo "ln_response_T$1.bin"; } ;;
  qkv)       prefix=qkv;  runs="4";   outs_for() { echo "qkv_response_T$1.bin"; } ;;
  attention) prefix=attn; runs="1 8"; outs_for() { echo "attn_response_T$1.bin"; } ;;
  mlp)       prefix=mlp;  runs="8";   outs_for() { echo "mlp_response_gelu_T$1.bin mlp_response_T$1.bin"; } ;;
  *) echo "usage: bash scripts/regress_mode.sh layernorm|qkv|attention|mlp (got '$mode')"
     echo "REGRESS_FAIL"; exit 1 ;;
esac
TAG=$(echo "$mode" | tr '[:lower:]' '[:upper:]')
STAGE="data/live/regress_$mode"
fail=0

die() { echo "$1"; echo "REGRESS_${TAG}_FAIL"; exit 1; }

[ -e "$STAGE" ] && die "ERROR: $STAGE exists; remove it to rerun (refusing to overwrite)"
[ -f orchestrator ] || die "MISSING orchestrator"
for T in $runs; do
  [ -f "data/${prefix}_input_T$T.bin" ] || die "MISSING data/${prefix}_input_T$T.bin"
  for o in $(outs_for "$T"); do
    [ -f "data/$o" ] || die "MISSING June oracle data/$o"
  done
done
mkdir -p "$STAGE/data" || die "ERROR: cannot create $STAGE"
for T in $runs; do
  cp "data/${prefix}_input_T$T.bin" "$STAGE/data/" || die "ERROR: staging failed"
done

for T in $runs; do
  # Run from inside $STAGE so the orchestrator's hardcoded data/<x>_response
  # paths land in $STAGE/data/, never in data/.
  (cd "$STAGE" && ../../../orchestrator --mode "$mode" --T "$T" > "orch_T$T.txt" 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $mode T=$T: orchestrator exit $rc, see $STAGE/orch_T$T.txt"; fail=1
  fi
  for o in $(outs_for "$T"); do
    if [ ! -f "$STAGE/data/$o" ]; then
      echo "FAIL $mode T=$T: no fresh $o"; fail=1
    elif cmp -s "$STAGE/data/$o" "data/$o"; then
      echo "PASS $mode T=$T: $o byte-identical to June data/$o"
    else
      echo "FAIL $mode T=$T: $o differs from June data/$o"; fail=1
    fi
  done
done

if [ "$fail" -eq 0 ]; then echo "REGRESS_${TAG}_PASS"; else echo "REGRESS_${TAG}_FAIL"; exit 1; fi
