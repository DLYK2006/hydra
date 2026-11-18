#!/usr/bin/env bash
# M7 step 0 (DRAFT): run one locked checker against the WSL2 worker_linux on localhost.
# The originals honor no host override (orchestrator.cpp:255, g3_check_v2.sh WORKER_HOST,
# gen_hydra.py's default, kv_overflow.py HOST are all hardcoded to 10.0.0.62) and are never
# edited. EVERY invocation regenerates the localhost copies from the originals as they are on disk
# now, into a fresh data/live/linux_checkers_<UTC>/, prints the full `diff -u` against the
# originals, then runs the requested copy. Nothing is reused between invocations.
# Each copy differs from its original ONLY in the lines substituted below: the worker host, the
# orchestrator binary (orchestrator_local, rebuilt each time from a copy of orchestrator.cpp with
# only the IP literal changed), the staging dirs (data/live/linux_<UTC>_*, so neither the Mac
# runs' dirs nor earlier local runs are touched) and the paths of copied helper scripts. Every
# substitution must change exactly the stated number of lines, or the script stops before
# running anything.
# Usage (WSL2, repo root, worker_linux already launched behind its gate in the right mode):
#   bash scripts/m7_local_checkers.sh g3_check_v2
#   bash scripts/m7_local_checkers.sh regress_mode layernorm|qkv|attention|mlp
#   bash scripts/m7_local_checkers.sh kv_check
set -eu
CHECK=${1:?g3_check_v2|regress_mode|kv_check}; shift
case "$CHECK" in g3_check_v2|regress_mode|kv_check) ;; *) echo "ERROR: unknown checker '$CHECK'"; exit 1 ;; esac
RUN=$(date -u +%Y%m%dT%H%M%SZ)
OUT=data/live/linux_checkers_$RUN
P=data/live/linux_${RUN}_           # staging-dir prefix for this invocation
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 1; }
mkdir -p "$OUT"

# mk SRC [SED_EXPR COUNT]...: copy SRC into $OUT; each SED_EXPR must change exactly COUNT lines.
mk() {
  local src=$1 dst="$OUT/$(basename "$1")"; shift
  cp "$src" "$dst"
  while [ $# -gt 0 ]; do
    local e=$1 n=$2 prev got; shift 2
    prev=$(mktemp); cp "$dst" "$prev"
    sed -i "$e" "$dst"
    got=$(diff "$prev" "$dst" | grep -c '^>' || true); rm -f "$prev"
    [ "$got" -eq "$n" ] || { echo "ERROR: $src: '$e' changed $got lines, want $n"; exit 1; }
  done
  diff -u "$src" "$dst" >> "$OUT/DIFF.txt" || true
}

mk orchestrator.cpp 's|"10\.0\.0\.62"|"127.0.0.1"|' 1
mk scripts/g3_check_v2.sh \
  's|^WORKER_HOST=10\.0\.0\.62 .*|WORKER_HOST=127.0.0.1   # M7 local copy|' 1 \
  's|^STAGE=data/live/g3v2$|STAGE='"$P"'g3v2|' 1 \
  's|^for f in orchestrator data/|for f in orchestrator_local data/|' 1 \
  's|\.\./\.\./\.\./orchestrator --mode|../../../orchestrator_local --mode|' 1
mk scripts/regress_mode.sh \
  's|^STAGE="data/live/regress_\$mode"$|STAGE="'"$P"'regress_$mode"|' 1 \
  's#^\[ -f orchestrator \] || die "MISSING orchestrator"$#[ -f orchestrator_local ] || die "MISSING orchestrator_local"#' 1 \
  's|\.\./\.\./\.\./orchestrator --mode|../../../orchestrator_local --mode|' 1
mk scripts/g5_check.sh \
  's|^STAGE=data/live/g5$|STAGE='"$P"'g5|' 1 \
  's|gen_hydra\.py --out "\$STAGE" >|gen_hydra.py --host 127.0.0.1 --out "$STAGE" >|' 1
mk scripts/kv_check.sh \
  's|^STAGE=data/live/kv$|STAGE='"$P"'kv|' 1 \
  's|^\[ -e data/live/g5 \] && die "ERROR: data/live/g5 exists|[ -e '"$P"'g5 ] \&\& die "ERROR: '"$P"'g5 exists|' 1 \
  's|^bash scripts/g5_check\.sh >|bash '"$OUT"'/g5_check.sh >|' 1 \
  's|cmp -s "data/live/g5/\$f"|cmp -s "'"$P"'g5/$f"|' 1 \
  's|"\$PY" gen_hydra\.py --out "\$out" "\$@"|"$PY" gen_hydra.py --host 127.0.0.1 --out "$out" "$@"|' 1 \
  's|^"\$PY" scripts/kv_overflow\.py|"$PY" '"$OUT"'/kv_overflow.py|' 1
mk scripts/kv_overflow.py \
  's|^sys\.path\.insert(0, os\.path\.dirname(os\.path\.dirname(os\.path\.abspath(__file__))))$|sys.path.insert(0, os.path.abspath("."))   # M7 local copy: run from the repo root|' 1 \
  's|^HOST, PORT, H = "10\.0\.0\.62", 50051, 768 .*|HOST, PORT, H = "127.0.0.1", 50051, 768   # M7 local copy|' 1

# orchestrator_local: the copied source, same compiler and flags as the Makefile's orchestrator.
# Rebuilt on every invocation (a derived artifact, gitignored), so it always matches the copy.
g++ -O2 -std=c++17 -Wall -Wextra -I. "$OUT/orchestrator.cpp" hydra.pb.cc -lprotobuf -o orchestrator_local

# Everything below is also saved to $OUT/run.log (hashes, the diff, the checker's full output).
set +e
{
  sha256sum orchestrator_local "$OUT/orchestrator.cpp" orchestrator.cpp
  echo "== diff -u (locked originals -> $OUT copies); $(grep -c '^[-+][^-+]' "$OUT/DIFF.txt") changed lines"
  cat "$OUT/DIFF.txt"
  echo "== running $OUT/$CHECK.sh $*"
  bash "$OUT/$CHECK.sh" "$@"
  echo "== checker exit $?"
} 2>&1 | tee "$OUT/run.log"
exit "$(tail -1 "$OUT/run.log" | sed -n 's/^== checker exit //p')"
