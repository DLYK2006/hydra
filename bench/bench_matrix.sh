#!/usr/bin/env bash
# Timed KV benchmark matrix against ONE already-launched Mac block worker (timing build; the
# loop applies the launch gate first). Run from repo root on WSL2, detached (tmux), e.g.
#   tmux new-session -d -s hydra_bench 'bash bench/bench_matrix.sh BATCH 12 6'
# Usage: bash bench/bench_matrix.sh BATCH N_DECODE [REPS=6] [--dry-run]
#   --dry-run: print the schedule that would be used and exit (no checks, nothing written).
# RESUMABLE: rerunning with the same BATCH/N/REPS skips every run whose "done" line is in
# matrix.log; a run that started but never finished has its partial dir/log RENAMED to
# *.aborted_<UTC> (never deleted) and is rerun. A different schedule for an existing BATCH is
# refused. Only one instance per BATCH (flock).
# Writes only bench/raw/BATCH/ (gitignored):
#   schedule.csv (written once, before the first run), specs_{wsl,windows,mac}_<UTC>.txt,
#   ping_<tag>_<UTC>.txt, runs/<run_id>/ (bench_run.py outputs) + runs/<run_id>.log,
#   matrix.log; on completion SHA256SUMS (all raw files except itself, matrix.log and the lock)
#   and matrix.log.sha256 (hash of the final matrix.log).
# Order: anchor pair (fixed G5 prompt; must reproduce data/gen_ref_tokens.txt), discarded
# warmup pair (L=32), then per rep the 4 lengths in a rotated order; per length on/off back to
# back, first side = (rep + length index) % 2. With an EVEN number of reps every length starts
# "on" and "off" equally often; within each rep two lengths start each way.
# Any run failure stops the matrix (rerun to resume). Every Windows python.exe /
# powershell.exe / ssh call gets < /dev/null: under WSL interop a Windows process otherwise
# consumes the loop's stdin (the schedule, read on fd 3).
set -u
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
M=yonghowlee@10.0.0.62
BATCH=${1:?batch}; N=${2:?n_decode}; REPS=${3:-6}; MODE=${4:-run}; COOLDOWN=30
OUT=bench/raw/$BATCH
LENS=(32 128 512 $((1024 - N)))
ts() { date -u +%Y%m%dT%H%M%SZ; }

gen_schedule() {   # columns: order,run_id,rep,prompt_file,kv_cache,n_decode
  local k=0 r i j L a b
  echo "order,run_id,rep,prompt_file,kv_cache,n_decode"
  emit() { k=$((k + 1)); echo "$k,$1,$2,$3,$4,$5"; }
  emit anchor_on 0 bench/prompts/prompt_g5.txt on 19;  emit anchor_off 0 bench/prompts/prompt_g5.txt off 19
  emit warmup_on 0 bench/prompts/prompt_L32.txt on "$N"; emit warmup_off 0 bench/prompts/prompt_L32.txt off "$N"
  for r in $(seq 1 "$REPS"); do
    for i in 0 1 2 3; do
      j=$(( (i + r - 1) % 4 )); L=${LENS[$j]}   # j = length index; slot i rotates per rep
      # keyed on (rep + length index), NOT the slot: keying on the slot gives each length the
      # same first side every rep.
      if [ $(( (r + j) % 2 )) -eq 0 ]; then a=on; b=off; else a=off; b=on; fi
      emit "r${r}_L${L}_$a" "$r" "bench/prompts/prompt_L$L.txt" "$a" "$N"
      emit "r${r}_L${L}_$b" "$r" "bench/prompts/prompt_L$L.txt" "$b" "$N"
    done
  done
}
SCHED=$(gen_schedule)
if [ "$MODE" = --dry-run ]; then echo "$SCHED"; exit 0; fi
[ "$MODE" = run ] || { echo "ERROR: unknown 4th argument '$MODE'"; exit 1; }
[ $((REPS % 2)) -eq 0 ] || echo "WARNING: odd REPS=$REPS -> per-length on/off-first split is unbalanced"

for f in bench/prompts/prompt_g5.txt $(for L in "${LENS[@]}"; do echo bench/prompts/prompt_L$L.txt; done) \
         bench/prompts/SHA256SUMS bench/bench_run.py bench/bench_matrix.sh bench/summarize.py gen_hydra.py \
         data/gen_ref_tokens.txt; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  git cat-file -e "HEAD:$f" 2>/dev/null && git diff --quiet HEAD -- "$f" || { echo "ERROR: $f not committed/unmodified"; exit 1; }
done
(cd bench/prompts && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR: prompt sha256 mismatch"; exit 1; }

if [ -e "$OUT" ]; then
  [ "$(cat "$OUT/schedule.csv" 2>/dev/null)" = "$SCHED" ] \
    || { echo "ERROR: $OUT exists with a different (or no) schedule; refusing"; exit 1; }
  [ -e "$OUT/SHA256SUMS" ] && { echo "ERROR: $OUT is already complete (SHA256SUMS present)"; exit 1; }
  RESUME=1
else
  mkdir -p "$OUT/runs"; RESUME=0
fi
exec 9> "$OUT/.matrix.lock"
flock -n 9 || { echo "ERROR: another bench_matrix.sh holds $OUT/.matrix.lock"; exit 1; }
die() { echo "$1" | tee -a "$OUT/matrix.log"; echo "BENCH_MATRIX_FAIL" | tee -a "$OUT/matrix.log"; exit 1; }

# SPECS (the client runs on Windows Python, so the Windows host is recorded too). Any failure
# stops the matrix; files are UTC-stamped, never overwritten.
specs() { local t; t=$(ts)
  { lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket' && free -b && uname -a && head -2 /etc/os-release; } \
    > "$OUT/specs_wsl_$t.txt" 2>&1 || die "ERROR: WSL specs failed (see $OUT/specs_wsl_$t.txt)"
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed; Get-CimInstance Win32_ComputerSystem | Format-List TotalPhysicalMemory; Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version" \
    > "$OUT/specs_windows_$t.txt" 2>&1 < /dev/null || die "ERROR: Windows specs failed (see $OUT/specs_windows_$t.txt)"
  ssh -o BatchMode=yes $M "sysctl machdep.cpu.brand_string hw.ncpu hw.physicalcpu hw.memsize; sw_vers" \
    > "$OUT/specs_mac_$t.txt" 2>&1 < /dev/null || die "ERROR: Mac specs failed (see $OUT/specs_mac_$t.txt)"
  echo "$(date -u +%FT%TZ) specs recorded ($t)" >> "$OUT/matrix.log"; }
if [ "$RESUME" -eq 0 ]; then
  echo "$SCHED" > "$OUT/schedule.csv"
  echo "$(date -u +%FT%TZ) BEGIN batch=$BATCH n_decode=$N reps=$REPS git HEAD $(git rev-parse HEAD)" > "$OUT/matrix.log"
  specs
else
  echo "$(date -u +%FT%TZ) RESUME git HEAD $(git rev-parse HEAD)" >> "$OUT/matrix.log"
  grep -q "specs recorded" "$OUT/matrix.log" || specs   # an earlier attempt died during specs
fi

ping_tag() { local f="$OUT/ping_$1_$(ts).txt"
             { date -u +%FT%TZ; ping -c 20 10.0.0.62; } > "$f" 2>&1 < /dev/null || die "ERROR: ping $1 failed (see $f)"
             tail -1 "$f" | sed "s/^/ping $1: /" >> "$OUT/matrix.log"; }
check_anchors() {   # both anchors must reproduce the G1/G5 tokens before anything else runs
  local want a; want=$(grep -v '^#' data/gen_ref_tokens.txt | awk '{print $2}')
  for a in anchor_on anchor_off; do
    [ "$(cat "$OUT/runs/$a/tokens.txt" 2>/dev/null)" = "$want" ] || die "FAIL $a tokens != data/gen_ref_tokens.txt"
  done
  echo "PASS anchors: on and off reproduce data/gen_ref_tokens.txt" | tee -a "$OUT/matrix.log"; }

n_sched=0; prev_rep=-1; anchors_ok=0
while IFS=, read -r ord id rep pf mode nd <&3; do
  n_sched=$((n_sched + 1))
  # anchor gate: before the first non-anchor row, in EVERY invocation (fresh or resumed; anchors
  # either just ran or were skipped as done earlier)
  if [ "$anchors_ok" -eq 0 ] && [ "$id" != anchor_on ] && [ "$id" != anchor_off ]; then
    check_anchors; anchors_ok=1
  fi
  if grep -q " done  $ord $id\$" "$OUT/matrix.log"; then
    echo "$(date -u +%FT%TZ) skip  $ord $id (done)" >> "$OUT/matrix.log"; continue
  fi
  for p in "$OUT/runs/$id" "$OUT/runs/$id.log"; do   # partial attempt: rename, never delete
    [ -e "$p" ] && { mv "$p" "$p.aborted_$(ts)"; echo "$(date -u +%FT%TZ) renamed partial $p" >> "$OUT/matrix.log"; }
  done
  if [ "$rep" != "$prev_rep" ]; then
    if [ "$prev_rep" != -1 ]; then ping_tag "rep${prev_rep}_after"; ping_tag "rep${rep}_before"
    elif [ "$RESUME" -eq 1 ]; then ping_tag "rep${rep}_resume"   # first ping after a resume: may be mid-rep
    else ping_tag "rep${rep}_before"; fi
    prev_rep=$rep
  fi
  echo "$(date -u +%FT%TZ) start $ord $id" >> "$OUT/matrix.log"
  "$PY" bench/bench_run.py --prompt-file "$pf" --kv-cache "$mode" --n-decode "$nd" \
      --run-id "$id" --out "$OUT/runs/$id" > "$OUT/runs/$id.log" 2>&1 < /dev/null \
    || die "FAIL run $id (see $OUT/runs/$id.log); stopping (rerun the same command to resume)"
  echo "$(date -u +%FT%TZ) done  $ord $id" >> "$OUT/matrix.log"
  sleep "$COOLDOWN"
done 3< <(echo "$SCHED" | tail -n +2)
ping_tag "rep${REPS}_after"
n_done=$(grep -c ' done  ' "$OUT/matrix.log")
[ "$n_done" -eq "$n_sched" ] || die "FAIL: $n_done of $n_sched scheduled runs completed"
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name matrix.log ! -name matrix.log.sha256 ! -name .matrix.lock \
   | sort | xargs -d '\n' sha256sum) > "$OUT/SHA256SUMS"
echo "BENCH_MATRIX_DONE ($n_done/$n_sched runs)" | tee -a "$OUT/matrix.log"
sha256sum "$OUT/matrix.log" | awk '{print $1}' > "$OUT/matrix.log.sha256"
