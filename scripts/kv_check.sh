#!/usr/bin/env bash
# KV-cache checker. Hand-reviewed; do not edit. Run from repo root on WSL2, with the Mac worker
# FRESHLY launched from the KV build in --mode block. Covers gates a-e against that one
# worker. Gate f (G3 regression: g3_check_v2.sh + regress_mode.sh x4) needs the worker
# relaunched per mode, so the loop runs those existing scripts separately.
#   [stage A] cache OFF: scripts/g5_check.sh passes, LOCK byte-identical to frozen.
#   [stage B] cache ON (cold, first cached sequence on this worker; head on the newest row
#             only): 20 tokens = G1; prefill rows byte-identical to the mirror; each step's
#             newest block-11 row byte-identical to the mirror AND to the frozen cache-off run
#             (data/live/g5_run1); logits within the a-priori float64 dot-product bound of
#             data/g5_hydra_logits.bin, element by element; the client really used the cache
#             (prefill T=11, then T=1 at start_pos 10+s).
#   [stage C] 64 tokens greedy, cache ON vs OFF (hard): token IDs identical; prefill rows and
#             each step's newest block-11 row byte-identical; logits within the same bound.
#   [stage D] stale cache: a different prompt (cache ON), then the fixed prompt (cache ON)
#             must be byte-identical (tokens, logits, hidden) to stage B's cold run.
#   [stage E] protocol + overflow (scripts/kv_overflow.py): oversize prefill rejected; a
#             rejected overflow leaves the cache unchanged; replay/gap/new-seq-not-at-0
#             rejected, each followed by an accepted aligned packet; chunked (4+4) prefill
#             byte-identical to a single 8-row prefill; fill to exactly 1024 accepted, the
#             1025th row rejected; the worker still serves a new sequence afterwards.
# Stage D's other prompt is 9 tokens (+5 generated = 13 cached positions), longer than the
# fixed prompt's 11: positions 11-12 stay stale after the fixed prompt's prefill, so a
# worker that does not reset or that reads past the current length is exposed.
set -u
# Windows Python: WSL2's native Python has no numpy/torch (CLAUDE.md, M1 notes).
# Relative paths work only because the repo lives on /mnt/c (interop maps cwd).
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
STAGE=data/live/kv
OTHER_PROMPT="The quick brown fox jumps over the lazy dog"
fail=0

die() { echo "$1"; echo "KV_CHECK_FAIL"; exit 1; }

echo "[stage PRECHECK]"
[ -e "$STAGE" ] && die "ERROR: $STAGE exists; remove it to rerun (refusing to overwrite)"
[ -e data/live/g5 ] && die "ERROR: data/live/g5 exists; stage A runs g5_check.sh, which needs it absent"
for f in tokens.txt logits_f64.bin hidden_f32.bin; do
  [ -f "data/live/g5_run1/$f" ] || die "MISSING data/live/g5_run1/$f (pre-KV G5 run 1 outputs, stage A baseline)"
done
for f in gen_hydra.py hydra_pb2.py scripts/kv_check.sh scripts/kv_compare.py scripts/kv_overflow.py \
         scripts/g5_check.sh scripts/g5_compare.py scripts/g5_compare_hidden.py \
         data/g5_mirror_hidden.bin data/g5_mirror_hidden_index.txt data/g5_hydra_logits.bin \
         data/g5_hydra_logits.sha256 data/gen_ref_tokens.txt; do
  [ -f "$f" ] || die "MISSING $f"
done
git rev-parse --verify -q HEAD > /dev/null || die "ERROR: git unavailable or not in a repo with a HEAD"
for f in gen_hydra.py scripts/kv_check.sh scripts/kv_compare.py scripts/kv_overflow.py \
         scripts/g5_check.sh scripts/g5_compare.py scripts/g5_compare_hidden.py \
         data/g5_mirror_hidden.bin data/g5_mirror_hidden_index.txt data/g5_hydra_logits.bin \
         data/g5_hydra_logits.sha256 data/gen_ref_tokens.txt; do
  git cat-file -e "HEAD:$f" 2>/dev/null || die "ERROR: $f is not committed"
  git diff --quiet HEAD -- "$f" || die "ERROR: $f differs from its committed version"
done
# hydra_pb2.py is generated and gitignored: require that it carries the new fields.
"$PY" -W error -c "import hydra_pb2; p = hydra_pb2.TensorPacket(sequence_id=1, start_pos=2); assert p.sequence_id == 1 and p.start_pos == 2" \
  || die "ERROR: hydra_pb2.py lacks sequence_id/start_pos; regenerate it (protoc --python_out=. hydra.proto)"
mkdir -p "$STAGE" || die "ERROR: cannot create $STAGE"

# gen <outdir> <args...>: one gen_hydra.py run; the client refuses to overwrite its outputs.
gen() { local out=$1; shift
  "$PY" gen_hydra.py --out "$out" "$@" > "$out.log" 2>&1; local rc=$?
  tr -d '\r' < "$out.log" | grep -E "^Wrote |ERROR|Error|rejected|bad reply" ; return $rc; }

echo "[stage A] cache OFF via scripts/g5_check.sh"
bash scripts/g5_check.sh > "$STAGE/a_g5_check.log" 2>&1; arc=$?
tr -d '\r' < "$STAGE/a_g5_check.log" | grep -E "G5 hidden-state gate|G5 compare|PASS logits byte-identical|G5_CHECK_"
if [ "$arc" -eq 0 ] && grep -q "PASS logits byte-identical to frozen" "$STAGE/a_g5_check.log" \
   && grep -q "^G5_CHECK_PASS" "$STAGE/a_g5_check.log"; then echo "PASS stage A"
else echo "FAIL stage A (exit $arc)"; fail=1; fi
# Cache-off outputs must be byte-identical to the pre-KV client's G5 run 1 (header included).
for f in tokens.txt logits_f64.bin hidden_f32.bin; do
  if cmp -s "data/live/g5/$f" "data/live/g5_run1/$f"; then echo "PASS stage A: $f byte-identical to data/live/g5_run1/$f"
  else echo "FAIL stage A: $f differs from data/live/g5_run1/$f"; fail=1; fi
done

echo "[stage B] cache ON, cold, fixed prompt, 20 tokens"
if gen "$STAGE/b_on20" --kv-cache on; then
  "$PY" scripts/kv_compare.py b "$STAGE/b_on20" 2>&1 | tr -d '\r' | tee "$STAGE/b_compare.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
else echo "FAIL stage B: gen_hydra.py failed, see $STAGE/b_on20.log"; fail=1; fi

echo "[stage C] 64 tokens greedy, cache ON vs OFF"
if gen "$STAGE/c_on64" --kv-cache on --n-gen 64 && gen "$STAGE/c_off64" --kv-cache off --n-gen 64; then
  "$PY" scripts/kv_compare.py c "$STAGE/c_on64" "$STAGE/c_off64" 2>&1 | tr -d '\r' | tee "$STAGE/c_compare.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
else echo "FAIL stage C: gen_hydra.py failed, see $STAGE/c_*64.log"; fail=1; fi

echo "[stage D] stale cache: other prompt, then fixed prompt == stage B cold run"
if gen "$STAGE/d_other" --kv-cache on --prompt "$OTHER_PROMPT" --n-gen 5 \
   && gen "$STAGE/d_fixed" --kv-cache on; then
  dok=1
  for f in tokens.txt logits_f64.bin hidden_f32.bin; do
    if cmp -s "$STAGE/d_fixed/$f" "$STAGE/b_on20/$f"; then echo "PASS stage D: $f byte-identical to cold run"
    else echo "FAIL stage D: $f differs from cold run"; dok=0; fi
  done
  [ "$dok" -eq 1 ] || fail=1
else echo "FAIL stage D: gen_hydra.py failed, see $STAGE/d_*.log"; fail=1; fi

echo "[stage E] overflow past 1024"
"$PY" scripts/kv_overflow.py 2>&1 | tr -d '\r'; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL stage E"; fail=1; }

if [ "$fail" -eq 0 ]; then echo "KV_CHECK_PASS"; else echo "KV_CHECK_FAIL"; exit 1; fi
