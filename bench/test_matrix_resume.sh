#!/usr/bin/env bash
# Offline test of bench/bench_matrix.sh scheduling + resume (no worker, no network, no Python):
# runs a copy in a throwaway git repo with stubbed python.exe/ssh/ping/powershell and COOLDOWN=0.
# The stub client fails once on r1_L128_on (first attempt only), after writing a partial dir.
# Checks: the failed first pass stops with BENCH_MATRIX_FAIL; a concurrent instance is refused by
# the lock; the resume skips every completed run, renames the partial attempt, finishes with
# BENCH_MATRIX_DONE, and the "done" order equals schedule.csv; --dry-run prints the schedule.
# Usage (WSL2, repo root): bash bench/test_matrix_resume.sh
set -u
SRC=$(pwd)
T=$(mktemp -d); cd "$T"
git init -q . && git config user.email t@t && git config user.name t
mkdir -p bench/prompts data stub
cp "$SRC/bench/bench_matrix.sh" bench/
sed -i 's#^PY=.*#PY='"$T"'/stub/python.exe#; s#COOLDOWN=30#COOLDOWN=0#' bench/bench_matrix.sh
for f in bench_run.py summarize.py; do echo "# stub" > bench/$f; done
echo "# stub" > gen_hydra.py
printf '# h\n%4d  %7d  x\n' 0 11 1 12 > data/gen_ref_tokens.txt
for L in g5 32 128 512 1012; do printf '# p\n1\n2\n' > bench/prompts/prompt_L$L.txt; done
mv bench/prompts/prompt_Lg5.txt bench/prompts/prompt_g5.txt
(cd bench/prompts && sha256sum prompt_* > SHA256SUMS)
cat > stub/python.exe <<'EOF'
#!/usr/bin/env bash
# stub bench_run.py: args ... --run-id ID --out DIR
while [ $# -gt 0 ]; do case $1 in --run-id) id=$2; shift;; --out) out=$2; shift;; esac; shift; done
if [ "$id" = r1_L128_on ] && [ ! -e "$(dirname "$out")/../.failed_once" ]; then
  mkdir -p "$out"; echo partial > "$out/packets.csv"; touch "$(dirname "$out")/../.failed_once"; echo "boom"; exit 3
fi
mkdir -p "$out"
case $id in
  anchor_*) if [ -e "$STUB_BAD_ANCHOR" ]; then printf '11\n99\n' > "$out/tokens.txt"; else printf '11\n12\n' > "$out/tokens.txt"; fi;;
  *) echo 7 > "$out/tokens.txt";;
esac
echo "{}" > "$out/run.json"; sleep 0.2; echo "Wrote $out"
EOF
for c in ssh ping powershell.exe; do printf '#!/usr/bin/env bash\necho "stub %s $*"\n' "$c" > stub/$c; done
chmod +x stub/*
export PATH="$T/stub:$PATH"
git add -A && git commit -qm t
ok=1
bash bench/bench_matrix.sh B 12 6 > pass1.out 2>&1; rc1=$?
grep -q BENCH_MATRIX_FAIL pass1.out && [ $rc1 -ne 0 ] && echo "PASS first pass stops on the failing run (rc=$rc1)" || { echo "FAIL first pass"; ok=0; }
# concurrent instance: hold the lock, try a second run
( exec 9> bench/raw/B/.matrix.lock; flock 9; sleep 3 ) & sleep 0.5
bash bench/bench_matrix.sh B 12 6 > lock.out 2>&1; rcl=$?
grep -q "another bench_matrix.sh holds" lock.out && echo "PASS concurrent instance refused" || { echo "FAIL lock: $(cat lock.out)"; ok=0; }
wait
bash bench/bench_matrix.sh B 12 6 > pass2.out 2>&1; rc2=$?
tail -1 pass2.out
[ $rc2 -eq 0 ] && grep -q "BENCH_MATRIX_DONE (52/52 runs)" pass2.out && echo "PASS resume completes 52/52" || { echo "FAIL resume rc=$rc2"; ok=0; }
nskip=$(grep -c " skip  " bench/raw/B/matrix.log); want_skip=$(( $(grep -n ",r1_L128_on," bench/raw/B/schedule.csv | cut -d: -f1) - 2 ))
[ "$nskip" -eq "$want_skip" ] && echo "PASS resume skipped the $want_skip runs done before the failure" || { echo "FAIL skipped $nskip"; ok=0; }
ls -d bench/raw/B/runs/r1_L128_on.aborted_* bench/raw/B/runs/r1_L128_on.log.aborted_* > /dev/null 2>&1 \
  && echo "PASS partial attempt renamed (dir + log), not deleted" || { echo "FAIL partial not renamed"; ok=0; }
done_ids=$(grep -E '^\S+ done  ' bench/raw/B/matrix.log | awk '{print $4}')
[ "$done_ids" = "$(tail -n +2 bench/raw/B/schedule.csv | cut -d, -f2)" ] && echo "PASS done order == schedule.csv" || { echo "FAIL done order"; ok=0; }
[ "$(sha256sum bench/raw/B/matrix.log | awk '{print $1}')" = "$(cat bench/raw/B/matrix.log.sha256)" ] && echo "PASS matrix.log.sha256" || { echo "FAIL matrix.log.sha256"; ok=0; }
bash bench/bench_matrix.sh B 12 6 > pass3.out 2>&1; grep -q "already complete" pass3.out && echo "PASS completed batch refuses rerun" || { echo "FAIL rerun of complete batch"; ok=0; }
[ "$(bash bench/bench_matrix.sh X 12 6 --dry-run | wc -l)" -eq 53 ] && echo "PASS --dry-run prints header + 52 rows" || { echo "FAIL dry-run"; ok=0; }
# anchor gate must also hold on RESUME (anchors skipped as "done")
export STUB_BAD_ANCHOR="$T/bad_anchor"; touch "$STUB_BAD_ANCHOR"
bash bench/bench_matrix.sh C 12 6 > anc1.out 2>&1; ra=$?
bash bench/bench_matrix.sh C 12 6 > anc2.out 2>&1; rb=$?
if [ $ra -ne 0 ] && [ $rb -ne 0 ] && grep -q "FAIL anchor_on tokens" anc1.out && grep -q "FAIL anchor_on tokens" anc2.out \
   && grep -q "skip  1 anchor_on" bench/raw/C/matrix.log && ! grep -q "start 3 warmup_on" bench/raw/C/matrix.log; then
  echo "PASS bad anchors stop the first pass AND the resume (no warmup/timed run started)"
else echo "FAIL anchor gate on resume"; ok=0; fi
grep -q "specs recorded" bench/raw/C/matrix.log && ls bench/raw/C/specs_mac_*.txt > /dev/null 2>&1 \
  && echo "PASS specs recorded once, stamped" || { echo "FAIL specs"; ok=0; }
# batch B resumed at r1_L128_on (mid rep 1): its first ping after RESUME must be labelled rep1_resume
awk '/ RESUME /{r=1} r && /^ping /{print; exit}' bench/raw/B/matrix.log | grep -q "^ping rep1_resume:" \
  && echo "PASS resume ping labelled rep1_resume (mid-rep)" || { echo "FAIL resume ping label"; ok=0; }
cd "$SRC"; rm -rf "$T"
[ $ok -eq 1 ] && echo "MATRIX_RESUME_TEST PASS" || { echo "MATRIX_RESUME_TEST FAIL"; exit 1; }
