# Builds results.csv (one row per timed run) and RESULTS.md (generated tables) from
# bench/raw/BATCH only. Every number is computed here from a raw file named in the row's src_*
# columns; nothing is typed in by hand.
# Usage (WSL2, repo root; PY = /mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe):
#   $PY bench/summarize.py BATCH [--out-dir bench]   writes OUT/results.csv, OUT/RESULTS.md
#                                                     (refuses to overwrite either)
#   $PY bench/summarize.py BATCH --verify [--out-dir bench]
#       regenerates into a temp dir and requires OUT/results.csv and the generated part of
#       OUT/RESULTS.md (everything above "## Findings") to be byte-identical; then every number
#       in the Findings section (signed) must appear as a whole token in the generated tables or
#       results.csv. Integers under 3 characters can only be weakly checked and are listed.
# Percentiles: p50/p95 are ONLY defined over the pooled decode steps of all runs in a cell
# (nearest rank). results.csv carries per-run means, not per-run percentiles (with N=12 a per-run
# p95 would just be the per-run maximum).
# Inputs in bench/raw/BATCH/: schedule.csv, matrix.log, matrix.log.sha256, SHA256SUMS,
#   runs/<id>/{packets,steps}.csv, tokens.txt, run.json; worker_log.txt (verbatim copy of the Mac
#   worker log, approved cat) and worker_log.mac_sha256 (Mac `shasum -a 256` of that log).
# Hard checks (any failure -> exit 1, nothing written):
#   W0 every file in SHA256SUMS matches, and matrix.log matches matrix.log.sha256 (stops before
#      parsing anything else);
#   W1 worker_log.txt sha256 == the Mac's;
#   W2 per run, the worker TIMING lines of the run's device_id match packets.csv 1:1 in order on
#      (layer, T, sequence_id, start_pos), and the worker's Received-line count for that device
#      equals its TIMING-line count -> every compute_us comes from the worker log;
#   W3 prompt identity: run.json prompt_file == the schedule's, its prompt_sha256 == the file's
#      sha256 == bench/prompts/SHA256SUMS, and prompt_len == the file's id count; then the
#      WORKER-side pattern matches the mode: cache off -> every packet seq=0 and
#      T == prompt_len + step (full context; an uncached run that silently used the cache shows
#      T=1 or seq!=0 here); cache on -> one nonzero seq, step 0 T=prompt_len start_pos 0,
#      step s>=1 T=1 start_pos prompt_len+s-1;
#   W4 tokens identical between the on and off run of the same (rep, prompt_len);
#   W5 every scheduled run completed exactly once, and the order of matrix.log "done" lines ==
#      schedule.csv order (aborted, renamed attempts are allowed and ignored);
#   W6 anchor_on and anchor_off tokens == data/gen_ref_tokens.txt;
#   W7 the generated Machines/RTT/Profiler sections have their sources: exactly 3 specs_*.txt,
#      >= 1 profile/*.sample.log (profile.sh, after the matrix) and "ping " lines in matrix.log.
import argparse
import csv
import filecmp
import glob
import hashlib
import json
import os
import re
import statistics as stx
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("batch")
ap.add_argument("--out-dir", default="bench")
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()
RAW = f"bench/raw/{a.batch}"
MARK = "## Findings"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def pct(xs, q):   # nearest-rank percentile: smallest x with at least q% of samples <= x
    xs = sorted(xs)
    return xs[max(0, -(-len(xs) * q // 100) - 1)]


def stop(fail):
    print("\n".join(fail)); print("SUMMARIZE_FAIL"); sys.exit(1)


def build(out_dir):
    fail = []
    for ln in open(f"{RAW}/SHA256SUMS"):
        h, name = ln.split(maxsplit=1)
        p = os.path.join(RAW, name.strip())
        if not os.path.isfile(p) or sha(p) != h:
            fail.append(f"W0 {p}: missing or sha256 mismatch")
    if sha(f"{RAW}/matrix.log") != open(f"{RAW}/matrix.log.sha256").read().split()[0]:
        fail.append("W0 matrix.log != matrix.log.sha256")
    log = open(f"{RAW}/worker_log.txt", "rb").read()
    if hashlib.sha256(log).hexdigest() != open(f"{RAW}/worker_log.mac_sha256").read().split()[0]:
        fail.append("W1 worker_log.txt sha256 != Mac shasum")
    if fail:   # raw data not trustworthy: stop before parsing any of it
        stop(fail)
    tim = re.compile(r"^TIMING dev=(\S+) layer=(\d+) T=(-?\d+) seq=(\d+) start_pos=(-?\d+) compute_us=(\d+)$")
    rec = re.compile(r"^Received TensorPacket #\d+ from '([^']*)'")
    wl, nrec = {}, {}
    for ln in log.decode(errors="replace").splitlines():
        m = tim.match(ln)
        if m:
            wl.setdefault(m[1], []).append(tuple(int(v) for v in m.groups()[1:]))
        m = rec.match(ln)
        if m:
            nrec[m[1]] = nrec.get(m[1], 0) + 1
    psums = {l.split()[1]: l.split()[0] for l in open("bench/prompts/SHA256SUMS") if l.strip()}

    allsched = rows(f"{RAW}/schedule.csv")
    done = [l.split()[3] for l in open(f"{RAW}/matrix.log") if re.match(r"^\S+ done  \d+ \S+$", l)]
    if done != [r["run_id"] for r in allsched]:
        fail.append(f"W5 matrix.log done order ({len(done)} runs) != schedule.csv ({len(allsched)} runs)")
    want_tok = "".join(f"{l.split()[1]}\n" for l in open("data/gen_ref_tokens.txt") if l.strip() and not l.startswith("#"))
    for anc in ("anchor_on", "anchor_off"):
        p = f"{RAW}/runs/{anc}/tokens.txt"
        if not os.path.isfile(p) or open(p).read() != want_tok:
            fail.append(f"W6 {anc} tokens != data/gen_ref_tokens.txt")
    sched = [r for r in allsched if r["rep"] != "0"]          # anchors + warmups excluded from metrics
    pos_in_pair = {}                                          # 1 = ran first in its on/off pair
    for i in range(0, len(sched), 2):
        pos_in_pair[sched[i]["run_id"]], pos_in_pair[sched[i + 1]["run_id"]] = 1, 2

    out, tokens, pooled = [], {}, {}
    for s in sched:
        rid, mode, pf = s["run_id"], s["kv_cache"], s["prompt_file"]
        d = f"{RAW}/runs/{rid}"
        if not os.path.isfile(f"{d}/run.json"):
            fail.append(f"W5 missing run {rid}"); continue
        meta = json.load(open(f"{d}/run.json"))
        L, n = meta["prompt_len"], meta["n_decode"]
        n_ids = sum(1 for l in open(pf) if l.strip() and not l.startswith("#"))
        if (meta["prompt_file"] != pf or meta["prompt_sha256"] != sha(pf)
                or psums.get(os.path.basename(pf)) != sha(pf) or n_ids != L or str(n) != s["n_decode"]):
            fail.append(f"W3 {rid}: prompt/n_decode identity mismatch (run.json vs schedule vs prompts/SHA256SUMS)"); continue
        pk, st = rows(f"{d}/packets.csv"), rows(f"{d}/steps.csv")
        ntok = sum(1 for l in open(f"{d}/tokens.txt") if l.strip())
        if len(st) != n + 1 or ntok != n + 1 or [int(x["step"]) for x in st] != list(range(n + 1)):
            fail.append(f"W3 {rid}: steps.csv {len(st)} rows / tokens {ntok}, want {n + 1} each"); continue
        w = wl.get(meta["device_id"], [])
        key = [(int(p["layer"]), int(p["T"]), int(p["sequence_id"]), int(p["start_pos"])) for p in pk]
        if [t[:4] for t in w] != key or nrec.get(meta["device_id"], 0) != len(w):
            fail.append(f"W2 {rid}: worker TIMING {len(w)} / Received {nrec.get(meta['device_id'], 0)} "
                        f"vs packets.csv {len(pk)} (or tuples differ)"); continue
        if mode == "off":
            want = [(b, L + k, 0, 0) for k in range(n + 1) for b in range(12)]
        else:
            sq = int(meta["sequence_id"])
            want = [(b, L if k == 0 else 1, sq, 0 if k == 0 else L + k - 1) for k in range(n + 1) for b in range(12)]
            if sq == 0:
                fail.append(f"W3 {rid}: cache on with sequence_id 0")
        if [t[:4] for t in w] != want:
            fail.append(f"W3 {rid}: worker T/seq/start_pos pattern wrong for cache {mode}"); continue
        comp = [t[4] for t in w]
        stw = [int(x["step_wall_us"]) for x in st]
        wst = [sum(comp[12 * i:12 * i + 12]) for i in range(n + 1)]
        nst = [x - y for x, y in zip(stw, wst)]
        dec, dw, dn = stw[1:], wst[1:], nst[1:]
        cell = pooled.setdefault((L, mode), ([], [], []))
        cell[0].extend(dec); cell[1].extend(dw); cell[2].extend(dn)
        tokens[(s["rep"], L, mode)] = open(f"{d}/tokens.txt").read()
        out.append(dict(
            run_id=rid, order=s["order"], rep=s["rep"], pair_position=pos_in_pair[rid],
            prompt_len=L, kv_cache=mode, n_decode=n,
            prefill_s=stw[0] / 1e6, prefill_worker_s=wst[0] / 1e6, prefill_net_client_s=nst[0] / 1e6,
            decode_mean_ms=stx.mean(dec) / 1e3, decode_worker_mean_ms=stx.mean(dw) / 1e3,
            decode_net_client_mean_ms=stx.mean(dn) / 1e3,
            decode_tok_per_s=n / (sum(dec) / 1e6), decode_tok_per_s_worker_only=n / (sum(dw) / 1e6),
            decode_worker_s_total=sum(dw) / 1e6, decode_net_client_s_total=sum(dn) / 1e6,
            req_bytes_total=sum(int(p["req_bytes"]) for p in pk),
            reply_bytes_total=sum(int(p["reply_bytes"]) for p in pk),
            src_steps=f"{d}/steps.csv", src_packets=f"{d}/packets.csv",
            src_worker=f"{RAW}/worker_log.txt#dev={meta['device_id']}"))
    for r in out:
        other = tokens.get((r["rep"], r["prompt_len"], "off" if r["kv_cache"] == "on" else "on"))
        r["tokens_match_pair"] = other == tokens[(r["rep"], r["prompt_len"], r["kv_cache"])]
        if other is None:
            fail.append(f"W4 {r['run_id']}: no valid on/off partner (partner failed an earlier check)")
        elif not r["tokens_match_pair"]:
            fail.append(f"W4 {r['run_id']}: tokens differ from its on/off pair (check top-2 margin first)")
    if fail:
        stop(fail)
    specs = sorted(glob.glob(f"{RAW}/specs_*.txt"))
    profs = sorted(glob.glob(f"{RAW}/profile/*.sample.log"))
    pings = [l.strip() for l in open(f"{RAW}/matrix.log") if l.startswith("ping ")]
    if len(specs) != 3 or not profs or not pings:
        stop([f"W7 need 3 specs_*.txt, >=1 profile/*.sample.log and ping lines in matrix.log "
              f"(have {len(specs)}, {len(profs)}, {len(pings)})"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        wr.writeheader(); wr.writerows(out)
    g3 = lambda x: f"{x:.3g}"
    md = [f"# Hydra KV benchmark — batch {a.batch}", "",
          "Generated by bench/summarize.py; do not edit above the Findings heading "
          "(`summarize.py BATCH --verify` regenerates and diffs it).", "",
          "## Per-run values (median over the cell's runs; [min-max])", "",
          "| prompt | cache | runs | prefill s | prefill worker s | decode mean ms | worker ms | net+client ms | tok/s | tok/s worker-only |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    cells = sorted({(r["prompt_len"], r["kv_cache"]) for r in out}, key=lambda c: (c[0], c[1] != "off"))
    for L, mode in cells:
        g = [r for r in out if r["prompt_len"] == L and r["kv_cache"] == mode]
        c = lambda k: f"{g3(stx.median(r[k] for r in g))} [{g3(min(r[k] for r in g))}-{g3(max(r[k] for r in g))}]"
        md.append(f"| {L} | {mode} | {len(g)} | {c('prefill_s')} | {c('prefill_worker_s')} | {c('decode_mean_ms')} | "
                  f"{c('decode_worker_mean_ms')} | {c('decode_net_client_mean_ms')} | {c('decode_tok_per_s')} | "
                  f"{c('decode_tok_per_s_worker_only')} |")
    md += ["", "## Decode latency percentiles (pooled: every decode step of every run in the cell; nearest rank)", "",
           "| prompt | cache | samples | p50 ms | p95 ms | worker p50 ms | worker p95 ms | net+client p50 ms | net+client p95 ms |",
           "|---|---|---|---|---|---|---|---|---|"]
    for L, mode in cells:
        t, w_, n_ = pooled[(L, mode)]
        md.append(f"| {L} | {mode} | {len(t)} | {g3(pct(t, 50) / 1e3)} | {g3(pct(t, 95) / 1e3)} | "
                  f"{g3(pct(w_, 50) / 1e3)} | {g3(pct(w_, 95) / 1e3)} | {g3(pct(n_, 50) / 1e3)} | {g3(pct(n_, 95) / 1e3)} |")
    md += ["", "## Order/drift check (decode mean ms when the run went first vs second in its on/off pair)", "",
           "| prompt | cache | first: n, median | second: n, median |", "|---|---|---|---|"]
    for L, mode in cells:
        f1 = [r["decode_mean_ms"] for r in out if (r["prompt_len"], r["kv_cache"], r["pair_position"]) == (L, mode, 1)]
        f2 = [r["decode_mean_ms"] for r in out if (r["prompt_len"], r["kv_cache"], r["pair_position"]) == (L, mode, 2)]
        md.append(f"| {L} | {mode} | {len(f1)}, {g3(stx.median(f1)) if f1 else '-'} | {len(f2)}, {g3(stx.median(f2)) if f2 else '-'} |")
    # Speedup = off/on pooled mean step time (= on/off tokens/s). Saved time splits exactly into
    # worker + net+client, since net+client is defined as step wall - worker.
    md += ["", "## Decode speedup, cache on vs off (pooled means over every decode step in the cell)", "",
           "| prompt | off ms/token | on ms/token | speedup x | saved ms/token | saved in worker compute ms | % | saved in net+client ms | % |",
           "|---|---|---|---|---|---|---|---|---|"]
    mm = lambda xs: stx.mean(xs) / 1e3
    for L in sorted({c[0] for c in cells}):
        off, on = pooled[(L, "off")], pooled[(L, "on")]
        sv, sw, sn = mm(off[0]) - mm(on[0]), mm(off[1]) - mm(on[1]), mm(off[2]) - mm(on[2])
        md.append(f"| {L} | {g3(mm(off[0]))} | {g3(mm(on[0]))} | {g3(mm(off[0]) / mm(on[0]))} | {g3(sv)} | "
                  f"{g3(sw)} | {g3(100 * sw / sv)} | {g3(sn)} | {g3(100 * sn / sv)} |")
    avgs = [float(l.split("=")[1].split("/")[1]) for l in pings]
    md += ["", f"## RTT WSL2 -> Mac (`ping -c 20` before/after every rep; verbatim from matrix.log)", "", "```"] + pings + \
          ["```", "", f"Median of the {len(avgs)} per-ping-run avg RTTs: {g3(stx.median(avgs))} ms "
                      f"(range {g3(min(avgs))}-{g3(max(avgs))} ms).",
           "", "## Machines (verbatim from specs_*.txt, blank lines dropped)", ""]
    for p in specs:
        md += [f"`{os.path.basename(p)}`:", "", "```"] + \
              [l.rstrip() for l in open(p, encoding="utf-8", errors="replace") if l.strip()] + ["```", ""]
    md += ["## Profiler: top 5 by self time (macOS `sample`, 10 s at 1 ms; separate worker after the matrix, "
           "never timed)", ""]
    for p in profs:
        sec, on_ = [], False
        for l in open(p, encoding="utf-8", errors="replace"):
            if l.startswith("Sort by top of stack"):
                on_ = True
            elif on_ and not l.strip():
                break
            elif on_:
                sec.append(l.strip().rsplit(None, 1))
        tot = sum(int(c) for _, c in sec)
        md += [f"`profile/{os.path.basename(p)}`: share of the {tot} samples listed in its top-of-stack "
               "section (entries with >= 5 samples)", "", "| function | samples | share % |", "|---|---|---|"]
        md += [f"| `{fn}` | {c} | {g3(100 * int(c) / tot)} |" for fn, c in sec[:5]] + [""]
    md += ["", MARK, "", "(hand-written after review; every number must appear in the tables above "
           "or in results.csv — enforced by --verify)", ""]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", newline="\n", encoding="utf-8") as f:   # not cp1252
        f.write("\n".join(md))
    return out_dir


csv_p, md_p = os.path.join(a.out_dir, "results.csv"), os.path.join(a.out_dir, "RESULTS.md")
if a.verify:
    tmp = build(tempfile.mkdtemp(prefix="hydra_bench_verify_"))
    ok = filecmp.cmp(f"{tmp}/results.csv", csv_p, shallow=False)
    gen_new = open(f"{tmp}/RESULTS.md", encoding="utf-8").read().split(MARK)[0]
    have = open(md_p, encoding="utf-8").read()
    ok &= have.split(MARK)[0] == gen_new
    findings = have.split(MARK, 1)[1] if MARK in have else ""
    num = r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    known = set(re.findall(num, gen_new + open(csv_p).read()))
    found = re.findall(num, findings)
    bad = [x for x in found if x not in known]
    weak = sorted({x for x in found if x in known and "." not in x and len(x.lstrip("-")) < 3})
    print(f"regenerated tables/csv identical: {ok}; untraceable numbers in Findings: {bad or 'none'}; "
          f"weakly checked small integers: {weak or 'none'}")
    sys.exit(0 if ok and not bad else 1)
for p in (csv_p, md_p):
    if os.path.exists(p):
        sys.exit(f"ERROR: {p} exists; refusing to overwrite")
print(f"Wrote {build(a.out_dir)}/results.csv and RESULTS.md")
