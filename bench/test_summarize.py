# Offline test of bench/summarize.py on fabricated raw batches (no worker, no network).
# Positive cases must pass; every negative case must fail WITH THE EXPECTED CHECK (W0..W7).
# Usage: /mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe bench/test_summarize.py
import hashlib, json, os, shutil, subprocess, sys, tempfile

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarize.py")
sha = lambda b: hashlib.sha256(b).hexdigest()
GEN_REF = [464, 318, 257, 1332]                  # fake gen_ref token ids for the anchors
N = 3


def make(root, tamper=None):
    raw = os.path.join(root, "bench", "raw", "B")
    os.makedirs(os.path.join(raw, "runs"))
    os.makedirs(os.path.join(root, "bench", "prompts"))
    os.makedirs(os.path.join(root, "data"))
    shutil.copy(SRC, os.path.join(root, "bench", "summarize.py"))
    with open(os.path.join(root, "data", "gen_ref_tokens.txt"), "w", newline="\n") as f:
        f.write("# step  token_id  decoded\n" + "".join(f"{i:4d}  {t:7d}  'x'\n" for i, t in enumerate(GEN_REF)))
    prompts = {"prompt_g5.txt": [1, 2], "prompt_L4.txt": [5, 6, 7, 8], "prompt_L6.txt": [5, 6, 7, 8, 9, 10]}
    sums = ""
    for name, ids in prompts.items():
        body = ("# test\n" + "".join(f"{i}\n" for i in ids)).encode()
        open(os.path.join(root, "bench", "prompts", name), "wb").write(body)
        sums += f"{sha(body)}  {name}\n"
    open(os.path.join(root, "bench", "prompts", "SHA256SUMS"), "w", newline="\n").write(sums)
    sched = [("anchor_on", "0", "on", "bench/prompts/prompt_g5.txt", len(GEN_REF) - 1),
             ("anchor_off", "0", "off", "bench/prompts/prompt_g5.txt", len(GEN_REF) - 1)]
    for r in (1, 2):
        for L in (4, 6):
            first = "on" if (r + L) % 4 == 0 else "off"
            for m in (first, "off" if first == "on" else "on"):
                sched.append((f"r{r}_L{L}_{m}", str(r), m, f"bench/prompts/prompt_L{L}.txt", N))
    wlog, mlog = ["Worker listening on port 50051..."], ["2026-11-02T00:00:00Z BEGIN",
                  "ping rep0_before: rtt min/avg/max/mdev = 6.021/59.822/199.312/43.364 ms"]
    for host in ("mac", "windows", "wsl"):
        open(os.path.join(raw, f"specs_{host}_20261102T000000Z.txt"), "w", newline="\n").write(f"\n{host} cpu: test\n")
    with open(os.path.join(raw, "schedule.csv"), "w", newline="\n") as f:
        f.write("order,run_id,rep,prompt_file,kv_cache,n_decode\n")
        f.writelines(f"{i},{rid},{rep},{pf},{m},{nd}\n" for i, (rid, rep, m, pf, nd) in enumerate(sched, 1))
    for i, (rid, rep, m, pf, nd) in enumerate(sched, 1):
        L = len(prompts[os.path.basename(pf)])
        seq = 1000 + i if m == "on" else 0
        dev = f"bench-{rid}-20261102T000000Z-{i}"
        if tamper == "aborted_retry" and rid == "r1_L6_on":   # an earlier aborted attempt: other dev id
            mlog.append(f"2026-11-02T00:00:00Z start {i} {rid}")
            for b in range(5):
                wlog.append(f"Received TensorPacket #{b+1} from '{dev}-old' (layer {b}), 768 floats, shape [{L},768] seq=77 start_pos=0")
                wlog.append(f"TIMING dev={dev}-old layer={b} T={L} seq=77 start_pos=0 compute_us=5")
            os.makedirs(os.path.join(raw, "runs", rid + ".aborted_20261102T000000Z"))
        mlog.append(f"2026-11-02T00:00:00Z start {i} {rid}")
        d = os.path.join(raw, "runs", rid); os.makedirs(d)
        pk, st = [], []
        for s in range(nd + 1):
            T, sp = (L + s, 0) if m == "off" else ((L, 0) if s == 0 else (1, L + s - 1))
            if tamper in ("uncached_uses_cache", "uncached_consistent") and rid == "r1_L4_off" and s > 0:
                T, sp = 1, L + s - 1
            for b in range(12):
                pk.append((s, b, seq, sp, T, 100 * T, 100 * T, 2000 + T))
                if tamper == "drop_timing" and rid == "r2_L6_on" and s == 2 and b == 5:
                    continue
                wlog.append(f"Received TensorPacket #{b+1} from '{dev}' (layer {b}), {T*768} floats, shape [{T},768] seq={seq} start_pos={sp}")
                wlog.append(f"TIMING dev={dev} layer={b} T={T} seq={seq} start_pos={sp} compute_us={1000 + 10 * T}")
            st.append((s, T, 7 + s, 30000 + 1000 * T + 17 * s))
        if tamper == "uncached_uses_cache" and rid == "r1_L4_off":   # client record looks uncached
            pk = [(s_, b_, sq, 0, 4 + s_, x, y, z) for (s_, b_, sq, sp_, T_, x, y, z) in pk]
        with open(os.path.join(d, "packets.csv"), "w", newline="\n") as f:
            f.write("step,layer,sequence_id,start_pos,T,req_bytes,reply_bytes,client_wall_us\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in pk)
        with open(os.path.join(d, "steps.csv"), "w", newline="\n") as f:
            f.write("step,T_sent,token,step_wall_us\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in st)
        toks = "".join(f"{t}\n" for t in GEN_REF) if rep == "0" else "".join(f"{7+s}\n" for s in range(nd + 1))
        if tamper == "token_mismatch" and rid == "r2_L4_on":
            toks = toks.replace("9\n", "99\n")
        if tamper == "anchor_wrong" and rid == "anchor_on":
            toks = toks.replace("318\n", "319\n")
        open(os.path.join(d, "tokens.txt"), "w", newline="\n").write(toks)
        pbytes = open(os.path.join(root, pf), "rb").read()
        meta = dict(run_id=rid, device_id=dev, kv_cache=m, prompt_len=L, n_decode=nd, sequence_id=seq,
                    prompt_file=pf, prompt_sha256=sha(pbytes))
        if tamper == "wrong_prompt" and rid == "r2_L6_off":
            meta["prompt_file"] = "bench/prompts/prompt_L4.txt"
        json.dump(meta, open(os.path.join(d, "run.json"), "w"))
        mlog.append(f"2026-11-02T00:00:01Z done  {i} {rid}")
    if tamper == "order":
        a_, b_ = [k for k, l in enumerate(mlog) if " done  " in l][4:6]
        mlog[a_], mlog[b_] = mlog[b_], mlog[a_]
    mlog.append("BENCH_MATRIX_DONE")
    mtext = ("\n".join(mlog) + "\n").encode()
    open(os.path.join(raw, "matrix.log"), "wb").write(mtext)
    open(os.path.join(raw, "matrix.log.sha256"), "w").write(sha(mtext) + "\n")
    files = sorted(os.path.relpath(os.path.join(p, f), raw).replace("\\", "/") for p, _, fs in os.walk(raw)
                   for f in fs if f not in ("matrix.log", "matrix.log.sha256"))
    with open(os.path.join(raw, "SHA256SUMS"), "w", newline="\n") as f:
        f.writelines(f"{sha(open(os.path.join(raw, rel), 'rb').read())}  ./{rel}\n" for rel in files)
    if tamper == "edit_raw":
        p = os.path.join(raw, "runs", "r1_L6_on", "steps.csv")
        open(p, "w").write(open(p).read().replace("3", "2", 1))
    if tamper == "edit_matrix_log":
        open(os.path.join(raw, "matrix.log"), "ab").write(b"x\n")
    wl = ("\n".join(wlog) + "\n").encode()
    open(os.path.join(raw, "worker_log.txt"), "wb").write(wl)
    open(os.path.join(raw, "worker_log.mac_sha256"), "w").write(sha(wl) + "  worker_log.txt\n")
    if tamper != "no_profile":   # written after SHA256SUMS, as profile.sh does after the matrix
        os.makedirs(os.path.join(raw, "profile"))
        open(os.path.join(raw, "profile", "off4.sample.log"), "w", newline="\n").write(
            "Call graph:\n    1 main\n\nSort by top of stack, same collapsed (when >= 5):\n"
            "        matmul_nt(float const*)  (in worker)        30\n        __recvfrom  (in libsystem_kernel.dylib)        10\n\n"
            "Binary Images:\n")


def run(root, *args):
    p = subprocess.run([sys.executable, "bench/summarize.py", "B", *args], cwd=root, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()


ok = True
def check(label, cond, out):
    global ok
    ok &= cond
    print(f"{'PASS' if cond else 'FAIL'} {label}: {out[-1] if out else ''}")

root = tempfile.mkdtemp(); make(root)
rc, out = run(root); check("positive batch", rc == 0, out)
rc, out = run(root, "--verify"); check("--verify clean", rc == 0, out)
md = os.path.join(root, "bench", "RESULTS.md")
open(md, "a", encoding="utf-8").write("Cached decode was 123.456 ms faster, and -36 is not real.\n")
rc, out = run(root, "--verify"); check("--verify rejects invented numbers", rc == 1 and "123.456" in out[-1] and "-36" in out[-1], out)
rc, out = run(root); check("refuses to overwrite", rc == 1 and "refusing to overwrite" in out[-1], out)
r2 = tempfile.mkdtemp(); make(r2, "aborted_retry")
rc, out = run(r2); check("aborted attempt (renamed dir + other dev id) tolerated", rc == 0, out)
expect = {"uncached_uses_cache": "W2", "uncached_consistent": "W3", "drop_timing": "W2", "token_mismatch": "W4",
          "order": "W5", "edit_raw": "W0", "edit_matrix_log": "W0", "anchor_wrong": "W6", "wrong_prompt": "W3",
          "no_profile": "W7"}
for t, code in expect.items():
    r2 = tempfile.mkdtemp(); make(r2, t)
    rc, out = run(r2)
    first = next((l for l in out if l[:1] == "W"), "")
    check(f"negative {t} -> {code}", rc == 1 and any(l.startswith(code) for l in out), [first])
print("TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
