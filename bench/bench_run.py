# One timed greedy run against the Mac block worker.
# Same generation semantics as gen_hydra.py (embeddings, cache on/off packets, float64 head;
# cache on = head on the newest row), plus timing. Writes, only after the whole run succeeds:
#   OUT/packets.csv  step,layer,sequence_id,start_pos,T,req_bytes,reply_bytes,client_wall_us
#                    client_wall_us = build+serialize+send+recv+parse of that packet
#   OUT/steps.csv    step,T_sent,token,step_wall_us   (step_wall = embed + 12 packets incl.
#                    connect + head + argmax; step 0 = prefill, steps >= 1 = decode tokens)
#   OUT/tokens.txt   one token id per line
#   OUT/run.json     run metadata (device_id identifies this run's lines in the worker log)
# Refuses if OUT exists. Prints "STEP <s> START" (flushed) for the profiler wrapper.
# Usage (WSL2, repo root; normally called by bench_matrix.sh / profile.sh):
#   /mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe bench/bench_run.py \
#     --prompt-file bench/prompts/prompt_L512.txt --kv-cache on --n-decode 12 --run-id X --out DIR
import argparse
import hashlib
import json
import os
import secrets
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from transformers import GPT2Model                                      # noqa: E402
from gen_hydra import send_packet, recv_packet, hydra_pb2, HIDDEN, N_BLOCKS   # noqa: E402

CTX = 1024
ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--kv-cache", choices=["on", "off"], required=True)
ap.add_argument("--n-decode", type=int, required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--host", default="10.0.0.62")
ap.add_argument("--port", type=int, default=50051)
ap.add_argument("--timeout", type=float, default=300.0)   # an uncached T~1024 packet takes ~6 s
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f"ERROR: {a.out} exists; refusing to overwrite")
cache = a.kv_cache == "on"
raw = open(a.prompt_file, "rb").read()
prompt = [int(l) for l in raw.decode().splitlines() if l.strip() and not l.startswith("#")]
L = len(prompt)
if L + a.n_decode > CTX:
    sys.exit(f"ERROR: prompt {L} + {a.n_decode} decode tokens > {CTX} positions")

model = GPT2Model.from_pretrained("gpt2").eval()
head = lambda x: F.layer_norm(x.double(), [HIDDEN], weight=model.ln_f.weight.double(),
                              bias=model.ln_f.bias.double()) @ model.wte.weight.double().T
dev = f"bench-{a.run_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"   # unique per attempt
seq = (secrets.randbits(63) | 1) if cache else 0
pk, st, toks = [], [], []
t_run = time.time()
with torch.no_grad():
    for step in range(a.n_decode + 1):
        print(f"STEP {step} START", flush=True)
        s0 = time.perf_counter_ns()
        if cache and step > 0:
            start = L + step - 1
            ids, pos = torch.tensor([[toks[-1]]]), torch.tensor([[start]])
        else:
            start = 0
            ids = torch.tensor([prompt + toks])
            pos = torch.arange(ids.shape[1]).unsqueeze(0)
        x = (model.wte(ids) + model.wpe(pos)).squeeze(0).float()
        T = x.shape[0]
        with socket.create_connection((a.host, a.port), timeout=a.timeout) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for n in range(N_BLOCKS):
                t0 = time.perf_counter_ns()
                pkt = hydra_pb2.TensorPacket(layer_index=n, device_id=dev)
                if seq:
                    pkt.sequence_id, pkt.start_pos = seq, start
                pkt.data.extend(x.reshape(-1).tolist())
                pkt.shape.extend([T, HIDDEN])
                send_packet(s, pkt)
                r = recv_packet(s)
                if list(r.shape) != [T, HIDDEN] or r.layer_index != n or len(r.data) != T * HIDDEN:
                    sys.exit(f"ERROR step {step} block {n}: bad/rejected reply shape={list(r.shape)}")
                x = torch.from_numpy(np.array(r.data, dtype=np.float32)).view(T, HIDDEN)
                t1 = time.perf_counter_ns()
                pk.append((step, n, seq, start, T, 4 + pkt.ByteSize(), 4 + r.ByteSize(), (t1 - t0) // 1000))
        tid = int(head(x[-1:] if cache else x)[-1].argmax())
        toks.append(tid)
        st.append((step, T, tid, (time.perf_counter_ns() - s0) // 1000))

os.makedirs(a.out)
with open(os.path.join(a.out, "packets.csv"), "w", newline="\n") as f:
    f.write("step,layer,sequence_id,start_pos,T,req_bytes,reply_bytes,client_wall_us\n")
    f.writelines(",".join(map(str, r)) + "\n" for r in pk)
with open(os.path.join(a.out, "steps.csv"), "w", newline="\n") as f:
    f.write("step,T_sent,token,step_wall_us\n")
    f.writelines(",".join(map(str, r)) + "\n" for r in st)
with open(os.path.join(a.out, "tokens.txt"), "w", newline="\n") as f:
    f.writelines(f"{t}\n" for t in toks)
with open(os.path.join(a.out, "run.json"), "w", newline="\n") as f:
    json.dump(dict(run_id=a.run_id, device_id=dev, kv_cache=a.kv_cache, prompt_len=L,
                   n_decode=a.n_decode, sequence_id=seq, prompt_file=a.prompt_file,
                   prompt_sha256=hashlib.sha256(raw).hexdigest(), start_unix=t_run,
                   end_unix=time.time(), python=sys.version.split()[0]), f, indent=1)
print(f"Wrote {a.out}")
