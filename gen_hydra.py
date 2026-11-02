#!/usr/bin/env python3
"""
gen_hydra.py  --  G4: greedy GPT-2 generation through the Hydra Mac worker.

Usage (from repo root, Windows Python; the Mac worker runs the G3/KV build in --mode block):
  python gen_hydra.py --out DIR [--host 10.0.0.62] [--port 50051] [--timeout 10]
                      [--kv-cache on|off] [--n-gen 20] [--prompt TEXT]

--kv-cache off (default): exactly the G4 client below (no sequence_id; same wire bytes,
  arithmetic and outputs). --kv-cache on: one random nonzero sequence_id per run; step 0
  prefills the whole prompt (start_pos 0), every later step sends ONE row,
  wte[last token] + wpe[p] at start_pos p = T_prompt + step - 1; the worker attends over its
  cache. The head runs on the newest row only, and hidden_f32.bin holds only the rows the
  worker returned (T_prompt prefill rows, then one row per later step).

Writes, only after all N_GEN steps succeed (nothing on any failure):
  DIR/tokens.txt     -- one row per step, same row format as data/gen_ref_tokens.txt
                        ("%4d  %7d  %r", gen_gpt2_ref.py), LF line endings
  DIR/logits_f64.bin -- raw float64 [N_GEN, 50257] row-major, last-position logits
  DIR/hidden_f32.bin -- raw float32, every step's FULL block-11 output (all T rows,
                        the input to ln_f), steps concatenated: 410 x 768 for the
                        fixed prompt (T = 11..30); compared byte-for-byte with
                        data/g5_mirror_hidden.bin by scripts/g5_compare_hidden.py
  DIR/sends.txt      -- one line per packet sent: "step layer sequence_id start_pos T"
Refuses to run if any output exists.

Per generation step (no KV cache; the full context is resent every step):
  1. ids = prompt + tokens generated so far;  x = (wte(ids) + wpe(0..T-1)) in float32,
     [T, 768] -- the same expression as gen_gpt2_ref.py forward_double().
  2. ONE TCP connection; for N in 0..11: send TensorPacket(layer_index=N, data=x,
     shape=[T, 768]); receive the reply; validate it; x = reply. Block N+1's input is
     block N's output, so the 12 round trips are strictly sequential.
  3. logits = head_f64(x)[-1]: float64 ln_f + LM head (precision convention, CLAUDE.md),
     op-for-op as gen_gpt2_ref.py head_f64().
  4. next token = argmax(logits) (greedy).

Wire format (must match orchestrator.cpp / worker.cpp):
  [4-byte unsigned length, network byte order (big-endian)] [serialized TensorPacket]
  - orchestrator.cpp:38-43 sendPacket(): htonl(uint32 size), then the payload.
  - orchestrator.cpp:74-86 recvPacket(): 4-byte prefix, ntohl, then exactly msg_len bytes.
  - worker.cpp: one connection may carry many packets (the per-connection `while (true)`
    loop; EOF at a message boundary ends it). Lengths > MAX_MSG_LEN (4 MiB) close the
    connection.
  Receive timeout: socket timeout (default 10 s, --timeout) on connect and on every recv,
  the analogue of orchestrator.cpp's SO_RCVTIMEO (RECV_TIMEOUT_SEC = 5, :195, :273-280).
  A timeout raises; nothing is written.
TCP_NODELAY is set (Engineering Standards).
"""

import argparse
import os
import secrets
import socket
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

try:
    import hydra_pb2
except ImportError as e:
    sys.exit(f"ERROR: cannot import hydra_pb2 ({e}). Generate it on WSL2 from the repo root:\n"
             "  protoc --python_out=. hydra.proto")

PROMPT = "Hello, how are you doing today? I hope everything"   # fixed for G1-G5
N_GEN = 20
HIDDEN = 768
N_BLOCKS = 12
DEVICE_ID = "gen-hydra-py"
MAX_MSG_LEN = 4 << 20   # worker.cpp MAX_MSG_LEN; a larger reply is a framing error


def send_packet(sock, pkt):
    payload = pkt.SerializeToString()
    if len(payload) > MAX_MSG_LEN:
        raise ValueError(f"packet {len(payload)} bytes exceeds MAX_MSG_LEN {MAX_MSG_LEN}")
    # sendall loops over partial sends; PEP 475 retries EINTR.
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))          # socket.timeout propagates
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


def recv_packet(sock):
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    if n > MAX_MSG_LEN:
        raise ValueError(f"reply length prefix {n} exceeds MAX_MSG_LEN {MAX_MSG_LEN}")
    pkt = hydra_pb2.TensorPacket()
    pkt.ParseFromString(recv_exact(sock, n))
    return pkt


def run_blocks(x, host, port, timeout, sends, seq=0, start_pos=0):
    """Send x [T, 768] float32 through blocks 0..11 on one connection; return the output.
    seq != 0: KV-cached sequence, x holds positions start_pos.. . Appends one
    (layer, sequence_id, start_pos, T) tuple per packet sent to `sends`."""
    T = x.shape[0]
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for n in range(N_BLOCKS):
            pkt = hydra_pb2.TensorPacket(layer_index=n, device_id=DEVICE_ID)
            if seq:   # unset when 0, so the cache-off wire bytes are unchanged
                pkt.sequence_id = seq
                pkt.start_pos = start_pos
            pkt.data.extend(x.reshape(-1).tolist())    # float32 -> Python float: exact
            pkt.shape.extend([T, HIDDEN])
            send_packet(s, pkt)
            sends.append((pkt.layer_index, pkt.sequence_id, pkt.start_pos, pkt.shape[0]))
            r = recv_packet(s)
            if len(r.shape) == 0:
                raise RuntimeError(f"worker rejected layer_index {r.layer_index}")
            if (r.layer_index != n or r.device_id != "worker-mac"
                    or list(r.shape) != [T, HIDDEN] or len(r.data) != T * HIDDEN):
                raise RuntimeError(f"bad reply for block {n}: layer_index={r.layer_index} "
                                   f"device_id={r.device_id!r} shape={list(r.shape)} "
                                   f"data={len(r.data)}")
            x = torch.from_numpy(np.array(r.data, dtype=np.float32)).view(T, HIDDEN)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="10.0.0.62")   # Mac worker, as in orchestrator.cpp:255
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--kv-cache", choices=["on", "off"], default="off")
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    cache = a.kv_cache == "on"

    tok_path = os.path.join(a.out, "tokens.txt")
    log_path = os.path.join(a.out, "logits_f64.bin")
    hid_path = os.path.join(a.out, "hidden_f32.bin")
    snd_path = os.path.join(a.out, "sends.txt")
    for p in (tok_path, log_path, hid_path, snd_path):
        if os.path.exists(p):
            sys.exit(f"ERROR: {p} exists; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()

    def head_f64(x):   # gen_gpt2_ref.py head_f64(), same ops
        x = F.layer_norm(x.double(), [HIDDEN],
                         weight=model.ln_f.weight.double(), bias=model.ln_f.bias.double())
        return x @ model.wte.weight.double().T

    prompt_ids = tokenizer.encode(a.prompt)
    seq = (secrets.randbits(63) | 1) if cache else 0   # nonzero; fresh per run
    tokens, rows, hidden, sends = [], [], [], []
    with torch.no_grad():
        for step in range(a.n_gen):
            if cache and step > 0:   # decode: only the newest token, at its absolute position
                start = len(prompt_ids) + step - 1
                ids = torch.tensor([[tokens[-1]]], dtype=torch.long)
                pos = torch.tensor([[start]], dtype=torch.long)
            else:                    # cache off, or the cache-on prefill: the whole context
                start = 0
                ids = torch.tensor([prompt_ids + tokens], dtype=torch.long)
                pos = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)
            T = ids.shape[1]
            x = (model.wte(ids) + model.wpe(pos)).squeeze(0).float()
            n0 = len(sends)
            x = run_blocks(x, a.host, a.port, a.timeout, sends, seq, start)
            sends[n0:] = [(step,) + t for t in sends[n0:]]
            hidden.append(x.numpy().astype(np.float32))   # block-11 output, [T, 768]
            # cache on: head on the newest row only (owner decision; bounded in kv_compare.py)
            logits = head_f64(x[-1:] if cache else x)[-1]   # [50257] float64
            tid = int(logits.argmax())
            tokens.append(tid)
            rows.append(logits)
            print(f"step {step:2d}  T={T:2d}  token={tid:5d} {tokenizer.decode([tid])!r}")

    torch.stack(rows).numpy().astype(np.float64).tofile(log_path)
    np.concatenate(hidden, axis=0).astype(np.float32).tofile(hid_path)
    with open(tok_path, "w", newline="\n") as f:
        f.write("# Hydra greedy tokens -- gen_hydra.py (blocks on the Mac worker)\n")
        f.write(f"# PROMPT = {a.prompt!r}\n# N_GEN  = {a.n_gen}\n# step  token_id  decoded\n")
        for step, tid in enumerate(tokens):
            f.write(f"{step:4d}  {tid:7d}  {repr(tokenizer.decode([tid]))}\n")
    with open(snd_path, "w", newline="\n") as f:
        f.write("# step layer sequence_id start_pos T\n")
        f.writelines(" ".join(map(str, t)) + "\n" for t in sends)
    print(f"Wrote {tok_path}, {log_path}, {hid_path} and {snd_path}")


if __name__ == "__main__":
    main()
