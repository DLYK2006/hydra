# Cost probe (planning only, not committed). One uncached 12-block step at T=960 (seq 0),
# then one cached prefill at T=960 (fresh nonzero seq). Per packet: request/reply bytes on
# the wire (4-byte prefix included) and client wall time send-start -> reply parsed.
import os, secrets, socket, sys, time
sys.path.insert(0, r"C:\Users\ASUS\Documents\hydra")
import numpy as np
from gen_hydra import send_packet, recv_packet, hydra_pb2

HOST, PORT, H, T = "10.0.0.62", 50051, 768, 960
x0 = np.random.default_rng(0).standard_normal((T, H)).astype(np.float32)


def step(label, seq):
    x = x0
    total = time.perf_counter()
    with socket.create_connection((HOST, PORT), timeout=1800) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for n in range(12):
            pkt = hydra_pb2.TensorPacket(layer_index=n, device_id="cost-probe")
            if seq:
                pkt.sequence_id, pkt.start_pos = seq, 0
            pkt.data.extend(x.reshape(-1).tolist())
            pkt.shape.extend([T, H])
            req = 4 + pkt.ByteSize()
            t0 = time.perf_counter()
            send_packet(s, pkt)
            r = recv_packet(s)
            dt = time.perf_counter() - t0
            if list(r.shape) != [T, H]:
                raise SystemExit(f"{label} layer {n}: rejected / bad reply shape {list(r.shape)}")
            print(f"{label} layer {n:2d} T={T} seq={seq} req_bytes={req} reply_bytes={4 + r.ByteSize()} wall_s={dt:.3f}", flush=True)
            x = np.array(r.data, dtype=np.float32).reshape(T, H)
    print(f"{label} TOTAL 12 packets wall_s={time.perf_counter() - total:.3f}", flush=True)


step("uncached", 0)
step("cached_prefill", secrets.randbits(63) | 1)
