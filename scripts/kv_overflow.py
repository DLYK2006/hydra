# KV-cache protocol + overflow test. Hand-reviewed; do not edit. Called by scripts/kv_check.sh
# (stage E). Talks to the block worker directly, layer 0 only, one connection per request,
# using gen_hydra.py's framing helpers. KV_CAPACITY = 1024 positions.
#   Every rejection below is followed by an accepted, aligned packet on the ALREADY
#   ESTABLISHED sequence D, which proves the rejection changed no cache state (no reset,
#   no adopted sequence, no partial append).
#   E0 establish:                   seq D prefill 4                        -> accepted (len 4)
#   E1 oversize prefill (T check):  new seq, start_pos 0, T=1025            -> rejected;
#                                   then seq D start_pos 4 T=1 -> accepted (len 5)
#   E2 overflow append (bounds):    seq D start_pos 5 T=1020 (5+1020=1025)  -> rejected;
#                                   then seq D start_pos 5 T=1 -> accepted (len 6)
#   E3 alignment (check 5):         seq D replay start_pos 0 -> rejected; gap start_pos 7
#                                   (len 6) -> rejected; aligned start_pos 6 -> accepted
#                                   (len 7)
#   E4 new seq must start at 0:     new seq E at start_pos 5 -> rejected; then seq D
#                                   start_pos 7 T=1 -> accepted (len 8; no reset/adoption);
#                                   then seq E at start_pos 0 -> accepted (reset to E)
#   E5 multi-row append values:     seq F prefill 8 rows in one packet; seq G prefill the
#                                   same rows as 4 + append 4 -> reply rows byte-identical
#                                   (tests the causal limit and K/V placement for T>1 at
#                                   start_pos > 0, not just shapes)
#   E6 fill to capacity:            seq B prefill 1000; append 24 at 1000 (exactly 1024)
#                                   -> accepted; append 1 at 1024 -> rejected
#   E7 worker alive:                new seq prefill 4 -> accepted
# Input rows: fixed-seed float32 normal samples; their values only matter for E5.
import os
import secrets
import socket
import sys

# The repo root is not on sys.path when this file runs as scripts/kv_overflow.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                            # noqa: E402
from gen_hydra import send_packet, recv_packet, hydra_pb2     # noqa: E402

HOST, PORT, H = "10.0.0.62", 50051, 768   # same hardcoded worker as orchestrator.cpp:255
TIMEOUT = 600.0   # a 1000-row prefill through one block takes far longer than a decode step
rng = np.random.default_rng(0)


def rows(T):
    return rng.standard_normal((T, H)).astype(np.float32)


def request(seq, start_pos, x):
    pkt = hydra_pb2.TensorPacket(layer_index=0, device_id="kv-protocol-test",
                                 sequence_id=seq, start_pos=start_pos)
    pkt.data.extend(x.reshape(-1).tolist())
    pkt.shape.extend([x.shape[0], H])
    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_packet(s, pkt)
        return recv_packet(s)


ok = True


def expect(label, reply, accepted, T=None):
    global ok
    got_acc = len(reply.shape) != 0
    good = got_acc == accepted and (not accepted or (list(reply.shape) == [T, H]
                                                      and len(reply.data) == T * H))
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {label}: "
          f"{'accepted ' + str(list(reply.shape)) + f' ({len(reply.data)} floats)' if got_acc else 'rejected'}")
    return reply


new_seq = lambda: secrets.randbits(63) | 1

seq_d = new_seq()
expect("E0 seq D prefill T=4", request(seq_d, 0, rows(4)), True, 4)

expect("E1 new seq prefill T=1025 (> capacity)", request(new_seq(), 0, rows(1025)), False)
expect("E1 seq D start_pos 4 T=1 after rejected new-seq prefill (no reset)", request(seq_d, 4, rows(1)), True, 1)

expect("E2 seq D append start_pos 5 T=1020 (overflow)", request(seq_d, 5, rows(1020)), False)
expect("E2 seq D start_pos 5 T=1 after rejected overflow (no partial append)", request(seq_d, 5, rows(1)), True, 1)

expect("E3 seq D replay start_pos 0", request(seq_d, 0, rows(1)), False)
expect("E3 seq D gap start_pos 7 (len 6)", request(seq_d, 7, rows(1)), False)
expect("E3 seq D aligned start_pos 6", request(seq_d, 6, rows(1)), True, 1)

seq_e = new_seq()
expect("E4 new seq E starting at start_pos 5", request(seq_e, 5, rows(1)), False)
expect("E4 seq D start_pos 7 T=1 after rejected new seq (no reset/adoption)", request(seq_d, 7, rows(1)), True, 1)
expect("E4 seq E at start_pos 0 (reset to E)", request(seq_e, 0, rows(2)), True, 2)

x8 = rows(8)
one = expect("E5 seq F prefill 8 rows in one packet", request(new_seq(), 0, x8), True, 8)
seq_g = new_seq()
a = expect("E5 seq G prefill first 4 rows", request(seq_g, 0, x8[:4]), True, 4)
b = expect("E5 seq G append last 4 rows at start_pos 4", request(seq_g, 4, x8[4:]), True, 4)
same = (np.array(one.data, np.float32).tobytes()
        == np.concatenate([np.array(a.data, np.float32), np.array(b.data, np.float32)]).tobytes())
ok &= same
print(f"{'PASS' if same else 'FAIL'} E5 chunked (4+4) reply rows byte-identical to single 8-row prefill")

seq_b = new_seq()
expect("E6 seq B prefill T=1000", request(seq_b, 0, rows(1000)), True, 1000)
expect("E6 seq B append T=24 at start_pos 1000 (fills 1024)", request(seq_b, 1000, rows(24)), True, 24)
expect("E6 seq B append T=1 at start_pos 1024 (overflow)", request(seq_b, 1024, rows(1)), False)

expect("E7 new seq prefill T=4 (worker alive)", request(new_seq(), 0, rows(4)), True, 4)
print("KV protocol/overflow test:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
