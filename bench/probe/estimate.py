# Runtime model for bench/PLAN.md. Every input is read from a raw file on disk:
#   cost_probe_20261102T0157Z.log (per-packet wall times, T=960) and ping_20261102T0209Z.txt (RTT).
# One probe point -> [Likely]-grade. LOAD (client startup) is the only assumed constant [Guessing].
import os, re, sys
D = os.path.dirname(os.path.abspath(__file__))
probe = open(os.path.join(D, "cost_probe_20261102T0157Z.log")).read()
ping = open(os.path.join(D, "ping_20261102T0209Z.txt")).read()
tot = dict(re.findall(r"^(\w+) TOTAL 12 packets wall_s=([\d.]+)", probe, re.M))
U, C = float(tot["uncached"]) / 12, float(tot["cached_prefill"]) / 12
F = float(re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", ping)[1]) / 1e3
LOAD, COOLDOWN, ANCHOR_WARMUP = 15.0, 30.0, 4       # s, s, extra runs (anchor pair + warmup pair)
A = 2 * (U - C)
pair, row = A / 960**2, (U - A) / 960
pkt_full = lambda T: F + row * T + pair * T * T
pkt_pref = lambda T: F + row * T + pair * T * (T + 1) / 2
pkt_dec = lambda p: F + row + pair * (p + 1)
print(f"inputs: uncached {tot['uncached']} s, cached prefill {tot['cached_prefill']} s, ping avg {F*1e3:.3f} ms")
print(f"row={row*1e3:.2f} ms/row/block  pair={pair*1e6:.2f} us  A(960)={A:.2f} s")
for N in (8, 12, 16, 24, 32):
    Ls = (32, 128, 512, 1024 - N)
    off = {L: 12 * sum(pkt_full(L + s) for s in range(N + 1)) for L in Ls}
    on = {L: 12 * (pkt_pref(L) + sum(pkt_dec(L + s - 1) for s in range(1, N + 1))) for L in Ls}
    rep = sum(off.values()) + sum(on.values()) + 8 * (LOAD + COOLDOWN)
    small = 4 * (LOAD + COOLDOWN) + 2 * 12 * 30 * pkt_full(30)   # anchor (T<=30) + warmup (L=32), generous
    cells = "  ".join(f"L{L}:off {off[L]/60:.1f}m/on {on[L]/60:.1f}m" for L in Ls)
    print(f"N={N:2d} longest L={1024-N}: rep={rep/60:.1f} min; total 5 reps={(5*rep+small)/3600:.2f} h, "
          f"6 reps={(6*rep+small)/3600:.2f} h | {cells}")
print(f"cached decode per token ~ {12*pkt_dec(512)*1e3:.0f} ms (12 x RTT = {12*F*1e3:.0f} ms); "
      f"uncached step T=512 ~ {12*pkt_full(512):.1f} s, T=1008 ~ {12*pkt_full(1008):.1f} s, T=1024 ~ {12*pkt_full(1024):.1f} s")
