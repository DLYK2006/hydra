# Model estimate for the first timed on/off pair (r1_L32_off then r1_L32_on, N=12), for the
# goal's stop rule (measured vs estimate within +-50%). Same model and inputs as estimate.py:
# cost_probe_20261102T0157Z.log and ping_20261102T0209Z.txt. Compared quantity: the sum of the
# runs' step walls (steps.csv step_wall_us: embed + 12 packets + head, all steps) -- process
# startup (model load, an assumed constant) is excluded so the check tests the probe model only.
# Usage: python bench/probe/first_pair_estimate.py [N_DECODE=12] [L=32]
#        python bench/probe/first_pair_estimate.py 12 32 RUNDIR_OFF RUNDIR_ON   (adds measured + ratio)
import csv, os, re, sys
D = os.path.dirname(os.path.abspath(__file__))
probe = open(os.path.join(D, "cost_probe_20261102T0157Z.log")).read()
ping = open(os.path.join(D, "ping_20261102T0209Z.txt")).read()
tot = dict(re.findall(r"^(\w+) TOTAL 12 packets wall_s=([\d.]+)", probe, re.M))
U, C = float(tot["uncached"]) / 12, float(tot["cached_prefill"]) / 12
F = float(re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", ping)[1]) / 1e3
A = 2 * (U - C); pair, row = A / 960**2, (U - A) / 960
pkt_full = lambda T: F + row * T + pair * T * T
pkt_pref = lambda T: F + row * T + pair * T * (T + 1) / 2
pkt_dec = lambda p: F + row + pair * (p + 1)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
L = int(sys.argv[2]) if len(sys.argv) > 2 else 32
off = 12 * sum(pkt_full(L + s) for s in range(N + 1))
on = 12 * (pkt_pref(L) + sum(pkt_dec(L + s - 1) for s in range(1, N + 1)))
print(f"model: L={L} N={N} off={off:.2f} s on={on:.2f} s pair={off + on:.2f} s (step walls only, no process startup)")
if len(sys.argv) > 4:
    meas = [sum(int(r["step_wall_us"]) for r in csv.DictReader(open(os.path.join(d, "steps.csv")))) / 1e6 for d in sys.argv[3:5]]
    ratio = sum(meas) / (off + on)
    print(f"measured: off={meas[0]:.2f} s on={meas[1]:.2f} s pair={sum(meas):.2f} s; ratio measured/model={ratio:.3f} "
          f"-> {'WITHIN' if 0.5 <= ratio <= 1.5 else 'OUTSIDE'} +-50%")
