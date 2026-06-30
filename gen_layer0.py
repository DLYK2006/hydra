#!/usr/bin/env python3
"""
gen_layer0.py -- generates the fixed input vector and layer-0 weights for the
real-tensor milestone (Option B, first cut: one linear layer, y = W@x + b).

Deterministic (random.seed(0)) so the values here can be hand-checked against
the orchestrator's printed wire output after the round trip.

No numpy dependency (not installed in WSL2) -- stdlib random + struct only.

Layout (all values float32, explicit little-endian -- both WSL2 and the Mac
are x86_64, but pinning the byte order makes the files portable regardless of
which machine the generator runs on):

  data/input.bin          -- x: IN_DIM (4) floats
  data/layer0_weights.bin -- W: OUT_DIM*IN_DIM (8) floats, row-major
                                (W[i][j] at index i*IN_DIM + j)
                             b: OUT_DIM (2) floats
"""
import os
import random
import struct

IN_DIM = 4
OUT_DIM = 2

random.seed(0)


def f32(v):
    """Round-trip through float32 so printed values match what's on disk
    and what the C++ worker will compute in float32 arithmetic."""
    return struct.unpack('<f', struct.pack('<f', v))[0]


x = [f32(random.uniform(-1.0, 1.0)) for _ in range(IN_DIM)]
W = [[f32(random.uniform(-1.0, 1.0)) for _ in range(IN_DIM)] for _ in range(OUT_DIM)]
b = [f32(random.uniform(-1.0, 1.0)) for _ in range(OUT_DIM)]

y = [b[i] + sum(W[i][j] * x[j] for j in range(IN_DIM)) for i in range(OUT_DIM)]

os.makedirs("data", exist_ok=True)

with open("data/input.bin", "wb") as f:
    f.write(struct.pack(f"<{IN_DIM}f", *x))

with open("data/layer0_weights.bin", "wb") as f:
    W_flat = [W[i][j] for i in range(OUT_DIM) for j in range(IN_DIM)]
    f.write(struct.pack(f"<{OUT_DIM * IN_DIM}f", *W_flat))
    f.write(struct.pack(f"<{OUT_DIM}f", *b))

print(f"x (input, IN_DIM={IN_DIM}):")
print(" ", x)
print(f"W (weights, {OUT_DIM}x{IN_DIM}, row-major):")
for row in W:
    print(" ", row)
print(f"b (bias, OUT_DIM={OUT_DIM}):")
print(" ", b)
print("expected y = W @ x + b:")
print(" ", y)
print()
print("wrote data/input.bin (", IN_DIM * 4, "bytes )")
print("wrote data/layer0_weights.bin (", (OUT_DIM * IN_DIM + OUT_DIM) * 4, "bytes )")
