# Project Hydra

I wanted to understand distributed inference by building it instead of reading about it. So Hydra runs a full GPT-2 transformer block across two physical machines, a Windows/WSL2 laptop and a MacBook, talking over raw TCP sockets, with every piece written from scratch in C++. No ML frameworks, no gRPC, no cloud. Just sockets, Protocol Buffers, and a lot of checking the math against PyTorch.

The orchestrator slices up the work and ships activation tensors over the wire. The worker holds the layer weights, runs the computation, and sends the result back. Every kernel is verified numerically against a double-precision PyTorch reference before I move on.

**What it is, honestly:** Hydra runs *one* complete transformer block end to end across two machines. It's not a full-model server, and it doesn't do multi-worker pipeline parallelism yet. That's where it's headed next. What it does demonstrate is the hard part most tutorials skip: the transport, serialization, and numerical-precision layers of distributed inference, built from the ground up and verified at every step.

## What's actually built

The transport layer is the part I'm most proud of, because getting TCP right is fiddlier than it looks. TCP gives you a stream of bytes, not messages. It won't tell you where one message ends and the next begins, so every packet is length-prefixed: four bytes saying how big the message is, then the message. On top of that there's the unglamorous reliability work that separates real network code from tutorial code. Looping on partial reads and writes. Handling `EINTR` when a signal interrupts a syscall. Ignoring `SIGPIPE` so a dropped connection becomes an error instead of killing the process. Disabling Nagle's algorithm with `TCP_NODELAY` so small request/response messages go out immediately. A receive timeout so a stalled peer can't hang the orchestrator forever. Graceful shutdown via `sigaction` that lets the worker finish cleanly on Ctrl-C. Messages go over the wire as Protocol Buffers, which keeps them self-describing and handles cross-platform encoding. Worth noting the two machines run *different* protobuf versions, so each regenerates its own code locally.

Then the actual transformer, built one verified primitive at a time:

| Milestone | Component | Max diff vs PyTorch | Notes |
|-----------|-----------|---------------------|-------|
| M1 | LayerNorm | 1.19e-07 | Double-accumulated mean/variance |
| M2 | QKV projection | 0.00e+00 | Bit-identical, Conv1D `[in,out]` weight layout |
| M3 | Single-head attention (T=1) | 0.00e+00 | Structural test; softmax/mask dormant at T=1 |
| M4 | Masked multi-head attention | 0.00e+00 | Real softmax plus causal mask, confirmed active |
| M5 | MLP (tanh-GELU) | 2.86e-06 | First non-zero diff, and there's a good reason |
| M6 | **Full transformer block** | **3.81e-06** | Pre-LN ordering plus residual connections |

That M5 number is my favorite part. Everything up to it came out bit-identical to PyTorch, but the MLP uses GELU, which calls `tanh`, and `tanh` isn't computed identically across libraries. So each output differs by about 1e-7, and the matmul that follows amplifies that across a 3072-term sum to roughly 3e-6. The diff isn't a bug. It's the predictable signature of two correct-but-different `tanh` implementations. Being able to explain *why* a result is the number it is, rather than just "it passed," was the whole point of building it this way.

## How it fits together

```
┌──────────────────────────┐      TCP / Protocol Buffers      ┌──────────────────────────┐
│   Orchestrator (WSL2)     │  >> length-prefixed packet >>    │      Worker (macOS)       │
│   loads input, sends      │                                  │   loads weights, computes │
│   activations, gets       │  << length-prefixed response <<  │   y = f(x), loops on      │
│   the result back         │                                  │   accept()                │
└──────────────────────────┘                                  └──────────────────────────┘
```

The worker starts with a fixed role (`--mode layernorm|qkv|attention|mlp|block`), the same way real pipeline parallelism assigns node roles up front rather than negotiating them per request.

## Running it

You'll need a C++17 compiler, Protocol Buffers (`protoc` plus libprotobuf), and Python with `torch`, `transformers`, and `safetensors` for the weight extraction and reference generation.

```bash
make proto              # generate protobuf code, run on EACH machine separately,
                        # never copy the generated files across (version mismatch)
make                    # build

python extract_gpt2_block.py   # pull real GPT-2 weights, generate the reference

# on the worker machine:
./worker --mode block

# on the orchestrator machine:
./orchestrator --mode block --T 8

python compare_block.py 8      # check the result against PyTorch
```

One detail that matters for the verification to mean anything: the references are generated in double precision to match the kernels. If you compare a double-accumulated kernel against a float32 reference, the two disagree by about 1e-5 just from precision differences, right at the tolerance, causing false failures with no actual bug. Matching the precision class makes the tolerance measure correctness instead.

## Some decisions I made on purpose

**Raw sockets over gRPC.** gRPC would've handed me framing, retries, and TLS for free, and buried every transport decision under an abstraction. The entire reason this project exists is to handle partial reads, `EINTR`, `SIGPIPE`, Nagle, and framing myself, and to know exactly what's on the wire.

**The Conv1D weight trap.** This one cost me time and taught me the most. GPT-2 stores its projection weights as `[in, out]` and computes `x @ W`, the transpose of PyTorch's standard `Linear`, which is `[out, in]` and `W @ x`. Get the convention wrong and the output is the right *shape* but the wrong *numbers*. Nothing crashes, nothing warns you, only the numerical comparison catches it. I wrote a dedicated kernel for GPT-2's layout rather than reusing my earlier one, which would have silently applied the wrong convention.

## What's not done yet

- One block. Multi-block and multi-worker pipeline parallelism is the next phase.
- fp32 only, no quantization.
- No fault recovery: if a worker drops mid-request, the pipeline stalls.
- The graceful-shutdown accept loop has a small TOCTOU race (the clean fix is `pselect`/`ppoll`).
