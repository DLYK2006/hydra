#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>

#include "hydra.pb.h"

// Reads exactly `len` bytes into buf, looping over short reads and retrying
// on EINTR -- mirrors sendAll() on the orchestrator side, since recv() is
// likewise not guaranteed to fill the buffer in one call.
// Returns:
//   (ssize_t)len  - all `len` bytes read
//   0             - peer closed before any bytes of this call arrived
//                   (clean EOF at a message boundary)
//   -1            - recv() error, or peer closed mid-message (truncated)
ssize_t recvAll(int sock, char* buf, size_t len) {
    size_t totalRead = 0;
    while (totalRead < len) {
        ssize_t n = recv(sock, buf + totalRead, len - totalRead, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return (totalRead == 0) ? 0 : -1;
        }
        totalRead += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(totalRead);
}

// send() may write fewer bytes than requested for larger payloads, and can
// be interrupted by a signal mid-write -- mirrors sendAll() on the
// orchestrator side. Loop until the whole buffer is on the wire.
ssize_t sendAll(int sock, const char* buf, size_t len) {
    size_t totalSent = 0;
    while (totalSent < len) {
        ssize_t n = send(sock, buf + totalSent, len - totalSent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        totalSent += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(totalSent);
}

// Frame the response with the same 4-byte length prefix (network byte
// order) the orchestrator uses for requests -- mirrors sendPacket() on the
// orchestrator side, so recvPacket() there can read it back with identical
// framing logic.
bool sendPacket(int sock, const std::string& serialized) {
    uint32_t len = htonl(static_cast<uint32_t>(serialized.size()));
    if (sendAll(sock, reinterpret_cast<const char*>(&len), sizeof(len)) < 0) {
        return false;
    }
    return sendAll(sock, serialized.data(), serialized.size()) >= 0;
}

// Reads exactly `count` float32 values from a raw little-endian binary file
// with no header. x86_64/ARM64 are both little-endian for float storage, so
// bytes land directly into a float buffer with no byte-swap. A size mismatch
// is a hard error -- a wrong-size weight file would otherwise load silently
// and produce garbage output for every connection.
bool loadFloatsFromFile(const std::string& path, size_t count, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cout << "Failed to open " << path << std::endl;
        return false;
    }

    std::streamsize size = f.tellg();
    std::streamsize expected = static_cast<std::streamsize>(count * sizeof(float));
    if (size != expected) {
        std::cout << "Unexpected size for " << path << ": got " << size
                  << " bytes, expected " << expected << std::endl;
        return false;
    }

    f.seekg(0, std::ios::beg);
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), size);
    if (f.gcount() != size) {
        std::cout << "Short read on " << path << std::endl;
        return false;
    }
    return true;
}

// Compute LayerNorm over one row of `hidden_dim` values: y = (x - mean) /
// sqrt(var + eps) * gamma + beta.
//
// Double accumulators for mean and variance: fp32 catastrophic cancellation
// is measurable when summing 768 squared deviations. The mean and variance
// are computed in double; the final normalized value is cast back to float
// because gamma and beta are fp32 (as stored in the model file) so there is
// no benefit to carrying more precision past that multiply.
void layernorm(const float* x, const float* gamma, const float* beta,
               float* y, int hidden_dim) {
    constexpr double eps = 1e-5;

    double sum = 0.0;
    for (int i = 0; i < hidden_dim; i++) sum += static_cast<double>(x[i]);
    double mean = sum / hidden_dim;

    double var_sum = 0.0;
    for (int i = 0; i < hidden_dim; i++) {
        double d = static_cast<double>(x[i]) - mean;
        var_sum += d * d;
    }
    double inv_std = 1.0 / std::sqrt(var_sum / hidden_dim + eps);

    for (int i = 0; i < hidden_dim; i++) {
        y[i] = static_cast<float>(
            (static_cast<double>(x[i]) - mean) * inv_std
            * static_cast<double>(gamma[i])
            + static_cast<double>(beta[i]));
    }
}

// matmul_nt: y = x @ W + b
// GPT-2 Conv1D convention: W stored [in_dim, out_dim] row-major (opposite of
// MNIST where W is [out, in]).  k-outer, j-inner loop so W[k*out_dim .. +out_dim)
// is accessed contiguously per k step — good cache behaviour.
// acc[] stays in double through the entire k loop: 768 fma steps accumulate
// without any intermediate float rounding.  Cast to float only once, after
// all 768 contributions are in.  Rounding to float on each k step (the
// prior version) was equivalent to float accumulation and caused ~2-3e-5 error.
void matmul_nt(const float* x, const float* W, const float* b,
               float* y, int T, int in_dim, int out_dim) {
    std::vector<double> acc(static_cast<size_t>(out_dim));
    for (int t = 0; t < T; t++) {
        // Initialise accumulator row with bias (double, no rounding yet).
        for (int j = 0; j < out_dim; j++)
            acc[j] = static_cast<double>(b[j]);

        // Accumulate all in_dim contributions in double before touching y[].
        for (int k = 0; k < in_dim; k++) {
            double x_tk = static_cast<double>(x[t * in_dim + k]);
            const float* W_row = W + k * out_dim;
            for (int j = 0; j < out_dim; j++)
                acc[j] += x_tk * static_cast<double>(W_row[j]);
        }

        // Single float cast per output element after all 768 terms are in.
        for (int j = 0; j < out_dim; j++)
            y[t * out_dim + j] = static_cast<float>(acc[j]);
    }
}

// Scaled dot-product multi-head attention (M3+).
// qkv: [T, 3*HIDDEN_DIM] row-major from matmul_nt(c_attn); within each row
//      the columns are laid out as Q(HIDDEN_DIM) | K(HIDDEN_DIM) | V(HIDDEN_DIM).
//      Head h at token t:
//        Q_h = qkv + t*(3*H) + h*HEAD_DIM
//        K_h = qkv + t*(3*H) + H + h*HEAD_DIM
//        V_h = qkv + t*(3*H) + 2*H + h*HEAD_DIM
// out:  [T, HIDDEN_DIM] row-major -- all heads concatenated per token;
//       head h at token t occupies out + t*H + h*HEAD_DIM.
//
// Double accumulators throughout: QK dot products (HEAD_DIM=64 terms),
// softmax exp/sum, and V weighted sum all stay in float64. Cast to float32
// once per output element. At T=1: softmax([single_score]) = 1.0, so
// output per head equals V per head exactly (used as the M3 structural test).
void attn_heads(const float* qkv, float* out,
                int T, int n_heads, int head_dim) {
    int H = n_heads * head_dim;
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    std::vector<double> scores(static_cast<size_t>(T) * T);
    std::vector<double> acc(static_cast<size_t>(head_dim));

    for (int h = 0; h < n_heads; h++) {
        // Scaled QK^T: scores[qt*T + kt] = (Q_h[qt] . K_h[kt]) * scale
        for (int qt = 0; qt < T; qt++) {
            const float* Q_h = qkv + qt * (3 * H) + h * head_dim;
            for (int kt = 0; kt < T; kt++) {
                const float* K_h = qkv + kt * (3 * H) + H + h * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++)
                    dot += static_cast<double>(Q_h[d]) * static_cast<double>(K_h[d]);
                scores[qt * T + kt] = dot * scale;
            }
        }

        // Causal mask: token qt cannot attend to future tokens kt > qt.
        // Pre-softmax -1e9 makes exp(-1e9 - row_max) underflow to 0.0 in
        // float64, matching the Python reference's masked_fill(-1e9) exactly.
        // This loop is O(T^2) and dominates only at large T; for T=8 it is
        // 28 assignments and is negligible versus the score computation above.
        for (int qt = 0; qt < T; qt++)
            for (int kt = qt + 1; kt < T; kt++)
                scores[qt * T + kt] = -1e9;

        // Numerically-stable softmax row by row (subtract max before exp).
        for (int qt = 0; qt < T; qt++) {
            double* row = scores.data() + qt * T;
            double mx = row[0];
            for (int kt = 1; kt < T; kt++) if (row[kt] > mx) mx = row[kt];
            double sum = 0.0;
            for (int kt = 0; kt < T; kt++) { row[kt] = std::exp(row[kt] - mx); sum += row[kt]; }
            for (int kt = 0; kt < T; kt++) row[kt] /= sum;
        }

        // Weighted sum of V: out[qt*H + h*head_dim .. +head_dim]
        for (int qt = 0; qt < T; qt++) {
            const double* attn_row = scores.data() + qt * T;
            float* out_h = out + qt * H + h * head_dim;
            for (int d = 0; d < head_dim; d++) acc[d] = 0.0;
            for (int kt = 0; kt < T; kt++) {
                const float* V_h = qkv + kt * (3 * H) + 2 * H + h * head_dim;
                double w = attn_row[kt];
                for (int d = 0; d < head_dim; d++)
                    acc[d] += w * static_cast<double>(V_h[d]);
            }
            for (int d = 0; d < head_dim; d++)
                out_h[d] = static_cast<float>(acc[d]);
        }
    }
}

// KV cache: attention for T_new new rows at absolute positions start_pos ..
// start_pos+T_new-1, reading K/V for keys 0..p from the per-layer cache (which
// already holds the new rows). Kc, Vc: [KV_CAPACITY, H] row-major.
//
// Same per-row arithmetic as attn_heads(), in the same order: scores for keys
// 0..p ascending (double dot over d ascending, times scale); max from key 0;
// exp and sum ascending; divide; V accumulation ascending; one float cast.
// attn_heads() also visits the masked keys kt > p, but with score -1e9 they
// contribute exp() == 0.0 exactly to the sum and 0.0*V to the accumulator,
// which leaves both bitwise unchanged -- so skipping them here is bit-exact
// against the full recompute (verified by scripts/kv_check.sh gates B/C).
void attn_heads_cached(const float* qkv_new, int T_new, int start_pos,
                       const float* Kc, const float* Vc, float* out,
                       int n_heads, int head_dim) {
    int H = n_heads * head_dim;
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    std::vector<double> scores(static_cast<size_t>(start_pos) + T_new);
    std::vector<double> acc(static_cast<size_t>(head_dim));

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < T_new; i++) {
            const int p = start_pos + i;   // this query row's absolute position
            const float* Q_h = qkv_new + i * (3 * H) + h * head_dim;
            for (int kt = 0; kt <= p; kt++) {
                const float* K_h = Kc + static_cast<size_t>(kt) * H + h * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++)
                    dot += static_cast<double>(Q_h[d]) * static_cast<double>(K_h[d]);
                scores[kt] = dot * scale;
            }

            double mx = scores[0];
            for (int kt = 1; kt <= p; kt++) if (scores[kt] > mx) mx = scores[kt];
            double sum = 0.0;
            for (int kt = 0; kt <= p; kt++) { scores[kt] = std::exp(scores[kt] - mx); sum += scores[kt]; }
            for (int kt = 0; kt <= p; kt++) scores[kt] /= sum;

            float* out_h = out + i * H + h * head_dim;
            for (int d = 0; d < head_dim; d++) acc[d] = 0.0;
            for (int kt = 0; kt <= p; kt++) {
                const float* V_h = Vc + static_cast<size_t>(kt) * H + h * head_dim;
                double w = scores[kt];
                for (int d = 0; d < head_dim; d++)
                    acc[d] += w * static_cast<double>(V_h[d]);
            }
            for (int d = 0; d < head_dim; d++)
                out_h[d] = static_cast<float>(acc[d]);
        }
    }
}

// GELU with tanh approximation (GPT-2 variant):
//   gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// Must match PyTorch's F.gelu(x, approximate='tanh') -- NOT the default erf
// form.  GPT-2 was trained with the tanh approximation; using erf gives
// close-but-wrong outputs with no obvious error signal.
// Computed in double for consistency with the double-accumulated matmul
// convention: the input (float32 from matmul_nt output) is promoted to double,
// the polynomial and tanh run in double, and the result is cast back to float32.
inline float gelu(float x) {
    constexpr double kSqrt2OverPi = 0.7978845608028654;  // sqrt(2/pi)
    constexpr double kBeta        = 0.044715;
    double xd    = static_cast<double>(x);
    double inner = kSqrt2OverPi * (xd + kBeta * xd * xd * xd);
    return static_cast<float>(0.5 * xd * (1.0 + std::tanh(inner)));
}

// Linear mode dimensions: MNIST fc1 (784 -> 128)
constexpr int IN_DIM  = 784;
constexpr int OUT_DIM = 128;

// LayerNorm / QKV / attention / MLP mode: GPT-2 small hidden dim
constexpr int HIDDEN_DIM    = 768;
constexpr int QKV_IN_DIM    = 768;
constexpr int QKV_OUT_DIM   = 2304;  // 3 * 768
constexpr int ATTN_N_HEAD   = 12;
constexpr int ATTN_HEAD_DIM = 64;    // HIDDEN_DIM / ATTN_N_HEAD
constexpr int MLP_HIDDEN_DIM = 3072; // 4 * HIDDEN_DIM

// G3: block mode holds all 12 GPT-2 blocks and dispatches on layer_index.
constexpr int N_BLOCKS = 12;

// G3: cap on one framed message, checked before buffer.resize(). The largest
// legitimate inbound packet is [T, 768] fp32; at GPT-2's n_ctx = 1024 that is
// 3,145,728 data bytes plus < 64 bytes of protobuf tags/shape, so 4 MiB clears
// it (and implicitly bounds T at ~1365). Without a cap, a corrupt or hostile
// 4-byte length prefix makes the worker try a ~4 GiB allocation.
constexpr uint32_t MAX_MSG_LEN = 4u << 20;  // 4 MiB

// KV cache (block mode): positions per sequence (GPT-2 n_ctx).
constexpr int KV_CAPACITY = 1024;

// G3: one GPT-2 block's weights, loaded from data/h{NN}_<name>.bin (written by
// extract_gpt2_all_blocks.py; Conv1D matrices are [in, out] row-major).
struct BlockWeights {
    std::vector<float> ln1_gamma, ln1_beta;       // [HIDDEN_DIM]
    std::vector<float> c_attn_W, c_attn_b;        // [HIDDEN_DIM x QKV_OUT_DIM], [QKV_OUT_DIM]
    std::vector<float> attn_proj_W, attn_proj_b;  // [HIDDEN_DIM x HIDDEN_DIM], [HIDDEN_DIM]
    std::vector<float> ln2_gamma, ln2_beta;       // [HIDDEN_DIM]
    std::vector<float> c_fc_W, c_fc_b;            // [HIDDEN_DIM x MLP_HIDDEN_DIM], [MLP_HIDDEN_DIM]
    std::vector<float> mlp_proj_W, mlp_proj_b;    // [MLP_HIDDEN_DIM x HIDDEN_DIM], [HIDDEN_DIM]
};

// KV cache: one active sequence. Per layer, K and V rows for positions
// 0..len[L]-1, exactly the float32 values matmul_nt wrote into qkv_buf
// columns [H, 2H) and [2H, 3H). Allocated once at startup in block mode:
// 12 layers x 2 x 1024 x 768 x 4 B = 75,497,472 B.
struct KVCache {
    std::vector<float> K[N_BLOCKS], V[N_BLOCKS];   // [KV_CAPACITY x HIDDEN_DIM] each
    int len[N_BLOCKS] = {0};
    uint64_t seq = 0;                              // 0 = no active sequence
};

// Hardening milestone: minimal graceful shutdown. SIGINT/SIGTERM set this
// flag from the handler -- the only thing safe to do in a signal handler per
// POSIX async-signal-safety (no I/O, no close()). main()'s accept loop checks
// it and exits cleanly instead of running until SIGINT/kill tears down the
// process.
volatile sig_atomic_t g_shutdown_requested = 0;

void handleShutdownSignal(int /*signum*/) {
    g_shutdown_requested = 1;
}

int main(int argc, char* argv[]) {
    // --mode linear    : MNIST fc1 forward pass (default, backward-compat)
    // --mode layernorm : GPT-2 block-0 ln_1 forward pass
    std::string mode = "linear";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        }
    }
    if (mode != "linear" && mode != "layernorm" && mode != "qkv"
            && mode != "attention" && mode != "mlp" && mode != "block") {
        std::cout << "Unknown mode '" << mode
                  << "'. Use --mode linear|layernorm|qkv|attention|mlp|block"
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Worker mode: " << mode << std::endl;

    // See orchestrator.cpp: ignore SIGPIPE so a peer that disappears mid-I/O
    // surfaces as a normal errno/return-value error instead of killing us.
    signal(SIGPIPE, SIG_IGN);

    // Register SIGINT/SIGTERM via sigaction with sa_flags = 0 (no
    // SA_RESTART). signal()'s BSD semantics imply SA_RESTART, which would
    // silently restart accept() after the handler returns -- the accept loop
    // below would never observe the interruption. Without SA_RESTART,
    // accept() returns -1/EINTR when the signal arrives, which the loop
    // checks against g_shutdown_requested.
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handleShutdownSignal;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // Load weights once at startup, before opening the listening socket --
    // fail fast on a missing/wrong-size weight file rather than binding a
    // port we can't actually serve.
    std::vector<float> weights;   // linear mode: W[OUT_DIM x IN_DIM] + b[OUT_DIM]
    std::vector<float> ln_gamma;  // layernorm mode: scale [HIDDEN_DIM]
    std::vector<float> ln_beta;   // layernorm mode: shift [HIDDEN_DIM]
    std::vector<float> qkv_W;        // qkv/attention mode: W[QKV_IN_DIM x QKV_OUT_DIM], [in, out]
    std::vector<float> qkv_b;        // qkv/attention mode: bias [QKV_OUT_DIM]
    std::vector<float> attn_proj_W;  // attention mode: c_proj W[HIDDEN_DIM x HIDDEN_DIM], [in, out]
    std::vector<float> attn_proj_b;  // attention mode: c_proj bias [HIDDEN_DIM]
    std::vector<float> mlp_c_fc_W;   // mlp mode: c_fc W[HIDDEN_DIM x MLP_HIDDEN_DIM], [in, out]
    std::vector<float> mlp_c_fc_b;   // mlp mode: c_fc bias [MLP_HIDDEN_DIM]
    std::vector<float> mlp_c_proj_W; // mlp mode: c_proj W[MLP_HIDDEN_DIM x HIDDEN_DIM], [in, out]
    std::vector<float> mlp_c_proj_b; // mlp mode: c_proj bias [HIDDEN_DIM]
    std::vector<BlockWeights> blocks; // block mode: all N_BLOCKS blocks, indexed by layer_index
    KVCache kv;                       // block mode: KV cache for sequence_id != 0

    if (mode == "linear") {
        if (!loadFloatsFromFile("data/layer0_weights.bin",
                                OUT_DIM * IN_DIM + OUT_DIM, weights)) {
            return EXIT_FAILURE;
        }
    } else if (mode == "layernorm") {
        if (!loadFloatsFromFile("data/ln_gamma.bin", HIDDEN_DIM, ln_gamma)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/ln_beta.bin", HIDDEN_DIM, ln_beta)) {
            return EXIT_FAILURE;
        }
        std::cout << "Loaded ln_gamma and ln_beta (" << HIDDEN_DIM
                  << " floats each)" << std::endl;
    } else if (mode == "qkv") {
        if (!loadFloatsFromFile("data/qkv_W.bin",
                                QKV_IN_DIM * QKV_OUT_DIM, qkv_W)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/qkv_b.bin", QKV_OUT_DIM, qkv_b)) {
            return EXIT_FAILURE;
        }
        std::cout << "Loaded QKV W [" << QKV_IN_DIM << "," << QKV_OUT_DIM
                  << "] and b [" << QKV_OUT_DIM << "]" << std::endl;
    } else if (mode == "attention") {
        if (!loadFloatsFromFile("data/qkv_W.bin",
                                QKV_IN_DIM * QKV_OUT_DIM, qkv_W)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/qkv_b.bin", QKV_OUT_DIM, qkv_b)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/attn_proj_W.bin",
                                HIDDEN_DIM * HIDDEN_DIM, attn_proj_W)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/attn_proj_b.bin", HIDDEN_DIM, attn_proj_b)) {
            return EXIT_FAILURE;
        }
        std::cout << "Loaded c_attn W [" << QKV_IN_DIM << "," << QKV_OUT_DIM
                  << "] and c_proj W [" << HIDDEN_DIM << "," << HIDDEN_DIM << "]" << std::endl;
    } else if (mode == "mlp") {
        if (!loadFloatsFromFile("data/mlp_c_fc_W.bin",
                                HIDDEN_DIM * MLP_HIDDEN_DIM, mlp_c_fc_W)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/mlp_c_fc_b.bin", MLP_HIDDEN_DIM, mlp_c_fc_b)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/mlp_c_proj_W.bin",
                                MLP_HIDDEN_DIM * HIDDEN_DIM, mlp_c_proj_W)) {
            return EXIT_FAILURE;
        }
        if (!loadFloatsFromFile("data/mlp_c_proj_b.bin", HIDDEN_DIM, mlp_c_proj_b)) {
            return EXIT_FAILURE;
        }
        std::cout << "Loaded MLP c_fc W [" << HIDDEN_DIM << "," << MLP_HIDDEN_DIM
                  << "] and c_proj W [" << MLP_HIDDEN_DIM << "," << HIDDEN_DIM
                  << "]" << std::endl;
    } else {  // block: all N_BLOCKS blocks from data/h{NN}_<name>.bin
        blocks.resize(N_BLOCKS);
        size_t total_bytes = 0;
        for (int n = 0; n < N_BLOCKS; n++) {
            BlockWeights& w = blocks[n];
            // (file name, element count, destination) -- sizes are exact, so a
            // missing or truncated h-file fails here, before bind().
            const struct { const char* name; size_t count; std::vector<float>* dst; } files[] = {
                {"ln1_gamma",   HIDDEN_DIM,                                       &w.ln1_gamma},
                {"ln1_beta",    HIDDEN_DIM,                                       &w.ln1_beta},
                {"c_attn_W",    static_cast<size_t>(QKV_IN_DIM) * QKV_OUT_DIM,    &w.c_attn_W},
                {"c_attn_b",    QKV_OUT_DIM,                                      &w.c_attn_b},
                {"attn_proj_W", static_cast<size_t>(HIDDEN_DIM) * HIDDEN_DIM,     &w.attn_proj_W},
                {"attn_proj_b", HIDDEN_DIM,                                       &w.attn_proj_b},
                {"ln2_gamma",   HIDDEN_DIM,                                       &w.ln2_gamma},
                {"ln2_beta",    HIDDEN_DIM,                                       &w.ln2_beta},
                {"c_fc_W",      static_cast<size_t>(HIDDEN_DIM) * MLP_HIDDEN_DIM, &w.c_fc_W},
                {"c_fc_b",      MLP_HIDDEN_DIM,                                   &w.c_fc_b},
                {"mlp_proj_W",  static_cast<size_t>(MLP_HIDDEN_DIM) * HIDDEN_DIM, &w.mlp_proj_W},
                {"mlp_proj_b",  HIDDEN_DIM,                                       &w.mlp_proj_b},
            };
            for (const auto& f : files) {
                char path[64];
                std::snprintf(path, sizeof(path), "data/h%02d_%s.bin", n, f.name);
                if (!loadFloatsFromFile(path, f.count, *f.dst))
                    return EXIT_FAILURE;
                total_bytes += f.count * sizeof(float);
            }
        }
        std::cout << "Loaded " << N_BLOCKS << " blocks (" << total_bytes << " bytes)"
                  << std::endl;
        size_t kv_bytes = 0;
        for (int n = 0; n < N_BLOCKS; n++) {
            kv.K[n].assign(static_cast<size_t>(KV_CAPACITY) * HIDDEN_DIM, 0.0f);
            kv.V[n].assign(static_cast<size_t>(KV_CAPACITY) * HIDDEN_DIM, 0.0f);
            kv_bytes += (kv.K[n].size() + kv.V[n].size()) * sizeof(float);
        }
        std::cout << "KV cache: " << kv_bytes << " bytes (" << N_BLOCKS << " layers x K,V x "
                  << KV_CAPACITY << " x " << HIDDEN_DIM << " fp32)" << std::endl;
    }

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return EXIT_FAILURE;
    }

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(50051);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return EXIT_FAILURE;
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        return EXIT_FAILURE;
    }

    std::cout << "Worker listening on port 50051..." << std::endl;

    while (!g_shutdown_requested) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (new_socket < 0) {
            if (errno == EINTR) {
                if (g_shutdown_requested) break;
                continue;
            }
            perror("accept");
            continue;
        }

        // Per Engineering Standards: TCP_NODELAY is non-optional on both ends.
        int nodelay = 1;
        if (setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
            perror("Warning: setsockopt TCP_NODELAY");
        }

        int packets_received = 0;
        bool connection_ok = true;
        std::string buffer;

        while (true) {
            uint32_t net_len;
            ssize_t r = recvAll(new_socket, reinterpret_cast<char*>(&net_len), sizeof(net_len));
            if (r == 0) break;
            if (r < 0) {
                std::cout << "recv error or mid-message EOF while reading length prefix" << std::endl;
                connection_ok = false;
                break;
            }

            uint32_t msg_len = ntohl(net_len);
            // The payload can't be skipped without reading it, so framing is
            // lost: drop only this connection and return to accept().
            if (msg_len > MAX_MSG_LEN) {
                std::cout << "Rejecting message: length prefix " << msg_len
                          << " bytes exceeds MAX_MSG_LEN " << MAX_MSG_LEN
                          << "; closing connection" << std::endl;
                connection_ok = false;
                break;
            }
            buffer.resize(msg_len);

            r = recvAll(new_socket, buffer.data(), msg_len);
            if (r <= 0) {
                std::cout << "recv error or mid-message EOF (expected "
                          << msg_len << " bytes)" << std::endl;
                connection_ok = false;
                break;
            }

            hydra::TensorPacket packet;
            if (!packet.ParseFromString(buffer)) {
                std::cout << "Failed to parse TensorPacket (" << buffer.size()
                          << " bytes received)" << std::endl;
                connection_ok = false;
                break;
            }
            const auto t_parsed = std::chrono::steady_clock::now();   // bench: compute window start

            packets_received++;
            std::cout << "Received TensorPacket #" << packets_received << " from '"
                      << packet.device_id() << "' (layer " << packet.layer_index()
                      << "), " << packet.data_size() << " floats, shape [";
            for (int i = 0; i < packet.shape_size(); i++) {
                std::cout << packet.shape(i)
                          << (i + 1 < packet.shape_size() ? "," : "");
            }
            std::cout << "] seq=" << packet.sequence_id()
                      << " start_pos=" << packet.start_pos() << std::endl;

            hydra::TensorPacket response;
            response.set_layer_index(packet.layer_index());
            response.set_device_id("worker-mac");

            if (mode == "linear") {
                // --- linear mode: validate 1-D input against IN_DIM ---
                if (packet.data_size() != IN_DIM ||
                    (packet.shape_size() > 0 && packet.shape(0) != IN_DIM)) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << ", shape[0]=" << (packet.shape_size() > 0 ? packet.shape(0) : -1)
                              << ", expected IN_DIM=" << IN_DIM << std::endl;
                    connection_ok = false;
                    break;
                }

                // y = W@x + b  (weights stored row-major [OUT_DIM x IN_DIM] + bias)
                std::vector<float> y(OUT_DIM);
                for (int i = 0; i < OUT_DIM; i++) {
                    float acc = weights[OUT_DIM * IN_DIM + i];
                    for (int j = 0; j < IN_DIM; j++) {
                        acc += weights[i * IN_DIM + j] * packet.data(j);
                    }
                    y[i] = acc;
                }
                for (float v : y) response.add_data(v);
                response.add_shape(OUT_DIM);

            } else if (mode == "layernorm") {
                // --- layernorm mode: validate 2-D input [T, HIDDEN_DIM] ---
                //
                // Three distinct failure modes, each logged precisely so the
                // operator knows whether the orchestrator sent the wrong rank,
                // the wrong hidden dim, or a size that doesn't match the shape.
                if (packet.shape_size() != 2) {
                    std::cout << "Rejecting packet: layernorm mode requires shape_size=2"
                              << ", got shape_size=" << packet.shape_size() << std::endl;
                    connection_ok = false;
                    break;
                }
                int pkt_T = packet.shape(0);
                int pkt_H = packet.shape(1);
                if (pkt_H != HIDDEN_DIM) {
                    std::cout << "Rejecting packet: expected shape[1]=" << HIDDEN_DIM
                              << " (HIDDEN_DIM), got " << pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }
                if (packet.data_size() != pkt_T * pkt_H) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << " does not match shape[0]*shape[1]="
                              << pkt_T << "*" << pkt_H << "=" << pkt_T * pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }

                // Apply LayerNorm row by row. Each row is one token's hidden
                // state. Rows are independent -- no cross-row interaction in LN.
                std::vector<float> y(static_cast<size_t>(pkt_T) * HIDDEN_DIM);
                // RepeatedField::data() returns const float* to the contiguous
                // backing array -- the only correct way to pointer-index into a
                // repeated field, since data(i) returns by value in newer protobuf.
                const float* flat_in = packet.data().data();
                for (int t = 0; t < pkt_T; t++) {
                    const float* row_in  = flat_in + t * HIDDEN_DIM;
                    float*       row_out = &y[t * HIDDEN_DIM];
                    layernorm(row_in, ln_gamma.data(), ln_beta.data(),
                              row_out, HIDDEN_DIM);
                }

                for (float v : y) response.add_data(v);
                response.add_shape(pkt_T);
                response.add_shape(HIDDEN_DIM);

                std::cout << "Computed LayerNorm [" << pkt_T << "," << HIDDEN_DIM
                          << "], first 4 output values of row 0: ["
                          << y[0] << ", " << y[1] << ", "
                          << y[2] << ", " << y[3] << "]" << std::endl;

            } else if (mode == "qkv") {
                // --- qkv mode: validate 2-D input [T, QKV_IN_DIM] ---
                if (packet.shape_size() != 2) {
                    std::cout << "Rejecting packet: qkv mode requires shape_size=2"
                              << ", got shape_size=" << packet.shape_size() << std::endl;
                    connection_ok = false;
                    break;
                }
                int pkt_T = packet.shape(0);
                int pkt_H = packet.shape(1);
                if (pkt_H != QKV_IN_DIM) {
                    std::cout << "Rejecting packet: expected shape[1]=" << QKV_IN_DIM
                              << " (QKV_IN_DIM), got " << pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }
                if (packet.data_size() != pkt_T * pkt_H) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << " does not match shape[0]*shape[1]="
                              << pkt_T << "*" << pkt_H << "=" << pkt_T * pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }

                // y = x @ W + b  (GPT-2 Conv1D convention: [in, out] weight layout)
                std::vector<float> y(static_cast<size_t>(pkt_T) * QKV_OUT_DIM);
                const float* flat_in = packet.data().data();
                matmul_nt(flat_in, qkv_W.data(), qkv_b.data(),
                          y.data(), pkt_T, QKV_IN_DIM, QKV_OUT_DIM);

                // Log first 4 values of Q, K, V for row 0 separately so any
                // split indexing bug is immediately localised to one head.
                const int HEAD_DIM = QKV_IN_DIM;  // 768
                const char* head_names[3] = {"Q", "K", "V"};
                for (int split = 0; split < 3; split++) {
                    int col_off = split * HEAD_DIM;
                    std::cout << head_names[split] << " row0[0:4]: ["
                              << y[col_off]   << ", " << y[col_off+1] << ", "
                              << y[col_off+2] << ", " << y[col_off+3] << "]"
                              << std::endl;
                }

                for (float v : y) response.add_data(v);
                response.add_shape(pkt_T);
                response.add_shape(QKV_OUT_DIM);

            } else if (mode == "mlp") {
                // --- mlp mode: validate 2-D input [T, HIDDEN_DIM] ---
                if (packet.shape_size() != 2) {
                    std::cout << "Rejecting packet: mlp mode requires shape_size=2"
                              << ", got shape_size=" << packet.shape_size() << std::endl;
                    connection_ok = false;
                    break;
                }
                int pkt_T = packet.shape(0);
                int pkt_H = packet.shape(1);
                if (pkt_H != HIDDEN_DIM) {
                    std::cout << "Rejecting packet: expected shape[1]=" << HIDDEN_DIM
                              << " (HIDDEN_DIM), got " << pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }
                if (packet.data_size() != pkt_T * pkt_H) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << " does not match shape[0]*shape[1]="
                              << pkt_T << "*" << pkt_H << "=" << pkt_T * pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }

                const float* flat_in = packet.data().data();

                // Step 1: c_fc -- x [T,768] @ W [768,3072] + b -> [T,3072]
                // Heap-allocated because T varies per connection; avoids a stack
                // allocation of T*3072*4 bytes (~96 KB for T=8) per connection.
                std::vector<float> fc_out(static_cast<size_t>(pkt_T) * MLP_HIDDEN_DIM);
                matmul_nt(flat_in, mlp_c_fc_W.data(), mlp_c_fc_b.data(),
                          fc_out.data(), pkt_T, HIDDEN_DIM, MLP_HIDDEN_DIM);

                // Step 2: GELU elementwise (tanh approximation).
                // gelu() promotes float32 to double, computes polynomial+tanh in
                // double, casts back to float32.  Each element is independent; no
                // accumulation across elements (unlike matmul), so the precision
                // benefit is modest but consistent with the project convention.
                std::vector<float> gelu_out(static_cast<size_t>(pkt_T) * MLP_HIDDEN_DIM);
                for (size_t i = 0; i < static_cast<size_t>(pkt_T) * MLP_HIDDEN_DIM; ++i)
                    gelu_out[i] = gelu(fc_out[i]);

                // Step 3: c_proj -- gelu_out [T,3072] @ W [3072,768] + b -> [T,768]
                std::vector<float> proj_out(static_cast<size_t>(pkt_T) * HIDDEN_DIM);
                matmul_nt(gelu_out.data(), mlp_c_proj_W.data(), mlp_c_proj_b.data(),
                          proj_out.data(), pkt_T, MLP_HIDDEN_DIM, HIDDEN_DIM);

                std::cout << "Computed MLP [" << pkt_T << "," << HIDDEN_DIM << "]"
                          << "  fc[0:4]=[" << fc_out[0] << "," << fc_out[1]
                          << "," << fc_out[2] << "," << fc_out[3] << "]"
                          << "  gelu[0:4]=[" << gelu_out[0] << "," << gelu_out[1]
                          << "," << gelu_out[2] << "," << gelu_out[3] << "]"
                          << "  proj[0:4]=[" << proj_out[0] << "," << proj_out[1]
                          << "," << proj_out[2] << "," << proj_out[3] << "]"
                          << std::endl;

                // MLP sends TWO response packets so the orchestrator can save
                // both the post-GELU intermediate and the final output
                // independently.  The compare script diffs them separately,
                // which localises a GELU bug vs a second-matmul bug.

                // Packet 1: post-GELU intermediate [T, MLP_HIDDEN_DIM]
                {
                    hydra::TensorPacket gelu_resp;
                    gelu_resp.set_layer_index(packet.layer_index());
                    gelu_resp.set_device_id("worker-mac");
                    for (float v : gelu_out) gelu_resp.add_data(v);
                    gelu_resp.add_shape(pkt_T);
                    gelu_resp.add_shape(MLP_HIDDEN_DIM);
                    std::string s;
                    if (!gelu_resp.SerializeToString(&s)) {
                        std::cout << "Failed to serialize GELU response" << std::endl;
                        connection_ok = false;
                        break;
                    }
                    if (!sendPacket(new_socket, s)) {
                        std::cout << "Failed to send GELU response: "
                                  << strerror(errno) << std::endl;
                        connection_ok = false;
                        break;
                    }
                }

                // Packet 2: final c_proj output [T, HIDDEN_DIM]
                {
                    hydra::TensorPacket proj_resp;
                    proj_resp.set_layer_index(packet.layer_index());
                    proj_resp.set_device_id("worker-mac");
                    for (float v : proj_out) proj_resp.add_data(v);
                    proj_resp.add_shape(pkt_T);
                    proj_resp.add_shape(HIDDEN_DIM);
                    std::string s;
                    if (!proj_resp.SerializeToString(&s)) {
                        std::cout << "Failed to serialize proj response" << std::endl;
                        connection_ok = false;
                        break;
                    }
                    if (!sendPacket(new_socket, s)) {
                        std::cout << "Failed to send proj response: "
                                  << strerror(errno) << std::endl;
                        connection_ok = false;
                        break;
                    }
                }

                std::cout << "Sent 2 MLP response packets (gelu + proj) for layer "
                          << packet.layer_index() << std::endl;

                // Skip the generic single-packet serialize/send below; both
                // response packets are already on the wire.
                continue;

            } else if (mode == "block") {
                // Full GPT-2 transformer block (Pre-LN ordering):
                //   residual = x; x_n = ln1(x); attn_out = attn(x_n);
                //   x = residual + attn_out                (float32 residual add)
                //   residual = x; x_n = ln2(x); mlp_out  = mlp(x_n);
                //   x = residual + mlp_out                 (float32 residual add)
                //
                // All matmuls use double accumulators (matmul_nt / attn_heads).
                // LayerNorm uses double accumulators for mean/var.
                // Residual adds are float32; they do NOT reset divergence the way
                // inter-stage float32 casts do -- measured max diff 3.81e-06 at T=8
                // (not 0.0).

                // G3: dispatch on layer_index. Out of range -> error reply (the
                // response already carries layer_index and device_id from above;
                // no data, no shape is the "rejected" contract, since a valid
                // reply always has shape [T, 768]). Connection and worker stay up.
                const int L = packet.layer_index();
                if (L < 0 || L >= N_BLOCKS) {
                    std::cout << "Rejecting packet: layer_index " << L
                              << " outside [0, " << N_BLOCKS - 1 << "]; sending error reply"
                              << std::endl;
                    std::string err_serialized;
                    if (!response.SerializeToString(&err_serialized)) {
                        std::cout << "Failed to serialize error reply" << std::endl;
                        connection_ok = false;
                        break;
                    }
                    if (!sendPacket(new_socket, err_serialized)) {
                        std::cout << "Failed to send error reply: " << strerror(errno) << std::endl;
                        connection_ok = false;
                        break;
                    }
                    continue;
                }
                const BlockWeights& w = blocks[L];

                // KV cache (sequence_id != 0). Every check runs before any cache
                // state changes; every rejection sends the no-shape error reply and
                // keeps the connection (the sequence_id == 0 path below is unchanged:
                // its shape/size errors still close the connection). Sizes in int64.
                const uint64_t seq = packet.sequence_id();
                const bool cached = (seq != 0);
                const int start_pos = packet.start_pos();
                if (cached) {
                    std::string why;
                    bool new_seq = false;
                    const int64_t T64 = (packet.shape_size() == 2) ? packet.shape(0) : -1;
                    if (packet.shape_size() != 2 || packet.shape(1) != HIDDEN_DIM) {
                        why = "KV shape: need [T," + std::to_string(HIDDEN_DIM) + "]";
                    } else if (T64 < 1 || T64 > KV_CAPACITY) {
                        why = "KV shape: T=" + std::to_string(T64) + " outside [1, "
                              + std::to_string(KV_CAPACITY) + "]";
                    } else if (static_cast<int64_t>(packet.data_size()) != T64 * HIDDEN_DIM) {
                        why = "KV size: data_size=" + std::to_string(packet.data_size())
                              + " != T*H=" + std::to_string(T64 * HIDDEN_DIM);
                    } else if (start_pos < 0 || static_cast<int64_t>(start_pos) + T64 > KV_CAPACITY) {
                        why = "KV bounds: start_pos=" + std::to_string(start_pos) + " T="
                              + std::to_string(T64) + " capacity " + std::to_string(KV_CAPACITY);
                    } else {
                        new_seq = (seq != kv.seq);
                        const int cur_len = new_seq ? 0 : kv.len[L];
                        if (new_seq && start_pos != 0) {
                            why = "KV sequence: new sequence " + std::to_string(seq)
                                  + " must start at start_pos 0, got " + std::to_string(start_pos);
                        } else if (start_pos != cur_len) {
                            why = "KV alignment: start_pos " + std::to_string(start_pos)
                                  + " != cached length " + std::to_string(cur_len)
                                  + " (layer " + std::to_string(L) + ")";
                        }
                    }
                    if (!why.empty()) {
                        std::cout << "Rejecting packet: " << why << "; sending error reply" << std::endl;
                        std::string err_serialized;
                        if (!response.SerializeToString(&err_serialized)) {
                            std::cout << "Failed to serialize error reply" << std::endl;
                            connection_ok = false;
                            break;
                        }
                        if (!sendPacket(new_socket, err_serialized)) {
                            std::cout << "Failed to send error reply: " << strerror(errno) << std::endl;
                            connection_ok = false;
                            break;
                        }
                        continue;
                    }
                    if (new_seq) {   // all checks passed: now (and only now) reset
                        kv.seq = seq;
                        std::fill(kv.len, kv.len + N_BLOCKS, 0);
                        std::cout << "KV cache reset for sequence " << seq << std::endl;
                    }
                }

                if (packet.shape_size() != 2) {
                    std::cout << "Rejecting packet: block mode requires shape_size=2"
                              << ", got shape_size=" << packet.shape_size() << std::endl;
                    connection_ok = false;
                    break;
                }
                int pkt_T = packet.shape(0);
                int pkt_H = packet.shape(1);
                if (pkt_H != HIDDEN_DIM) {
                    std::cout << "Rejecting packet: expected shape[1]=" << HIDDEN_DIM
                              << " (HIDDEN_DIM), got " << pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }
                if (packet.data_size() != pkt_T * pkt_H) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << " does not match T*H=" << pkt_T * pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }

                const size_t elems = static_cast<size_t>(pkt_T) * HIDDEN_DIM;

                // x is the running hidden state; starts as a copy of the input.
                std::vector<float> x(packet.data().data(),
                                     packet.data().data() + elems);
                std::vector<float> x_n(elems);

                // ---- Sub-block 1: attention sub-layer ----

                // Step 1a: ln_1 applied row-by-row (each token independent)
                for (int t = 0; t < pkt_T; t++) {
                    layernorm(x.data()   + t * HIDDEN_DIM,
                              w.ln1_gamma.data(), w.ln1_beta.data(),
                              x_n.data() + t * HIDDEN_DIM, HIDDEN_DIM);
                }

                // Step 1b: QKV projection  [T,768] -> [T,2304]
                std::vector<float> qkv_buf(static_cast<size_t>(pkt_T) * QKV_OUT_DIM);
                matmul_nt(x_n.data(), w.c_attn_W.data(), w.c_attn_b.data(),
                          qkv_buf.data(), pkt_T, QKV_IN_DIM, QKV_OUT_DIM);

                // Step 1c: multi-head causal attention  [T,2304] -> [T,768]
                std::vector<float> attn_out(elems);
                if (cached) {
                    // Append this packet's K and V rows at start_pos.., then attend
                    // over keys 0..p from the cache.
                    for (int t = 0; t < pkt_T; t++) {
                        const float* row = qkv_buf.data() + static_cast<size_t>(t) * QKV_OUT_DIM;
                        const size_t dst = (static_cast<size_t>(start_pos) + t) * HIDDEN_DIM;
                        std::copy(row + HIDDEN_DIM, row + 2 * HIDDEN_DIM, kv.K[L].begin() + dst);
                        std::copy(row + 2 * HIDDEN_DIM, row + 3 * HIDDEN_DIM, kv.V[L].begin() + dst);
                    }
                    kv.len[L] = start_pos + pkt_T;
                    attn_heads_cached(qkv_buf.data(), pkt_T, start_pos,
                                      kv.K[L].data(), kv.V[L].data(), attn_out.data(),
                                      ATTN_N_HEAD, ATTN_HEAD_DIM);
                } else {
                    attn_heads(qkv_buf.data(), attn_out.data(),
                               pkt_T, ATTN_N_HEAD, ATTN_HEAD_DIM);
                }

                // Step 1d: c_proj  [T,768] -> [T,768]
                std::vector<float> attn_proj(elems);
                matmul_nt(attn_out.data(), w.attn_proj_W.data(), w.attn_proj_b.data(),
                          attn_proj.data(), pkt_T, HIDDEN_DIM, HIDDEN_DIM);

                // Step 1e: first residual add (float32)
                for (size_t i = 0; i < elems; ++i)
                    x[i] += attn_proj[i];

                // ---- Sub-block 2: MLP sub-layer ----

                // Step 2a: ln_2 applied row-by-row
                for (int t = 0; t < pkt_T; t++) {
                    layernorm(x.data()   + t * HIDDEN_DIM,
                              w.ln2_gamma.data(), w.ln2_beta.data(),
                              x_n.data() + t * HIDDEN_DIM, HIDDEN_DIM);
                }

                // Step 2b: c_fc  [T,768] -> [T,3072]
                std::vector<float> fc_out(static_cast<size_t>(pkt_T) * MLP_HIDDEN_DIM);
                matmul_nt(x_n.data(), w.c_fc_W.data(), w.c_fc_b.data(),
                          fc_out.data(), pkt_T, HIDDEN_DIM, MLP_HIDDEN_DIM);

                // Step 2c: GELU (tanh approximation) elementwise
                for (float& v : fc_out) v = gelu(v);

                // Step 2d: c_proj  [T,3072] -> [T,768]
                std::vector<float> mlp_out(elems);
                matmul_nt(fc_out.data(), w.mlp_proj_W.data(), w.mlp_proj_b.data(),
                          mlp_out.data(), pkt_T, MLP_HIDDEN_DIM, HIDDEN_DIM);

                // Step 2e: second residual add (float32)
                for (size_t i = 0; i < elems; ++i)
                    x[i] += mlp_out[i];

                std::cout << "Computed block layer " << L << " [" << pkt_T << "," << HIDDEN_DIM
                          << "]  out[0:4]=["
                          << x[0] << "," << x[1] << "," << x[2] << "," << x[3]
                          << "] seq=" << seq << " start_pos=" << start_pos
                          << " cache_len=" << (cached ? kv.len[L] : 0) << std::endl;

                for (float v : x) response.add_data(v);
                response.add_shape(pkt_T);
                response.add_shape(HIDDEN_DIM);

            } else {  // attention
                // --- attention mode: validate 2-D input [T, HIDDEN_DIM] ---
                if (packet.shape_size() != 2) {
                    std::cout << "Rejecting packet: attention mode requires shape_size=2"
                              << ", got shape_size=" << packet.shape_size() << std::endl;
                    connection_ok = false;
                    break;
                }
                int pkt_T = packet.shape(0);
                int pkt_H = packet.shape(1);
                if (pkt_H != HIDDEN_DIM) {
                    std::cout << "Rejecting packet: expected shape[1]=" << HIDDEN_DIM
                              << " (HIDDEN_DIM), got " << pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }
                if (packet.data_size() != pkt_T * pkt_H) {
                    std::cout << "Rejecting packet: data_size=" << packet.data_size()
                              << " does not match shape[0]*shape[1]="
                              << pkt_T << "*" << pkt_H << "=" << pkt_T * pkt_H << std::endl;
                    connection_ok = false;
                    break;
                }

                const float* flat_in = packet.data().data();

                // Step 1: QKV projection -- x [T,768] @ W_attn [768,2304] + b -> [T,2304]
                std::vector<float> qkv_buf(static_cast<size_t>(pkt_T) * QKV_OUT_DIM);
                matmul_nt(flat_in, qkv_W.data(), qkv_b.data(),
                          qkv_buf.data(), pkt_T, QKV_IN_DIM, QKV_OUT_DIM);

                // Step 2: Multi-head attention -- [T,2304] -> [T,768]
                std::vector<float> attn_out(static_cast<size_t>(pkt_T) * HIDDEN_DIM);
                attn_heads(qkv_buf.data(), attn_out.data(),
                           pkt_T, ATTN_N_HEAD, ATTN_HEAD_DIM);

                // Structural check at T=1: softmax([s]) = 1.0 so attn_out = V per head.
                // V at token 0 starts at qkv_buf[2*HIDDEN_DIM] (offset into the V slice,
                // head 0 is the first HEAD_DIM values). At T=1 these must equal attn_out[0:4].
                if (pkt_T == 1) {
                    const float* V_h0 = qkv_buf.data() + 2 * HIDDEN_DIM;
                    std::cout << "T=1 structural check (attn_out == V per head):" << std::endl;
                    std::cout << "  V_h0    [0:4]: ["
                              << V_h0[0]     << ", " << V_h0[1]     << ", "
                              << V_h0[2]     << ", " << V_h0[3]     << "]" << std::endl;
                    std::cout << "  attn_out[0:4]: ["
                              << attn_out[0] << ", " << attn_out[1] << ", "
                              << attn_out[2] << ", " << attn_out[3] << "]" << std::endl;
                }

                // Step 3: c_proj -- attn_out [T,768] @ W_proj [768,768] + b -> [T,768]
                std::vector<float> proj_out(static_cast<size_t>(pkt_T) * HIDDEN_DIM);
                matmul_nt(attn_out.data(), attn_proj_W.data(), attn_proj_b.data(),
                          proj_out.data(), pkt_T, HIDDEN_DIM, HIDDEN_DIM);

                std::cout << "Computed attention [" << pkt_T << "," << HIDDEN_DIM
                          << "], proj_out[0:4]: ["
                          << proj_out[0] << ", " << proj_out[1] << ", "
                          << proj_out[2] << ", " << proj_out[3] << "]" << std::endl;

                for (float v : proj_out) response.add_data(v);
                response.add_shape(pkt_T);
                response.add_shape(HIDDEN_DIM);
            }

            std::string response_serialized;
            if (!response.SerializeToString(&response_serialized)) {
                std::cout << "Failed to serialize response for layer "
                          << packet.layer_index() << std::endl;
                connection_ok = false;
                break;
            }

            const long long compute_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_parsed).count();   // bench: window end
            if (!sendPacket(new_socket, response_serialized)) {
                std::cout << "Failed to send response: " << strerror(errno) << std::endl;
                connection_ok = false;
                break;
            }

            std::cout << "Sent response for layer " << packet.layer_index()
                      << " back to orchestrator" << std::endl;
            std::cout << "TIMING dev=" << packet.device_id() << " layer=" << packet.layer_index()
                      << " T=" << (packet.shape_size() > 0 ? packet.shape(0) : -1)
                      << " seq=" << packet.sequence_id() << " start_pos=" << packet.start_pos()
                      << " compute_us=" << compute_us << std::endl;
        }

        close(new_socket);
        std::cout << "Connection closed (" << packets_received << " packet(s) processed"
                  << (connection_ok ? "" : ", with error")
                  << "). Waiting for next connection..." << std::endl;
    }

    std::cout << "Shutdown signal received, no longer accepting connections." << std::endl;
    close(server_fd);
    return 0;
}
