#include <iostream>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <cerrno>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sys/time.h>

#include "hydra.pb.h"

// send() may write fewer bytes than requested for larger payloads (the
// serialized packet can exceed the socket's send buffer), and can be
// interrupted by a signal mid-write. Loop until the whole buffer is on the wire.
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

// Frame each message with a 4-byte length prefix (network byte order) ahead
// of the serialized TensorPacket bytes, so the worker can recv() exactly one
// message at a time instead of relying on EOF to mark message boundaries.
bool sendPacket(int sock, const std::string& serialized) {
    uint32_t len = htonl(static_cast<uint32_t>(serialized.size()));
    if (sendAll(sock, reinterpret_cast<const char*>(&len), sizeof(len)) < 0) {
        return false;
    }
    return sendAll(sock, serialized.data(), serialized.size()) >= 0;
}

// Reads exactly `len` bytes into buf, looping over short reads and retrying
// on EINTR -- mirrors recvAll() on the worker side, since recv() is
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

// Reads one length-prefixed message and parses it into `out` -- the
// receive-side mirror of sendPacket(), applied here to the worker's
// response packets so the orchestrator can close the round trip.
// Returns false on any recv error, premature EOF, or parse failure.
bool recvPacket(int sock, hydra::TensorPacket& out) {
    uint32_t net_len;
    ssize_t r = recvAll(sock, reinterpret_cast<char*>(&net_len), sizeof(net_len));
    if (r <= 0) return false;

    uint32_t msg_len = ntohl(net_len);
    std::string buffer;
    buffer.resize(msg_len);

    r = recvAll(sock, buffer.data(), msg_len);
    if (r <= 0) return false;

    return out.ParseFromString(buffer);
}

// Reads exactly `count` float32 values from a raw little-endian binary file
// with no header. x86_64 is little-endian, so the bytes land directly into
// a float buffer with no byte-swap. A size mismatch is a hard error rather
// than a truncate/zero-pad -- a wrong-size input file would otherwise produce
// a packet that serializes fine but carries the wrong tensor.
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

// Write `count` float32 values to a raw little-endian binary file (no header).
// Used in layernorm mode to persist the worker's response so compare_ln.py
// can diff it against the PyTorch reference without parsing stdout.
bool saveFloatsToFile(const std::string& path, const hydra::TensorPacket& pkt) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cout << "Failed to open " << path << " for writing" << std::endl;
        return false;
    }
    for (int i = 0; i < pkt.data_size(); i++) {
        float v = pkt.data(i);
        f.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
    return f.good();
}

int main(int argc, char* argv[]) {
    // --mode linear    : MNIST fc1 forward pass (default, backward-compat)
    // --mode layernorm : GPT-2 block-0 ln_1 forward pass
    // --T N            : sequence length for layernorm mode (default 1)
    std::string mode = "linear";
    int T = 1;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (std::string(argv[i]) == "--T" && i + 1 < argc) {
            T = std::atoi(argv[++i]);
        }
    }
    if (mode != "linear" && mode != "layernorm" && mode != "qkv"
            && mode != "attention" && mode != "mlp" && mode != "block") {
        std::cout << "Unknown mode '" << mode
                  << "'. Use --mode linear|layernorm|qkv|attention|mlp|block"
                  << std::endl;
        return -1;
    }
    if ((mode == "layernorm" || mode == "qkv" || mode == "attention"
            || mode == "mlp" || mode == "block")
            && T < 1) {
        std::cout << "Invalid --T " << T << ": must be >= 1" << std::endl;
        return -1;
    }

    // Default disposition for SIGPIPE terminates the process the instant a
    // write hits a peer that has closed its end -- before send() can return
    // EPIPE. Ignore it so sendAll() observes the error via its return value
    // and errno instead of the process dying mid-send.
    signal(SIGPIPE, SIG_IGN);

    // Linear mode: MNIST fc1 (784 -> 128)
    constexpr int IN_DIM = 784;
    // LayerNorm mode: GPT-2 small hidden dim
    constexpr int HIDDEN_DIM = 768;

    // Hardening milestone: bound how long recv() will block waiting for the
    // worker's response. 5s is generous for a tiny layer over LAN but still
    // fails fast if the worker hangs.
    constexpr int RECV_TIMEOUT_SEC = 5;

    // Load input from disk. File and size depend on mode.
    std::vector<float> input_data;
    std::string input_path;
    size_t n_floats;

    if (mode == "linear") {
        input_path = "data/input.bin";
        n_floats   = IN_DIM;
    } else if (mode == "layernorm") {
        input_path = "data/ln_input_T" + std::to_string(T) + ".bin";
        n_floats   = static_cast<size_t>(T) * HIDDEN_DIM;
    } else if (mode == "qkv") {
        input_path = "data/qkv_input_T" + std::to_string(T) + ".bin";
        n_floats   = static_cast<size_t>(T) * HIDDEN_DIM;
    } else if (mode == "attention") {
        input_path = "data/attn_input_T" + std::to_string(T) + ".bin";
        n_floats   = static_cast<size_t>(T) * HIDDEN_DIM;
    } else if (mode == "mlp") {
        input_path = "data/mlp_input_T" + std::to_string(T) + ".bin";
        n_floats   = static_cast<size_t>(T) * HIDDEN_DIM;
    } else {  // block — raw token+pos embeddings, [T, 768]
        input_path = "data/block_input_T" + std::to_string(T) + ".bin";
        n_floats   = static_cast<size_t>(T) * HIDDEN_DIM;
    }

    if (!loadFloatsFromFile(input_path, n_floats, input_data)) {
        return -1;
    }

    hydra::TensorPacket packet;
    packet.set_layer_index(0);
    packet.set_device_id("orchestrator-wsl2");
    for (float v : input_data) {
        packet.add_data(v);
    }

    // Populate shape. Linear: [IN_DIM] (1-D, as before).
    // LayerNorm: [T, HIDDEN_DIM] (2-D) -- the proto field is identical; only
    // the number of dims populated here changes. The worker validates the
    // product matches data_size() for multi-dim tensors.
    if (mode == "linear") {
        packet.add_shape(IN_DIM);
    } else {
        packet.add_shape(T);
        packet.add_shape(HIDDEN_DIM);
    }

    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "Socket creation error" << std::endl;
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(50051);

    if (inet_pton(AF_INET, "10.0.0.62", &serv_addr.sin_addr) <= 0) {
        std::cout << "Invalid address / Address not supported" << std::endl;
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cout << "Connection Failed" << std::endl;
        return -1;
    }

    // Activation tensors go out as small, latency-sensitive sends. Without
    // TCP_NODELAY, Nagle's algorithm can hold a small packet waiting to
    // coalesce with more data, adding tens of ms of avoidable latency per hop.
    int nodelay = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
        std::cout << "Warning: failed to set TCP_NODELAY: " << strerror(errno) << std::endl;
    }

    // Hardening milestone: without SO_RCVTIMEO, recv() inside recvAll() blocks
    // forever if the worker hangs after receiving a packet but before replying.
    struct timeval recv_timeout;
    recv_timeout.tv_sec = RECV_TIMEOUT_SEC;
    recv_timeout.tv_usec = 0;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout)) < 0) {
        std::cout << "Warning: failed to set SO_RCVTIMEO: " << strerror(errno) << std::endl;
    }

    std::string serialized;
    if (!packet.SerializeToString(&serialized)) {
        std::cout << "Failed to serialize TensorPacket" << std::endl;
        close(sock);
        return -1;
    }

    std::cout << "Mode: " << mode;
    if (mode == "layernorm" || mode == "qkv" || mode == "attention"
            || mode == "mlp" || mode == "block")
        std::cout << "  T=" << T;
    std::cout << std::endl;
    std::cout << "Sending TensorPacket (" << serialized.size()
              << " bytes serialized), layer " << packet.layer_index()
              << ", " << packet.data_size() << " floats, shape [";
    for (int i = 0; i < packet.shape_size(); i++) {
        std::cout << packet.shape(i) << (i + 1 < packet.shape_size() ? "," : "");
    }
    std::cout << "]" << std::endl;

    if (!sendPacket(sock, serialized)) {
        std::cout << "Send failed: " << strerror(errno) << std::endl;
        close(sock);
        return -1;
    }

    // MLP mode: worker sends 2 packets (gelu intermediate, then proj output).
    // Handle before the generic single-packet path and return early.
    if (mode == "mlp") {
        auto recv_or_die = [&](hydra::TensorPacket& pkt, const char* label) -> bool {
            if (!recvPacket(sock, pkt)) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::cout << "Receive timed out waiting for " << label << std::endl;
                } else {
                    std::cout << "Failed to receive " << label
                              << ": " << strerror(errno) << std::endl;
                }
                return false;
            }
            return true;
        };

        // Packet 1: post-GELU intermediate [T, MLP_HIDDEN_DIM]
        hydra::TensorPacket gelu_resp;
        if (!recv_or_die(gelu_resp, "GELU response")) { close(sock); return -1; }
        std::cout << "GELU response:  layer " << gelu_resp.layer_index()
                  << " from '" << gelu_resp.device_id() << "', shape [";
        for (int i = 0; i < gelu_resp.shape_size(); i++)
            std::cout << gelu_resp.shape(i) << (i + 1 < gelu_resp.shape_size() ? "," : "");
        std::cout << "], " << gelu_resp.data_size() << " floats" << std::endl;
        std::string gelu_path = "data/mlp_response_gelu_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(gelu_path, gelu_resp)) {
            std::cout << "GELU intermediate written to " << gelu_path << std::endl;
        } else {
            std::cout << "Warning: failed to write " << gelu_path << std::endl;
        }

        // Packet 2: final c_proj output [T, HIDDEN_DIM]
        hydra::TensorPacket proj_resp;
        if (!recv_or_die(proj_resp, "proj response")) { close(sock); return -1; }
        std::cout << "Proj response:  layer " << proj_resp.layer_index()
                  << " from '" << proj_resp.device_id() << "', shape [";
        for (int i = 0; i < proj_resp.shape_size(); i++)
            std::cout << proj_resp.shape(i) << (i + 1 < proj_resp.shape_size() ? "," : "");
        const int max_print = 8;
        int n_print = proj_resp.data_size() < max_print ? proj_resp.data_size() : max_print;
        std::cout << "], data = [";
        for (int i = 0; i < n_print; i++)
            std::cout << proj_resp.data(i) << (i + 1 < n_print ? ", " : "");
        if (proj_resp.data_size() > max_print)
            std::cout << ", ... (" << proj_resp.data_size() << " total)";
        std::cout << "]" << std::endl;
        std::string proj_path = "data/mlp_response_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(proj_path, proj_resp)) {
            std::cout << "Final output written to " << proj_path << std::endl;
            std::cout << "Run: python compare_mlp.py " << T << std::endl;
        } else {
            std::cout << "Warning: failed to write " << proj_path << std::endl;
        }

        close(sock);
        return 0;
    }

    // All other modes: worker sends exactly one response packet.
    hydra::TensorPacket response;
    if (!recvPacket(sock, response)) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            std::cout << "Receive timed out after " << RECV_TIMEOUT_SEC
                      << "s waiting for worker response" << std::endl;
        } else {
            std::cout << "Failed to receive response: " << strerror(errno) << std::endl;
        }
        close(sock);
        return -1;
    }

    std::cout << "Received response for layer " << response.layer_index()
              << " from '" << response.device_id() << "', shape [";
    for (int i = 0; i < response.shape_size(); i++) {
        std::cout << response.shape(i) << (i + 1 < response.shape_size() ? "," : "");
    }
    // Print at most 8 values -- LayerNorm responses are 768-3072 floats and
    // don't fit usefully on a terminal. compare_ln.py diffs the binary files.
    const int max_print = 8;
    int n_print = response.data_size() < max_print ? response.data_size() : max_print;
    std::cout << "], data = [";
    for (int i = 0; i < n_print; i++) {
        std::cout << response.data(i) << (i + 1 < n_print ? ", " : "");
    }
    if (response.data_size() > max_print) {
        std::cout << ", ... (" << response.data_size() << " total)";
    }
    std::cout << "]" << std::endl;

    if (mode == "layernorm") {
        std::string out_path = "data/ln_response_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(out_path, response)) {
            std::cout << "Response written to " << out_path
                      << " -- run: python compare_ln.py " << T << std::endl;
        } else {
            std::cout << "Warning: failed to write " << out_path << std::endl;
        }
    } else if (mode == "qkv") {
        // Print first 4 values of Q, K, V from row 0 of the response for a
        // quick sanity-check before the compare script does the full diff.
        const int HEAD_DIM = 768;
        const char* head_names[3] = {"Q", "K", "V"};
        for (int split = 0; split < 3; split++) {
            int col_off = split * HEAD_DIM;
            std::cout << head_names[split] << " row0[0:4]: [";
            for (int j = 0; j < 4; j++) {
                std::cout << response.data(col_off + j);
                if (j < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::string out_path = "data/qkv_response_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(out_path, response)) {
            std::cout << "Response written to " << out_path
                      << " -- run: python compare_qkv.py " << T << std::endl;
        } else {
            std::cout << "Warning: failed to write " << out_path << std::endl;
        }
    } else if (mode == "attention") {
        std::string out_path = "data/attn_response_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(out_path, response)) {
            std::cout << "Response written to " << out_path
                      << " -- run: python compare_attn.py " << T << std::endl;
        } else {
            std::cout << "Warning: failed to write " << out_path << std::endl;
        }
    } else if (mode == "block") {
        std::string out_path = "data/block_response_T" + std::to_string(T) + ".bin";
        if (saveFloatsToFile(out_path, response)) {
            std::cout << "Response written to " << out_path
                      << " -- run: python compare_block.py " << T << std::endl;
        } else {
            std::cout << "Warning: failed to write " << out_path << std::endl;
        }
    } else {
        std::cout << "Compare against train_mnist_layer0.py's printed 'expected y = fc1(x)'."
                  << std::endl;
    }

    close(sock);
    return 0;
}
