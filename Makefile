# ==============================================================================
# Makefile — Project Hydra | Cross-Platform Build (WSL2 + macOS)
#
# Targets:
#   make proto        — regenerate hydra.pb.h / hydra.pb.cc from hydra.proto
#   make              — compile the orchestrator binary
#   make run          — compile and run orchestrator locally (WSL2 only)
#   make clean        — remove all build artifacts
#
# When you rename floatSum.cpp → orchestrator.cpp, update SRC below.
#
# Platform detection:
#   Linux (WSL2) — links with -lprotobuf
#   macOS        — links via pkg-config to pull in Abseil transitive deps
#                  required for protobuf v22+ (v7.34.1 on this Mac)
# ==============================================================================

CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
PROTOC   := protoc

# --- Platform detection -------------------------------------------------------
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    # macOS: protobuf v22+ requires Abseil; pkg-config pulls in all transitive deps
    LIBS := $(shell pkg-config --libs protobuf)
    CXX  := clang++
else
    # Linux / WSL2: system protobuf, direct link is sufficient
    LIBS := -lprotobuf
endif

# --- Source files -------------------------------------------------------------
# Currently floatSum.cpp. Rename this one line when you rename the file.
SRC        := orchestrator.cpp
PROTO_SRC  := hydra.pb.cc
PROTO_HDR  := hydra.pb.h
PROTO_DEF  := hydra.proto

# --- Output binary ------------------------------------------------------------
TARGET := orchestrator

# --- Targets ------------------------------------------------------------------

.PHONY: all proto run clean

all: $(TARGET)

# Compile the orchestrator binary
$(TARGET): $(SRC) $(PROTO_SRC) $(PROTO_HDR)
	$(CXX) $(CXXFLAGS) $(SRC) $(PROTO_SRC) $(LIBS) -o $(TARGET)
	@echo "[build] $(TARGET) ready (platform: $(UNAME))"

# Regenerate protobuf files from schema
# Run this whenever hydra.proto changes
# WARNING: always run on BOTH machines independently — never copy .pb files across
proto: $(PROTO_DEF)
	$(PROTOC) --cpp_out=. $(PROTO_DEF)
	@echo "[proto] regenerated $(PROTO_HDR) and $(PROTO_SRC)"

# Compile and immediately run (useful for local WSL2 testing)
run: $(TARGET)
	@echo "[run] launching $(TARGET)..."
	./$(TARGET)

# Remove all build artifacts
# Does NOT delete generated .pb files — those are regenerated per machine
clean:
	rm -f $(TARGET)
	@echo "[clean] done"
