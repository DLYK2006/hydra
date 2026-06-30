# Skill: Protobuf Workflow

Read this before making any change to `hydra.proto` or any `.pb` file.

---

## When This Applies

- Adding, removing, or renaming any field in `hydra.proto`
- Changing a field type or field number
- Adding a new message type
- Anything that would touch `hydra.pb.h` or `hydra.pb.cc`

---

## The Core Rule

The two machines run different protobuf versions (WSL2: 3.21.12, Mac: v7.34.1).
Generated C++ files are version-specific. **Never copy `.pb.h` or `.pb.cc` between machines.**
Both machines must regenerate independently from the same `hydra.proto` source.

---

## Step-by-Step Procedure

### Step 1 — Edit the schema on WSL2 only

Edit `hydra.proto` on WSL2. This is the source of truth.
Do not edit the Mac copy directly.

```bash
# On WSL2
nano hydra.proto   # or vim, or whatever editor
```

Rules for safe edits:
- Never reuse a field number that has been deleted — protobuf uses numbers on the wire, not names
- Never change an existing field number — this breaks wire compatibility silently
- New fields only get new numbers

### Step 2 — Regenerate on WSL2

```bash
# On WSL2, in the hydra/ directory
make proto
```

This runs `protoc --cpp_out=. hydra.proto` and overwrites `hydra.pb.h` and `hydra.pb.cc`.

Verify it succeeded:

```bash
# Should show today's timestamp on both files
ls -lh hydra.pb.h hydra.pb.cc
```

### Step 3 — Build and sanity-check on WSL2

```bash
# On WSL2
make clean && make
```

Must compile with zero errors and zero warnings before touching the Mac.
Do not proceed if WSL2 build is broken.

### Step 4 — Ask permission before SSH

**Always ask the user before SSH-ing into the Mac. No exceptions.**

Once permission is granted:

```bash
ssh user@10.0.0.62
```

### Step 5 — Copy the proto source to the Mac

Copy only `hydra.proto` — never the generated files.

```bash
# On WSL2, before SSH or in a separate terminal
scp hydra.proto user@10.0.0.62:/path/to/hydra/hydra.proto
```

### Step 6 — Regenerate on the Mac

```bash
# On Mac, in the hydra/ directory
make proto
```

The Mac Makefile uses `pkg-config --libs protobuf` instead of `-lprotobuf` because
protobuf v22+ requires Abseil as a transitive dependency. Do not change this.

Verify:

```bash
# On Mac
ls -lh hydra.pb.h hydra.pb.cc
protoc --version   # should show libprotoc 27.x or similar
```

### Step 7 — Build on the Mac

```bash
# On Mac, in the hydra/ directory
make clean && make
```

Zero errors, zero warnings. If it fails, fix it on the Mac — do not modify the
generated files by hand.

---

## Known Failure Modes

**Copying `.pb.h` or `.pb.cc` across machines**
Symptom: build succeeds but crashes at runtime with protobuf internal errors,
or fails with ABI mismatch errors. Fix: delete the copied files, regenerate locally.

**Running `make proto` on only one machine**
Symptom: one side has a field the other doesn't. Serialization succeeds but
deserialized fields are zero/empty on the receiver. Fix: always regenerate on both.

**Forgetting `pkg-config` on macOS**
Symptom: linker error about missing Abseil symbols (`absl::...`).
Fix: confirm the Makefile macOS branch uses `$(shell pkg-config --libs protobuf)`.

**Reusing a deleted field number**
Symptom: silent data corruption — old clients read the wrong field type.
Fix: never reuse field numbers. Use `reserved` to tombstone them:
```protobuf
reserved 4;
reserved "old_field_name";
```

**Editing `hydra.pb.h` or `hydra.pb.cc` by hand**
These files are generated. Any hand edit is overwritten by the next `make proto`.
Fix: all schema changes go through `hydra.proto` only.

---

## Verification Checklist

Before declaring the schema change done:

- [ ] `hydra.proto` edited on WSL2
- [ ] `make proto && make` passes on WSL2 with zero warnings
- [ ] `hydra.proto` (only) copied to Mac via `scp`
- [ ] `make proto && make` passes on Mac with zero warnings
- [ ] New field is visible in both `hydra.pb.h` files (check with `grep <fieldname> hydra.pb.h`)
- [ ] A test send/receive confirms the new field survives the wire round-trip
