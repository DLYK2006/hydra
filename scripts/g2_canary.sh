#!/usr/bin/env bash
# G2 canary. Hand-written; the agent must not edit this file. Run from repo root.
set -u
pass=0; fail=0
while read -r new old; do
  if cmp -s "data/$new.bin" "data/$old.bin"; then echo "MATCH $new == $old"; pass=$((pass+1))
  else echo "DIFF  $new != $old"; fail=$((fail+1)); fi
done <<'EOF'
h00_ln1_gamma ln_gamma
h00_ln1_beta ln_beta
h00_c_attn_W qkv_W
h00_c_attn_b qkv_b
h00_attn_proj_W attn_proj_W
h00_attn_proj_b attn_proj_b
h00_ln2_gamma block_ln2_gamma
h00_ln2_beta block_ln2_beta
h00_c_fc_W mlp_c_fc_W
h00_c_fc_b mlp_c_fc_b
h00_mlp_proj_W mlp_c_proj_W
h00_mlp_proj_b mlp_c_proj_b
EOF
n=$(ls data/h[0-9][0-9]_*.bin 2>/dev/null | wc -l)
echo "FILES $n/144"
size_bad=0
for f in data/h00_*.bin; do
  name=${f#data/h00_}; ref=$(stat -c %s "$f" 2>/dev/null || echo missing)
  for i in $(seq -w 1 11); do
    g="data/h${i}_${name}"
    if [ ! -f "$g" ] || [ "$(stat -c %s "$g")" != "$ref" ]; then echo "SIZE_BAD $g"; size_bad=$((size_bad+1)); fi
  done
done
echo "CANARY $pass/12 MATCH, $fail DIFF, $size_bad SIZE_BAD"
if [ "$fail" -eq 0 ] && [ "$n" -eq 144 ] && [ "$size_bad" -eq 0 ]; then echo "G2_CANARY_PASS"; else echo "G2_CANARY_FAIL"; exit 1; fi
