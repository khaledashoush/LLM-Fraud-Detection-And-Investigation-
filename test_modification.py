# -*- coding: utf-8 -*-
"""Quick verification: FraudGTLayer modification is safe and functional."""

import torch
from train_fraudgt import FraudGTLayer, GT

print("=" * 50)
print("Testing FraudGTLayer modification...")
print("=" * 50)

# Setup — ★ FIX: metadata format matching train_fraudgt.py exactly
metadata = [('node',), (('node', 'to', 'node'), ('node', 'rev_to', 'node'))]
layer = FraudGTLayer(64, 8, metadata, 0)
layer.eval()  # eval mode (dropout = identity)

# Test inputs
N, E = 10, 5
h = {'node': torch.randn(N, 64)}
ei_to = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
ei_rev = ei_to.flip(0)
edge_index_dict = {
    ('node', 'to', 'node'): ei_to,
    ('node', 'rev_to', 'node'): ei_rev,
}
edge_attr_dict = {
    ('node', 'to', 'node'): torch.randn(E, 64),
    ('node', 'rev_to', 'node'): torch.randn(E, 64),
}

# ═══ TEST 1: Flag exists and is False by default ═══
assert hasattr(layer, 'capture_attention'), "❌ flag not found!"
assert layer.capture_attention == False, "❌ flag not False by default!"
assert layer._attention_capture is None, "❌ capture not None initially!"
print("✅ TEST 1 PASSED: flag exists, default=False, capture=None")

# ═══ TEST 2: Forward WITHOUT capture — output same as original ═══
with torch.no_grad():
    h_out1, ea_out1 = layer(h, edge_index_dict, edge_attr_dict)
assert layer._attention_capture is None, "❌ captured without flag!"
print("✅ TEST 2 PASSED: forward works, no capture when flag=False")

# ═══ TEST 3: Forward WITH capture — attention stored ═══
layer.capture_attention = True
with torch.no_grad():
    h_out2, ea_out2 = layer(h, edge_index_dict, edge_attr_dict)
assert layer._attention_capture is not None, "❌ no capture when flag=True!"
cap = layer._attention_capture
assert "attn" in cap, "❌ 'attn' key missing!"
assert "dst_flat" in cap, "❌ 'dst_flat' key missing!"
assert "src_flat" in cap, "❌ 'src_flat' key missing!"
assert "edge_offsets" in cap, "❌ 'edge_offsets' key missing!"
print(f"✅ TEST 3 PASSED: capture works | attn shape={cap['attn'].shape} "
      f"(H={cap['attn'].shape[0]}, E_total={cap['attn'].shape[1]})")

# ═══ TEST 4: Outputs IDENTICAL with and without capture ═══
assert torch.allclose(h_out1['node'], h_out2['node'], atol=1e-6), \
    "❌ node outputs differ!"
for et in edge_attr_dict:
    assert torch.allclose(ea_out1[et], ea_out2[et], atol=1e-6), \
        f"❌ edge outputs differ for {et}!"
print("✅ TEST 4 PASSED: outputs identical (capture is purely observational)")

# ═══ TEST 5: Attention softmax sanity (sums to 1 per dst) ═══
attn = cap['attn']  # [H, E_total]
dst_flat = cap['dst_flat']
for test_dst in [0, 1, 5]:
    mask = dst_flat == test_dst
    if mask.sum() > 0:
        s = attn[:, mask].sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
            f"❌ attention doesn't sum to 1 at dst={test_dst}!"
print("✅ TEST 5 PASSED: attention softmax sums to 1.0 per destination")

# ═══ TEST 6: edge_offsets correctly maps positions ═══
offsets = cap['edge_offsets']
assert ('node', 'to', 'node') in offsets, "❌ 'to' relation missing!"
assert ('node', 'rev_to', 'node') in offsets, "❌ 'rev_to' relation missing!"
to_start, to_end = offsets[('node', 'to', 'node')]
assert to_start == 0 and to_end == E, f"❌ 'to' offsets wrong: {offsets}"
print(f"✅ TEST 6 PASSED: edge_offsets correct: to=(0,{E}), "
      f"rev_to=({E},{2*E})")

# ═══ TEST 7: Disable capture → clean state ═══
layer.capture_attention = False
layer._attention_capture = None
with torch.no_grad():
    h_out3, ea_out3 = layer(h, edge_index_dict, edge_attr_dict)
assert layer._attention_capture is None, "❌ captured after disabling!"
assert torch.allclose(h_out1['node'], h_out3['node'], atol=1e-6), \
    "❌ outputs differ after disable!"
print("✅ TEST 7 PASSED: disable → clean state, same outputs")

print("\n" + "=" * 50)
print("🎉 ALL 7 TESTS PASSED!")
print("   The modification is safe and functional.")
print("   Ready to run: python train_fraudgt.py")
print("=" * 50)