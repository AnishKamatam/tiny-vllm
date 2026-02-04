import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent directory to path to import layers
sys.path.insert(0, str(Path(__file__).parent.parent))
from layers.attention import Attention


def test_prefill():
    print("Testing prefill (full sequence, no cache)...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # Create attention module
    attn = Attention(hidden_size, num_heads, head_dim, bias=False)
    attn.eval()
    
    # Create random input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass through custom attention
    with torch.no_grad():
        output_custom, _ = attn(hidden_states, kv_cache=None, use_cache=False)
    
    # Forward pass using PyTorch's scaled_dot_product_attention
    with torch.no_grad():
        # Get QKV projections
        qkv = attn.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=2)
        
        # Reshape for multi-head attention
        B, T, _ = hidden_states.shape
        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, num_heads, head_dim).transpose(1, 2)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        
        # Use PyTorch's scaled_dot_product_attention
        output_pytorch = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            is_causal=False,  # We're providing explicit mask
        )
        
        # Reshape and apply output projection
        output_pytorch = output_pytorch.transpose(1, 2).contiguous()
        output_pytorch = output_pytorch.view(B, T, hidden_size)
        output_pytorch = attn.out_proj(output_pytorch)
    
    # Compare outputs
    max_diff = (output_custom - output_pytorch).abs().max().item()
    mean_diff = (output_custom - output_pytorch).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff:.2e}, expected < 1e-5"
    print("  ✓ Prefill test passed!\n")


def test_token_by_token_decode():
    print("Testing token-by-token decode with KV cache...")
    
    # Test parameters
    batch_size = 2
    prefill_len = 10
    decode_steps = 5
    hidden_size = 128
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # Create attention module
    attn = Attention(hidden_size, num_heads, head_dim, bias=False)
    attn.eval()
    
    # Step 1: Prefill (full sequence)
    prefill_states = torch.randn(batch_size, prefill_len, hidden_size)
    
    with torch.no_grad():
        # Custom attention prefill
        _, kv_cache = attn(prefill_states, kv_cache=None, use_cache=True)
        
        # Reference prefill using PyTorch
        qkv_ref = attn.qkv_proj(prefill_states)
        q_ref, k_ref, v_ref = qkv_ref.chunk(3, dim=2)
        B, T_prefill, _ = prefill_states.shape
        q_ref = q_ref.view(B, T_prefill, num_heads, head_dim).transpose(1, 2)
        k_ref = k_ref.view(B, T_prefill, num_heads, head_dim).transpose(1, 2)
        v_ref = v_ref.view(B, T_prefill, num_heads, head_dim).transpose(1, 2)
        
        causal_mask_ref = torch.tril(torch.ones(T_prefill, T_prefill, dtype=torch.bool))
        output_ref_prefill = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref,
            attn_mask=causal_mask_ref,
            is_causal=False,
        )
        output_ref_prefill = output_ref_prefill.transpose(1, 2).contiguous()
        output_ref_prefill = output_ref_prefill.view(B, T_prefill, hidden_size)
        output_ref_prefill = attn.out_proj(output_ref_prefill)
    
    # Step 2: Decode token-by-token
    all_outputs_custom = []
    all_outputs_reference = []
    
    # Store reference KV cache
    k_ref_cache = k_ref.clone()
    v_ref_cache = v_ref.clone()
    
    for step in range(decode_steps):
        # Single token input
        decode_token = torch.randn(batch_size, 1, hidden_size)
        
        with torch.no_grad():
            # Custom attention decode
            output_custom, kv_cache = attn(
                decode_token,
                kv_cache=kv_cache,
                use_cache=True
            )
            all_outputs_custom.append(output_custom)
            
            # Reference decode using PyTorch
            qkv_ref_decode = attn.qkv_proj(decode_token)
            q_ref_decode, k_ref_decode, v_ref_decode = qkv_ref_decode.chunk(3, dim=2)
            B, T_decode, _ = decode_token.shape
            q_ref_decode = q_ref_decode.view(B, T_decode, num_heads, head_dim).transpose(1, 2)
            k_ref_decode = k_ref_decode.view(B, T_decode, num_heads, head_dim).transpose(1, 2)
            v_ref_decode = v_ref_decode.view(B, T_decode, num_heads, head_dim).transpose(1, 2)
            
            # Concatenate with cache
            k_ref_full = torch.cat([k_ref_cache, k_ref_decode], dim=2)
            v_ref_full = torch.cat([v_ref_cache, v_ref_decode], dim=2)
            
            # Create mask: allow attention to all cached tokens + new token causally
            total_len = k_ref_cache.size(2) + T_decode
            causal_mask_decode = torch.tril(
                torch.ones(T_decode, total_len, dtype=torch.bool),
                diagonal=k_ref_cache.size(2)
            )
            
            output_ref_decode = F.scaled_dot_product_attention(
                q_ref_decode, k_ref_full, v_ref_full,
                attn_mask=causal_mask_decode,
                is_causal=False,
            )
            output_ref_decode = output_ref_decode.transpose(1, 2).contiguous()
            output_ref_decode = output_ref_decode.view(B, T_decode, hidden_size)
            output_ref_decode = attn.out_proj(output_ref_decode)
            all_outputs_reference.append(output_ref_decode)
            
            # Update reference cache
            k_ref_cache = k_ref_full
            v_ref_cache = v_ref_full
    
    # Compare all decode outputs
    max_diff_all = 0.0
    for i, (out_custom, out_ref) in enumerate(zip(all_outputs_custom, all_outputs_reference)):
        max_diff = (out_custom - out_ref).abs().max().item()
        mean_diff = (out_custom - out_ref).abs().mean().item()
        max_diff_all = max(max_diff_all, max_diff)
        print(f"  Decode step {i+1}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    print(f"  Overall max difference: {max_diff_all:.2e}")
    
    assert max_diff_all < 1e-5, f"Outputs differ by {max_diff_all:.2e}, expected < 1e-5"
    print("  ✓ Token-by-token decode test passed!\n")


def test_cache_consistency():
    """Test that cache is correctly maintained across steps"""
    print("Testing KV cache consistency...")
    
    batch_size = 2
    hidden_size = 128
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    attn = Attention(hidden_size, num_heads, head_dim, bias=False)
    attn.eval()
    
    # Prefill
    prefill = torch.randn(batch_size, 5, hidden_size)
    with torch.no_grad():
        _, cache1 = attn(prefill, kv_cache=None, use_cache=True)
    
    # Decode step 1
    token1 = torch.randn(batch_size, 1, hidden_size)
    with torch.no_grad():
        _, cache2 = attn(token1, kv_cache=cache1, use_cache=True)
    
    # Decode step 2
    token2 = torch.randn(batch_size, 1, hidden_size)
    with torch.no_grad():
        _, cache3 = attn(token2, kv_cache=cache2, use_cache=True)
    
    # Verify cache grows correctly
    assert cache1[0].size(2) == 5, "Prefill cache should have length 5"
    assert cache2[0].size(2) == 6, "After decode step 1, cache should have length 6"
    assert cache3[0].size(2) == 7, "After decode step 2, cache should have length 7"
    
    # Verify cache contains previous values
    assert torch.allclose(cache2[0][:, :, :5], cache1[0]), "Cache should preserve previous keys"
    assert torch.allclose(cache3[0][:, :, :6], cache2[0]), "Cache should preserve previous keys"
    
    print("  ✓ Cache consistency test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Attention Implementation")
    print("=" * 60 + "\n")
    
    try:
        test_prefill()
        test_token_by_token_decode()
        test_cache_consistency()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
