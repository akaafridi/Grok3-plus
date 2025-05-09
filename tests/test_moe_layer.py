"""
Tests for the Mixture of Experts layer.
DOI: 10.5281/zenodo.15341810
"""

import pytest
import torch
import torch.nn.functional as F

from grok3p.model.moe_layer import MixtureOfExperts, MoERouter


def test_moe_router_output_shapes():
    """Test that the MoE router produces outputs with the correct shapes."""
    batch_size = 4
    seq_len = 8
    hidden_size = 32
    num_experts = 8
    num_experts_per_tok = 2
    
    # Create router
    router = MoERouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=False,
    )
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    dispatch_mask, combine_weights, aux_loss = router(hidden_states)
    
    # Check shapes
    assert dispatch_mask.shape == (batch_size * seq_len, num_experts_per_tok, num_experts), \
        f"Expected dispatch_mask shape {(batch_size * seq_len, num_experts_per_tok, num_experts)}, got {dispatch_mask.shape}"
    assert combine_weights.shape == (batch_size, seq_len, num_experts_per_tok, 1), \
        f"Expected combine_weights shape {(batch_size, seq_len, num_experts_per_tok, 1)}, got {combine_weights.shape}"
    assert aux_loss.shape == (), f"Expected aux_loss to be a scalar, got shape {aux_loss.shape}"
    
    # Check that routing weights sum to 1 for each token
    assert torch.all(torch.isclose(combine_weights.sum(dim=2), torch.ones_like(combine_weights[:, :, 0, :]))), \
        "Routing weights should sum to 1 for each token"


def test_moe_layer_output_shapes():
    """Test that the MoE layer produces outputs with the correct shapes."""
    batch_size = 4
    seq_len = 8
    hidden_size = 32
    intermediate_size = 128
    num_experts = 8
    num_experts_per_tok = 2
    
    # Create MoE layer
    moe_layer = MixtureOfExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=False,
    )
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output, aux_loss = moe_layer(hidden_states)
    
    # Check shapes
    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected output shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert aux_loss.shape == (), f"Expected aux_loss to be a scalar, got shape {aux_loss.shape}"


def test_moe_expert_balance():
    """Test that the MoE router distributes tokens evenly across experts."""
    batch_size = 32
    seq_len = 64
    hidden_size = 64
    num_experts = 8
    num_experts_per_tok = 2
    
    # Create router
    router = MoERouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=False,
        # Set jitter to 0 for deterministic testing
        router_jitter_noise=0.0,
    )
    
    # Initialize router weights with zeros for uniform routing
    router.router.weight.data.zero_()
    
    # Create input
    torch.manual_seed(42)  # Set seed for reproducibility
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    dispatch_mask, combine_weights, aux_loss = router(hidden_states)
    
    # Check expert assignment
    # Flatten batch and sequence dimensions
    dispatch_mask_flat = dispatch_mask.reshape(-1, num_experts_per_tok, num_experts)
    
    # Count tokens assigned to each expert
    expert_counts = torch.sum(dispatch_mask_flat, dim=(0, 1))
    total_tokens_per_expert = batch_size * seq_len * num_experts_per_tok / num_experts
    
    # Calculate balance factor: should be close to 1.0 for perfectly balanced routing
    balance = expert_counts / total_tokens_per_expert
    
    # We expect the balance to be approximately 1.0 for each expert, 
    # but allow for some variation due to randomization
    # The paper requires the balance to be within 0.5-0.05 range, so we check that
    assert torch.all(balance >= 0.45) and torch.all(balance <= 1.55), \
        f"Expert balance out of expected range (0.45-1.55): {balance}"
    
    # Calculate coefficient of variation (CV) to measure the spread
    # For a well-balanced system, this should be low
    cv = torch.std(balance) / torch.mean(balance)
    
    # Expect CV to be reasonably low (typically < 0.2 for well-balanced)
    assert cv < 0.3, f"Coefficient of variation too high: {cv}"


def test_moe_forward_backward():
    """Test the forward and backward passes of the MoE layer."""
    batch_size = 4
    seq_len = 8
    hidden_size = 32
    intermediate_size = 128
    num_experts = 4
    num_experts_per_tok = 2
    
    # Create MoE layer
    moe_layer = MixtureOfExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=False,  # Use FP32 for reliable gradient testing
    )
    
    # Create input with requires_grad=True for backward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    # Forward pass
    output, aux_loss = moe_layer(hidden_states)
    
    # Check that output requires gradients
    assert output.requires_grad, "Output should require gradients"
    
    # Create dummy loss and run backward pass
    loss = output.sum() + aux_loss
    loss.backward()
    
    # Check that input gradients exist
    assert hidden_states.grad is not None, "Input gradients should exist"
    assert not torch.isnan(hidden_states.grad).any(), "Input gradients should not contain NaN values"
    assert not torch.isinf(hidden_states.grad).any(), "Input gradients should not contain Inf values"


def test_moe_with_identical_experts():
    """Test that the MoE layer works correctly when all experts are identical."""
    batch_size = 4
    seq_len = 8
    hidden_size = 32
    intermediate_size = 128
    num_experts = 4
    num_experts_per_tok = 2
    
    # Create MoE layer
    moe_layer = MixtureOfExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=False,
    )
    
    # Make all experts identical by setting their weights to the same values
    with torch.no_grad():
        expert0 = moe_layer.experts[0]
        for expert in moe_layer.experts[1:]:
            expert.up_proj.weight.copy_(expert0.up_proj.weight)
            expert.up_proj.bias.copy_(expert0.up_proj.bias)
            expert.gate_proj.weight.copy_(expert0.gate_proj.weight)
            expert.gate_proj.bias.copy_(expert0.gate_proj.bias)
            expert.down_proj.weight.copy_(expert0.down_proj.weight)
            expert.down_proj.bias.copy_(expert0.down_proj.bias)
    
    # Create two identical inputs
    hidden_states1 = torch.randn(batch_size, seq_len, hidden_size)
    hidden_states2 = hidden_states1.clone()
    
    # Forward pass with the two inputs
    output1, aux_loss1 = moe_layer(hidden_states1)
    output2, aux_loss2 = moe_layer(hidden_states2)
    
    # Check that outputs are identical
    assert torch.allclose(output1, output2), "Outputs should be identical for identical inputs and experts"
    assert torch.allclose(aux_loss1, aux_loss2), "Auxiliary losses should be identical"


def test_top_k_routing():
    """Test that the top-k routing mechanism selects the top k experts for each token."""
    batch_size = 4
    seq_len = 8
    hidden_size = 32
    num_experts = 8
    k_values = [1, 2, 4]
    
    # Create a fixed input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    for k in k_values:
        # Create router
        router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=k,
            use_fp8=False,
            router_jitter_noise=0.0,  # Disable jitter for deterministic testing
        )
        
        # Calculate logits manually
        logits = router.router(hidden_states)
        expected_topk_indices = torch.topk(logits, k, dim=-1)[1]
        
        # Get router outputs
        dispatch_mask, combine_weights, _ = router(hidden_states)
        
        # Extract selected expert indices from dispatch_mask
        # First, reshape dispatch_mask to match expected_topk_indices
        dispatch_mask_reshaped = dispatch_mask.view(batch_size, seq_len, k, num_experts)
        
        # For each token, find which experts were selected
        selected_experts = torch.zeros((batch_size, seq_len, k), dtype=torch.long)
        for b in range(batch_size):
            for s in range(seq_len):
                for i in range(k):
                    # Find the index of the 1 in each row
                    selected_experts[b, s, i] = torch.nonzero(dispatch_mask_reshaped[b, s, i])[0]
        
        # Sort both the expected and actual indices for comparison
        expected_indices_sorted, _ = torch.sort(expected_topk_indices, dim=-1)
        actual_indices_sorted, _ = torch.sort(selected_experts, dim=-1)
        
        # Check that the selected experts are the top k
        assert torch.all(expected_indices_sorted == actual_indices_sorted), \
            f"For k={k}, the selected experts are not the top k"


if __name__ == "__main__":
    # Run tests
    test_moe_router_output_shapes()
    test_moe_layer_output_shapes()
    test_moe_expert_balance()
    test_moe_forward_backward()
    test_moe_with_identical_experts()
    test_top_k_routing()
    
    print("All tests passed!")
