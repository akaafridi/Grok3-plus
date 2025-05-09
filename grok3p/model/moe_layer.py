"""
Mixture of Experts (MoE) implementation for Grok-3+ architecture.
Based on the research paper: DOI 10.5281/zenodo.15341810
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp8_layer import FP8Linear


class MoERouter(nn.Module):
    """
    Router for Mixture of Experts with Top-K routing.
    Implements load balancing and auxiliary loss for balanced expert usage.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        use_fp8: bool = True,
        router_jitter_noise: float = 0.0,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_weight = aux_loss_weight
        self.router_jitter_noise = router_jitter_noise
        
        # Router projection
        Linear = FP8Linear if use_fp8 else nn.Linear
        self.router = Linear(hidden_size, num_experts, bias=False)
        
        # Initialize with zeros to start with uniform routing
        self.router.weight.data.zero_()
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing probabilities for each token.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tuple of (
                dispatch_weights: Tensor for dispatching to experts [batch, seq_len, num_experts_per_tok, hidden_size]
                combine_weights: Tensor for combining expert outputs [batch, seq_len, num_experts_per_tok, hidden_size]
                aux_loss: Load balancing auxiliary loss
            )
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        # Add noise during training to improve load balancing
        if self.training and self.router_jitter_noise > 0:
            noise = torch.rand_like(router_logits) * self.router_jitter_noise
            router_logits = router_logits + noise
            
        # Reshape for easier manipulation
        router_logits = router_logits.reshape(-1, self.num_experts)  # [batch*seq_len, num_experts]
        
        # Get routing probabilities with softmax
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_probs, self.num_experts_per_tok, dim=-1
        )  # [batch*seq_len, num_experts_per_tok]
        
        # Normalize the routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        # The goal is to encourage uniform expert assignment
        # Calculate fraction of tokens routed to each expert
        # Shape: [batch*seq_len, num_experts]
        router_prob_per_expert = torch.zeros_like(routing_probs)
        router_prob_per_expert.scatter_add_(
            -1, selected_experts, routing_weights
        )
        
        # Calculate auxiliary loss 
        # We want each expert to process approximately the same fraction of tokens
        aux_loss = torch.mean(
            router_prob_per_expert.sum(dim=0) * router_prob_per_expert.mean(dim=0)
        ) * self.num_experts * self.aux_loss_weight
        
        # Reshape for sparse dispatch
        selected_experts = selected_experts.reshape(batch_size, seq_len, self.num_experts_per_tok)
        routing_weights = routing_weights.reshape(batch_size, seq_len, self.num_experts_per_tok)
        
        # One-hot encoding for expert selection
        # Shape: [batch, seq_len, num_experts_per_tok, num_experts]
        expert_mask = torch.zeros(
            (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
            device=hidden_states.device
        )
        expert_mask.scatter_(-1, selected_experts.unsqueeze(-1), 1)
        
        # Combine weights for aggregating expert outputs
        # Shape: [batch, seq_len, num_experts_per_tok, 1]
        combine_weights = routing_weights.unsqueeze(-1)
        
        # Create dispatch mask to send tokens to the right experts
        # Shape: [batch*seq_len, num_experts_per_tok, hidden_size]
        dispatch_mask = expert_mask.reshape(-1, self.num_experts_per_tok, self.num_experts)
        
        return dispatch_mask, combine_weights, aux_loss


class MoEExpert(nn.Module):
    """Expert network in the Mixture of Experts layer."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_fp8: bool = True,
    ):
        super().__init__()
        Linear = FP8Linear if use_fp8 else nn.Linear
        
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single expert.
        
        Args:
            hidden_states: Input tensor [batch_size, hidden_size]
            
        Returns:
            Output tensor [batch_size, hidden_size]
        """
        # SwiGLU activation
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        activated = gate * up
        
        # Project back to hidden size
        return self.down_proj(activated)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with Top-K routing.
    Routes each token to the top K experts and combines their outputs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        use_fp8: bool = True,
        router_jitter_noise: float = 0.1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Create router
        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            use_fp8=use_fp8,
            router_jitter_noise=router_jitter_noise,
            aux_loss_weight=aux_loss_weight,
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                use_fp8=use_fp8,
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Mixture of Experts layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tuple of (output tensor, aux_loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing weights and auxiliary loss
        dispatch_mask, combine_weights, aux_loss = self.router(hidden_states)
        
        # Reshape for expert processing
        # [batch*seq_len, hidden_size]
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)
        
        # Expert outputs container
        # Shape: [batch*seq_len, num_experts_per_tok, hidden_size]
        expert_outputs = torch.zeros(
            (batch_size * seq_len, self.num_experts_per_tok, hidden_size),
            device=hidden_states.device
        )
        
        # For each expert, process the tokens assigned to it
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are assigned to this expert
            # [batch*seq_len, num_experts_per_tok]
            expert_mask = dispatch_mask[:, :, expert_idx].bool()
            
            # Skip if no tokens are assigned to this expert
            if not torch.any(expert_mask):
                continue
                
            # Get flattened indices for tokens assigned to this expert
            token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            
            # Select the right tokens for this expert
            # [num_tokens, hidden_size]
            expert_inputs = hidden_states_flat[token_indices]
            
            # Process tokens through this expert
            # [num_tokens, hidden_size]
            processed = expert(expert_inputs)
            
            # Create a mask to identify where to put the expert outputs
            # [batch*seq_len, num_experts_per_tok, 1]
            assignment_mask = expert_mask.unsqueeze(-1).float()
            
            # Assign the processed tokens back to the correct positions
            # This is a sparse update - we only update the positions
            # where tokens were assigned to this expert
            expert_outputs = expert_outputs + assignment_mask * processed.unsqueeze(1)
        
        # Combine expert outputs using the routing weights
        # [batch*seq_len, num_experts_per_tok, hidden_size] * [batch*seq_len, num_experts_per_tok, 1]
        combined_output = torch.sum(expert_outputs * combine_weights, dim=1)
        
        # Reshape back to original dimensions
        # [batch, seq_len, hidden_size]
        combined_output = combined_output.reshape(batch_size, seq_len, hidden_size)
        
        return combined_output, aux_loss
