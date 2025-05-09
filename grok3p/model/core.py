"""
Core model implementation for Grok-3+ architecture.
Based on the research paper: DOI 10.5281/zenodo.15341810
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fp8_layer import FP8Linear
from .moe_layer import MixtureOfExperts


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for better extrapolation properties."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Initialize cached embeddings
        self._init_cached_embeddings()
        
    def _init_cached_embeddings(self):
        """Initialize the cached embeddings to None."""
        self.cached_embeddings = None
        self.cached_max_seq_len = None
        
    def forward(self, seq_len: int, device: torch.device):
        """
        Generate positional embeddings for the given sequence length.
        
        Args:
            seq_len: Length of the sequence
            device: Device to create the embeddings on
            
        Returns:
            Tuple of (cos, sin) embeddings shapes [seq_len, dim/2]
        """
        # Reuse cached embeddings if possible
        if self.cached_embeddings is not None and seq_len <= self.cached_max_seq_len:
            return (
                self.cached_embeddings[0][:seq_len],
                self.cached_embeddings[1][:seq_len]
            )
        
        # Generate position indices
        positions = torch.arange(seq_len, device=device).float()
        
        # Calculate frequencies
        freqs = torch.outer(positions, self.inv_freq)
        
        # Calculate embeddings: (seq_len, dim/2) for both cos and sin
        cos_embeddings = torch.cos(freqs)
        sin_embeddings = torch.sin(freqs)
        
        # Update cache
        self.cached_embeddings = (cos_embeddings, sin_embeddings)
        self.cached_max_seq_len = seq_len
        
        return cos_embeddings, sin_embeddings


def apply_rotary_embeddings(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor.
    
    Args:
        x: Input tensor of shape [batch, seq_len, heads, head_dim]
        cos: Cosine embeddings of shape [seq_len, head_dim/2]
        sin: Sine embeddings of shape [seq_len, head_dim/2]
        
    Returns:
        Tensor with rotary embeddings applied
    """
    batch, seq_len, n_heads, head_dim = x.shape
    
    # Reshape x for easier manipulation
    x_reshaped = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
    
    # Extract even and odd indices
    x_even = x_reshaped[..., 0]
    x_odd = x_reshaped[..., 1]
    
    # Rotate by applying cos and sin
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
    
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos
    
    # Interleave the rotated values
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    
    # Reshape back to original dimensions
    return x_rotated.reshape(batch, seq_len, n_heads, head_dim)


class Grok3pAttention(nn.Module):
    """Multi-head attention with FP8 precision and rotary embeddings."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        rotary_dim: Optional[int] = None,
        use_fp8: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        
        # Determine rotary dimension
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        
        # Initialize rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_dim, 
            max_position_embeddings=max_position_embeddings
        )
        
        # Use FP8 linear layers for QKV projection
        Linear = FP8Linear if use_fp8 else nn.Linear
        self.qkv_proj = Linear(hidden_size, 3 * hidden_size)
        self.output_proj = Linear(hidden_size, hidden_size)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch, 1, 1, seq_len]
            
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input to query, key, value
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, batch, seq_len, num_heads, head_dim]
        
        # Split into query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings
        if self.rotary_dim > 0:
            cos, sin = self.rotary_emb(seq_len, hidden_states.device)
            q = apply_rotary_embeddings(q, cos, sin)
            k = apply_rotary_embeddings(k, cos, sin)
        
        # Scaled dot-product attention
        # Reshape for batch matrix multiplication
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Get context vector
        context = torch.matmul(attention_probs, v)
        
        # Reshape back to [batch, seq_len, hidden_size]
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        # Project to output
        output = self.output_proj(context)
        
        return output


class Grok3pMLP(nn.Module):
    """Feed-forward layer with FP8 precision."""
    
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation function.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        activated = gate * up
        
        # Project back to hidden size
        return self.down_proj(activated)


class Grok3pBlock(nn.Module):
    """Transformer block with attention, MLP/MoE, and layer normalization."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_norm_eps: float = 1e-5,
        use_fp8: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.use_moe = use_moe
        
        # Pre-attention layernorm
        self.pre_attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Self-attention
        self.attention = Grok3pAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_fp8=use_fp8,
        )
        
        # Pre-FFN layernorm
        self.pre_ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward
        if use_moe:
            self.feed_forward = MixtureOfExperts(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                use_fp8=use_fp8,
            )
        else:
            self.feed_forward = Grok3pMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                use_fp8=use_fp8,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a transformer block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            If use_moe is False:
                Output tensor [batch, seq_len, hidden_size]
            If use_moe is True:
                Tuple of (output tensor, aux_loss)
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.pre_ffn_norm(hidden_states)
        
        if self.use_moe:
            hidden_states, aux_loss = self.feed_forward(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, aux_loss
        else:
            hidden_states = self.feed_forward(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states


class Grok3pModel(nn.Module):
    """Grok-3+ model with FP8 precision and optional MoE."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
        use_fp8: bool = True,
        use_moe: bool = False,
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_token_id = pad_token_id
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Transformer blocks
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_use_moe = use_moe and (moe_layers is None or layer_idx in moe_layers)
            self.layers.append(
                Grok3pBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    layer_norm_eps=layer_norm_eps,
                    use_fp8=use_fp8,
                    use_moe=layer_use_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
            )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # LM head
        if tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = FP8Linear(hidden_size, vocab_size) if use_fp8 else nn.Linear(hidden_size, vocab_size)
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with small values to improve training stability."""
        if isinstance(module, (nn.Linear, FP8Linear)):
            # Use truncated normal initialization for larger models
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_input_embeddings(self):
        """Get word embeddings for token ids."""
        return self.word_embeddings
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the Grok-3+ model.
        
        Args:
            input_ids: Token ids [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_aux_loss: Whether to return auxiliary MoE loss
            
        Returns:
            If return_aux_loss is False:
                Output tensor [batch, seq_len, vocab_size]
            If return_aux_loss is True:
                Tuple of (output tensor, aux_loss)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
            
        # Extend attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get word embeddings
        hidden_states = self.word_embeddings(input_ids)
        
        # Track auxiliary loss
        aux_loss = 0.0
        
        # Process through transformer blocks
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                if isinstance(layer, Grok3pBlock) and layer.use_moe:
                    # For MoE layers, handle auxiliary loss
                    layer_output, layer_aux_loss = checkpoint(
                        layer,
                        hidden_states,
                        extended_attention_mask,
                    )
                    hidden_states = layer_output
                    aux_loss += layer_aux_loss
                else:
                    hidden_states = checkpoint(
                        layer,
                        hidden_states,
                        extended_attention_mask,
                    )
                    # If checkpoint returned a tuple (for MoE layers)
                    if isinstance(hidden_states, tuple):
                        hidden_states, layer_aux_loss = hidden_states
                        aux_loss += layer_aux_loss
            else:
                layer_output = layer(hidden_states, extended_attention_mask)
                # Check if layer returned auxiliary loss (for MoE layers)
                if isinstance(layer_output, tuple):
                    hidden_states, layer_aux_loss = layer_output
                    aux_loss += layer_aux_loss
                else:
                    hidden_states = layer_output
                    
        # Apply final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Apply language modeling head
        if self.lm_head is None:
            # Tied weights
            logits = F.linear(hidden_states, self.word_embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)
            
        if return_aux_loss:
            return logits, aux_loss
        else:
            return logits
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token ids [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Probability threshold for nucleus sampling
            repetition_penalty: Penalty for repetition
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token id
            eos_token_id: End-of-sequence token id
            
        Returns:
            Generated token ids [batch, max_length]
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
            
        batch_size = input_ids.shape[0]
        cur_length = input_ids.shape[1]
        device = input_ids.device
        
        # Create attention mask
        attention_mask = torch.ones((batch_size, cur_length), device=device)
        
        # Create array to track which sequences are already done
        if eos_token_id is not None:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Start generation
        while cur_length < max_length:
            # Forward pass to get logits
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for batch_idx in range(batch_size):
                        for prev_token in set(input_ids[batch_idx].tolist()):
                            next_token_logits[batch_idx, prev_token] /= repetition_penalty
                
                # Sampling
                if do_sample:
                    # Top-k sampling
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("inf")
                        
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for batch_idx in range(batch_size):
                            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                            next_token_logits[batch_idx, indices_to_remove] = -float("inf")
                            
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    
                # Update input_ids, attention_mask
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
                cur_length += 1
                
                # Check if we're done
                if eos_token_id is not None:
                    eos_in_sents = next_tokens == eos_token_id
                    if eos_in_sents.any():
                        unfinished_sequences[eos_in_sents] = 0
                    if unfinished_sequences.max() == 0:
                        break
                        
        return input_ids
