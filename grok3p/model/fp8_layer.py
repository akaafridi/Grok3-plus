"""
FP8 Linear layer implementation for memory and compute efficiency.
Based on the research paper: DOI 10.5281/zenodo.15341810
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8ScalingFactor(nn.Module):
    """
    FP8 scaling factors to manage dynamic range.
    Tracks and adjusts scaling factors for weights and activations.
    """
    
    def __init__(
        self,
        init_scale: float = 1.0,
        amax_history_len: int = 16,
        scale_update_freq: int = 1000,
    ):
        super().__init__()
        self.init_scale = init_scale
        self.amax_history_len = amax_history_len
        self.scale_update_freq = scale_update_freq
        
        # Register buffers for scale values
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
        self.register_buffer("amax_history", torch.zeros(amax_history_len, dtype=torch.float32))
        self.register_buffer("step", torch.zeros(1, dtype=torch.int64))
        
    def update_scale(self, tensor: torch.Tensor) -> None:
        """
        Update scaling factor based on the tensor values.
        
        Args:
            tensor: Input tensor for calculating the scale
        """
        if not self.training:
            return
            
        with torch.no_grad():
            # Calculate max absolute value
            amax = torch.max(torch.abs(tensor.detach())).float()
            
            # Update history
            idx = (self.step % self.amax_history_len).item()
            self.amax_history[idx] = amax
            
            # Update scale periodically
            if (self.step % self.scale_update_freq).item() == 0 and self.step > 0:
                valid_entries = torch.nonzero(self.amax_history).numel()
                if valid_entries > 0:
                    # Use max value to avoid underflow
                    max_amax = torch.max(self.amax_history[:valid_entries])
                    # Range of FP8 is [-448, 448]
                    self.scale = 448.0 / max_amax
            
            # Increment step
            self.step += 1
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling to input tensor.
        
        Args:
            tensor: Input tensor to scale
            
        Returns:
            Scaled tensor
        """
        return tensor * self.scale
    
    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse scaling for backward pass.
        
        Args:
            tensor: Input tensor to inverse scale
            
        Returns:
            Inverse scaled tensor
        """
        return tensor / self.scale


def quantize_to_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Simulate FP8 quantization for forward pass.
    This function quantizes the tensor to 8-bit and then scales it back.
    
    Args:
        tensor: Input tensor to quantize
        scale: Scaling factor to apply before quantization
        
    Returns:
        Quantized tensor in FP32 format
    """
    # Apply scaling
    scaled_tensor = tensor * scale
    
    # Clamp values to FP8 range [-448, 448]
    scaled_tensor = torch.clamp(scaled_tensor, -448.0, 448.0)
    
    # Simulate FP8 by rounding to the nearest 8-bit representation (e4m3)
    # e4m3 format: 1-bit sign, 4-bit exponent, 3-bit mantissa
    # We can simulate this by using torch.round with appropriate scaling
    
    # Find the exponent
    abs_tensor = torch.abs(scaled_tensor)
    exponent = torch.floor(torch.log2(abs_tensor + 1e-10))
    
    # Calculate the quantization step for the mantissa
    mantissa_step = torch.pow(2.0, exponent - 3)
    
    # Quantize
    quantized = torch.round(scaled_tensor / mantissa_step) * mantissa_step
    
    # Scale back
    dequantized = quantized / scale
    
    return dequantized


def dequantize_from_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Simulate FP8 dequantization for backward pass.
    
    Args:
        tensor: Input tensor to dequantize
        scale: Scaling factor used during quantization
        
    Returns:
        Dequantized tensor
    """
    # Since we're simulating, we just need to apply the inverse scaling
    return tensor / scale


class FP8Linear(nn.Module):
    """
    Linear layer with FP8 precision for forward and backward passes.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_enable: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_enable = fp8_enable
        
        # Create weights in fp32 format
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        # Initialize scaling factors for FP8
        if fp8_enable:
            self.input_scaling = FP8ScalingFactor()
            self.weight_scaling = FP8ScalingFactor()
            self.output_scaling = FP8ScalingFactor()
        
        # Initialize weights and biases
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize weights and biases using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 emulation.
        
        Args:
            input: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        if not self.fp8_enable:
            # Use regular linear layer if FP8 is disabled
            return F.linear(input, self.weight, self.bias)
        
        # Update scaling factors
        self.input_scaling.update_scale(input)
        self.weight_scaling.update_scale(self.weight)
        
        # Quantize input and weights to FP8
        fp8_input = quantize_to_fp8(input, self.input_scaling.scale)
        fp8_weight = quantize_to_fp8(self.weight, self.weight_scaling.scale)
        
        # Compute matrix multiplication
        output = F.linear(fp8_input, fp8_weight, bias=None)
        
        # Apply inverse scaling
        output = output / (self.input_scaling.scale * self.weight_scaling.scale)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        # Update output scaling factor
        self.output_scaling.update_scale(output)
        
        return output
