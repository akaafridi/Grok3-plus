#!/usr/bin/env python
"""
Benchmark comparing FP8 vs FP16 vs FP32 performance.
DOI: 10.5281/zenodo.15341810
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from grok3p.model import Grok3pModel, FP8Linear
from grok3p.config import load_config


@dataclass
class BenchmarkArguments:
    """Arguments for benchmarking."""
    
    output_dir: str = field(
        default="./results/benchmarks",
        metadata={"help": "Directory to save benchmark results"}
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16],
        metadata={"help": "Batch sizes to benchmark"}
    )
    seq_lengths: List[int] = field(
        default_factory=lambda: [128, 512, 1024, 2048],
        metadata={"help": "Sequence lengths to benchmark"}
    )
    hidden_sizes: List[int] = field(
        default_factory=lambda: [1024, 2048, 4096],
        metadata={"help": "Hidden sizes to benchmark"}
    )
    num_layers: int = field(
        default=2,
        metadata={"help": "Number of layers to use for the benchmark models"}
    )
    warmup_steps: int = field(
        default=5,
        metadata={"help": "Number of warmup steps"}
    )
    test_steps: int = field(
        default=10,
        metadata={"help": "Number of test steps to average over"}
    )
    precision_modes: List[str] = field(
        default_factory=lambda: ["fp8", "fp16", "fp32"],
        metadata={"help": "Precision modes to benchmark"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run benchmarks on"}
    )
    compile: bool = field(
        default=False,
        metadata={"help": "Whether to use torch.compile for benchmarks"}
    )


def create_benchmark_model(hidden_size, precision_mode, num_layers=2, num_heads=8):
    """
    Create a model for benchmarking with the specified parameters.
    
    Args:
        hidden_size: Hidden size for the model
        precision_mode: One of "fp8", "fp16", "fp32"
        num_layers: Number of layers in the model
        num_heads: Number of attention heads
        
    Returns:
        Model with the specified configuration
    """
    use_fp8 = precision_mode == "fp8"
    
    model = Grok3pModel(
        vocab_size=32000,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=4096,
        use_fp8=use_fp8,
        use_moe=False,
        gradient_checkpointing=False,
    )
    
    return model


def benchmark_forward(model, batch_size, seq_len, hidden_size, precision_mode, device, warmup=5, steps=10, compile=False):
    """
    Benchmark forward pass performance.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size to use
        seq_len: Sequence length to use
        hidden_size: Hidden size of the model
        precision_mode: One of "fp8", "fp16", "fp32"
        device: Device to run on
        warmup: Number of warmup steps
        steps: Number of test steps
        compile: Whether to use torch.compile
        
    Returns:
        Dictionary with benchmark results
    """
    model = model.to(device)
    
    # Use torch.compile if requested (requires PyTorch 2.0+)
    if compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    # Generate random input data
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    
    # Trace memory usage
    if precision_mode == "fp16":
        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Record memory usage before model runs
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            if precision_mode == "fp16":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids, attention_mask)
    
    # Benchmark forward pass
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(steps):
            if precision_mode == "fp16":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids, attention_mask)
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate tokens per second
    total_tokens = batch_size * seq_len * steps
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    # Record peak memory usage
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated()
        memory_usage = (peak_mem - start_mem) / (1024 ** 2)  # MB
    else:
        memory_usage = 0
    
    results = {
        "precision_mode": precision_mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "tokens_per_second": tokens_per_second,
        "latency_ms": (elapsed_time / steps) * 1000,
        "memory_usage_mb": memory_usage,
    }
    
    return results


def benchmark_backward(model, batch_size, seq_len, hidden_size, precision_mode, device, warmup=5, steps=10, compile=False):
    """
    Benchmark backward pass performance.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size to use
        seq_len: Sequence length to use
        hidden_size: Hidden size of the model
        precision_mode: One of "fp8", "fp16", "fp32"
        device: Device to run on
        warmup: Number of warmup steps
        steps: Number of test steps
        compile: Whether to use torch.compile
        
    Returns:
        Dictionary with benchmark results
    """
    model = model.to(device)
    model.train()
    
    # Use torch.compile if requested (requires PyTorch 2.0+)
    if compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    # Generate random input data
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    # Trace memory usage
    if precision_mode == "fp16":
        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Record memory usage before model runs
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        
        if precision_mode == "fp16":
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD(model.parameters(), lr=0.1))
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            
            loss.backward()
            torch.optim.SGD(model.parameters(), lr=0.1).step()
    
    # Benchmark backward pass
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(steps):
        model.zero_grad()
        
        if precision_mode == "fp16":
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD(model.parameters(), lr=0.1))
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            
            loss.backward()
            torch.optim.SGD(model.parameters(), lr=0.1).step()
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate tokens per second
    total_tokens = batch_size * seq_len * steps
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    # Record peak memory usage
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated()
        memory_usage = (peak_mem - start_mem) / (1024 ** 2)  # MB
    else:
        memory_usage = 0
    
    results = {
        "precision_mode": precision_mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "tokens_per_second": tokens_per_second,
        "latency_ms": (elapsed_time / steps) * 1000,
        "memory_usage_mb": memory_usage,
    }
    
    return results


def plot_results(results, output_dir, plot_type="tokens_per_second"):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save plots
        plot_type: Type of plot to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier processing
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Plot tokens per second by precision mode and hidden size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for precision in df["precision_mode"].unique():
        data = df[df["precision_mode"] == precision]
        ax.plot(
            data["hidden_size"], 
            data["tokens_per_second"],
            marker="o",
            label=precision
        )
    
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Tokens Per Second")
    ax.set_title("Performance Comparison: FP8 vs FP16 vs FP32")
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_type}_by_hidden_size.png"))
    plt.close()
    
    # Plot memory usage by precision mode and hidden size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for precision in df["precision_mode"].unique():
        data = df[df["precision_mode"] == precision]
        ax.plot(
            data["hidden_size"], 
            data["memory_usage_mb"],
            marker="o",
            label=precision
        )
    
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Usage Comparison: FP8 vs FP16 vs FP32")
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_by_hidden_size.png"))
    plt.close()


def main():
    """Main benchmark function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FP8 vs FP16 vs FP32 benchmark")
    
    parser.add_argument("--output-dir", type=str, default="./results/benchmarks",
                        help="Directory to save benchmark results")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[1024, 2048, 4096],
                        help="Hidden sizes to benchmark")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers to use for the benchmark models")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--test-steps", type=int, default=10,
                        help="Number of test steps to average over")
    parser.add_argument("--precision-modes", type=str, nargs="+", default=["fp8", "fp16", "fp32"],
                        help="Precision modes to benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmarks on")
    parser.add_argument("--forward-only", action="store_true",
                        help="Only benchmark forward pass")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for benchmarks")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for different hidden sizes and precision modes
    all_results = []
    
    # Benchmark forward pass
    print("Benchmarking forward pass...")
    for hidden_size in args.hidden_sizes:
        for precision_mode in args.precision_modes:
            print(f"Testing {precision_mode} with hidden size {hidden_size}...")
            
            model = create_benchmark_model(
                hidden_size,
                precision_mode,
                num_layers=args.num_layers,
            )
            
            for batch_size in args.batch_sizes:
                for seq_len in args.seq_lengths:
                    if batch_size * seq_len > 32768 and args.device.startswith("cuda"):
                        # Skip configurations that might cause OOM
                        continue
                    
                    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
                    
                    result = benchmark_forward(
                        model,
                        batch_size,
                        seq_len,
                        hidden_size,
                        precision_mode,
                        args.device,
                        warmup=args.warmup_steps,
                        steps=args.test_steps,
                        compile=args.compile,
                    )
                    
                    result["pass_type"] = "forward"
                    all_results.append(result)
                    
                    print(f"    Tokens/sec: {result['tokens_per_second']:.2f}, "
                          f"Latency: {result['latency_ms']:.2f} ms, "
                          f"Memory: {result['memory_usage_mb']:.2f} MB")
    
    # Benchmark backward pass if requested
    if not args.forward_only:
        print("\nBenchmarking backward pass...")
        for hidden_size in args.hidden_sizes:
            for precision_mode in args.precision_modes:
                print(f"Testing {precision_mode} with hidden size {hidden_size}...")
                
                model = create_benchmark_model(
                    hidden_size,
                    precision_mode,
                    num_layers=args.num_layers,
                )
                
                for batch_size in args.batch_sizes:
                    for seq_len in args.seq_lengths:
                        if batch_size * seq_len > 16384 and args.device.startswith("cuda"):
                            # Skip configurations that might cause OOM
                            continue
                        
                        print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
                        
                        result = benchmark_backward(
                            model,
                            batch_size,
                            seq_len,
                            hidden_size,
                            precision_mode,
                            args.device,
                            warmup=args.warmup_steps,
                            steps=args.test_steps,
                            compile=args.compile,
                        )
                        
                        result["pass_type"] = "backward"
                        all_results.append(result)
                        
                        print(f"    Tokens/sec: {result['tokens_per_second']:.2f}, "
                              f"Latency: {result['latency_ms']:.2f} ms, "
                              f"Memory: {result['memory_usage_mb']:.2f} MB")
    
    # Save results to file
    with open(os.path.join(args.output_dir, "fp8_vs_fp16_benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(
        [r for r in all_results if r["pass_type"] == "forward"],
        args.output_dir,
        plot_type="forward_tokens_per_second"
    )
    
    if not args.forward_only:
        plot_results(
            [r for r in all_results if r["pass_type"] == "backward"],
            args.output_dir,
            plot_type="backward_tokens_per_second"
        )
    
    print(f"\nAll benchmark results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
