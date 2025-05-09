#!/usr/bin/env python
"""
Benchmark for scaling the number of experts in Mixture-of-Experts layers.
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

from grok3p.model import MixtureOfExperts
from grok3p.config import load_config


@dataclass
class BenchmarkArguments:
    """Arguments for MoE scaling benchmark."""
    
    output_dir: str = field(
        default="./results/benchmarks",
        metadata={"help": "Directory to save benchmark results"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size to use for the benchmark"}
    )
    seq_length: int = field(
        default=512,
        metadata={"help": "Sequence length to use for the benchmark"}
    )
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Hidden size to use for the benchmark"}
    )
    intermediate_size: int = field(
        default=4096,
        metadata={"help": "Intermediate size for the MoE FFN layers"}
    )
    expert_counts: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32],
        metadata={"help": "Number of experts to benchmark"}
    )
    experts_per_token: List[int] = field(
        default_factory=lambda: [1, 2, 4],
        metadata={"help": "Number of experts per token to benchmark"}
    )
    warmup_steps: int = field(
        default=5,
        metadata={"help": "Number of warmup steps"}
    )
    test_steps: int = field(
        default=10,
        metadata={"help": "Number of test steps to average over"}
    )
    use_fp8: bool = field(
        default=True,
        metadata={"help": "Whether to use FP8 precision"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run benchmarks on"}
    )


def benchmark_moe_scaling(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    batch_size: int,
    seq_length: int,
    use_fp8: bool,
    device: str,
    warmup_steps: int = 5,
    test_steps: int = 10,
):
    """
    Benchmark MoE layer with different numbers of experts.
    
    Args:
        hidden_size: Hidden size of the model
        intermediate_size: Intermediate size of the FFN layers
        num_experts: Number of experts in the MoE layer
        num_experts_per_tok: Number of experts to route each token to
        batch_size: Batch size for the benchmark
        seq_length: Sequence length for the benchmark
        use_fp8: Whether to use FP8 precision
        device: Device to run on
        warmup_steps: Number of warmup steps
        test_steps: Number of test steps
        
    Returns:
        Dictionary with benchmark results
    """
    # Create MoE layer
    moe_layer = MixtureOfExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_fp8=use_fp8,
    ).to(device)
    
    # Generate random input data
    input_data = torch.randn(batch_size, seq_length, hidden_size, device=device)
    
    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = moe_layer(input_data)
    
    # Record memory usage
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    # Benchmark forward pass
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(test_steps):
        with torch.no_grad():
            _ = moe_layer(input_data)
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate tokens per second
    total_tokens = batch_size * seq_length * test_steps
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    # Record memory usage
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated()
        memory_usage = (peak_mem - start_mem) / (1024 ** 2)  # MB
    else:
        memory_usage = 0
    
    # Gather results
    results = {
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "use_fp8": use_fp8,
        "tokens_per_second": tokens_per_second,
        "latency_ms": (elapsed_time / test_steps) * 1000,
        "memory_usage_mb": memory_usage,
    }
    
    return results


def plot_results(results, output_dir):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier processing
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Create plot for tokens per second vs number of experts
    for experts_per_tok in df["num_experts_per_tok"].unique():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = df[df["num_experts_per_tok"] == experts_per_tok]
        ax.plot(
            data["num_experts"],
            data["tokens_per_second"],
            marker="o",
            label=f"Top-{experts_per_tok} routing"
        )
        
        ax.set_xlabel("Number of Experts")
        ax.set_ylabel("Tokens Per Second")
        ax.set_title(f"MoE Scaling: Performance with Top-{experts_per_tok} Routing")
        ax.grid(True)
        
        if max(data["num_experts"]) > 8:
            ax.set_xscale("log", base=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"moe_scaling_top{experts_per_tok}.png"))
        plt.close()
    
    # Create plot comparing different top-k routing strategies
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for experts_per_tok in sorted(df["num_experts_per_tok"].unique()):
        data = df[df["num_experts_per_tok"] == experts_per_tok]
        
        ax.plot(
            data["num_experts"],
            data["tokens_per_second"],
            marker="o",
            label=f"Top-{experts_per_tok} routing"
        )
    
    ax.set_xlabel("Number of Experts")
    ax.set_ylabel("Tokens Per Second")
    ax.set_title("MoE Scaling: Performance with Different Routing Strategies")
    ax.legend()
    ax.grid(True)
    
    if max(df["num_experts"]) > 8:
        ax.set_xscale("log", base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "moe_scaling_comparison.png"))
    plt.close()
    
    # Create plot for memory usage
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for experts_per_tok in sorted(df["num_experts_per_tok"].unique()):
        data = df[df["num_experts_per_tok"] == experts_per_tok]
        
        ax.plot(
            data["num_experts"],
            data["memory_usage_mb"],
            marker="o",
            label=f"Top-{experts_per_tok} routing"
        )
    
    ax.set_xlabel("Number of Experts")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("MoE Scaling: Memory Usage")
    ax.legend()
    ax.grid(True)
    
    if max(df["num_experts"]) > 8:
        ax.set_xscale("log", base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "moe_scaling_memory.png"))
    plt.close()


def main():
    """Main benchmark function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MoE scaling benchmark")
    
    parser.add_argument("--output-dir", type=str, default="./results/benchmarks",
                        help="Directory to save benchmark results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size to use for the benchmark")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length to use for the benchmark")
    parser.add_argument("--hidden-size", type=int, default=1024,
                        help="Hidden size to use for the benchmark")
    parser.add_argument("--intermediate-size", type=int, default=4096,
                        help="Intermediate size for the MoE FFN layers")
    parser.add_argument("--expert-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Number of experts to benchmark")
    parser.add_argument("--experts-per-token", type=int, nargs="+", default=[1, 2, 4],
                        help="Number of experts per token to benchmark")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--test-steps", type=int, default=10,
                        help="Number of test steps to average over")
    parser.add_argument("--use-fp8", action="store_true", default=True,
                        help="Whether to use FP8 precision")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmarks on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for different expert counts
    all_results = []
    
    for num_experts_per_tok in args.experts_per_token:
        for num_experts in args.expert_counts:
            # Skip invalid combinations
            if num_experts_per_tok > num_experts:
                continue
                
            print(f"Benchmarking MoE with {num_experts} experts, top-{num_experts_per_tok} routing...")
            
            try:
                result = benchmark_moe_scaling(
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    use_fp8=args.use_fp8,
                    device=args.device,
                    warmup_steps=args.warmup_steps,
                    test_steps=args.test_steps,
                )
                
                all_results.append(result)
                
                print(f"  Tokens/sec: {result['tokens_per_second']:.2f}, "
                      f"Latency: {result['latency_ms']:.2f} ms, "
                      f"Memory: {result['memory_usage_mb']:.2f} MB")
            except Exception as e:
                print(f"Error benchmarking MoE with {num_experts} experts: {e}")
    
    # Save results to file
    with open(os.path.join(args.output_dir, "moe_scaling_benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(all_results, args.output_dir)
    
    print(f"\nAll benchmark results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
