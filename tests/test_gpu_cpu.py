"""Quick hardware diagnostic for CPU and GPU detection.

Run with:
	python tests/test_gpu_cpu.py
"""

from __future__ import annotations

import os
import platform
from typing import Final

import torch


BYTES_IN_GB: Final[int] = 1024**3


def print_cpu_info() -> None:
	"""Print basic CPU information."""
	print("=" * 60)
	print("CPU")
	print("=" * 60)
	print(f"Machine            : {platform.machine()}")
	print(f"Processor          : {platform.processor() or 'N/A'}")
	print(f"Logical cores      : {os.cpu_count()}")
	print(f"PyTorch threads    : {torch.get_num_threads()}")


def print_gpu_info() -> None:
	"""Print CUDA/GPU information if available in PyTorch."""
	print("\n" + "=" * 60)
	print("GPU (CUDA via PyTorch)")
	print("=" * 60)
	cuda_available = torch.cuda.is_available()
	print(f"CUDA available     : {cuda_available}")
	print(f"PyTorch version    : {torch.__version__}")
	print(f"CUDA (build)       : {torch.version.cuda}")

	if not cuda_available:
		print("No CUDA GPU detected by PyTorch.")
		return

	device_count = torch.cuda.device_count()
	print(f"Detected GPU(s)    : {device_count}")

	for gpu_index in range(device_count):
		props = torch.cuda.get_device_properties(gpu_index)
		total_mem_gb = props.total_memory / BYTES_IN_GB
		print(f"\nGPU #{gpu_index}")
		print(f"Name               : {props.name}")
		print(f"Compute capability : {props.major}.{props.minor}")
		print(f"Total memory (GB)  : {total_mem_gb:.2f}")
		print(f"Multiprocessors    : {props.multi_processor_count}")


def run_quick_torch_check() -> None:
	"""Run a tiny tensor operation on detected device to validate runtime."""
	print("\n" + "=" * 60)
	print("PyTorch runtime check")
	print("=" * 60)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device       : {device}")

	x = torch.rand((1024, 1024), device=device)
	y = torch.rand((1024, 1024), device=device)
	z = (x @ y).mean()
	print(f"Computation OK     : mean(x @ y) = {z.item():.6f}")


def main() -> None:
	print_cpu_info()
	print_gpu_info()
	run_quick_torch_check()


if __name__ == "__main__":
	main()
