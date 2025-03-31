import os
import sys
import platform
import psutil
import torch
import multiprocessing
import numpy as np
from pathlib import Path
from tabulate import tabulate


def get_cpu_info():
    """Get detailed CPU information"""
    info = {
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
        "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_arch": platform.processor(),
        "system": platform.system(),
        "platform": platform.platform(),
    }
    return info


def get_memory_info():
    """Get system memory information"""
    vm = psutil.virtual_memory()
    info = {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "percent_used": vm.percent
    }
    return info


def get_gpu_info():
    """Get GPU information using PyTorch"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    info = {
        "gpu_available": True,
        "gpu_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": []
    }

    for i in range(info["gpu_count"]):
        device_info = {
            "device_name": torch.cuda.get_device_name(i),
            "device_capability": torch.cuda.get_device_capability(i),
            "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),
            "device_id": i
        }
        # Try to get memory info - this may fail on some systems
        try:
            reserved = round(torch.cuda.memory_reserved(i) / (1024 ** 3), 2)
            allocated = round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2)
            device_info["reserved_memory_gb"] = reserved
            device_info["allocated_memory_gb"] = allocated
            device_info["free_memory_gb"] = device_info["total_memory_gb"] - allocated
        except:
            device_info["memory_info"] = "Not available"

        info["devices"].append(device_info)

    return info


def get_pytorch_info():
    """Get PyTorch build information"""
    info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "parallel_info": f"Native MP: {multiprocessing.cpu_count()} CPUs"
    }

    if hasattr(torch, 'get_num_threads'):
        info["num_threads"] = torch.get_num_threads()

    if hasattr(torch, 'get_num_interop_threads'):
        info["num_interop_threads"] = torch.get_num_interop_threads()

    return info


def run_tensor_benchmark(sizes=[(1000, 1000), (5000, 5000), (10000, 10000)]):
    """
    Run a simple matrix multiplication benchmark to
    measure relative CPU vs GPU performance
    """
    results = []

    # CPU benchmark
    for size in sizes:
        a = torch.randn(size)
        b = torch.randn(size)

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(2):
            _ = torch.matmul(a, b)

        # Measure
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(5):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        cpu_time = start.elapsed_time(end) / 5

        results.append({
            "size": f"{size[0]}x{size[1]}",
            "device": "CPU",
            "time_ms": cpu_time
        })

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        for size in sizes:
            a = torch.randn(size, device="cuda")
            b = torch.randn(size, device="cuda")

            # Warmup
            for _ in range(2):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()

            # Measure
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(5):
                _ = torch.matmul(a, b)
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / 5

            results.append({
                "size": f"{size[0]}x{size[1]}",
                "device": "GPU",
                "time_ms": gpu_time
            })

    return results


def suggest_optimal_settings():
    """
    Suggest optimal settings for pipeline parallelization
    based on system specs
    """
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    mem_info = get_memory_info()

    suggestions = {}

    # Num workers should be ~75% of physical cores
    physical_cores = cpu_info["cpu_count_physical"]
    suggestions["num_workers"] = max(1, int(physical_cores * 0.75))

    # If hyperthreading is available, we can use more threads
    hyperthreading_ratio = cpu_info["cpu_count_logical"] / cpu_info["cpu_count_physical"]
    thread_multiplier = min(3, max(2, round(hyperthreading_ratio * 1.5)))
    suggestions["num_threads_multiplier"] = thread_multiplier

    # Batch sizes based on GPU VRAM
    if gpu_info["gpu_available"]:
        # For tracklet batch size, use the GPU memory
        gpu_mem_gb = gpu_info["devices"][0]["total_memory_gb"]

        if gpu_mem_gb >= 16:
            suggestions["tracklet_batch_size"] = 64
            suggestions["image_batch_size"] = 256
        elif gpu_mem_gb >= 8:
            suggestions["tracklet_batch_size"] = 32
            suggestions["image_batch_size"] = 128
        elif gpu_mem_gb >= 4:
            suggestions["tracklet_batch_size"] = 16
            suggestions["image_batch_size"] = 64
        else:
            suggestions["tracklet_batch_size"] = 8
            suggestions["image_batch_size"] = 32

        # Adjust based on GPU model - some models perform better with different settings
        gpu_model = gpu_info["devices"][0]["device_name"].lower()

        if "3090" in gpu_model or "4090" in gpu_model or "a100" in gpu_model:
            # High-end GPUs can handle more
            suggestions["tracklet_batch_size"] *= 1.5
            suggestions["image_batch_size"] *= 1.5
        elif "1060" in gpu_model or "1650" in gpu_model or "1050" in gpu_model:
            # Lower-end GPUs need more conservative settings
            suggestions["tracklet_batch_size"] = max(4, int(suggestions["tracklet_batch_size"] * 0.5))
            suggestions["image_batch_size"] = max(16, int(suggestions["image_batch_size"] * 0.5))

    else:
        # CPU-only settings
        suggestions["tracklet_batch_size"] = 8
        suggestions["image_batch_size"] = 32

    # Round to integers
    suggestions["tracklet_batch_size"] = int(suggestions["tracklet_batch_size"])
    suggestions["image_batch_size"] = int(suggestions["image_batch_size"])

    return suggestions


def main():
    print("\n" + "=" * 80)
    print(" SYSTEM INFORMATION FOR PIPELINE OPTIMIZATION ".center(80, "="))
    print("=" * 80 + "\n")

    # Get all system information
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    gpu_info = get_gpu_info()
    pytorch_info = get_pytorch_info()

    # Display CPU information
    print("\n" + "CPU INFORMATION".center(80, "-"))
    cpu_table = [
        ["Physical Cores", cpu_info["cpu_count_physical"]],
        ["Logical Cores", cpu_info["cpu_count_logical"]],
        ["Max Frequency", f"{cpu_info['cpu_freq_max']} MHz" if cpu_info['cpu_freq_max'] != "Unknown" else "Unknown"],
        ["Current Frequency",
         f"{cpu_info['cpu_freq_current']} MHz" if cpu_info['cpu_freq_current'] != "Unknown" else "Unknown"],
        ["CPU Usage", f"{cpu_info['cpu_percent']}%"],
        ["Architecture", cpu_info["cpu_arch"]],
        ["System", cpu_info["system"]],
        ["Platform", cpu_info["platform"]]
    ]
    print(tabulate(cpu_table, tablefmt="fancy_grid"))

    # Display Memory information
    print("\n" + "MEMORY INFORMATION".center(80, "-"))
    memory_table = [
        ["Total Memory", f"{memory_info['total_gb']} GB"],
        ["Available Memory", f"{memory_info['available_gb']} GB"],
        ["Used Memory", f"{memory_info['used_gb']} GB"],
        ["Memory Usage", f"{memory_info['percent_used']}%"]
    ]
    print(tabulate(memory_table, tablefmt="fancy_grid"))

    # Display GPU information
    print("\n" + "GPU INFORMATION".center(80, "-"))
    if gpu_info["gpu_available"]:
        gpu_table = [
            ["GPU Count", gpu_info["gpu_count"]],
            ["CUDA Version", gpu_info["cuda_version"]],
            ["CuDNN Version", gpu_info["cudnn_version"]]
        ]
        print(tabulate(gpu_table, tablefmt="fancy_grid"))

        # Display information for each GPU
        for i, device in enumerate(gpu_info["devices"]):
            print(f"\nGPU {i}: {device['device_name']}")
            device_table = [
                ["Device Capability", f"{device['device_capability'][0]}.{device['device_capability'][1]}"],
                ["Total Memory", f"{device['total_memory_gb']} GB"]
            ]
            if "reserved_memory_gb" in device:
                device_table.extend([
                    ["Reserved Memory", f"{device['reserved_memory_gb']} GB"],
                    ["Allocated Memory", f"{device['allocated_memory_gb']} GB"],
                    ["Free Memory", f"{device['free_memory_gb']} GB"]
                ])
            print(tabulate(device_table, tablefmt="fancy_grid"))
    else:
        print("No CUDA-compatible GPU detected")

    # Display PyTorch information
    print("\n" + "PYTORCH INFORMATION".center(80, "-"))
    pytorch_table = [
        ["PyTorch Version", pytorch_info["version"]],
        ["CUDA Available", "Yes" if pytorch_info["cuda_available"] else "No"],
        ["Parallel Info", pytorch_info["parallel_info"]]
    ]
    if "num_threads" in pytorch_info:
        pytorch_table.append(["Num Threads", pytorch_info["num_threads"]])
    if "num_interop_threads" in pytorch_info:
        pytorch_table.append(["Num Interop Threads", pytorch_info["num_interop_threads"]])
    print(tabulate(pytorch_table, tablefmt="fancy_grid"))

    # Run benchmark tests
    print("\n" + "PERFORMANCE BENCHMARK".center(80, "-"))
    print("Running matrix multiplication benchmark...")
    benchmark_results = run_tensor_benchmark()

    benchmark_table = []
    for result in benchmark_results:
        benchmark_table.append([
            result["size"],
            result["device"],
            f"{result['time_ms']:.2f} ms"
        ])
    print(tabulate(benchmark_table, headers=["Matrix Size", "Device", "Time per Op"],
                   tablefmt="fancy_grid"))

    # Get and display suggested settings
    print("\n" + "SUGGESTED PIPELINE SETTINGS".center(80, "-"))
    suggestions = suggest_optimal_settings()
    suggestion_table = []
    for param, value in suggestions.items():
        suggestion_table.append([param, value])
    print(tabulate(suggestion_table, headers=["Parameter", "Suggested Value"],
                   tablefmt="fancy_grid"))

    # Generate code snippet for pipeline configuration
    print("\n" + "SUGGESTED PIPELINE CODE".center(80, "-"))
    code_snippet = f"""
# Suggested optimization parameters for your system
pipeline = CentralPipeline(
    # ... your other parameters ...
    num_workers={suggestions['num_workers']},
    tracklet_batch_size={suggestions['tracklet_batch_size']},
    image_batch_size={suggestions['image_batch_size']},
    num_threads_multiplier={suggestions['num_threads_multiplier']}
)
"""
    print(code_snippet)

    print("\n" + "=" * 80)
    print(" END OF SYSTEM INFORMATION ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred while collecting system information: {e}")
        import traceback

        traceback.print_exc()