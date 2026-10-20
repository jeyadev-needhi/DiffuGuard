import time
import torch


def timed_generate(generate_fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = generate_fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return result, elapsed


def summarize(times):
    times = [float(x) for x in times]

    return {
        "n": len(times),
        "mean_seconds": sum(times) / len(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }
