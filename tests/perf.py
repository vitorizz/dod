"""Shared benchmark helper for capture-path overhead tests."""

import time


def p99_ms(fn, n=10_000, warmup=1_000, label=None):
    """Call ``fn`` ``warmup`` times untimed, then ``n`` times timed.

    Prints p50 and p99 (visible with ``pytest -s``) and returns p99 in ms.
    """
    for _ in range(warmup):
        fn()
    samples = []
    clock = time.perf_counter_ns
    for _ in range(n):
        start = clock()
        fn()
        samples.append(clock() - start)
    samples.sort()
    p50 = samples[n // 2] / 1e6
    p99 = samples[min(n - 1, int(n * 0.99))] / 1e6
    print(f"{label or getattr(fn, '__name__', 'fn')}: p50={p50:.4f} ms p99={p99:.4f} ms (n={n})")
    return p99
