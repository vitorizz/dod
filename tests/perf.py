"""Shared benchmark helper for capture-path overhead tests."""

import time


def p99_ms(fn, n=10_000, warmup=1_000, label=None, pause_s=0.0, rounds=1):
    """Call ``fn`` ``warmup`` times untimed, then ``n`` times timed.

    ``pause_s`` sleeps (untimed) before each timed call, for paths that write
    so much per call that a tight loop measures the kernel's dirty-page
    writeback throttling rather than the call itself.

    ``rounds`` repeats the timed loop and returns the median round's p99, so
    one round hit by a burst of disk or scheduler noise on a shared CI runner
    doesn't decide the result.

    Prints p50 and p99 (visible with ``pytest -s``) and returns p99 in ms.
    """
    for _ in range(warmup):
        fn()
    clock = time.perf_counter_ns
    name = label or getattr(fn, "__name__", "fn")
    p99s = []
    for r in range(rounds):
        samples = []
        for _ in range(n):
            if pause_s:
                time.sleep(pause_s)
            start = clock()
            fn()
            samples.append(clock() - start)
        samples.sort()
        p50 = samples[n // 2] / 1e6
        p99s.append(samples[min(n - 1, int(n * 0.99))] / 1e6)
        tag = f" round {r + 1}/{rounds}" if rounds > 1 else ""
        print(f"{name}{tag}: p50={p50:.4f} ms p99={p99s[-1]:.4f} ms (n={n})")
    return sorted(p99s)[rounds // 2]
