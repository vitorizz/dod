"""Shared benchmark helpers for capture-path overhead tests.

The bars are set for CI. On a slower machine, ``DOD_PERF_SCALE=2 pytest -q``
divides every measured time by 2 before the test compares it to its bar,
which is the same as doubling every bar. CI never sets it.
"""

import math
import os
import time


def _scale():
    raw = os.environ.get("DOD_PERF_SCALE", "").strip()
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (0 < value < math.inf):
        raise ValueError(f"DOD_PERF_SCALE must be a positive number, got {raw!r}")
    return value


SCALE = _scale()
"""``DOD_PERF_SCALE``, default 1.0: multiplies every benchmark bar."""


def _rounds(fn, n, warmup, pause_s, rounds, inner):
    for _ in range(warmup):
        fn()
    clock = time.perf_counter_ns
    for _ in range(rounds):
        inner.clear()
        samples = []
        for _ in range(n):
            if pause_s:
                time.sleep(pause_s)
            start = clock()
            fn()
            samples.append(clock() - start)
        yield samples


def _p50_p99(samples):
    samples = sorted(samples)
    n = len(samples)
    return samples[n // 2] / 1e6, samples[min(n - 1, int(n * 0.99))] / 1e6


def _report(name, r, rounds, n, p50, p99):
    tag = f" round {r + 1}/{rounds}" if rounds > 1 else ""
    scaled = f", p99/DOD_PERF_SCALE={p99 / SCALE:.4f} ms" if SCALE != 1 else ""
    print(f"{name}{tag}: p50={p50:.4f} ms p99={p99:.4f} ms (n={n}{scaled})")


def p50_p99_ms(fn, n=10_000, warmup=1_000, label=None):
    """One timed loop like ``p99_ms``; returns ``(p50, p99)`` in ms, both divided by ``SCALE``."""
    name = label or getattr(fn, "__name__", "fn")
    samples = next(_rounds(fn, n, warmup, 0.0, 1, []))
    p50, p99 = _p50_p99(samples)
    _report(name, 0, 1, n, p50, p99)
    return p50 / SCALE, p99 / SCALE


def p99_ms(fn, n=10_000, warmup=1_000, label=None, pause_s=0.0, rounds=1):
    """Call ``fn`` ``warmup`` times untimed, then ``n`` times timed.

    ``pause_s`` sleeps (untimed) before each timed call, for paths that write
    so much per call that a tight loop measures the kernel's dirty-page
    writeback throttling rather than the call itself.

    ``rounds`` repeats the timed loop and returns the median round's p99, so
    one round hit by a burst of disk or scheduler noise on a shared CI runner
    doesn't decide the result.

    Prints p50 and p99 (visible with ``pytest -s``) and returns p99 in ms,
    divided by ``SCALE``.
    """
    name = label or getattr(fn, "__name__", "fn")
    p99s = []
    for r, samples in enumerate(_rounds(fn, n, warmup, pause_s, rounds, [])):
        p50, p99 = _p50_p99(samples)
        _report(name, r, rounds, n, p50, p99)
        p99s.append(p99)
    return sorted(p99s)[rounds // 2] / SCALE


def split_p99_ms(fn, inner, n=10_000, warmup=1_000, labels=("outer", "inner"),
                 pause_s=0.0, rounds=1):
    """Like ``p99_ms``, but split each call into two parts.

    ``fn`` must append to the list ``inner`` the nanoseconds spent in one
    nested part of each call, exactly once per call. Returns the median
    round's p99 of the rest of the call and of the nested part, in ms,
    each divided by ``SCALE``.
    """
    outer_p99s, inner_p99s = [], []
    for r, samples in enumerate(_rounds(fn, n, warmup, pause_s, rounds, inner)):
        assert len(inner) == len(samples), "fn must record one inner time per call"
        rest = [total - part for total, part in zip(samples, inner)]
        for label, part, p99s in ((labels[0], rest, outer_p99s), (labels[1], inner, inner_p99s)):
            p50, p99 = _p50_p99(part)
            _report(label, r, rounds, n, p50, p99)
            p99s.append(p99)
    return sorted(outer_p99s)[rounds // 2] / SCALE, sorted(inner_p99s)[rounds // 2] / SCALE
