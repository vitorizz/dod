import itertools

import perf
import pytest


@pytest.mark.parametrize("raw, expected", [(None, 1.0), ("", 1.0), ("2", 2.0), (" 1.5 ", 1.5)])
def test_scale_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("DOD_PERF_SCALE", raising=False)
    else:
        monkeypatch.setenv("DOD_PERF_SCALE", raw)
    assert perf._scale() == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "nan", "inf"])
def test_scale_rejects_non_positive_or_non_numbers(monkeypatch, raw):
    monkeypatch.setenv("DOD_PERF_SCALE", raw)
    with pytest.raises(ValueError, match="DOD_PERF_SCALE"):
        perf._scale()


def fake_clock(monkeypatch, step_ns):
    ticks = itertools.count(0, step_ns)
    monkeypatch.setattr(perf.time, "perf_counter_ns", lambda: next(ticks))


def test_measured_times_are_divided_by_scale(monkeypatch):
    # Every timed call takes one fake 1 ms tick; the inner part none.
    fake_clock(monkeypatch, 1_000_000)
    monkeypatch.setattr(perf, "SCALE", 1.0)
    assert perf.p99_ms(lambda: None, n=10, warmup=0) == 1.0
    monkeypatch.setattr(perf, "SCALE", 2.0)
    assert perf.p99_ms(lambda: None, n=10, warmup=0) == 0.5
    inner = []
    assert perf.split_p99_ms(lambda: inner.append(0), inner, n=10, warmup=0) == (0.5, 0.0)
