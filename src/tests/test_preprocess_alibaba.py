"""Tests for the Alibaba Cluster Trace v2018 preprocessing pipeline."""

import importlib.util
from pathlib import Path

import pandas as pd


_SCRIPT = Path(__file__).parents[2] / "dataset" / "preprocess_alibaba.py"
_SPEC = importlib.util.spec_from_file_location("preprocess_alibaba", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
identify_flash_events = _MODULE.identify_flash_events
preprocess = _MODULE.preprocess


def _write_fixture(raw_dir):
    machine_usage = pd.DataFrame([
        ["m1", 0.0, 5, 10, 0, 0, 0, 0, 0],
        ["m2", 0.0, 5, 10, 0, 0, 0, 0, 0],
        ["m1", 0.1, 60, 10, 0, 0, 0, 0, 0],
        ["m2", 0.1, 60, 10, 0, 0, 0, 0, 0],
        ["m1", 0.2, 5, 10, 0, 0, 0, 0, 0],
        ["m2", 0.2, 5, 10, 0, 0, 0, 0, 0],
    ])
    machine_usage.to_csv(raw_dir / "machine_usage.csv", index=False, header=False)
    batch_instances = pd.DataFrame([
        ["i1", "t", "j", "type", "terminated", 0.1, 1, "m1", 0, 1, 1, 1, 1, 1],
        ["i2", "t", "j", "type", "terminated", 0.1, 1, "m1", 0, 1, 1, 1, 1, 1],
    ])
    batch_instances.to_csv(raw_dir / "batch_instance.csv", index=False, header=False)


def test_preprocess_creates_summary_with_ten_x_event(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_fixture(raw_dir)
    output = tmp_path / "alibaba_summary.csv"

    summary = preprocess(raw_dir, output, threshold_multiplier=5, chunksize=2)

    assert output.is_file()
    assert len(summary) == 1
    assert summary.loc[0, "burst_multiplier"] == 12.0
    assert summary.loc[0, "peak_arrival_rate_per_second"] == 20.0
    assert summary.loc[0, "arrival_count"] == 2


def test_event_threshold_is_strictly_greater_than_five_x():
    cpu = pd.Series([5.0, 25.0, 25.1, 5.0, 5.0], index=[0.0, 0.1, 0.2, 0.3, 0.4])
    events = identify_flash_events(cpu, pd.Series(dtype="float64"), threshold_multiplier=5)

    assert len(events) == 1
    assert events.loc[0, "start_time_s"] == 0.2
