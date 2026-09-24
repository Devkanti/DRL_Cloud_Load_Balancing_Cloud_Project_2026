"""Preprocess Alibaba Cluster Trace v2018 for FlashBalanceAI.

The script reads the official headerless ``machine_usage.csv`` and
``batch_instance.csv`` files in chunks. It produces an event-level CSV where
each event is a contiguous interval whose cluster CPU usage is greater than a
multiple of the median baseline. Batch-instance start times are resampled to
100 ms and reported as a workload-arrival proxy.

The trace does not contain end-user HTTP requests. Consequently, the arrival
series represents batch workload starts and must be cited as a workload proxy,
not as HTTP request telemetry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd


MACHINE_USAGE_COLUMNS = [
    "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
    "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent",
]
BATCH_INSTANCE_COLUMNS = [
    "instance_name", "task_name", "job_name", "task_type", "status",
    "start_time", "end_time", "machine_id", "seq_no", "total_seq_no",
    "cpu_avg", "cpu_max", "mem_avg", "mem_max",
]
SUMMARY_COLUMNS = [
    "event_id", "start_time_s", "end_time_s", "duration_s",
    "baseline_cpu_percent", "peak_cpu_percent", "burst_multiplier",
    "peak_arrival_rate_per_second", "arrival_count",
]


def _read_trace(path: Path, columns: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    """Read an official trace CSV, accepting either positional or named columns."""
    sample = pd.read_csv(path, nrows=1)
    use_header = set(columns).issubset(sample.columns)
    return pd.read_csv(
        path,
        header=0 if use_header else None,
        names=None if use_header else columns,
        chunksize=chunksize,
        low_memory=False,
    )


def load_cluster_cpu(machine_usage_path: Path, chunksize: int = 1_000_000) -> pd.Series:
    """Return mean cluster CPU utilisation indexed by 100 ms trace time."""
    aggregates: list[pd.DataFrame] = []
    for chunk in _read_trace(machine_usage_path, MACHINE_USAGE_COLUMNS, chunksize):
        data = chunk[["time_stamp", "cpu_util_percent"]].apply(pd.to_numeric, errors="coerce").dropna()
        data = data[data["cpu_util_percent"].between(0, 100)]
        data["time_bin"] = (data["time_stamp"] * 10).round().astype("int64") / 10
        aggregates.append(data.groupby("time_bin")["cpu_util_percent"].agg(["sum", "count"]))

    if not aggregates:
        raise ValueError("machine_usage.csv contains no valid CPU observations")
    combined = pd.concat(aggregates).groupby(level=0).sum()
    return (combined["sum"] / combined["count"]).sort_index()


def load_arrival_rates(batch_instance_path: Path, chunksize: int = 1_000_000) -> pd.Series:
    """Return batch-instance starts per second, resampled into 100 ms bins."""
    counts: list[pd.Series] = []
    for chunk in _read_trace(batch_instance_path, BATCH_INSTANCE_COLUMNS, chunksize):
        starts = pd.to_numeric(chunk["start_time"], errors="coerce").dropna()
        starts = starts[starts >= 0]
        bins = (starts * 10).round().astype("int64") / 10
        counts.append(bins.value_counts())

    if not counts:
        return pd.Series(dtype="float64", name="arrival_rate_per_second")
    arrivals = pd.concat(counts).groupby(level=0).sum().sort_index() * 10.0
    arrivals.name = "arrival_rate_per_second"
    return arrivals


def identify_flash_events(
    cluster_cpu: pd.Series,
    arrival_rates: pd.Series,
    threshold_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Identify contiguous CPU intervals above ``threshold_multiplier`` baseline."""
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive")
    if cluster_cpu.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    baseline = float(cluster_cpu.median())
    if baseline <= 0:
        raise ValueError("CPU baseline is zero; burst multipliers are undefined")
    is_event = cluster_cpu > baseline * threshold_multiplier
    group = is_event.ne(is_event.shift()).cumsum()
    events: list[dict[str, float | int]] = []
    for _, values in cluster_cpu[is_event].groupby(group[is_event]):
        start, end = float(values.index.min()), float(values.index.max())
        event_arrivals = arrival_rates[(arrival_rates.index >= start) & (arrival_rates.index <= end)]
        peak = float(values.max())
        events.append({
            "event_id": len(events) + 1,
            "start_time_s": start,
            "end_time_s": end,
            "duration_s": round(end - start + 0.1, 3),
            "baseline_cpu_percent": baseline,
            "peak_cpu_percent": peak,
            "burst_multiplier": peak / baseline,
            "peak_arrival_rate_per_second": float(event_arrivals.max()) if not event_arrivals.empty else 0.0,
            "arrival_count": int((event_arrivals / 10.0).sum()),
        })
    return pd.DataFrame(events, columns=SUMMARY_COLUMNS)


def preprocess(
    raw_dir: Path,
    output_path: Path,
    threshold_multiplier: float = 5.0,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Create and save the flash-event summary from an extracted v2018 trace."""
    machine_usage = raw_dir / "machine_usage.csv"
    batch_instance = raw_dir / "batch_instance.csv"
    missing = [str(path) for path in (machine_usage, batch_instance) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required trace file(s): " + ", ".join(missing))

    cpu = load_cluster_cpu(machine_usage, chunksize)
    arrivals = load_arrival_rates(batch_instance, chunksize)
    summary = identify_flash_events(cpu, arrivals, threshold_multiplier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Alibaba Cluster Trace v2018.")
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/alibaba_cluster_trace"))
    parser.add_argument("--output", type=Path, default=Path("dataset/alibaba_summary.csv"))
    parser.add_argument("--threshold-multiplier", type=float, default=5.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--s3-uri", help="Optional s3://bucket/key destination for the summary CSV.")
    args = parser.parse_args()

    summary = preprocess(args.raw_dir, args.output, args.threshold_multiplier, args.chunksize)
    print(f"Wrote {len(summary)} flash events to {args.output}")
    if not summary.empty:
        print(f"Maximum CPU burst multiplier: {summary['burst_multiplier'].max():.2f}x")
    if args.s3_uri:
        if not args.s3_uri.startswith("s3://"):
            raise ValueError("--s3-uri must use s3://bucket/key format")
        import boto3

        bucket, key = args.s3_uri.removeprefix("s3://").split("/", 1)
        boto3.client("s3").upload_file(str(args.output), bucket, key)
        print(f"Uploaded summary to {args.s3_uri}")


if __name__ == "__main__":
    main()
