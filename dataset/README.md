# Alibaba Cluster Trace v2018

## Source and terms

This project uses the [Alibaba Cluster Trace Program v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018) as a real-world workload reference. Alibaba documents a production cluster trace covering approximately 4,000 machines over eight days, delivered as six CSV tables. The official schema identifies `machine_usage.csv` as the source of CPU observations and `batch_instance.csv` as the source of workload start times.

Download the trace from the official program after completing its access survey, or run its published [`fetchData.sh`](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2018/fetchData.sh). The official archive is approximately 48–49 GB compressed and about 280 GB extracted. Store the extracted `machine_usage.csv` and `batch_instance.csv` in `dataset/alibaba_cluster_trace/`; raw CSV and archive files are ignored by Git.

The official repository does not publish a separate open-source licence file for v2018. Use is therefore subject to the trace program’s access terms and any terms presented during download. Do not redistribute the raw data through this repository.

## Processing

Run from the repository root:

```powershell
python dataset/preprocess_alibaba.py --raw-dir dataset/alibaba_cluster_trace --output dataset/alibaba_summary.csv --s3-uri s3://YOUR_BUCKET/dataset/alibaba_summary.csv
```

The script reads CSVs in chunks, averages valid machine CPU observations in 100 ms timestamp bins, and uses the median cluster CPU as the baseline. It marks each contiguous interval above `5 × baseline` as a flash event. Batch-instance `start_time` values are counted in the same 100 ms bins and converted to starts per second.

The output table contains each event’s start/end time, duration, baseline and peak CPU, CPU burst multiplier, peak batch-start rate, and total batch starts. The trace does not include end-user HTTP request arrivals. Its batch workload starts are therefore a workload-arrival proxy and must not be labelled as HTTP RPS.

## Validation protocol

After processing the actual raw trace, inspect `burst_multiplier`. The Phase 2 claim that the trace supports a `10×` scenario is valid only if the generated summary includes an event at or above `10.0`. The trace’s CPU metric is capped at 100 percent, so it cannot by itself demonstrate a `100×` CPU burst; the project’s 50× and 100× HTTP scenarios remain controlled stress-test multipliers rather than values directly inferred from this dataset.

The summary is intentionally generated locally from the downloaded trace and is not committed until the team has verified the raw-data provenance and result.
