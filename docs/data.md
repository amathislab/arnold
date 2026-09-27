# Data and checkpoints

Download the [Zenodo v3 release](https://zenodo.org/records/21807280) from the
repository root using Python 3.8 or newer:

```bash
# Benchmark results (3.2 MB).
python scripts/fetch_data.py --profile benchmarks

# Model checkpoints, expert policies and Kinesis assets (3.4 GB).
python scripts/fetch_data.py --profile models

# Download both profiles.
python scripts/fetch_data.py --profile all
```

| Data | Installed directory |
| --- | --- |
| Model checkpoints | `data/final_checkpoints/` |
| Expert policies | `data/expert_policies/` |
| Kinesis assets | `data/kinesis/` |
| Benchmark results | `data/final_benchmarks/` |
| Extra benchmark results and MT learning curves | `data/final_benchmarks_extra/` |

Archives are cached in `data/cache/archives/21807280/`.

```bash
# List downloads.
python scripts/fetch_data.py --profile models --list

# Verify cached archive checksums.
python scripts/fetch_data.py --profile benchmarks --verify-only

# Select a cache and installation directory.
python scripts/fetch_data.py --profile benchmarks --cache-dir /path/to/archives --root /path/to/install
```

Continue with [Training](training.md), [Evaluation](evaluation.md), or
[Replicate plots](replicate-plots.md).
