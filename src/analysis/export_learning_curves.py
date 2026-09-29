"""Export a selected TensorBoard scalar without smoothing."""
import argparse
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="+", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metric", choices=["solved_fraction", "reward"], default="solved_fraction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.events:
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        points = events.Scalars(args.tag)
        if points:
            first_step = min(point.step for point in points)
            rows = [row for row in rows if row["step"] < first_step]
        rows.extend(dict(panel=args.panel, method=args.method, seed=args.seed, task=args.task,
                         stage=args.stage, step=point.step, value=point.value, metric=args.metric)
                    for point in points)
    data = pd.DataFrame(rows)
    data = data.drop_duplicates(["step"], keep="last").sort_values("step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
