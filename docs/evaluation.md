# Evaluation

`src/benchmark.py` evaluates the performance of pretrained models, including those trained
with OBC and Arnold, as well as the expert policies.

## Evaluating OBC and Arnold models

Specify the path to the saved model (`.zip` file) and the task to evaluate:

```bash
python src/benchmark.py \
    --load path/to/your/model.zip \
    --task <task_names> \
    --arnold \
    --num_episodes <number_of_episodes> \
    --deterministic \
    --device <cpu_or_cuda> \
    --render
```

| Flag | Meaning |
| --- | --- |
| `--load` | Path to your trained model checkpoint. |
| `--task` | One or many of the [available tasks](tasks.md). |
| `--arnold` | Include if the model was trained with Arnold. |
| `--num_episodes` | How many episodes to run for evaluation. |
| `--deterministic` | Take deterministic actions from the policy. |
| `--device` | `cpu` or `cuda`. |
| `--render` | Optionally render to video. |

!!! warning "Rendering on macOS"
    `--render` requires running `mjpython` instead of `python`.

### Example: an Arnold model

```bash
python src/benchmark.py \
    --load data/student_policies/arnold/rl_model_64670238_steps.zip \
    --task kinesis \
    --arnold \
    --num_episodes 10 \
    --deterministic \
    --device cpu
```

### Example: an OBC model

```bash
python src/benchmark.py \
    --load data/student_policies/obc/rl_model_54974700_steps.zip \
    --task relocate \
    --arnold \
    --num_episodes 10 \
    --deterministic \
    --device cpu
```

## Evaluating expert policies

For an expert policy, only the task needs to be specified — the checkpoint is resolved from
`data/expert_policies/`:

```bash
python src/benchmark.py \
    --task <task_name> \
    --expert \
    --num_episodes <number_of_episodes> \
    --deterministic \
    --device <cpu_or_cuda>
```

Include the `--expert` flag to indicate you are testing an expert policy, and set
`<task_name>` to one of the [available tasks](tasks.md).

### Example

```bash
python src/benchmark.py \
    --task relocate \
    --expert \
    --num_episodes 10 \
    --deterministic \
    --device cpu \
    --render
```

## Saving results for the plotting scripts

The plotting scripts read benchmark result JSONs from `data/final_benchmarks/` and
`data/final_benchmarks_extra/`. To regenerate them rather than using the released ones, add
`--save_results` and point `--out_dir` at the directory the relevant plot expects — see
[CSI-Finetuning](csi-finetuning.md) and [Multi-task RL baselines](mt-baselines.md) for the per-arm
destinations.
