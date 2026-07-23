# Tasks

Arnold is trained and evaluated on 14 tasks spanning **four musculoskeletal models** from the
MyoSuite library. Any `<task_name>` argument in the training, evaluation and plotting scripts
must be one of the code names below.

!!! note "Code names vs. paper names"
    The task identifiers used in the code differ from the names used in the paper. The
    mapping is given in the tables below — in particular `reorient` is *Die reorient*,
    `relocate` is *Object relocation*, and `kinesis` is *Walk to point*.

## MyoElbow

Six muscles, one joint — the simplest of the four models.

| Code name | Paper name | Description | Max steps | Envs |
| --- | --- | --- | --- | --- |
| `elbow_pose` | Elbow pose | Rotate the elbow to point the hand at a random target location. | 100 | 2 |

## MyoHand

39 muscles, 23 joints. Eleven of the 14 tasks use this model — five finger-reaching tasks and
six object-manipulation tasks.

| Code name | Paper name | Description | Max steps | Envs |
| --- | --- | --- | --- | --- |
| `hand_thumb_reach` | Thumb reach | Point the tip of the thumb at a random target location. | 100 | 2 |
| `hand_index_reach` | Index reach | Point the tip of the index finger at a random target location. | 100 | 2 |
| `hand_middle_reach` | Middle reach | Point the tip of the middle finger at a random target location. | 100 | 2 |
| `hand_ring_reach` | Ring reach | Point the tip of the ring finger at a random target location. | 100 | 2 |
| `hand_little_reach` | Little reach | Point the tip of the little finger at a random target location. | 100 | 2 |
| `pen` | Pen reorient | Rotate a pen (cylinder) to a random desired orientation. | 100 | 2 |
| `reorient` | Die reorient | Rotate a die (cube) to a random desired orientation. | 150 | 2 |
| `baoding_p1_cw` | Baoding CW | Rotate two Baoding balls clockwise. Initial phase and target rotation speed are fixed. | 200 | 4 |
| `baoding_p1_ccw` | Baoding CCW | Rotate two Baoding balls counter-clockwise. Initial phase and target rotation speed are fixed. | 200 | 4 |
| `baoding_p2` | Baoding hard | Rotate two Baoding balls. Initial phase fixed; rotation direction and target speed vary. | 200 | 6 |
| `baoding_p2_overlap` | Baoding harder | Rotate two Baoding balls. Initial phase, rotation direction and target speed all vary. | 200 | 6 |

These 11 tasks are exactly the set used for the [CSI analysis](csi-analysis.md), which
relies on all of them sharing the same 39-muscle action space.

## MyoArm

63 muscles, 38 joints — the MyoHand extended with upper arm, pectoral and shoulder muscles.

| Code name | Paper name | Description | Max steps | Envs |
| --- | --- | --- | --- | --- |
| `relocate` | Object relocation | Grasp a dynamically generated object, lift it, and place it inside a box. | 150 | 6 |

## MyoLeg

80 muscles, 28 joints — a model of the human lower body.

| Code name | Paper name | Description | Max steps | Envs |
| --- | --- | --- | --- | --- |
| `kinesis` | Walk to point | Walk the body to a random target location without falling. | 150 | 12 |

Requires the `data/kinesis/` MuJoCo assets from Zenodo.

## Environment allocation

The **Envs** column above is the number of parallel environments the paper allocates to each
task during multi-task training. The allocation is deliberately imbalanced: the Baoding,
Object relocation and Walk to point tasks need substantially more environment interactions,
so they get more of the rollout budget.

The training scripts encode this by **repeating a task name in `--tasks`** — each occurrence
gets its own `--num_envs_per_task` environments. With `--num_envs_per_task 2`, a task listed
three times receives 6 parallel environments. See [Training](training.md).

## Full list

For copy-pasting into a `--tasks` argument (one occurrence each — not the imbalanced
allocation):

```text
hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach
reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose
relocate kinesis
```
