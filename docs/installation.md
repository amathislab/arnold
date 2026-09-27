# Installation

Run these commands from the repository root.

## Docker

```bash
docker build -t arnold_image docker-cuda
docker run -it --rm --gpus all \
    -v "$PWD":/arnold -w /arnold \
    arnold_image /bin/bash
```

For CPU use, omit `--gpus all` and pass `--device cpu` to experiment scripts.

## Conda

```bash
conda env create -f environment.yml
conda activate arnold
pip install imitation==1.0.0
```

Install `imitation` last. On Linux, install the rendering libraries:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libosmesa6
```

On macOS, use `mjpython` for commands with `--render`.

Continue with [Data and checkpoints](data.md).
