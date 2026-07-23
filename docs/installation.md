# Installation

To reproduce the training experiments there are two supported ways to set up an environment.

!!! info "Expected time"
    The installation usually takes less than 30 minutes on a modern computer with a fast
    internet connection.

## Method 1: Docker

The provided `Dockerfile` builds an image that can run all the Arnold experiments. This
assumes Docker is installed on your system.

Navigate to the directory containing the `Dockerfile` (`docker-cuda`) and build the image:

```bash
docker build -t arnold_image .
```

Once the image is built, run a container:

```bash
docker run -it --rm arnold_image /bin/bash
```

This starts an interactive session inside the container, from which you can execute the
training or evaluation scripts.

## Method 2: Conda environment

Alternatively, create a conda environment and install the dependencies manually.

```bash
conda create -n arnold python=3.8
conda activate arnold
pip install \
    cloudpickle==1.2.2\
    gym==0.13.0\
    gymnasium==0.29.1\
    h5py==3.7.0\
    wandb\
    tqdm\
    numpy\
    ipdb

pip install stable-baselines3==2.2.1
pip install MyoSuite==2.2.0
pip install imitation==1.0.0
pip install sb3-contrib==2.2.1
pip install Shimmy==1.3.0
pip install imageio
```

### System packages

You may need to install some OpenGL-related system packages:

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6
```

!!! warning "Rendering on macOS"
    Any command that passes `--render` must be run with `mjpython` instead of `python`.
    This is a MuJoCo requirement on macOS.

## Next steps

The code is installed, but the checkpoints and benchmark results are not — continue with
[Data and checkpoints](data.md).
