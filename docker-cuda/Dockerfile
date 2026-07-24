FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# System dependencies.
#   - libgl1-mesa-*/libglew/libosmesa6: OpenGL + headless (OSMesa) rendering for MuJoCo
#   - libxml2-dev/libxslt1-dev/zlib1g-dev: needed to build lxml from source
#     (dm_control -> lxml, which has no cp38 wheel on every platform)
#   - python3-dev/gcc/g++: needed to build the source-only wheels (gym 0.13.0, lxml)
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    g++ \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libxml2-dev \
    libxslt1-dev \
    patchelf \
    python3 \
    python3-dev \
    python3-pip \
    software-properties-common \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The README (and every script) invokes `python`, which Ubuntu 20.04 does not provide.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

ENV PYTHONUNBUFFERED=1
# Render headlessly by default. Override with `-e MUJOCO_GL=egl` when a GPU is attached.
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

COPY requirements.txt /tmp/requirements.txt

# These must stay three separate steps: Ubuntu 20.04 ships pip 20.0.2, which crashes
# on legacy version strings found on PyPI as soon as a modern `setuptools` is importable.
# Upgrade pip on its own first, then let the new pip do the rest.
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade setuptools wheel
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# imitation must be installed in a second pass, because pip cannot resolve it together
# with gym (see the note at the top of requirements.txt). This step upgrades cloudpickle
# from 1.2.2 to 3.x, which is the version everything actually runs against.
RUN python -m pip install --no-cache-dir imitation==1.0.0

# Fail the build early if the environment is not actually importable.
RUN python -c "import gym, myosuite, torch, stable_baselines3, imitation, sb3_contrib; print('env ok')"

# The repository itself is bind-mounted at run time (see README), so that `data/`
# and the training outputs live on the host and survive the container.
WORKDIR /arnold
ENV PYTHONPATH=/arnold/src

CMD ["/bin/bash"]
