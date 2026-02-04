# PARC
![demo](./doc/images/teaser.jpg)
Project page: https://michaelx.io/parc

# Installation

## Quick Start with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install PARC:

```bash
# Clone the repository
git clone https://github.com/mshoe/PARC.git
cd PARC

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
```

### With mjlab Backend (MuJoCo Warp - GPU Required)

For GPU-accelerated motion tracking using mjlab:

```bash
# Install with mjlab support
uv sync --extra mjlab

# Or add mjlab to existing installation
uv add mjlab
```

### Platform-Specific Dependencies

```bash
# Linux (with CUDA)
uv sync --extra linux

# macOS (evaluation only, no GPU training)
uv sync --extra macos
```

### Development Setup

```bash
# Install with dev dependencies
uv sync --extra dev

# Or install everything
uv sync --extra all
```

## Alternative: pip Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# With mjlab backend
pip install -e ".[mjlab]"
```

## Legacy: Conda Installation

For Isaac Gym compatibility (deprecated):

```bash
conda create -n parc python=3.8.20
conda activate parc
pip install -r requirements.txt
```

If PyTorch doesn't detect CUDA:
```bash
pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Running PARC

```bash
# Run motionscope
python scripts/run_motionscope.py

# Or using the installed command
run-motionscope
```

## Motionscope
Motionscope is my motion/terrain editor, as well as motion generator visualizer, built on top of Polyscope: https://polyscope.run/py/. You can run it by running:
```
python scripts/run_motionscope.py
```

Before that, you must edit the config file `parc/motionscope/motionscope_config.yaml` to load the motion you want, as well as optionally load an MDM model.

## Dataset and Models
Download the datasets from the initial iteration and each stage of PARC.


### New release: https://huggingface.co/datasets/mxucg/PARC
Two separate PARC experiments (Dec 2024 with 4 iterations, April 2025 with 5 iterations).
New small model (~30 mb).


These files are loaded with anim/motion_lib.py and anim/kin_char_model.py.
You can view them with scripts/run_motionscope.py, by editing the "motion_filepath" param in parc/motionscope/motionscope_config.yaml

If you only want the data without installing the whole repo, check out the script: scripts/read_motion_data.py
You should only need numpy (and maybe pytorch?) to read the data.


(Old release:
https://1sfu-my.sharepoint.com/:f:/g/personal/mxa23_sfu_ca/Et16uLMFxoRKouibvBa7LbwBEmX5_iI5a8dZyiMc0wmSTA?e=ihma1b
The password is "PARC". The file format is only compatible with v0.1 PARC release.)

## User configuration
All configuration files reference data, checkpoints, and generated outputs through a `$DATA_DIR` placeholder. Set this base
directory in `user_config.yaml` at the repository root:

```
DATA_DIR: "/absolute/path/to/your/data"
```

`DATA_DIR` must be an existing absolute path. The training and pipeline scripts will automatically replace `$DATA_DIR` in YAML
configs with the configured value when they load them.


## Motion Tracking

PARC supports two physics backends for motion tracking:

### mjlab Backend (Recommended)

[mjlab](https://github.com/mujocolab/mjlab) combines Isaac Lab's manager-based API with MuJoCo Warp for GPU-accelerated simulation. This is the recommended backend for new projects.

```bash
# Install with mjlab support
uv sync --extra mjlab

# Run the example
python PARC/motion_tracker/envs/mj_launch_example.py \
    --char-file data/characters/humanoid.xml \
    --motion-file data/motions/walk.yaml \
    --num-envs 4096
```

Key features:
- Manager-based RL API (modular observations, rewards, events)
- Direct PyTorch tensor access via `env.scene.data`
- MuJoCo Warp GPU backend for fast parallel simulation
- Custom PD actuators for exponential map control

Example usage:
```python
from parc.motion_tracker.envs import MJCharEnv, MJCharEnvCfg

cfg = MJCharEnvCfg(
    char_file="path/to/humanoid.xml",
    motion_file="path/to/motions.yaml",
    scene=MJCharSceneCfg(num_envs=4096),
)
env = MJCharEnv(cfg=cfg)
obs, info = env.reset()
```

### Isaac Gym Backend (Legacy)

The original implementation using Isaac Gym (deprecated by NVIDIA). Use this for compatibility with existing checkpoints.

If you still wish to use Isaac Gym, install it from: https://developer.nvidia.com/isaac-gym

```bash
# Create conda environment for Isaac Gym
conda create -n parc python=3.8.20
conda activate parc
pip install -r requirements.txt
```

The isaac gym helper yaml should look like this:
```yaml
name: parc
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8.20
  - pytorch=2.20.0
  - torchvision=0.9.1
  - cudatoolkit=11.1
  - pyyaml>=5.3.1
  - scipy>=1.5.0
  - tensorboard>=2.2.1
```

### Environment Architecture

The blocky terrain style motion tracking environment works by:
1. Loading terrain-motion pairs from your dataset
2. Laying them out as a grid in the simulation
3. Each agent tracks a specific motion-terrain pair
4. Position offsets are managed per agent for reward/observation computation

For more details, see [MimicKit](https://github.com/xbpeng/MimicKit)

## Codebase Guide
* [PARC Guide](doc/parc_guide.md)

## Citation
If you find PARC helpful, please consider citing the references in the [citation document](./doc/cite.md).