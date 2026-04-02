# ReplicateAnyScene
ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment

## Installation

### Prerequisites

- A linux 64-bits architecture
- A NVIDIA GPU with at least 48 Gb of VRAM.

**Create a new conda environment:**

```bash
conda env create -f environments/default.yml
conda deactivate
conda activate ReplicateAnyScene
```

**Install SAM3D-related dependencies.**

```bash
cd sam-3d-objects
# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

**Install SAM3,VGGT,Qwen3VL dependencies.**

```bash
cd ../sam3
pip install .[dev,notebooks,train]
cd ../vggt
pip install -e .
pip install transformers==4.57.3
cd ..
```

**Download required models.**

```bash
mkdir models
hf download facebook/VGGT-1B  --local-dir models/VGGT
# You can use other VLMs as alternatives
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir models/Qwen3VL
```

## Usage

### Run the pipeline entirely

```bash
python main.py 