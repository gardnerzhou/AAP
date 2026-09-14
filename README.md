# Active-Associative Prompting (AAP)

Core implementation of **"Active-Associative Prompting: Closed-Loop Correction with Cohort-Level Association for Scribble-Supervised Medical Image Segmentation."**

AAP contains two main components:
- **ACS**: cohort-level prototype association with DPP-based memory curation.
- **AIS**: prediction-dependent corrective prompting with a PPO agent trained from scribble-derived rewards.

## Setup

```bash
conda env create -f environment.yml
conda activate medsam2
bash checkpoints/download_ckpts.sh
```

## Data

Set the dataset paths in `cfg.py`. The experiments use **SUN-SEG**, **ISIC2018**, and **CHAOS**.

## Core code

- `func_2d/PPO_prompt.py`: PPO actor-critic prompting policy.
- `func_2d/utils_memory.py`: DPP-based cohort memory curation.
- `func_2d/utils_reinfor.py`: prompt-related utilities.
- `func_2d/postprocess.py`: segmentation post-processing.
- `cfg.py`: experiment configuration.

Pretrained SAM2 weights are not included in this repository.
