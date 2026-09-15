# Active-Associative Prompting (AAP)

Official code for **Active-Associative Prompting: Closed-Loop Correction with Cohort-Level Association for Scribble-Supervised Medical Image Segmentation**.

AAP contains two components:
- **ACS**: cohort-level prototype association with DPP-based memory curation.
- **AIS**: prediction-dependent corrective prompting with a PPO agent trained from scribble-derived rewards.

## Setup

```bash
conda env create -f environment.yml
conda activate medsam2
bash checkpoints/download_ckpts.sh
```

The project uses SAM2. The SAM2 source tree should be available as `sam2_train/` and the pretrained checkpoint path should be set in the command line or `cfg.py`.

## Data

Set dataset paths in `cfg.py`. Experiments use **SUN-SEG**, **ISIC2018**, and **CHAOS**.

## Training

```bash
python train_2d_sunseg.py -dataset SUNSEG -exp_name aap_sunseg
```

## Testing

```bash
python test.py -dataset SUNSEG -exp_name aap_sunseg
```

Pretrained weights, datasets, logs, and experiment outputs are not included in this repository.
