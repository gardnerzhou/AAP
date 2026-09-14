# Active-Associative Prompting (AAP)

Official implementation of **"Active-Associative Prompting: Closed-Loop Correction with Cohort-Level Association for Scribble-Supervised Medical Image Segmentation."**

AAP combines:
- **ACS**: cohort-level prototype association with DPP-based memory curation.
- **AIS**: prediction-dependent corrective prompting with a PPO agent trained from scribble-derived rewards.

## Setup

```bash
conda env create -f environment.yml
conda activate medsam2
bash checkpoints/download_ckpts.sh
```

## Data

Set dataset paths in `cfg.py`. The experiments use **SUN-SEG**, **ISIC2018**, and **CHAOS**.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_2d_sunseg.py \
  -dataset SUNSEG \
  -exp_name aap_sunseg \
  -use_memory_attention True \
  -positive_prompt_num 2 \
  -negative_prompt_num 1 \
  -memory_bank_size 64 \
  -num_memory_used 8 \
  -ppo_lr_actor 3e-4 \
  -ppo_lr_critic 3e-4 \
  -ppo_gamma 0.99 \
  -ppo_entropy_coef 0.01 \
  -lr 1e-5 \
  -num_epochs 6
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -exp_name aap_sunseg
```

## Checkpoints

SAM2 pretrained weights are not included. Run `checkpoints/download_ckpts.sh` to download the required checkpoint.

## License

Apache License 2.0. See `LICENSE`.
