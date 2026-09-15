# Active-Associative Prompting (AAP)

Official code for **Active-Associative Prompting: Closed-Loop Correction with Cohort-Level Association for Scribble-Supervised Medical Image Segmentation**.

AAP contains two components:
- **ACS**: cohort-level prototype association with DPP-based memory curation.
- **AIS**: prediction-dependent corrective prompting with a PPO agent trained from scribble-derived rewards.

## Setup

```bash
conda env create -f environment.yml
conda activate medsam2
python tools/prepare_sam2.py
bash checkpoints/download_ckpts.sh
```

The pretrained SAM2 checkpoint is downloaded to `checkpoints/sam2_hiera_large.pt`.

## Data

Set the dataset path in `cfg.py` or pass it on the command line. Experiments use **SUN-SEG**, **ISIC2018**, and **CHAOS**.

## Training

Example for the SUN-SEG setting used in the paper:

```bash
python train_2d_sunseg.py \
  -dataset SUNSEG \
  -exp_name aap_sunseg \
  -raw_data_path /path/to/SUN-SEG \
  -sam_ckpt checkpoints/sam2_hiera_large.pt \
  -sam_config sam2_hiera_l \
  -use_memory_attention True \
  -memory_bank_size 64 \
  -num_memory_used 8 \
  -positive_prompt_num 2 \
  -negative_prompt_num 1 \
  -ppo_lr_actor 3e-4 \
  -ppo_lr_critic 3e-4 \
  -ppo_gamma 0.99 \
  -ppo_eps_clip 0.2 \
  -ppo_entropy_coef 0.01 \
  -use_prompt_dist_reward True \
  -lr 1e-5 \
  -per_step_backprop True \
  -num_epochs 6
```

## Testing

```bash
python test.py \
  -dataset SUNSEG \
  -exp_name aap_sunseg \
  -save_path logs \
  -test_save_path test_save \
  -args_yaml_path logs/aap_sunseg/args.yaml
```
