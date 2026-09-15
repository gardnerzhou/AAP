import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-vis', type=bool, default=True, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='click', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=1, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq', type=int, default=1, help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')
    parser.add_argument('-dataset', default='SUNSEG', type=str, help='dataset name: SUNSEG, ISIC2017, ISIC2018, REFUGE, PancreasCT, CHAOS')
    parser.add_argument('-sam_ckpt', type=str, default='checkpoints/sam2_hiera_large.pt', help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_l', help='sam checkpoint config')
    parser.add_argument('-video_length', type=int, default=2, help='video length')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-6, help='initial learning rate')
    parser.add_argument('-weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('-weights', type=str, default=0, help='weights file to test')
    parser.add_argument('-multimask_output', type=int, default=1, help='number of masks output')
    parser.add_argument('-memory_bank_size', type=int, default=64, help='memory bank size')
    parser.add_argument('-num_memory_used', type=int, default=8, help='number of memories used in attention')
    parser.add_argument('-num_epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('-loss_pos', type=int, default=3, help='number of iterations for cluster')
    parser.add_argument('-use_tv_loss', type=bool, default=False)
    parser.add_argument('-tv_lambda', type=float, default=0.01)
    parser.add_argument('-use_btv_loss', type=bool, default=False)
    parser.add_argument('-btv_lambda', type=float, default=0.01)
    parser.add_argument('-loss_reward_weight', type=float, default=0)
    parser.add_argument('-improvement_reward_weight', type=float, default=1.0)
    parser.add_argument('-positive_prompt_num', type=int, default=2)
    parser.add_argument('-negative_prompt_num', type=int, default=1)
    parser.add_argument('-ppo_lr_actor', type=float, default=1e-6)
    parser.add_argument('-ppo_lr_critic', type=float, default=1e-6)
    parser.add_argument('-ppo_gamma', type=float, default=0.95)
    parser.add_argument('-ppo_K_epochs', type=int, default=4)
    parser.add_argument('-ppo_eps_clip', type=float, default=0.2)
    parser.add_argument('-ppo_entropy_coef', type=float, default=0.01)
    parser.add_argument('-ppo_input_feature_size', type=int, default=64)
    parser.add_argument('-use_memory_attention', type=bool, default=False)
    parser.add_argument('-freeze_sam', type=bool, default=False)
    parser.add_argument('-use_prompt_dist_reward', type=bool, default=False)
    parser.add_argument('-prompt_dist_reward_weight', type=float, default=0.1)
    parser.add_argument('-use_entropy_constraint', type=bool, default=True)
    parser.add_argument('-entropy_constraint_weight', type=float, default=1.0)
    parser.add_argument('-use_modified_mask_decoder', type=bool, default=False)
    parser.add_argument('-use_extra_postprocess', type=bool, default=False)
    parser.add_argument('-postprocess_min_area_ratio', type=float, default=0.005)
    parser.add_argument('-postprocess_morph_kernel_size', type=int, default=7)
    parser.add_argument('-postprocess_keep_largest_only', type=bool, default=True)
    parser.add_argument('-use_prompt_warmup', type=bool, default=False)
    parser.add_argument('-prompt_warmup_epochs', type=int, default=10)
    parser.add_argument('-per_step_backprop', type=bool, default=False)
    parser.add_argument('-save_path', type=str, default='logs')
    parser.add_argument('-test_save_path', type=str, default='test_save')
    parser.add_argument('-exp_name', type=str, default='aap_sunseg')
    parser.add_argument('-raw_data_path', type=str, default='data/SUN-SEG')
    parser.add_argument('-isic2017_path', type=str, default='data/ISIC2017')
    parser.add_argument('-isic2018_path', type=str, default='data/ISIC2018')
    parser.add_argument('-pancreas_ct_path', type=str, default='data/Pancreas-CT')
    parser.add_argument('-chaos_path', type=str, default='data/CHAOS')
    parser.add_argument('-memory_path', type=str, default='')
    parser.add_argument('-args_yaml_path', type=str, default=None)

    opt = parser.parse_args()
    opt.save_path = os.path.join(opt.save_path, opt.exp_name)
    opt.test_save_path = os.path.join(opt.test_save_path, opt.exp_name)

    if opt.dataset == 'ISIC2017':
        opt.data_path = opt.isic2017_path
    elif opt.dataset == 'ISIC2018':
        opt.data_path = opt.isic2018_path
    elif opt.dataset == 'REFUGE':
        opt.refuge_path = getattr(opt, 'refuge_path', 'data/REFUGE')
        opt.data_path = opt.refuge_path
    elif opt.dataset == 'PancreasCT':
        opt.data_path = opt.pancreas_ct_path
    elif opt.dataset == 'CHAOS':
        opt.data_path = opt.chaos_path

    opt.memory_path = os.path.join(opt.save_path, 'memory.pkl')
    return opt
