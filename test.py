# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""
import os
import time

import torch
from torch.utils.data import DataLoader
# from torch.amp import autocast, GradScaler

import cfg
import func_2d.function as function
from func_2d.dataset import *
from func_2d.utils import *
from func_2d.augmentation import get_test_transform
import yaml
import pickle
from func_2d.PPO_prompt import PPO_prompt
import random



def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id):
	"""确保每个worker的随机种子一致，保证数据加载顺序一致"""
	np.random.seed(42 + worker_id)
	random.seed(42 + worker_id)  
    
def main():
    
    # use bfloat16 for the entire work
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()
    
    # 设置 CUDA_VISIBLE_DEVICES
    original_gpu_device = getattr(args, 'gpu_device', 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(original_gpu_device)
    
    # 注意：设置 CUDA_VISIBLE_DEVICES 后，GPU 编号从 0 开始
    args.gpu_device = 0
    
    # os.makedirs(args.save_path,exist_ok=False)
    os.makedirs(args.test_save_path,exist_ok=True)
    
    # 确定 args.yaml 的路径
    if args.args_yaml_path is not None:
        args_yaml_path = args.args_yaml_path
    else:
        args_yaml_path = os.path.join(args.save_path, 'args.yaml')
    
    with open(args_yaml_path, 'r') as f:
        args_config=yaml.load(f.read(), yaml.FullLoader)
    
    for key, value in args_config.items():
        if hasattr(args, key):
            # 注意：不要覆盖 gpu_device，保持为 0（因为 CUDA_VISIBLE_DEVICES 已经设置）
            if key != 'gpu_device':
                setattr(args, key, value)
    
    # args.ppo_agent = False
    
    '''load pretrained model'''
    # args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.save_path)
    logger.info(args)
    
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    
    # 仅确定PPO训练后的权重路径（不再考虑BC预训练）
    best_ppo_path = os.path.join(args.save_path, 'best_ppo.pth')
    use_ppo_trained_weights = os.path.exists(best_ppo_path)
    
    # 测试阶段：根据是否有PPO训练权重决定参数（不再考虑BC预训练）
    if use_ppo_trained_weights:
        # 使用PPO训练后的权重：使用训练时的参数
        ppo_lr_actor = 1e-4  # 测试时学习率不影响，但保持一致性
        ppo_lr_critic = 1e-4
        ppo_entropy_coef = getattr(args, 'ppo_entropy_coef', 0.01)  # 测试时熵系数不影响
        logger.info("测试阶段：将加载PPO训练后的完整权重（best_ppo.pth）")
    else:
        # 没有PPO训练权重：使用随机初始化的权重
        ppo_lr_actor = 1e-4
        ppo_lr_critic = 1e-4
        ppo_entropy_coef = getattr(args, 'ppo_entropy_coef', 0.01)
        logger.info("测试阶段：未找到PPO训练权重，将使用随机初始化的权重")
    
    # 创建ppo_agent，参数与train_2d_sunseg.py保持一致
    # 注意：train_2d_sunseg.py中训练时创建的ppo_agent包含use_weak_label_supervision等参数
    # 测试阶段只有在没有PPO和BC权重时才重新创建（不包含这些参数）
    # 但test.py需要创建ppo_agent，为了与训练时保持一致，应该包含这些参数
    
    ppo_agent = PPO_prompt(
        state_dim=259,
        lr_actor=ppo_lr_actor,
        lr_critic=ppo_lr_critic,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        device='cuda',
        input_feature_size=64,
        original_image_size=args.image_size,
        entropy_coef=ppo_entropy_coef
    )
    
    # 设置image encoder lora微调
    # lora_net = LoRA_sam2(net,rank=args.lora_rank)
    # net = lora_net.sam

    # optimisation
    # parameters = list(net.image_encoder.parameters()) + list(
    #     net.sam_mask_decoder.parameters())+ list(net.memory_attention.parameters())+ list(net.memory_encoder.parameters())
    
    # for param in net.image_encoder.parameters():
    #     param.requires_grad = False
    # # for param in net.sam_prompt_encoder.parameters():
    # #     param.requires_grad = False
    
    # parameters = net.parameters()
    # parameters = filter(lambda p : p.requires_grad, net.parameters())
    
    # optimizer = optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs,eta_min=1e-5)


    '''segmentation data'''
    # 使用数据增强模块，测试时使用无增强的transform
    transform_test, transform_test_msk = get_test_transform(
        image_size=args.image_size
    )

    if args.dataset == 'SUNSEG':
        refuge_test_dataset = SUNSEG(args, args.raw_data_path, args.raw_data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test', json=True)
        refuge_val_dataset = SUNSEG(args, args.raw_data_path, args.raw_data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid', json=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)      
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)

    elif args.dataset in ('ISIC2017', 'ISIC2018'):
        refuge_test_dataset = ISIC(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = ISIC(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'REFUGE':
        # REFUGE 数据集：<data_path> 下含 train/val/test，每个子目录中有 image/gt/scribble
        refuge_test_dataset = REFUGE(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = REFUGE(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'PancreasCT':
        # PancreasCT 数据集：<data_path> 下含 train/val/test，每个子目录中有 image/gt/scribble
        refuge_test_dataset = PancreasCT(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = PancreasCT(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'CHAOS':
        # CHAOS 数据集：<data_path> 下含 train/val/test，每个子目录中有 image/gt/scribble
        refuge_test_dataset = CHAOS(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = CHAOS(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=args.b, pin_memory=True, worker_init_fn=worker_init_fn)

    best_model_path = os.path.join(args.save_path, 'best.pth')
    if not os.path.exists(best_model_path):
        logger.error(f"模型权重文件不存在: {best_model_path}")
        raise FileNotFoundError(f"模型权重文件不存在: {best_model_path}")
    
    checkpoint = torch.load(best_model_path)
    
    # 获取模型 state_dict 和 _parameters
    model_state_dict = checkpoint['model']
    model_parameters = checkpoint['parameter']
    
    # 关键：在测试前，确保模型完全重置状态（与train_2d_sunseg.py保持一致）
    # 训练过程中，模型可能处于train()模式，BatchNorm的running stats可能被更新
    # 即使重新加载权重，如果模型状态不对，可能影响结果
    
    # 1. 先设置为eval模式（避免BatchNorm等层使用训练时的running stats）
    net.eval()
    
    # 2. 清理CUDA缓存，确保没有残留状态
    torch.cuda.empty_cache()
    
    # 3. 重新加载模型权重（确保所有状态都被重置为保存时的状态）
    # 注意：state_dict中应该包含BatchNorm的running_mean和running_var
    # 注意：保存模型时调用了net.float()，所以加载后也需要调用net.float()以保持一致
    net.load_state_dict(model_state_dict)
    net.float()  # 与保存时的状态保持一致（保存时调用了net.float()）
    
    # 4. 再次确保模型处于eval模式（双重保险）
    net.eval()
    
    for param in net.image_encoder.parameters():
        param.requires_grad = False
    
    # kmeans参数（与train_2d_sunseg.py保持一致）
    kmeans_target = None
    kmeans_background = None
    # 如果需要，可以手动设置 _parameters
    # for name, param in model_parameters.items():
    #     if name in net._parameters:
    #         net._parameters[name] = param
    
    # 只根据是否存在PPO训练权重加载（不再回退到BC预训练权重）
    if use_ppo_trained_weights:
        logger.info(f"测试阶段：加载PPO训练后的完整权重: {best_ppo_path}")
        ppo_agent.load(best_ppo_path)
        logger.info("PPO训练权重加载完成！")
    
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    
    # 重置PPO agent的buffer（确保状态一致）
    ppo_agent.reset_buffer()
    
    # 加载Memory Bank（与train_2d_sunseg.py保持一致，在PPO加载和reset之后）
    if os.path.exists(args.memory_path):
        with open(args.memory_path, 'rb') as f:
            memory_bank = pickle.load(f)
    else:
        logger.warning(f"Memory Bank文件不存在: {args.memory_path}，使用空列表")
        memory_bank = []
    
    # 最终确保模型处于eval模式（三重保险）
    net.eval()
    
    # 与train_2d_sunseg.py保持一致的测试调用
    # 注意：train_2d_sunseg.py使用epoch变量（最后一个epoch），test.py使用0作为占位符
    # 参数顺序和值都与train_2d_sunseg.py保持一致
    # 与验证时保持一致，使用autocast
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            (eiou, edice, pos_prompt_acc, neg_prompt_acc) = function.test_sam(args, nice_test_loader, 0, net, kmeans_target, kmeans_background, memory_bank, is_save=False, is_show=True, ppo_agent=ppo_agent)
    # with torch.no_grad():
    #     (eiou, edice) = function.validation_sam(args, nice_test_loader, 0, net, kmeans_target,kmeans_background,memory_bank)
    logger.info(f'Test: IOU: {eiou}, DICE: {edice} || @ epoch final.')
    # writer.close()



if __name__ == '__main__':
    seed_torch()
    # torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()


# 尝试冻结 image encoder


# python train_2d_sunseg.py -net sam2 -exp_name S_SUNSEG_full -vis 1 -sam_ckpt /data/whl/Medical-SAM2-main/checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1