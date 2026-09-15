# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.amp import autocast, GradScaler
import cfg
import func_2d.function as function  
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *
from func_2d.augmentation import get_train_augmentation, get_test_transform
import yaml
import pickle
from func_2d.PPO_prompt import *
import random


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
    #torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id):
	"""确保每个worker的随机种子一致，保证数据加载顺序一致"""
	np.random.seed(42 + worker_id)
	random.seed(42 + worker_id)


def main():

    args = cfg.parse_args()
    
    os.makedirs(args.save_path,exist_ok=True)
    # os.makedirs(args.test_save_path,exist_ok=True)
    os.makedirs(args.test_save_path,exist_ok=True)
    
    with open(os.path.join(args.save_path,'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    
    # net = torch.compile(net)
    # 设置image encoder lora微调
    # lora_net = LoRA_sam2(net,rank=args.lora_rank)
    # net = lora_net.sam

    # optimisation
    # parameters = list(net.image_encoder.parameters()) + list(
    #     net.sam_mask_decoder.parameters())+ list(net.memory_attention.parameters())+ list(net.memory_encoder.parameters())
    
    # 根据开关决定是否冻结整个 SAM
    if args.freeze_sam:
        print("[INFO] 冻结整个 SAM 主干网络，只训练 PPO Prompt Generator")
        for param in net.parameters():
            param.requires_grad = False
    else:
        print("[INFO] 正常训练所有参数")
        # 保持原来的 image_encoder 冻结
        for param in net.image_encoder.parameters():
            param.requires_grad = False
    
    
    # for param in net.parameters():
    #     param.data = param.data.float() 
    # for param in net.sam_prompt_encoder.parameters():
    #     param.requires_grad = False
    
    parameters = list(filter(lambda p : p.requires_grad, net.parameters()))
    
    optimizer = None
    if len(parameters) > 0:
        optimizer = optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    else:
        print("[INFO] 没有需要优化的 SAM 参数，跳过 SAM 优化器创建")
    
    '''load pretrained model'''

    logger = create_logger(args.save_path)
    logger.info(args)

    '''segmentation data'''
    transform_train, transform_train_msk = get_train_augmentation(
        image_size=args.image_size,
        use_augmentation=True
    )
    
    transform_test, transform_test_msk = get_test_transform(
        image_size=args.image_size
    )

    if args.dataset == 'SUNSEG':
        refuge_train_dataset = SUNSEG(args, args.raw_data_path, args.raw_data_path, transform=transform_train, transform_msk=transform_train_msk, mode='train', json=True)
        refuge_test_dataset = SUNSEG(args, args.raw_data_path, args.raw_data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test', json=True)
        refuge_val_dataset = SUNSEG(args, args.raw_data_path, args.raw_data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid', json=True)
        
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn) 
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    elif args.dataset in ('ISIC2017', 'ISIC2018'):
        refuge_train_dataset = ISIC(args, args.data_path, transform=transform_train, transform_msk=transform_train_msk, mode='train')
        refuge_test_dataset = ISIC(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = ISIC(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'REFUGE':
        refuge_train_dataset = REFUGE(args, args.data_path, transform=transform_train, transform_msk=transform_train_msk, mode='train')
        refuge_test_dataset = REFUGE(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = REFUGE(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'PancreasCT':
        refuge_train_dataset = PancreasCT(args, args.data_path, transform=transform_train, transform_msk=transform_train_msk, mode='train')
        refuge_test_dataset = PancreasCT(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = PancreasCT(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'CHAOS':
        refuge_train_dataset = CHAOS(args, args.data_path, transform=transform_train, transform_msk=transform_train_msk, mode='train')
        refuge_test_dataset = CHAOS(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='test')
        refuge_val_dataset = CHAOS(args, args.data_path, transform=transform_test, transform_msk=transform_test_msk, mode='valid')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
        nice_val_loader = DataLoader(refuge_val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    best_dice = 0.0
    kmeans_target = None
    kmeans_background = None
    ppo_lr_actor = getattr(args, 'ppo_lr_actor', 1e-5)
    ppo_lr_critic = getattr(args, 'ppo_lr_critic', 1e-5)
    ppo_gamma = getattr(args, 'ppo_gamma', 0.99)
    ppo_K_epochs = getattr(args, 'ppo_K_epochs', 4)
    ppo_eps_clip = getattr(args, 'ppo_eps_clip', 0.2)
    ppo_entropy_coef = getattr(args, 'ppo_entropy_coef', 0.05)
    ppo_input_feature_size = getattr(args, 'ppo_input_feature_size', 64)
    logger.info(f"PPO参数: lr_actor={ppo_lr_actor}, lr_critic={ppo_lr_critic}, gamma={ppo_gamma}, K_epochs={ppo_K_epochs}, eps_clip={ppo_eps_clip}, entropy_coef={ppo_entropy_coef}")

    ppo_agent = PPO_prompt(
        state_dim=259,
        lr_actor=ppo_lr_actor,
        lr_critic=ppo_lr_critic,
        gamma=ppo_gamma,
        K_epochs=ppo_K_epochs,
        eps_clip=ppo_eps_clip,
        device='cuda',
        input_feature_size=ppo_input_feature_size,
        original_image_size=args.image_size,
        entropy_coef=ppo_entropy_coef
    )

    decrease = 0
    dice_last_epoch = 0
    memory_bank = []
    best_epoch = 0
    scaler = torch.amp.GradScaler('cuda')
    metrics_csv_path = os.path.join(args.save_path, 'training_metrics.csv')
    import csv
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_iou', 'val_dice', 
                            'pos_prompt_acc', 'neg_prompt_acc', 
                            'ppo_loss', 'dist_entropy'])
    
    for epoch in range(args.num_epochs):
        net.train()
        time_start = time.time()
        ppo_agent.policy.train()
        ppo_agent.policy_old.train()
        loss, memory_bank, avg_ppo_loss, avg_dist_entropy = function.train_sam(args, net, optimizer, nice_train_loader, epoch, kmeans_target,kmeans_background,memory_bank,scaler,ppo_agent)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        logger.info(f'time_for_training: {time_end - time_start}')

        if hasattr(ppo_agent, "bad_grad_steps_epoch") and hasattr(ppo_agent, "param_reset_steps_epoch"):
            if ppo_agent.bad_grad_steps_epoch > 0 or ppo_agent.param_reset_steps_epoch > 0:
                logger.info(f"PPO Epoch {epoch}: bad_grad_steps={ppo_agent.bad_grad_steps_epoch}, param_reset_steps={ppo_agent.param_reset_steps_epoch}")
            ppo_agent.bad_grad_steps_epoch = 0
            ppo_agent.param_reset_steps_epoch = 0

        net.eval()
        ppo_agent.policy.eval()
        ppo_agent.policy_old.eval()
        val_iou = None
        val_dice = None
        pos_prompt_acc = None
        neg_prompt_acc = None
        if (epoch % args.val_freq == 0 or epoch == args.num_epochs-1):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    (eiou, edice, pos_prompt_acc, neg_prompt_acc) = function.validation_sam(args, nice_val_loader, epoch, net, kmeans_target,kmeans_background,memory_bank,is_test=False,ppo_agent=ppo_agent)
                    ppo_agent.reset_buffer()
            logger.info(f'Validation: IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            val_iou = eiou
            val_dice = edice
            if edice > best_dice:
                best_dice = edice
                best_epoch = epoch 
                net.float()
                torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.save_path, 'best.pth'))
                ppo_agent.save(os.path.join(args.save_path, 'best_ppo.pth'))
                with open(args.memory_path, 'wb') as f:
                    pickle.dump(memory_bank, f)
        ppo_loss = avg_ppo_loss
        dist_entropy = avg_dist_entropy
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss if loss is not None else '', val_iou if val_iou is not None else '', val_dice if val_dice is not None else '', pos_prompt_acc if pos_prompt_acc is not None else '', neg_prompt_acc if neg_prompt_acc is not None else '', ppo_loss if ppo_loss is not None else '', dist_entropy if dist_entropy is not None else ''])

    if len(memory_bank) > 0:
        with open(args.memory_path, 'wb') as f:
            pickle.dump(memory_bank, f)
        logger.info(f"训练结束，已保存最新的Memory Bank（条目数: {len(memory_bank)}）")

    checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
    model_state_dict = checkpoint['model']
    best_ppo_path = os.path.join(args.save_path, 'best_ppo.pth')
    use_ppo_trained_weights = os.path.exists(best_ppo_path)
    ppo_agent = PPO_prompt(state_dim=259,lr_actor=ppo_lr_actor,lr_critic=ppo_lr_critic,gamma=ppo_gamma,K_epochs=ppo_K_epochs,eps_clip=ppo_eps_clip,device='cuda',input_feature_size=ppo_input_feature_size,original_image_size=args.image_size,entropy_coef=ppo_entropy_coef)
    net.eval()
    torch.cuda.empty_cache()
    net.load_state_dict(model_state_dict)
    net.float()
    net.eval()
    if use_ppo_trained_weights:
        ppo_agent.load(best_ppo_path)
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    ppo_agent.reset_buffer()
    if os.path.exists(args.memory_path):
        with open(args.memory_path, 'rb') as f:
            memory_bank = pickle.load(f)
    else:
        memory_bank = []
    net.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            (eiou, edice, pos_prompt_acc, neg_prompt_acc) = function.test_sam(args, nice_test_loader, 0, net, kmeans_target,kmeans_background,memory_bank,is_save=False,is_show=True,ppo_agent=ppo_agent)
    logger.info(f'Best Test: IOU: {eiou}, DICE: {edice} || @ at epoch {best_epoch}.')
    fp = open(os.path.join(args.save_path,'best.txt'),'w')
    fp.write(f'Best Test: IOU: {eiou}, DICE: {edice} || @ at epoch {best_epoch}.')
    fp.close()

if __name__ == '__main__':
    seed_torch()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()
