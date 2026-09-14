import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
import os
import glob
import torchvision.models as models
import numpy as np
import math


# --- 通用 Norm 工具：统一使用 BatchNorm2d ---

def make_norm(num_channels: int, use_group_norm: bool = True, num_groups: int = 8) -> nn.Module:
    """
    统一返回 BatchNorm2d。
    保留原有函数签名以兼容调用，但忽略 use_group_norm / num_groups。
    """
    return nn.BatchNorm2d(num_channels)

# --- 2. 共享骨干网络 + Actor/Critic 分支头 设计 ---

class SharedBackbone(nn.Module):
    """
    共享特征提取骨干网络
    由3个卷积块组成，每个块包含 Conv2d、BatchNorm2d、ReLU
    输入通道为 C（拼接后的通道数），逐步压缩至128通道，空间尺寸保持64x64
    输出共享特征张量 (B, 128, 64, 64)
    """
    def __init__(self, in_channels=259, use_group_norm: bool = True):
        super().__init__()

        self.input_norm = make_norm(in_channels, use_group_norm=use_group_norm)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            make_norm(256, use_group_norm=use_group_norm),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, padding=1, bias=False),
            make_norm(192, use_group_norm=use_group_norm),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            make_norm(128, use_group_norm=use_group_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[SharedBackbone] NaN/Inf at input (state concat)")

        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ActorHead(nn.Module):
    def __init__(self, input_feature_size=64, target_heatmap_size=1024, use_group_norm: bool = True):
        super().__init__()
        self.input_feature_size = input_feature_size
        self.target_heatmap_size = target_heatmap_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            make_norm(64, use_group_norm=use_group_norm),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1)
        self.upsample = nn.Upsample(size=(target_heatmap_size, target_heatmap_size), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = F.softmax(x, dim=1)
        return x


class CriticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 input_feature_size=64,
                 original_image_size=1024,
                 use_group_norm: bool = True):

        super(ActorCritic, self).__init__()
        self.original_image_size = original_image_size
        self.target_heatmap_size = original_image_size
        self.scale_factor = original_image_size / self.target_heatmap_size

        self.shared_backbone = SharedBackbone(state_dim, use_group_norm=use_group_norm)
        self.actor_head = ActorHead(input_feature_size, self.target_heatmap_size, use_group_norm=use_group_norm)
        self.critic_head = CriticHead()

    def _convert_index_to_coords(self, action_indices, B):
        heatmap_w = self.target_heatmap_size
        action_indices_exp = action_indices.unsqueeze(-1)
        action_coords_y = action_indices_exp // heatmap_w
        action_coords_x = action_indices_exp % heatmap_w
        heatmap_coords = torch.cat([action_coords_y, action_coords_x], dim=1)
        original_coords = heatmap_coords.float() * (self.original_image_size - 1) / (self.target_heatmap_size - 1)
        original_coords = torch.clamp(original_coords, 0, self.original_image_size - 1)
        return original_coords

    def act(self, state, deterministic=False, head:"str"="pos", compute_critic=True):
        B = state.shape[0]
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("Warning: state contains nan or inf")

        shared_features = self.shared_backbone(state)
        probs_heatmap = self.actor_head(shared_features)

        state_value = None
        if compute_critic:
            state_value = self.critic_head(shared_features)
            state_value = state_value.detach()

        head_idx = 0 if head == "pos" else 1
        probs = probs_heatmap[:, head_idx:head_idx+1, :, :]
        flat_probs = probs.view(B, -1)

        eps = 1e-8
        flat_probs = torch.clamp(flat_probs, min=eps, max=1.0 - eps)
        flat_logits = torch.log(flat_probs)

        if torch.isnan(flat_logits).any() or torch.isinf(flat_logits).any():
            print("Warning: flat_logits contains nan or inf")
        flat_logits = torch.where(torch.isfinite(flat_logits), flat_logits, torch.zeros_like(flat_logits))
        flat_logits = torch.clamp(flat_logits, min=-50.0, max=50.0)

        dist = Categorical(logits=flat_logits)

        if deterministic:
            action_index = torch.argmax(flat_logits, dim=1)
        else:
            action_index = dist.sample()

        action_logprob = dist.log_prob(action_index)
        original_coords = self._convert_index_to_coords(action_index, B)
        return original_coords.detach(), action_index.detach(), action_logprob.detach(), state_value

    def evaluate(self, state, action_indices, head_types:torch.Tensor):
        B = state.shape[0]
        shared_features = self.shared_backbone(state)
        probs_heatmap = self.actor_head(shared_features)
        state_values = self.critic_head(shared_features)

        flat_probs_all = probs_heatmap.view(B, 2, -1)
        flat_probs_pos = flat_probs_all[:, 0, :]
        flat_probs_neg = flat_probs_all[:, 1, :]
        head_types_bool = head_types.to(state.device).view(-1, 1).bool()
        flat_probs = torch.where(head_types_bool, flat_probs_pos, flat_probs_neg)

        eps = 1e-8
        flat_probs = torch.clamp(flat_probs, min=eps, max=1.0 - eps)
        flat_logits = torch.log(flat_probs)
        flat_logits = torch.where(torch.isfinite(flat_logits), flat_logits, torch.zeros_like(flat_logits))
        flat_logits = torch.clamp(flat_logits, min=-50.0, max=50.0)

        dist = Categorical(logits=flat_logits)
        action_logprobs = dist.log_prob(action_indices)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.head_types = []

    def clear(self):
        self.__init__()


class PPO_prompt:
    def __init__(self,
                 state_dim,
                 lr_actor,
                 lr_critic,
                 gamma,
                 K_epochs,
                 eps_clip,
                 device,
                 input_feature_size=64,
                 original_image_size=1024,
                 entropy_coef=0.01,
                 use_group_norm: bool = True):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.use_group_norm = use_group_norm
        self.original_image_size = original_image_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim,
            input_feature_size,
            original_image_size,
            use_group_norm=use_group_norm
        ).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.shared_backbone.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(
            state_dim,
            input_feature_size,
            original_image_size,
            use_group_norm=use_group_norm
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.bad_grad_steps_epoch = 0
        self.param_reset_steps_epoch = 0

    def _select_action_with_head(self, state, deterministic:bool, head:str):
        B = state.shape[0]
        with torch.no_grad():
            original_coords, action_index, action_logprob, state_val = self.policy_old.act(state, deterministic=deterministic, head=head, compute_critic=False)

        self.buffer.states.append(state.detach())
        self.buffer.actions.append(action_index.detach())
        self.buffer.logprobs.append(action_logprob.detach())
        head_type_val = 1 if head == "pos" else 0
        self.buffer.head_types.append(torch.full((B,), head_type_val, device=state.device, dtype=torch.long))
        return original_coords.cpu().numpy()

    def select_action_pos(self, state, deterministic=False):
        return self._select_action_with_head(state, deterministic, head="pos")

    def select_action_neg(self, state, deterministic=False):
        return self._select_action_with_head(state, deterministic, head="neg")

    def select_action(self, state, deterministic=False):
        return self._select_action_with_head(state, deterministic, head="pos")

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.cat(self.buffer.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_heads = torch.cat(self.buffer.head_types, dim=0).detach().to(self.device)

        with torch.no_grad():
            shared_features = self.policy.shared_backbone(old_states)
            old_state_values = self.policy.critic_head(shared_features).detach()
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(old_states)
        indices = torch.randperm(num_samples, device=self.device)

        old_states = old_states[indices]
        old_actions = old_actions[indices]
        old_logprobs = old_logprobs[indices]
        old_heads = old_heads[indices]
        rewards = rewards[indices]
        advantages = advantages[indices]

        bad_grad_steps = 0
        param_reset_steps = 0
        last_ppo_loss = None
        last_dist_entropy = None

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_heads
            )

            total_logprobs = logprobs
            state_values = torch.squeeze(state_values)
            log_ratio = (total_logprobs - old_logprobs.detach()).clamp(min=-20.0, max=20.0)
            ratios = torch.exp(log_ratio)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            ppo_loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy.mean()
            loss = ppo_loss.mean()

            last_ppo_loss = ppo_loss.mean().item()
            last_dist_entropy = dist_entropy.mean().item()

            self.optimizer.zero_grad()
            loss.backward()

            bad_grad = False
            for name, param in self.policy.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    bad_grad = True
                    break

            if bad_grad:
                bad_grad_steps += 1
                self.optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            param_reset_this_step = False
            for name, param in self.policy.named_parameters():
                if not torch.isfinite(param.data).all():
                    with torch.no_grad():
                        if param.data.dim() >= 2:
                            nn.init.kaiming_normal_(param.data)
                        else:
                            param.data.zero_()
                    param_reset_this_step = True

            if param_reset_this_step:
                param_reset_steps += 1

        self.bad_grad_steps_epoch += bad_grad_steps
        self.param_reset_steps_epoch += param_reset_steps

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {
            'ppo_loss': last_ppo_loss,
            'dist_entropy': last_dist_entropy
        }

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def reset_buffer(self):
        self.buffer.clear()
