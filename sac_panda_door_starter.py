# Spring 2026, 535510 Reinforcement Learning
# HW3: SAC
# Instructor: Ping-Chun Hsieh (National Yang Ming Chiao Tung University)

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

import robosuite as suite
from robosuite.wrappers import GymWrapper

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set the hidden layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)

        # set log_std layer
        self.log_std_layer = nn.Linear(128, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(128, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        # get mean
        mu = self.mu_layer(x).tanh()

        # get std
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of SAC paper
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return the deterministic mean action (no sampling)."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        mu = self.mu_layer(x).tanh()
        return mu.tanh()


class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class CriticV(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(CriticV, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value
    
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class SACAgent:
    """SAC agent interacting with environment.

    Attrtibutes:
        actor (nn.Module): actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        vf (nn.Module): critic model to predict state values
        vf_target (nn.Module): target critic model to predict state values
        vf_optimizer (Optimizer): optimizer for training vf
        qf_1 (nn.Module): critic model to predict state-action values
        qf_2 (nn.Module): critic model to predict state-action values
        qf_1_optimizer (Optimizer): optimizer for training qf_1
        qf_2_optimizer (Optimizer): optimizer for training qf_2
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        policy_update_freq (int): policy update frequency
        device (torch.device): cpu / gpu
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env, args=None):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size        
        self.memory = ReplayBuffer(obs_dim, action_dim, self.memory_size, self.batch_size)
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.lr = args.lr
        self.initial_random_steps = args.initial_random_steps
        self.policy_update_freq = args.policy_update_freq
        self.seed = args.seed
        self.num_steps = args.num_steps
        self.eval_interval = args.eval_interval
        self.eval_episodes = args.eval_episodes
        self.checkpoint_dir = args.checkpoint_dir
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # automatic entropy tuning
        self.target_entropy = -np.prod((action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        # actor
        self.actor = Actor(obs_dim, action_dim).to(self.device)

        # v function
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        # q function
        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.lr)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=self.lr)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
            )

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by stochastic gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)

        # train alpha (dual problem)
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # q function loss
        mask = 1 - done
        ########## Your Code (~10 lines)##########
        #frozen target V network 
        next_v = self.vf_target(next_state)

        #zero out terminal states
        next_v = next_v * mask

        #Bellman target
        q_target = (reward + self.gamma * next_v).detach()

        #Q predictions for stored (s, a) 
        qf_1_pred = self.qf_1(state, action)
        qf_2_pred = self.qf_2(state, action)

        #MSE loss
        qf_1_loss = F.mse_loss(qf_1_pred, q_target)
        qf_2_loss = F.mse_loss(qf_2_pred, q_target)


        ########## End of Your Code ##########

        # v function loss
        ########## Your Code (~5 lines)##########
        #min of C networks
        q_pred = torch.min(
            self.qf_1(state, new_action),
            self.qf_2(state, new_action)
        )

        #soft value target
        v_target = (q_pred - alpha * log_prob).detach()

        #MSE loss between V network output and soft value target
        vf_loss = F.mse_loss(self.vf(state), v_target)


        ########## End of Your Code ##########

        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            ########## Your Code (<5 lines)##########

            #push policy to high-Q actions 
            actor_loss = (alpha * log_prob - self.qf_1(state, new_action)).mean()

            ########## End of Your Code ##########

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(())

        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, vf_loss.data, alpha_loss.data

    def evaluate(self, num_episodes: int = 20) -> Tuple[float, float]:
        """Evaluate the agent deterministically for a fixed number of episodes.

        The actor mean action (tanh of mu) is used so evaluation is
        deterministic and comparable across checkpoints.

        Args:
            num_episodes: number of evaluation episodes.

        Returns:
            Tuple of (mean return, std return) across episodes.
        """
        self.actor.eval()
        eval_scores = []

        for ep_idx in range(num_episodes):
            state = self._reset_env()
            done = False
            ep_score = 0.0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action = self.actor.get_deterministic_action(state_tensor).cpu().numpy()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_score += reward
                state = next_state

            eval_scores.append(ep_score)

        self.actor.train()
        mean_score = float(np.mean(eval_scores))
        std_score = float(np.std(eval_scores))
        print(eval_scores)
        return mean_score, std_score

    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """Save model weights and optimizer states to a checkpoint file.

        Args:
            checkpoint_dir: directory where checkpoints are written.
            step: current training step, used in the filename.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
        torch.save(
            {
                "step": step,
                "actor": self.actor.state_dict(),
                "vf": self.vf.state_dict(),
                "vf_target": self.vf_target.state_dict(),
                "qf_1": self.qf_1.state_dict(),
                "qf_2": self.qf_2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "vf_optimizer": self.vf_optimizer.state_dict(),
                "qf_1_optimizer": self.qf_1_optimizer.state_dict(),
                "qf_2_optimizer": self.qf_2_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )
        print(f"\n[Checkpoint] Saved to {path}")

    def _reset_env(self) -> np.ndarray:
        """Reset the environment, handling both (obs, info) and obs-only return."""
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def train(self):
        """Train the agent."""
        self.is_test = False

        state = self._reset_env()
        actor_losses, qf_losses, vf_losses, alpha_losses = [], [], [], []
        scores = []
        score = 0
        ep = 0
        for self.total_step in tqdm(range(1, self.num_steps + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            actor_loss = 0
            qf_loss  = 0
            vf_loss = 0
            alpha_loss = 0
            
            # if episode ends
            if done:
                state = self._reset_env()
                scores.append(score)
                ep += 1
                print(f"\n Episode {ep} (Total step = {self.total_step}): Total Reward = {score}")
                # W&B logging
                wandb.log({
                    "episode": ep,
                    "return": score
                    }) 
                score = 0
                
            # if training is ready
            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                losses = self.update_model()
                actor_loss = losses[0].cpu().numpy()
                qf_loss = losses[1].cpu().numpy()                                                
                vf_loss = losses[2].cpu().numpy()
                alpha_loss = losses[3].cpu().numpy()
                actor_losses.append(actor_loss)
                qf_losses.append(qf_loss)
                vf_losses.append(vf_loss)
                alpha_losses.append(alpha_loss)

            # W&B logging
            wandb.log({
                "step": self.total_step,
                "actor loss": actor_loss,
                "q loss": qf_loss,
                "v loss": vf_loss,
                "alpha loss": alpha_loss
                })

            # --- Checkpoint + Evaluation every eval_interval steps ---
            if self.total_step % self.eval_interval == 0:
                # Save checkpoint
                self.save_checkpoint(self.checkpoint_dir, self.total_step)

                # Evaluate for eval_episodes episodes
                eval_mean, eval_std = self.evaluate(num_episodes=self.eval_episodes)
                print(
                    f"\n[Eval @ step {self.total_step}] "
                    f"Mean Return = {eval_mean:.2f} ± {eval_std:.2f} "
                    f"over {self.eval_episodes} episodes"
                )
                wandb.log({
                    "step": self.total_step,
                    "eval/mean_return": eval_mean,
                    "eval/std_return": eval_std,
                })

                # Reset training env after eval so robosuite doesn't see a
                # terminated/stale episode on the next train step
                state = self._reset_env()
                score = 0

        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.is_test = False

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="door-sac")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--initial-random-steps", type=int, default=5000)
    parser.add_argument("--memory-size", type=int, default=300000)
    parser.add_argument("--num-steps", type=int, default=500000)
    parser.add_argument("--policy-update-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="Save a checkpoint and run evaluation every this many steps.")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes used for each evaluation.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_door",
                        help="Directory to save checkpoint files.")
    args = parser.parse_args()

    # Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    seed_torch(args.seed)

    # W&B init
    wandb.init(project="RL-HW3-SAC-Door", name=args.wandb_run_name, save_code=True)

    # Robosuite Door Opening environment
    raw_env = suite.make(
        env_name="Door",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(raw_env)

    agent = SACAgent(env, args)
    agent.train()