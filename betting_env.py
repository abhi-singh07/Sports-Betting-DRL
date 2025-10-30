import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import torch  # For DRL integration

# Suppress pandas future warnings
pd.set_option('future.no_silent_downcasting', True)

class BettingEnv(gym.Env):
    """
    Custom Betting Environment (Gym-compatible).
    - Episode: One season of matches.
    - State: [Implied_Prob_H/D/A, Home/Away_form, Home/Away_winrate, bankroll_norm].
    - Actions: Discrete {0:skip, 1:home, 2:draw, 3:away} or Continuous [0-1 stake, 0-3 outcome].
    - Reward: Profit (win: stake*(odds-1), loss: -stake).
    - Transition: Next match state; outcome from data (or sample implied probs if stochastic).
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, data_dir='data/epl/', split='train', initial_bankroll=1000.0,
                 stochastic=False, continuous_actions=False):
        super(BettingEnv, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.initial_bankroll = initial_bankroll
        self.stake = 1.0  # Fixed stake (fraction of bankroll if continuous)
        self.stochastic = stochastic  # Sample outcomes from implied probs?
        self.continuous_actions = continuous_actions  # Stake continuous?
        
        # Load data
        self.df = pd.read_csv(f"{data_dir}{split}.csv")
        self.n_matches = len(self.df)
        self.current_step = 0
        self.bankroll = initial_bankroll
        self.y_map = {'H': 0, 'D': 1, 'A': 2}  # FIXED: Outcome mapper
        self.odds_cols = ['B365H', 'B365D', 'B365A']
        self.feature_cols = ['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A',
                             'Home_form_rolling', 'Away_form_rolling',
                             'Home_win_rate_rolling', 'Away_win_rate_rolling']
        
        # Spaces
        n_features = len(self.feature_cols) + 1  # + bankroll norm
        if self.continuous_actions:
            self.action_space = spaces.Box(low=np.array([0.0, 0]), high=np.array([1.0, 3]), dtype=np.float32)  # [stake, outcome_idx]
        else:
            self.action_space = spaces.Discrete(4)  # 0=skip,1=H,2=D,3=A (stake=1 fixed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        
        self.last_action = None  # For render
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.bankroll = self.initial_bankroll
        self.last_action = None
        obs = self._get_obs()
        info = {'bankroll': self.bankroll}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= self.n_matches:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].fillna(0).values
        bankroll_norm = self.bankroll / self.initial_bankroll
        return np.append(features, bankroll_norm).astype(np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        if self.current_step >= self.n_matches:
            return self._get_obs(), 0.0, True, True, {'bankroll': self.bankroll}
        
        row = self.df.iloc[self.current_step]
        implied_probs = row[['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']].values
        odds = row[self.odds_cols].fillna(2.0).values
        true_outcome = self.y_map[row['FTR']]  # FIXED: Use y_map
        
        # Parse action
        if self.continuous_actions:
            stake_frac, outcome_idx = action
            stake_frac = np.clip(stake_frac, 0, 1)  # Fraction of bankroll
            stake_abs = stake_frac * self.bankroll
            outcome_idx = int(np.clip(outcome_idx, 0, 2))
            if stake_abs == 0:
                outcome_idx = -1  # Skip if no stake
        else:
            stake_abs = self.stake if action > 0 else 0
            outcome_idx = action - 1 if action > 0 else -1  # -1=skip
        
        # Outcome: True or sampled?
        if self.stochastic:
            outcome_idx = np.random.choice([0,1,2], p=implied_probs)
        else:
            outcome_idx = true_outcome  # Deterministic from data
        
        # Reward
        reward = 0
        terminated = False
        if outcome_idx != -1 and stake_abs > 0:
            if outcome_idx == (action - 1 if not self.continuous_actions else outcome_idx):
                reward = stake_abs * (odds[outcome_idx] - 1)
            else:  # Loss
                reward = -stake_abs
            self.bankroll += reward
            if self.bankroll <= 0:
                terminated = True
        
        # Next state
        self.current_step += 1
        truncated = self.current_step >= self.n_matches
        if truncated:
            reward += 0.01 * (self.bankroll / self.initial_bankroll - 1)  # Bonus for final bankroll
        
        obs = self._get_obs()
        info = {'bankroll': self.bankroll, 'step': self.current_step, 'action': action}
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}: Bankroll {self.bankroll:.2f}, Action {self.last_action}")

    def close(self):
        plt.close('all')

# Example: Multi-episode sim + plot bankroll
def run_sample_episode(env, n_episodes=5):
    bankrolls = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_bank = [env.initial_bankroll]
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # Random policy
            obs, rew, term, trunc, info = env.step(action)
            episode_bank.append(info['bankroll'])
            total_reward += rew
            done = term or trunc
            env.render()
        bankrolls.append(episode_bank)
        roi = (info['bankroll'] / env.initial_bankroll - 1) * 100
        print(f"Ep {ep}: Final Bankroll {info['bankroll']:.2f}, ROI {roi:.1f}%, Total Reward {total_reward:.2f}")
    
    # Plot
    plt.figure(figsize=(10,6))
    for i, bk in enumerate(bankrolls):
        plt.plot(bk, label=f'Episode {i+1}')
    plt.xlabel('Match')
    plt.ylabel('Bankroll')
    plt.title('Sample Betting Episodes (Random Policy)')
    plt.legend()
    plt.savefig('bankroll_curves.png')
    plt.show()

# DRL Integration Example: Train DQN from baselines.py
def train_dqn_on_env(env, episodes=50):
    try:
        from baselines import BettingBaselines  # Assume in same dir
        baselines = BettingBaselines()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n if isinstance(env.action_space, spaces.Discrete) else 3
        dqn_model = baselines.dqn_stub(state_size=state_size, action_size=action_size, episodes=episodes)
        
        # Simple eval episode
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            q_values = dqn_model(state_t)
            action = torch.argmax(q_values).item()
            obs, rew, term, trunc, _ = env.step(action)
            total_reward += rew
            done = term or trunc
        print(f"DQN Eval: Total Reward {total_reward:.2f} (Profit), Final Bankroll {env.bankroll:.2f}")
        return total_reward
    except ImportError:
        print("baselines.py not found—run DQN stub separately.")
        return 0

if __name__ == "__main__":
    # Create env (discrete, deterministic)
    env = BettingEnv(split='train', stochastic=False, continuous_actions=False)
    print(f"Env created: {env.split} split, {env.n_matches} steps/episode.")
    print(f"State space: {env.observation_space.shape}, Action space: {env.action_space}")
    
    # Sample episodes
    run_sample_episode(env, n_episodes=3)
    
    # Train DQN example (uncomment to run)
    # train_dqn_on_env(env, episodes=20)
    
    env.close()