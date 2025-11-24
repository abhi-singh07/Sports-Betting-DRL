import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import torch

# This line is removed/commented out to fix the OptionError in older pandas versions
# pd.set_option('future.no_silent_downcasting', True)


class BettingEnv(gym.Env):
    """ 
    Custom Betting Environment (Gym-compatible).
    - Episode: Represents one full season of matches.
    - State: [Implied_Prob_H/D/A, Home/Away_form, Home/Away_winrate, bankroll_norm].
    - Actions: Discrete {0: skip, 1: home, 2: draw, 3: away}
                or Continuous [stake_fraction (0–1), outcome_index (0–3)].
    - Reward: Betting profit (win: stake * (odds − 1), loss: −stake).
    - Transition: Move to next match; outcome taken from dataset
                    or sampled from implied probabilities when stochastic=True.
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data_dir='data/epl/',
        split='train',
        initial_bankroll=1000.0,
        stochastic=False,
        continuous_actions=False,
        stake=1.0 # <-- ADDED: Default fixed stake for discrete action
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.stochastic = stochastic
        self.continuous_actions = continuous_actions
        self.stake = stake # <-- ADDED: Store the stake

        # Load data
        self.df = pd.read_csv(f"{data_dir}{split}.csv")
        self.odds_cols = ['B365H', 'B365D', 'B365A']

        self.feature_cols = [
            'Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A',
            'Home_form_rolling', 'Away_form_rolling',
            'Home_win_rate_rolling', 'Away_win_rate_rolling'
        ]

        self.state_cols = self.feature_cols
        n_features = len(self.state_cols) + 1  # + bankroll norm

        # Actions
        if continuous_actions:
            # [stake_fraction, outcome_index(0-2)]
            self.action_space = spaces.Box(
                low=np.array([0.0, 0]),
                high=np.array([1.0, 3]),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(4)  # 0 skip, 1 H, 2 D, 3 A

        # Observations
        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(n_features,),
            dtype=np.float32
        )

        self.y_map = {'H': 0, 'D': 1, 'A': 2}

        self.current_step = 0
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bankroll = self.initial_bankroll
        self.current_step = 0

        obs = self._get_obs()
        return obs, {'bankroll': self.bankroll}


    def _get_obs(self):
        if self.current_step >= len(self.df):
            # End-of-episode obs
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        row = self.df.iloc[self.current_step]

        features = row[self.feature_cols].fillna(0).values

        bankroll_norm = self.bankroll / self.initial_bankroll

        obs = np.append(features, bankroll_norm).astype(np.float32)
        return obs


    def step(self, action):
        # Terminal condition for over-step
        if self.current_step >= len(self.df):
            return self._get_obs(), 0.0, True, True, {'bankroll': self.bankroll}

        row = self.df.iloc[self.current_step]

        implied_probs = row[['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']].values
        odds = row[self.odds_cols].fillna(2.0).values
        true_outcome = self.y_map[row['FTR']]

        # Parse action
        if self.continuous_actions:
            stake_frac, pred_idx = action
            stake_frac = float(np.clip(stake_frac, 0.0, 1.0))
            stake_abs = stake_frac * self.bankroll
            pred_idx = int(np.clip(pred_idx, 0, 2))
            skip_flag = (stake_abs <= 1e-8)
        else:
            skip_flag = (action == 0)
            pred_idx = action - 1 if not skip_flag else -1
            stake_abs = self.stake if not skip_flag else 0.0 # <-- FIXED: Use self.stake

        if skip_flag:
            reward = 0.0
        else:
            realized = (
                np.random.choice(3, p=implied_probs)
                if self.stochastic else true_outcome
            )

            # Win/loss
            if realized == pred_idx:
                profit = stake_abs * (odds[pred_idx] - 1)
                reward = profit
            else:
                reward = -stake_abs

            # Update bankroll safely
            self.bankroll += reward
            if self.bankroll <= 0:
                self.bankroll = 0
                terminated = True
                truncated = False
                obs = self._get_obs()
                return obs, reward, terminated, truncated, {'bankroll': self.bankroll}

        # Step forward
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= len(self.df))

        if truncated:
            reward += 0.01 * (self.bankroll / self.initial_bankroll - 1)

        obs = self._get_obs()
        info = {'bankroll': self.bankroll, 'step': self.current_step}

        return obs, float(reward), terminated, truncated, info


    def render(self):
        print(f"[Step {self.current_step}] Bankroll={self.bankroll:.2f}")

    def close(self):
        plt.close('all')