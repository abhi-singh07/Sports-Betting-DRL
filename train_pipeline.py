import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from betting_env import BettingEnv
from baselines import BettingBaselines
from football_data_preprocessor import FootballDataImporter

class PipelineEvaluator:
    def __init__(self, data_dir='data/epl/', n_episodes=20, total_timesteps=10000): # <-- INCREASED TIMESTEPS to 10000
        self.data_dir = data_dir
        self.n_episodes = n_episodes
        self.total_timesteps = total_timesteps
        self.results = {}

    def create_env(self, split='train', **kwargs):
        if not os.path.exists(f"{self.data_dir}{split}.csv"):
            raise FileNotFoundError(f"Missing {split}.csv. Run the data loading pipeline first.")
            
        return BettingEnv(data_dir=self.data_dir, split=split, **kwargs)

    def train_dqn(self, env):
        """Train DQN (custom from baselines stub)."""
        try:
            from baselines import BettingBaselines 
            baselines = BettingBaselines(data_dir=self.data_dir)
            # DQN episodes kept low as the stub iterates over the entire dataset per episode
            model = baselines.dqn_stub(state_size=env.observation_space.shape[0],
                                       action_size=env.action_space.n, episodes=50) 
            
            # Eval: Run n_episodes for mean/std
            rewards = []
            for _ in range(self.n_episodes):
                obs, _ = env.reset()
                ep_reward = 0
                done = False
                while not done:
                    state_t = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = model(state_t)
                    action = torch.argmax(q_values).item()
                    obs, rew, term, trunc, _ = env.step(action)
                    ep_reward += rew
                    done = term or trunc
                rewards.append(ep_reward)
            return np.mean(rewards), np.std(rewards)
        except Exception as e:
            print(f"DQN Error: {e} — Skipping.")
            return 0, 0

    def train_a2c(self, env):
        """Train A2C (SB3)."""
        try:
            vec_env = make_vec_env(lambda: env, n_envs=1)
            # LEARNING RATE REDUCED for stability
            model = A2C('MlpPolicy', vec_env, verbose=0, device='cpu', learning_rate=5e-4) 
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"A2C Error: {e} — Using dummy.")
            return 0, 0

    def train_ppo(self, env):
        """Train PPO (SB3)."""
        try:
            vec_env = make_vec_env(lambda: env, n_envs=1)
            # LEARNING RATE REDUCED and n_steps increased for stability
            model = PPO('MlpPolicy', vec_env, verbose=0, device='cpu', learning_rate=3e-4, n_steps=2048)
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"PPO Error: {e} — Using dummy.")
            return 0, 0

    def train_ddpg(self, env):
        """DDPG for continuous actions (SB3)."""
        try:
            env_cont = self.create_env(continuous_actions=True) 
            vec_env = make_vec_env(lambda: env_cont, n_envs=1)
            model = DDPG('MlpPolicy', vec_env, verbose=0, device='cpu', learning_rate=1e-3)
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"DDPG Error: {e} — Using dummy. Check continuous action support.")
            return 0, 0

    def eval_baselines(self):
        """Eval non-DRL baselines (from baselines.py)."""
        baselines = BettingBaselines(data_dir=self.data_dir)
        fav = baselines.always_favorite()
        nobet = baselines.no_bet()
        sup = baselines.supervised_threshold()
        dyna = baselines.dyna_q_baseline(episodes=200)
        sarsa = baselines.sarsa_baseline(episodes=200) # <-- NEW: SARSA
        return {
            'Favorite': fav,
            'No Bet': nobet,
            'Supervised': sup,
            'Dyna-Q': dyna,
            'SARSA': sarsa # <-- NEW: Include SARSA results
        }

    def compute_metrics(self, mean_reward, std_reward, total_matches):
        """ROI, Sharpe Ratio, etc. from rewards."""
        # Use a dummy environment instance to get the fixed parameters
        dummy_env = self.create_env(split='train')
        stake = dummy_env.stake 
        initial_br = dummy_env.initial_bankroll
        
        # Approximate ROI based on total possible turnover (max profit/loss)
        total_possible_turnover = total_matches * stake
        
        mean_roi = (mean_reward / total_possible_turnover) * 100
        sharpe = mean_reward / (std_reward + 1e-8)
        
        # Placeholder for complex metrics
        hit_rate = 0.0
        drawdown = 0.0
        
        return {'Mean Reward ($)': mean_reward, 'Mean ROI (%)': mean_roi, 'Sharpe Ratio': sharpe, 'Hit Rate (%)': hit_rate, 'Max DD (%)': drawdown}


    def run_pipeline(self):
        """
        1. Ensure data is ready. 
        2. Train/Eval DRL agents on train split. 
        3. Eval Baselines on test split.
        """
        
        # --- 1. Data Preparation Check (Unchanged) ---
        print("Ensuring data splits are current...")
        importer = FootballDataImporter(save_dir=self.data_dir)
        train_seasons = [2017, 2018, 2019, 2020] 
        val_seasons = [2021]
        test_seasons = [2022]
        
        try:
            importer.load_and_preprocess_all_by_season(
                train_seasons=train_seasons, 
                val_seasons=val_seasons, 
                test_seasons=test_seasons, 
                division='E0'
            )
        except Exception as e:
            print(f"Data preparation failed: {e}. Ensure you have internet and correct seasons are available.")
            return

        print("\n--- 2. Training DRL Algorithms (Train Set) ---")
        env_train = self.create_env(split='train')
        total_matches_train = len(pd.read_csv(f"{self.data_dir}train.csv"))
        
        # Train DQN
        dqn_mean, dqn_std = self.train_dqn(env_train)
        self.results['DQN'] = self.compute_metrics(dqn_mean, dqn_std, total_matches_train)
        
        # Train A2C
        a2c_mean, a2c_std = self.train_a2c(env_train)
        self.results['A2C'] = self.compute_metrics(a2c_mean, a2c_std, total_matches_train)

        # Train PPO
        ppo_mean, ppo_std = self.train_ppo(env_train)
        self.results['PPO'] = self.compute_metrics(ppo_mean, ppo_std, total_matches_train)
        
        # Train DDPG
        ddpg_mean, ddpg_std = self.train_ddpg(env_train)
        self.results['DDPG'] = self.compute_metrics(ddpg_mean, ddpg_std, total_matches_train)
        
        
        # --- 3. Evaluate Baselines (Test Set) ---
        print("\n--- 3. Evaluating Baselines (Test Set) ---")
        base_results = self.eval_baselines()
        
        for name, metrics in base_results.items():
            self.results[f'Baseline: {name}'] = {
                'Mean Reward ($)': metrics['Total Profit'],
                'Mean ROI (%)': metrics['ROI (%)'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Hit Rate (%)': metrics['Hit Rate (%)'],
                'Max DD (%)': metrics['Max Drawdown (%)']
            }
        
        # --- 4. Final Comparison Table ---
        results_df = pd.DataFrame(self.results).T
        results_df.index = [i.replace('Baseline: ', '') for i in results_df.index]
        
        print("\n=== Algorithm Comparison (Train/Test Mix) ===")
        print(results_df.round(2).to_string())
        results_df.to_csv('results_table_full_enhanced.csv')
        
        # --- 5. Plotting ROI and Sharpe ---
        print("\nGenerating comparison plots...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: ROI
        rois = results_df['Mean ROI (%)']
        rois.plot(kind='bar', ax=axes[0], color=['blue', 'green', 'orange', 'red', 'darkgray', 'gray', 'silver', 'lightgray'])
        axes[0].set_title('Mean Return on Investment (ROI %)')
        axes[0].set_ylabel('Mean ROI (%)')
        axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Sharpe Ratio
        sharpes = results_df['Sharpe Ratio']
        sharpes.plot(kind='bar', ax=axes[1], color=['blue', 'green', 'orange', 'red', 'darkgray', 'gray', 'silver', 'lightgray'])
        axes[1].set_title('Risk-Adjusted Performance (Sharpe Ratio)')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('comparison_plots_full_enhanced.png', dpi=150)
        plt.show()
        print("\n✓ Results saved to 'results_table_full_enhanced.csv' and 'comparison_plots_full_enhanced.png'")

if __name__ == "__main__":
    # Note: Training time will increase significantly due to 10000 timesteps.
    evaluator = PipelineEvaluator(n_episodes=10, total_timesteps=10000) 
    evaluator.run_pipeline()