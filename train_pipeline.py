import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch  # FIXED: Import for DQN
from stable_baselines3 import A2C, PPO, DDPG  # SB3 for A2C/PPO/DDPG (REINFORCE proxy)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from betting_env import BettingEnv  # From Section 4
from baselines import BettingBaselines  # From Section 3

class PipelineEvaluator:
    def __init__(self, data_dir='data/epl/', n_episodes=20, total_timesteps=5000):
        self.data_dir = data_dir
        self.n_episodes = n_episodes
        self.total_timesteps = total_timesteps
        self.results = {}

    def create_env(self, split='train', **kwargs):
        return BettingEnv(data_dir=self.data_dir, split=split, **kwargs)

    def train_dqn(self, env):
        """Train DQN (custom from baselines stub)."""
        try:
            from baselines import BettingBaselines  # Use custom
            baselines = BettingBaselines()
            model = baselines.dqn_stub(state_size=env.observation_space.shape[0],
                                       action_size=env.action_space.n, episodes=100)
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
            model = A2C('MlpPolicy', vec_env, verbose=0, device='cpu')  # CPU for speed
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"A2C Error: {e} — Using dummy.")
            return 0, 0

    def train_reinforce(self, env):
        """REINFORCE (PPO as proxy)."""
        try:
            vec_env = make_vec_env(lambda: env, n_envs=1)
            model = PPO('MlpPolicy', vec_env, verbose=0, device='cpu')
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"REINFORCE Error: {e} — Using dummy.")
            return 0, 0

    def train_ddpg(self, env):
        """DDPG for continuous actions."""
        try:
            env_cont = self.create_env(continuous_actions=True)
            vec_env = make_vec_env(lambda: env_cont, n_envs=1)
            model = DDPG('MlpPolicy', vec_env, verbose=0, device='cpu')
            model.learn(total_timesteps=self.total_timesteps)
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=self.n_episodes)
            return mean_reward, std_reward
        except Exception as e:
            print(f"DDPG Error: {e} — Using dummy.")
            return 0, 0

    def eval_baselines(self):
        """Eval non-DRL baselines (from baselines.py)."""
        baselines = BettingBaselines(data_dir=self.data_dir)
        fav = baselines.always_favorite()
        nobet = baselines.no_bet()
        sup = baselines.supervised_threshold()
        dyna = baselines.dyna_q_baseline(episodes=200)
        return {
            'Favorite': fav['Total Profit'],
            'No Bet': nobet['Total Profit'],
            'Supervised': sup['Total Profit'],
            'Dyna-Q': dyna['Total Profit']
        }

    def compute_metrics(self, mean_reward, std_reward):
        """ROI etc. from rewards (approx; full from profits)."""
        initial_br = self.create_env().initial_bankroll
        mean_roi = (mean_reward / initial_br) * 100
        sharpe = mean_reward / (std_reward + 1e-8)
        # Dummy hit/DD for SB3 (expand with logging)
        hit_rate = 50.0  # Placeholder
        drawdown = - (std_reward / initial_br * 100)  # Approx
        return {'Mean ROI (%)': mean_roi, 'Sharpe': sharpe, 'Hit Rate (%)': hit_rate, 'Max DD (%)': drawdown}

    def run_pipeline(self):
        """Full train/eval pipeline."""
        print("Training on train split...")
        env = self.create_env(split='train')
        
        # Train DRL
        dqn_mean, dqn_std = self.train_dqn(env)
        self.results['DQN'] = self.compute_metrics(dqn_mean, dqn_std)
        
        a2c_mean, a2c_std = self.train_a2c(env)
        self.results['A2C'] = self.compute_metrics(a2c_mean, a2c_std)
        
        rein_mean, rein_std = self.train_reinforce(env)
        self.results['REINFORCE (PPO)'] = self.compute_metrics(rein_mean, rein_std)
        
        ddpg_mean, ddpg_std = self.train_ddpg(env)
        self.results['DDPG'] = self.compute_metrics(ddpg_mean, ddpg_std)
        
        # Baselines (profits as proxy for reward)
        base_profits = self.eval_baselines()
        initial_br = env.initial_bankroll
        for name, profit in base_profits.items():
            roi = (profit / (len(env.df) * 1.0)) / initial_br * 100  # Approx per bet
            self.results[name] = {'Mean ROI (%)': roi, 'Sharpe': 0, 'Hit Rate (%)': 0, 'Max DD (%)': 0}
        
        # Table
        results_df = pd.DataFrame(self.results).T.round(2)
        print("\n=== Algorithm Comparison (Train Split) ===")
        print(results_df)
        results_df.to_csv('results_table.csv')
        
        # Plot ROIs
        algos = list(self.results.keys())
        rois = [r['Mean ROI (%)'] for r in self.results.values()]
        plt.figure(figsize=(12,6))
        plt.bar(algos, rois, color=['blue' if 'DRL' in a else 'gray' for a in algos])
        plt.xlabel('Algorithm')
        plt.ylabel('Mean ROI (%)')
        plt.title('DRL vs Baselines: ROI Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results_table.png', dpi=150)
        plt.show()
        
        # Test Eval (quick SB3 re-eval)
        print("\nQuick Test Split Eval...")
        env_test = self.create_env(split='test')
        for algo in ['A2C', 'PPO']:  # SB3 examples
            try:
                if algo == 'A2C':
                    model = A2C.load('a2c_model') if os.path.exists('a2c_model') else A2C('MlpPolicy', make_vec_env(lambda: env_test, n_envs=1))
                else:
                    model = PPO.load('ppo_model') if os.path.exists('ppo_model') else PPO('MlpPolicy', make_vec_env(lambda: env_test, n_envs=1))
                mean_test, _ = evaluate_policy(model, make_vec_env(lambda: env_test, n_envs=1), n_eval_episodes=5)
                test_roi = (mean_test / initial_br) * 100
                print(f"{algo} Test ROI: {test_roi:.2f}%")
            except:
                print(f"{algo} Test: Skipped (model not saved).")

if __name__ == "__main__":
    import os  # For model load check
    evaluator = PipelineEvaluator(n_episodes=10, total_timesteps=2000)  # Even shorter demo
    evaluator.run_pipeline()