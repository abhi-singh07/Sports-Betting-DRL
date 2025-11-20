from baselines import BettingBaselines
from collections import deque
from gymnasium import spaces
import gymnasium as gym
import torch
from torch import nn, optim
import numpy as np
import random


class BettingAlgorithms(BettingBaselines):
    def __init__(self, data_dir='data/epl/', split='train'):
        super().__init__(data_dir=data_dir)
        self.split = split
        self._setup_env_data()
        
    def _setup_env_data(self):
        """Setup environment-specific data based on split"""
        self.load_data(self.split)
        
        if self.split == 'train':
            self.X = self.X_train
            self.y = self.y_train
            self.odds = self.train_df[self.odds_cols].fillna(2.0).values
        else:
            self.X = self.X_test
            self.y = self.y_test
            self.odds = self.odds_test
        
        # Define spaces
        obs_dim = self.X.shape[1] + 1  # +1 for bankroll
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        self.current_index = 0
        self.bankroll = self.initial_bankroll
    
    def get_state(self):
        """Get current state with bankroll"""
        if self.current_index >= len(self.X):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        base_features = self.X[self.current_index]
        bankroll_norm = self.bankroll / self.initial_bankroll
        state = np.append(base_features, bankroll_norm)
        return state.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        self.current_index = 0
        self.bankroll = self.initial_bankroll
        return self.get_state(), {}
    
    def step(self, action):
        outcome = self.y[self.current_index]
        odds = self.odds[self.current_index]
        
        reward = 0
        done = False
        
        if action > 0:
            if action - 1 == outcome:
                profit = self.stake * (odds[action - 1] - 1)
                reward = profit
                self.bankroll += profit
            else:
                loss = self.stake
                reward = -loss
                self.bankroll -= loss
            
            if self.bankroll <= 0:
                done = True
                reward = -10  # Bankruptcy penalty
        
        self.current_index += 1
        if self.current_index >= len(self.X):
            done = True
        
        next_state = self.get_state()
        info = {'bankroll': self.bankroll, 'step': self.current_index}
        
        return next_state, reward, done, False, info
    
    # DQN AGENT
    def dqn_agent(self):

        class DQNAgent:
            def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
                self.state_size = state_size
                self.action_size = action_size
                self.gamma = gamma
                self.epsilon = 1.0
                self.epsilon_min = 0.01
                self.epsilon_decay = 0.995

                # Networks
                self.model = self._build_model()
                self.target_model = self._build_model()
                self.update_target_model()

                # Optim + replay
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                self.memory = deque(maxlen=10000)
                self.batch_size = 32
                self.loss_fn = nn.MSELoss()

            def _build_model(self):
                model = nn.Sequential(
                    nn.Linear(self.state_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.action_size),
                )
                return model

            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

            def act(self, state):
                if np.random.rand() < self.epsilon:
                    return np.random.randint(self.action_size)
                with torch.no_grad():
                    q_vals = self.model(torch.tensor(state).float()).numpy()
                return int(np.argmax(q_vals))

            def replay(self):
                if len(self.memory) < self.batch_size:
                    return

                batch = random.sample(self.memory, self.batch_size)

                states = np.array([b[0] for b in batch])
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])
                next_states = np.array([b[3] for b in batch])
                dones = np.array([b[4] for b in batch])

                # Convert numpy arrays to tensors (much faster)
                states = torch.from_numpy(states).float()
                actions = torch.from_numpy(actions).long()
                rewards = torch.from_numpy(rewards).float()
                next_states = torch.from_numpy(next_states).float()
                dones = torch.from_numpy(dones).float()

                # Q(s,a)
                q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Target Q
                with torch.no_grad():
                    next_q = self.target_model(next_states).max(dim=1)[0]
                target = rewards + (1 - dones) * self.gamma * next_q

                loss = self.loss_fn(q_values, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Decay exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            def update_target_model(self):
                self.target_model.load_state_dict(self.model.state_dict())

            def train(self, env, episodes=100):
                for ep in range(episodes):
                    state, _ = env.reset()
                    done = False
                    total_reward = 0

                    while not done:
                        action = self.act(state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        self.remember(state, action, reward, next_state, done)
                        self.replay()

                        state = next_state
                        total_reward += reward

                    if ep % 10 == 0:
                        self.update_target_model()
                        print(f"Episode {ep}: Reward={total_reward:.2f}, Eps={self.epsilon:.3f}")

            def test(self, env, episodes=10):
                rewards = []
                for ep in range(episodes):
                    state, _ = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action = np.argmax(
                            self.model(torch.tensor(state).float()).detach().numpy()
                        )
                        next_state, reward, term, trunc, _ = env.step(action)
                        done = term or trunc
                        state = next_state
                        total_reward += reward
                    rewards.append(total_reward)
                return rewards

        # ---- Instantiate DQN agent ----
        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n
        agent = DQNAgent(state_dim, action_dim)

        # ---- Train & Test ----
        agent.train(self, episodes=100)
        return agent
    

    # A2C AGENT
    def a2c_agent(self):

        class ActorNetwork(nn.Module):
            """Policy network - outputs action probabilities"""
            def __init__(self, state_size, action_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, action_size)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                action_probs = torch.softmax(self.fc3(x), dim=-1)
                return action_probs

        class CriticNetwork(nn.Module):
            """Value network - outputs state value V(s)"""
            def __init__(self, state_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                value = self.fc3(x)
                return value

        class A2CAgent:
            def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                        entropy_coef=0.01):
                self.state_size = state_size
                self.action_size = action_size
                self.gamma = gamma
                self.entropy_coef = entropy_coef  # Encourage exploration
                
                # Actor and Critic networks
                self.actor = ActorNetwork(state_size, action_size)
                self.critic = CriticNetwork(state_size)
                
                # Separate optimizers
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
            
            def act(self, state):
                """Sample action from policy and return log probability"""
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                
                # Create categorical distribution and sample
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
                return action.item()
            
            def evaluate_action(self, state, action):
                """Get log probability and entropy for a state-action pair"""
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = torch.LongTensor([action])
                
                action_probs = self.actor(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                
                log_prob = dist.log_prob(action_tensor)
                entropy = dist.entropy()
                
                return log_prob, entropy
            
            def compute_returns(self, rewards, dones, next_value):
                """
                Compute discounted returns (targets for critic)
                Uses bootstrapping with next_value for the last state
                """
                returns = []
                R = next_value
                
                for reward, done in zip(reversed(rewards), reversed(dones)):
                    if done:
                        R = 0  # Terminal state has no future value
                    R = reward + self.gamma * R
                    returns.insert(0, R)
                
                return torch.FloatTensor(returns)
            
            def train(self, env, episodes=100):
                """Train A2C agent"""
                for ep in range(episodes):
                    state, _ = env.reset()
                    done = False
                    
                    # Storage for episode
                    log_probs = []
                    values = []
                    rewards = []
                    entropies = []
                    dones_list = []
                    
                    total_reward = 0
                    steps = 0
                    
                    # Collect trajectory
                    while not done:
                        # Get action from policy
                        action = self.act(state)
                        
                        # Get value estimate and log probability
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        value = self.critic(state_tensor)
                        log_prob, entropy = self.evaluate_action(state, action)
                        
                        # Take action in environment
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        # Store
                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(reward)
                        entropies.append(entropy)
                        dones_list.append(done)
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                    
                    # Bootstrap value for last state (if not terminal)
                    if not dones_list[-1]:
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                        next_value = self.critic(next_state_tensor).item()
                    else:
                        next_value = 0
                    
                    # Compute returns (Monte Carlo targets)
                    returns = self.compute_returns(rewards, dones_list, next_value)
                    
                    # Convert lists to tensors
                    log_probs = torch.cat(log_probs)
                    values = torch.cat(values).squeeze()
                    entropies = torch.cat(entropies)
                    
                    # Compute advantages: A(s,a) = Q(s,a) - V(s) ≈ R - V(s)
                    advantages = returns - values.detach()
                    
                    # Actor loss: -log π(a|s) * A(s,a) - entropy_bonus
                    actor_loss = -(log_probs * advantages).mean()
                    actor_loss -= self.entropy_coef * entropies.mean()  # Entropy bonus
                    
                    # Critic loss: MSE between predicted value and actual return
                    critic_loss = ((values - returns) ** 2).mean()
                    
                    # Update actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # Gradient clipping
                    self.actor_optimizer.step()
                    
                    # Update critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
                    
                    # Logging
                    if ep % 10 == 0:
                        print(f"Episode {ep}: Reward={total_reward:.2f}, "
                            f"Steps={steps}, "
                            f"Actor Loss={actor_loss.item():.3f}, "
                            f"Critic Loss={critic_loss.item():.3f}")
            
            def test(self, env, episodes=10):
                """Test A2C agent (greedy policy)"""
                rewards = []
                
                for ep in range(episodes):
                    state, _ = env.reset()
                    done = False
                    total_reward = 0
                    steps = 0
                    actions_taken = []
                    
                    while not done:
                        # Greedy action selection (use mode of distribution)
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action_probs = self.actor(state_tensor)
                        
                        # Take action with highest probability (greedy)
                        action = torch.argmax(action_probs).item()
                        
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                        actions_taken.append(action)
                    
                    rewards.append(total_reward)
                    
                    # Log first episode details
                    if ep == 0:
                        print(f"\n[Test Episode 0 Details]")
                        print(f"  Steps: {steps}")
                        print(f"  Total reward: {total_reward:.2f}")
                        print(f"  Actions: Skip={actions_taken.count(0)}, "
                            f"H={actions_taken.count(1)}, D={actions_taken.count(2)}, "
                            f"A={actions_taken.count(3)}")
                
                return rewards
            
        # ---- Instantiate and train ----
        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n
        agent = A2CAgent(state_dim, action_dim)
        
        agent.train(self, episodes=100)
            
        return agent
        


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print("="*70)
    print(" SPORTS BETTING RL: DQN vs A2C COMPARISON")
    print("="*70)
    
    # 1. ENVIRONMENT VERIFICATION
    print("\n" + "="*70)
    print("1. ENVIRONMENT VERIFICATION")
    print("="*70)
    
    env_train = BettingAlgorithms(data_dir='data/epl/', split='train')
    env_test = BettingAlgorithms(data_dir='data/epl/', split='test')
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(env_train.X)} matches")
    print(f"  Test:  {len(env_test.X)} matches")
    print(f"  Train outcomes (H/D/A): {np.bincount(env_train.y)}")
    print(f"  Test outcomes (H/D/A):  {np.bincount(env_test.y)}")
    print(f"  State dimension: {env_train.observation_space.shape[0]}")
    print(f"  Action space: {env_train.action_space.n} (Skip, Home, Draw, Away)")
    
    # 2. BASELINE PERFORMANCE
    print("\n" + "="*70)
    print("2. BASELINE PERFORMANCE (Test Set)")
    print("="*70)
    
    baselines = BettingBaselines(data_dir='data/epl/')
    
    print("\nRunning baselines...")
    baseline_results = {}
    
    # Always Favorite
    fav_result = baselines.always_favorite()
    baseline_results['Always Favorite'] = fav_result
    print(f"  ✓ Always Favorite: Profit={fav_result['Total Profit']:.2f}, "
          f"ROI={fav_result['ROI (%)']:.2f}%")
    
    # No Bet
    nobet_result = baselines.no_bet()
    baseline_results['No Bet'] = nobet_result
    print(f"  ✓ No Bet (Control): Profit={nobet_result['Total Profit']:.2f}, "
          f"ROI={nobet_result['ROI (%)']:.2f}%")
    
    # Supervised Learning
    sup_result = baselines.supervised_threshold(threshold=0.6)
    baseline_results['Supervised (LR)'] = sup_result
    print(f"  ✓ Supervised (threshold=0.6): Profit={sup_result['Total Profit']:.2f}, "
          f"ROI={sup_result['ROI (%)']:.2f}%")
    
    # Dyna-Q (model-based)
    print("\n  Running Dyna-Q (this may take a minute)...")
    dyna_result = baselines.dyna_q_baseline(episodes=200)
    baseline_results['Dyna-Q'] = dyna_result
    print(f"  ✓ Dyna-Q: Profit={dyna_result['Total Profit']:.2f}, "
          f"ROI={dyna_result['ROI (%)']:.2f}%")
    
    # 3. TRAIN DQN
    print("\n" + "="*70)
    print("3. TRAINING DQN AGENT")
    print("="*70)
    
    # Reset environment for training
    env_train_dqn = BettingAlgorithms(data_dir='data/epl/', split='train')
    
    # Train DQN
    print("\nTraining DQN for 100 episodes...")
    dqn_agent = env_train_dqn.dqn_agent()
    
    # Test DQN
    print("\nEvaluating DQN on test set...")
    env_test_dqn = BettingAlgorithms(data_dir='data/epl/', split='test')
    dqn_test_rewards = dqn_agent.test(env_test_dqn, episodes=10)
    
    dqn_mean = np.mean(dqn_test_rewards)
    dqn_std = np.std(dqn_test_rewards)
    
    print(f"\nDQN Test Results:")
    print(f"  Mean Reward: {dqn_mean:.2f}")
    print(f"  Std Reward:  {dqn_std:.2f}")
    print(f"  Min/Max:     {np.min(dqn_test_rewards):.2f} / {np.max(dqn_test_rewards):.2f}")
    
    # 4. TRAIN A2C
    print("\n" + "="*70)
    print("4. TRAINING A2C AGENT")
    print("="*70)
    
    # Reset environment for training
    env_train_a2c = BettingAlgorithms(data_dir='data/epl/', split='train')
    
    # Train A2C
    print("\nTraining A2C for 100 episodes...")
    a2c_agent = env_train_a2c.a2c_agent()
    
    # Test A2C
    print("\nEvaluating A2C on test set...")
    env_test_a2c = BettingAlgorithms(data_dir='data/epl/', split='test')
    a2c_test_rewards = a2c_agent.test(env_test_a2c, episodes=10)
    
    a2c_mean = np.mean(a2c_test_rewards)
    a2c_std = np.std(a2c_test_rewards)
    
    print(f"\nA2C Test Results:")
    print(f"  Mean Reward: {a2c_mean:.2f}")
    print(f"  Std Reward:  {a2c_std:.2f}")
    print(f"  Min/Max:     {np.min(a2c_test_rewards):.2f} / {np.max(a2c_test_rewards):.2f}")
    
    # 5. COMPARISON TABLE
    print("\n" + "="*70)
    print("5. FINAL COMPARISON (All Methods)")
    print("="*70)
    
    # Compile all results
    all_results = {}
    
    # Baselines
    for name, result in baseline_results.items():
        all_results[name] = result
    
    # DQN
    dqn_metrics = {
        'Total Profit': dqn_mean,
        'ROI (%)': (dqn_mean / (len(env_test.X) * env_test.stake)) * 100 if dqn_mean != 0 else 0,
        'Sharpe Ratio': dqn_mean / (dqn_std + 1e-8),
        'Hit Rate (%)': 0,  # Would need to track wins/losses
        'Max Drawdown (%)': 0,  # Would need to track cumulative
    }
    all_results['DQN'] = dqn_metrics
    
    # A2C
    a2c_metrics = {
        'Total Profit': a2c_mean,
        'ROI (%)': (a2c_mean / (len(env_test.X) * env_test.stake)) * 100 if a2c_mean != 0 else 0,
        'Sharpe Ratio': a2c_mean / (a2c_std + 1e-8),
        'Hit Rate (%)': 0,
        'Max Drawdown (%)': 0,
    }
    all_results['A2C'] = a2c_metrics
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(all_results).T
    results_df = results_df[['Total Profit', 'ROI (%)', 'Sharpe Ratio', 'Hit Rate (%)', 'Max Drawdown (%)']]
    
    print("\n" + results_df.to_string())
    
    # Save results
    results_df.to_csv('results_comparison.csv')
    print("\n✓ Results saved to 'results_comparison.csv'")
    
    # 6. VISUALIZATIONS
    print("\n" + "="*70)
    print("6. GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot 1: Profit Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Total Profit
    ax1 = axes[0, 0]
    methods = list(all_results.keys())
    profits = [all_results[m]['Total Profit'] for m in methods]
    colors = ['gray', 'lightgray', 'silver', 'darkgray', 'blue', 'green']
    
    bars = ax1.bar(range(len(methods)), profits, color=colors)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Total Profit ($)')
    ax1.set_title('Total Profit Comparison')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, profit) in enumerate(zip(bars, profits)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{profit:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)
    
    # Subplot 2: ROI Comparison
    ax2 = axes[0, 1]
    rois = [all_results[m]['ROI (%)'] for m in methods]
    bars = ax2.bar(range(len(methods)), rois, color=colors)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('ROI (%)')
    ax2.set_title('Return on Investment (ROI)')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, roi) in enumerate(zip(bars, rois)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{roi:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)
    
    # Subplot 3: Sharpe Ratio
    ax3 = axes[1, 0]
    sharpes = [all_results[m].get('Sharpe Ratio', 0) for m in methods]
    bars = ax3.bar(range(len(methods)), sharpes, color=colors)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Performance (Sharpe Ratio)')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (bar, sharpe) in enumerate(zip(bars, sharpes)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{sharpe:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)
    
    # Subplot 4: DQN vs A2C Test Rewards Distribution
    ax4 = axes[1, 1]
    ax4.hist([dqn_test_rewards, a2c_test_rewards], 
             label=['DQN', 'A2C'], 
             bins=10, 
             alpha=0.7, 
             color=['blue', 'green'])
    ax4.set_xlabel('Test Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('DQN vs A2C: Test Reward Distribution')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to 'comparison_plots.png'")
    plt.show()
    
    # 7. STATISTICAL SIGNIFICANCE TEST
    print("\n" + "="*70)
    print("7. STATISTICAL ANALYSIS")
    print("="*70)
    
    from scipy import stats
    
    # T-test: DQN vs A2C
    if len(dqn_test_rewards) > 1 and len(a2c_test_rewards) > 1:
        t_stat, p_value = stats.ttest_ind(dqn_test_rewards, a2c_test_rewards)
        print(f"\nT-Test (DQN vs A2C):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")
        
        if p_value < 0.05:
            winner = "DQN" if dqn_mean > a2c_mean else "A2C"
            print(f"  → {winner} is statistically significantly better (p < 0.05)")
        else:
            print(f"  → No statistically significant difference (p >= 0.05)")
    else:
        print("\nInsufficient test episodes for statistical testing (need > 1 episode)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((dqn_std**2 + a2c_std**2) / 2)
    cohens_d = (dqn_mean - a2c_mean) / (pooled_std + 1e-8)
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        print("  → Small effect size")
    elif abs(cohens_d) < 0.5:
        print("  → Medium effect size")
    else:
        print("  → Large effect size")
    
    # 8. SUMMARY & RECOMMENDATIONS
    print("\n" + "="*70)
    print("8. SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    # Find best performer
    best_method = max(all_results.items(), key=lambda x: x[1]['Total Profit'])
    worst_method = min(all_results.items(), key=lambda x: x[1]['Total Profit'])
    
    print(f"\n📊 Best Performer: {best_method[0]}")
    print(f"   Profit: ${best_method[1]['Total Profit']:.2f}")
    print(f"   ROI: {best_method[1]['ROI (%)']:.2f}%")
    
    print(f"\n📉 Worst Performer: {worst_method[0]}")
    print(f"   Profit: ${worst_method[1]['Total Profit']:.2f}")
    print(f"   ROI: {worst_method[1]['ROI (%)']:.2f}%")
    
    # Compare RL methods to baselines
    print(f"\n🤖 Deep RL Performance:")
    print(f"   DQN:  ${dqn_mean:.2f} profit")
    print(f"   A2C:  ${a2c_mean:.2f} profit")
    
    best_baseline = max(baseline_results.items(), key=lambda x: x[1]['Total Profit'])
    print(f"\n📈 Best Baseline: {best_baseline[0]} (${best_baseline[1]['Total Profit']:.2f})")
    
    if dqn_mean > best_baseline[1]['Total Profit'] or a2c_mean > best_baseline[1]['Total Profit']:
        print("\n✅ Deep RL methods outperform traditional baselines!")
    else:
        print("\n⚠️  Baselines outperform Deep RL - possible overfitting or insufficient training")
    
    # Data size warning
    if len(env_test.X) < 50:
        print(f"\n⚠️  WARNING: Small test set ({len(env_test.X)} matches)")
        print("   Results may not be statistically reliable")
        print("   Recommendation: Download more seasons of data")
    
    print("\n" + "="*70)
    print(" COMPARISON COMPLETE")
    print("="*70)
    
    print("\n📁 Output files:")
    print("   - results_comparison.csv")
    print("   - comparison_plots.png")