import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress sklearn deprecation
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class BettingBaselines:
    def __init__(self, data_dir='data/epl/'):
        self.data_dir = data_dir
        self.initial_bankroll = 1000.0
        self.stake = 1.0  # Fixed stake per bet
        self.odds_cols = ['B365H', 'B365D', 'B365A']  # Proxy odds
        self.train_df = None  # Cache
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.odds_test = None

    def load_data(self, split='test', cache=True):
        """Load preprocessed split; prepare features/odds/outcomes. Cache for efficiency."""
        if split == 'train' and cache and self.train_df is not None:
            return self.train_df
        if split == 'test' and cache and self.test_df is not None:
            return self.test_df
        
        df = pd.read_csv(f"{self.data_dir}{split}.csv")
        # State features: probs + form (expandable)
        feature_cols = ['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A', 
                        'Home_form_rolling', 'Away_form_rolling', 
                        'Home_win_rate_rolling', 'Away_win_rate_rolling']
        X = df[feature_cols].fillna(0).values
        y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values  # Numeric labels: 0=H,1=D,2=A
        odds = df[self.odds_cols].fillna(2.0).values  # Shape: (n_matches, 3)
        
        if split == 'train':
            self.X_train = X
            self.y_train = y
            self.train_df = df
        else:
            self.X_test = X
            self.y_test = y
            self.odds_test = odds
            self.test_df = df
        
        return df

    def simulate_bets(self, actions):
        """actions: list [0=skip,1=H,2=D,3=A]. Returns profits array."""
        self.load_data('test')  # Ensure cached
        profits = np.zeros(len(actions))
        for i, action in enumerate(actions):
            if action == 0:  # Skip
                continue
            outcome_idx = self.y_test[i]  # 0=H,1=D,2=A
            if action - 1 == outcome_idx:  # Win
                odd = self.odds_test[i, action - 1]
                profits[i] = self.stake * (odd - 1)
            else:  # Loss
                profits[i] = -self.stake
        self.bankroll = self.initial_bankroll + np.sum(profits)
        return profits

    def compute_metrics(self, profits):
        """ROI, Profit, Sharpe, Hit Rate."""
        total_profit = np.sum(profits)
        bets = np.sum(profits != 0)
        turnover = bets * self.stake
        roi = (total_profit / turnover * 100) if turnover > 0 else 0
        returns = profits / self.initial_bankroll  # Simplified returns
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)  # Avoid div0
        hit_rate = np.sum(profits > 0) / bets if bets > 0 else 0
        drawdown = np.min(np.cumsum(profits)) / self.initial_bankroll * 100 if np.any(profits) else 0
        return {
            'Total Profit': total_profit,
            'ROI (%)': roi,
            'Sharpe Ratio': sharpe,
            'Hit Rate (%)': hit_rate * 100,
            'Max Drawdown (%)': drawdown
        }

    # Baseline 1: Always bet on favorite (min odds)
    def always_favorite(self):
        self.load_data('test')
        actions = []
        for i in range(len(self.test_df)):
            min_idx = np.argmin(self.odds_test[i])
            actions.append(min_idx + 1)  # 1=H,2=D,3=A
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)

    # Baseline 2: No-bet (control)
    def no_bet(self):
        profits = np.zeros(len(self.load_data('test')))
        return self.compute_metrics(profits)

    # Baseline 3: Supervised (LR probs > threshold)
    def supervised_threshold(self, threshold=0.6):
        self.load_data('train')
        self.load_data('test')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        model = LogisticRegression(multi_class='auto', max_iter=500)  # Fixed: 'auto' for future-proof
        model.fit(X_train_scaled, self.y_train)
        
        X_test_scaled = scaler.transform(self.X_test)
        probs = model.predict_proba(X_test_scaled)
        actions = []
        for prob_row in probs:
            max_prob = np.max(prob_row)
            if max_prob > threshold:
                action = np.argmax(prob_row) + 1  # 1=H,2=D,3=A
            else:
                action = 0  # Skip
            actions.append(action)
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)
    
    # Model-Based Baseline: Dyna-Q (tabular Q + learned transitions)
    def dyna_q_baseline(self, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.load_data('train')
        n_states = len(self.train_df)  # Simplified: state = match index
        n_actions = 4  # 0=skip,1=H,2=D,3=A
        Q = np.zeros((n_states, n_actions))
        model = {}  # (s,a) -> list of (r, s') samples
        
        for _ in range(episodes):
            for s in range(n_states):
                # Epsilon-greedy
                if random.random() < epsilon:
                    a = random.randint(0, n_actions - 1)
                else:
                    a = np.argmax(Q[s])
                
                # Interact with env (train data)
                r = 0
                if a > 0:
                    outcome_idx = self.y_train[s]
                    if a - 1 == outcome_idx:
                        r = self.stake * (self.odds_test[s % len(self.odds_test), a - 1] - 1)  # Approx odds
                    else:
                        r = -self.stake
                s_next = min(s + 1, n_states - 1)
                
                # Q-update
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
                
                # Update model: Collect samples for transitions
                key = (s, a)
                if key not in model:
                    model[key] = []
                model[key].append((r, s_next))
                
                # Planning: n steps from model
                for _ in range(planning_steps):
                    if len(model) > 0:
                        sample_key = random.choice(list(model.keys()))
                        sample_a = sample_key[1]
                        sample_r, sample_s_next = random.choice(model[sample_key])
                        sample_s_next_next = min(sample_s_next + 1, n_states - 1)
                        Q[sample_key[0], sample_a] += alpha * (sample_r + gamma * np.max(Q[sample_s_next]) - Q[sample_key[0], sample_a])
        
        # Test on test set (approx states by modulo)
        self.load_data('test')
        actions = [np.argmax(Q[min(s % n_states, n_states - 1)]) for s in range(len(self.test_df))]
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)

    # DRL Stub: DQN (integrate with env later)
    def dqn_stub(self, state_size=7, action_size=4, episodes=100):
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_size)
                )
            def forward(self, x):
                return self.net(x)
        
        model = DQN(state_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        replay = deque(maxlen=10000)
        loss = torch.tensor(0.0)  # Initialize to avoid UnboundLocalError
        
        # Dummy training loop (replace with env steps)
        for ep in range(episodes):
            state = torch.rand(1, state_size)  # Dummy state
            action = torch.argmax(model(state)).item()
            reward = random.uniform(-1, 2)  # Dummy
            next_state = torch.rand(1, state_size)
            replay.append((state, action, reward, next_state, False))
            
            if len(replay) > 32:
                batch = random.sample(replay, 32)
                # Q-learning update (simplified)
                states = torch.cat([b[0] for b in batch])
                actions_b = torch.tensor([b[1] for b in batch]).unsqueeze(1)
                qs = model(states).gather(1, actions_b)
                targets = torch.tensor([b[2] for b in batch])
                loss = nn.MSELoss()(qs.squeeze(), targets.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"DQN Ep {ep}: Loss {loss.item():.3f}")
            else:
                print(f"DQN Ep {ep}: Buffer filling... (size {len(replay)})")
        return model

    # DRL Stub: A2C (placeholder)
    def a2c_stub(self, episodes=100):
        print("A2C Stub: Implement Actor-Critic (policy/value nets, advantage updates).")
        print("Use for continuous stakes later (e.g., via PPO).")
        return None

if __name__ == "__main__":
    baselines = BettingBaselines()
    
    print("=== Baseline Results (Test Set) ===")
    fav = baselines.always_favorite()
    nobet = baselines.no_bet()
    sup = baselines.supervised_threshold(threshold=0.6)
    dyna = baselines.dyna_q_baseline(episodes=200)
    
    results_df = pd.DataFrame({
        'Always Favorite': pd.Series(fav),
        'No Bet': pd.Series(nobet),
        'Supervised (Thresh=0.6)': pd.Series(sup),
        'Dyna-Q (Model-Based)': pd.Series(dyna)
    }).T
    print(results_df.round(2))
    
    # DRL Stubs (run for demo)
    print("\n=== DRL Stubs (Dummy Training) ===")
    dqn_model = baselines.dqn_stub()
    a2c_model = baselines.a2c_stub()