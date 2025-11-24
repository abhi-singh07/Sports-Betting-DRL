import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# --- Helper Class for Tabular RL (Discretization) ---
class TabularRL:
    def __init__(self, X, n_bins=5):
        # X: feature matrix (excluding bankroll)
        self.X = X
        self.n_bins = n_bins
        
        # Calculate min/max and bins based on the training data X
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        
        # Define the bin boundaries for each feature dimension
        self.bins = [np.linspace(self.min_vals[i], self.max_vals[i], self.n_bins + 1)[1:-1] for i in range(X.shape[1])]
    
    def discretize_state(self, state_cont):
        """Converts a continuous state vector (features only) to a single discrete integer index."""
        if state_cont is None or len(state_cont) == 0:
            return 0
            
        discretized = []
        for i, val in enumerate(state_cont):
            # np.digitize returns the index of the bin the value falls into
            bin_index = np.digitize(val, self.bins[i])
            discretized.append(bin_index)
            
        # Combine the bin indices into a unique integer state index
        state_index = 0
        multiplier = 1
        # The number of possible indices (including the index before the first bin) is n_bins + 1
        base = self.n_bins + 1 
        
        for index in discretized:
            state_index += index * multiplier
            multiplier *= base
            
        return state_index
        
    def get_state_space_size(self):
        """Returns the maximum possible number of discrete states."""
        return (self.n_bins + 1) ** self.X.shape[1]
# --- End TabularRL Helper Class ---


class BettingBaselines:
    def __init__(self, data_dir='data/epl/', stake=1.0):
        self.data_dir = data_dir
        self.initial_bankroll = 1000.0
        self.stake = stake  # Fixed stake per bet
        self.odds_cols = ['B365H', 'B365D', 'B365A']
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.odds_test = None
        
        # Define feature columns used for all algorithms (used for Tabular RL discretization)
        self.feature_cols = ['Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A', 
                            'Home_form_rolling', 'Away_form_rolling', 
                            'Home_win_rate_rolling', 'Away_win_rate_rolling']
        

    def load_data(self, split='test', cache=True):
        """Load preprocessed split; prepare features/odds/outcomes. Cache for efficiency."""
        if split == 'train' and cache and self.train_df is not None:
            return self.train_df
        if split == 'test' and cache and self.test_df is not None:
            return self.test_df
        
        df = pd.read_csv(f"{self.data_dir}{split}.csv")
        # Features (X) now only include the list in self.feature_cols (no bankroll or other context features)
        X = df[self.feature_cols].fillna(0).values
        y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values  # Numeric labels: 0=H,1=D,2=A
        odds = df[self.odds_cols].fillna(2.0).values
        
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
        self.load_data('test')
        profits = np.zeros(len(actions))
        
        # NOTE: This uses the test data outcomes (self.y_test) and odds (self.odds_test)
        for i, action in enumerate(actions):
            if action == 0:
                continue
            outcome_idx = self.y_test[i]
            if action - 1 == outcome_idx:
                odd = self.odds_test[i, action - 1]
                profits[i] = self.stake * (odd - 1)
            else:
                profits[i] = -self.stake
                
        self.bankroll = self.initial_bankroll + np.sum(profits)
        return profits

    def compute_metrics(self, profits):
        """ROI, Profit, Sharpe, Hit Rate, and Max Drawdown."""
        total_profit = np.sum(profits)
        bets = np.sum(profits != 0)
        turnover = bets * self.stake
        roi = (total_profit / turnover * 100) if turnover > 0 else 0
        returns = profits / self.initial_bankroll
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        hit_rate = np.sum(profits > 0) / bets if bets > 0 else 0
        
        # Max Drawdown Calculation
        cumulative_profits = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative_profits)
        drawdown_abs = np.min(cumulative_profits - peak) if np.any(profits) else 0
        drawdown_pct = (drawdown_abs / self.initial_bankroll) * 100
        
        return {
            'Total Profit': total_profit,
            'ROI (%)': roi,
            'Sharpe Ratio': sharpe,
            'Hit Rate (%)': hit_rate * 100,
            'Max Drawdown (%)': drawdown_pct
        }

    # Baseline 1: Always bet on favorite (min odds)
    def always_favorite(self):
        self.load_data('test')
        actions = []
        for i in range(len(self.test_df)):
            min_idx = np.argmin(self.odds_test[i])
            actions.append(min_idx + 1)
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
        model = LogisticRegression(multi_class='auto', max_iter=500)
        model.fit(X_train_scaled, self.y_train)
        
        X_test_scaled = scaler.transform(self.X_test)
        probs = model.predict_proba(X_test_scaled)
        
        actions = []
        for prob_row in probs:
            max_prob = np.max(prob_row)
            if max_prob > threshold:
                action = np.argmax(prob_row) + 1
            else:
                action = 0
            actions.append(action)
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)
    
    # Baseline 4: Dyna-Q (Model-Based Tabular RL with Discretization)
    def dyna_q_baseline(self, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.load_data('train')
        
        # Use TabularRL helper to discretize the feature space (5 bins for each of the 7 features)
        tabular_helper = TabularRL(self.X_train, n_bins=5)
        n_states = tabular_helper.get_state_space_size()
        n_actions = 4
        
        Q = np.zeros((n_states, n_actions))
        model = {}
        
        for _ in range(episodes):
            for i in range(len(self.X_train)):
                s = tabular_helper.discretize_state(self.X_train[i])
                
                if random.random() < epsilon:
                    a = random.randint(0, n_actions - 1)
                else:
                    s_safe = min(s, Q.shape[0] - 1)
                    a = np.argmax(Q[s_safe])
                
                # Interact with env (train data)
                r = 0
                if a > 0:
                    outcome_idx = self.y_train[i]
                    if a - 1 == outcome_idx:
                        odd = self.odds_test[i % len(self.odds_test), a - 1] 
                        r = self.stake * (odd - 1)
                    else:
                        r = -self.stake
                
                # Next state (index i+1 in the training set)
                s_next_index = min(i + 1, len(self.X_train) - 1)
                s_next = tabular_helper.discretize_state(self.X_train[s_next_index])
                
                # Q-update
                s_safe, s_next_safe = min(s, Q.shape[0] - 1), min(s_next, Q.shape[0] - 1)
                Q[s_safe, a] += alpha * (r + gamma * np.max(Q[s_next_safe]) - Q[s_safe, a])
                
                # Update model
                key = (s_safe, a)
                if key not in model:
                    model[key] = []
                # Store sample only once per transition to avoid memory issues
                if (r, s_next_safe) not in model[key]: 
                    model[key].append((r, s_next_safe))
                
                # Planning: n steps from model
                for _ in range(planning_steps):
                    if len(model) > 0:
                        sample_key = random.choice(list(model.keys()))
                        sample_s, sample_a = sample_key
                        sample_r, sample_s_next = random.choice(model[sample_key])
                        
                        Q[sample_s, sample_a] += alpha * (sample_r + gamma * np.max(Q[sample_s_next]) - Q[sample_s, sample_a])
        
        # Test Phase (using test data and discretization from train)
        self.load_data('test')
        actions = []
        for i in range(len(self.X_test)):
            s = tabular_helper.discretize_state(self.X_test[i])
            s_safe = min(s, Q.shape[0] - 1)
            actions.append(np.argmax(Q[s_safe]))
            
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)
    
    # Baseline 5: SARSA (On-Policy Tabular RL with Discretization)
    def sarsa_baseline(self, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.load_data('train')
        
        tabular_helper = TabularRL(self.X_train, n_bins=5)
        n_states = tabular_helper.get_state_space_size()
        n_actions = 4
        
        Q = np.zeros((n_states, n_actions))
        
        def epsilon_greedy(s):
            s_safe = min(s, Q.shape[0] - 1)
            if random.random() < epsilon:
                return random.randint(0, n_actions - 1)
            return np.argmax(Q[s_safe])

        # Training Phase
        for _ in range(episodes):
            
            # Start state and action
            s = tabular_helper.discretize_state(self.X_train[0])
            a = epsilon_greedy(s)
            s_safe = min(s, Q.shape[0] - 1)
            
            for i in range(len(self.X_train) - 1):
                
                # 1. Take action 'a' from state 's' and get reward 'r'
                r = 0
                if a > 0:
                    outcome_idx = self.y_train[i]
                    if a - 1 == outcome_idx:
                        odd = self.odds_test[i % len(self.odds_test), a - 1]
                        r = self.stake * (odd - 1)
                    else:
                        r = -self.stake
                
                # 2. Observe next state s'
                s_next = tabular_helper.discretize_state(self.X_train[i + 1])
                s_next_safe = min(s_next, Q.shape[0] - 1)
                
                # 3. Choose next action a' using the same policy (epsilon-greedy)
                a_next = epsilon_greedy(s_next)
                
                # 4. SARSA Update: Q(s, a) += alpha * (R + gamma * Q(s', a') - Q(s, a))
                Q[s_safe, a] += alpha * (r + gamma * Q[s_next_safe, a_next] - Q[s_safe, a])
                
                # 5. Move to next step
                s, a = s_next, a_next
                s_safe = s_next_safe
                
        # Test Phase
        self.load_data('test')
        actions = []
        for i in range(len(self.X_test)):
            s = tabular_helper.discretize_state(self.X_test[i])
            s_safe = min(s, Q.shape[0] - 1) 
            actions.append(np.argmax(Q[s_safe]))
            
        profits = self.simulate_bets(actions)
        return self.compute_metrics(profits)


    # DRL Stub (Unchanged, removed verbose printout)
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
        
        # Dummy training loop
        for ep in range(episodes):
            state = torch.rand(1, state_size)
            action = torch.argmax(model(state)).item()
            reward = random.uniform(-1, 2)
            next_state = torch.rand(1, state_size)
            replay.append((state, action, reward, next_state, False))
            
            if len(replay) > 32:
                batch = random.sample(replay, 32)
                states = torch.cat([b[0] for b in batch])
                actions_b = torch.tensor([b[1] for b in batch]).unsqueeze(1)
                qs = model(states).gather(1, actions_b)
                targets = torch.tensor([b[2] for b in batch])
                loss = nn.MSELoss()(qs.squeeze(), targets.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model