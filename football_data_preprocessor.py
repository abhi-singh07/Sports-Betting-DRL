import pandas as pd
import os
import requests
from io import StringIO
import numpy as np
from collections import defaultdict, deque

class FootballDataImporter:
    def __init__(self, save_dir='data/'):
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def preprocess_data(self, df):
        """
        Preprocesses the football data DataFrame.
        Includes proposal steps: sort by date, drop incompletes, rolling features,
        implied probabilities, and time-based splits.
        """
        print("\nPreprocessing data...")
        if df is None or df.empty:
            return None

        # Step 1: Sort by date and handle datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Step 2: Drop missing or incomplete rows (essential columns)
        essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'Season']
        df = df.dropna(subset=essential_cols)
        
        # Map FTR to numeric for easier computation (H=1 win for home, D=0, A=-1 loss for home)
        df['FTR_numeric'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

        # Step 3: Compute rolling team features (last 5 matches: form, goals scored/conceded, win rate)
        # Full implementation: Chronological iteration, track per-team history (deque for last 5)
        print("Computing rolling team features...")
        team_histories = defaultdict(lambda: {'results': deque(maxlen=5), 'goals_scored': deque(maxlen=5), 'goals_conceded': deque(maxlen=5)})
        
        # For each team, store home/away perspective separately? No—use unified history, but adjust for opponent view
        # Here: Track overall last 5 for each team (as form is team-specific)
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Update home team history (their result: FTR_numeric, scored FTHG, conceded FTAG)
            home_hist = team_histories[home_team]
            home_hist['results'].append(row['FTR_numeric'])
            home_hist['goals_scored'].append(row['FTHG'])
            home_hist['goals_conceded'].append(row['FTAG'])
            
            # Update away team history (their result: -FTR_numeric (win if A), scored FTAG, conceded FTHG)
            away_hist = team_histories[away_team]
            away_hist['results'].append(-row['FTR_numeric'])  # Invert for away perspective
            away_hist['goals_scored'].append(row['FTAG'])
            away_hist['goals_conceded'].append(row['FTHG'])
        
        # Now backfill rolling stats (since we can't update future, but df is sorted, we can assign post-loop? Wait, need 2-pass)
        # Better: 2-pass - first compute all histories, then assign to df
        # Wait, above loop updates after match, but for rolling UP TO match, need to assign before update? No:
        # For rolling at match time, compute from prior history.
        # Reset and re-loop with assignment before update:
        team_histories = defaultdict(lambda: {'results': deque(maxlen=5), 'goals_scored': deque(maxlen=5), 'goals_conceded': deque(maxlen=5)})
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Assign current rolling for home (from prior matches)
            home_hist = team_histories[home_team]
            df.at[idx, 'Home_form_rolling'] = np.mean(home_hist['results']) if home_hist['results'] else 0.0
            df.at[idx, 'Home_goals_scored_rolling'] = np.mean(home_hist['goals_scored']) if home_hist['goals_scored'] else 0.0
            df.at[idx, 'Home_goals_conceded_rolling'] = np.mean(home_hist['goals_conceded']) if home_hist['goals_conceded'] else 0.0
            df.at[idx, 'Home_win_rate_rolling'] = np.mean([r == 1 for r in home_hist['results']]) if home_hist['results'] else 0.0
            
            # Assign for away
            away_hist = team_histories[away_team]
            df.at[idx, 'Away_form_rolling'] = np.mean(away_hist['results']) if away_hist['results'] else 0.0
            df.at[idx, 'Away_goals_scored_rolling'] = np.mean(away_hist['goals_scored']) if away_hist['goals_scored'] else 0.0
            df.at[idx, 'Away_goals_conceded_rolling'] = np.mean(away_hist['goals_conceded']) if away_hist['goals_conceded'] else 0.0
            df.at[idx, 'Away_win_rate_rolling'] = np.mean([r == 1 for r in away_hist['results']]) if away_hist['results'] else 0.0
            
            # Now update histories WITH this match for future rows
            home_hist['results'].append(row['FTR_numeric'])
            home_hist['goals_scored'].append(row['FTHG'])
            home_hist['goals_conceded'].append(row['FTAG'])
            
            away_hist['results'].append(-row['FTR_numeric'])
            away_hist['goals_scored'].append(row['FTAG'])
            away_hist['goals_conceded'].append(row['FTHG'])
        
        print("Rolling features computed (full chronological).")

        # Step 4: Convert bookmaker odds to implied probabilities (FIXED: row-wise normalization across H/D/A)
        print("Computing implied probabilities...")
        odds_bookmakers = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
        outcomes = ['H', 'D', 'A']
        implied_probs = pd.DataFrame(index=df.index)
        
        for outcome in outcomes:
            odds_cols = [f"{bm}{outcome}" for bm in odds_bookmakers if f"{bm}{outcome}" in df.columns]
            if odds_cols:
                avg_odds = df[odds_cols].mean(axis=1)
                implied_probs[f'raw_{outcome}'] = 1 / avg_odds
            else:
                implied_probs[f'raw_{outcome}'] = 1  # Uniform fallback odds=1 (prob=1, but will normalize)
        
        # Row-wise normalization: sum raw probs across outcomes, then divide
        raw_sum = implied_probs[[f'raw_{o}' for o in outcomes]].sum(axis=1)
        for outcome in outcomes:
            df[f'Implied_Prob_{outcome}'] = implied_probs[f'raw_{outcome}'] / raw_sum
        
        # Verify: Add check
        check_sum = df[[f'Implied_Prob_{o}' for o in outcomes]].sum(axis=1)
        print(f"Implied probs normalized (mean row sum: {check_sum.mean():.3f})")

        # Step 5: Time-based splits (70% train, 15% val, 15% test by date quantiles)
        q70 = df['Date'].quantile(0.7)
        q85 = df['Date'].quantile(0.85)
        train = df[df['Date'] <= q70].copy()
        val = df[(df['Date'] > q70) & (df['Date'] <= q85)].copy()
        test = df[df['Date'] > q85].copy()
        
        print(f"Train: {len(train)} matches, Val: {len(val)}, Test: {len(test)}")
        
        # Save splits (and full preprocessed)
        train.to_csv(os.path.join(self.save_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(self.save_dir, 'val.csv'), index=False)
        test.to_csv(os.path.join(self.save_dir, 'test.csv'), index=False)
        df.to_csv(os.path.join(self.save_dir, 'full_preprocessed.csv'), index=False)

        print(f"Matches after cleaning: {len(df)}")
        return df

    def download_season_data(self, season, division='E0'):
        """
        Downloads football data for a given season and division.
        FIRST checks if local file exists; if yes, loads it. Otherwise downloads.
        """
        season_code = f"{str(season)[-2:]}{str(season + 1)[-2:]}"
        filename = os.path.join(self.save_dir, f"{division}_{season_code}.csv")
        
        if os.path.exists(filename):
            print(f"Local file found for season {season}-{season+1}. Loading from {filename}...")
            df = pd.read_csv(filename)
            if 'Season' not in df.columns:
                df['Season'] = f"{season}-{season + 1}"
            df = self.preprocess_data(df)
            print(f"Data for season {season}-{season+1} loaded and preprocessed from local file.")
            return df
        else:
            # Proceed with download if not found
            url = f"{self.base_url}/{season_code}/{division}.csv"
            print(f"Downloading season {season}-{season+1} data from {url}...")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                df['Season'] = f"{season}-{season + 1}"
                df = self.preprocess_data(df)
                df.to_csv(filename, index=False)
                print(f"Data for season {season}-{season+1} downloaded and saved to {filename}.")
                return df
            except requests.exceptions.RequestException as e:
                print(f"Failed to download data for season {season}-{season+1}: {e}")
                return None

    def load_and_preprocess_all(self, seasons=[2022], division='E0'):
        """
        Load multiple seasons and concatenate, then preprocess once.
        Checks each season's file: loads if exists, else downloads.
        """
        dfs = []
        for season in seasons:
            df = self.download_season_data(season, division)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            return self.preprocess_data(combined_df)
        return None

    def get_available_columns(self, df):
        """
        Print information about available columns.
        """
        if df is None:
            print("No DataFrame provided.")
            return
        print("\n=== Available Columns ===")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in sorted(df.columns):  # Sorted for readability
            print(f" - {col}")
        # Show sample of odds columns
        odds_cols = [col for col in df.columns if any(col.startswith(bm) for bm in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC'])]
        if odds_cols:
            print(f"\nBookmaker odds columns ({len(odds_cols)}):")
            for col in odds_cols[:10]:  # Show first 10
                print(f" - {col}")
            if len(odds_cols) > 10:
                print(f" ... and {len(odds_cols) - 10} more")

if __name__ == "__main__":
    importer = FootballDataImporter(save_dir='data/epl/')
    # Download/load and preprocess 2022-23 season (checks local first)
    df = importer.download_season_data(season=2022, division='E0')
    if df is not None:
        print("=== Sample Data ===")
        sample_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                       'Home_form_rolling', 'Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']
        print(df[sample_cols].head())
        print("=== Basic Statistics ===")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Number of teams: {df['HomeTeam'].nunique()}")
        print(f"Total matches: {len(df)}")
        print(f"\nResult distribution:")
        print(df['FTR'].value_counts())
        importer.get_available_columns(df)