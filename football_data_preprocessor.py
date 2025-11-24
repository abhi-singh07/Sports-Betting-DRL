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
        Includes steps: sort by date, drop incompletes, rolling features,
        and implied probabilities. Season splitting is removed and done externally.
        """
        print("\nPreprocessing data...")
        if df is None or df.empty:
            return None

        # Step 1: Sort by date and handle datetime
        # Preserve original 'Season' column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Step 2: Drop missing or incomplete rows (essential columns)
        essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'Season']
        df = df.dropna(subset=essential_cols)
        
        # Map FTR to numeric for easier computation (H=1 win for home, D=0, A=-1 loss for home)
        df['FTR_numeric'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1}).infer_objects()

        # Step 3: Compute rolling team features (last 5 matches)
        print("Computing rolling team features...")
        team_histories = defaultdict(lambda: {'results': deque(maxlen=5), 'goals_scored': deque(maxlen=5), 'goals_conceded': deque(maxlen=5)})
        
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # --- Assignment (use history BEFORE this match) ---
            home_hist = team_histories[home_team]
            df.loc[idx, 'Home_form_rolling'] = np.mean(home_hist['results']) if home_hist['results'] else 0.0
            df.loc[idx, 'Home_win_rate_rolling'] = np.mean([r == 1 for r in home_hist['results']]) if home_hist['results'] else 0.0
            
            away_hist = team_histories[away_team]
            df.loc[idx, 'Away_form_rolling'] = np.mean(away_hist['results']) if away_hist['results'] else 0.0
            df.loc[idx, 'Away_win_rate_rolling'] = np.mean([r == 1 for r in away_hist['results']]) if away_hist['results'] else 0.0
            
            # --- Update histories WITH this match for future rows ---
            home_hist['results'].append(row['FTR_numeric'])
            home_hist['goals_scored'].append(row['FTHG'])
            home_hist['goals_conceded'].append(row['FTAG'])
            
            away_hist['results'].append(-row['FTR_numeric'])
            away_hist['goals_scored'].append(row['FTAG'])
            away_hist['goals_conceded'].append(row['FTHG'])
        
        # Drop temporary cols
        df = df.drop(columns=['FTR_numeric'], errors='ignore')
        df[['Home_form_rolling', 'Away_form_rolling', 'Home_win_rate_rolling', 'Away_win_rate_rolling']] = \
            df[['Home_form_rolling', 'Away_form_rolling', 'Home_win_rate_rolling', 'Away_win_rate_rolling']].fillna(0)
        
        print("Rolling features computed (full chronological).")

        # Step 4: Convert bookmaker odds to implied probabilities
        print("Computing implied probabilities...")
        odds_bookmakers = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
        outcomes = ['H', 'D', 'A']
        
        implied_probs_dict = {}
        
        for outcome in outcomes:
            odds_cols = [f"{bm}{outcome}" for bm in odds_bookmakers if f"{bm}{outcome}" in df.columns]
            if odds_cols:
                # Use the mean of available odds for a robust estimate
                df[f'Avg_Odds_{outcome}'] = df[odds_cols].mean(axis=1)
                implied_probs_dict[f'raw_{outcome}'] = 1 / df[f'Avg_Odds_{outcome}']
            else:
                implied_probs_dict[f'raw_{outcome}'] = 0.33  # Uniform fallback

        implied_probs = pd.DataFrame(implied_probs_dict, index=df.index)
        
        # Row-wise normalization (to remove bookmaker margin)
        raw_sum = implied_probs[[f'raw_{o}' for o in outcomes]].sum(axis=1)
        for outcome in outcomes:
            df[f'Implied_Prob_{outcome}'] = implied_probs[f'raw_{outcome}'] / raw_sum

        print(f"Matches after cleaning: {len(df)}")
        return df

    def download_season_data(self, season, division='E0'):
        """
        Downloads football data for a given season and division.
        """
        season_code = f"{str(season)[-2:]}{str(season + 1)[-2:]}"
        filename = os.path.join(self.save_dir, f"{division}_{season_code}.csv")
        
        if os.path.exists(filename):
            # Load local file
            print(f"Local file found for season {season}-{season+1}. Loading from {filename}...")
            df = pd.read_csv(filename)
            # Ensure 'Season' column is correct (important for splitting)
            if 'Season' not in df.columns or df['Season'].isna().all():
                df['Season'] = f"{season}-{season + 1}"
            return self.preprocess_data(df)
        else:
            # Download if not found
            url = f"{self.base_url}/{season_code}/{division}.csv"
            print(f"Downloading season {season}-{season+1} data from {url}...")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text), encoding='unicode_escape')
                df['Season'] = f"{season}-{season + 1}"
                
                # Preprocess immediately after download
                df = self.preprocess_data(df)
                
                # Save the unprocessed data locally for caching
                df.to_csv(filename, index=False)
                print(f"Data for season {season}-{season+1} downloaded, preprocessed, and saved to {filename}.")
                return df
            except requests.exceptions.RequestException as e:
                print(f"Failed to download data for season {season}-{season+1}: {e}")
                return None
    
    def load_and_preprocess_all_by_season(self, train_seasons, val_seasons, test_seasons, division='E0'):
        """
        Load data for specified seasons and create train/val/test splits.
        """
        dfs = []
        # 1. Download/Load all necessary seasons
        all_seasons = sorted(list(set(train_seasons + val_seasons + test_seasons)))
        for season in all_seasons:
            df = self.download_season_data(season, division)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            print("No data loaded. Exiting.")
            return None, None, None

        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 2. Split by season label (E.g., 2020 -> "2020-2021")
        train_df = combined_df[combined_df['Season'].isin([f"{s}-{s+1}" for s in train_seasons])].copy()
        val_df = combined_df[combined_df['Season'].isin([f"{s}-{s+1}" for s in val_seasons])].copy()
        test_df = combined_df[combined_df['Season'].isin([f"{s}-{s+1}" for s in test_seasons])].copy()
        
        # 3. Save splits
        train_df.to_csv(os.path.join(self.save_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.save_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.save_dir, 'test.csv'), index=False)
        
        print(f"\n--- Data Split Summary ---")
        print(f"Train: {len(train_df)} matches (Seasons: {train_seasons})")
        print(f"Val:   {len(val_df)} matches (Seasons: {val_seasons})")
        print(f"Test:  {len(test_df)} matches (Seasons: {test_seasons})")
        print(f"--------------------------")
        
        return train_df, val_df, test_df
    
    def get_available_columns(self, df):
        # ... (rest of the method unchanged)
        if df is None:
            print("No DataFrame provided.")
            return
        print("\n=== Available Columns ===")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in sorted(df.columns):
            print(f" - {col}")
        odds_cols = [col for col in df.columns if any(col.startswith(bm) for bm in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC'])]
        if odds_cols:
            print(f"\nBookmaker odds columns ({len(odds_cols)}):")
            for col in odds_cols[:10]:
                print(f" - {col}")
            if len(odds_cols) > 10:
                print(f" ... and {len(odds_cols) - 10} more")

if __name__ == "__main__":
    importer = FootballDataImporter(save_dir='data/epl/')
    
    # 🚨 New run command for season-based splitting
    print("Running multi-season data pipeline...")
    train_seasons = [2017, 2018, 2019, 2020]
    val_seasons = [2021]
    test_seasons = [2022]
    
    train_df, val_df, test_df = importer.load_and_preprocess_all_by_season(
        train_seasons=train_seasons, 
        val_seasons=val_seasons, 
        test_seasons=test_seasons, 
        division='E0'
    )
    
    if train_df is not None:
        print("\n=== Sample Train Data ===")
        sample_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTR', 
                       'Home_form_rolling', 'Implied_Prob_H', 'Implied_Prob_D', 'Implied_Prob_A']
        print(train_df[sample_cols].head())
        importer.get_available_columns(train_df)