import pandas as pd
import os
import requests
from io import StringIO

class FootballDataImporter:
    def __init__(self, save_dir='data/'):
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def preprocess_data(self, df):
        """
        Preprocesses the football data DataFrame.
        """ 
        print("\nPreprocessing data...")

        if df is None:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date').reset_index(drop=True)
        essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
        df = df.dropna(subset=essential_cols)
        
        odds_cols = []
        for bookmaker in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']:
            for outcome in ['H', 'D', 'A']:
                col = f"{bookmaker}{outcome}"
                if col in df.columns:
                    odds_cols.append(col)
        
        print(f"Found {len(odds_cols)} bookmaker odds columns")
        print(f"Matches after cleaning: {len(df)}")
        
        return df


    def download_season_data(self, season, division='E0'):
        """
        Downloads football data for a given season and division.
        """
        
        season_code = f"{str(season)[-2:]}{str(season + 1)[-2:]}"
        url = f"{self.base_url}/{season_code}/{division}.csv"
        print(f"Downloading season {season}-{season+1} data from {url}...")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))
            df['Season'] = f"{season}-{season + 1}"
            df = self.preprocess_data(df)

            filename = os.path.join(self.save_dir, f"{division}_{season_code}.csv")
            df.to_csv(filename, index=False)
            print(f"Data for season {season}-{season+1} saved to {filename}.")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Failed to download data for season {season}-{season+1}: {e}")
            return None
            
    def get_available_columns(self, df):
        """
        Print information about available columns
        """
        print("\n=== Available Columns ===")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Show sample of odds columns
        odds_cols = [col for col in df.columns if any(col.startswith(bm) for bm in ['B365', 'BW', 'IW', 'PS', 'WH'])]
        if odds_cols:
            print(f"\nBookmaker odds columns ({len(odds_cols)}):")
            for col in odds_cols[:10]:  # Show first 10
                print(f"  - {col}")
            if len(odds_cols) > 10:
                print(f"  ... and {len(odds_cols) - 10} more")


if __name__ == "__main__":
    importer = FootballDataImporter(save_dir='data/epl/')
    
    df = importer.download_season_data(season=2022, division='E0')

    if df is not None:
        print("===Sample Data===")
        print(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].head())


        print("=== Basic Statistics ===")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Number of teams: {df['HomeTeam'].nunique()}")
        print(f"Total matches: {len(df)}")
        print(f"\nResult distribution:")
        print(df['FTR'].value_counts())
    