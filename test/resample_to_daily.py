"""
Resample 1-minute parquet data to daily OHLCV.
Reads from data/raw/, writes to data/processed/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import glob
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, get_daily_path


def main():
    print("=" * 60)
    print("RESAMPLING 1-MINUTE DATA TO DAILY")
    print("=" * 60)

    # Create processed directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Find all parquet files in raw data directory
    parquet_files = glob.glob(str(RAW_DATA_DIR / "*.parquet"))

    if not parquet_files:
        print(f"ERROR: No parquet files found in {RAW_DATA_DIR}")
        print("Please place your Binance data files in this directory.")
        return

    print(f"\nFound {len(parquet_files)} files to process")

    for file_path in parquet_files:
        # Extract symbol from filename (e.g., BTCUSDT_1m_... -> BTC)
        filename = Path(file_path).name
        symbol_raw = filename.split('_')[0]
        symbol = symbol_raw.replace('USDT', '').upper()

        print(f"\n{'=' * 40}")
        print(f"PROCESSING {symbol}")
        print(f"{'=' * 40}")
        print(f"File: {filename}")

        # Load 1-minute parquet data
        df_1m = pd.read_parquet(file_path)

        # Ensure open_time is datetime and set as index
        if 'open_time' in df_1m.columns:
            df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
            df_1m.set_index('open_time', inplace=True)

        print(f"Records loaded: {len(df_1m):,}")
        print(f"Date range: {df_1m.index.min().date()} to {df_1m.index.max().date()}")

        # Resample to daily: OHLCV aggregation
        df_daily = df_1m.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"Daily records generated: {len(df_daily):,}")

        # Calculate log returns
        df_daily['log_return'] = np.log(df_daily['close'] / df_daily['close'].shift(1))

        # Calculate basic metrics
        if len(df_daily) > 1:
            total_return = (df_daily['close'].iloc[-1] / df_daily['close'].iloc[0] - 1) * 100
            print(f"\nPeriod total return: {total_return:.2f}%")
            print(f"Initial price: ${df_daily['close'].iloc[0]:,.2f}")
            print(f"Final price: ${df_daily['close'].iloc[-1]:,.2f}")

        # Save to feather (fast, binary format)
        daily_path = get_daily_path(symbol)
        df_daily.reset_index().to_feather(daily_path)
        print(f"\nSAVED: {daily_path}")

        # Also save as CSV for easy inspection
        csv_path = daily_path.with_suffix('.csv')
        df_daily.to_csv(csv_path)
        print(f"BACKUP: {csv_path}")

        # Display sample
        print("\nSAMPLE DAILY DATA:")
        print(df_daily[['open', 'high', 'low', 'close', 'volume', 'log_return']].head())

    print("\n" + "=" * 60)
    print("RESAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput directory: {PROCESSED_DATA_DIR}")
    print("\nNext step: python test/calibrate_all_jumps.py")


if __name__ == "__main__":
    main()