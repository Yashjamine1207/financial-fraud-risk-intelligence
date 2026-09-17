"""
Create temporal train/validation/test splits using TransactionDT.

This script:
- Sorts data by TransactionDT (timestamp)
- Splits chronologically: 70% train, 15% validation, 15% test
- Ensures no data leakage from future to past
- Saves split indices and metadata for reproducibility

The test set represents the most recent transactions and must remain untouched
until final evaluation (no model selection or threshold tuning on it).
"""

import json
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Temporal split function
# -----------------------------------------------------------------------------

def create_temporal_splits(df: pd.DataFrame, time_col: str = 'TransactionDT',
                           train_ratio: float = 0.70, val_ratio: float = 0.15,
                           test_ratio: float = 0.15) -> dict:
    """
    Split dataframe chronologically by timestamp column.
    
    Args:
        df: Input dataframe (must include time_col)
        time_col: Timestamp column for ordering
        train_ratio: Proportion for training (default 70%)
        val_ratio: Proportion for validation (default 15%)
        test_ratio: Proportion for testing (default 15%)
    
    Returns:
        Dict with train/val/test dataframes and split metadata
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Sort by timestamp
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    # Metadata
    metadata = {
        'total_rows': n,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'train_ratio_actual': len(train_df) / n,
        'val_ratio_actual': len(val_df) / n,
        'test_ratio_actual': len(test_df) / n,
        'time_col': time_col,
        'train_time_range': {
            'min': int(train_df[time_col].min()),
            'max': int(train_df[time_col].max())
        },
        'val_time_range': {
            'min': int(val_df[time_col].min()),
            'max': int(val_df[time_col].max())
        },
        'test_time_range': {
            'min': int(test_df[time_col].min()),
            'max': int(test_df[time_col].max())
        },
        'fraud_rate_train': float(train_df['isFraud'].mean()),
        'fraud_rate_val': float(val_df['isFraud'].mean()),
        'fraud_rate_test': float(test_df['isFraud'].mean())
    }
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'metadata': metadata
    }


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """
    Load joined transaction data and create temporal train/val/test splits.
    Save splits to Parquet and metadata to JSON.
    """
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "interim" / "joined_transactions" / "train_joined.parquet"
    output_dir = project_root / "data" / "interim" / "temporal_splits"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")
    
    # Create splits
    print("\nCreating temporal splits (70/15/15)...")
    splits = create_temporal_splits(df, time_col='TransactionDT')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits to Parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    test_path = output_dir / "test.parquet"
    
    print("\nSaving splits:")
    print(f"  Train: {train_path} ({len(splits['train']):,} rows)")
    print(f"  Val:   {val_path} ({len(splits['val']):,} rows)")
    print(f"  Test:  {test_path} ({len(splits['test']):,} rows)")
    
    splits['train'].to_parquet(train_path, index=False)
    splits['val'].to_parquet(val_path, index=False)
    splits['test'].to_parquet(test_path, index=False)
    
    # Save metadata
    metadata_path = output_dir / "split_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(splits['metadata'], f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Print summary
    meta = splits['metadata']
    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total rows: {meta['total_rows']:,}")
    print(f"Train: {meta['train_rows']:,} ({meta['train_ratio_actual']:.1%})")
    print(f"Val:   {meta['val_rows']:,} ({meta['val_ratio_actual']:.1%})")
    print(f"Test:  {meta['test_rows']:,} ({meta['test_ratio_actual']:.1%})")
    print("\nTime ranges (TransactionDT):")
    print(f"  Train: {meta['train_time_range']['min']:,} to {meta['train_time_range']['max']:,}")
    print(f"  Val:   {meta['val_time_range']['min']:,} to {meta['val_time_range']['max']:,}")
    print(f"  Test:  {meta['test_time_range']['min']:,} to {meta['test_time_range']['max']:,}")
    print("\nFraud rates:")
    print(f"  Train: {meta['fraud_rate_train']:.2%}")
    print(f"  Val:   {meta['fraud_rate_val']:.2%}")
    print(f"  Test:  {meta['fraud_rate_test']:.2%}")
    print("\n⚠️  IMPORTANT: Test set must remain untouched until final evaluation!")


if __name__ == "__main__":
    main()