"""
Exploratory Data Analysis: Fraud patterns by amount, time, and categorical fields.

This script:
- Analyzes transaction amount distributions (fraud vs. non-fraud)
- Examines temporal patterns (hour, day of week)
- Calculates fraud rates by ProductCD, card type, email domain
- Profiles high-cardinality fields (top categories by fraud rate)
- Saves summary statistics and plots for documentation

Focus: Identify behavioural patterns without making invalid temporal assumptions.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Analysis functions
# -----------------------------------------------------------------------------

def analyze_amount_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze transaction amount distribution overall and by fraud label.
    """
    results = {
        'overall': {
            'mean': float(df['TransactionAmt'].mean()),
            'median': float(df['TransactionAmt'].median()),
            'std': float(df['TransactionAmt'].std()),
            'min': float(df['TransactionAmt'].min()),
            'max': float(df['TransactionAmt'].max()),
            'q25': float(df['TransactionAmt'].quantile(0.25)),
            'q75': float(df['TransactionAmt'].quantile(0.75))
        },
        'by_fraud': {}
    }
    
    for label in sorted(df['isFraud'].unique()):
        subset = df[df['isFraud'] == label]['TransactionAmt']
        results['by_fraud'][label] = {
            'mean': float(subset.mean()),
            'median': float(subset.median()),
            'std': float(subset.std()),
            'count': len(subset)
        }
    
    return results


def analyze_temporal_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze fraud rates by hour of day and day of week.
    """
    df = df.copy()
    
    # Convert TransactionDT to hours (assuming seconds from epoch-like reference)
    # IEEE-CIS: TransactionDT is seconds since a reference point
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    
    # Fraud rate by hour
    fraud_by_hour = df.groupby('hour')['isFraud'].agg(['mean', 'count']).reset_index()
    fraud_by_hour.columns = ['hour', 'fraud_rate', 'transaction_count']
    
    # Fraud rate by day of week
    fraud_by_dow = df.groupby('day_of_week')['isFraud'].agg(['mean', 'count']).reset_index()
    fraud_by_dow.columns = ['day_of_week', 'fraud_rate', 'transaction_count']
    
    return {
        'fraud_by_hour': fraud_by_hour.to_dict('records'),
        'fraud_by_day_of_week': fraud_by_dow.to_dict('records')
    }


def analyze_categorical_fraud_rates(df: pd.DataFrame, columns: list) -> dict:
    """
    Calculate fraud rates and transaction counts for categorical columns.
    """
    results = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Group by category
        grouped = df.groupby(col)['isFraud'].agg(['mean', 'count', 'sum']).reset_index()
        grouped.columns = ['category', 'fraud_rate', 'transaction_count', 'fraud_count']
        grouped = grouped.sort_values('fraud_rate', ascending=False)
        
        results[col] = {
            'unique_categories': int(df[col].nunique()),
            'null_count': int(df[col].isna().sum()),
            'top_20_by_fraud_rate': grouped.head(20).to_dict('records')
        }
    
    return results


def profile_high_cardinality_fields(df: pd.DataFrame, columns: list, 
                                    top_n: int = 10) -> dict:
    """
    Profile high-cardinality fields: top categories by frequency and fraud rate.
    """
    results = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # By frequency
        freq = df[col].value_counts().head(top_n)
        
        # By fraud rate (min 100 transactions)
        fraud_rates = df.groupby(col)['isFraud'].agg(['mean', 'count'])
        fraud_rates = fraud_rates[fraud_rates['count'] >= 100].sort_values('mean', ascending=False).head(top_n)
        
        results[col] = {
            'top_by_frequency': {str(cat): int(count) for cat, count in freq.items()},
            'top_by_fraud_rate': {
                str(cat): {
                    'fraud_rate': float(row['mean']),
                    'count': int(row['count'])
                }
                for cat, row in fraud_rates.iterrows()
            }
        }
    
    return results


# -----------------------------------------------------------------------------
# Visualization functions
# -----------------------------------------------------------------------------

def plot_amount_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot transaction amount distribution (log scale) by fraud label.
    """
    plt.figure(figsize=(10, 6))
    
    for label in sorted(df['isFraud'].unique()):
        subset = df[df['isFraud'] == label]['TransactionAmt']
        subset_log = np.log1p(subset[subset > 0])  # Log transform, exclude zeros
        sns.kdeplot(subset_log, label=f'Fraud={label}', fill=True, alpha=0.3)
    
    plt.xlabel('Log(Transaction Amount + 1)')
    plt.ylabel('Density')
    plt.title('Transaction Amount Distribution by Fraud Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fraud_rate_by_hour(df: pd.DataFrame, output_path: Path):
    """
    Plot fraud rate by hour of day.
    """
    df = df.copy()
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    
    fraud_by_hour = df.groupby('hour')['isFraud'].mean().reset_index()
    
    plt.figure(figsize=(12, 5))
    plt.bar(fraud_by_hour['hour'], fraud_by_hour['isFraud'], color='steelblue', alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate')
    plt.title('Fraud Rate by Hour of Day')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fraud_rate_by_productcd(df: pd.DataFrame, output_path: Path):
    """
    Plot fraud rate and transaction count by ProductCD.
    """
    grouped = df.groupby('ProductCD')['isFraud'].agg(['mean', 'count']).reset_index()
    
    _, ax1 = plt.subplots(figsize=(10, 6))
    
    # Fraud rate (bar)
    ax1.bar(grouped['ProductCD'], grouped['mean'], color='coral', alpha=0.7, label='Fraud Rate')
    ax1.set_xlabel('ProductCD')
    ax1.set_ylabel('Fraud Rate', color='coral')
    ax1.tick_params(axis='y', labelcolor='coral')
    ax1.set_title('Fraud Rate and Transaction Count by ProductCD')
    
    # Transaction count (line on secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(grouped['ProductCD'], grouped['count'], color='steelblue', marker='o', 
             linewidth=2, markersize=8, label='Transaction Count')
    ax2.set_ylabel('Transaction Count', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """
    Load train split and perform comprehensive EDA on fraud patterns.
    Save statistics to JSON and plots to reports/figures/.
    """
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "interim" / "temporal_splits" / "train.parquet"
    output_dir_stats = project_root / "data" / "interim" / "eda_statistics"
    output_dir_figs = project_root / "reports" / "figures"
    
    print(f"Loading train split from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")
    
    # Create output directories
    output_dir_stats.mkdir(parents=True, exist_ok=True)
    output_dir_figs.mkdir(parents=True, exist_ok=True)
    
    # 1. Amount distribution
    print("\n1. Analyzing amount distribution...")
    amount_stats = analyze_amount_distribution(df)
    
    # 2. Temporal patterns
    print("2. Analyzing temporal patterns...")
    temporal_stats = analyze_temporal_patterns(df)
    
    # 3. Categorical fraud rates
    print("3. Analyzing categorical fraud rates...")
    categorical_cols = ['ProductCD', 'card4', 'card6', 'id_30', 'id_31', 'id_33', 'id_34']
    categorical_stats = analyze_categorical_fraud_rates(df, categorical_cols)
    
    # 4. High-cardinality field profiling
    print("4. Profiling high-cardinality fields...")
    high_card_cols = ['R_emaildomain', 'P_emaildomain', 'DeviceInfo', 'id_31', 'id_33']
    high_card_stats = profile_high_cardinality_fields(df, high_card_cols)
    
    # 5. Compile statistics
    eda_stats = {
        'amount_distribution': amount_stats,
        'temporal_patterns': temporal_stats,
        'categorical_fraud_rates': categorical_stats,
        'high_cardinality_profiles': high_card_stats
    }
    
    # Save statistics
    stats_path = output_dir_stats / "fraud_patterns.json"
    
    # Custom JSON encoder to handle numpy types
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    eda_stats_serializable = convert_to_serializable(eda_stats)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(eda_stats_serializable, f, indent=2, default=str)
    print(f"\nStatistics saved to: {stats_path}")
    
    # 6. Generate plots
    print("\n5. Generating plots...")
    
    plot_amount_distribution(df, output_dir_figs / "amount_distribution_by_fraud.png")
    print("  - amount_distribution_by_fraud.png")
    
    plot_fraud_rate_by_hour(df, output_dir_figs / "fraud_rate_by_hour.png")
    print("  - fraud_rate_by_hour.png")
    
    plot_fraud_rate_by_productcd(df, output_dir_figs / "fraud_rate_by_productcd.png")
    print("  - fraud_rate_by_productcd.png")
    
    # 7. Print key findings
    print("\n" + "=" * 60)
    print("EDA FRAUD PATTERNS SUMMARY")
    print("=" * 60)
    
    print("\nAmount distribution:")
    print(f"  Overall mean: ${amount_stats['overall']['mean']:.2f}")
    print(f"  Overall median: ${amount_stats['overall']['median']:.2f}")
    print(f"  Fraud mean: ${amount_stats['by_fraud'][1]['mean']:.2f}")
    print(f"  Non-fraud mean: ${amount_stats['by_fraud'][0]['mean']:.2f}")
    
    print("\nFraud rate by ProductCD:")
    for row in categorical_stats.get('ProductCD', {}).get('top_20_by_fraud_rate', [])[:5]:
        print(f"  {row['category']}: {row['fraud_rate']:.1%} ({row['transaction_count']:,} txns)")
    
    print("\nTop email domains by fraud rate (min 100 txns):")
    for col in ['R_emaildomain', 'P_emaildomain']:
        if col in high_card_stats:
            top_domain = next(
                iter(high_card_stats[col]["top_by_fraud_rate"].items())
            )
            print(f"  {col}: {top_domain[0]} → {top_domain[1]['fraud_rate']:.1%} fraud rate")
    
    print(f"\nPlots saved to: {output_dir_figs}")


if __name__ == "__main__":
    main()