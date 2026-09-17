"""
Audit missingness patterns in IEEE-CIS data by fraud class and time.

This script:
- Calculates null rates overall and stratified by isFraud
- Identifies columns with significantly different missingness between classes
- Analyzes missingness trends over time (TransactionDT bins)
- Saves structured audit results for leakage documentation

Helps identify which columns are reliable vs. which are too sparse for modelling.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# -----------------------------------------------------------------------------
# Missingness analysis functions
# -----------------------------------------------------------------------------

def calculate_null_rates(
    df: pd.DataFrame,
    group_col: str | None = None,
) -> pd.Series:
    """
    Calculate null rates per column, optionally stratified by a grouping column.
    
    Args:
        df: Input dataframe
        group_col: Optional column to stratify by (e.g., 'isFraud')
    
    Returns:
        Series with null rates per column
    """
    return df.isna().sum() / len(df)


def calculate_stratified_null_rates(df: pd.DataFrame, group_col: str = 'isFraud') -> dict:
    """
    Calculate null rates stratified by a grouping column.
    
    Returns dict with overall and per-group null rates.
    """
    results = {}
    
    # Overall
    results['overall'] = (df.isna().sum() / len(df)).to_dict()
    
    # By group
    for group in sorted(df[group_col].unique()):
        subset = df[df[group_col] == group]
        results[f'group_{group}'] = (subset.isna().sum() / len(subset)).to_dict()
    
    return results


def identify_differential_missingness(df: pd.DataFrame, group_col: str = 'isFraud', 
                                      threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify columns where missingness differs significantly between groups.
    Uses chi-squared test for independence.
    
    Args:
        df: Input dataframe
        group_col: Grouping column (e.g., 'isFraud')
        threshold: P-value threshold for significance
    
    Returns:
        DataFrame with columns showing differential missingness statistics
    """
    results = []
    
    for col in df.columns:
        if col == group_col:
            continue
        
        # Create contingency table: missing vs. group
        contingency = pd.crosstab(df[col].isna(), df[group_col])
        
        # Need at least 2x2 table with sufficient counts
        if contingency.shape != (2, 2) or contingency.min().min() < 5:
            continue
        
        # Chi-squared test
        chi2, p_value, dof, _expected = chi2_contingency(contingency)
        
        # Calculate missingness rates by group
        missing_by_group = df.groupby(group_col)[col].apply(lambda x: x.isna().mean())
        
        results.append({
            'column': col,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < threshold,
            'missing_rate_group_0': missing_by_group.get(0, np.nan),
            'missing_rate_group_1': missing_by_group.get(1, np.nan),
            'difference': abs(missing_by_group.get(0, 0) - missing_by_group.get(1, 0))
        })
    
    return pd.DataFrame(results).sort_values('difference', ascending=False)


def analyze_temporal_missingness(df: pd.DataFrame, time_col: str = 'TransactionDT',
                                 n_bins: int = 10) -> dict:
    """
    Analyze how missingness changes over time.
    
    Args:
        df: Input dataframe
        time_col: Timestamp column
        n_bins: Number of time bins
    
    Returns:
        Dict with temporal missingness patterns for high-null columns
    """
    # Create time bins
    df = df.copy()
    df['time_bin'] = pd.qcut(df[time_col], q=n_bins, labels=False, duplicates='drop')
    
    # Focus on columns with >5% nulls
    null_rates = df.isna().sum() / len(df)
    sparse_cols = null_rates[null_rates > 0.05].index.tolist()
    
    temporal_patterns = {}
    
    for col in sparse_cols[:20]:  # Limit to top 20 sparsest columns
        missing_by_bin = df.groupby('time_bin')[col].apply(lambda x: x.isna().mean())
        temporal_patterns[col] = {
            'null_rate_by_bin': missing_by_bin.to_dict(),
            'overall_null_rate': null_rates[col],
            'trend': 'increasing' if missing_by_bin.iloc[-1] > missing_by_bin.iloc[0] else 
                     'decreasing' if missing_by_bin.iloc[-1] < missing_by_bin.iloc[0] else 'stable'
        }
    
    return temporal_patterns


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """
    Load joined transaction data and perform comprehensive missingness audit.
    Save results to data/interim/missingness_audit.json and missingness_summary.csv
    """
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "interim" / "joined_transactions" / "train_joined.parquet"
    output_json = project_root / "data" / "interim" / "missingness_audit.json"
    output_csv = project_root / "data" / "interim" / "missingness_summary.csv"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # 1. Overall null rates
    print("\n1. Calculating overall null rates...")
    overall_nulls = calculate_null_rates(df)
    high_null_cols = overall_nulls[overall_nulls > 0.95]
    print(f"   Columns with >95% nulls: {len(high_null_cols)}")
    
    # 2. Stratified by fraud class
    print("\n2. Calculating null rates by fraud class...")
    stratified_nulls = calculate_stratified_null_rates(df, group_col='isFraud')
    
    # 3. Differential missingness (chi-squared test)
    print("\n3. Testing for differential missingness between fraud classes...")
    diff_missing = identify_differential_missingness(df, group_col='isFraud')
    significant_cols = diff_missing[diff_missing['significant']]
    print(f"   Columns with significant differential missingness (p<0.05): {len(significant_cols)}")
    
    # 4. Temporal missingness patterns
    print("\n4. Analyzing temporal missingness patterns...")
    temporal_patterns = analyze_temporal_missingness(df)
    
    # 5. Compile audit report
    audit_report = {
        'summary': {
            'total_columns': len(df.columns),
            'columns_with_any_nulls': int((df.isna().any()).sum()),
            'columns_with_>95%_nulls': len(high_null_cols),
            'columns_with_differential_missingness': len(significant_cols)
        },
        'overall_null_rates': overall_nulls.to_dict(),
        'stratified_null_rates': stratified_nulls,
        'high_null_columns': high_null_cols.index.tolist(),
        'differential_missingness': {
            'significant_columns': significant_cols['column'].tolist(),
            'top_20_by_difference': diff_missing.head(20)[['column', 'difference', 'p_value']].to_dict('records')
        },
        'temporal_patterns': temporal_patterns
    }
    
    # Save JSON report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2, default=str)
    print(f"\nAudit report saved to: {output_json}")
    
    # Save CSV summary for quick review
    summary_df = overall_nulls.reset_index()
    summary_df.columns = ['column', 'null_rate']
    summary_df['high_null_flag'] = summary_df['column'].isin(high_null_cols.index)
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary CSV saved to: {output_csv}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("MISSINGNESS AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total columns: {audit_report['summary']['total_columns']}")
    print(f"Columns with any nulls: {audit_report['summary']['columns_with_any_nulls']}")
    print(f"Columns with >95% nulls: {audit_report['summary']['columns_with_>95%_nulls']}")
    print(f"Columns with differential missingness: {audit_report['summary']['columns_with_differential_missingness']}")
    
    print("\nTop 10 sparsest columns:")
    top_10_sparse = overall_nulls.nlargest(10)
    for col, rate in top_10_sparse.items():
        print(f"  {col}: {rate:.1%}")
    
    print("\nTop 10 columns with largest fraud-class missingness difference:")
    for row in diff_missing.head(10).itertuples(index=False):
        print(f"  {row.column}: {row.difference:.1%} difference (p={row.p_value:.4f})")


if __name__ == "__main__":
    main()