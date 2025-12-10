import pandas as pd
import numpy as np
from scipy import stats

# --- Helper Function for General Metrics ---
def calculate_metrics(df):
    """Calculates the core risk and profit metrics required for A/B testing."""
    df_copy = df.copy()
    
    # 1. Claim Frequency (Binary: 1 if claim > 0, 0 otherwise)
    df_copy['HasClaim'] = np.where(df_copy['TotalClaims'] > 0, 1, 0)
    
    # 2. Margin (Profit)
    df_copy['Margin'] = df_copy['TotalPremium'] - df_copy['TotalClaims']
    
    return df_copy


# --- 1. Testing Categorical Risk (Claim Frequency) using Chi-squared ---
def test_claim_frequency(df, grouping_col, group_a, group_b, alpha=0.05):
    """
    Tests H₀: There are no risk differences (Claim Frequency) between two groups.
    Uses a Chi-squared test on the contingency table of claims vs. no-claims.
    ... [omitted detailed function body for brevity, assuming you have the full code]
    """
    
    # 1. Segmentation
    group_a_data = df[df[grouping_col] == group_a]
    group_b_data = df[df[grouping_col] == group_b]
    
    if group_a_data.empty or group_b_data.empty:
        print(f"Error: One of the groups ({group_a} or {group_b}) is empty.")
        return np.nan

    # Calculate contingency table (claims/no-claims for A and B)
    claims_a = group_a_data['HasClaim'].sum()
    policies_a = len(group_a_data)
    no_claims_a = policies_a - claims_a

    claims_b = group_b_data['HasClaim'].sum()
    policies_b = len(group_b_data)
    no_claims_b = policies_b - claims_b
    
    contingency_table = np.array([
        [claims_a, claims_b],
        [no_claims_a, no_claims_b]
    ])
    
    # Run the Chi-squared Test 
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # Results reporting logic here...
    
    return p_value


# --- 2. Testing Numerical Differences (Margin or Severity) using T-test ---
def test_numerical_difference(df, grouping_col, metric_col, group_a, group_b, alpha=0.05, filter_claims=False):
    """
    Tests H₀: There is no significant difference in the mean of a numerical metric 
    (Margin or Claim Severity) between two groups. Uses a two-sample T-test.
    ... [omitted detailed function body for brevity, assuming you have the full code]
    """
    
    df_segment = df.copy()
    
    if filter_claims:
        df_segment = df_segment[df_segment['HasClaim'] == 1]
        
    # 1. Segmentation
    data_a = df_segment[df_segment[grouping_col] == group_a][metric_col].dropna()
    data_b = df_segment[df_segment[grouping_col] == group_b][metric_col].dropna()
    
    if data_a.empty or data_b.empty:
        print(f"Error: Not enough data for one or both groups in {metric_col} comparison.")
        return np.nan

    # Run the T-test 
    t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)

    # Results reporting logic here...
        
    return p_value