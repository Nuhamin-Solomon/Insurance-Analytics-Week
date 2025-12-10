import pandas as pd
import numpy as np
from scipy import stats

# --- Helper Function for General Metrics ---
def calculate_metrics(df):
    """
    Calculates the core risk and profit metrics required for A/B testing: 
    Claim Frequency (HasClaim) and Margin.
    """
    if df is None:
        return None
        
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
    
    Args:
        df (pd.DataFrame): DataFrame containing 'HasClaim' and the grouping column.
        grouping_col (str): The column used to define the A/B groups (e.g., 'Province').
        group_a (str): Name of the control group.
        group_b (str): Name of the test group.
        alpha (float): Significance level (default 0.05).
        
    Returns:
        float: The p-value from the statistical test.
    """
    
    # 1. Segmentation
    group_a_data = df[df[grouping_col] == group_a]
    group_b_data = df[df[grouping_col] == group_b]
    
    if group_a_data.empty or group_b_data.empty:
        print(f"Error: One of the groups ({group_a} or {group_b}) is empty.")
        return np.nan

    # 2. Contingency Table (Policies with/without claims for Group A and Group B)
    claims_a = group_a_data['HasClaim'].sum()
    policies_a = len(group_a_data)
    no_claims_a = policies_a - claims_a

    claims_b = group_b_data['HasClaim'].sum()
    policies_b = len(group_b_data)
    no_claims_b = policies_b - claims_b
    
    contingency_table = np.array([
        [claims_a, claims_b],       # Row 1: Claims Count
        [no_claims_a, no_claims_b]  # Row 2: No Claims Count
    ])
    
    # 3. Chi-squared Test
    # This test determines if there's a significant association between the 'group' 
    # and the 'outcome' (having a claim).
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # 4. Results Reporting
    freq_a = claims_a / policies_a
    freq_b = claims_b / policies_b
    
    print(f"\n--- Chi-squared Test for Claim Frequency: {group_a} vs {group_b} ---")
    print(f"Metric (Frequency): {group_a}: {freq_a:.4f} | {group_b}: {freq_b:.4f}")
    print(f"Chi2 Stat: {chi2:.3f}, P-value: {p_value:.5f}")
    
    if p_value < alpha:
        print(f"-> Conclusion: Reject H₀. Significant difference in risk/frequency found (p < {alpha}).")
    else:
        print(f"-> Conclusion: Fail to reject H₀. No significant difference in risk/frequency found (p >= {alpha}).")

    return p_value


# --- 2. Testing Numerical Differences (Margin or Severity) using T-test ---
def test_numerical_difference(df, grouping_col, metric_col, group_a, group_b, alpha=0.05, filter_claims=False):
    """
    Tests H₀: There is no significant difference in the mean of a numerical metric 
    (Margin or Claim Severity) between two groups. Uses a two-sample T-test.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metric column.
        grouping_col (str): The column used to define the A/B groups.
        metric_col (str): The numerical column to test (e.g., 'Margin', 'TotalClaims').
        group_a (str): Name of the control group.
        group_b (str): Name of the test group.
        alpha (float): Significance level (default 0.05).
        filter_claims (bool): If True, only includes data where 'HasClaim' == 1 (for Severity).
        
    Returns:
        float: The p-value from the statistical test.
    """
    
    df_segment = df.copy()
    
    if filter_claims:
        # For Claim Severity, we only look at records where a claim occurred
        df_segment = df_segment[df_segment['HasClaim'] == 1]
        print(f"Note: Only policies with claims are used for {metric_col} analysis (Claim Severity).")
        
    # 1. Segmentation
    data_a = df_segment[df_segment[grouping_col] == group_a][metric_col].dropna()
    data_b = df_segment[df_segment[grouping_col] == group_b][metric_col].dropna()
    
    if data_a.empty or data_b.empty:
        print(f"Error: Not enough data for one or both groups in {metric_col} comparison.")
        return np.nan

    # 2. T-test (Welch's T-test, assuming unequal variances is safer)
    # This test compares the means of two independent groups.
    t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)

    # 3. Results Reporting
    mean_a = data_a.mean()
    mean_b = data_b.mean()
    
    print(f"\n--- T-Test for Mean {metric_col}: {group_a} vs {group_b} ---")
    print(f"Metric (Mean): {group_a}: {mean_a:.2f} | {group_b}: {mean_b:.2f}")
    print(f"T-Stat: {t_stat:.3f}, P-value: {p_value:.5f}")

    if p_value < alpha:
        print(f"-> Conclusion: Reject H₀. Significant difference in mean {metric_col} found (p < {alpha}).")
    else:
        print(f"-> Conclusion: Fail to reject H₀. No significant difference in mean {metric_col} found (p >= {alpha}).")
        
    return p_value