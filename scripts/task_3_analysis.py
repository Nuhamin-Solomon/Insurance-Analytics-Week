# ============================
# Task 3: Risk Drivers Analysis
# ============================

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ----------------------------
# 1. Load Data
# ----------------------------
df = pd.read_csv("data/processed/insurance_data_processed.csv")

# ----------------------------
# 2. Calculate Metrics
# ----------------------------
df['ClaimFrequency'] = (df['ClaimCount'] > 0).astype(int)
df['ClaimSeverity'] = df.apply(lambda x: x['TotalClaims']/x['ClaimCount'] if x['ClaimCount']>0 else 0, axis=1)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# ----------------------------
# 3. Data Segmentation
# ----------------------------
def segment_data(df, feature, category1, category2):
    groupA = df[df[feature] == category1]
    groupB = df[df[feature] == category2]
    return groupA, groupB

# ----------------------------
# 4. Statistical Tests
# ----------------------------
def t_test(groupA, groupB, metric):
    t_stat, p_value = ttest_ind(groupA[metric], groupB[metric], equal_var=False)
    return t_stat, p_value

def chi2_test(df, feature, target):
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p, dof, ex = chi2_contingency(contingency)
    return chi2, p

def interpret_result(p_value, alpha=0.05):
    if p_value < alpha:
        return "Reject H₀ → Statistically significant difference"
    else:
        return "Fail to reject H₀ → No significant difference"

# ----------------------------
# 5. Apply Tests (Examples)
# ----------------------------

# Gender differences
groupA, groupB = segment_data(df, 'Gender', 'Female', 'Male')
t_stat_severity, p_severity = t_test(groupA, groupB, 'ClaimSeverity')
chi2_freq, p_freq = chi2_test(df, 'Gender', 'ClaimFrequency')

# Province differences (example: Gauteng vs Western Cape)
groupA_prov, groupB_prov = segment_data(df, 'Province', 'Gauteng', 'Western Cape')
t_stat_margin, p_margin = t_test(groupA_prov, groupB_prov, 'Margin')
chi2_freq_prov, p_freq_prov = chi2_test(df[df['Province'].isin(['Gauteng','Western Cape'])], 'Province', 'ClaimFrequency')

# ----------------------------
# 6. Print Results
# ----------------------------
print("=== Gender ===")
print(f"Claim Severity: t-stat={t_stat_severity:.2f}, p-value={p_severity:.4f} → {interpret_result(p_severity)}")
print(f"Claim Frequency: chi2={chi2_freq:.2f}, p-value={p_freq:.4f} → {interpret_result(p_freq)}\n")

print("=== Province (Gauteng vs Western Cape) ===")
print(f"Margin: t-stat={t_stat_margin:.2f}, p-value={p_margin:.4f} → {interpret_result(p_margin)}")
print(f"Claim Frequency: chi2={chi2_freq_prov:.2f}, p-value={p_freq_prov:.4f} → {interpret_result(p_freq_prov)}\n")

# ----------------------------
# 7. Visualizations
# ----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x='Gender', y='ClaimFrequency', data=df)
plt.title("Claim Frequency by Gender")
plt.savefig("results/figures/claim_frequency_gender.png")
plt.close()

plt.figure(figsize=(8,4))
sns.boxplot(x='Province', y='ClaimSeverity', data=df)
plt.title("Claim Severity by Province")
plt.savefig("results/figures/claim_severity_province.png")
plt.close()

# ----------------------------
# 8. Save Report
# ----------------------------
report = f"""
## Task 3 Statistical Analysis Report

### Gender
- Claim Frequency p-value: {p_freq:.4f} → {interpret_result(p_freq)}
- Claim Severity p-value: {p_severity:.4f} → {interpret_result(p_severity)}

### Province (Gauteng vs Western Cape)
- Claim Frequency p-value: {p_freq_prov:.4f} → {interpret_result(p_freq_prov)}
- Margin p-value: {p_margin:.4f} → {interpret_result(p_margin)}
"""

with open("results/reports/Task_3_report.md", "w") as f:
    f.write(report)

print("Report saved to results/reports/Task_3_report.md")
print("Figures saved to results/figures/")
