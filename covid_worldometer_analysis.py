# covid_worldometer_analysis.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Load data
# ----------------------------
INPUT_XLSX = "worldometer_data.xlsx"
SHEET_NAME = "worldometer_data"

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

# Clean numeric columns
def to_numeric_safe(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("+", "", regex=False)
         .str.replace("N/A", "", regex=False)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan})
         .astype(float)
    )

for c in [
    "Population","TotalCases","NewCases","TotalDeaths","NewDeaths",
    "TotalRecovered","NewRecovered","ActiveCases","Serious,Critical",
    "Tot Cases/1M pop","Deaths/1M pop","TotalTests","Tests/1M pop"
]:
    if c in df.columns:
        df[c] = to_numeric_safe(df[c])

# ----------------------------
# Derived metrics
# ----------------------------
df["CFR"] = df["TotalDeaths"] / df["TotalCases"]
df["RecoveryRate"] = df["TotalRecovered"] / df["TotalCases"]
df["CasesPer1M_calc"] = df["TotalCases"]*1_000_000 / df["Population"]
df["DeathsPer1M_calc"] = df["TotalDeaths"]*1_000_000 / df["Population"]
df["TestsPer1M_calc"] = df["TotalTests"]*1_000_000 / df["Population"]

# ----------------------------
# Descriptive stats
# ----------------------------
print("\n=== Descriptive Statistics ===")
print(df.describe(include=[np.number]).T)

# ----------------------------
# Visualizations (inline)
# ----------------------------
# Top 15 by TotalCases
top_cases = df[["Country/Region","TotalCases"]].dropna().sort_values("TotalCases", ascending=False).head(15)
plt.figure()
plt.bar(top_cases["Country/Region"], top_cases["TotalCases"])
plt.xticks(rotation=60, ha="right")
plt.title("Top 15 Countries by Total Cases")
plt.show()

# Top 20 CFR
cfr_df = df[df["TotalCases"] >= 1000][["Country/Region","CFR"]].dropna().sort_values("CFR", ascending=False).head(20)
plt.figure()
plt.bar(cfr_df["Country/Region"], cfr_df["CFR"])
plt.xticks(rotation=60, ha="right")
plt.title("Top 20 CFR (≥1000 cases)")
plt.show()

# Scatter: Tests vs Cases per 1M
plt.figure()
plt.scatter(df["TestsPer1M_calc"], df["CasesPer1M_calc"], alpha=0.6)
plt.xlabel("Tests per 1M")
plt.ylabel("Cases per 1M")
plt.title("Tests vs Cases per 1M")
plt.show()

# ----------------------------
# Correlation
# ----------------------------
print("\n=== Correlation Matrix ===")
print(df.corr(numeric_only=True))

plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
im = plt.imshow(corr.values, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# Clustering
# ----------------------------
features = ["CasesPer1M_calc","DeathsPer1M_calc","TestsPer1M_calc","CFR","RecoveryRate"]
cluster_df = df[["Country/Region"] + features].dropna()
if len(cluster_df) > 10:
    X = StandardScaler().fit_transform(cluster_df[features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df["Cluster"] = km.fit_predict(X)

    print("\n=== Sample Clustering Output ===")
    print(cluster_df.head())

    plt.figure()
    plt.scatter(cluster_df["CasesPer1M_calc"], cluster_df["DeathsPer1M_calc"],
                c=cluster_df["Cluster"], cmap="tab10", alpha=0.7)
    plt.xlabel("Cases per 1M")
    plt.ylabel("Deaths per 1M")
    plt.title("KMeans Clustering (k=4)")
    plt.show()

# ----------------------------
# Regression
# ----------------------------
reg_df = df[["DeathsPer1M_calc","CasesPer1M_calc","TestsPer1M_calc","CFR","RecoveryRate"]].dropna()
if len(reg_df) > 20:
    X = reg_df[["CasesPer1M_calc","TestsPer1M_calc","CFR","RecoveryRate"]].values
    y = reg_df["DeathsPer1M_calc"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = r2_score(y, pred)

    print("\n=== Regression Coefficients ===")
    for feat, coef in zip(["CasesPer1M","TestsPer1M","CFR","RecoveryRate"], model.coef_):
        print(f"{feat}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R² Score: {r2:.3f}")

    plt.figure()
    plt.scatter(y, pred, alpha=0.6)
    plt.xlabel("Actual Deaths/1M")
    plt.ylabel("Predicted Deaths/1M")
    plt.title(f"Regression Fit (R²={r2:.3f})")
    plt.show()

print("\n=== Analysis Complete ===")
# ----------------------------
# Continent-wise Summaries
# ----------------------------
if "Continent" in df.columns:
    print("\n=== Continent-wise Summary (mean values) ===")
    cont_summary = df.groupby("Continent")[[
        "TotalCases","TotalDeaths","TotalRecovered",
        "CasesPer1M_calc","DeathsPer1M_calc",
        "TestsPer1M_calc","CFR","RecoveryRate"
    ]].mean(numeric_only=True).round(2)
    print(cont_summary)

    print("\n=== Continent-wise Summary (median values) ===")
    cont_summary_med = df.groupby("Continent")[[
        "TotalCases","TotalDeaths","TotalRecovered",
        "CasesPer1M_calc","DeathsPer1M_calc",
        "TestsPer1M_calc","CFR","RecoveryRate"
    ]].median(numeric_only=True).round(2)
    print(cont_summary_med)
else:
    print("\n(No 'Continent' column found, skipping continent-wise summary.)")

# ----------------------------
# Ranking Tables
# ----------------------------
print("\n=== Top 10 Countries by Recovery Rate ===")
print(df[["Country/Region","RecoveryRate"]]
      .dropna()
      .sort_values("RecoveryRate", ascending=False)
      .head(10))

print("\n=== Top 10 Countries by CFR (≥1000 cases) ===")
print(df[df["TotalCases"] >= 1000][["Country/Region","CFR"]]
      .dropna()
      .sort_values("CFR", ascending=False)
      .head(10))

print("\n=== Top 10 Countries by Tests per Million ===")
print(df[["Country/Region","TestsPer1M_calc"]]
      .dropna()
      .sort_values("TestsPer1M_calc", ascending=False)
      .head(10))

print("\n=== Top 10 Countries by Deaths per Million ===")
print(df[["Country/Region","DeathsPer1M_calc"]]
      .dropna()
      .sort_values("DeathsPer1M_calc", ascending=False)
      .head(10))

# ----------------------------
# Monte Carlo Simulation
# ----------------------------
print("\n=== Monte Carlo Simulation: Predicting Deaths per 1M ===")

mc_df = df[["CasesPer1M_calc","CFR","RecoveryRate"]].dropna()

if not mc_df.empty:
    np.random.seed(42)
    sims = 10000
    sampled_cases = np.random.choice(mc_df["CasesPer1M_calc"], sims, replace=True)
    sampled_cfr = np.random.choice(mc_df["CFR"], sims, replace=True)
    sampled_recovery = np.random.choice(mc_df["RecoveryRate"], sims, replace=True)

    # Predicted deaths per 1M = cases_per_1M * CFR
    simulated_deaths = sampled_cases * sampled_cfr

    print(f"Simulated {sims} possible outcomes of deaths per 1M.")
    print(f"Mean: {np.mean(simulated_deaths):.2f}, Median: {np.median(simulated_deaths):.2f}")
    print(f"5th percentile: {np.percentile(simulated_deaths,5):.2f}, 95th percentile: {np.percentile(simulated_deaths,95):.2f}")

    plt.figure()
    plt.hist(simulated_deaths, bins=50, color="skyblue", edgecolor="black")
    plt.title("Monte Carlo Simulation of Deaths per 1M")
    plt.xlabel("Deaths per 1M")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("Not enough data for Monte Carlo simulation.")

# ----------------------------
# Active vs Recovered Ratios
# ----------------------------
if "ActiveCases" in df.columns and "TotalCases" in df.columns:
    print("\n=== Active vs Recovered vs Deaths Ratios ===")
    ratio_df = df[["Country/Region","ActiveCases","TotalRecovered","TotalDeaths","TotalCases"]].dropna()
    ratio_df["Active%"] = ratio_df["ActiveCases"]/ratio_df["TotalCases"]*100
    ratio_df["Recovered%"] = ratio_df["TotalRecovered"]/ratio_df["TotalCases"]*100
    ratio_df["Deaths%"] = ratio_df["TotalDeaths"]/ratio_df["TotalCases"]*100

    print(ratio_df[["Country/Region","Active%","Recovered%","Deaths%"]].head(10))

    plt.figure()
    avg_ratios = ratio_df[["Active%","Recovered%","Deaths%"]].mean()
    avg_ratios.plot(kind="bar", color=["orange","green","red"])
    plt.title("Average Case Distribution Across Countries")
    plt.ylabel("Percentage")
    plt.show()
