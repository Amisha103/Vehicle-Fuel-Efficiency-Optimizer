import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 1️⃣ Load Results
# =========================
print("📂 Loading results...")

best_config_path = "results/best_engine_configuration.csv"
eff_log_path = "results/efficiency_log.csv"
correlation_path = "results/correlation_matrix.csv" if os.path.exists("results/correlation_matrix.csv") else None

if os.path.exists(best_config_path):
    best_config = pd.read_csv(best_config_path)
    print("\n🏁 Best Configuration Summary:")
    print(best_config.head())
else:
    raise FileNotFoundError("❌ 'best_engine_configuration.csv' not found in results/ folder.")

if os.path.exists(eff_log_path):
    eff_log = pd.read_csv(eff_log_path)
    print("\n📈 Loaded efficiency log successfully!")
else:
    eff_log = None
    print("\n⚠️ No efficiency log found!")

# =========================
# 2️⃣ Plot Efficiency Evolution
# =========================
if eff_log is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(eff_log["Generation"], eff_log["Best_Efficiency"], marker="o", linewidth=2, color="teal")
    plt.title("Evolution of Best Efficiency Over Generations", fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Best Efficiency (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/efficiency_evolution_line.png")
    plt.show()
else:
    print("⚠️ Skipping efficiency evolution plot (no log).")

# =========================
# 3️⃣ Feature Importance (Bar Chart)
# =========================
# Try loading from the correlation CSV if available
if correlation_path:
    corr = pd.read_csv(correlation_path, index_col=0)
    if "num_Gross_Indicated_Thermal_Efficiency" in corr.columns:
        corr_target = corr["num_Gross_Indicated_Thermal_Efficiency"].abs().sort_values(ascending=False)
        top_features = corr_target.head(10)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        plt.title("Top 10 Most Correlated Features", fontsize=14)
        plt.xlabel("Correlation with Efficiency")
        plt.tight_layout()
        plt.savefig("results/top_correlated_features.png")
        plt.show()

# =========================
# 4️⃣ 3D Scatter Plot
# =========================
# Using top 3 numeric features (if present in best config)
try:
    numeric_features = [col for col in best_config["Unnamed: 0"].values if "num_" in col][:3]
    values = best_config.iloc[:3, 1].values

    if len(numeric_features) == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(values[0], values[1], values[2], color='crimson', s=100)
        ax.set_xlabel(numeric_features[0])
        ax.set_ylabel(numeric_features[1])
        ax.set_zlabel(numeric_features[2])
        ax.set_title("3D Visualization of Key Engine Parameters", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/3d_feature_scatter.png")
        plt.show()
    else:
        print("⚠️ Not enough numeric features for 3D scatter plot.")
except Exception as e:
    print(f"⚠️ Could not create 3D plot: {e}")

# =========================
# 5️⃣ Heatmap (Optional)
# =========================
try:
    data = pd.read_csv("data/processed_engine_data.csv")
    numeric_data = data.select_dtypes(include=["number"])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap of Engine Parameters", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/feature_correlation_heatmap.png")
    plt.show()
except Exception as e:
    print(f"⚠️ Could not create heatmap: {e}")

print("\n✅ All visualizations generated successfully in 'results/' folder!")
