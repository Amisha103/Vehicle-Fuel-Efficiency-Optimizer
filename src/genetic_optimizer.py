import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import random
import os
import warnings

warnings.filterwarnings("ignore")


POPULATION_SIZE = 40      # Number of individuals in each generation
GENERATIONS = 20          # Total evolutionary iterations
MUTATION_RATE = 0.1       # Probability of mutation
ELITE_COUNT = 5           # How many top individuals survive unchanged
RANDOM_SEED = 42          # Reproducibility

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



data = pd.read_csv("data/processed_engine_data.csv")

target_col = "num_Gross_Indicated_Thermal_Efficiency"
X = data.drop(columns=[target_col])
y = data[target_col]


categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"Encoding categorical columns: {categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
else:
    print("No categorical columns detected.")


print("Scaling numeric features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================

print("🌲 Training RandomForest model...")
rf = RandomForestRegressor(
    n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1
)
rf.fit(X_scaled, y)

# =========================
# 4️⃣ Genetic Algorithm Functions
# =========================

def create_individual():
    """
    Each individual represents an engine configuration.
    Each gene (value between 0 and 1) corresponds to a normalized feature.
    """
    return np.random.rand(X_scaled.shape[1])


def evaluate_fitness(individual):
    """
    Fitness function:
    Predicts the Gross Indicated Thermal Efficiency using the trained model.
    Higher efficiency = better fitness.
    """
    return rf.predict([individual])[0]


def tournament_selection(population, fitness, tournament_size=5):
    """
    Parent selection method — Tournament Selection.
    Randomly samples 'tournament_size' individuals and picks the fittest.
    ✅ Advantage: Balances exploration & exploitation and avoids premature convergence.
    """
    selected = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = selected[np.argmax(fitness[selected])]
    return population[best_idx]


def arithmetic_crossover(parent1, parent2):
    """
    Crossover method — Arithmetic Crossover.
    Produces an offspring that is a weighted blend of two parents.
    ✅ Advantage: Preserves useful traits from both parents and maintains numerical stability.
    """
    alpha = np.random.uniform(0.3, 0.7)  # Random blending ratio
    child = alpha * np.array(parent1) + (1 - alpha) * np.array(parent2)
    return np.clip(child, 0, 1)  # Keep within normalized bounds


def gaussian_mutation(individual):
    """
    Mutation method — Gaussian Mutation.
    Slightly perturbs genes by adding Gaussian noise.
    ✅ Advantage: Introduces small variations for better exploration without disrupting good genes.
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += np.random.normal(0, 0.1)
            individual[i] = np.clip(individual[i], 0, 1)
    return individual


# =========================
# 5️⃣ Main GA Evolution Loop
# =========================
print("⚙️ Running Genetic Algorithm...")

population = [create_individual() for _ in range(POPULATION_SIZE)]
efficiency_log = []

for gen in range(GENERATIONS):
    # Calculate fitness for entire population
    fitness = np.array([evaluate_fitness(ind) for ind in population])
    best_idx = np.argmax(fitness)
    best_fit = fitness[best_idx]

    efficiency_log.append({"Generation": gen + 1, "Best_Efficiency": best_fit})

    # Display progress
    if (gen + 1) % 2 == 0 or gen == 0:
        print(f"🧬 Generation {gen+1}/{GENERATIONS} | Best Efficiency: {best_fit:.4f}")

    # Elitism — keep top-performing individuals unchanged
    elites = [population[i] for i in np.argsort(fitness)[-ELITE_COUNT:]]

    # Generate next generation
    new_population = elites.copy()
    while len(new_population) < POPULATION_SIZE:
        # Parent selection via tournament method
        p1 = tournament_selection(np.array(population), fitness)
        p2 = tournament_selection(np.array(population), fitness)

        # Crossover
        child = arithmetic_crossover(p1, p2)

        # Mutation
        child = gaussian_mutation(child)

        new_population.append(child)

    population = new_population

# =========================
# 6️⃣ Final Results
# =========================
final_fitness = np.array([evaluate_fitness(ind) for ind in population])
best_idx = np.argmax(final_fitness)
best_individual = population[best_idx]
best_efficiency = final_fitness[best_idx]

# Convert best solution back to original feature scale
best_config = pd.Series(scaler.inverse_transform([best_individual])[0], index=X.columns)

print("\n🏁 Best Engine Configuration Found:")
print(best_config)
print(f"\n🔥 Predicted Gross Indicated Thermal Efficiency: {best_efficiency:.4f}")

# =========================
# 7️⃣ Save Outputs
# =========================
os.makedirs("results", exist_ok=True)
best_config.to_csv("results/best_engine_configuration.csv")
pd.DataFrame(efficiency_log).to_csv("results/efficiency_log.csv", index=False)

print("\n✅ Saved results to 'results/' folder (best configuration + efficiency log)")
print("📊 Now you can visualize results using: python src/visualize_results.py")
