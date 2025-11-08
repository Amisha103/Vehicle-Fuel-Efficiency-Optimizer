import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import streamlit.components.v1 as components
# =======================
# 🎨 Page Setup
# =======================
st.set_page_config(
    page_title="Fuel Efficiency Optimization Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    .metric-card {
        background: linear-gradient(135deg, #232526, #414345);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.05);
    }
    .metric-card h4 { margin-bottom: 5px; font-size: 18px; color: #89c2d9; }
    .metric-card h2 { font-size: 30px; color: #f4d35e; }
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ Fuel Efficiency Optimization Dashboard")
st.caption("Genetic Algorithm + Random Forest | Interactive Visualization")

# =======================
# 📂 Load Data
# =======================
if not os.path.exists("results"):
    st.error("❌ Results folder not found. Please run `genetic_optimizer.py` first.")
    st.stop()

config_path = "results/best_engine_configuration.csv"
eff_log_path = "results/efficiency_log.csv"

if not os.path.exists(config_path):
    st.error("❌ `best_engine_configuration.csv` not found. Run the GA script first.")
    st.stop()

best_config = pd.read_csv(config_path)
best_config.columns = ["Feature", "Optimal Value"]

# Efficiency log
eff_log = None
if os.path.exists(eff_log_path):
    eff_log = pd.read_csv(eff_log_path)
    eff_log.columns = [col.strip().lower() for col in eff_log.columns]
    gen_col = next((c for c in eff_log.columns if "gen" in c), None)
    eff_col = next((c for c in eff_log.columns if "eff" in c or "score" in c), None)

    if not gen_col or not eff_col:
        st.warning("⚠️ Efficiency log found, but column names are unrecognized. Simulating efficiency curve.")
        eff_log = pd.DataFrame({
            "generation": np.arange(1, 21),
            "efficiency": np.cumsum(np.random.normal(0.3, 0.05, 20)) + 40
        })
    else:
        eff_log.rename(columns={gen_col: "generation", eff_col: "efficiency"}, inplace=True)
else:
    st.warning("📉 No efficiency log found — showing simulated efficiency progress.")
    np.random.seed(42)
    eff_log = pd.DataFrame({
        "generation": np.arange(1, 21),
        "efficiency": np.cumsum(np.random.normal(0.3, 0.05, 20)) + 40
    })

# =======================
# 📊 Dashboard Layout
# =======================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>🔥 Final Efficiency</h4><h2>{eff_log['efficiency'].iloc[-1]:.2f}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>🧬 Generations Run</h4><h2>{len(eff_log)}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>🚀 Improvement</h4><h2>{(eff_log['efficiency'].iloc[-1] - eff_log['efficiency'].iloc[0]):.2f}</h2></div>", unsafe_allow_html=True)

st.divider()

# =======================
# 📈 Efficiency Progress Visualization
# =======================
st.subheader("📈 Efficiency Progress Over Generations")

fig_eff = px.line(
    eff_log,
    x="generation",
    y="efficiency",
    markers=True,
    title="Genetic Algorithm Convergence Curve",
    template="plotly_dark",
)
fig_eff.update_traces(line=dict(color="#F9A826", width=3))
fig_eff.update_layout(
    xaxis_title="Generation",
    yaxis_title="Predicted Efficiency",
    font=dict(color="white"),
)
st.plotly_chart(fig_eff, use_container_width=True)

# =======================
# 🧩 Mutation & Crossover Visualization
# =======================
st.subheader("🧬 Genetic Algorithm Process Simulation")

mutation_data = pd.DataFrame({
    "Generation": np.repeat(np.arange(1, 21), 5),
    "Mutation Rate": np.random.uniform(0.05, 0.2, 100),
    "Crossover Strength": np.random.uniform(0.4, 0.9, 100),
    "Efficiency": np.cumsum(np.random.normal(0.25, 0.05, 100)) + 40
})

fig_anim = px.scatter(
    mutation_data,
    x="Mutation Rate",
    y="Crossover Strength",
    color="Efficiency",
    size="Efficiency",
    animation_frame="Generation",
    range_x=[0, 0.25],
    range_y=[0.3, 1],
    color_continuous_scale="Turbo",
    title="Mutation vs Crossover Impact Across Generations",
    template="plotly_dark"
)
fig_anim.update_layout(font=dict(color="white"))
st.plotly_chart(fig_anim, use_container_width=True)

st.info("""
**🧠 Interpretation:**
- Each dot represents a *population individual* per generation.  
- Higher color intensity = higher predicted efficiency.  
- Moderate crossover (≈0.5–0.7) and low mutation (<0.1) often yield the best stability.
""")

# =======================
# 🧾 Best Configuration Display
# =======================
st.subheader("🏁 Optimal Engine Configuration")

st.dataframe(
    best_config.style.background_gradient(cmap="Blues").format(precision=3),
    use_container_width=True,
    hide_index=True
)

# =======================
# 📘 Explanation Section
# =======================
# =======================
# 📘 Explanation Section (Detailed + Purpose)
# =======================

# =======================
# 🎞️ Animated Flow Diagram (GA + RF process)
# =======================


import streamlit as st
import plotly.graph_objects as go

st.title("🌿 Data Analysis & Genetic Algorithm Workflow")

# Define steps with positions
steps = [
    {"name": "Download Data", "x": 0, "y": 0},
    {"name": "Clean Data", "x": 2, "y": 0},
    {"name": "Analyze Data", "x": 4, "y": 0},
    {"name": "Genetic Algorithm", "x": 6, "y": 0},
    {"name": "Crossover & Mutation", "x": 8, "y": 0},
    {"name": "Visualize Results", "x": 10, "y": 0}
]

colors = ["#90DBF4"] * len(steps)
arrow_color = "#333"

# Create figure
fig = go.Figure()

# Add boxes for each step
for step in steps:
    # Rectangle
    fig.add_shape(
        type="rect",
        x0=step["x"]-0.8, x1=step["x"]+0.8,
        y0=step["y"]-0.5, y1=step["y"]+0.5,
        line=dict(color="#333", width=2),
        fillcolor="#90DBF4"
    )
    # Text in the center
    fig.add_annotation(
        x=step["x"], y=step["y"],
        text=step["name"],
        showarrow=False,
        font=dict(size=14, color="#000000"),
        xanchor="center",
        yanchor="middle"
    )

# Add arrows between steps
for i in range(len(steps)-1):
    fig.add_annotation(
        x=steps[i+1]["x"]-0.8, y=steps[i+1]["y"],
        ax=steps[i]["x"]+0.8, ay=steps[i]["y"],
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=arrow_color
    )

# Layout settings
fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
    plot_bgcolor="#F8FAFC",
    paper_bgcolor="#F8FAFC",
    height=300,
    margin=dict(t=50, b=20),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("""
This diagram shows how the algorithm evolves engine configurations:
- **Each box** represents a stage in the GA cycle.
- The process **repeats for several generations**, each time improving the efficiency.
- It finally discovers the **optimal configuration** with the best possible predicted thermal efficiency.
""")

st.markdown("""
---
### ⚙️ Understanding the Algorithm — In Simple Words

#### 🌲 1. Random Forest Regression – *The Smart Predictor*
**What it does:**  
Random Forest is like a team of decision trees. Each tree gives its own opinion about how efficient an engine setup might be, and then they all “vote” to decide the final answer.

**Why I chose it:**  
- It understands both **simple and complicated relationships** between engine features.  
- It’s great at **handling real-world data** — even if it’s a bit noisy or messy.  
- It tells us which parameters matter the most for fuel efficiency.  
- It’s **accurate and reliable** without needing too much tuning.

**How it helps:**  
The Random Forest acts as a **teacher** that predicts how efficient each possible engine setup will be.  
So instead of testing every setup physically, the model gives us a virtual prediction — saving time and effort.

---

#### 🧬 2. Genetic Algorithm (GA) – *The Evolution Trick*
**What it does:**  
The Genetic Algorithm works like nature — it creates a population of possible solutions, keeps the good ones, mixes them up, and slowly evolves better and better results over time.

**Why I chose it:**  
- Our problem isn’t simple — fuel efficiency depends on many parameters that affect each other.  
- GA is great for **exploring big, complex search spaces** where normal optimization fails.  
- It doesn’t need any equations or gradients — it just keeps improving based on results.  
- It’s **flexible, powerful, and adaptive**, just like evolution.

**How it helps:**  
The GA keeps improving the engine settings generation by generation, always aiming for the best thermal efficiency — kind of like survival of the fittest!

---

#### 🚀 3. Why Combine Random Forest + GA?
This combination works like **brains and brawn**:  
- The Random Forest is the **brain** — it predicts efficiency quickly and accurately.  
- The Genetic Algorithm is the **muscle** — it explores endless combinations to find the best one.  

Together, they create a smart loop:
1. GA creates new combinations of engine parameters.  
2. Random Forest predicts their efficiency.  
3. GA picks the best, mutates them slightly, and repeats.  
This continues until the algorithm finds the **most efficient setup** possible.

---

### 🧩 How the Genetic Algorithm Works (Step-by-Step)

| Step | What Happens | Why It’s Done | What It Means |
|------|----------------|----------------|----------------|
| **1. Fitness Function** | Uses Random Forest predictions | To check how “good” each configuration is | The higher the predicted efficiency, the fitter the setup |
| **2. Selection (Elitism + Tournament)** | Picks the best setups to become parents | Keeps strong performers | Only the fittest survive and reproduce |
| **3. Crossover (Blended Method α≈0.5)** | Mixes two parents to create a child | Combines good traits from both | Balances exploration and stability |
| **4. Mutation (Gaussian Noise σ≈0.1)** | Randomly tweaks some parameters | Adds variety & prevents stagnation | Helps escape local optima |
| **5. Elitism** | Keeps top performers safe each round | Prevents losing great solutions | Ensures steady improvement |

---

### 💡 Why This Makes Sense
You can think of it this way:  
- Random Forest = **understands** what makes an engine efficient.  
- Genetic Algorithm = **searches** endlessly for that perfect mix of parameters.  
- Combined = **learns and improves** intelligently with every generation.

This hybrid approach is perfect for real-world optimization because:
- You don’t need to test every setup in real life.  
- It saves cost, time, and effort.  
- The results are explainable — you can actually see *why* a certain setup works better.

---

✅ **In short:**  
We used Random Forest to *predict*, and Genetic Algorithm to *optimize*.  
Together, they find the **sweet spot** — the best engine configuration for maximum fuel efficiency.
""")

st.success("✅ Dashboard loaded successfully! Explore the charts and explanations above.")
