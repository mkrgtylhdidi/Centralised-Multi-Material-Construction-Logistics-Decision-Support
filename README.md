# 🏗️ DQN-Based Construction Logistics Decision Support

This repository provides a prototype **construction logistics decision-support tool** based on a trained **Deep Q-Network (DQN)**.

The system recommends a coordinated daily truck allocation for three construction materials under shared delivery-capacity and storage constraints.

---

## 🔗 Try the Live App

👉 [Click here to try the Streamlit app](https://construction-dqn-app-2lmryrp2uwbgms7aijzf6x.streamlit.app/)

The app allows users to select the available daily truck capacity and enter current project and site information. The trained DQN then returns a recommended truck allocation for the three material types.

---

## 🚚 Materials and Delivery Capacity

The prototype considers three construction materials:

- **Material A — Rebar**
- **Material B — Formwork**
- **Material C — Mesh / palletised material**

Users can select one of three daily truck-capacity limits:

- **V = 4 trucks/day**
- **V = 5 trucks/day**
- **V = 6 trucks/day**

A separately trained DQN model is loaded for each capacity level.

The corresponding joint-action spaces contain:

- V = 4 → **35 feasible allocations**
- V = 5 → **56 feasible allocations**
- V = 6 → **84 feasible allocations**

---

## 🧠 Operational Information Used by the DQN

The interface uses routinely available project and site-management information.

For each material, the user provides:

- Current inventory
- Current backlog
- Planned demand today
- Planned demand tomorrow
- Known incoming delivery

The user also enters the current project day.

The interface automatically calculates shared storage utilisation and internally constructs the **17-dimensional operational state** used by the trained DQN.

The user does **not** need to specify reinforcement-learning parameters or manually estimate latent uncertainty variables such as weather severity, productivity factors, or disruption probabilities.

---

## ⚙️ How to Use

1. Select the available daily truck capacity: **4, 5, or 6 trucks/day**
2. Enter the current project day
3. Enter the current inventory, backlog, planned demand and incoming delivery information for Materials A, B and C
4. Click **Generate delivery recommendation**
5. The system returns the recommended coordinated truck allocation for the three materials

The interface also reports the corresponding ordered quantities and current shared-storage utilisation.

---

## 📦 Trained Models

The repository contains three separately trained DQN models:

- `dqn_V4_seed_42.pth`
- `dqn_V5_seed_42.pth`
- `dqn_V6_seed_42.pth`

Each model corresponds to a different daily site-level truck-capacity constraint.

---

## 📁 Files Included

- `app.py` — Streamlit decision-support interface
- `dqn_V4_seed_42.pth` — trained DQN for V = 4
- `dqn_V5_seed_42.pth` — trained DQN for V = 5
- `dqn_V6_seed_42.pth` — trained DQN for V = 6
- `requirements.txt` — Python dependencies
- `DQN_training_and_evaluation.ipynb` — training and evaluation notebook *(to be added)*

---

## 💻 Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🔬 Reproducibility

The accompanying notebook provides the final environment configuration, DQN training procedure, multi-scenario evaluation and truck-capacity sensitivity analysis used in the associated study.

The Streamlit interface is intended as a prototype implementation of the trained decision policy rather than a standalone commercial planning system.

---

## 📌 Notes

The recommendations are generated under the assumptions and case-study parameters described in the associated manuscript, including the 21-day planning horizon, shared storage constraint and one-day replenishment structure.

---
