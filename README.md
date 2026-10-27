# 🏗️ Centralised Multi-Material DQN for Construction Logistics

This repository contains the implementation and decision-support prototype for a **centralised Deep Q-Network (DQN) framework for multi-material construction logistics planning**.

Unlike a single-material ordering model, the framework makes **one coordinated site-level decision across three material types** while accounting for shared truck availability and shared on-site storage.

The current case study considers a **21-day planning horizon** and jointly allocates delivery trucks to:

- **Material A — Rebar**
- **Material B — Formwork**
- **Material C — Mesh / palletised material**

---

## 🔗 Try the Live Decision-Support App

👉 **[Launch the Streamlit application](https://centralised-multi-material-construction-logistics-decision-sup.streamlit.app/)**

The application allows the user to select the available daily fleet capacity and enter current project and site information. The corresponding trained DQN then recommends a **joint truck allocation across all three materials**.

---

## 🚚 Centralised Site-Level Decision

The DQN does not make three independent ordering decisions.

Instead, each action specifies a coordinated allocation:

```text
(A trucks, B trucks, C trucks)
```

subject to the shared site-level constraint:

```text
A trucks + B trucks + C trucks ≤ daily fleet capacity
```

This allows the policy to allocate limited delivery capacity across competing material requirements.

---

## 🚛 Selectable Daily Fleet Capacity

The prototype provides separately trained DQN policies for three daily fleet-capacity (\(N_{\max}\)) settings:

- **\(N_{\max}\) = 4 trucks/day**
- **\(N_{\max}\) = 5 trucks/day**
- **\(N_{\max}\) = 6 trucks/day**

The corresponding joint-action spaces contain:

- \(N_{\max}\) = 4 → **35 feasible joint allocations**
- \(N_{\max}\) = 5 → **56 feasible joint allocations**
- \(N_{\max}\) = 6 → **84 feasible joint allocations**

Selecting a fleet size in the Streamlit interface automatically loads the corresponding trained DQN model.

---

## 📅 Planning Horizon

The case study represents a **21-day construction planning horizon**.

The DQN uses the current project position together with near-term planned demand and current logistics conditions to support daily delivery decisions throughout the planning period.

---

## 🧠 Operational State Information

The centralised DQN uses a **17-dimensional operational state**.

The state contains:

- Current project day
- Shared storage utilisation

For each of the three materials:

- Current inventory
- Current backlog
- Planned demand today
- Planned demand tomorrow
- Known incoming delivery

The interface automatically derives shared storage utilisation from the current inventories.

The user therefore interacts with familiar project and site-management information rather than with a technical reinforcement-learning state vector.

The application does **not** require the user to manually estimate latent uncertainty variables such as weather severity, productivity factors, disruption probabilities or subjective risk levels.

---

## 🏗️ Shared Site Constraints

The three material flows are coordinated under common site-level constraints.

The framework considers:

- Shared daily truck capacity
- Shared site-storage capacity
- Material-specific truck capacities
- Material-specific storage footprints
- Inventory and backlog
- Planned near-term demand
- Known incoming deliveries

This allows the DQN to represent the trade-offs created when several materials compete for limited construction-logistics resources.

---

## ⚙️ How to Use the App

1. Select the available daily fleet capacity: **4, 5 or 6 trucks/day**
2. Enter the current project day
3. Enter inventory, backlog, planned demand and incoming delivery information for Materials A, B and C
4. Click **Generate delivery recommendation**
5. The trained DQN returns the recommended joint allocation of trucks across the three materials

The interface also displays:

- Current shared-storage use
- Storage utilisation
- Recommended truck allocation for each material
- Corresponding ordered quantities

---

## 📦 Trained DQN Models

The repository contains separately trained models for each fleet-capacity setting:

- `dqn_V4_seed_42.pth`
- `dqn_V5_seed_42.pth`
- `dqn_V6_seed_42.pth`

Each model has the same 17-dimensional operational state but a different output dimension corresponding to its feasible joint-action space.

---

## 🔄 Relationship to the Earlier Single-Material Prototype

An earlier public repository, [`dqn-order-app`](https://github.com/mkrgtylhdidi/dqn-order-app), demonstrates a **single-material reinforcement-steel ordering model**.

The present repository addresses a different and broader decision problem.

The main differences are:

| Earlier prototype | Current framework |
|---|---|
| Single material | Three materials |
| Independent order-quantity decision | Centralised joint truck-allocation decision |
| One trained ordering model | Separate DQN policies for V = 4, 5 and 6 |
| Material-level storage | Shared site-level storage constraint |
| 8 state features | 17-dimensional centralised operational state |
| Discrete order quantity | Joint `(A, B, C)` fleet allocation |
| Single-material logistics support | Coordinated site-level logistics support |

This repository therefore focuses on **multi-material coordination under shared construction-logistics constraints**.

---

## 📁 Files Included

- `app.py` — Streamlit decision-support interface
- `dqn_V4_seed_42.pth` — trained DQN for V = 4
- `dqn_V5_seed_42.pth` — trained DQN for V = 5
- `dqn_V6_seed_42.pth` — trained DQN for V = 6
- `requirements.txt` — Python dependencies
- `Centralised_Multi_Material_DQN_Construction_Logistics.ipynb` — final training and evaluation notebook

---

## 💻 Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔬 Reproducibility

The accompanying notebook provides the final environment configuration, DQN training procedure, multi-scenario evaluation, benchmark evaluation and truck-capacity sensitivity analysis used in the associated study.

The Streamlit interface is a prototype implementation of the trained decision policies and is intended to demonstrate how the framework can be translated into an operational construction-logistics decision-support tool.

---

## 📌 Notes

The recommendations are generated under the assumptions and case-study parameters described in the associated manuscript, including the **21-day planning horizon**, shared truck-capacity constraint, shared storage constraint and one-day replenishment structure.
