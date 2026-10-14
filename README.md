# DQN-Based Construction Logistics Decision Support

This repository contains the code and trained models supporting a centralised Deep Q-Network (DQN) framework for construction material logistics planning under shared delivery and storage constraints.

## Streamlit decision-support prototype

The Streamlit interface provides daily truck-allocation recommendations for three construction materials:

- Material A: Rebar
- Material B: Formwork
- Material C: Mesh / palletised material

The user selects the available daily truck capacity:

- V = 4 trucks/day
- V = 5 trucks/day
- V = 6 trucks/day

A separately trained DQN policy is loaded for each capacity level.

## Operational inputs

The user provides routinely available project and site-management information:

- Current project day
- Current inventory
- Current backlog
- Planned demand today
- Planned demand tomorrow
- Known incoming delivery

The interface automatically derives shared storage utilisation and constructs the 17-dimensional state used by the trained DQN.

The user does not need to specify reinforcement-learning parameters or estimate latent uncertainty variables such as weather severity, productivity factors, or disruption probabilities.

## Output

The trained DQN returns a coordinated joint truck allocation for Materials A, B and C subject to the selected daily truck-capacity limit.

## Trained models

The repository contains separately trained models for:

- `dqn_V4_seed_42.pth`
- `dqn_V5_seed_42.pth`
- `dqn_V6_seed_42.pth`

The corresponding action-space dimensions are:

- V = 4: 35 feasible joint allocations
- V = 5: 56 feasible joint allocations
- V = 6: 84 feasible joint allocations

## Running the Streamlit app

Install the required packages:

```bash
pip install -r requirements.txt
