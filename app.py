# ============================================================
# Construction Logistics Decision Support Interface
# Centralised DQN
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Construction Logistics Decision Support",
    page_icon="🏗️",
    layout="wide"
)


# ============================================================
# 2. MODEL / CASE CONSTANTS
# ============================================================

STATE_DIM = 17

# Shared site storage capacity
SHARED_STORAGE_CAPACITY = 20.0  # m²

# Storage footprint per unit
STORAGE_FOOTPRINT = {
    "A": 1.0,
    "B": 1.5,
    "C": 1.2,
}

# Quantity transported by one truck
TRUCK_CAPACITY = {
    "A": 3.0,
    "B": 2.0,
    "C": 4.0,
}

# Folder containing this app.py file
APP_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    4: APP_DIR / "dqn_V4_seed_42.pth",
    5: APP_DIR / "dqn_V5_seed_42.pth",
    6: APP_DIR / "dqn_V6_seed_42.pth",
}


# ============================================================
# 3. DQN NETWORK
# Must exactly match the network used during training
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. BUILD JOINT ACTION SPACE
# Must exactly match the training environment
# ============================================================

def build_joint_action_space(vehicle_limit):

    joint_actions = []

    for a in range(vehicle_limit + 1):
        for b in range(vehicle_limit + 1):
            for c in range(vehicle_limit + 1):

                if a + b + c <= vehicle_limit:
                    joint_actions.append(
                        (a, b, c)
                    )

    return joint_actions


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(vehicle_limit):

    model_path = MODEL_PATHS[vehicle_limit]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Check model metadata
    # --------------------------------------------------------

    saved_vehicle_limit = checkpoint["vehicle_limit"]
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    if saved_vehicle_limit != vehicle_limit:
        raise ValueError(
            f"Selected V={vehicle_limit}, but model was "
            f"trained for V={saved_vehicle_limit}."
        )

    if state_dim != STATE_DIM:
        raise ValueError(
            f"Expected state dimension {STATE_DIM}, "
            f"but checkpoint contains {state_dim}."
        )

    # --------------------------------------------------------
    # Build matching action space
    # --------------------------------------------------------

    joint_actions = build_joint_action_space(
        vehicle_limit
    )

    if len(joint_actions) != action_dim:
        raise ValueError(
            f"Action-space mismatch. "
            f"Checkpoint expects {action_dim} actions, "
            f"but generated action space contains "
            f"{len(joint_actions)}."
        )

    # --------------------------------------------------------
    # Rebuild network and load trained weights
    # --------------------------------------------------------

    model = QNetwork(
        state_dim=state_dim,
        action_dim=action_dim
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint, joint_actions


# ============================================================
# 6. USER INTERFACE HEADER
# ============================================================

st.title(
    "Construction Logistics Decision Support"
)

st.write(
    """
    This prototype uses a trained Deep Q-Network (DQN) to
    recommend a coordinated daily truck allocation for three
    construction materials under a shared delivery-capacity
    constraint.
    """
)


# ============================================================
# 7. DAILY TRUCK CAPACITY SELECTION
# ============================================================

st.subheader("1. Site delivery capacity")

vehicle_limit = st.selectbox(
    "Maximum number of trucks available per day",
    options=[4, 5, 6],
    index=0,
    help=(
        "Select the maximum number of trucks that can be "
        "allocated across Materials A, B and C during one day."
    )
)

try:

    model, checkpoint, joint_actions = load_model(
        vehicle_limit
    )

    st.success(
        f"Decision policy loaded for a daily capacity "
        f"of {vehicle_limit} trucks."
    )

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# ============================================================
# 8. OPERATIONAL INPUT FORM
# ============================================================

st.subheader("2. Current project and site information")

st.caption(
    "Enter information available from the current programme, "
    "inventory records and delivery information."
)

with st.form("logistics_input_form"):

    # --------------------------------------------------------
    # Project day
    # Human-facing day is 1-21.
    # DQN state uses 0-20 internally.
    # --------------------------------------------------------

    project_day = st.number_input(
        "Current project day",
        min_value=1,
        max_value=21,
        value=1,
        step=1,
        help="Day within the 21-day structural-phase planning horizon."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Material columns
    # --------------------------------------------------------

    col_a, col_b, col_c = st.columns(3)

    # ========================================================
    # MATERIAL A
    # ========================================================

    with col_a:

        st.markdown("### Material A — Rebar")

        a_inventory = st.number_input(
            "Current inventory (A)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_inventory"
        )

        a_backlog = st.number_input(
            "Current backlog (A)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_backlog"
        )

        a_today = st.number_input(
            "Planned demand today (A)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_today"
        )

        a_tomorrow = st.number_input(
            "Planned demand tomorrow (A)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_tomorrow"
        )

        a_pipeline = st.number_input(
            "Known incoming delivery / pipeline (A)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_pipeline"
        )

    # ========================================================
    # MATERIAL B
    # ========================================================

    with col_b:

        st.markdown("### Material B — Formwork")

        b_inventory = st.number_input(
            "Current inventory (B)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_inventory"
        )

        b_backlog = st.number_input(
            "Current backlog (B)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_backlog"
        )

        b_today = st.number_input(
            "Planned demand today (B)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_today"
        )

        b_tomorrow = st.number_input(
            "Planned demand tomorrow (B)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_tomorrow"
        )

        b_pipeline = st.number_input(
            "Known incoming delivery / pipeline (B)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_pipeline"
        )

    # ========================================================
    # MATERIAL C
    # ========================================================

    with col_c:

        st.markdown("### Material C — Mesh / Palletised Material")

        c_inventory = st.number_input(
            "Current inventory (C)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_inventory"
        )

        c_backlog = st.number_input(
            "Current backlog (C)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_backlog"
        )

        c_today = st.number_input(
            "Planned demand today (C)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_today"
        )

        c_tomorrow = st.number_input(
            "Planned demand tomorrow (C)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_tomorrow"
        )

        c_pipeline = st.number_input(
            "Known incoming delivery / pipeline (C)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_pipeline"
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Generate delivery recommendation",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 9. GENERATE DQN RECOMMENDATION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Convert human-facing project day to DQN state day
    #
    # Human:
    # Day 1 ... Day 21
    #
    # Training state:
    # 0 ... 20
    # --------------------------------------------------------

    day_state = project_day - 1

    # --------------------------------------------------------
    # Calculate shared storage automatically
    #
    # This reproduces:
    #
    # shared_storage_used =
    #     sum(inventory[p] * storage_footprint[p])
    #
    # shared_storage_utilisation =
    #     shared_storage_used / shared_storage_capacity
    # --------------------------------------------------------

    shared_storage_used = (
        a_inventory * STORAGE_FOOTPRINT["A"]
        + b_inventory * STORAGE_FOOTPRINT["B"]
        + c_inventory * STORAGE_FOOTPRINT["C"]
    )

    shared_storage_utilisation = (
        shared_storage_used
        / SHARED_STORAGE_CAPACITY
    )

    # --------------------------------------------------------
    # Reconstruct EXACT 17-dimensional DQN state
    #
    # [
    #     day,
    #     shared_storage_utilisation,
    #
    #     A_inventory,
    #     A_backlog,
    #     A_planned_today,
    #     A_planned_tomorrow,
    #     A_pipeline,
    #
    #     B_inventory,
    #     B_backlog,
    #     B_planned_today,
    #     B_planned_tomorrow,
    #     B_pipeline,
    #
    #     C_inventory,
    #     C_backlog,
    #     C_planned_today,
    #     C_planned_tomorrow,
    #     C_pipeline
    # ]
    # --------------------------------------------------------

    state = np.array(
        [
            day_state,
            shared_storage_utilisation,

            a_inventory,
            a_backlog,
            a_today,
            a_tomorrow,
            a_pipeline,

            b_inventory,
            b_backlog,
            b_today,
            b_tomorrow,
            b_pipeline,

            c_inventory,
            c_backlog,
            c_today,
            c_tomorrow,
            c_pipeline,
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(state) != STATE_DIM:

        st.error(
            f"State construction error: "
            f"expected {STATE_DIM} values, "
            f"received {len(state)}."
        )

        st.stop()

    # --------------------------------------------------------
    # DQN forward pass
    # --------------------------------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model(
            state_tensor
        )

        action_index = torch.argmax(
            q_values,
            dim=1
        ).item()

    # --------------------------------------------------------
    # Convert action index back into truck allocation
    # --------------------------------------------------------

    recommended_action = joint_actions[
        action_index
    ]

    a_trucks, b_trucks, c_trucks = (
        recommended_action
    )

    # --------------------------------------------------------
    # Corresponding delivery quantities
    # --------------------------------------------------------

    a_quantity = (
        a_trucks * TRUCK_CAPACITY["A"]
    )

    b_quantity = (
        b_trucks * TRUCK_CAPACITY["B"]
    )

    c_quantity = (
        c_trucks * TRUCK_CAPACITY["C"]
    )

    total_trucks = sum(
        recommended_action
    )

    # ========================================================
    # 10. DISPLAY CURRENT STORAGE INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Current site storage"
    )

    storage_col1, storage_col2 = st.columns(2)

    storage_col1.metric(
        "Shared storage currently used",
        f"{shared_storage_used:.1f} m²"
    )

    storage_col2.metric(
        "Storage utilisation",
        f"{shared_storage_utilisation * 100:.1f}%"
    )

    if shared_storage_used > SHARED_STORAGE_CAPACITY:

        st.warning(
            f"Current material inventory exceeds the "
            f"{SHARED_STORAGE_CAPACITY:.0f} m² shared "
            f"storage threshold."
        )

    # ========================================================
    # 11. DISPLAY RECOMMENDED ALLOCATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "3. Recommended truck allocation"
    )

    st.write(
        f"Recommended allocation under the selected "
        f"daily capacity of **{vehicle_limit} trucks**:"
    )

    rec_a, rec_b, rec_c = st.columns(3)

    rec_a.metric(
        "Material A — Rebar",
        f"{a_trucks} truck(s)",
        help=(
            f"Equivalent ordered quantity: "
            f"{a_quantity:.1f} units"
        )
    )

    rec_b.metric(
        "Material B — Formwork",
        f"{b_trucks} truck(s)",
        help=(
            f"Equivalent ordered quantity: "
            f"{b_quantity:.1f} units"
        )
    )

    rec_c.metric(
        "Material C — Mesh / Palletised",
        f"{c_trucks} truck(s)",
        help=(
            f"Equivalent ordered quantity: "
            f"{c_quantity:.1f} units"
        )
    )

    st.success(
        f"Total recommended allocation: "
        f"{total_trucks} of "
        f"{vehicle_limit} available trucks."
    )

    # --------------------------------------------------------
    # Delivery quantity summary
    # --------------------------------------------------------

    st.write(
        "**Corresponding ordered quantities:**"
    )

    st.table(
        {
            "Material": [
                "A — Rebar",
                "B — Formwork",
                "C — Mesh / Palletised"
            ],
            "Recommended trucks": [
                a_trucks,
                b_trucks,
                c_trucks
            ],
            "Quantity": [
                a_quantity,
                b_quantity,
                c_quantity
            ]
        }
    )

    st.caption(
        "The recommendation represents the joint truck-allocation "
        "decision generated by the trained DQN under the model's "
        "one-day replenishment assumption."
    )

    # ========================================================
    # 12. OPTIONAL TECHNICAL INFORMATION
    # ========================================================

    with st.expander(
        "Technical information"
    ):

        st.write(
            f"Selected vehicle limit: V={vehicle_limit}"
        )

        st.write(
            f"Model action dimension: "
            f"{checkpoint['action_dim']}"
        )

        st.write(
            f"Training seed: "
            f"{checkpoint['seed']}"
        )

        st.write(
            f"Selected action index: "
            f"{action_index}"
        )

        st.write(
            "Model-ready 17-dimensional state:"
        )

        st.code(
            np.array2string(
                state,
                precision=3,
                separator=", "
            )
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Prototype decision-support interface for construction "
    "material logistics planning. Recommendations are generated "
    "from separately trained DQN policies for daily truck "
    "capacity limits V=4, V=5 and V=6."
)