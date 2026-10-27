# ============================================================
# Centralised Multi-Material Construction Logistics Decision Support
# Deep Q-Network (DQN) Interface
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
    page_title="Centralised Multi-Material Construction Logistics Decision Support",
    page_icon="🏗️",
    layout="wide"
)

# ============================================================
# 2. MODEL / CASE CONSTANTS
# ============================================================

STATE_DIM = 17

SHARED_STORAGE_CAPACITY = 20.0  # m²

STORAGE_FOOTPRINT = {
    "A": 1.0,
    "B": 1.5,
    "C": 1.2,
}

TRUCK_CAPACITY = {
    "A": 3.0,
    "B": 2.0,
    "C": 4.0,
}

APP_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    4: APP_DIR / "dqn_V4_seed_42.pth",
    5: APP_DIR / "dqn_V5_seed_42.pth",
    6: APP_DIR / "dqn_V6_seed_42.pth",
}


# ============================================================
# 3. DQN NETWORK
# Must exactly match training architecture
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
# Must exactly match training environment
# ============================================================

def build_joint_action_space(vehicle_limit):

    joint_actions = []

    for a in range(vehicle_limit + 1):
        for b in range(vehicle_limit + 1):
            for c in range(vehicle_limit + 1):

                if a + b + c <= vehicle_limit:
                    joint_actions.append((a, b, c))

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
# 6. HEADER
# ============================================================

st.title("Centralised Multi-Material Construction Logistics Decision Support")

st.caption(
    "This prototype uses a centralised Deep Q-Network (DQN) to recommend "
    "coordinated daily truck allocations for three construction materials "
    "under shared delivery-capacity and storage constraints."
)


# ============================================================
# 7. SITE DELIVERY CAPACITY
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

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# Make selected capacity visually prominent
# ------------------------------------------------------------

capacity_col1, capacity_col2 = st.columns(2)

capacity_col1.metric(
    "Selected daily truck capacity",
    f"{vehicle_limit} trucks/day"
)

capacity_col2.metric(
    "Available DQN action space",
    f"{len(joint_actions)} feasible allocations"
)

st.caption(
    f"Using the trained DQN policy for V={vehicle_limit}."
)


# ============================================================
# 8. OPERATIONAL INPUT FORM
# ============================================================

st.subheader(
    "2. Current project and site information"
)

st.caption(
    "Enter information available from the current programme, "
    "inventory records and delivery information."
)

with st.form("logistics_input_form"):

    # --------------------------------------------------------
    # Project day
    # Human-facing day = 1-21
    # DQN internal state = 0-20
    # --------------------------------------------------------

    project_day = st.number_input(
        "Current project day",
        min_value=1,
        max_value=21,
        value=1,
        step=1,
        help="Day within the 21-day planning horizon."
    )

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    # ========================================================
    # MATERIAL A
    # ========================================================

    with col_a:

        st.markdown("### Material A — Rebar")

        a_inventory = st.number_input(
            "Current inventory",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_inventory"
        )

        a_backlog = st.number_input(
            "Current backlog",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_backlog"
        )

        a_today = st.number_input(
            "Planned demand today",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_today"
        )

        a_tomorrow = st.number_input(
            "Planned demand tomorrow",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="a_tomorrow"
        )

        a_pipeline = st.number_input(
            "Incoming delivery",
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
            "Current inventory",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_inventory"
        )

        b_backlog = st.number_input(
            "Current backlog",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_backlog"
        )

        b_today = st.number_input(
            "Planned demand today",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_today"
        )

        b_tomorrow = st.number_input(
            "Planned demand tomorrow",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_tomorrow"
        )

        b_pipeline = st.number_input(
            "Incoming delivery",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="b_pipeline"
        )

    # ========================================================
    # MATERIAL C
    # ========================================================

    with col_c:

        st.markdown(
            "### Material C — Mesh / Palletised"
        )

        c_inventory = st.number_input(
            "Current inventory",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_inventory"
        )

        c_backlog = st.number_input(
            "Current backlog",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_backlog"
        )

        c_today = st.number_input(
            "Planned demand today",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_today"
        )

        c_tomorrow = st.number_input(
            "Planned demand tomorrow",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="c_tomorrow"
        )

        c_pipeline = st.number_input(
            "Incoming delivery",
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
    # Convert human-facing day to training state
    # --------------------------------------------------------

    day_state = project_day - 1

    # --------------------------------------------------------
    # Calculate shared storage automatically
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
    # Exact 17-dimensional state used during training
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
    # Convert action index to truck allocation
    # --------------------------------------------------------

    recommended_action = joint_actions[
        action_index
    ]

    a_trucks, b_trucks, c_trucks = (
        recommended_action
    )

    # --------------------------------------------------------
    # Convert trucks to material quantities
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
    # 10. CURRENT SITE STORAGE
    # ========================================================

    st.markdown("---")

    storage_col1, storage_col2 = st.columns(2)

    storage_col1.metric(
        "Current shared storage",
        f"{shared_storage_used:.1f} m²"
    )

    storage_col2.metric(
        "Storage utilisation",
        f"{shared_storage_utilisation * 100:.1f}%"
    )

    if shared_storage_used > SHARED_STORAGE_CAPACITY:

        st.warning(
            f"Current material inventory exceeds the "
            f"{SHARED_STORAGE_CAPACITY:.0f} m² "
            f"shared-storage threshold."
        )


    # ========================================================
    # 11. RECOMMENDED DELIVERY ALLOCATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "3. Recommended delivery allocation"
    )

    st.caption(
        "DQN recommendation based on the current operational state."
    )

    rec_a, rec_b, rec_c = st.columns(3)

    with rec_a:

        st.metric(
            "Rebar — Material A",
            f"{a_trucks} truck(s)"
        )

        st.caption(
            f"Ordered quantity: "
            f"{a_quantity:.1f} tonnes"
        )

    with rec_b:

        st.metric(
            "Formwork — Material B",
            f"{b_trucks} truck(s)"
        )

        st.caption(
            f"Ordered quantity: "
            f"{b_quantity:.1f} sets"
        )

    with rec_c:

        st.metric(
            "Mesh / Palletised — Material C",
            f"{c_trucks} truck(s)"
        )

        st.caption(
            f"Ordered quantity: "
            f"{c_quantity:.1f} units"
        )

    st.success(
        f"Recommended total allocation: "
        f"{total_trucks} of "
        f"{vehicle_limit} available trucks."
    )

    st.caption(
        "The recommendation represents the coordinated "
        "joint truck-allocation decision generated by the "
        "trained DQN under the model's one-day replenishment "
        "assumption."
    )


    # ========================================================
    # 12. TECHNICAL INFORMATION
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
