import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="RecomLab | Recommender Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL UI THEME
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --primary: #4F46E5;
        --primary-dark: #4338CA;
        --ink: #172033;
        --muted: #64748B;
        --line: #E2E8F0;
        --surface: #FFFFFF;
        --canvas: #F5F7FB;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263247;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #CBD5E1;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: #1F2937;
        border-color: #374151;
        border-radius: 9px;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1F2937;
        border: 1px dashed #4B5563;
        border-radius: 12px;
        padding: 10px;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.45px;
    }

    h1 { font-size: 2.05rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }

    p, li, label {
        color: #334155;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 19px 20px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.85rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.78rem;
    }

    .hero {
        background: linear-gradient(115deg, #111827 0%, #202B50 58%, #4338CA 100%);
        padding: 28px 32px;
        border-radius: 20px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 12px 28px rgba(30, 41, 90, 0.12);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        margin: 5px 0 8px 0;
    }

    .hero p {
        color: #DCE5FF !important;
        margin: 0;
        font-size: 0.98rem;
    }

    .eyebrow {
        color: #C7D2FE;
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.6px;
    }

    .section-kicker {
        color: #64748B;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 1.15px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    .surface-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: 20px 22px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
        margin-bottom: 12px;
    }

    .surface-card h3 {
        margin-top: 0;
    }

    .muted {
        color: #64748B;
        font-size: 0.9rem;
    }

    .pill {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .rank-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: #EEF2FF;
        color: #4338CA;
        font-weight: 800;
        margin-right: 10px;
    }

    .recommendation-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 13px 15px;
        margin: 8px 0;
        gap: 12px;
    }

    .score {
        color: #047857;
        background: #D1FAE5;
        padding: 5px 9px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .stButton > button {
        background: var(--primary);
        color: white;
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 0.55rem 1.05rem;
        font-weight: 700;
        transition: 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        color: white;
        border-color: var(--primary-dark);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 0.15rem #C7D2FE;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        border-radius: 9px 9px 0 0;
        font-weight: 600;
        padding: 10px 16px;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #E2E8F0;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        .hero { padding: 22px 20px; }
        .hero h1 { font-size: 1.55rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MOCK RECOMMENDER ENGINE
# IMPORTANT: This is a demo simulator, not a trained ML model.
# ============================================================
class MockRecSys:
    """Demo recommender simulator. Replace with actual model fitting later."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.unique_users = []
        self.unique_items = []

    def fit(self, df):
        self.unique_users = sorted(df["User_ID"].dropna().unique().tolist(), key=str)
        self.unique_items = sorted(df["Item_ID"].dropna().unique().tolist(), key=str)

    def get_macro_metrics(self):
        # Illustrative values only; these are NOT calculated from the dataset.
        demo_metrics = {
            "Collaborative Filtering": {
                "RMSE": 0.84, "Precision@K": 76.5, "Recall@K": 71.2
            },
            "Matrix Factorization (SVD)": {
                "RMSE": 0.72, "Precision@K": 82.1, "Recall@K": 78.4
            },
            "Neural Matrix Factorization (NeuMF)": {
                "RMSE": 0.65, "Precision@K": 88.9, "Recall@K": 84.1
            },
            "LightGCN / NGCF": {
                "RMSE": 0.59, "Precision@K": 92.3, "Recall@K": 89.7
            },
        }
        return demo_metrics.get(self.model_name, demo_metrics["Collaborative Filtering"])

    def recommend_for_user(self, user_id, top_k=5):
        if not self.unique_items:
            return []
        user_num = "".join(filter(str.isdigit, str(user_id)))
        rng = np.random.default_rng(int(user_num) if user_num else 42)
        item_indices = rng.permutation(len(self.unique_items))[:top_k]
        chosen_items = [self.unique_items[i] for i in item_indices]
        scores = sorted(rng.uniform(0.70, 0.99, len(chosen_items)), reverse=True)
        return list(zip(chosen_items, scores))

    def get_global_top_items(self, original_df, top_k=5):
        stats = original_df.groupby("Item_ID", dropna=True).agg(
            Average_Rating=("Rating", "mean"),
            Total_Interactions=("Rating", "count"),
        ).reset_index()
        stats = stats.sort_values(
            by=["Average_Rating", "Total_Interactions"],
            ascending=[False, False],
        )
        return stats.head(top_k)


# ============================================================
# DATA HELPERS
# ============================================================
REQUIRED_COLUMNS = ["User_ID", "Item_ID", "Rating"]


def make_demo_data(domain):
    rng = np.random.default_rng(42)

    if domain == "University Elective Papers":
        items = ["Advanced AI", "Cloud Computing", "Quantum Physics",
                 "Blockchain", "Cyber Security"]
        users = [f"Student_{i}" for i in rng.integers(1, 30, 180)]
        ratings = rng.integers(3, 6, 180)
    elif domain == "Fashion Recommendation":
        items = ["Denim Jacket", "Running Shoes", "Leather Boots",
                 "Casual T-Shirt", "Summer Dress"]
        users = [f"Shopper_{i}" for i in rng.integers(1, 30, 180)]
        ratings = rng.integers(1, 6, 180)
    elif domain == "Restaurant Selection":
        items = ["Sushi House", "Pizza Garden", "Burger Joint",
                 "Taco Express", "Tandoori Night"]
        users = [f"Diner_{i}" for i in rng.integers(1, 30, 180)]
        ratings = rng.integers(1, 6, 180)
    else:
        items = [f"Item_{i}" for i in range(500, 525)]
        users = [f"User_{i}" for i in rng.integers(100, 130, 180)]
        ratings = rng.integers(1, 6, 180)

    return pd.DataFrame({
        "User_ID": users,
        "Item_ID": rng.choice(items, len(users)),
        "Rating": ratings,
    })


def validate_dataset(data):
    if data is None or data.empty:
        return False, "The dataset is empty."

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        return False, (
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Your CSV must contain User_ID, Item_ID, and Rating columns."
        )

    cleaned = data[REQUIRED_COLUMNS].copy()
    cleaned = cleaned.dropna(subset=["User_ID", "Item_ID", "Rating"])
    cleaned["Rating"] = pd.to_numeric(cleaned["Rating"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Rating"])

    if cleaned.empty:
        return False, "No usable rows remain after removing missing or invalid values."

    if cleaned["User_ID"].nunique() < 2:
        return False, "At least two unique users are required."
    if cleaned["Item_ID"].nunique() < 2:
        return False, "At least two unique items are required."
    if len(cleaned) < 5:
        return False, "Please provide at least five valid interaction rows."

    return True, cleaned.reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 22px 0;">
            <div style="font-size: 1.65rem; font-weight: 800; color: #FFFFFF;">
                🎓 RecomLab
            </div>
            <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 3px;">
                RECOMMENDATION ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ Workspace settings")

    user_profile = st.radio(
        "Experience level",
        ["Novice (Simple Results)", "Expert (Detailed Mechanics)"],
        help="Expert mode includes additional dataset and model configuration details.",
    )

    st.markdown("---")
    st.markdown("### 📁 Dataset")

    domain = st.selectbox(
        "Choose a domain",
        [
            "Custom Upload",
            "University Elective Papers",
            "Fashion Recommendation",
            "Restaurant Selection",
        ],
    )

    uploaded_file = None
    if domain == "Custom Upload":
        uploaded_file = st.file_uploader(
            "Upload interaction data",
            type=["csv"],
            help="Required columns: User_ID, Item_ID, Rating",
        )
        st.caption("CSV format · Maximum upload size depends on your Streamlit configuration.")

    test_size_percent = st.slider(
        "Test split",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        format="%d%%",
        help="Percentage of rows reserved for testing.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Model configuration")
    model_option = st.selectbox(
        "Recommendation algorithm",
        (
            "Collaborative Filtering",
            "Matrix Factorization (SVD)",
            "Neural Matrix Factorization (NeuMF)",
            "LightGCN / NGCF",
        ),
    )

    top_k = st.select_slider(
        "Number of recommendations (K)",
        options=[3, 5, 10],
        value=5,
    )

    run_model = st.button(
        "🚀 Run recommendation demo",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("RecomLab · Academic prototype")
    st.caption("Demo metrics and rankings are simulated.")


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================
data_error = None

if domain == "Custom Upload":
    if uploaded_file is not None:
        try:
            source_df = pd.read_csv(uploaded_file)
            valid, result = validate_dataset(source_df)
            if valid:
                df = result
                data_source_label = uploaded_file.name
            else:
                df = None
                data_error = result
        except Exception as exc:
            df = None
            data_error = f"Could not read the CSV file: {exc}"
    else:
        df = make_demo_data("Custom Upload")
        data_source_label = "Built-in demo dataset"
else:
    df = make_demo_data(domain)
    data_source_label = f"Built-in {domain} demo dataset"

# If a custom CSV is invalid, don't silently train on fallback data.
if data_error:
    st.error(data_error)
    st.info("Correct the CSV columns or upload another file. Required columns: User_ID, Item_ID, Rating.")
    st.stop()

# ============================================================
# HEADER / HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">LEARNING PLATFORM · PHASE 1–3</div>
        <h1>Recommender System Workspace</h1>
        <p>Explore user-item interactions, compare recommendation approaches,
        and inspect personalized and global item rankings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compact dataset status row
status_col1, status_col2, status_col3 = st.columns([1.2, 1, 1])
with status_col1:
    st.markdown(f"**📁 Dataset:** {data_source_label}")
with status_col2:
    st.markdown(f"**👥 Users:** {df['User_ID'].nunique():,}")
with status_col3:
    st.markdown(f"**🧾 Interactions:** {len(df):,}")

# ============================================================
# RUN DEMO ENGINE
# ============================================================
if run_model:
    try:
        if len(df) < 5:
            st.error("The dataset needs at least five valid rows.")
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size_percent / 100.0,
                random_state=42,
            )
            rec_engine = MockRecSys(model_option)
            rec_engine.fit(train_df)

            st.session_state["recomlab_result"] = {
                "engine": rec_engine,
                "train_df": train_df,
                "test_df": test_df,
                "model_option": model_option,
                "domain": domain,
                "test_size_percent": test_size_percent,
                "top_k": top_k,
                "data_source_label": data_source_label,
                "dataset_signature": (
                    len(df),
                    int(df["User_ID"].nunique()),
                    int(df["Item_ID"].nunique()),
                    str(data_source_label),
                ),
            }
            st.session_state["recomlab_notice"] = "Recommendation demo prepared successfully."
    except ValueError as exc:
        st.error(f"Could not split the dataset: {exc}")

result_state = st.session_state.get("recomlab_result")

# Mark old result as stale if key settings/dataset changed
current_signature = (
    len(df),
    int(df["User_ID"].nunique()),
    int(df["Item_ID"].nunique()),
    str(data_source_label),
)
result_is_current = (
    result_state is not None
    and result_state["dataset_signature"] == current_signature
    and result_state["model_option"] == model_option
    and result_state["test_size_percent"] == test_size_percent
    and result_state["top_k"] == top_k
    and result_state["domain"] == domain
)

if result_state is not None and not result_is_current:
    st.warning("Your settings or dataset have changed. Run the demo again to refresh the results.")

if st.session_state.get("recomlab_notice") and result_is_current:
    st.success(st.session_state["recomlab_notice"])

# ============================================================
# NAVIGATION TABS
# ============================================================
overview_tab, evaluation_tab, recommendations_tab, learning_tab = st.tabs(
    ["📊 Overview", "📈 Evaluation", "✨ Recommendations", "📘 Learn"]
)

# ============================================================
# OVERVIEW TAB
# ============================================================
with overview_tab:
    st.markdown('<div class="section-kicker">WORKSPACE OVERVIEW</div>', unsafe_allow_html=True)
    st.header("Your recommendation workspace")

    if result_is_current:
        engine = result_state["engine"]
        metrics = engine.get_macro_metrics()

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Illustrative demo value, not computed from test predictions.")
        metric2.metric("Precision@K", f"{metrics['Precision@K']:.1f}%")
        metric3.metric("Recall@K", f"{metrics['Recall@K']:.1f}%")

        st.caption("⚠️ These are illustrative preset metrics. This prototype does not train the selected algorithm or calculate these metrics from held-out predictions.")

        st.markdown("### Dataset snapshot")
        snap1, snap2, snap3, snap4 = st.columns(4)
        snap1.metric("Total rows", f"{len(df):,}")
        snap2.metric("Unique users", f"{df['User_ID'].nunique():,}")
        snap3.metric("Unique items", f"{df['Item_ID'].nunique():,}")
        snap4.metric("Train / test", f"{len(result_state['train_df']):,} / {len(result_state['test_df']):,}")

        with st.container(border=True):
            st.markdown("#### Preview interaction data")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="surface-card">
                <span class="pill">GETTING STARTED</span>
                <h3 style="margin-top: 14px;">Run your first recommendation demo</h3>
                <p class="muted">1. Select a domain or upload a CSV in the sidebar.</p>
                <p class="muted">2. Choose an algorithm and test split.</p>
                <p class="muted">3. Click <b>Run recommendation demo</b> to explore the results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Dataset preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# ============================================================
# EVALUATION TAB
# ============================================================
with evaluation_tab:
    st.markdown('<div class="section-kicker">MODEL INSIGHTS</div>', unsafe_allow_html=True)
    st.header("Model evaluation")

    if result_is_current:
        engine = result_state["engine"]
        metrics = engine.get_macro_metrics()

        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### Current demo configuration")
            st.write(f"**Algorithm:** {result_state['model_option']}")
            st.write(f"**Domain:** {result_state['domain']}")
            st.write(f"**Training rows:** {len(result_state['train_df']):,}")
            st.write(f"**Test rows:** {len(result_state['test_df']):,}")
            st.write(f"**Test split:** {result_state['test_size_percent']}%")
            st.write(f"**K:** {result_state['top_k']}")
        with right:
            st.markdown("#### Illustrative metric values")
            metric_df = pd.DataFrame({
                "Metric": ["RMSE", "Precision@K", "Recall@K"],
                "Value": [
                    metrics["RMSE"],
                    metrics["Precision@K"],
                    metrics["Recall@K"],
                ],
            })
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
            st.caption("Do not use these values as experimental results in a report or publication.")

        if user_profile == "Expert (Detailed Mechanics)":
            st.markdown("#### Technical details")
            st.code(
                f"Model label: {result_state['model_option']}\n"
                f"Training rows: {len(result_state['train_df'])}\n"
                f"Test rows: {len(result_state['test_df'])}\n"
                f"Split strategy: random row split (random_state=42)\n"
                f"Engine type: MockRecSys (simulated)",
                language="text",
            )
            st.info("For a genuine evaluation, implement model.fit(), predict() on test rows, and compute RMSE from actual predictions. Precision@K and Recall@K require a defined relevance threshold and ranking protocol.")
    else:
        st.info("Run the recommendation demo from the sidebar to view evaluation details.")

# ============================================================
# RECOMMENDATIONS TAB
# ============================================================
with recommendations_tab:
    st.markdown('<div class="section-kicker">DISCOVERY</div>', unsafe_allow_html=True)
    st.header("Recommendation explorer")
    st.write("Inspect a user's ranked items or explore highly rated and popular items globally.")

    if result_is_current:
        engine = result_state["engine"]
        train_df = result_state["train_df"]

        perspective = st.radio(
            "Recommendation perspective",
            ["Personalized for a user", "Global popular items"],
            horizontal=True,
        )

        if perspective == "Personalized for a user":
            available_users = engine.unique_users
            if not available_users:
                st.warning("No users are available in the training data.")
            else:
                user_col, count_col = st.columns([2, 1])
                with user_col:
                    selected_user = st.selectbox(
                        "Select target user",
                        available_users,
                        format_func=str,
                    )
                with count_col:
                    st.markdown("**Recommendations**")
                    st.markdown(f'<span class="pill">Top {result_state["top_k"]}</span>', unsafe_allow_html=True)

                recommendations = engine.recommend_for_user(
                    selected_user,
                    top_k=result_state["top_k"],
                )

                st.markdown(
                    f'<div class="surface-card"><span class="pill">PERSONALIZED VIEW</span>'
                    f'<h3 style="margin-top: 12px;">Top picks for {selected_user}</h3>'
                    f'<p class="muted">Illustrative ranking scores generated by the demo simulator.</p></div>',
                    unsafe_allow_html=True,
                )

                if recommendations:
                    for rank, (item, score) in enumerate(recommendations, start=1):
                        st.markdown(
                            f'<div class="recommendation-row">'
                            f'<div><span class="rank-number">{rank}</span>'
                            f'<strong>{item}</strong></div>'
                            f'<span class="score">Match score · {score:.2f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No recommendations are available for this user.")
        else:
            global_top = engine.get_global_top_items(
                train_df,
                top_k=result_state["top_k"],
            )
            st.markdown("#### Highest-rated items in the training data")
            if global_top.empty:
                st.info("No item interactions are available.")
            else:
                display_top = global_top.rename(columns={
                    "Item_ID": "Item",
                    "Average_Rating": "Average rating",
                    "Total_Interactions": "Interactions",
                }).copy()
                display_top["Average rating"] = display_top["Average rating"].round(2)
                display_top.insert(0, "Rank", range(1, len(display_top) + 1))
                st.dataframe(display_top, use_container_width=True, hide_index=True)
                st.caption("Global ranking is calculated from training-data average ratings, with interaction count as a tie-breaker.")
    else:
        st.info("Run the recommendation demo to explore personalized and global results.")

# ============================================================
# LEARNING TAB
# ============================================================
with learning_tab:
    st.markdown('<div class="section-kicker">LEARNING CENTER</div>', unsafe_allow_html=True)
    st.header("Understand the recommendation workflow")

    with st.expander("1. What is the user-item interaction dataset?", expanded=True):
        st.write(
            "Each row represents an interaction between a user and an item. "
            "User_ID identifies the user, Item_ID identifies the item, and Rating "
            "records the user's rating or preference signal."
        )
        st.code("User_ID,Item_ID,Rating\nUser_1,Item_101,5\nUser_1,Item_104,3\nUser_2,Item_101,4", language="csv")

    with st.expander("2. What do the metrics mean?"):
        st.markdown(
            """
            - **RMSE:** Measures the typical magnitude of rating-prediction errors. Lower is better.
            - **Precision@K:** Fraction of the top-K recommended items that are relevant.
            - **Recall@K:** Fraction of all relevant items that appear in the top-K recommendations.

            Precision@K and Recall@K require a clearly defined relevance threshold and evaluation protocol.
            """
        )

    with st.expander("3. What do the algorithms represent?"):
        st.markdown(
            """
            - **Collaborative Filtering:** Uses patterns in user-item interactions to find similar users or items.
            - **Matrix Factorization (SVD):** Represents users and items in a lower-dimensional latent-factor space.
            - **NeuMF:** Combines neural interaction learning with matrix-factorization-style representations.
            - **LightGCN / NGCF:** Use graph-based propagation over user-item interaction graphs.

            The current application only selects algorithm labels; it does not implement or train these algorithms.
            """
        )

    with st.expander("4. CSV data requirements"):
        st.write("Your CSV must contain these exact column names:")
        st.dataframe(
            pd.DataFrame({
                "Column": ["User_ID", "Item_ID", "Rating"],
                "Purpose": [
                    "Unique or repeated user identifier",
                    "Unique or repeated item identifier",
                    "Numeric rating or interaction score",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.caption("RecomLab · Learning Platform for Recommender Systems")
with footer_right:
    st.caption("Academic prototype · Demo engine")
