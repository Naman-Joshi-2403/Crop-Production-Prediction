import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
from dotenv import load_dotenv
import os

login_user = os.path.expanduser("~")
load_dotenv("dev.env")
clean_csv_output_path = os.getenv("OUTPUT_DATASET_PATH")


st.set_page_config(page_title="Crop Production EDA", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(login_user, clean_csv_output_path))  # Update path if needed
    return df

df = load_data()

with st.sidebar:
    selected = option_menu(
        menu_title="EDA Dashboard",
        options=[
            "Overview",
            "Area Analysis",
            "Crop Analysis",
            "Yearly Trends",
            "Correlation",
            "Summary"
        ],
        icons=["bar-chart", "globe", "flower1", "graph-up", "activity","file-text"],
        default_index=0
    )


if selected == "Overview":
    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Unique Areas", df["area"].nunique())
    col3.metric("Unique Crops", df["item"].nunique())

    st.subheader("Sample Data")
    st.dataframe(df ,use_container_width=True, hide_index=True)


elif selected == "Area Analysis":
    st.title("🌍 Area-wise Production Analysis")

    area_prod = (
        df.groupby("area")["production_tons"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.subheader("Top 15 Areas by Total Production")
    st.bar_chart(area_prod)


elif selected == "Crop Analysis":
    st.title("🌾 Crop-wise Production Analysis")

    crop_prod = (
        df.groupby("item")["production_tons"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.subheader("Top 15 Crops by Production")
    st.bar_chart(crop_prod)


elif selected == "Yearly Trends":
    st.title("📈 Production Trend Over Years")

    selected_crop = st.selectbox(
        "Select Crop",
        sorted(df["item"].unique())
    )

    trend_df = (
        df[df["item"] == selected_crop]
        .groupby("year")["production_tons"]
        .sum()
    )

    st.line_chart(trend_df)


elif selected == "Correlation":
    st.title("🔍 Correlation Heatmap")

    numeric_cols = [
        "area_harvested_ha",
        "yield_kg_per_ha",
        "production_tons"
    ]

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )

    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )

    st.plotly_chart(fig, use_container_width=True)


elif selected == "Summary":

    st.title("📌 EDA Summary & Key Insights")

    st.markdown("""
### 🌾 Overall Findings

- Crop production is mainly influenced by **area harvested** and **yield**.
- Some crops contribute significantly more to total production than others.
- A few regions dominate agricultural output.

---

### 🌍 Area-wise Insights

- Certain areas consistently show high production levels.
- Regional productivity differs due to land size and crop selection.
- Some regions have high yield even with smaller land area.

---

### 🌾 Crop-wise Insights

- High-yield crops generate more production per hectare.
- Some crops are widely cultivated but produce moderate output.
- Production varies significantly across crop types.

---

### 📈 Yearly Trends

- Crop production shows variation across years.
- Some crops show increasing trends over time.
- External factors like climate and policies may influence yearly changes.

---

### 🔍 Correlation Insights

- Production has a strong positive relationship with area harvested.
- Yield also positively impacts production.
- Both land usage and efficiency are important for higher output.

---

### 🤖 Recommended Machine Learning Model

After analyzing the relationships between features, we observed:

- Production has strong positive correlation with area harvested.
- Yield also significantly impacts total production.
- Relationships between variables are mostly linear.

#### ✅ Best Suitable Model: Linear Regression

**Why Linear Regression?**

- The target variable (production) is continuous.
- Strong linear relationship exists between features and target.
- It is simple, interpretable, and performs well for regression problems.
- Easy to explain to stakeholders and policymakers.

---

#### 🔎 Alternative Models for Better Accuracy

- **Random Forest Regressor** → Handles non-linearity and complex interactions.
- **XGBoost Regressor** → Provides higher accuracy for large datasets.
- **Decision Tree Regressor** → Easy to visualize but may overfit.

---

### 🎯 Final Recommendation

Start with **Linear Regression** for interpretability and baseline performance.
Then compare with **Random Forest or XGBoost** to improve accuracy.
""")