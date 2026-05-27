# ==========================================================
# 🌍 GLOBAL DEVELOPMENT CLUSTER ANALYZER
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Global Development Cluster Analyzer",
    page_icon="🌍",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00FFAA;
    text-align: center;
}

h2 {
    color: #00C4FF;
}

.stButton>button {
    background-color: #00C4FF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>

""", unsafe_allow_html=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(
    "World_development_mesurement.xlsx"
)

# ==========================================================
# LOAD MODELS
# ==========================================================

model = pickle.load(
    open('clustering_model.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

pca = pickle.load(
    open('pca.pkl', 'rb')
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌍 Navigation")

menu = st.sidebar.radio(

    "Go To",

    [
        "Home",
        "Dataset Overview",
        "Cluster Visualization",
        "Country Cluster Search",
        "Cluster Insights"
    ]

)


# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "Home":

    st.title("🌍 Global Development Cluster Analyzer")
    st.toast(
    "Dashboard Loaded Successfully 🚀"
)    
    st.image(
    "https://images.unsplash.com/photo-1521295121783-8a321d551ad2",
    use_column_width=True
    )
    st.balloons()
    st.markdown("""

    ## 📌 Project Objective

    This project groups countries into meaningful
    development clusters using Machine Learning.

    ### 🔥 Algorithms Used
    - KMeans Clustering
    - Hierarchical Clustering
    - DBSCAN

    ### 📊 Features Used
    - GDP Per Capita
    - Digital Development
    - Health Efficiency
    - CO2 Efficiency
    - Birth Rate
    - Energy Usage

    """)


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

elif menu == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        df.select_dtypes(include=np.number).corr(),
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)


# ==========================================================
# CLUSTER VISUALIZATION
# ==========================================================

elif menu == "Cluster Visualization":

    st.title("🌍 Country Cluster Visualization")
    
    fig = px.scatter(

        df,
        x='PC1',
        y='PC2',
        color='KMeans_Cluster',
        hover_name='Country',

        title='KMeans Country Clusters',

        color_continuous_scale='viridis'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# COUNTRY SEARCH
# ==========================================================

elif menu == "Country Cluster Search":

    st.title("🔍 Country Cluster Search")
    
    country = st.selectbox(

        "Select Country",

        sorted(df['Country'].unique())

    )

    result = df[
        df['Country'] == country
    ]

    st.subheader("Country Information")

    st.write(result)


# ==========================================================
# CLUSTER INSIGHTS
# ==========================================================

elif menu == "Cluster Insights":

    st.title("🧠 Cluster Insights")
    
    cluster_profile = (

        df
        .groupby('KMeans_Cluster')
        .mean(numeric_only=True)

    )

    st.dataframe(cluster_profile)

    st.markdown("""

    ## 🌍 Cluster Interpretation

    ### Cluster 0
    Developed Economies

    ### Cluster 1
    Emerging Economies

    ### Cluster 2
    Underdeveloped Economies

    """)
