import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "amazon_music_final_clusters.csv"
    )

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "amazon_music_final_clusters.csv was not found. "
        "Keep the CSV file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# FEATURE LIST
# ============================================================

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]


available_features = [
    feature
    for feature in features
    if feature in df.columns
]


# ============================================================
# FIND CLUSTER COLUMN
# ============================================================

cluster_column = None

possible_cluster_columns = [
    "kmeans_cluster",
    "Cluster",
    "cluster",
    "cluster_label",
    "kmeans_labels"
]

for column in possible_cluster_columns:

    if column in df.columns:

        cluster_column = column
        break


if cluster_column is None:

    st.error(
        "K-Means cluster column was not found."
    )

    st.stop()


df[cluster_column] = pd.to_numeric(
    df[cluster_column],
    errors="coerce"
)


# ============================================================
# FIND CLUSTER NAME COLUMN
# ============================================================

cluster_name_column = None

possible_name_columns = [
    "Cluster_Name",
    "cluster_name",
    "Cluster Name"
]

for column in possible_name_columns:

    if column in df.columns:

        cluster_name_column = column
        break


# ============================================================
# CLUSTER NAME MAPPING
# ============================================================

default_cluster_names = {

    0: "Spoken & Story Content",

    1: "Calm / Classic Music",

    2: "Energetic / Pop-Oriented Music"
}


def get_cluster_name(cluster):

    if cluster_name_column is not None:

        values = df.loc[
            df[cluster_column] == cluster,
            cluster_name_column
        ].dropna()

        if len(values) > 0:

            return str(values.iloc[0])

    return default_cluster_names.get(
        int(cluster),
        f"Cluster {int(cluster)}"
    )


# ============================================================
# COMMON VALUES
# ============================================================

clusters = sorted(
    df[cluster_column]
    .dropna()
    .unique()
)

total_songs = len(df)

number_of_clusters = len(clusters)

# Final K-Means result
final_silhouette = 0.2465
final_inertia = 658335.08


# ============================================================
# ELBOW METHOD DATA
# ============================================================

k_values = list(range(2, 11))

inertia_values = [
    84084.55626590706,
    71868.01805893448,
    62045.27172780161,
    56678.519918066275,
    52246.993847888676,
    48445.048768041335,
    45836.647599769465,
    43523.87533092957,
    41600.73389642952
]


# ============================================================
# SILHOUETTE DATA
# ============================================================

silhouette_values = [
    0.2086,
    0.2465,
    0.2335,
    0.2045,
    0.1616,
    0.2049,
    0.1713,
    0.1722,
    0.1783
]


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎵 Amazon Music Clustering'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Unsupervised Learning for Audio-Based Song Grouping'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎵 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Cluster Overview",
        "Model Evaluation",
        "Cluster Analysis",
        "PCA Visualization",
        "Song Explorer",
        "Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Amazon Music Clustering"
)

st.sidebar.caption(
    "K-Means | K = 3"
)

st.sidebar.caption(
    "Silhouette Score = 0.2465"
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="section-title">'
        'Project Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project uses **K-Means clustering** to automatically
        group songs based on their audio characteristics.

        Genre was **not used as an input feature** during clustering.
        Instead, audio features such as danceability, energy,
        acousticness, speechiness, tempo and valence were used to
        discover natural groups in the dataset.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Songs",
        f"{total_songs:,}"
    )

    col2.metric(
        "Clusters",
        number_of_clusters
    )

    col3.metric(
        "Silhouette Score",
        f"{final_silhouette:.4f}"
    )

    col4.metric(
        "Final Inertia",
        f"{final_inertia:,.2f}"
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Project Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        **Data Exploration → Data Cleaning → Feature Selection →
        StandardScaler → K-Means → Elbow Method → Silhouette Score →
        PCA → Cluster Interpretation → Business Insights**
        """
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Final Clusters'
        '</div>',
        unsafe_allow_html=True
    )

    for cluster in clusters:

        count = (
            df[cluster_column] == cluster
        ).sum()

        st.write(
            f"**Cluster {int(cluster)} — "
            f"{get_cluster_name(cluster)}:** "
            f"{count:,} songs"
        )


# ============================================================
# CLUSTER OVERVIEW
# ============================================================

elif page == "Cluster Overview":

    st.markdown(
        '<div class="section-title">'
        '📊 Cluster Overview'
        '</div>',
        unsafe_allow_html=True
    )

    cluster_counts = (
        df[cluster_column]
        .value_counts()
        .sort_index()
    )

    overview_data = []

    for cluster in clusters:

        count = cluster_counts.get(
            cluster,
            0
        )

        percentage = (
            count / total_songs
        ) * 100

        overview_data.append(
            {
                "Cluster": int(cluster),

                "Cluster Name":
                    get_cluster_name(cluster),

                "Songs":
                    int(count),

                "Percentage":
                    round(
                        percentage,
                        2
                    )
            }
        )

    overview_df = pd.DataFrame(
        overview_data
    )

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Cluster Size"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        overview_df["Cluster Name"],
        overview_df["Songs"]
    )

    ax.set_xlabel(
        "Cluster"
    )

    ax.set_ylabel(
        "Number of Songs"
    )

    ax.set_title(
        "Songs Distribution Across Clusters"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.markdown(
        '<div class="section-title">'
        '📈 Model Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The **Elbow Method** and **Silhouette Score** were used
        to evaluate the appropriate number of clusters for
        K-Means clustering.
        """
    )

    # ========================================================
    # ELBOW METHOD
    # ========================================================

    st.subheader(
        "1. Elbow Method"
    )

    elbow_df = pd.DataFrame(
        {
            "K": k_values,
            "Inertia": inertia_values
        }
    )

    st.dataframe(
        elbow_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    # Highlight K=3
    k3_index = k_values.index(3)

    ax.scatter(
        3,
        inertia_values[k3_index],
        s=100,
        zorder=5
    )

    ax.annotate(
        "Selected K = 3",
        (
            3,
            inertia_values[k3_index]
        ),
        xytext=(
            4,
            inertia_values[k3_index] + 5000
        ),
        arrowprops=dict(
            arrowstyle="->"
        )
    )

    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_title(
        "Elbow Method"
    )

    ax.set_xticks(
        k_values
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        """
        The Elbow Method evaluates how inertia decreases as
        the number of clusters increases. K = 3 was selected
        for the final K-Means model based on the overall
        evaluation and project requirement.
        """
    )

    st.divider()

    # ========================================================
    # SILHOUETTE SCORE
    # ========================================================

    st.subheader(
        "2. Silhouette Score"
    )

    silhouette_df = pd.DataFrame(
        {
            "K": k_values,
            "Silhouette Score":
                silhouette_values
        }
    )

    st.dataframe(
        silhouette_df,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    # Highlight K=3
    ax.scatter(
        3,
        0.2465,
        s=100,
        zorder=5
    )

    ax.annotate(
        "Best K = 3",
        (
            3,
            0.2465
        ),
        xytext=(
            4,
            0.255
        ),
        arrowprops=dict(
            arrowstyle="->"
        )
    )

    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    ax.set_title(
        "Silhouette Score vs Number of Clusters"
    )

    ax.set_xticks(
        k_values
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.success(
        """
        K = 3 produced the highest Silhouette Score
        of 0.2465 among the evaluated K values.
        """
    )

    st.divider()

    # ========================================================
    # FINAL MODEL
    # ========================================================

    st.subheader(
        "3. Final K-Means Model"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected K",
        "3"
    )

    col2.metric(
        "Silhouette Score",
        "0.2465"
    )

    col3.metric(
        "n_init",
        "10"
    )

    st.write(
        """
        **Final configuration:**

        - Algorithm: K-Means
        - Number of clusters: 3
        - Random state: 42
        - n_init: 10
        - Silhouette Score: 0.2465
        """
    )


# ============================================================
# CLUSTER ANALYSIS
# ============================================================

elif page == "Cluster Analysis":

    st.markdown(
        '<div class="section-title">'
        '🔍 Cluster Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    if not available_features:

        st.warning(
            "Audio feature columns were not found."
        )

    else:

        # ====================================================
        # RAW CLUSTER PROFILE
        # ====================================================

        st.subheader(
            "Average Audio Features by Cluster"
        )

        cluster_profile = (
            df.groupby(
                cluster_column
            )[available_features]
            .mean()
            .round(3)
        )

        display_profile = (
            cluster_profile.copy()
        )

        display_profile.index = [
            get_cluster_name(cluster)
            for cluster
            in display_profile.index
        ]

        st.dataframe(
            display_profile,
            use_container_width=True
        )

        st.divider()

        # ====================================================
        # STANDARDIZED PROFILE
        # ====================================================

        st.subheader(
            "Standardized Cluster Comparison"
        )

        st.info(
            """
            Standardization is used for visualization so
            features with different scales, such as duration_ms
            and tempo, do not dominate the chart.
            """
        )

        profile_scaled = (
            cluster_profile
            - cluster_profile.mean()
        ) / cluster_profile.std()

        profile_chart = (
            profile_scaled.T
        )

        profile_chart.columns = [
            get_cluster_name(cluster)
            for cluster
            in profile_chart.columns
        ]

        # ====================================================
        # ALL CLUSTERS CHART
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        profile_chart.plot(
            kind="bar",
            ax=ax
        )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.set_xlabel(
            "Audio Feature"
        )

        ax.set_ylabel(
            "Standardized Value"
        )

        ax.set_title(
            "Standardized Audio Feature Comparison"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        ax.legend(
            title="Cluster",
            bbox_to_anchor=(
                1.02,
                1
            ),
            loc="upper left"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ====================================================
        # SELECT CLUSTER
        # ====================================================

        st.subheader(
            "Individual Cluster Analysis"
        )

        selected_cluster = st.selectbox(
            "Select Cluster",
            clusters,
            format_func=lambda x:
                f"Cluster {int(x)} — "
                f"{get_cluster_name(x)}"
        )

        selected_data = df[
            df[cluster_column]
            == selected_cluster
        ]

        # ====================================================
        # KPIs
        # ====================================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Songs",
            f"{len(selected_data):,}"
        )

        if "energy" in selected_data.columns:

            col2.metric(
                "Average Energy",
                f"{selected_data['energy'].mean():.3f}"
            )

        if "danceability" in selected_data.columns:

            col3.metric(
                "Average Danceability",
                f"{selected_data['danceability'].mean():.3f}"
            )

        # ====================================================
        # SELECTED PROFILE
        # ====================================================

        selected_profile = (
            profile_scaled.loc[
                selected_cluster
            ]
            .sort_values(
                ascending=False
            )
        )

        st.subheader(
            "Selected Cluster Profile"
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.bar(
            selected_profile.index,
            selected_profile.values
        )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.set_xlabel(
            "Audio Feature"
        )

        ax.set_ylabel(
            "Standardized Value"
        )

        ax.set_title(
            f"Audio Profile — "
            f"{get_cluster_name(selected_cluster)}"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# PCA VISUALIZATION
# ============================================================

elif page == "PCA Visualization":

    st.markdown(
        '<div class="section-title">'
        '📈 PCA Visualization'
        '</div>',
        unsafe_allow_html=True
    )

    if len(available_features) < 2:

        st.warning(
            "Not enough features for PCA."
        )

    else:

        with st.spinner(
            "Creating PCA visualization..."
        ):

            X = df[
                available_features
            ].copy()

            valid_mask = (
                X.notna().all(axis=1)
            )

            X_valid = X.loc[
                valid_mask
            ]

            cluster_valid = df.loc[
                valid_mask,
                cluster_column
            ]

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(
                X_valid
            )

            pca = PCA(
                n_components=2,
                random_state=42
            )

            X_pca = pca.fit_transform(
                X_scaled
            )

        explained_1 = (
            pca.explained_variance_ratio_[0]
            * 100
        )

        explained_2 = (
            pca.explained_variance_ratio_[1]
            * 100
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "PC1 Explained Variance",
            f"{explained_1:.2f}%"
        )

        col2.metric(
            "PC2 Explained Variance",
            f"{explained_2:.2f}%"
        )

        pca_df = pd.DataFrame(
            {
                "PC1":
                    X_pca[:, 0],

                "PC2":
                    X_pca[:, 1],

                "Cluster":
                    cluster_valid.values
            }
        )

        max_points = 15000

        if len(pca_df) > max_points:

            pca_plot = pca_df.sample(
                max_points,
                random_state=42
            )

        else:

            pca_plot = pca_df

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        for cluster in clusters:

            points = pca_plot[
                pca_plot["Cluster"]
                == cluster
            ]

            ax.scatter(
                points["PC1"],
                points["PC2"],
                label=(
                    f"Cluster {int(cluster)} — "
                    f"{get_cluster_name(cluster)}"
                ),
                alpha=0.5,
                s=12
            )

        ax.set_xlabel(
            f"PC1 ({explained_1:.2f}%)"
        )

        ax.set_ylabel(
            f"PC2 ({explained_2:.2f}%)"
        )

        ax.set_title(
            "2D PCA Visualization of Song Clusters"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            """
            PCA is used only for visualization.
            K-Means clustering was performed using
            the standardized audio features.
            """
        )


# ============================================================
# SONG EXPLORER
# ============================================================

elif page == "Song Explorer":

    st.markdown(
        '<div class="section-title">'
        '🎧 Song Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    selected_cluster = st.selectbox(
        "Choose a Cluster",
        clusters,
        format_func=lambda x:
            f"Cluster {int(x)} — "
            f"{get_cluster_name(x)}"
    )

    selected_data = df[
        df[cluster_column]
        == selected_cluster
    ].copy()

    st.write(
        f"**{len(selected_data):,} songs** "
        f"found in "
        f"**{get_cluster_name(selected_cluster)}**."
    )

    # ========================================================
    # SEARCH
    # ========================================================

    search_text = st.text_input(
        "Search by song or artist"
    )

    if search_text:

        searchable_columns = []

        for column in [
            "track_name",
            "track",
            "song_name",
            "name",
            "artist_name",
            "artist"
        ]:

            if column in selected_data.columns:

                searchable_columns.append(
                    column
                )

        if searchable_columns:

            mask = pd.Series(
                False,
                index=selected_data.index
            )

            for column in searchable_columns:

                mask = (
                    mask
                    |
                    selected_data[column]
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                )

            selected_data = selected_data[
                mask
            ]

    # ========================================================
    # DISPLAY
    # ========================================================

    preferred_columns = [
        "track_name",
        "track",
        "song_name",
        "name",
        "artist_name",
        "artist",
        "genre",
        cluster_column
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in selected_data.columns
    ]

    for feature in available_features:

        if feature not in display_columns:

            display_columns.append(
                feature
            )

    display_columns = display_columns[:12]

    if display_columns:

        st.dataframe(
            selected_data[
                display_columns
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Showing up to 100 songs."
        )

    else:

        st.dataframe(
            selected_data.head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.markdown(
        '<div class="section-title">'
        '💡 Business Insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Cluster 0 — Spoken & Story Content"
    )

    st.write(
        """
        Contains **12,513 songs** and has very high
        speechiness (0.830). This indicates a strong
        association with spoken, narration and story-based
        audio content.
        """
    )

    st.subheader(
        "Cluster 1 — Calm / Classic Music"
    )

    st.write(
        """
        Contains **30,807 songs** with high acousticness
        (0.749), lower energy (0.311) and low speechiness
        (0.060). This represents a calmer and more acoustic
        listening profile.
        """
    )

    st.subheader(
        "Cluster 2 — Energetic / Pop-Oriented Music"
    )

    st.write(
        """
        Contains **52,517 songs**, making it the largest
        cluster. It has higher energy (0.694), tempo
        (124.911 BPM), danceability (0.627) and valence
        (0.666), representing energetic and upbeat music.
        """
    )

    st.divider()

    st.subheader(
        "Potential Applications"
    )

    st.write(
        """
        **🎵 Automated Playlist Creation**

        Songs can be grouped into playlists based on their
        discovered audio profiles.

        **🔎 Music Discovery**

        Users can explore songs belonging to different
        audio-based clusters.

        **🤖 Recommendation Systems**

        Cluster membership can be used as one signal for
        finding similar songs.

        **📂 Content Organization**

        Large music libraries can be automatically organized
        into meaningful audio groups.
        """
    )

    st.divider()

    st.subheader(
        "Machine Learning Approach"
    )

    st.success(
        """
        Genre was not used as an input feature during
        clustering. K-Means discovered groups using
        audio characteristics, while genre information
        was used afterward for interpretation and validation.
        """
    )

    st.subheader(
        "Model Comparison"
    )

    st.write(
        """
        DBSCAN and Hierarchical Clustering were evaluated
        as comparison approaches. K-Means remained the
        primary model because the project specifically
        focuses on K-Means, Elbow Method and Silhouette Score.
        """
    )

    st.subheader(
        "Final Conclusion"
    )

    st.success(
        """
        The project successfully grouped 95,837 songs into
        three meaningful audio-based clusters using
        unsupervised K-Means clustering.
        """
    )