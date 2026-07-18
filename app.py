import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.caption("by Riya Gupta | Labmentix")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "EDA & Insights",
    "RFM Analysis",
    "Elbow Method",
    "Customer Segmentation",
    "Product Recommendation",
    "Customer Prediction",
    "Business Insights"
])

uploaded_file = st.sidebar.file_uploader("Upload your own retail CSV (optional)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin1')
else:
    df = pd.read_csv('online_retail.csv', encoding='latin1')

if True:
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df.drop_duplicates()
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    if page == "Overview":
        st.title("🛒 Shopper Spectrum")
        st.markdown("### Customer Segmentation & Product Recommendation")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", f"{df['CustomerID'].nunique():,}")
        col2.metric("Products", f"{df['Description'].nunique():,}")
        col3.metric("Invoices", f"{df['InvoiceNo'].nunique():,}")
        col4.metric("Revenue", f"£{df['TotalPrice'].sum():,.0f}")

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10))

        st.markdown("### Data Quality Check (Before Cleaning)")
        if uploaded_file is not None:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            raw_df = pd.read_csv('online_retail.csv', encoding='latin1')
        st.write("Missing Values per Column:")
        st.dataframe(raw_df.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

    elif page == "EDA & Insights":
        st.title("📊 EDA & Insights")

        st.markdown("### Top 10 Countries by Sales")
        country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(10,4))
        country_sales.plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Total Revenue")
        st.pyplot(fig1)

        st.markdown("### Top 10 Best-Selling Products")
        top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(10,4))
        top_products.plot(kind='barh', ax=ax2, color='orange')
        ax2.invert_yaxis()
        ax2.set_xlabel("Total Quantity Sold")
        st.pyplot(fig2)

        st.markdown("### Monthly Sales Trend")
        df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('Month')['TotalPrice'].sum()
        fig3, ax3 = plt.subplots(figsize=(10,4))
        monthly_sales.plot(kind='line', marker='o', ax=ax3, color='green')
        ax3.set_ylabel("Total Revenue")
        st.pyplot(fig3)

    elif page == "RFM Analysis":
        st.title("📈 RFM Analysis")

        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()

        st.markdown("### RFM Table (first 10 customers)")
        st.dataframe(rfm.head(10))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Recency Distribution**")
            fig, ax = plt.subplots()
            ax.hist(rfm['Recency'], bins=30, color='skyblue')
            st.pyplot(fig)
        with col2:
            st.markdown("**Frequency Distribution**")
            fig, ax = plt.subplots()
            ax.hist(rfm['Frequency'], bins=30, color='lightgreen')
            st.pyplot(fig)
        with col3:
            st.markdown("**Monetary Distribution**")
            fig, ax = plt.subplots()
            ax.hist(rfm['Monetary'], bins=30, color='salmon')
            st.pyplot(fig)

    elif page == "Elbow Method":
        st.title("📉 Elbow Method for Optimal Clusters")

        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        inertia = []
        silhouette_scores = []
        k_range = range(2, 8)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(rfm_scaled)
            inertia.append(km.inertia_)
            silhouette_scores.append(silhouette_score(rfm_scaled, labels))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Elbow Curve**")
            fig, ax = plt.subplots()
            ax.plot(list(k_range), inertia, marker='o')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            st.pyplot(fig)
        with col2:
            st.markdown("**Silhouette Scores**")
            fig, ax = plt.subplots()
            ax.plot(list(k_range), silhouette_scores, marker='o', color='green')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Silhouette Score")
            st.pyplot(fig)

        st.info("Based on analysis, k=4 gives a good balance of cluster quality and business interpretability.")

    elif page == "Customer Segmentation":
        st.title("👥 Customer Segmentation")

        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        cluster_summary = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
        ordered_clusters = cluster_summary.index.tolist()
        label_names = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
        cluster_labels = {cluster: label_names[i] for i, cluster in enumerate(ordered_clusters)}
        rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

        st.markdown("### Segment Distribution")
        segment_counts = rfm['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(8,4))
        segment_counts.plot(kind='bar', color=['#4CAF50','#2196F3','#FF9800','#F44336'], ax=ax)
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

        st.markdown("### Cluster Averages")
        st.dataframe(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1))

        st.markdown("### Customer-Level Segments")
        st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(20))

    elif page == "Product Recommendation":
        st.title("🛍️ Product Recommendation")

        @st.cache_data
        def get_similarity_matrix(data):
            customer_product = data.pivot_table(
                index='CustomerID',
                columns='Description',
                values='Quantity',
                aggfunc='sum',
                fill_value=0
            )
            product_customer = customer_product.T
            sim_matrix = cosine_similarity(product_customer)
            sim_df = pd.DataFrame(sim_matrix, index=product_customer.index, columns=product_customer.index)
            return sim_df

        similarity_df = get_similarity_matrix(df)

        product_list = sorted(similarity_df.columns.tolist())
        selected_product = st.selectbox("Select a product", product_list)

        if st.button("Get Recommendations"):
            similar_scores = similarity_df[selected_product].sort_values(ascending=False)
            similar_scores = similar_scores.drop(selected_product)
            top_5 = similar_scores.head(5)

            st.markdown("### Top 5 Similar Products")
            for product, score in top_5.items():
                st.write(f"**{product}** — similarity: {score:.2f}")

    elif page == "Customer Prediction":
        st.title("🔮 Customer Segment Prediction")

        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        cluster_summary = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
        ordered_clusters = cluster_summary.index.tolist()
        label_names = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
        cluster_labels = {cluster: label_names[i] for i, cluster in enumerate(ordered_clusters)}

        st.markdown("Enter customer details to predict their segment:")

        col1, col2, col3 = st.columns(3)
        with col1:
            recency_input = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        with col2:
            frequency_input = st.number_input("Frequency (number of orders)", min_value=1, value=5)
        with col3:
            monetary_input = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

        if st.button("Predict Segment"):
            input_scaled = scaler.transform([[recency_input, frequency_input, monetary_input]])
            predicted_cluster = kmeans.predict(input_scaled)[0]
            predicted_segment = cluster_labels[predicted_cluster]
            st.success(f"Predicted Segment: **{predicted_segment}**")

    elif page == "Business Insights":
        st.title("📋 Business Insights")

        total_revenue = df['TotalPrice'].sum()
        avg_spending = df.groupby('CustomerID')['TotalPrice'].sum().mean()
        top_country = df.groupby('Country')['TotalPrice'].sum().idxmax()
        best_product = df.groupby('Description')['Quantity'].sum().idxmax()

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
        col2.metric("Average Customer Spending", f"£{avg_spending:,.0f}")

        st.markdown("### Key Takeaways")
        st.write(f"📈 **Total Revenue:** £{total_revenue:,.0f}")
        st.write(f"🌍 **Top Country:** {top_country}")
        st.write(f"🛍️ **Best Selling Product:** {best_product}")
        st.write("👑 **High-value customers** contribute the maximum share of revenue despite being the smallest group.")
        st.write("⚠️ **At-risk customers** need targeted retention campaigns (discounts, win-back emails).")
        st.write("📊 **Regular customers** can be converted into premium/high-value customers with loyalty programs.")
        st.write("🛒 **Occasional customers** form the largest base — good target for engagement campaigns to increase frequency.")

else:
    st.title("🛒 Shopper Spectrum")
    st.info("👈 Please upload a CSV file from the sidebar to get started.")