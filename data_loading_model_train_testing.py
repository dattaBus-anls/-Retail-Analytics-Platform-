## Environment Setup and Data loading ##
# Install required libraries (run this cell if using Google Colab)
# !pip install pandas numpy matplotlib seaborn scikit-learn mlxtend plotly -q

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# For association rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# For visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("All libraries imported successfully!")


# Load the Superstore dataset
print("Loading Superstore dataset...")

try:
    # Load from local file
    df = pd.read_csv('Sample-Superstore.csv', encoding='latin1')
    print("✅ Dataset loaded successfully from local file!")
except FileNotFoundError:
    print("❌ File 'Sample-Superstore.csv' not found in current directory")
    print("Please ensure the Superstore dataset is in the same folder as this script")
    df = None
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None


# Verify data loading and check structure
if 'df' in locals() and not df.empty:
    print(f"Data loaded successfully! Shape: {df.shape}")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for required columns
    required_cols = ['Customer ID', 'Order ID', 'Order Date', 'Sales', 'Quantity', 
                    'Product Name', 'Category', 'Sub-Category', 'Region', 'Segment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        print("Please check your dataset format")
    else:
        print("✅ All required columns present!")

    # Dataset Verification Section:
    # ========================
    
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION & QUALITY ASSESSMENT")
    print("="*60)
    
    # Basic statistical summary
    print("\n📈 DATASET STATISTICAL SUMMARY:")
    print("-" * 40)
    print(df.describe())
    
    # Check data types
    print(f"\n🔍 DATA TYPES:")
    print("-" * 20)
    print(df.dtypes)
    
    # Check for missing values
    print(f"\n🔍 MISSING DATA ANALYSIS:")
    print("-" * 30)
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percentage.values
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_summary) > 0:
        print("⚠️ MISSING DATA FOUND:")
        print(missing_summary.to_string(index=False))
        
        print(f"\n🔧 HANDLING MISSING DATA:")
        print("-" * 25)
        
        # Handle missing data for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_numeric = missing_summary[missing_summary['Column'].isin(numeric_cols)]
        
        if len(missing_numeric) > 0:
            print("Filling missing numeric values with column averages...")
            for col in missing_numeric['Column']:
                avg_value = df[col].mean()
                df[col].fillna(avg_value, inplace=True)
                print(f"  ✅ {col}: filled {missing_numeric[missing_numeric['Column']==col]['Missing_Count'].iloc[0]} missing values with average ({avg_value:.2f})")
        
        # Handle missing data for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        missing_categorical = missing_summary[missing_summary['Column'].isin(categorical_cols)]
        
        if len(missing_categorical) > 0:
            print("\nFilling missing categorical values with mode (most frequent)...")
            for col in missing_categorical['Column']:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                print(f"  ✅ {col}: filled {missing_categorical[missing_categorical['Column']==col]['Missing_Count'].iloc[0]} missing values with mode ('{mode_value}')")
        
        # Verify missing data handling
        print(f"\n✅ MISSING DATA VERIFICATION AFTER CLEANING:")
        print("-" * 45)
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing == 0:
            print("🎉 All missing data successfully handled!")
            print("Dataset is now ready for analysis.")
        else:
            print(f"⚠️ Still {remaining_missing} missing values remaining.")
            print("Remaining missing data:")
            print(df.isnull().sum()[df.isnull().sum() > 0])
    
    else:
        print("✅ No missing data found! Dataset is clean.")
    
    # Final dataset summary after cleaning
    print(f"\n📋 FINAL DATASET SUMMARY:")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Show sample of cleaned data
    print(f"\n📋 SAMPLE OF CLEANED DATA:")
    print("-" * 30)
    print(df.head(3))
    
    print("\n" + "="*60)
    print("✅ DATA EXPLORATION & CLEANING COMPLETE!")
    print("="*60)
    
    # END OF NEW SECTION
    # ==================

else:
    print("❌ Dataset not loaded. Please check your data source.")


## Data Preprocessing and Feature Engineering for Superstore Dataset ##

def preprocess_superstore_data(df):
    """
    Comprehensive data preprocessing for Superstore dataset
    """
    print("Starting data preprocessing...")
    print(f"Original data shape: {df.shape}")
    
    # Create a copy to work with
    df_clean = df.copy()
    
    # 1. Data Cleaning
    print("\n1. DATA CLEANING")
    
    # Remove rows with missing Customer ID (can't segment unknown customers)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['Customer ID'])
    print(f"Removed {initial_rows - len(df_clean)} rows with missing Customer ID")
    
    # Remove negative quantities and sales (returns/errors)
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['Sales'] > 0]
    print(f"After removing invalid quantities/sales: {df_clean.shape}")
    
    # Clean Customer ID (remove any special characters)
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(str).str.strip()
    
    # Convert Order Date to datetime
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
    
    # 2. Feature Engineering for RFM Analysis
    print("\n2. RFM FEATURE ENGINEERING")
    
    # Define reference date (latest date in dataset + 1 day)
    reference_date = df_clean['Order Date'].max() + timedelta(days=1)
    print(f"Reference date for recency calculation: {reference_date.strftime('%Y-%m-%d')}")
    
    # Calculate RFM metrics per customer
    rfm = df_clean.groupby('Customer ID').agg({
        'Order Date': lambda x: (reference_date - x.max()).days,  # Recency
        'Order ID': 'nunique',  # Frequency (unique orders)
        'Sales': 'sum'  # Monetary (total sales)
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # 3. Additional Behavioral Features
    print("3. ADDITIONAL BEHAVIORAL FEATURES")
    
    # Average Order Value
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    
    # Product diversity (unique products per customer)
    product_diversity = df_clean.groupby('Customer ID')['Product ID'].nunique().reset_index()
    product_diversity.columns = ['Customer ID', 'UniqueProducts']
    rfm = rfm.merge(product_diversity, on='Customer ID')
    
    # Category diversity (unique categories per customer)
    category_diversity = df_clean.groupby('Customer ID')['Category'].nunique().reset_index()
    category_diversity.columns = ['Customer ID', 'UniqueCategories']
    rfm = rfm.merge(category_diversity, on='Customer ID')
    
    # Sub-category diversity
    subcategory_diversity = df_clean.groupby('Customer ID')['Sub-Category'].nunique().reset_index()
    subcategory_diversity.columns = ['Customer ID', 'UniqueSubCategories']
    rfm = rfm.merge(subcategory_diversity, on='Customer ID')
    
    # Geographic features
    customer_geo = df_clean.groupby('Customer ID').agg({
        'State': 'first',
        'Region': 'first',
        'Segment': 'first'
    }).reset_index()
    rfm = rfm.merge(customer_geo, on='Customer ID')
    
    # Days span (days between first and last order)
    date_span = df_clean.groupby('Customer ID')['Order Date'].agg(['min', 'max']).reset_index()
    date_span['DaysSpan'] = (date_span['max'] - date_span['min']).dt.days
    date_span = date_span[['Customer ID', 'DaysSpan']]
    rfm = rfm.merge(date_span, on='Customer ID')
    
    # Total items purchased
    total_items = df_clean.groupby('Customer ID')['Quantity'].sum().reset_index()
    total_items.columns = ['Customer ID', 'TotalItems']
    rfm = rfm.merge(total_items, on='Customer ID')
    
    # Profitability metrics
    profit_metrics = df_clean.groupby('Customer ID')['Profit'].sum().reset_index()
    profit_metrics.columns = ['Customer ID', 'TotalProfit']
    rfm = rfm.merge(profit_metrics, on='Customer ID')
    rfm['AvgProfit'] = rfm['TotalProfit'] / rfm['Frequency']
    
    print(f"\nFinal RFM table shape: {rfm.shape}")
    print(f"Enhanced features: {list(rfm.columns)}")
    
    # 4. Data Quality Summary
    print("\n4. DATA QUALITY SUMMARY")
    print(f"Original transactions: {len(df):,}")
    print(f"Clean transactions: {len(df_clean):,}")
    print(f"Data retention rate: {len(df_clean)/len(df)*100:.1f}%")
    print(f"Unique customers: {rfm['Customer ID'].nunique():,}")
    print(f"Unique products: {df_clean['Product ID'].nunique():,}")
    print(f"Date range: {df_clean['Order Date'].min().strftime('%Y-%m-%d')} to {df_clean['Order Date'].max().strftime('%Y-%m-%d')}")
    print(f"Categories: {df_clean['Category'].nunique()} ({', '.join(df_clean['Category'].unique())})")
    print(f"Segments: {df_clean['Segment'].nunique()} ({', '.join(df_clean['Segment'].unique())})")
    
    return df_clean, rfm

# Visualize RFM distributions
def visualize_rfm_distributions(rfm):
    """
    Create comprehensive visualizations of RFM distributions
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Customer Features - Superstore Analysis', fontsize=16, y=1.02)
    
    # Recency
    axes[0, 0].hist(rfm['Recency'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Recency (Days since last purchase)')
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Number of Customers')
    
    # Frequency
    axes[0, 1].hist(rfm['Frequency'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Frequency (Number of orders)')
    axes[0, 1].set_xlabel('Number of Orders')
    axes[0, 1].set_ylabel('Number of Customers')
    
    # Monetary
    axes[0, 2].hist(rfm['Monetary'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Monetary (Total sales)')
    axes[0, 2].set_xlabel('Total Sales ($)')
    axes[0, 2].set_ylabel('Number of Customers')
    
    # Average Order Value
    axes[1, 0].hist(rfm['AvgOrderValue'], bins=50, color='gold', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Average Order Value')
    axes[1, 0].set_xlabel('Average Order Value ($)')
    axes[1, 0].set_ylabel('Number of Customers')
    
    # Unique Products
    axes[1, 1].hist(rfm['UniqueProducts'], bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Product Diversity')
    axes[1, 1].set_xlabel('Number of Unique Products')
    axes[1, 1].set_ylabel('Number of Customers')
    
    # Total Profit
    axes[1, 2].hist(rfm['TotalProfit'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Total Profit Generated')
    axes[1, 2].set_xlabel('Total Profit ($)')
    axes[1, 2].set_ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\nRFM SUMMARY STATISTICS")
    print("=" * 50)
    print(rfm[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'TotalProfit']].describe().round(2))

print("Data preprocessing functions ready!")
print("Run: df_clean, rfm = preprocess_superstore_data(df) when you load your data")


## Customer Segmentation with K-Means Clustering for Superstore Dataset

def prepare_clustering_data(rfm, remove_outliers=True):
    """
    Prepare data for clustering by handling outliers and scaling
    """
    print("PREPARING DATA FOR CLUSTERING")
    print("=" * 40)
    
    # Select features for clustering
    clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 
                          'UniqueProducts', 'UniqueCategories', 'TotalItems', 'TotalProfit']
    
    X = rfm[clustering_features].copy()
    print(f"Selected features: {clustering_features}")
    print(f"Original data shape: {X.shape}")
    
    if remove_outliers:
        # Handle outliers using IQR method
        print("\nRemoving outliers using IQR method...")
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        X_clean = X[mask]
        rfm_clean = rfm[mask].copy()
        
        print(f"Original customers: {len(X)}")
        print(f"After outlier removal: {len(X_clean)}")
        print(f"Outliers removed: {len(X) - len(X_clean)} ({(len(X) - len(X_clean))/len(X)*100:.1f}%)")
    else:
        X_clean = X
        rfm_clean = rfm.copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    print(f"\nFeatures standardized (mean=0, std=1)")
    print(f"Final clustering data shape: {X_scaled.shape}")
    
    return X_clean, X_scaled, rfm_clean, scaler, clustering_features

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Find optimal number of clusters using elbow method and silhouette analysis
    """
    print("\nFINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 40)
    
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    davies_bouldin_scores = []
    K_range = range(2, max_k + 1)
    
    print("Testing different numbers of clusters...")
    for k in K_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, 
                       n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        
        print(f"k={k}: Silhouette Score = {silhouette_scores[-1]:.3f}")
    
    # Plot evaluation metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cluster Evaluation Metrics - Superstore Dataset', fontsize=16)
    
    # Elbow curve
    ax1.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    # Davies-Bouldin Index (lower is better)
    ax3.plot(K_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Davies-Bouldin Score')
    ax3.set_title('Davies-Bouldin Index (Lower = Better)')
    ax3.grid(True, alpha=0.3)
    
    # Combined view
    ax4.plot(K_range, np.array(silhouette_scores), 'r-', label='Silhouette Score', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(K_range, np.array(davies_bouldin_scores), 'g-', label='Davies-Bouldin', linewidth=2)
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Silhouette Score', color='r')
    ax4_twin.set_ylabel('Davies-Bouldin Score', color='g')
    ax4.set_title('Combined Metrics')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    optimal_k_davies = K_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\nOPTIMAL CLUSTER RECOMMENDATIONS:")
    print(f"Best k by Silhouette Score: {optimal_k_silhouette} (Score: {max(silhouette_scores):.3f})")
    print(f"Best k by Davies-Bouldin: {optimal_k_davies} (Score: {min(davies_bouldin_scores):.3f})")
    
    # Business recommendation
    if optimal_k_silhouette == optimal_k_davies:
        recommended_k = optimal_k_silhouette
    else:
        # For business interpretability, typically choose between 4-6 clusters
        business_range = [k for k in [4, 5, 6] if k in K_range]
        if business_range:
            business_scores = [(k, silhouette_scores[k-2]) for k in business_range]
            recommended_k = max(business_scores, key=lambda x: x[1])[0]
        else:
            recommended_k = optimal_k_silhouette
    
    print(f"RECOMMENDED k for business analysis: {recommended_k}")
    
    return recommended_k, silhouette_scores, wcss

def apply_clustering(X_scaled, rfm_clean, optimal_k, clustering_features, scaler):
    """
    Apply K-Means clustering and analyze results
    """
    print(f"\nAPPLYING K-MEANS CLUSTERING (k={optimal_k})")
    print("=" * 50)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, 
                   n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    rfm_clustered = rfm_clean.copy()
    rfm_clustered['Cluster'] = cluster_labels
    
    # Calculate cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=clustering_features)
    cluster_centers_df['Cluster'] = range(optimal_k)
    
    print("CLUSTER CENTERS (Original Scale):")
    print(cluster_centers_df.round(2))
    
    # Cluster sizes and characteristics
    print(f"\nCLUSTER SIZES:")
    cluster_sizes = rfm_clustered['Cluster'].value_counts().sort_index()
    total_customers = len(rfm_clustered)
    
    for cluster in range(optimal_k):
        size = cluster_sizes[cluster]
        percentage = size / total_customers * 100
        print(f"Cluster {cluster}: {size:,} customers ({percentage:.1f}%)")
    
    # Calculate cluster validation metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    print(f"\nCLUSTER VALIDATION METRICS:")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
    
    return rfm_clustered, cluster_centers_df, kmeans

def visualize_clusters(X_scaled, rfm_clustered, clustering_features):
    """
    Visualize clusters using PCA and create comprehensive plots
    """
    print("\nVISUALIZING CUSTOMER SEGMENTS")
    print("=" * 35)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_labels = rfm_clustered['Cluster'].values
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # PCA Scatter Plot
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Customer Segments (PCA Visualization)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # Cluster size pie chart
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
           autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Cluster Size Distribution')
    
    # RFM by Cluster - Monetary
    ax3 = plt.subplot(2, 3, 3)
    rfm_clustered.boxplot(column='Monetary', by='Cluster', ax=ax3)
    plt.title('Monetary Value by Cluster')
    plt.suptitle('')  # Remove automatic title
    plt.xticks(rotation=0)
    
    # RFM by Cluster - Frequency
    ax4 = plt.subplot(2, 3, 4)
    rfm_clustered.boxplot(column='Frequency', by='Cluster', ax=ax4)
    plt.title('Purchase Frequency by Cluster')
    plt.suptitle('')
    plt.xticks(rotation=0)
    
    # RFM by Cluster - Recency
    ax5 = plt.subplot(2, 3, 5)
    rfm_clustered.boxplot(column='Recency', by='Cluster', ax=ax5)
    plt.title('Recency by Cluster')
    plt.suptitle('')
    plt.xticks(rotation=0)
    
    # Segment distribution by business segment
    ax6 = plt.subplot(2, 3, 6)
    segment_cluster = pd.crosstab(rfm_clustered['Segment'], rfm_clustered['Cluster'])
    segment_cluster.plot(kind='bar', ax=ax6, colormap='viridis')
    plt.title('Business Segment vs Customer Clusters')
    plt.xlabel('Business Segment')
    plt.ylabel('Number of Customers')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total variance explained by 2 PCA components: {sum(pca.explained_variance_ratio_):.1%}")
    
    return X_pca, pca

def create_customer_personas(rfm_clustered, cluster_centers_df):
    """
    Create detailed customer personas based on cluster characteristics
    """
    print("\nCUSTOMER PERSONAS ANALYSIS")
    print("=" * 40)
    
    # Calculate cluster profiles
    numeric_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 
                       'UniqueProducts', 'UniqueCategories', 'TotalItems', 'TotalProfit']
    
    cluster_profiles = rfm_clustered.groupby('Cluster')[numeric_features].mean()
    
    # Business segment distribution by cluster
    segment_dist = pd.crosstab(rfm_clustered['Cluster'], rfm_clustered['Segment'], normalize='index') * 100
    
    # Regional distribution by cluster
    region_dist = pd.crosstab(rfm_clustered['Cluster'], rfm_clustered['Region'], normalize='index') * 100
    
    personas = {}
    cluster_names = {}
    
    for cluster_id in sorted(rfm_clustered['Cluster'].unique()):
        profile = cluster_profiles.loc[cluster_id]
        size = len(rfm_clustered[rfm_clustered['Cluster'] == cluster_id])
        
        # Determine persona based on RFM characteristics
        if profile['Monetary'] >= cluster_profiles['Monetary'].quantile(0.8) and \
           profile['Frequency'] >= cluster_profiles['Frequency'].quantile(0.6):
            name = "Champions"
            description = "High-value, frequent customers - Best segment for retention"
            strategy = "VIP treatment, exclusive offers, loyalty rewards"
            
        elif profile['Frequency'] >= cluster_profiles['Frequency'].quantile(0.7):
            name = "Loyal Customers"
            description = "Regular customers with consistent purchase behavior"
            strategy = "Upselling, cross-category promotions, referral programs"
            
        elif profile['Monetary'] >= cluster_profiles['Monetary'].quantile(0.7) and \
             profile['Recency'] <= cluster_profiles['Recency'].quantile(0.3):
            name = "Big Spenders"
            description = "Recent high-value customers with potential for growth"
            strategy = "Premium product recommendations, personalized service"
            
        elif profile['Recency'] <= cluster_profiles['Recency'].quantile(0.4):
            name = "Potential Loyalists"
            description = "Recent customers showing engagement potential"
            strategy = "Onboarding campaigns, product education, engagement boost"
            
        elif profile['Recency'] >= cluster_profiles['Recency'].quantile(0.8):
            name = "At Risk"
            description = "Previously valuable customers who haven't purchased recently"
            strategy = "Win-back campaigns, special discounts, re-engagement"
            
        else:
            name = "New Customers"
            description = "Recently acquired customers with unknown potential"
            strategy = "Welcome series, category exploration, habit formation"
        
        personas[cluster_id] = {
            'name': name,
            'description': description,
            'strategy': strategy,
            'size': size,
            'percentage': size / len(rfm_clustered) * 100,
            'profile': profile.to_dict()
        }
        
        cluster_names[cluster_id] = name
    
    # Display personas
    for cluster_id, persona in personas.items():
        print(f"\n🎯 CLUSTER {cluster_id}: {persona['name'].upper()}")
        print("─" * 60)
        print(f"Size: {persona['size']:,} customers ({persona['percentage']:.1f}%)")
        print(f"Description: {persona['description']}")
        print(f"Strategy: {persona['strategy']}")
        
        print(f"\nKey Metrics:")
        print(f"  • Recency: {persona['profile']['Recency']:.0f} days")
        print(f"  • Frequency: {persona['profile']['Frequency']:.1f} orders")
        print(f"  • Monetary: ${persona['profile']['Monetary']:,.0f}")
        print(f"  • Avg Order Value: ${persona['profile']['AvgOrderValue']:,.0f}")
        print(f"  • Product Diversity: {persona['profile']['UniqueProducts']:.1f} products")
        print(f"  • Total Profit: ${persona['profile']['TotalProfit']:,.0f}")
        
        # Top business segment in this cluster
        top_segment = segment_dist.loc[cluster_id].idxmax()
        segment_pct = segment_dist.loc[cluster_id, top_segment]
        print(f"  • Primary Segment: {top_segment} ({segment_pct:.0f}%)")
        
        # FIXED: Complete the missing part
        top_region = region_dist.loc[cluster_id].idxmax()
        region_pct = region_dist.loc[cluster_id, top_region]
        print(f"  • Primary Region: {top_region} ({region_pct:.0f}%)")
    
    return personas, cluster_names

# FIXED: Add missing alternative clustering comparison function
def alternative_clustering_comparison(X_scaled, optimal_k):
    """
    Compare K-Means with alternative clustering methods
    """
    print("\nALTERNATIVE CLUSTERING METHODS COMPARISON")
    print("=" * 50)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # DBSCAN (density-based)
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    # Use elbow method for eps
    eps = np.percentile(distances, 95)  # Use 95th percentile as eps
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Calculate comparison metrics
    hier_vs_kmeans = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    
    print("CLUSTERING METHOD COMPARISON:")
    print(f"K-Means clusters: {len(np.unique(kmeans_labels))}")
    print(f"Hierarchical clusters: {len(np.unique(hierarchical_labels))}")
    print(f"DBSCAN clusters: {len(np.unique(dbscan_labels[dbscan_labels != -1]))}")
    print(f"DBSCAN noise points: {np.sum(dbscan_labels == -1)}")
    
    print(f"\nSimilarity Scores:")
    print(f"K-Means vs Hierarchical: {hier_vs_kmeans:.3f}")
    print(f"(1.0 = identical, 0.0 = random)")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means visualization
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[0].set_title('K-Means Clustering')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    
    # Hierarchical visualization
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7)
    axes[1].set_title('Hierarchical Clustering')
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    
    # DBSCAN visualization
    scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
    axes[2].set_title('DBSCAN Clustering')
    axes[2].set_xlabel('First Principal Component')
    axes[2].set_ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.show()
    
    return hierarchical_labels, dbscan_labels

# FIXED: Add missing perform_customer_segmentation function
def perform_customer_segmentation(df):
    """
    Complete customer segmentation pipeline
    """
    print("🚀 STARTING CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 60)
    
    # Step 1: Preprocess data
    df_clean, rfm = preprocess_superstore_data(df)
    
    # Step 2: Prepare clustering data
    X_clean, X_scaled, rfm_clean, scaler, clustering_features = prepare_clustering_data(rfm)
    
    # Step 3: Find optimal clusters
    optimal_k, silhouette_scores, wcss = find_optimal_clusters(X_scaled)
    
    # Step 4: Apply clustering
    rfm_clustered, cluster_centers_df, kmeans_model = apply_clustering(
        X_scaled, rfm_clean, optimal_k, clustering_features, scaler)
    
    # Step 5: Visualize results
    X_pca, pca_model = visualize_clusters(X_scaled, rfm_clustered, clustering_features)
    
    # Step 6: Create personas
    personas, cluster_names = create_customer_personas(rfm_clustered, cluster_centers_df)
    
    # Step 7: Compare with alternative methods
    hier_labels, dbscan_labels = alternative_clustering_comparison(X_scaled, optimal_k)
    
    print("\n✅ CUSTOMER SEGMENTATION COMPLETE!")
    print("Results stored in variables for further analysis.")
    
    return {
        'df_clean': df_clean,
        'rfm_clustered': rfm_clustered,
        'cluster_centers': cluster_centers_df,
        'personas': personas,
        'cluster_names': cluster_names,
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'pca_model': pca_model,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'clustering_features': clustering_features
    }


## Market Basket Analysis with Apriori Algorithm for Superstore Dataset

def prepare_market_basket_data(df_clean):
    """
    Prepare Superstore data for market basket analysis
    """
    print("PREPARING MARKET BASKET ANALYSIS DATA")
    print("=" * 45)
    
    # For Superstore, we'll analyze product associations
    # Create transaction matrix at product level
    print("Creating transaction matrix...")
    
    # Group by Order ID to create baskets
    basket_data = df_clean.groupby(['Order ID', 'Product Name'])['Quantity'].sum().unstack().fillna(0)
    
    # Convert to binary matrix (0/1)
    # 1 = product was purchased in this order, 0 = not purchased
    # basket_binary = basket_data.applymap(lambda x: 1 if x > 0 else 0)
    basket_binary = (basket_data > 0)

    min_orders = 15
    product_freq = basket_binary.sum(axis=0)
    kept_cols = product_freq[product_freq >= min_orders].index
    basket_binary = basket_binary[kept_cols]
    print(f"Kept {len(kept_cols)} products after frequency pruning (≥ {min_orders} orders)")
    
    print(f"Transaction matrix shape: {basket_binary.shape}")
    print(f"Number of transactions (orders): {basket_binary.shape[0]:,}")
    print(f"Number of unique products: {basket_binary.shape[1]:,}")
    
    # Show data quality metrics
    avg_items_per_order = basket_binary.sum(axis=1).mean()
    max_items_per_order = basket_binary.sum(axis=1).max()
    min_items_per_order = basket_binary.sum(axis=1).min()
    
    print(f"\nBasket Analysis Summary:")
    print(f"Average items per order: {avg_items_per_order:.1f}")
    print(f"Maximum items per order: {max_items_per_order}")
    print(f"Minimum items per order: {min_items_per_order}")
    
    # Remove single-item transactions for association analysis
    multi_item_mask = basket_binary.sum(axis=1) > 1
    basket_binary_filtered = basket_binary[multi_item_mask]
    
    print(f"\nAfter filtering single-item transactions:")
    print(f"Transactions for analysis: {basket_binary_filtered.shape[0]:,}")
    print(f"Filtered out: {basket_binary.shape[0] - basket_binary_filtered.shape[0]:,} single-item orders")
    
    # Show sample transactions
    print(f"\nSample transaction matrix (first 5 orders, first 5 products):")
    sample_products = basket_binary.columns[:5]
    sample_data = basket_binary.iloc[:5][sample_products]
    print(sample_data)
    
    return basket_binary_filtered, basket_binary

def perform_apriori_analysis(basket_binary, min_support=0.01, min_confidence=0.25):
    """
    Perform Apriori algorithm for association rule mining
    """
    print(f"\nPERFORMING APRIORI ANALYSIS")
    print("=" * 35)
    
    print(f"Parameters:")
    print(f"Minimum Support: {min_support:.2%} (items in at least {min_support*100:.1f}% of transactions)")
    print(f"Minimum Confidence: {min_confidence:.2%}")
    
    # Find frequent itemsets
    print("\nFinding frequent itemsets...")
    frequent_itemsets = apriori(basket_binary, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("⚠️ No frequent itemsets found! Try reducing min_support.")
        return None, None
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    # Show top frequent itemsets
    print("\nTop 15 Most Frequent Itemsets:")
    top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(15)
    for idx, row in top_itemsets.iterrows():
        items = ', '.join(list(row['itemsets']))
        support_pct = row['support'] * 100
        print(f"  {support_pct:5.1f}% | {items}")
    
    # Generate association rules
    print(f"\nGenerating association rules...")
    rules = association_rules(frequent_itemsets, metric="confidence", 
                             min_threshold=min_confidence)
    
    if rules.empty:
        print("⚠️ No association rules found! Try reducing min_confidence.")
        return frequent_itemsets, None
    
    # Add additional metrics
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    # Calculate additional business metrics
    rules['support_pct'] = rules['support'] * 100
    rules['confidence_pct'] = rules['confidence'] * 100
    rules['lift_strength'] = pd.cut(rules['lift'], 
                                   bins=[0, 1.5, 2.5, float('inf')], 
                                   labels=['Weak', 'Moderate', 'Strong'])
    
    print(f"Generated {len(rules)} association rules")
    
    # Rules summary by strength
    lift_summary = rules['lift_strength'].value_counts()
    print(f"\nRules by Lift Strength:")
    for strength, count in lift_summary.items():
        print(f"  {strength}: {count} rules")
    
    return frequent_itemsets, rules

def analyze_association_rules(rules, top_n=20):
    """
    Analyze and display top association rules with business insights
    """
    print(f"\nTOP {top_n} ASSOCIATION RULES ANALYSIS")
    print("=" * 50)
    
    # Sort rules by lift (most interesting first)
    top_rules = rules.sort_values(['lift', 'confidence'], ascending=False).head(top_n)
    
    print("🔍 STRONGEST PRODUCT ASSOCIATIONS:")
    print("─" * 70)
    
    for idx, (_, rule) in enumerate(top_rules.iterrows(), 1):
        antecedent = ', '.join(list(rule['antecedents']))
        consequent = ', '.join(list(rule['consequents']))
        
        print(f"\n{idx:2d}. RULE:")
        print(f"    If customer buys: {antecedent}")
        print(f"    Then likely to buy: {consequent}")
        print(f"    📊 Support: {rule['support_pct']:.1f}% | "
              f"Confidence: {rule['confidence_pct']:.1f}% | "
              f"Lift: {rule['lift']:.2f}")
        
        # Business interpretation
        if rule['lift'] > 3:
            strength = "Very Strong"
        elif rule['lift'] > 2:
            strength = "Strong"
        elif rule['lift'] > 1.5:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        print(f"    💪 Association Strength: {strength}")
    
    # Category-level analysis
    print(f"\n📈 RULE QUALITY DISTRIBUTION:")
    print(f"Average Confidence: {rules['confidence'].mean():.1%}")
    print(f"Average Lift: {rules['lift'].mean():.2f}")
    print(f"Rules with Lift > 2: {len(rules[rules['lift'] > 2])} ({len(rules[rules['lift'] > 2])/len(rules):.1%})")
    print(f"Rules with Confidence > 50%: {len(rules[rules['confidence'] > 0.5])} ({len(rules[rules['confidence'] > 0.5])/len(rules):.1%})")

def visualize_market_basket_results(rules, frequent_itemsets):
    """
    Create comprehensive visualizations for market basket analysis
    """
    print("\nCREATING MARKET BASKET VISUALIZATIONS")
    print("=" * 40)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Support vs Confidence scatter plot (sized by lift)
    ax1 = plt.subplot(3, 3, 1)
    scatter = plt.scatter(rules['support'], rules['confidence'], 
                         c=rules['lift'], s=rules['lift']*30, 
                         alpha=0.6, cmap='viridis')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence\n(size and color = Lift)')
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, alpha=0.3)
    
    # 2. Lift distribution
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(rules['lift'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Lift')
    plt.ylabel('Number of Rules')
    plt.title('Distribution of Lift Values')
    plt.axvline(rules['lift'].mean(), color='red', linestyle='--', 
                label=f'Mean: {rules["lift"].mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Confidence distribution
    ax3 = plt.subplot(3, 3, 3)
    plt.hist(rules['confidence'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Rules')
    plt.title('Distribution of Confidence Values')
    plt.axvline(rules['confidence'].mean(), color='red', linestyle='--',
                label=f'Mean: {rules["confidence"].mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Support distribution
    ax4 = plt.subplot(3, 3, 4)
    plt.hist(rules['support'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
    plt.xlabel('Support')
    plt.ylabel('Number of Rules')
    plt.title('Distribution of Support Values')
    plt.grid(True, alpha=0.3)
    
    # 5. Frequent itemsets by size
    ax5 = plt.subplot(3, 3, 5)
    itemset_sizes = frequent_itemsets['itemsets'].apply(len)
    size_counts = itemset_sizes.value_counts().sort_index()
    plt.bar(size_counts.index, size_counts.values, color='gold', alpha=0.7, edgecolor='black')
    plt.xlabel('Itemset Size')
    plt.ylabel('Number of Itemsets')
    plt.title('Frequent Itemsets by Size')
    plt.grid(True, alpha=0.3)
    
    # 6. Top 10 rules by lift (horizontal bar chart)
    ax6 = plt.subplot(3, 3, 6)
    top_10_rules = rules.sort_values('lift', ascending=False).head(10)
    rule_labels = []
    for _, rule in top_10_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))[:20] + "..." if len(', '.join(list(rule['antecedents']))) > 20 else ', '.join(list(rule['antecedents']))
        con = ', '.join(list(rule['consequents']))[:20] + "..." if len(', '.join(list(rule['consequents']))) > 20 else ', '.join(list(rule['consequents']))
        rule_labels.append(f"{ant} → {con}")
    
    y_pos = np.arange(len(rule_labels))
    plt.barh(y_pos, top_10_rules['lift'], color='purple', alpha=0.7)
    plt.yticks(y_pos, rule_labels, fontsize=8)
    plt.xlabel('Lift')
    plt.title('Top 10 Rules by Lift')
    plt.grid(True, alpha=0.3)
    
    # 7. Rule strength categorization
    ax7 = plt.subplot(3, 3, 7)
    strength_counts = rules['lift_strength'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(strength_counts.values, labels=strength_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    plt.title('Rules by Association Strength')
    
    # 8. Confidence vs Lift scatter
    ax8 = plt.subplot(3, 3, 8)
    plt.scatter(rules['confidence'], rules['lift'], alpha=0.6, color='orange')
    plt.xlabel('Confidence')
    plt.ylabel('Lift')
    plt.title('Confidence vs Lift')
    plt.grid(True, alpha=0.3)
    
    # 9. Most frequent single items
    ax9 = plt.subplot(3, 3, 9)
    single_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1].copy()
    single_items['item_name'] = single_items['itemsets'].apply(lambda x: list(x)[0])
    top_single = single_items.sort_values('support', ascending=False).head(10)
    
    item_names = [name[:15] + "..." if len(name) > 15 else name for name in top_single['item_name']]
    plt.barh(range(len(item_names)), top_single['support'], color='lightcoral', alpha=0.7)
    plt.yticks(range(len(item_names)), item_names, fontsize=8)
    plt.xlabel('Support')
    plt.title('Top 10 Most Frequent Products')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_product_recommendation_engine(rules):
    """
    Create a product recommendation engine based on association rules
    """
    print("\n🤖 PRODUCT RECOMMENDATION ENGINE")
    print("=" * 40)
    
    def get_product_recommendations(product_name, max_recommendations=5):
        """
        Get product recommendations for a given product
        """
        # Find rules where the product is in the antecedent
        relevant_rules = rules[rules['antecedents'].apply(
            lambda x: product_name in [item for item in x])]
        
        if len(relevant_rules) == 0:
            return f"No recommendations found for '{product_name}'"
        
        # Sort by lift and confidence
        relevant_rules = relevant_rules.sort_values(
            ['lift', 'confidence'], ascending=False).head(max_recommendations)
        
        recommendations = []
        for idx, rule in relevant_rules.iterrows():
            consequent_items = list(rule['consequents'])
            for item in consequent_items:
                recommendations.append({
                    'Recommended_Product': item,
                    'Confidence': f"{rule['confidence']:.1%}",
                    'Lift': f"{rule['lift']:.2f}",
                    'Support': f"{rule['support']:.2%}"
                })
        
        return pd.DataFrame(recommendations).drop_duplicates('Recommended_Product').head(max_recommendations)
    
    # Test the recommendation engine
    print("Testing Product Recommendation Engine:")
    print("─" * 40)
    
    # Get list of products that appear in rules
    all_products = set()
    for rule_antecedents in rules['antecedents']:
        all_products.update(rule_antecedents)
    
    if all_products:
        # Test with a few popular products
        test_products = list(all_products)[:3]
        
        for product in test_products:
            print(f"\n🛒 Recommendations for: '{product[:50]}{'...' if len(product) > 50 else ''}'")
            recommendations = get_product_recommendations(product)
            if isinstance(recommendations, str):
                print(f"   {recommendations}")
            else:
                for idx, rec in recommendations.iterrows():
                    print(f"   • {rec['Recommended_Product'][:40]}")
                    print(f"     Confidence: {rec['Confidence']} | Lift: {rec['Lift']}")
    
    return get_product_recommendations


def analyze_cross_selling_opportunities(rules, df_clean):
    """
    Identify cross-selling opportunities by category and segment
    """
    print("\n💰 CROSS-SELLING OPPORTUNITIES ANALYSIS")
    print("=" * 45)
    
    # Create product-to-category mapping
    product_category_map = df_clean.groupby('Product Name')[['Category', 'Sub-Category']].first().to_dict()
    
    # Add category information to rules
    rules_with_categories = rules.copy()
    
    def get_categories(itemset):
        categories = []
        for item in itemset:
            if item in product_category_map['Category']:
                categories.append(product_category_map['Category'][item])
        return list(set(categories))
    
    rules_with_categories['antecedent_categories'] = rules_with_categories['antecedents'].apply(get_categories)
    rules_with_categories['consequent_categories'] = rules_with_categories['consequents'].apply(get_categories)
    
    # Find cross-category associations
    cross_category_rules = rules_with_categories[
        rules_with_categories.apply(
            lambda row: len(set(row['antecedent_categories']).intersection(
                set(row['consequent_categories']))) == 0, axis=1
        )
    ]
    
    if len(cross_category_rules) > 0:
        print(f"Found {len(cross_category_rules)} cross-category association rules")
        print("\nTop 10 Cross-Category Opportunities:")
        
        top_cross_category = cross_category_rules.sort_values('lift', ascending=False).head(10)
        
        for idx, (_, rule) in enumerate(top_cross_category.iterrows(), 1):
            antecedent = ', '.join(list(rule['antecedents']))[:30]
            consequent = ', '.join(list(rule['consequents']))[:30]
            ant_cat = ', '.join(rule['antecedent_categories'])
            con_cat = ', '.join(rule['consequent_categories'])
            
            print(f"\n{idx}. {ant_cat} → {con_cat}")
            print(f"   Products: {antecedent} → {consequent}")
            print(f"   Confidence: {rule['confidence']:.1%} | Lift: {rule['lift']:.2f}")
    
    # Revenue impact analysis
    print(f"\n💲 REVENUE IMPACT ANALYSIS")
    
    # Calculate potential revenue from recommendations
    avg_order_value = df_clean.groupby('Order ID')['Sales'].sum().mean()
    total_orders = df_clean['Order ID'].nunique()
    
    print(f"Average Order Value: ${avg_order_value:,.2f}")
    print(f"Total Orders in Dataset: {total_orders:,}")
    
    # Estimate impact of top rules
    high_impact_rules = rules[
        (rules['confidence'] > 0.3) & 
        (rules['lift'] > 2) & 
        (rules['support'] > 0.01)
    ]
    
    if len(high_impact_rules) > 0:
        potential_additional_sales = len(high_impact_rules) * avg_order_value * 0.1  # Conservative 10% uplift
        print(f"\nHigh-Impact Rules: {len(high_impact_rules)}")
        print(f"Estimated Additional Revenue Opportunity: ${potential_additional_sales:,.2f}")


def analyze_temporal_associations(rules, df_clean):
    """
    Simple seasonal/temporal check for 1→1 rules.
    Groups orders by year-month and computes monthly confidence P(consequent | antecedent).
    """
    if rules is None or getattr(rules, "empty", True) or len(rules) == 0:
        print("No rules to analyze temporally.")
        return

    df_tmp = df_clean.copy()
    df_tmp['Order Date'] = pd.to_datetime(df_tmp['Order Date'])
    df_tmp['ym'] = df_tmp['Order Date'].dt.to_period('M').astype(str)

    # Map each order to its set of products
    order_items = df_tmp.groupby('Order ID')['Product Name'].apply(set)

    # Focus on 1→1 rules
    simple_rules = rules[
        (rules['antecedents'].apply(len) == 1) &
        (rules['consequents'].apply(len) == 1)
    ]
    if simple_rules.empty:
        print("No simple 1→1 rules for temporal analysis.")
        return

    print("\n🗓️ Temporal pattern check (monthly confidence for first few months):")
    for _, r in simple_rules.iterrows():
        a = list(r['antecedents'])[0]
        c = list(r['consequents'])[0]

        monthly_conf = []
        for ym, ids in df_tmp.groupby('ym')['Order ID'].unique().items():
            subset = order_items[order_items.index.isin(ids)]
            a_mask = subset.apply(lambda s: a in s)
            a_count = a_mask.sum()
            if a_count == 0:
                continue
            both = subset[a_mask].apply(lambda s: c in s).sum()
            monthly_conf.append((ym, both / a_count))

        if monthly_conf:
            monthly_conf.sort()
            preview = ", ".join([f"{m}: {v:.2f}" for m, v in monthly_conf[:6]])
            print(f"  {a} → {c}: {preview}{' ...' if len(monthly_conf) > 6 else ''}")


def perform_market_basket_analysis(df_clean, min_support=0.01, min_confidence=0.25):
    """
    Complete market basket analysis pipeline for Superstore dataset
    """
    print("🛒 STARTING MARKET BASKET ANALYSIS")
    print("=" * 50)
    
    # Step 1: Prepare data
    basket_binary, basket_full = prepare_market_basket_data(df_clean)
    
    # Step 2: Perform Apriori analysis
    frequent_itemsets, rules = perform_apriori_analysis(
        basket_binary, min_support=min_support, min_confidence=min_confidence)

    # Handle no/empty rules
    if rules is None or getattr(rules, "empty", True):
        print("⚠️ No rules with current thresholds. Retrying with looser thresholds...")
        retry_support = max(min_support / 2, 0.001)
        retry_conf   = max(min_confidence / 2, 0.05)
        frequent_itemsets2, rules2 = perform_apriori_analysis(
            basket_binary, min_support=retry_support, min_confidence=retry_conf)

        if rules2 is None or getattr(rules2, "empty", True):
            print("❌ Still no rules found. Proceeding without market-basket outputs.")
            return {
                'frequent_itemsets': frequent_itemsets if frequent_itemsets is not None else pd.DataFrame(),
                'rules': pd.DataFrame(),
                'basket_binary': basket_binary,
                'recommendation_engine': lambda *args, **kwargs: "No recommendations (no rules).",
                'min_support': min_support,
                'min_confidence': min_confidence
            }
        else:
            frequent_itemsets, rules = frequent_itemsets2, rules2

    # Step 3: Analyze rules
    analyze_association_rules(rules)

    # Step 4: Create visualizations
    visualize_market_basket_results(rules, frequent_itemsets)

    # Step 5: Build recommendation engine
    recommendation_engine = create_product_recommendation_engine(rules)

    # Optional: simple temporal pattern scan
    analyze_temporal_associations(rules, df_clean)
    print("\n✅ MARKET BASKET ANALYSIS COMPLETE!")
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'basket_binary': basket_binary,
        'recommendation_engine': recommendation_engine,
        'min_support': min_support,
        'min_confidence': min_confidence
    }

print("Market Basket Analysis module ready!")
print("Usage: basket_results = perform_market_basket_analysis(df_clean)")
print("\nNote: Adjust min_support and min_confidence parameters based on your dataset size")


## Business Insights and Recommendations for Superstore Analysis

def generate_comprehensive_business_insights(segmentation_results, basket_results, df_clean):
    """
    Generate comprehensive business insights combining customer segmentation and market basket analysis
    """
    print("🎯 COMPREHENSIVE BUSINESS INSIGHTS REPORT")
    print("=" * 60)
    
    rfm_clustered = segmentation_results['rfm_clustered']
    personas = segmentation_results['personas']
    # rules = basket_results['rules']
    rules = basket_results['rules'] if ('rules' in basket_results) else pd.DataFrame()

    # Executive Summary
    print("\n📋 EXECUTIVE SUMMARY")
    print("─" * 30)
    
    total_customers = len(rfm_clustered)
    total_revenue = df_clean['Sales'].sum()
    total_profit = df_clean['Profit'].sum()
    total_orders = df_clean['Order ID'].nunique()
    avg_order_value = df_clean.groupby('Order ID')['Sales'].sum().mean()
    
    print(f"📊 Business Overview:")
    print(f"   • Total Customers: {total_customers:,}")
    print(f"   • Total Revenue: ${total_revenue:,.0f}")
    print(f"   • Total Profit: ${total_profit:,.0f}")
    print(f"   • Total Orders: {total_orders:,}")
    print(f"   • Average Order Value: ${avg_order_value:,.2f}")
    print(f"   • Profit Margin: {total_profit/total_revenue:.1%}")
    
    print(f"\n🎯 Key Findings:")
    print(f"   • Identified {len(personas)} distinct customer segments")
    print(f"   • Discovered {len(rules)} product association rules")
    print(f"   • {len(rules[rules['lift'] > 2])} high-impact cross-selling opportunities")
    
    return total_customers, total_revenue, total_profit, avg_order_value

def analyze_segment_performance(rfm_clustered, personas, df_clean):
    """
    Analyze performance metrics by customer segment
    """
    print("\n💼 CUSTOMER SEGMENT PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Merge transaction data with clusters
    df_with_clusters = df_clean.merge(rfm_clustered[['Customer ID', 'Cluster']], on='Customer ID')
    
    # Calculate segment metrics
    segment_metrics = []
    
    for cluster_id, persona in personas.items():
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        metrics = {
            'Segment': persona['name'],
            'Customer_Count': len(cluster_data['Customer ID'].unique()),
            'Total_Revenue': cluster_data['Sales'].sum(),
            'Total_Profit': cluster_data['Profit'].sum(),
            'Avg_Order_Value': cluster_data.groupby('Order ID')['Sales'].sum().mean(),
            'Orders_Per_Customer': len(cluster_data) / len(cluster_data['Customer ID'].unique()),
            'Profit_Margin': cluster_data['Profit'].sum() / cluster_data['Sales'].sum() if cluster_data['Sales'].sum() > 0 else 0,
            'Revenue_Per_Customer': cluster_data['Sales'].sum() / len(cluster_data['Customer ID'].unique()),
            'Primary_Category': cluster_data['Category'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
            'Primary_Region': cluster_data['Region'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
        }
        segment_metrics.append(metrics)
    
    segment_df = pd.DataFrame(segment_metrics)
    segment_df['Revenue_Share'] = (segment_df['Total_Revenue'] / segment_df['Total_Revenue'].sum()) * 100
    segment_df['Customer_Share'] = (segment_df['Customer_Count'] / segment_df['Customer_Count'].sum()) * 100
    
    print("📈 SEGMENT PERFORMANCE DASHBOARD:")
    print("─" * 80)
    
    for _, row in segment_df.iterrows():
        print(f"\n🏷️  {row['Segment'].upper()}")
        print(f"   Customers: {row['Customer_Count']:,} ({row['Customer_Share']:.1f}% of total)")
        print(f"   Revenue: ${row['Total_Revenue']:,.0f} ({row['Revenue_Share']:.1f}% of total)")
        print(f"   Revenue per Customer: ${row['Revenue_Per_Customer']:,.0f}")
        print(f"   Average Order Value: ${row['Avg_Order_Value']:,.0f}")
        print(f"   Profit Margin: {row['Profit_Margin']:.1%}")
        print(f"   Primary Category: {row['Primary_Category']}")
        print(f"   Primary Region: {row['Primary_Region']}")
    
    return segment_df

def create_targeted_marketing_strategies(personas, segment_df, rules):
    """
    Create targeted marketing strategies for each customer segment
    """
    print("\n🎯 TARGETED MARKETING STRATEGIES")
    print("=" * 40)
    
    # High-value product recommendations from association rules
    high_value_rules = rules[rules['lift'] > 2].sort_values('confidence', ascending=False).head(10)
    
    strategies = {}
    
    for cluster_id, persona in personas.items():
        segment_data = segment_df[segment_df['Segment'] == persona['name']].iloc[0]
        
        print(f"\n🎪 {persona['name'].upper()} SEGMENT STRATEGY")
        print("─" * 50)
        
        print(f"📊 Segment Profile:")
        print(f"   • Size: {segment_data['Customer_Count']:,} customers")
        print(f"   • Revenue Impact: ${segment_data['Total_Revenue']:,.0f}")
        print(f"   • Customer Value: ${segment_data['Revenue_Per_Customer']:,.0f}")
        
        # Strategy based on segment characteristics
        if persona['name'] == 'Champions':
            strategy = {
                'focus': 'Retention & Advocacy',
                'tactics': [
                    'VIP customer program with exclusive benefits',
                    'Early access to new products',
                    'Personalized account management',
                    'Referral incentive programs',
                    'Premium customer support channel'
                ],
                'kpis': ['Customer Lifetime Value', 'Retention Rate', 'Referral Count'],
                'budget_allocation': '25%'
            }
        
        elif persona['name'] == 'Loyal Customers':
            strategy = {
                'focus': 'Upselling & Cross-selling',
                'tactics': [
                    'Category expansion campaigns',
                    'Bundle product recommendations',
                    'Seasonal promotion targeting',
                    'Loyalty point accelerators',
                    'Purchase frequency incentives'
                ],
                'kpis': ['Average Order Value', 'Purchase Frequency', 'Category Penetration'],
                'budget_allocation': '30%'
            }
        
        elif persona['name'] == 'Big Spenders':
            strategy = {
                'focus': 'Frequency Increase',
                'tactics': [
                    'Premium product showcases',
                    'Exclusive high-value product access',
                    'Bulk purchase incentives',
                    'Corporate account development',
                    'Personalized high-value recommendations'
                ],
                'kpis': ['Purchase Frequency', 'Average Order Value', 'Revenue Growth'],
                'budget_allocation': '20%'
            }
        
        elif persona['name'] == 'Potential Loyalists':
            strategy = {
                'focus': 'Engagement & Development',
                'tactics': [
                    'Onboarding email sequences',
                    'Product education content',
                    'First purchase incentives',
                    'Category exploration campaigns',
                    'Progressive engagement rewards'
                ],
                'kpis': ['Engagement Rate', 'Second Purchase Rate', 'Time to Loyalty'],
                'budget_allocation': '15%'
            }
        
        elif persona['name'] == 'At Risk':
            strategy = {
                'focus': 'Win-back & Re-engagement',
                'tactics': [
                    'Win-back discount campaigns',
                    'Product recommendation refresh',
                    'Survey for feedback collection',
                    'Limited-time exclusive offers',
                    'Channel preference optimization'
                ],
                'kpis': ['Reactivation Rate', 'Win-back ROI', 'Engagement Recovery'],
                'budget_allocation': '8%'
            }
        
        else:  # New Customers
            strategy = {
                'focus': 'Acquisition & Onboarding',
                'tactics': [
                    'Welcome series automation',
                    'First-time buyer incentives',
                    'Product recommendation engine',
                    'Category introduction campaigns',
                    'Social proof and reviews'
                ],
                'kpis': ['Conversion Rate', 'First Purchase Value', 'Onboarding Completion'],
                'budget_allocation': '2%'
            }
        
        strategies[persona['name']] = strategy
        
        print(f"🎯 Strategy Focus: {strategy['focus']}")
        print(f"💰 Recommended Budget Allocation: {strategy['budget_allocation']}")
        print(f"📈 Key Performance Indicators:")
        for kpi in strategy['kpis']:
            print(f"   • {kpi}")
        
        print(f"🛠️ Tactical Recommendations:")
        for tactic in strategy['tactics']:
            print(f"   • {tactic}")
    
    return strategies

def calculate_roi_projections(segment_df, strategies):
    """
    Calculate ROI projections for marketing strategies
    """
    print("\n💰 ROI PROJECTIONS & BUSINESS IMPACT")
    print("=" * 45)
    
    total_revenue = segment_df['Total_Revenue'].sum()
    
    # Conservative improvement estimates by segment type
    improvement_rates = {
        'Champions': {'revenue_lift': 0.15, 'retention_improvement': 0.95},
        'Loyal Customers': {'revenue_lift': 0.25, 'retention_improvement': 0.90},
        'Big Spenders': {'revenue_lift': 0.30, 'retention_improvement': 0.85},
        'Potential Loyalists': {'revenue_lift': 0.35, 'retention_improvement': 0.70},
        'At Risk': {'revenue_lift': 0.20, 'retention_improvement': 0.50},
        'New Customers': {'revenue_lift': 0.40, 'retention_improvement': 0.60}
    }
    
    total_projected_lift = 0
    annual_marketing_budget = total_revenue * 0.05  # Assume 5% of revenue for marketing
    
    print("📊 PROJECTED ANNUAL IMPACT:")
    print("─" * 50)
    
    roi_summary = []
    
    for _, row in segment_df.iterrows():
        segment_name = row['Segment']
        if segment_name in improvement_rates:
            current_revenue = row['Total_Revenue']
            customer_count = row['Customer_Count']
            
            # Calculate improvements
            revenue_lift = improvement_rates[segment_name]['revenue_lift']
            retention_rate = improvement_rates[segment_name]['retention_improvement']
            
            # Projected revenue increase
            revenue_increase = current_revenue * revenue_lift
            
            # Marketing budget allocation
            if segment_name in strategies:
                budget_pct = float(strategies[segment_name]['budget_allocation'].strip('%')) / 100
                allocated_budget = annual_marketing_budget * budget_pct
            
                # ROI calculation
                roi = (revenue_increase / allocated_budget) if allocated_budget > 0 else 0
            
                roi_summary.append({
                    'Segment': segment_name,
                    'Current_Revenue': current_revenue,
                    'Projected_Increase': revenue_increase,
                    'Marketing_Budget': allocated_budget,
                    'ROI': roi,
                    'Customers': customer_count
                })
                
                total_projected_lift += revenue_increase
                
                print(f"\n🏷️  {segment_name.upper()}")
                print(f"   Current Annual Revenue: ${current_revenue:,.0f}")
                print(f"   Projected Revenue Increase: ${revenue_increase:,.0f} (+{revenue_lift:.0%})")
                print(f"   Marketing Budget Allocation: ${allocated_budget:,.0f}")
                print(f"   Projected ROI: {roi:.1f}x")
    
    # Overall summary
    total_marketing_budget = sum([item['Marketing_Budget'] for item in roi_summary])
    overall_roi = total_projected_lift / total_marketing_budget if total_marketing_budget > 0 else 0
    
    print(f"\n🎯 OVERALL PROGRAM SUMMARY:")
    print(f"   Total Current Revenue: ${total_revenue:,.0f}")
    print(f"   Total Projected Revenue Increase: ${total_projected_lift:,.0f}")
    print(f"   Total Marketing Investment: ${total_marketing_budget:,.0f}")
    print(f"   Overall Program ROI: {overall_roi:.1f}x")
    print(f"   Revenue Growth Rate: {total_projected_lift/total_revenue:.1%}")
    
    return roi_summary, total_projected_lift, overall_roi

def create_implementation_roadmap(strategies, roi_summary):
    """
    Create a practical implementation roadmap
    """
    print("\n🗺️ IMPLEMENTATION ROADMAP")
    print("=" * 35)
    
    print("📅 PHASE 1: FOUNDATION (Months 1-2)")
    print("─" * 40)
    print("• Set up customer segmentation in CRM system")
    print("• Implement product recommendation engine")
    print("• Create segment-specific email templates")
    print("• Establish KPI tracking dashboard")
    print("• Train customer service team on segments")
    
    print("\n📅 PHASE 2: HIGH-IMPACT SEGMENTS (Months 2-4)")
    print("─" * 45)
    
    # Sort segments by ROI for prioritization
    roi_df = pd.DataFrame(roi_summary)
    if len(roi_df) > 0:
        priority_segments = roi_df.sort_values('ROI', ascending=False)
        
        for _, segment in priority_segments.head(3).iterrows():
            segment_name = segment['Segment']
            print(f"• Launch {segment_name} targeted campaigns (ROI: {segment['ROI']:.1f}x)")
            if segment_name in strategies:
                print(f"  Focus: {strategies[segment_name]['focus']}")
    
    print("\n📅 PHASE 3: OPTIMIZATION (Months 4-6)")
    print("─" * 40)
    print("• A/B test campaign messaging by segment")
    print("• Optimize product recommendation algorithms")
    print("• Refine segment definitions based on performance")
    print("• Expand successful tactics to similar segments")
    print("• Implement advanced personalization")
    
    print("\n📅 PHASE 4: SCALE & INNOVATION (Months 6-12)")
    print("─" * 45)
    print("• Roll out to remaining segments")
    print("• Implement predictive analytics")
    print("• Develop dynamic segmentation")
    print("• Create customer journey automation")
    print("• Launch loyalty program enhancements")
    
    print("\n🎖️ SUCCESS METRICS TO TRACK:")
    print("─" * 30)
    print("• Customer Lifetime Value by segment")
    print("• Segment migration rates (up/down)")
    print("• Cross-sell/upsell conversion rates")
    print("• Campaign ROI by segment")
    print("• Customer satisfaction scores")
    print("• Revenue per segment")
    print("• Customer retention rates")

def generate_executive_summary_report(total_customers, total_revenue, total_projected_lift, overall_roi, personas):
    """
    Generate final executive summary for leadership
    """
    print("\n📋 EXECUTIVE SUMMARY REPORT")
    print("=" * 40)
    
    print("🎯 BUSINESS OPPORTUNITY:")
    print(f"Through advanced customer analytics, we identified ${total_projected_lift:,.0f} in")
    print(f"annual revenue growth opportunity ({total_projected_lift/total_revenue:.1%} increase)")
    print(f"with an overall program ROI of {overall_roi:.1f}x.")
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"• {total_customers:,} customers segmented into {len(personas)} actionable groups")
    print(f"• {len([p for p in personas.values() if 'Champion' in p['name'] or 'Loyal' in p['name']])} high-value segments drive majority of revenue")
    print(f"• Significant cross-selling opportunities identified")
    print(f"• Clear differentiation in customer behaviors and preferences")
    
    print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
    print(f"1. Implement customer segmentation in CRM systems immediately")
    print(f"2. Launch targeted campaigns for high-ROI segments first")
    print(f"3. Deploy product recommendation engine for cross-selling")
    print(f"4. Establish segment-specific retention programs")
    print(f"5. Create personalized customer experience by segment")
    
    print(f"\n⚡ IMMEDIATE ACTIONS (Next 30 days):")
    print(f"• Approve marketing budget allocation by segment")
    print(f"• Begin CRM system integration for segmentation")
    print(f"• Start development of recommendation engine")
    print(f"• Train customer-facing teams on segment strategies")
    print(f"• Establish performance tracking dashboard")

def perform_complete_business_analysis(segmentation_results, basket_results, df_clean):
    """
    Perform complete business analysis combining all insights
    """
    print("🚀 PERFORMING COMPLETE BUSINESS ANALYSIS")
    print("=" * 60)
    
    # Generate comprehensive insights
    total_customers, total_revenue, total_profit, avg_order_value = generate_comprehensive_business_insights(
        segmentation_results, basket_results, df_clean)
    
    # Analyze segment performance
    segment_df = analyze_segment_performance(
        segmentation_results['rfm_clustered'], 
        segmentation_results['personas'], 
        df_clean)
    
    # Create marketing strategies
    strategies = create_targeted_marketing_strategies(
        segmentation_results['personas'], 
        segment_df, 
        basket_results['rules'])
    
    # Calculate ROI projections
    roi_summary, total_projected_lift, overall_roi = calculate_roi_projections(segment_df, strategies)
    
    # Create implementation roadmap
    create_implementation_roadmap(strategies, roi_summary)
    
    # Generate executive summary
    generate_executive_summary_report(
        total_customers, total_revenue, total_projected_lift, 
        overall_roi, segmentation_results['personas'])
    
    print("\n✅ COMPLETE BUSINESS ANALYSIS FINISHED!")
    
    return {
        'segment_performance': segment_df,
        'marketing_strategies': strategies,
        'roi_projections': roi_summary,
        'total_projected_lift': total_projected_lift,
        'overall_roi': overall_roi,
        'business_metrics': {
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'avg_order_value': avg_order_value
        }
    }

print("Business Insights module ready!")
print("Usage: business_results = perform_complete_business_analysis(segmentation_results, basket_results, df_clean)")


## Main Execution Pipeline for Superstore Customer Intelligence Project

def save_results_to_files(segmentation_results, basket_results, business_results, df_clean):
    """
    Save all analysis results to files for documentation and further use
    """
    print("💾 SAVING RESULTS TO FILES")
    print("=" * 30)
    
    try:
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save customer segments
        rfm_clustered = segmentation_results['rfm_clustered']
        rfm_clustered.to_csv(f'customer_segments_{timestamp}.csv', index=False)
        print("✅ Customer segments saved to CSV")
        
        # Save association rules
        rules = basket_results['rules']
        rules_export = rules.copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_export.to_csv(f'association_rules_{timestamp}.csv', index=False)
        print("✅ Association rules saved to CSV")
        
        # Save business summary
        segment_performance = business_results['segment_performance']
        segment_performance.to_csv(f'segment_performance_{timestamp}.csv', index=False)
        print("✅ Segment performance saved to CSV")
        
        # Save cluster centers
        cluster_centers = segmentation_results['cluster_centers']
        cluster_centers.to_csv(f'cluster_centers_{timestamp}.csv', index=False)
        print("✅ Cluster centers saved to CSV")
        
        print(f"\n📁 All results saved with timestamp: {timestamp}")
        
    except Exception as e:
        print(f"⚠️ Error saving files: {e}")
        print("Results are still available in memory for analysis")

def create_business_report_summary():
    """
    Create a formatted business report summary
    """
    report = """
    
    🏢 SUPERSTORE CUSTOMER INTELLIGENCE PROJECT
    ==========================================
    
    📋 PROJECT OVERVIEW
    This analysis leveraged unsupervised machine learning to segment customers 
    and discover product associations, providing actionable insights for marketing 
    strategy and revenue optimization.
    
    🎯 METHODOLOGY
    • Customer Segmentation: RFM analysis + K-Means clustering
    • Market Basket Analysis: Apriori algorithm for association rules
    • Business Intelligence: Performance analysis and strategy development
    
    📊 KEY DELIVERABLES
    1. Customer segment identification and personas
    2. Product association rules and recommendations
    3. Targeted marketing strategies by segment
    4. ROI projections and implementation roadmap
    5. Interactive dashboards and visualizations
    
    💼 BUSINESS IMPACT
    • Identified distinct customer segments with unique characteristics
    • Discovered high-value cross-selling opportunities
    • Developed data-driven marketing strategies
    • Projected significant revenue growth through targeted approaches
    
    🚀 NEXT STEPS
    • Implement customer segmentation in CRM systems
    • Deploy product recommendation engine
    • Launch targeted marketing campaigns
    • Monitor KPIs and optimize strategies
    
    """
    
    return report

def run_complete_analysis(df, min_support=0.01, min_confidence=0.25, save_files=True):
    """
    Run the complete customer intelligence analysis pipeline
    """
    print("🎉 STARTING COMPLETE SUPERSTORE CUSTOMER INTELLIGENCE ANALYSIS")
    print("=" * 80)
    
    try:
        # Phase 1: Data Preprocessing and Customer Segmentation
        print("\n🔄 PHASE 1: CUSTOMER SEGMENTATION")
        segmentation_results = perform_customer_segmentation(df)
        
        # Phase 2: Market Basket Analysis
        print("\n🔄 PHASE 2: MARKET BASKET ANALYSIS")
        basket_results = perform_market_basket_analysis(
            segmentation_results['df_clean'], 
            min_support=min_support, 
            min_confidence=min_confidence
        )
        
        if basket_results is None:
            print("❌ Market basket analysis failed. Continuing with segmentation results only.")
            basket_results = {'rules': pd.DataFrame(), 'frequent_itemsets': pd.DataFrame()}
        
        # Phase 3: Business Intelligence and Strategy
        print("\n🔄 PHASE 3: BUSINESS INTELLIGENCE")
        business_results = perform_complete_business_analysis(
            segmentation_results, basket_results, segmentation_results['df_clean'])
        
        # Phase 4: Save Results
        if save_files:
            print("\n🔄 PHASE 4: SAVING RESULTS")
            save_results_to_files(segmentation_results, basket_results, 
                                business_results, segmentation_results['df_clean'])
        
        # Generate final report
        report_summary = create_business_report_summary()
        print(report_summary)
        
        print("\n🎊 ANALYSIS COMPLETE! 🎊")
        print("=" * 50)
        print("All results are available in the returned dictionary.")
        print("Check the generated CSV files for detailed data export.")
        
        return {
            'segmentation_results': segmentation_results,
            'basket_results': basket_results,
            'business_results': business_results,
            'report_summary': report_summary,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ ERROR IN ANALYSIS PIPELINE: {e}")
        print("Please check your data format and try again.")
        return {'success': False, 'error': str(e)}

def display_quick_start_guide():
    """
    Display quick start guide for using the analysis
    """
    guide = """
    
    🚀 QUICK START GUIDE
    ===================
    
    1. LOAD YOUR DATA:
       df = pd.read_csv('Sample-Superstore.csv', encoding='latin1')
    
    2. RUN COMPLETE ANALYSIS:
       results = run_complete_analysis(df)
    
    3. ACCESS RESULTS:
       • Customer Segments: results['segmentation_results']['rfm_clustered']
       • Association Rules: results['basket_results']['rules']
       • Business Metrics: results['business_results']['segment_performance']
    
    4. ADJUST PARAMETERS (if needed):
       results = run_complete_analysis(df, min_support=0.005, min_confidence=0.2)
    
    5. EXPLORE INDIVIDUAL COMPONENTS:
       • segmentation_results = perform_customer_segmentation(df)
       • basket_results = perform_market_basket_analysis(df_clean)
       • business_results = perform_complete_business_analysis(...)
    
    📊 PARAMETER GUIDELINES:
    • min_support: 0.005-0.02 (0.5%-2% of transactions)
    • min_confidence: 0.2-0.5 (20%-50% confidence)
    • Larger datasets → lower min_support
    • Smaller datasets → higher min_support
    
    🎯 TROUBLESHOOTING:
    • "No frequent itemsets found" → Reduce min_support
    • "No association rules found" → Reduce min_confidence
    • Memory issues → Use data sampling for large datasets
    
    📁 OUTPUT FILES:
    • customer_segments_TIMESTAMP.csv
    • association_rules_TIMESTAMP.csv
    • segment_performance_TIMESTAMP.csv
    • cluster_centers_TIMESTAMP.csv
    
    """
    
    print(guide)

# Initialize the analysis environment
def initialize_analysis_environment():
    """
    Initialize the analysis environment and check requirements
    """
    print("🔧 INITIALIZING SUPERSTORE CUSTOMER INTELLIGENCE ENVIRONMENT")
    print("=" * 65)
    
    # Check if all required modules are available
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'mlxtend', 'plotly'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - Available")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - Missing")
    
    if missing_modules:
        print(f"\n⚠️  Install missing modules: pip install {' '.join(missing_modules)}")
        return False
    
    print("\n✅ All required modules are available!")
    print("\n📚 AVAILABLE FUNCTIONS:")
    print("─" * 25)
    print("• run_complete_analysis(df) - Full analysis pipeline")
    print("• perform_customer_segmentation(df) - Customer segmentation only") 
    print("• perform_market_basket_analysis(df_clean) - Market basket only")
    print("• perform_complete_business_analysis(...) - Business insights only")
    print("• display_quick_start_guide() - Show usage guide")
    
    return True

# Main execution check
if __name__ == "__main__":
    # Initialize environment
    env_ready = initialize_analysis_environment()
    
    if env_ready:
        # Display quick start guide
        display_quick_start_guide()
        
        print("\n🎯 READY TO ANALYZE!")
        print("Load your Superstore dataset and run: results = run_complete_analysis(df)")
    else:
        print("\n❌ Environment setup incomplete. Please install missing modules.")

# Example usage template
example_usage = '''

# EXAMPLE USAGE:
# ==============

# 1. Load your Superstore dataset
df = pd.read_csv('Sample-Superstore.csv', encoding='latin1')

# 2. Run complete analysis
results = run_complete_analysis(df)

# 3. Access specific results
customer_segments = results['segmentation_results']['rfm_clustered']
association_rules = results['basket_results']['rules']
business_metrics = results['business_results']['segment_performance']
personas = results['segmentation_results']['personas']

# 4. Display key insights
print("Customer Segments:")
print(customer_segments['Cluster'].value_counts())

print("\nTop Association Rules:")
print(association_rules.head())

print("\nSegment Performance:")
print(business_metrics)

'''

print("\n" + "="*60)
print("🎯 SUPERSTORE CUSTOMER INTELLIGENCE SYSTEM READY!")
print("="*60)
print("All analysis modules loaded successfully!")
print("Use run_complete_analysis(df) to start your analysis.")
print("="*60)
