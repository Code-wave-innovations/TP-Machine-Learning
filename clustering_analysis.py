import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from scipy.stats import f_oneway

# 1. Chargement des données
df = pd.read_csv('Mall_Customers.csv')

# Vérification des valeurs manquantes
print("Valeurs manquantes :")
print(df.isnull().sum())

# 2. Prétraitement et Feature Engineering
# Catégorisation de l'âge
def categorize_age(age):
    if age < 20:
        return 'Très jeune'
    elif 20 <= age < 35:
        return 'Jeune'
    elif 35 <= age < 50:
        return 'Adulte'
    elif 50 <= age < 65:
        return 'Mature'
    else:
        return 'Senior'

df['Age_Category'] = df['Age'].apply(categorize_age)
print("\nRépartition par catégorie d'âge :")
print(df['Age_Category'].value_counts())

# Sélection des features pour le clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 3. EDA Univariée
# Histogrammes
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogramme de {feature}')
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot de {feature}')
plt.tight_layout()
plt.show()

# Matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de Corrélation')
plt.show()

# Scatter plot matrix
sns.pairplot(df[features])
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()

# 4. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Fonctions existantes (test_cluster_stability, bootstrap_clustering, etc.)
def test_cluster_stability(X, max_clusters=6):
    stability_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        silhouette_avg = silhouette_score(X, kmeans.fit_predict(X))
        stability_scores.append(silhouette_avg)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), stability_scores, marker='o')
    plt.title('Stabilité des Clusters')
    plt.xlabel('Nombre de Clusters')
    plt.ylabel('Score de Silhouette')
    plt.show()
    
    return stability_scores

def bootstrap_clustering(X, n_clusters, n_iterations=100):
    cluster_labels_list = []
    
    for _ in range(n_iterations):
        X_resampled = resample(X, replace=True, n_samples=len(X))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_resampled)
        cluster_labels_list.append(cluster_labels)
    
    stability = np.mean([np.allclose(cluster_labels_list[i], cluster_labels_list[j]) 
                         for i in range(len(cluster_labels_list)) 
                         for j in range(i+1, len(cluster_labels_list))])
    
    print(f"Stabilité des clusters pour {n_clusters} clusters : {stability:.2%}")
    return stability

def test_cluster_significance(X, cluster_labels, features):
    f_stats = []
    p_values = []
    
    for feature in features:
        cluster_groups = [X[cluster_labels == i][feature] for i in np.unique(cluster_labels)]
        
        f_stat, p_value = f_oneway(*cluster_groups)
        f_stats.append(f_stat)
        p_values.append(p_value)
    
    results_df = pd.DataFrame({
        'Feature': features,
        'F-Statistic': f_stats,
        'p-value': p_values
    })
    
    print("Test de significativité des clusters :")
    print(results_df)
    return results_df

def plot_cluster_variations(X, max_clusters=6):
    plt.figure(figsize=(15, 5))
    
    # Méthode du coude
    plt.subplot(1, 3, 1)
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Méthode du Coude')
    plt.xlabel('Nombre de Clusters')
    plt.ylabel('Inertie')
    
    # Score de Silhouette
    plt.subplot(1, 3, 2)
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Score de Silhouette')
    plt.xlabel('Nombre de Clusters')
    plt.ylabel('Score')
    
    # Variance expliquée par PCA
    plt.subplot(1, 3, 3)
    explained_variances = []
    for n_components in range(1, len(features) + 1):
        pca = PCA(n_components=n_components)
        pca.fit(X)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
    plt.plot(range(1, len(features) + 1), explained_variances, marker='o')
    plt.title('Variance Expliquée (PCA)')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Expliquée')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

def cluster_insights(df, cluster_labels, features):
    # Profil des clusters
    cluster_profile = df.copy()
    cluster_profile['Cluster'] = cluster_labels
    
    print("\nProfil des clusters :")
    print(cluster_profile.groupby('Cluster')[features].mean())
    
    # Distribution par genre dans chaque cluster
    print("\nRépartition par genre dans chaque cluster :")
    print(cluster_profile.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0))
    
    # Suggestions d'amélioration
    print("\nSuggestions d'amélioration :")
    print("1. Affiner les catégories d'âge")
    print("2. Explorer d'autres algorithmes de clustering")
    print("3. Collecter des données démographiques supplémentaires")

# 6. Réduction de dimension et visualisation des clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 7. Clustering
print("Test de stabilité des clusters")
stability_results = test_cluster_stability(X_scaled)

print("\nTest de robustesse par bootstrapping")
for n_clusters in range(2, 7):
    bootstrap_clustering(X_scaled, n_clusters)

print("\nClustering avec 4 clusters")
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Visualisation des clusters en 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Clusters de Clients (PCA)')
plt.colorbar(scatter)
plt.xlabel('Première Composante Principale')
plt.ylabel('Deuxième Composante Principale')
plt.show()

# 8. Analyse des centroïdes
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df.index.name = 'Cluster'
print("\nCentroïdes des clusters :")
print(centroid_df)

# Test de significativité
features_to_test = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
significance_results = test_cluster_significance(df[features_to_test], cluster_labels, features_to_test)

# Visualisation des variations de clusters
plot_cluster_variations(X_scaled)

# Insights finaux
cluster_insights(df, cluster_labels, features)