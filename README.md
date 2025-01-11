

# Wine Quality Clustering using K-Means and PCA

This project analyzes and clusters wine samples from the **Wine Quality dataset** based on their chemical properties. It employs **K-Means Clustering** for unsupervised learning, **PCA** for dimensionality reduction, and various metrics like the **Silhouette Score** to evaluate clustering performance.

## Project Overview

This project demonstrates:
1. **Data Preprocessing**:
   - Standardizing numerical features for clustering.
   - Handling data transformations to optimize clustering results.
2. **Clustering Analysis**:
   - Using the **Elbow Method** to determine the optimal number of clusters.
   - Evaluating clusters with the **Silhouette Score**.
3. **Dimensionality Reduction**:
   - Applying **Principal Component Analysis (PCA)** to reduce feature dimensions.
   - Visualizing clusters in a lower-dimensional space for interpretability.
4. **Visualization**:
   - Displaying clusters and centroids in feature and principal component spaces.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## Dataset

The dataset contains chemical properties of red wine, such as:
- **Features**:
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, and `alcohol`.
- **Target Variable**:
  - The `quality` column was dropped to perform unsupervised clustering.

---

## Project Workflow

### Step 1: Data Preprocessing
- Standardized the dataset using **StandardScaler** to normalize features for clustering.

### Step 2: Optimal Clusters (Elbow Method)
- Calculated the **Within-Cluster Sum of Squares (WCSS)** for different cluster counts to find the optimal number of clusters:
```python
for k in range(1, 15):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit_predict(scaled_df)
    wcss.append(model.inertia_)
plt.plot(wcss)
```
- The elbow point was identified at **6 clusters**.

### Step 3: K-Means Clustering
- Applied **K-Means Clustering** with the optimal cluster count.
- Added cluster labels to the dataset:
```python
model = KMeans(n_clusters=6, n_init=10, random_state=42)
label = model.fit_predict(scaled_df)
df['cluster'] = label
```

### Step 4: Dimensionality Reduction with PCA
1. Performed **PCA** to identify the top components explaining variance in the dataset:
   - Retained **4 components**, explaining approximately **80% variance**.
   - Visualized the cumulative explained variance:
   ```python
   plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
   ```
2. Re-applied K-Means on PCA-reduced data for clustering:
   - Evaluated clusters using the **Silhouette Score** for varying cluster counts.

### Step 5: Visualization
- Visualized clusters and centroids in both feature space and PCA-reduced space:
```python
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels_pca, cmap='viridis', alpha=0.6)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, label='centroids')
plt.title("KMeans on PCA")
```

---

## Results

### Optimal Clusters
- The **Elbow Method** identified **6 clusters** as optimal for the dataset.
- Re-running K-Means on PCA-reduced data produced well-separated clusters.

### Silhouette Score
- Clustering on PCA-reduced data achieved the highest silhouette score of **0.29** with **5 clusters**, indicating moderately well-separated clusters.

### Visualizations
1. **Feature Space Clustering**:
   - Visualized clusters in the original feature space.
2. **PCA-Reduced Space Clustering**:
   - Visualized clusters in a 2D PCA-reduced space for interpretability.

---

## Conclusion

This project demonstrates:
- The importance of feature scaling for clustering algorithms like K-Means.
- The use of **PCA** for dimensionality reduction to improve clustering performance and visualization.
- Metrics like **WCSS** and **Silhouette Score** to evaluate and optimize clusters.

### Future Improvements:
- Test advanced clustering algorithms like **DBSCAN** or **Hierarchical Clustering**.
- Analyze the influence of different PCA components on clustering results.
- Compare results with supervised learning by incorporating the `quality` target variable.

---

## License

This project is licensed under the MIT License.

---

