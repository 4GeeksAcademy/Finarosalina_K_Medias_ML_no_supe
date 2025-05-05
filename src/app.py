#!/usr/bin/env python
# coding: utf-8

# # Explore here

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import warnings


# In[7]:


data= pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
data.head()


# In[8]:


data.to_csv('/workspaces/Finarosalina_K_Medias_ML_no_supe/data/raw/raw_data.csv')


# In[9]:


# solo nos interesan las columnas Latitude, Longitude y MedInc.
data.drop(columns=['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedHouseVal'], inplace=True)


# In[10]:


data.describe()


# Viendo los valores de MedInc, está claro que tiene outliers por la parte alta, conviene observarlos y si no son demasiados eliminarlos.

# In[11]:


sns.boxplot(x=data['MedInc'])
plt.title("Distribución de MedInc (Ingreso Mediano)")
plt.show()


# In[12]:


# Filtrar por debajo del percentil 95
data_filtered = data[data['MedInc'] < data['MedInc'].quantile(0.95)]
data_filtered.shape


# In[13]:


plt.figure(figsize=(5, 4))

#  Longitud vs Latitud, codificando MedInc con color
scatter = plt.scatter(data_filtered['Longitude'], data_filtered['Latitude'],
                      c=data_filtered['MedInc'], cmap='viridis', s=50)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribución geográfica codificada por ingreso (MedInc)")
plt.colorbar(scatter, label='MedInc')  # Muestra la escala de colores

plt.show()


# In[14]:


X = data_filtered[['Latitude', 'Longitude', 'MedInc']]


# In[15]:


X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  


# In[16]:


inercias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    inercias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inercias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del codo para encontrar el k óptimo')
plt.grid(True)
plt.show()


# In[17]:


# viendo la gráfica dudaba entre k=2 y K=4, pero se consigue mejor Silhouette Score con k=2

kmeans = KMeans(n_clusters=2, random_state=42)
train_labels = kmeans.fit_predict(X_train_scaled)


# In[18]:


from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_train_scaled, train_labels)
print(f"Silhouette Score para el conjunto de entrenamiento: {silhouette_avg}")


# In[19]:


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled) 

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_pca)
train_labels = kmeans.labels_

plt.figure(figsize=(6, 4))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_labels, cmap='Set1', s=50)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters con PCA (2D) en el conjunto de entrenamiento")
plt.colorbar(scatter, label='Cluster')
plt.show()

print(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_)}")



# Visualmetne se aprecia la mejora, además que con una componente menos se explican el 97% de los datos, lo que es logico, pq hay una cierta relación.

# In[20]:


# Pero aunque tenga peor score, el proyecto pide crear y entrenar K-Means con k=6

# Entrenamiento de K-Means con k=6

kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init=10)  # n_init para evitar warnings
clusters = kmeans_6.fit_predict(X_train_scaled)


X_train['cluster'] = pd.Categorical(clusters)

print("\nMuestra de datos con clusters asignados:")
print(X_train[['Latitude', 'Longitude', 'MedInc', 'cluster']].head())
print("\nDistribución de clusters:")
print(X_train['cluster'].value_counts().sort_index())
plt.figure(figsize=(9, 6))
plt.scatter(X_train['Longitude'], X_train['Latitude'], 
            c=X_train['cluster'], cmap='viridis', s=20, alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters en Datos de Entrenamiento (k=6)')

# Mostrar centros de clusters (en escala original)
centers_original = scaler.inverse_transform(kmeans_6.cluster_centers_)
plt.scatter(centers_original[:, 1], centers_original[:, 0], 
            c='red', s=200, marker='X', alpha=0.8, label='Centroides')
plt.legend()
plt.show()

print(f"Silhouette Score (train): {silhouette_score(X_train_scaled, X_train['cluster']):.4f}")


X_test['cluster'] = kmeans_6.predict(X_test_scaled)

# Visualización conjunta
plt.figure(figsize=(9, 6))
plt.scatter(X_train['Longitude'], X_train['Latitude'], 
            c=X_train['cluster'], cmap='viridis', s=20, alpha=0.5, label='Train')
plt.scatter(X_test['Longitude'], X_test['Latitude'], 
            c=X_test['cluster'], cmap='viridis', s=50, marker='x', label='Test')
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Comparación Train vs Test')
plt.legend()
plt.show()

#  test
print(f"Silhouette Score (test): {silhouette_score(X_test_scaled, X_test['cluster']):.4f}")

# Análisis comparativo
print("\nDistribución en train:")
print(X_train['cluster'].value_counts().sort_index())
print("\nDistribución en test:")
print(X_test['cluster'].value_counts().sort_index())


# In[21]:


plt.figure(figsize=(9, 5))
scatter = plt.scatter(
    x=X_train['Longitude'],
    y=X_train['Latitude'],
    c=X_train['cluster'].cat.codes,  # códigos numéricos de las categorías
    cmap='tab10',  # Mejor paleta para distinguir clusters
    s=15,
    alpha=0.7,
    edgecolor='k',
    linewidth=0.3
)

# Añadir centroides
centers = scaler.inverse_transform(kmeans_6.cluster_centers_)
plt.scatter(
    x=centers[:, 1],  # Longitude
    y=centers[:, 0],  # Latitude
    c='black',
    marker='X',
    s=200,
    label='Centroides'
)

plt.colorbar(scatter, label='Cluster', ticks=range(6))
plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
plt.title('Distribución Geográfica de Clusters de Viviendas (k=6)', fontsize=14)
plt.grid(alpha=0.2)
plt.legend()

# Anotar los centroides
for i, center in enumerate(centers):
    plt.annotate(
        f'C{i}',
        xy=(center[1], center[0]),
        xytext=(5, 5),
        textcoords='offset points',
        weight='bold'
    )

plt.tight_layout()
plt.show()


# In[22]:


# Análisis de ingresos por cluster
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='MedInc', data=X_train)
plt.title('Distribución de Ingresos por Cluster')
plt.show()


# In[23]:


import joblib

joblib.dump(kmeans_6, 'kmeans_6_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# 
# A la vista de los gráficos yo diría que sobran centroides, porque hay varios conjuntos de datos que se solapan, por lo que yo lo haría de nuevo con K= 2

# In[24]:


# 1. Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 2. Entrenamiento de K-Means con K=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
train_labels = kmeans.fit_predict(X_train_pca)

# 3. Asignación de clusters al DataFrame (CORRECCIÓN: usando train_labels en lugar de clusters)
X_train['cluster'] = pd.Categorical(train_labels)

# 4. Visualización mejorada
plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    X_train_pca[:, 0], 
    X_train_pca[:, 1], 
    c=X_train['cluster'].cat.codes,  # Usando los códigos categóricos
    cmap='Set1', 
    s=50,
    alpha=0.8,
    edgecolor='k',
    linewidth=0.5
)

# Centroides con anotaciones
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='yellow',
    marker='X',
    s=250,
    edgecolor='k',
    linewidth=1.5,
    label='Centroides'
)

# Añadir etiquetas a los centroides
for i, centroid in enumerate(centroids):
    plt.text(
        centroid[0], 
        centroid[1]+0.1,  # Pequeño desplazamiento vertical
        f'Centroide {i}',
        fontsize=10,
        ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
    )

plt.xlabel("Componente Principal 1", fontsize=12)
plt.ylabel("Componente Principal 2", fontsize=12)
plt.title("Clusters con PCA (2D) - K=2\nVarianza Total Explicada: {:.2%}".format(
    np.sum(pca.explained_variance_ratio_)), fontsize=14)
plt.colorbar(scatter, label='Cluster', ticks=[0, 1])
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Análisis de varianza
print("\nAnálisis de Componentes Principales:")
print("----------------------------------")
print(f"Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Varianza total explicada: {np.sum(pca.explained_variance_ratio_):.2%}")

# 6. Interpretación de componentes
print("\nContribución de variables a los componentes:")
components_df = pd.DataFrame(
    pca.components_,
    columns=['Latitude', 'Longitude', 'MedInc'],
    index=['PC1', 'PC2']
)
print(components_df)

# 7. Proyección del test al espacio PCA entrenado
X_test_pca = pca.transform(X_test_scaled)

# 8. Predecir clusters para test
test_labels = kmeans.predict(X_test_pca)

# 9. Asignar los clusters al DataFrame de test
X_test['cluster'] = pd.Categorical(test_labels)

# 10. Visualización de los clusters en el espacio PCA (Train + Test)
plt.figure(figsize=(7, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
            c=train_labels, cmap='Set1', s=30, alpha=0.6, label='Train')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
            c=test_labels, cmap='Set1', marker='x', s=50, label='Test')
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='yellow', s=250, marker='X', edgecolor='k', linewidth=1.5, label='Centroides')
plt.title('Clusters (K=2) en PCA - Train + Test')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 11. Guardar el modelo y PCA (si aún no lo has hecho)
import joblib
joblib.dump(kmeans, 'kmeans_k2_model.pkl')
joblib.dump(pca, 'pca_k2_model.pkl')
joblib.dump(scaler, 'scaler_k2.pkl')


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Entradas para modelos supervisados
X_train_sup = X_train_pca
X_test_sup = X_test_pca
y_train = train_labels
y_test = test_labels

# 2. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sup, y_train)
rf_preds = rf.predict(X_test_sup)

print("🔎 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))

# 3. K-Nearest Neighbors (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sup, y_train)
knn_preds = knn.predict(X_test_sup)

print("\n🔎 K-Nearest Neighbors Results:")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_preds))
print("Classification Report:\n", classification_report(y_test, knn_preds))



# In[29]:


import os

save_path = '/workspaces/Finarosalina_K_Medias_ML_no_supe/models'
os.makedirs(save_path, exist_ok=True)

# Guardar modelos
joblib.dump(rf, f'{save_path}/random_forest_model.pkl')
joblib.dump(knn, f'{save_path}/knn_model.pkl')


# In[32]:


import nbformat
from nbconvert import PythonExporter


notebook_path = "/workspaces/Finarosalina_K_Medias_ML_no_supe/src/explore.ipynb"

python_script_path = "/workspaces/Finarosalina_K_Medias_ML_no_supe/src/app.py"


with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(notebook)


with open(python_script_path, "w", encoding="utf-8") as f:
    f.write(python_code)

print(f"Notebook convertido y guardado como: {python_script_path}")

