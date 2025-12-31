import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-hierarchical-clustering',
	title: 'Hierarchical Clustering',
	difficulty: 'medium',
	tags: ['sklearn', 'clustering', 'hierarchical'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Hierarchical Clustering

Hierarchical clustering builds a tree of clusters through agglomerative merging.

## Task

Implement three functions:
1. \`train_agglomerative(X, n_clusters, linkage)\` - Train agglomerative clustering
2. \`get_linkage_matrix(X)\` - Return linkage matrix for dendrogram
3. \`cut_dendrogram(Z, n_clusters)\` - Cut dendrogram at n clusters

## Example

\`\`\`python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

model = train_agglomerative(X, n_clusters=2, linkage='ward')
Z = get_linkage_matrix(X)
labels = cut_dendrogram(Z, n_clusters=2)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

def train_agglomerative(X: np.ndarray, n_clusters: int, linkage_type: str = 'ward'):
    """Train agglomerative clustering. Return fitted model."""
    # Your code here
    pass

def get_linkage_matrix(X: np.ndarray) -> np.ndarray:
    """Return linkage matrix for dendrogram visualization."""
    # Your code here
    pass

def cut_dendrogram(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut dendrogram at n clusters. Return cluster labels."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

def train_agglomerative(X: np.ndarray, n_clusters: int, linkage_type: str = 'ward'):
    """Train agglomerative clustering. Return fitted model."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    model.fit(X)
    return model

def get_linkage_matrix(X: np.ndarray) -> np.ndarray:
    """Return linkage matrix for dendrogram visualization."""
    return linkage(X, method='ward')

def cut_dendrogram(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut dendrogram at n clusters. Return cluster labels."""
    return fcluster(Z, n_clusters, criterion='maxclust')
`,

	testCode: `import numpy as np
import unittest

class TestHierarchicalClustering(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    def test_train_returns_model(self):
        model = train_agglomerative(self.X, 2)
        self.assertIsNotNone(model)

    def test_model_has_labels(self):
        model = train_agglomerative(self.X, 2)
        self.assertEqual(len(model.labels_), 6)

    def test_linkage_matrix_shape(self):
        Z = get_linkage_matrix(self.X)
        self.assertEqual(Z.shape[0], 5)  # n-1 merges
        self.assertEqual(Z.shape[1], 4)

    def test_cut_dendrogram_returns_labels(self):
        Z = get_linkage_matrix(self.X)
        labels = cut_dendrogram(Z, 2)
        self.assertEqual(len(labels), 6)

    def test_correct_cluster_count(self):
        Z = get_linkage_matrix(self.X)
        labels = cut_dendrogram(Z, 3)
        self.assertEqual(len(set(labels)), 3)

    def test_linkage_matrix_returns_numpy(self):
        Z = get_linkage_matrix(self.X)
        self.assertIsInstance(Z, np.ndarray)

    def test_cut_dendrogram_returns_numpy(self):
        Z = get_linkage_matrix(self.X)
        labels = cut_dendrogram(Z, 2)
        self.assertIsInstance(labels, np.ndarray)

    def test_different_linkage_types(self):
        model_ward = train_agglomerative(self.X, 2, 'ward')
        model_complete = train_agglomerative(self.X, 2, 'complete')
        self.assertIsNotNone(model_ward)
        self.assertIsNotNone(model_complete)

    def test_different_n_clusters(self):
        model_2 = train_agglomerative(self.X, 2)
        model_3 = train_agglomerative(self.X, 3)
        self.assertEqual(model_2.n_clusters, 2)
        self.assertEqual(model_3.n_clusters, 3)

    def test_labels_unique_count(self):
        model = train_agglomerative(self.X, 3)
        self.assertEqual(len(set(model.labels_)), 3)
`,

	hint1: 'Use AgglomerativeClustering(n_clusters=n, linkage=type).fit(X)',
	hint2: 'Use scipy.cluster.hierarchy.linkage and fcluster for dendrogram operations',

	whyItMatters: `Hierarchical clustering is useful for:

- **Dendrograms**: Visualize cluster hierarchy
- **No k needed upfront**: Decide clusters after seeing structure
- **Taxonomy**: Natural for hierarchical data
- **Interpretability**: Understand how clusters merge

Essential for exploratory clustering analysis.`,

	translations: {
		ru: {
			title: 'Иерархическая кластеризация',
			description: `# Иерархическая кластеризация

Иерархическая кластеризация строит дерево кластеров через агломеративное слияние.

## Задача

Реализуйте три функции:
1. \`train_agglomerative(X, n_clusters, linkage)\` - Обучить агломеративную кластеризацию
2. \`get_linkage_matrix(X)\` - Вернуть матрицу связей для дендрограммы
3. \`cut_dendrogram(Z, n_clusters)\` - Обрезать дендрограмму на n кластеров

## Пример

\`\`\`python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

model = train_agglomerative(X, n_clusters=2, linkage='ward')
Z = get_linkage_matrix(X)
labels = cut_dendrogram(Z, n_clusters=2)
\`\`\``,
			hint1: 'Используйте AgglomerativeClustering(n_clusters=n, linkage=type).fit(X)',
			hint2: 'Используйте scipy.cluster.hierarchy.linkage и fcluster',
			whyItMatters: `Иерархическая кластеризация полезна для:

- **Дендрограммы**: Визуализация иерархии кластеров
- **Не нужен k заранее**: Решите после просмотра структуры
- **Таксономия**: Естественна для иерархических данных`,
		},
		uz: {
			title: 'Ierarxik klasterlash',
			description: `# Ierarxik klasterlash

Ierarxik klasterlash aglomerativ birlashtirish orqali klasterlar daraxtini quradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_agglomerative(X, n_clusters, linkage)\` - Aglomerativ klasterlashni o'rgatish
2. \`get_linkage_matrix(X)\` - Dendrogramma uchun bog'lanish matritsasini qaytarish
3. \`cut_dendrogram(Z, n_clusters)\` - Dendrogrammani n klasterga kesish

## Misol

\`\`\`python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

model = train_agglomerative(X, n_clusters=2, linkage='ward')
Z = get_linkage_matrix(X)
labels = cut_dendrogram(Z, n_clusters=2)
\`\`\``,
			hint1: "AgglomerativeClustering(n_clusters=n, linkage=type).fit(X) dan foydalaning",
			hint2: "scipy.cluster.hierarchy.linkage va fcluster dan foydalaning",
			whyItMatters: `Ierarxik klasterlash quyidagilar uchun foydali:

- **Dendrogrammalar**: Klaster ierarxiyasini vizualizatsiya qilish
- **Oldindan k kerak emas**: Strukturani ko'rgandan keyin qaror qiling
- **Taksonomiya**: Ierarxik ma'lumotlar uchun tabiiy`,
		},
	},
};

export default task;
