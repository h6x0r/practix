import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-kmeans',
	title: 'K-Means Clustering',
	difficulty: 'easy',
	tags: ['sklearn', 'clustering', 'kmeans'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# K-Means Clustering

K-Means partitions data into k clusters by minimizing within-cluster variance.

## Task

Implement three functions:
1. \`train_kmeans(X, n_clusters)\` - Train K-Means and return model
2. \`get_cluster_centers(model)\` - Return cluster centroids
3. \`find_optimal_k(X, k_range)\` - Find best k using elbow method (inertia)

## Example

\`\`\`python
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

model = train_kmeans(X, n_clusters=2)
centers = get_cluster_centers(model)  # [[1, 2], [10, 2]]
inertias = find_optimal_k(X, range(1, 6))  # {1: 100, 2: 10, ...}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.cluster import KMeans

def train_kmeans(X: np.ndarray, n_clusters: int):
    """Train K-Means clustering. Return fitted model."""
    # Your code here
    pass

def get_cluster_centers(model) -> np.ndarray:
    """Return cluster centroids."""
    # Your code here
    pass

def find_optimal_k(X: np.ndarray, k_range) -> dict:
    """Find inertia for each k. Return {k: inertia}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.cluster import KMeans

def train_kmeans(X: np.ndarray, n_clusters: int):
    """Train K-Means clustering. Return fitted model."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model

def get_cluster_centers(model) -> np.ndarray:
    """Return cluster centroids."""
    return model.cluster_centers_

def find_optimal_k(X: np.ndarray, k_range) -> dict:
    """Find inertia for each k. Return {k: inertia}."""
    inertias = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias[k] = model.inertia_
    return inertias
`,

	testCode: `import numpy as np
import unittest

class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    def test_train_returns_model(self):
        model = train_kmeans(self.X, 2)
        self.assertIsNotNone(model)

    def test_model_has_labels(self):
        model = train_kmeans(self.X, 2)
        self.assertEqual(len(model.labels_), 6)

    def test_get_cluster_centers_shape(self):
        model = train_kmeans(self.X, 2)
        centers = get_cluster_centers(model)
        self.assertEqual(centers.shape, (2, 2))

    def test_find_optimal_k_returns_dict(self):
        result = find_optimal_k(self.X, range(1, 4))
        self.assertIsInstance(result, dict)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_inertia_decreases(self):
        result = find_optimal_k(self.X, range(1, 4))
        self.assertGreater(result[1], result[2])

    def test_two_clusters_separates(self):
        model = train_kmeans(self.X, 2)
        labels = model.labels_
        # First 3 should be same cluster, last 3 different
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])

    def test_centers_returns_numpy(self):
        model = train_kmeans(self.X, 2)
        centers = get_cluster_centers(model)
        self.assertIsInstance(centers, np.ndarray)

    def test_model_can_predict(self):
        model = train_kmeans(self.X, 2)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_inertia_non_negative(self):
        result = find_optimal_k(self.X, range(1, 4))
        for inertia in result.values():
            self.assertGreaterEqual(inertia, 0)

    def test_different_k_values(self):
        model_2 = train_kmeans(self.X, 2)
        model_3 = train_kmeans(self.X, 3)
        self.assertEqual(model_2.n_clusters, 2)
        self.assertEqual(model_3.n_clusters, 3)
`,

	hint1: 'Use KMeans(n_clusters=k).fit(X) for clustering',
	hint2: 'Use model.inertia_ for within-cluster sum of squares',

	whyItMatters: `K-Means is fundamental for:

- **Customer segmentation**: Group customers by behavior
- **Image compression**: Reduce color palette
- **Feature engineering**: Create cluster-based features
- **Anomaly detection**: Points far from centroids

The most used clustering algorithm.`,

	translations: {
		ru: {
			title: 'Кластеризация K-Means',
			description: `# Кластеризация K-Means

K-Means разбивает данные на k кластеров минимизируя внутрикластерную дисперсию.

## Задача

Реализуйте три функции:
1. \`train_kmeans(X, n_clusters)\` - Обучить K-Means и вернуть модель
2. \`get_cluster_centers(model)\` - Вернуть центроиды кластеров
3. \`find_optimal_k(X, k_range)\` - Найти лучший k методом локтя

## Пример

\`\`\`python
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

model = train_kmeans(X, n_clusters=2)
centers = get_cluster_centers(model)  # [[1, 2], [10, 2]]
inertias = find_optimal_k(X, range(1, 6))  # {1: 100, 2: 10, ...}
\`\`\``,
			hint1: 'Используйте KMeans(n_clusters=k).fit(X) для кластеризации',
			hint2: 'Используйте model.inertia_ для внутрикластерной суммы квадратов',
			whyItMatters: `K-Means фундаментален для:

- **Сегментация клиентов**: Группировка по поведению
- **Сжатие изображений**: Уменьшение цветовой палитры
- **Feature engineering**: Создание признаков на основе кластеров`,
		},
		uz: {
			title: 'K-Means klasterlash',
			description: `# K-Means klasterlash

K-Means klaster ichidagi dispersiyani minimallashtirish orqali ma'lumotlarni k klasterga ajratadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_kmeans(X, n_clusters)\` - K-Means ni o'rgatish va modelni qaytarish
2. \`get_cluster_centers(model)\` - Klaster markazlarini qaytarish
3. \`find_optimal_k(X, k_range)\` - Tirsak usuli bilan eng yaxshi k ni topish

## Misol

\`\`\`python
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

model = train_kmeans(X, n_clusters=2)
centers = get_cluster_centers(model)  # [[1, 2], [10, 2]]
inertias = find_optimal_k(X, range(1, 6))  # {1: 100, 2: 10, ...}
\`\`\``,
			hint1: "Klasterlash uchun KMeans(n_clusters=k).fit(X) dan foydalaning",
			hint2: "Klaster ichidagi kvadratlar yig'indisi uchun model.inertia_ dan foydalaning",
			whyItMatters: `K-Means quyidagilar uchun asosiy:

- **Mijozlarni segmentatsiya qilish**: Xatti-harakatlar bo'yicha guruhlash
- **Rasm siqish**: Rang palitrasini kamaytirish
- **Feature engineering**: Klaster asosida xususiyatlar yaratish`,
		},
	},
};

export default task;
