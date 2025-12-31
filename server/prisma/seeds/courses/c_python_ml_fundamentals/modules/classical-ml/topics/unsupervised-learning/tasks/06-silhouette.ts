import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-silhouette',
	title: 'Cluster Evaluation',
	difficulty: 'medium',
	tags: ['sklearn', 'clustering', 'evaluation'],
	estimatedTime: '12m',
	isPremium: false,
	order: 6,
	description: `# Cluster Evaluation

Evaluate clustering quality using silhouette score and other metrics.

## Task

Implement three functions:
1. \`compute_silhouette(X, labels)\` - Compute silhouette score
2. \`silhouette_per_sample(X, labels)\` - Get per-sample silhouette values
3. \`find_optimal_clusters(X, k_range)\` - Find best k using silhouette

## Example

\`\`\`python
from sklearn.metrics import silhouette_score, silhouette_samples

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
labels = np.array([0, 0, 0, 1, 1, 1])

score = compute_silhouette(X, labels)  # ~0.7
per_sample = silhouette_per_sample(X, labels)
best_k = find_optimal_clusters(X, range(2, 5))  # 2
\`\`\``,

	initialCode: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score. Return score (-1 to 1)."""
    # Your code here
    pass

def silhouette_per_sample(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Get per-sample silhouette values. Return array of scores."""
    # Your code here
    pass

def find_optimal_clusters(X: np.ndarray, k_range) -> int:
    """Find best k using silhouette score. Return optimal k."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score. Return score (-1 to 1)."""
    return silhouette_score(X, labels)

def silhouette_per_sample(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Get per-sample silhouette values. Return array of scores."""
    return silhouette_samples(X, labels)

def find_optimal_clusters(X: np.ndarray, k_range) -> int:
    """Find best k using silhouette score. Return optimal k."""
    best_k = 2
    best_score = -1
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k
`,

	testCode: `import numpy as np
import unittest

class TestClusterEvaluation(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        self.labels = np.array([0, 0, 0, 1, 1, 1])

    def test_compute_silhouette_in_range(self):
        score = compute_silhouette(self.X, self.labels)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_good_clustering_high_score(self):
        score = compute_silhouette(self.X, self.labels)
        self.assertGreater(score, 0.5)

    def test_per_sample_shape(self):
        per_sample = silhouette_per_sample(self.X, self.labels)
        self.assertEqual(len(per_sample), 6)

    def test_per_sample_in_range(self):
        per_sample = silhouette_per_sample(self.X, self.labels)
        for s in per_sample:
            self.assertGreaterEqual(s, -1)
            self.assertLessEqual(s, 1)

    def test_find_optimal_returns_int(self):
        k = find_optimal_clusters(self.X, range(2, 4))
        self.assertIsInstance(k, int)

    def test_finds_correct_k(self):
        k = find_optimal_clusters(self.X, range(2, 5))
        self.assertEqual(k, 2)

    def test_compute_silhouette_returns_float(self):
        score = compute_silhouette(self.X, self.labels)
        self.assertIsInstance(score, float)

    def test_per_sample_returns_numpy(self):
        per_sample = silhouette_per_sample(self.X, self.labels)
        self.assertIsInstance(per_sample, np.ndarray)

    def test_optimal_k_in_range(self):
        k_range = range(2, 5)
        k = find_optimal_clusters(self.X, k_range)
        self.assertIn(k, k_range)

    def test_bad_clustering_low_score(self):
        bad_labels = np.array([0, 1, 0, 1, 0, 1])  # Mixed labels
        score = compute_silhouette(self.X, bad_labels)
        self.assertLess(score, 0.5)
`,

	hint1: 'Use silhouette_score(X, labels) from sklearn.metrics',
	hint2: 'Use silhouette_samples for per-point analysis, iterate k to find best',

	whyItMatters: `Cluster evaluation is critical for:

- **Model selection**: Choose optimal number of clusters
- **Quality assessment**: Validate clustering results
- **Debugging**: Identify poorly clustered samples
- **Comparison**: Compare different algorithms

Essential for reliable unsupervised learning.`,

	translations: {
		ru: {
			title: 'Оценка кластеризации',
			description: `# Оценка кластеризации

Оценка качества кластеризации с помощью silhouette score и других метрик.

## Задача

Реализуйте три функции:
1. \`compute_silhouette(X, labels)\` - Вычислить silhouette score
2. \`silhouette_per_sample(X, labels)\` - Получить silhouette для каждого образца
3. \`find_optimal_clusters(X, k_range)\` - Найти лучший k используя silhouette

## Пример

\`\`\`python
from sklearn.metrics import silhouette_score, silhouette_samples

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
labels = np.array([0, 0, 0, 1, 1, 1])

score = compute_silhouette(X, labels)  # ~0.7
per_sample = silhouette_per_sample(X, labels)
best_k = find_optimal_clusters(X, range(2, 5))  # 2
\`\`\``,
			hint1: 'Используйте silhouette_score(X, labels) из sklearn.metrics',
			hint2: 'Используйте silhouette_samples для анализа по точкам',
			whyItMatters: `Оценка кластеризации критична для:

- **Выбор модели**: Выбор оптимального числа кластеров
- **Оценка качества**: Валидация результатов кластеризации
- **Отладка**: Определение плохо кластеризованных образцов`,
		},
		uz: {
			title: 'Klasterlashni baholash',
			description: `# Klasterlashni baholash

Silhouette ball va boshqa metrikalar yordamida klasterlash sifatini baholash.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_silhouette(X, labels)\` - Silhouette ballini hisoblash
2. \`silhouette_per_sample(X, labels)\` - Har bir namuna uchun silhouette qiymatlarini olish
3. \`find_optimal_clusters(X, k_range)\` - Silhouette yordamida eng yaxshi k ni topish

## Misol

\`\`\`python
from sklearn.metrics import silhouette_score, silhouette_samples

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
labels = np.array([0, 0, 0, 1, 1, 1])

score = compute_silhouette(X, labels)  # ~0.7
per_sample = silhouette_per_sample(X, labels)
best_k = find_optimal_clusters(X, range(2, 5))  # 2
\`\`\``,
			hint1: "sklearn.metrics dan silhouette_score(X, labels) dan foydalaning",
			hint2: "Nuqta bo'yicha tahlil uchun silhouette_samples dan foydalaning",
			whyItMatters: `Klasterlashni baholash quyidagilar uchun muhim:

- **Model tanlash**: Optimal klasterlar sonini tanlash
- **Sifat baholash**: Klasterlash natijalarini tasdiqlash
- **Debugging**: Yomon klasterlangan namunalarni aniqlash`,
		},
	},
};

export default task;
