import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-dbscan',
	title: 'DBSCAN Clustering',
	difficulty: 'medium',
	tags: ['sklearn', 'clustering', 'dbscan'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# DBSCAN Clustering

DBSCAN finds clusters of varying shapes and identifies outliers automatically.

## Task

Implement three functions:
1. \`train_dbscan(X, eps, min_samples)\` - Train DBSCAN
2. \`get_outliers(model, X)\` - Return indices of outlier points
3. \`count_clusters(model)\` - Return number of clusters (excluding noise)

## Example

\`\`\`python
from sklearn.cluster import DBSCAN

X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [100, 100]])

model = train_dbscan(X, eps=2, min_samples=2)
outliers = get_outliers(model, X)  # [5] - index of [100, 100]
n_clusters = count_clusters(model)  # 2
\`\`\``,

	initialCode: `import numpy as np
from sklearn.cluster import DBSCAN

def train_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """Train DBSCAN clustering. Return fitted model."""
    # Your code here
    pass

def get_outliers(model, X: np.ndarray) -> np.ndarray:
    """Return indices of outlier points (label = -1)."""
    # Your code here
    pass

def count_clusters(model) -> int:
    """Return number of clusters (excluding noise)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.cluster import DBSCAN

def train_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """Train DBSCAN clustering. Return fitted model."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model

def get_outliers(model, X: np.ndarray) -> np.ndarray:
    """Return indices of outlier points (label = -1)."""
    return np.where(model.labels_ == -1)[0]

def count_clusters(model) -> int:
    """Return number of clusters (excluding noise)."""
    labels = model.labels_
    return len(set(labels)) - (1 if -1 in labels else 0)
`,

	testCode: `import numpy as np
import unittest

class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [100, 100]])

    def test_train_returns_model(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        self.assertIsNotNone(model)

    def test_model_has_labels(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        self.assertEqual(len(model.labels_), 6)

    def test_get_outliers_finds_outlier(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        outliers = get_outliers(model, self.X)
        self.assertIn(5, outliers)

    def test_count_clusters_correct(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        n = count_clusters(model)
        self.assertEqual(n, 2)

    def test_no_outliers_with_large_eps(self):
        model = train_dbscan(self.X, eps=200, min_samples=2)
        outliers = get_outliers(model, self.X)
        self.assertEqual(len(outliers), 0)

    def test_get_outliers_returns_numpy(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        outliers = get_outliers(model, self.X)
        self.assertIsInstance(outliers, np.ndarray)

    def test_count_clusters_returns_int(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        n = count_clusters(model)
        self.assertIsInstance(n, int)

    def test_small_eps_more_outliers(self):
        model_small = train_dbscan(self.X, eps=0.5, min_samples=2)
        model_large = train_dbscan(self.X, eps=2, min_samples=2)
        outliers_small = get_outliers(model_small, self.X)
        outliers_large = get_outliers(model_large, self.X)
        self.assertGreaterEqual(len(outliers_small), len(outliers_large))

    def test_labels_contain_noise(self):
        model = train_dbscan(self.X, eps=2, min_samples=2)
        self.assertIn(-1, model.labels_)

    def test_different_params(self):
        model1 = train_dbscan(self.X, eps=1, min_samples=2)
        model2 = train_dbscan(self.X, eps=5, min_samples=3)
        self.assertIsNotNone(model1)
        self.assertIsNotNone(model2)
`,

	hint1: 'Use DBSCAN(eps=eps, min_samples=n).fit(X) for clustering',
	hint2: 'Outliers have label -1, count unique labels excluding -1',

	whyItMatters: `DBSCAN is powerful for:

- **Arbitrary shapes**: Finds non-spherical clusters
- **No k needed**: Automatically determines cluster count
- **Outlier detection**: Built-in anomaly identification
- **Density-based**: Works with varying densities

Essential when cluster shape is unknown.`,

	translations: {
		ru: {
			title: 'Кластеризация DBSCAN',
			description: `# Кластеризация DBSCAN

DBSCAN находит кластеры произвольной формы и автоматически определяет выбросы.

## Задача

Реализуйте три функции:
1. \`train_dbscan(X, eps, min_samples)\` - Обучить DBSCAN
2. \`get_outliers(model, X)\` - Вернуть индексы выбросов
3. \`count_clusters(model)\` - Вернуть количество кластеров (без шума)

## Пример

\`\`\`python
from sklearn.cluster import DBSCAN

X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [100, 100]])

model = train_dbscan(X, eps=2, min_samples=2)
outliers = get_outliers(model, X)  # [5] - index of [100, 100]
n_clusters = count_clusters(model)  # 2
\`\`\``,
			hint1: 'Используйте DBSCAN(eps=eps, min_samples=n).fit(X)',
			hint2: 'Выбросы имеют метку -1, считайте уникальные метки исключая -1',
			whyItMatters: `DBSCAN мощный для:

- **Произвольные формы**: Находит несферические кластеры
- **Не нужен k**: Автоматически определяет количество кластеров
- **Обнаружение выбросов**: Встроенная идентификация аномалий`,
		},
		uz: {
			title: 'DBSCAN klasterlash',
			description: `# DBSCAN klasterlash

DBSCAN turli shakllarning klasterlarini topadi va outlierlarni avtomatik aniqlaydi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_dbscan(X, eps, min_samples)\` - DBSCAN ni o'rgatish
2. \`get_outliers(model, X)\` - Outlier nuqtalarining indekslarini qaytarish
3. \`count_clusters(model)\` - Klasterlar sonini qaytarish (shovqinsiz)

## Misol

\`\`\`python
from sklearn.cluster import DBSCAN

X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [100, 100]])

model = train_dbscan(X, eps=2, min_samples=2)
outliers = get_outliers(model, X)  # [5] - index of [100, 100]
n_clusters = count_clusters(model)  # 2
\`\`\``,
			hint1: "DBSCAN(eps=eps, min_samples=n).fit(X) dan foydalaning",
			hint2: "Outlierlar -1 tegiga ega, -1 ni istisno qilib noyob teglarni sanang",
			whyItMatters: `DBSCAN quyidagilar uchun kuchli:

- **Ixtiyoriy shakllar**: Sferik bo'lmagan klasterlarni topadi
- **k kerak emas**: Klaster sonini avtomatik aniqlaydi
- **Outlier aniqlash**: O'rnatilgan anomaliya identifikatsiyasi`,
		},
	},
};

export default task;
