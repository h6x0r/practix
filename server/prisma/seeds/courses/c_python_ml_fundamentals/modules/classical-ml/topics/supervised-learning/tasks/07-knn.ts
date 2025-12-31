import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-knn',
	title: 'K-Nearest Neighbors',
	difficulty: 'easy',
	tags: ['sklearn', 'knn', 'classification'],
	estimatedTime: '12m',
	isPremium: false,
	order: 7,
	description: `# K-Nearest Neighbors

KNN classifies based on the majority class of nearest neighbors.

## Task

Implement three functions:
1. \`train_knn(X, y, n_neighbors)\` - Train KNN classifier
2. \`find_optimal_k(X, y, k_range)\` - Find best k via cross-validation
3. \`get_neighbors(model, X_sample, k)\` - Return indices of k nearest neighbors

## Example

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_knn(X, y, n_neighbors=3)
best_k = find_optimal_k(X, y, range(1, 10))  # e.g., 5
neighbors = get_neighbors(model, X[[0]], k=3)  # indices
\`\`\``,

	initialCode: `import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
    """Train KNN classifier. Return fitted model."""
    # Your code here
    pass

def find_optimal_k(X: np.ndarray, y: np.ndarray, k_range) -> int:
    """Find best k via cross-validation. Return optimal k."""
    # Your code here
    pass

def get_neighbors(model, X_sample: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
    """Train KNN classifier. Return fitted model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

def find_optimal_k(X: np.ndarray, y: np.ndarray, k_range) -> int:
    """Find best k via cross-validation. Return optimal k."""
    best_k = 1
    best_score = 0
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv=5).mean()
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def get_neighbors(model, X_sample: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors."""
    distances, indices = model.kneighbors(X_sample, n_neighbors=k)
    return indices
`,

	testCode: `import numpy as np
import unittest

class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 2]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_train_returns_model(self):
        model = train_knn(self.X, self.y, 3)
        self.assertIsNotNone(model)

    def test_model_can_predict(self):
        model = train_knn(self.X, self.y, 3)
        pred = model.predict(self.X[[0]])
        self.assertEqual(len(pred), 1)

    def test_find_optimal_k_returns_int(self):
        k = find_optimal_k(self.X, self.y, range(1, 5))
        self.assertIsInstance(k, int)
        self.assertGreaterEqual(k, 1)

    def test_get_neighbors_shape(self):
        model = train_knn(self.X, self.y, 3)
        indices = get_neighbors(model, self.X[[0]], 3)
        self.assertEqual(indices.shape[1], 3)

    def test_get_neighbors_includes_self(self):
        model = train_knn(self.X, self.y, 3)
        indices = get_neighbors(model, self.X[[0]], 3)
        self.assertIn(0, indices[0])

    def test_model_has_n_neighbors(self):
        model = train_knn(self.X, self.y, 5)
        self.assertEqual(model.n_neighbors, 5)

    def test_get_neighbors_returns_numpy(self):
        model = train_knn(self.X, self.y, 3)
        indices = get_neighbors(model, self.X[[0]], 3)
        self.assertIsInstance(indices, np.ndarray)

    def test_different_k_values(self):
        model_3 = train_knn(self.X, self.y, 3)
        model_5 = train_knn(self.X, self.y, 5)
        self.assertEqual(model_3.n_neighbors, 3)
        self.assertEqual(model_5.n_neighbors, 5)

    def test_optimal_k_in_range(self):
        k_range = range(1, 5)
        k = find_optimal_k(self.X, self.y, k_range)
        self.assertIn(k, k_range)

    def test_predictions_shape(self):
        model = train_knn(self.X, self.y, 3)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
`,

	hint1: 'Use KNeighborsClassifier(n_neighbors=k).fit(X, y) for training',
	hint2: 'Use model.kneighbors(X, n_neighbors=k) to find neighbors',

	whyItMatters: `KNN is useful for:

- **Simplicity**: No training phase, just store data
- **Non-parametric**: Makes no assumptions about data
- **Anomaly detection**: Points far from neighbors are unusual
- **Baseline**: Quick model to compare against

Great for prototyping and understanding data.`,

	translations: {
		ru: {
			title: 'K ближайших соседей',
			description: `# K ближайших соседей

KNN классифицирует на основе класса большинства ближайших соседей.

## Задача

Реализуйте три функции:
1. \`train_knn(X, y, n_neighbors)\` - Обучить KNN классификатор
2. \`find_optimal_k(X, y, k_range)\` - Найти лучший k через кросс-валидацию
3. \`get_neighbors(model, X_sample, k)\` - Вернуть индексы k ближайших соседей

## Пример

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_knn(X, y, n_neighbors=3)
best_k = find_optimal_k(X, y, range(1, 10))  # e.g., 5
neighbors = get_neighbors(model, X[[0]], k=3)  # indices
\`\`\``,
			hint1: 'Используйте KNeighborsClassifier(n_neighbors=k).fit(X, y)',
			hint2: 'Используйте model.kneighbors(X, n_neighbors=k) для поиска соседей',
			whyItMatters: `KNN полезен для:

- **Простота**: Нет фазы обучения, просто хранение данных
- **Непараметрический**: Не делает предположений о данных
- **Обнаружение аномалий**: Точки далеко от соседей необычны`,
		},
		uz: {
			title: "K eng yaqin qo'shnilar",
			description: `# K eng yaqin qo'shnilar

KNN eng yaqin qo'shnilarning ko'pchilik klassi asosida klassifikatsiya qiladi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_knn(X, y, n_neighbors)\` - KNN klassifikatorini o'rgatish
2. \`find_optimal_k(X, y, k_range)\` - Kross-validatsiya orqali eng yaxshi k ni topish
3. \`get_neighbors(model, X_sample, k)\` - k eng yaqin qo'shnilar indekslarini qaytarish

## Misol

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_knn(X, y, n_neighbors=3)
best_k = find_optimal_k(X, y, range(1, 10))  # e.g., 5
neighbors = get_neighbors(model, X[[0]], k=3)  # indices
\`\`\``,
			hint1: "O'rgatish uchun KNeighborsClassifier(n_neighbors=k).fit(X, y) dan foydalaning",
			hint2: "Qo'shnilarni topish uchun model.kneighbors(X, n_neighbors=k) dan foydalaning",
			whyItMatters: `KNN quyidagilar uchun foydali:

- **Oddiylik**: O'rganish bosqichi yo'q, faqat ma'lumotlarni saqlash
- **Noparametrik**: Ma'lumotlar haqida hech qanday taxmin qilmaydi
- **Anomaliya aniqlash**: Qo'shnilardan uzoqdagi nuqtalar g'ayrioddiy`,
		},
	},
};

export default task;
