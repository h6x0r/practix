import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-pca',
	title: 'Principal Component Analysis',
	difficulty: 'medium',
	tags: ['sklearn', 'pca', 'dimensionality-reduction'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Principal Component Analysis

PCA reduces dimensionality while preserving maximum variance.

## Task

Implement three functions:
1. \`fit_pca(X, n_components)\` - Fit PCA and return transformer
2. \`get_explained_variance(pca)\` - Return explained variance ratios
3. \`find_components_for_variance(X, target_variance)\` - Find n_components for target variance

## Example

\`\`\`python
from sklearn.decomposition import PCA

X = np.random.randn(100, 10)

pca = fit_pca(X, n_components=3)
X_reduced = pca.transform(X)  # (100, 3)
variance = get_explained_variance(pca)  # [0.3, 0.2, 0.15]
n = find_components_for_variance(X, 0.95)  # e.g., 7
\`\`\``,

	initialCode: `import numpy as np
from sklearn.decomposition import PCA

def fit_pca(X: np.ndarray, n_components: int):
    """Fit PCA. Return fitted transformer."""
    # Your code here
    pass

def get_explained_variance(pca) -> np.ndarray:
    """Return explained variance ratios."""
    # Your code here
    pass

def find_components_for_variance(X: np.ndarray, target_variance: float) -> int:
    """Find minimum n_components to explain target_variance (0-1)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.decomposition import PCA

def fit_pca(X: np.ndarray, n_components: int):
    """Fit PCA. Return fitted transformer."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca

def get_explained_variance(pca) -> np.ndarray:
    """Return explained variance ratios."""
    return pca.explained_variance_ratio_

def find_components_for_variance(X: np.ndarray, target_variance: float) -> int:
    """Find minimum n_components to explain target_variance (0-1)."""
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumsum >= target_variance) + 1)
`,

	testCode: `import numpy as np
import unittest

class TestPCA(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10)

    def test_fit_returns_pca(self):
        pca = fit_pca(self.X, 3)
        self.assertIsNotNone(pca)

    def test_transform_shape(self):
        pca = fit_pca(self.X, 3)
        X_t = pca.transform(self.X)
        self.assertEqual(X_t.shape, (100, 3))

    def test_explained_variance_shape(self):
        pca = fit_pca(self.X, 3)
        var = get_explained_variance(pca)
        self.assertEqual(len(var), 3)

    def test_variance_sums_less_than_one(self):
        pca = fit_pca(self.X, 3)
        var = get_explained_variance(pca)
        self.assertLessEqual(sum(var), 1.0)

    def test_find_components_returns_int(self):
        n = find_components_for_variance(self.X, 0.8)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_more_variance_needs_more_components(self):
        n1 = find_components_for_variance(self.X, 0.5)
        n2 = find_components_for_variance(self.X, 0.9)
        self.assertLessEqual(n1, n2)

    def test_variance_returns_numpy(self):
        pca = fit_pca(self.X, 3)
        var = get_explained_variance(pca)
        self.assertIsInstance(var, np.ndarray)

    def test_pca_has_components(self):
        pca = fit_pca(self.X, 3)
        self.assertTrue(hasattr(pca, 'components_'))
        self.assertEqual(pca.components_.shape[0], 3)

    def test_variance_decreasing(self):
        pca = fit_pca(self.X, 5)
        var = get_explained_variance(pca)
        for i in range(len(var) - 1):
            self.assertGreaterEqual(var[i], var[i+1])

    def test_components_within_range(self):
        n = find_components_for_variance(self.X, 0.8)
        self.assertTrue(1 <= n <= 10)
`,

	hint1: 'Use PCA(n_components=n).fit(X) for dimensionality reduction',
	hint2: 'Use pca.explained_variance_ratio_ and cumsum to find components for target variance',

	whyItMatters: `PCA is essential for:

- **Dimensionality reduction**: Speed up training, reduce overfitting
- **Visualization**: Project high-dim data to 2D/3D
- **Noise reduction**: Remove low-variance components
- **Feature extraction**: Create uncorrelated features

Foundation of unsupervised feature learning.`,

	translations: {
		ru: {
			title: 'Метод главных компонент',
			description: `# Метод главных компонент

PCA снижает размерность сохраняя максимум дисперсии.

## Задача

Реализуйте три функции:
1. \`fit_pca(X, n_components)\` - Обучить PCA и вернуть трансформер
2. \`get_explained_variance(pca)\` - Вернуть доли объяснённой дисперсии
3. \`find_components_for_variance(X, target_variance)\` - Найти n_components для целевой дисперсии

## Пример

\`\`\`python
from sklearn.decomposition import PCA

X = np.random.randn(100, 10)

pca = fit_pca(X, n_components=3)
X_reduced = pca.transform(X)  # (100, 3)
variance = get_explained_variance(pca)  # [0.3, 0.2, 0.15]
n = find_components_for_variance(X, 0.95)  # e.g., 7
\`\`\``,
			hint1: 'Используйте PCA(n_components=n).fit(X) для снижения размерности',
			hint2: 'Используйте pca.explained_variance_ratio_ и cumsum',
			whyItMatters: `PCA необходим для:

- **Снижение размерности**: Ускорение обучения, уменьшение переобучения
- **Визуализация**: Проекция многомерных данных в 2D/3D
- **Уменьшение шума**: Удаление низкодисперсных компонент`,
		},
		uz: {
			title: "Asosiy komponentlar tahlili",
			description: `# Asosiy komponentlar tahlili

PCA maksimal dispersiyani saqlab o'lchamlilikni kamaytiradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`fit_pca(X, n_components)\` - PCA ni o'rgatish va transformerni qaytarish
2. \`get_explained_variance(pca)\` - Tushuntirilgan dispersiya nisbatlarini qaytarish
3. \`find_components_for_variance(X, target_variance)\` - Maqsadli dispersiya uchun n_components topish

## Misol

\`\`\`python
from sklearn.decomposition import PCA

X = np.random.randn(100, 10)

pca = fit_pca(X, n_components=3)
X_reduced = pca.transform(X)  # (100, 3)
variance = get_explained_variance(pca)  # [0.3, 0.2, 0.15]
n = find_components_for_variance(X, 0.95)  # e.g., 7
\`\`\``,
			hint1: "O'lchamlilikni kamaytirish uchun PCA(n_components=n).fit(X) dan foydalaning",
			hint2: "pca.explained_variance_ratio_ va cumsum dan foydalaning",
			whyItMatters: `PCA quyidagilar uchun zarur:

- **O'lchamlilikni kamaytirish**: O'qitishni tezlashtirish, ortiqcha moslashishni kamaytirish
- **Vizualizatsiya**: Yuqori o'lchamli ma'lumotlarni 2D/3D ga proyeksiya qilish
- **Shovqinni kamaytirish**: Past dispersiyali komponentlarni olib tashlash`,
		},
	},
};

export default task;
