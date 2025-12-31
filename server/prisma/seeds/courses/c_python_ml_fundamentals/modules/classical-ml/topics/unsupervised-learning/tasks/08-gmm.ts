import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-gmm',
	title: 'Gaussian Mixture Models',
	difficulty: 'hard',
	tags: ['sklearn', 'gmm', 'clustering'],
	estimatedTime: '15m',
	isPremium: true,
	order: 8,
	description: `# Gaussian Mixture Models

GMM performs soft clustering using Gaussian distributions.

## Task

Implement three functions:
1. \`train_gmm(X, n_components)\` - Train GMM
2. \`get_cluster_probabilities(model, X)\` - Get soft assignment probabilities
3. \`select_n_components(X, n_range)\` - Select best n using BIC

## Example

\`\`\`python
from sklearn.mixture import GaussianMixture

X = np.random.randn(100, 2)

gmm = train_gmm(X, n_components=3)
probs = get_cluster_probabilities(gmm, X)  # (100, 3) soft assignments
best_n = select_n_components(X, range(1, 6))  # e.g., 3
\`\`\``,

	initialCode: `import numpy as np
from sklearn.mixture import GaussianMixture

def train_gmm(X: np.ndarray, n_components: int):
    """Train Gaussian Mixture Model. Return fitted model."""
    # Your code here
    pass

def get_cluster_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Get soft assignment probabilities. Return (n_samples, n_components)."""
    # Your code here
    pass

def select_n_components(X: np.ndarray, n_range) -> int:
    """Select best n_components using BIC. Return optimal n."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.mixture import GaussianMixture

def train_gmm(X: np.ndarray, n_components: int):
    """Train Gaussian Mixture Model. Return fitted model."""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    return gmm

def get_cluster_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Get soft assignment probabilities. Return (n_samples, n_components)."""
    return model.predict_proba(X)

def select_n_components(X: np.ndarray, n_range) -> int:
    """Select best n_components using BIC. Return optimal n."""
    best_n = 1
    best_bic = np.inf
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_n = n
    return best_n
`,

	testCode: `import numpy as np
import unittest

class TestGMM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [5, 5]
        ])

    def test_train_returns_model(self):
        gmm = train_gmm(self.X, 2)
        self.assertIsNotNone(gmm)

    def test_model_can_predict(self):
        gmm = train_gmm(self.X, 2)
        labels = gmm.predict(self.X)
        self.assertEqual(len(labels), 100)

    def test_probabilities_shape(self):
        gmm = train_gmm(self.X, 3)
        probs = get_cluster_probabilities(gmm, self.X)
        self.assertEqual(probs.shape, (100, 3))

    def test_probabilities_sum_to_one(self):
        gmm = train_gmm(self.X, 2)
        probs = get_cluster_probabilities(gmm, self.X)
        for row in probs:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_select_returns_int(self):
        n = select_n_components(self.X, range(1, 4))
        self.assertIsInstance(n, int)

    def test_finds_reasonable_n(self):
        n = select_n_components(self.X, range(1, 5))
        self.assertIn(n, [1, 2, 3])

    def test_probabilities_returns_numpy(self):
        gmm = train_gmm(self.X, 2)
        probs = get_cluster_probabilities(gmm, self.X)
        self.assertIsInstance(probs, np.ndarray)

    def test_model_has_means(self):
        gmm = train_gmm(self.X, 2)
        self.assertTrue(hasattr(gmm, 'means_'))
        self.assertEqual(gmm.means_.shape[0], 2)

    def test_probabilities_in_range(self):
        gmm = train_gmm(self.X, 2)
        probs = get_cluster_probabilities(gmm, self.X)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_select_n_in_range(self):
        n_range = range(1, 5)
        n = select_n_components(self.X, n_range)
        self.assertIn(n, n_range)
`,

	hint1: 'Use GaussianMixture(n_components=n).fit(X) for training',
	hint2: 'Use model.predict_proba(X) for soft assignments, model.bic(X) for selection',

	whyItMatters: `GMM is valuable for:

- **Soft clustering**: Probabilistic cluster assignments
- **Density estimation**: Model data distribution
- **Generative model**: Sample new data points
- **Uncertainty**: Know confidence in assignments

Bridge between clustering and probability.`,

	translations: {
		ru: {
			title: 'Гауссовы смеси',
			description: `# Гауссовы смеси

GMM выполняет мягкую кластеризацию с использованием гауссовых распределений.

## Задача

Реализуйте три функции:
1. \`train_gmm(X, n_components)\` - Обучить GMM
2. \`get_cluster_probabilities(model, X)\` - Получить вероятности принадлежности
3. \`select_n_components(X, n_range)\` - Выбрать лучший n используя BIC

## Пример

\`\`\`python
from sklearn.mixture import GaussianMixture

X = np.random.randn(100, 2)

gmm = train_gmm(X, n_components=3)
probs = get_cluster_probabilities(gmm, X)  # (100, 3) soft assignments
best_n = select_n_components(X, range(1, 6))  # e.g., 3
\`\`\``,
			hint1: 'Используйте GaussianMixture(n_components=n).fit(X)',
			hint2: 'Используйте model.predict_proba(X) для мягких присвоений, model.bic(X) для выбора',
			whyItMatters: `GMM ценен для:

- **Мягкая кластеризация**: Вероятностные присвоения кластеров
- **Оценка плотности**: Моделирование распределения данных
- **Генеративная модель**: Генерация новых точек`,
		},
		uz: {
			title: "Gauss aralashmasi modellari",
			description: `# Gauss aralashmasi modellari

GMM Gauss taqsimotlari yordamida yumshoq klasterlashni amalga oshiradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_gmm(X, n_components)\` - GMM ni o'rgatish
2. \`get_cluster_probabilities(model, X)\` - Yumshoq tayinlash ehtimolliklarini olish
3. \`select_n_components(X, n_range)\` - BIC yordamida eng yaxshi n ni tanlash

## Misol

\`\`\`python
from sklearn.mixture import GaussianMixture

X = np.random.randn(100, 2)

gmm = train_gmm(X, n_components=3)
probs = get_cluster_probabilities(gmm, X)  # (100, 3) soft assignments
best_n = select_n_components(X, range(1, 6))  # e.g., 3
\`\`\``,
			hint1: "O'rgatish uchun GaussianMixture(n_components=n).fit(X) dan foydalaning",
			hint2: "Yumshoq tayinlashlar uchun model.predict_proba(X), tanlash uchun model.bic(X) dan foydalaning",
			whyItMatters: `GMM quyidagilar uchun qimmatli:

- **Yumshoq klasterlash**: Ehtimollik asosidagi klaster tayinlashlari
- **Zichlik baholash**: Ma'lumotlar taqsimotini modellashtirish
- **Generativ model**: Yangi ma'lumotlar nuqtalarini namuna olish`,
		},
	},
};

export default task;
