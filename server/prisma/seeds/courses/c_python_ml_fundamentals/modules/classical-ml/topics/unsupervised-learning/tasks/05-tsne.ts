import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-tsne',
	title: 't-SNE Visualization',
	difficulty: 'medium',
	tags: ['sklearn', 'tsne', 'visualization'],
	estimatedTime: '12m',
	isPremium: false,
	order: 5,
	description: `# t-SNE Visualization

t-SNE creates 2D/3D visualizations preserving local structure.

## Task

Implement three functions:
1. \`apply_tsne(X, n_components, perplexity)\` - Apply t-SNE transformation
2. \`compare_perplexities(X, perplexities)\` - Compare different perplexity values
3. \`tsne_with_pca_init(X, n_components)\` - Use PCA initialization for stability

## Example

\`\`\`python
from sklearn.manifold import TSNE

X = np.random.randn(100, 50)

X_2d = apply_tsne(X, n_components=2, perplexity=30)
embeddings = compare_perplexities(X, [5, 30, 50])  # {5: X_2d, 30: X_2d, ...}
X_stable = tsne_with_pca_init(X, n_components=2)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def apply_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30):
    """Apply t-SNE transformation. Return embedded coordinates."""
    # Your code here
    pass

def compare_perplexities(X: np.ndarray, perplexities: list) -> dict:
    """Compare perplexities. Return {perplexity: embedding}."""
    # Your code here
    pass

def tsne_with_pca_init(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Use PCA initialization for stable t-SNE. Return embedded coordinates."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def apply_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30):
    """Apply t-SNE transformation. Return embedded coordinates."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(X)

def compare_perplexities(X: np.ndarray, perplexities: list) -> dict:
    """Compare perplexities. Return {perplexity: embedding}."""
    embeddings = {}
    for p in perplexities:
        tsne = TSNE(n_components=2, perplexity=p, random_state=42)
        embeddings[p] = tsne.fit_transform(X)
    return embeddings

def tsne_with_pca_init(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Use PCA initialization for stable t-SNE. Return embedded coordinates."""
    tsne = TSNE(n_components=n_components, init='pca', random_state=42)
    return tsne.fit_transform(X)
`,

	testCode: `import numpy as np
import unittest

class TestTSNE(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 20)

    def test_apply_tsne_shape(self):
        X_2d = apply_tsne(self.X, n_components=2, perplexity=10)
        self.assertEqual(X_2d.shape, (50, 2))

    def test_apply_tsne_3d(self):
        X_3d = apply_tsne(self.X, n_components=3, perplexity=10)
        self.assertEqual(X_3d.shape, (50, 3))

    def test_compare_perplexities_returns_dict(self):
        result = compare_perplexities(self.X, [5, 10])
        self.assertIsInstance(result, dict)
        self.assertIn(5, result)
        self.assertIn(10, result)

    def test_compare_perplexities_shapes(self):
        result = compare_perplexities(self.X, [5, 10])
        for emb in result.values():
            self.assertEqual(emb.shape, (50, 2))

    def test_tsne_with_pca_init_shape(self):
        X_2d = tsne_with_pca_init(self.X, n_components=2)
        self.assertEqual(X_2d.shape, (50, 2))

    def test_apply_tsne_returns_numpy(self):
        X_2d = apply_tsne(self.X, n_components=2, perplexity=10)
        self.assertIsInstance(X_2d, np.ndarray)

    def test_compare_perplexities_keys(self):
        perps = [5, 10, 15]
        result = compare_perplexities(self.X, perps)
        for p in perps:
            self.assertIn(p, result)

    def test_tsne_with_pca_init_returns_numpy(self):
        X_2d = tsne_with_pca_init(self.X, n_components=2)
        self.assertIsInstance(X_2d, np.ndarray)

    def test_different_perplexity_values(self):
        X_5 = apply_tsne(self.X, n_components=2, perplexity=5)
        X_15 = apply_tsne(self.X, n_components=2, perplexity=15)
        self.assertEqual(X_5.shape, X_15.shape)

    def test_samples_preserved(self):
        X_2d = apply_tsne(self.X, n_components=2, perplexity=10)
        self.assertEqual(X_2d.shape[0], self.X.shape[0])
`,

	hint1: 'Use TSNE(n_components=n, perplexity=p).fit_transform(X)',
	hint2: 'Use init="pca" for reproducible results',

	whyItMatters: `t-SNE is essential for:

- **Cluster visualization**: See natural groupings in data
- **Embedding quality**: Validate learned representations
- **Presentation**: Create intuitive data visualizations
- **Debugging**: Understand what models learn

The standard for high-dimensional data visualization.`,

	translations: {
		ru: {
			title: 'Визуализация t-SNE',
			description: `# Визуализация t-SNE

t-SNE создаёт 2D/3D визуализации сохраняя локальную структуру.

## Задача

Реализуйте три функции:
1. \`apply_tsne(X, n_components, perplexity)\` - Применить преобразование t-SNE
2. \`compare_perplexities(X, perplexities)\` - Сравнить разные значения perplexity
3. \`tsne_with_pca_init(X, n_components)\` - Использовать PCA инициализацию

## Пример

\`\`\`python
from sklearn.manifold import TSNE

X = np.random.randn(100, 50)

X_2d = apply_tsne(X, n_components=2, perplexity=30)
embeddings = compare_perplexities(X, [5, 30, 50])  # {5: X_2d, 30: X_2d, ...}
X_stable = tsne_with_pca_init(X, n_components=2)
\`\`\``,
			hint1: 'Используйте TSNE(n_components=n, perplexity=p).fit_transform(X)',
			hint2: 'Используйте init="pca" для воспроизводимых результатов',
			whyItMatters: `t-SNE необходим для:

- **Визуализация кластеров**: Видеть естественные группировки
- **Качество эмбеддингов**: Валидация выученных представлений
- **Презентации**: Создание интуитивных визуализаций`,
		},
		uz: {
			title: 't-SNE vizualizatsiyasi',
			description: `# t-SNE vizualizatsiyasi

t-SNE mahalliy strukturani saqlab 2D/3D vizualizatsiyalarni yaratadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`apply_tsne(X, n_components, perplexity)\` - t-SNE transformatsiyasini qo'llash
2. \`compare_perplexities(X, perplexities)\` - Turli perplexity qiymatlarini taqqoslash
3. \`tsne_with_pca_init(X, n_components)\` - Barqarorlik uchun PCA initsializatsiyasidan foydalanish

## Misol

\`\`\`python
from sklearn.manifold import TSNE

X = np.random.randn(100, 50)

X_2d = apply_tsne(X, n_components=2, perplexity=30)
embeddings = compare_perplexities(X, [5, 30, 50])  # {5: X_2d, 30: X_2d, ...}
X_stable = tsne_with_pca_init(X, n_components=2)
\`\`\``,
			hint1: "TSNE(n_components=n, perplexity=p).fit_transform(X) dan foydalaning",
			hint2: 'Takrorlanadigan natijalar uchun init="pca" dan foydalaning',
			whyItMatters: `t-SNE quyidagilar uchun zarur:

- **Klaster vizualizatsiyasi**: Ma'lumotlarda tabiiy guruhlanishlarni ko'rish
- **Embedding sifati**: O'rganilgan ifodalashlarni tasdiqlash
- **Taqdimot**: Intuitiv ma'lumotlar vizualizatsiyalarini yaratish`,
		},
	},
};

export default task;
