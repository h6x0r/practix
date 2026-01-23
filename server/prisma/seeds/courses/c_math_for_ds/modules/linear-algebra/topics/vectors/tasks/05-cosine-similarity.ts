import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-cosine-similarity',
	title: 'Cosine Similarity',
	difficulty: 'medium',
	tags: ['python', 'math', 'linear-algebra', 'vectors', 'numpy', 'similarity'],
	estimatedTime: '20m',
	isPremium: false,
	order: 5,
	description: `# Cosine Similarity

Implement a function that computes the cosine similarity between two vectors.

## Background

Cosine similarity measures the cosine of the angle between two vectors:

\`\`\`
cos(θ) = (a · b) / (||a|| × ||b||)
\`\`\`

- **Range**: [-1, 1]
- **1**: Vectors point same direction (identical)
- **0**: Vectors are perpendicular (unrelated)
- **-1**: Vectors point opposite directions

## Use Cases

- **Text similarity**: Compare document embeddings
- **Recommendation**: Find similar users/items
- **Image search**: Match feature vectors

## Example

\`\`\`python
v1 = np.array([1, 0, 0])
v2 = np.array([1, 0, 0])
result = cosine_similarity(v1, v2)  # 1.0 (identical)

v3 = np.array([0, 1, 0])
result = cosine_similarity(v1, v3)  # 0.0 (perpendicular)
\`\`\`
`,
	initialCode: `import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)
`,
	testCode: `import unittest
import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

def assert_close(actual, expected, places=5, msg=""):
    """Helper for scalar comparison with clear error message"""
    if abs(actual - expected) > 10**(-places):
        raise AssertionError(f"Expected {expected}, got {actual}")

class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 3])
        result = cosine_similarity(v1, v2)
        assert_close(result, 1.0)

    def test_parallel_vectors(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([2, 4, 6])
        result = cosine_similarity(v1, v2)
        assert_close(result, 1.0)

    def test_perpendicular_vectors(self):
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        result = cosine_similarity(v1, v2)
        assert_close(result, 0.0)

    def test_opposite_vectors(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([-1, -2, -3])
        result = cosine_similarity(v1, v2)
        assert_close(result, -1.0)

    def test_45_degree_angle(self):
        v1 = np.array([1, 0])
        v2 = np.array([1, 1])
        result = cosine_similarity(v1, v2)
        assert_close(result, np.sqrt(2) / 2)

    def test_unit_vectors(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = cosine_similarity(v1, v2)
        assert_close(result, 0.0)

    def test_float_vectors(self):
        v1 = np.array([0.5, 0.5, 0.5])
        v2 = np.array([1.0, 1.0, 1.0])
        result = cosine_similarity(v1, v2)
        assert_close(result, 1.0)

    def test_different_magnitudes(self):
        v1 = np.array([1, 0])
        v2 = np.array([100, 0])
        result = cosine_similarity(v1, v2)
        assert_close(result, 1.0)

    def test_range_bounds(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = cosine_similarity(v1, v2)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_3d_vectors(self):
        v1 = np.array([1, 1, 1])
        v2 = np.array([1, 1, 1])
        result = cosine_similarity(v1, v2)
        assert_close(result, 1.0)

`,
	hint1: 'Formula: dot(v1, v2) / (norm(v1) * norm(v2))',
	hint2: 'Use np.dot() for dot product and np.linalg.norm() for vector magnitudes. Handle zero vectors.',
	whyItMatters: `Cosine similarity is THE metric for comparing embeddings. BERT, GPT, and all transformer models produce embeddings compared via cosine similarity. Vector databases like Pinecone and Faiss use it for semantic search. **Production Pattern:** When building RAG systems, retrieved documents are ranked by cosine similarity to the query embedding.`,
	translations: {
		ru: {
			title: 'Косинусное сходство',
			description: `# Косинусное сходство

Реализуйте функцию вычисления косинусного сходства между двумя векторами.

## Теория

Косинусное сходство измеряет косинус угла между двумя векторами:

\`\`\`
cos(θ) = (a · b) / (||a|| × ||b||)
\`\`\`

- **Диапазон**: [-1, 1]
- **1**: Векторы направлены одинаково
- **0**: Векторы перпендикулярны
- **-1**: Векторы направлены противоположно

## Применение

- Сходство текстов
- Рекомендательные системы
- Поиск изображений
`,
			hint1: 'Формула: dot(v1, v2) / (norm(v1) * norm(v2))',
			hint2: 'Используйте np.dot() для скалярного произведения и np.linalg.norm() для норм.',
			whyItMatters: `Косинусное сходство - ГЛАВНАЯ метрика для сравнения эмбеддингов. BERT, GPT и все трансформеры используют его. **Production Pattern:** В RAG системах документы ранжируются по косинусному сходству с запросом.`,
		},
		uz: {
			title: 'Kosinus o\'xshashligi',
			description: `# Kosinus o'xshashligi

Ikki vektor orasidagi kosinus o'xshashligini hisoblaydigan funksiyani amalga oshiring.

## Nazariya

Kosinus o'xshashligi ikki vektor orasidagi burchakning kosinusini o'lchaydi.

## Qo'llanilishi

- Matn o'xshashligi
- Tavsiya tizimlari
- Rasm qidirish
`,
			hint1: 'Formula: dot(v1, v2) / (norm(v1) * norm(v2))',
			hint2: 'np.dot() va np.linalg.norm() dan foydalaning.',
			whyItMatters: `Kosinus o'xshashligi embeddinglarni solishtirish uchun asosiy metrika. Barcha transformer modellari undan foydalanadi.`,
		},
	},
};

export default task;
