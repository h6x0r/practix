import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-normalize-vector',
	title: 'Normalize Vector',
	difficulty: 'easy',
	tags: ['python', 'math', 'linear-algebra', 'vectors', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Normalize Vector

Implement a function that normalizes a vector to unit length (L2 norm = 1).

## Background

A normalized vector (unit vector) has length 1:

\`\`\`
v_normalized = v / ||v||₂
\`\`\`

## Why Normalize?

- **Direction only**: Unit vectors represent direction without magnitude
- **Stable training**: Normalized inputs help neural networks train faster
- **Cosine similarity**: Dot product of unit vectors = cosine of angle between them

## Example

\`\`\`python
v = np.array([3, 4])
result = normalize_vector(v)  # [0.6, 0.8]
# Verify: ||result|| = √(0.36 + 0.64) = 1
\`\`\`
`,
	initialCode: `import numpy as np

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Unit vector (same direction, length = 1)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Unit vector (same direction, length = 1)
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
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

class TestNormalizeVector(unittest.TestCase):
    def test_3_4_vector(self):
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        assert_array_close(result, np.array([0.6, 0.8]))

    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        norm = np.linalg.norm(result)
        assert_close(norm, 1.0)

    def test_already_unit(self):
        v = np.array([1.0, 0.0])
        result = normalize_vector(v)
        assert_array_close(result, v)

    def test_3d_vector(self):
        v = np.array([1.0, 2.0, 2.0])
        result = normalize_vector(v)
        norm = np.linalg.norm(result)
        assert_close(norm, 1.0)

    def test_preserves_direction(self):
        v = np.array([2.0, 4.0, 6.0])
        result = normalize_vector(v)
        expected = v / np.linalg.norm(v)
        assert_array_close(result, expected)

    def test_negative_values(self):
        v = np.array([-3.0, -4.0])
        result = normalize_vector(v)
        norm = np.linalg.norm(result)
        assert_close(norm, 1.0)

    def test_single_element(self):
        v = np.array([5.0])
        result = normalize_vector(v)
        assert_close(result[0], 1.0)

    def test_large_values(self):
        v = np.array([1000.0, 0.0])
        result = normalize_vector(v)
        assert_array_close(result, np.array([1.0, 0.0]))

    def test_small_values(self):
        v = np.array([0.001, 0.001])
        result = normalize_vector(v)
        norm = np.linalg.norm(result)
        assert_close(norm, 1.0)

    def test_returns_numpy_array(self):
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

`,
	hint1: 'Divide the vector by its L2 norm: v / np.linalg.norm(v)',
	hint2: 'Handle the zero vector case - you cannot normalize a zero vector.',
	whyItMatters: `Normalization is critical in ML. Word embeddings like Word2Vec are normalized before computing similarities. Layer normalization and batch normalization use similar concepts. **Production Pattern:** In vector databases for similarity search, all vectors are normalized so dot product equals cosine similarity.`,
	translations: {
		ru: {
			title: 'Нормализация вектора',
			description: `# Нормализация вектора

Реализуйте функцию нормализации вектора до единичной длины (L2 норма = 1).

## Теория

Нормализованный вектор (единичный вектор) имеет длину 1:

\`\`\`
v_normalized = v / ||v||₂
\`\`\`

## Зачем нормализовать?

- **Только направление**: Единичные векторы представляют направление без величины
- **Стабильное обучение**: Нормализованные входы помогают быстрее обучать нейросети
- **Косинусное сходство**: Скалярное произведение единичных векторов = косинус угла
`,
			hint1: 'Разделите вектор на его L2 норму: v / np.linalg.norm(v)',
			hint2: 'Обработайте случай нулевого вектора - нельзя нормализовать нулевой вектор.',
			whyItMatters: `Нормализация критична в ML. Эмбеддинги слов нормализуются перед вычислением схожести. **Production Pattern:** В векторных базах данных все векторы нормализуются, чтобы скалярное произведение равнялось косинусному сходству.`,
		},
		uz: {
			title: 'Vektorni normallashtirish',
			description: `# Vektorni normallashtirish

Vektorni birlik uzunlikka (L2 norma = 1) normallashtiradigan funksiyani amalga oshiring.

## Nazariya

Normallashtirilgan vektor (birlik vektor) uzunligi 1:

\`\`\`
v_normalized = v / ||v||₂
\`\`\`
`,
			hint1: 'Vektorni uning L2 normasiga bo\'ling: v / np.linalg.norm(v)',
			hint2: 'Nol vektor holatini ko\'rib chiqing.',
			whyItMatters: `Normallashtirish ML da juda muhim. So'z embeddinglar o'xshashlikni hisoblashdan oldin normallashtiriladi.`,
		},
	},
};

export default task;
