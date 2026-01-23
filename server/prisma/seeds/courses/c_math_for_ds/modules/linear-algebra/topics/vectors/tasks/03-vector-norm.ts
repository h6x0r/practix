import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-vector-norm',
	title: 'Vector Norm (L2)',
	difficulty: 'easy',
	tags: ['python', 'math', 'linear-algebra', 'vectors', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Vector Norm (L2)

Implement a function that computes the L2 norm (Euclidean length) of a vector.

## Background

The L2 norm (also called Euclidean norm or magnitude) is:

\`\`\`
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
\`\`\`

## Use Cases

- **Normalization**: Dividing by norm creates unit vectors
- **Distance**: L2 norm of difference = Euclidean distance
- **Regularization**: L2 regularization penalizes large weights

## Example

\`\`\`python
v = np.array([3, 4])
result = vector_norm(v)  # √(9 + 16) = √25 = 5
\`\`\`
`,
	initialCode: `import numpy as np

def vector_norm(v: np.ndarray) -> float:
    """
    Compute the L2 (Euclidean) norm of a vector.

    Args:
        v: Input vector

    Returns:
        L2 norm (magnitude) of the vector
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def vector_norm(v: np.ndarray) -> float:
    """
    Compute the L2 (Euclidean) norm of a vector.

    Args:
        v: Input vector

    Returns:
        L2 norm (magnitude) of the vector
    """
    return np.linalg.norm(v)
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

class TestVectorNorm(unittest.TestCase):
    def test_3_4_triangle(self):
        v = np.array([3, 4])
        result = vector_norm(v)
        assert_close(result, 5.0)

    def test_unit_vector(self):
        v = np.array([1, 0, 0])
        result = vector_norm(v)
        assert_close(result, 1.0)

    def test_3d_vector(self):
        v = np.array([1, 2, 2])
        result = vector_norm(v)
        assert_close(result, 3.0)

    def test_zero_vector(self):
        v = np.array([0, 0, 0])
        result = vector_norm(v)
        assert_close(result, 0.0)

    def test_negative_values(self):
        v = np.array([-3, -4])
        result = vector_norm(v)
        assert_close(result, 5.0)

    def test_float_values(self):
        v = np.array([1.0, 1.0])
        result = vector_norm(v)
        assert_close(result, np.sqrt(2))

    def test_single_element(self):
        v = np.array([5])
        result = vector_norm(v)
        assert_close(result, 5.0)

    def test_large_vector(self):
        v = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        result = vector_norm(v)
        assert_close(result, 3.0)

    def test_mixed_values(self):
        v = np.array([2, -1, 2])
        result = vector_norm(v)
        assert_close(result, 3.0)

    def test_returns_positive(self):
        v = np.array([-5, -12])
        result = vector_norm(v)
        assert result > 0
        assert_close(result, 13.0)

`,
	hint1: 'Use np.linalg.norm() to compute the L2 norm directly.',
	hint2: 'Alternatively: np.sqrt(np.sum(v ** 2)) or np.sqrt(np.dot(v, v))',
	whyItMatters: `Vector norms are essential for regularization and normalization. L2 regularization (weight decay) adds the squared norm of weights to the loss. Batch normalization normalizes using L2 norms. **Production Pattern:** Embedding vectors are often normalized to unit length before comparison in similarity search.`,
	translations: {
		ru: {
			title: 'Норма вектора (L2)',
			description: `# Норма вектора (L2)

Реализуйте функцию вычисления L2 нормы (евклидовой длины) вектора.

## Теория

L2 норма (евклидова норма или длина):

\`\`\`
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
\`\`\`

## Применение

- **Нормализация**: Деление на норму создаёт единичные векторы
- **Расстояние**: L2 норма разности = евклидово расстояние
- **Регуляризация**: L2 регуляризация штрафует большие веса
`,
			hint1: 'Используйте np.linalg.norm() для прямого вычисления L2 нормы.',
			hint2: 'Альтернативно: np.sqrt(np.sum(v ** 2)) или np.sqrt(np.dot(v, v))',
			whyItMatters: `Нормы векторов важны для регуляризации и нормализации. L2 регуляризация добавляет квадрат нормы весов к функции потерь. **Production Pattern:** Эмбеддинги часто нормализуют до единичной длины перед сравнением.`,
		},
		uz: {
			title: 'Vektor normasi (L2)',
			description: `# Vektor normasi (L2)

Vektorning L2 normasini (Evklid uzunligi) hisoblaydigan funksiyani amalga oshiring.

## Nazariya

L2 norma (Evklid normasi):

\`\`\`
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
\`\`\`

## Qo'llanilishi

- Normalizatsiya
- Masofani hisoblash
- Regularizatsiya
`,
			hint1: 'L2 normasini to\'g\'ridan-to\'g\'ri hisoblash uchun np.linalg.norm() dan foydalaning.',
			hint2: 'Muqobil: np.sqrt(np.sum(v ** 2))',
			whyItMatters: `Vektor normalari regularizatsiya va normalizatsiya uchun muhim. L2 regularizatsiya og'irliklar normasining kvadratini yo'qotish funksiyasiga qo'shadi.`,
		},
	},
};

export default task;
