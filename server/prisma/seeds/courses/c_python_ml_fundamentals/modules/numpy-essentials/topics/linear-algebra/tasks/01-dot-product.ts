import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-dot-product',
	title: 'Dot Product',
	difficulty: 'easy',
	tags: ['numpy', 'linear-algebra', 'dot-product'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Dot Product

The dot product is a fundamental operation in linear algebra, computing the sum of element-wise products.

## Task

Implement three functions:
1. \`vector_dot(a, b)\` - Dot product of two 1D vectors
2. \`matrix_vector_dot(matrix, vector)\` - Matrix-vector multiplication
3. \`cosine_similarity(a, b)\` - Cosine similarity between two vectors

## Example

\`\`\`python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

vector_dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

matrix = np.array([[1, 2], [3, 4]])
vector = np.array([1, 2])
matrix_vector_dot(matrix, vector)  # [5, 11]

cosine_similarity(a, b)  # 0.9746 (angle between vectors)
\`\`\``,

	initialCode: `import numpy as np

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1D vectors."""
    # Your code here
    pass

def matrix_vector_dot(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Matrix-vector multiplication."""
    # Your code here
    pass

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1D vectors."""
    return np.dot(a, b)

def matrix_vector_dot(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Matrix-vector multiplication."""
    return np.dot(matrix, vector)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
`,

	testCode: `import numpy as np
import unittest

class TestDotProduct(unittest.TestCase):
    def test_vector_dot_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = vector_dot(a, b)
        self.assertEqual(result, 32)

    def test_vector_dot_zeros(self):
        a = np.array([1, 2, 3])
        b = np.array([0, 0, 0])
        result = vector_dot(a, b)
        self.assertEqual(result, 0)

    def test_vector_dot_orthogonal(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        result = vector_dot(a, b)
        self.assertEqual(result, 0)

    def test_matrix_vector_basic(self):
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([1, 2])
        result = matrix_vector_dot(matrix, vector)
        np.testing.assert_array_equal(result, [5, 11])

    def test_matrix_vector_identity(self):
        matrix = np.eye(3)
        vector = np.array([1, 2, 3])
        result = matrix_vector_dot(matrix, vector)
        np.testing.assert_array_equal(result, vector)

    def test_matrix_vector_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        vector = np.array([1, 1, 1])
        result = matrix_vector_dot(matrix, vector)
        self.assertEqual(result.shape, (2,))

    def test_cosine_same_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(a, a)
        self.assertAlmostEqual(result, 1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = cosine_similarity(a, b)
        self.assertAlmostEqual(result, 0.0)

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        result = cosine_similarity(a, b)
        self.assertAlmostEqual(result, -1.0)

    def test_cosine_range(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = cosine_similarity(a, b)
        self.assertTrue(-1 <= result <= 1)
`,

	hint1: 'Use np.dot(a, b) for dot product',
	hint2: 'Cosine similarity: dot(a,b) / (norm(a) * norm(b))',

	whyItMatters: `Dot product is everywhere in ML:

- **Neural network forward pass**: Weights dot inputs
- **Embeddings similarity**: Compare word/sentence vectors
- **Attention mechanism**: Query-key dot products
- **Recommendation systems**: User-item similarity scores

This is the most common operation in deep learning.`,

	translations: {
		ru: {
			title: 'Скалярное произведение',
			description: `# Скалярное произведение

Скалярное произведение — фундаментальная операция линейной алгебры, вычисляющая сумму поэлементных произведений.

## Задача

Реализуйте три функции:
1. \`vector_dot(a, b)\` - Скалярное произведение двух 1D векторов
2. \`matrix_vector_dot(matrix, vector)\` - Умножение матрицы на вектор
3. \`cosine_similarity(a, b)\` - Косинусное сходство между векторами

## Пример

\`\`\`python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

vector_dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

matrix = np.array([[1, 2], [3, 4]])
vector = np.array([1, 2])
matrix_vector_dot(matrix, vector)  # [5, 11]

cosine_similarity(a, b)  # 0.9746 (угол между векторами)
\`\`\``,
			hint1: 'Используйте np.dot(a, b) для скалярного произведения',
			hint2: 'Косинусное сходство: dot(a,b) / (norm(a) * norm(b))',
			whyItMatters: `Скалярное произведение повсюду в ML:

- **Прямой проход нейросети**: Веса умножаются на входы
- **Сходство эмбеддингов**: Сравнение векторов слов/предложений
- **Механизм внимания**: Скалярные произведения query-key
- **Рекомендательные системы**: Оценки сходства user-item`,
		},
		uz: {
			title: "Skalyar ko'paytma",
			description: `# Skalyar ko'paytma

Skalyar ko'paytma — chiziqli algebraning asosiy operatsiyasi, elementli ko'paytmalar yig'indisini hisobaydi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`vector_dot(a, b)\` - Ikki 1D vektorning skalyar ko'paytmasi
2. \`matrix_vector_dot(matrix, vector)\` - Matritsani vektorga ko'paytirish
3. \`cosine_similarity(a, b)\` - Vektorlar orasidagi kosinus o'xshashligi

## Misol

\`\`\`python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

vector_dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

matrix = np.array([[1, 2], [3, 4]])
vector = np.array([1, 2])
matrix_vector_dot(matrix, vector)  # [5, 11]

cosine_similarity(a, b)  # 0.9746 (vektorlar orasidagi burchak)
\`\`\``,
			hint1: "Skalyar ko'paytma uchun np.dot(a, b) dan foydalaning",
			hint2: "Kosinus o'xshashligi: dot(a,b) / (norm(a) * norm(b))",
			whyItMatters: `Skalyar ko'paytma ML da hamma joyda:

- **Neyron tarmoq to'g'ri o'tishi**: Vaznlar kirishlarga ko'paytiriladi
- **Embedding o'xshashligi**: So'z/gap vektorlarini taqqoslash
- **Attention mexanizmi**: Query-key skalyar ko'paytmalari`,
		},
	},
};

export default task;
