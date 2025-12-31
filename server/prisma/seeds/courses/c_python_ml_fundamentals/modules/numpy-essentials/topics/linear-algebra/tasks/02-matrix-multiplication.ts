import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-matrix-multiplication',
	title: 'Matrix Multiplication',
	difficulty: 'medium',
	tags: ['numpy', 'linear-algebra', 'matrix'],
	estimatedTime: '12m',
	isPremium: false,
	order: 2,
	description: `# Matrix Multiplication

Matrix multiplication is the backbone of neural network computations. Understanding shapes is crucial.

## Task

Implement three functions:
1. \`matmul(A, B)\` - Standard matrix multiplication
2. \`batch_matmul(A, B)\` - Batched matrix multiplication (3D arrays)
3. \`outer_product(a, b)\` - Outer product of two vectors

## Example

\`\`\`python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

matmul(A, B)  # [[19, 22], [43, 50]]

# Batch: (batch, m, k) @ (batch, k, n) -> (batch, m, n)
A_batch = np.random.randn(4, 2, 3)  # 4 matrices of (2, 3)
B_batch = np.random.randn(4, 3, 2)  # 4 matrices of (3, 2)
batch_matmul(A_batch, B_batch)  # (4, 2, 2)

a = np.array([1, 2, 3])
b = np.array([4, 5])
outer_product(a, b)  # [[4, 5], [8, 10], [12, 15]]
\`\`\``,

	initialCode: `import numpy as np

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard matrix multiplication."""
    # Your code here
    pass

def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched matrix multiplication for 3D arrays."""
    # Your code here
    pass

def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of two vectors."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard matrix multiplication."""
    return A @ B

def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched matrix multiplication for 3D arrays."""
    return A @ B

def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of two vectors."""
    return np.outer(a, b)
`,

	testCode: `import numpy as np
import unittest

class TestMatrixMultiplication(unittest.TestCase):
    def test_matmul_basic(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_matmul_identity(self):
        A = np.array([[1, 2], [3, 4]])
        I = np.eye(2)
        result = matmul(A, I)
        np.testing.assert_array_equal(result, A)

    def test_matmul_rectangular(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        B = np.array([[1], [2], [3]])  # (3, 1)
        result = matmul(A, B)
        self.assertEqual(result.shape, (2, 1))

    def test_batch_matmul_shape(self):
        A = np.random.randn(4, 2, 3)
        B = np.random.randn(4, 3, 5)
        result = batch_matmul(A, B)
        self.assertEqual(result.shape, (4, 2, 5))

    def test_batch_matmul_values(self):
        A = np.array([[[1, 2], [3, 4]]])
        B = np.array([[[1, 0], [0, 1]]])
        result = batch_matmul(A, B)
        np.testing.assert_array_equal(result[0], A[0])

    def test_outer_product_basic(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = outer_product(a, b)
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        np.testing.assert_array_equal(result, expected)

    def test_outer_product_shape(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([1, 2])
        result = outer_product(a, b)
        self.assertEqual(result.shape, (4, 2))

    def test_outer_product_zeros(self):
        a = np.array([1, 2])
        b = np.array([0, 0])
        result = outer_product(a, b)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_matmul_associative(self):
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 4)
        C = np.random.randn(4, 2)
        r1 = matmul(matmul(A, B), C)
        r2 = matmul(A, matmul(B, C))
        np.testing.assert_array_almost_equal(r1, r2)

    def test_batch_matmul_consistency(self):
        A = np.random.randn(3, 2, 4)
        B = np.random.randn(3, 4, 5)
        result = batch_matmul(A, B)
        for i in range(3):
            expected = matmul(A[i], B[i])
            np.testing.assert_array_almost_equal(result[i], expected)
`,

	hint1: 'Use @ operator or np.matmul for matrix multiplication',
	hint2: 'np.outer(a, b) computes the outer product',

	whyItMatters: `Matrix multiplication powers all neural networks:

- **Dense layers**: output = weights @ input + bias
- **Batch processing**: Multiply entire batch at once
- **Attention**: Q @ K.T for attention scores
- **Gradient computation**: Backprop uses transposed matmul

GPUs are optimized specifically for matrix multiplication.`,

	translations: {
		ru: {
			title: 'Матричное умножение',
			description: `# Матричное умножение

Матричное умножение — основа вычислений нейронных сетей. Понимание форм критически важно.

## Задача

Реализуйте три функции:
1. \`matmul(A, B)\` - Стандартное матричное умножение
2. \`batch_matmul(A, B)\` - Батчевое матричное умножение (3D массивы)
3. \`outer_product(a, b)\` - Внешнее произведение двух векторов

## Пример

\`\`\`python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

matmul(A, B)  # [[19, 22], [43, 50]]

# Batch: (batch, m, k) @ (batch, k, n) -> (batch, m, n)
A_batch = np.random.randn(4, 2, 3)
B_batch = np.random.randn(4, 3, 2)
batch_matmul(A_batch, B_batch)  # (4, 2, 2)

a = np.array([1, 2, 3])
b = np.array([4, 5])
outer_product(a, b)  # [[4, 5], [8, 10], [12, 15]]
\`\`\``,
			hint1: 'Используйте оператор @ или np.matmul для матричного умножения',
			hint2: 'np.outer(a, b) вычисляет внешнее произведение',
			whyItMatters: `Матричное умножение питает все нейросети:

- **Dense слои**: output = weights @ input + bias
- **Batch обработка**: Умножение всего батча за раз
- **Attention**: Q @ K.T для attention scores
- **Вычисление градиентов**: Backprop использует транспонированное matmul`,
		},
		uz: {
			title: "Matritsali ko'paytirish",
			description: `# Matritsali ko'paytirish

Matritsali ko'paytirish neyron tarmoq hisoblashlarining asosidir. Shakllarni tushunish muhim.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`matmul(A, B)\` - Standart matritsali ko'paytirish
2. \`batch_matmul(A, B)\` - Batchli matritsali ko'paytirish (3D massivlar)
3. \`outer_product(a, b)\` - Ikki vektorning tashqi ko'paytmasi

## Misol

\`\`\`python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

matmul(A, B)  # [[19, 22], [43, 50]]

a = np.array([1, 2, 3])
b = np.array([4, 5])
outer_product(a, b)  # [[4, 5], [8, 10], [12, 15]]
\`\`\``,
			hint1: "Matritsali ko'paytirish uchun @ operatori yoki np.matmul dan foydalaning",
			hint2: "np.outer(a, b) tashqi ko'paytmani hisoblaydi",
			whyItMatters: `Matritsali ko'paytirish barcha neyron tarmoqlarni quvvatlaydi:

- **Dense qatlamlar**: output = weights @ input + bias
- **Batch ishlov berish**: Butun batchni bir vaqtda ko'paytirish
- **Attention**: Attention scores uchun Q @ K.T`,
		},
	},
};

export default task;
