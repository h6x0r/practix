import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-matrix-multiply',
	title: 'Matrix Multiplication',
	difficulty: 'medium',
	tags: ['python', 'math', 'linear-algebra', 'matrices', 'numpy'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Matrix Multiplication

Implement a function that multiplies two matrices.

## Background

Matrix multiplication is NOT element-wise. For A (m×n) and B (n×p):
- Result C has shape (m×p)
- Each element: C[i,j] = sum(A[i,k] * B[k,j]) for all k

## Rules

- A's columns must equal B's rows
- Order matters: A @ B ≠ B @ A (generally)

## Example

\`\`\`python
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

C = A @ B = [[1*5+2*7, 1*6+2*8],
             [3*5+4*7, 3*6+4*8]]
          = [[19, 22],
             [43, 50]]
\`\`\`
`,
	initialCode: `import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product matrix (m x p)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product matrix (m x p)
    """
    return A @ B  # or np.matmul(A, B) or np.dot(A, B)
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

class TestMatrixMultiply(unittest.TestCase):
    def test_2x2_matrices(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        assert_array_close(result, expected)

    def test_identity_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        I = np.eye(2)
        result = matrix_multiply(A, I)
        assert_array_close(result, A)

    def test_different_shapes(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        B = np.array([[1], [2], [3]])  # 3x1
        result = matrix_multiply(A, B)
        expected = np.array([[14], [32]])  # 2x1
        assert_array_close(result, expected)

    def test_vector_multiplication(self):
        A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
        v = np.array([[1], [1]])  # 2x1
        result = matrix_multiply(A, v)
        expected = np.array([[3], [7], [11]])
        assert_array_close(result, expected)

    def test_3x3_matrices(self):
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = matrix_multiply(A, B)
        assert_array_close(result, B)

    def test_float_values(self):
        A = np.array([[0.5, 1.5], [2.5, 3.5]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = matrix_multiply(A, B)
        assert_array_close(result, A)

    def test_negative_values(self):
        A = np.array([[1, -1], [-1, 1]])
        B = np.array([[1, 1], [1, 1]])
        result = matrix_multiply(A, B)
        expected = np.array([[0, 0], [0, 0]])
        assert_array_close(result, expected)

    def test_result_shape(self):
        A = np.array([[1, 2, 3]])  # 1x3
        B = np.array([[1], [2], [3]])  # 3x1
        result = matrix_multiply(A, B)
        assert result.shape == (1, 1, f"Expected (1, 1, got {result.shape}")

    def test_large_matrices(self):
        A = np.ones((3, 4))
        B = np.ones((4, 2))
        result = matrix_multiply(A, B)
        expected = np.full((3, 2), 4.0)
        assert_array_close(result, expected)

    def test_returns_numpy_array(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 0], [0, 1]])
        result = matrix_multiply(A, B)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

`,
	hint1: 'Use the @ operator for matrix multiplication: A @ B',
	hint2: 'Alternatively, use np.matmul(A, B) or np.dot(A, B) for 2D arrays.',
	whyItMatters: `Matrix multiplication IS neural network forward pass. Every layer computes output = activation(weights @ input + bias). The entire network is just a sequence of matrix multiplications. **Production Pattern:** GPUs are optimized for matrix multiplication - that's why they accelerate deep learning so dramatically.`,
	translations: {
		ru: {
			title: 'Умножение матриц',
			description: `# Умножение матриц

Реализуйте функцию умножения двух матриц.

## Теория

Умножение матриц НЕ поэлементное. Для A (m×n) и B (n×p):
- Результат C имеет форму (m×p)
- Каждый элемент: C[i,j] = сумма(A[i,k] * B[k,j])

## Правила

- Столбцы A должны равняться строкам B
- Порядок важен: A @ B ≠ B @ A
`,
			hint1: 'Используйте оператор @ для умножения матриц: A @ B',
			hint2: 'Альтернативно: np.matmul(A, B) или np.dot(A, B)',
			whyItMatters: `Умножение матриц - ЭТО прямой проход нейросети. Каждый слой вычисляет output = activation(weights @ input + bias). **Production Pattern:** GPU оптимизированы для умножения матриц - поэтому они так ускоряют глубокое обучение.`,
		},
		uz: {
			title: 'Matritsalarni ko\'paytirish',
			description: `# Matritsalarni ko'paytirish

Ikki matritsani ko'paytiradigan funksiyani amalga oshiring.

## Nazariya

Matritsa ko'paytirish elementma-element emas. A (m×n) va B (n×p) uchun:
- Natija C shakli (m×p)
`,
			hint1: 'Matritsa ko\'paytirish uchun @ operatoridan foydalaning: A @ B',
			hint2: 'Muqobil: np.matmul(A, B)',
			whyItMatters: `Matritsa ko'paytirish neyron tarmoqlarning oldinga o'tish jarayonidir. GPU'lar matritsa ko'paytirish uchun optimallashtirilgan.`,
		},
	},
};

export default task;
