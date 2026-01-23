import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-matrix-inverse',
	title: 'Matrix Inverse',
	difficulty: 'medium',
	tags: ['python', 'math', 'linear-algebra', 'matrices', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 3,
	description: `# Matrix Inverse

Implement a function that computes the inverse of a square matrix.

## Background

The inverse A⁻¹ satisfies: A @ A⁻¹ = A⁻¹ @ A = I (identity)

## Key Points

- Only square matrices can have inverses
- Not all square matrices are invertible (singular matrices)
- det(A) ≠ 0 for invertible matrices

## Use Cases

- **Solving linear equations**: x = A⁻¹ @ b
- **Covariance matrix operations**
- **Least squares solutions**

## Example

\`\`\`python
A = [[4, 7],
     [2, 6]]

A⁻¹ = [[0.6, -0.7],
       [-0.2, 0.4]]
\`\`\`
`,
	initialCode: `import numpy as np

def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square matrix.

    Args:
        A: Square matrix

    Returns:
        Inverse matrix A⁻¹
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square matrix.

    Args:
        A: Square matrix

    Returns:
        Inverse matrix A⁻¹
    """
    return np.linalg.inv(A)
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

class TestMatrixInverse(unittest.TestCase):
    def test_2x2_matrix(self):
        A = np.array([[4.0, 7.0], [2.0, 6.0]])
        result = matrix_inverse(A)
        expected = np.array([[0.6, -0.7], [-0.2, 0.4]])
        assert_array_close(result, expected)

    def test_identity_inverse(self):
        I = np.eye(3)
        result = matrix_inverse(I)
        assert_array_close(result, I)

    def test_inverse_property(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = matrix_inverse(A)
        product = A @ A_inv
        assert_array_close(product, np.eye(2))

    def test_3x3_matrix(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        result = matrix_inverse(A)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1/3]])
        assert_array_close(result, expected)

    def test_double_inverse(self):
        A = np.array([[2.0, 1.0], [1.0, 1.0]])
        result = matrix_inverse(matrix_inverse(A))
        assert_array_close(result, A)

    def test_inverse_times_original(self):
        A = np.array([[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]])
        A_inv = matrix_inverse(A)
        I = np.eye(3)
        np.testing.assert_array_almost_equal(A @ A_inv, I)

    def test_simple_2x2(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = matrix_inverse(A)
        expected = np.array([[1.0, 0.0], [0.0, 0.5]])
        assert_array_close(result, expected)

    def test_inverse_right_multiply(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv = matrix_inverse(A)
        np.testing.assert_array_almost_equal(A_inv @ A, np.eye(2))

    def test_returns_numpy_array(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = matrix_inverse(A)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_determinant_nonzero(self):
        A = np.array([[2.0, 1.0], [4.0, 3.0]])
        A_inv = matrix_inverse(A)
        self.assertIsNotNone(A_inv)

`,
	hint1: 'Use np.linalg.inv(A) to compute the matrix inverse.',
	hint2: 'The inverse exists only for square matrices with non-zero determinant.',
	whyItMatters: `Matrix inversion is key in optimization. Normal equations for linear regression: θ = (XᵀX)⁻¹Xᵀy. Covariance matrix inversion in Gaussian processes. **Production Pattern:** In practice, we avoid explicit inversion - use np.linalg.solve(A, b) instead of inv(A) @ b for numerical stability.`,
	translations: {
		ru: {
			title: 'Обратная матрица',
			description: `# Обратная матрица

Реализуйте функцию вычисления обратной матрицы.

## Теория

Обратная матрица A⁻¹ удовлетворяет: A @ A⁻¹ = A⁻¹ @ A = I (единичная)

## Ключевые моменты

- Только квадратные матрицы могут иметь обратные
- Не все квадратные матрицы обратимы (сингулярные матрицы)
- det(A) ≠ 0 для обратимых матриц
`,
			hint1: 'Используйте np.linalg.inv(A) для вычисления обратной матрицы.',
			hint2: 'Обратная матрица существует только для квадратных матриц с ненулевым определителем.',
			whyItMatters: `Обратная матрица ключевая в оптимизации. Нормальные уравнения линейной регрессии: θ = (XᵀX)⁻¹Xᵀy. **Production Pattern:** На практике избегают явного обращения - используют np.linalg.solve(A, b) для численной устойчивости.`,
		},
		uz: {
			title: 'Teskari matritsa',
			description: `# Teskari matritsa

Teskari matritsani hisoblaydigan funksiyani amalga oshiring.

## Nazariya

Teskari matritsa A⁻¹ shartni qanoatlantiradi: A @ A⁻¹ = I (birlik)
`,
			hint1: 'Teskari matritsani hisoblash uchun np.linalg.inv(A) dan foydalaning.',
			hint2: 'Teskari matritsa faqat nol bo\'lmagan determinantli kvadrat matritsalar uchun mavjud.',
			whyItMatters: `Teskari matritsa optimizatsiyada muhim. Chiziqli regressiyaning normal tenglamalari: θ = (XᵀX)⁻¹Xᵀy.`,
		},
	},
};

export default task;
