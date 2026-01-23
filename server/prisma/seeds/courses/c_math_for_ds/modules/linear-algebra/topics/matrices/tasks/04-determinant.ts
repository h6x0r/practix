import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-determinant',
	title: 'Matrix Determinant',
	difficulty: 'medium',
	tags: ['python', 'math', 'linear-algebra', 'matrices', 'numpy'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Matrix Determinant

Implement a function that computes the determinant of a square matrix.

## Background

The determinant is a scalar value that encodes properties of a matrix:
- **det = 0**: Matrix is singular (not invertible)
- **det ≠ 0**: Matrix is invertible
- **|det|**: Scaling factor of the transformation

## 2×2 Formula

\`\`\`
det([[a, b], [c, d]]) = ad - bc
\`\`\`

## Example

\`\`\`python
A = [[1, 2],
     [3, 4]]
det(A) = 1*4 - 2*3 = -2
\`\`\`
`,
	initialCode: `import numpy as np

def determinant(A: np.ndarray) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        A: Square matrix

    Returns:
        Determinant value
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def determinant(A: np.ndarray) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        A: Square matrix

    Returns:
        Determinant value
    """
    return np.linalg.det(A)
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

class TestDeterminant(unittest.TestCase):
    def test_2x2_simple(self):
        A = np.array([[1, 2], [3, 4]])
        result = determinant(A)
        assert_close(result, -2.0)

    def test_identity_matrix(self):
        I = np.eye(3)
        result = determinant(I)
        assert_close(result, 1.0)

    def test_singular_matrix(self):
        A = np.array([[1, 2], [2, 4]])
        result = determinant(A)
        assert_close(result, 0.0)

    def test_diagonal_matrix(self):
        A = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        result = determinant(A)
        assert_close(result, 24.0)

    def test_3x3_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = determinant(A)
        assert_close(result, 0.0)

    def test_negative_determinant(self):
        A = np.array([[0, 1], [1, 0]])
        result = determinant(A)
        assert_close(result, -1.0)

    def test_positive_determinant(self):
        A = np.array([[3, 1], [1, 2]])
        result = determinant(A)
        assert_close(result, 5.0)

    def test_float_values(self):
        A = np.array([[1.5, 2.5], [0.5, 1.5]])
        result = determinant(A)
        assert_close(result, 1.0)

    def test_4x4_identity(self):
        I = np.eye(4)
        result = determinant(I)
        assert_close(result, 1.0)

    def test_returns_scalar(self):
        A = np.array([[1, 2], [3, 4]])
        result = determinant(A)
        assert np.isscalar(result or result.shape == ())

`,
	hint1: 'Use np.linalg.det(A) to compute the determinant.',
	hint2: 'For 2x2 matrices: det([[a,b],[c,d]]) = a*d - b*c',
	whyItMatters: `Determinants detect singularity (when matrix operations fail). In optimization, the Hessian determinant indicates if we're at a minimum, maximum, or saddle point. **Production Pattern:** Before inverting a matrix, check if determinant is non-zero to avoid numerical instability.`,
	translations: {
		ru: {
			title: 'Определитель матрицы',
			description: `# Определитель матрицы

Реализуйте функцию вычисления определителя квадратной матрицы.

## Теория

Определитель - скалярное значение, кодирующее свойства матрицы:
- **det = 0**: Матрица сингулярна (необратима)
- **det ≠ 0**: Матрица обратима
- **|det|**: Коэффициент масштабирования преобразования
`,
			hint1: 'Используйте np.linalg.det(A) для вычисления определителя.',
			hint2: 'Для матриц 2x2: det([[a,b],[c,d]]) = a*d - b*c',
			whyItMatters: `Определители определяют сингулярность (когда матричные операции не работают). В оптимизации определитель гессиана показывает тип экстремума. **Production Pattern:** Перед обращением матрицы проверяйте ненулевой определитель.`,
		},
		uz: {
			title: 'Matritsa determinanti',
			description: `# Matritsa determinanti

Kvadrat matritsaning determinantini hisoblaydigan funksiyani amalga oshiring.

## Nazariya

Determinant - matritsa xususiyatlarini kodlovchi skalyar qiymat.
`,
			hint1: 'Determinantni hisoblash uchun np.linalg.det(A) dan foydalaning.',
			hint2: '2x2 matritsalar uchun: det([[a,b],[c,d]]) = a*d - b*c',
			whyItMatters: `Determinantlar singularlikni aniqlaydi. Matritsani teskari qilishdan oldin determinantning nol emasligini tekshiring.`,
		},
	},
};

export default task;
