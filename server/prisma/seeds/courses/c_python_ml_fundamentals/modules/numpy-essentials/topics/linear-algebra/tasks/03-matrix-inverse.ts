import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-matrix-inverse',
	title: 'Matrix Inverse and Determinant',
	difficulty: 'medium',
	tags: ['numpy', 'linear-algebra', 'inverse'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Matrix Inverse and Determinant

The inverse of a matrix A is the matrix A⁻¹ such that A @ A⁻¹ = I (identity matrix).

## Task

Implement three functions:
1. \`compute_inverse(A)\` - Compute matrix inverse
2. \`compute_determinant(A)\` - Compute matrix determinant
3. \`is_invertible(A)\` - Check if matrix is invertible (det ≠ 0)

## Example

\`\`\`python
A = np.array([[4, 7], [2, 6]])

compute_inverse(A)  # [[ 0.6, -0.7], [-0.2,  0.4]]
compute_determinant(A)  # 10.0

A @ compute_inverse(A)  # [[1, 0], [0, 1]] (identity)

singular = np.array([[1, 2], [2, 4]])
is_invertible(singular)  # False (rows are linearly dependent)
\`\`\``,

	initialCode: `import numpy as np

def compute_inverse(A: np.ndarray) -> np.ndarray:
    """Compute matrix inverse."""
    # Your code here
    pass

def compute_determinant(A: np.ndarray) -> float:
    """Compute matrix determinant."""
    # Your code here
    pass

def is_invertible(A: np.ndarray) -> bool:
    """Check if matrix is invertible (det ≠ 0)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def compute_inverse(A: np.ndarray) -> np.ndarray:
    """Compute matrix inverse."""
    return np.linalg.inv(A)

def compute_determinant(A: np.ndarray) -> float:
    """Compute matrix determinant."""
    return np.linalg.det(A)

def is_invertible(A: np.ndarray) -> bool:
    """Check if matrix is invertible (det ≠ 0)."""
    return np.abs(np.linalg.det(A)) > 1e-10
`,

	testCode: `import numpy as np
import unittest

class TestMatrixInverse(unittest.TestCase):
    def test_inverse_basic(self):
        A = np.array([[4.0, 7.0], [2.0, 6.0]])
        inv = compute_inverse(A)
        result = A @ inv
        np.testing.assert_array_almost_equal(result, np.eye(2))

    def test_inverse_3x3(self):
        A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float)
        inv = compute_inverse(A)
        result = A @ inv
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_inverse_identity(self):
        I = np.eye(3)
        inv = compute_inverse(I)
        np.testing.assert_array_almost_equal(inv, I)

    def test_determinant_2x2(self):
        A = np.array([[4.0, 7.0], [2.0, 6.0]])
        det = compute_determinant(A)
        self.assertAlmostEqual(det, 10.0)

    def test_determinant_identity(self):
        I = np.eye(4)
        det = compute_determinant(I)
        self.assertAlmostEqual(det, 1.0)

    def test_determinant_singular(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        det = compute_determinant(A)
        self.assertAlmostEqual(det, 0.0)

    def test_is_invertible_true(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        self.assertTrue(is_invertible(A))

    def test_is_invertible_false(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        self.assertFalse(is_invertible(A))

    def test_is_invertible_identity(self):
        I = np.eye(3)
        self.assertTrue(is_invertible(I))

    def test_inverse_inverse(self):
        A = np.array([[1, 2], [3, 5]], dtype=float)
        inv_inv = compute_inverse(compute_inverse(A))
        np.testing.assert_array_almost_equal(inv_inv, A)
`,

	hint1: 'Use np.linalg.inv() for inverse, np.linalg.det() for determinant',
	hint2: 'A matrix is invertible if its determinant is not zero',

	whyItMatters: `Matrix inverse is used in:

- **Linear regression**: (X.T @ X)⁻¹ @ X.T @ y for closed-form solution
- **Kalman filters**: State estimation in robotics and tracking
- **Principal Component Analysis**: Whitening transformations
- **Numerical optimization**: Newton's method uses Hessian inverse

Understanding singularity helps debug ill-conditioned problems.`,

	translations: {
		ru: {
			title: 'Обратная матрица и определитель',
			description: `# Обратная матрица и определитель

Обратная матрица A — это матрица A⁻¹, такая что A @ A⁻¹ = I (единичная матрица).

## Задача

Реализуйте три функции:
1. \`compute_inverse(A)\` - Вычислить обратную матрицу
2. \`compute_determinant(A)\` - Вычислить определитель
3. \`is_invertible(A)\` - Проверить обратимость (det ≠ 0)

## Пример

\`\`\`python
A = np.array([[4, 7], [2, 6]])

compute_inverse(A)  # [[ 0.6, -0.7], [-0.2,  0.4]]
compute_determinant(A)  # 10.0

A @ compute_inverse(A)  # [[1, 0], [0, 1]] (единичная)

singular = np.array([[1, 2], [2, 4]])
is_invertible(singular)  # False (строки линейно зависимы)
\`\`\``,
			hint1: 'Используйте np.linalg.inv() для обратной, np.linalg.det() для определителя',
			hint2: 'Матрица обратима, если её определитель не равен нулю',
			whyItMatters: `Обратная матрица используется в:

- **Линейная регрессия**: (X.T @ X)⁻¹ @ X.T @ y для закрытой формы
- **Фильтры Калмана**: Оценка состояния в робототехнике
- **PCA**: Отбеливающие преобразования
- **Численная оптимизация**: Метод Ньютона использует обратный гессиан`,
		},
		uz: {
			title: "Teskari matritsa va determinant",
			description: `# Teskari matritsa va determinant

A matritsaning teskari matritsasi A⁻¹ bo'lib, A @ A⁻¹ = I (birlik matritsa).

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_inverse(A)\` - Teskari matritsani hisoblash
2. \`compute_determinant(A)\` - Determinantni hisoblash
3. \`is_invertible(A)\` - Teskarilanish mumkinligini tekshirish (det ≠ 0)

## Misol

\`\`\`python
A = np.array([[4, 7], [2, 6]])

compute_inverse(A)  # [[ 0.6, -0.7], [-0.2,  0.4]]
compute_determinant(A)  # 10.0

singular = np.array([[1, 2], [2, 4]])
is_invertible(singular)  # False
\`\`\``,
			hint1: "Teskari uchun np.linalg.inv(), determinant uchun np.linalg.det() dan foydalaning",
			hint2: "Matritsa determinanti nolga teng bo'lmasa teskarilanuvchan",
			whyItMatters: `Teskari matritsa quyidagilarda ishlatiladi:

- **Chiziqli regressiya**: Yopiq shakl yechimi uchun (X.T @ X)⁻¹ @ X.T @ y
- **Kalman filtrlari**: Robototexnika va kuzatishda holat baholash
- **PCA**: Oqartirish transformatsiyalari`,
		},
	},
};

export default task;
