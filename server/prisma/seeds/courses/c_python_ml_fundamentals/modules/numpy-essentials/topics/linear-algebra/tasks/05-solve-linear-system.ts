import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-solve-linear-system',
	title: 'Solving Linear Systems',
	difficulty: 'medium',
	tags: ['numpy', 'linear-algebra', 'equations'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Solving Linear Systems

Solving Ax = b is one of the most common operations in scientific computing.

## Task

Implement three functions:
1. \`solve_system(A, b)\` - Solve Ax = b for x
2. \`least_squares_solution(A, b)\` - Solve overdetermined system (more equations than unknowns)
3. \`solve_multiple_rhs(A, B)\` - Solve AX = B for multiple right-hand sides

## Example

\`\`\`python
# System: 2x + y = 5, x + 3y = 7
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

solve_system(A, b)  # [1.6, 1.8] -> x=1.6, y=1.8

# Overdetermined: 3 equations, 2 unknowns
A_over = np.array([[1, 1], [2, 1], [1, 2]])
b_over = np.array([3, 4, 4])
least_squares_solution(A_over, b_over)  # Best fit solution

# Multiple right-hand sides
B = np.array([[5, 10], [7, 14]])
solve_multiple_rhs(A, B)  # Solutions for both columns of B
\`\`\``,

	initialCode: `import numpy as np

def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b for x."""
    # Your code here
    pass

def least_squares_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve overdetermined system using least squares."""
    # Your code here
    pass

def solve_multiple_rhs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve AX = B for multiple right-hand sides."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b for x."""
    return np.linalg.solve(A, b)

def least_squares_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve overdetermined system using least squares."""
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

def solve_multiple_rhs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve AX = B for multiple right-hand sides."""
    return np.linalg.solve(A, B)
`,

	testCode: `import numpy as np
import unittest

class TestSolveLinearSystem(unittest.TestCase):
    def test_solve_system_2x2(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = solve_system(A, b)
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_solve_system_3x3(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        b = np.array([6, 15, 25], dtype=float)
        x = solve_system(A, b)
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_solve_system_identity(self):
        I = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = solve_system(I, b)
        np.testing.assert_array_almost_equal(x, b)

    def test_least_squares_overdetermined(self):
        A = np.array([[1, 1], [2, 1], [1, 2]], dtype=float)
        b = np.array([3, 4, 4], dtype=float)
        x = least_squares_solution(A, b)
        self.assertEqual(len(x), 2)

    def test_least_squares_exact(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([3, 4], dtype=float)
        x = least_squares_solution(A, b)
        np.testing.assert_array_almost_equal(x, b)

    def test_least_squares_minimizes_residual(self):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        b = np.array([1, 2, 2], dtype=float)
        x = least_squares_solution(A, b)
        residual = np.linalg.norm(A @ x - b)
        # Check that random solution has higher residual
        random_x = np.array([0.5, 0.5])
        random_residual = np.linalg.norm(A @ random_x - b)
        self.assertLess(residual, random_residual + 0.1)

    def test_solve_multiple_rhs_basic(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        B = np.array([[5.0, 10.0], [7.0, 14.0]])
        X = solve_multiple_rhs(A, B)
        np.testing.assert_array_almost_equal(A @ X, B)

    def test_solve_multiple_rhs_shape(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6, 7], [8, 9, 10]], dtype=float)
        X = solve_multiple_rhs(A, B)
        self.assertEqual(X.shape, (2, 3))

    def test_solve_multiple_columns_independent(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        B = np.array([[1, 2], [3, 4]], dtype=float)
        X = solve_multiple_rhs(A, B)
        np.testing.assert_array_almost_equal(X, B)

    def test_solve_system_values(self):
        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)
        x = solve_system(A, b)
        self.assertAlmostEqual(x[0], 2.0, places=5)
        self.assertAlmostEqual(x[1], 3.0, places=5)
`,

	hint1: 'Use np.linalg.solve() for square systems',
	hint2: 'Use np.linalg.lstsq() for least squares solution',

	whyItMatters: `Solving linear systems is everywhere:

- **Linear regression**: Solve (X.T @ X) @ w = X.T @ y
- **Neural network initialization**: Solve for initial weights
- **Physics simulations**: Solve for forces, velocities
- **Optimization**: Newton's method solves linear systems each step

Efficient solvers handle millions of equations in seconds.`,

	translations: {
		ru: {
			title: 'Решение линейных систем',
			description: `# Решение линейных систем

Решение Ax = b — одна из самых частых операций в научных вычислениях.

## Задача

Реализуйте три функции:
1. \`solve_system(A, b)\` - Решить Ax = b для x
2. \`least_squares_solution(A, b)\` - Решить переопределённую систему
3. \`solve_multiple_rhs(A, B)\` - Решить AX = B для нескольких правых частей

## Пример

\`\`\`python
# Система: 2x + y = 5, x + 3y = 7
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

solve_system(A, b)  # [1.6, 1.8]

# Переопределённая: 3 уравнения, 2 неизвестных
A_over = np.array([[1, 1], [2, 1], [1, 2]])
b_over = np.array([3, 4, 4])
least_squares_solution(A_over, b_over)  # Наилучшее приближение
\`\`\``,
			hint1: 'Используйте np.linalg.solve() для квадратных систем',
			hint2: 'Используйте np.linalg.lstsq() для метода наименьших квадратов',
			whyItMatters: `Решение линейных систем везде:

- **Линейная регрессия**: Решение (X.T @ X) @ w = X.T @ y
- **Инициализация нейросетей**: Решение для начальных весов
- **Физические симуляции**: Решение для сил, скоростей
- **Оптимизация**: Метод Ньютона решает линейные системы на каждом шаге`,
		},
		uz: {
			title: "Chiziqli sistemalarni yechish",
			description: `# Chiziqli sistemalarni yechish

Ax = b ni yechish ilmiy hisoblashda eng keng tarqalgan operatsiyalardan biri.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`solve_system(A, b)\` - x uchun Ax = b ni yechish
2. \`least_squares_solution(A, b)\` - Ortiqcha aniqlangan sistemani yechish
3. \`solve_multiple_rhs(A, B)\` - Bir nechta o'ng tomonlar uchun AX = B ni yechish

## Misol

\`\`\`python
# Sistema: 2x + y = 5, x + 3y = 7
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

solve_system(A, b)  # [1.6, 1.8]
\`\`\``,
			hint1: "Kvadrat sistemalar uchun np.linalg.solve() dan foydalaning",
			hint2: "Eng kichik kvadratlar yechimi uchun np.linalg.lstsq() dan foydalaning",
			whyItMatters: `Chiziqli sistemalarni yechish hamma joyda:

- **Chiziqli regressiya**: (X.T @ X) @ w = X.T @ y ni yechish
- **Neyron tarmoq initializatsiyasi**: Boshlang'ich vaznlar uchun yechish
- **Fizika simulyatsiyalari**: Kuchlar, tezliklar uchun yechish`,
		},
	},
};

export default task;
