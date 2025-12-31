import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-eigenvalues',
	title: 'Eigenvalues and Eigenvectors',
	difficulty: 'hard',
	tags: ['numpy', 'linear-algebra', 'eigenvalues'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Eigenvalues and Eigenvectors

Eigenvectors are vectors that only get scaled (not rotated) when multiplied by a matrix. The scaling factor is the eigenvalue.

## Task

Implement three functions:
1. \`compute_eigen(A)\` - Return eigenvalues and eigenvectors
2. \`verify_eigen(A, eigenvalue, eigenvector)\` - Verify A @ v = λ @ v
3. \`power_iteration(A, num_iterations)\` - Find dominant eigenvalue/vector

## Example

\`\`\`python
A = np.array([[4, -2], [1, 1]])

eigenvalues, eigenvectors = compute_eigen(A)
# eigenvalues: [3, 2]
# eigenvectors: columns are eigenvectors

# Verify: A @ v = λ * v
verify_eigen(A, 3, eigenvectors[:, 0])  # True

# Power iteration finds largest eigenvalue
val, vec = power_iteration(A, 100)  # val ≈ 3
\`\`\``,

	initialCode: `import numpy as np

def compute_eigen(A: np.ndarray) -> tuple:
    """Return (eigenvalues, eigenvectors) of matrix A."""
    # Your code here
    pass

def verify_eigen(A: np.ndarray, eigenvalue: float, eigenvector: np.ndarray) -> bool:
    """Verify that A @ v ≈ eigenvalue * v."""
    # Your code here
    pass

def power_iteration(A: np.ndarray, num_iterations: int = 100) -> tuple:
    """Find dominant eigenvalue and eigenvector using power iteration."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def compute_eigen(A: np.ndarray) -> tuple:
    """Return (eigenvalues, eigenvectors) of matrix A."""
    return np.linalg.eig(A)

def verify_eigen(A: np.ndarray, eigenvalue: float, eigenvector: np.ndarray) -> bool:
    """Verify that A @ v ≈ eigenvalue * v."""
    lhs = A @ eigenvector
    rhs = eigenvalue * eigenvector
    return np.allclose(lhs, rhs)

def power_iteration(A: np.ndarray, num_iterations: int = 100) -> tuple:
    """Find dominant eigenvalue and eigenvector using power iteration."""
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        Av = A @ v
        v = Av / np.linalg.norm(Av)

    eigenvalue = (v @ A @ v) / (v @ v)
    return eigenvalue, v
`,

	testCode: `import numpy as np
import unittest

class TestEigenvalues(unittest.TestCase):
    def test_compute_eigen_2x2(self):
        A = np.array([[4.0, -2.0], [1.0, 1.0]])
        vals, vecs = compute_eigen(A)
        self.assertEqual(len(vals), 2)
        self.assertEqual(vecs.shape, (2, 2))

    def test_compute_eigen_symmetric(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        vals, vecs = compute_eigen(A)
        # Symmetric matrices have real eigenvalues
        self.assertTrue(np.allclose(vals.imag, 0))

    def test_compute_eigen_identity(self):
        I = np.eye(3)
        vals, vecs = compute_eigen(I)
        np.testing.assert_array_almost_equal(vals, [1, 1, 1])

    def test_verify_eigen_true(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = verify_eigen(A, 2.0, np.array([1.0, 0.0]))
        self.assertTrue(result)

    def test_verify_eigen_false(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = verify_eigen(A, 5.0, np.array([1.0, 0.0]))
        self.assertFalse(result)

    def test_verify_eigen_computed(self):
        A = np.array([[4.0, -2.0], [1.0, 1.0]])
        vals, vecs = compute_eigen(A)
        for i in range(len(vals)):
            result = verify_eigen(A, vals[i], vecs[:, i])
            self.assertTrue(result)

    def test_power_iteration_dominant(self):
        A = np.array([[3.0, 1.0], [0.0, 2.0]])
        val, vec = power_iteration(A, 100)
        # Dominant eigenvalue is 3
        self.assertAlmostEqual(abs(val), 3.0, places=1)

    def test_power_iteration_symmetric(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        val, vec = power_iteration(A, 100)
        # Check eigenvector property
        Av = A @ vec
        ratio = Av / vec
        self.assertTrue(np.allclose(ratio, ratio[0]))

    def test_power_iteration_convergence(self):
        A = np.array([[5.0, 0.0], [0.0, 1.0]])
        val, vec = power_iteration(A, 50)
        self.assertAlmostEqual(val, 5.0, places=2)

    def test_eigenvalue_sum_trace(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        vals, _ = compute_eigen(A)
        # Sum of eigenvalues equals trace
        self.assertAlmostEqual(np.sum(vals), np.trace(A))
`,

	hint1: 'Use np.linalg.eig() to compute eigenvalues and eigenvectors',
	hint2: 'Power iteration: repeatedly multiply by A and normalize',

	whyItMatters: `Eigendecomposition is essential for:

- **PCA**: Principal components are eigenvectors of covariance matrix
- **PageRank**: Website importance is dominant eigenvector
- **Spectral clustering**: Uses eigenvalues of graph Laplacian
- **Stability analysis**: Eigenvalues determine system stability

Understanding eigen-decomposition unlocks many ML algorithms.`,

	translations: {
		ru: {
			title: 'Собственные значения и векторы',
			description: `# Собственные значения и векторы

Собственные векторы — это векторы, которые при умножении на матрицу только масштабируются (не вращаются). Коэффициент масштабирования — собственное значение.

## Задача

Реализуйте три функции:
1. \`compute_eigen(A)\` - Вернуть собственные значения и векторы
2. \`verify_eigen(A, eigenvalue, eigenvector)\` - Проверить A @ v = λ @ v
3. \`power_iteration(A, num_iterations)\` - Найти доминирующее собственное значение/вектор

## Пример

\`\`\`python
A = np.array([[4, -2], [1, 1]])

eigenvalues, eigenvectors = compute_eigen(A)
# eigenvalues: [3, 2]

# Проверка: A @ v = λ * v
verify_eigen(A, 3, eigenvectors[:, 0])  # True

# Степенная итерация находит наибольшее собственное значение
val, vec = power_iteration(A, 100)  # val ≈ 3
\`\`\``,
			hint1: 'Используйте np.linalg.eig() для вычисления собственных значений и векторов',
			hint2: 'Степенная итерация: многократное умножение на A и нормализация',
			whyItMatters: `Разложение на собственные значения необходимо для:

- **PCA**: Главные компоненты — собственные векторы ковариационной матрицы
- **PageRank**: Важность сайта — доминирующий собственный вектор
- **Spectral clustering**: Использует собственные значения лапласиана графа
- **Анализ стабильности**: Собственные значения определяют стабильность системы`,
		},
		uz: {
			title: "Xususiy qiymatlar va vektorlar",
			description: `# Xususiy qiymatlar va vektorlar

Xususiy vektorlar matritsaga ko'paytirilganda faqat masshtablanadigan (aylanmaydigan) vektorlardir. Masshtablash koeffitsiyenti xususiy qiymatdir.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_eigen(A)\` - Xususiy qiymatlar va vektorlarni qaytarish
2. \`verify_eigen(A, eigenvalue, eigenvector)\` - A @ v = λ @ v ni tekshirish
3. \`power_iteration(A, num_iterations)\` - Dominant xususiy qiymat/vektorni topish

## Misol

\`\`\`python
A = np.array([[4, -2], [1, 1]])

eigenvalues, eigenvectors = compute_eigen(A)

verify_eigen(A, 3, eigenvectors[:, 0])  # True

val, vec = power_iteration(A, 100)  # val ≈ 3
\`\`\``,
			hint1: "Xususiy qiymatlar va vektorlarni hisoblash uchun np.linalg.eig() dan foydalaning",
			hint2: "Daraja iteratsiyasi: takroriy A ga ko'paytirish va normalizatsiya",
			whyItMatters: `Eigendecomposition quyidagilar uchun zarur:

- **PCA**: Asosiy komponentlar kovariatsiya matritsasining xususiy vektorlari
- **PageRank**: Sayt ahamiyati dominant xususiy vektor
- **Spectral clustering**: Graf laplasianining xususiy qiymatlaridan foydalanadi`,
		},
	},
};

export default task;
