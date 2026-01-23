import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-eigenvalues',
	title: 'Eigenvalues & Eigenvectors',
	difficulty: 'hard',
	tags: ['python', 'math', 'linear-algebra', 'matrices', 'numpy', 'eigenvalues'],
	estimatedTime: '25m',
	isPremium: true,
	order: 5,
	description: `# Eigenvalues & Eigenvectors

Implement a function that computes eigenvalues and eigenvectors of a matrix.

## Background

For matrix A, eigenvector v and eigenvalue λ satisfy:
\`\`\`
A @ v = λ * v
\`\`\`

The eigenvector v doesn't change direction when transformed by A - it only scales by λ.

## Use Cases in ML

- **PCA**: Principal components are eigenvectors of covariance matrix
- **PageRank**: Dominant eigenvector of link matrix
- **Spectral clustering**: Eigenvectors of graph Laplacian

## Example

\`\`\`python
A = [[2, 1],
     [1, 2]]
eigenvalues: [3, 1]
eigenvectors: [[0.707, -0.707], [0.707, 0.707]]
\`\`\`
`,
	initialCode: `import numpy as np
from typing import Tuple

def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a matrix.

    Args:
        A: Square matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
        - eigenvalues: 1D array of eigenvalues
        - eigenvectors: 2D array where each column is an eigenvector
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Tuple

def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a matrix.

    Args:
        A: Square matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
        - eigenvalues: 1D array of eigenvalues
        - eigenvectors: 2D array where each column is an eigenvector
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors
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

class TestEigenDecomposition(unittest.TestCase):
    def test_symmetric_matrix(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        eigenvalues, eigenvectors = eigen_decomposition(A)
        assert len(eigenvalues) == 2, f"Expected 2, got {len(eigenvalues)}"
        assert eigenvectors.shape == (2, 2, f"Expected (2, 2, got {eigenvectors.shape}")

    def test_eigenvalue_equation(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        eigenvalues, eigenvectors = eigen_decomposition(A)
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            np.testing.assert_array_almost_equal(A @ v, lam * v)

    def test_identity_matrix(self):
        I = np.eye(3)
        eigenvalues, _ = eigen_decomposition(I)
        np.testing.assert_array_almost_equal(sorted(eigenvalues.real), [1, 1, 1])

    def test_diagonal_matrix(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        eigenvalues, _ = eigen_decomposition(A)
        assert 1.99 < min(eigenvalues.real < 2.01 or 2.99 < max(eigenvalues.real) < 3.01)

    def test_3x3_matrix(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        eigenvalues, eigenvectors = eigen_decomposition(A)
        assert len(eigenvalues) == 3, f"Expected 3, got {len(eigenvalues)}"

    def test_returns_tuple(self):
        A = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = eigen_decomposition(A)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"
        assert len(result) == 2, f"Expected 2, got {len(result)}"

    def test_eigenvector_shape(self):
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        _, eigenvectors = eigen_decomposition(A)
        assert eigenvectors.shape == (3, 3, f"Expected (3, 3, got {eigenvectors.shape}")

    def test_eigenvalue_count(self):
        A = np.eye(4)
        eigenvalues, _ = eigen_decomposition(A)
        assert len(eigenvalues) == 4, f"Expected 4, got {len(eigenvalues)}"

    def test_real_eigenvalues_for_symmetric(self):
        A = np.array([[4.0, 2.0], [2.0, 1.0]])
        eigenvalues, _ = eigen_decomposition(A)
        np.testing.assert_array_almost_equal(eigenvalues.imag, [0, 0])

    def test_reconstruction(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        eigenvalues, eigenvectors = eigen_decomposition(A)
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        np.testing.assert_array_almost_equal(reconstructed.real, A)

`,
	hint1: 'Use np.linalg.eig(A) to compute eigenvalues and eigenvectors.',
	hint2: 'The function returns (eigenvalues, eigenvectors) where eigenvectors[:,i] corresponds to eigenvalues[i].',
	whyItMatters: `Eigendecomposition is the heart of PCA (Principal Component Analysis). The principal components that preserve most variance are the eigenvectors with largest eigenvalues. **Production Pattern:** In recommendation systems, matrix factorization techniques like SVD use eigendecomposition to find latent features.`,
	translations: {
		ru: {
			title: 'Собственные значения и векторы',
			description: `# Собственные значения и векторы

Реализуйте функцию вычисления собственных значений и векторов матрицы.

## Теория

Для матрицы A собственный вектор v и собственное значение λ удовлетворяют:
\`\`\`
A @ v = λ * v
\`\`\`

## Применение в ML

- **PCA**: Главные компоненты - собственные векторы ковариационной матрицы
- **PageRank**: Доминирующий собственный вектор матрицы ссылок
- **Спектральная кластеризация**: Собственные векторы лапласиана графа
`,
			hint1: 'Используйте np.linalg.eig(A) для вычисления собственных значений и векторов.',
			hint2: 'Функция возвращает (eigenvalues, eigenvectors), где eigenvectors[:,i] соответствует eigenvalues[i].',
			whyItMatters: `Разложение по собственным значениям - основа PCA. Главные компоненты с наибольшей дисперсией - собственные векторы с наибольшими собственными значениями. **Production Pattern:** SVD в рекомендательных системах использует собственное разложение.`,
		},
		uz: {
			title: 'Xususiy qiymatlar va vektorlar',
			description: `# Xususiy qiymatlar va vektorlar

Matritsaning xususiy qiymatlari va vektorlarini hisoblaydigan funksiyani amalga oshiring.

## ML da qo'llanilishi

- **PCA**: Asosiy komponentlar kovariatsiya matritsasining xususiy vektorlari
- **PageRank**: Havola matritsasining dominant xususiy vektori
`,
			hint1: 'Xususiy qiymatlar va vektorlarni hisoblash uchun np.linalg.eig(A) dan foydalaning.',
			hint2: 'Funksiya (eigenvalues, eigenvectors) qaytaradi.',
			whyItMatters: `Xususiy qiymatlarga ajratish PCA ning asosi. Eng katta dispersiyani saqlaydigan asosiy komponentlar eng katta xususiy qiymatli xususiy vektorlardir.`,
		},
	},
};

export default task;
