import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-svd-decomposition',
	title: 'Singular Value Decomposition (SVD)',
	difficulty: 'hard',
	tags: ['numpy', 'linear-algebra', 'svd', 'decomposition'],
	estimatedTime: '20m',
	isPremium: true,
	order: 6,
	description: `# Singular Value Decomposition (SVD)

SVD decomposes any matrix A into A = U @ S @ V.T, where U and V are orthogonal and S is diagonal.

## Task

Implement three functions:
1. \`compute_svd(A)\` - Return U, singular values, and V.T
2. \`low_rank_approximation(A, k)\` - Reconstruct A using only top k singular values
3. \`compute_rank(A)\` - Compute matrix rank using SVD

## Example

\`\`\`python
A = np.array([[1, 2], [3, 4], [5, 6]])

U, s, Vt = compute_svd(A)
# A ≈ U @ np.diag(s) @ Vt

# Low-rank approximation
A_approx = low_rank_approximation(A, k=1)  # Rank-1 approximation

# Matrix rank
compute_rank(A)  # 2 (full rank for 3x2 matrix)
\`\`\``,

	initialCode: `import numpy as np

def compute_svd(A: np.ndarray) -> tuple:
    """Return (U, singular_values, Vt) from SVD."""
    # Your code here
    pass

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct A using only top k singular values."""
    # Your code here
    pass

def compute_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Compute matrix rank using SVD."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def compute_svd(A: np.ndarray) -> tuple:
    """Return (U, singular_values, Vt) from SVD."""
    return np.linalg.svd(A)

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct A using only top k singular values."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def compute_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Compute matrix rank using SVD."""
    s = np.linalg.svd(A, compute_uv=False)
    return np.sum(s > tol)
`,

	testCode: `import numpy as np
import unittest

class TestSVDDecomposition(unittest.TestCase):
    def test_svd_reconstruction(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        U, s, Vt = compute_svd(A)
        reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_array_almost_equal(A, reconstructed)

    def test_svd_shapes(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        U, s, Vt = compute_svd(A)
        self.assertEqual(U.shape[0], 2)
        self.assertEqual(Vt.shape[1], 3)

    def test_svd_orthogonal_U(self):
        A = np.random.randn(4, 3)
        U, s, Vt = compute_svd(A)
        UtU = U.T @ U
        np.testing.assert_array_almost_equal(UtU, np.eye(UtU.shape[0]))

    def test_low_rank_shape(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        A_approx = low_rank_approximation(A, k=2)
        self.assertEqual(A_approx.shape, A.shape)

    def test_low_rank_k1(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        A_approx = low_rank_approximation(A, k=1)
        # Rank-1 matrix: rows are proportional
        ratio = A_approx[0, :] / A_approx[1, :]
        self.assertAlmostEqual(ratio[0], ratio[1])

    def test_low_rank_full_rank(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        A_approx = low_rank_approximation(A, k=2)
        np.testing.assert_array_almost_equal(A_approx, A)

    def test_compute_rank_full(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        rank = compute_rank(A)
        self.assertEqual(rank, 2)

    def test_compute_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        rank = compute_rank(A)
        self.assertEqual(rank, 1)

    def test_compute_rank_rectangular(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        rank = compute_rank(A)
        self.assertEqual(rank, 2)

    def test_svd_singular_values_sorted(self):
        A = np.random.randn(5, 4)
        U, s, Vt = compute_svd(A)
        self.assertTrue(all(s[i] >= s[i+1] for i in range(len(s)-1)))
`,

	hint1: 'Use np.linalg.svd() for decomposition',
	hint2: 'For low-rank: use only first k columns of U, k singular values, k rows of Vt',

	whyItMatters: `SVD is fundamental to modern ML:

- **Dimensionality reduction**: PCA via SVD is the most common technique
- **Recommendation systems**: Matrix factorization for Netflix-style recommendations
- **Image compression**: Approximate images with low-rank matrices
- **NLP**: Latent Semantic Analysis uses SVD on term-document matrices
- **Noise reduction**: Filter out small singular values

SVD is one of the most important algorithms in data science.`,

	translations: {
		ru: {
			title: 'Сингулярное разложение (SVD)',
			description: `# Сингулярное разложение (SVD)

SVD разлагает любую матрицу A в A = U @ S @ V.T, где U и V ортогональны, а S диагональна.

## Задача

Реализуйте три функции:
1. \`compute_svd(A)\` - Вернуть U, сингулярные значения и V.T
2. \`low_rank_approximation(A, k)\` - Восстановить A используя только top k сингулярных значений
3. \`compute_rank(A)\` - Вычислить ранг матрицы с помощью SVD

## Пример

\`\`\`python
A = np.array([[1, 2], [3, 4], [5, 6]])

U, s, Vt = compute_svd(A)
# A ≈ U @ np.diag(s) @ Vt

# Низкоранговое приближение
A_approx = low_rank_approximation(A, k=1)

# Ранг матрицы
compute_rank(A)  # 2
\`\`\``,
			hint1: 'Используйте np.linalg.svd() для разложения',
			hint2: 'Для низкого ранга: используйте только первые k столбцов U, k сингулярных значений, k строк Vt',
			whyItMatters: `SVD фундаментально для современного ML:

- **Снижение размерности**: PCA через SVD — самая распространённая техника
- **Рекомендательные системы**: Матричная факторизация для рекомендаций
- **Сжатие изображений**: Аппроксимация изображений низкоранговыми матрицами
- **NLP**: Латентный семантический анализ использует SVD`,
		},
		uz: {
			title: "Singular qiymat dekompozitsiyasi (SVD)",
			description: `# Singular qiymat dekompozitsiyasi (SVD)

SVD har qanday A matritsani A = U @ S @ V.T ga ajratadi, bu yerda U va V ortogonal, S esa diagonal.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_svd(A)\` - U, singular qiymatlar va V.T ni qaytarish
2. \`low_rank_approximation(A, k)\` - Faqat top k singular qiymatlardan foydalanib A ni qayta tiklash
3. \`compute_rank(A)\` - SVD yordamida matritsa rangini hisoblash

## Misol

\`\`\`python
A = np.array([[1, 2], [3, 4], [5, 6]])

U, s, Vt = compute_svd(A)

A_approx = low_rank_approximation(A, k=1)

compute_rank(A)  # 2
\`\`\``,
			hint1: "Dekompozitsiya uchun np.linalg.svd() dan foydalaning",
			hint2: "Past rang uchun: faqat U ning birinchi k ustunlari, k singular qiymatlar, Vt ning k qatorlaridan foydalaning",
			whyItMatters: `SVD zamonaviy ML uchun asosiydir:

- **O'lchamni kamaytirish**: SVD orqali PCA eng keng tarqalgan texnika
- **Tavsiya tizimlari**: Tavsiyalar uchun matritsa faktorizatsiyasi
- **Tasvir siqish**: Rasmlarni past rangdagi matritsalar bilan taxminlash`,
		},
	},
};

export default task;
