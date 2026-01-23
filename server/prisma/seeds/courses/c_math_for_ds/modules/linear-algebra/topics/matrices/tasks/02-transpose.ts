import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-matrix-transpose',
	title: 'Matrix Transpose',
	difficulty: 'easy',
	tags: ['python', 'math', 'linear-algebra', 'matrices', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Matrix Transpose

Implement a function that transposes a matrix (swaps rows and columns).

## Background

Transpose flips a matrix over its diagonal:
- Rows become columns
- A[i,j] becomes Aᵀ[j,i]
- Shape (m×n) becomes (n×m)

## Example

\`\`\`python
A = [[1, 2, 3],
     [4, 5, 6]]   # 2x3

Aᵀ = [[1, 4],
      [2, 5],
      [3, 6]]      # 3x2
\`\`\`
`,
	initialCode: `import numpy as np

def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix.

    Args:
        A: Input matrix

    Returns:
        Transposed matrix
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix.

    Args:
        A: Input matrix

    Returns:
        Transposed matrix
    """
    return A.T
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

class TestTranspose(unittest.TestCase):
    def test_2x3_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose(A)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert_array_close(result, expected)

    def test_square_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        result = transpose(A)
        expected = np.array([[1, 3], [2, 4]])
        assert_array_close(result, expected)

    def test_symmetric_matrix(self):
        A = np.array([[1, 2], [2, 1]])
        result = transpose(A)
        assert_array_close(result, A)

    def test_row_vector(self):
        A = np.array([[1, 2, 3]])
        result = transpose(A)
        expected = np.array([[1], [2], [3]])
        assert_array_close(result, expected)

    def test_column_vector(self):
        A = np.array([[1], [2], [3]])
        result = transpose(A)
        expected = np.array([[1, 2, 3]])
        assert_array_close(result, expected)

    def test_shape_change(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4
        result = transpose(A)
        assert result.shape == (4, 2, f"Expected (4, 2, got {result.shape}")

    def test_double_transpose(self):
        A = np.array([[1, 2], [3, 4], [5, 6]])
        result = transpose(transpose(A))
        assert_array_close(result, A)

    def test_identity_matrix(self):
        I = np.eye(3)
        result = transpose(I)
        assert_array_close(result, I)

    def test_float_values(self):
        A = np.array([[1.5, 2.5], [3.5, 4.5]])
        result = transpose(A)
        expected = np.array([[1.5, 3.5], [2.5, 4.5]])
        assert_array_close(result, expected)

    def test_returns_numpy_array(self):
        A = np.array([[1, 2], [3, 4]])
        result = transpose(A)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

`,
	hint1: 'Use the .T attribute: A.T',
	hint2: 'Alternatively: np.transpose(A) or A.transpose()',
	whyItMatters: `Transpose is used constantly in backpropagation. When computing gradients, you transpose weight matrices. In attention mechanisms, you compute Q @ Kᵀ (query times key transpose). **Production Pattern:** Reshape operations in neural networks often involve transposes to align dimensions correctly.`,
	translations: {
		ru: {
			title: 'Транспонирование матрицы',
			description: `# Транспонирование матрицы

Реализуйте функцию транспонирования матрицы (меняет строки и столбцы местами).

## Теория

Транспонирование отражает матрицу относительно диагонали:
- Строки становятся столбцами
- A[i,j] становится Aᵀ[j,i]
- Форма (m×n) становится (n×m)
`,
			hint1: 'Используйте атрибут .T: A.T',
			hint2: 'Альтернативно: np.transpose(A)',
			whyItMatters: `Транспонирование постоянно используется при обратном распространении ошибки. При вычислении градиентов транспонируются матрицы весов. **Production Pattern:** В attention механизмах вычисляется Q @ Kᵀ.`,
		},
		uz: {
			title: 'Matritsani transpozitsiya qilish',
			description: `# Matritsani transpozitsiya qilish

Matritsani transpozitsiya qiladigan (satr va ustunlarni almashtiradi) funksiyani amalga oshiring.
`,
			hint1: '.T atributidan foydalaning: A.T',
			hint2: 'Muqobil: np.transpose(A)',
			whyItMatters: `Transpozitsiya orqaga tarqalishda doimiy ishlatiladi. Gradientlarni hisoblashda og'irlik matritsalari transpozitsiya qilinadi.`,
		},
	},
};

export default task;
