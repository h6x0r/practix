import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-matrix-transpose',
	title: 'Matrix Transpose',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'lists', 'matrix'],
	estimatedTime: '15m',
	isPremium: false,
	order: 10,

	description: `# Matrix Transpose

A matrix transpose flips a matrix over its diagonal, swapping rows and columns.

## Task

Implement the function \`transpose(matrix)\` that returns the transpose of a 2D matrix.

## Requirements

- Convert rows to columns and columns to rows
- Handle non-square matrices (rows and columns may differ)
- Return empty list for empty input

## Examples

\`\`\`python
>>> transpose([[1, 2, 3], [4, 5, 6]])
[[1, 4], [2, 5], [3, 6]]

>>> transpose([[1, 2], [3, 4], [5, 6]])
[[1, 3, 5], [2, 4, 6]]

>>> transpose([[1]])
[[1]]

>>> transpose([])
[]
\`\`\``,

	initialCode: `def transpose(matrix: list[list]) -> list[list]:
    """Transpose a 2D matrix (swap rows and columns).

    Args:
        matrix: 2D list where all rows have the same length

    Returns:
        Transposed matrix (rows become columns)
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def transpose(matrix: list[list]) -> list[list]:
    """Transpose a 2D matrix (swap rows and columns).

    Args:
        matrix: 2D list where all rows have the same length

    Returns:
        Transposed matrix (rows become columns)
    """
    # Handle empty matrix
    if not matrix or not matrix[0]:
        return []

    # Get dimensions
    rows = len(matrix)
    cols = len(matrix[0])

    # Create transposed matrix using list comprehension
    # For each column index, collect all values from that column
    return [[matrix[row][col] for row in range(rows)] for col in range(cols)]`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """2x3 matrix"""
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_2(self):
        """3x2 matrix"""
        self.assertEqual(transpose([[1, 2], [3, 4], [5, 6]]), [[1, 3, 5], [2, 4, 6]])

    def test_3(self):
        """1x1 matrix"""
        self.assertEqual(transpose([[1]]), [[1]])

    def test_4(self):
        """Empty matrix"""
        self.assertEqual(transpose([]), [])

    def test_5(self):
        """Square 2x2 matrix"""
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_6(self):
        """Square 3x3 matrix"""
        result = transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(result, [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    def test_7(self):
        """1 row matrix"""
        self.assertEqual(transpose([[1, 2, 3]]), [[1], [2], [3]])

    def test_8(self):
        """1 column matrix"""
        self.assertEqual(transpose([[1], [2], [3]]), [[1, 2, 3]])

    def test_9(self):
        """With negative numbers"""
        self.assertEqual(transpose([[-1, 2], [3, -4]]), [[-1, 3], [2, -4]])

    def test_10(self):
        """Transpose twice returns original"""
        original = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(transpose(transpose(original)), original)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'The element at [row][col] in the original becomes [col][row] in the transposed matrix.',
	hint2: 'Use nested list comprehension: [[matrix[row][col] for row in range(rows)] for col in range(cols)].',

	whyItMatters: `Matrix operations are fundamental to data science, graphics, and machine learning.

**Production Pattern:**

\`\`\`python
def transpose_with_zip(matrix: list[list]) -> list[list]:
    """Pythonic transpose using zip."""
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]

def rotate_90_clockwise(matrix: list[list]) -> list[list]:
    """Rotate matrix 90 degrees clockwise."""
    # Transpose then reverse each row
    transposed = [list(row) for row in zip(*matrix)]
    return [row[::-1] for row in transposed]

def matrix_multiply(a: list[list], b: list[list]) -> list[list]:
    """Multiply two matrices."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError("Incompatible dimensions")

    result = [[0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result

# In practice, use numpy for matrix operations:
# import numpy as np
# transposed = np.array(matrix).T
\`\`\`

**Practical Benefits:**
- Image processing uses matrix transformations
- Neural networks rely heavily on matrix operations
- Understanding indexing helps with data manipulation`,

	translations: {
		ru: {
			title: 'Транспонирование матрицы',
			description: `# Транспонирование матрицы

Транспонирование матрицы — это отражение матрицы по диагонали с обменом строк и столбцов.

## Задача

Реализуйте функцию \`transpose(matrix)\`, которая возвращает транспонированную 2D матрицу.

## Требования

- Преобразуйте строки в столбцы и столбцы в строки
- Обработайте неквадратные матрицы (строки и столбцы могут различаться)
- Для пустого ввода верните пустой список

## Примеры

\`\`\`python
>>> transpose([[1, 2, 3], [4, 5, 6]])
[[1, 4], [2, 5], [3, 6]]

>>> transpose([[1, 2], [3, 4], [5, 6]])
[[1, 3, 5], [2, 4, 6]]

>>> transpose([[1]])
[[1]]

>>> transpose([])
[]
\`\`\``,
			hint1: 'Элемент в [row][col] в оригинале становится [col][row] в транспонированной.',
			hint2: 'Используйте вложенный list comprehension: [[matrix[row][col] for row in range(rows)] for col in range(cols)].',
			whyItMatters: `Операции с матрицами фундаментальны для data science и машинного обучения.

**Продакшен паттерн:**

\`\`\`python
def transpose_with_zip(matrix: list[list]) -> list[list]:
    """Pythonic транспонирование через zip."""
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]

def rotate_90_clockwise(matrix: list[list]) -> list[list]:
    """Поворот матрицы на 90° по часовой."""
    transposed = [list(row) for row in zip(*matrix)]
    return [row[::-1] for row in transposed]
\`\`\`

**Практические преимущества:**
- Обработка изображений использует матричные преобразования
- Нейросети опираются на операции с матрицами`,
		},
		uz: {
			title: 'Matritsani transpozitsiyalash',
			description: `# Matritsani transpozitsiyalash

Matritsani transpozitsiyalash — matritsani diagonali bo'ylab aks ettirish, qatorlar va ustunlarni almashtirish.

## Vazifa

2D matritsaning transpozitsiyasini qaytaruvchi \`transpose(matrix)\` funksiyasini amalga oshiring.

## Talablar

- Qatorlarni ustunlarga va ustunlarni qatorlarga aylantiring
- Kvadrat bo'lmagan matritsalarni ishlang (qatorlar va ustunlar farq qilishi mumkin)
- Bo'sh kirish uchun bo'sh ro'yxat qaytaring

## Misollar

\`\`\`python
>>> transpose([[1, 2, 3], [4, 5, 6]])
[[1, 4], [2, 5], [3, 6]]

>>> transpose([[1, 2], [3, 4], [5, 6]])
[[1, 3, 5], [2, 4, 6]]

>>> transpose([[1]])
[[1]]

>>> transpose([])
[]
\`\`\``,
			hint1: "Asl [row][col] dagi element transpozitsiyada [col][row] bo'ladi.",
			hint2: 'Ichma-ich list comprehension ishlating: [[matrix[row][col] for row in range(rows)] for col in range(cols)].',
			whyItMatters: `Matritsa amallari data science va mashinali o'rganish uchun asosiy.

**Ishlab chiqarish patterni:**

\`\`\`python
def transpose_with_zip(matrix: list[list]) -> list[list]:
    """zip orqali Pythonic transpozitsiya."""
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]

def rotate_90_clockwise(matrix: list[list]) -> list[list]:
    """Matritsani 90° soat yo'nalishida aylantirish."""
    transposed = [list(row) for row in zip(*matrix)]
    return [row[::-1] for row in transposed]
\`\`\`

**Amaliy foydalari:**
- Rasm qayta ishlash matritsa almashtirishlaridan foydalanadi
- Neyron tarmoqlar matritsa amallariga tayanadi`,
		},
	},
};

export default task;
