import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-array-transpose',
	title: 'Array Transpose',
	difficulty: 'easy',
	tags: ['numpy', 'transpose', 'matrix'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# Array Transpose

Transposing swaps rows and columns in 2D arrays, or reorders axes in higher dimensions.

## Task

Implement three functions:
1. \`transpose_2d(arr)\` - Transpose a 2D array
2. \`swap_axes(arr, axis1, axis2)\` - Swap two axes of an array
3. \`transpose_for_matmul(arr)\` - Transpose last two dimensions for matrix multiplication

## Example

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

transpose_2d(arr)  # [[1, 4], [2, 5], [3, 6]] shape (3, 2)

arr3d = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
swap_axes(arr3d, 0, 2)  # shape (4, 3, 2)
\`\`\``,

	initialCode: `import numpy as np

def transpose_2d(arr: np.ndarray) -> np.ndarray:
    """Transpose a 2D array."""
    # Your code here
    pass

def swap_axes(arr: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
    """Swap two axes of an array."""
    # Your code here
    pass

def transpose_for_matmul(arr: np.ndarray) -> np.ndarray:
    """Transpose last two dimensions for matrix multiplication."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def transpose_2d(arr: np.ndarray) -> np.ndarray:
    """Transpose a 2D array."""
    return arr.T

def swap_axes(arr: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
    """Swap two axes of an array."""
    return np.swapaxes(arr, axis1, axis2)

def transpose_for_matmul(arr: np.ndarray) -> np.ndarray:
    """Transpose last two dimensions for matrix multiplication."""
    return np.swapaxes(arr, -1, -2)
`,

	testCode: `import numpy as np
import unittest

class TestArrayTranspose(unittest.TestCase):
    def test_transpose_2d_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose_2d(arr)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_transpose_2d_shape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose_2d(arr)
        self.assertEqual(result.shape, (3, 2))

    def test_transpose_2d_square(self):
        arr = np.array([[1, 2], [3, 4]])
        result = transpose_2d(arr)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_swap_axes_3d(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = swap_axes(arr, 0, 2)
        self.assertEqual(result.shape, (4, 3, 2))

    def test_swap_axes_same(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = swap_axes(arr, 1, 1)
        np.testing.assert_array_equal(result, arr)

    def test_swap_axes_2d(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = swap_axes(arr, 0, 1)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_transpose_for_matmul_3d(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = transpose_for_matmul(arr)
        self.assertEqual(result.shape, (2, 4, 3))

    def test_transpose_for_matmul_4d(self):
        arr = np.arange(48).reshape(2, 2, 3, 4)
        result = transpose_for_matmul(arr)
        self.assertEqual(result.shape, (2, 2, 4, 3))

    def test_transpose_double(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose_2d(transpose_2d(arr))
        np.testing.assert_array_equal(result, arr)

    def test_transpose_preserves_data(self):
        arr = np.array([[1, 2], [3, 4]])
        result = transpose_2d(arr)
        self.assertEqual(result[0, 1], 3)
        self.assertEqual(result[1, 0], 2)
`,

	hint1: 'Use arr.T for simple 2D transpose',
	hint2: 'Use np.swapaxes(arr, axis1, axis2) to swap specific axes',

	whyItMatters: `Transpose operations are crucial for:

- **Matrix multiplication**: Ensure compatible shapes for dot products
- **Attention mechanisms**: Transpose queries/keys in transformers
- **Convolution layers**: Rearrange (N, C, H, W) to (N, H, W, C)
- **Batch operations**: Process multiple samples efficiently

Understanding axis manipulation is key to debugging shape errors in deep learning.`,

	translations: {
		ru: {
			title: 'Транспонирование массивов',
			description: `# Транспонирование массивов

Транспонирование меняет местами строки и столбцы в 2D массивах или переупорядочивает оси в многомерных массивах.

## Задача

Реализуйте три функции:
1. \`transpose_2d(arr)\` - Транспонировать 2D массив
2. \`swap_axes(arr, axis1, axis2)\` - Поменять местами две оси массива
3. \`transpose_for_matmul(arr)\` - Транспонировать последние два измерения для матричного умножения

## Пример

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

transpose_2d(arr)  # [[1, 4], [2, 5], [3, 6]] shape (3, 2)

arr3d = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
swap_axes(arr3d, 0, 2)  # shape (4, 3, 2)
\`\`\``,
			hint1: 'Используйте arr.T для простого 2D транспонирования',
			hint2: 'Используйте np.swapaxes(arr, axis1, axis2) для обмена конкретных осей',
			whyItMatters: `Операции транспонирования критичны для:

- **Матричное умножение**: Обеспечение совместимых форм для скалярных произведений
- **Механизмы внимания**: Транспонирование queries/keys в трансформерах
- **Свёрточные слои**: Перестановка (N, C, H, W) в (N, H, W, C)
- **Batch операции**: Эффективная обработка множества сэмплов`,
		},
		uz: {
			title: "Massivlarni transponlash",
			description: `# Massivlarni transponlash

Transponlash 2D massivlarda qatorlar va ustunlarni almashtiradi yoki ko'p o'lchovli massivlarda o'qlarni qayta tartiblaydi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`transpose_2d(arr)\` - 2D massivni transponlash
2. \`swap_axes(arr, axis1, axis2)\` - Massivning ikkita o'qini almashtirish
3. \`transpose_for_matmul(arr)\` - Matritsali ko'paytirish uchun oxirgi ikki o'lchamni transponlash

## Misol

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

transpose_2d(arr)  # [[1, 4], [2, 5], [3, 6]] shape (3, 2)

arr3d = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
swap_axes(arr3d, 0, 2)  # shape (4, 3, 2)
\`\`\``,
			hint1: "Oddiy 2D transponlash uchun arr.T dan foydalaning",
			hint2: "Ma'lum o'qlarni almashtirish uchun np.swapaxes(arr, axis1, axis2) dan foydalaning",
			whyItMatters: `Transponlash operatsiyalari quyidagilar uchun muhim:

- **Matritsali ko'paytirish**: Skalyar ko'paytmalar uchun mos shakllarni ta'minlash
- **Attention mexanizmlari**: Transformerlarda queries/keys ni transponlash
- **Konvolyutsion qatlamlar**: (N, C, H, W) ni (N, H, W, C) ga qayta tartiblash`,
		},
	},
};

export default task;
