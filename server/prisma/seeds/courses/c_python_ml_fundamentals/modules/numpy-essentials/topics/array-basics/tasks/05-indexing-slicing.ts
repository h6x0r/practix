import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-indexing-slicing',
	title: 'Array Indexing and Slicing',
	difficulty: 'medium',
	tags: ['numpy', 'indexing', 'slicing'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Array Indexing and Slicing

NumPy supports powerful indexing and slicing operations similar to Python lists but extended to multiple dimensions.

## Task

Implement four functions:
1. \`get_element(arr, row, col)\` - Get element at position [row, col]
2. \`get_row(arr, row)\` - Get entire row
3. \`get_column(arr, col)\` - Get entire column
4. \`get_subarray(arr, r1, r2, c1, c2)\` - Get subarray from [r1:r2, c1:c2]

## Example

\`\`\`python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

get_element(arr, 1, 2)  # 6
get_row(arr, 0)         # [1, 2, 3]
get_column(arr, 1)      # [2, 5, 8]
get_subarray(arr, 0, 2, 1, 3)  # [[2, 3], [5, 6]]
\`\`\``,

	initialCode: `import numpy as np

def get_element(arr: np.ndarray, row: int, col: int):
    """Get element at position [row, col]."""
    # Your code here
    pass

def get_row(arr: np.ndarray, row: int) -> np.ndarray:
    """Get entire row."""
    # Your code here
    pass

def get_column(arr: np.ndarray, col: int) -> np.ndarray:
    """Get entire column."""
    # Your code here
    pass

def get_subarray(arr: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> np.ndarray:
    """Get subarray from [r1:r2, c1:c2]."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def get_element(arr: np.ndarray, row: int, col: int):
    """Get element at position [row, col]."""
    return arr[row, col]

def get_row(arr: np.ndarray, row: int) -> np.ndarray:
    """Get entire row."""
    return arr[row, :]

def get_column(arr: np.ndarray, col: int) -> np.ndarray:
    """Get entire column."""
    return arr[:, col]

def get_subarray(arr: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> np.ndarray:
    """Get subarray from [r1:r2, c1:c2]."""
    return arr[r1:r2, c1:c2]
`,

	testCode: `import numpy as np
import unittest

class TestIndexingSlicing(unittest.TestCase):
    def setUp(self):
        self.arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_get_element_center(self):
        result = get_element(self.arr, 1, 1)
        self.assertEqual(result, 5)

    def test_get_element_corner(self):
        result = get_element(self.arr, 0, 0)
        self.assertEqual(result, 1)

    def test_get_element_last(self):
        result = get_element(self.arr, 2, 2)
        self.assertEqual(result, 9)

    def test_get_row_first(self):
        result = get_row(self.arr, 0)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_get_row_last(self):
        result = get_row(self.arr, 2)
        np.testing.assert_array_equal(result, [7, 8, 9])

    def test_get_column_first(self):
        result = get_column(self.arr, 0)
        np.testing.assert_array_equal(result, [1, 4, 7])

    def test_get_column_last(self):
        result = get_column(self.arr, 2)
        np.testing.assert_array_equal(result, [3, 6, 9])

    def test_get_subarray_top_left(self):
        result = get_subarray(self.arr, 0, 2, 0, 2)
        expected = np.array([[1, 2], [4, 5]])
        np.testing.assert_array_equal(result, expected)

    def test_get_subarray_bottom_right(self):
        result = get_subarray(self.arr, 1, 3, 1, 3)
        expected = np.array([[5, 6], [8, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_get_subarray_single_row(self):
        result = get_subarray(self.arr, 1, 2, 0, 3)
        expected = np.array([[4, 5, 6]])
        np.testing.assert_array_equal(result, expected)
`,

	hint1: 'Use arr[row, col] for 2D indexing',
	hint2: 'Use : for all elements in a dimension, e.g., arr[:, col]',

	whyItMatters: `Efficient data access is fundamental to ML performance:

- **Feature extraction**: Select specific columns (features) from datasets
- **Batch processing**: Extract rows (samples) for mini-batch training
- **Image processing**: Crop regions of interest from images
- **Time series**: Select windows of data for sequence models

Understanding NumPy indexing prevents unnecessary data copies and memory overhead.`,

	translations: {
		ru: {
			title: 'Индексация и срезы массивов',
			description: `# Индексация и срезы массивов

NumPy поддерживает мощные операции индексации и срезов, аналогичные спискам Python, но расширенные для многомерных массивов.

## Задача

Реализуйте четыре функции:
1. \`get_element(arr, row, col)\` - Получить элемент на позиции [row, col]
2. \`get_row(arr, row)\` - Получить всю строку
3. \`get_column(arr, col)\` - Получить весь столбец
4. \`get_subarray(arr, r1, r2, c1, c2)\` - Получить подмассив [r1:r2, c1:c2]

## Пример

\`\`\`python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

get_element(arr, 1, 2)  # 6
get_row(arr, 0)         # [1, 2, 3]
get_column(arr, 1)      # [2, 5, 8]
get_subarray(arr, 0, 2, 1, 3)  # [[2, 3], [5, 6]]
\`\`\``,
			hint1: 'Используйте arr[row, col] для 2D индексации',
			hint2: 'Используйте : для всех элементов измерения, например arr[:, col]',
			whyItMatters: `Эффективный доступ к данным — основа производительности ML:

- **Извлечение признаков**: Выбор конкретных столбцов (признаков) из датасетов
- **Batch обработка**: Извлечение строк (сэмплов) для mini-batch обучения
- **Обработка изображений**: Обрезка областей интереса
- **Временные ряды**: Выбор окон данных для sequence моделей`,
		},
		uz: {
			title: "Massiv indekslash va kesish",
			description: `# Massiv indekslash va kesish

NumPy Python ro'yxatlariga o'xshash, lekin ko'p o'lchovli massivlar uchun kengaytirilgan kuchli indekslash va kesish amallarini qo'llab-quvvatlaydi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`get_element(arr, row, col)\` - [row, col] pozitsiyasidagi elementni olish
2. \`get_row(arr, row)\` - Butun qatorni olish
3. \`get_column(arr, col)\` - Butun ustunni olish
4. \`get_subarray(arr, r1, r2, c1, c2)\` - [r1:r2, c1:c2] dan pastki massivni olish

## Misol

\`\`\`python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

get_element(arr, 1, 2)  # 6
get_row(arr, 0)         # [1, 2, 3]
get_column(arr, 1)      # [2, 5, 8]
get_subarray(arr, 0, 2, 1, 3)  # [[2, 3], [5, 6]]
\`\`\``,
			hint1: "2D indekslash uchun arr[row, col] dan foydalaning",
			hint2: "O'lchamdagi barcha elementlar uchun : dan foydalaning, masalan arr[:, col]",
			whyItMatters: `Samarali ma'lumotlarga kirish ML ishlashi uchun asosiydir:

- **Xususiyatlarni ajratib olish**: Datasetlardan ma'lum ustunlarni (xususiyatlarni) tanlash
- **Batch ishlov berish**: Mini-batch o'qitish uchun qatorlarni (namunalarni) ajratib olish
- **Tasvir ishlov berish**: Rasmlardan qiziqish sohalarini kesish`,
		},
	},
};

export default task;
