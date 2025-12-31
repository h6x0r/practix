import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-fancy-indexing',
	title: 'Fancy Indexing',
	difficulty: 'medium',
	tags: ['numpy', 'indexing', 'advanced'],
	estimatedTime: '20m',
	isPremium: false,
	order: 7,
	description: `# Fancy Indexing

Fancy indexing uses arrays of indices to access multiple elements at once. This is powerful for reordering and selecting specific elements.

## Task

Implement three functions:
1. \`select_by_indices(arr, indices)\` - Select elements at given indices
2. \`reorder_rows(arr, order)\` - Reorder rows of 2D array
3. \`select_diagonal(arr)\` - Select diagonal elements using fancy indexing

## Example

\`\`\`python
arr = np.array([10, 20, 30, 40, 50])
select_by_indices(arr, [0, 2, 4])  # [10, 30, 50]

matrix = np.array([[1, 2], [3, 4], [5, 6]])
reorder_rows(matrix, [2, 0, 1])  # [[5, 6], [1, 2], [3, 4]]

square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
select_diagonal(square)  # [1, 5, 9]
\`\`\``,

	initialCode: `import numpy as np

def select_by_indices(arr: np.ndarray, indices: list) -> np.ndarray:
    """Select elements at given indices."""
    # Your code here
    pass

def reorder_rows(arr: np.ndarray, order: list) -> np.ndarray:
    """Reorder rows of 2D array according to order list."""
    # Your code here
    pass

def select_diagonal(arr: np.ndarray) -> np.ndarray:
    """Select diagonal elements using fancy indexing."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def select_by_indices(arr: np.ndarray, indices: list) -> np.ndarray:
    """Select elements at given indices."""
    return arr[indices]

def reorder_rows(arr: np.ndarray, order: list) -> np.ndarray:
    """Reorder rows of 2D array according to order list."""
    return arr[order]

def select_diagonal(arr: np.ndarray) -> np.ndarray:
    """Select diagonal elements using fancy indexing."""
    n = min(arr.shape[0], arr.shape[1])
    indices = np.arange(n)
    return arr[indices, indices]
`,

	testCode: `import numpy as np
import unittest

class TestFancyIndexing(unittest.TestCase):
    def test_select_indices_basic(self):
        arr = np.array([10, 20, 30, 40, 50])
        result = select_by_indices(arr, [0, 2, 4])
        np.testing.assert_array_equal(result, [10, 30, 50])

    def test_select_indices_reverse(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = select_by_indices(arr, [4, 3, 2, 1, 0])
        np.testing.assert_array_equal(result, [5, 4, 3, 2, 1])

    def test_select_indices_repeated(self):
        arr = np.array([10, 20, 30])
        result = select_by_indices(arr, [0, 0, 2, 2])
        np.testing.assert_array_equal(result, [10, 10, 30, 30])

    def test_reorder_rows_basic(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        result = reorder_rows(matrix, [2, 0, 1])
        expected = np.array([[5, 6], [1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_reorder_rows_reverse(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        result = reorder_rows(matrix, [2, 1, 0])
        expected = np.array([[5, 6], [3, 4], [1, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_diagonal_3x3(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = select_diagonal(arr)
        np.testing.assert_array_equal(result, [1, 5, 9])

    def test_diagonal_2x2(self):
        arr = np.array([[10, 20], [30, 40]])
        result = select_diagonal(arr)
        np.testing.assert_array_equal(result, [10, 40])

    def test_diagonal_4x4(self):
        arr = np.arange(16).reshape(4, 4)
        result = select_diagonal(arr)
        np.testing.assert_array_equal(result, [0, 5, 10, 15])

    def test_select_single_index(self):
        arr = np.array([100, 200, 300])
        result = select_by_indices(arr, [1])
        np.testing.assert_array_equal(result, [200])

    def test_reorder_same_order(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = reorder_rows(matrix, [0, 1])
        np.testing.assert_array_equal(result, matrix)
`,

	hint1: 'Use arr[list_of_indices] to select multiple elements',
	hint2: 'For diagonal: arr[np.arange(n), np.arange(n)]',

	whyItMatters: `Fancy indexing is essential for:

- **Batch selection**: Select specific samples from a dataset by their indices
- **Data shuffling**: Randomly reorder data for training
- **Embedding lookup**: Index into embedding matrices (word2vec, etc.)
- **Sparse operations**: Efficiently work with sparse data

This technique is used extensively in deep learning frameworks for batch processing.`,

	translations: {
		ru: {
			title: 'Расширенная индексация',
			description: `# Расширенная индексация

Расширенная индексация использует массивы индексов для доступа к нескольким элементам сразу. Это мощный инструмент для переупорядочивания и выбора конкретных элементов.

## Задача

Реализуйте три функции:
1. \`select_by_indices(arr, indices)\` - Выбрать элементы по заданным индексам
2. \`reorder_rows(arr, order)\` - Переупорядочить строки 2D массива
3. \`select_diagonal(arr)\` - Выбрать диагональные элементы

## Пример

\`\`\`python
arr = np.array([10, 20, 30, 40, 50])
select_by_indices(arr, [0, 2, 4])  # [10, 30, 50]

matrix = np.array([[1, 2], [3, 4], [5, 6]])
reorder_rows(matrix, [2, 0, 1])  # [[5, 6], [1, 2], [3, 4]]

square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
select_diagonal(square)  # [1, 5, 9]
\`\`\``,
			hint1: 'Используйте arr[список_индексов] для выбора нескольких элементов',
			hint2: 'Для диагонали: arr[np.arange(n), np.arange(n)]',
			whyItMatters: `Расширенная индексация необходима для:

- **Выбор батча**: Выбор конкретных сэмплов из датасета по их индексам
- **Перемешивание данных**: Случайное переупорядочивание данных для обучения
- **Поиск эмбеддингов**: Индексация матриц эмбеддингов (word2vec и т.д.)
- **Разреженные операции**: Эффективная работа с разреженными данными`,
		},
		uz: {
			title: "Murakkab indekslash",
			description: `# Murakkab indekslash

Murakkab indekslash bir vaqtning o'zida bir nechta elementlarga kirish uchun indekslar massivlaridan foydalanadi. Bu qayta tartiblash va ma'lum elementlarni tanlash uchun kuchli.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`select_by_indices(arr, indices)\` - Berilgan indekslardagi elementlarni tanlash
2. \`reorder_rows(arr, order)\` - 2D massiv qatorlarini qayta tartiblash
3. \`select_diagonal(arr)\` - Diagonal elementlarni tanlash

## Misol

\`\`\`python
arr = np.array([10, 20, 30, 40, 50])
select_by_indices(arr, [0, 2, 4])  # [10, 30, 50]

matrix = np.array([[1, 2], [3, 4], [5, 6]])
reorder_rows(matrix, [2, 0, 1])  # [[5, 6], [1, 2], [3, 4]]

square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
select_diagonal(square)  # [1, 5, 9]
\`\`\``,
			hint1: "Bir nechta elementni tanlash uchun arr[indekslar_ro'yxati] dan foydalaning",
			hint2: "Diagonal uchun: arr[np.arange(n), np.arange(n)]",
			whyItMatters: `Murakkab indekslash quyidagilar uchun zarur:

- **Batch tanlash**: Datasetdan indekslar bo'yicha ma'lum namunalarni tanlash
- **Ma'lumotlarni aralashtirish**: O'qitish uchun ma'lumotlarni tasodifiy qayta tartiblash
- **Embedding qidirish**: Embedding matritsalarini indekslash (word2vec va boshqalar)`,
		},
	},
};

export default task;
