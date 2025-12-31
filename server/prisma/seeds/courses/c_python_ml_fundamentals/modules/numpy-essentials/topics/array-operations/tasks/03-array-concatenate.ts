import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-array-concatenate',
	title: 'Array Concatenation',
	difficulty: 'easy',
	tags: ['numpy', 'concatenate', 'stack'],
	estimatedTime: '12m',
	isPremium: false,
	order: 3,
	description: `# Array Concatenation

Combining arrays is essential for building datasets and merging results.

## Task

Implement three functions:
1. \`concat_horizontal(arr1, arr2)\` - Concatenate arrays horizontally (along columns)
2. \`concat_vertical(arr1, arr2)\` - Concatenate arrays vertically (along rows)
3. \`stack_arrays(arrays)\` - Stack list of arrays along new axis

## Example

\`\`\`python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_horizontal(a, b)  # [[1, 2, 5, 6], [3, 4, 7, 8]]
concat_vertical(a, b)    # [[1, 2], [3, 4], [5, 6], [7, 8]]

arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
stack_arrays(arrays)  # [[1, 2], [3, 4], [5, 6]]
\`\`\``,

	initialCode: `import numpy as np

def concat_horizontal(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Concatenate arrays horizontally (along columns)."""
    # Your code here
    pass

def concat_vertical(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Concatenate arrays vertically (along rows)."""
    # Your code here
    pass

def stack_arrays(arrays: list) -> np.ndarray:
    """Stack list of arrays along new axis."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def concat_horizontal(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Concatenate arrays horizontally (along columns)."""
    return np.concatenate([arr1, arr2], axis=1)

def concat_vertical(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Concatenate arrays vertically (along rows)."""
    return np.concatenate([arr1, arr2], axis=0)

def stack_arrays(arrays: list) -> np.ndarray:
    """Stack list of arrays along new axis."""
    return np.stack(arrays)
`,

	testCode: `import numpy as np
import unittest

class TestArrayConcatenate(unittest.TestCase):
    def test_concat_horizontal_basic(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = concat_horizontal(a, b)
        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_concat_horizontal_shape(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5], [6]])
        result = concat_horizontal(a, b)
        self.assertEqual(result.shape, (2, 3))

    def test_concat_vertical_basic(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = concat_vertical(a, b)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_concat_vertical_shape(self):
        a = np.array([[1, 2, 3]])
        b = np.array([[4, 5, 6], [7, 8, 9]])
        result = concat_vertical(a, b)
        self.assertEqual(result.shape, (3, 3))

    def test_stack_arrays_basic(self):
        arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = stack_arrays(arrays)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_stack_arrays_shape(self):
        arrays = [np.array([1, 2, 3]) for _ in range(4)]
        result = stack_arrays(arrays)
        self.assertEqual(result.shape, (4, 3))

    def test_stack_2d_arrays(self):
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        result = stack_arrays(arrays)
        self.assertEqual(result.shape, (2, 2, 2))

    def test_concat_preserves_order(self):
        a = np.array([[1, 2]])
        b = np.array([[3, 4]])
        result = concat_vertical(a, b)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[1, 0], 3)

    def test_stack_single_array(self):
        arrays = [np.array([1, 2, 3])]
        result = stack_arrays(arrays)
        self.assertEqual(result.shape, (1, 3))

    def test_concat_1d_vertical(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.concatenate([a, b])
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])
`,

	hint1: 'Use np.concatenate([arr1, arr2], axis=0) for vertical, axis=1 for horizontal',
	hint2: 'Use np.stack(arrays) to create a new axis when stacking',

	whyItMatters: `Concatenation is fundamental for:

- **Dataset building**: Combine train/validation/test splits
- **Feature engineering**: Merge different feature sets
- **Batch creation**: Stack individual samples into batches
- **Data augmentation**: Combine original and augmented data

Efficient concatenation avoids memory issues with large datasets.`,

	translations: {
		ru: {
			title: 'Конкатенация массивов',
			description: `# Конкатенация массивов

Объединение массивов необходимо для построения датасетов и слияния результатов.

## Задача

Реализуйте три функции:
1. \`concat_horizontal(arr1, arr2)\` - Конкатенация массивов горизонтально (по столбцам)
2. \`concat_vertical(arr1, arr2)\` - Конкатенация массивов вертикально (по строкам)
3. \`stack_arrays(arrays)\` - Сложить список массивов вдоль новой оси

## Пример

\`\`\`python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_horizontal(a, b)  # [[1, 2, 5, 6], [3, 4, 7, 8]]
concat_vertical(a, b)    # [[1, 2], [3, 4], [5, 6], [7, 8]]

arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
stack_arrays(arrays)  # [[1, 2], [3, 4], [5, 6]]
\`\`\``,
			hint1: 'Используйте np.concatenate([arr1, arr2], axis=0) для вертикальной, axis=1 для горизонтальной',
			hint2: 'Используйте np.stack(arrays) для создания новой оси при складывании',
			whyItMatters: `Конкатенация фундаментальна для:

- **Построение датасетов**: Объединение train/validation/test разбиений
- **Feature engineering**: Слияние разных наборов признаков
- **Создание батчей**: Складывание отдельных сэмплов в батчи
- **Аугментация данных**: Объединение оригинальных и аугментированных данных`,
		},
		uz: {
			title: "Massivlarni birlashtirish",
			description: `# Massivlarni birlashtirish

Massivlarni birlashtirish datasetlarni yaratish va natijalarni qo'shish uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`concat_horizontal(arr1, arr2)\` - Massivlarni gorizontal birlashtirish (ustunlar bo'ylab)
2. \`concat_vertical(arr1, arr2)\` - Massivlarni vertikal birlashtirish (qatorlar bo'ylab)
3. \`stack_arrays(arrays)\` - Massivlar ro'yxatini yangi o'q bo'ylab yig'ish

## Misol

\`\`\`python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_horizontal(a, b)  # [[1, 2, 5, 6], [3, 4, 7, 8]]
concat_vertical(a, b)    # [[1, 2], [3, 4], [5, 6], [7, 8]]

arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
stack_arrays(arrays)  # [[1, 2], [3, 4], [5, 6]]
\`\`\``,
			hint1: "Vertikal uchun np.concatenate([arr1, arr2], axis=0), gorizontal uchun axis=1 dan foydalaning",
			hint2: "Yig'ishda yangi o'q yaratish uchun np.stack(arrays) dan foydalaning",
			whyItMatters: `Birlashtirish quyidagilar uchun asosiydir:

- **Dataset yaratish**: Train/validation/test bo'linmalarini birlashtirish
- **Feature engineering**: Turli xususiyat to'plamlarini birlashtirish
- **Batch yaratish**: Alohida namunalarni batchlarga yig'ish`,
		},
	},
};

export default task;
