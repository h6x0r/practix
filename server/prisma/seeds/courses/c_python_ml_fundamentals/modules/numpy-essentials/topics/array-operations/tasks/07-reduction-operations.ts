import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-reduction-operations',
	title: 'Reduction Operations',
	difficulty: 'medium',
	tags: ['numpy', 'reduction', 'aggregation'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,
	description: `# Reduction Operations

Reduction operations aggregate array values along specified axes, producing smaller arrays.

## Task

Implement four functions:
1. \`compute_stats(arr)\` - Return dict with mean, std, min, max
2. \`row_sums(arr)\` - Sum along each row
3. \`column_means(arr)\` - Mean of each column
4. \`argmax_per_row(arr)\` - Index of max value in each row

## Example

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

compute_stats(arr)  # {'mean': 3.5, 'std': 1.7, 'min': 1, 'max': 6}
row_sums(arr)       # [6, 15]
column_means(arr)   # [2.5, 3.5, 4.5]
argmax_per_row(arr) # [2, 2]
\`\`\``,

	initialCode: `import numpy as np

def compute_stats(arr: np.ndarray) -> dict:
    """Return dict with mean, std, min, max of entire array."""
    # Your code here
    pass

def row_sums(arr: np.ndarray) -> np.ndarray:
    """Sum along each row."""
    # Your code here
    pass

def column_means(arr: np.ndarray) -> np.ndarray:
    """Mean of each column."""
    # Your code here
    pass

def argmax_per_row(arr: np.ndarray) -> np.ndarray:
    """Index of max value in each row."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def compute_stats(arr: np.ndarray) -> dict:
    """Return dict with mean, std, min, max of entire array."""
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max())
    }

def row_sums(arr: np.ndarray) -> np.ndarray:
    """Sum along each row."""
    return arr.sum(axis=1)

def column_means(arr: np.ndarray) -> np.ndarray:
    """Mean of each column."""
    return arr.mean(axis=0)

def argmax_per_row(arr: np.ndarray) -> np.ndarray:
    """Index of max value in each row."""
    return arr.argmax(axis=1)
`,

	testCode: `import numpy as np
import unittest

class TestReductionOperations(unittest.TestCase):
    def test_compute_stats_basic(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = compute_stats(arr)
        self.assertAlmostEqual(result['mean'], 3.0)
        self.assertAlmostEqual(result['min'], 1.0)
        self.assertAlmostEqual(result['max'], 5.0)

    def test_compute_stats_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = compute_stats(arr)
        self.assertAlmostEqual(result['mean'], 2.5)
        self.assertAlmostEqual(result['min'], 1.0)
        self.assertAlmostEqual(result['max'], 4.0)

    def test_compute_stats_keys(self):
        arr = np.array([1, 2, 3])
        result = compute_stats(arr)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('min', result)
        self.assertIn('max', result)

    def test_row_sums_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = row_sums(arr)
        np.testing.assert_array_equal(result, [6, 15])

    def test_row_sums_shape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = row_sums(arr)
        self.assertEqual(result.shape, (3,))

    def test_row_sums_single_row(self):
        arr = np.array([[1, 2, 3, 4]])
        result = row_sums(arr)
        np.testing.assert_array_equal(result, [10])

    def test_column_means_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = column_means(arr)
        np.testing.assert_array_almost_equal(result, [2.5, 3.5, 4.5])

    def test_column_means_shape(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = column_means(arr)
        self.assertEqual(result.shape, (2,))

    def test_argmax_basic(self):
        arr = np.array([[1, 3, 2], [6, 4, 5]])
        result = argmax_per_row(arr)
        np.testing.assert_array_equal(result, [1, 0])

    def test_argmax_all_same(self):
        arr = np.array([[5, 5, 5], [3, 3, 3]])
        result = argmax_per_row(arr)
        self.assertEqual(result.shape, (2,))

    def test_argmax_last_column(self):
        arr = np.array([[1, 2, 10], [1, 2, 10]])
        result = argmax_per_row(arr)
        np.testing.assert_array_equal(result, [2, 2])

    def test_compute_stats_std(self):
        arr = np.array([1, 1, 1, 1])
        result = compute_stats(arr)
        self.assertAlmostEqual(result['std'], 0.0)
`,

	hint1: 'Use axis=1 for row operations, axis=0 for column operations',
	hint2: 'Use argmax() to find index of maximum value',

	whyItMatters: `Reduction operations are fundamental to ML:

- **Statistics**: Compute mean/std for normalization
- **Loss aggregation**: Sum losses across batch
- **Accuracy**: Compare argmax predictions to labels
- **Attention weights**: Sum to 1 along sequence dimension

Efficient reductions are critical for processing large datasets.`,

	translations: {
		ru: {
			title: 'Операции редукции',
			description: `# Операции редукции

Операции редукции агрегируют значения массива вдоль указанных осей, создавая меньшие массивы.

## Задача

Реализуйте четыре функции:
1. \`compute_stats(arr)\` - Вернуть dict с mean, std, min, max
2. \`row_sums(arr)\` - Сумма по каждой строке
3. \`column_means(arr)\` - Среднее каждого столбца
4. \`argmax_per_row(arr)\` - Индекс максимального значения в каждой строке

## Пример

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

compute_stats(arr)  # {'mean': 3.5, 'std': 1.7, 'min': 1, 'max': 6}
row_sums(arr)       # [6, 15]
column_means(arr)   # [2.5, 3.5, 4.5]
argmax_per_row(arr) # [2, 2]
\`\`\``,
			hint1: 'Используйте axis=1 для операций по строкам, axis=0 для столбцов',
			hint2: 'Используйте argmax() для поиска индекса максимального значения',
			whyItMatters: `Операции редукции фундаментальны для ML:

- **Статистика**: Вычисление mean/std для нормализации
- **Агрегация loss**: Суммирование потерь по батчу
- **Accuracy**: Сравнение argmax предсказаний с метками
- **Веса внимания**: Сумма = 1 вдоль измерения последовательности`,
		},
		uz: {
			title: "Reduktion operatsiyalari",
			description: `# Reduktion operatsiyalari

Reduktion operatsiyalari massiv qiymatlarini ko'rsatilgan o'qlar bo'ylab jamlaydi, kichikroq massivlar yaratadi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`compute_stats(arr)\` - mean, std, min, max bilan dict qaytarish
2. \`row_sums(arr)\` - Har bir qator bo'ylab yig'indi
3. \`column_means(arr)\` - Har bir ustun o'rtachasi
4. \`argmax_per_row(arr)\` - Har bir qatorda maksimal qiymat indeksi

## Misol

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

compute_stats(arr)  # {'mean': 3.5, 'std': 1.7, 'min': 1, 'max': 6}
row_sums(arr)       # [6, 15]
column_means(arr)   # [2.5, 3.5, 4.5]
argmax_per_row(arr) # [2, 2]
\`\`\``,
			hint1: "Qator operatsiyalari uchun axis=1, ustun operatsiyalari uchun axis=0 dan foydalaning",
			hint2: "Maksimal qiymat indeksini topish uchun argmax() dan foydalaning",
			whyItMatters: `Reduktion operatsiyalari ML uchun asosiydir:

- **Statistika**: Normalizatsiya uchun mean/std hisoblash
- **Loss jamlash**: Batch bo'ylab losslarni yig'ish
- **Accuracy**: Argmax bashoratlarini labellar bilan taqqoslash`,
		},
	},
};

export default task;
