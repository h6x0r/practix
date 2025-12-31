import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-array-split',
	title: 'Array Splitting',
	difficulty: 'easy',
	tags: ['numpy', 'split', 'partition'],
	estimatedTime: '10m',
	isPremium: false,
	order: 4,
	description: `# Array Splitting

Splitting arrays is the inverse of concatenation. It's used for creating data splits and partitions.

## Task

Implement three functions:
1. \`split_equal(arr, n)\` - Split array into n equal parts along axis 0
2. \`split_at_indices(arr, indices)\` - Split array at specified indices
3. \`split_train_test(arr, train_ratio)\` - Split into train/test sets

## Example

\`\`\`python
arr = np.arange(12).reshape(4, 3)

split_equal(arr, 2)  # [array([[0,1,2],[3,4,5]]), array([[6,7,8],[9,10,11]])]
split_at_indices(arr, [1, 3])  # Split at rows 1 and 3

data = np.arange(100)
train, test = split_train_test(data, 0.8)  # 80 train, 20 test
\`\`\``,

	initialCode: `import numpy as np

def split_equal(arr: np.ndarray, n: int) -> list:
    """Split array into n equal parts along axis 0."""
    # Your code here
    pass

def split_at_indices(arr: np.ndarray, indices: list) -> list:
    """Split array at specified indices along axis 0."""
    # Your code here
    pass

def split_train_test(arr: np.ndarray, train_ratio: float) -> tuple:
    """Split into train/test sets based on ratio."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def split_equal(arr: np.ndarray, n: int) -> list:
    """Split array into n equal parts along axis 0."""
    return np.array_split(arr, n)

def split_at_indices(arr: np.ndarray, indices: list) -> list:
    """Split array at specified indices along axis 0."""
    return np.split(arr, indices)

def split_train_test(arr: np.ndarray, train_ratio: float) -> tuple:
    """Split into train/test sets based on ratio."""
    split_idx = int(len(arr) * train_ratio)
    return arr[:split_idx], arr[split_idx:]
`,

	testCode: `import numpy as np
import unittest

class TestArraySplit(unittest.TestCase):
    def test_split_equal_basic(self):
        arr = np.arange(12).reshape(4, 3)
        result = split_equal(arr, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (2, 3))
        self.assertEqual(result[1].shape, (2, 3))

    def test_split_equal_uneven(self):
        arr = np.arange(10)
        result = split_equal(arr, 3)
        self.assertEqual(len(result), 3)
        total = sum(len(r) for r in result)
        self.assertEqual(total, 10)

    def test_split_equal_into_one(self):
        arr = np.array([1, 2, 3, 4])
        result = split_equal(arr, 1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], arr)

    def test_split_at_indices_basic(self):
        arr = np.arange(10)
        result = split_at_indices(arr, [3, 7])
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], [0, 1, 2])
        np.testing.assert_array_equal(result[1], [3, 4, 5, 6])
        np.testing.assert_array_equal(result[2], [7, 8, 9])

    def test_split_at_indices_single(self):
        arr = np.arange(6)
        result = split_at_indices(arr, [3])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], [0, 1, 2])
        np.testing.assert_array_equal(result[1], [3, 4, 5])

    def test_split_train_test_ratio(self):
        arr = np.arange(100)
        train, test = split_train_test(arr, 0.8)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_split_train_test_half(self):
        arr = np.arange(50)
        train, test = split_train_test(arr, 0.5)
        self.assertEqual(len(train), 25)
        self.assertEqual(len(test), 25)

    def test_split_train_test_preserves_order(self):
        arr = np.arange(10)
        train, test = split_train_test(arr, 0.6)
        np.testing.assert_array_equal(train, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(test, [6, 7, 8, 9])

    def test_split_equal_4_parts(self):
        arr = np.arange(20)
        result = split_equal(arr, 4)
        self.assertEqual(len(result), 4)
        for r in result:
            self.assertEqual(len(r), 5)

    def test_split_preserves_dtype(self):
        arr = np.array([1.5, 2.5, 3.5, 4.5])
        result = split_equal(arr, 2)
        self.assertEqual(result[0].dtype, arr.dtype)
`,

	hint1: 'Use np.array_split for equal parts (handles uneven splits)',
	hint2: 'For train/test: calculate split index as int(len(arr) * ratio)',

	whyItMatters: `Array splitting is essential for:

- **Train/Val/Test splits**: Divide data for model evaluation
- **Cross-validation**: Create k folds for robust evaluation
- **Batch processing**: Split large datasets for memory efficiency
- **Distributed computing**: Partition data across workers

Proper splitting ensures unbiased model evaluation.`,

	translations: {
		ru: {
			title: 'Разделение массивов',
			description: `# Разделение массивов

Разделение массивов — обратная операция конкатенации. Используется для создания разбиений данных.

## Задача

Реализуйте три функции:
1. \`split_equal(arr, n)\` - Разделить массив на n равных частей по оси 0
2. \`split_at_indices(arr, indices)\` - Разделить массив по указанным индексам
3. \`split_train_test(arr, train_ratio)\` - Разделить на train/test наборы

## Пример

\`\`\`python
arr = np.arange(12).reshape(4, 3)

split_equal(arr, 2)  # [array([[0,1,2],[3,4,5]]), array([[6,7,8],[9,10,11]])]
split_at_indices(arr, [1, 3])  # Разделение по строкам 1 и 3

data = np.arange(100)
train, test = split_train_test(data, 0.8)  # 80 train, 20 test
\`\`\``,
			hint1: 'Используйте np.array_split для равных частей (обрабатывает неравные разбиения)',
			hint2: 'Для train/test: вычислите индекс разделения как int(len(arr) * ratio)',
			whyItMatters: `Разделение массивов необходимо для:

- **Train/Val/Test разбиения**: Разделение данных для оценки модели
- **Кросс-валидация**: Создание k фолдов для надёжной оценки
- **Batch обработка**: Разделение больших датасетов для эффективности памяти
- **Распределённые вычисления**: Партиционирование данных между воркерами`,
		},
		uz: {
			title: "Massivlarni bo'lish",
			description: `# Massivlarni bo'lish

Massivlarni bo'lish birlashtirishning teskari operatsiyasi. Ma'lumotlar bo'linmalarini yaratish uchun ishlatiladi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`split_equal(arr, n)\` - Massivni 0 o'q bo'ylab n teng qismga bo'lish
2. \`split_at_indices(arr, indices)\` - Massivni ko'rsatilgan indekslarda bo'lish
3. \`split_train_test(arr, train_ratio)\` - Train/test to'plamlariga bo'lish

## Misol

\`\`\`python
arr = np.arange(12).reshape(4, 3)

split_equal(arr, 2)  # [array([[0,1,2],[3,4,5]]), array([[6,7,8],[9,10,11]])]
split_at_indices(arr, [1, 3])  # 1 va 3-qatorlarda bo'lish

data = np.arange(100)
train, test = split_train_test(data, 0.8)  # 80 train, 20 test
\`\`\``,
			hint1: "Teng qismlar uchun np.array_split dan foydalaning (teng bo'lmagan bo'linishlarni boshqaradi)",
			hint2: "Train/test uchun: bo'linish indeksini int(len(arr) * ratio) sifatida hisoblang",
			whyItMatters: `Massivlarni bo'lish quyidagilar uchun zarur:

- **Train/Val/Test bo'linmalari**: Modelni baholash uchun ma'lumotlarni bo'lish
- **Cross-validation**: Ishonchli baholash uchun k fold yaratish
- **Batch ishlov berish**: Xotira samaradorligi uchun katta datasetlarni bo'lish`,
		},
	},
};

export default task;
