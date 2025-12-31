import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-broadcasting-basics',
	title: 'Broadcasting Basics',
	difficulty: 'medium',
	tags: ['numpy', 'broadcasting', 'operations'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Broadcasting Basics

Broadcasting allows NumPy to perform operations on arrays with different shapes automatically.

## Task

Implement three functions:
1. \`add_scalar(arr, scalar)\` - Add scalar to all elements
2. \`normalize_rows(arr)\` - Subtract row means from each row
3. \`scale_columns(arr, scales)\` - Multiply each column by corresponding scale

## Example

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

add_scalar(arr, 10)  # [[11, 12, 13], [14, 15, 16]]

# Row means: [2, 5]
normalize_rows(arr)  # [[-1, 0, 1], [-1, 0, 1]]

scales = np.array([1, 2, 3])
scale_columns(arr, scales)  # [[1, 4, 9], [4, 10, 18]]
\`\`\``,

	initialCode: `import numpy as np

def add_scalar(arr: np.ndarray, scalar: float) -> np.ndarray:
    """Add scalar to all elements."""
    # Your code here
    pass

def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Subtract row means from each row."""
    # Your code here
    pass

def scale_columns(arr: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Multiply each column by corresponding scale."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def add_scalar(arr: np.ndarray, scalar: float) -> np.ndarray:
    """Add scalar to all elements."""
    return arr + scalar

def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Subtract row means from each row."""
    row_means = arr.mean(axis=1, keepdims=True)
    return arr - row_means

def scale_columns(arr: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Multiply each column by corresponding scale."""
    return arr * scales
`,

	testCode: `import numpy as np
import unittest

class TestBroadcastingBasics(unittest.TestCase):
    def test_add_scalar_basic(self):
        arr = np.array([[1, 2], [3, 4]])
        result = add_scalar(arr, 10)
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result, expected)

    def test_add_scalar_negative(self):
        arr = np.array([1, 2, 3])
        result = add_scalar(arr, -1)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_add_scalar_float(self):
        arr = np.array([1, 2, 3])
        result = add_scalar(arr, 0.5)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])

    def test_normalize_rows_basic(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_rows(arr)
        expected = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_rows_zero_mean(self):
        arr = np.array([[1.0, 2.0, 3.0]])
        result = normalize_rows(arr)
        self.assertAlmostEqual(result.mean(), 0.0)

    def test_normalize_rows_single_row(self):
        arr = np.array([[10.0, 20.0, 30.0]])
        result = normalize_rows(arr)
        expected = np.array([[-10.0, 0.0, 10.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_columns_basic(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        scales = np.array([1, 2, 3])
        result = scale_columns(arr, scales)
        expected = np.array([[1, 4, 9], [4, 10, 18]])
        np.testing.assert_array_equal(result, expected)

    def test_scale_columns_ones(self):
        arr = np.array([[1, 2], [3, 4]])
        scales = np.array([1, 1])
        result = scale_columns(arr, scales)
        np.testing.assert_array_equal(result, arr)

    def test_scale_columns_zeros(self):
        arr = np.array([[1, 2], [3, 4]])
        scales = np.array([0, 0])
        result = scale_columns(arr, scales)
        np.testing.assert_array_equal(result, np.zeros_like(arr))

    def test_original_unchanged(self):
        arr = np.array([[1, 2], [3, 4]])
        original = arr.copy()
        _ = add_scalar(arr, 10)
        np.testing.assert_array_equal(arr, original)
`,

	hint1: 'Broadcasting works automatically when shapes are compatible',
	hint2: 'Use keepdims=True with mean() to preserve shape for broadcasting',

	whyItMatters: `Broadcasting is the heart of vectorized operations:

- **Normalization**: Subtract mean/divide by std efficiently
- **Feature scaling**: Apply different scales to features
- **Bias addition**: Add bias vectors to batch outputs
- **Attention scores**: Multiply queries and keys efficiently

Understanding broadcasting eliminates slow Python loops.`,

	translations: {
		ru: {
			title: 'Основы Broadcasting',
			description: `# Основы Broadcasting

Broadcasting позволяет NumPy автоматически выполнять операции над массивами с разными формами.

## Задача

Реализуйте три функции:
1. \`add_scalar(arr, scalar)\` - Добавить скаляр ко всем элементам
2. \`normalize_rows(arr)\` - Вычесть средние значения строк из каждой строки
3. \`scale_columns(arr, scales)\` - Умножить каждый столбец на соответствующий масштаб

## Пример

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

add_scalar(arr, 10)  # [[11, 12, 13], [14, 15, 16]]

# Средние строк: [2, 5]
normalize_rows(arr)  # [[-1, 0, 1], [-1, 0, 1]]

scales = np.array([1, 2, 3])
scale_columns(arr, scales)  # [[1, 4, 9], [4, 10, 18]]
\`\`\``,
			hint1: 'Broadcasting работает автоматически когда формы совместимы',
			hint2: 'Используйте keepdims=True с mean() для сохранения формы при broadcasting',
			whyItMatters: `Broadcasting — сердце векторизованных операций:

- **Нормализация**: Эффективное вычитание среднего/деление на std
- **Масштабирование признаков**: Применение разных масштабов к признакам
- **Добавление смещения**: Добавление bias векторов к выходам батча
- **Attention scores**: Эффективное умножение queries и keys`,
		},
		uz: {
			title: "Broadcasting asoslari",
			description: `# Broadcasting asoslari

Broadcasting NumPy ga turli shakldagi massivlar ustida avtomatik operatsiyalar bajarishga imkon beradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`add_scalar(arr, scalar)\` - Barcha elementlarga skalyar qo'shish
2. \`normalize_rows(arr)\` - Har bir qatordan qator o'rtachalarini ayirish
3. \`scale_columns(arr, scales)\` - Har bir ustunni mos masshtabga ko'paytirish

## Misol

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

add_scalar(arr, 10)  # [[11, 12, 13], [14, 15, 16]]

# Qator o'rtachalari: [2, 5]
normalize_rows(arr)  # [[-1, 0, 1], [-1, 0, 1]]

scales = np.array([1, 2, 3])
scale_columns(arr, scales)  # [[1, 4, 9], [4, 10, 18]]
\`\`\``,
			hint1: "Broadcasting shakllar mos kelganda avtomatik ishlaydi",
			hint2: "Broadcasting uchun shaklni saqlash uchun mean() bilan keepdims=True dan foydalaning",
			whyItMatters: `Broadcasting vektorlashtirilgan operatsiyalarning markazidir:

- **Normalizatsiya**: O'rtachani ayirish/std ga bo'lishni samarali bajarish
- **Xususiyatlarni masshtablash**: Xususiyatlarga turli masshtablarni qo'llash
- **Bias qo'shish**: Batch chiqishlariga bias vektorlarini qo'shish`,
		},
	},
};

export default task;
