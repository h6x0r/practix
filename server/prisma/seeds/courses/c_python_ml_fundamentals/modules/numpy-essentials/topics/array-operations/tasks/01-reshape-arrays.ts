import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-reshape-arrays',
	title: 'Reshaping Arrays',
	difficulty: 'easy',
	tags: ['numpy', 'reshape', 'dimensions'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Reshaping Arrays

Reshaping allows you to change the dimensions of an array without changing its data. This is fundamental for preparing data for ML models.

## Task

Implement three functions:
1. \`flatten_array(arr)\` - Convert any array to 1D
2. \`reshape_to_2d(arr, rows, cols)\` - Reshape to specified 2D shape
3. \`add_dimension(arr)\` - Add a new axis at position 0

## Example

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flatten_array(arr)  # [1, 2, 3, 4, 5, 6]
reshape_to_2d(np.arange(6), 2, 3)  # [[0, 1, 2], [3, 4, 5]]
add_dimension(arr)  # shape: (1, 2, 3)
\`\`\``,

	initialCode: `import numpy as np

def flatten_array(arr: np.ndarray) -> np.ndarray:
    """Convert any array to 1D."""
    # Your code here
    pass

def reshape_to_2d(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reshape array to specified 2D shape."""
    # Your code here
    pass

def add_dimension(arr: np.ndarray) -> np.ndarray:
    """Add a new axis at position 0."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def flatten_array(arr: np.ndarray) -> np.ndarray:
    """Convert any array to 1D."""
    return arr.flatten()

def reshape_to_2d(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reshape array to specified 2D shape."""
    return arr.reshape(rows, cols)

def add_dimension(arr: np.ndarray) -> np.ndarray:
    """Add a new axis at position 0."""
    return arr[np.newaxis, ...]
`,

	testCode: `import numpy as np
import unittest

class TestReshapeArrays(unittest.TestCase):
    def test_flatten_2d(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = flatten_array(arr)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_flatten_3d(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = flatten_array(arr)
        np.testing.assert_array_equal(result, np.arange(24))

    def test_flatten_already_1d(self):
        arr = np.array([1, 2, 3])
        result = flatten_array(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_reshape_to_2d_basic(self):
        arr = np.arange(6)
        result = reshape_to_2d(arr, 2, 3)
        expected = np.array([[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(result, expected)

    def test_reshape_to_2d_square(self):
        arr = np.arange(9)
        result = reshape_to_2d(arr, 3, 3)
        self.assertEqual(result.shape, (3, 3))

    def test_reshape_to_2d_column(self):
        arr = np.arange(4)
        result = reshape_to_2d(arr, 4, 1)
        self.assertEqual(result.shape, (4, 1))

    def test_add_dimension_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = add_dimension(arr)
        self.assertEqual(result.shape, (1, 2, 2))

    def test_add_dimension_1d(self):
        arr = np.array([1, 2, 3])
        result = add_dimension(arr)
        self.assertEqual(result.shape, (1, 3))

    def test_add_dimension_values(self):
        arr = np.array([[1, 2], [3, 4]])
        result = add_dimension(arr)
        np.testing.assert_array_equal(result[0], arr)

    def test_reshape_preserves_data(self):
        arr = np.arange(12)
        result = reshape_to_2d(arr, 3, 4)
        np.testing.assert_array_equal(result.flatten(), arr)
`,

	hint1: 'Use arr.flatten() or arr.ravel() for 1D conversion',
	hint2: 'Use arr[np.newaxis, ...] or np.expand_dims(arr, 0) to add dimension',

	whyItMatters: `Reshaping is essential in ML for:

- **Batch processing**: Models expect input shape (batch_size, features)
- **Image processing**: Convert (H, W, C) to (C, H, W) for different frameworks
- **Feature vectors**: Flatten 2D/3D data for traditional ML algorithms
- **Broadcasting**: Add dimensions to enable element-wise operations

Incorrect shapes cause 90% of deep learning debugging time.`,

	translations: {
		ru: {
			title: 'Изменение формы массивов',
			description: `# Изменение формы массивов

Изменение формы позволяет изменить размерность массива без изменения его данных. Это фундаментально для подготовки данных для ML моделей.

## Задача

Реализуйте три функции:
1. \`flatten_array(arr)\` - Преобразовать любой массив в 1D
2. \`reshape_to_2d(arr, rows, cols)\` - Изменить форму на указанную 2D
3. \`add_dimension(arr)\` - Добавить новую ось в позицию 0

## Пример

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flatten_array(arr)  # [1, 2, 3, 4, 5, 6]
reshape_to_2d(np.arange(6), 2, 3)  # [[0, 1, 2], [3, 4, 5]]
add_dimension(arr)  # shape: (1, 2, 3)
\`\`\``,
			hint1: 'Используйте arr.flatten() или arr.ravel() для преобразования в 1D',
			hint2: 'Используйте arr[np.newaxis, ...] или np.expand_dims(arr, 0) для добавления измерения',
			whyItMatters: `Изменение формы необходимо в ML для:

- **Batch обработка**: Модели ожидают форму входа (batch_size, features)
- **Обработка изображений**: Конвертация (H, W, C) в (C, H, W) для разных фреймворков
- **Векторы признаков**: Выпрямление 2D/3D данных для классических ML алгоритмов
- **Broadcasting**: Добавление измерений для поэлементных операций`,
		},
		uz: {
			title: "Massiv shaklini o'zgartirish",
			description: `# Massiv shaklini o'zgartirish

Shaklni o'zgartirish ma'lumotlarni o'zgartirmasdan massiv o'lchamlarini o'zgartirish imkonini beradi. Bu ML modellari uchun ma'lumotlarni tayyorlashda asosiydir.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`flatten_array(arr)\` - Har qanday massivni 1D ga aylantirish
2. \`reshape_to_2d(arr, rows, cols)\` - Ko'rsatilgan 2D shaklga o'zgartirish
3. \`add_dimension(arr)\` - 0 pozitsiyasiga yangi o'q qo'shish

## Misol

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flatten_array(arr)  # [1, 2, 3, 4, 5, 6]
reshape_to_2d(np.arange(6), 2, 3)  # [[0, 1, 2], [3, 4, 5]]
add_dimension(arr)  # shape: (1, 2, 3)
\`\`\``,
			hint1: "1D ga aylantirish uchun arr.flatten() yoki arr.ravel() dan foydalaning",
			hint2: "O'lcham qo'shish uchun arr[np.newaxis, ...] yoki np.expand_dims(arr, 0) dan foydalaning",
			whyItMatters: `Shaklni o'zgartirish ML da quyidagilar uchun zarur:

- **Batch ishlov berish**: Modellar (batch_size, features) kirish shaklini kutadi
- **Tasvir ishlov berish**: Turli frameworklar uchun (H, W, C) ni (C, H, W) ga aylantirish
- **Xususiyat vektorlari**: Klassik ML algoritmlari uchun 2D/3D ma'lumotlarni tekislash`,
		},
	},
};

export default task;
