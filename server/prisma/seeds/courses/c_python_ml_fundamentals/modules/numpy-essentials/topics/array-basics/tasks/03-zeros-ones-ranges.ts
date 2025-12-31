import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-zeros-ones-ranges',
	title: 'Zeros, Ones, and Ranges',
	difficulty: 'easy',
	tags: ['numpy', 'arrays', 'initialization'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,
	description: `# Zeros, Ones, and Ranges

NumPy provides convenient functions to create arrays filled with specific values or sequences.

## Task

Implement three functions:
1. \`create_zeros(shape)\` - Create an array filled with zeros
2. \`create_ones(shape)\` - Create an array filled with ones
3. \`create_range(start, stop, step)\` - Create an array with evenly spaced values

## Example

\`\`\`python
zeros = create_zeros((2, 3))
# [[0. 0. 0.]
#  [0. 0. 0.]]

ones = create_ones((3,))
# [1. 1. 1.]

range_arr = create_range(0, 10, 2)
# [0 2 4 6 8]
\`\`\`

## Requirements

- Use \`np.zeros()\`, \`np.ones()\`, and \`np.arange()\``,

	initialCode: `import numpy as np

def create_zeros(shape: tuple) -> np.ndarray:
    """Create an array filled with zeros."""
    # Your code here
    pass

def create_ones(shape: tuple) -> np.ndarray:
    """Create an array filled with ones."""
    # Your code here
    pass

def create_range(start: int, stop: int, step: int) -> np.ndarray:
    """Create an array with evenly spaced values."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def create_zeros(shape: tuple) -> np.ndarray:
    """Create an array filled with zeros."""
    return np.zeros(shape)

def create_ones(shape: tuple) -> np.ndarray:
    """Create an array filled with ones."""
    return np.ones(shape)

def create_range(start: int, stop: int, step: int) -> np.ndarray:
    """Create an array with evenly spaced values."""
    return np.arange(start, stop, step)
`,

	testCode: `import numpy as np
import unittest

class TestArrayCreation(unittest.TestCase):
    def test_zeros_1d(self):
        result = create_zeros((5,))
        expected = np.zeros((5,))
        np.testing.assert_array_equal(result, expected)

    def test_zeros_2d(self):
        result = create_zeros((2, 3))
        expected = np.zeros((2, 3))
        np.testing.assert_array_equal(result, expected)

    def test_zeros_3d(self):
        result = create_zeros((2, 3, 4))
        self.assertEqual(result.shape, (2, 3, 4))

    def test_ones_1d(self):
        result = create_ones((4,))
        expected = np.ones((4,))
        np.testing.assert_array_equal(result, expected)

    def test_ones_2d(self):
        result = create_ones((3, 2))
        expected = np.ones((3, 2))
        np.testing.assert_array_equal(result, expected)

    def test_range_basic(self):
        result = create_range(0, 10, 2)
        expected = np.arange(0, 10, 2)
        np.testing.assert_array_equal(result, expected)

    def test_range_step_1(self):
        result = create_range(0, 5, 1)
        expected = np.arange(0, 5, 1)
        np.testing.assert_array_equal(result, expected)

    def test_range_negative(self):
        result = create_range(-5, 5, 2)
        expected = np.arange(-5, 5, 2)
        np.testing.assert_array_equal(result, expected)

    def test_zeros_dtype(self):
        result = create_zeros((2, 2))
        self.assertEqual(result.dtype, np.float64)

    def test_ones_dtype(self):
        result = create_ones((2, 2))
        self.assertEqual(result.dtype, np.float64)
`,

	hint1: 'Use np.zeros(), np.ones(), np.arange()',
	hint2: 'Shape is a tuple, arange takes start, stop, step',

	whyItMatters: `These initialization functions are used constantly in ML:

- \`np.zeros()\` - Initialize weight matrices, create placeholder arrays
- \`np.ones()\` - Create bias vectors, masking arrays
- \`np.arange()\` - Generate indices, create time series, batch processing

You'll use these in every neural network, data preprocessing pipeline, and feature engineering task.`,

	translations: {
		ru: {
			title: 'Нули, единицы и диапазоны',
			description: `# Нули, единицы и диапазоны

NumPy предоставляет удобные функции для создания массивов с определёнными значениями или последовательностями.

## Задача

Реализуйте три функции:
1. \`create_zeros(shape)\` - Создать массив, заполненный нулями
2. \`create_ones(shape)\` - Создать массив, заполненный единицами
3. \`create_range(start, stop, step)\` - Создать массив с равномерно распределёнными значениями

## Пример

\`\`\`python
zeros = create_zeros((2, 3))
# [[0. 0. 0.]
#  [0. 0. 0.]]

ones = create_ones((3,))
# [1. 1. 1.]

range_arr = create_range(0, 10, 2)
# [0 2 4 6 8]
\`\`\`

## Требования

- Используйте \`np.zeros()\`, \`np.ones()\` и \`np.arange()\``,
			hint1: 'Используйте np.zeros(), np.ones(), np.arange()',
			hint2: 'Shape - это кортеж, arange принимает start, stop, step',
			whyItMatters: `Эти функции инициализации постоянно используются в ML:

- \`np.zeros()\` - Инициализация матриц весов, создание placeholder массивов
- \`np.ones()\` - Создание векторов смещения, массивов масок
- \`np.arange()\` - Генерация индексов, создание временных рядов, batch обработка`,
		},
		uz: {
			title: "Nollar, birliklar va diapazonlar",
			description: `# Nollar, birliklar va diapazonlar

NumPy ma'lum qiymatlar yoki ketma-ketliklar bilan to'ldirilgan massivlarni yaratish uchun qulay funksiyalarni taqdim etadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`create_zeros(shape)\` - Nollar bilan to'ldirilgan massiv yaratish
2. \`create_ones(shape)\` - Birliklar bilan to'ldirilgan massiv yaratish
3. \`create_range(start, stop, step)\` - Teng masofadagi qiymatlar bilan massiv yaratish

## Misol

\`\`\`python
zeros = create_zeros((2, 3))
# [[0. 0. 0.]
#  [0. 0. 0.]]

ones = create_ones((3,))
# [1. 1. 1.]

range_arr = create_range(0, 10, 2)
# [0 2 4 6 8]
\`\`\`

## Talablar

- \`np.zeros()\`, \`np.ones()\` va \`np.arange()\` dan foydalaning`,
			hint1: "np.zeros(), np.ones(), np.arange() dan foydalaning",
			hint2: "Shape - bu tuple, arange start, stop, step qabul qiladi",
			whyItMatters: `Bu initsializatsiya funksiyalari ML da doimo ishlatiladi:

- \`np.zeros()\` - Vazn matritsalarini initsializatsiya qilish, placeholder massivlar yaratish
- \`np.ones()\` - Bias vektorlarini yaratish, mask massivlari
- \`np.arange()\` - Indekslarni generatsiya qilish, vaqt qatorlarini yaratish`,
		},
	},
};

export default task;
