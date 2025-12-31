import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-create-array-from-list',
	title: 'Create Array from List',
	difficulty: 'easy',
	tags: ['numpy', 'arrays', 'fundamentals'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Create NumPy Array from List

NumPy arrays are the foundation of scientific computing in Python. They are more efficient than Python lists for numerical operations.

## Task

Implement the \`create_array\` function that takes a Python list and returns a NumPy array.

## Example

\`\`\`python
result = create_array([1, 2, 3, 4, 5])
print(result)  # [1 2 3 4 5]
print(type(result))  # <class 'numpy.ndarray'>
\`\`\`

## Requirements

- Use \`np.array()\` to convert the list
- Return the NumPy array`,

	initialCode: `import numpy as np

def create_array(data: list) -> np.ndarray:
    # TODO: Convert the list to a NumPy array using np.array()

    pass
`,

	solutionCode: `import numpy as np

def create_array(data: list) -> np.ndarray:
    """
    Convert a Python list to a NumPy array.

    Args:
        data: A Python list of numbers

    Returns:
        A NumPy array containing the same elements
    """
    return np.array(data)
`,

	testCode: `import numpy as np
import unittest

class TestCreateArray(unittest.TestCase):
    def test_basic_integers(self):
        result = create_array([1, 2, 3, 4, 5])
        expected = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result, expected, "Expected [1 2 3 4 5]")

    def test_empty_list(self):
        result = create_array([])
        expected = np.array([])
        np.testing.assert_array_equal(result, expected, "Expected empty array")

    def test_single_element(self):
        result = create_array([42])
        expected = np.array([42])
        np.testing.assert_array_equal(result, expected, "Expected [42]")

    def test_floats(self):
        result = create_array([1.5, 2.5, 3.5])
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(result, expected, err_msg="Expected [1.5 2.5 3.5]")

    def test_negative_numbers(self):
        result = create_array([-1, -2, -3])
        expected = np.array([-1, -2, -3])
        np.testing.assert_array_equal(result, expected, "Expected [-1 -2 -3]")

    def test_mixed_int_float(self):
        result = create_array([1, 2.5, 3])
        expected = np.array([1, 2.5, 3])
        np.testing.assert_array_almost_equal(result, expected, err_msg="Expected [1.0 2.5 3.0]")

    def test_large_numbers(self):
        result = create_array([1000000, 2000000, 3000000])
        expected = np.array([1000000, 2000000, 3000000])
        np.testing.assert_array_equal(result, expected, "Expected large numbers array")

    def test_returns_ndarray(self):
        result = create_array([1, 2, 3])
        self.assertIsInstance(result, np.ndarray, "Expected numpy.ndarray type")

    def test_zeros(self):
        result = create_array([0, 0, 0])
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected, "Expected [0 0 0]")

    def test_long_list(self):
        data = list(range(100))
        result = create_array(data)
        expected = np.array(data)
        np.testing.assert_array_equal(result, expected, "Expected array of 0-99")
`,

	hint1: 'Use np.array() to convert lists to arrays',
	hint2: 'The function takes a list and returns np.array(list)',

	whyItMatters: `NumPy arrays are the backbone of data science in Python. Unlike Python lists, NumPy arrays:

- Store data contiguously in memory for faster access
- Support vectorized operations (operations on entire arrays at once)
- Use less memory for large datasets
- Are required by virtually all ML libraries (scikit-learn, TensorFlow, PyTorch)

Understanding array creation is the first step to working with real-world data pipelines.`,

	translations: {
		ru: {
			title: 'Создание массива из списка',
			description: `# Создание NumPy массива из списка

Массивы NumPy — основа научных вычислений в Python. Они эффективнее списков Python для числовых операций.

## Задача

Реализуйте функцию \`create_array\`, которая принимает список Python и возвращает массив NumPy.

## Пример

\`\`\`python
result = create_array([1, 2, 3, 4, 5])
print(result)  # [1 2 3 4 5]
print(type(result))  # <class 'numpy.ndarray'>
\`\`\`

## Требования

- Используйте \`np.array()\` для преобразования списка
- Верните массив NumPy`,
			hint1: 'Используйте np.array() для преобразования списков в массивы',
			hint2: 'Функция принимает список и возвращает np.array(список)',
			whyItMatters: `Массивы NumPy — основа data science в Python. В отличие от списков Python, массивы NumPy:

- Хранят данные непрерывно в памяти для быстрого доступа
- Поддерживают векторизованные операции (операции над всем массивом сразу)
- Используют меньше памяти для больших данных
- Требуются практически всеми ML библиотеками (scikit-learn, TensorFlow, PyTorch)`,
		},
		uz: {
			title: "Ro'yxatdan massiv yaratish",
			description: `# Python ro'yxatidan NumPy massivi yaratish

NumPy massivlari Python'da ilmiy hisob-kitoblarning asosidir. Ular Python ro'yxatlaridan raqamli amallar uchun samaraliroq.

## Topshiriq

Python ro'yxatini qabul qilib, NumPy massivini qaytaruvchi \`create_array\` funksiyasini amalga oshiring.

## Misol

\`\`\`python
result = create_array([1, 2, 3, 4, 5])
print(result)  # [1 2 3 4 5]
print(type(result))  # <class 'numpy.ndarray'>
\`\`\`

## Talablar

- Ro'yxatni o'zgartirish uchun \`np.array()\` dan foydalaning
- NumPy massivini qaytaring`,
			hint1: "Ro'yxatlarni massivlarga o'zgartirish uchun np.array() dan foydalaning",
			hint2: "Funksiya ro'yxatni qabul qiladi va np.array(ro'yxat) ni qaytaradi",
			whyItMatters: `NumPy massivlari Python'da data science asosi. Python ro'yxatlaridan farqli o'laroq, NumPy massivlari:

- Ma'lumotlarni tez kirish uchun xotirada ketma-ket saqlaydi
- Vektorlashtirilgan amallarni qo'llab-quvvatlaydi
- Katta ma'lumotlar uchun kamroq xotira ishlatadi
- Deyarli barcha ML kutubxonalari tomonidan talab qilinadi`,
		},
	},
};

export default task;
