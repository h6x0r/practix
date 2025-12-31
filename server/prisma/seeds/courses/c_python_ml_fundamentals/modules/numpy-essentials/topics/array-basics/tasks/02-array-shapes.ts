import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-array-shapes',
	title: 'Array Shapes and Dimensions',
	difficulty: 'easy',
	tags: ['numpy', 'arrays', 'shapes'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# Array Shapes and Dimensions

NumPy arrays have a \`shape\` attribute that tells you the size of each dimension. Understanding shapes is crucial for data manipulation.

## Task

Implement the \`get_array_info\` function that returns a dictionary with the array's shape, number of dimensions, and total size.

## Example

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])
info = get_array_info(arr)
print(info)
# {'shape': (2, 3), 'ndim': 2, 'size': 6}
\`\`\`

## Requirements

- Return a dictionary with keys: 'shape', 'ndim', 'size'
- Use array attributes: \`.shape\`, \`.ndim\`, \`.size\``,

	initialCode: `import numpy as np

def get_array_info(arr: np.ndarray) -> dict:
    # TODO: Return dict with 'shape', 'ndim', and 'size' of the array

    pass
`,

	solutionCode: `import numpy as np

def get_array_info(arr: np.ndarray) -> dict:
    """
    Get information about a NumPy array.

    Args:
        arr: A NumPy array

    Returns:
        A dictionary with 'shape', 'ndim', and 'size' keys
    """
    return {
        'shape': arr.shape,
        'ndim': arr.ndim,
        'size': arr.size
    }
`,

	testCode: `import numpy as np
import unittest

class TestArrayInfo(unittest.TestCase):
    def test_2d_array(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (2, 3), "Expected shape (2, 3)")
        self.assertEqual(result['ndim'], 2, "Expected ndim 2")
        self.assertEqual(result['size'], 6, "Expected size 6")

    def test_1d_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (5,), "Expected shape (5,)")
        self.assertEqual(result['ndim'], 1, "Expected ndim 1")
        self.assertEqual(result['size'], 5, "Expected size 5")

    def test_3d_array(self):
        arr = np.zeros((2, 3, 4))
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (2, 3, 4), "Expected shape (2, 3, 4)")
        self.assertEqual(result['ndim'], 3, "Expected ndim 3")
        self.assertEqual(result['size'], 24, "Expected size 24")

    def test_scalar_like(self):
        arr = np.array(42)
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (), "Expected shape ()")
        self.assertEqual(result['ndim'], 0, "Expected ndim 0")
        self.assertEqual(result['size'], 1, "Expected size 1")

    def test_empty_array(self):
        arr = np.array([])
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (0,), "Expected shape (0,)")
        self.assertEqual(result['ndim'], 1, "Expected ndim 1")
        self.assertEqual(result['size'], 0, "Expected size 0")

    def test_column_vector(self):
        arr = np.array([[1], [2], [3]])
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (3, 1), "Expected shape (3, 1)")
        self.assertEqual(result['ndim'], 2, "Expected ndim 2")
        self.assertEqual(result['size'], 3, "Expected size 3")

    def test_row_vector(self):
        arr = np.array([[1, 2, 3]])
        result = get_array_info(arr)
        self.assertEqual(result['shape'], (1, 3), "Expected shape (1, 3)")
        self.assertEqual(result['ndim'], 2, "Expected ndim 2")
        self.assertEqual(result['size'], 3, "Expected size 3")

    def test_returns_dict(self):
        arr = np.array([1, 2, 3])
        result = get_array_info(arr)
        self.assertIsInstance(result, dict, "Expected dict type")

    def test_has_required_keys(self):
        arr = np.array([1, 2, 3])
        result = get_array_info(arr)
        self.assertIn('shape', result, "Expected 'shape' key")
        self.assertIn('ndim', result, "Expected 'ndim' key")
        self.assertIn('size', result, "Expected 'size' key")

    def test_large_array(self):
        arr = np.zeros((10, 20, 30))
        result = get_array_info(arr)
        self.assertEqual(result['size'], 6000, "Expected size 6000")
`,

	hint1: 'Use .shape, .ndim, and .size attributes',
	hint2: 'Return a dict with keys: shape, ndim, size',

	whyItMatters: `Understanding array shapes is essential for:

- Debugging dimension mismatches in ML pipelines
- Reshaping data for model input (e.g., images, sequences)
- Verifying data after loading from files
- Broadcasting operations between arrays of different shapes

Most ML errors come from shape mismatches - knowing how to inspect shapes saves hours of debugging.`,

	translations: {
		ru: {
			title: 'Формы и размерности массивов',
			description: `# Формы и размерности массивов

Массивы NumPy имеют атрибут \`shape\`, который показывает размер каждого измерения. Понимание форм критически важно для манипуляции данными.

## Задача

Реализуйте функцию \`get_array_info\`, которая возвращает словарь с формой массива, количеством измерений и общим размером.

## Пример

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])
info = get_array_info(arr)
print(info)
# {'shape': (2, 3), 'ndim': 2, 'size': 6}
\`\`\`

## Требования

- Верните словарь с ключами: 'shape', 'ndim', 'size'
- Используйте атрибуты массива: \`.shape\`, \`.ndim\`, \`.size\``,
			hint1: 'Используйте атрибуты .shape, .ndim и .size',
			hint2: 'Верните словарь с ключами: shape, ndim, size',
			whyItMatters: `Понимание форм массивов необходимо для:

- Отладки несоответствия размерностей в ML пайплайнах
- Изменения формы данных для ввода модели (изображения, последовательности)
- Проверки данных после загрузки из файлов
- Операций broadcasting между массивами разных форм`,
		},
		uz: {
			title: "Massiv shakllari va o'lchamlari",
			description: `# Massiv shakllari va o'lchamlari

NumPy massivlari har bir o'lchamning hajmini ko'rsatuvchi \`shape\` atributiga ega. Shakllarni tushunish ma'lumotlarni boshqarish uchun juda muhim.

## Topshiriq

Massivning shakli, o'lchamlar soni va umumiy hajmini o'z ichiga olgan lug'atni qaytaruvchi \`get_array_info\` funksiyasini amalga oshiring.

## Misol

\`\`\`python
arr = np.array([[1, 2, 3], [4, 5, 6]])
info = get_array_info(arr)
print(info)
# {'shape': (2, 3), 'ndim': 2, 'size': 6}
\`\`\`

## Talablar

- Kalitlar bilan lug'at qaytaring: 'shape', 'ndim', 'size'
- Massiv atributlaridan foydalaning: \`.shape\`, \`.ndim\`, \`.size\``,
			hint1: ".shape, .ndim va .size atributlaridan foydalaning",
			hint2: "shape, ndim, size kalitlari bilan lug'at qaytaring",
			whyItMatters: `Massiv shakllarini tushunish quyidagilar uchun zarur:

- ML pipeline'larda o'lcham nomutanosibliklarini debug qilish
- Model kirishi uchun ma'lumotlarni qayta shakllantirish
- Fayllardan yuklangandan keyin ma'lumotlarni tekshirish
- Turli shakldagi massivlar orasida broadcasting amallari`,
		},
	},
};

export default task;
