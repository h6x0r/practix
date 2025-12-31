import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-boolean-indexing',
	title: 'Boolean Indexing',
	difficulty: 'medium',
	tags: ['numpy', 'indexing', 'filtering'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,
	description: `# Boolean Indexing

Boolean indexing allows you to select elements based on conditions. This is extremely powerful for filtering data.

## Task

Implement three functions:
1. \`filter_positive(arr)\` - Return only positive values
2. \`filter_range(arr, low, high)\` - Return values in range [low, high]
3. \`replace_negatives(arr, value)\` - Replace negative values with given value

## Example

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2, 3])

filter_positive(arr)           # [1, 2, 3]
filter_range(arr, -1, 2)       # [-1, 0, 1, 2]
replace_negatives(arr, 0)      # [0, 0, 0, 1, 2, 3]
\`\`\`

## Requirements

- Use boolean conditions like \`arr > 0\`
- For replace_negatives, return a new array (don't modify original)`,

	initialCode: `import numpy as np

def filter_positive(arr: np.ndarray) -> np.ndarray:
    """Return only positive values."""
    # Your code here
    pass

def filter_range(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return values in range [low, high]."""
    # Your code here
    pass

def replace_negatives(arr: np.ndarray, value: float) -> np.ndarray:
    """Replace negative values with given value (return new array)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def filter_positive(arr: np.ndarray) -> np.ndarray:
    """Return only positive values."""
    return arr[arr > 0]

def filter_range(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return values in range [low, high]."""
    return arr[(arr >= low) & (arr <= high)]

def filter_range(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return values in range [low, high]."""
    return arr[(arr >= low) & (arr <= high)]

def replace_negatives(arr: np.ndarray, value: float) -> np.ndarray:
    """Replace negative values with given value (return new array)."""
    result = arr.copy()
    result[result < 0] = value
    return result
`,

	testCode: `import numpy as np
import unittest

class TestBooleanIndexing(unittest.TestCase):
    def test_filter_positive_basic(self):
        arr = np.array([-2, -1, 0, 1, 2, 3])
        result = filter_positive(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_filter_positive_all_negative(self):
        arr = np.array([-3, -2, -1])
        result = filter_positive(arr)
        self.assertEqual(len(result), 0)

    def test_filter_positive_all_positive(self):
        arr = np.array([1, 2, 3])
        result = filter_positive(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_filter_range_basic(self):
        arr = np.array([-2, -1, 0, 1, 2, 3])
        result = filter_range(arr, -1, 2)
        np.testing.assert_array_equal(result, [-1, 0, 1, 2])

    def test_filter_range_inclusive(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = filter_range(arr, 2, 4)
        np.testing.assert_array_equal(result, [2, 3, 4])

    def test_filter_range_floats(self):
        arr = np.array([0.5, 1.5, 2.5, 3.5])
        result = filter_range(arr, 1.0, 3.0)
        np.testing.assert_array_equal(result, [1.5, 2.5])

    def test_replace_negatives_basic(self):
        arr = np.array([-2, -1, 0, 1, 2])
        result = replace_negatives(arr, 0)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 2])

    def test_replace_negatives_no_modify(self):
        arr = np.array([-2, -1, 0, 1, 2])
        result = replace_negatives(arr, 0)
        self.assertEqual(arr[0], -2)  # Original unchanged

    def test_replace_negatives_custom_value(self):
        arr = np.array([-5, -3, 0, 2, 4])
        result = replace_negatives(arr, 99)
        np.testing.assert_array_equal(result, [99, 99, 0, 2, 4])

    def test_replace_negatives_no_negatives(self):
        arr = np.array([1, 2, 3])
        result = replace_negatives(arr, 0)
        np.testing.assert_array_equal(result, [1, 2, 3])
`,

	hint1: 'Use arr[arr > 0] to filter by condition',
	hint2: 'Combine conditions with & operator: (arr >= low) & (arr <= high)',

	whyItMatters: `Boolean indexing is used everywhere in data science:

- **Outlier removal**: Filter values outside acceptable ranges
- **Missing data handling**: Identify and replace NaN/invalid values
- **Feature selection**: Select samples meeting specific criteria
- **Data cleaning**: Remove corrupted or invalid entries

This is one of the most commonly used NumPy operations in real ML pipelines.`,

	translations: {
		ru: {
			title: 'Булева индексация',
			description: `# Булева индексация

Булева индексация позволяет выбирать элементы на основе условий. Это чрезвычайно мощный инструмент для фильтрации данных.

## Задача

Реализуйте три функции:
1. \`filter_positive(arr)\` - Вернуть только положительные значения
2. \`filter_range(arr, low, high)\` - Вернуть значения в диапазоне [low, high]
3. \`replace_negatives(arr, value)\` - Заменить отрицательные значения на заданное

## Пример

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2, 3])

filter_positive(arr)           # [1, 2, 3]
filter_range(arr, -1, 2)       # [-1, 0, 1, 2]
replace_negatives(arr, 0)      # [0, 0, 0, 1, 2, 3]
\`\`\`

## Требования

- Используйте булевы условия типа \`arr > 0\`
- Для replace_negatives возвращайте новый массив (не изменяйте оригинал)`,
			hint1: 'Используйте arr[arr > 0] для фильтрации по условию',
			hint2: 'Комбинируйте условия оператором &: (arr >= low) & (arr <= high)',
			whyItMatters: `Булева индексация используется повсюду в data science:

- **Удаление выбросов**: Фильтрация значений вне допустимых диапазонов
- **Обработка пропущенных данных**: Идентификация и замена NaN/невалидных значений
- **Отбор признаков**: Выбор сэмплов по определённым критериям
- **Очистка данных**: Удаление повреждённых или невалидных записей`,
		},
		uz: {
			title: "Mantiqiy indekslash",
			description: `# Mantiqiy indekslash

Mantiqiy indekslash shartlar asosida elementlarni tanlash imkonini beradi. Bu ma'lumotlarni filtrlash uchun juda kuchli.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`filter_positive(arr)\` - Faqat musbat qiymatlarni qaytarish
2. \`filter_range(arr, low, high)\` - [low, high] oralig'idagi qiymatlarni qaytarish
3. \`replace_negatives(arr, value)\` - Manfiy qiymatlarni berilgan qiymat bilan almashtirish

## Misol

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2, 3])

filter_positive(arr)           # [1, 2, 3]
filter_range(arr, -1, 2)       # [-1, 0, 1, 2]
replace_negatives(arr, 0)      # [0, 0, 0, 1, 2, 3]
\`\`\`

## Talablar

- \`arr > 0\` kabi mantiqiy shartlardan foydalaning
- replace_negatives uchun yangi massiv qaytaring (asl massivni o'zgartirmang)`,
			hint1: "Shart bo'yicha filtrlash uchun arr[arr > 0] dan foydalaning",
			hint2: "Shartlarni & operatori bilan birlashtiring: (arr >= low) & (arr <= high)",
			whyItMatters: `Mantiqiy indekslash data science'da hamma joyda ishlatiladi:

- **Outlierlarni olib tashlash**: Qabul qilinadigan diapazonlardan tashqaridagi qiymatlarni filtrlash
- **Yo'qolgan ma'lumotlarni boshqarish**: NaN/noto'g'ri qiymatlarni aniqlash va almashtirish
- **Xususiyatlarni tanlash**: Ma'lum mezonlarga javob beradigan namunalarni tanlash`,
		},
	},
};

export default task;
