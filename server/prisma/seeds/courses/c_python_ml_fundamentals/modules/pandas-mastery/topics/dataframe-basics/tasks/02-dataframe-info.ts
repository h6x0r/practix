import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-dataframe-info',
	title: 'DataFrame Inspection',
	difficulty: 'easy',
	tags: ['pandas', 'dataframe', 'inspection'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# DataFrame Inspection

Before analyzing data, you need to understand its structure, types, and basic statistics.

## Task

Implement four functions:
1. \`get_shape(df)\` - Return (rows, columns) tuple
2. \`get_column_types(df)\` - Return dict of column names to their dtypes
3. \`get_memory_usage(df)\` - Return total memory usage in bytes
4. \`get_summary_stats(df)\` - Return dict with count, mean, std for numeric columns

## Example

\`\`\`python
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30], 'salary': [50000, 60000]})

get_shape(df)  # (2, 3)
get_column_types(df)  # {'name': 'object', 'age': 'int64', 'salary': 'int64'}
get_memory_usage(df)  # ~256 bytes
get_summary_stats(df)  # {'count': 2, 'mean': {'age': 27.5, 'salary': 55000}, ...}
\`\`\``,

	initialCode: `import pandas as pd

def get_shape(df: pd.DataFrame) -> tuple:
    """Return (rows, columns) tuple."""
    # Your code here
    pass

def get_column_types(df: pd.DataFrame) -> dict:
    """Return dict of column names to their dtype strings."""
    # Your code here
    pass

def get_memory_usage(df: pd.DataFrame) -> int:
    """Return total memory usage in bytes."""
    # Your code here
    pass

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return dict with count, mean, std for numeric columns."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def get_shape(df: pd.DataFrame) -> tuple:
    """Return (rows, columns) tuple."""
    return df.shape

def get_column_types(df: pd.DataFrame) -> dict:
    """Return dict of column names to their dtype strings."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def get_memory_usage(df: pd.DataFrame) -> int:
    """Return total memory usage in bytes."""
    return df.memory_usage(deep=True).sum()

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return dict with count, mean, std for numeric columns."""
    numeric_df = df.select_dtypes(include=['number'])
    return {
        'count': len(df),
        'mean': numeric_df.mean().to_dict(),
        'std': numeric_df.std().to_dict()
    }
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestDataFrameInfo(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0]
        })

    def test_get_shape_basic(self):
        result = get_shape(self.df)
        self.assertEqual(result, (3, 3))

    def test_get_shape_empty(self):
        df = pd.DataFrame()
        result = get_shape(df)
        self.assertEqual(result, (0, 0))

    def test_get_shape_single_column(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        result = get_shape(df)
        self.assertEqual(result, (5, 1))

    def test_get_column_types_basic(self):
        result = get_column_types(self.df)
        self.assertEqual(result['name'], 'object')
        self.assertIn('int', result['age'])

    def test_get_column_types_keys(self):
        result = get_column_types(self.df)
        self.assertEqual(set(result.keys()), {'name', 'age', 'salary'})

    def test_get_memory_usage_positive(self):
        result = get_memory_usage(self.df)
        self.assertGreater(result, 0)

    def test_get_memory_usage_type(self):
        result = get_memory_usage(self.df)
        self.assertIsInstance(result, (int, np.integer))

    def test_get_summary_stats_count(self):
        result = get_summary_stats(self.df)
        self.assertEqual(result['count'], 3)

    def test_get_summary_stats_mean(self):
        result = get_summary_stats(self.df)
        self.assertAlmostEqual(result['mean']['age'], 30.0)
        self.assertAlmostEqual(result['mean']['salary'], 60000.0)

    def test_get_summary_stats_std(self):
        result = get_summary_stats(self.df)
        self.assertIn('std', result)
        self.assertIn('age', result['std'])
`,

	hint1: 'Use df.shape, df.dtypes, df.memory_usage()',
	hint2: 'Use df.select_dtypes(include=["number"]) for numeric columns only',

	whyItMatters: `DataFrame inspection is crucial for:

- **Data quality**: Detect missing values and wrong types
- **Memory optimization**: Choose appropriate dtypes
- **Feature selection**: Understand data distributions
- **Debugging**: Verify data after transformations

Always inspect data before building models.`,

	translations: {
		ru: {
			title: 'Инспекция DataFrame',
			description: `# Инспекция DataFrame

Прежде чем анализировать данные, нужно понять их структуру, типы и базовую статистику.

## Задача

Реализуйте четыре функции:
1. \`get_shape(df)\` - Вернуть кортеж (строки, столбцы)
2. \`get_column_types(df)\` - Вернуть dict имён столбцов к их типам
3. \`get_memory_usage(df)\` - Вернуть общее использование памяти в байтах
4. \`get_summary_stats(df)\` - Вернуть dict с count, mean, std для числовых столбцов

## Пример

\`\`\`python
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30], 'salary': [50000, 60000]})

get_shape(df)  # (2, 3)
get_column_types(df)  # {'name': 'object', 'age': 'int64', 'salary': 'int64'}
\`\`\``,
			hint1: 'Используйте df.shape, df.dtypes, df.memory_usage()',
			hint2: 'Используйте df.select_dtypes(include=["number"]) только для числовых столбцов',
			whyItMatters: `Инспекция DataFrame критична для:

- **Качество данных**: Обнаружение пропущенных значений и неправильных типов
- **Оптимизация памяти**: Выбор подходящих типов данных
- **Отбор признаков**: Понимание распределений данных
- **Отладка**: Проверка данных после преобразований`,
		},
		uz: {
			title: "DataFrame tekshiruvi",
			description: `# DataFrame tekshiruvi

Ma'lumotlarni tahlil qilishdan oldin ularning strukturasini, turlarini va asosiy statistikasini tushunish kerak.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`get_shape(df)\` - (qatorlar, ustunlar) tuple qaytarish
2. \`get_column_types(df)\` - Ustun nomlari va ularning dtype lari dict qaytarish
3. \`get_memory_usage(df)\` - Umumiy xotira ishlatilishini baytlarda qaytarish
4. \`get_summary_stats(df)\` - Raqamli ustunlar uchun count, mean, std bilan dict qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30], 'salary': [50000, 60000]})

get_shape(df)  # (2, 3)
get_column_types(df)  # {'name': 'object', 'age': 'int64', 'salary': 'int64'}
get_memory_usage(df)  # ~256 bytes
get_summary_stats(df)  # {'count': 2, 'mean': {'age': 27.5, 'salary': 55000}, ...}
\`\`\``,
			hint1: "df.shape, df.dtypes, df.memory_usage() dan foydalaning",
			hint2: "Faqat raqamli ustunlar uchun df.select_dtypes(include=['number']) dan foydalaning",
			whyItMatters: `DataFrame tekshiruvi quyidagilar uchun muhim:

- **Ma'lumotlar sifati**: Yo'qolgan qiymatlar va noto'g'ri turlarni aniqlash
- **Xotira optimallashtirish**: Mos dtype larni tanlash
- **Xususiyatlarni tanlash**: Ma'lumotlar taqsimotini tushunish`,
		},
	},
};

export default task;
