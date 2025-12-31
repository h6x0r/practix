import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-missing-values',
	title: 'Handling Missing Values',
	difficulty: 'medium',
	tags: ['pandas', 'missing', 'nan', 'data-cleaning'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Handling Missing Values

Real-world data often contains missing values (NaN). Handling them correctly is crucial for data quality.

## Task

Implement four functions:
1. \`count_missing(df)\` - Return dict of column names to missing value counts
2. \`drop_missing_rows(df, subset)\` - Drop rows with NaN in specified columns
3. \`fill_with_value(df, column, value)\` - Fill NaN in column with given value
4. \`fill_with_mean(df, column)\` - Fill NaN in column with column mean

## Example

\`\`\`python
df = pd.DataFrame({
    'a': [1, None, 3, None],
    'b': [4, 5, None, 7]
})

count_missing(df)  # {'a': 2, 'b': 1}
drop_missing_rows(df, ['a'])  # Keeps rows where 'a' is not NaN
fill_with_value(df, 'b', 0)  # Replace NaN in 'b' with 0
fill_with_mean(df, 'a')  # Replace NaN in 'a' with mean(a)
\`\`\``,

	initialCode: `import pandas as pd

def count_missing(df: pd.DataFrame) -> dict:
    """Return dict of column names to missing value counts."""
    # Your code here
    pass

def drop_missing_rows(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Drop rows with NaN in specified columns."""
    # Your code here
    pass

def fill_with_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Fill NaN in column with given value."""
    # Your code here
    pass

def fill_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill NaN in column with column mean."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def count_missing(df: pd.DataFrame) -> dict:
    """Return dict of column names to missing value counts."""
    return df.isnull().sum().to_dict()

def drop_missing_rows(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Drop rows with NaN in specified columns."""
    return df.dropna(subset=subset)

def fill_with_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Fill NaN in column with given value."""
    df = df.copy()
    df[column] = df[column].fillna(value)
    return df

def fill_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill NaN in column with column mean."""
    df = df.copy()
    df[column] = df[column].fillna(df[column].mean())
    return df
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestMissingValues(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10.0, 20.0, np.nan, 40.0, 50.0],
            'c': ['x', 'y', 'z', None, 'w']
        })

    def test_count_missing_basic(self):
        result = count_missing(self.df)
        self.assertEqual(result['a'], 2)
        self.assertEqual(result['b'], 1)
        self.assertEqual(result['c'], 1)

    def test_count_missing_no_missing(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = count_missing(df)
        self.assertEqual(result['x'], 0)

    def test_drop_missing_basic(self):
        result = drop_missing_rows(self.df, ['a'])
        self.assertEqual(len(result), 3)
        self.assertFalse(result['a'].isnull().any())

    def test_drop_missing_multiple(self):
        result = drop_missing_rows(self.df, ['a', 'b'])
        self.assertEqual(len(result), 2)

    def test_fill_with_value_basic(self):
        result = fill_with_value(self.df, 'a', 0)
        self.assertFalse(result['a'].isnull().any())
        self.assertEqual(result['a'].iloc[1], 0)

    def test_fill_with_value_no_modify(self):
        original = self.df.copy()
        _ = fill_with_value(self.df, 'a', 0)
        self.assertTrue(self.df['a'].isnull().any())

    def test_fill_with_mean_basic(self):
        result = fill_with_mean(self.df, 'a')
        self.assertFalse(result['a'].isnull().any())
        expected_mean = (1 + 3 + 5) / 3
        self.assertAlmostEqual(result['a'].iloc[1], expected_mean)

    def test_fill_with_mean_no_modify(self):
        original = self.df.copy()
        _ = fill_with_mean(self.df, 'b')
        self.assertTrue(self.df['b'].isnull().any())

    def test_count_missing_all_columns(self):
        result = count_missing(self.df)
        self.assertEqual(len(result), 3)

    def test_fill_preserves_non_missing(self):
        result = fill_with_value(self.df, 'a', 999)
        self.assertEqual(result['a'].iloc[0], 1.0)
        self.assertEqual(result['a'].iloc[2], 3.0)
`,

	hint1: 'Use df.isnull().sum() to count missing, df.dropna() to drop',
	hint2: 'Use df.fillna() for filling; use .copy() to avoid modifying original',

	whyItMatters: `Missing value handling is critical for:

- **Data quality**: NaN can break calculations and models
- **Imputation strategies**: Mean, median, mode, or advanced methods
- **Feature engineering**: Missing indicator columns
- **Model compatibility**: Many algorithms can't handle NaN

Proper handling improves model accuracy significantly.`,

	translations: {
		ru: {
			title: 'Обработка пропущенных значений',
			description: `# Обработка пропущенных значений

Реальные данные часто содержат пропущенные значения (NaN). Правильная обработка критична для качества данных.

## Задача

Реализуйте четыре функции:
1. \`count_missing(df)\` - Вернуть dict имён столбцов к количеству пропущенных значений
2. \`drop_missing_rows(df, subset)\` - Удалить строки с NaN в указанных столбцах
3. \`fill_with_value(df, column, value)\` - Заполнить NaN в столбце заданным значением
4. \`fill_with_mean(df, column)\` - Заполнить NaN в столбце средним значением столбца

## Пример

\`\`\`python
df = pd.DataFrame({
    'a': [1, None, 3, None],
    'b': [4, 5, None, 7]
})

count_missing(df)  # {'a': 2, 'b': 1}
\`\`\``,
			hint1: 'Используйте df.isnull().sum() для подсчёта, df.dropna() для удаления',
			hint2: 'Используйте df.fillna() для заполнения; используйте .copy() чтобы не изменять оригинал',
			whyItMatters: `Обработка пропущенных значений критична для:

- **Качество данных**: NaN могут сломать вычисления и модели
- **Стратегии импутации**: Среднее, медиана, мода или продвинутые методы
- **Feature engineering**: Столбцы-индикаторы пропущенных значений
- **Совместимость моделей**: Многие алгоритмы не могут обрабатывать NaN`,
		},
		uz: {
			title: "Yo'qolgan qiymatlarni boshqarish",
			description: `# Yo'qolgan qiymatlarni boshqarish

Haqiqiy ma'lumotlar ko'pincha yo'qolgan qiymatlarni (NaN) o'z ichiga oladi. Ularni to'g'ri boshqarish ma'lumotlar sifati uchun muhim.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`count_missing(df)\` - Ustun nomlari va yo'qolgan qiymatlar sonini dict qaytarish
2. \`drop_missing_rows(df, subset)\` - Ko'rsatilgan ustunlarda NaN bo'lgan qatorlarni o'chirish
3. \`fill_with_value(df, column, value)\` - Ustundagi NaN ni berilgan qiymat bilan to'ldirish
4. \`fill_with_mean(df, column)\` - Ustundagi NaN ni ustun o'rtachasi bilan to'ldirish

## Misol

\`\`\`python
df = pd.DataFrame({
    'a': [1, None, 3, None],
    'b': [4, 5, None, 7]
})

count_missing(df)  # {'a': 2, 'b': 1}
drop_missing_rows(df, ['a'])  # Keeps rows where 'a' is not NaN
fill_with_value(df, 'b', 0)  # Replace NaN in 'b' with 0
fill_with_mean(df, 'a')  # Replace NaN in 'a' with mean(a)
\`\`\``,
			hint1: "Hisoblash uchun df.isnull().sum(), o'chirish uchun df.dropna() dan foydalaning",
			hint2: "To'ldirish uchun df.fillna() dan foydalaning; asl nusxani o'zgartirmaslik uchun .copy() ishlating",
			whyItMatters: `Yo'qolgan qiymatlarni boshqarish quyidagilar uchun muhim:

- **Ma'lumotlar sifati**: NaN hisob-kitoblar va modellarni buzishi mumkin
- **Imputatsiya strategiyalari**: O'rtacha, mediana, moda yoki ilg'or usullar
- **Model mosligi**: Ko'p algoritmlar NaN ni qayta ishlay olmaydi`,
		},
	},
};

export default task;
