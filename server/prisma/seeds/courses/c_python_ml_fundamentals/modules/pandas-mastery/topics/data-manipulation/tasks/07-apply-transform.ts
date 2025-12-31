import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-apply-transform',
	title: 'Apply and Transform',
	difficulty: 'medium',
	tags: ['pandas', 'apply', 'transform', 'lambda'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,
	description: `# Apply and Transform

Apply allows custom functions on DataFrame elements, rows, or columns. Transform returns same-shape results.

## Task

Implement four functions:
1. \`apply_to_column(df, column, func)\` - Apply function to each value in column
2. \`apply_to_rows(df, func)\` - Apply function to each row
3. \`apply_to_df(df, func)\` - Apply function to entire DataFrame (element-wise)
4. \`transform_normalize(df, columns)\` - Normalize columns to 0-1 range

## Example

\`\`\`python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

apply_to_column(df, 'a', lambda x: x ** 2)  # a: [1, 4, 9]
apply_to_rows(df, lambda row: row['a'] + row['b'])  # [11, 22, 33]
apply_to_df(df, lambda x: x * 2)  # All values doubled
transform_normalize(df, ['a', 'b'])  # Scale to [0, 1]
\`\`\``,

	initialCode: `import pandas as pd

def apply_to_column(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
    """Apply function to each value in column."""
    # Your code here
    pass

def apply_to_rows(df: pd.DataFrame, func) -> pd.Series:
    """Apply function to each row, return Series."""
    # Your code here
    pass

def apply_to_df(df: pd.DataFrame, func) -> pd.DataFrame:
    """Apply function element-wise to entire DataFrame."""
    # Your code here
    pass

def transform_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns to 0-1 range."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def apply_to_column(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
    """Apply function to each value in column."""
    df = df.copy()
    df[column] = df[column].apply(func)
    return df

def apply_to_rows(df: pd.DataFrame, func) -> pd.Series:
    """Apply function to each row, return Series."""
    return df.apply(func, axis=1)

def apply_to_df(df: pd.DataFrame, func) -> pd.DataFrame:
    """Apply function element-wise to entire DataFrame."""
    return df.applymap(func)

def transform_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns to 0-1 range."""
    df = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    return df
`,

	testCode: `import pandas as pd
import unittest

class TestApplyTransform(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

    def test_apply_to_column_square(self):
        result = apply_to_column(self.df, 'a', lambda x: x ** 2)
        self.assertEqual(result['a'].tolist(), [1, 4, 9, 16])

    def test_apply_to_column_preserves_other(self):
        result = apply_to_column(self.df, 'a', lambda x: x * 2)
        self.assertEqual(result['b'].tolist(), [10, 20, 30, 40])

    def test_apply_to_rows_sum(self):
        result = apply_to_rows(self.df, lambda row: row['a'] + row['b'])
        self.assertEqual(result.tolist(), [11, 22, 33, 44])

    def test_apply_to_rows_max(self):
        result = apply_to_rows(self.df, lambda row: max(row))
        self.assertEqual(result.tolist(), [10, 20, 30, 40])

    def test_apply_to_df_double(self):
        result = apply_to_df(self.df, lambda x: x * 2)
        self.assertEqual(result['a'].tolist(), [2, 4, 6, 8])
        self.assertEqual(result['b'].tolist(), [20, 40, 60, 80])

    def test_apply_to_df_string(self):
        df = pd.DataFrame({'x': ['a', 'b'], 'y': ['c', 'd']})
        result = apply_to_df(df, str.upper)
        self.assertEqual(result['x'].tolist(), ['A', 'B'])

    def test_transform_normalize_range(self):
        result = transform_normalize(self.df, ['a', 'b'])
        self.assertAlmostEqual(result['a'].min(), 0)
        self.assertAlmostEqual(result['a'].max(), 1)

    def test_transform_normalize_values(self):
        result = transform_normalize(self.df, ['a'])
        self.assertAlmostEqual(result['a'].iloc[0], 0)
        self.assertAlmostEqual(result['a'].iloc[-1], 1)

    def test_transform_normalize_preserves_other(self):
        result = transform_normalize(self.df, ['a'])
        self.assertEqual(result['b'].tolist(), [10, 20, 30, 40])

    def test_no_modify_original(self):
        _ = apply_to_column(self.df, 'a', lambda x: x * 100)
        self.assertEqual(self.df['a'].iloc[0], 1)
`,

	hint1: 'Use df["col"].apply(func) for column, df.apply(func, axis=1) for rows',
	hint2: 'Normalize: (x - min) / (max - min)',

	whyItMatters: `Apply and transform are essential for:

- **Feature engineering**: Create complex derived features
- **Custom transformations**: Apply domain-specific logic
- **Data normalization**: Scale features for ML models
- **Vectorized operations**: Apply functions efficiently

These are the Swiss Army knives of Pandas data manipulation.`,

	translations: {
		ru: {
			title: 'Apply и Transform',
			description: `# Apply и Transform

Apply позволяет применять пользовательские функции к элементам DataFrame, строкам или столбцам. Transform возвращает результат той же формы.

## Задача

Реализуйте четыре функции:
1. \`apply_to_column(df, column, func)\` - Применить функцию к каждому значению в столбце
2. \`apply_to_rows(df, func)\` - Применить функцию к каждой строке
3. \`apply_to_df(df, func)\` - Применить функцию ко всему DataFrame (поэлементно)
4. \`transform_normalize(df, columns)\` - Нормализовать столбцы к диапазону 0-1

## Пример

\`\`\`python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

apply_to_column(df, 'a', lambda x: x ** 2)  # a: [1, 4, 9]
apply_to_rows(df, lambda row: row['a'] + row['b'])  # [11, 22, 33]
apply_to_df(df, lambda x: x * 2)  # All values doubled
transform_normalize(df, ['a', 'b'])  # Scale to [0, 1]
\`\`\``,
			hint1: 'Используйте df["col"].apply(func) для столбца, df.apply(func, axis=1) для строк',
			hint2: 'Нормализация: (x - min) / (max - min)',
			whyItMatters: `Apply и transform необходимы для:

- **Feature engineering**: Создание сложных производных признаков
- **Пользовательские преобразования**: Применение domain-specific логики
- **Нормализация данных**: Масштабирование признаков для ML моделей
- **Векторизованные операции**: Эффективное применение функций`,
		},
		uz: {
			title: "Apply va Transform",
			description: `# Apply va Transform

Apply DataFrame elementlariga, qatorlariga yoki ustunlariga maxsus funksiyalarni qo'llash imkonini beradi. Transform bir xil shakldagi natijalarni qaytaradi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`apply_to_column(df, column, func)\` - Ustundagi har bir qiymatga funksiya qo'llash
2. \`apply_to_rows(df, func)\` - Har bir qatorga funksiya qo'llash
3. \`apply_to_df(df, func)\` - Butun DataFrame ga funksiya qo'llash (elementli)
4. \`transform_normalize(df, columns)\` - Ustunlarni 0-1 diapazoniga normalizatsiya qilish

## Misol

\`\`\`python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

apply_to_column(df, 'a', lambda x: x ** 2)  # a: [1, 4, 9]
apply_to_rows(df, lambda row: row['a'] + row['b'])  # [11, 22, 33]
apply_to_df(df, lambda x: x * 2)  # All values doubled
transform_normalize(df, ['a', 'b'])  # Scale to [0, 1]
\`\`\``,
			hint1: "Ustun uchun df['col'].apply(func), qatorlar uchun df.apply(func, axis=1) dan foydalaning",
			hint2: "Normalizatsiya: (x - min) / (max - min)",
			whyItMatters: `Apply va transform quyidagilar uchun zarur:

- **Feature engineering**: Murakkab hosil xususiyatlarni yaratish
- **Maxsus transformatsiyalar**: Domain-specific mantiqni qo'llash
- **Ma'lumotlarni normalizatsiya qilish**: ML modellari uchun xususiyatlarni masshtablash`,
		},
	},
};

export default task;
