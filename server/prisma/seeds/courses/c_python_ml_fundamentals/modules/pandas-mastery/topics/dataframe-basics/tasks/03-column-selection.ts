import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-column-selection',
	title: 'Selecting Columns',
	difficulty: 'easy',
	tags: ['pandas', 'selection', 'columns'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,
	description: `# Selecting Columns

Selecting specific columns is one of the most common DataFrame operations.

## Task

Implement three functions:
1. \`select_single_column(df, col_name)\` - Return a single column as Series
2. \`select_multiple_columns(df, col_names)\` - Return DataFrame with selected columns
3. \`select_by_dtype(df, dtype)\` - Return DataFrame with columns of specified dtype

## Example

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'salary': [50000.0, 60000.0]
})

select_single_column(df, 'name')  # Series: ['Alice', 'Bob']
select_multiple_columns(df, ['name', 'age'])  # DataFrame with 2 columns
select_by_dtype(df, 'int64')  # DataFrame with only 'age' column
\`\`\``,

	initialCode: `import pandas as pd

def select_single_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a single column as Series."""
    # Your code here
    pass

def select_multiple_columns(df: pd.DataFrame, col_names: list) -> pd.DataFrame:
    """Return DataFrame with selected columns."""
    # Your code here
    pass

def select_by_dtype(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """Return DataFrame with columns of specified dtype."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def select_single_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a single column as Series."""
    return df[col_name]

def select_multiple_columns(df: pd.DataFrame, col_names: list) -> pd.DataFrame:
    """Return DataFrame with selected columns."""
    return df[col_names]

def select_by_dtype(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """Return DataFrame with columns of specified dtype."""
    return df.select_dtypes(include=[dtype])
`,

	testCode: `import pandas as pd
import unittest

class TestColumnSelection(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0],
            'active': [True, False, True]
        })

    def test_select_single_basic(self):
        result = select_single_column(self.df, 'name')
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result), ['Alice', 'Bob', 'Charlie'])

    def test_select_single_numeric(self):
        result = select_single_column(self.df, 'age')
        self.assertEqual(result.tolist(), [25, 30, 35])

    def test_select_single_length(self):
        result = select_single_column(self.df, 'salary')
        self.assertEqual(len(result), 3)

    def test_select_multiple_basic(self):
        result = select_multiple_columns(self.df, ['name', 'age'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['name', 'age'])

    def test_select_multiple_shape(self):
        result = select_multiple_columns(self.df, ['name', 'salary'])
        self.assertEqual(result.shape, (3, 2))

    def test_select_multiple_order(self):
        result = select_multiple_columns(self.df, ['salary', 'name'])
        self.assertEqual(list(result.columns), ['salary', 'name'])

    def test_select_by_dtype_int(self):
        result = select_by_dtype(self.df, 'int64')
        self.assertIn('age', result.columns)

    def test_select_by_dtype_float(self):
        result = select_by_dtype(self.df, 'float64')
        self.assertIn('salary', result.columns)

    def test_select_by_dtype_bool(self):
        result = select_by_dtype(self.df, 'bool')
        self.assertIn('active', result.columns)

    def test_select_by_dtype_shape(self):
        result = select_by_dtype(self.df, 'object')
        self.assertEqual(result.shape[0], 3)
`,

	hint1: 'Use df["column"] for single column, df[["col1", "col2"]] for multiple',
	hint2: 'Use df.select_dtypes(include=[dtype]) for dtype-based selection',

	whyItMatters: `Column selection is essential for:

- **Feature selection**: Choose relevant features for models
- **Data preprocessing**: Process different column types separately
- **Memory efficiency**: Work with subset of large datasets
- **Data exploration**: Focus on specific aspects of data

This is one of the most frequent operations in data analysis.`,

	translations: {
		ru: {
			title: 'Выбор столбцов',
			description: `# Выбор столбцов

Выбор конкретных столбцов — одна из самых частых операций с DataFrame.

## Задача

Реализуйте три функции:
1. \`select_single_column(df, col_name)\` - Вернуть один столбец как Series
2. \`select_multiple_columns(df, col_names)\` - Вернуть DataFrame с выбранными столбцами
3. \`select_by_dtype(df, dtype)\` - Вернуть DataFrame со столбцами указанного типа

## Пример

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'salary': [50000.0, 60000.0]
})

select_single_column(df, 'name')  # Series: ['Alice', 'Bob']
select_multiple_columns(df, ['name', 'age'])  # DataFrame с 2 столбцами
\`\`\``,
			hint1: 'Используйте df["column"] для одного столбца, df[["col1", "col2"]] для нескольких',
			hint2: 'Используйте df.select_dtypes(include=[dtype]) для выбора по типу',
			whyItMatters: `Выбор столбцов необходим для:

- **Отбор признаков**: Выбор релевантных признаков для моделей
- **Предобработка данных**: Раздельная обработка столбцов разных типов
- **Эффективность памяти**: Работа с подмножеством больших датасетов
- **Исследование данных**: Фокус на конкретных аспектах данных`,
		},
		uz: {
			title: "Ustunlarni tanlash",
			description: `# Ustunlarni tanlash

Ma'lum ustunlarni tanlash DataFrame ning eng ko'p ishlatiladigan operatsiyalaridan biri.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`select_single_column(df, col_name)\` - Bitta ustunni Series sifatida qaytarish
2. \`select_multiple_columns(df, col_names)\` - Tanlangan ustunlar bilan DataFrame qaytarish
3. \`select_by_dtype(df, dtype)\` - Ko'rsatilgan dtype ustunlari bilan DataFrame qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'salary': [50000.0, 60000.0]
})

select_single_column(df, 'name')  # Series: ['Alice', 'Bob']
select_multiple_columns(df, ['name', 'age'])  # DataFrame with 2 columns
select_by_dtype(df, 'int64')  # DataFrame with only 'age' column
\`\`\``,
			hint1: "Bitta ustun uchun df['column'], bir nechta uchun df[['col1', 'col2']] dan foydalaning",
			hint2: "dtype bo'yicha tanlash uchun df.select_dtypes(include=[dtype]) dan foydalaning",
			whyItMatters: `Ustunlarni tanlash quyidagilar uchun zarur:

- **Xususiyatlarni tanlash**: Modellar uchun tegishli xususiyatlarni tanlash
- **Ma'lumotlarni oldindan qayta ishlash**: Turli ustun turlarini alohida qayta ishlash
- **Xotira samaradorligi**: Katta datasetlarning quyi to'plami bilan ishlash`,
		},
	},
};

export default task;
