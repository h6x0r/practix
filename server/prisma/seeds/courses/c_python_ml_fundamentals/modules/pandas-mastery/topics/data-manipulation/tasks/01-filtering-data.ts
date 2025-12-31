import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-filtering-data',
	title: 'Filtering Data',
	difficulty: 'easy',
	tags: ['pandas', 'filtering', 'boolean'],
	estimatedTime: '12m',
	isPremium: false,
	order: 1,
	description: `# Filtering Data

Filtering allows you to select rows based on conditions - essential for data analysis.

## Task

Implement four functions:
1. \`filter_by_value(df, column, value)\` - Filter rows where column equals value
2. \`filter_by_range(df, column, min_val, max_val)\` - Filter rows where column is in range
3. \`filter_by_multiple(df, conditions)\` - Filter by multiple column conditions
4. \`filter_by_isin(df, column, values)\` - Filter where column value is in list

## Example

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'NY']
})

filter_by_value(df, 'city', 'NY')  # Alice, Charlie
filter_by_range(df, 'age', 26, 34)  # Bob
filter_by_multiple(df, {'city': 'NY', 'age': 25})  # Alice
filter_by_isin(df, 'city', ['NY', 'SF'])  # Alice, Charlie
\`\`\``,

	initialCode: `import pandas as pd

def filter_by_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Filter rows where column equals value."""
    # Your code here
    pass

def filter_by_range(df: pd.DataFrame, column: str, min_val, max_val) -> pd.DataFrame:
    """Filter rows where column is in range [min_val, max_val]."""
    # Your code here
    pass

def filter_by_multiple(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Filter by multiple column conditions (AND logic)."""
    # Your code here
    pass

def filter_by_isin(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Filter where column value is in list."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def filter_by_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Filter rows where column equals value."""
    return df[df[column] == value]

def filter_by_range(df: pd.DataFrame, column: str, min_val, max_val) -> pd.DataFrame:
    """Filter rows where column is in range [min_val, max_val]."""
    return df[(df[column] >= min_val) & (df[column] <= max_val)]

def filter_by_multiple(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Filter by multiple column conditions (AND logic)."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in conditions.items():
        mask = mask & (df[col] == val)
    return df[mask]

def filter_by_isin(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Filter where column value is in list."""
    return df[df[column].isin(values)]
`,

	testCode: `import pandas as pd
import unittest

class TestFilteringData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 28],
            'city': ['NY', 'LA', 'NY', 'SF']
        })

    def test_filter_by_value_basic(self):
        result = filter_by_value(self.df, 'city', 'NY')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['city'] == 'NY'))

    def test_filter_by_value_single(self):
        result = filter_by_value(self.df, 'name', 'Bob')
        self.assertEqual(len(result), 1)

    def test_filter_by_value_none(self):
        result = filter_by_value(self.df, 'city', 'Chicago')
        self.assertEqual(len(result), 0)

    def test_filter_by_range_basic(self):
        result = filter_by_range(self.df, 'age', 26, 32)
        self.assertEqual(len(result), 2)  # Bob, Diana

    def test_filter_by_range_inclusive(self):
        result = filter_by_range(self.df, 'age', 25, 25)
        self.assertEqual(len(result), 1)

    def test_filter_by_multiple_basic(self):
        result = filter_by_multiple(self.df, {'city': 'NY', 'age': 25})
        self.assertEqual(len(result), 1)
        self.assertEqual(result['name'].iloc[0], 'Alice')

    def test_filter_by_multiple_no_match(self):
        result = filter_by_multiple(self.df, {'city': 'NY', 'age': 30})
        self.assertEqual(len(result), 0)

    def test_filter_by_isin_basic(self):
        result = filter_by_isin(self.df, 'city', ['NY', 'SF'])
        self.assertEqual(len(result), 3)

    def test_filter_by_isin_single(self):
        result = filter_by_isin(self.df, 'city', ['LA'])
        self.assertEqual(len(result), 1)

    def test_filter_preserves_columns(self):
        result = filter_by_value(self.df, 'city', 'NY')
        self.assertEqual(list(result.columns), ['name', 'age', 'city'])
`,

	hint1: 'Use df[df["column"] == value] for equality filtering',
	hint2: 'Combine conditions with & operator: (cond1) & (cond2)',

	whyItMatters: `Data filtering is essential for:

- **Data exploration**: Focus on subsets of interest
- **Data cleaning**: Remove outliers or invalid rows
- **Feature engineering**: Create filtered aggregations
- **Model training**: Select training samples by criteria

This is one of the most frequent data manipulation operations.`,

	translations: {
		ru: {
			title: 'Фильтрация данных',
			description: `# Фильтрация данных

Фильтрация позволяет выбирать строки на основе условий — необходимо для анализа данных.

## Задача

Реализуйте четыре функции:
1. \`filter_by_value(df, column, value)\` - Фильтр строк где столбец равен значению
2. \`filter_by_range(df, column, min_val, max_val)\` - Фильтр строк где столбец в диапазоне
3. \`filter_by_multiple(df, conditions)\` - Фильтр по нескольким условиям столбцов
4. \`filter_by_isin(df, column, values)\` - Фильтр где значение столбца в списке

## Пример

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'NY']
})

filter_by_value(df, 'city', 'NY')  # Alice, Charlie
filter_by_range(df, 'age', 26, 34)  # Bob
filter_by_multiple(df, {'city': 'NY', 'age': 25})  # Alice
filter_by_isin(df, 'city', ['NY', 'SF'])  # Alice, Charlie
\`\`\``,
			hint1: 'Используйте df[df["column"] == value] для фильтрации по равенству',
			hint2: 'Комбинируйте условия оператором &: (cond1) & (cond2)',
			whyItMatters: `Фильтрация данных необходима для:

- **Исследование данных**: Фокус на интересующих подмножествах
- **Очистка данных**: Удаление выбросов или невалидных строк
- **Feature engineering**: Создание фильтрованных агрегаций
- **Обучение модели**: Выбор обучающих сэмплов по критериям`,
		},
		uz: {
			title: "Ma'lumotlarni filtrlash",
			description: `# Ma'lumotlarni filtrlash

Filtrlash shartlar asosida qatorlarni tanlash imkonini beradi - ma'lumotlar tahlili uchun zarur.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`filter_by_value(df, column, value)\` - Ustun qiymatga teng bo'lgan qatorlarni filtrlash
2. \`filter_by_range(df, column, min_val, max_val)\` - Ustun diapazon ichida bo'lgan qatorlarni filtrlash
3. \`filter_by_multiple(df, conditions)\` - Bir nechta ustun shartlari bo'yicha filtrlash
4. \`filter_by_isin(df, column, values)\` - Ustun qiymati ro'yxatda bo'lganlarni filtrlash

## Misol

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'NY']
})

filter_by_value(df, 'city', 'NY')  # Alice, Charlie
filter_by_range(df, 'age', 26, 34)  # Bob
filter_by_multiple(df, {'city': 'NY', 'age': 25})  # Alice
filter_by_isin(df, 'city', ['NY', 'SF'])  # Alice, Charlie
\`\`\``,
			hint1: "Tenglik bo'yicha filtrlash uchun df[df['column'] == value] dan foydalaning",
			hint2: "Shartlarni & operatori bilan birlashtiring: (cond1) & (cond2)",
			whyItMatters: `Ma'lumotlarni filtrlash quyidagilar uchun zarur:

- **Ma'lumotlarni tadqiq qilish**: Qiziqarli quyi to'plamlarga e'tibor
- **Ma'lumotlarni tozalash**: Outlierlar yoki noto'g'ri qatorlarni olib tashlash
- **Model o'qitish**: Mezonlar bo'yicha o'qitish namunalarini tanlash`,
		},
	},
};

export default task;
