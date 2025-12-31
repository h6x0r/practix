import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-groupby-basics',
	title: 'GroupBy Basics',
	difficulty: 'medium',
	tags: ['pandas', 'groupby', 'aggregation'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# GroupBy Basics

GroupBy splits data into groups, applies a function to each group, and combines results - the "split-apply-combine" pattern.

## Task

Implement four functions:
1. \`group_and_sum(df, group_col, value_col)\` - Sum values by group
2. \`group_and_mean(df, group_col, value_col)\` - Mean values by group
3. \`group_and_count(df, group_col)\` - Count rows per group
4. \`group_multiple_agg(df, group_col, value_col)\` - Return sum, mean, min, max per group

## Example

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

group_and_sum(df, 'category', 'value')  # A: 30, B: 120
group_and_mean(df, 'category', 'value')  # A: 15, B: 40
group_and_count(df, 'category')  # A: 2, B: 3
group_multiple_agg(df, 'category', 'value')  # DataFrame with sum, mean, min, max
\`\`\``,

	initialCode: `import pandas as pd

def group_and_sum(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Sum values by group."""
    # Your code here
    pass

def group_and_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Mean values by group."""
    # Your code here
    pass

def group_and_count(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Count rows per group."""
    # Your code here
    pass

def group_multiple_agg(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return DataFrame with sum, mean, min, max per group."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def group_and_sum(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Sum values by group."""
    return df.groupby(group_col)[value_col].sum()

def group_and_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Mean values by group."""
    return df.groupby(group_col)[value_col].mean()

def group_and_count(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Count rows per group."""
    return df.groupby(group_col).size()

def group_multiple_agg(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return DataFrame with sum, mean, min, max per group."""
    return df.groupby(group_col)[value_col].agg(['sum', 'mean', 'min', 'max'])
`,

	testCode: `import pandas as pd
import unittest

class TestGroupByBasics(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

    def test_group_and_sum_basic(self):
        result = group_and_sum(self.df, 'category', 'value')
        self.assertEqual(result['A'], 30)
        self.assertEqual(result['B'], 120)

    def test_group_and_sum_type(self):
        result = group_and_sum(self.df, 'category', 'value')
        self.assertIsInstance(result, pd.Series)

    def test_group_and_mean_basic(self):
        result = group_and_mean(self.df, 'category', 'value')
        self.assertAlmostEqual(result['A'], 15.0)
        self.assertAlmostEqual(result['B'], 40.0)

    def test_group_and_count_basic(self):
        result = group_and_count(self.df, 'category')
        self.assertEqual(result['A'], 2)
        self.assertEqual(result['B'], 3)

    def test_group_multiple_agg_columns(self):
        result = group_multiple_agg(self.df, 'category', 'value')
        self.assertIn('sum', result.columns)
        self.assertIn('mean', result.columns)
        self.assertIn('min', result.columns)
        self.assertIn('max', result.columns)

    def test_group_multiple_agg_values(self):
        result = group_multiple_agg(self.df, 'category', 'value')
        self.assertEqual(result.loc['A', 'sum'], 30)
        self.assertEqual(result.loc['B', 'min'], 30)
        self.assertEqual(result.loc['B', 'max'], 50)

    def test_single_group(self):
        df = pd.DataFrame({'cat': ['X', 'X'], 'val': [5, 10]})
        result = group_and_sum(df, 'cat', 'val')
        self.assertEqual(result['X'], 15)

    def test_many_groups(self):
        df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D'], 'val': [1, 2, 3, 4]})
        result = group_and_count(df, 'cat')
        self.assertEqual(len(result), 4)

    def test_group_preserves_index(self):
        result = group_and_sum(self.df, 'category', 'value')
        self.assertTrue('A' in result.index)
        self.assertTrue('B' in result.index)

    def test_group_multiple_agg_shape(self):
        result = group_multiple_agg(self.df, 'category', 'value')
        self.assertEqual(result.shape, (2, 4))
`,

	hint1: 'Use df.groupby(column)[value].sum() for basic aggregation',
	hint2: 'Use .agg(["sum", "mean", "min", "max"]) for multiple aggregations',

	whyItMatters: `GroupBy is fundamental for:

- **Data analysis**: Calculate statistics per category
- **Feature engineering**: Create aggregated features
- **Business metrics**: Revenue by region, users by country
- **Time series**: Daily/weekly/monthly aggregations

This is the most important Pandas operation for data analysis.`,

	translations: {
		ru: {
			title: 'Основы GroupBy',
			description: `# Основы GroupBy

GroupBy разделяет данные на группы, применяет функцию к каждой группе и объединяет результаты — паттерн "split-apply-combine".

## Задача

Реализуйте четыре функции:
1. \`group_and_sum(df, group_col, value_col)\` - Сумма значений по группам
2. \`group_and_mean(df, group_col, value_col)\` - Среднее значений по группам
3. \`group_and_count(df, group_col)\` - Количество строк в группе
4. \`group_multiple_agg(df, group_col, value_col)\` - Вернуть sum, mean, min, max по группам

## Пример

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

group_and_sum(df, 'category', 'value')  # A: 30, B: 120
group_and_mean(df, 'category', 'value')  # A: 15, B: 40
group_and_count(df, 'category')  # A: 2, B: 3
group_multiple_agg(df, 'category', 'value')  # DataFrame with sum, mean, min, max
\`\`\``,
			hint1: 'Используйте df.groupby(column)[value].sum() для базовой агрегации',
			hint2: 'Используйте .agg(["sum", "mean", "min", "max"]) для нескольких агрегаций',
			whyItMatters: `GroupBy фундаментален для:

- **Анализ данных**: Вычисление статистик по категориям
- **Feature engineering**: Создание агрегированных признаков
- **Бизнес метрики**: Доход по регионам, пользователи по странам
- **Временные ряды**: Дневные/недельные/месячные агрегации`,
		},
		uz: {
			title: "GroupBy asoslari",
			description: `# GroupBy asoslari

GroupBy ma'lumotlarni guruhlarga ajratadi, har bir guruhga funksiya qo'llaydi va natijalarni birlashtiradi - "split-apply-combine" patterni.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`group_and_sum(df, group_col, value_col)\` - Guruh bo'yicha qiymatlar yig'indisi
2. \`group_and_mean(df, group_col, value_col)\` - Guruh bo'yicha qiymatlar o'rtachasi
3. \`group_and_count(df, group_col)\` - Guruhdagi qatorlar soni
4. \`group_multiple_agg(df, group_col, value_col)\` - Guruh bo'yicha sum, mean, min, max qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

group_and_sum(df, 'category', 'value')  # A: 30, B: 120
group_and_mean(df, 'category', 'value')  # A: 15, B: 40
group_and_count(df, 'category')  # A: 2, B: 3
group_multiple_agg(df, 'category', 'value')  # DataFrame with sum, mean, min, max
\`\`\``,
			hint1: "Asosiy agregatsiya uchun df.groupby(column)[value].sum() dan foydalaning",
			hint2: "Bir nechta agregatsiya uchun .agg(['sum', 'mean', 'min', 'max']) dan foydalaning",
			whyItMatters: `GroupBy quyidagilar uchun asosiydir:

- **Ma'lumotlar tahlili**: Kategoriya bo'yicha statistikalarni hisoblash
- **Feature engineering**: Agregatsiyalangan xususiyatlarni yaratish
- **Biznes metrikalari**: Mintaqa bo'yicha daromad, mamlakat bo'yicha foydalanuvchilar`,
		},
	},
};

export default task;
