import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-custom-aggregation',
	title: 'Custom Aggregation',
	difficulty: 'hard',
	tags: ['pandas', 'aggregation', 'custom', 'advanced'],
	estimatedTime: '18m',
	isPremium: true,
	order: 6,
	description: `# Custom Aggregation

Sometimes built-in aggregations aren't enough. Learn to create custom aggregation functions.

## Task

Implement four functions:
1. \`agg_with_named(df, group_col, agg_dict)\` - Named aggregations with dict
2. \`agg_custom_func(df, group_col, value_col, func)\` - Apply custom function per group
3. \`agg_multiple_cols(df, group_col, agg_specs)\` - Different aggregations per column
4. \`agg_with_transform(df, group_col, value_col)\` - Return group stats aligned with original

## Example

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'count': [1, 2, 3, 4]
})

# Named aggregations
agg_dict = {'total': ('value', 'sum'), 'avg': ('value', 'mean')}
agg_with_named(df, 'category', agg_dict)

# Custom function: range (max - min)
agg_custom_func(df, 'category', 'value', lambda x: x.max() - x.min())

# Different aggregations per column
agg_specs = {'value': ['sum', 'mean'], 'count': 'sum'}
agg_multiple_cols(df, 'category', agg_specs)

# Transform: add group mean as new column
agg_with_transform(df, 'category', 'value')  # Adds 'value_group_mean'
\`\`\``,

	initialCode: `import pandas as pd

def agg_with_named(df: pd.DataFrame, group_col: str, agg_dict: dict) -> pd.DataFrame:
    """Named aggregations. agg_dict: {'new_name': ('column', 'agg_func')}"""
    # Your code here
    pass

def agg_custom_func(df: pd.DataFrame, group_col: str, value_col: str, func) -> pd.Series:
    """Apply custom function per group."""
    # Your code here
    pass

def agg_multiple_cols(df: pd.DataFrame, group_col: str, agg_specs: dict) -> pd.DataFrame:
    """Different aggregations per column. agg_specs: {'col': ['agg1', 'agg2']}"""
    # Your code here
    pass

def agg_with_transform(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return original df with group mean added as new column."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def agg_with_named(df: pd.DataFrame, group_col: str, agg_dict: dict) -> pd.DataFrame:
    """Named aggregations. agg_dict: {'new_name': ('column', 'agg_func')}"""
    return df.groupby(group_col).agg(**agg_dict)

def agg_custom_func(df: pd.DataFrame, group_col: str, value_col: str, func) -> pd.Series:
    """Apply custom function per group."""
    return df.groupby(group_col)[value_col].agg(func)

def agg_multiple_cols(df: pd.DataFrame, group_col: str, agg_specs: dict) -> pd.DataFrame:
    """Different aggregations per column. agg_specs: {'col': ['agg1', 'agg2']}"""
    return df.groupby(group_col).agg(agg_specs)

def agg_with_transform(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return original df with group mean added as new column."""
    df = df.copy()
    df[f'{value_col}_group_mean'] = df.groupby(group_col)[value_col].transform('mean')
    return df
`,

	testCode: `import pandas as pd
import unittest

class TestCustomAggregation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40],
            'count': [1, 2, 3, 4]
        })

    def test_agg_with_named_basic(self):
        agg_dict = {'total': ('value', 'sum'), 'avg': ('value', 'mean')}
        result = agg_with_named(self.df, 'category', agg_dict)
        self.assertIn('total', result.columns)
        self.assertIn('avg', result.columns)

    def test_agg_with_named_values(self):
        agg_dict = {'total': ('value', 'sum')}
        result = agg_with_named(self.df, 'category', agg_dict)
        self.assertEqual(result.loc['A', 'total'], 30)
        self.assertEqual(result.loc['B', 'total'], 70)

    def test_agg_custom_func_range(self):
        result = agg_custom_func(self.df, 'category', 'value', lambda x: x.max() - x.min())
        self.assertEqual(result['A'], 10)
        self.assertEqual(result['B'], 10)

    def test_agg_custom_func_sum_squares(self):
        result = agg_custom_func(self.df, 'category', 'value', lambda x: (x**2).sum())
        self.assertEqual(result['A'], 100 + 400)

    def test_agg_multiple_cols_basic(self):
        agg_specs = {'value': ['sum', 'mean'], 'count': 'sum'}
        result = agg_multiple_cols(self.df, 'category', agg_specs)
        self.assertIn('value', result.columns)
        self.assertIn('count', result.columns)

    def test_agg_multiple_cols_values(self):
        agg_specs = {'value': 'sum', 'count': 'sum'}
        result = agg_multiple_cols(self.df, 'category', agg_specs)
        self.assertEqual(result.loc['A', 'count'], 3)

    def test_agg_with_transform_column(self):
        result = agg_with_transform(self.df, 'category', 'value')
        self.assertIn('value_group_mean', result.columns)

    def test_agg_with_transform_values(self):
        result = agg_with_transform(self.df, 'category', 'value')
        # A's mean is 15, B's mean is 35
        self.assertAlmostEqual(result['value_group_mean'].iloc[0], 15.0)
        self.assertAlmostEqual(result['value_group_mean'].iloc[2], 35.0)

    def test_agg_with_transform_length(self):
        result = agg_with_transform(self.df, 'category', 'value')
        self.assertEqual(len(result), len(self.df))

    def test_agg_custom_func_type(self):
        result = agg_custom_func(self.df, 'category', 'value', 'sum')
        self.assertIsInstance(result, pd.Series)
`,

	hint1: 'Use df.groupby(col).agg(**{"name": ("col", "func")}) for named aggregations',
	hint2: 'Use .transform() to get same-length result aligned with original DataFrame',

	whyItMatters: `Custom aggregation enables:

- **Complex metrics**: Calculate domain-specific statistics
- **Feature engineering**: Create sophisticated aggregated features
- **Business logic**: Implement custom business rules
- **Performance optimization**: Combine multiple operations efficiently

This is what separates basic analysis from advanced data engineering.`,

	translations: {
		ru: {
			title: 'Пользовательская агрегация',
			description: `# Пользовательская агрегация

Иногда встроенных агрегаций недостаточно. Научитесь создавать пользовательские функции агрегации.

## Задача

Реализуйте четыре функции:
1. \`agg_with_named(df, group_col, agg_dict)\` - Именованные агрегации со словарём
2. \`agg_custom_func(df, group_col, value_col, func)\` - Применить пользовательскую функцию к группе
3. \`agg_multiple_cols(df, group_col, agg_specs)\` - Разные агрегации для каждого столбца
4. \`agg_with_transform(df, group_col, value_col)\` - Вернуть статистики группы выровненные с оригиналом

## Пример

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'count': [1, 2, 3, 4]
})

# Named aggregations
agg_dict = {'total': ('value', 'sum'), 'avg': ('value', 'mean')}
agg_with_named(df, 'category', agg_dict)

# Custom function: range (max - min)
agg_custom_func(df, 'category', 'value', lambda x: x.max() - x.min())

# Different aggregations per column
agg_specs = {'value': ['sum', 'mean'], 'count': 'sum'}
agg_multiple_cols(df, 'category', agg_specs)

# Transform: add group mean as new column
agg_with_transform(df, 'category', 'value')  # Adds 'value_group_mean'
\`\`\``,
			hint1: 'Используйте df.groupby(col).agg(**{"name": ("col", "func")}) для именованных агрегаций',
			hint2: 'Используйте .transform() для получения результата той же длины',
			whyItMatters: `Пользовательская агрегация позволяет:

- **Сложные метрики**: Вычисление domain-specific статистик
- **Feature engineering**: Создание сложных агрегированных признаков
- **Бизнес логика**: Реализация пользовательских бизнес правил
- **Оптимизация производительности**: Эффективное объединение операций`,
		},
		uz: {
			title: "Maxsus agregatsiya",
			description: `# Maxsus agregatsiya

Ba'zida o'rnatilgan agregatsiyalar yetarli emas. Maxsus agregatsiya funksiyalarini yaratishni o'rganing.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`agg_with_named(df, group_col, agg_dict)\` - Dict bilan nomlangan agregatsiyalar
2. \`agg_custom_func(df, group_col, value_col, func)\` - Guruhga maxsus funksiya qo'llash
3. \`agg_multiple_cols(df, group_col, agg_specs)\` - Har bir ustun uchun turli agregatsiyalar
4. \`agg_with_transform(df, group_col, value_col)\` - Asl bilan tekislangan guruh statistikasini qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40],
    'count': [1, 2, 3, 4]
})

# Named aggregations
agg_dict = {'total': ('value', 'sum'), 'avg': ('value', 'mean')}
agg_with_named(df, 'category', agg_dict)

# Custom function: range (max - min)
agg_custom_func(df, 'category', 'value', lambda x: x.max() - x.min())

# Different aggregations per column
agg_specs = {'value': ['sum', 'mean'], 'count': 'sum'}
agg_multiple_cols(df, 'category', agg_specs)

# Transform: add group mean as new column
agg_with_transform(df, 'category', 'value')  # Adds 'value_group_mean'
\`\`\``,
			hint1: "Nomlangan agregatsiyalar uchun df.groupby(col).agg(**{'name': ('col', 'func')}) dan foydalaning",
			hint2: "Bir xil uzunlikdagi natija olish uchun .transform() dan foydalaning",
			whyItMatters: `Maxsus agregatsiya quyidagilarni imkon beradi:

- **Murakkab metrikalar**: Domain-specific statistikalarni hisoblash
- **Feature engineering**: Murakkab agregatsiyalangan xususiyatlarni yaratish
- **Biznes mantiqi**: Maxsus biznes qoidalarini amalga oshirish`,
		},
	},
};

export default task;
