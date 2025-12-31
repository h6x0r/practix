import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-multi-index',
	title: 'Multi-level Indexing',
	difficulty: 'hard',
	tags: ['pandas', 'multiindex', 'hierarchical'],
	estimatedTime: '18m',
	isPremium: false,
	order: 4,
	description: `# Multi-level Indexing

MultiIndex allows hierarchical indexing for complex data structures and efficient grouping.

## Task

Implement four functions:
1. \`create_multi_index(df, cols)\` - Set multiple columns as hierarchical index
2. \`select_level(df, level, value)\` - Select rows by value at specific level
3. \`reset_index_level(df, level)\` - Reset specific level back to column
4. \`group_by_level(df, level)\` - Group and sum by index level

## Example

\`\`\`python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'sales': [100, 150, 120, 180]
})

mi_df = create_multi_index(df, ['year', 'quarter'])
#                 sales
# year quarter
# 2022 Q1         100
#      Q2         150
# 2023 Q1         120
#      Q2         180

select_level(mi_df, 'year', 2023)  # Q1: 120, Q2: 180
\`\`\``,

	initialCode: `import pandas as pd

def create_multi_index(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Set multiple columns as hierarchical index."""
    # Your code here
    pass

def select_level(df: pd.DataFrame, level: str, value) -> pd.DataFrame:
    """Select rows by value at specific level."""
    # Your code here
    pass

def reset_index_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Reset specific level back to column."""
    # Your code here
    pass

def group_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Group and sum by index level."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def create_multi_index(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Set multiple columns as hierarchical index."""
    return df.set_index(cols)

def select_level(df: pd.DataFrame, level: str, value) -> pd.DataFrame:
    """Select rows by value at specific level."""
    return df.xs(value, level=level)

def reset_index_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Reset specific level back to column."""
    return df.reset_index(level=level)

def group_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Group and sum by index level."""
    return df.groupby(level=level).sum()
`,

	testCode: `import pandas as pd
import unittest

class TestMultiIndex(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'year': [2022, 2022, 2023, 2023],
            'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
            'sales': [100, 150, 120, 180]
        })

    def test_create_multi_index_basic(self):
        result = create_multi_index(self.df, ['year', 'quarter'])
        self.assertTrue(isinstance(result.index, pd.MultiIndex))

    def test_create_multi_index_levels(self):
        result = create_multi_index(self.df, ['year', 'quarter'])
        self.assertEqual(result.index.names, ['year', 'quarter'])

    def test_create_multi_index_values(self):
        result = create_multi_index(self.df, ['year', 'quarter'])
        self.assertEqual(result.loc[(2022, 'Q1'), 'sales'], 100)

    def test_select_level_basic(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = select_level(mi_df, 'year', 2023)
        self.assertEqual(len(result), 2)

    def test_select_level_values(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = select_level(mi_df, 'year', 2022)
        self.assertEqual(result.loc['Q1', 'sales'], 100)

    def test_reset_index_level_basic(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = reset_index_level(mi_df, 'year')
        self.assertIn('year', result.columns)

    def test_reset_index_level_keeps_other(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = reset_index_level(mi_df, 'year')
        self.assertEqual(result.index.name, 'quarter')

    def test_group_by_level_basic(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = group_by_level(mi_df, 'year')
        self.assertEqual(result.loc[2022, 'sales'], 250)
        self.assertEqual(result.loc[2023, 'sales'], 300)

    def test_group_by_level_shape(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = group_by_level(mi_df, 'year')
        self.assertEqual(len(result), 2)

    def test_select_other_level(self):
        mi_df = create_multi_index(self.df, ['year', 'quarter'])
        result = select_level(mi_df, 'quarter', 'Q1')
        self.assertEqual(len(result), 2)
`,

	hint1: 'Use df.set_index(cols) to create MultiIndex, df.xs(value, level=level) to select',
	hint2: 'Use df.reset_index(level=level) to convert level back to column',

	whyItMatters: `MultiIndex is powerful for:

- **Hierarchical data**: Geographic (country/city), temporal (year/month)
- **Efficient selection**: Fast access to subgroups
- **Multi-dimensional analysis**: Panel data, time series by entity
- **Memory efficiency**: More compact than repeated columns

Essential for complex datasets with natural hierarchies.`,

	translations: {
		ru: {
			title: 'Многоуровневая индексация',
			description: `# Многоуровневая индексация

MultiIndex позволяет иерархическую индексацию для сложных структур данных и эффективной группировки.

## Задача

Реализуйте четыре функции:
1. \`create_multi_index(df, cols)\` - Установить несколько столбцов как иерархический индекс
2. \`select_level(df, level, value)\` - Выбрать строки по значению на конкретном уровне
3. \`reset_index_level(df, level)\` - Сбросить конкретный уровень обратно в столбец
4. \`group_by_level(df, level)\` - Группировать и суммировать по уровню индекса

## Пример

\`\`\`python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'sales': [100, 150, 120, 180]
})

mi_df = create_multi_index(df, ['year', 'quarter'])
#                 sales
# year quarter
# 2022 Q1         100
#      Q2         150
# 2023 Q1         120
#      Q2         180

select_level(mi_df, 'year', 2023)  # Q1: 120, Q2: 180
\`\`\``,
			hint1: 'Используйте df.set_index(cols) для создания MultiIndex, df.xs(value, level=level) для выбора',
			hint2: 'Используйте df.reset_index(level=level) для конвертации уровня обратно в столбец',
			whyItMatters: `MultiIndex мощный для:

- **Иерархические данные**: Географические (страна/город), временные (год/месяц)
- **Эффективная выборка**: Быстрый доступ к подгруппам
- **Многомерный анализ**: Панельные данные, временные ряды по сущностям
- **Эффективность памяти**: Компактнее чем повторяющиеся столбцы`,
		},
		uz: {
			title: "Ko'p darajali indekslash",
			description: `# Ko'p darajali indekslash

MultiIndex murakkab ma'lumotlar strukturalari va samarali guruhlash uchun ierarxik indekslash imkonini beradi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`create_multi_index(df, cols)\` - Bir nechta ustunlarni ierarxik indeks sifatida o'rnatish
2. \`select_level(df, level, value)\` - Ma'lum darajadagi qiymat bo'yicha qatorlarni tanlash
3. \`reset_index_level(df, level)\` - Ma'lum darajani ustun holatiga qaytarish
4. \`group_by_level(df, level)\` - Indeks darajasi bo'yicha guruhlash va jamlash

## Misol

\`\`\`python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'sales': [100, 150, 120, 180]
})

mi_df = create_multi_index(df, ['year', 'quarter'])
#                 sales
# year quarter
# 2022 Q1         100
#      Q2         150
# 2023 Q1         120
#      Q2         180

select_level(mi_df, 'year', 2023)  # Q1: 120, Q2: 180
\`\`\``,
			hint1: "MultiIndex yaratish uchun df.set_index(cols), tanlash uchun df.xs(value, level=level) dan foydalaning",
			hint2: "Darajani ustun holatiga aylantirish uchun df.reset_index(level=level) dan foydalaning",
			whyItMatters: `MultiIndex quyidagilar uchun kuchli:

- **Ierarxik ma'lumotlar**: Geografik (mamlakat/shahar), vaqtinchalik (yil/oy)
- **Samarali tanlash**: Quyi guruhlarga tez kirish
- **Ko'p o'lchovli tahlil**: Panel ma'lumotlari, ob'ekt bo'yicha vaqt qatorlari`,
		},
	},
};

export default task;
