import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-pivot-tables',
	title: 'Pivot Tables',
	difficulty: 'medium',
	tags: ['pandas', 'pivot', 'reshape'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Pivot Tables

Pivot tables reshape data from long to wide format, summarizing values across dimensions.

## Task

Implement three functions:
1. \`simple_pivot(df, index, columns, values)\` - Create basic pivot table
2. \`pivot_with_agg(df, index, columns, values, aggfunc)\` - Pivot with custom aggregation
3. \`unpivot(df, id_vars, value_vars)\` - Melt wide format back to long

## Example

\`\`\`python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

simple_pivot(df, 'date', 'product', 'sales')
#           A    B
# 2023-01  100  150
# 2023-02  120  180

# Unpivot: wide to long
wide = pd.DataFrame({'id': [1], 'A': [10], 'B': [20]})
unpivot(wide, ['id'], ['A', 'B'])  # id, variable, value columns
\`\`\``,

	initialCode: `import pandas as pd

def simple_pivot(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Create basic pivot table."""
    # Your code here
    pass

def pivot_with_agg(df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str) -> pd.DataFrame:
    """Pivot with custom aggregation function."""
    # Your code here
    pass

def unpivot(df: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    """Melt wide format back to long format."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def simple_pivot(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Create basic pivot table."""
    return df.pivot(index=index, columns=columns, values=values)

def pivot_with_agg(df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str) -> pd.DataFrame:
    """Pivot with custom aggregation function."""
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)

def unpivot(df: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    """Melt wide format back to long format."""
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)
`,

	testCode: `import pandas as pd
import unittest

class TestPivotTables(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'product': ['A', 'B', 'A', 'B'],
            'sales': [100, 150, 120, 180]
        })

    def test_simple_pivot_basic(self):
        result = simple_pivot(self.df, 'date', 'product', 'sales')
        self.assertEqual(result.loc['2023-01', 'A'], 100)
        self.assertEqual(result.loc['2023-02', 'B'], 180)

    def test_simple_pivot_shape(self):
        result = simple_pivot(self.df, 'date', 'product', 'sales')
        self.assertEqual(result.shape, (2, 2))

    def test_simple_pivot_columns(self):
        result = simple_pivot(self.df, 'date', 'product', 'sales')
        self.assertTrue('A' in result.columns)
        self.assertTrue('B' in result.columns)

    def test_pivot_with_agg_sum(self):
        df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-01'],
            'product': ['A', 'A', 'B'],
            'sales': [10, 20, 30]
        })
        result = pivot_with_agg(df, 'date', 'product', 'sales', 'sum')
        self.assertEqual(result.loc['2023-01', 'A'], 30)

    def test_pivot_with_agg_mean(self):
        df = pd.DataFrame({
            'date': ['2023-01', '2023-01'],
            'product': ['A', 'A'],
            'sales': [10, 20]
        })
        result = pivot_with_agg(df, 'date', 'product', 'sales', 'mean')
        self.assertAlmostEqual(result.loc['2023-01', 'A'], 15.0)

    def test_unpivot_basic(self):
        wide = pd.DataFrame({'id': [1, 2], 'A': [10, 20], 'B': [30, 40]})
        result = unpivot(wide, ['id'], ['A', 'B'])
        self.assertEqual(len(result), 4)

    def test_unpivot_columns(self):
        wide = pd.DataFrame({'id': [1], 'A': [10], 'B': [20]})
        result = unpivot(wide, ['id'], ['A', 'B'])
        self.assertIn('variable', result.columns)
        self.assertIn('value', result.columns)

    def test_unpivot_values(self):
        wide = pd.DataFrame({'id': [1], 'X': [100]})
        result = unpivot(wide, ['id'], ['X'])
        self.assertEqual(result['value'].iloc[0], 100)

    def test_pivot_index_preserved(self):
        result = simple_pivot(self.df, 'date', 'product', 'sales')
        self.assertTrue('2023-01' in result.index)

    def test_unpivot_roundtrip(self):
        wide = pd.DataFrame({'id': [1], 'A': [10]})
        long = unpivot(wide, ['id'], ['A'])
        self.assertEqual(len(long), 1)
`,

	hint1: 'Use df.pivot() for simple reshape, df.pivot_table() when aggregation needed',
	hint2: 'Use pd.melt() to convert wide format to long format',

	whyItMatters: `Pivot tables are essential for:

- **Reporting**: Create summary tables for business reports
- **Feature engineering**: Reshape data for time series features
- **Data exploration**: View data from different perspectives
- **Cross-tabulation**: Analyze relationships between categories

Excel-style pivot tables in Python enable powerful data summarization.`,

	translations: {
		ru: {
			title: 'Сводные таблицы',
			description: `# Сводные таблицы

Сводные таблицы преобразуют данные из длинного в широкий формат, суммируя значения по измерениям.

## Задача

Реализуйте три функции:
1. \`simple_pivot(df, index, columns, values)\` - Создать базовую сводную таблицу
2. \`pivot_with_agg(df, index, columns, values, aggfunc)\` - Сводная с пользовательской агрегацией
3. \`unpivot(df, id_vars, value_vars)\` - Преобразовать широкий формат обратно в длинный

## Пример

\`\`\`python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

simple_pivot(df, 'date', 'product', 'sales')
#           A    B
# 2023-01  100  150
# 2023-02  120  180

# Unpivot: wide to long
wide = pd.DataFrame({'id': [1], 'A': [10], 'B': [20]})
unpivot(wide, ['id'], ['A', 'B'])  # id, variable, value columns
\`\`\``,
			hint1: 'Используйте df.pivot() для простого преобразования, df.pivot_table() когда нужна агрегация',
			hint2: 'Используйте pd.melt() для конвертации широкого формата в длинный',
			whyItMatters: `Сводные таблицы необходимы для:

- **Отчётность**: Создание сводных таблиц для бизнес-отчётов
- **Feature engineering**: Преобразование данных для признаков временных рядов
- **Исследование данных**: Просмотр данных с разных перспектив
- **Кросс-табуляция**: Анализ связей между категориями`,
		},
		uz: {
			title: "Pivot jadvallar",
			description: `# Pivot jadvallar

Pivot jadvallar ma'lumotlarni uzun formatdan keng formatga o'zgartiradi, o'lchamlar bo'yicha qiymatlarni jamlaydi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_pivot(df, index, columns, values)\` - Asosiy pivot jadval yaratish
2. \`pivot_with_agg(df, index, columns, values, aggfunc)\` - Maxsus agregatsiya bilan pivot
3. \`unpivot(df, id_vars, value_vars)\` - Keng formatni uzun formatga qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

simple_pivot(df, 'date', 'product', 'sales')
#           A    B
# 2023-01  100  150
# 2023-02  120  180

# Unpivot: wide to long
wide = pd.DataFrame({'id': [1], 'A': [10], 'B': [20]})
unpivot(wide, ['id'], ['A', 'B'])  # id, variable, value columns
\`\`\``,
			hint1: "Oddiy o'zgartirish uchun df.pivot(), agregatsiya kerak bo'lganda df.pivot_table() dan foydalaning",
			hint2: "Keng formatni uzun formatga aylantirish uchun pd.melt() dan foydalaning",
			whyItMatters: `Pivot jadvallar quyidagilar uchun zarur:

- **Hisobot**: Biznes hisobotlari uchun umumlashtiruvchi jadvallar yaratish
- **Feature engineering**: Vaqt qatorlari xususiyatlari uchun ma'lumotlarni qayta shakllantirish
- **Ma'lumotlarni o'rganish**: Ma'lumotlarni turli nuqtai nazardan ko'rish`,
		},
	},
};

export default task;
