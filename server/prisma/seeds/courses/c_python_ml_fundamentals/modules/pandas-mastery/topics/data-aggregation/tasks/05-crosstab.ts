import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-crosstab',
	title: 'Cross-tabulation',
	difficulty: 'medium',
	tags: ['pandas', 'crosstab', 'contingency'],
	estimatedTime: '12m',
	isPremium: false,
	order: 5,
	description: `# Cross-tabulation

Cross-tabulation creates contingency tables showing frequency of variable combinations.

## Task

Implement three functions:
1. \`simple_crosstab(df, col1, col2)\` - Create frequency cross-tabulation
2. \`crosstab_with_values(df, col1, col2, values, aggfunc)\` - Cross-tab with value aggregation
3. \`crosstab_normalized(df, col1, col2, normalize)\` - Normalized crosstab (proportions)

## Example

\`\`\`python
df = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'product': ['A', 'B', 'A', 'A', 'A'],
    'amount': [100, 200, 150, 100, 250]
})

simple_crosstab(df, 'gender', 'product')
#         A  B
# gender
# F       2  0
# M       2  1

crosstab_with_values(df, 'gender', 'product', 'amount', 'sum')
#         A    B
# gender
# F      250    0
# M      350  200

crosstab_normalized(df, 'gender', 'product', 'index')  # Row proportions
\`\`\``,

	initialCode: `import pandas as pd

def simple_crosstab(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create frequency cross-tabulation."""
    # Your code here
    pass

def crosstab_with_values(df: pd.DataFrame, col1: str, col2: str, values: str, aggfunc: str) -> pd.DataFrame:
    """Cross-tab with value aggregation."""
    # Your code here
    pass

def crosstab_normalized(df: pd.DataFrame, col1: str, col2: str, normalize: str) -> pd.DataFrame:
    """Normalized crosstab. normalize: 'index', 'columns', or 'all'."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def simple_crosstab(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create frequency cross-tabulation."""
    return pd.crosstab(df[col1], df[col2])

def crosstab_with_values(df: pd.DataFrame, col1: str, col2: str, values: str, aggfunc: str) -> pd.DataFrame:
    """Cross-tab with value aggregation."""
    return pd.crosstab(df[col1], df[col2], values=df[values], aggfunc=aggfunc)

def crosstab_normalized(df: pd.DataFrame, col1: str, col2: str, normalize: str) -> pd.DataFrame:
    """Normalized crosstab. normalize: 'index', 'columns', or 'all'."""
    return pd.crosstab(df[col1], df[col2], normalize=normalize)
`,

	testCode: `import pandas as pd
import unittest

class TestCrosstab(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'gender': ['M', 'M', 'F', 'F', 'M'],
            'product': ['A', 'B', 'A', 'A', 'A'],
            'amount': [100, 200, 150, 100, 250]
        })

    def test_simple_crosstab_basic(self):
        result = simple_crosstab(self.df, 'gender', 'product')
        self.assertEqual(result.loc['F', 'A'], 2)
        self.assertEqual(result.loc['M', 'B'], 1)

    def test_simple_crosstab_zeros(self):
        result = simple_crosstab(self.df, 'gender', 'product')
        self.assertEqual(result.loc['F', 'B'], 0)

    def test_simple_crosstab_shape(self):
        result = simple_crosstab(self.df, 'gender', 'product')
        self.assertEqual(result.shape, (2, 2))

    def test_crosstab_with_values_sum(self):
        result = crosstab_with_values(self.df, 'gender', 'product', 'amount', 'sum')
        self.assertEqual(result.loc['F', 'A'], 250)
        self.assertEqual(result.loc['M', 'A'], 350)

    def test_crosstab_with_values_mean(self):
        result = crosstab_with_values(self.df, 'gender', 'product', 'amount', 'mean')
        self.assertAlmostEqual(result.loc['F', 'A'], 125.0)

    def test_crosstab_normalized_index(self):
        result = crosstab_normalized(self.df, 'gender', 'product', 'index')
        # Each row should sum to 1
        for idx in result.index:
            self.assertAlmostEqual(result.loc[idx].sum(), 1.0)

    def test_crosstab_normalized_columns(self):
        result = crosstab_normalized(self.df, 'gender', 'product', 'columns')
        # Each column should sum to 1
        for col in result.columns:
            self.assertAlmostEqual(result[col].sum(), 1.0)

    def test_crosstab_normalized_all(self):
        result = crosstab_normalized(self.df, 'gender', 'product', 'all')
        # All values should sum to 1
        self.assertAlmostEqual(result.values.sum(), 1.0)

    def test_simple_crosstab_index_name(self):
        result = simple_crosstab(self.df, 'gender', 'product')
        self.assertEqual(result.index.name, 'gender')

    def test_crosstab_with_values_handles_nan(self):
        result = crosstab_with_values(self.df, 'gender', 'product', 'amount', 'sum')
        # F has no B purchases, should be NaN or 0
        self.assertTrue(result.loc['F', 'B'] == 0 or pd.isna(result.loc['F', 'B']))
`,

	hint1: 'Use pd.crosstab(df[col1], df[col2]) for frequency tables',
	hint2: 'Add values= and aggfunc= for value aggregation, normalize= for proportions',

	whyItMatters: `Cross-tabulation is essential for:

- **Categorical analysis**: Understand relationships between categories
- **Chi-square tests**: Prepare data for statistical tests
- **Market research**: Customer segments by product preferences
- **Survey analysis**: Response patterns across demographics

This is fundamental to exploratory data analysis.`,

	translations: {
		ru: {
			title: 'Кросс-табуляция',
			description: `# Кросс-табуляция

Кросс-табуляция создаёт таблицы сопряжённости, показывающие частоту комбинаций переменных.

## Задача

Реализуйте три функции:
1. \`simple_crosstab(df, col1, col2)\` - Создать частотную кросс-табуляцию
2. \`crosstab_with_values(df, col1, col2, values, aggfunc)\` - Кросс-таблица с агрегацией значений
3. \`crosstab_normalized(df, col1, col2, normalize)\` - Нормализованная кросс-таблица (пропорции)

## Пример

\`\`\`python
df = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'product': ['A', 'B', 'A', 'A', 'A'],
    'amount': [100, 200, 150, 100, 250]
})

simple_crosstab(df, 'gender', 'product')
#         A  B
# gender
# F       2  0
# M       2  1

crosstab_with_values(df, 'gender', 'product', 'amount', 'sum')
#         A    B
# gender
# F      250    0
# M      350  200

crosstab_normalized(df, 'gender', 'product', 'index')  # Row proportions
\`\`\``,
			hint1: 'Используйте pd.crosstab(df[col1], df[col2]) для частотных таблиц',
			hint2: 'Добавьте values= и aggfunc= для агрегации значений, normalize= для пропорций',
			whyItMatters: `Кросс-табуляция необходима для:

- **Анализ категорий**: Понимание связей между категориями
- **Тесты хи-квадрат**: Подготовка данных для статистических тестов
- **Маркетинговые исследования**: Сегменты клиентов по предпочтениям
- **Анализ опросов**: Паттерны ответов по демографии`,
		},
		uz: {
			title: "Kross-tabulyatsiya",
			description: `# Kross-tabulyatsiya

Kross-tabulyatsiya o'zgaruvchilar kombinatsiyalarining chastotasini ko'rsatadigan contingency jadvallarini yaratadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_crosstab(df, col1, col2)\` - Chastotali kross-tabulyatsiya yaratish
2. \`crosstab_with_values(df, col1, col2, values, aggfunc)\` - Qiymat agregatsiyasi bilan kross-jadval
3. \`crosstab_normalized(df, col1, col2, normalize)\` - Normallashtirilgan kross-jadval (nisbatlar)

## Misol

\`\`\`python
df = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'product': ['A', 'B', 'A', 'A', 'A'],
    'amount': [100, 200, 150, 100, 250]
})

simple_crosstab(df, 'gender', 'product')
#         A  B
# gender
# F       2  0
# M       2  1

crosstab_with_values(df, 'gender', 'product', 'amount', 'sum')
#         A    B
# gender
# F      250    0
# M      350  200

crosstab_normalized(df, 'gender', 'product', 'index')  # Row proportions
\`\`\``,
			hint1: "Chastota jadvallari uchun pd.crosstab(df[col1], df[col2]) dan foydalaning",
			hint2: "Qiymat agregatsiyasi uchun values= va aggfunc=, nisbatlar uchun normalize= qo'shing",
			whyItMatters: `Kross-tabulyatsiya quyidagilar uchun zarur:

- **Kategorik tahlil**: Kategoriyalar orasidagi munosabatlarni tushunish
- **Xi-kvadrat testlari**: Statistik testlar uchun ma'lumotlarni tayyorlash
- **Marketing tadqiqotlari**: Mahsulot afzalliklari bo'yicha mijoz segmentlari`,
		},
	},
};

export default task;
