import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-data-types',
	title: 'Working with Data Types',
	difficulty: 'medium',
	tags: ['pandas', 'dtypes', 'conversion'],
	estimatedTime: '12m',
	isPremium: false,
	order: 6,
	description: `# Working with Data Types

Proper data types ensure correct operations and memory efficiency.

## Task

Implement four functions:
1. \`convert_to_numeric(df, column)\` - Convert column to numeric type
2. \`convert_to_datetime(df, column, format)\` - Convert column to datetime
3. \`convert_to_category(df, column)\` - Convert column to category type
4. \`optimize_memory(df)\` - Downcast numeric types to save memory

## Example

\`\`\`python
df = pd.DataFrame({
    'price': ['10.5', '20.0', '15.5'],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'color': ['red', 'blue', 'red']
})

convert_to_numeric(df, 'price')  # price becomes float64
convert_to_datetime(df, 'date', '%Y-%m-%d')  # date becomes datetime64
convert_to_category(df, 'color')  # color becomes category (saves memory)
optimize_memory(df)  # Downcast int64 to int32/int16 where possible
\`\`\``,

	initialCode: `import pandas as pd

def convert_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to numeric type."""
    # Your code here
    pass

def convert_to_datetime(df: pd.DataFrame, column: str, format: str) -> pd.DataFrame:
    """Convert column to datetime."""
    # Your code here
    pass

def convert_to_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to category type."""
    # Your code here
    pass

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to save memory."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def convert_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to numeric type."""
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def convert_to_datetime(df: pd.DataFrame, column: str, format: str) -> pd.DataFrame:
    """Convert column to datetime."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format=format)
    return df

def convert_to_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to category type."""
    df = df.copy()
    df[column] = df[column].astype('category')
    return df

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to save memory."""
    df = df.copy()
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestDataTypes(unittest.TestCase):
    def test_convert_to_numeric_basic(self):
        df = pd.DataFrame({'x': ['1.5', '2.5', '3.5']})
        result = convert_to_numeric(df, 'x')
        self.assertTrue(np.issubdtype(result['x'].dtype, np.floating))

    def test_convert_to_numeric_int(self):
        df = pd.DataFrame({'x': ['1', '2', '3']})
        result = convert_to_numeric(df, 'x')
        self.assertTrue(np.issubdtype(result['x'].dtype, np.number))

    def test_convert_to_numeric_no_modify(self):
        df = pd.DataFrame({'x': ['1', '2', '3']})
        _ = convert_to_numeric(df, 'x')
        self.assertEqual(df['x'].dtype, object)

    def test_convert_to_datetime_basic(self):
        df = pd.DataFrame({'d': ['2023-01-01', '2023-01-02']})
        result = convert_to_datetime(df, 'd', '%Y-%m-%d')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['d']))

    def test_convert_to_datetime_values(self):
        df = pd.DataFrame({'d': ['2023-06-15']})
        result = convert_to_datetime(df, 'd', '%Y-%m-%d')
        self.assertEqual(result['d'].iloc[0].month, 6)

    def test_convert_to_category_basic(self):
        df = pd.DataFrame({'c': ['a', 'b', 'a', 'b', 'a']})
        result = convert_to_category(df, 'c')
        self.assertEqual(result['c'].dtype.name, 'category')

    def test_convert_to_category_unique(self):
        df = pd.DataFrame({'c': ['x', 'y', 'z', 'x']})
        result = convert_to_category(df, 'c')
        self.assertEqual(len(result['c'].cat.categories), 3)

    def test_optimize_memory_runs(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        result = optimize_memory(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_optimize_memory_smaller(self):
        df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.int64)})
        original_mem = df.memory_usage(deep=True).sum()
        result = optimize_memory(df)
        new_mem = result.memory_usage(deep=True).sum()
        self.assertLessEqual(new_mem, original_mem)

    def test_convert_preserves_values(self):
        df = pd.DataFrame({'x': ['10', '20', '30']})
        result = convert_to_numeric(df, 'x')
        self.assertEqual(result['x'].tolist(), [10, 20, 30])
`,

	hint1: 'Use pd.to_numeric(), pd.to_datetime(), .astype("category")',
	hint2: 'Use pd.to_numeric(col, downcast="integer"/"float") to optimize',

	whyItMatters: `Proper data types are essential for:

- **Memory efficiency**: Category vs object saves 90%+ memory
- **Performance**: Numeric operations on proper types are faster
- **Correctness**: Date operations require datetime type
- **Feature engineering**: Different types enable different operations

Large datasets require careful type management.`,

	translations: {
		ru: {
			title: 'Работа с типами данных',
			description: `# Работа с типами данных

Правильные типы данных обеспечивают корректные операции и эффективность памяти.

## Задача

Реализуйте четыре функции:
1. \`convert_to_numeric(df, column)\` - Конвертировать столбец в числовой тип
2. \`convert_to_datetime(df, column, format)\` - Конвертировать столбец в datetime
3. \`convert_to_category(df, column)\` - Конвертировать столбец в категориальный тип
4. \`optimize_memory(df)\` - Уменьшить числовые типы для экономии памяти

## Пример

\`\`\`python
df = pd.DataFrame({
    'price': ['10.5', '20.0', '15.5'],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'color': ['red', 'blue', 'red']
})

convert_to_numeric(df, 'price')  # price становится float64
convert_to_datetime(df, 'date', '%Y-%m-%d')  # date становится datetime64
\`\`\``,
			hint1: 'Используйте pd.to_numeric(), pd.to_datetime(), .astype("category")',
			hint2: 'Используйте pd.to_numeric(col, downcast="integer"/"float") для оптимизации',
			whyItMatters: `Правильные типы данных необходимы для:

- **Эффективность памяти**: Category vs object экономит 90%+ памяти
- **Производительность**: Числовые операции на правильных типах быстрее
- **Корректность**: Операции с датами требуют тип datetime
- **Feature engineering**: Разные типы позволяют разные операции`,
		},
		uz: {
			title: "Ma'lumot turlari bilan ishlash",
			description: `# Ma'lumot turlari bilan ishlash

To'g'ri ma'lumot turlari to'g'ri operatsiyalar va xotira samaradorligini ta'minlaydi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`convert_to_numeric(df, column)\` - Ustunni raqamli turga aylantirish
2. \`convert_to_datetime(df, column, format)\` - Ustunni datetime ga aylantirish
3. \`convert_to_category(df, column)\` - Ustunni kategoriya turiga aylantirish
4. \`optimize_memory(df)\` - Xotirani tejash uchun raqamli turlarni kichraytirish

## Misol

\`\`\`python
df = pd.DataFrame({
    'price': ['10.5', '20.0', '15.5'],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'color': ['red', 'blue', 'red']
})

convert_to_numeric(df, 'price')  # price becomes float64
convert_to_datetime(df, 'date', '%Y-%m-%d')  # date becomes datetime64
convert_to_category(df, 'color')  # color becomes category (saves memory)
optimize_memory(df)  # Downcast int64 to int32/int16 where possible
\`\`\``,
			hint1: "pd.to_numeric(), pd.to_datetime(), .astype('category') dan foydalaning",
			hint2: "Optimallashtirish uchun pd.to_numeric(col, downcast='integer'/'float') dan foydalaning",
			whyItMatters: `To'g'ri ma'lumot turlari quyidagilar uchun zarur:

- **Xotira samaradorligi**: Category vs object 90%+ xotira tejaydi
- **Ishlash**: To'g'ri turlardagi raqamli operatsiyalar tezroq
- **To'g'rilik**: Sana operatsiyalari datetime turini talab qiladi`,
		},
	},
};

export default task;
