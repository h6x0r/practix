import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-merge-join',
	title: 'Merging and Joining DataFrames',
	difficulty: 'medium',
	tags: ['pandas', 'merge', 'join', 'combine'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Merging and Joining DataFrames

Combining data from multiple sources is a core data engineering skill.

## Task

Implement four functions:
1. \`inner_merge(df1, df2, on)\` - Inner join on column
2. \`left_merge(df1, df2, on)\` - Left join on column
3. \`concat_rows(dfs)\` - Concatenate DataFrames vertically
4. \`merge_on_multiple(df1, df2, on_cols)\` - Merge on multiple columns

## Example

\`\`\`python
users = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
orders = pd.DataFrame({'user_id': [1, 1, 2], 'product': ['A', 'B', 'C']})

inner_merge(users, orders, left_on='id', right_on='user_id')  # Users with orders
left_merge(users, orders, left_on='id', right_on='user_id')   # All users, orders if exist

df1 = pd.DataFrame({'x': [1, 2]})
df2 = pd.DataFrame({'x': [3, 4]})
concat_rows([df1, df2])  # x: [1, 2, 3, 4]
\`\`\``,

	initialCode: `import pandas as pd

def inner_merge(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Inner join on columns."""
    # Your code here
    pass

def left_merge(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Left join on columns."""
    # Your code here
    pass

def concat_rows(dfs: list) -> pd.DataFrame:
    """Concatenate DataFrames vertically."""
    # Your code here
    pass

def merge_on_multiple(df1: pd.DataFrame, df2: pd.DataFrame, on_cols: list) -> pd.DataFrame:
    """Inner merge on multiple columns."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def inner_merge(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Inner join on columns."""
    return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how='inner')

def left_merge(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Left join on columns."""
    return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how='left')

def concat_rows(dfs: list) -> pd.DataFrame:
    """Concatenate DataFrames vertically."""
    return pd.concat(dfs, ignore_index=True)

def merge_on_multiple(df1: pd.DataFrame, df2: pd.DataFrame, on_cols: list) -> pd.DataFrame:
    """Inner merge on multiple columns."""
    return pd.merge(df1, df2, on=on_cols, how='inner')
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestMergeJoin(unittest.TestCase):
    def setUp(self):
        self.users = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        self.orders = pd.DataFrame({
            'user_id': [1, 1, 2],
            'product': ['A', 'B', 'C']
        })

    def test_inner_merge_basic(self):
        result = inner_merge(self.users, self.orders, 'id', 'user_id')
        self.assertEqual(len(result), 3)
        self.assertNotIn(3, result['id'].values)

    def test_inner_merge_columns(self):
        result = inner_merge(self.users, self.orders, 'id', 'user_id')
        self.assertIn('name', result.columns)
        self.assertIn('product', result.columns)

    def test_left_merge_keeps_all_left(self):
        result = left_merge(self.users, self.orders, 'id', 'user_id')
        self.assertEqual(len(result['id'].unique()), 3)

    def test_left_merge_has_nan(self):
        result = left_merge(self.users, self.orders, 'id', 'user_id')
        charlie = result[result['name'] == 'Charlie']
        self.assertTrue(charlie['product'].isna().all())

    def test_concat_rows_basic(self):
        df1 = pd.DataFrame({'x': [1, 2]})
        df2 = pd.DataFrame({'x': [3, 4]})
        result = concat_rows([df1, df2])
        self.assertEqual(len(result), 4)
        self.assertEqual(result['x'].tolist(), [1, 2, 3, 4])

    def test_concat_rows_reset_index(self):
        df1 = pd.DataFrame({'x': [1]}, index=[0])
        df2 = pd.DataFrame({'x': [2]}, index=[0])
        result = concat_rows([df1, df2])
        self.assertEqual(list(result.index), [0, 1])

    def test_merge_on_multiple_basic(self):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'val1': [10, 20]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 5], 'val2': [100, 200]})
        result = merge_on_multiple(df1, df2, ['a', 'b'])
        self.assertEqual(len(result), 1)

    def test_concat_three_dfs(self):
        dfs = [pd.DataFrame({'x': [i]}) for i in range(3)]
        result = concat_rows(dfs)
        self.assertEqual(len(result), 3)

    def test_inner_merge_empty(self):
        df1 = pd.DataFrame({'id': [1, 2]})
        df2 = pd.DataFrame({'id': [3, 4]})
        result = inner_merge(df1, df2, 'id', 'id')
        self.assertEqual(len(result), 0)

    def test_left_merge_all_match(self):
        df1 = pd.DataFrame({'id': [1, 2], 'x': ['a', 'b']})
        df2 = pd.DataFrame({'id': [1, 2], 'y': ['c', 'd']})
        result = left_merge(df1, df2, 'id', 'id')
        self.assertFalse(result['y'].isna().any())
`,

	hint1: 'Use pd.merge(df1, df2, left_on=, right_on=, how=)',
	hint2: 'Use pd.concat(dfs, ignore_index=True) for vertical concatenation',

	whyItMatters: `Data merging is essential for:

- **Data integration**: Combine data from multiple sources
- **Feature enrichment**: Add external features to dataset
- **Relational data**: Join tables like SQL
- **Data pipelines**: Combine results from parallel processing

Understanding join types prevents data loss and duplication.`,

	translations: {
		ru: {
			title: 'Слияние и соединение DataFrame',
			description: `# Слияние и соединение DataFrame

Объединение данных из нескольких источников — ключевой навык data engineering.

## Задача

Реализуйте четыре функции:
1. \`inner_merge(df1, df2, on)\` - Inner join по столбцу
2. \`left_merge(df1, df2, on)\` - Left join по столбцу
3. \`concat_rows(dfs)\` - Конкатенация DataFrame вертикально
4. \`merge_on_multiple(df1, df2, on_cols)\` - Слияние по нескольким столбцам

## Пример

\`\`\`python
users = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
orders = pd.DataFrame({'user_id': [1, 1, 2], 'product': ['A', 'B', 'C']})

inner_merge(users, orders, left_on='id', right_on='user_id')  # Users with orders
left_merge(users, orders, left_on='id', right_on='user_id')   # All users, orders if exist

df1 = pd.DataFrame({'x': [1, 2]})
df2 = pd.DataFrame({'x': [3, 4]})
concat_rows([df1, df2])  # x: [1, 2, 3, 4]
\`\`\``,
			hint1: 'Используйте pd.merge(df1, df2, left_on=, right_on=, how=)',
			hint2: 'Используйте pd.concat(dfs, ignore_index=True) для вертикальной конкатенации',
			whyItMatters: `Слияние данных необходимо для:

- **Интеграция данных**: Объединение данных из разных источников
- **Обогащение признаков**: Добавление внешних признаков к датасету
- **Реляционные данные**: Соединение таблиц как в SQL
- **Data pipelines**: Объединение результатов параллельной обработки`,
		},
		uz: {
			title: "DataFramelarni birlashtirish va qo'shish",
			description: `# DataFramelarni birlashtirish va qo'shish

Bir nechta manbalardan ma'lumotlarni birlashtirish data engineering ning asosiy ko'nikmasi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`inner_merge(df1, df2, on)\` - Ustun bo'yicha Inner join
2. \`left_merge(df1, df2, on)\` - Ustun bo'yicha Left join
3. \`concat_rows(dfs)\` - DataFramelarni vertikal birlashtirish
4. \`merge_on_multiple(df1, df2, on_cols)\` - Bir nechta ustunlar bo'yicha birlashtirish

## Misol

\`\`\`python
users = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
orders = pd.DataFrame({'user_id': [1, 1, 2], 'product': ['A', 'B', 'C']})

inner_merge(users, orders, left_on='id', right_on='user_id')  # Users with orders
left_merge(users, orders, left_on='id', right_on='user_id')   # All users, orders if exist

df1 = pd.DataFrame({'x': [1, 2]})
df2 = pd.DataFrame({'x': [3, 4]})
concat_rows([df1, df2])  # x: [1, 2, 3, 4]
\`\`\``,
			hint1: "pd.merge(df1, df2, left_on=, right_on=, how=) dan foydalaning",
			hint2: "Vertikal birlashtirish uchun pd.concat(dfs, ignore_index=True) dan foydalaning",
			whyItMatters: `Ma'lumotlarni birlashtirish quyidagilar uchun zarur:

- **Ma'lumotlar integratsiyasi**: Bir nechta manbalardan ma'lumotlarni birlashtirish
- **Xususiyatlarni boyitish**: Datasetga tashqi xususiyatlarni qo'shish
- **Relyatsion ma'lumotlar**: SQL kabi jadvallarni qo'shish`,
		},
	},
};

export default task;
