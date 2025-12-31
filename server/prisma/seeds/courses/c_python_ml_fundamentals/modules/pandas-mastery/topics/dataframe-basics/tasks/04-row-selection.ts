import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-row-selection',
	title: 'Selecting Rows',
	difficulty: 'easy',
	tags: ['pandas', 'selection', 'rows', 'iloc', 'loc'],
	estimatedTime: '12m',
	isPremium: false,
	order: 4,
	description: `# Selecting Rows

Pandas provides multiple ways to select rows: by position (iloc) or by label (loc).

## Task

Implement four functions:
1. \`select_by_position(df, start, end)\` - Select rows by position range [start:end]
2. \`select_by_index(df, indices)\` - Select rows at specific integer positions
3. \`select_by_label(df, labels)\` - Select rows by index labels
4. \`select_head_tail(df, n)\` - Return first n and last n rows combined

## Example

\`\`\`python
df = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])

select_by_position(df, 1, 4)  # Rows at positions 1, 2, 3
select_by_index(df, [0, 2, 4])  # Rows at positions 0, 2, 4
select_by_label(df, ['a', 'c', 'e'])  # Rows with labels a, c, e
select_head_tail(df, 2)  # First 2 + last 2 rows
\`\`\``,

	initialCode: `import pandas as pd

def select_by_position(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Select rows by position range [start:end]."""
    # Your code here
    pass

def select_by_index(df: pd.DataFrame, indices: list) -> pd.DataFrame:
    """Select rows at specific integer positions."""
    # Your code here
    pass

def select_by_label(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Select rows by index labels."""
    # Your code here
    pass

def select_head_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return first n and last n rows combined."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def select_by_position(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Select rows by position range [start:end]."""
    return df.iloc[start:end]

def select_by_index(df: pd.DataFrame, indices: list) -> pd.DataFrame:
    """Select rows at specific integer positions."""
    return df.iloc[indices]

def select_by_label(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Select rows by index labels."""
    return df.loc[labels]

def select_head_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return first n and last n rows combined."""
    return pd.concat([df.head(n), df.tail(n)])
`,

	testCode: `import pandas as pd
import unittest

class TestRowSelection(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'x': [10, 20, 30, 40, 50]},
            index=['a', 'b', 'c', 'd', 'e']
        )

    def test_select_by_position_basic(self):
        result = select_by_position(self.df, 1, 4)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['x'].tolist(), [20, 30, 40])

    def test_select_by_position_from_start(self):
        result = select_by_position(self.df, 0, 2)
        self.assertEqual(result['x'].tolist(), [10, 20])

    def test_select_by_position_to_end(self):
        result = select_by_position(self.df, 3, 5)
        self.assertEqual(result['x'].tolist(), [40, 50])

    def test_select_by_index_basic(self):
        result = select_by_index(self.df, [0, 2, 4])
        self.assertEqual(result['x'].tolist(), [10, 30, 50])

    def test_select_by_index_single(self):
        result = select_by_index(self.df, [2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result['x'].iloc[0], 30)

    def test_select_by_index_reverse(self):
        result = select_by_index(self.df, [4, 2, 0])
        self.assertEqual(result['x'].tolist(), [50, 30, 10])

    def test_select_by_label_basic(self):
        result = select_by_label(self.df, ['a', 'c', 'e'])
        self.assertEqual(result['x'].tolist(), [10, 30, 50])

    def test_select_by_label_single(self):
        result = select_by_label(self.df, ['b'])
        self.assertEqual(len(result), 1)

    def test_select_head_tail_basic(self):
        result = select_head_tail(self.df, 2)
        self.assertEqual(len(result), 4)
        self.assertIn(10, result['x'].tolist())
        self.assertIn(50, result['x'].tolist())

    def test_select_head_tail_n1(self):
        result = select_head_tail(self.df, 1)
        self.assertEqual(len(result), 2)
`,

	hint1: 'Use df.iloc[] for position-based, df.loc[] for label-based selection',
	hint2: 'Use pd.concat([df.head(n), df.tail(n)]) to combine head and tail',

	whyItMatters: `Row selection is fundamental for:

- **Data sampling**: Select random or specific samples
- **Train/test splits**: Divide data for model evaluation
- **Time series**: Select date ranges for analysis
- **Debugging**: Inspect specific rows with issues

Understanding iloc vs loc prevents common indexing errors.`,

	translations: {
		ru: {
			title: 'Выбор строк',
			description: `# Выбор строк

Pandas предоставляет несколько способов выбора строк: по позиции (iloc) или по метке (loc).

## Задача

Реализуйте четыре функции:
1. \`select_by_position(df, start, end)\` - Выбрать строки по диапазону позиций [start:end]
2. \`select_by_index(df, indices)\` - Выбрать строки по конкретным целочисленным позициям
3. \`select_by_label(df, labels)\` - Выбрать строки по меткам индекса
4. \`select_head_tail(df, n)\` - Вернуть первые n и последние n строк вместе

## Пример

\`\`\`python
df = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])

select_by_position(df, 1, 4)  # Строки на позициях 1, 2, 3
select_by_index(df, [0, 2, 4])  # Строки на позициях 0, 2, 4
select_by_label(df, ['a', 'c', 'e'])  # Строки с метками a, c, e
\`\`\``,
			hint1: 'Используйте df.iloc[] для позиционного, df.loc[] для выбора по меткам',
			hint2: 'Используйте pd.concat([df.head(n), df.tail(n)]) для объединения начала и конца',
			whyItMatters: `Выбор строк фундаментален для:

- **Сэмплирование данных**: Выбор случайных или конкретных сэмплов
- **Train/test разбиение**: Разделение данных для оценки модели
- **Временные ряды**: Выбор диапазонов дат для анализа
- **Отладка**: Проверка конкретных строк с проблемами`,
		},
		uz: {
			title: "Qatorlarni tanlash",
			description: `# Qatorlarni tanlash

Pandas qatorlarni tanlashning bir necha usullarini taqdim etadi: pozitsiya bo'yicha (iloc) yoki yorliq bo'yicha (loc).

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`select_by_position(df, start, end)\` - Pozitsiya diapazoni bo'yicha qatorlarni tanlash [start:end]
2. \`select_by_index(df, indices)\` - Ma'lum butun son pozitsiyalaridagi qatorlarni tanlash
3. \`select_by_label(df, labels)\` - Indeks yorliqlari bo'yicha qatorlarni tanlash
4. \`select_head_tail(df, n)\` - Birinchi n va oxirgi n qatorlarni birlashtirib qaytarish

## Misol

\`\`\`python
df = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])

select_by_position(df, 1, 4)  # Rows at positions 1, 2, 3
select_by_index(df, [0, 2, 4])  # Rows at positions 0, 2, 4
select_by_label(df, ['a', 'c', 'e'])  # Rows with labels a, c, e
select_head_tail(df, 2)  # First 2 + last 2 rows
\`\`\``,
			hint1: "Pozitsiya asosida df.iloc[], yorliq asosida df.loc[] dan foydalaning",
			hint2: "Bosh va dumni birlashtirish uchun pd.concat([df.head(n), df.tail(n)]) dan foydalaning",
			whyItMatters: `Qatorlarni tanlash quyidagilar uchun asosiydir:

- **Ma'lumotlarni namunalash**: Tasodifiy yoki ma'lum namunalarni tanlash
- **Train/test bo'linmalari**: Modelni baholash uchun ma'lumotlarni bo'lish
- **Vaqt qatorlari**: Tahlil uchun sana diapazonlarini tanlash`,
		},
	},
};

export default task;
