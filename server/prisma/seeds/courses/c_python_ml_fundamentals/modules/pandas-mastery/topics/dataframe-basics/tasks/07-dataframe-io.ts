import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-dataframe-io',
	title: 'Reading and Writing Data',
	difficulty: 'easy',
	tags: ['pandas', 'io', 'csv', 'json'],
	estimatedTime: '10m',
	isPremium: false,
	order: 7,
	description: `# Reading and Writing Data

Pandas can read from and write to many file formats. CSV and JSON are the most common.

## Task

Implement four functions:
1. \`read_csv_data(csv_string)\` - Parse CSV string into DataFrame
2. \`read_json_data(json_string)\` - Parse JSON string into DataFrame
3. \`to_csv_string(df)\` - Convert DataFrame to CSV string
4. \`to_json_string(df)\` - Convert DataFrame to JSON string

## Example

\`\`\`python
csv_string = "name,age\\nAlice,25\\nBob,30"
read_csv_data(csv_string)
#     name  age
# 0  Alice   25
# 1    Bob   30

json_string = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
read_json_data(json_string)

df = pd.DataFrame({'x': [1, 2]})
to_csv_string(df)  # "x\\n1\\n2\\n"
to_json_string(df)  # '[{"x": 1}, {"x": 2}]'
\`\`\``,

	initialCode: `import pandas as pd
from io import StringIO

def read_csv_data(csv_string: str) -> pd.DataFrame:
    """Parse CSV string into DataFrame."""
    # Your code here
    pass

def read_json_data(json_string: str) -> pd.DataFrame:
    """Parse JSON string into DataFrame."""
    # Your code here
    pass

def to_csv_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string (no index)."""
    # Your code here
    pass

def to_json_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to JSON string (records format)."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd
from io import StringIO

def read_csv_data(csv_string: str) -> pd.DataFrame:
    """Parse CSV string into DataFrame."""
    return pd.read_csv(StringIO(csv_string))

def read_json_data(json_string: str) -> pd.DataFrame:
    """Parse JSON string into DataFrame."""
    return pd.read_json(StringIO(json_string))

def to_csv_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string (no index)."""
    return df.to_csv(index=False)

def to_json_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to JSON string (records format)."""
    return df.to_json(orient='records')
`,

	testCode: `import pandas as pd
import json
import unittest

class TestDataFrameIO(unittest.TestCase):
    def test_read_csv_basic(self):
        csv = "name,age\\nAlice,25\\nBob,30"
        df = read_csv_data(csv)
        self.assertEqual(list(df.columns), ['name', 'age'])
        self.assertEqual(len(df), 2)

    def test_read_csv_values(self):
        csv = "x,y\\n1,2\\n3,4"
        df = read_csv_data(csv)
        self.assertEqual(df['x'].tolist(), [1, 3])

    def test_read_json_basic(self):
        j = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
        df = read_json_data(j)
        self.assertEqual(len(df), 2)
        self.assertIn('a', df.columns)

    def test_read_json_values(self):
        j = '[{"x": 10}, {"x": 20}]'
        df = read_json_data(j)
        self.assertEqual(df['x'].tolist(), [10, 20])

    def test_to_csv_basic(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = to_csv_string(df)
        self.assertIn('x', result)
        self.assertIn('1', result)

    def test_to_csv_no_index(self):
        df = pd.DataFrame({'a': [1]})
        result = to_csv_string(df)
        lines = result.strip().split('\\n')
        self.assertEqual(lines[0], 'a')

    def test_to_json_basic(self):
        df = pd.DataFrame({'x': [1, 2]})
        result = to_json_string(df)
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)

    def test_to_json_format(self):
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = to_json_string(df)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)
        self.assertIn('a', parsed[0])

    def test_roundtrip_csv(self):
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        csv = to_csv_string(df)
        df2 = read_csv_data(csv)
        pd.testing.assert_frame_equal(df, df2)

    def test_roundtrip_json(self):
        df = pd.DataFrame({'x': [1, 2]})
        j = to_json_string(df)
        df2 = read_json_data(j)
        self.assertEqual(df['x'].tolist(), df2['x'].tolist())
`,

	hint1: 'Use pd.read_csv(StringIO(string)) to read from string',
	hint2: 'Use df.to_csv(index=False) and df.to_json(orient="records")',

	whyItMatters: `Data I/O is the first step in any data pipeline:

- **Data loading**: Import data from files, APIs, databases
- **Data export**: Save processed data for sharing or storage
- **Interoperability**: Exchange data between tools and systems
- **Reproducibility**: Save intermediate results

Understanding I/O options (CSV, JSON, Parquet, etc.) is essential.`,

	translations: {
		ru: {
			title: 'Чтение и запись данных',
			description: `# Чтение и запись данных

Pandas может читать и записывать во многие форматы файлов. CSV и JSON — самые распространённые.

## Задача

Реализуйте четыре функции:
1. \`read_csv_data(csv_string)\` - Парсить CSV строку в DataFrame
2. \`read_json_data(json_string)\` - Парсить JSON строку в DataFrame
3. \`to_csv_string(df)\` - Конвертировать DataFrame в CSV строку
4. \`to_json_string(df)\` - Конвертировать DataFrame в JSON строку

## Пример

\`\`\`python
csv_string = "name,age\\nAlice,25\\nBob,30"
read_csv_data(csv_string)
#     name  age
# 0  Alice   25
# 1    Bob   30
\`\`\``,
			hint1: 'Используйте pd.read_csv(StringIO(string)) для чтения из строки',
			hint2: 'Используйте df.to_csv(index=False) и df.to_json(orient="records")',
			whyItMatters: `Ввод/вывод данных — первый шаг в любом data pipeline:

- **Загрузка данных**: Импорт данных из файлов, API, баз данных
- **Экспорт данных**: Сохранение обработанных данных
- **Интероперабельность**: Обмен данными между инструментами
- **Воспроизводимость**: Сохранение промежуточных результатов`,
		},
		uz: {
			title: "Ma'lumotlarni o'qish va yozish",
			description: `# Ma'lumotlarni o'qish va yozish

Pandas ko'p fayl formatlaridan o'qish va yozish mumkin. CSV va JSON eng keng tarqalgan.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`read_csv_data(csv_string)\` - CSV satrini DataFrame ga parse qilish
2. \`read_json_data(json_string)\` - JSON satrini DataFrame ga parse qilish
3. \`to_csv_string(df)\` - DataFrame ni CSV satriga aylantirish
4. \`to_json_string(df)\` - DataFrame ni JSON satriga aylantirish

## Misol

\`\`\`python
csv_string = "name,age\\nAlice,25\\nBob,30"
read_csv_data(csv_string)
#     name  age
# 0  Alice   25
# 1    Bob   30

json_string = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
read_json_data(json_string)

df = pd.DataFrame({'x': [1, 2]})
to_csv_string(df)  # "x\\n1\\n2\\n"
to_json_string(df)  # '[{"x": 1}, {"x": 2}]'
\`\`\``,
			hint1: "Satrdan o'qish uchun pd.read_csv(StringIO(string)) dan foydalaning",
			hint2: "df.to_csv(index=False) va df.to_json(orient='records') dan foydalaning",
			whyItMatters: `Ma'lumotlar kiritish/chiqarish har qanday data pipeline dagi birinchi qadam:

- **Ma'lumotlarni yuklash**: Fayllar, API, ma'lumotlar bazalaridan import
- **Ma'lumotlarni eksport qilish**: Qayta ishlangan ma'lumotlarni saqlash
- **Interoperabellik**: Asboblar va tizimlar o'rtasida ma'lumot almashish`,
		},
	},
};

export default task;
