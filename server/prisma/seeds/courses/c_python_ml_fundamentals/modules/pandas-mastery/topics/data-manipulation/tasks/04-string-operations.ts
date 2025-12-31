import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-string-operations',
	title: 'String Operations',
	difficulty: 'medium',
	tags: ['pandas', 'strings', 'text'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# String Operations

Pandas provides vectorized string operations through the .str accessor.

## Task

Implement four functions:
1. \`to_lowercase(df, column)\` - Convert column to lowercase
2. \`extract_pattern(df, column, pattern)\` - Extract regex pattern matches
3. \`contains_substring(df, column, substring)\` - Filter rows containing substring
4. \`split_column(df, column, delimiter)\` - Split column into multiple columns

## Example

\`\`\`python
df = pd.DataFrame({'email': ['Alice@GMAIL.COM', 'Bob@Yahoo.com']})

to_lowercase(df, 'email')  # ['alice@gmail.com', 'bob@yahoo.com']

df = pd.DataFrame({'text': ['ID: 123', 'ID: 456']})
extract_pattern(df, 'text', r'\\d+')  # ['123', '456']

df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
split_column(df, 'name', ' ')  # Creates 'name_0' and 'name_1' columns
\`\`\``,

	initialCode: `import pandas as pd

def to_lowercase(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to lowercase."""
    # Your code here
    pass

def extract_pattern(df: pd.DataFrame, column: str, pattern: str) -> pd.DataFrame:
    """Extract first regex pattern match as new column 'extracted'."""
    # Your code here
    pass

def contains_substring(df: pd.DataFrame, column: str, substring: str) -> pd.DataFrame:
    """Filter rows where column contains substring (case-insensitive)."""
    # Your code here
    pass

def split_column(df: pd.DataFrame, column: str, delimiter: str) -> pd.DataFrame:
    """Split column into multiple columns named column_0, column_1, etc."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def to_lowercase(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert column to lowercase."""
    df = df.copy()
    df[column] = df[column].str.lower()
    return df

def extract_pattern(df: pd.DataFrame, column: str, pattern: str) -> pd.DataFrame:
    """Extract first regex pattern match as new column 'extracted'."""
    df = df.copy()
    df['extracted'] = df[column].str.extract(f'({pattern})', expand=False)
    return df

def contains_substring(df: pd.DataFrame, column: str, substring: str) -> pd.DataFrame:
    """Filter rows where column contains substring (case-insensitive)."""
    return df[df[column].str.contains(substring, case=False, na=False)]

def split_column(df: pd.DataFrame, column: str, delimiter: str) -> pd.DataFrame:
    """Split column into multiple columns named column_0, column_1, etc."""
    df = df.copy()
    splits = df[column].str.split(delimiter, expand=True)
    splits.columns = [f'{column}_{i}' for i in range(splits.shape[1])]
    return pd.concat([df, splits], axis=1)
`,

	testCode: `import pandas as pd
import unittest

class TestStringOperations(unittest.TestCase):
    def test_to_lowercase_basic(self):
        df = pd.DataFrame({'text': ['HELLO', 'World', 'PyThOn']})
        result = to_lowercase(df, 'text')
        self.assertEqual(result['text'].tolist(), ['hello', 'world', 'python'])

    def test_to_lowercase_mixed(self):
        df = pd.DataFrame({'x': ['ABC123', 'def456']})
        result = to_lowercase(df, 'x')
        self.assertEqual(result['x'].iloc[0], 'abc123')

    def test_extract_pattern_numbers(self):
        df = pd.DataFrame({'text': ['ID: 123', 'ID: 456']})
        result = extract_pattern(df, 'text', r'\\d+')
        self.assertEqual(result['extracted'].tolist(), ['123', '456'])

    def test_extract_pattern_letters(self):
        df = pd.DataFrame({'text': ['abc123', 'def456']})
        result = extract_pattern(df, 'text', r'[a-z]+')
        self.assertEqual(result['extracted'].tolist(), ['abc', 'def'])

    def test_contains_substring_basic(self):
        df = pd.DataFrame({'text': ['hello world', 'goodbye', 'hello there']})
        result = contains_substring(df, 'text', 'hello')
        self.assertEqual(len(result), 2)

    def test_contains_substring_case_insensitive(self):
        df = pd.DataFrame({'text': ['HELLO', 'hello', 'hi']})
        result = contains_substring(df, 'text', 'hello')
        self.assertEqual(len(result), 2)

    def test_split_column_basic(self):
        df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
        result = split_column(df, 'name', ' ')
        self.assertIn('name_0', result.columns)
        self.assertIn('name_1', result.columns)
        self.assertEqual(result['name_0'].tolist(), ['John', 'Jane'])

    def test_split_column_multiple(self):
        df = pd.DataFrame({'path': ['a/b/c', 'x/y/z']})
        result = split_column(df, 'path', '/')
        self.assertEqual(result['path_2'].tolist(), ['c', 'z'])

    def test_no_modify_original_lowercase(self):
        df = pd.DataFrame({'text': ['HELLO']})
        _ = to_lowercase(df, 'text')
        self.assertEqual(df['text'].iloc[0], 'HELLO')

    def test_contains_no_match(self):
        df = pd.DataFrame({'text': ['abc', 'def']})
        result = contains_substring(df, 'text', 'xyz')
        self.assertEqual(len(result), 0)
`,

	hint1: 'Use df["column"].str.lower(), .str.contains(), etc.',
	hint2: 'Use .str.extract(pattern) for regex extraction, .str.split(expand=True) for splitting',

	whyItMatters: `String operations are essential for:

- **Text preprocessing**: Clean text data for NLP
- **Feature extraction**: Extract patterns from unstructured text
- **Data standardization**: Normalize formats (emails, names)
- **Parsing**: Extract structured data from strings

Most real-world data contains text that needs processing.`,

	translations: {
		ru: {
			title: 'Операции со строками',
			description: `# Операции со строками

Pandas предоставляет векторизованные строковые операции через аксессор .str.

## Задача

Реализуйте четыре функции:
1. \`to_lowercase(df, column)\` - Преобразовать столбец в нижний регистр
2. \`extract_pattern(df, column, pattern)\` - Извлечь совпадения regex паттерна
3. \`contains_substring(df, column, substring)\` - Фильтр строк содержащих подстроку
4. \`split_column(df, column, delimiter)\` - Разделить столбец на несколько столбцов

## Пример

\`\`\`python
df = pd.DataFrame({'email': ['Alice@GMAIL.COM', 'Bob@Yahoo.com']})

to_lowercase(df, 'email')  # ['alice@gmail.com', 'bob@yahoo.com']

df = pd.DataFrame({'text': ['ID: 123', 'ID: 456']})
extract_pattern(df, 'text', r'\\d+')  # ['123', '456']

df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
split_column(df, 'name', ' ')  # Creates 'name_0' and 'name_1' columns
\`\`\``,
			hint1: 'Используйте df["column"].str.lower(), .str.contains() и т.д.',
			hint2: 'Используйте .str.extract(pattern) для regex, .str.split(expand=True) для разделения',
			whyItMatters: `Операции со строками необходимы для:

- **Предобработка текста**: Очистка текстовых данных для NLP
- **Извлечение признаков**: Извлечение паттернов из неструктурированного текста
- **Стандартизация данных**: Нормализация форматов (email, имена)
- **Парсинг**: Извлечение структурированных данных из строк`,
		},
		uz: {
			title: "Satr operatsiyalari",
			description: `# Satr operatsiyalari

Pandas .str aksessori orqali vektorlashtirilgan satr operatsiyalarini taqdim etadi.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`to_lowercase(df, column)\` - Ustunni kichik harflarga aylantirish
2. \`extract_pattern(df, column, pattern)\` - Regex pattern mosliklarini ajratib olish
3. \`contains_substring(df, column, substring)\` - Quyi satrni o'z ichiga olgan qatorlarni filtrlash
4. \`split_column(df, column, delimiter)\` - Ustunni bir nechta ustunlarga bo'lish

## Misol

\`\`\`python
df = pd.DataFrame({'email': ['Alice@GMAIL.COM', 'Bob@Yahoo.com']})

to_lowercase(df, 'email')  # ['alice@gmail.com', 'bob@yahoo.com']

df = pd.DataFrame({'text': ['ID: 123', 'ID: 456']})
extract_pattern(df, 'text', r'\\d+')  # ['123', '456']

df = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
split_column(df, 'name', ' ')  # Creates 'name_0' and 'name_1' columns
\`\`\``,
			hint1: "df['column'].str.lower(), .str.contains() va boshqalardan foydalaning",
			hint2: "regex uchun .str.extract(pattern), bo'lish uchun .str.split(expand=True) dan foydalaning",
			whyItMatters: `Satr operatsiyalari quyidagilar uchun zarur:

- **Matn oldindan qayta ishlash**: NLP uchun matn ma'lumotlarini tozalash
- **Xususiyatlarni ajratib olish**: Strukturalanmagan matndan patternlarni ajratib olish
- **Ma'lumotlarni standartlashtirish**: Formatlarni normalizatsiya qilish`,
		},
	},
};

export default task;
