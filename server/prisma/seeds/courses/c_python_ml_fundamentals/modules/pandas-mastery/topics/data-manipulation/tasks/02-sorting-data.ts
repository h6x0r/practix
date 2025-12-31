import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-sorting-data',
	title: 'Sorting Data',
	difficulty: 'easy',
	tags: ['pandas', 'sorting', 'ordering'],
	estimatedTime: '10m',
	isPremium: false,
	order: 2,
	description: `# Sorting Data

Sorting organizes data by one or more columns - essential for analysis and presentation.

## Task

Implement three functions:
1. \`sort_by_column(df, column, ascending)\` - Sort by single column
2. \`sort_by_multiple(df, columns, ascending_list)\` - Sort by multiple columns
3. \`get_top_n(df, column, n)\` - Get top n rows by column value

## Example

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [30, 25, 35],
    'score': [85, 90, 85]
})

sort_by_column(df, 'age', True)  # Bob, Alice, Charlie (ascending)
sort_by_multiple(df, ['score', 'age'], [False, True])  # By score desc, then age asc
get_top_n(df, 'age', 2)  # Charlie, Alice (oldest 2)
\`\`\``,

	initialCode: `import pandas as pd

def sort_by_column(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort by single column."""
    # Your code here
    pass

def sort_by_multiple(df: pd.DataFrame, columns: list, ascending_list: list) -> pd.DataFrame:
    """Sort by multiple columns with different sort orders."""
    # Your code here
    pass

def get_top_n(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    """Get top n rows by column value (highest first)."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def sort_by_column(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort by single column."""
    return df.sort_values(by=column, ascending=ascending)

def sort_by_multiple(df: pd.DataFrame, columns: list, ascending_list: list) -> pd.DataFrame:
    """Sort by multiple columns with different sort orders."""
    return df.sort_values(by=columns, ascending=ascending_list)

def get_top_n(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    """Get top n rows by column value (highest first)."""
    return df.nlargest(n, column)
`,

	testCode: `import pandas as pd
import unittest

class TestSortingData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [30, 25, 35, 28],
            'score': [85, 90, 85, 92]
        })

    def test_sort_by_column_asc(self):
        result = sort_by_column(self.df, 'age', True)
        self.assertEqual(result['age'].iloc[0], 25)
        self.assertEqual(result['age'].iloc[-1], 35)

    def test_sort_by_column_desc(self):
        result = sort_by_column(self.df, 'age', False)
        self.assertEqual(result['age'].iloc[0], 35)
        self.assertEqual(result['age'].iloc[-1], 25)

    def test_sort_by_column_string(self):
        result = sort_by_column(self.df, 'name', True)
        self.assertEqual(result['name'].iloc[0], 'Alice')

    def test_sort_by_multiple_basic(self):
        result = sort_by_multiple(self.df, ['score', 'age'], [False, True])
        self.assertEqual(result['name'].iloc[0], 'Diana')  # Highest score

    def test_sort_by_multiple_tiebreak(self):
        result = sort_by_multiple(self.df, ['score', 'age'], [True, True])
        # Score 85: Alice(30) and Charlie(35), Alice comes first (younger)
        scores_85 = result[result['score'] == 85]
        self.assertEqual(scores_85['name'].iloc[0], 'Alice')

    def test_get_top_n_basic(self):
        result = get_top_n(self.df, 'age', 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['age'].iloc[0], 35)

    def test_get_top_n_score(self):
        result = get_top_n(self.df, 'score', 1)
        self.assertEqual(result['name'].iloc[0], 'Diana')

    def test_sort_preserves_data(self):
        result = sort_by_column(self.df, 'age', True)
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result['name']), set(self.df['name']))

    def test_get_top_n_all(self):
        result = get_top_n(self.df, 'age', 4)
        self.assertEqual(len(result), 4)

    def test_sort_no_modify_original(self):
        original_first = self.df['age'].iloc[0]
        _ = sort_by_column(self.df, 'age', True)
        self.assertEqual(self.df['age'].iloc[0], original_first)
`,

	hint1: 'Use df.sort_values(by=column, ascending=True/False)',
	hint2: 'Use df.nlargest(n, column) for top n by value',

	whyItMatters: `Sorting is fundamental for:

- **Data presentation**: Display rankings, leaderboards
- **Data analysis**: Find top/bottom performers
- **Time series**: Order by date for temporal analysis
- **Deduplication**: Sort before removing duplicates

Efficient sorting algorithms are built into Pandas.`,

	translations: {
		ru: {
			title: 'Сортировка данных',
			description: `# Сортировка данных

Сортировка организует данные по одному или нескольким столбцам — необходимо для анализа и презентации.

## Задача

Реализуйте три функции:
1. \`sort_by_column(df, column, ascending)\` - Сортировка по одному столбцу
2. \`sort_by_multiple(df, columns, ascending_list)\` - Сортировка по нескольким столбцам
3. \`get_top_n(df, column, n)\` - Получить top n строк по значению столбца

## Пример

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [30, 25, 35],
    'score': [85, 90, 85]
})

sort_by_column(df, 'age', True)  # Bob, Alice, Charlie (ascending)
sort_by_multiple(df, ['score', 'age'], [False, True])  # By score desc, then age asc
get_top_n(df, 'age', 2)  # Charlie, Alice (oldest 2)
\`\`\``,
			hint1: 'Используйте df.sort_values(by=column, ascending=True/False)',
			hint2: 'Используйте df.nlargest(n, column) для top n по значению',
			whyItMatters: `Сортировка фундаментальна для:

- **Презентация данных**: Отображение рейтингов, лидербордов
- **Анализ данных**: Поиск лучших/худших
- **Временные ряды**: Упорядочивание по дате
- **Дедупликация**: Сортировка перед удалением дубликатов`,
		},
		uz: {
			title: "Ma'lumotlarni saralash",
			description: `# Ma'lumotlarni saralash

Saralash ma'lumotlarni bir yoki bir nechta ustunlar bo'yicha tartibga soladi - tahlil va taqdimot uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`sort_by_column(df, column, ascending)\` - Bitta ustun bo'yicha saralash
2. \`sort_by_multiple(df, columns, ascending_list)\` - Bir nechta ustunlar bo'yicha saralash
3. \`get_top_n(df, column, n)\` - Ustun qiymati bo'yicha eng yuqori n qatorni olish

## Misol

\`\`\`python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [30, 25, 35],
    'score': [85, 90, 85]
})

sort_by_column(df, 'age', True)  # Bob, Alice, Charlie (ascending)
sort_by_multiple(df, ['score', 'age'], [False, True])  # By score desc, then age asc
get_top_n(df, 'age', 2)  # Charlie, Alice (oldest 2)
\`\`\``,
			hint1: "df.sort_values(by=column, ascending=True/False) dan foydalaning",
			hint2: "Qiymat bo'yicha top n uchun df.nlargest(n, column) dan foydalaning",
			whyItMatters: `Saralash quyidagilar uchun asosiydir:

- **Ma'lumotlarni taqdim etish**: Reytinglar, liderlik jadvallari
- **Ma'lumotlar tahlili**: Eng yaxshi/yomon natijalarni topish
- **Vaqt qatorlari**: Vaqtinchalik tahlil uchun sana bo'yicha tartiblash`,
		},
	},
};

export default task;
