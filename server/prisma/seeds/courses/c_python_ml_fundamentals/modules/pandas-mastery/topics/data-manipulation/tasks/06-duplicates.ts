import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-duplicates',
	title: 'Handling Duplicates',
	difficulty: 'easy',
	tags: ['pandas', 'duplicates', 'cleaning'],
	estimatedTime: '10m',
	isPremium: false,
	order: 6,
	description: `# Handling Duplicates

Duplicate data can skew analysis and waste resources. Detecting and removing duplicates is essential.

## Task

Implement four functions:
1. \`find_duplicates(df)\` - Return DataFrame of duplicate rows
2. \`count_duplicates(df, subset)\` - Count duplicates based on subset of columns
3. \`drop_duplicates_keep_first(df, subset)\` - Remove duplicates, keeping first occurrence
4. \`drop_duplicates_keep_last(df, subset)\` - Remove duplicates, keeping last occurrence

## Example

\`\`\`python
df = pd.DataFrame({
    'id': [1, 2, 2, 3],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie'],
    'score': [90, 85, 88, 92]
})

find_duplicates(df)  # Row with id=2, name='Bob' (second occurrence)
count_duplicates(df, ['id', 'name'])  # 1
drop_duplicates_keep_first(df, ['id'])  # Keeps first Bob
\`\`\``,

	initialCode: `import pandas as pd

def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of duplicate rows (all occurrences)."""
    # Your code here
    pass

def count_duplicates(df: pd.DataFrame, subset: list) -> int:
    """Count number of duplicate rows based on subset of columns."""
    # Your code here
    pass

def drop_duplicates_keep_first(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Remove duplicates based on subset, keeping first occurrence."""
    # Your code here
    pass

def drop_duplicates_keep_last(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Remove duplicates based on subset, keeping last occurrence."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of duplicate rows (all occurrences)."""
    return df[df.duplicated(keep=False)]

def count_duplicates(df: pd.DataFrame, subset: list) -> int:
    """Count number of duplicate rows based on subset of columns."""
    return df.duplicated(subset=subset).sum()

def drop_duplicates_keep_first(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Remove duplicates based on subset, keeping first occurrence."""
    return df.drop_duplicates(subset=subset, keep='first')

def drop_duplicates_keep_last(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Remove duplicates based on subset, keeping last occurrence."""
    return df.drop_duplicates(subset=subset, keep='last')
`,

	testCode: `import pandas as pd
import unittest

class TestDuplicates(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 2, 3, 3],
            'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
            'score': [90, 85, 88, 92, 95]
        })

    def test_find_duplicates_basic(self):
        result = find_duplicates(self.df)
        self.assertEqual(len(result), 4)  # Both Bob rows and both Charlie rows

    def test_find_duplicates_none(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = find_duplicates(df)
        self.assertEqual(len(result), 0)

    def test_count_duplicates_basic(self):
        count = count_duplicates(self.df, ['id', 'name'])
        self.assertEqual(count, 2)  # 2 duplicate rows

    def test_count_duplicates_single_col(self):
        count = count_duplicates(self.df, ['id'])
        self.assertEqual(count, 2)

    def test_count_duplicates_none(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        count = count_duplicates(df, ['x'])
        self.assertEqual(count, 0)

    def test_drop_keep_first_basic(self):
        result = drop_duplicates_keep_first(self.df, ['id'])
        self.assertEqual(len(result), 3)
        bob_score = result[result['name'] == 'Bob']['score'].iloc[0]
        self.assertEqual(bob_score, 85)  # First Bob

    def test_drop_keep_last_basic(self):
        result = drop_duplicates_keep_last(self.df, ['id'])
        self.assertEqual(len(result), 3)
        bob_score = result[result['name'] == 'Bob']['score'].iloc[0]
        self.assertEqual(bob_score, 88)  # Last Bob

    def test_drop_preserves_non_duplicates(self):
        result = drop_duplicates_keep_first(self.df, ['id'])
        self.assertTrue(any(result['name'] == 'Alice'))

    def test_drop_multiple_columns(self):
        result = drop_duplicates_keep_first(self.df, ['id', 'name'])
        self.assertEqual(len(result), 3)

    def test_find_all_duplicates(self):
        df = pd.DataFrame({'x': [1, 1, 1]})
        result = find_duplicates(df)
        self.assertEqual(len(result), 3)
`,

	hint1: 'Use df.duplicated() to find duplicates, df.drop_duplicates() to remove',
	hint2: 'Use keep=False in duplicated() to mark all duplicates, not just subsequent ones',

	whyItMatters: `Duplicate handling is critical for:

- **Data quality**: Ensure each record is unique
- **Accurate analysis**: Prevent double-counting in aggregations
- **Storage efficiency**: Remove redundant data
- **Model training**: Avoid biased training from repeated samples

Deduplication is a standard step in data preprocessing.`,

	translations: {
		ru: {
			title: 'Обработка дубликатов',
			description: `# Обработка дубликатов

Дублирующиеся данные могут исказить анализ и тратить ресурсы. Обнаружение и удаление дубликатов необходимо.

## Задача

Реализуйте четыре функции:
1. \`find_duplicates(df)\` - Вернуть DataFrame дублирующихся строк
2. \`count_duplicates(df, subset)\` - Посчитать дубликаты по подмножеству столбцов
3. \`drop_duplicates_keep_first(df, subset)\` - Удалить дубликаты, оставив первое вхождение
4. \`drop_duplicates_keep_last(df, subset)\` - Удалить дубликаты, оставив последнее вхождение

## Пример

\`\`\`python
df = pd.DataFrame({
    'id': [1, 2, 2, 3],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie'],
    'score': [90, 85, 88, 92]
})

find_duplicates(df)  # Row with id=2, name='Bob' (second occurrence)
count_duplicates(df, ['id', 'name'])  # 1
drop_duplicates_keep_first(df, ['id'])  # Keeps first Bob
\`\`\``,
			hint1: 'Используйте df.duplicated() для поиска, df.drop_duplicates() для удаления',
			hint2: 'Используйте keep=False в duplicated() чтобы отметить все дубликаты',
			whyItMatters: `Обработка дубликатов критична для:

- **Качество данных**: Обеспечение уникальности каждой записи
- **Точный анализ**: Предотвращение двойного счёта в агрегациях
- **Эффективность хранения**: Удаление избыточных данных
- **Обучение модели**: Избежание смещения от повторных сэмплов`,
		},
		uz: {
			title: "Dublikatlarni boshqarish",
			description: `# Dublikatlarni boshqarish

Takroriy ma'lumotlar tahlilni buzishi va resurslarni sarflashi mumkin. Dublikatlarni aniqlash va olib tashlash zarur.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`find_duplicates(df)\` - Dublikat qatorlar DataFrame ni qaytarish
2. \`count_duplicates(df, subset)\` - Ustunlar quyi to'plami asosida dublikatlarni hisoblash
3. \`drop_duplicates_keep_first(df, subset)\` - Dublikatlarni olib tashlash, birinchisini saqlash
4. \`drop_duplicates_keep_last(df, subset)\` - Dublikatlarni olib tashlash, oxirgisini saqlash

## Misol

\`\`\`python
df = pd.DataFrame({
    'id': [1, 2, 2, 3],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie'],
    'score': [90, 85, 88, 92]
})

find_duplicates(df)  # Row with id=2, name='Bob' (second occurrence)
count_duplicates(df, ['id', 'name'])  # 1
drop_duplicates_keep_first(df, ['id'])  # Keeps first Bob
\`\`\``,
			hint1: "Topish uchun df.duplicated(), olib tashlash uchun df.drop_duplicates() dan foydalaning",
			hint2: "Barcha dublikatlarni belgilash uchun duplicated() da keep=False dan foydalaning",
			whyItMatters: `Dublikatlarni boshqarish quyidagilar uchun muhim:

- **Ma'lumotlar sifati**: Har bir yozuv noyobligini ta'minlash
- **Aniq tahlil**: Agregatsiyalarda ikki marta hisoblashni oldini olish
- **Saqlash samaradorligi**: Ortiqcha ma'lumotlarni olib tashlash`,
		},
	},
};

export default task;
