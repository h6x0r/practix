import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-group-by',
	title: 'Group By Key',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'dictionaries', 'grouping'],
	estimatedTime: '15m',
	isPremium: false,
	order: 11,

	description: `# Group By Key

Group a list of items by a specified key.

## Task

Implement the function \`group_by(items, key)\` that groups a list of dictionaries by a specified key.

## Requirements

- \`items\` is a list of dictionaries
- \`key\` is a string representing the dictionary key to group by
- Return a dictionary where keys are unique values of the grouping key
- Each value is a list of items with that key value
- Items missing the key should be grouped under \`None\`

## Examples

\`\`\`python
>>> students = [
...     {"name": "Alice", "grade": "A"},
...     {"name": "Bob", "grade": "B"},
...     {"name": "Charlie", "grade": "A"},
... ]
>>> group_by(students, "grade")
{
    "A": [{"name": "Alice", "grade": "A"}, {"name": "Charlie", "grade": "A"}],
    "B": [{"name": "Bob", "grade": "B"}]
}
\`\`\``,

	initialCode: `def group_by(items: list[dict], key: str) -> dict:
    """Group a list of dictionaries by a specified key.

    Args:
        items: List of dictionaries to group
        key: The dictionary key to group by

    Returns:
        Dictionary with grouped items
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def group_by(items: list[dict], key: str) -> dict:
    """Group a list of dictionaries by a specified key.

    Args:
        items: List of dictionaries to group
        key: The dictionary key to group by

    Returns:
        Dictionary with grouped items
    """
    result = {}

    for item in items:
        # Get the value for the grouping key
        # Use None if the key doesn't exist
        group_key = item.get(key, None)

        # Initialize list for this group if not exists
        if group_key not in result:
            result[group_key] = []

        # Add item to its group
        result[group_key].append(item)

    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic grouping"""
        students = [
            {"name": "Alice", "grade": "A"},
            {"name": "Bob", "grade": "B"},
            {"name": "Charlie", "grade": "A"},
        ]
        result = group_by(students, "grade")
        self.assertEqual(len(result["A"]), 2)
        self.assertEqual(len(result["B"]), 1)

    def test_2(self):
        """Single group"""
        items = [{"x": 1}, {"x": 1}, {"x": 1}]
        result = group_by(items, "x")
        self.assertEqual(len(result[1]), 3)

    def test_3(self):
        """Empty list"""
        self.assertEqual(group_by([], "any"), {})

    def test_4(self):
        """Missing key groups under None"""
        items = [{"a": 1}, {"b": 2}, {"a": 3}]
        result = group_by(items, "a")
        self.assertEqual(len(result.get(None, [])), 1)

    def test_5(self):
        """Numbers as group keys"""
        items = [{"score": 90}, {"score": 80}, {"score": 90}]
        result = group_by(items, "score")
        self.assertEqual(len(result[90]), 2)

    def test_6(self):
        """All items have unique keys"""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = group_by(items, "id")
        self.assertEqual(len(result), 3)

    def test_7(self):
        """Boolean group keys"""
        items = [{"active": True}, {"active": False}, {"active": True}]
        result = group_by(items, "active")
        self.assertEqual(len(result[True]), 2)

    def test_8(self):
        """Group by nested-friendly key"""
        items = [{"type": "user"}, {"type": "admin"}, {"type": "user"}]
        result = group_by(items, "type")
        self.assertEqual(len(result["user"]), 2)

    def test_9(self):
        """Items preserved correctly"""
        items = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
        result = group_by(items, "name")
        self.assertEqual(result["A"][0]["value"], 1)

    def test_10(self):
        """All missing key"""
        items = [{"x": 1}, {"x": 2}]
        result = group_by(items, "y")
        self.assertEqual(len(result.get(None, [])), 2)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use item.get(key, None) to safely get the grouping key value, handling missing keys.',
	hint2: 'Check if the group exists in result before appending. Use: if group_key not in result: result[group_key] = []',

	whyItMatters: `Grouping data is one of the most common operations in data processing and analytics.

**Production Pattern:**

\`\`\`python
from collections import defaultdict
from itertools import groupby as itertools_groupby

def group_by_efficient(items: list[dict], key: str) -> dict:
    """Efficient grouping using defaultdict."""
    result = defaultdict(list)
    for item in items:
        result[item.get(key)].append(item)
    return dict(result)

def group_by_multiple_keys(items: list[dict], keys: list[str]) -> dict:
    """Group by multiple keys (creates nested structure)."""
    if not keys:
        return items

    result = defaultdict(list)
    for item in items:
        key_value = item.get(keys[0])
        result[key_value].append(item)

    if len(keys) > 1:
        for key_value in result:
            result[key_value] = group_by_multiple_keys(result[key_value], keys[1:])

    return dict(result)

def aggregate_groups(items: list[dict], group_key: str, agg_key: str) -> dict:
    """Group and aggregate (like SQL GROUP BY with SUM)."""
    groups = defaultdict(list)
    for item in items:
        groups[item.get(group_key)].append(item.get(agg_key, 0))

    return {k: sum(v) for k, v in groups.items()}

# Example: Sum sales by category
# aggregate_groups(sales, "category", "amount")
\`\`\`

**Practical Benefits:**
- Data aggregation for reports and dashboards
- Organizing API responses by category
- Preparing data for charts and visualizations`,

	translations: {
		ru: {
			title: 'Группировка по ключу',
			description: `# Группировка по ключу

Группируйте список элементов по указанному ключу.

## Задача

Реализуйте функцию \`group_by(items, key)\`, которая группирует список словарей по указанному ключу.

## Требования

- \`items\` — список словарей
- \`key\` — строка, представляющая ключ для группировки
- Верните словарь, где ключи — уникальные значения группирующего ключа
- Каждое значение — список элементов с этим значением ключа
- Элементы без ключа группируются под \`None\`

## Примеры

\`\`\`python
>>> students = [
...     {"name": "Alice", "grade": "A"},
...     {"name": "Bob", "grade": "B"},
...     {"name": "Charlie", "grade": "A"},
... ]
>>> group_by(students, "grade")
{
    "A": [{"name": "Alice", "grade": "A"}, {"name": "Charlie", "grade": "A"}],
    "B": [{"name": "Bob", "grade": "B"}]
}
\`\`\``,
			hint1: 'Используйте item.get(key, None) для безопасного получения значения ключа.',
			hint2: 'Проверьте существование группы перед добавлением: if group_key not in result: result[group_key] = []',
			whyItMatters: `Группировка данных — одна из самых частых операций в обработке данных.

**Продакшен паттерн:**

\`\`\`python
from collections import defaultdict

def group_by_efficient(items: list[dict], key: str) -> dict:
    """Эффективная группировка через defaultdict."""
    result = defaultdict(list)
    for item in items:
        result[item.get(key)].append(item)
    return dict(result)

def aggregate_groups(items: list[dict], group_key: str, agg_key: str) -> dict:
    """Группировка с агрегацией (как SQL GROUP BY + SUM)."""
    groups = defaultdict(list)
    for item in items:
        groups[item.get(group_key)].append(item.get(agg_key, 0))
    return {k: sum(v) for k, v in groups.items()}
\`\`\`

**Практические преимущества:**
- Агрегация данных для отчётов и дашбордов
- Организация ответов API по категориям`,
		},
		uz: {
			title: "Kalit bo'yicha guruhlash",
			description: `# Kalit bo'yicha guruhlash

Elementlar ro'yxatini belgilangan kalit bo'yicha guruhlang.

## Vazifa

Lug'atlar ro'yxatini belgilangan kalit bo'yicha guruhlash uchun \`group_by(items, key)\` funksiyasini amalga oshiring.

## Talablar

- \`items\` — lug'atlar ro'yxati
- \`key\` — guruhlash kalitini ifodalovchi satr
- Kalitlar guruhlash kalitining noyob qiymatlari bo'lgan lug'at qaytaring
- Har bir qiymat o'sha kalit qiymatiga ega elementlar ro'yxati
- Kaliti yo'q elementlar \`None\` ostida guruhlanadi

## Misollar

\`\`\`python
>>> students = [
...     {"name": "Alice", "grade": "A"},
...     {"name": "Bob", "grade": "B"},
...     {"name": "Charlie", "grade": "A"},
... ]
>>> group_by(students, "grade")
{
    "A": [{"name": "Alice", "grade": "A"}, {"name": "Charlie", "grade": "A"}],
    "B": [{"name": "Bob", "grade": "B"}]
}
\`\`\``,
			hint1: "Kalit qiymatini xavfsiz olish uchun item.get(key, None) ishlating.",
			hint2: "Qo'shishdan oldin guruh mavjudligini tekshiring: if group_key not in result: result[group_key] = []",
			whyItMatters: `Ma'lumotlarni guruhlash ma'lumotlarni qayta ishlashda eng keng tarqalgan amallardan biri.

**Ishlab chiqarish patterni:**

\`\`\`python
from collections import defaultdict

def group_by_efficient(items: list[dict], key: str) -> dict:
    """defaultdict orqali samarali guruhlash."""
    result = defaultdict(list)
    for item in items:
        result[item.get(key)].append(item)
    return dict(result)

def aggregate_groups(items: list[dict], group_key: str, agg_key: str) -> dict:
    """Agregatsiya bilan guruhlash (SQL GROUP BY + SUM kabi)."""
    groups = defaultdict(list)
    for item in items:
        groups[item.get(group_key)].append(item.get(agg_key, 0))
    return {k: sum(v) for k, v in groups.items()}
\`\`\`

**Amaliy foydalari:**
- Hisobotlar va dashboardlar uchun ma'lumotlarni jamlash
- API javoblarini kategoriya bo'yicha tashkil etish`,
		},
	},
};

export default task;
