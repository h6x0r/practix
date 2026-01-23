import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-set-operations',
	title: 'Set Operations',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'sets', 'collections'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,

	description: `# Set Operations

Sets are unordered collections of unique elements with powerful operations.

## Task

Implement the function \`analyze_lists(list_a, list_b)\` that returns a dictionary with set operations on two lists.

## Requirements

Return a dictionary with:
- \`"union"\`: All unique elements from both lists
- \`"intersection"\`: Elements that appear in both lists
- \`"only_in_a"\`: Elements only in list_a
- \`"only_in_b"\`: Elements only in list_b
- \`"symmetric_diff"\`: Elements in either but not both

All values should be sorted lists.

## Examples

\`\`\`python
>>> analyze_lists([1, 2, 3], [2, 3, 4])
{
    "union": [1, 2, 3, 4],
    "intersection": [2, 3],
    "only_in_a": [1],
    "only_in_b": [4],
    "symmetric_diff": [1, 4]
}
\`\`\``,

	initialCode: `def analyze_lists(list_a: list, list_b: list) -> dict:
    """Perform set operations on two lists.

    Args:
        list_a: First list
        list_b: Second list

    Returns:
        Dictionary with union, intersection, only_in_a,
        only_in_b, and symmetric_diff as sorted lists
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def analyze_lists(list_a: list, list_b: list) -> dict:
    """Perform set operations on two lists.

    Args:
        list_a: First list
        list_b: Second list

    Returns:
        Dictionary with union, intersection, only_in_a,
        only_in_b, and symmetric_diff as sorted lists
    """
    # Convert lists to sets for efficient operations
    set_a = set(list_a)
    set_b = set(list_b)

    # Perform set operations and convert results to sorted lists
    return {
        # Union: all unique elements from both sets
        "union": sorted(set_a | set_b),

        # Intersection: elements in both sets
        "intersection": sorted(set_a & set_b),

        # Difference: elements only in set_a
        "only_in_a": sorted(set_a - set_b),

        # Difference: elements only in set_b
        "only_in_b": sorted(set_b - set_a),

        # Symmetric difference: elements in either but not both
        "symmetric_diff": sorted(set_a ^ set_b),
    }`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic overlapping lists"""
        result = analyze_lists([1, 2, 3], [2, 3, 4])
        self.assertEqual(result["union"], [1, 2, 3, 4])
        self.assertEqual(result["intersection"], [2, 3])
        self.assertEqual(result["only_in_a"], [1])
        self.assertEqual(result["only_in_b"], [4])
        self.assertEqual(result["symmetric_diff"], [1, 4])

    def test_2(self):
        """No overlap"""
        result = analyze_lists([1, 2], [3, 4])
        self.assertEqual(result["intersection"], [])
        self.assertEqual(result["union"], [1, 2, 3, 4])

    def test_3(self):
        """Complete overlap"""
        result = analyze_lists([1, 2, 3], [1, 2, 3])
        self.assertEqual(result["intersection"], [1, 2, 3])
        self.assertEqual(result["only_in_a"], [])
        self.assertEqual(result["only_in_b"], [])

    def test_4(self):
        """Empty first list"""
        result = analyze_lists([], [1, 2])
        self.assertEqual(result["union"], [1, 2])
        self.assertEqual(result["only_in_a"], [])

    def test_5(self):
        """Empty second list"""
        result = analyze_lists([1, 2], [])
        self.assertEqual(result["union"], [1, 2])
        self.assertEqual(result["only_in_b"], [])

    def test_6(self):
        """Both empty"""
        result = analyze_lists([], [])
        self.assertEqual(result["union"], [])

    def test_7(self):
        """With duplicates in input"""
        result = analyze_lists([1, 1, 2, 2], [2, 2, 3, 3])
        self.assertEqual(result["union"], [1, 2, 3])

    def test_8(self):
        """Strings"""
        result = analyze_lists(["a", "b"], ["b", "c"])
        self.assertEqual(result["intersection"], ["b"])

    def test_9(self):
        """Subset relationship"""
        result = analyze_lists([1, 2, 3, 4], [2, 3])
        self.assertEqual(result["only_in_b"], [])

    def test_10(self):
        """Single element lists"""
        result = analyze_lists([1], [2])
        self.assertEqual(result["symmetric_diff"], [1, 2])

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Convert lists to sets first: set_a = set(list_a). Then use set operators: | for union, & for intersection.',
	hint2: 'Use set_a - set_b for difference, and set_a ^ set_b for symmetric difference. Convert results back to sorted lists.',

	whyItMatters: `Set operations are essential for data analysis, deduplication, and comparison.

**Production Pattern:**

\`\`\`python
def find_permission_changes(
    old_perms: list[str],
    new_perms: list[str]
) -> dict[str, list[str]]:
    """Find what permissions were added, removed, or kept."""
    old_set = set(old_perms)
    new_set = set(new_perms)

    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
        "unchanged": sorted(old_set & new_set),
    }

def find_common_tags(articles: list[dict]) -> set[str]:
    """Find tags that appear in ALL articles."""
    if not articles:
        return set()

    # Start with first article's tags
    common = set(articles[0].get("tags", []))

    # Intersect with each subsequent article
    for article in articles[1:]:
        common &= set(article.get("tags", []))

    return common

def deduplicate_preserve_order(items: list) -> list:
    """Remove duplicates while preserving original order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
\`\`\`

**Practical Benefits:**
- Permission systems use set operations for access control
- Finding common/different features across datasets
- Efficient membership testing O(1) average`,

	translations: {
		ru: {
			title: 'Операции с множествами',
			description: `# Операции с множествами

Множества — неупорядоченные коллекции уникальных элементов с мощными операциями.

## Задача

Реализуйте функцию \`analyze_lists(list_a, list_b)\`, которая возвращает словарь с операциями множеств над двумя списками.

## Требования

Верните словарь с:
- \`"union"\`: Все уникальные элементы из обоих списков
- \`"intersection"\`: Элементы, присутствующие в обоих списках
- \`"only_in_a"\`: Элементы только в list_a
- \`"only_in_b"\`: Элементы только в list_b
- \`"symmetric_diff"\`: Элементы в одном, но не в обоих

Все значения должны быть отсортированными списками.

## Примеры

\`\`\`python
>>> analyze_lists([1, 2, 3], [2, 3, 4])
{
    "union": [1, 2, 3, 4],
    "intersection": [2, 3],
    "only_in_a": [1],
    "only_in_b": [4],
    "symmetric_diff": [1, 4]
}
\`\`\``,
			hint1: 'Сначала преобразуйте списки во множества: set_a = set(list_a). Используйте операторы: | для объединения, & для пересечения.',
			hint2: 'Используйте set_a - set_b для разности и set_a ^ set_b для симметричной разности.',
			whyItMatters: `Операции с множествами необходимы для анализа данных и сравнений.

**Продакшен паттерн:**

\`\`\`python
def find_permission_changes(
    old_perms: list[str],
    new_perms: list[str]
) -> dict[str, list[str]]:
    """Найти добавленные, удалённые и сохранённые права."""
    old_set = set(old_perms)
    new_set = set(new_perms)

    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
        "unchanged": sorted(old_set & new_set),
    }
\`\`\`

**Практические преимущества:**
- Системы прав используют операции множеств
- Поиск общих/различных признаков в данных
- Эффективная проверка принадлежности O(1)`,
		},
		uz: {
			title: "To'plam amallari",
			description: `# To'plam amallari

To'plamlar — kuchli amallarga ega tartibsiz noyob elementlar to'plami.

## Vazifa

Ikki ro'yxat ustida to'plam amallarini qaytaruvchi \`analyze_lists(list_a, list_b)\` funksiyasini amalga oshiring.

## Talablar

Quyidagilarga ega lug'at qaytaring:
- \`"union"\`: Ikkala ro'yxatdagi barcha noyob elementlar
- \`"intersection"\`: Ikkala ro'yxatda ham mavjud elementlar
- \`"only_in_a"\`: Faqat list_a dagi elementlar
- \`"only_in_b"\`: Faqat list_b dagi elementlar
- \`"symmetric_diff"\`: Birida bor, lekin ikkalasida ham yo'q elementlar

Barcha qiymatlar tartiblangan ro'yxatlar bo'lishi kerak.

## Misollar

\`\`\`python
>>> analyze_lists([1, 2, 3], [2, 3, 4])
{
    "union": [1, 2, 3, 4],
    "intersection": [2, 3],
    "only_in_a": [1],
    "only_in_b": [4],
    "symmetric_diff": [1, 4]
}
\`\`\``,
			hint1: "Avval ro'yxatlarni to'plamlarga aylantiring: set_a = set(list_a). Operatorlardan foydalaning: | birlashma, & kesishma uchun.",
			hint2: "Farq uchun set_a - set_b, simmetrik farq uchun set_a ^ set_b ishlating.",
			whyItMatters: `To'plam amallari ma'lumotlarni tahlil qilish va taqqoslash uchun zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
def find_permission_changes(
    old_perms: list[str],
    new_perms: list[str]
) -> dict[str, list[str]]:
    """Qo'shilgan, olib tashlangan va saqlanib qolgan ruxsatlarni topish."""
    old_set = set(old_perms)
    new_set = set(new_perms)

    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
        "unchanged": sorted(old_set & new_set),
    }
\`\`\`

**Amaliy foydalari:**
- Ruxsat tizimlari to'plam amallaridan foydalanadi
- Ma'lumotlardagi umumiy/farqli xususiyatlarni topish
- Samarali tegishlilik tekshiruvi O(1)`,
		},
	},
};

export default task;
