import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-merge-dicts',
	title: 'Merge Dictionaries',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'dictionaries', 'merge'],
	estimatedTime: '10m',
	isPremium: false,
	order: 5,

	description: `# Merge Dictionaries

Learn different ways to merge dictionaries in Python.

## Task

Implement the function \`merge_dicts(dict1, dict2)\` that merges two dictionaries.

## Requirements

- If a key exists in both, use the value from dict2
- Return a NEW dictionary (don't modify originals)
- Handle empty dictionaries

## Examples

\`\`\`python
>>> merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
{"a": 1, "b": 3, "c": 4}

>>> merge_dicts({"x": 10}, {})
{"x": 10}

>>> merge_dicts({}, {"y": 20})
{"y": 20}
\`\`\``,

	initialCode: `def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries, with dict2 values taking precedence.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1)

    Returns:
        New merged dictionary
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries, with dict2 values taking precedence.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1)

    Returns:
        New merged dictionary
    """
    # Python 3.9+ way: return dict1 | dict2
    # Compatible way:
    result = dict1.copy()
    result.update(dict2)
    return result`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Overlapping keys"""
        result = merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        self.assertEqual(result, {"a": 1, "b": 3, "c": 4})

    def test_2(self):
        """First dict only"""
        result = merge_dicts({"x": 10}, {})
        self.assertEqual(result, {"x": 10})

    def test_3(self):
        """Second dict only"""
        result = merge_dicts({}, {"y": 20})
        self.assertEqual(result, {"y": 20})

    def test_4(self):
        """Both empty"""
        self.assertEqual(merge_dicts({}, {}), {})

    def test_5(self):
        """No overlap"""
        result = merge_dicts({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_6(self):
        """Original not modified"""
        original = {"a": 1}
        merge_dicts(original, {"a": 2})
        self.assertEqual(original, {"a": 1})

    def test_7(self):
        """Multiple overlaps"""
        result = merge_dicts({"a": 1, "b": 2, "c": 3}, {"a": 10, "c": 30})
        self.assertEqual(result, {"a": 10, "b": 2, "c": 30})

    def test_8(self):
        """String keys and values"""
        result = merge_dicts({"name": "Alice"}, {"name": "Bob", "age": "25"})
        self.assertEqual(result, {"name": "Bob", "age": "25"})

    def test_9(self):
        """Numeric keys"""
        result = merge_dicts({1: "one"}, {2: "two"})
        self.assertEqual(result, {1: "one", 2: "two"})

    def test_10(self):
        """Large dicts"""
        d1 = {i: i for i in range(100)}
        d2 = {i: i * 2 for i in range(50, 150)}
        result = merge_dicts(d1, d2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[50], 100)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use dict1.copy() to create a new dictionary, then update() with dict2.',
	hint2: 'In Python 3.9+, you can simply use: dict1 | dict2',

	whyItMatters: `Merging dictionaries is common when combining configurations, defaults, and overrides.

**Production Pattern:**

\`\`\`python
def get_config(user_config: dict = None) -> dict:
    """Merge user config with defaults."""
    defaults = {
        "timeout": 30,
        "retries": 3,
        "debug": False,
    }
    return {**defaults, **(user_config or {})}

def merge_api_responses(responses: list[dict]) -> dict:
    """Combine multiple API responses into one."""
    result = {}
    for response in responses:
        result.update(response)
    return result

# Python 3.9+ syntax: config = defaults | user_config
\`\`\`

**Practical Benefits:**
- Config management uses dict merging extensively
- The {**d1, **d2} syntax is clear and concise
- Preserves immutability by creating new dicts`,

	translations: {
		ru: {
			title: 'Объединение словарей',
			description: `# Объединение словарей

Изучите разные способы объединения словарей в Python.

## Задача

Реализуйте функцию \`merge_dicts(dict1, dict2)\`, которая объединяет два словаря.

## Требования

- Если ключ есть в обоих, используйте значение из dict2
- Верните НОВЫЙ словарь (не изменяйте оригиналы)
- Обработайте пустые словари

## Примеры

\`\`\`python
>>> merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
{"a": 1, "b": 3, "c": 4}

>>> merge_dicts({"x": 10}, {})
{"x": 10}

>>> merge_dicts({}, {"y": 20})
{"y": 20}
\`\`\``,
			hint1: 'Используйте dict1.copy() для создания нового словаря, затем update() с dict2.',
			hint2: 'В Python 3.9+: dict1 | dict2',
			whyItMatters: `Объединение словарей часто используется при работе с конфигурациями.

**Продакшен паттерн:**

\`\`\`python
def get_config(user_config: dict = None) -> dict:
    """Объединение пользовательского конфига с дефолтами."""
    defaults = {
        "timeout": 30,
        "retries": 3,
        "debug": False,
    }
    return {**defaults, **(user_config or {})}
\`\`\`

**Практические преимущества:**
- Управление конфигурацией использует объединение словарей
- Синтаксис {**d1, **d2} понятен и компактен
- Сохраняет иммутабельность создавая новые словари`,
		},
		uz: {
			title: "Lug'atlarni birlashtirish",
			description: `# Lug'atlarni birlashtirish

Python da lug'atlarni birlashtirishning turli usullarini o'rganing.

## Vazifa

Ikki lug'atni birlashtiruvchi \`merge_dicts(dict1, dict2)\` funksiyasini amalga oshiring.

## Talablar

- Agar kalit ikkalasida ham bo'lsa, dict2 dan qiymatni ishlating
- YANGI lug'at qaytaring (asllarini o'zgartirmang)
- Bo'sh lug'atlarni ham ishlov bering

## Misollar

\`\`\`python
>>> merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
{"a": 1, "b": 3, "c": 4}

>>> merge_dicts({"x": 10}, {})
{"x": 10}

>>> merge_dicts({}, {"y": 20})
{"y": 20}
\`\`\``,
			hint1: "Yangi lug'at yaratish uchun dict1.copy() dan, keyin dict2 bilan update() dan foydalaning.",
			hint2: "Python 3.9+ da: dict1 | dict2",
			whyItMatters: `Lug'atlarni birlashtirish konfiguratsiyalar bilan ishlashda tez-tez ishlatiladi.

**Ishlab chiqarish patterni:**

\`\`\`python
def get_config(user_config: dict = None) -> dict:
    """Foydalanuvchi konfiguratsiyasini standartlar bilan birlashtirish."""
    defaults = {
        "timeout": 30,
        "retries": 3,
        "debug": False,
    }
    return {**defaults, **(user_config or {})}
\`\`\`

**Amaliy foydalari:**
- Konfiguratsiyani boshqarish lug'atlarni birlashtirishdan foydalanadi
- {**d1, **d2} sintaksisi aniq va ixcham
- Yangi lug'atlar yaratib o'zgarmaslikni saqlaydi`,
		},
	},
};

export default task;
