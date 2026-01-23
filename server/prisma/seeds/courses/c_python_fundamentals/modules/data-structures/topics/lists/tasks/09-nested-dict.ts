import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-nested-dict',
	title: 'Nested Dictionary Access',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'dictionaries', 'nested'],
	estimatedTime: '15m',
	isPremium: false,
	order: 9,

	description: `# Nested Dictionary Access

Safely navigate and extract data from nested dictionary structures.

## Task

Implement the function \`get_nested(data, path, default)\` that safely retrieves a value from a nested dictionary.

## Requirements

- \`path\` is a string with keys separated by dots: \`"a.b.c"\`
- Return the value at the path if it exists
- Return \`default\` if any key in the path doesn't exist
- Handle both dict and list indexing (use integers for lists)

## Examples

\`\`\`python
>>> data = {"user": {"name": "Alice", "scores": [85, 90, 95]}}
>>> get_nested(data, "user.name", None)
"Alice"

>>> get_nested(data, "user.scores.1", None)
90

>>> get_nested(data, "user.email", "N/A")
"N/A"

>>> get_nested(data, "invalid.path", "default")
"default"
\`\`\``,

	initialCode: `def get_nested(data: dict, path: str, default=None):
    """Safely get a value from a nested dictionary using dot notation.

    Args:
        data: Nested dictionary structure
        path: Dot-separated path like "user.profile.name"
        default: Value to return if path not found

    Returns:
        Value at path or default if not found
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def get_nested(data: dict, path: str, default=None):
    """Safely get a value from a nested dictionary using dot notation.

    Args:
        data: Nested dictionary structure
        path: Dot-separated path like "user.profile.name"
        default: Value to return if path not found

    Returns:
        Value at path or default if not found
    """
    # Split path into individual keys
    keys = path.split(".")

    # Start with the full data structure
    current = data

    # Navigate through each key in the path
    for key in keys:
        try:
            # Try to access as dictionary first
            if isinstance(current, dict):
                current = current[key]
            # Try to access as list with integer index
            elif isinstance(current, list):
                current = current[int(key)]
            else:
                # Can't navigate further
                return default
        except (KeyError, IndexError, ValueError, TypeError):
            # Key doesn't exist or index is invalid
            return default

    return current`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def setUp(self):
        self.data = {
            "user": {
                "name": "Alice",
                "profile": {
                    "age": 30,
                    "city": "NYC"
                },
                "scores": [85, 90, 95]
            },
            "settings": {
                "theme": "dark"
            }
        }

    def test_1(self):
        """Simple nested access"""
        self.assertEqual(get_nested(self.data, "user.name", None), "Alice")

    def test_2(self):
        """Deep nested access"""
        self.assertEqual(get_nested(self.data, "user.profile.city", None), "NYC")

    def test_3(self):
        """List index access"""
        self.assertEqual(get_nested(self.data, "user.scores.1", None), 90)

    def test_4(self):
        """Missing key returns default"""
        self.assertEqual(get_nested(self.data, "user.email", "N/A"), "N/A")

    def test_5(self):
        """Invalid path returns default"""
        self.assertEqual(get_nested(self.data, "invalid.path", "default"), "default")

    def test_6(self):
        """Top-level key"""
        result = get_nested(self.data, "settings", None)
        self.assertEqual(result["theme"], "dark")

    def test_7(self):
        """Empty path handling"""
        self.assertEqual(get_nested(self.data, "", "default"), "default")

    def test_8(self):
        """List first element"""
        self.assertEqual(get_nested(self.data, "user.scores.0", None), 85)

    def test_9(self):
        """Out of bounds list index"""
        self.assertEqual(get_nested(self.data, "user.scores.99", "N/A"), "N/A")

    def test_10(self):
        """Integer value in dict"""
        self.assertEqual(get_nested(self.data, "user.profile.age", None), 30)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Split the path into keys using path.split("."). Then iterate through each key, accessing the nested value.',
	hint2: 'Use try/except to handle missing keys. Check if current is a list and convert key to int for list indexing.',

	whyItMatters: `Safely accessing nested data is essential when working with JSON APIs and configuration files.

**Production Pattern:**

\`\`\`python
from typing import Any
from functools import reduce

def get_nested_safe(data: dict, path: str, default: Any = None) -> Any:
    """Production-ready nested dict access."""
    try:
        return reduce(
            lambda d, key: d[int(key)] if isinstance(d, list) else d[key],
            path.split("."),
            data
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return default

def set_nested(data: dict, path: str, value: Any) -> dict:
    """Set a value in a nested dict, creating intermediate dicts as needed."""
    keys = path.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
    return data

def deep_get_with_default(data: dict, *keys, default=None):
    """Get nested value with multiple key arguments."""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data

# Example: deep_get_with_default(response, "data", "users", 0, "name")
\`\`\`

**Practical Benefits:**
- API responses often have deeply nested structures
- Configuration files use nested hierarchies
- Avoiding KeyError exceptions improves robustness`,

	translations: {
		ru: {
			title: 'Доступ к вложенным словарям',
			description: `# Доступ к вложенным словарям

Безопасная навигация и извлечение данных из вложенных структур словарей.

## Задача

Реализуйте функцию \`get_nested(data, path, default)\`, которая безопасно извлекает значение из вложенного словаря.

## Требования

- \`path\` — строка с ключами, разделёнными точками: \`"a.b.c"\`
- Верните значение по пути, если оно существует
- Верните \`default\`, если какой-либо ключ в пути не существует
- Обработайте и словари, и списки (используйте целые числа для списков)

## Примеры

\`\`\`python
>>> data = {"user": {"name": "Alice", "scores": [85, 90, 95]}}
>>> get_nested(data, "user.name", None)
"Alice"

>>> get_nested(data, "user.scores.1", None)
90

>>> get_nested(data, "user.email", "N/A")
"N/A"
\`\`\``,
			hint1: 'Разбейте путь на ключи: path.split("."). Затем переберите каждый ключ.',
			hint2: 'Используйте try/except для обработки отсутствующих ключей. Для списков конвертируйте ключ в int.',
			whyItMatters: `Безопасный доступ к вложенным данным необходим при работе с JSON API.

**Продакшен паттерн:**

\`\`\`python
def set_nested(data: dict, path: str, value) -> dict:
    """Установить значение во вложенном словаре."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return data
\`\`\`

**Практические преимущества:**
- Ответы API часто имеют глубокую вложенность
- Конфигурационные файлы используют вложенные иерархии`,
		},
		uz: {
			title: "Ichma-ich lug'atlarga kirish",
			description: `# Ichma-ich lug'atlarga kirish

Ichma-ich lug'at tuzilmalaridan xavfsiz navigatsiya va ma'lumotlarni olish.

## Vazifa

Ichma-ich lug'atdan qiymatni xavfsiz oluvchi \`get_nested(data, path, default)\` funksiyasini amalga oshiring.

## Talablar

- \`path\` nuqta bilan ajratilgan kalitlar satri: \`"a.b.c"\`
- Agar yo'l mavjud bo'lsa, qiymatni qaytaring
- Yo'ldagi biror kalit mavjud bo'lmasa, \`default\` qaytaring
- Lug'at va ro'yxat indekslarini ham ishlang (ro'yxatlar uchun butun sonlar)

## Misollar

\`\`\`python
>>> data = {"user": {"name": "Alice", "scores": [85, 90, 95]}}
>>> get_nested(data, "user.name", None)
"Alice"

>>> get_nested(data, "user.scores.1", None)
90

>>> get_nested(data, "user.email", "N/A")
"N/A"
\`\`\``,
			hint1: "Yo'lni kalitlarga ajrating: path.split(\".\"). So'ng har bir kalit bo'ylab yuring.",
			hint2: "Mavjud bo'lmagan kalitlar uchun try/except ishlating. Ro'yxatlar uchun kalitni int ga aylantiring.",
			whyItMatters: `Ichma-ich ma'lumotlarga xavfsiz kirish JSON API bilan ishlashda zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
def set_nested(data: dict, path: str, value) -> dict:
    """Ichma-ich lug'atda qiymat o'rnatish."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return data
\`\`\`

**Amaliy foydalari:**
- API javoblari ko'pincha chuqur ichma-ich bo'ladi
- Konfiguratsiya fayllari ichma-ich ierarxiyalardan foydalanadi`,
		},
	},
};

export default task;
