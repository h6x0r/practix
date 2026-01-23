import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-type-checking',
	title: 'Type Checking',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'types', 'isinstance'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,

	description: `# Type Checking

Python is dynamically typed, but you can check types at runtime using \`type()\` and \`isinstance()\`.

## Task

Implement the function \`get_type_name(value)\` that returns a human-readable type name.

## Requirements

Return the following strings based on the type:
- \`int\` → \`"integer"\`
- \`float\` → \`"float"\`
- \`str\` → \`"string"\`
- \`bool\` → \`"boolean"\`
- \`list\` → \`"list"\`
- \`dict\` → \`"dictionary"\`
- \`None\` → \`"none"\`
- anything else → \`"unknown"\`

## Examples

\`\`\`python
>>> get_type_name(42)
"integer"

>>> get_type_name("hello")
"string"

>>> get_type_name([1, 2, 3])
"list"

>>> get_type_name(None)
"none"
\`\`\``,

	initialCode: `def get_type_name(value) -> str:
    """Return a human-readable name for the type of value.

    Args:
        value: Any Python value

    Returns:
        A string describing the type
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def get_type_name(value) -> str:
    """Return a human-readable name for the type of value.

    Args:
        value: Any Python value

    Returns:
        A string describing the type
    """
    # Check bool BEFORE int (bool is subclass of int)
    if value is None:
        return "none"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "list"
    elif isinstance(value, dict):
        return "dictionary"
    else:
        return "unknown"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Integer type"""
        self.assertEqual(get_type_name(42), "integer")

    def test_2(self):
        """String type"""
        self.assertEqual(get_type_name("hello"), "string")

    def test_3(self):
        """Float type"""
        self.assertEqual(get_type_name(3.14), "float")

    def test_4(self):
        """Boolean True"""
        self.assertEqual(get_type_name(True), "boolean")

    def test_5(self):
        """Boolean False"""
        self.assertEqual(get_type_name(False), "boolean")

    def test_6(self):
        """List type"""
        self.assertEqual(get_type_name([1, 2, 3]), "list")

    def test_7(self):
        """Dictionary type"""
        self.assertEqual(get_type_name({"a": 1}), "dictionary")

    def test_8(self):
        """None type"""
        self.assertEqual(get_type_name(None), "none")

    def test_9(self):
        """Empty string is still string"""
        self.assertEqual(get_type_name(""), "string")

    def test_10(self):
        """Tuple is unknown"""
        self.assertEqual(get_type_name((1, 2)), "unknown")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use isinstance(value, type) to check if a value is of a specific type.',
	hint2: 'Important: Check for bool BEFORE int, because bool is a subclass of int in Python!',

	whyItMatters: `Type checking is essential for writing robust code that handles different inputs gracefully.

**Production Pattern:**

\`\`\`python
TYPE_HANDLERS = {
    int: lambda x: f"Processing integer: {x}",
    str: lambda x: f"Processing string: {x}",
    list: lambda x: f"Processing list with {len(x)} items",
}

def process(value):
    handler = TYPE_HANDLERS.get(type(value))
    if handler:
        return handler(value)
    raise TypeError(f"Unsupported type: {type(value)}")
\`\`\`

**Practical Benefits:**
- Type validation prevents runtime errors
- isinstance() handles inheritance correctly
- Clear error messages help debugging`,

	translations: {
		ru: {
			title: 'Проверка типов',
			description: `# Проверка типов

Python — динамически типизированный язык, но вы можете проверять типы во время выполнения с помощью \`type()\` и \`isinstance()\`.

## Задача

Реализуйте функцию \`get_type_name(value)\`, которая возвращает читаемое название типа.

## Требования

Верните следующие строки в зависимости от типа:
- \`int\` → \`"integer"\`
- \`float\` → \`"float"\`
- \`str\` → \`"string"\`
- \`bool\` → \`"boolean"\`
- \`list\` → \`"list"\`
- \`dict\` → \`"dictionary"\`
- \`None\` → \`"none"\`
- всё остальное → \`"unknown"\`

## Примеры

\`\`\`python
>>> get_type_name(42)
"integer"

>>> get_type_name("hello")
"string"

>>> get_type_name(None)
"none"
\`\`\``,
			hint1: 'Используйте isinstance(value, type) для проверки типа значения.',
			hint2: 'Важно: проверяйте bool ДО int, потому что bool — подкласс int в Python!',
			whyItMatters: `Проверка типов необходима для написания надёжного кода.

**Продакшен паттерн:**

\`\`\`python
TYPE_HANDLERS = {
    int: lambda x: f"Обработка целого: {x}",
    str: lambda x: f"Обработка строки: {x}",
    list: lambda x: f"Обработка списка с {len(x)} элементами",
}

def process(value):
    handler = TYPE_HANDLERS.get(type(value))
    if handler:
        return handler(value)
    raise TypeError(f"Неподдерживаемый тип: {type(value)}")
\`\`\`

**Практические преимущества:**
- Валидация типов предотвращает ошибки
- isinstance() корректно обрабатывает наследование
- Понятные сообщения помогают отладке`,
		},
		uz: {
			title: 'Turlarni tekshirish',
			description: `# Turlarni tekshirish

Python dinamik tiplangan til, lekin siz \`type()\` va \`isinstance()\` yordamida turlarni tekshirishingiz mumkin.

## Vazifa

Qiymat turining o'qilishi oson nomini qaytaruvchi \`get_type_name(value)\` funksiyasini amalga oshiring.

## Talablar

Turga qarab quyidagi satrlarni qaytaring:
- \`int\` → \`"integer"\`
- \`float\` → \`"float"\`
- \`str\` → \`"string"\`
- \`bool\` → \`"boolean"\`
- \`list\` → \`"list"\`
- \`dict\` → \`"dictionary"\`
- \`None\` → \`"none"\`
- boshqalar → \`"unknown"\`

## Misollar

\`\`\`python
>>> get_type_name(42)
"integer"

>>> get_type_name("hello")
"string"

>>> get_type_name(None)
"none"
\`\`\``,
			hint1: "Qiymat turini tekshirish uchun isinstance(value, type) dan foydalaning.",
			hint2: "Muhim: bool ni int dan OLDIN tekshiring, chunki bool Python da int ning kichik klassi!",
			whyItMatters: `Turlarni tekshirish ishonchli kod yozish uchun zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
TYPE_HANDLERS = {
    int: lambda x: f"Butun sonni qayta ishlash: {x}",
    str: lambda x: f"Satrni qayta ishlash: {x}",
    list: lambda x: f"{len(x)} ta element bilan ro'yxatni qayta ishlash",
}

def process(value):
    handler = TYPE_HANDLERS.get(type(value))
    if handler:
        return handler(value)
    raise TypeError(f"Qo'llab-quvvatlanmaydigan tur: {type(value)}")
\`\`\`

**Amaliy foydalari:**
- Tur tekshiruvi xatolarni oldini oladi
- isinstance() merosni to'g'ri boshqaradi
- Aniq xabarlar disk raskadrovkaga yordam beradi`,
		},
	},
};

export default task;
