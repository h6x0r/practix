import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-str-repr',
	title: 'String Representation',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'oop', 'dunder'],
	estimatedTime: '10m',
	isPremium: false,
	order: 3,

	description: `# String Representation

Learn to customize how objects appear as strings using \`__str__\` and \`__repr__\`.

## Task

Create a class \`Person\` with proper string representations.

## Requirements

- \`__init__(self, name, age)\`: Initialize with name and age
- \`__str__(self)\`: Return user-friendly string like "Alice (30 years old)"
- \`__repr__(self)\`: Return developer-friendly string like "Person('Alice', 30)"

## Examples

\`\`\`python
>>> p = Person("Alice", 30)
>>> str(p)
"Alice (30 years old)"
>>> repr(p)
"Person('Alice', 30)"
>>> print(p)
Alice (30 years old)
\`\`\``,

	initialCode: `class Person:
    """A class representing a person."""

    def __init__(self, name: str, age: int):
        # TODO: Initialize attributes
        pass

    def __str__(self) -> str:
        # TODO: Return user-friendly string
        pass

    def __repr__(self) -> str:
        # TODO: Return developer-friendly string
        pass`,

	solutionCode: `class Person:
    """A class representing a person."""

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self) -> str:
        # User-friendly: for end users and print()
        return f"{self.name} ({self.age} years old)"

    def __repr__(self) -> str:
        # Developer-friendly: shows how to recreate the object
        return f"Person('{self.name}', {self.age})"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        p = Person("Alice", 30)
        self.assertEqual(str(p), "Alice (30 years old)")

    def test_2(self):
        p = Person("Alice", 30)
        self.assertEqual(repr(p), "Person('Alice', 30)")

    def test_3(self):
        p = Person("Bob", 25)
        self.assertEqual(str(p), "Bob (25 years old)")

    def test_4(self):
        p = Person("Bob", 25)
        self.assertEqual(repr(p), "Person('Bob', 25)")

    def test_5(self):
        p = Person("Test", 0)
        self.assertEqual(str(p), "Test (0 years old)")

    def test_6(self):
        p = Person("Name With Spaces", 50)
        self.assertIn("Name With Spaces", str(p))

    def test_7(self):
        p = Person("Test", 100)
        self.assertIn("100", str(p))

    def test_8(self):
        p = Person("A", 1)
        self.assertEqual(p.name, "A")

    def test_9(self):
        p = Person("A", 1)
        self.assertEqual(p.age, 1)

    def test_10(self):
        p = Person("Test", 42)
        self.assertIn("Person(", repr(p))

if __name__ == '__main__':
    unittest.main()`,

	hint1: '__str__ is called by print() and str(). It should be human-readable.',
	hint2: '__repr__ is for debugging. It should ideally show how to create the same object.',

	whyItMatters: `Good string representations make debugging easier and improve code readability.

**Production Pattern:**

\`\`\`python
@dataclass
class User:
    id: int
    username: str
    email: str

    def __str__(self) -> str:
        return f"@{self.username}"

    def __repr__(self) -> str:
        return f"User(id={self.id}, username='{self.username}')"
\`\`\``,

	translations: {
		ru: {
			title: 'Строковое представление',
			description: `# Строковое представление

Научитесь настраивать отображение объектов как строк с помощью \`__str__\` и \`__repr__\`.

## Задача

Создайте класс \`Person\` с правильными строковыми представлениями.

## Требования

- \`__init__(self, name, age)\`: Инициализация с именем и возрастом
- \`__str__(self)\`: Возврат читаемой строки "Alice (30 years old)"
- \`__repr__(self)\`: Возврат строки для разработчика "Person('Alice', 30)"`,
			hint1: '__str__ вызывается print() и str(). Должен быть читаемым для человека.',
			hint2: '__repr__ для отладки. Должен показывать, как создать такой же объект.',
			whyItMatters: `Хорошие строковые представления упрощают отладку и улучшают читаемость кода.`,
		},
		uz: {
			title: "Satr ko'rinishi",
			description: `# Satr ko'rinishi

\`__str__\` va \`__repr__\` yordamida ob'ektlarning satr sifatida ko'rinishini sozlashni o'rganing.

## Vazifa

To'g'ri satr ko'rinishlariga ega \`Person\` klassini yarating.

## Talablar

- \`__init__(self, name, age)\`: Ism va yosh bilan ishga tushirish
- \`__str__(self)\`: Foydalanuvchi uchun o'qilishi oson satr "Alice (30 years old)"
- \`__repr__(self)\`: Dasturchi uchun satr "Person('Alice', 30)"`,
			hint1: "__str__ print() va str() tomonidan chaqiriladi. Odam uchun o'qilishi oson bo'lishi kerak.",
			hint2: "__repr__ nosozliklarni tuzatish uchun. Xuddi shunday ob'ektni qanday yaratishni ko'rsatishi kerak.",
			whyItMatters: `Yaxshi satr ko'rinishlari nosozliklarni tuzatishni osonlashtiradi va kod o'qilishini yaxshilaydi.`,
		},
	},
};

export default task;
