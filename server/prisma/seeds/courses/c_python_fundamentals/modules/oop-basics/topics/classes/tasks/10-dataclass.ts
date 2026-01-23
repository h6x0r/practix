import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-dataclass',
	title: 'Data Classes',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'oop', 'dataclass'],
	estimatedTime: '10m',
	isPremium: false,
	order: 10,

	description: `# Data Classes

The \`dataclass\` decorator automatically generates \`__init__\`, \`__repr__\`, and \`__eq__\`.

## Task

Create a \`User\` class using the dataclass decorator with methods.

## Requirements

- Use \`@dataclass\` decorator
- Fields: \`username\` (str), \`email\` (str), \`age\` (int with default 0)
- Method \`is_adult(self)\`: Return True if age >= 18
- Method \`email_domain(self)\`: Return the domain part of email

## Examples

\`\`\`python
>>> u = User("alice", "alice@example.com", 25)
>>> u.is_adult()
True
>>> u.email_domain()
"example.com"

>>> u2 = User("bob", "bob@test.org")  # age defaults to 0
>>> u2.age
0
\`\`\``,

	initialCode: `from dataclasses import dataclass

@dataclass
class User:
    """User class using dataclass decorator."""
    # TODO: Define fields with types
    # username: str
    # email: str
    # age: int = 0

    def is_adult(self) -> bool:
        # TODO: Return True if age >= 18
        pass

    def email_domain(self) -> str:
        # TODO: Return domain part after @
        pass`,

	solutionCode: `from dataclasses import dataclass

@dataclass
class User:
    """User class using dataclass decorator."""
    username: str
    email: str
    age: int = 0

    def is_adult(self) -> bool:
        return self.age >= 18

    def email_domain(self) -> str:
        return self.email.split("@")[1]`,

	testCode: `import unittest
from dataclasses import dataclass

class Test(unittest.TestCase):
    def test_1(self):
        u = User("alice", "alice@example.com", 25)
        self.assertTrue(u.is_adult())

    def test_2(self):
        u = User("alice", "alice@example.com", 25)
        self.assertEqual(u.email_domain(), "example.com")

    def test_3(self):
        u = User("bob", "bob@test.org")
        self.assertEqual(u.age, 0)

    def test_4(self):
        u = User("bob", "bob@test.org")
        self.assertFalse(u.is_adult())

    def test_5(self):
        u = User("test", "test@gmail.com", 18)
        self.assertTrue(u.is_adult())

    def test_6(self):
        u = User("test", "test@gmail.com", 17)
        self.assertFalse(u.is_adult())

    def test_7(self):
        u = User("test", "user@company.co.uk", 30)
        self.assertEqual(u.email_domain(), "company.co.uk")

    def test_8(self):
        u = User("x", "x@y.z")
        self.assertEqual(u.username, "x")

    def test_9(self):
        u = User("x", "x@y.z")
        self.assertEqual(u.email, "x@y.z")

    def test_10(self):
        u1 = User("a", "a@b.c", 10)
        u2 = User("a", "a@b.c", 10)
        self.assertEqual(u1, u2)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use @dataclass before class definition. Define fields with type annotations: username: str',
	hint2: 'For default values, use: age: int = 0. Methods work normally within dataclasses.',

	whyItMatters: `Dataclasses reduce boilerplate and provide automatic comparison, representation, and hashing.

**Production Pattern:**

\`\`\`python
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)  # Immutable
class Config:
    host: str
    port: int = 8080
    debug: bool = False

@dataclass
class User:
    id: int
    name: str
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Validation after init
        if not self.name.strip():
            raise ValueError("Name cannot be empty")
\`\`\``,

	translations: {
		ru: {
			title: 'Классы данных',
			description: `# Классы данных

Декоратор \`dataclass\` автоматически генерирует \`__init__\`, \`__repr__\` и \`__eq__\`.

## Задача

Создайте класс \`User\` с декоратором dataclass и методами.`,
			hint1: 'Используйте @dataclass перед определением класса. Определите поля с аннотациями: username: str',
			hint2: 'Для значений по умолчанию: age: int = 0. Методы работают как обычно.',
			whyItMatters: `Dataclass уменьшает шаблонный код и автоматизирует сравнение и представление.`,
		},
		uz: {
			title: "Ma'lumotlar klasslari",
			description: `# Ma'lumotlar klasslari

\`dataclass\` dekoratori avtomatik \`__init__\`, \`__repr__\` va \`__eq__\` yaratadi.

## Vazifa

dataclass dekoratori va metodlarga ega \`User\` klassini yarating.`,
			hint1: "Klass ta'rifidan oldin @dataclass ishlating. Tip annotatsiyalari bilan maydonlarni aniqlang: username: str",
			hint2: "Standart qiymatlar uchun: age: int = 0. Metodlar dataclass ichida oddiy ishlaydi.",
			whyItMatters: `Dataclass shablon kodini kamaytiradi va avtomatik taqqoslash va ko'rinishni ta'minlaydi.`,
		},
	},
};

export default task;
