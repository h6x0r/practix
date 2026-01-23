import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-hello-world',
	title: 'Hello World',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'functions', 'strings'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,

	description: `# Hello World

Welcome to Python! Your first task is to write a function that returns a greeting message.

## Task

Implement the function \`greet(name)\` that takes a person's name and returns a greeting string.

## Requirements

- The function should return the string: \`"Hello, {name}!"\`
- If the name is empty, return \`"Hello, World!"\`

## Examples

\`\`\`python
>>> greet("Alice")
"Hello, Alice!"

>>> greet("")
"Hello, World!"

>>> greet("Python")
"Hello, Python!"
\`\`\``,

	initialCode: `def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name to greet. If empty, use "World".

    Returns:
        A greeting string in format "Hello, {name}!"
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name to greet. If empty, use "World".

    Returns:
        A greeting string in format "Hello, {name}!"
    """
    # Use "World" as default if name is empty
    if not name:
        name = "World"
    return f"Hello, {name}!"`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic greeting"""
        self.assertEqual(greet("Alice"), "Hello, Alice!")

    def test_2(self):
        """Empty name returns World"""
        self.assertEqual(greet(""), "Hello, World!")

    def test_3(self):
        """Single character name"""
        self.assertEqual(greet("A"), "Hello, A!")

    def test_4(self):
        """Name with spaces"""
        self.assertEqual(greet("John Doe"), "Hello, John Doe!")

    def test_5(self):
        """Name with numbers"""
        self.assertEqual(greet("User123"), "Hello, User123!")

    def test_6(self):
        """Lowercase name"""
        self.assertEqual(greet("python"), "Hello, python!")

    def test_7(self):
        """Uppercase name"""
        self.assertEqual(greet("PYTHON"), "Hello, PYTHON!")

    def test_8(self):
        """Name with special characters"""
        self.assertEqual(greet("O'Brien"), "Hello, O'Brien!")

    def test_9(self):
        """Long name"""
        self.assertEqual(greet("Alexander the Great"), "Hello, Alexander the Great!")

    def test_10(self):
        """Unicode name"""
        self.assertEqual(greet("Мария"), "Hello, Мария!")

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use an if statement to check if the name is empty. An empty string is "falsy" in Python.',
	hint2: 'Use f-strings for formatting: f"Hello, {name}!"',

	whyItMatters: `This simple task introduces three fundamental Python concepts:

1. **Functions**: The building blocks of reusable code
2. **Conditional logic**: Making decisions with if statements
3. **String formatting**: Creating dynamic text with f-strings

**Production Pattern:**

\`\`\`python
def create_response(user_name: str = "") -> dict:
    """Create a personalized API response."""
    greeting = f"Welcome, {user_name or 'Guest'}!"
    return {"message": greeting, "status": "success"}
\`\`\`

**Practical Benefits:**
- Functions make code reusable and testable
- Default values prevent errors from missing data
- F-strings are the modern, readable way to format text`,

	translations: {
		ru: {
			title: 'Привет, мир',
			description: `# Привет, мир

Добро пожаловать в Python! Ваша первая задача — написать функцию, которая возвращает приветственное сообщение.

## Задача

Реализуйте функцию \`greet(name)\`, которая принимает имя и возвращает строку приветствия.

## Требования

- Функция должна возвращать строку: \`"Hello, {name}!"\`
- Если имя пустое, вернуть \`"Hello, World!"\`

## Примеры

\`\`\`python
>>> greet("Alice")
"Hello, Alice!"

>>> greet("")
"Hello, World!"

>>> greet("Python")
"Hello, Python!"
\`\`\``,
			hint1: 'Используйте if для проверки, пустое ли имя. Пустая строка — это "ложное" значение в Python.',
			hint2: 'Используйте f-строки для форматирования: f"Hello, {name}!"',
			whyItMatters: `Эта простая задача знакомит с тремя фундаментальными концепциями Python:

1. **Функции**: строительные блоки переиспользуемого кода
2. **Условная логика**: принятие решений с помощью if
3. **Форматирование строк**: создание динамического текста с f-строками

**Продакшен паттерн:**

\`\`\`python
def create_response(user_name: str = "") -> dict:
    """Создание персонализированного API ответа."""
    greeting = f"Добро пожаловать, {user_name or 'Гость'}!"
    return {"message": greeting, "status": "success"}
\`\`\`

**Практические преимущества:**
- Функции делают код переиспользуемым и тестируемым
- Значения по умолчанию предотвращают ошибки
- F-строки — современный и читаемый способ форматирования`,
		},
		uz: {
			title: 'Salom, dunyo',
			description: `# Salom, dunyo

Python ga xush kelibsiz! Birinchi vazifangiz — salomlash xabarini qaytaruvchi funksiya yozish.

## Vazifa

\`greet(name)\` funksiyasini amalga oshiring. U ismni qabul qilib, salomlash satrini qaytaradi.

## Talablar

- Funksiya \`"Hello, {name}!"\` satrini qaytarishi kerak
- Agar ism bo'sh bo'lsa, \`"Hello, World!"\` qaytaring

## Misollar

\`\`\`python
>>> greet("Alice")
"Hello, Alice!"

>>> greet("")
"Hello, World!"

>>> greet("Python")
"Hello, Python!"
\`\`\``,
			hint1: "Ism bo'sh ekanligini tekshirish uchun if dan foydalaning. Bo'sh satr Python da \"yolg'on\" qiymat hisoblanadi.",
			hint2: 'Formatlash uchun f-satrlardan foydalaning: f"Hello, {name}!"',
			whyItMatters: `Bu oddiy vazifa Python ning uchta asosiy tushunchasini o'rgatadi:

1. **Funksiyalar**: qayta ishlatiladigan kod bloklari
2. **Shartli mantiq**: if yordamida qaror qabul qilish
3. **Satr formatlash**: f-satrlar bilan dinamik matn yaratish

**Ishlab chiqarish patterni:**

\`\`\`python
def create_response(user_name: str = "") -> dict:
    """Shaxsiylashtirilgan API javobi yaratish."""
    greeting = f"Xush kelibsiz, {user_name or 'Mehmon'}!"
    return {"message": greeting, "status": "success"}
\`\`\`

**Amaliy foydalari:**
- Funksiyalar kodni qayta ishlatiluvchi va test qilinadigan qiladi
- Standart qiymatlar xatolarni oldini oladi
- F-satrlar zamonaviy va o'qilishi oson formatlash usuli`,
		},
	},
};

export default task;
