import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-stack-operations',
	title: 'Valid Parentheses',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'stack', 'strings'],
	estimatedTime: '15m',
	isPremium: false,
	order: 12,

	description: `# Valid Parentheses

Use a stack to validate matching brackets in an expression.

## Task

Implement the function \`is_valid_brackets(s)\` that checks if a string has valid matching brackets.

## Requirements

- Handle three types: \`()\`, \`{}\`, \`[]\`
- Every opening bracket must have a matching closing bracket
- Brackets must close in the correct order
- Return \`True\` for valid, \`False\` for invalid

## Examples

\`\`\`python
>>> is_valid_brackets("()")
True

>>> is_valid_brackets("()[]{}")
True

>>> is_valid_brackets("([{}])")
True

>>> is_valid_brackets("(]")
False

>>> is_valid_brackets("([)]")
False

>>> is_valid_brackets("")
True
\`\`\``,

	initialCode: `def is_valid_brackets(s: str) -> bool:
    """Check if brackets in string are valid and properly nested.

    Args:
        s: String containing brackets (){}[]

    Returns:
        True if all brackets are properly matched and nested
    """
    # TODO: Implement using a stack
    pass`,

	solutionCode: `def is_valid_brackets(s: str) -> bool:
    """Check if brackets in string are valid and properly nested.

    Args:
        s: String containing brackets (){}[]

    Returns:
        True if all brackets are properly matched and nested
    """
    # Stack to keep track of opening brackets
    stack = []

    # Mapping of closing to opening brackets
    bracket_pairs = {
        ")": "(",
        "}": "{",
        "]": "[",
    }

    for char in s:
        # If it's a closing bracket
        if char in bracket_pairs:
            # Stack must not be empty and top must match
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            # Pop the matching opening bracket
            stack.pop()

        # If it's an opening bracket
        elif char in "({[":
            stack.append(char)

    # Valid only if all brackets are matched (stack is empty)
    return len(stack) == 0`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Simple parentheses"""
        self.assertTrue(is_valid_brackets("()"))

    def test_2(self):
        """Multiple types"""
        self.assertTrue(is_valid_brackets("()[]{}"))

    def test_3(self):
        """Nested brackets"""
        self.assertTrue(is_valid_brackets("([{}])"))

    def test_4(self):
        """Mismatched types"""
        self.assertFalse(is_valid_brackets("(]"))

    def test_5(self):
        """Wrong nesting order"""
        self.assertFalse(is_valid_brackets("([)]"))

    def test_6(self):
        """Empty string"""
        self.assertTrue(is_valid_brackets(""))

    def test_7(self):
        """Only opening"""
        self.assertFalse(is_valid_brackets("((("))

    def test_8(self):
        """Only closing"""
        self.assertFalse(is_valid_brackets(")))"))

    def test_9(self):
        """Complex valid"""
        self.assertTrue(is_valid_brackets("{[()]}()[]"))

    def test_10(self):
        """Single bracket"""
        self.assertFalse(is_valid_brackets("["))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use a list as a stack. Push opening brackets, pop when you see a closing bracket.',
	hint2: 'Create a dict mapping closing to opening brackets. Check if stack top matches when you see a closing bracket.',

	whyItMatters: `The stack data structure is essential for parsing, expression evaluation, and syntax validation.

**Production Pattern:**

\`\`\`python
def evaluate_expression(tokens: list[str]) -> int:
    """Evaluate arithmetic expression using two stacks."""
    values = []
    operators = []

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}

    def apply_operator():
        right = values.pop()
        left = values.pop()
        op = operators.pop()
        if op == "+":
            values.append(left + right)
        elif op == "-":
            values.append(left - right)
        elif op == "*":
            values.append(left * right)
        elif op == "/":
            values.append(left // right)

    for token in tokens:
        if token.isdigit():
            values.append(int(token))
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()  # Remove "("
        elif token in precedence:
            while (operators and operators[-1] != "(" and
                   operators[-1] in precedence and
                   precedence[operators[-1]] >= precedence[token]):
                apply_operator()
            operators.append(token)

    while operators:
        apply_operator()

    return values[0]

def find_matching_tags(html: str) -> list[tuple[str, str]]:
    """Find matching HTML open/close tags."""
    import re
    stack = []
    matches = []
    pattern = r'<(/?)(\w+)[^>]*>'

    for match in re.finditer(pattern, html):
        is_closing = match.group(1) == "/"
        tag_name = match.group(2)

        if is_closing:
            if stack and stack[-1][0] == tag_name:
                open_tag = stack.pop()
                matches.append((open_tag[1], match.start()))
        else:
            stack.append((tag_name, match.start()))

    return matches
\`\`\`

**Practical Benefits:**
- Syntax highlighting in code editors
- HTML/XML validation and parsing
- Expression evaluation in calculators`,

	translations: {
		ru: {
			title: 'Валидные скобки',
			description: `# Валидные скобки

Используйте стек для проверки соответствия скобок в выражении.

## Задача

Реализуйте функцию \`is_valid_brackets(s)\`, которая проверяет правильность скобок в строке.

## Требования

- Обработайте три типа: \`()\`, \`{}\`, \`[]\`
- Каждая открывающая скобка должна иметь соответствующую закрывающую
- Скобки должны закрываться в правильном порядке
- Верните \`True\` для валидной строки, \`False\` — для невалидной

## Примеры

\`\`\`python
>>> is_valid_brackets("()")
True

>>> is_valid_brackets("()[]{}")
True

>>> is_valid_brackets("([{}])")
True

>>> is_valid_brackets("(]")
False

>>> is_valid_brackets("([)]")
False

>>> is_valid_brackets("")
True
\`\`\``,
			hint1: 'Используйте список как стек. Добавляйте открывающие скобки, извлекайте при закрывающих.',
			hint2: 'Создайте словарь отображения закрывающих в открывающие. Проверяйте вершину стека.',
			whyItMatters: `Стек необходим для парсинга, вычисления выражений и валидации синтаксиса.

**Продакшен паттерн:**

\`\`\`python
def evaluate_expression(tokens: list[str]) -> int:
    """Вычисление арифметического выражения с двумя стеками."""
    values = []
    operators = []
    # ... реализация с приоритетом операторов

def find_matching_tags(html: str) -> list[tuple]:
    """Поиск соответствующих HTML тегов."""
    import re
    stack = []
    matches = []
    # ... парсинг с использованием стека
\`\`\`

**Практические преимущества:**
- Подсветка синтаксиса в редакторах кода
- Валидация и парсинг HTML/XML
- Вычисление выражений в калькуляторах`,
		},
		uz: {
			title: "To'g'ri qavslar",
			description: `# To'g'ri qavslar

Ifodalardagi qavslar mosligi ni tekshirish uchun stackdan foydalaning.

## Vazifa

Satrdagi qavslar to'g'riligini tekshiruvchi \`is_valid_brackets(s)\` funksiyasini amalga oshiring.

## Talablar

- Uchta turni ishlang: \`()\`, \`{}\`, \`[]\`
- Har bir ochiluvchi qavsning mos yopiluvchisi bo'lishi kerak
- Qavslar to'g'ri tartibda yopilishi kerak
- To'g'ri bo'lsa \`True\`, noto'g'ri bo'lsa \`False\` qaytaring

## Misollar

\`\`\`python
>>> is_valid_brackets("()")
True

>>> is_valid_brackets("()[]{}")
True

>>> is_valid_brackets("([{}])")
True

>>> is_valid_brackets("(]")
False

>>> is_valid_brackets("([)]")
False

>>> is_valid_brackets("")
True
\`\`\``,
			hint1: "Ro'yxatdan stack sifatida foydalaning. Ochiluvchi qavslarni qo'shing, yopiluvchida oling.",
			hint2: "Yopiluvchilarni ochiluvchilarga bog'lovchi lug'at yarating. Stack tepasini tekshiring.",
			whyItMatters: `Stack ma'lumot tuzilmasi parsing, ifoda hisoblash va sintaksis tekshiruvi uchun zarur.

**Ishlab chiqarish patterni:**

\`\`\`python
def evaluate_expression(tokens: list[str]) -> int:
    """Arifmetik ifodani ikki stack bilan hisoblash."""
    values = []
    operators = []
    # ... operator ustunligi bilan amalga oshirish

def find_matching_tags(html: str) -> list[tuple]:
    """Mos HTML teglarini topish."""
    import re
    stack = []
    matches = []
    # ... stack yordamida parsing
\`\`\`

**Amaliy foydalari:**
- Kod muharrirlarida sintaksis yoritish
- HTML/XML tekshiruvi va parsing
- Kalkulyatorlarda ifodalarni hisoblash`,
		},
	},
};

export default task;
