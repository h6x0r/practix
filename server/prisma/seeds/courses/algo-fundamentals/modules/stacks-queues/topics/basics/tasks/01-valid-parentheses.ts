import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-valid-parentheses',
	title: 'Valid Parentheses',
	difficulty: 'easy',
	tags: ['python', 'stack', 'string'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if a string of parentheses is valid.

**Problem:**

Given a string \`s\` containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

A string is valid if:
1. Open brackets must be closed by the same type of brackets
2. Open brackets must be closed in the correct order
3. Every close bracket has a corresponding open bracket

**Examples:**

\`\`\`
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

Input: s = "([)]"
Output: false

Input: s = "{[]}"
Output: true
\`\`\`

**Stack Approach:**

1. For each character:
   - If opening bracket: push to stack
   - If closing bracket: pop from stack and check if matching
2. Stack should be empty at the end

**Time Complexity:** O(n)
**Space Complexity:** O(n)`,
	initialCode: `def is_valid(s: str) -> bool:
    # TODO: Check if parentheses string is valid

    return False`,
	solutionCode: `def is_valid(s: str) -> bool:
    """
    Check if parentheses string is valid.

    Args:
        s: String containing only '(', ')', '{', '}', '[', ']'

    Returns:
        True if valid, False otherwise
    """
    # Map closing brackets to their opening counterparts
    matching = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    # Stack to track opening brackets
    stack = []

    for char in s:
        # If opening bracket, push to stack
        if char in '([{':
            stack.append(char)
        else:
            # Closing bracket - check if matches top of stack
            if not stack:
                return False  # No opening bracket to match

            # Pop from stack
            top = stack.pop()

            # Check if matching
            if top != matching[char]:
                return False

    # Stack should be empty (all brackets matched)
    return len(stack) == 0`,
	testCode: `import pytest
from solution import is_valid


class TestIsValid:
    def test_simple(self):
        """Test simple pair"""
        assert is_valid("()") == True

    def test_multiple(self):
        """Test multiple pairs"""
        assert is_valid("()[]{}") == True

    def test_mismatch(self):
        """Test mismatched brackets"""
        assert is_valid("(]") == False

    def test_wrong_order(self):
        """Test wrong closing order"""
        assert is_valid("([)]") == False

    def test_nested(self):
        """Test nested brackets"""
        assert is_valid("{[]}") == True

    def test_empty(self):
        """Test empty string"""
        assert is_valid("") == True

    def test_single_open(self):
        """Test single opening bracket"""
        assert is_valid("(") == False

    def test_single_close(self):
        """Test single closing bracket"""
        assert is_valid(")") == False

    def test_deep_nested(self):
        """Test deeply nested brackets"""
        assert is_valid("((([[[]]])))")  == True

    def test_complex_valid(self):
        """Test complex valid pattern"""
        assert is_valid("{[()]}") == True`,
	hint1: `Use a stack to keep track of opening brackets. When you see an opening bracket, push it. When you see a closing bracket, pop and check if it matches.`,
	hint2: `Create a map to match closing brackets to their opening counterparts. After processing all characters, the stack should be empty for a valid string.`,
	whyItMatters: `This classic problem teaches stack fundamentals with a practical application.

**Why This Matters:**

**1. Stack = LIFO**

Last In, First Out is perfect for matching:
\`\`\`python
# Most recent opening bracket must close first
"([" + "])"  # Works: ] closes [, ) closes (
"([" + "))"  # Fails: ) would close [ (wrong type)
\`\`\`

**2. Real-World Applications**

- Code editors (syntax highlighting, error detection)
- Compilers (parsing expressions)
- HTML/XML validators
- Mathematical expression evaluation

**3. Extended Problems**

This pattern extends to:
- Minimum Add to Make Parentheses Valid
- Longest Valid Parentheses
- Score of Parentheses
- Remove Invalid Parentheses

**4. Python Stack Implementation**

\`\`\`python
# Python uses list as stack:
stack = []

# Push
stack.append(value)

# Pop
top = stack.pop()

# Peek
top = stack[-1]

# IsEmpty
len(stack) == 0
# or: not stack
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Правильные скобки',
			description: `Определите, является ли строка со скобками правильной.

**Задача:**

Дана строка \`s\`, содержащая только символы '(', ')', '{', '}', '[' и ']', определите, является ли входная строка правильной.

Строка правильная, если:
1. Открывающие скобки закрываются скобками того же типа
2. Открывающие скобки закрываются в правильном порядке
3. Каждая закрывающая скобка имеет соответствующую открывающую

**Примеры:**

\`\`\`
Вход: s = "()"
Выход: true

Вход: s = "()[]{}"
Выход: true

Вход: s = "(]"
Выход: false
\`\`\`

**Подход со стеком:**

1. Для каждого символа:
   - Если открывающая скобка: push в стек
   - Если закрывающая: pop из стека и проверить соответствие
2. Стек должен быть пуст в конце

**Временная сложность:** O(n)
**Пространственная сложность:** O(n)`,
			hint1: `Используйте стек для отслеживания открывающих скобок. Когда видите открывающую - push. Когда видите закрывающую - pop и проверяйте соответствие.`,
			hint2: `Создайте map для сопоставления закрывающих скобок с открывающими. После обработки всех символов стек должен быть пуст для правильной строки.`,
			whyItMatters: `Эта классическая задача учит основам стека с практическим применением.

**Почему это важно:**

**1. Стек = LIFO**

Last In, First Out идеально подходит для сопоставления скобок.

**2. Реальные применения**

- Редакторы кода
- Компиляторы
- Валидаторы HTML/XML`,
			solutionCode: `def is_valid(s: str) -> bool:
    """
    Проверяет, является ли строка скобок правильной.

    Args:
        s: Строка, содержащая только '(', ')', '{', '}', '[', ']'

    Returns:
        True если правильная, иначе False
    """
    # Сопоставляем закрывающие скобки с открывающими
    matching = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    # Стек для отслеживания открывающих скобок
    stack = []

    for char in s:
        # Если открывающая скобка, добавляем в стек
        if char in '([{':
            stack.append(char)
        else:
            # Закрывающая скобка - проверяем соответствие с вершиной стека
            if not stack:
                return False  # Нет открывающей скобки для сопоставления

            # Извлекаем из стека
            top = stack.pop()

            # Проверяем соответствие
            if top != matching[char]:
                return False

    # Стек должен быть пуст (все скобки сопоставлены)
    return len(stack) == 0`
		},
		uz: {
			title: 'To\'g\'ri qavslar',
			description: `Qavslar satrining to'g'ri ekanligini aniqlang.

**Masala:**

Faqat '(', ')', '{', '}', '[' va ']' belgilarini o'z ichiga olgan \`s\` satr berilgan, kirish satrining to'g'ri ekanligini aniqlang.

Satr to'g'ri, agar:
1. Ochiluvchi qavslar bir xil turdagi qavslar bilan yopilsa
2. Ochiluvchi qavslar to'g'ri tartibda yopilsa
3. Har bir yopiluvchi qavsning mos ochiluvchi qavsi bo'lsa

**Misollar:**

\`\`\`
Kirish: s = "()"
Chiqish: true

Kirish: s = "()[]{}"
Chiqish: true

Kirish: s = "(]"
Chiqish: false
\`\`\`

**Stek yondashuvi:**

1. Har bir belgi uchun:
   - Agar ochiluvchi qavs: stekga push qiling
   - Agar yopiluvchi qavs: stekdan pop qiling va mosligini tekshiring
2. Stek oxirida bo'sh bo'lishi kerak

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Ochiluvchi qavslarni kuzatish uchun stekdan foydalaning. Ochiluvchi qavsni ko'rsangiz - push. Yopiluvchi qavsni ko'rsangiz - pop qiling va mosligini tekshiring.`,
			hint2: `Yopiluvchi qavslarni ochiluvchi qavslar bilan moslashtirish uchun map yarating. Barcha belgilarni qayta ishlagandan keyin to'g'ri satr uchun stek bo'sh bo'lishi kerak.`,
			whyItMatters: `Bu klassik masala amaliy qo'llanilish bilan stek asoslarini o'rgatadi.

**Bu nima uchun muhim:**

**1. Stek = LIFO**

Last In, First Out qavslarni moslashtirish uchun mukammal.

**2. Haqiqiy dunyo qo'llanilishlari**

- Kod muharrirlari
- Kompilyatorlar
- HTML/XML validatorlari`,
			solutionCode: `def is_valid(s: str) -> bool:
    """
    Qavslar satri to'g'ri ekanligini tekshiradi.

    Args:
        s: Faqat '(', ')', '{', '}', '[', ']' dan iborat satr

    Returns:
        To'g'ri bo'lsa True, aks holda False
    """
    # Yopiluvchi qavslarni ochiluvchi qavslar bilan moslashtiramiz
    matching = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    # Ochiluvchi qavslarni kuzatish uchun stek
    stack = []

    for char in s:
        # Agar ochiluvchi qavs bo'lsa, stekga qo'shamiz
        if char in '([{':
            stack.append(char)
        else:
            # Yopiluvchi qavs - stek tepasi bilan mosligini tekshiramiz
            if not stack:
                return False  # Moslashtirish uchun ochiluvchi qavs yo'q

            # Stekdan olamiz
            top = stack.pop()

            # Mosligini tekshiramiz
            if top != matching[char]:
                return False

    # Stek bo'sh bo'lishi kerak (barcha qavslar moslashtirilgan)
    return len(stack) == 0`
		}
	}
};

export default task;
