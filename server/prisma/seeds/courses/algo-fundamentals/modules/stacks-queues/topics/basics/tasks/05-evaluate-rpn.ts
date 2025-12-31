import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-evaluate-rpn',
	title: 'Evaluate Reverse Polish Notation',
	difficulty: 'medium',
	tags: ['python', 'stack', 'expression'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Evaluate an expression in Reverse Polish Notation.

**Problem:**

Evaluate the value of an arithmetic expression in Reverse Polish Notation (postfix).

Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Division truncates toward zero (integer division).

**Examples:**

\`\`\`
Input: tokens = ["2", "1", "+", "3", "*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Input: tokens = ["4", "13", "5", "/", "+"]
Output: 6
Explanation: (4 + (13 / 5)) = (4 + 2) = 6

Input: tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
Output: 22
\`\`\`

**Stack Approach:**

1. For each token:
   - If number: push to stack
   - If operator: pop two operands, compute, push result
2. Final result is the only element in stack

**Time Complexity:** O(n)
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def eval_rpn(tokens: List[str]) -> int:
    # TODO: Evaluate Reverse Polish Notation expression

    return 0`,
	solutionCode: `from typing import List


def eval_rpn(tokens: List[str]) -> int:
    """
    Evaluate Reverse Polish Notation expression.

    Args:
        tokens: List of tokens (numbers and operators)

    Returns:
        Result of the expression
    """
    stack = []

    for token in tokens:
        if token == "+":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif token == "-":
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif token == "*":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif token == "/":
            b = stack.pop()
            a = stack.pop()
            # Python division truncates toward negative infinity
            # We need to truncate toward zero like integer division
            stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]`,
	testCode: `import pytest
from solution import eval_rpn


class TestEvalRPN:
    def test_simple(self):
        """Test simple expression"""
        assert eval_rpn(["2", "1", "+", "3", "*"]) == 9

    def test_division(self):
        """Test with division"""
        assert eval_rpn(["4", "13", "5", "/", "+"]) == 6

    def test_complex(self):
        """Test complex expression"""
        tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
        assert eval_rpn(tokens) == 22

    def test_single(self):
        """Test single number"""
        assert eval_rpn(["5"]) == 5

    def test_subtraction(self):
        """Test subtraction"""
        assert eval_rpn(["4", "3", "-"]) == 1

    def test_negative_result(self):
        """Test negative result"""
        assert eval_rpn(["3", "4", "-"]) == -1

    def test_negative_division(self):
        """Test division truncates toward zero"""
        assert eval_rpn(["7", "-3", "/"]) == -2

    def test_multiplication(self):
        """Test multiplication"""
        assert eval_rpn(["5", "6", "*"]) == 30

    def test_all_operations(self):
        """Test all four operations"""
        assert eval_rpn(["15", "7", "1", "1", "+", "-", "/", "3", "*", "2", "1", "1", "+", "+", "-"]) == 5

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert eval_rpn(["-5", "2", "+"]) == -3`,
	hint1: `When you see an operator, pop two values from the stack. Remember the order: the second popped value (a) is the left operand, and the first popped (b) is the right operand.`,
	hint2: `Use int() to convert string numbers to integers. Use if/elif statements to handle the four operators.`,
	whyItMatters: `This classic problem demonstrates stack-based expression evaluation.

**Why This Matters:**

**1. Why RPN?**

Reverse Polish Notation eliminates need for parentheses:
\`\`\`
Infix:   (2 + 1) * 3
Postfix: 2 1 + 3 *  <- No parentheses needed!

Infix:   2 + 1 * 3  <- Needs operator precedence
Postfix: 2 1 3 * +  <- Order of operations is explicit
\`\`\`

**2. Compiler Design**

Expression evaluation pipeline:
\`\`\`
Infix -> Parse Tree -> Postfix -> Evaluate
"2+3"    (tree)      "2 3 +"    5
\`\`\`

**3. Calculator Design**

HP calculators use RPN because:
- No equals button needed
- No parentheses needed
- Can see intermediate results

**4. Order Matters**

For non-commutative operations:
\`\`\`python
# "3 4 -" means 3 - 4, not 4 - 3
# Pop b=4, pop a=3, compute a-b = 3-4 = -1
b = stack.pop()  # Second operand
a = stack.pop()  # First operand
result = a - b   # Order matters!
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Вычисление обратной польской нотации',
			description: `Вычислите выражение в обратной польской нотации.

**Задача:**

Вычислите значение арифметического выражения в обратной польской нотации (постфиксной).

Допустимые операторы: +, -, *, /. Каждый операнд может быть целым числом или другим выражением.

Деление усекается к нулю (целочисленное деление).

**Примеры:**

\`\`\`
Вход: tokens = ["2", "1", "+", "3", "*"]
Выход: 9
Объяснение: ((2 + 1) * 3) = 9

Вход: tokens = ["4", "13", "5", "/", "+"]
Выход: 6
Объяснение: (4 + (13 / 5)) = (4 + 2) = 6
\`\`\`

**Подход со стеком:**

1. Для каждого токена:
   - Если число: push в стек
   - Если оператор: pop два операнда, вычислить, push результат
2. Финальный результат - единственный элемент в стеке

**Временная сложность:** O(n)
**Пространственная сложность:** O(n)`,
			hint1: `Когда видите оператор, извлеките два значения из стека. Помните порядок: второе извлечённое (a) - левый операнд, первое (b) - правый.`,
			hint2: `Используйте int() для преобразования строковых чисел в целые. Используйте if/elif для обработки четырёх операторов.`,
			whyItMatters: `Эта классическая задача демонстрирует вычисление выражений на основе стека.

**Почему это важно:**

**1. Почему RPN?**

Обратная польская нотация устраняет необходимость в скобках.

**2. Проектирование компиляторов**

Конвейер вычисления выражений: Infix -> Parse Tree -> Postfix -> Evaluate`,
			solutionCode: `from typing import List


def eval_rpn(tokens: List[str]) -> int:
    """
    Вычисляет выражение в обратной польской нотации.

    Args:
        tokens: Список токенов (числа и операторы)

    Returns:
        Результат выражения
    """
    stack = []

    for token in tokens:
        if token == "+":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif token == "-":
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif token == "*":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif token == "/":
            b = stack.pop()
            a = stack.pop()
            # Деление усекается к нулю
            stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]`
		},
		uz: {
			title: 'Teskari Polsha yozuvini hisoblash',
			description: `Teskari Polsha yozuvidagi ifodani hisoblang.

**Masala:**

Teskari Polsha yozuvidagi (postfiks) arifmetik ifoda qiymatini hisoblang.

Yaroqli operatorlar: +, -, *, /. Har bir operand butun son yoki boshqa ifoda bo'lishi mumkin.

Bo'lish nolga qarab qisqartiriladi (butun sonli bo'lish).

**Misollar:**

\`\`\`
Kirish: tokens = ["2", "1", "+", "3", "*"]
Chiqish: 9
Tushuntirish: ((2 + 1) * 3) = 9

Kirish: tokens = ["4", "13", "5", "/", "+"]
Chiqish: 6
Tushuntirish: (4 + (13 / 5)) = (4 + 2) = 6
\`\`\`

**Stek yondashuvi:**

1. Har bir token uchun:
   - Agar raqam: stekga push qiling
   - Agar operator: ikkita operandni pop qiling, hisoblang, natijani push qiling
2. Yakuniy natija stekdagi yagona element

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Operator ko'rsangiz, stekdan ikkita qiymat oling. Tartibni eslang: ikkinchi olingan (a) chap operand, birinchi (b) o'ng operand.`,
			hint2: `Satrli raqamlarni butun sonlarga aylantirish uchun int() dan foydalaning. To'rtta operatorni qayta ishlash uchun if/elif dan foydalaning.`,
			whyItMatters: `Bu klassik masala stekga asoslangan ifoda hisoblashni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Nima uchun RPN?**

Teskari Polsha yozuvi qavslarga ehtiyojni yo'q qiladi.

**2. Kompilyator loyihalash**

Ifoda hisoblash quvuri: Infix -> Parse Tree -> Postfix -> Evaluate`,
			solutionCode: `from typing import List


def eval_rpn(tokens: List[str]) -> int:
    """
    Teskari Polsha yozuvidagi ifodani hisoblaydi.

    Args:
        tokens: Tokenlar ro'yxati (raqamlar va operatorlar)

    Returns:
        Ifoda natijasi
    """
    stack = []

    for token in tokens:
        if token == "+":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif token == "-":
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif token == "*":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif token == "/":
            b = stack.pop()
            a = stack.pop()
            # Bo'lish nolga qarab qisqartiriladi
            stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]`
		}
	}
};

export default task;
