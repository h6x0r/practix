import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-letter-combinations',
	title: 'Letter Combinations of Phone Number',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'string'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Generate all letter combinations from a phone number.

**Problem:**

Given a string containing digits from 2-9, return all possible letter combinations that the number could represent on a phone keypad.

**Phone Keypad Mapping:**

\`\`\`
2 -> abc    3 -> def
4 -> ghi    5 -> jkl    6 -> mno
7 -> pqrs   8 -> tuv    9 -> wxyz
\`\`\`

**Examples:**

\`\`\`
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Input: digits = ""
Output: []

Input: digits = "2"
Output: ["a","b","c"]
\`\`\`

**Visualization:**

\`\`\`
digits = "23"

                 ""
       /         |         \\
      a          b          c       (digit 2: abc)
    / | \\      / | \\      / | \\
   ad ae af   bd be bf   cd ce cf   (digit 3: def)

Total: 3 × 3 = 9 combinations
\`\`\`

**Constraints:**
- 0 <= digits.length <= 4
- digits[i] is a digit in the range ['2', '9']

**Time Complexity:** O(4^n × n) where n = number of digits
**Space Complexity:** O(n) for recursion stack`,
	initialCode: `from typing import List

def letter_combinations(digits: str) -> List[str]:
    # TODO: Generate all letter combinations from phone digits

    return []`,
	solutionCode: `from typing import List

def letter_combinations(digits: str) -> List[str]:
    """
    Generate all letter combinations from phone digits.
    """
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index: int, current: str) -> None:
        if index == len(digits):
            result.append(current)
            return

        for letter in mapping[digits[index]]:
            backtrack(index + 1, current + letter)

    backtrack(0, '')
    return result


# Iterative approach
def letter_combinations_iterative(digits: str) -> List[str]:
    """Generate combinations iteratively."""
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = ['']

    for digit in digits:
        result = [combo + letter for combo in result for letter in mapping[digit]]

    return result


# Using product from itertools
from itertools import product

def letter_combinations_product(digits: str) -> List[str]:
    """Using itertools.product."""
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    letters = [mapping[d] for d in digits]
    return [''.join(p) for p in product(*letters)]`,
	testCode: `import pytest
from solution import letter_combinations


class TestLetterCombinations:
    def test_two_digits(self):
        """Test with two digits"""
        result = letter_combinations("23")
        assert len(result) == 9  # 3 × 3
        expected = ["ad","ae","af","bd","be","bf","cd","ce","cf"]
        assert sorted(result) == sorted(expected)

    def test_empty_input(self):
        """Test empty input"""
        assert letter_combinations("") == []

    def test_single_digit(self):
        """Test single digit"""
        assert sorted(letter_combinations("2")) == ['a', 'b', 'c']

    def test_digit_7(self):
        """Test digit 7 (4 letters)"""
        result = letter_combinations("7")
        assert result == ['p', 'q', 'r', 's']

    def test_correct_count(self):
        """Test correct number of combinations"""
        # 2->3, 3->3, 7->4, 9->4
        result = letter_combinations("279")
        assert len(result) == 3 * 4 * 4  # 48

    def test_all_same_digits(self):
        """Test same digit repeated"""
        result = letter_combinations("222")
        assert len(result) == 27  # 3^3

    def test_combination_length(self):
        """Test each combination has correct length"""
        result = letter_combinations("234")
        for combo in result:
            assert len(combo) == 3

    def test_valid_characters(self):
        """Test combinations contain only valid characters"""
        result = letter_combinations("23")
        valid_chars = set('abcdef')
        for combo in result:
            for char in combo:
                assert char in valid_chars

    def test_three_digits(self):
        """Test three digits"""
        result = letter_combinations("234")
        assert len(result) == 27  # 3 × 3 × 3

    def test_no_duplicates(self):
        """Test no duplicate combinations"""
        result = letter_combinations("234")
        assert len(result) == len(set(result))`,
	hint1: `Create a mapping from digit to letters. Use backtracking: for each digit position, try all possible letters and recurse to the next position.`,
	hint2: `Base case: when index equals digits length, add current string to result. For each letter in mapping[digits[index]], recurse with index+1 and current+letter.`,
	whyItMatters: `This problem demonstrates backtracking on a tree where each level has different choices. It's commonly used for generating combinations from multiple independent sets.

**Why This Matters:**

**1. Cartesian Product Pattern**

\`\`\`python
# When combining choices from multiple independent sets
# Use the same pattern:

def cartesian_product(sets):
    result = [[]]
    for s in sets:
        result = [r + [elem] for r in result for elem in s]
    return result

# Example: sizes × colors × styles
sizes = ['S', 'M', 'L']
colors = ['red', 'blue']
styles = ['A', 'B']
# Result: 3 × 2 × 2 = 12 combinations
\`\`\`

**2. Multiple Approaches**

\`\`\`python
# 1. Backtracking (recursive)
def backtrack(index, current):
    if index == len(digits):
        result.append(current)
        return
    for letter in mapping[digits[index]]:
        backtrack(index + 1, current + letter)

# 2. Iterative
result = ['']
for digit in digits:
    result = [r + l for r in result for l in mapping[digit]]

# 3. itertools.product
from itertools import product
letters = [mapping[d] for d in digits]
result = [''.join(p) for p in product(*letters)]
\`\`\`

**3. Similar Problems**

\`\`\`python
# Generate all passwords from character sets
# Generate all URLs from path segments
# Generate all test case combinations
# Multiple choice questionnaire scoring
\`\`\`

**4. Complexity Analysis**

\`\`\`
For n digits:
- Minimum combinations: 3^n (all digits have 3 letters)
- Maximum combinations: 4^n (if using 7 and 9 which have 4 letters)

Time: O(4^n × n)
- 4^n combinations
- n characters per combination

Space: O(n) for recursion stack
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Буквенные комбинации телефона',
			description: `Сгенерируйте все буквенные комбинации для телефонного номера.

**Задача:**

Дана строка с цифрами от 2 до 9. Верните все возможные буквенные комбинации.

**Соответствие клавиш:**

\`\`\`
2 -> abc    3 -> def
4 -> ghi    5 -> jkl    6 -> mno
7 -> pqrs   8 -> tuv    9 -> wxyz
\`\`\`

**Примеры:**

\`\`\`
Вход: digits = "23"
Выход: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Вход: digits = ""
Выход: []
\`\`\`

**Ограничения:**
- 0 <= digits.length <= 4

**Временная сложность:** O(4^n × n)
**Пространственная сложность:** O(n)`,
			hint1: `Создайте соответствие цифра -> буквы. Бэктрекинг: для каждой позиции пробуйте все буквы и рекурсивно переходите к следующей.`,
			hint2: `Базовый случай: когда индекс равен длине digits, добавьте текущую строку. Для каждой буквы в mapping[digits[index]] рекурсия с index+1 и current+letter.`,
			whyItMatters: `Задача демонстрирует бэктрекинг на дереве с разными выборами на каждом уровне.

**Почему это важно:**

**1. Паттерн декартова произведения**

При комбинировании выборов из независимых множеств.

**2. Три подхода**

Рекурсивный бэктрекинг, итеративный, itertools.product.

**3. Похожие задачи**

Генерация паролей, URL, тестовых случаев.`,
			solutionCode: `from typing import List

def letter_combinations(digits: str) -> List[str]:
    """Генерирует все буквенные комбинации для цифр телефона."""
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index: int, current: str) -> None:
        if index == len(digits):
            result.append(current)
            return

        for letter in mapping[digits[index]]:
            backtrack(index + 1, current + letter)

    backtrack(0, '')
    return result`
		},
		uz: {
			title: 'Telefon raqamining harf kombinatsiyalari',
			description: `Telefon raqami uchun barcha harf kombinatsiyalarini yarating.

**Masala:**

2-9 raqamlarini o'z ichiga olgan satr berilgan. Barcha mumkin bo'lgan harf kombinatsiyalarini qaytaring.

**Tugmalar mos kelishi:**

\`\`\`
2 -> abc    3 -> def
4 -> ghi    5 -> jkl    6 -> mno
7 -> pqrs   8 -> tuv    9 -> wxyz
\`\`\`

**Misollar:**

\`\`\`
Kirish: digits = "23"
Chiqish: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Kirish: digits = ""
Chiqish: []
\`\`\`

**Cheklovlar:**
- 0 <= digits.length <= 4

**Vaqt murakkabligi:** O(4^n × n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Raqamdan harflarga moslik yarating. Backtracking: har bir pozitsiya uchun barcha harflarni sinab ko'ring va keyingisiga rekursiv o'ting.`,
			hint2: `Asosiy holat: indeks digits uzunligiga teng bo'lganda joriy satrni qo'shing. mapping[digits[index]] dagi har bir harf uchun index+1 va current+letter bilan rekursiya.`,
			whyItMatters: `Masala har bir darajada turli tanlovlarga ega daraxtda backtracking ni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Dekart ko'paytmasi patterni**

Mustaqil to'plamlardan tanlovlarni birlashtirganda.

**2. Uchta yondashuv**

Rekursiv backtracking, iterativ, itertools.product.

**3. O'xshash masalalar**

Parollar, URL, test holatlari yaratish.`,
			solutionCode: `from typing import List

def letter_combinations(digits: str) -> List[str]:
    """Telefon raqamlari uchun barcha harf kombinatsiyalarini yaratadi."""
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index: int, current: str) -> None:
        if index == len(digits):
            result.append(current)
            return

        for letter in mapping[digits[index]]:
            backtrack(index + 1, current + letter)

    backtrack(0, '')
    return result`
		}
	}
};

export default task;
