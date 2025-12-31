import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-single-number',
	title: 'Single Number',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'xor', 'array'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the element that appears only once in an array where every other element appears twice.

**Problem:**

Given a **non-empty** array of integers \`nums\`, every element appears **twice** except for one. Find that single one.

You must implement a solution with **linear runtime complexity** and use only **constant extra space**.

**Examples:**

\`\`\`
Input: nums = [2, 2, 1]
Output: 1

Input: nums = [4, 1, 2, 1, 2]
Output: 4

Input: nums = [1]
Output: 1
\`\`\`

**Key Insight:**

XOR has special properties:
- \`a ^ a = 0\` (same numbers cancel out)
- \`a ^ 0 = a\` (XOR with 0 returns same number)
- XOR is commutative and associative

So XOR all numbers: pairs cancel out, leaving the single number.

**Constraints:**
- 1 <= nums.length <= 3 * 10^4
- -3 * 10^4 <= nums[i] <= 3 * 10^4
- Each element appears twice except for one

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def single_number(nums: List[int]) -> int:
    # TODO: Find the element that appears only once (others appear twice)

    return 0`,
	solutionCode: `from typing import List
from functools import reduce
import operator

def single_number(nums: List[int]) -> int:
    """
    Find the element that appears only once using XOR.
    """
    result = 0
    for num in nums:
        result ^= num
    return result


# Using reduce
def single_number_reduce(nums: List[int]) -> int:
    """Using functools.reduce with XOR."""
    return reduce(operator.xor, nums)


# Using reduce with lambda
def single_number_lambda(nums: List[int]) -> int:
    """Using reduce with lambda."""
    return reduce(lambda a, b: a ^ b, nums)


# Single Number II: Every element appears 3 times except one
def single_number_ii(nums: List[int]) -> int:
    """
    Find element appearing once when others appear 3 times.
    Count bits at each position modulo 3.
    """
    result = 0
    for i in range(32):
        bit_sum = 0
        for num in nums:
            # Count 1s at position i
            bit_sum += (num >> i) & 1

        # If bit_sum % 3 != 0, single number has 1 at this position
        if bit_sum % 3:
            # Handle negative numbers (sign bit)
            if i == 31:
                result -= (1 << i)
            else:
                result |= (1 << i)

    return result


# Single Number III: Two elements appear once, others twice
def single_number_iii(nums: List[int]) -> List[int]:
    """
    Find two elements appearing once when others appear twice.
    """
    # XOR all numbers: result is a ^ b
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit (a and b differ at this bit)
    diff_bit = xor_all & (-xor_all)

    # Partition numbers by this bit and XOR separately
    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num

    return [a, b]`,
	testCode: `import pytest
from solution import single_number


class TestSingleNumber:
    def test_basic_case(self):
        """Test basic case"""
        assert single_number([2, 2, 1]) == 1

    def test_multiple_pairs(self):
        """Test with multiple pairs"""
        assert single_number([4, 1, 2, 1, 2]) == 4

    def test_single_element(self):
        """Test single element array"""
        assert single_number([1]) == 1

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert single_number([-1, -1, 5]) == 5

    def test_single_at_start(self):
        """Test single at start"""
        assert single_number([1, 2, 2]) == 1

    def test_single_at_end(self):
        """Test single at end"""
        assert single_number([2, 2, 3]) == 3

    def test_larger_array(self):
        """Test larger array"""
        assert single_number([5, 3, 5, 7, 3, 9, 7]) == 9

    def test_zero_as_single(self):
        """Test zero as single number"""
        assert single_number([0, 1, 1]) == 0

    def test_all_same_except_one(self):
        """Test many pairs of same number"""
        assert single_number([7, 7, 7, 7, 5, 7, 7]) != 5  # Invalid input per constraints
        # Testing valid case:
        assert single_number([7, 7, 5]) == 5

    def test_mixed_positive_negative(self):
        """Test mixed positive and negative"""
        assert single_number([-1, 2, -1, 3, 2]) == 3`,
	hint1: `XOR has the property that a ^ a = 0 (same numbers cancel each other out) and a ^ 0 = a. What happens if you XOR all numbers together?`,
	hint2: `Since every number except one appears twice, all pairs will cancel out (x ^ x = 0), leaving only the single number. Initialize result = 0 and XOR each number.`,
	whyItMatters: `Single Number is the gateway problem to bit manipulation. Understanding XOR properties opens up a whole class of elegant O(1) space solutions for problems involving duplicates.

**Why This Matters:**

**1. XOR Properties**

\`\`\`python
# Fundamental XOR properties:
a ^ a = 0       # Same numbers cancel
a ^ 0 = a       # Identity
a ^ b = b ^ a   # Commutative
(a ^ b) ^ c = a ^ (b ^ c)  # Associative

# These properties make XOR perfect for:
# - Finding unique elements
# - Swapping without temp variable
# - Encryption (simple XOR cipher)
\`\`\`

**2. Single Number Variants**

\`\`\`python
# Single Number II: Others appear 3 times
# Count bits at each position mod 3
for i in range(32):
    bit_count = sum((num >> i) & 1 for num in nums)
    if bit_count % 3:
        result |= (1 << i)

# Single Number III: Two unique numbers
# XOR all -> a ^ b
# Find differing bit
# Partition and XOR separately
\`\`\`

**3. Related Tricks**

\`\`\`python
# Swap without temp:
a ^= b
b ^= a
a ^= b

# Check if bit is set:
if num & (1 << i):
    # bit i is set

# XOR from 0 to n:
# Useful for missing number problems
\`\`\`

**4. Applications**

\`\`\`python
# Find missing number in sequence
# Detect errors in data transmission
# Simple encryption/decryption
# Memory-efficient duplicate detection
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Одиночное число',
			description: `Найдите элемент, который появляется только один раз, когда все остальные появляются дважды.

**Задача:**

Дан **непустой** массив целых чисел \`nums\`, каждый элемент появляется **дважды** кроме одного. Найдите это одиночное число.

Решение должно работать за **линейное время** и использовать **O(1) дополнительной памяти**.

**Примеры:**

\`\`\`
Вход: nums = [2, 2, 1]
Выход: 1

Вход: nums = [4, 1, 2, 1, 2]
Выход: 4
\`\`\`

**Ключевая идея:**

XOR имеет особые свойства:
- \`a ^ a = 0\` (одинаковые числа сокращаются)
- \`a ^ 0 = a\`
- XOR коммутативен и ассоциативен

Применив XOR ко всем числам, пары сократятся, останется одиночное число.

**Ограничения:**
- 1 <= nums.length <= 3 * 10^4

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `XOR имеет свойство a ^ a = 0. Что произойдёт если применить XOR ко всем числам?`,
			hint2: `Так как каждое число кроме одного появляется дважды, пары сократятся (x ^ x = 0), останется только одиночное число.`,
			whyItMatters: `Single Number - вводная задача в битовые манипуляции. Понимание свойств XOR открывает класс элегантных O(1) решений.

**Почему это важно:**

**1. Свойства XOR**

a ^ a = 0, a ^ 0 = a, коммутативность и ассоциативность.

**2. Варианты задачи**

Single Number II (остальные по 3 раза), Single Number III (два уникальных).

**3. Применения**

Поиск пропущенного числа, обнаружение ошибок, шифрование.`,
			solutionCode: `from typing import List

def single_number(nums: List[int]) -> int:
    """Находит элемент, появляющийся только один раз, используя XOR."""
    result = 0
    for num in nums:
        result ^= num
    return result`
		},
		uz: {
			title: 'Yagona son',
			description: `Boshqalari ikki marta paydo bo'lganda faqat bir marta paydo bo'ladigan elementni toping.

**Masala:**

**Bo'sh bo'lmagan** butun sonlar massivi \`nums\` berilgan, har bir element bittasidan tashqari **ikki marta** paydo bo'ladi. O'sha yagona sonni toping.

Yechim **chiziqli vaqt** murakkabligida va **O(1) qo'shimcha xotira** ishlatishi kerak.

**Misollar:**

\`\`\`
Kirish: nums = [2, 2, 1]
Chiqish: 1

Kirish: nums = [4, 1, 2, 1, 2]
Chiqish: 4
\`\`\`

**Asosiy tushuncha:**

XOR maxsus xususiyatlarga ega:
- \`a ^ a = 0\` (bir xil sonlar qisqaradi)
- \`a ^ 0 = a\`
- XOR kommutativ va assotsiativ

Barcha sonlarga XOR qo'llab, juftliklar qisqaradi, yagona son qoladi.

**Cheklovlar:**
- 1 <= nums.length <= 3 * 10^4

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `XOR xususiyati a ^ a = 0. Barcha sonlarga XOR qo'llasangiz nima bo'ladi?`,
			hint2: `Har bir son bittasidan tashqari ikki marta paydo bo'lgani uchun juftliklar qisqaradi (x ^ x = 0), faqat yagona son qoladi.`,
			whyItMatters: `Single Number - bit manipulyatsiyasiga kirish masalasi. XOR xususiyatlarini tushunish O(1) yechimlar sinfini ochadi.

**Bu nima uchun muhim:**

**1. XOR xususiyatlari**

a ^ a = 0, a ^ 0 = a, kommutativlik va assotsiativlik.

**2. Masala variantlari**

Single Number II (qolganlari 3 marta), Single Number III (ikkita noyob).

**3. Qo'llanishlar**

Yo'qolgan sonni topish, xatolarni aniqlash, shifrlash.`,
			solutionCode: `from typing import List

def single_number(nums: List[int]) -> int:
    """XOR yordamida faqat bir marta paydo bo'ladigan elementni topadi."""
    result = 0
    for num in nums:
        result ^= num
    return result`
		}
	}
};

export default task;
