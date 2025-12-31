import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-missing-number',
	title: 'Missing Number',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'xor', 'array', 'math'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the missing number in an array containing n distinct numbers from 0 to n.

**Problem:**

Given an array \`nums\` containing \`n\` distinct numbers in the range \`[0, n]\`, return the only number in the range that is missing from the array.

**Examples:**

\`\`\`
Input: nums = [3, 0, 1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0, 3].
2 is missing since it is not in the array.

Input: nums = [0, 1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0, 2].
2 is missing since it is not in the array.

Input: nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
Output: 8
\`\`\`

**Key Insight:**

Use XOR! XOR all numbers from 0 to n, then XOR with all elements in the array. Since a ^ a = 0, all paired numbers cancel out, leaving only the missing number.

**Alternative:** Use math: sum(0 to n) - sum(nums) = missing number.

**Constraints:**
- n == nums.length
- 1 <= n <= 10^4
- 0 <= nums[i] <= n
- All numbers are unique

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def missing_number(nums: List[int]) -> int:
    # TODO: Find the missing number in range [0, n]

    return 0`,
	solutionCode: `from typing import List
from functools import reduce

def missing_number(nums: List[int]) -> int:
    """
    Find missing number using XOR.
    """
    n = len(nums)
    result = n  # Include n in the XOR
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result


# Approach 2: Using math (Gauss formula)
def missing_number_math(nums: List[int]) -> int:
    """Using sum formula: n * (n + 1) / 2."""
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# Approach 3: Using reduce with XOR
def missing_number_reduce(nums: List[int]) -> int:
    """Using reduce to XOR all numbers."""
    n = len(nums)
    # XOR all numbers from 0 to n
    xor_range = reduce(lambda a, b: a ^ b, range(n + 1))
    # XOR with all elements in nums
    xor_nums = reduce(lambda a, b: a ^ b, nums)
    return xor_range ^ xor_nums


# Approach 4: Using set (O(n) space)
def missing_number_set(nums: List[int]) -> int:
    """Using set difference."""
    num_set = set(nums)
    n = len(nums)
    for i in range(n + 1):
        if i not in num_set:
            return i
    return -1


# Approach 5: Cyclic sort (modifies input)
def missing_number_cyclic_sort(nums: List[int]) -> int:
    """Place each number at its index, find empty spot."""
    n = len(nums)

    # Place each number at its index if possible
    i = 0
    while i < n:
        correct_idx = nums[i]
        if correct_idx < n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Find index with wrong number
    for i in range(n):
        if nums[i] != i:
            return i

    return n


# Approach 6: Binary search (if sorted)
def missing_number_binary_search(nums: List[int]) -> int:
    """Binary search on sorted array."""
    nums_sorted = sorted(nums)
    left, right = 0, len(nums)

    while left < right:
        mid = (left + right) // 2
        if nums_sorted[mid] > mid:
            right = mid
        else:
            left = mid + 1

    return left


# Find multiple missing numbers
def find_all_missing(nums: List[int], upper: int) -> List[int]:
    """Find all missing numbers in range [1, upper]."""
    present = set(nums)
    return [i for i in range(1, upper + 1) if i not in present]`,
	testCode: `import pytest
from solution import missing_number


class TestMissingNumber:
    def test_basic_case(self):
        """Test basic case"""
        assert missing_number([3, 0, 1]) == 2

    def test_missing_at_end(self):
        """Test missing number at end"""
        assert missing_number([0, 1]) == 2

    def test_larger_array(self):
        """Test larger array"""
        assert missing_number([9, 6, 4, 2, 3, 5, 7, 0, 1]) == 8

    def test_missing_zero(self):
        """Test missing zero"""
        assert missing_number([1]) == 0

    def test_missing_one(self):
        """Test missing one"""
        assert missing_number([0]) == 1

    def test_missing_middle(self):
        """Test missing in middle"""
        assert missing_number([0, 1, 3]) == 2

    def test_sorted_array(self):
        """Test sorted array"""
        assert missing_number([0, 1, 2, 3, 5]) == 4

    def test_reverse_sorted(self):
        """Test reverse sorted array"""
        assert missing_number([5, 4, 3, 2, 0]) == 1

    def test_single_element_zero(self):
        """Test single element is zero"""
        assert missing_number([0]) == 1

    def test_large_missing(self):
        """Test larger range"""
        nums = list(range(100))
        nums.remove(50)
        assert missing_number(nums) == 50`,
	hint1: `XOR properties: a ^ a = 0 and a ^ 0 = a. If you XOR all indices 0 to n with all elements, paired numbers cancel out.`,
	hint2: `Initialize result = n (the length), then XOR both the index i and nums[i] for each element. The unpaired number (missing one) remains.`,
	whyItMatters: `Missing Number elegantly demonstrates how XOR can solve problems without extra space. It's a classic example of using mathematical properties for O(1) space solutions.

**Why This Matters:**

**1. XOR for Finding Unique Elements**

\`\`\`python
# XOR properties make it perfect for this:
# a ^ a = 0 (duplicates cancel)
# a ^ 0 = a (identity)

# XOR(0, 1, 2, ..., n) ^ XOR(nums) = missing
# All present numbers appear twice (once in each XOR)
# Missing number appears only once
\`\`\`

**2. Math Alternative**

\`\`\`python
# Gauss formula: sum(0 to n) = n * (n + 1) / 2
# This avoids overflow issues in some languages
# In Python, integers have arbitrary precision

# Be careful: for very large n, sum might overflow in other languages
# XOR approach never overflows
\`\`\`

**3. Related Problems**

\`\`\`python
# Single Number: one unique, others appear twice
# Missing Two Numbers: two missing from range
# Find Duplicate: one number appears twice
# Find All Duplicates: multiple duplicates

# All can use XOR or math properties
\`\`\`

**4. Interview Patterns**

\`\`\`python
# When you see:
# - "Find missing/unique element"
# - "O(1) space required"
# - "Numbers in specific range"

# Think about:
# - XOR approach
# - Math formula approach
# - Cyclic sort (if modification allowed)
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Пропущенное число',
			description: `Найдите пропущенное число в массиве из n различных чисел от 0 до n.

**Задача:**

Дан массив \`nums\` из \`n\` различных чисел в диапазоне \`[0, n]\`. Верните единственное пропущенное число.

**Примеры:**

\`\`\`
Вход: nums = [3, 0, 1]
Выход: 2
Объяснение: n = 3, все числа в диапазоне [0, 3]. 2 отсутствует.

Вход: nums = [0, 1]
Выход: 2

Вход: nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
Выход: 8
\`\`\`

**Ключевая идея:**

Используйте XOR! XOR всех чисел от 0 до n, затем XOR с элементами массива. Так как a ^ a = 0, парные числа сокращаются, остаётся пропущенное.

**Альтернатива:** Математика: sum(0 до n) - sum(nums) = пропущенное.

**Ограничения:**
- n == nums.length
- 1 <= n <= 10^4

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Свойства XOR: a ^ a = 0 и a ^ 0 = a. Если применить XOR ко всем индексам 0 до n и всем элементам, парные сократятся.`,
			hint2: `Инициализируйте result = n, затем XOR и индекса i, и nums[i] для каждого элемента. Непарное число (пропущенное) останется.`,
			whyItMatters: `Missing Number демонстрирует как XOR решает задачи без дополнительной памяти - классический пример O(1) пространственных решений.

**Почему это важно:**

**1. XOR для поиска уникальных элементов**

Свойства XOR: a ^ a = 0, a ^ 0 = a. Все присутствующие числа появляются дважды и сокращаются.

**2. Математическая альтернатива**

Формула Гаусса: sum(0 до n) = n * (n + 1) / 2. XOR не переполняется.

**3. Связанные задачи**

Single Number, Missing Two, Find Duplicate - все используют XOR или математику.

**4. Паттерны собеседований**

Когда видите "найти пропущенный элемент" + "O(1) память" - думайте о XOR.`,
			solutionCode: `from typing import List

def missing_number(nums: List[int]) -> int:
    """Находит пропущенное число используя XOR."""
    n = len(nums)
    result = n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result`
		},
		uz: {
			title: "Yo'qolgan son",
			description: `0 dan n gacha n ta turli sonlardan iborat massivda yo'qolgan sonni toping.

**Masala:**

\`[0, n]\` diapazonidagi \`n\` ta turli sonlardan iborat \`nums\` massivi berilgan. Diapazondagi yagona yo'qolgan sonni qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [3, 0, 1]
Chiqish: 2
Tushuntirish: n = 3, barcha sonlar [0, 3] diapazonida. 2 yo'q.

Kirish: nums = [0, 1]
Chiqish: 2

Kirish: nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
Chiqish: 8
\`\`\`

**Asosiy tushuncha:**

XOR ishlating! 0 dan n gacha barcha sonlarni XOR qiling, keyin massiv elementlari bilan XOR qiling. a ^ a = 0 bo'lgani uchun juft sonlar qisqaradi, yo'qolgan son qoladi.

**Alternativ:** Matematika: sum(0 dan n) - sum(nums) = yo'qolgan.

**Cheklovlar:**
- n == nums.length
- 1 <= n <= 10^4

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `XOR xususiyatlari: a ^ a = 0 va a ^ 0 = a. Barcha indekslar 0 dan n gacha va barcha elementlarga XOR qo'llasangiz, juftlar qisqaradi.`,
			hint2: `result = n bilan boshlang, keyin har bir element uchun i va nums[i] ni XOR qiling. Juft bo'lmagan son (yo'qolgan) qoladi.`,
			whyItMatters: `Missing Number XOR qanday qilib qo'shimcha xotirasiz masalalarni hal qilishini ko'rsatadi - O(1) fazoviy yechimlarning klassik misoli.

**Bu nima uchun muhim:**

**1. Noyob elementlarni topish uchun XOR**

XOR xususiyatlari: a ^ a = 0, a ^ 0 = a. Barcha mavjud sonlar ikki marta paydo bo'lib qisqaradi.

**2. Matematik alternativ**

Gauss formulasi: sum(0 dan n) = n * (n + 1) / 2. XOR hech qachon overflow bo'lmaydi.

**3. Bog'liq masalalar**

Single Number, Missing Two, Find Duplicate - barchasi XOR yoki matematika ishlatadi.

**4. Intervyu patternlari**

"Yo'qolgan elementni topish" + "O(1) xotira" ko'rsangiz - XOR haqida o'ylang.`,
			solutionCode: `from typing import List

def missing_number(nums: List[int]) -> int:
    """XOR yordamida yo'qolgan sonni topadi."""
    n = len(nums)
    result = n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result`
		}
	}
};

export default task;
