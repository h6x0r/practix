import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-subsets',
	title: 'Subsets',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'bit-manipulation'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Generate all possible subsets (power set) of a given array.

**Problem:**

Given an integer array \`nums\` of **unique** elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in **any order**.

**Examples:**

\`\`\`
Input: nums = [1, 2, 3]
Output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]

Input: nums = [0]
Output: [[], [0]]

Input: nums = [1, 2]
Output: [[], [1], [2], [1, 2]]
\`\`\`

**Visualization (Backtracking Tree):**

\`\`\`
nums = [1, 2, 3]

                    []
           /        |        \\
         [1]       [2]       [3]
        /   \\       |
     [1,2] [1,3]  [2,3]
       |
   [1,2,3]

At each step, decide: include current element or not
\`\`\`

**Key Insight:**

For each element, we have two choices:
1. **Include** it in the current subset
2. **Exclude** it from the current subset

This gives us 2^n total subsets.

**Constraints:**
- 1 <= nums.length <= 10
- -10 <= nums[i] <= 10
- All elements are **unique**

**Time Complexity:** O(n × 2^n) - generate 2^n subsets, each takes O(n) to copy
**Space Complexity:** O(n) for recursion stack`,
	initialCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    # TODO: Generate all possible subsets (power set)

    return []`,
	solutionCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets using backtracking.
    """
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        # Add current subset to result
        result.append(current.copy())

        # Try adding each remaining element
        for i in range(start, len(nums)):
            current.append(nums[i])      # Choose
            backtrack(i + 1, current)    # Explore
            current.pop()                 # Unchoose

    backtrack(0, [])
    return result


# Iterative approach
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """Generate subsets iteratively."""
    result = [[]]

    for num in nums:
        # For each existing subset, create a new one with current number
        result += [subset + [num] for subset in result]

    return result


# Bit manipulation approach
def subsets_bitmask(nums: List[int]) -> List[List[int]]:
    """Generate subsets using bit manipulation."""
    n = len(nums)
    result = []

    # Each number from 0 to 2^n-1 represents a subset
    for mask in range(1 << n):  # 2^n possibilities
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Check if i-th bit is set
                subset.append(nums[i])
        result.append(subset)

    return result


# Subsets II: With duplicates
def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """Generate subsets when nums may contain duplicates."""
    nums.sort()  # Sort to group duplicates
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        result.append(current.copy())

        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result`,
	testCode: `import pytest
from solution import subsets


class TestSubsets:
    def test_three_elements(self):
        """Test with three elements"""
        result = subsets([1, 2, 3])
        assert len(result) == 8  # 2^3 = 8
        assert [] in result
        assert [1, 2, 3] in result

    def test_single_element(self):
        """Test with single element"""
        result = subsets([0])
        assert sorted(result, key=len) == [[], [0]]

    def test_two_elements(self):
        """Test with two elements"""
        result = subsets([1, 2])
        assert len(result) == 4
        assert [] in result
        assert [1] in result
        assert [2] in result
        assert [1, 2] in result

    def test_empty_subset_always_included(self):
        """Test that empty subset is always included"""
        for n in range(1, 5):
            result = subsets(list(range(n)))
            assert [] in result

    def test_full_set_always_included(self):
        """Test that full set is always included"""
        nums = [1, 2, 3, 4]
        result = subsets(nums)
        assert sorted(nums) in [sorted(s) for s in result]

    def test_correct_count(self):
        """Test that we get 2^n subsets"""
        for n in range(1, 6):
            nums = list(range(n))
            result = subsets(nums)
            assert len(result) == 2 ** n

    def test_no_duplicates(self):
        """Test that there are no duplicate subsets"""
        result = subsets([1, 2, 3])
        # Convert to tuples for comparison
        tuples = [tuple(sorted(s)) for s in result]
        assert len(tuples) == len(set(tuples))

    def test_negative_numbers(self):
        """Test with negative numbers"""
        result = subsets([-1, 0, 1])
        assert len(result) == 8
        assert [-1, 0, 1] in result

    def test_larger_input(self):
        """Test with larger input"""
        result = subsets([1, 2, 3, 4, 5])
        assert len(result) == 32  # 2^5

    def test_subset_elements(self):
        """Test that subsets contain correct elements"""
        result = subsets([5, 10])
        assert [5] in result
        assert [10] in result
        assert [5, 10] in result or [10, 5] in result`,
	hint1: `Use backtracking: at each position, you can either include or exclude the current element. Start with an empty subset and build up.`,
	hint2: `The backtracking template: add current subset to result, then for each remaining element, add it, recurse, and remove it (backtrack). Use start index to avoid duplicates.`,
	whyItMatters: `Subsets is the foundation of backtracking. Understanding this pattern unlocks solutions to permutations, combinations, and countless other problems.

**Why This Matters:**

**1. Backtracking Template**

\`\`\`python
def backtrack(start, current):
    result.append(current.copy())    # Process current state

    for i in range(start, len(nums)):
        current.append(nums[i])      # Make choice
        backtrack(i + 1, current)    # Explore
        current.pop()                 # Undo choice

# This template works for:
# - Subsets
# - Combinations
# - Permutations (with modification)
\`\`\`

**2. Three Approaches**

\`\`\`python
# 1. Backtracking (recursive)
# Time: O(n * 2^n), Space: O(n)

# 2. Iterative
# Start with [[]], for each num, add it to all existing subsets
result = [[]]
for num in nums:
    result += [s + [num] for s in result]

# 3. Bit manipulation
# Each subset maps to a binary number 0 to 2^n-1
for mask in range(1 << n):
    subset = [nums[i] for i in range(n) if mask & (1 << i)]
\`\`\`

**3. Handling Duplicates (Subsets II)**

\`\`\`python
# For [1, 2, 2], we don't want [1, 2] twice
def subsets_with_dup(nums):
    nums.sort()  # Key: sort first!
    result = []

    def backtrack(start, current):
        result.append(current.copy())
        for i in range(start, len(nums)):
            # Skip same element at same level
            if i > start and nums[i] == nums[i-1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
\`\`\`

**4. Applications**

\`\`\`python
# Find all subsets with sum = target
def subsets_with_sum(nums, target):
    result = []
    def backtrack(start, current, total):
        if total == target:
            result.append(current.copy())
            return
        if total > target:
            return
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current, total + nums[i])
            current.pop()
    backtrack(0, [], 0)
    return result
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Подмножества',
			description: `Сгенерируйте все возможные подмножества (булеан) массива.

**Задача:**

Дан массив целых чисел \`nums\` с **уникальными** элементами. Верните все возможные подмножества (булеан).

Результат не должен содержать дубликатов подмножеств.

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3]
Выход: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]

Вход: nums = [0]
Выход: [[], [0]]
\`\`\`

**Ключевая идея:**

Для каждого элемента два выбора:
1. **Включить** его в текущее подмножество
2. **Исключить** его

Это даёт 2^n подмножеств.

**Ограничения:**
- 1 <= nums.length <= 10
- Все элементы **уникальны**

**Временная сложность:** O(n × 2^n)
**Пространственная сложность:** O(n)`,
			hint1: `Используйте бэктрекинг: на каждой позиции можно включить или исключить элемент. Начните с пустого подмножества.`,
			hint2: `Шаблон: добавьте текущее подмножество в результат, затем для каждого оставшегося элемента добавьте его, рекурсия, удалите (откат).`,
			whyItMatters: `Subsets - основа бэктрекинга. Понимание этого паттерна открывает решения перестановок, комбинаций и многих других задач.

**Почему это важно:**

**1. Шаблон бэктрекинга**

Этот шаблон работает для подмножеств, комбинаций и перестановок.

**2. Три подхода**

Рекурсивный, итеративный и с битовыми масками.

**3. Обработка дубликатов (Subsets II)**

Сортировка + пропуск одинаковых элементов на одном уровне.`,
			solutionCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """Генерирует все подмножества с помощью бэктрекинга."""
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        result.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result`
		},
		uz: {
			title: 'Kichik to\'plamlar',
			description: `Massivning barcha mumkin bo'lgan kichik to'plamlarini (quvvat to'plami) yarating.

**Masala:**

**Noyob** elementlarga ega butun sonlar massivi \`nums\` berilgan. Barcha mumkin bo'lgan kichik to'plamlarni qaytaring.

Natijada takroriy kichik to'plamlar bo'lmasligi kerak.

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3]
Chiqish: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]

Kirish: nums = [0]
Chiqish: [[], [0]]
\`\`\`

**Asosiy tushuncha:**

Har bir element uchun ikkita tanlov:
1. Joriy kichik to'plamga **qo'shish**
2. **Qo'shmaslik**

Bu 2^n ta kichik to'plam beradi.

**Cheklovlar:**
- 1 <= nums.length <= 10
- Barcha elementlar **noyob**

**Vaqt murakkabligi:** O(n × 2^n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Backtracking ishlating: har bir pozitsiyada elementni qo'shish yoki qo'shmaslik mumkin. Bo'sh kichik to'plamdan boshlang.`,
			hint2: `Shablon: joriy kichik to'plamni natijaga qo'shing, keyin har bir qolgan element uchun qo'shing, rekursiya, olib tashlang (orqaga qaytish).`,
			whyItMatters: `Subsets backtracking ning asosidir. Bu patternni tushunish permutatsiyalar, kombinatsiyalar va ko'plab boshqa masalalarni yechish imkonini beradi.

**Bu nima uchun muhim:**

**1. Backtracking shabloni**

Bu shablon kichik to'plamlar, kombinatsiyalar va permutatsiyalar uchun ishlaydi.

**2. Uchta yondashuv**

Rekursiv, iterativ va bit maskalari bilan.

**3. Dublikatlarni qayta ishlash (Subsets II)**

Saralash + bir xil darajadagi bir xil elementlarni o'tkazib yuborish.`,
			solutionCode: `from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """Backtracking yordamida barcha kichik to'plamlarni yaratadi."""
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        result.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result`
		}
	}
};

export default task;
