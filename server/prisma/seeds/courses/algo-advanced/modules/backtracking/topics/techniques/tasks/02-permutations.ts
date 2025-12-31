import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-permutations',
	title: 'Permutations',
	difficulty: 'medium',
	tags: ['python', 'backtracking', 'recursion', 'permutation'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Generate all possible permutations of a given array.

**Problem:**

Given an array \`nums\` of distinct integers, return all possible permutations in any order.

**Examples:**

\`\`\`
Input: nums = [1, 2, 3]
Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

Input: nums = [0, 1]
Output: [[0, 1], [1, 0]]

Input: nums = [1]
Output: [[1]]
\`\`\`

**Visualization (Decision Tree):**

\`\`\`
nums = [1, 2, 3]

                    []
         /          |          \\
       [1]         [2]         [3]
      /   \\       /   \\       /   \\
   [1,2] [1,3] [2,1] [2,3] [3,1] [3,2]
     |     |     |     |     |     |
 [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

Each level: choose one of remaining elements
\`\`\`

**Key Difference from Subsets:**

- **Subsets**: Elements can be skipped, order doesn't matter
- **Permutations**: All elements must be used, order matters

**Constraints:**
- 1 <= nums.length <= 6
- -10 <= nums[i] <= 10
- All integers are **unique**

**Time Complexity:** O(n! × n) - n! permutations, each takes O(n) to copy
**Space Complexity:** O(n) for recursion stack`,
	initialCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    # TODO: Generate all permutations of nums

    return []`,
	solutionCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations using backtracking.
    """
    result = []

    def backtrack(current: List[int], remaining: set) -> None:
        if not remaining:
            result.append(current.copy())
            return

        for num in list(remaining):
            current.append(num)
            remaining.remove(num)
            backtrack(current, remaining)
            remaining.add(num)
            current.pop()

    backtrack([], set(nums))
    return result


# Alternative: Swap approach (more efficient)
def permute_swap(nums: List[int]) -> List[List[int]]:
    """Generate permutations by swapping elements."""
    result = []

    def backtrack(start: int) -> None:
        if start == len(nums):
            result.append(nums.copy())
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result


# Using visited array
def permute_visited(nums: List[int]) -> List[List[int]]:
    """Generate permutations using visited array."""
    result = []
    n = len(nums)
    visited = [False] * n

    def backtrack(current: List[int]) -> None:
        if len(current) == n:
            result.append(current.copy())
            return

        for i in range(n):
            if not visited[i]:
                visited[i] = True
                current.append(nums[i])
                backtrack(current)
                current.pop()
                visited[i] = False

    backtrack([])
    return result


# Permutations II: With duplicates
def permute_unique(nums: List[int]) -> List[List[int]]:
    """Generate unique permutations when nums has duplicates."""
    result = []
    nums.sort()
    visited = [False] * len(nums)

    def backtrack(current: List[int]) -> None:
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
            # Skip if used OR skip duplicate (same value, previous not used)
            if visited[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and not visited[i-1]:
                continue

            visited[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            visited[i] = False

    backtrack([])
    return result`,
	testCode: `import pytest
from solution import permute


class TestPermutations:
    def test_three_elements(self):
        """Test with three elements"""
        result = permute([1, 2, 3])
        assert len(result) == 6  # 3! = 6
        expected = [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
        for perm in expected:
            assert perm in result

    def test_two_elements(self):
        """Test with two elements"""
        result = permute([0, 1])
        assert len(result) == 2
        assert [0, 1] in result
        assert [1, 0] in result

    def test_single_element(self):
        """Test with single element"""
        result = permute([1])
        assert result == [[1]]

    def test_correct_count(self):
        """Test that we get n! permutations"""
        import math
        for n in range(1, 6):
            nums = list(range(n))
            result = permute(nums)
            assert len(result) == math.factorial(n)

    def test_no_duplicates(self):
        """Test that there are no duplicate permutations"""
        result = permute([1, 2, 3])
        tuples = [tuple(p) for p in result]
        assert len(tuples) == len(set(tuples))

    def test_all_elements_used(self):
        """Test that all elements are used in each permutation"""
        nums = [1, 2, 3, 4]
        result = permute(nums)
        for perm in result:
            assert sorted(perm) == sorted(nums)

    def test_negative_numbers(self):
        """Test with negative numbers"""
        result = permute([-1, 0, 1])
        assert len(result) == 6

    def test_order_matters(self):
        """Test that order matters"""
        result = permute([1, 2])
        assert [1, 2] in result
        assert [2, 1] in result
        assert len(result) == 2

    def test_larger_input(self):
        """Test with larger input"""
        result = permute([1, 2, 3, 4, 5])
        assert len(result) == 120  # 5!

    def test_four_elements(self):
        """Test with four elements"""
        result = permute([1, 2, 3, 4])
        assert len(result) == 24  # 4! = 24
        # Verify all permutations are unique
        unique = set(tuple(p) for p in result)
        assert len(unique) == 24`,
	hint1: `Unlike subsets, you must use ALL elements. Track which elements are already used with a set or visited array.`,
	hint2: `Backtrack when current permutation has all elements. For each unused element, add it to current, recurse, then remove it (backtrack).`,
	whyItMatters: `Permutations is a classic backtracking problem. Understanding it helps solve scheduling, arrangement, and optimization problems.

**Why This Matters:**

**1. Permutation vs Subset vs Combination**

\`\`\`python
# Given [1, 2, 3]:

# Subsets (2^n): Order doesn't matter, don't use all
# [], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]

# Combinations (n choose k): Order doesn't matter, use exactly k
# k=2: [1,2], [1,3], [2,3]

# Permutations (n!): Order matters, use all
# [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
\`\`\`

**2. Swap Technique**

\`\`\`python
def permute_swap(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums.copy())
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]  # Choose
            backtrack(start + 1)                          # Explore
            nums[start], nums[i] = nums[i], nums[start]  # Unchoose

    backtrack(0)
    return result

# More efficient: no extra space for tracking used elements
\`\`\`

**3. Handling Duplicates (Permutations II)**

\`\`\`python
def permute_unique(nums):
    nums.sort()  # Sort to group duplicates
    result = []
    visited = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current.copy())
            return

        for i in range(len(nums)):
            if visited[i]:
                continue
            # Key: skip duplicate if previous same element not used
            if i > 0 and nums[i] == nums[i-1] and not visited[i-1]:
                continue

            visited[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            visited[i] = False

    backtrack([])
    return result
\`\`\`

**4. Applications**

\`\`\`python
# Anagrams generation
# Task scheduling with constraints
# Arrangement problems
# Brute-force search

# Next Permutation (LeetCode 31)
def next_permutation(nums):
    # Find rightmost ascending pair
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        # Find rightmost larger element
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    # Reverse suffix
    nums[i + 1:] = reversed(nums[i + 1:])
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Перестановки',
			description: `Сгенерируйте все возможные перестановки массива.

**Задача:**

Дан массив различных целых чисел \`nums\`. Верните все возможные перестановки.

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3]
Выход: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

Вход: nums = [0, 1]
Выход: [[0, 1], [1, 0]]
\`\`\`

**Ключевое отличие от подмножеств:**

- **Подмножества**: Элементы можно пропускать, порядок не важен
- **Перестановки**: Все элементы должны использоваться, порядок важен

**Ограничения:**
- 1 <= nums.length <= 6
- Все числа **уникальны**

**Временная сложность:** O(n! × n)
**Пространственная сложность:** O(n)`,
			hint1: `В отличие от подмножеств, нужно использовать ВСЕ элементы. Отслеживайте использованные элементы множеством или массивом visited.`,
			hint2: `Бэктрек когда текущая перестановка содержит все элементы. Для каждого неиспользованного элемента: добавьте, рекурсия, удалите.`,
			whyItMatters: `Перестановки - классическая задача бэктрекинга. Понимание помогает решать задачи планирования и оптимизации.

**Почему это важно:**

**1. Перестановка vs Подмножество vs Комбинация**

Понимание различий критично для выбора правильного алгоритма.

**2. Техника обмена (Swap)**

Более эффективна: не требует дополнительной памяти для отслеживания.

**3. Обработка дубликатов (Permutations II)**

Сортировка + пропуск если предыдущий одинаковый элемент не использован.`,
			solutionCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """Генерирует все перестановки с помощью бэктрекинга."""
    result = []

    def backtrack(current: List[int], remaining: set) -> None:
        if not remaining:
            result.append(current.copy())
            return

        for num in list(remaining):
            current.append(num)
            remaining.remove(num)
            backtrack(current, remaining)
            remaining.add(num)
            current.pop()

    backtrack([], set(nums))
    return result`
		},
		uz: {
			title: 'Permutatsiyalar',
			description: `Massivning barcha mumkin bo'lgan permutatsiyalarini yarating.

**Masala:**

Turli butun sonlardan iborat \`nums\` massivi berilgan. Barcha mumkin bo'lgan permutatsiyalarni qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3]
Chiqish: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

Kirish: nums = [0, 1]
Chiqish: [[0, 1], [1, 0]]
\`\`\`

**Kichik to'plamlardan asosiy farqi:**

- **Kichik to'plamlar**: Elementlarni o'tkazib yuborish mumkin, tartib muhim emas
- **Permutatsiyalar**: BARCHA elementlar ishlatilishi kerak, tartib muhim

**Cheklovlar:**
- 1 <= nums.length <= 6
- Barcha sonlar **noyob**

**Vaqt murakkabligi:** O(n! × n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Kichik to'plamlardan farqli, BARCHA elementlarni ishlatish kerak. Ishlatilgan elementlarni to'plam yoki visited massiv bilan kuzating.`,
			hint2: `Joriy permutatsiya barcha elementlarni o'z ichiga olganda backtrack qiling. Har bir ishlatilmagan element uchun: qo'shing, rekursiya, olib tashlang.`,
			whyItMatters: `Permutatsiyalar klassik backtracking masalasi. Tushunish rejalashtirish va optimallashtirish masalalarini yechishga yordam beradi.

**Bu nima uchun muhim:**

**1. Permutatsiya vs Kichik to'plam vs Kombinatsiya**

Farqlarni tushunish to'g'ri algoritmni tanlash uchun muhim.

**2. Almashtirish texnikasi (Swap)**

Ko'proq samarali: kuzatish uchun qo'shimcha xotira talab qilmaydi.

**3. Dublikatlarni qayta ishlash (Permutations II)**

Saralash + oldingi bir xil element ishlatilmagan bo'lsa o'tkazib yuborish.`,
			solutionCode: `from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """Backtracking yordamida barcha permutatsiyalarni yaratadi."""
    result = []

    def backtrack(current: List[int], remaining: set) -> None:
        if not remaining:
            result.append(current.copy())
            return

        for num in list(remaining):
            current.append(num)
            remaining.remove(num)
            backtrack(current, remaining)
            remaining.add(num)
            current.pop()

    backtrack([], set(nums))
    return result`
		}
	}
};

export default task;
