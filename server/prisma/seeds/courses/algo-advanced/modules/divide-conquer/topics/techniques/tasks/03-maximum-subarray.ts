import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-maximum-subarray',
	title: 'Maximum Subarray',
	difficulty: 'medium',
	tags: ['python', 'divide-conquer', 'array', 'dynamic-programming'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the contiguous subarray with the largest sum using divide and conquer.

**Problem:**

Given an integer array \`nums\`, find the subarray with the largest sum and return its sum.

Solve this using the **divide and conquer** approach.

**Examples:**

\`\`\`
Input: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Output: 6

Explanation: Subarray [4, -1, 2, 1] has the largest sum = 6.

Input: nums = [1]
Output: 1

Input: nums = [5, 4, -1, 7, 8]
Output: 23

Explanation: Entire array has the largest sum = 23.
\`\`\`

**Divide and Conquer Approach:**

\`\`\`
[-2, 1, -3, 4, -1, 2, 1, -5, 4]

Maximum subarray is either:
1. Entirely in left half: max_left
2. Entirely in right half: max_right
3. Crossing the middle: max_cross

Answer = max(max_left, max_right, max_cross)
\`\`\`

**Key Insight:**

For crossing case, find max sum starting from middle going left AND going right, then combine.

**Constraints:**
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4

**Time Complexity:** O(n log n) for D&C, O(n) for Kadane's
**Space Complexity:** O(log n) for recursion`,
	initialCode: `from typing import List

def max_subarray_dc(nums: List[int]) -> int:
    # TODO: Find maximum subarray sum using divide and conquer

    return 0`,
	solutionCode: `from typing import List

def max_subarray_dc(nums: List[int]) -> int:
    """
    Find maximum subarray sum using divide and conquer.
    """
    def find_cross_max(lo: int, mid: int, hi: int) -> int:
        # Find max sum going left from mid
        left_sum = float('-inf')
        curr_sum = 0
        for i in range(mid, lo - 1, -1):
            curr_sum += nums[i]
            left_sum = max(left_sum, curr_sum)

        # Find max sum going right from mid + 1
        right_sum = float('-inf')
        curr_sum = 0
        for i in range(mid + 1, hi + 1):
            curr_sum += nums[i]
            right_sum = max(right_sum, curr_sum)

        return left_sum + right_sum

    def divide_conquer(lo: int, hi: int) -> int:
        # Base case: single element
        if lo == hi:
            return nums[lo]

        mid = (lo + hi) // 2

        # Maximum subarray in left half
        left_max = divide_conquer(lo, mid)

        # Maximum subarray in right half
        right_max = divide_conquer(mid + 1, hi)

        # Maximum subarray crossing the middle
        cross_max = find_cross_max(lo, mid, hi)

        return max(left_max, right_max, cross_max)

    return divide_conquer(0, len(nums) - 1)


# Kadane's algorithm (O(n) - optimal)
def max_subarray_kadane(nums: List[int]) -> int:
    """Find maximum subarray sum using Kadane's algorithm."""
    max_sum = nums[0]
    curr_sum = nums[0]

    for i in range(1, len(nums)):
        # Either extend current subarray or start new
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)

    return max_sum


# Return subarray indices
def max_subarray_with_indices(nums: List[int]) -> tuple:
    """Return max sum and subarray indices."""
    max_sum = nums[0]
    curr_sum = nums[0]
    start = end = 0
    temp_start = 0

    for i in range(1, len(nums)):
        if nums[i] > curr_sum + nums[i]:
            curr_sum = nums[i]
            temp_start = i
        else:
            curr_sum = curr_sum + nums[i]

        if curr_sum > max_sum:
            max_sum = curr_sum
            start = temp_start
            end = i

    return max_sum, start, end, nums[start:end + 1]


# Circular max subarray
def max_subarray_circular(nums: List[int]) -> int:
    """Find max subarray in circular array."""
    # Case 1: Max subarray doesn't wrap around
    max_kadane = max_subarray_kadane(nums)

    # Case 2: Max subarray wraps around
    # = total_sum - min_subarray
    total = sum(nums)
    min_sum = nums[0]
    curr_min = nums[0]

    for i in range(1, len(nums)):
        curr_min = min(nums[i], curr_min + nums[i])
        min_sum = min(min_sum, curr_min)

    # If all negative, max_kadane is the answer
    if total == min_sum:
        return max_kadane

    return max(max_kadane, total - min_sum)`,
	testCode: `import pytest
from solution import max_subarray_dc


class TestMaximumSubarray:
    def test_basic_case(self):
        """Test basic case"""
        assert max_subarray_dc([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6

    def test_single_element(self):
        """Test single element"""
        assert max_subarray_dc([1]) == 1

    def test_all_positive(self):
        """Test all positive elements"""
        assert max_subarray_dc([5, 4, -1, 7, 8]) == 23

    def test_all_negative(self):
        """Test all negative elements"""
        assert max_subarray_dc([-1, -2, -3]) == -1

    def test_single_negative(self):
        """Test single negative element"""
        assert max_subarray_dc([-5]) == -5

    def test_two_elements(self):
        """Test two elements"""
        assert max_subarray_dc([1, 2]) == 3
        assert max_subarray_dc([-1, 2]) == 2
        assert max_subarray_dc([1, -2]) == 1

    def test_mixed_array(self):
        """Test mixed positive and negative"""
        assert max_subarray_dc([1, 2, -1, 3, -2]) == 5

    def test_alternating(self):
        """Test alternating positive/negative"""
        assert max_subarray_dc([-2, 1, -1, 2, -1, 1]) == 2

    def test_zero_in_array(self):
        """Test with zeros"""
        assert max_subarray_dc([0, -1, 0, 2]) == 2

    def test_cross_middle(self):
        """Test where max crosses the middle"""
        assert max_subarray_dc([1, -1, 3]) == 3
        assert max_subarray_dc([-1, 2, 1, -1]) == 3`,
	hint1: `The maximum subarray is either entirely in the left half, entirely in the right half, or it crosses the middle. Handle each case separately.`,
	hint2: `For the crossing case, start from the middle and expand left to find max left sum, then expand right to find max right sum. The crossing max is their sum.`,
	whyItMatters: `Maximum Subarray demonstrates how divide and conquer can solve problems that also have simpler solutions. Understanding both approaches deepens algorithm knowledge.

**Why This Matters:**

**1. D&C Problem Structure**

\`\`\`python
# Three possibilities for max subarray:
# 1. Entirely in left half
# 2. Entirely in right half
# 3. Crosses the middle

# The "crossing" case is the key insight
# Must start from middle and expand both directions

def find_cross_max(lo, mid, hi):
    # Expand left from mid
    left_sum = -inf
    curr = 0
    for i in range(mid, lo - 1, -1):
        curr += nums[i]
        left_sum = max(left_sum, curr)

    # Expand right from mid + 1
    right_sum = -inf
    curr = 0
    for i in range(mid + 1, hi + 1):
        curr += nums[i]
        right_sum = max(right_sum, curr)

    return left_sum + right_sum
\`\`\`

**2. Comparison with Kadane's**

\`\`\`python
# Kadane's: O(n) time, O(1) space
max_sum = curr_sum = nums[0]
for num in nums[1:]:
    curr_sum = max(num, curr_sum + num)
    max_sum = max(max_sum, curr_sum)

# D&C: O(n log n) time, O(log n) space
# More complex but demonstrates D&C paradigm
\`\`\`

**3. When D&C is Useful**

\`\`\`python
# Even though Kadane's is faster, D&C approach:
# - Helps understand divide and conquer
# - Extends to variations (2D max subarray)
# - Can be parallelized easily
# - Teaching tool for algorithm design
\`\`\`

**4. Variations**

\`\`\`python
# Circular max subarray: max(kadane, total - min_subarray)
# 2D max subarray: Use 1D max subarray as subroutine
# Max product subarray: Track both max and min
# At most k elements: Sliding window
\`\`\`

**5. Interview Tips**

\`\`\`python
# Often asked to solve both ways
# D&C shows strong algorithm fundamentals
# Kadane's shows practical optimization
# Know trade-offs between approaches
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Максимальный подмассив',
			description: `Найдите подмассив с наибольшей суммой методом "разделяй и властвуй".

**Задача:**

Дан массив целых чисел \`nums\`. Найдите подмассив с наибольшей суммой и верните эту сумму.

Решите с использованием метода **"разделяй и властвуй"**.

**Примеры:**

\`\`\`
Вход: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Выход: 6

Объяснение: Подмассив [4, -1, 2, 1] имеет наибольшую сумму = 6.

Вход: nums = [5, 4, -1, 7, 8]
Выход: 23
\`\`\`

**Подход "разделяй и властвуй":**

Максимальный подмассив либо:
1. Полностью в левой половине
2. Полностью в правой половине
3. Пересекает середину

**Ограничения:**
- 1 <= nums.length <= 10^5

**Временная сложность:** O(n log n) для D&C, O(n) для Kadane
**Пространственная сложность:** O(log n)`,
			hint1: `Максимальный подмассив либо полностью слева, полностью справа, либо пересекает середину. Обработайте каждый случай отдельно.`,
			hint2: `Для пересечения: от середины расширяйтесь влево для макс суммы слева, вправо для макс суммы справа. Сумма пересечения = их сумма.`,
			whyItMatters: `Maximum Subarray показывает, как D&C решает задачи, которые имеют более простые решения. Понимание обоих подходов углубляет знание алгоритмов.

**Почему это важно:**

**1. Структура задачи D&C**

Три возможности: слева, справа или пересекает середину.

**2. Сравнение с Kadane**

Kadane: O(n). D&C: O(n log n), но демонстрирует парадигму.

**3. Вариации**

Круговой массив, 2D, максимальное произведение.`,
			solutionCode: `from typing import List

def max_subarray_dc(nums: List[int]) -> int:
    """Находит максимальную сумму подмассива методом D&C."""
    def find_cross_max(lo: int, mid: int, hi: int) -> int:
        left_sum = float('-inf')
        curr_sum = 0
        for i in range(mid, lo - 1, -1):
            curr_sum += nums[i]
            left_sum = max(left_sum, curr_sum)

        right_sum = float('-inf')
        curr_sum = 0
        for i in range(mid + 1, hi + 1):
            curr_sum += nums[i]
            right_sum = max(right_sum, curr_sum)

        return left_sum + right_sum

    def divide_conquer(lo: int, hi: int) -> int:
        if lo == hi:
            return nums[lo]

        mid = (lo + hi) // 2
        left_max = divide_conquer(lo, mid)
        right_max = divide_conquer(mid + 1, hi)
        cross_max = find_cross_max(lo, mid, hi)

        return max(left_max, right_max, cross_max)

    return divide_conquer(0, len(nums) - 1)`
		},
		uz: {
			title: 'Maksimal qism massiv',
			description: `"Bo'l va hukmronlik qil" usuli bilan eng katta yig'indili qism massivni toping.

**Masala:**

\`nums\` butun sonlar massivi berilgan. Eng katta yig'indili qism massivni toping va yig'indini qaytaring.

**"Bo'l va hukmronlik qil"** yondashuvi bilan yeching.

**Misollar:**

\`\`\`
Kirish: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Chiqish: 6

Izoh: [4, -1, 2, 1] qism massiv eng katta yig'indiga ega = 6.

Kirish: nums = [5, 4, -1, 7, 8]
Chiqish: 23
\`\`\`

**"Bo'l va hukmronlik qil" yondashuvi:**

Maksimal qism massiv yoki:
1. To'liq chap yarmida
2. To'liq o'ng yarmida
3. O'rtadan o'tadi

**Cheklovlar:**
- 1 <= nums.length <= 10^5

**Vaqt murakkabligi:** O(n log n) D&C uchun, O(n) Kadane uchun
**Xotira murakkabligi:** O(log n)`,
			hint1: `Maksimal qism massiv to'liq chapda, to'liq o'ngda yoki o'rtadan o'tadi. Har bir holatni alohida ishlang.`,
			hint2: `O'tish uchun: o'rtadan chapga maksimal yig'indi, o'ngga maksimal yig'indi toping. O'tish maksimumi = ularning yig'indisi.`,
			whyItMatters: `Maximum Subarray D&C soddaroq yechimlari ham bor masalalarni qanday hal qilishini ko'rsatadi.

**Bu nima uchun muhim:**

**1. D&C masala tuzilishi**

Uchta imkoniyat: chapda, o'ngda yoki o'rtadan o'tadi.

**2. Kadane bilan taqqoslash**

Kadane: O(n). D&C: O(n log n), lekin paradigmani ko'rsatadi.

**3. Variatsiyalar**

Aylanma massiv, 2D, maksimal ko'paytma.`,
			solutionCode: `from typing import List

def max_subarray_dc(nums: List[int]) -> int:
    """D&C usuli bilan maksimal qism massiv yig'indisini topadi."""
    def find_cross_max(lo: int, mid: int, hi: int) -> int:
        left_sum = float('-inf')
        curr_sum = 0
        for i in range(mid, lo - 1, -1):
            curr_sum += nums[i]
            left_sum = max(left_sum, curr_sum)

        right_sum = float('-inf')
        curr_sum = 0
        for i in range(mid + 1, hi + 1):
            curr_sum += nums[i]
            right_sum = max(right_sum, curr_sum)

        return left_sum + right_sum

    def divide_conquer(lo: int, hi: int) -> int:
        if lo == hi:
            return nums[lo]

        mid = (lo + hi) // 2
        left_max = divide_conquer(lo, mid)
        right_max = divide_conquer(mid + 1, hi)
        cross_max = find_cross_max(lo, mid, hi)

        return max(left_max, right_max, cross_max)

    return divide_conquer(0, len(nums) - 1)`
		}
	}
};

export default task;
