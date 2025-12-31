import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-maximum-product-subarray',
	title: 'Maximum Product Subarray',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'array', 'kadane'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the contiguous subarray with the largest product.

**Problem:**

Given an integer array \`nums\`, find a contiguous non-empty subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

A **subarray** is a contiguous subsequence of the array.

**Examples:**

\`\`\`
Input: nums = [2, 3, -2, 4]
Output: 6
Explanation: [2, 3] has the largest product 6

Input: nums = [-2, 0, -1]
Output: 0
Explanation: Cannot include both -2 and -1 (0 in between)
Best is [0] or [-2] or [-1], max is 0

Input: nums = [-2, 3, -4]
Output: 24
Explanation: [-2, 3, -4] = 24 (two negatives make positive)

Input: nums = [2, -5, -2, -4, 3]
Output: 24
Explanation: [-5, -2, -4, 3] = 120? No, [-2, -4, 3] = 24
Actually: [2, -5, -2, -4] = 80
\`\`\`

**Key Insight:**

Unlike Maximum Subarray (sum), we need to track both:
- **Maximum product** ending at current position
- **Minimum product** ending at current position

Why? A large negative × negative = large positive!

\`\`\`
max_dp[i] = max(nums[i], max_dp[i-1] * nums[i], min_dp[i-1] * nums[i])
min_dp[i] = min(nums[i], max_dp[i-1] * nums[i], min_dp[i-1] * nums[i])
\`\`\`

**Constraints:**
- 1 <= nums.length <= 2 × 10^4
- -10 <= nums[i] <= 10
- Product fits in 32-bit integer

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def max_product(nums: List[int]) -> int:
    # TODO: Find maximum product of contiguous subarray

    return 0`,
	solutionCode: `from typing import List


def max_product(nums: List[int]) -> int:
    """
    Find maximum product of contiguous subarray.

    Args:
        nums: List of integers

    Returns:
        Maximum product of any contiguous subarray
    """
    if not nums:
        return 0

    max_prod = nums[0]
    min_prod = nums[0]
    result = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]

        # Need to use temp because max_prod is updated before min_prod calculation
        temp_max = max(num, max_prod * num, min_prod * num)
        temp_min = min(num, max_prod * num, min_prod * num)

        max_prod = temp_max
        min_prod = temp_min

        result = max(result, max_prod)

    return result


# Alternative: Handle zeros explicitly
def max_product_v2(nums: List[int]) -> int:
    """Version that explicitly handles zeros."""
    if not nums:
        return 0

    result = nums[0]
    max_prod = min_prod = 1

    for num in nums:
        if num == 0:
            max_prod = min_prod = 1
            result = max(result, 0)
        else:
            temp = max_prod
            max_prod = max(num, max_prod * num, min_prod * num)
            min_prod = min(num, temp * num, min_prod * num)
            result = max(result, max_prod)

    return result


# Alternative: Two-pass approach
def max_product_two_pass(nums: List[int]) -> int:
    """
    Two-pass: left-to-right and right-to-left.
    Handles cases where zeros split the array.
    """
    if not nums:
        return 0

    result = nums[0]

    # Left to right
    prod = 1
    for num in nums:
        prod *= num
        result = max(result, prod)
        if prod == 0:
            prod = 1

    # Right to left
    prod = 1
    for num in reversed(nums):
        prod *= num
        result = max(result, prod)
        if prod == 0:
            prod = 1

    return result`,
	testCode: `import pytest
from solution import max_product


class TestMaximumProductSubarray:
    def test_basic_example(self):
        """Test [2,3,-2,4] -> 6"""
        assert max_product([2, 3, -2, 4]) == 6

    def test_with_zero(self):
        """Test [-2,0,-1] -> 0"""
        assert max_product([-2, 0, -1]) == 0

    def test_two_negatives(self):
        """Test [-2,3,-4] -> 24"""
        assert max_product([-2, 3, -4]) == 24

    def test_single_positive(self):
        """Test single positive element"""
        assert max_product([5]) == 5

    def test_single_negative(self):
        """Test single negative element"""
        assert max_product([-3]) == -3

    def test_single_zero(self):
        """Test single zero"""
        assert max_product([0]) == 0

    def test_all_positive(self):
        """Test all positive numbers"""
        assert max_product([1, 2, 3, 4]) == 24

    def test_all_negative_even(self):
        """Test all negative, even count"""
        assert max_product([-1, -2, -3, -4]) == 24

    def test_all_negative_odd(self):
        """Test all negative, odd count"""
        assert max_product([-1, -2, -3]) == 6

    def test_complex_case(self):
        """Test [2,-5,-2,-4,3] -> 24"""
        assert max_product([2, -5, -2, -4, 3]) == 24`,
	hint1: `Unlike sum, negative numbers can become positive when multiplied by another negative. Track both the maximum AND minimum product ending at each position.`,
	hint2: `For each number: new_max = max(num, max_prod * num, min_prod * num). Use temporary variables to avoid overwriting before calculating min.`,
	whyItMatters: `Maximum Product Subarray shows how to adapt Kadane's algorithm when negative numbers change the game. It's a FAANG favorite.

**Why This Matters:**

**1. Adapting Kadane's Algorithm**

\`\`\`python
# Maximum Sum Subarray (Kadane's)
def max_sum(nums):
    max_so_far = curr_max = nums[0]
    for num in nums[1:]:
        curr_max = max(num, curr_max + num)
        max_so_far = max(max_so_far, curr_max)
    return max_so_far

# Maximum Product: Need both max AND min
# Because negative * negative = positive!
def max_product(nums):
    max_prod = min_prod = result = nums[0]
    for num in nums[1:]:
        temp = max_prod
        max_prod = max(num, max_prod * num, min_prod * num)
        min_prod = min(num, temp * num, min_prod * num)
        result = max(result, max_prod)
    return result
\`\`\`

**2. Why Track Minimum?**

\`\`\`python
# Example: [-2, 3, -4]
#
# At -2: max=-2, min=-2
# At 3:  max=3 (3 > -6), min=-6 (-2*3)
# At -4: max=24 (-6*-4), min=-12 (3*-4)
#
# The minimum (-6) led to the maximum (24)!
\`\`\`

**3. Handling Edge Cases**

\`\`\`python
# Zero resets the product
# nums = [2, 3, 0, 4, 5]
# Effectively: [2, 3] and [4, 5] are separate subarrays

# All negatives
# nums = [-2, -3, -4]
# Even count: product of all = 24
# Odd count: drop one from either end
\`\`\`

**4. Two-Pass Approach**

\`\`\`python
# Alternative: scan left-to-right and right-to-left
def max_product_two_pass(nums):
    result = float('-inf')

    # Left to right
    prod = 1
    for num in nums:
        prod *= num
        result = max(result, prod)
        if prod == 0:
            prod = 1

    # Right to left (handles odd negative counts)
    prod = 1
    for num in reversed(nums):
        prod *= num
        result = max(result, prod)
        if prod == 0:
            prod = 1

    return result
\`\`\`

**5. Related Problems**

\`\`\`python
# Maximum Subarray (sum version) - LeetCode 53
# Product of Array Except Self - LeetCode 238
# Maximum Product of Three Numbers - LeetCode 628
# Subarray Product Less Than K - LeetCode 713
\`\`\`

**6. Interview Tips**

- Always ask about negative numbers and zeros
- Draw out examples with negatives
- Mention the "track min too" insight
- Discuss the two-pass alternative`,
	order: 10,
	translations: {
		ru: {
			title: 'Максимальное произведение подмассива',
			description: `Найдите непрерывный подмассив с наибольшим произведением.

**Задача:**

Дан массив целых чисел \`nums\`, найдите непрерывный непустой подмассив с наибольшим произведением и верните это произведение.

**Подмассив** - это непрерывная подпоследовательность массива.

**Примеры:**

\`\`\`
Вход: nums = [2, 3, -2, 4]
Выход: 6
Объяснение: [2, 3] имеет наибольшее произведение 6

Вход: nums = [-2, 0, -1]
Выход: 0
Объяснение: Нельзя включить оба -2 и -1 (0 между ними)

Вход: nums = [-2, 3, -4]
Выход: 24
Объяснение: [-2, 3, -4] = 24 (два отрицательных дают положительное)
\`\`\`

**Ключевая идея:**

В отличие от максимальной суммы, нужно отслеживать оба:
- **Максимальное произведение** до текущей позиции
- **Минимальное произведение** до текущей позиции

Почему? Большое отрицательное × отрицательное = большое положительное!

**Ограничения:**
- 1 <= nums.length <= 2 × 10^4
- -10 <= nums[i] <= 10

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `В отличие от суммы, отрицательные числа могут стать положительными при умножении на другое отрицательное. Отслеживайте И максимальное, И минимальное произведение.`,
			hint2: `Для каждого числа: new_max = max(num, max_prod * num, min_prod * num). Используйте временные переменные, чтобы не перезаписать значения.`,
			whyItMatters: `Maximum Product Subarray показывает, как адаптировать алгоритм Кадане, когда отрицательные числа меняют правила игры.

**Почему это важно:**

**1. Адаптация алгоритма Кадане**

Для суммы: отслеживаем только максимум.
Для произведения: отслеживаем И максимум, И минимум!

**2. Почему отслеживать минимум?**

Минимум (большое отрицательное) может стать максимумом при умножении на отрицательное.

**3. Обработка краевых случаев**

- Ноль сбрасывает произведение
- Все отрицательные: чётное количество → произведение всех

**4. Связанные задачи**

Maximum Subarray, Product of Array Except Self, Maximum Product of Three Numbers.`,
			solutionCode: `from typing import List


def max_product(nums: List[int]) -> int:
    """
    Находит максимальное произведение непрерывного подмассива.

    Args:
        nums: Список целых чисел

    Returns:
        Максимальное произведение любого непрерывного подмассива
    """
    if not nums:
        return 0

    max_prod = nums[0]
    min_prod = nums[0]
    result = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]

        # Нужна временная переменная
        temp_max = max(num, max_prod * num, min_prod * num)
        temp_min = min(num, max_prod * num, min_prod * num)

        max_prod = temp_max
        min_prod = temp_min

        result = max(result, max_prod)

    return result`
		},
		uz: {
			title: 'Maksimal ko\'paytma qism massiv',
			description: `Eng katta ko'paytmaga ega uzluksiz qism massivni toping.

**Masala:**

Butun sonlar massivi \`nums\` berilgan, eng katta ko'paytmaga ega uzluksiz bo'sh bo'lmagan qism massivni toping va ko'paytmani qaytaring.

**Qism massiv** - bu massivning uzluksiz qism ketma-ketligi.

**Misollar:**

\`\`\`
Kirish: nums = [2, 3, -2, 4]
Chiqish: 6
Izoh: [2, 3] eng katta ko'paytma 6 ga ega

Kirish: nums = [-2, 0, -1]
Chiqish: 0
Izoh: -2 va -1 ni birgalikda qo'shib bo'lmaydi (orada 0)

Kirish: nums = [-2, 3, -4]
Chiqish: 24
Izoh: [-2, 3, -4] = 24 (ikkita manfiy musbat beradi)
\`\`\`

**Asosiy tushuncha:**

Maksimal yig'indidan farqli, ikkisini ham kuzatish kerak:
- Joriy pozitsiyada tugaydigan **Maksimal ko'paytma**
- Joriy pozitsiyada tugaydigan **Minimal ko'paytma**

Nima uchun? Katta manfiy × manfiy = katta musbat!

**Cheklovlar:**
- 1 <= nums.length <= 2 × 10^4
- -10 <= nums[i] <= 10

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Yig'indidan farqli, manfiy sonlar boshqa manfiyga ko'paytirilganda musbat bo'lishi mumkin. Har bir pozitsiyada tugaydigan VA maksimal, VA minimal ko'paytmani kuzating.`,
			hint2: `Har bir son uchun: new_max = max(num, max_prod * num, min_prod * num). Qiymatlarni qayta yozmaslik uchun vaqtinchalik o'zgaruvchilardan foydalaning.`,
			whyItMatters: `Maximum Product Subarray manfiy sonlar o'yin qoidalarini o'zgartirganida Kadane algoritmini qanday moslashtirish kerakligini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Kadane algoritmini moslashtirish**

Yig'indi uchun: faqat maksimumni kuzatamiz.
Ko'paytma uchun: VA maksimum, VA minimumni kuzatamiz!

**2. Nima uchun minimumni kuzatish kerak?**

Minimum (katta manfiy) manfiyga ko'paytirilganda maksimum bo'lishi mumkin.

**3. Chekka holatlarni qayta ishlash**

- Nol ko'paytmani qayta boshlaydi
- Hammasi manfiy: juft son → hammasining ko'paytmasi

**4. Bog'liq masalalar**

Maximum Subarray, Product of Array Except Self, Maximum Product of Three Numbers.`,
			solutionCode: `from typing import List


def max_product(nums: List[int]) -> int:
    """
    Uzluksiz qism massivning maksimal ko'paytmasini topadi.

    Args:
        nums: Butun sonlar ro'yxati

    Returns:
        Istalgan uzluksiz qism massivning maksimal ko'paytmasi
    """
    if not nums:
        return 0

    max_prod = nums[0]
    min_prod = nums[0]
    result = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]

        # Vaqtinchalik o'zgaruvchi kerak
        temp_max = max(num, max_prod * num, min_prod * num)
        temp_min = min(num, max_prod * num, min_prod * num)

        max_prod = temp_max
        min_prod = temp_min

        result = max(result, max_prod)

    return result`
		}
	}
};

export default task;
