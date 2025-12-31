import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-house-robber',
	title: 'House Robber',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'array', 'optimization'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the maximum amount of money you can rob without alerting the police.

**Problem:**

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed.

**Constraint:** Adjacent houses have security systems connected - if two adjacent houses are broken into on the same night, the police will be alerted.

Given an integer array \`nums\` representing the amount of money in each house, return the maximum amount you can rob tonight **without alerting the police**.

**Examples:**

\`\`\`
Input: nums = [1, 2, 3, 1]
Output: 4
Explanation: Rob house 1 (money = 1) and house 3 (money = 3)
Total = 1 + 3 = 4

Input: nums = [2, 7, 9, 3, 1]
Output: 12
Explanation: Rob house 1 (money = 2), house 3 (money = 9), house 5 (money = 1)
Total = 2 + 9 + 1 = 12

Input: nums = [2, 1, 1, 2]
Output: 4
Explanation: Rob house 1 and house 4 (2 + 2 = 4)
\`\`\`

**Key Insight:**

At each house, you have two choices:
1. **Rob it:** Add current house + best from 2 houses back
2. **Skip it:** Take best from previous house

\`dp[i] = max(dp[i-1], dp[i-2] + nums[i])\`

**Constraints:**
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 400

**Time Complexity:** O(n)
**Space Complexity:** O(1) with optimization`,
	initialCode: `from typing import List

def rob(nums: List[int]) -> int:
    # TODO: Find maximum money that can be robbed (no adjacent houses)

    return 0`,
	solutionCode: `from typing import List


def rob(nums: List[int]) -> int:
    """
    Find maximum money that can be robbed.

    Args:
        nums: Amount of money in each house

    Returns:
        Maximum amount that can be robbed without alerting police
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    # Space-optimized approach
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr

    return prev1


# Alternative: Full DP array
def rob_dp_array(nums: List[int]) -> int:
    """Full DP array approach for clarity."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[n - 1]


# House Robber II: Circular street
def rob_circular(nums: List[int]) -> int:
    """
    Houses are in a circle - first and last are adjacent.
    Solution: max of (rob 0 to n-2) and (rob 1 to n-1)
    """
    if len(nums) == 1:
        return nums[0]

    def rob_range(start: int, end: int) -> int:
        prev2, prev1 = 0, 0
        for i in range(start, end + 1):
            curr = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, curr
        return prev1

    return max(rob_range(0, len(nums) - 2), rob_range(1, len(nums) - 1))`,
	testCode: `import pytest
from solution import rob


class TestHouseRobber:
    def test_example_1(self):
        """Test [1,2,3,1] -> 4"""
        assert rob([1, 2, 3, 1]) == 4

    def test_example_2(self):
        """Test [2,7,9,3,1] -> 12"""
        assert rob([2, 7, 9, 3, 1]) == 12

    def test_two_houses(self):
        """Test [2,1,1,2] -> 4"""
        assert rob([2, 1, 1, 2]) == 4

    def test_single_house(self):
        """Test single house"""
        assert rob([5]) == 5

    def test_two_houses_first_bigger(self):
        """Test two houses, first bigger"""
        assert rob([2, 1]) == 2

    def test_two_houses_second_bigger(self):
        """Test two houses, second bigger"""
        assert rob([1, 2]) == 2

    def test_all_same(self):
        """Test all same values"""
        assert rob([3, 3, 3, 3]) == 6

    def test_increasing(self):
        """Test increasing values"""
        assert rob([1, 2, 3, 4, 5]) == 9  # 1 + 3 + 5

    def test_decreasing(self):
        """Test decreasing values"""
        assert rob([5, 4, 3, 2, 1]) == 9  # 5 + 3 + 1

    def test_alternating(self):
        """Test alternating high-low"""
        assert rob([10, 1, 10, 1, 10]) == 30  # 10 + 10 + 10`,
	hint1: `At each house, you either rob it (add to best from 2 houses back) or skip it (take best from previous house). dp[i] = max(dp[i-1], dp[i-2] + nums[i])`,
	hint2: `You only need the last two values, not the entire array. Use two variables: prev2 (2 houses back) and prev1 (1 house back), update them as you iterate.`,
	whyItMatters: `House Robber is a classic DP problem that demonstrates the "include or exclude" pattern. It's frequently asked at FAANG interviews.

**Why This Matters:**

**1. Include/Exclude Pattern**

\`\`\`python
# For each element, decide: include it or exclude it
# Include: can't include adjacent, so look 2 back
# Exclude: take the best so far

dp[i] = max(
    dp[i-1],           # Exclude current
    dp[i-2] + nums[i]  # Include current
)

# This pattern appears in many problems:
# - Maximum sum non-adjacent elements
# - Delete and Earn
# - Pizza with 3n slices
\`\`\`

**2. Space Optimization Pattern**

\`\`\`python
# From O(n) to O(1) space
# Only need last two values

prev2, prev1 = nums[0], max(nums[0], nums[1])
for i in range(2, n):
    prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
\`\`\`

**3. House Robber Variations**

\`\`\`python
# House Robber II: Circular arrangement
# Can't rob both first and last house
def rob_circular(nums):
    if len(nums) == 1:
        return nums[0]
    return max(rob(nums[:-1]), rob(nums[1:]))

# House Robber III: Binary tree
# Can't rob parent and child
def rob_tree(root):
    def dfs(node):
        if not node:
            return (0, 0)  # (rob this, don't rob this)

        left = dfs(node.left)
        right = dfs(node.right)

        rob_this = node.val + left[1] + right[1]
        dont_rob = max(left) + max(right)

        return (rob_this, dont_rob)

    return max(dfs(root))
\`\`\`

**4. Connection to Other Problems**

\`\`\`python
# Delete and Earn (LeetCode 740)
# Transform into House Robber by counting element values
def delete_and_earn(nums):
    if not nums:
        return 0

    max_num = max(nums)
    points = [0] * (max_num + 1)
    for num in nums:
        points[num] += num

    # Now it's House Robber on points array!
    return rob(points)

# Maximum Alternating Subsequence Sum
# Similar include/exclude logic
\`\`\`

**5. Interview Tips**

\`\`\`python
# Always clarify:
# - Can houses have 0 money? (Yes)
# - What if array is empty? (Return 0)
# - Are there negative values? (Usually no)

# Start with O(n) space solution, then optimize
# Mention the circular variant (shows depth of knowledge)
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Ограбление домов',
			description: `Найдите максимальную сумму денег, которую можно украсть, не вызвав полицию.

**Задача:**

Вы профессиональный грабитель, планирующий ограбить дома вдоль улицы. В каждом доме спрятана определённая сумма денег.

**Ограничение:** Соседние дома связаны системой безопасности - если два соседних дома ограблены в одну ночь, вызывается полиция.

Дан массив целых чисел \`nums\`, представляющий сумму денег в каждом доме. Верните максимальную сумму, которую можно украсть сегодня ночью **без вызова полиции**.

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3, 1]
Выход: 4
Объяснение: Ограбить дом 1 (1) и дом 3 (3). Итого = 4

Вход: nums = [2, 7, 9, 3, 1]
Выход: 12
Объяснение: Ограбить дома 1, 3, 5 (2 + 9 + 1 = 12)
\`\`\`

**Ключевая идея:**

У каждого дома два выбора:
1. **Ограбить:** Добавить текущий + лучшее от 2 домов назад
2. **Пропустить:** Взять лучшее от предыдущего дома

\`dp[i] = max(dp[i-1], dp[i-2] + nums[i])\`

**Ограничения:**
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 400

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) с оптимизацией`,
			hint1: `У каждого дома: либо грабим (добавляем к лучшему от 2 домов назад), либо пропускаем (берём лучшее от предыдущего). dp[i] = max(dp[i-1], dp[i-2] + nums[i])`,
			hint2: `Нужны только два последних значения. Используйте две переменные: prev2 и prev1, обновляйте их при итерации.`,
			whyItMatters: `House Robber - классическая задача DP, демонстрирующая паттерн "включить или исключить". Часто спрашивается на FAANG-интервью.

**Почему это важно:**

**1. Паттерн включить/исключить**

Для каждого элемента решаем: включить или исключить.

**2. Паттерн оптимизации памяти**

От O(n) к O(1) - нужны только два последних значения.

**3. Вариации House Robber**

- House Robber II: Круговое расположение
- House Robber III: Бинарное дерево

**4. Связь с другими задачами**

Delete and Earn, Maximum Alternating Subsequence Sum.`,
			solutionCode: `from typing import List


def rob(nums: List[int]) -> int:
    """
    Находит максимальную сумму денег для ограбления.

    Args:
        nums: Сумма денег в каждом доме

    Returns:
        Максимальная сумма без вызова полиции
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    # Оптимизированный по памяти подход
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr

    return prev1`
		},
		uz: {
			title: 'Uylarni o\'g\'irlash',
			description: `Politsiyani chaqirmasdan o'g'irlash mumkin bo'lgan maksimal pul miqdorini toping.

**Masala:**

Siz ko'cha bo'ylab uylarni o'g'irlashni rejalashtirgan professional o'g'risiz. Har bir uyda ma'lum miqdorda pul yashirilgan.

**Cheklov:** Qo'shni uylar xavfsizlik tizimi bilan bog'langan - agar bir kechada ikkita qo'shni uy o'g'irlansa, politsiya chaqiriladi.

Har bir uydagi pul miqdorini ifodalovchi butun sonlar massivi \`nums\` berilgan. Bugun kechqurun **politsiyani chaqirmasdan** o'g'irlash mumkin bo'lgan maksimal summani qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3, 1]
Chiqish: 4
Izoh: 1-uy (1) va 3-uyni (3) o'g'irlash. Jami = 4

Kirish: nums = [2, 7, 9, 3, 1]
Chiqish: 12
Izoh: 1, 3, 5-uylarni o'g'irlash (2 + 9 + 1 = 12)
\`\`\`

**Asosiy tushuncha:**

Har bir uyda ikkita tanlov:
1. **O'g'irlash:** Joriy + 2 uy oldindagi eng yaxshisini qo'shish
2. **O'tkazib yuborish:** Oldingi uydagi eng yaxshisini olish

\`dp[i] = max(dp[i-1], dp[i-2] + nums[i])\`

**Cheklovlar:**
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 400

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) optimallashtirish bilan`,
			hint1: `Har bir uyda: o'g'irlaymiz (2 uy oldindagi eng yaxshisiga qo'shamiz) yoki o'tkazib yuboramiz (oldingi eng yaxshisini olamiz). dp[i] = max(dp[i-1], dp[i-2] + nums[i])`,
			hint2: `Faqat oxirgi ikki qiymat kerak. Ikkita o'zgaruvchi ishlating: prev2 va prev1, iteratsiya qilganda yangilang.`,
			whyItMatters: `House Robber - "qo'shish yoki chiqarib tashlash" patternini ko'rsatadigan klassik DP masalasi. FAANG intervyularida tez-tez so'raladi.

**Bu nima uchun muhim:**

**1. Qo'shish/chiqarib tashlash patterni**

Har bir element uchun qaror qilamiz: qo'shish yoki chiqarib tashlash.

**2. Xotira optimallashtirish patterni**

O(n) dan O(1) ga - faqat oxirgi ikki qiymat kerak.

**3. House Robber variatsiyalari**

- House Robber II: Doira shaklida joylashish
- House Robber III: Binar daraxt

**4. Boshqa masalalar bilan bog'liqlik**

Delete and Earn, Maximum Alternating Subsequence Sum.`,
			solutionCode: `from typing import List


def rob(nums: List[int]) -> int:
    """
    O'g'irlash mumkin bo'lgan maksimal pul miqdorini topadi.

    Args:
        nums: Har bir uydagi pul miqdori

    Returns:
        Politsiyani chaqirmasdan o'g'irlash mumkin bo'lgan maksimal summa
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    # Xotira bo'yicha optimallashtirilgan yondashuv
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = curr

    return prev1`
		}
	}
};

export default task;
