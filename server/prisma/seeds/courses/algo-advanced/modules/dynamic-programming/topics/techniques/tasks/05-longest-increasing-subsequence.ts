import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-longest-increasing-subsequence',
	title: 'Longest Increasing Subsequence',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'binary-search', 'subsequence'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the length of the longest strictly increasing subsequence.

**Problem:**

Given an integer array \`nums\`, return the length of the longest strictly increasing subsequence.

A **subsequence** is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.

**Examples:**

\`\`\`
Input: nums = [10, 9, 2, 5, 3, 7, 101, 18]
Output: 4
Explanation: The longest increasing subsequence is [2, 3, 7, 101] or [2, 3, 7, 18]

Input: nums = [0, 1, 0, 3, 2, 3]
Output: 4
Explanation: The longest increasing subsequence is [0, 1, 2, 3]

Input: nums = [7, 7, 7, 7, 7]
Output: 1
Explanation: All elements are the same, LIS is any single element
\`\`\`

**Approaches:**

1. **O(n²) DP:** For each element, find longest LIS ending at that element
2. **O(n log n) Binary Search:** Maintain sorted sequence, use binary search

**DP Approach (O(n²)):**

\`\`\`
nums = [10, 9, 2, 5, 3, 7, 101, 18]
dp   = [1,  1, 1, 2, 2, 3,  4,   4]

dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
\`\`\`

**Constraints:**
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4

**Time Complexity:** O(n²) for DP, O(n log n) with binary search
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def length_of_lis(nums: List[int]) -> int:
    # TODO: Find length of longest strictly increasing subsequence

    return 0`,
	solutionCode: `from typing import List
import bisect


def length_of_lis(nums: List[int]) -> int:
    """
    Find length of longest increasing subsequence.

    Args:
        nums: List of integers

    Returns:
        Length of the longest strictly increasing subsequence
    """
    if not nums:
        return 0

    n = len(nums)
    # dp[i] = length of LIS ending at index i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


# O(n log n) approach using binary search
def length_of_lis_binary_search(nums: List[int]) -> int:
    """
    Optimized LIS using binary search.

    Maintain a list 'tails' where tails[i] is the smallest tail
    element for LIS of length i+1.
    """
    if not nums:
        return 0

    tails = []

    for num in nums:
        # Find position to insert/replace
        pos = bisect.bisect_left(tails, num)

        if pos == len(tails):
            # num is larger than all tails, extend LIS
            tails.append(num)
        else:
            # Replace existing tail with smaller value
            tails[pos] = num

    return len(tails)


# Version that reconstructs the actual LIS
def lis_with_sequence(nums: List[int]) -> tuple:
    """Returns both length and the actual LIS."""
    if not nums:
        return 0, []

    n = len(nums)
    dp = [1] * n
    parent = [-1] * n  # Track previous element in LIS

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # Find index of maximum dp value
    max_length = max(dp)
    max_idx = dp.index(max_length)

    # Reconstruct LIS
    lis = []
    idx = max_idx
    while idx != -1:
        lis.append(nums[idx])
        idx = parent[idx]

    return max_length, list(reversed(lis))`,
	testCode: `import pytest
from solution import length_of_lis


class TestLongestIncreasingSubsequence:
    def test_basic_example(self):
        """Test [10,9,2,5,3,7,101,18] -> 4"""
        assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4

    def test_with_duplicates(self):
        """Test [0,1,0,3,2,3] -> 4"""
        assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4

    def test_all_same(self):
        """Test all same elements"""
        assert length_of_lis([7, 7, 7, 7, 7]) == 1

    def test_already_sorted(self):
        """Test already increasing"""
        assert length_of_lis([1, 2, 3, 4, 5]) == 5

    def test_decreasing(self):
        """Test strictly decreasing"""
        assert length_of_lis([5, 4, 3, 2, 1]) == 1

    def test_single_element(self):
        """Test single element"""
        assert length_of_lis([10]) == 1

    def test_two_elements_increasing(self):
        """Test two elements increasing"""
        assert length_of_lis([1, 2]) == 2

    def test_two_elements_decreasing(self):
        """Test two elements decreasing"""
        assert length_of_lis([2, 1]) == 1

    def test_alternating(self):
        """Test alternating pattern"""
        assert length_of_lis([1, 3, 2, 4, 3, 5]) == 4

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert length_of_lis([-2, -1, 0, 1, 2]) == 5
        assert length_of_lis([5, -1, 3, -2, 4]) == 3`,
	hint1: `Define dp[i] as the length of the longest increasing subsequence ending at index i. Initialize all dp values to 1 (each element is its own subsequence).`,
	hint2: `For each element at index i, check all previous elements at index j. If nums[j] < nums[i], you can extend the subsequence ending at j. Update dp[i] = max(dp[i], dp[j] + 1).`,
	whyItMatters: `LIS is a classic DP problem that appears in many variations at FAANG interviews. The O(n log n) optimization is a great example of combining DP with binary search.

**Why This Matters:**

**1. Two Approaches**

\`\`\`python
# O(n²) DP - Easy to understand
dp = [1] * n
for i in range(n):
    for j in range(i):
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], dp[j] + 1)

# O(n log n) Binary Search - Interview optimization
tails = []
for num in nums:
    pos = bisect.bisect_left(tails, num)
    if pos == len(tails):
        tails.append(num)
    else:
        tails[pos] = num
return len(tails)
\`\`\`

**2. Binary Search Intuition**

Maintain smallest possible tail for each LIS length:

\`\`\`
nums = [10, 9, 2, 5, 3, 7]
tails = []

Process 10: tails = [10]
Process 9:  tails = [9]   (9 < 10, replace)
Process 2:  tails = [2]   (2 < 9, replace)
Process 5:  tails = [2, 5]  (5 > 2, extend)
Process 3:  tails = [2, 3]  (3 replaces 5)
Process 7:  tails = [2, 3, 7]  (7 > 3, extend)

Answer: len(tails) = 3
\`\`\`

**3. Related FAANG Problems**

\`\`\`python
# Longest Increasing Subsequence Count
# How many LIS of maximum length exist?

# Russian Doll Envelopes (2D LIS)
# Sort by width, then LIS on heights

# Maximum Sum Increasing Subsequence
# Track sum instead of length

# Number of Longest Increasing Subsequence
dp = [1] * n
count = [1] * n
for i in range(n):
    for j in range(i):
        if nums[j] < nums[i]:
            if dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                count[i] = count[j]
            elif dp[j] + 1 == dp[i]:
                count[i] += count[j]
\`\`\`

**4. Patience Sorting Connection**

The binary search approach is related to patience sorting:
- Each pile represents subsequences of different lengths
- Top cards form an increasing sequence

**5. Real-World Applications**

- Version control (finding longest unchanged code section)
- Stock market analysis (longest growth periods)
- Data compression (finding patterns)`,
	order: 5,
	translations: {
		ru: {
			title: 'Наибольшая возрастающая подпоследовательность',
			description: `Найдите длину наибольшей строго возрастающей подпоследовательности.

**Задача:**

Дан массив целых чисел \`nums\`, верните длину наибольшей строго возрастающей подпоследовательности.

**Подпоследовательность** - это массив, который можно получить из другого массива удалением некоторых или ни одного элемента без изменения порядка оставшихся элементов.

**Примеры:**

\`\`\`
Вход: nums = [10, 9, 2, 5, 3, 7, 101, 18]
Выход: 4
Объяснение: Наибольшая возрастающая подпоследовательность [2, 3, 7, 101]

Вход: nums = [7, 7, 7, 7, 7]
Выход: 1
Объяснение: Все элементы одинаковы
\`\`\`

**Подходы:**

1. **O(n²) DP:** Для каждого элемента находим наибольшую LIS, заканчивающуюся на нём
2. **O(n log n) Бинарный поиск:** Поддерживаем отсортированную последовательность

**Ограничения:**
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4

**Временная сложность:** O(n²) для DP, O(n log n) с бинарным поиском
**Пространственная сложность:** O(n)`,
			hint1: `Определите dp[i] как длину наибольшей возрастающей подпоследовательности, заканчивающейся на индексе i. Инициализируйте все значения dp единицами.`,
			hint2: `Для каждого элемента с индексом i проверьте все предыдущие элементы с индексом j. Если nums[j] < nums[i], можно расширить подпоследовательность. Обновите dp[i] = max(dp[i], dp[j] + 1).`,
			whyItMatters: `LIS - классическая задача DP, которая встречается во многих вариациях на FAANG-интервью. O(n log n) оптимизация - отличный пример комбинации DP с бинарным поиском.

**Почему это важно:**

**1. Два подхода**

- O(n²) DP - легко понять
- O(n log n) бинарный поиск - оптимизация для интервью

**2. Интуиция бинарного поиска**

Поддерживаем наименьший возможный хвост для каждой длины LIS.

**3. Связанные задачи FAANG**

- Russian Doll Envelopes (2D LIS)
- Maximum Sum Increasing Subsequence
- Number of Longest Increasing Subsequence

**4. Применения в реальном мире**

- Контроль версий
- Анализ фондового рынка
- Сжатие данных`,
			solutionCode: `from typing import List


def length_of_lis(nums: List[int]) -> int:
    """
    Находит длину наибольшей возрастающей подпоследовательности.

    Args:
        nums: Список целых чисел

    Returns:
        Длина наибольшей строго возрастающей подпоследовательности
    """
    if not nums:
        return 0

    n = len(nums)
    # dp[i] = длина LIS, заканчивающейся на индексе i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)`
		},
		uz: {
			title: 'Eng uzun o\'suvchi qism ketma-ketlik',
			description: `Eng uzun qat'iy o'suvchi qism ketma-ketlikning uzunligini toping.

**Masala:**

Butun sonlar massivi \`nums\` berilgan, eng uzun qat'iy o'suvchi qism ketma-ketlik uzunligini qaytaring.

**Qism ketma-ketlik** - bu boshqa massivdan ba'zi yoki hech qanday elementlarni o'chirmasdan qolgan elementlar tartibini o'zgartirmasdan olinishi mumkin bo'lgan massiv.

**Misollar:**

\`\`\`
Kirish: nums = [10, 9, 2, 5, 3, 7, 101, 18]
Chiqish: 4
Izoh: Eng uzun o'suvchi qism ketma-ketlik [2, 3, 7, 101]

Kirish: nums = [7, 7, 7, 7, 7]
Chiqish: 1
Izoh: Barcha elementlar bir xil
\`\`\`

**Yondashuvlar:**

1. **O(n²) DP:** Har bir element uchun unda tugaydigan eng uzun LIS ni topamiz
2. **O(n log n) Binar qidiruv:** Saralangan ketma-ketlikni saqlaymiz

**Cheklovlar:**
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4

**Vaqt murakkabligi:** O(n²) DP uchun, O(n log n) binar qidiruv bilan
**Xotira murakkabligi:** O(n)`,
			hint1: `dp[i] ni i indeksida tugaydigan eng uzun o'suvchi qism ketma-ketlik uzunligi sifatida aniqlang. Barcha dp qiymatlarini 1 ga boshlang.`,
			hint2: `Har bir i indeksidagi element uchun barcha oldingi j indeksidagi elementlarni tekshiring. Agar nums[j] < nums[i] bo'lsa, j da tugaydigan qism ketma-ketlikni kengaytirish mumkin. dp[i] = max(dp[i], dp[j] + 1) ni yangilang.`,
			whyItMatters: `LIS - FAANG intervyularida ko'p variatsiyalarda uchraydigan klassik DP masalasi. O(n log n) optimallashtirish DP ni binar qidiruv bilan birlashtirish uchun ajoyib misol.

**Bu nima uchun muhim:**

**1. Ikki yondashuv**

- O(n²) DP - tushunish oson
- O(n log n) binar qidiruv - intervyu uchun optimallashtirish

**2. Binar qidiruv intuitsiyasi**

Har bir LIS uzunligi uchun eng kichik mumkin bo'lgan oxirini saqlaymiz.

**3. Bog'liq FAANG masalalari**

- Russian Doll Envelopes (2D LIS)
- Maximum Sum Increasing Subsequence
- Number of Longest Increasing Subsequence

**4. Haqiqiy dunyo qo'llanishlari**

- Versiya nazorati
- Fond bozori tahlili
- Ma'lumotlarni siqish`,
			solutionCode: `from typing import List


def length_of_lis(nums: List[int]) -> int:
    """
    Eng uzun o'suvchi qism ketma-ketlik uzunligini topadi.

    Args:
        nums: Butun sonlar ro'yxati

    Returns:
        Eng uzun qat'iy o'suvchi qism ketma-ketlik uzunligi
    """
    if not nums:
        return 0

    n = len(nums)
    # dp[i] = i indeksida tugaydigan LIS uzunligi
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)`
		}
	}
};

export default task;
