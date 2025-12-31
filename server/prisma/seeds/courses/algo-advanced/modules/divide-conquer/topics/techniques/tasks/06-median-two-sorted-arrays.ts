import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-median-two-arrays',
	title: 'Median of Two Sorted Arrays',
	difficulty: 'hard',
	tags: ['python', 'divide-conquer', 'binary-search', 'array'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the median of two sorted arrays in O(log(m+n)) time.

**Problem:**

Given two sorted arrays \`nums1\` and \`nums2\` of size \`m\` and \`n\` respectively, return the **median** of the two sorted arrays.

The overall run time complexity should be **O(log(m+n))**.

**Examples:**

\`\`\`
Input: nums1 = [1, 3], nums2 = [2]
Output: 2.00000

Explanation: merged array = [1, 2, 3], median is 2.

Input: nums1 = [1, 2], nums2 = [3, 4]
Output: 2.50000

Explanation: merged array = [1, 2, 3, 4], median is (2 + 3) / 2 = 2.5.

Input: nums1 = [], nums2 = [1]
Output: 1.00000
\`\`\`

**Key Insight:**

Binary search on the smaller array to find the correct partition. At the correct partition:
- All elements on the left side ≤ all elements on the right side
- Left side has exactly half of total elements

**Visualization:**

\`\`\`
nums1: [1, 3, 5, 7]  partition at i=2
nums2: [2, 4, 6]     partition at j=1

Left: [1, 3] + [2] = 3 elements
Right: [5, 7] + [4, 6] = 4 elements

Check: max(left) <= min(right)
       max(3, 2) <= min(5, 4) → 3 <= 4 ✓
\`\`\`

**Constraints:**
- nums1.length == m
- nums2.length == n
- 0 <= m <= 1000
- 0 <= n <= 1000
- 1 <= m + n <= 2000
- -10^6 <= nums1[i], nums2[i] <= 10^6

**Time Complexity:** O(log(min(m, n)))
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    # TODO: Find median of two sorted arrays (O(log(min(m,n))))

    return 0.0`,
	solutionCode: `from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Find median of two sorted arrays in O(log(min(m,n))).
    """
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    half = (m + n + 1) // 2  # Elements in left partition

    while lo <= hi:
        i = (lo + hi) // 2  # Partition index in nums1
        j = half - i         # Partition index in nums2

        # Get boundary elements
        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]

        # Check if partition is correct
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found correct partition
            if (m + n) % 2 == 1:
                return max(max_left1, max_left2)
            else:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
        elif max_left1 > min_right2:
            hi = i - 1  # Move partition left in nums1
        else:
            lo = i + 1  # Move partition right in nums1

    return 0.0  # Should never reach here


# Alternative: Find kth element (generalization)
def find_kth_element(nums1: List[int], nums2: List[int], k: int) -> int:
    """Find kth smallest element in union of two sorted arrays."""
    m, n = len(nums1), len(nums2)

    if m > n:
        return find_kth_element(nums2, nums1, k)

    if m == 0:
        return nums2[k - 1]

    if k == 1:
        return min(nums1[0], nums2[0])

    # Take min(k//2, length) elements from each array
    i = min(m, k // 2)
    j = min(n, k // 2)

    if nums1[i - 1] < nums2[j - 1]:
        # Discard first i elements of nums1
        return find_kth_element(nums1[i:], nums2, k - i)
    else:
        # Discard first j elements of nums2
        return find_kth_element(nums1, nums2[j:], k - j)


def find_median_kth(nums1: List[int], nums2: List[int]) -> float:
    """Find median using kth element approach."""
    total = len(nums1) + len(nums2)

    if total % 2 == 1:
        return find_kth_element(nums1, nums2, total // 2 + 1)
    else:
        left = find_kth_element(nums1, nums2, total // 2)
        right = find_kth_element(nums1, nums2, total // 2 + 1)
        return (left + right) / 2


# Simple merge approach (O(m+n) for comparison)
def find_median_merge(nums1: List[int], nums2: List[int]) -> float:
    """O(m+n) merge approach."""
    merged = []
    i = j = 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1

    merged.extend(nums1[i:])
    merged.extend(nums2[j:])

    n = len(merged)
    if n % 2 == 1:
        return merged[n // 2]
    else:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2`,
	testCode: `import pytest
from solution import find_median_sorted_arrays


class TestMedianTwoSortedArrays:
    def test_odd_total(self):
        """Test odd total length"""
        assert find_median_sorted_arrays([1, 3], [2]) == 2.0

    def test_even_total(self):
        """Test even total length"""
        assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5

    def test_one_empty(self):
        """Test one empty array"""
        assert find_median_sorted_arrays([], [1]) == 1.0
        assert find_median_sorted_arrays([1], []) == 1.0

    def test_single_elements(self):
        """Test single element arrays"""
        assert find_median_sorted_arrays([1], [2]) == 1.5

    def test_same_elements(self):
        """Test same elements"""
        assert find_median_sorted_arrays([1, 1], [1, 1]) == 1.0

    def test_no_overlap(self):
        """Test non-overlapping arrays"""
        assert find_median_sorted_arrays([1, 2], [3, 4, 5, 6]) == 3.5

    def test_interleaved(self):
        """Test interleaved elements"""
        assert find_median_sorted_arrays([1, 3, 5], [2, 4, 6]) == 3.5

    def test_different_sizes(self):
        """Test very different sizes"""
        assert find_median_sorted_arrays([1], [2, 3, 4, 5, 6]) == 3.5

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert find_median_sorted_arrays([-5, -3, -1], [0, 2, 4]) == -0.5

    def test_larger_arrays(self):
        """Test with larger arrays"""
        nums1 = [1, 3, 5, 7, 9]
        nums2 = [2, 4, 6, 8, 10]
        assert find_median_sorted_arrays(nums1, nums2) == 5.5`,
	hint1: `Binary search on the smaller array. For partition at index i in nums1, the partition in nums2 is at j = (total+1)//2 - i. This ensures left side has half the elements.`,
	hint2: `Valid partition when max(left1, left2) <= min(right1, right2). If max_left1 > min_right2, move partition left in nums1. Use float('-inf') and float('inf') for boundary cases.`,
	whyItMatters: `Median of Two Sorted Arrays is one of the most challenging divide and conquer problems. It demonstrates advanced binary search techniques and is a common hard interview question.

**Why This Matters:**

**1. The Partition Concept**

\`\`\`python
# We want to split combined arrays into two equal halves
# Left half: elements from nums1[:i] + nums2[:j]
# Right half: elements from nums1[i:] + nums2[j:]

# Total elements: m + n
# Left half size: (m + n + 1) // 2

# If we fix partition i in nums1:
# j = half - i  (to get correct total on left)
\`\`\`

**2. Validity Check**

\`\`\`python
# Correct partition means:
# max(left) <= min(right)

# Which translates to:
# nums1[i-1] <= nums2[j]     (max_left1 <= min_right2)
# nums2[j-1] <= nums1[i]     (max_left2 <= min_right1)

# If not valid, adjust i using binary search
\`\`\`

**3. Edge Cases**

\`\`\`python
# Handle empty partitions:
max_left1 = float('-inf') if i == 0 else nums1[i-1]
min_right1 = float('inf') if i == m else nums1[i]

# This elegantly handles:
# - One array being empty
# - Partition at array boundaries
\`\`\`

**4. Why O(log(min(m,n)))**

\`\`\`python
# Binary search on smaller array
# Each iteration halves search space
# Always search on min(m, n) size

# Swap arrays if needed:
if len(nums1) > len(nums2):
    nums1, nums2 = nums2, nums1
\`\`\`

**5. Generalization: Kth Element**

\`\`\`python
# Find kth element in union of sorted arrays
# Median is just k = (m+n+1)//2 and (m+n+2)//2

# This technique works for any kth element query
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Медиана двух отсортированных массивов',
			description: `Найдите медиану двух отсортированных массивов за O(log(m+n)).

**Задача:**

Даны два отсортированных массива \`nums1\` и \`nums2\` размеров \`m\` и \`n\`. Верните медиану объединённого массива.

Сложность должна быть **O(log(m+n))**.

**Примеры:**

\`\`\`
Вход: nums1 = [1, 3], nums2 = [2]
Выход: 2.00000

Объяснение: объединённый массив = [1, 2, 3], медиана = 2.

Вход: nums1 = [1, 2], nums2 = [3, 4]
Выход: 2.50000

Объяснение: медиана = (2 + 3) / 2 = 2.5.
\`\`\`

**Ключевая идея:**

Бинарный поиск по меньшему массиву для нахождения правильного разбиения. При правильном разбиении:
- Все элементы слева ≤ всем элементам справа
- Слева ровно половина элементов

**Ограничения:**
- 0 <= m, n <= 1000
- 1 <= m + n <= 2000

**Временная сложность:** O(log(min(m, n)))
**Пространственная сложность:** O(1)`,
			hint1: `Бинарный поиск по меньшему массиву. Для разбиения в позиции i в nums1, разбиение в nums2 в j = (total+1)//2 - i.`,
			hint2: `Правильное разбиение когда max(left1, left2) <= min(right1, right2). Используйте float('-inf') и float('inf') для краевых случаев.`,
			whyItMatters: `Median of Two Sorted Arrays - одна из сложнейших задач "разделяй и властвуй". Частый вопрос на собеседованиях уровня hard.

**Почему это важно:**

**1. Концепция разбиения**

Делим объединённые массивы на две равные половины.

**2. Проверка валидности**

max(левые) <= min(правые).

**3. Обобщение**

Техника работает для любого k-го элемента.`,
			solutionCode: `from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """Находит медиану двух отсортированных массивов."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    half = (m + n + 1) // 2

    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i

        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 1:
                return max(max_left1, max_left2)
            else:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
        elif max_left1 > min_right2:
            hi = i - 1
        else:
            lo = i + 1

    return 0.0`
		},
		uz: {
			title: 'Ikki saralangan massivning medianasi',
			description: `O(log(m+n)) da ikki saralangan massivning medianasini toping.

**Masala:**

\`m\` va \`n\` o'lchamli ikkita saralangan massiv \`nums1\` va \`nums2\` berilgan. Birlashtirilgan massivning medianasini qaytaring.

Murakkablik **O(log(m+n))** bo'lishi kerak.

**Misollar:**

\`\`\`
Kirish: nums1 = [1, 3], nums2 = [2]
Chiqish: 2.00000

Izoh: birlashtirilgan massiv = [1, 2, 3], mediana = 2.

Kirish: nums1 = [1, 2], nums2 = [3, 4]
Chiqish: 2.50000

Izoh: mediana = (2 + 3) / 2 = 2.5.
\`\`\`

**Asosiy tushuncha:**

To'g'ri bo'linishni topish uchun kichik massiv bo'yicha binar qidiruv. To'g'ri bo'linishda:
- Chapdagi barcha elementlar ≤ o'ngdagilar
- Chapda aynan yarim element

**Cheklovlar:**
- 0 <= m, n <= 1000
- 1 <= m + n <= 2000

**Vaqt murakkabligi:** O(log(min(m, n)))
**Xotira murakkabligi:** O(1)`,
			hint1: `Kichik massiv bo'yicha binar qidiruv. nums1 da i pozitsiyasida bo'linish uchun, nums2 da j = (total+1)//2 - i.`,
			hint2: `To'g'ri bo'linish max(left1, left2) <= min(right1, right2) bo'lganda. Chegaraviy holatlar uchun float('-inf') va float('inf') ishlating.`,
			whyItMatters: `Median of Two Sorted Arrays - eng qiyin "bo'l va hukmronlik qil" masalalaridan biri. Hard darajadagi tez-tez so'raladigan suhbat savoli.

**Bu nima uchun muhim:**

**1. Bo'linish konsepti**

Birlashtirilgan massivlarni ikkita teng yarmga bo'lamiz.

**2. To'g'rilikni tekshirish**

max(chaplar) <= min(o'nglar).

**3. Umumlashtirish**

Texnika istalgan k-chi element uchun ishlaydi.`,
			solutionCode: `from typing import List

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """Ikki saralangan massivning medianasini topadi."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    half = (m + n + 1) // 2

    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i

        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 1:
                return max(max_left1, max_left2)
            else:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
        elif max_left1 > min_right2:
            hi = i - 1
        else:
            lo = i + 1

    return 0.0`
		}
	}
};

export default task;
