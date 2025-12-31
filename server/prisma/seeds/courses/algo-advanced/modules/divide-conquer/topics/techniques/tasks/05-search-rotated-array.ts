import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-search-rotated',
	title: 'Search in Rotated Sorted Array',
	difficulty: 'medium',
	tags: ['python', 'divide-conquer', 'binary-search', 'array'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Search for a target in a rotated sorted array.

**Problem:**

You are given an integer array \`nums\` sorted in ascending order (with distinct values), that has been rotated at some unknown pivot.

Given the array after rotation and an integer \`target\`, return the index of \`target\` if it is in \`nums\`, or \`-1\` if it is not.

You must write an algorithm with **O(log n)** runtime complexity.

**Examples:**

\`\`\`
Input: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
Output: 4

Input: nums = [4, 5, 6, 7, 0, 1, 2], target = 3
Output: -1

Input: nums = [1], target = 0
Output: -1
\`\`\`

**Visualization:**

\`\`\`
Original sorted: [0, 1, 2, 4, 5, 6, 7]
Rotated at 4:    [4, 5, 6, 7, 0, 1, 2]
                     ^
                   pivot

At any point, one half is sorted:
[4, 5, 6, 7, 0, 1, 2]
    sorted    not sorted
OR
[6, 7, 0, 1, 2, 4, 5]
not sorted    sorted
\`\`\`

**Key Insight:**

At any midpoint, at least one half of the array is properly sorted. Determine which half is sorted, then check if target lies in that sorted half.

**Constraints:**
- 1 <= nums.length <= 5000
- -10^4 <= nums[i] <= 10^4
- All values of nums are unique
- nums was sorted and rotated between 1 and n times

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    # TODO: Search for target in rotated sorted array (O(log n))

    return -1`,
	solutionCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    """
    Search for target in rotated sorted array.
    """
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[lo] <= nums[mid]:
            # Left half [lo, mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1  # Target in left half
            else:
                lo = mid + 1  # Target in right half
        else:
            # Right half [mid, hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1  # Target in right half
            else:
                hi = mid - 1  # Target in left half

    return -1


# Find minimum in rotated array
def find_min(nums: List[int]) -> int:
    """Find minimum element (pivot point)."""
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = (lo + hi) // 2

        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid

    return nums[lo]


# Find rotation count
def find_rotation_count(nums: List[int]) -> int:
    """Find how many times array was rotated."""
    lo, hi = 0, len(nums) - 1

    # If not rotated
    if nums[lo] <= nums[hi]:
        return 0

    while lo < hi:
        mid = (lo + hi) // 2

        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid

    return lo


# Search with duplicates allowed
def search_with_duplicates(nums: List[int], target: int) -> bool:
    """Search when duplicates are allowed."""
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return True

        # Handle duplicates at boundaries
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
        elif nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

    return False


# Two-pass approach (find pivot first)
def search_two_pass(nums: List[int], target: int) -> int:
    """Find pivot first, then binary search."""
    n = len(nums)

    # Find pivot (minimum element)
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid

    pivot = lo

    # Determine which half to search
    if target >= nums[pivot] and target <= nums[n - 1]:
        lo, hi = pivot, n - 1
    else:
        lo, hi = 0, pivot - 1

    # Standard binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1`,
	testCode: `import pytest
from solution import search


class TestSearchRotatedArray:
    def test_basic_case(self):
        """Test basic rotated array"""
        assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4

    def test_not_found(self):
        """Test target not in array"""
        assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1

    def test_single_element_found(self):
        """Test single element found"""
        assert search([1], 1) == 0

    def test_single_element_not_found(self):
        """Test single element not found"""
        assert search([1], 0) == -1

    def test_two_elements(self):
        """Test two elements"""
        assert search([3, 1], 1) == 1
        assert search([3, 1], 3) == 0
        assert search([1, 3], 3) == 1

    def test_target_at_boundaries(self):
        """Test target at first or last position"""
        assert search([4, 5, 6, 7, 0, 1, 2], 4) == 0
        assert search([4, 5, 6, 7, 0, 1, 2], 2) == 6

    def test_no_rotation(self):
        """Test array not rotated"""
        assert search([1, 2, 3, 4, 5], 4) == 3

    def test_full_rotation(self):
        """Test rotated n times (back to original)"""
        assert search([1, 2, 3, 4, 5], 1) == 0

    def test_pivot_at_various_positions(self):
        """Test with different rotation amounts"""
        assert search([2, 3, 4, 5, 1], 1) == 4
        assert search([5, 1, 2, 3, 4], 5) == 0
        assert search([3, 4, 5, 1, 2], 4) == 1

    def test_target_at_pivot(self):
        """Test target is the minimum element"""
        assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4`,
	hint1: `At any midpoint, one half of the array is always sorted. Compare nums[lo] with nums[mid] to determine which half is sorted.`,
	hint2: `Once you know which half is sorted, check if target lies within that sorted range. If yes, search that half; otherwise search the other half.`,
	whyItMatters: `Search in Rotated Sorted Array is a classic binary search variation that tests understanding of the fundamental divide and conquer principle even when data has irregularities.

**Why This Matters:**

**1. Key Observation**

\`\`\`python
# In a rotated sorted array, at any midpoint:
# - Either left half [lo, mid] is sorted
# - Or right half [mid, hi] is sorted
# - Never both unsorted!

# Determine sorted half:
if nums[lo] <= nums[mid]:
    # Left half is sorted
else:
    # Right half is sorted
\`\`\`

**2. Decision Making**

\`\`\`python
# Once we know sorted half, check if target is there:

# If left sorted:
if nums[lo] <= target < nums[mid]:
    hi = mid - 1  # Search left
else:
    lo = mid + 1  # Search right

# If right sorted:
if nums[mid] < target <= nums[hi]:
    lo = mid + 1  # Search right
else:
    hi = mid - 1  # Search left
\`\`\`

**3. Related Problems**

\`\`\`python
# Find Minimum in Rotated Array
# Find Rotation Count
# Search in Rotated Array II (with duplicates)
# Rotated Binary Search variations
\`\`\`

**4. Handling Duplicates**

\`\`\`python
# When duplicates exist:
if nums[lo] == nums[mid] == nums[hi]:
    lo += 1
    hi -= 1
    # Worst case becomes O(n)
\`\`\`

**5. Two-Pass Alternative**

\`\`\`python
# Approach 2: Find pivot first
# 1. Binary search to find minimum (pivot)
# 2. Determine which half target is in
# 3. Standard binary search on that half

# Same O(log n) complexity, sometimes clearer
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Поиск в повёрнутом массиве',
			description: `Найдите элемент в повёрнутом отсортированном массиве.

**Задача:**

Дан отсортированный массив целых чисел (с уникальными значениями), повёрнутый на неизвестный pivot.

Найдите индекс \`target\` или верните \`-1\` если не найден.

Алгоритм должен работать за **O(log n)**.

**Примеры:**

\`\`\`
Вход: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
Выход: 4

Вход: nums = [4, 5, 6, 7, 0, 1, 2], target = 3
Выход: -1
\`\`\`

**Ключевая идея:**

В любой точке одна половина массива отсортирована. Определите какая, затем проверьте находится ли target в этой половине.

**Ограничения:**
- 1 <= nums.length <= 5000
- Все значения уникальны

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `В любой точке одна половина всегда отсортирована. Сравните nums[lo] с nums[mid] чтобы определить какая.`,
			hint2: `Зная отсортированную половину, проверьте находится ли target в ней. Если да - ищите там, иначе в другой половине.`,
			whyItMatters: `Search in Rotated Sorted Array - классическая вариация бинарного поиска, проверяющая понимание принципа "разделяй и властвуй".

**Почему это важно:**

**1. Ключевое наблюдение**

В любой точке одна из половин отсортирована.

**2. Принятие решений**

Определив отсортированную половину, проверяем диапазон.

**3. Связанные задачи**

Find Minimum, Find Rotation Count, Search with Duplicates.`,
			solutionCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    """Ищет target в повёрнутом отсортированном массиве."""
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

    return -1`
		},
		uz: {
			title: "Aylantirilgan massivda qidirish",
			description: `Aylantirilgan saralangan massivda elementni toping.

**Masala:**

Noma'lum pivot da aylantirilgan saralangan butun sonlar massivi (noyob qiymatlar bilan) berilgan.

\`target\` indeksini toping yoki topilmasa \`-1\` qaytaring.

Algoritm **O(log n)** da ishlashi kerak.

**Misollar:**

\`\`\`
Kirish: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
Chiqish: 4

Kirish: nums = [4, 5, 6, 7, 0, 1, 2], target = 3
Chiqish: -1
\`\`\`

**Asosiy tushuncha:**

Istalgan nuqtada massivning bir yarmi saralangan. Qaysi ekanini aniqlang, keyin target o'sha yarmida ekanligini tekshiring.

**Cheklovlar:**
- 1 <= nums.length <= 5000
- Barcha qiymatlar noyob

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Istalgan nuqtada bir yarm har doim saralangan. Qaysi ekanini aniqlash uchun nums[lo] ni nums[mid] bilan solishtiring.`,
			hint2: `Saralangan yarmni bilgach, target o'sha diapazondaligini tekshiring. Ha bo'lsa - u yerda qidiring, aks holda boshqa yarmda.`,
			whyItMatters: `Search in Rotated Sorted Array - "bo'l va hukmronlik qil" prinsipi tushunilishini tekshiruvchi klassik binar qidiruv variatsiyasi.

**Bu nima uchun muhim:**

**1. Asosiy kuzatuv**

Istalgan nuqtada yarmlardan biri saralangan.

**2. Qaror qabul qilish**

Saralangan yarmni aniqlagach, diapazonni tekshiramiz.

**3. Bog'liq masalalar**

Find Minimum, Find Rotation Count, Search with Duplicates.`,
			solutionCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    """Aylantirilgan saralangan massivda target ni qidiradi."""
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

    return -1`
		}
	}
};

export default task;
