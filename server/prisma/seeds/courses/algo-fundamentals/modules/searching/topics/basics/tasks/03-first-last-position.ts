import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-first-last-position',
	title: 'First and Last Position',
	difficulty: 'medium',
	tags: ['python', 'searching', 'binary-search'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the starting and ending position of a target value in a sorted array.

**Problem:**

Given an array of integers \`nums\` sorted in non-decreasing order, find the starting and ending position of a given \`target\` value.

If \`target\` is not found in the array, return \`[-1, -1]\`.

You must write an algorithm with O(log n) runtime complexity.

**Examples:**

\`\`\`
Input: nums = [5, 7, 7, 8, 8, 10], target = 8
Output: [3, 4]
Explanation: 8 first appears at index 3, last at index 4

Input: nums = [5, 7, 7, 8, 8, 10], target = 6
Output: [-1, -1]
Explanation: 6 is not in the array

Input: nums = [], target = 0
Output: [-1, -1]
\`\`\`

**Approach:**

Use two binary searches:
1. Find the **leftmost** (first) occurrence
2. Find the **rightmost** (last) occurrence

For leftmost: when you find target, continue searching left
For rightmost: when you find target, continue searching right

**Constraints:**
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
- nums is sorted in non-decreasing order

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def search_range(nums: List[int], target: int) -> List[int]:
    # TODO: Find first and last position of target in sorted array

    return [-1, -1]`,
	solutionCode: `from typing import List

def search_range(nums: List[int], target: int) -> List[int]:
    """
    Find first and last position of target in sorted array.

    Args:
        nums: Sorted array (may have duplicates)
        target: Value to find

    Returns:
        [first_index, last_index] or [-1, -1] if not found
    """
    def find_bound(is_left: bool) -> int:
        left, right = 0, len(nums) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                result = mid  # Found one, but keep searching
                if is_left:
                    right = mid - 1  # Continue searching left
                else:
                    left = mid + 1   # Continue searching right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    return [find_bound(True), find_bound(False)]


# Alternative: Using bisect module
import bisect

def search_range_bisect(nums: List[int], target: int) -> List[int]:
    """Using Python's bisect module."""
    left = bisect.bisect_left(nums, target)

    # Check if target exists
    if left >= len(nums) or nums[left] != target:
        return [-1, -1]

    right = bisect.bisect_right(nums, target) - 1
    return [left, right]


# Single binary search approach
def search_range_optimized(nums: List[int], target: int) -> List[int]:
    """Find left bound, then scan or binary search for right."""
    def find_left():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    left_idx = find_left()

    # Check if target exists
    if left_idx >= len(nums) or nums[left_idx] != target:
        return [-1, -1]

    # Find right bound
    def find_right():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    return [left_idx, find_right()]`,
	testCode: `import pytest
from solution import search_range

class TestSearchRange:
    def test_found_multiple(self):
        """Test with multiple occurrences"""
        assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]

    def test_not_found(self):
        """Test target not in array"""
        assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]

    def test_empty_array(self):
        """Test empty array"""
        assert search_range([], 0) == [-1, -1]

    def test_single_element_found(self):
        """Test single element - found"""
        assert search_range([1], 1) == [0, 0]

    def test_single_element_not_found(self):
        """Test single element - not found"""
        assert search_range([1], 0) == [-1, -1]

    def test_all_same(self):
        """Test array with all same elements"""
        assert search_range([2, 2, 2, 2, 2], 2) == [0, 4]

    def test_single_occurrence(self):
        """Test single occurrence in larger array"""
        assert search_range([1, 2, 3, 4, 5], 3) == [2, 2]

    def test_at_boundaries(self):
        """Test target at array boundaries"""
        assert search_range([1, 1, 2, 3, 4], 1) == [0, 1]
        assert search_range([1, 2, 3, 4, 4], 4) == [3, 4]

    def test_long_sequence(self):
        """Test longer sequence of duplicates"""
        assert search_range([1, 2, 2, 2, 2, 2, 2, 3], 2) == [1, 6]

    def test_target_larger_than_all(self):
        """Test target larger than all elements"""
        assert search_range([1, 2, 3, 4, 5], 10) == [-1, -1]`,
	hint1: `You need two binary searches: one to find the leftmost occurrence, one for the rightmost. When you find the target, don't stop - continue searching in the appropriate direction.`,
	hint2: `For leftmost: when nums[mid] == target, set result = mid and continue with right = mid - 1. For rightmost: set result = mid and continue with left = mid + 1.`,
	whyItMatters: `Finding bounds in sorted arrays with duplicates is a fundamental skill for range queries.

**Why This Matters:**

**1. Count Occurrences**

\`\`\`python
# Number of occurrences = right - left + 1
first, last = search_range(nums, target)
count = last - first + 1 if first != -1 else 0
\`\`\`

**2. bisect_left vs bisect_right**

\`\`\`python
import bisect

# bisect_left: first position where element can be inserted
# bisect_right: last position where element can be inserted

nums = [1, 2, 2, 2, 3]
bisect.bisect_left(nums, 2)   # Returns 1 (first 2)
bisect.bisect_right(nums, 2)  # Returns 4 (after last 2)
\`\`\`

**3. Template for Finding Bounds**

\`\`\`python
# When you find target, don't return immediately
# Instead, record the position and keep searching
result = -1
while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
        result = mid
        # For leftmost: right = mid - 1
        # For rightmost: left = mid + 1
\`\`\`

**4. Applications**

- Finding time ranges in logs
- Database range queries
- Finding all elements in a range`,
	order: 3,
	translations: {
		ru: {
			title: 'Первая и последняя позиция',
			description: `Найдите начальную и конечную позицию целевого значения в отсортированном массиве.

**Задача:**

Дан массив целых чисел \`nums\`, отсортированный по неубыванию. Найдите начальную и конечную позицию заданного \`target\`.

Если \`target\` не найден, верните \`[-1, -1]\`.

Алгоритм должен работать за O(log n).

**Примеры:**

\`\`\`
Вход: nums = [5, 7, 7, 8, 8, 10], target = 8
Выход: [3, 4]
Объяснение: 8 впервые появляется на индексе 3, последний раз на 4

Вход: nums = [5, 7, 7, 8, 8, 10], target = 6
Выход: [-1, -1]

Вход: nums = [], target = 0
Выход: [-1, -1]
\`\`\`

**Подход:**

Используйте два бинарных поиска:
1. Найти **самое левое** (первое) вхождение
2. Найти **самое правое** (последнее) вхождение

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `Нужны два бинарных поиска: один для самого левого, один для самого правого вхождения. Найдя target, не останавливайтесь - продолжайте искать в нужном направлении.`,
			hint2: `Для левого: когда nums[mid] == target, сохраните result = mid и продолжите с right = mid - 1. Для правого: сохраните result = mid и продолжите с left = mid + 1.`,
			whyItMatters: `Поиск границ в отсортированных массивах с дубликатами - фундаментальный навык для диапазонных запросов.

**Почему это важно:**

**1. Подсчёт вхождений**

Количество вхождений = right - left + 1.

**2. bisect_left vs bisect_right**

bisect_left: первая позиция для вставки. bisect_right: последняя позиция.

**3. Применения**

Поиск диапазонов времени в логах, диапазонные запросы в БД.`,
			solutionCode: `from typing import List

def search_range(nums: List[int], target: int) -> List[int]:
    """
    Находит первую и последнюю позицию target в отсортированном массиве.
    """
    def find_bound(is_left: bool) -> int:
        left, right = 0, len(nums) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                result = mid
                if is_left:
                    right = mid - 1  # Продолжаем искать слева
                else:
                    left = mid + 1   # Продолжаем искать справа
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    return [find_bound(True), find_bound(False)]`
		},
		uz: {
			title: 'Birinchi va oxirgi pozitsiya',
			description: `Tartiblangan massivda maqsadli qiymatning boshlang'ich va oxirgi pozitsiyasini toping.

**Masala:**

Kamaymaydigan tartibda tartiblangan butun sonlar massivi \`nums\` berilgan. Berilgan \`target\` qiymatining boshlang'ich va oxirgi pozitsiyasini toping.

Agar \`target\` topilmasa, \`[-1, -1]\` qaytaring.

O(log n) vaqt murakkabligida algoritm yozing.

**Misollar:**

\`\`\`
Kirish: nums = [5, 7, 7, 8, 8, 10], target = 8
Chiqish: [3, 4]
Tushuntirish: 8 birinchi marta indeks 3 da, oxirgi marta indeks 4 da

Kirish: nums = [5, 7, 7, 8, 8, 10], target = 6
Chiqish: [-1, -1]

Kirish: nums = [], target = 0
Chiqish: [-1, -1]
\`\`\`

**Yondashuv:**

Ikki binar qidiruv ishlating:
1. **Eng chapdagi** (birinchi) topilishni toping
2. **Eng o'ngdagi** (oxirgi) topilishni toping

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikki binar qidiruv kerak: biri eng chapdagi, biri eng o'ngdagi topilish uchun. target ni topganingizda to'xtamang - kerakli yo'nalishda qidirishni davom ettiring.`,
			hint2: `Chap uchun: nums[mid] == target bo'lganda, result = mid saqlang va right = mid - 1 bilan davom eting. O'ng uchun: result = mid saqlang va left = mid + 1 bilan davom eting.`,
			whyItMatters: `Dublikatlari bor tartiblangan massivlarda chegaralarni topish diapazon so'rovlari uchun asosiy ko'nikma.

**Bu nima uchun muhim:**

**1. Topilishlar sonini hisoblash**

Topilishlar soni = right - left + 1.

**2. bisect_left vs bisect_right**

bisect_left: qo'shish uchun birinchi pozitsiya. bisect_right: oxirgi pozitsiya.

**3. Qo'llanishlar**

Loglarda vaqt diapazonlarini topish, ma'lumotlar bazasida diapazon so'rovlari.`,
			solutionCode: `from typing import List

def search_range(nums: List[int], target: int) -> List[int]:
    """
    Tartiblangan massivda target ning birinchi va oxirgi pozitsiyasini topadi.
    """
    def find_bound(is_left: bool) -> int:
        left, right = 0, len(nums) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                result = mid
                if is_left:
                    right = mid - 1  # Chapda qidirishni davom etamiz
                else:
                    left = mid + 1   # O'ngda qidirishni davom etamiz
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    return [find_bound(True), find_bound(False)]`
		}
	}
};

export default task;
