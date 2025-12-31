import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-quick-sort',
	title: 'Quick Sort',
	difficulty: 'medium',
	tags: ['python', 'divide-conquer', 'sorting', 'recursion', 'partition'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Quick Sort algorithm with partitioning.

**Problem:**

Implement quick sort to sort an array in ascending order.

Quick sort picks a pivot element, partitions the array around it, then recursively sorts the partitions.

**Examples:**

\`\`\`
Input: nums = [3, 6, 8, 10, 1, 2, 1]
Output: [1, 1, 2, 3, 6, 8, 10]

Input: nums = [5, 2, 3, 1]
Output: [1, 2, 3, 5]

Input: nums = [1]
Output: [1]
\`\`\`

**Visualization:**

\`\`\`
[3, 6, 8, 10, 1, 2, 1]  pivot = 3

After partition:
[1, 2, 1] [3] [6, 8, 10]
   <3      =3     >3

Recursively sort left and right partitions
\`\`\`

**Algorithm:**
1. Choose a pivot element
2. Partition: elements < pivot go left, > pivot go right
3. Recursively sort left and right partitions

**Constraints:**
- 1 <= nums.length <= 5 * 10^4
- -5 * 10^4 <= nums[i] <= 5 * 10^4

**Time Complexity:** O(n log n) average, O(n²) worst
**Space Complexity:** O(log n) for recursion stack`,
	initialCode: `from typing import List

def quick_sort(nums: List[int]) -> List[int]:
    # TODO: Sort array using quick sort algorithm

    return []`,
	solutionCode: `from typing import List
import random

def quick_sort(nums: List[int]) -> List[int]:
    """
    Sort array using quick sort.
    """
    def partition(lo: int, hi: int) -> int:
        """Lomuto partition scheme."""
        pivot = nums[hi]
        i = lo

        for j in range(lo, hi):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    def quick_sort_helper(lo: int, hi: int) -> None:
        if lo < hi:
            pivot_idx = partition(lo, hi)
            quick_sort_helper(lo, pivot_idx - 1)
            quick_sort_helper(pivot_idx + 1, hi)

    quick_sort_helper(0, len(nums) - 1)
    return nums


# Hoare partition (more efficient)
def quick_sort_hoare(nums: List[int]) -> List[int]:
    """Quick sort with Hoare partition scheme."""
    def partition(lo: int, hi: int) -> int:
        pivot = nums[lo + (hi - lo) // 2]
        i, j = lo - 1, hi + 1

        while True:
            i += 1
            while nums[i] < pivot:
                i += 1

            j -= 1
            while nums[j] > pivot:
                j -= 1

            if i >= j:
                return j

            nums[i], nums[j] = nums[j], nums[i]

    def sort(lo: int, hi: int) -> None:
        if lo < hi:
            p = partition(lo, hi)
            sort(lo, p)
            sort(p + 1, hi)

    sort(0, len(nums) - 1)
    return nums


# Randomized quick sort (better average case)
def quick_sort_randomized(nums: List[int]) -> List[int]:
    """Randomized pivot selection."""
    def partition(lo: int, hi: int) -> int:
        # Random pivot to avoid worst case
        rand_idx = random.randint(lo, hi)
        nums[rand_idx], nums[hi] = nums[hi], nums[rand_idx]

        pivot = nums[hi]
        i = lo

        for j in range(lo, hi):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    def sort(lo: int, hi: int) -> None:
        if lo < hi:
            p = partition(lo, hi)
            sort(lo, p - 1)
            sort(p + 1, hi)

    sort(0, len(nums) - 1)
    return nums


# Three-way partition (handles duplicates efficiently)
def quick_sort_3way(nums: List[int]) -> List[int]:
    """Dutch National Flag partitioning for duplicates."""
    def sort(lo: int, hi: int) -> None:
        if lo >= hi:
            return

        lt, gt = lo, hi
        pivot = nums[lo]
        i = lo + 1

        while i <= gt:
            if nums[i] < pivot:
                nums[lt], nums[i] = nums[i], nums[lt]
                lt += 1
                i += 1
            elif nums[i] > pivot:
                nums[i], nums[gt] = nums[gt], nums[i]
                gt -= 1
            else:
                i += 1

        sort(lo, lt - 1)
        sort(gt + 1, hi)

    sort(0, len(nums) - 1)
    return nums


# Quick select (kth smallest element)
def quick_select(nums: List[int], k: int) -> int:
    """Find kth smallest element using partition."""
    def partition(lo: int, hi: int) -> int:
        rand_idx = random.randint(lo, hi)
        nums[rand_idx], nums[hi] = nums[hi], nums[rand_idx]

        pivot = nums[hi]
        i = lo

        for j in range(lo, hi):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    k = k - 1  # Convert to 0-indexed
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        p = partition(lo, hi)
        if p == k:
            return nums[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1

    return nums[lo]`,
	testCode: `import pytest
from solution import quick_sort


class TestQuickSort:
    def test_basic_case(self):
        """Test basic sorting"""
        assert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]

    def test_simple(self):
        """Test simple case"""
        assert quick_sort([5, 2, 3, 1]) == [1, 2, 3, 5]

    def test_single_element(self):
        """Test single element"""
        assert quick_sort([1]) == [1]

    def test_empty_array(self):
        """Test empty array"""
        assert quick_sort([]) == []

    def test_already_sorted(self):
        """Test already sorted array"""
        assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        """Test reverse sorted array"""
        assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test with duplicates"""
        assert quick_sort([3, 3, 3, 1, 1, 2, 2]) == [1, 1, 2, 2, 3, 3, 3]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert quick_sort([-3, -1, -2, 0, 2, 1]) == [-3, -2, -1, 0, 1, 2]

    def test_all_same(self):
        """Test all same elements"""
        assert quick_sort([5, 5, 5, 5]) == [5, 5, 5, 5]

    def test_two_elements(self):
        """Test two elements"""
        assert quick_sort([2, 1]) == [1, 2]
        assert quick_sort([1, 2]) == [1, 2]

    def test_large_array(self):
        """Test larger array"""
        import random
        nums = random.sample(range(1000), 100)
        assert quick_sort(nums.copy()) == sorted(nums)`,
	hint1: `Choose a pivot element (typically last element for Lomuto partition). Partition the array so elements smaller than pivot are on the left, larger on the right.`,
	hint2: `In the partition function, maintain pointer i for the "boundary" of smaller elements. Swap when you find an element smaller than pivot. Finally, place pivot at its correct position.`,
	whyItMatters: `Quick Sort is one of the most widely used sorting algorithms due to its excellent average-case performance and in-place nature. Understanding partitioning is key to many interview problems.

**Why This Matters:**

**1. Partition Schemes**

\`\`\`python
# Lomuto (simpler, pivot at end):
# - One pointer, compare with pivot
# - Pivot ends up in final position

# Hoare (faster, pivot in middle):
# - Two pointers from both ends
# - More swaps but fewer comparisons

# Dutch National Flag (for duplicates):
# - Three regions: <pivot, =pivot, >pivot
# - O(n) even with many duplicates
\`\`\`

**2. Pivot Selection Strategies**

\`\`\`python
# Fixed (first/last): O(n²) on sorted input
# Random: Expected O(n log n)
# Median-of-three: Good practical choice
# Median-of-medians: Guaranteed O(n log n) but slower

# Randomized prevents adversarial inputs
pivot_idx = random.randint(lo, hi)
\`\`\`

**3. Quick Select (kth Element)**

\`\`\`python
# Use partition to find kth element in O(n) average
# Only recurse on one side (where k is)
# Useful for finding median, top-k, etc.
\`\`\`

**4. When to Use Quick Sort**

\`\`\`python
# Pros:
# - O(1) extra space (in-place)
# - Excellent cache performance
# - Fast in practice

# Cons:
# - O(n²) worst case
# - Not stable
# - Recursion overhead

# Use when:
# - Memory is limited
# - Average case matters more than worst case
# - Data fits in cache
\`\`\`

**5. Interview Applications**

\`\`\`python
# Kth largest/smallest element
# Top K frequent elements
# Sort colors (Dutch National Flag)
# Wiggle sort
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Быстрая сортировка',
			description: `Реализуйте алгоритм быстрой сортировки.

**Задача:**

Реализуйте quick sort для сортировки массива по возрастанию.

Быстрая сортировка выбирает опорный элемент, разделяет массив вокруг него, затем рекурсивно сортирует части.

**Примеры:**

\`\`\`
Вход: nums = [3, 6, 8, 10, 1, 2, 1]
Выход: [1, 1, 2, 3, 6, 8, 10]

Вход: nums = [5, 2, 3, 1]
Выход: [1, 2, 3, 5]
\`\`\`

**Алгоритм:**
1. Выбрать опорный элемент
2. Разделение: элементы < опорного слева, > справа
3. Рекурсивно отсортировать левую и правую части

**Ограничения:**
- 1 <= nums.length <= 5 * 10^4

**Временная сложность:** O(n log n) в среднем, O(n²) худший случай
**Пространственная сложность:** O(log n) для стека`,
			hint1: `Выберите опорный элемент. Разделите массив так, чтобы меньшие были слева, большие справа.`,
			hint2: `В partition поддерживайте указатель i для границы меньших элементов. Меняйте местами когда находите элемент меньше опорного.`,
			whyItMatters: `Quick Sort - один из самых используемых алгоритмов сортировки благодаря отличной средней производительности.

**Почему это важно:**

**1. Схемы разделения**

Lomuto (проще), Hoare (быстрее), Dutch National Flag (для дубликатов).

**2. Выбор опорного элемента**

Случайный выбор предотвращает худший случай.

**3. Quick Select**

Нахождение k-го элемента за O(n) в среднем.`,
			solutionCode: `from typing import List

def quick_sort(nums: List[int]) -> List[int]:
    """Сортирует массив быстрой сортировкой."""
    def partition(lo: int, hi: int) -> int:
        pivot = nums[hi]
        i = lo

        for j in range(lo, hi):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    def quick_sort_helper(lo: int, hi: int) -> None:
        if lo < hi:
            pivot_idx = partition(lo, hi)
            quick_sort_helper(lo, pivot_idx - 1)
            quick_sort_helper(pivot_idx + 1, hi)

    quick_sort_helper(0, len(nums) - 1)
    return nums`
		},
		uz: {
			title: 'Tez tartiblash',
			description: `Quick Sort algoritmini amalga oshiring.

**Masala:**

Massivni o'sish tartibida saralash uchun quick sort ni amalga oshiring.

Quick sort tayanch elementni tanlaydi, massivni uning atrofida bo'ladi, keyin qismlarni rekursiv saralaydi.

**Misollar:**

\`\`\`
Kirish: nums = [3, 6, 8, 10, 1, 2, 1]
Chiqish: [1, 1, 2, 3, 6, 8, 10]

Kirish: nums = [5, 2, 3, 1]
Chiqish: [1, 2, 3, 5]
\`\`\`

**Algoritm:**
1. Tayanch elementni tanlash
2. Bo'lish: tayanch dan kichiklar chapda, kattalar o'ngda
3. Chap va o'ng qismlarni rekursiv saralash

**Cheklovlar:**
- 1 <= nums.length <= 5 * 10^4

**Vaqt murakkabligi:** O(n log n) o'rtacha, O(n²) eng yomon
**Xotira murakkabligi:** O(log n) stek uchun`,
			hint1: `Tayanch elementni tanlang. Massivni kichiklar chapda, kattalar o'ngda bo'ladigan qilib bo'ling.`,
			hint2: `partition da kichik elementlar chegarasi uchun i ko'rsatkichini saqlang. Tayanchdan kichik element topganda almashtiring.`,
			whyItMatters: `Quick Sort - a'lo o'rtacha samaradorlik tufayli eng ko'p ishlatiladigan tartiblash algoritmlaridan biri.

**Bu nima uchun muhim:**

**1. Bo'lish sxemalari**

Lomuto (sodda), Hoare (tezroq), Dutch National Flag (dublikatlar uchun).

**2. Tayanch tanlash**

Tasodifiy tanlash eng yomon holatni oldini oladi.

**3. Quick Select**

k-chi elementni o'rtacha O(n) da topish.`,
			solutionCode: `from typing import List

def quick_sort(nums: List[int]) -> List[int]:
    """Massivni tez tartiblash bilan saralaydi."""
    def partition(lo: int, hi: int) -> int:
        pivot = nums[hi]
        i = lo

        for j in range(lo, hi):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    def quick_sort_helper(lo: int, hi: int) -> None:
        if lo < hi:
            pivot_idx = partition(lo, hi)
            quick_sort_helper(lo, pivot_idx - 1)
            quick_sort_helper(pivot_idx + 1, hi)

    quick_sort_helper(0, len(nums) - 1)
    return nums`
		}
	}
};

export default task;
