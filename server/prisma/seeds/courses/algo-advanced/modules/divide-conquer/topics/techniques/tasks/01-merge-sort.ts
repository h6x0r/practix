import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'divide-conquer-merge-sort',
	title: 'Merge Sort',
	difficulty: 'medium',
	tags: ['python', 'divide-conquer', 'sorting', 'recursion'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the classic Merge Sort algorithm.

**Problem:**

Implement the merge sort algorithm to sort an array in ascending order.

Merge sort divides the array into halves, recursively sorts each half, then merges the sorted halves.

**Examples:**

\`\`\`
Input: nums = [5, 2, 3, 1]
Output: [1, 2, 3, 5]

Input: nums = [5, 1, 1, 2, 0, 0]
Output: [0, 0, 1, 1, 2, 5]

Input: nums = [1]
Output: [1]
\`\`\`

**Visualization:**

\`\`\`
[5, 2, 3, 1]
      |
   divide
   /    \\
[5,2]  [3,1]
  |      |
divide  divide
 / \\    / \\
[5][2] [3][1]
  |      |
 merge  merge
  |      |
[2,5]  [1,3]
   \\    /
   merge
     |
[1,2,3,5]
\`\`\`

**Algorithm:**
1. **Divide**: Split array into two halves
2. **Conquer**: Recursively sort each half
3. **Combine**: Merge two sorted halves

**Constraints:**
- 1 <= nums.length <= 5 * 10^4
- -5 * 10^4 <= nums[i] <= 5 * 10^4

**Time Complexity:** O(n log n) - always
**Space Complexity:** O(n) for auxiliary array`,
	initialCode: `from typing import List

def merge_sort(nums: List[int]) -> List[int]:
    # TODO: Sort array using merge sort algorithm

    return []`,
	solutionCode: `from typing import List

def merge_sort(nums: List[int]) -> List[int]:
    """
    Sort array using merge sort.
    """
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


# In-place merge sort (space optimized)
def merge_sort_inplace(nums: List[int]) -> List[int]:
    """In-place merge sort using auxiliary array."""
    aux = [0] * len(nums)

    def merge_helper(lo: int, mid: int, hi: int) -> None:
        # Copy to auxiliary array
        for k in range(lo, hi + 1):
            aux[k] = nums[k]

        i, j = lo, mid + 1

        for k in range(lo, hi + 1):
            if i > mid:
                nums[k] = aux[j]
                j += 1
            elif j > hi:
                nums[k] = aux[i]
                i += 1
            elif aux[i] <= aux[j]:
                nums[k] = aux[i]
                i += 1
            else:
                nums[k] = aux[j]
                j += 1

    def sort(lo: int, hi: int) -> None:
        if hi <= lo:
            return
        mid = lo + (hi - lo) // 2
        sort(lo, mid)
        sort(mid + 1, hi)
        merge_helper(lo, mid, hi)

    sort(0, len(nums) - 1)
    return nums


# Bottom-up merge sort (iterative)
def merge_sort_bottom_up(nums: List[int]) -> List[int]:
    """Iterative bottom-up merge sort."""
    n = len(nums)
    aux = [0] * n

    def merge_bu(lo: int, mid: int, hi: int) -> None:
        for k in range(lo, hi + 1):
            aux[k] = nums[k]

        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            if i > mid:
                nums[k] = aux[j]
                j += 1
            elif j > hi:
                nums[k] = aux[i]
                i += 1
            elif aux[i] <= aux[j]:
                nums[k] = aux[i]
                i += 1
            else:
                nums[k] = aux[j]
                j += 1

    size = 1
    while size < n:
        lo = 0
        while lo < n - size:
            merge_bu(lo, lo + size - 1, min(lo + 2 * size - 1, n - 1))
            lo += 2 * size
        size *= 2

    return nums


# Count inversions using merge sort
def count_inversions(nums: List[int]) -> int:
    """Count inversions in array using merge sort."""
    inversions = [0]

    def merge_count(left: List[int], right: List[int]) -> List[int]:
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
                inversions[0] += len(left) - i

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def sort_count(arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = sort_count(arr[:mid])
        right = sort_count(arr[mid:])
        return merge_count(left, right)

    sort_count(nums.copy())
    return inversions[0]`,
	testCode: `import pytest
from solution import merge_sort


class TestMergeSort:
    def test_basic_case(self):
        """Test basic sorting"""
        assert merge_sort([5, 2, 3, 1]) == [1, 2, 3, 5]

    def test_with_duplicates(self):
        """Test with duplicate values"""
        assert merge_sort([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]

    def test_single_element(self):
        """Test single element"""
        assert merge_sort([1]) == [1]

    def test_empty_array(self):
        """Test empty array"""
        assert merge_sort([]) == []

    def test_already_sorted(self):
        """Test already sorted array"""
        assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        """Test reverse sorted array"""
        assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert merge_sort([-3, -1, -2, 0, 2, 1]) == [-3, -2, -1, 0, 1, 2]

    def test_all_same(self):
        """Test all same elements"""
        assert merge_sort([5, 5, 5, 5]) == [5, 5, 5, 5]

    def test_two_elements(self):
        """Test two elements"""
        assert merge_sort([2, 1]) == [1, 2]
        assert merge_sort([1, 2]) == [1, 2]

    def test_stability(self):
        """Test that equal elements maintain relative order"""
        # Merge sort is stable - equal elements stay in original order
        result = merge_sort([3, 1, 4, 1, 5, 9, 2, 6])
        assert result == sorted([3, 1, 4, 1, 5, 9, 2, 6])

    def test_large_array(self):
        """Test larger array"""
        import random
        nums = random.sample(range(1000), 100)
        assert merge_sort(nums.copy()) == sorted(nums)`,
	hint1: `Divide the array into two halves at the midpoint. Recursively sort each half. Base case: arrays of length 0 or 1 are already sorted.`,
	hint2: `The merge function uses two pointers to compare elements from left and right arrays. Always take the smaller element. Don't forget to add remaining elements after one array is exhausted.`,
	whyItMatters: `Merge Sort is a fundamental divide-and-conquer algorithm with guaranteed O(n log n) performance. It's stable, parallelizable, and forms the basis for external sorting and many other algorithms.

**Why This Matters:**

**1. Divide and Conquer Paradigm**

\`\`\`python
# Classic D&C structure:
def divide_and_conquer(problem):
    # Base case
    if is_simple(problem):
        return solve_directly(problem)

    # Divide
    subproblems = divide(problem)

    # Conquer
    subsolutions = [divide_and_conquer(p) for p in subproblems]

    # Combine
    return combine(subsolutions)
\`\`\`

**2. Time Complexity Analysis**

\`\`\`python
# Recurrence: T(n) = 2T(n/2) + O(n)

# Master theorem: a=2, b=2, f(n)=n
# n^(log_b(a)) = n^1 = n
# f(n) = Theta(n^1)
# Case 2: T(n) = O(n log n)

# Always O(n log n) - best, average, worst!
\`\`\`

**3. Stability Property**

\`\`\`python
# Merge sort is stable - equal elements maintain relative order
# Important for:
# - Sorting by multiple keys
# - Database operations
# - Linked list sorting (O(1) space possible)

# Key: use <= not < in merge comparison
if left[i] <= right[j]:  # Stable
if left[i] < right[j]:   # NOT stable
\`\`\`

**4. Applications**

\`\`\`python
# Count inversions (modified merge)
# External sorting (huge files)
# Linked list sorting
# Parallel sorting (natural division)
# Tim Sort (hybrid with insertion sort)
\`\`\`

**5. Comparison with Quick Sort**

\`\`\`python
# Merge Sort:
# - O(n log n) always
# - O(n) extra space
# - Stable
# - Good for linked lists

# Quick Sort:
# - O(n log n) average, O(n²) worst
# - O(log n) space (in-place)
# - Not stable
# - Better cache performance
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Сортировка слиянием',
			description: `Реализуйте классический алгоритм сортировки слиянием.

**Задача:**

Реализуйте алгоритм merge sort для сортировки массива по возрастанию.

Сортировка слиянием делит массив пополам, рекурсивно сортирует каждую половину, затем сливает отсортированные части.

**Примеры:**

\`\`\`
Вход: nums = [5, 2, 3, 1]
Выход: [1, 2, 3, 5]

Вход: nums = [5, 1, 1, 2, 0, 0]
Выход: [0, 0, 1, 1, 2, 5]
\`\`\`

**Алгоритм:**
1. **Разделение**: Разбить массив на две половины
2. **Завоевание**: Рекурсивно отсортировать каждую
3. **Объединение**: Слить две отсортированные половины

**Ограничения:**
- 1 <= nums.length <= 5 * 10^4

**Временная сложность:** O(n log n) - всегда
**Пространственная сложность:** O(n)`,
			hint1: `Разделите массив на две половины. Рекурсивно отсортируйте каждую. Базовый случай: массивы длины 0 или 1 уже отсортированы.`,
			hint2: `Функция merge использует два указателя для сравнения. Всегда берите меньший элемент. Не забудьте добавить оставшиеся элементы.`,
			whyItMatters: `Merge Sort - фундаментальный алгоритм "разделяй и властвуй" с гарантированной O(n log n) производительностью.

**Почему это важно:**

**1. Парадигма "разделяй и властвуй"**

Классическая структура D&C: разделить, завоевать, объединить.

**2. Анализ сложности**

T(n) = 2T(n/2) + O(n) → O(n log n) всегда.

**3. Стабильность**

Равные элементы сохраняют относительный порядок.`,
			solutionCode: `from typing import List

def merge_sort(nums: List[int]) -> List[int]:
    """Сортирует массив методом слияния."""
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Сливает два отсортированных массива."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result`
		},
		uz: {
			title: 'Birlashtirish tartiblash',
			description: `Klassik Merge Sort algoritmini amalga oshiring.

**Masala:**

Massivni o'sish tartibida saralash uchun merge sort algoritmini amalga oshiring.

Merge sort massivni ikkiga bo'ladi, har birini rekursiv saralaydi, keyin saralangan qismlarni birlashtiradi.

**Misollar:**

\`\`\`
Kirish: nums = [5, 2, 3, 1]
Chiqish: [1, 2, 3, 5]

Kirish: nums = [5, 1, 1, 2, 0, 0]
Chiqish: [0, 0, 1, 1, 2, 5]
\`\`\`

**Algoritm:**
1. **Bo'lish**: Massivni ikkiga ajrating
2. **Zabt etish**: Har birini rekursiv saralang
3. **Birlashtirish**: Saralangan qismlarni qo'shing

**Cheklovlar:**
- 1 <= nums.length <= 5 * 10^4

**Vaqt murakkabligi:** O(n log n) - har doim
**Xotira murakkabligi:** O(n)`,
			hint1: `Massivni ikkiga bo'ling. Har birini rekursiv saralang. Asosiy holat: 0 yoki 1 uzunlikdagi massivlar saralangan.`,
			hint2: `merge funksiyasi solishtirish uchun ikkita ko'rsatkich ishlatadi. Har doim kichik elementni oling. Qolgan elementlarni qo'shishni unutmang.`,
			whyItMatters: `Merge Sort - kafolatlangan O(n log n) samaradorlikka ega asosiy "bo'l va hukmronlik qil" algoritmi.

**Bu nima uchun muhim:**

**1. "Bo'l va hukmronlik qil" paradigmasi**

Klassik D&C tuzilishi: bo'lish, zabt etish, birlashtirish.

**2. Murakkablik tahlili**

T(n) = 2T(n/2) + O(n) → O(n log n) har doim.

**3. Barqarorlik**

Teng elementlar nisbiy tartibni saqlaydi.`,
			solutionCode: `from typing import List

def merge_sort(nums: List[int]) -> List[int]:
    """Massivni birlashtirish usuli bilan saralaydi."""
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Ikki saralangan massivni birlashtiradi."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result`
		}
	}
};

export default task;
