import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-binary-search',
	title: 'Binary Search',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'algorithms', 'search'],
	estimatedTime: '15m',
	isPremium: false,
	order: 10,

	description: `# Binary Search

Binary search is an efficient algorithm for finding an item in a **sorted** list.

## Task

Implement the function \`binary_search(arr, target)\` that finds the index of a target value in a sorted array.

## Requirements

- The input array is sorted in ascending order
- Return the index of the target if found
- Return -1 if the target is not in the array
- The algorithm should be O(log n)

## Examples

\`\`\`python
>>> binary_search([1, 3, 5, 7, 9], 5)
2

>>> binary_search([1, 3, 5, 7, 9], 6)
-1

>>> binary_search([2, 4, 6, 8, 10], 2)
0

>>> binary_search([], 5)
-1
\`\`\``,

	initialCode: `def binary_search(arr: list[int], target: int) -> int:
    """Find the index of target in a sorted array using binary search.

    Binary search works by repeatedly dividing the search space in half.

    Args:
        arr: Sorted list of integers (ascending order)
        target: Value to search for

    Returns:
        Index of target if found, -1 otherwise
    """
    # TODO: Implement binary search
    pass`,

	solutionCode: `def binary_search(arr: list[int], target: int) -> int:
    """Find the index of target in a sorted array using binary search.

    Binary search works by repeatedly dividing the search space in half.

    Args:
        arr: Sorted list of integers (ascending order)
        target: Value to search for

    Returns:
        Index of target if found, -1 otherwise
    """
    # Initialize search boundaries
    left = 0
    right = len(arr) - 1

    # Continue while search space is valid
    while left <= right:
        # Calculate middle index (avoid integer overflow)
        mid = left + (right - left) // 2

        # Check if target is at middle
        if arr[mid] == target:
            return mid

        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1

        # If target is smaller, ignore right half
        else:
            right = mid - 1

    # Target not found
    return -1`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Target in middle"""
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 5), 2)

    def test_2(self):
        """Target not found"""
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 6), -1)

    def test_3(self):
        """Target at start"""
        self.assertEqual(binary_search([2, 4, 6, 8, 10], 2), 0)

    def test_4(self):
        """Empty array"""
        self.assertEqual(binary_search([], 5), -1)

    def test_5(self):
        """Target at end"""
        self.assertEqual(binary_search([1, 2, 3, 4, 5], 5), 4)

    def test_6(self):
        """Single element found"""
        self.assertEqual(binary_search([42], 42), 0)

    def test_7(self):
        """Single element not found"""
        self.assertEqual(binary_search([42], 10), -1)

    def test_8(self):
        """Two elements, find first"""
        self.assertEqual(binary_search([1, 2], 1), 0)

    def test_9(self):
        """Two elements, find second"""
        self.assertEqual(binary_search([1, 2], 2), 1)

    def test_10(self):
        """Large array"""
        arr = list(range(0, 1000, 2))  # Even numbers 0-998
        self.assertEqual(binary_search(arr, 500), 250)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use two pointers: left starting at 0, right at len(arr)-1. Calculate mid = (left + right) // 2.',
	hint2: 'If arr[mid] < target, search the right half (left = mid + 1). If arr[mid] > target, search left half (right = mid - 1).',

	whyItMatters: `Binary search is a fundamental algorithm that reduces O(n) search to O(log n).

**Production Pattern:**

\`\`\`python
import bisect

def find_insertion_point(sorted_list: list, value) -> int:
    """Find where to insert value to keep list sorted."""
    return bisect.bisect_left(sorted_list, value)

def binary_search_first(arr: list[int], target: int) -> int:
    """Find the first occurrence of target (for duplicates)."""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(arr) and arr[left] == target else -1

def search_rotated_array(arr: list[int], target: int) -> int:
    """Search in rotated sorted array."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
\`\`\`

**Practical Benefits:**
- Database indexes use binary search trees
- Python's bisect module provides optimized binary search
- Foundation for more complex search algorithms`,

	translations: {
		ru: {
			title: 'Бинарный поиск',
			description: `# Бинарный поиск

Бинарный поиск — эффективный алгоритм поиска элемента в **отсортированном** списке.

## Задача

Реализуйте функцию \`binary_search(arr, target)\`, которая находит индекс целевого значения в отсортированном массиве.

## Требования

- Входной массив отсортирован по возрастанию
- Верните индекс цели, если она найдена
- Верните -1, если цель отсутствует в массиве
- Алгоритм должен иметь сложность O(log n)

## Примеры

\`\`\`python
>>> binary_search([1, 3, 5, 7, 9], 5)
2

>>> binary_search([1, 3, 5, 7, 9], 6)
-1

>>> binary_search([2, 4, 6, 8, 10], 2)
0

>>> binary_search([], 5)
-1
\`\`\``,
			hint1: 'Используйте два указателя: left = 0, right = len(arr)-1. Вычислите mid = (left + right) // 2.',
			hint2: 'Если arr[mid] < target — ищите в правой половине (left = mid + 1). Если arr[mid] > target — в левой (right = mid - 1).',
			whyItMatters: `Бинарный поиск — фундаментальный алгоритм, снижающий сложность с O(n) до O(log n).

**Продакшен паттерн:**

\`\`\`python
import bisect

def find_insertion_point(sorted_list: list, value) -> int:
    """Найти позицию для вставки значения с сохранением сортировки."""
    return bisect.bisect_left(sorted_list, value)

def binary_search_first(arr: list[int], target: int) -> int:
    """Найти первое вхождение цели (для дубликатов)."""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(arr) and arr[left] == target else -1
\`\`\`

**Практические преимущества:**
- Индексы баз данных используют деревья бинарного поиска
- Модуль bisect в Python предоставляет оптимизированный бинарный поиск`,
		},
		uz: {
			title: 'Binar qidiruv',
			description: `# Binar qidiruv

Binar qidiruv — **tartiblangan** ro'yxatda element topish uchun samarali algoritm.

## Vazifa

Tartiblangan massivda maqsad qiymat indeksini topuvchi \`binary_search(arr, target)\` funksiyasini amalga oshiring.

## Talablar

- Kirish massivi o'sish tartibida tartiblangan
- Maqsad topilsa, uning indeksini qaytaring
- Maqsad massivda yo'q bo'lsa, -1 qaytaring
- Algoritm O(log n) bo'lishi kerak

## Misollar

\`\`\`python
>>> binary_search([1, 3, 5, 7, 9], 5)
2

>>> binary_search([1, 3, 5, 7, 9], 6)
-1

>>> binary_search([2, 4, 6, 8, 10], 2)
0

>>> binary_search([], 5)
-1
\`\`\``,
			hint1: "Ikki ko'rsatkich ishlating: left = 0, right = len(arr)-1. mid = (left + right) // 2 hisoblang.",
			hint2: "Agar arr[mid] < target — o'ng yarmida qidiring (left = mid + 1). Agar arr[mid] > target — chap yarmida (right = mid - 1).",
			whyItMatters: `Binar qidiruv — O(n) qidiruvni O(log n) ga kamaytiradigan asosiy algoritm.

**Ishlab chiqarish patterni:**

\`\`\`python
import bisect

def find_insertion_point(sorted_list: list, value) -> int:
    """Tartibni saqlash uchun qiymatni qo'yish joyini topish."""
    return bisect.bisect_left(sorted_list, value)

def binary_search_first(arr: list[int], target: int) -> int:
    """Maqsadning birinchi uchrashuvini topish (dublikatlar uchun)."""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(arr) and arr[left] == target else -1
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar bazasi indekslari binar qidiruv daraxtlaridan foydalanadi
- Python ning bisect moduli optimallashtirilgan binar qidiruvni taqdim etadi`,
		},
	},
};

export default task;
