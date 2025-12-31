import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-selection-sort',
	title: 'Selection Sort',
	difficulty: 'easy',
	tags: ['python', 'sorting', 'array', 'comparison'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the selection sort algorithm.

**Problem:**

Given an array of integers, sort it in ascending order using selection sort.

**How Selection Sort Works:**

Divide the array into sorted and unsorted parts. Repeatedly find the minimum element from the unsorted part and move it to the end of the sorted part.

**Examples:**

\`\`\`
Input: [64, 25, 12, 22, 11]
Output: [11, 12, 22, 25, 64]

Step by step:
[64, 25, 12, 22, 11] - Find min (11), swap with first
[11, 25, 12, 22, 64] - Find min in rest (12), swap with second
[11, 12, 25, 22, 64] - Find min in rest (22), swap with third
[11, 12, 22, 25, 64] - Find min in rest (25), already in place
[11, 12, 22, 25, 64] - Done
\`\`\`

**Key Properties:**

- Makes minimum number of swaps: O(n)
- Always O(n^2) comparisons regardless of input
- Not stable (relative order of equal elements may change)

**Time Complexity:** O(n^2) always
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def selection_sort(arr: List[int]) -> List[int]:
    # TODO: Implement selection sort algorithm

    return arr`,
	solutionCode: `from typing import List


def selection_sort(arr: List[int]) -> List[int]:
    """
    Sort array using selection sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list in ascending order
    """
    n = len(arr)

    for i in range(n - 1):
        # Find minimum element in unsorted part
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Swap minimum with first unsorted element
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr`,
	testCode: `import pytest
from solution import selection_sort


class TestSelectionSort:
    def test_normal(self):
        """Test normal unsorted array"""
        arr = [64, 25, 12, 22, 11]
        assert selection_sort(arr.copy()) == [11, 12, 22, 25, 64]

    def test_already_sorted(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        assert selection_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        assert selection_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test array with duplicates"""
        arr = [3, 1, 4, 1, 5]
        assert selection_sort(arr.copy()) == [1, 1, 3, 4, 5]

    def test_single(self):
        """Test single element"""
        assert selection_sort([1]) == [1]

    def test_empty(self):
        """Test empty array"""
        assert selection_sort([]) == []

    def test_two_elements(self):
        """Test two elements"""
        assert selection_sort([2, 1]) == [1, 2]

    def test_negatives(self):
        """Test array with negatives"""
        arr = [-5, 3, -1, 0, 2]
        assert selection_sort(arr.copy()) == [-5, -1, 0, 2, 3]

    def test_large_array(self):
        """Test larger array"""
        arr = [64, 25, 12, 22, 11, 90, 88, 45, 50, 34]
        assert selection_sort(arr.copy()) == [11, 12, 22, 25, 34, 45, 50, 64, 88, 90]

    def test_all_negative(self):
        """Test array with all negative numbers"""
        arr = [-10, -5, -20, -1, -15]
        assert selection_sort(arr.copy()) == [-20, -15, -10, -5, -1]`,
	hint1: `For each position i, find the index of minimum element in the range [i, n-1], then swap.`,
	hint2: `Track minIdx instead of minValue. This way you can easily swap arr[i] and arr[minIdx].`,
	whyItMatters: `Selection sort minimizes swaps and helps understand the sorted/unsorted partition pattern.

**Why This Matters:**

**1. Minimum Swaps**

Selection sort makes at most n-1 swaps:
\`\`\`python
# Each iteration makes exactly 0 or 1 swap
if min_idx != i:
    arr[i], arr[min_idx] = arr[min_idx], arr[i]
\`\`\`

Useful when writes are expensive (e.g., flash memory).

**2. Sorted/Unsorted Partition**

This pattern appears in many algorithms:
\`\`\`python
# [sorted part | unsorted part]
# Each iteration moves boundary right
for i in range(n - 1):
    # arr[0:i] is sorted
    # arr[i:n] is unsorted
\`\`\`

**3. Finding Minimum Pattern**

\`\`\`python
min_idx = i
for j in range(i + 1, n):
    if arr[j] < arr[min_idx]:
        min_idx = j
# Now arr[min_idx] is minimum in range
\`\`\`

**4. Stability**

Selection sort is NOT stable:
\`\`\`
[5a, 3, 5b, 2] - two 5s
After sorting: [2, 3, 5b, 5a] - order changed!
\`\`\`

**5. When to Use**

- When minimizing swaps matters
- Educational purposes
- Small arrays
- Never for large datasets`,
	order: 2,
	translations: {
		ru: {
			title: 'Сортировка выбором',
			description: `Реализуйте алгоритм сортировки выбором.

**Задача:**

Дан массив целых чисел, отсортируйте его по возрастанию используя сортировку выбором.

**Как работает сортировка выбором:**

Разделите массив на отсортированную и неотсортированную части. Многократно находите минимальный элемент в неотсортированной части и перемещайте его в конец отсортированной части.

**Примеры:**

\`\`\`
Вход: [64, 25, 12, 22, 11]
Выход: [11, 12, 22, 25, 64]

Пошагово:
[64, 25, 12, 22, 11] - Найти мин (11), обменять с первым
[11, 25, 12, 22, 64] - Найти мин в остатке (12), обменять со вторым
[11, 12, 25, 22, 64] - Найти мин в остатке (22), обменять с третьим
[11, 12, 22, 25, 64] - Готово
\`\`\`

**Временная сложность:** O(n^2) всегда
**Пространственная сложность:** O(1)`,
			hint1: `Для каждой позиции i найдите индекс минимального элемента в диапазоне [i, n-1], затем обменяйте.`,
			hint2: `Отслеживайте minIdx вместо minValue. Так вы легко обменяете arr[i] и arr[minIdx].`,
			whyItMatters: `Сортировка выбором минимизирует обмены и помогает понять паттерн разделения на отсортированную/неотсортированную части.

**Почему это важно:**

**1. Минимум обменов**

Сортировка выбором делает максимум n-1 обменов.

**2. Паттерн разделения**

Этот паттерн встречается во многих алгоритмах.

**3. Паттерн поиска минимума**

Базовый алгоритм поиска минимума в диапазоне.`,
			solutionCode: `from typing import List


def selection_sort(arr: List[int]) -> List[int]:
    """
    Сортирует массив сортировкой выбором.

    Args:
        arr: Список целых чисел для сортировки

    Returns:
        Отсортированный список по возрастанию
    """
    n = len(arr)

    for i in range(n - 1):
        # Найти минимальный элемент в неотсортированной части
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Обменять минимум с первым неотсортированным элементом
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr`
		},
		uz: {
			title: 'Tanlash saralashi',
			description: `Tanlash saralash algoritmini amalga oshiring.

**Masala:**

Butun sonlar massivi berilgan, uni tanlash saralash yordamida o'sish tartibida saralang.

**Tanlash saralash qanday ishlaydi:**

Massivni saralangan va saralanmagan qismlarga bo'ling. Saralanmagan qismda minimal elementni toping va saralangan qism oxiriga o'tkazing.

**Misollar:**

\`\`\`
Kirish: [64, 25, 12, 22, 11]
Chiqish: [11, 12, 22, 25, 64]

Qadamma-qadam:
[64, 25, 12, 22, 11] - Min topish (11), birinchi bilan almashtirish
[11, 25, 12, 22, 64] - Qolganida min topish (12), ikkinchi bilan almashtirish
[11, 12, 25, 22, 64] - Qolganida min topish (22), uchinchi bilan almashtirish
[11, 12, 22, 25, 64] - Tayyor
\`\`\`

**Vaqt murakkabligi:** O(n^2) doimo
**Xotira murakkabligi:** O(1)`,
			hint1: `Har bir i pozitsiyasi uchun [i, n-1] diapazonidagi minimal element indeksini toping, keyin almashtiring.`,
			hint2: `minValue o'rniga minIdx ni kuzating. Shunday qilib arr[i] va arr[minIdx] ni oson almashtirasiz.`,
			whyItMatters: `Tanlash saralash almashishlarni minimallashtiradi va saralangan/saralanmagan bo'linish patternini tushunishga yordam beradi.

**Bu nima uchun muhim:**

**1. Minimal almashishlar**

Tanlash saralash maksimal n-1 ta almashtirish qiladi.

**2. Bo'linish patterni**

Bu pattern ko'plab algoritmlarda uchraydi.

**3. Minimum topish patterni**

Diapazonda minimum topish asosiy algoritmi.`,
			solutionCode: `from typing import List


def selection_sort(arr: List[int]) -> List[int]:
    """
    Massivni tanlash saralash algoritmi bilan saralaydi.

    Args:
        arr: Saralash uchun butun sonlar ro'yxati

    Returns:
        O'sish tartibida saralangan ro'yxat
    """
    n = len(arr)

    for i in range(n - 1):
        # Saralanmagan qismda minimal elementni topish
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Minimumni birinchi saralanmagan element bilan almashtirish
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr`
		}
	}
};

export default task;
