import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-quick-sort',
	title: 'Quick Sort',
	difficulty: 'medium',
	tags: ['python', 'sorting', 'array', 'divide-conquer', 'recursion'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the quick sort algorithm.

**Problem:**

Given an array of integers, sort it in ascending order using quick sort.

**How Quick Sort Works:**

Quick sort uses divide-and-conquer with partitioning:

1. **Choose pivot:** Select an element as pivot (often last element)
2. **Partition:** Rearrange so elements < pivot are left, elements > pivot are right
3. **Recurse:** Apply quick sort to left and right partitions

**Examples:**

\`\`\`
Input: [10, 7, 8, 9, 1, 5]
Output: [1, 5, 7, 8, 9, 10]

Partition with pivot = 5:
[10, 7, 8, 9, 1, 5]
        |
[1] [5] [10, 7, 8, 9]  <- 1 < 5, pivot in middle, rest > 5

Recurse on [1] (done) and [10, 7, 8, 9]:
Pivot = 9:
[10, 7, 8, 9]
[7, 8] [9] [10]

Continue until sorted...
\`\`\`

**Key Properties:**

- In-place sorting (O(log n) extra space for recursion)
- Not stable
- Very fast in practice

**Time Complexity:** O(n log n) average, O(n^2) worst case
**Space Complexity:** O(log n) average (stack)`,
	initialCode: `from typing import List

def quick_sort(arr: List[int]) -> List[int]:
    # TODO: Implement quick sort algorithm

    return arr


def _quick_sort_helper(arr: List[int], low: int, high: int) -> None:
    # TODO: Recursive helper for quick sort
    pass


def _partition(arr: List[int], low: int, high: int) -> int:
    # TODO: Partition array around pivot and return pivot index

    return 0`,
	solutionCode: `from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    """
    Sort array using quick sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list in ascending order
    """
    if len(arr) <= 1:
        return arr
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr: List[int], low: int, high: int) -> None:
    """Helper function for recursive quick sort."""
    if low < high:
        # Partition and get pivot index
        pivot_idx = _partition(arr, low, high)

        # Recursively sort left and right partitions
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _partition(arr: List[int], low: int, high: int) -> int:
    """Partition array and return pivot index."""
    pivot = arr[high]  # Choose last element as pivot
    i = low  # Index of smaller element boundary

    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Place pivot at correct position
    arr[i], arr[high] = arr[high], arr[i]
    return i`,
	testCode: `import pytest
from solution import quick_sort


class TestQuickSort:
    def test_normal(self):
        """Test normal unsorted array"""
        arr = [10, 7, 8, 9, 1, 5]
        assert quick_sort(arr) == [1, 5, 7, 8, 9, 10]

    def test_already_sorted(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        assert quick_sort(arr) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        assert quick_sort(arr) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test array with duplicates"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        assert quick_sort(arr) == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_single(self):
        """Test single element"""
        assert quick_sort([1]) == [1]

    def test_empty(self):
        """Test empty array"""
        assert quick_sort([]) == []

    def test_two_elements(self):
        """Test two elements"""
        assert quick_sort([2, 1]) == [1, 2]

    def test_negatives(self):
        """Test array with negatives"""
        arr = [-5, 3, -1, 0, 2]
        assert quick_sort(arr) == [-5, -1, 0, 2, 3]

    def test_all_same(self):
        """Test array with all same elements"""
        arr = [5, 5, 5, 5]
        assert quick_sort(arr) == [5, 5, 5, 5]

    def test_large_array(self):
        """Test larger array"""
        arr = [10, 7, 8, 9, 1, 5, 15, 3, 12, 20]
        assert quick_sort(arr) == [1, 3, 5, 7, 8, 9, 10, 12, 15, 20]`,
	hint1: `Use Lomuto partition scheme: choose last element as pivot, maintain index i as boundary of elements less than pivot.`,
	hint2: `In partition, scan with j from low to high-1. If arr[j] < pivot, swap arr[i] and arr[j], then increment i. Finally swap pivot to position i.`,
	whyItMatters: `Quick sort is one of the most important sorting algorithms in practice.

**Why This Matters:**

**1. Fastest in Practice**

Despite O(n^2) worst case, quick sort is often fastest:
\`\`\`python
# Good cache locality
# In-place (no extra arrays)
# Small constant factors
\`\`\`

**2. The Partition Operation**

The key insight of quick sort:
\`\`\`python
# After partition:
# - All elements < pivot are on left
# - Pivot is in correct final position
# - All elements > pivot are on right
\`\`\`

**3. Pivot Selection Strategies**

\`\`\`python
import random

# Bad: always first/last (O(n^2) on sorted arrays)
pivot = arr[high]

# Better: median of three
mid = (low + high) // 2
# Choose median of arr[low], arr[mid], arr[high]

# Best: random
pivot = arr[random.randint(low, high)]
\`\`\`

**4. Avoiding Worst Case**

\`\`\`python
# Worst case: already sorted + bad pivot
# [1, 2, 3, 4, 5] with pivot = 5
# Partition: [1, 2, 3, 4] | 5 | []
# T(n) = T(n-1) + O(n) = O(n^2)

# Solution: randomized pivot or median-of-three
\`\`\`

**5. Three-Way Partition (for duplicates)**

\`\`\`python
# Dutch National Flag algorithm
# Partition into: [< pivot] [= pivot] [> pivot]
# Handles duplicates efficiently
\`\`\`

**6. Real-World Usage**

- C qsort, C++ std::sort
- Python's sorted() uses Timsort (hybrid)
- When cache efficiency matters`,
	order: 5,
	translations: {
		ru: {
			title: 'Быстрая сортировка',
			description: `Реализуйте алгоритм быстрой сортировки.

**Задача:**

Дан массив целых чисел, отсортируйте его по возрастанию используя быструю сортировку.

**Как работает быстрая сортировка:**

Быстрая сортировка использует "разделяй и властвуй" с разбиением:

1. **Выбор опорного элемента:** Выберите элемент как опорный (часто последний)
2. **Разбиение:** Перестройте так, чтобы элементы < опорного слева, > опорного справа
3. **Рекурсия:** Применить быструю сортировку к левой и правой частям

**Примеры:**

\`\`\`
Вход: [10, 7, 8, 9, 1, 5]
Выход: [1, 5, 7, 8, 9, 10]
\`\`\`

**Ключевые свойства:**

- Сортировка на месте (O(log n) доп. памяти для рекурсии)
- Не стабильная
- Очень быстрая на практике

**Временная сложность:** O(n log n) средняя, O(n^2) худшая
**Пространственная сложность:** O(log n) средняя (стек)`,
			hint1: `Используйте схему Ломуто: выберите последний элемент как опорный, поддерживайте индекс i как границу элементов меньше опорного.`,
			hint2: `В partition сканируйте j от low до high-1. Если arr[j] < pivot, обменяйте arr[i] и arr[j], затем увеличьте i. В конце обменяйте pivot на позицию i.`,
			whyItMatters: `Быстрая сортировка - один из важнейших алгоритмов сортировки на практике.

**Почему это важно:**

**1. Самая быстрая на практике**

Несмотря на O(n^2) в худшем случае, часто самая быстрая.

**2. Операция разбиения**

Ключевая идея быстрой сортировки.

**3. Стратегии выбора опорного**

Плохой выбор: всегда первый/последний.
Лучше: медиана трёх или случайный.

**4. Использование**

C qsort, C++ std::sort, Go sort.Slice.`,
			solutionCode: `from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    """
    Сортирует массив быстрой сортировкой.

    Args:
        arr: Список целых чисел для сортировки

    Returns:
        Отсортированный список по возрастанию
    """
    if len(arr) <= 1:
        return arr
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr: List[int], low: int, high: int) -> None:
    """Вспомогательная функция для рекурсивной быстрой сортировки."""
    if low < high:
        # Разбиение и получение индекса опорного
        pivot_idx = _partition(arr, low, high)

        # Рекурсивно сортируем левую и правую части
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _partition(arr: List[int], low: int, high: int) -> int:
    """Разбиение массива и возврат индекса опорного."""
    pivot = arr[high]  # Выбираем последний элемент как опорный
    i = low  # Индекс границы меньших элементов

    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Помещаем опорный на правильную позицию
    arr[i], arr[high] = arr[high], arr[i]
    return i`
		},
		uz: {
			title: 'Tez saralash',
			description: `Tez saralash algoritmini amalga oshiring.

**Masala:**

Butun sonlar massivi berilgan, uni tez saralash yordamida o'sish tartibida saralang.

**Tez saralash qanday ishlaydi:**

Tez saralash bo'linish bilan "bo'l va hukmronlik qil" dan foydalanadi:

1. **Pivot tanlash:** Elementni pivot sifatida tanlang (ko'pincha oxirgi element)
2. **Bo'linish:** < pivot elementlar chapda, > pivot elementlar o'ngda bo'ladigan qilib qayta tartibga keltiring
3. **Rekursiya:** Chap va o'ng bo'linmalarga tez saralashni qo'llang

**Misollar:**

\`\`\`
Kirish: [10, 7, 8, 9, 1, 5]
Chiqish: [1, 5, 7, 8, 9, 10]
\`\`\`

**Asosiy xususiyatlar:**

- Joyida saralash (rekursiya uchun O(log n) qo'shimcha xotira)
- Barqaror emas
- Amalda juda tez

**Vaqt murakkabligi:** O(n log n) o'rtacha, O(n^2) eng yomon
**Xotira murakkabligi:** O(log n) o'rtacha (stek)`,
			hint1: `Lomuto bo'linish sxemasidan foydalaning: oxirgi elementni pivot sifatida tanlang, pivotdan kichik elementlar chegarasi sifatida i indeksini saqlang.`,
			hint2: `Partition da j ni low dan high-1 gacha skanerlang. Agar arr[j] < pivot bo'lsa, arr[i] va arr[j] ni almashtiring, keyin i ni oshiring. Oxirida pivotni i pozitsiyasiga almashtiring.`,
			whyItMatters: `Tez saralash amalda eng muhim saralash algoritmlaridan biri.

**Bu nima uchun muhim:**

**1. Amalda eng tez**

O(n^2) eng yomon holatga qaramay, ko'pincha eng tez.

**2. Bo'linish operatsiyasi**

Tez saralashning asosiy g'oyasi.

**3. Pivot tanlash strategiyalari**

Yomon tanlov: doimo birinchi/oxirgi.
Yaxshiroq: uchtaning medianasi yoki tasodifiy.

**4. Foydalanish**

C qsort, C++ std::sort, Go sort.Slice.`,
			solutionCode: `from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    """
    Massivni tez saralash algoritmi bilan saralaydi.

    Args:
        arr: Saralash uchun butun sonlar ro'yxati

    Returns:
        O'sish tartibida saralangan ro'yxat
    """
    if len(arr) <= 1:
        return arr
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr: List[int], low: int, high: int) -> None:
    """Rekursiv tez saralash uchun yordamchi funktsiya."""
    if low < high:
        # Bo'linish va pivot indeksini olish
        pivot_idx = _partition(arr, low, high)

        # Chap va o'ng bo'linmalarni rekursiv saralash
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _partition(arr: List[int], low: int, high: int) -> int:
    """Massivni bo'lish va pivot indeksini qaytarish."""
    pivot = arr[high]  # Oxirgi elementni pivot sifatida tanlaymiz
    i = low  # Kichikroq elementlar chegarasi indeksi

    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Pivotni to'g'ri joyga qo'yamiz
    arr[i], arr[high] = arr[high], arr[i]
    return i`
		}
	}
};

export default task;
