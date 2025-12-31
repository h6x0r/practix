import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-binary-search-sorted',
	title: 'Binary Search',
	difficulty: 'easy',
	tags: ['python', 'search', 'array', 'divide-conquer'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement binary search on a sorted array.

**Problem:**

Given a **sorted** array of integers and a target value, return the index of the target if found, or -1 if not found.

**How Binary Search Works:**

Binary search works by repeatedly dividing the search interval in half:

1. Compare target with middle element
2. If equal, return the index
3. If target is smaller, search left half
4. If target is larger, search right half
5. Repeat until found or interval is empty

**Examples:**

\`\`\`
Input: arr = [-1, 0, 3, 5, 9, 12], target = 9
Output: 4

Search process:
mid = 2, arr[2] = 3 < 9, search right [5, 9, 12]
mid = 4, arr[4] = 9 = target, return 4

Input: arr = [-1, 0, 3, 5, 9, 12], target = 2
Output: -1 (not found)

Input: arr = [5], target = 5
Output: 0
\`\`\`

**Key Insight:**

Each comparison eliminates half of the remaining elements.

**Time Complexity:** O(log n)
**Space Complexity:** O(1) iterative, O(log n) recursive`,
	initialCode: `from typing import List

def binary_search(arr: List[int], target: int) -> int:
    # TODO: Implement binary search to find target in sorted array

    return -1`,
	solutionCode: `from typing import List


def binary_search(arr: List[int], target: int) -> int:
    """
    Find target in sorted array.

    Args:
        arr: Sorted list of integers
        target: Value to find

    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1`,
	testCode: `import pytest
from solution import binary_search


class TestBinarySearch:
    def test_found_middle(self):
        """Test finding element in middle"""
        assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4

    def test_found_first(self):
        """Test finding first element"""
        assert binary_search([1, 2, 3, 4, 5], 1) == 0

    def test_found_last(self):
        """Test finding last element"""
        assert binary_search([1, 2, 3, 4, 5], 5) == 4

    def test_not_found(self):
        """Test element not in array"""
        assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1

    def test_single_found(self):
        """Test single element found"""
        assert binary_search([5], 5) == 0

    def test_single_not_found(self):
        """Test single element not found"""
        assert binary_search([5], 3) == -1

    def test_empty(self):
        """Test empty array"""
        assert binary_search([], 5) == -1

    def test_two_elements_first(self):
        """Test two elements, find first"""
        assert binary_search([1, 2], 1) == 0

    def test_two_elements_second(self):
        """Test two elements, find second"""
        assert binary_search([1, 2], 2) == 1

    def test_negatives(self):
        """Test array with negatives"""
        assert binary_search([-10, -5, 0, 5, 10], -5) == 1`,
	hint1: `Use two pointers: left and right. Calculate mid carefully to avoid overflow: mid = left + (right - left) / 2`,
	hint2: `The loop condition is left <= right (with equals). When arr[mid] doesn't match, update left = mid + 1 or right = mid - 1.`,
	whyItMatters: `Binary search is one of the most important algorithms in computer science.

**Why This Matters:**

**1. Logarithmic Efficiency**

Each step halves the search space:
\`\`\`
n = 1,000,000 elements
Binary search: ~20 comparisons (log2 1M ≈ 20)
Linear search: up to 1,000,000 comparisons
\`\`\`

**2. Common Bug: Integer Overflow**

\`\`\`python
# WRONG: can overflow if left + right > MaxInt (in other languages)
mid = (left + right) // 2

# CORRECT: no overflow
mid = left + (right - left) // 2
\`\`\`

**3. Off-by-One Errors**

Common mistakes:
\`\`\`python
# Loop condition: use <= not <
while left <= right:
    ...

# Update: use mid+1 and mid-1, not just mid
left = mid + 1   # search right half
right = mid - 1  # search left half
\`\`\`

**4. Variations**

\`\`\`python
# Find first occurrence
# Find last occurrence
# Find insertion point
# Lower bound / upper bound
\`\`\`

**5. Applications**

- Database indexing (B-trees use binary search)
- Git bisect (find bug-introducing commit)
- Dictionary lookup
- Square root calculation
- Search in rotated sorted array

**6. Recursive Version**

\`\`\`python
def binary_search_recursive(arr: list, target: int, left: int, right: int) -> int:
    if left > right:
        return -1
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    return binary_search_recursive(arr, target, left, mid - 1)
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Бинарный поиск',
			description: `Реализуйте бинарный поиск в отсортированном массиве.

**Задача:**

Дан **отсортированный** массив целых чисел и целевое значение, верните индекс цели если найдена, или -1 если не найдена.

**Как работает бинарный поиск:**

Бинарный поиск работает путём многократного деления интервала поиска пополам:

1. Сравнить цель со средним элементом
2. Если равны, вернуть индекс
3. Если цель меньше, искать в левой половине
4. Если цель больше, искать в правой половине
5. Повторять пока не найдено или интервал пуст

**Примеры:**

\`\`\`
Вход: arr = [-1, 0, 3, 5, 9, 12], target = 9
Выход: 4

Вход: arr = [-1, 0, 3, 5, 9, 12], target = 2
Выход: -1 (не найдено)
\`\`\`

**Ключевая идея:**

Каждое сравнение исключает половину оставшихся элементов.

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1) итеративный, O(log n) рекурсивный`,
			hint1: `Используйте два указателя: left и right. Вычисляйте mid аккуратно для избежания переполнения: mid = left + (right - left) / 2`,
			hint2: `Условие цикла left <= right (с равенством). Когда arr[mid] не совпадает, обновляйте left = mid + 1 или right = mid - 1.`,
			whyItMatters: `Бинарный поиск - один из важнейших алгоритмов в информатике.

**Почему это важно:**

**1. Логарифмическая эффективность**

Каждый шаг делит пространство поиска пополам.

**2. Частая ошибка: переполнение**

Неправильно: mid := (left + right) / 2
Правильно: mid := left + (right - left) / 2

**3. Ошибки на единицу**

Условие цикла: используйте <= не <
Обновление: используйте mid+1 и mid-1

**4. Применения**

Индексация баз данных, git bisect, поиск в словаре.`,
			solutionCode: `from typing import List


def binary_search(arr: List[int], target: int) -> int:
    """
    Находит цель в отсортированном массиве.

    Args:
        arr: Отсортированный список целых чисел
        target: Искомое значение

    Returns:
        Индекс цели если найдена, -1 иначе
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1`
		},
		uz: {
			title: 'Binar qidiruv',
			description: `Saralangan massivda binar qidiruvni amalga oshiring.

**Masala:**

**Saralangan** butun sonlar massivi va maqsad qiymat berilgan, agar topilsa maqsadning indeksini qaytaring, yoki topilmasa -1 qaytaring.

**Binar qidiruv qanday ishlaydi:**

Binar qidiruv qidiruv oralig'ini takroriy ikki marta bo'lish orqali ishlaydi:

1. Maqsadni o'rta element bilan solishtirish
2. Agar teng bo'lsa, indeksni qaytarish
3. Agar maqsad kichikroq bo'lsa, chap yarmida qidirish
4. Agar maqsad kattaroq bo'lsa, o'ng yarmida qidirish
5. Topilguncha yoki oraliq bo'sh bo'lguncha takrorlash

**Misollar:**

\`\`\`
Kirish: arr = [-1, 0, 3, 5, 9, 12], target = 9
Chiqish: 4

Kirish: arr = [-1, 0, 3, 5, 9, 12], target = 2
Chiqish: -1 (topilmadi)
\`\`\`

**Asosiy tushuncha:**

Har bir taqqoslash qolgan elementlarning yarmini yo'q qiladi.

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1) iterativ, O(log n) rekursiv`,
			hint1: `Ikkita ko'rsatkichdan foydalaning: left va right. mid ni to'lib ketishdan qochish uchun ehtiyotkorlik bilan hisoblang: mid = left + (right - left) / 2`,
			hint2: `Sikl sharti left <= right (tenglik bilan). arr[mid] mos kelmasa, left = mid + 1 yoki right = mid - 1 ni yangilang.`,
			whyItMatters: `Binar qidiruv kompyuter fanida eng muhim algoritmlardan biri.

**Bu nima uchun muhim:**

**1. Logarifmik samaradorlik**

Har bir qadam qidiruv maydonini ikki marta kamaytiradi.

**2. Keng tarqalgan xato: to'lib ketish**

Noto'g'ri: mid := (left + right) / 2
To'g'ri: mid := left + (right - left) / 2

**3. Birga xatolar**

Sikl sharti: <= dan foydalaning, < emas
Yangilash: mid+1 va mid-1 dan foydalaning

**4. Qo'llanilish**

Ma'lumotlar bazasi indeksatsiyasi, git bisect, lug'atda qidirish.`,
			solutionCode: `from typing import List


def binary_search(arr: List[int], target: int) -> int:
    """
    Saralangan massivda maqsadni topadi.

    Args:
        arr: Saralangan butun sonlar ro'yxati
        target: Izlanayotgan qiymat

    Returns:
        Maqsad indeksi agar topilsa, aks holda -1
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1`
		}
	}
};

export default task;
