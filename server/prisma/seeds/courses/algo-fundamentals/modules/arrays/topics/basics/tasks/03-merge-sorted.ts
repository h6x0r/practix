import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-merge-sorted',
	title: 'Merge Sorted Arrays',
	difficulty: 'easy',
	tags: ['python', 'arrays', 'two-pointers', 'merge'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Merge two sorted arrays into one sorted array.

**Problem:**

Given two sorted integer arrays \`nums1\` and \`nums2\`, merge them into a single sorted array.

**Examples:**

\`\`\`
Input: nums1 = [1, 3, 5], nums2 = [2, 4, 6]
Output: [1, 2, 3, 4, 5, 6]

Input: nums1 = [1, 2, 3], nums2 = [4, 5, 6]
Output: [1, 2, 3, 4, 5, 6]

Input: nums1 = [], nums2 = [1, 2, 3]
Output: [1, 2, 3]
\`\`\`

**Two Pointers Approach:**

Use two pointers, one for each array:
1. Compare elements at both pointers
2. Add the smaller one to result
3. Move that pointer forward
4. Repeat until both arrays exhausted

\`\`\`python
i, j = 0, 0
while i < len(nums1) and j < len(nums2):
    if nums1[i] <= nums2[j]:
        result.append(nums1[i])
        i += 1
    else:
        result.append(nums2[j])
        j += 1
# Append remaining elements
\`\`\`

**Time Complexity:** O(n + m)
**Space Complexity:** O(n + m) for result array`,
	initialCode: `from typing import List

def merge_sorted(nums1: List[int], nums2: List[int]) -> List[int]:
    # TODO: Merge two sorted arrays into one sorted array

    return []`,
	solutionCode: `from typing import List

def merge_sorted(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Merge two sorted arrays into one sorted array.

    Args:
        nums1: First sorted list of integers
        nums2: Second sorted list of integers

    Returns:
        Merged sorted list containing all elements
    """
    result = []

    # Two pointers
    i, j = 0, 0

    # Merge while both arrays have elements
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1

    # Append remaining elements from nums1
    while i < len(nums1):
        result.append(nums1[i])
        i += 1

    # Append remaining elements from nums2
    while j < len(nums2):
        result.append(nums2[j])
        j += 1

    return result`,
	testCode: `import pytest
from solution import merge_sorted

class TestMergeSorted:
    def test_interleaved(self):
        """Test interleaved elements"""
        assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]

    def test_sequential(self):
        """Test sequential arrays (no overlap)"""
        assert merge_sorted([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]

    def test_empty_first(self):
        """Test with empty first array"""
        assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]

    def test_empty_second(self):
        """Test with empty second array"""
        assert merge_sorted([1, 2, 3], []) == [1, 2, 3]

    def test_both_empty(self):
        """Test with both arrays empty"""
        assert merge_sorted([], []) == []

    def test_duplicates(self):
        """Test with duplicate elements"""
        assert merge_sorted([1, 2, 2, 3], [2, 2, 4]) == [1, 2, 2, 2, 2, 3, 4]

    def test_single_elements(self):
        """Test with single element arrays"""
        assert merge_sorted([1], [2]) == [1, 2]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert merge_sorted([-5, -1, 0], [-3, -2, 1]) == [-5, -3, -2, -1, 0, 1]

    def test_different_lengths(self):
        """Test with arrays of very different lengths"""
        assert merge_sorted([1, 2], [3, 4, 5, 6, 7, 8]) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_all_duplicates(self):
        """Test with all duplicate elements"""
        assert merge_sorted([1, 1, 1], [1, 1, 1]) == [1, 1, 1, 1, 1, 1]`,
	hint1: `Create a result list and use two index variables i and j. Use a while loop with condition: while i < len(nums1) and j < len(nums2).`,
	hint2: `After the main loop, one array might still have elements. Add two more loops to append any remaining elements from nums1 (while i < len(nums1)) and nums2 (while j < len(nums2)).`,
	whyItMatters: `The two-pointer merge technique is fundamental to many algorithms.

**Why This Matters:**

**1. Core of Merge Sort**

This is the "merge" step in merge sort:
\`\`\`python
def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)  # <-- This function!
\`\`\`

**2. Two Pointer Pattern**

This teaches a critical pattern:
- Process two sequences in parallel
- Make decisions based on current elements
- Move pointers independently

**3. Common Interview Problems**

- Merge K sorted lists (heap + merge)
- Intersection of two sorted arrays
- Union of two sorted arrays
- Merge intervals (sort + merge)

**4. Real-World Applications**

- Database merge joins
- External sorting (merge files from disk)
- Git merge (conflict resolution)
- Log aggregation systems`,
	order: 3,
	translations: {
		ru: {
			title: 'Слияние отсортированных массивов',
			description: `Объедините два отсортированных массива в один отсортированный массив.

**Задача:**

Даны два отсортированных массива целых чисел \`nums1\` и \`nums2\`, объедините их в один отсортированный массив.

**Примеры:**

\`\`\`
Вход: nums1 = [1, 3, 5], nums2 = [2, 4, 6]
Выход: [1, 2, 3, 4, 5, 6]

Вход: nums1 = [1, 2, 3], nums2 = [4, 5, 6]
Выход: [1, 2, 3, 4, 5, 6]

Вход: nums1 = [], nums2 = [1, 2, 3]
Выход: [1, 2, 3]
\`\`\`

**Подход двух указателей:**

Используйте два указателя, по одному для каждого массива:
1. Сравните элементы по обоим указателям
2. Добавьте меньший в результат
3. Сдвиньте этот указатель вперёд
4. Повторяйте пока оба массива не исчерпаны

\`\`\`python
i, j = 0, 0
while i < len(nums1) and j < len(nums2):
    if nums1[i] <= nums2[j]:
        result.append(nums1[i])
        i += 1
    else:
        result.append(nums2[j])
        j += 1
# Добавляем оставшиеся элементы
\`\`\`

**Временная сложность:** O(n + m)
**Пространственная сложность:** O(n + m) для результирующего массива`,
			hint1: `Создайте список результата и используйте две переменные индекса i и j. Используйте цикл while с условием: while i < len(nums1) and j < len(nums2).`,
			hint2: `После основного цикла в одном массиве ещё могут быть элементы. Добавьте ещё два цикла для добавления оставшихся элементов из nums1 (while i < len(nums1)) и nums2 (while j < len(nums2)).`,
			whyItMatters: `Техника слияния с двумя указателями фундаментальна для многих алгоритмов.

**Почему это важно:**

**1. Основа Merge Sort**

Это шаг "слияния" в merge sort:
\`\`\`python
def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)  # <-- Эта функция!
\`\`\`

**2. Паттерн двух указателей**

Это учит критическому паттерну:
- Обрабатывать две последовательности параллельно
- Принимать решения на основе текущих элементов
- Двигать указатели независимо

**3. Распространённые задачи на интервью**

- Слияние K отсортированных списков (heap + merge)
- Пересечение двух отсортированных массивов
- Объединение двух отсортированных массивов
- Слияние интервалов (сортировка + слияние)

**4. Применения в реальном мире**

- Merge join в базах данных
- Внешняя сортировка (слияние файлов с диска)
- Git merge (разрешение конфликтов)
- Системы агрегации логов`,
			solutionCode: `from typing import List

def merge_sorted(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Объединяет два отсортированных массива в один отсортированный.

    Args:
        nums1: Первый отсортированный список целых чисел
        nums2: Второй отсортированный список целых чисел

    Returns:
        Объединённый отсортированный список со всеми элементами
    """
    result = []

    # Два указателя
    i, j = 0, 0

    # Слияние пока в обоих массивах есть элементы
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1

    # Добавляем оставшиеся элементы из nums1
    while i < len(nums1):
        result.append(nums1[i])
        i += 1

    # Добавляем оставшиеся элементы из nums2
    while j < len(nums2):
        result.append(nums2[j])
        j += 1

    return result`
		},
		uz: {
			title: 'Saralangan massivlarni birlashtirish',
			description: `Ikkita saralangan massivni bitta saralangan massivga birlashtiring.

**Masala:**

Ikkita saralangan butun sonlar massivi \`nums1\` va \`nums2\` berilgan, ularni bitta saralangan massivga birlashtiring.

**Misollar:**

\`\`\`
Kirish: nums1 = [1, 3, 5], nums2 = [2, 4, 6]
Chiqish: [1, 2, 3, 4, 5, 6]

Kirish: nums1 = [1, 2, 3], nums2 = [4, 5, 6]
Chiqish: [1, 2, 3, 4, 5, 6]

Kirish: nums1 = [], nums2 = [1, 2, 3]
Chiqish: [1, 2, 3]
\`\`\`

**Ikki ko'rsatkich yondashuvi:**

Har bir massiv uchun bittadan, ikkita ko'rsatkichdan foydalaning:
1. Ikkala ko'rsatkichdagi elementlarni solishtiring
2. Kichigini natijaga qo'shing
3. O'sha ko'rsatkichni oldinga siljiting
4. Ikkala massiv tugaguncha takrorlang

\`\`\`python
i, j = 0, 0
while i < len(nums1) and j < len(nums2):
    if nums1[i] <= nums2[j]:
        result.append(nums1[i])
        i += 1
    else:
        result.append(nums2[j])
        j += 1
# Qolgan elementlarni qo'shamiz
\`\`\`

**Vaqt murakkabligi:** O(n + m)
**Xotira murakkabligi:** O(n + m) natija massivi uchun`,
			hint1: `Natija ro'yxatini yarating va ikkita indeks o'zgaruvchisi i va j dan foydalaning. while tsiklidan shartli foydalaning: while i < len(nums1) and j < len(nums2).`,
			hint2: `Asosiy tsikldan keyin bitta massivda hali ham elementlar bo'lishi mumkin. nums1 dan qolgan elementlarni qo'shish uchun yana ikkita tsikl qo'shing (while i < len(nums1)) va nums2 uchun (while j < len(nums2)).`,
			whyItMatters: `Ikki ko'rsatkichli birlashtirish texnikasi ko'plab algoritmlar uchun asosiy.

**Bu nima uchun muhim:**

**1. Merge Sort asosi**

Bu merge sort dagi "birlashtirish" qadami:
\`\`\`python
def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)  # <-- Bu funksiya!
\`\`\`

**2. Ikki ko'rsatkich patterni**

Bu muhim patternni o'rgatadi:
- Ikkita ketma-ketlikni parallel qayta ishlash
- Joriy elementlar asosida qarorlar qabul qilish
- Ko'rsatkichlarni mustaqil harakatlantirish

**3. Intervyu uchun keng tarqalgan masalalar**

- K ta saralangan ro'yxatlarni birlashtirish (heap + merge)
- Ikkita saralangan massivning kesishmasi
- Ikkita saralangan massivning birlashmasi
- Intervallarni birlashtirish (saralash + birlashtirish)

**4. Haqiqiy dunyo qo'llanilishi**

- Ma'lumotlar bazasida merge join
- Tashqi saralash (diskdan fayllarni birlashtirish)
- Git merge (konfliktlarni hal qilish)
- Log agregatsiya tizimlari`,
			solutionCode: `from typing import List

def merge_sorted(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Ikkita saralangan massivni bitta saralanganga birlashtiradi.

    Args:
        nums1: Birinchi saralangan butun sonlar ro'yxati
        nums2: Ikkinchi saralangan butun sonlar ro'yxati

    Returns:
        Barcha elementlarni o'z ichiga olgan birlashtirilgan saralangan ro'yxat
    """
    result = []

    # Ikki ko'rsatkich
    i, j = 0, 0

    # Ikkala massivda elementlar bor ekan birlashtirish
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1

    # nums1 dan qolgan elementlarni qo'shamiz
    while i < len(nums1):
        result.append(nums1[i])
        i += 1

    # nums2 dan qolgan elementlarni qo'shamiz
    while j < len(nums2):
        result.append(nums2[j])
        j += 1

    return result`
		}
	}
};

export default task;
