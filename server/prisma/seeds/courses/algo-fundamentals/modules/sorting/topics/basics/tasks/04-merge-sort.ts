import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-merge-sort',
	title: 'Merge Sort',
	difficulty: 'medium',
	tags: ['python', 'sorting', 'array', 'divide-conquer', 'recursion'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the merge sort algorithm.

**Problem:**

Given an array of integers, sort it in ascending order using merge sort.

**How Merge Sort Works:**

Merge sort uses divide-and-conquer strategy:

1. **Divide:** Split the array into two halves
2. **Conquer:** Recursively sort each half
3. **Combine:** Merge the two sorted halves into one sorted array

**Examples:**

\`\`\`
Input: [38, 27, 43, 3, 9, 82, 10]
Output: [3, 9, 10, 27, 38, 43, 82]

Divide phase:
[38, 27, 43, 3, 9, 82, 10]
       /              \\
[38, 27, 43, 3]    [9, 82, 10]
    /      \\         /      \\
[38, 27]  [43, 3]  [9, 82]  [10]
 /   \\    /   \\    /   \\
[38] [27] [43] [3] [9] [82] [10]

Merge phase:
[27, 38] [3, 43] [9, 82] [10]
    \\      /        \\      /
[3, 27, 38, 43]  [9, 10, 82]
        \\            /
  [3, 9, 10, 27, 38, 43, 82]
\`\`\`

**Key Properties:**

- Stable sorting algorithm
- Guaranteed O(n log n) performance
- Requires O(n) extra space

**Time Complexity:** O(n log n) always
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    # TODO: Implement merge sort using divide-and-conquer

    return arr


def merge(left: List[int], right: List[int]) -> List[int]:
    # TODO: Merge two sorted lists into one sorted list

    return []`,
	solutionCode: `from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    """
    Sort array using merge sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list in ascending order
    """
    if len(arr) <= 1:
        return arr

    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Conquer
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Combine two sorted arrays into one sorted array.

    Args:
        left: First sorted list
        right: Second sorted list

    Returns:
        Merged sorted list
    """
    result = []
    i, j = 0, 0

    # Compare elements from both arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result`,
	testCode: `import pytest
from solution import merge_sort


class TestMergeSort:
    def test_normal(self):
        """Test normal unsorted array"""
        arr = [38, 27, 43, 3, 9, 82, 10]
        assert merge_sort(arr) == [3, 9, 10, 27, 38, 43, 82]

    def test_already_sorted(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        assert merge_sort(arr) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        assert merge_sort(arr) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test array with duplicates"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        assert merge_sort(arr) == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_single(self):
        """Test single element"""
        assert merge_sort([1]) == [1]

    def test_empty(self):
        """Test empty array"""
        assert merge_sort([]) == []

    def test_two_elements(self):
        """Test two elements"""
        assert merge_sort([2, 1]) == [1, 2]

    def test_negatives(self):
        """Test array with negatives"""
        arr = [-5, 3, -1, 0, 2]
        assert merge_sort(arr) == [-5, -1, 0, 2, 3]

    def test_large_array(self):
        """Test larger array"""
        arr = [38, 27, 43, 3, 9, 82, 10, 15, 6, 25]
        assert merge_sort(arr) == [3, 6, 9, 10, 15, 25, 27, 38, 43, 82]

    def test_all_same(self):
        """Test array with all same elements"""
        arr = [3, 3, 3, 3, 3]
        assert merge_sort(arr) == [3, 3, 3, 3, 3]`,
	hint1: `Split the array in half using arr[:mid] and arr[mid:]. Recursively sort both halves, then merge.`,
	hint2: `In the merge function, use two pointers (i for left, j for right). Compare elements and add the smaller one to result. Don't forget remaining elements.`,
	whyItMatters: `Merge sort introduces divide-and-conquer and guaranteed O(n log n) performance.

**Why This Matters:**

**1. Divide and Conquer**

The classic example of divide-and-conquer:
\`\`\`python
# 1. Divide: split in half
# 2. Conquer: recursively solve subproblems
# 3. Combine: merge solutions
\`\`\`

**2. Guaranteed Performance**

Always O(n log n), unlike quicksort:
\`\`\`
                MergeSort    QuickSort
Best case:      O(n log n)   O(n log n)
Average:        O(n log n)   O(n log n)
Worst case:     O(n log n)   O(n^2)
\`\`\`

**3. The Merge Operation**

Linear time merge of sorted arrays:
\`\`\`python
# Two sorted arrays -> one sorted array in O(n)
# This is the key insight!
\`\`\`

**4. Stable Sort**

Preserves relative order (use <= not <):
\`\`\`python
if left[i] <= right[j]:  # <= makes it stable
    result.append(left[i])
\`\`\`

**5. Real-World Usage**

- External sorting (when data doesn't fit in memory)
- Linked list sorting (no random access needed)
- Counting inversions
- Standard library sorts (often hybrid with merge sort)

**6. Time Complexity Analysis**

\`\`\`
T(n) = 2*T(n/2) + O(n)
     = 2*[2*T(n/4) + O(n/2)] + O(n)
     = 4*T(n/4) + O(n) + O(n)
     ...
     = n*T(1) + O(n)*log(n)
     = O(n log n)
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Сортировка слиянием',
			description: `Реализуйте алгоритм сортировки слиянием.

**Задача:**

Дан массив целых чисел, отсортируйте его по возрастанию используя сортировку слиянием.

**Как работает сортировка слиянием:**

Сортировка слиянием использует стратегию "разделяй и властвуй":

1. **Разделение:** Разбить массив на две половины
2. **Покорение:** Рекурсивно отсортировать каждую половину
3. **Объединение:** Слить две отсортированные половины в один отсортированный массив

**Примеры:**

\`\`\`
Вход: [38, 27, 43, 3, 9, 82, 10]
Выход: [3, 9, 10, 27, 38, 43, 82]
\`\`\`

**Ключевые свойства:**

- Стабильный алгоритм сортировки
- Гарантированная производительность O(n log n)
- Требует O(n) дополнительной памяти

**Временная сложность:** O(n log n) всегда
**Пространственная сложность:** O(n)`,
			hint1: `Разделите массив пополам используя arr[:mid] и arr[mid:]. Рекурсивно отсортируйте обе половины, затем слейте.`,
			hint2: `В функции merge используйте два указателя (i для left, j для right). Сравнивайте элементы и добавляйте меньший в результат. Не забудьте оставшиеся элементы.`,
			whyItMatters: `Сортировка слиянием знакомит с "разделяй и властвуй" и гарантированной O(n log n).

**Почему это важно:**

**1. Разделяй и властвуй**

Классический пример стратегии "разделяй и властвуй".

**2. Гарантированная производительность**

Всегда O(n log n), в отличие от быстрой сортировки.

**3. Операция слияния**

Слияние отсортированных массивов за линейное время.

**4. Стабильная сортировка**

Сохраняет относительный порядок.`,
			solutionCode: `from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    """
    Сортирует массив сортировкой слиянием.

    Args:
        arr: Список целых чисел для сортировки

    Returns:
        Отсортированный список по возрастанию
    """
    if len(arr) <= 1:
        return arr

    # Разделение
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Покорение
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Объединяет два отсортированных массива в один.

    Args:
        left: Первый отсортированный список
        right: Второй отсортированный список

    Returns:
        Объединённый отсортированный список
    """
    result = []
    i, j = 0, 0

    # Сравниваем элементы из обоих массивов
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Добавляем оставшиеся элементы
    result.extend(left[i:])
    result.extend(right[j:])

    return result`
		},
		uz: {
			title: 'Birlashtirish saralashi',
			description: `Birlashtirish saralash algoritmini amalga oshiring.

**Masala:**

Butun sonlar massivi berilgan, uni birlashtirish saralash yordamida o'sish tartibida saralang.

**Birlashtirish saralash qanday ishlaydi:**

Birlashtirish saralash "bo'l va hukmronlik qil" strategiyasidan foydalanadi:

1. **Bo'lish:** Massivni ikki yarmiga bo'lish
2. **Hukmronlik:** Har bir yarmini rekursiv saralash
3. **Birlashtirish:** Ikki saralangan yarmini bitta saralangan massivga birlashtirish

**Misollar:**

\`\`\`
Kirish: [38, 27, 43, 3, 9, 82, 10]
Chiqish: [3, 9, 10, 27, 38, 43, 82]
\`\`\`

**Asosiy xususiyatlar:**

- Barqaror saralash algoritmi
- Kafolatlangan O(n log n) samaradorlik
- O(n) qo'shimcha xotira talab qiladi

**Vaqt murakkabligi:** O(n log n) doimo
**Xotira murakkabligi:** O(n)`,
			hint1: `Massivni arr[:mid] va arr[mid:] yordamida yarmiga bo'ling. Ikkala yarmini rekursiv saralang, keyin birlashtiring.`,
			hint2: `Merge funktsiyasida ikkita ko'rsatkichdan foydalaning (left uchun i, right uchun j). Elementlarni solishtiring va kichigini natijaga qo'shing. Qolgan elementlarni unutmang.`,
			whyItMatters: `Birlashtirish saralash "bo'l va hukmronlik qil" va kafolatlangan O(n log n) bilan tanishtiradi.

**Bu nima uchun muhim:**

**1. Bo'l va hukmronlik qil**

"Bo'l va hukmronlik qil" strategiyasining klassik misoli.

**2. Kafolatlangan samaradorlik**

Tez saralashdan farqli o'laroq, doimo O(n log n).

**3. Birlashtirish operatsiyasi**

Saralangan massivlarni chiziqli vaqtda birlashtirish.

**4. Barqaror saralash**

Nisbiy tartibni saqlaydi.`,
			solutionCode: `from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    """
    Massivni birlashtirish saralash algoritmi bilan saralaydi.

    Args:
        arr: Saralash uchun butun sonlar ro'yxati

    Returns:
        O'sish tartibida saralangan ro'yxat
    """
    if len(arr) <= 1:
        return arr

    # Bo'lish
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Hukmronlik
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Ikki saralangan massivni bitta saralangan massivga birlashtiradi.

    Args:
        left: Birinchi saralangan ro'yxat
        right: Ikkinchi saralangan ro'yxat

    Returns:
        Birlashtirilgan saralangan ro'yxat
    """
    result = []
    i, j = 0, 0

    # Ikkala massivdan elementlarni solishtiramiz
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Qolgan elementlarni qo'shamiz
    result.extend(left[i:])
    result.extend(right[j:])

    return result`
		}
	}
};

export default task;
