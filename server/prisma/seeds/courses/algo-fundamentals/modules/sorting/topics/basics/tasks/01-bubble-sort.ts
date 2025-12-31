import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-bubble-sort',
	title: 'Bubble Sort',
	difficulty: 'easy',
	tags: ['python', 'sorting', 'array', 'comparison'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the bubble sort algorithm.

**Problem:**

Given an array of integers, sort it in ascending order using bubble sort.

**How Bubble Sort Works:**

Repeatedly step through the list, compare adjacent elements and swap them if they are in the wrong order. The pass through the list is repeated until the list is sorted.

After each pass, the largest unsorted element "bubbles up" to its correct position.

**Examples:**

\`\`\`
Input: [64, 34, 25, 12, 22, 11, 90]
Output: [11, 12, 22, 25, 34, 64, 90]

Pass 1: [34, 25, 12, 22, 11, 64, 90] - 90 bubbles to end
Pass 2: [25, 12, 22, 11, 34, 64, 90] - 64 in place
Pass 3: [12, 22, 11, 25, 34, 64, 90] - 34 in place
...continues until sorted
\`\`\`

**Optimization:**

If no swaps occur in a pass, the array is already sorted - exit early.

**Time Complexity:** O(n^2) average and worst, O(n) best (already sorted)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def bubble_sort(arr: List[int]) -> List[int]:
    # TODO: Implement bubble sort algorithm

    return arr`,
	solutionCode: `from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Sort array using bubble sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list in ascending order
    """
    n = len(arr)

    for i in range(n - 1):
        swapped = False

        # Last i elements are already in place
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swaps, array is sorted
        if not swapped:
            break

    return arr`,
	testCode: `import pytest
from solution import bubble_sort


class TestBubbleSort:
    def test_normal(self):
        """Test normal unsorted array"""
        arr = [64, 34, 25, 12, 22, 11, 90]
        assert bubble_sort(arr.copy()) == [11, 12, 22, 25, 34, 64, 90]

    def test_already_sorted(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        assert bubble_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        assert bubble_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test array with duplicates"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        assert bubble_sort(arr.copy()) == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_single(self):
        """Test single element"""
        assert bubble_sort([1]) == [1]

    def test_empty(self):
        """Test empty array"""
        assert bubble_sort([]) == []

    def test_two_elements(self):
        """Test two elements"""
        assert bubble_sort([2, 1]) == [1, 2]

    def test_negatives(self):
        """Test array with negative numbers"""
        arr = [-5, 3, -1, 0, 2, -3]
        assert bubble_sort(arr.copy()) == [-5, -3, -1, 0, 2, 3]

    def test_large_numbers(self):
        """Test array with large numbers"""
        arr = [1000, 500, 2000, 100, 1500]
        assert bubble_sort(arr.copy()) == [100, 500, 1000, 1500, 2000]

    def test_many_duplicates(self):
        """Test array with many duplicate elements"""
        arr = [5, 2, 5, 2, 5, 2, 5]
        assert bubble_sort(arr.copy()) == [2, 2, 2, 5, 5, 5, 5]`,
	hint1: `Use two nested loops. The outer loop runs n-1 times, the inner loop compares adjacent elements.`,
	hint2: `Optimize with a "swapped" flag. If no swaps happen in a pass, the array is sorted and you can exit early.`,
	whyItMatters: `Bubble sort is the simplest sorting algorithm to understand.

**Why This Matters:**

**1. Learning Foundation**

Bubble sort is often the first sorting algorithm taught:
\`\`\`python
# Core idea: compare neighbors, swap if wrong order
if arr[j] > arr[j + 1]:
    arr[j], arr[j + 1] = arr[j + 1], arr[j]
\`\`\`

**2. Understanding O(n^2)**

Nested loops demonstrate quadratic complexity:
\`\`\`python
for i in range(n):           # n iterations
    for j in range(n - i):   # ~n iterations
        # Total: ~n^2 operations
\`\`\`

**3. Optimization Technique**

The "swapped" flag is a common optimization pattern:
\`\`\`python
# Early termination when no work is done
if not swapped:
    break  # Already sorted!
\`\`\`

**4. When to Use**

- Educational purposes
- Very small arrays
- Nearly sorted arrays (with optimization)
- Never for production with large data

**5. Comparison with Others**

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Bubble    | O(n) | O(n^2)  | O(n^2)| O(1)  |
| Selection | O(n^2)| O(n^2) | O(n^2)| O(1)  |
| Insertion | O(n) | O(n^2)  | O(n^2)| O(1)  |`,
	order: 1,
	translations: {
		ru: {
			title: 'Пузырьковая сортировка',
			description: `Реализуйте алгоритм пузырьковой сортировки.

**Задача:**

Дан массив целых чисел, отсортируйте его по возрастанию используя пузырьковую сортировку.

**Как работает пузырьковая сортировка:**

Многократно проходим по списку, сравниваем соседние элементы и меняем их местами, если они в неправильном порядке. Проход повторяется до тех пор, пока список не отсортирован.

После каждого прохода наибольший неотсортированный элемент "всплывает" на своё место.

**Примеры:**

\`\`\`
Вход: [64, 34, 25, 12, 22, 11, 90]
Выход: [11, 12, 22, 25, 34, 64, 90]
\`\`\`

**Оптимизация:**

Если за проход не было обменов, массив уже отсортирован - выходим досрочно.

**Временная сложность:** O(n^2) средний и худший, O(n) лучший
**Пространственная сложность:** O(1)`,
			hint1: `Используйте два вложенных цикла. Внешний выполняется n-1 раз, внутренний сравнивает соседние элементы.`,
			hint2: `Оптимизируйте флагом "swapped". Если за проход не было обменов, массив отсортирован и можно выйти досрочно.`,
			whyItMatters: `Пузырьковая сортировка - простейший для понимания алгоритм сортировки.

**Почему это важно:**

**1. Основа обучения**

Пузырьковая сортировка часто первый изучаемый алгоритм сортировки.

**2. Понимание O(n^2)**

Вложенные циклы демонстрируют квадратичную сложность.

**3. Техника оптимизации**

Флаг "swapped" - распространённый паттерн оптимизации.`,
			solutionCode: `from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Сортирует массив пузырьковой сортировкой.

    Args:
        arr: Список целых чисел для сортировки

    Returns:
        Отсортированный список по возрастанию
    """
    n = len(arr)

    for i in range(n - 1):
        swapped = False

        # Последние i элементов уже на месте
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # Если обменов не было, массив отсортирован
        if not swapped:
            break

    return arr`
		},
		uz: {
			title: 'Pufakchali saralash',
			description: `Pufakchali saralash algoritmini amalga oshiring.

**Masala:**

Butun sonlar massivi berilgan, uni pufakchali saralash yordamida o'sish tartibida saralang.

**Pufakchali saralash qanday ishlaydi:**

Ro'yxatni takroriy o'ting, qo'shni elementlarni solishtiring va agar noto'g'ri tartibda bo'lsa almashiring. Ro'yxat saralanmaguncha o'tish takrorlanadi.

Har bir o'tishdan so'ng, eng katta saralanmagan element o'z joyiga "ko'tariladi".

**Misollar:**

\`\`\`
Kirish: [64, 34, 25, 12, 22, 11, 90]
Chiqish: [11, 12, 22, 25, 34, 64, 90]
\`\`\`

**Optimizatsiya:**

Agar o'tishda almashish bo'lmasa, massiv allaqachon saralangan - erta chiqing.

**Vaqt murakkabligi:** O(n^2) o'rtacha va eng yomon, O(n) eng yaxshi
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikkita ichma-ich sikl ishlating. Tashqi sikl n-1 marta ishlaydi, ichki sikl qo'shni elementlarni solishtiradi.`,
			hint2: `"swapped" bayrog'i bilan optimizatsiya qiling. Agar o'tishda almashish bo'lmasa, massiv saralangan va erta chiqishingiz mumkin.`,
			whyItMatters: `Pufakchali saralash tushunish uchun eng oddiy saralash algoritmi.

**Bu nima uchun muhim:**

**1. O'rganish asosi**

Pufakchali saralash ko'pincha birinchi o'rgatilgan saralash algoritmi.

**2. O(n^2) ni tushunish**

Ichma-ich sikllar kvadratik murakkablikni ko'rsatadi.

**3. Optimizatsiya texnikasi**

"swapped" bayrog'i keng tarqalgan optimizatsiya patterni.`,
			solutionCode: `from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Massivni pufakchali saralash algoritmi bilan saralaydi.

    Args:
        arr: Saralash uchun butun sonlar ro'yxati

    Returns:
        O'sish tartibida saralangan ro'yxat
    """
    n = len(arr)

    for i in range(n - 1):
        swapped = False

        # Oxirgi i elementlar allaqachon joyida
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # Agar almashish bo'lmasa, massiv saralangan
        if not swapped:
            break

    return arr`
		}
	}
};

export default task;
