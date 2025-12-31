import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-insertion-sort',
	title: 'Insertion Sort',
	difficulty: 'easy',
	tags: ['python', 'sorting', 'array', 'comparison'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the insertion sort algorithm.

**Problem:**

Given an array of integers, sort it in ascending order using insertion sort.

**How Insertion Sort Works:**

Build the sorted array one element at a time. Take each element and insert it into its correct position in the already-sorted portion.

Think of sorting playing cards: you pick up cards one by one and insert each into the correct position among the cards already in your hand.

**Examples:**

\`\`\`
Input: [5, 2, 4, 6, 1, 3]
Output: [1, 2, 3, 4, 5, 6]

Step by step:
[5] 2 4 6 1 3 - Start with first element
[2, 5] 4 6 1 3 - Insert 2 before 5
[2, 4, 5] 6 1 3 - Insert 4 between 2 and 5
[2, 4, 5, 6] 1 3 - Insert 6 at end
[1, 2, 4, 5, 6] 3 - Insert 1 at beginning
[1, 2, 3, 4, 5, 6] - Insert 3 between 2 and 4
\`\`\`

**Key Properties:**

- Stable sorting (preserves relative order of equal elements)
- Adaptive (efficient for nearly sorted arrays)
- Online (can sort as data arrives)

**Time Complexity:** O(n^2) worst/average, O(n) best (nearly sorted)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def insertion_sort(arr: List[int]) -> List[int]:
    # TODO: Implement insertion sort algorithm

    return arr`,
	solutionCode: `from typing import List


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Sort array using insertion sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list in ascending order
    """
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Shift elements greater than key to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # Insert key at correct position
        arr[j + 1] = key

    return arr`,
	testCode: `import pytest
from solution import insertion_sort


class TestInsertionSort:
    def test_normal(self):
        """Test normal unsorted array"""
        arr = [5, 2, 4, 6, 1, 3]
        assert insertion_sort(arr.copy()) == [1, 2, 3, 4, 5, 6]

    def test_already_sorted(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        assert insertion_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        assert insertion_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_duplicates(self):
        """Test array with duplicates"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        assert insertion_sort(arr.copy()) == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_single(self):
        """Test single element"""
        assert insertion_sort([1]) == [1]

    def test_empty(self):
        """Test empty array"""
        assert insertion_sort([]) == []

    def test_two_elements(self):
        """Test two elements"""
        assert insertion_sort([2, 1]) == [1, 2]

    def test_nearly_sorted(self):
        """Test nearly sorted array"""
        arr = [1, 2, 4, 3, 5]
        assert insertion_sort(arr.copy()) == [1, 2, 3, 4, 5]

    def test_negatives(self):
        """Test array with negative numbers"""
        arr = [-3, 5, -1, 0, 2, -5]
        assert insertion_sort(arr.copy()) == [-5, -3, -1, 0, 2, 5]

    def test_all_same(self):
        """Test array with all same elements"""
        arr = [7, 7, 7, 7, 7]
        assert insertion_sort(arr.copy()) == [7, 7, 7, 7, 7]`,
	hint1: `Save the current element (key) first. Then shift all larger elements right until you find the correct position.`,
	hint2: `Use a while loop that continues while j >= 0 AND arr[j] > key. This shifts elements and finds the insertion point.`,
	whyItMatters: `Insertion sort is efficient for small or nearly-sorted data and introduces important concepts.

**Why This Matters:**

**1. Best for Small Data**

Insertion sort is often used for small arrays:
\`\`\`python
# Many sorting libraries switch to insertion sort
# for small subarrays (e.g., n < 10-20)
if n < 10:
    return insertion_sort(arr)
return merge_sort(arr)
\`\`\`

**2. Adaptive Algorithm**

Runs in O(n) for nearly sorted data:
\`\`\`python
# If array is sorted, inner loop never executes
# [1, 2, 3, 4, 5] -> O(n) comparisons
\`\`\`

**3. Online Sorting**

Can sort as elements arrive:
\`\`\`python
# Insert new element into sorted list
def insert(sorted_list: list, new_element: int) -> list:
    sorted_list.append(new_element)
    j = len(sorted_list) - 2
    while j >= 0 and sorted_list[j] > new_element:
        sorted_list[j + 1] = sorted_list[j]
        j -= 1
    sorted_list[j + 1] = new_element
    return sorted_list
\`\`\`

**4. Stable Sort**

Preserves relative order of equal elements:
\`\`\`
[5a, 3, 5b, 2] -> [2, 3, 5a, 5b]
# 5a still comes before 5b
\`\`\`

**5. Used in Practice**

- Timsort (Python, Java) uses insertion sort for small runs
- Quick sort switches to insertion sort for small subarrays
- Shell sort is a generalization of insertion sort`,
	order: 3,
	translations: {
		ru: {
			title: 'Сортировка вставками',
			description: `Реализуйте алгоритм сортировки вставками.

**Задача:**

Дан массив целых чисел, отсортируйте его по возрастанию используя сортировку вставками.

**Как работает сортировка вставками:**

Строим отсортированный массив по одному элементу за раз. Берём каждый элемент и вставляем его в правильную позицию в уже отсортированной части.

Как сортировка игральных карт: вы берёте карты по одной и вставляете каждую на правильное место среди карт уже в руке.

**Примеры:**

\`\`\`
Вход: [5, 2, 4, 6, 1, 3]
Выход: [1, 2, 3, 4, 5, 6]

Пошагово:
[5] 2 4 6 1 3 - Начинаем с первого элемента
[2, 5] 4 6 1 3 - Вставляем 2 перед 5
[2, 4, 5] 6 1 3 - Вставляем 4 между 2 и 5
[2, 4, 5, 6] 1 3 - Вставляем 6 в конец
[1, 2, 4, 5, 6] 3 - Вставляем 1 в начало
[1, 2, 3, 4, 5, 6] - Вставляем 3 между 2 и 4
\`\`\`

**Временная сложность:** O(n^2) худший/средний, O(n) лучший
**Пространственная сложность:** O(1)`,
			hint1: `Сначала сохраните текущий элемент (key). Затем сдвигайте все большие элементы вправо пока не найдёте правильную позицию.`,
			hint2: `Используйте цикл while, который продолжается пока j >= 0 И arr[j] > key. Это сдвигает элементы и находит точку вставки.`,
			whyItMatters: `Сортировка вставками эффективна для малых или почти отсортированных данных.

**Почему это важно:**

**1. Лучшая для малых данных**

Сортировка вставками часто используется для малых массивов.

**2. Адаптивный алгоритм**

Работает за O(n) для почти отсортированных данных.

**3. Онлайн сортировка**

Может сортировать по мере поступления элементов.

**4. Стабильная сортировка**

Сохраняет относительный порядок равных элементов.`,
			solutionCode: `from typing import List


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Сортирует массив сортировкой вставками.

    Args:
        arr: Список целых чисел для сортировки

    Returns:
        Отсортированный список по возрастанию
    """
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Сдвигаем элементы больше key вправо
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # Вставляем key на правильную позицию
        arr[j + 1] = key

    return arr`
		},
		uz: {
			title: 'Qo\'yish saralashi',
			description: `Qo'yish saralash algoritmini amalga oshiring.

**Masala:**

Butun sonlar massivi berilgan, uni qo'yish saralash yordamida o'sish tartibida saralang.

**Qo'yish saralash qanday ishlaydi:**

Saralangan massivni birma-bir element bilan quramiz. Har bir elementni allaqachon saralangan qismdagi to'g'ri joyiga qo'yamiz.

O'yin kartalarini saralash kabi: kartalarni birma-bir olasiz va har birini qo'lingizdagi kartalar orasidagi to'g'ri joyga qo'yasiz.

**Misollar:**

\`\`\`
Kirish: [5, 2, 4, 6, 1, 3]
Chiqish: [1, 2, 3, 4, 5, 6]

Qadamma-qadam:
[5] 2 4 6 1 3 - Birinchi element bilan boshlaymiz
[2, 5] 4 6 1 3 - 2 ni 5 oldiga qo'yamiz
[2, 4, 5] 6 1 3 - 4 ni 2 va 5 orasiga qo'yamiz
[2, 4, 5, 6] 1 3 - 6 ni oxiriga qo'yamiz
[1, 2, 4, 5, 6] 3 - 1 ni boshiga qo'yamiz
[1, 2, 3, 4, 5, 6] - 3 ni 2 va 4 orasiga qo'yamiz
\`\`\`

**Vaqt murakkabligi:** O(n^2) eng yomon/o'rtacha, O(n) eng yaxshi
**Xotira murakkabligi:** O(1)`,
			hint1: `Avval joriy elementni (key) saqlang. Keyin to'g'ri joyni topguncha katta elementlarni o'ngga siljiting.`,
			hint2: `j >= 0 VA arr[j] > key bo'lguncha davom etadigan while siklidan foydalaning. Bu elementlarni siljitadi va qo'yish nuqtasini topadi.`,
			whyItMatters: `Qo'yish saralash kichik yoki deyarli saralangan ma'lumotlar uchun samarali.

**Bu nima uchun muhim:**

**1. Kichik ma'lumotlar uchun eng yaxshi**

Qo'yish saralash ko'pincha kichik massivlar uchun ishlatiladi.

**2. Adaptiv algoritm**

Deyarli saralangan ma'lumotlar uchun O(n) da ishlaydi.

**3. Onlayn saralash**

Elementlar kelganda saralashi mumkin.

**4. Barqaror saralash**

Teng elementlarning nisbiy tartibini saqlaydi.`,
			solutionCode: `from typing import List


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Massivni qo'yish saralash algoritmi bilan saralaydi.

    Args:
        arr: Saralash uchun butun sonlar ro'yxati

    Returns:
        O'sish tartibida saralangan ro'yxat
    """
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # key dan katta elementlarni o'ngga siljitamiz
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # key ni to'g'ri joyga qo'yamiz
        arr[j + 1] = key

    return arr`
		}
	}
};

export default task;
