import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-search-2d-matrix',
	title: 'Search a 2D Matrix',
	difficulty: 'medium',
	tags: ['python', 'searching', 'binary-search', 'matrix'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Search for a target value in a sorted 2D matrix.

**Problem:**

You are given an \`m x n\` integer matrix \`matrix\` with the following properties:
- Each row is sorted in non-decreasing order
- The first integer of each row is greater than the last integer of the previous row

Given an integer \`target\`, return \`true\` if target is in the matrix, otherwise \`false\`.

You must write an algorithm with O(log(m * n)) runtime complexity.

**Examples:**

\`\`\`
Input: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 3
Output: true

Input: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 13
Output: false
\`\`\`

**Visual:**

\`\`\`
[1,  3,  5,  7]     Flattened: [1, 3, 5, 7, 10, 11, 16, 20, 24, 30, 34, 50]
[10, 11, 16, 20]    Index:     [0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11]
[24, 30, 34, 50]
\`\`\`

**Key Insight:**

Treat the 2D matrix as a 1D sorted array:
- Total elements: m * n
- Index i maps to matrix[i // n][i % n]

Then apply standard binary search!

**Constraints:**
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 100
- -10^4 <= matrix[i][j], target <= 10^4

**Time Complexity:** O(log(m * n))
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def search_matrix(matrix: List[List[int]], target: int) -> bool:
    # TODO: Search for target in sorted 2D matrix

    return False`,
	solutionCode: `from typing import List

def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Search for target in sorted 2D matrix.

    Args:
        matrix: Sorted 2D matrix
        target: Value to find

    Returns:
        True if target exists, False otherwise
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = (left + right) // 2
        # Convert 1D index to 2D coordinates
        row, col = mid // n, mid % n
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False


# Two binary searches approach
def search_matrix_two_searches(matrix: List[List[int]], target: int) -> bool:
    """
    First binary search to find row, then search within row.
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])

    # Binary search for the row
    top, bottom = 0, m - 1
    while top <= bottom:
        mid_row = (top + bottom) // 2
        if matrix[mid_row][0] > target:
            bottom = mid_row - 1
        elif matrix[mid_row][n - 1] < target:
            top = mid_row + 1
        else:
            # Target might be in this row
            break

    if top > bottom:
        return False

    # Binary search within the row
    row = (top + bottom) // 2
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if matrix[row][mid] == target:
            return True
        elif matrix[row][mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False


# Staircase search (for row-sorted + col-sorted matrices)
def search_matrix_staircase(matrix: List[List[int]], target: int) -> bool:
    """
    Start from top-right corner and move left or down.
    Works for matrices sorted by row AND column independently.
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1  # Start at top-right

    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            row += 1  # Move down
        else:
            col -= 1  # Move left

    return False`,
	testCode: `import pytest
from solution import search_matrix

class TestSearchMatrix:
    def test_found_in_first_row(self):
        """Test target in first row"""
        matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]]
        assert search_matrix(matrix, 3) == True

    def test_not_found(self):
        """Test target not in matrix"""
        matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]]
        assert search_matrix(matrix, 13) == False

    def test_found_in_last_row(self):
        """Test target in last row"""
        matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]]
        assert search_matrix(matrix, 30) == True

    def test_first_element(self):
        """Test first element"""
        matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]]
        assert search_matrix(matrix, 1) == True

    def test_last_element(self):
        """Test last element"""
        matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]]
        assert search_matrix(matrix, 50) == True

    def test_single_row(self):
        """Test single row matrix"""
        assert search_matrix([[1, 3, 5]], 3) == True
        assert search_matrix([[1, 3, 5]], 2) == False

    def test_single_column(self):
        """Test single column matrix"""
        assert search_matrix([[1], [3], [5]], 3) == True
        assert search_matrix([[1], [3], [5]], 2) == False

    def test_single_element(self):
        """Test single element matrix"""
        assert search_matrix([[5]], 5) == True
        assert search_matrix([[5]], 1) == False

    def test_empty_matrix(self):
        """Test empty matrix"""
        assert search_matrix([], 1) == False
        assert search_matrix([[]], 1) == False

    def test_larger_matrix(self):
        """Test with larger matrix"""
        matrix = [[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]]
        assert search_matrix(matrix, 5) == True
        assert search_matrix(matrix, 20) == False`,
	hint1: `The key insight is that this sorted 2D matrix can be viewed as a single sorted 1D array. If the matrix is m x n, element at index i in the flattened array is at matrix[i // n][i % n].`,
	hint2: `Apply standard binary search on indices 0 to m*n-1. For each mid index, convert to row and column: row = mid // n, col = mid % n. Then compare matrix[row][col] with target.`,
	whyItMatters: `Search in 2D Matrix shows how to extend binary search to multi-dimensional data.

**Why This Matters:**

**1. Index Transformation**

\`\`\`python
# 2D to 1D (flatten)
index = row * n + col

# 1D to 2D (unflatten)
row = index // n
col = index % n

# This works for any 2D -> 1D mapping!
\`\`\`

**2. Problem Variations**

\`\`\`python
# Type I: Rows sorted + next row > prev row (this problem)
# -> Treat as single sorted array

# Type II: Rows sorted + columns sorted (independently)
# -> Use staircase search from corner

# Different properties -> different algorithms!
\`\`\`

**3. The Staircase Search**

\`\`\`python
# For Type II matrices, start at top-right:
# - If target > current: move down
# - If target < current: move left
# O(m + n) time, works for broader class of matrices
\`\`\`

**4. Real-World Applications**

- Spreadsheet cell lookup
- Image pixel search
- Game board state checking
- Database table operations`,
	order: 6,
	translations: {
		ru: {
			title: 'Поиск в 2D матрице',
			description: `Найдите целевое значение в отсортированной 2D матрице.

**Задача:**

Дана целочисленная матрица \`m x n\` со следующими свойствами:
- Каждая строка отсортирована по неубыванию
- Первое число каждой строки больше последнего числа предыдущей строки

Дано целое число \`target\`, верните \`true\` если target есть в матрице, иначе \`false\`.

Алгоритм должен работать за O(log(m * n)).

**Примеры:**

\`\`\`
Вход: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 3
Выход: true

Вход: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 13
Выход: false
\`\`\`

**Ключевая идея:**

Рассматривайте 2D матрицу как 1D отсортированный массив:
- Всего элементов: m * n
- Индекс i соответствует matrix[i // n][i % n]

Затем применяйте стандартный бинарный поиск!

**Временная сложность:** O(log(m * n))
**Пространственная сложность:** O(1)`,
			hint1: `Ключевая идея в том, что эта отсортированная 2D матрица может рассматриваться как один отсортированный 1D массив. Элемент с индексом i в развёрнутом массиве находится в matrix[i // n][i % n].`,
			hint2: `Применяйте стандартный бинарный поиск по индексам от 0 до m*n-1. Для каждого mid преобразуйте в строку и столбец: row = mid // n, col = mid % n.`,
			whyItMatters: `Поиск в 2D матрице показывает как расширить бинарный поиск на многомерные данные.

**Почему это важно:**

**1. Преобразование индексов**

2D в 1D: index = row * n + col. 1D в 2D: row = index // n, col = index % n.

**2. Вариации задачи**

Тип I: строки отсортированы + следующая строка > предыдущей - как один массив.
Тип II: строки и столбцы отсортированы независимо - поиск "лестницей" из угла.

**3. Применения**

Поиск в таблицах, операции с изображениями, проверка состояния игрового поля.`,
			solutionCode: `from typing import List

def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Ищет target в отсортированной 2D матрице.
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = (left + right) // 2
        # Преобразуем 1D индекс в 2D координаты
        row, col = mid // n, mid % n
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False`
		},
		uz: {
			title: '2D matritsada qidiruv',
			description: `Tartiblangan 2D matritsada maqsadli qiymatni toping.

**Masala:**

Quyidagi xususiyatlarga ega \`m x n\` butun sonlar matritsasi berilgan:
- Har bir qator kamaymaydigan tartibda tartiblangan
- Har bir qatorning birinchi soni oldingi qatorning oxirgi sonidan katta

Butun son \`target\` berilgan, agar target matritsada bo'lsa \`true\`, aks holda \`false\` qaytaring.

O(log(m * n)) vaqt murakkabligida algoritm yozing.

**Misollar:**

\`\`\`
Kirish: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 3
Chiqish: true

Kirish: matrix = [[1,3,5,7],[10,11,16,20],[24,30,34,50]], target = 13
Chiqish: false
\`\`\`

**Asosiy tushuncha:**

2D matritsani 1D tartiblangan massiv sifatida ko'ring:
- Jami elementlar: m * n
- Indeks i matrix[i // n][i % n] ga mos keladi

Keyin standart binar qidiruvni qo'llang!

**Vaqt murakkabligi:** O(log(m * n))
**Xotira murakkabligi:** O(1)`,
			hint1: `Asosiy tushuncha shuki, bu tartiblangan 2D matritsa bitta tartiblangan 1D massiv sifatida ko'rilishi mumkin. Tekislangan massivda i indeksidagi element matrix[i // n][i % n] da joylashgan.`,
			hint2: `0 dan m*n-1 gacha indekslarda standart binar qidiruv qo'llang. Har bir mid uchun qator va ustunga o'zgartiring: row = mid // n, col = mid % n.`,
			whyItMatters: `2D matritsada qidiruv binar qidiruvni ko'p o'lchamli ma'lumotlarga qanday kengaytirishni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Indeks o'zgartirishlari**

2D dan 1D ga: index = row * n + col. 1D dan 2D ga: row = index // n, col = index % n.

**2. Masala variantlari**

I tur: qatorlar tartiblangan + keyingi qator > oldingi - bitta massiv sifatida.
II tur: qatorlar va ustunlar mustaqil tartiblangan - burchakdan "zinalarmon" qidiruv.

**3. Qo'llanishlar**

Jadvallarda qidiruv, rasm operatsiyalari, o'yin taxtasi holatini tekshirish.`,
			solutionCode: `from typing import List

def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Tartiblangan 2D matritsada target ni qidiradi.
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = (left + right) // 2
        # 1D indeksni 2D koordinatalarga o'zgartiramiz
        row, col = mid // n, mid % n
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False`
		}
	}
};

export default task;
