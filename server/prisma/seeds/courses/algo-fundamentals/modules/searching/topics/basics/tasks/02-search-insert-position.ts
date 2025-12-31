import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-search-insert-position',
	title: 'Search Insert Position',
	difficulty: 'easy',
	tags: ['python', 'searching', 'binary-search'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the index where a target should be inserted in a sorted array.

**Problem:**

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

**Examples:**

\`\`\`
Input: nums = [1, 3, 5, 6], target = 5
Output: 2
Explanation: 5 is found at index 2

Input: nums = [1, 3, 5, 6], target = 2
Output: 1
Explanation: 2 should be inserted at index 1 (between 1 and 3)

Input: nums = [1, 3, 5, 6], target = 7
Output: 4
Explanation: 7 should be inserted at index 4 (at the end)

Input: nums = [1, 3, 5, 6], target = 0
Output: 0
Explanation: 0 should be inserted at index 0 (at the beginning)
\`\`\`

**Key Insight:**

This is finding the "lower bound" - the first position where we can insert target while keeping the array sorted. When binary search ends, \`left\` points to this position.

**Constraints:**
- 1 <= nums.length <= 10^4
- -10^4 <= nums[i] <= 10^4
- nums contains distinct values sorted in ascending order

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    # TODO: Return index of target or where it should be inserted

    return 0`,
	solutionCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Find index to insert target in sorted array.

    Args:
        nums: Sorted array of distinct integers
        target: Value to find or insert

    Returns:
        Index where target is or should be inserted
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Found exact match
        elif nums[mid] < target:
            left = mid + 1  # Search right
        else:
            right = mid - 1  # Search left

    # When loop ends, left is the insertion point
    # left > right, so left points to first element >= target
    return left


# Using bisect module (Python standard library)
import bisect

def search_insert_bisect(nums: List[int], target: int) -> int:
    """Using Python's bisect module."""
    return bisect.bisect_left(nums, target)


# Alternative: Find lower bound explicitly
def search_insert_lower_bound(nums: List[int], target: int) -> int:
    """Find lower bound - first index where nums[i] >= target."""
    left, right = 0, len(nums)

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left`,
	testCode: `import pytest
from solution import search_insert

class TestSearchInsert:
    def test_found_in_array(self):
        """Test when target exists"""
        assert search_insert([1, 3, 5, 6], 5) == 2

    def test_insert_middle(self):
        """Test insertion in middle"""
        assert search_insert([1, 3, 5, 6], 2) == 1

    def test_insert_end(self):
        """Test insertion at end"""
        assert search_insert([1, 3, 5, 6], 7) == 4

    def test_insert_beginning(self):
        """Test insertion at beginning"""
        assert search_insert([1, 3, 5, 6], 0) == 0

    def test_single_element_found(self):
        """Test single element - found"""
        assert search_insert([1], 1) == 0

    def test_single_element_before(self):
        """Test single element - insert before"""
        assert search_insert([1], 0) == 0

    def test_single_element_after(self):
        """Test single element - insert after"""
        assert search_insert([1], 2) == 1

    def test_two_elements(self):
        """Test two element array"""
        assert search_insert([1, 3], 2) == 1
        assert search_insert([1, 3], 0) == 0
        assert search_insert([1, 3], 4) == 2

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert search_insert([-5, -3, 0, 3, 5], -4) == 1
        assert search_insert([-5, -3, 0, 3, 5], -6) == 0

    def test_large_array(self):
        """Test with larger array"""
        nums = list(range(0, 100, 5))  # [0, 5, 10, 15, ..., 95]
        assert search_insert(nums, 47) == 10
        assert search_insert(nums, 100) == 20`,
	hint1: `This is standard binary search with a twist: when the target isn't found, the left pointer ends up at the correct insertion position.`,
	hint2: `After the while loop (when left > right), left points to the first element >= target, which is exactly where we should insert. Just return left.`,
	whyItMatters: `Search Insert Position teaches the "lower bound" concept - essential for many binary search variations.

**Why This Matters:**

**1. Lower Bound vs Upper Bound**

\`\`\`python
# Lower bound: first index where nums[i] >= target
# Upper bound: first index where nums[i] > target

# Python's bisect module:
bisect.bisect_left(nums, target)   # lower bound
bisect.bisect_right(nums, target)  # upper bound
\`\`\`

**2. The "Insertion Point" Insight**

After binary search terminates:
- \`left\` = insertion point (first element >= target)
- \`right\` = last element < target

**3. Common Applications**

- Finding range of equal elements
- Counting elements in range
- Finding floor/ceiling values
- Implementing sorted containers

**4. Template for Bound Problems**

\`\`\`python
# Lower bound template
left, right = 0, len(nums)
while left < right:
    mid = (left + right) // 2
    if nums[mid] < target:
        left = mid + 1
    else:
        right = mid
return left
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Позиция для вставки',
			description: `Найдите индекс, куда должен быть вставлен элемент в отсортированном массиве.

**Задача:**

Дан отсортированный массив уникальных целых чисел и целевое значение, верните индекс если цель найдена. Если нет, верните индекс, куда она должна быть вставлена.

Алгоритм должен работать за O(log n).

**Примеры:**

\`\`\`
Вход: nums = [1, 3, 5, 6], target = 5
Выход: 2 (5 найдено на индексе 2)

Вход: nums = [1, 3, 5, 6], target = 2
Выход: 1 (2 должно быть вставлено на индекс 1)

Вход: nums = [1, 3, 5, 6], target = 7
Выход: 4 (вставить в конец)

Вход: nums = [1, 3, 5, 6], target = 0
Выход: 0 (вставить в начало)
\`\`\`

**Ключевая идея:**

Это поиск "нижней границы" - первой позиции, куда можно вставить target сохраняя порядок. После бинарного поиска \`left\` указывает на эту позицию.

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `Это стандартный бинарный поиск с нюансом: когда цель не найдена, указатель left оказывается на правильной позиции вставки.`,
			hint2: `После цикла while (когда left > right), left указывает на первый элемент >= target, что и есть позиция вставки. Просто верните left.`,
			whyItMatters: `Search Insert Position учит концепции "нижней границы" - важной для многих вариаций бинарного поиска.

**Почему это важно:**

**1. Нижняя vs Верхняя граница**

Нижняя граница: первый индекс где nums[i] >= target. Верхняя граница: первый индекс где nums[i] > target.

**2. Точка вставки**

После завершения бинарного поиска left = точка вставки (первый элемент >= target).

**3. Применения**

Поиск диапазона равных элементов, подсчёт элементов, floor/ceiling значения.`,
			solutionCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Находит индекс для вставки target в отсортированный массив.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Нашли точное совпадение
        elif nums[mid] < target:
            left = mid + 1  # Ищем справа
        else:
            right = mid - 1  # Ищем слева

    # По окончании цикла left указывает на позицию вставки
    return left`
		},
		uz: {
			title: 'Qo\'shish pozitsiyasi',
			description: `Tartiblangan massivda elementni qo'shish kerak bo'lgan indeksni toping.

**Masala:**

Tartiblangan noyob butun sonlar massivi va maqsadli qiymat berilgan, agar maqsad topilsa indeksni qaytaring. Agar topilmasa, tartibni saqlab qo'shilishi kerak bo'lgan indeksni qaytaring.

O(log n) vaqt murakkabligida algoritm yozing.

**Misollar:**

\`\`\`
Kirish: nums = [1, 3, 5, 6], target = 5
Chiqish: 2 (5 indeks 2 da topildi)

Kirish: nums = [1, 3, 5, 6], target = 2
Chiqish: 1 (2 indeks 1 ga qo'shilishi kerak)

Kirish: nums = [1, 3, 5, 6], target = 7
Chiqish: 4 (oxiriga qo'shish)

Kirish: nums = [1, 3, 5, 6], target = 0
Chiqish: 0 (boshiga qo'shish)
\`\`\`

**Asosiy tushuncha:**

Bu "pastki chegara" topish - tartibni saqlab target qo'shish mumkin bo'lgan birinchi pozitsiya. Binar qidiruvdan keyin \`left\` bu pozitsiyaga ishora qiladi.

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Bu standart binar qidiruv, lekin farqi: maqsad topilmaganda, left ko'rsatkichi to'g'ri qo'shish pozitsiyasida bo'ladi.`,
			hint2: `while tsiklidan keyin (left > right bo'lganda), left birinchi element >= target ga ishora qiladi, bu aynan qo'shish pozitsiyasi. Shunchaki left qaytaring.`,
			whyItMatters: `Search Insert Position "pastki chegara" tushunchasini o'rgatadi - binar qidiruvning ko'plab variantlari uchun muhim.

**Bu nima uchun muhim:**

**1. Pastki vs Yuqori chegara**

Pastki chegara: nums[i] >= target bo'lgan birinchi indeks. Yuqori chegara: nums[i] > target bo'lgan birinchi indeks.

**2. Qo'shish nuqtasi**

Binar qidiruv tugaganidan keyin left = qo'shish nuqtasi (birinchi element >= target).

**3. Qo'llanishlar**

Teng elementlar diapazonini topish, elementlarni hisoblash, floor/ceiling qiymatlari.`,
			solutionCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Tartiblangan massivga target qo'shish uchun indeksni topadi.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Aniq mos keldi
        elif nums[mid] < target:
            left = mid + 1  # O'ngda qidiramiz
        else:
            right = mid - 1  # Chapda qidiramiz

    # Tsikl tugaganida left qo'shish pozitsiyasiga ishora qiladi
    return left`
		}
	}
};

export default task;
