import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-binary-search',
	title: 'Binary Search',
	difficulty: 'easy',
	tags: ['python', 'searching', 'binary-search', 'divide-and-conquer'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement binary search to find a target value in a sorted array.

**Problem:**

Given a sorted array of integers \`nums\` and a target value \`target\`, return the index of \`target\` if it exists, otherwise return \`-1\`.

**Examples:**

\`\`\`
Input: nums = [-1, 0, 3, 5, 9, 12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1, 0, 3, 5, 9, 12], target = 2
Output: -1
Explanation: 2 does not exist in nums

Input: nums = [5], target = 5
Output: 0
\`\`\`

**Algorithm:**

Binary search divides the search space in half each iteration:
1. Set left = 0, right = len(nums) - 1
2. While left <= right:
   - Calculate mid = (left + right) // 2
   - If nums[mid] == target, return mid
   - If nums[mid] < target, search right half (left = mid + 1)
   - If nums[mid] > target, search left half (right = mid - 1)
3. Return -1 if not found

**Constraints:**
- 1 <= nums.length <= 10^4
- All integers in nums are unique
- nums is sorted in ascending order

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    # TODO: Find target in sorted array using binary search

    return -1`,
	solutionCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    """
    Search for target in sorted array using binary search.

    Args:
        nums: Sorted array of integers
        target: Value to find

    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        # Avoid overflow: mid = left + (right - left) // 2
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Found!
        elif nums[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half

    return -1  # Not found


# Recursive version
def binary_search_recursive(nums: List[int], target: int) -> int:
    """Recursive binary search implementation."""
    def search(left: int, right: int) -> int:
        if left > right:
            return -1

        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            return search(mid + 1, right)
        else:
            return search(left, mid - 1)

    return search(0, len(nums) - 1)`,
	testCode: `import pytest
from solution import binary_search

class TestBinarySearch:
    def test_found_in_middle(self):
        """Test finding element in middle"""
        assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4

    def test_not_found(self):
        """Test element not in array"""
        assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1

    def test_single_element_found(self):
        """Test single element array - found"""
        assert binary_search([5], 5) == 0

    def test_single_element_not_found(self):
        """Test single element array - not found"""
        assert binary_search([5], 3) == -1

    def test_first_element(self):
        """Test finding first element"""
        assert binary_search([1, 2, 3, 4, 5], 1) == 0

    def test_last_element(self):
        """Test finding last element"""
        assert binary_search([1, 2, 3, 4, 5], 5) == 4

    def test_two_elements(self):
        """Test two element array"""
        assert binary_search([1, 3], 1) == 0
        assert binary_search([1, 3], 3) == 1
        assert binary_search([1, 3], 2) == -1

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert binary_search([-10, -5, 0, 5, 10], -5) == 1

    def test_large_array(self):
        """Test with larger array"""
        nums = list(range(0, 1000, 2))  # Even numbers 0-998
        assert binary_search(nums, 500) == 250
        assert binary_search(nums, 501) == -1

    def test_target_smaller_than_all(self):
        """Test target smaller than all elements"""
        assert binary_search([5, 10, 15, 20, 25], 1) == -1`,
	hint1: `Use two pointers (left and right) and repeatedly halve the search space. Calculate mid = (left + right) // 2 and compare nums[mid] with target.`,
	hint2: `If nums[mid] < target, the target must be in the right half, so set left = mid + 1. If nums[mid] > target, search the left half with right = mid - 1.`,
	whyItMatters: `Binary Search is the foundation of efficient searching and appears in countless algorithms and system designs.

**Why This Matters:**

**1. O(log n) is Powerful**

\`\`\`python
# Linear search: O(n)
# For 1 billion elements: 1,000,000,000 operations

# Binary search: O(log n)
# For 1 billion elements: ~30 operations

# That's 33 million times faster!
\`\`\`

**2. The Template is Universal**

\`\`\`python
# This pattern applies to many problems:
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    if condition(mid):
        # adjust left or right
    else:
        # adjust the other
\`\`\`

**3. Common Variations**

- Find first/last occurrence
- Find insertion position
- Search in rotated array
- Find peak element
- Search answer space (optimization)

**4. Real-World Applications**

- Database indexing (B-trees use binary search)
- Version control (git bisect)
- IP routing tables
- Auto-complete suggestions`,
	order: 1,
	translations: {
		ru: {
			title: 'Бинарный поиск',
			description: `Реализуйте бинарный поиск для нахождения целевого значения в отсортированном массиве.

**Задача:**

Дан отсортированный массив целых чисел \`nums\` и целевое значение \`target\`, верните индекс \`target\` если он существует, иначе верните \`-1\`.

**Примеры:**

\`\`\`
Вход: nums = [-1, 0, 3, 5, 9, 12], target = 9
Выход: 4
Объяснение: 9 существует в nums с индексом 4

Вход: nums = [-1, 0, 3, 5, 9, 12], target = 2
Выход: -1
Объяснение: 2 не существует в nums
\`\`\`

**Алгоритм:**

Бинарный поиск делит пространство поиска пополам на каждой итерации:
1. Установите left = 0, right = len(nums) - 1
2. Пока left <= right:
   - Вычислите mid = (left + right) // 2
   - Если nums[mid] == target, верните mid
   - Если nums[mid] < target, ищите в правой половине
   - Если nums[mid] > target, ищите в левой половине
3. Верните -1 если не найдено

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `Используйте два указателя (left и right) и последовательно уменьшайте пространство поиска вдвое. Вычислите mid = (left + right) // 2 и сравните nums[mid] с target.`,
			hint2: `Если nums[mid] < target, цель в правой половине, установите left = mid + 1. Если nums[mid] > target, ищите в левой половине с right = mid - 1.`,
			whyItMatters: `Бинарный поиск - основа эффективного поиска, появляется в бесчисленных алгоритмах и системном дизайне.

**Почему это важно:**

**1. O(log n) - это мощно**

Для 1 миллиарда элементов: линейный поиск - миллиард операций, бинарный - около 30.

**2. Шаблон универсален**

Этот паттерн применяется ко многим задачам: поиск первого/последнего вхождения, позиция вставки, поиск в повёрнутом массиве.

**3. Применение в реальном мире**

Индексирование баз данных (B-деревья), git bisect, таблицы IP-маршрутизации.`,
			solutionCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    """
    Поиск target в отсортированном массиве бинарным поиском.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Нашли!
        elif nums[mid] < target:
            left = mid + 1  # Ищем в правой половине
        else:
            right = mid - 1  # Ищем в левой половине

    return -1  # Не найдено`
		},
		uz: {
			title: 'Binar qidiruv',
			description: `Tartiblangan massivda maqsadli qiymatni topish uchun binar qidiruvni amalga oshiring.

**Masala:**

Tartiblangan butun sonlar massivi \`nums\` va maqsadli qiymat \`target\` berilgan, agar mavjud bo'lsa \`target\` indeksini, aks holda \`-1\` qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [-1, 0, 3, 5, 9, 12], target = 9
Chiqish: 4
Tushuntirish: 9 nums da mavjud va uning indeksi 4

Kirish: nums = [-1, 0, 3, 5, 9, 12], target = 2
Chiqish: -1
Tushuntirish: 2 nums da mavjud emas
\`\`\`

**Algoritm:**

Binar qidiruv har bir iteratsiyada qidiruv maydonini yarmiga bo'ladi:
1. left = 0, right = len(nums) - 1 o'rnating
2. left <= right bo'lgunicha:
   - mid = (left + right) // 2 hisoblang
   - Agar nums[mid] == target bo'lsa, mid qaytaring
   - Agar nums[mid] < target bo'lsa, o'ng yarmida qidiring
   - Agar nums[mid] > target bo'lsa, chap yarmida qidiring
3. Topilmasa -1 qaytaring

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikki ko'rsatkich (left va right) ishlating va qidiruv maydonini ketma-ket yarmiga qisqartiring. mid = (left + right) // 2 hisoblang va nums[mid] ni target bilan solishtiring.`,
			hint2: `Agar nums[mid] < target bo'lsa, maqsad o'ng yarmida, left = mid + 1 o'rnating. Agar nums[mid] > target bo'lsa, right = mid - 1 bilan chap yarmida qidiring.`,
			whyItMatters: `Binar qidiruv - samarali qidiruvning asosi, son-sanoqsiz algoritmlarda va tizim dizaynida uchraydi.

**Bu nima uchun muhim:**

**1. O(log n) kuchli**

1 milliard element uchun: chiziqli qidiruv - milliard operatsiya, binar - taxminan 30.

**2. Shablon universal**

Bu pattern ko'p masalalarga qo'llaniladi: birinchi/oxirgi topish, qo'shish pozitsiyasi, aylantirilgan massivda qidiruv.

**3. Haqiqiy dunyoda qo'llanilishi**

Ma'lumotlar bazasi indekslash (B-daraxtlar), git bisect, IP marshrutlash jadvallari.`,
			solutionCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    """
    Binar qidiruv yordamida tartiblangan massivda target ni qidiradi.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Topdik!
        elif nums[mid] < target:
            left = mid + 1  # O'ng yarmida qidiramiz
        else:
            right = mid - 1  # Chap yarmida qidiramiz

    return -1  # Topilmadi`
		}
	}
};

export default task;
