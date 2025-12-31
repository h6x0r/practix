import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-find-minimum-rotated',
	title: 'Find Minimum in Rotated Array',
	difficulty: 'medium',
	tags: ['python', 'searching', 'binary-search', 'array'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the minimum element in a rotated sorted array.

**Problem:**

Suppose an array of length \`n\` sorted in ascending order is **rotated** between 1 and n times. For example, the array \`nums = [0, 1, 2, 4, 5, 6, 7]\` might become:
- \`[4, 5, 6, 7, 0, 1, 2]\` if rotated 4 times
- \`[0, 1, 2, 4, 5, 6, 7]\` if rotated 7 times

Given the sorted rotated array \`nums\` of **unique** elements, return the minimum element.

You must write an algorithm that runs in O(log n) time.

**Examples:**

\`\`\`
Input: nums = [3, 4, 5, 1, 2]
Output: 1
Explanation: Original array was [1, 2, 3, 4, 5], rotated 3 times

Input: nums = [4, 5, 6, 7, 0, 1, 2]
Output: 0

Input: nums = [11, 13, 15, 17]
Output: 11
Explanation: Array was not rotated (or rotated n times)
\`\`\`

**Key Insight:**

Compare \`nums[mid]\` with \`nums[right]\`:
- If \`nums[mid] > nums[right]\`: minimum is in right half
- Otherwise: minimum is in left half (including mid)

**Constraints:**
- n == nums.length
- 1 <= n <= 5000
- All integers are unique
- nums is sorted and rotated

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def find_min(nums: List[int]) -> int:
    # TODO: Find minimum element in rotated sorted array

    return 0`,
	solutionCode: `from typing import List

def find_min(nums: List[int]) -> int:
    """
    Find minimum element in rotated sorted array.

    Args:
        nums: Rotated sorted array with unique elements

    Returns:
        The minimum element
    """
    left, right = 0, len(nums) - 1

    # If not rotated or single element
    if nums[left] <= nums[right]:
        return nums[left]

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            # nums[mid] is in the "larger" sorted portion
            # Minimum must be to the right
            left = mid + 1
        else:
            # nums[mid] is in the "smaller" sorted portion
            # Minimum is at mid or to the left
            right = mid

    return nums[left]


# Alternative: Compare with nums[0]
def find_min_compare_first(nums: List[int]) -> int:
    """Compare mid with first element."""
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] >= nums[0]:
            # Still in the larger sorted portion
            left = mid + 1
        else:
            # In the smaller sorted portion
            right = mid

    # Check if array wasn't rotated
    if nums[left] > nums[0]:
        return nums[0]

    return nums[left]


# Find rotation point (index of minimum)
def find_rotation_point(nums: List[int]) -> int:
    """Find the index of minimum element (rotation point)."""
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return left`,
	testCode: `import pytest
from solution import find_min

class TestFindMinRotated:
    def test_rotated_middle(self):
        """Test rotated with min in middle"""
        assert find_min([3, 4, 5, 1, 2]) == 1

    def test_rotated_beginning(self):
        """Test min near beginning of rotated part"""
        assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0

    def test_not_rotated(self):
        """Test array that's not rotated"""
        assert find_min([11, 13, 15, 17]) == 11

    def test_single_element(self):
        """Test single element"""
        assert find_min([1]) == 1

    def test_two_elements_rotated(self):
        """Test two elements - rotated"""
        assert find_min([2, 1]) == 1

    def test_two_elements_not_rotated(self):
        """Test two elements - not rotated"""
        assert find_min([1, 2]) == 1

    def test_rotated_once(self):
        """Test array rotated once"""
        assert find_min([2, 3, 4, 5, 1]) == 1

    def test_rotated_to_original(self):
        """Test fully rotated (back to original)"""
        assert find_min([1, 2, 3, 4, 5]) == 1

    def test_min_at_end(self):
        """Test minimum at last position"""
        assert find_min([2, 3, 4, 5, 6, 1]) == 1

    def test_large_rotated_array(self):
        """Test with larger rotated array"""
        assert find_min([7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]) == 1`,
	hint1: `In a rotated sorted array, there are two sorted portions. Compare nums[mid] with nums[right] to determine which portion mid is in.`,
	hint2: `If nums[mid] > nums[right], the minimum must be in the right half (mid+1 to right). Otherwise, the minimum is in the left half (left to mid, inclusive of mid).`,
	whyItMatters: `Finding minimum in rotated array teaches binary search with structural properties beyond simple ordering.

**Why This Matters:**

**1. Understanding Rotation**

\`\`\`python
# Original: [1, 2, 3, 4, 5]
# Rotated:  [3, 4, 5, 1, 2]
#           ─────┬─ ───┬──
#           larger    smaller
#           portion   portion

# The minimum is where the "break" happens
\`\`\`

**2. The Comparison Key**

\`\`\`python
# Why compare with nums[right]?
# If nums[mid] > nums[right]: mid is in larger portion
# If nums[mid] < nums[right]: mid is in smaller portion

# This works because:
# - If not rotated, nums[mid] <= nums[right] always
# - If rotated, the inflection point divides the array
\`\`\`

**3. Related Problems**

- Search in Rotated Sorted Array
- Find Rotation Count
- Search in Rotated Array II (with duplicates)

**4. Real-World Applications**

- Log rotation in system administration
- Circular buffer analysis
- Time series with wrap-around`,
	order: 5,
	translations: {
		ru: {
			title: 'Минимум в повёрнутом массиве',
			description: `Найдите минимальный элемент в повёрнутом отсортированном массиве.

**Задача:**

Пусть массив длины \`n\`, отсортированный по возрастанию, **повёрнут** от 1 до n раз. Например, \`nums = [0, 1, 2, 4, 5, 6, 7]\` может стать:
- \`[4, 5, 6, 7, 0, 1, 2]\` при 4 поворотах
- \`[0, 1, 2, 4, 5, 6, 7]\` при 7 поворотах

Дан повёрнутый отсортированный массив \`nums\` из **уникальных** элементов, верните минимальный элемент.

Алгоритм должен работать за O(log n).

**Примеры:**

\`\`\`
Вход: nums = [3, 4, 5, 1, 2]
Выход: 1

Вход: nums = [4, 5, 6, 7, 0, 1, 2]
Выход: 0

Вход: nums = [11, 13, 15, 17]
Выход: 11 (массив не повёрнут)
\`\`\`

**Ключевая идея:**

Сравните \`nums[mid]\` с \`nums[right]\`:
- Если \`nums[mid] > nums[right]\`: минимум в правой половине
- Иначе: минимум в левой половине (включая mid)

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `В повёрнутом отсортированном массиве есть две отсортированные части. Сравните nums[mid] с nums[right] чтобы определить в какой части находится mid.`,
			hint2: `Если nums[mid] > nums[right], минимум в правой половине (mid+1 до right). Иначе минимум в левой половине (left до mid включительно).`,
			whyItMatters: `Поиск минимума в повёрнутом массиве учит бинарному поиску со структурными свойствами помимо простого порядка.

**Почему это важно:**

**1. Понимание поворота**

Есть две отсортированные части: большая и меньшая. Минимум там, где "разрыв".

**2. Ключ сравнения**

Сравнение с nums[right] определяет в какой части находится mid.

**3. Применение**

Ротация логов, кольцевые буферы, временные ряды с переносом.`,
			solutionCode: `from typing import List

def find_min(nums: List[int]) -> int:
    """
    Находит минимальный элемент в повёрнутом отсортированном массиве.
    """
    left, right = 0, len(nums) - 1

    # Если не повёрнут или один элемент
    if nums[left] <= nums[right]:
        return nums[left]

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            # nums[mid] в "большей" отсортированной части
            # Минимум должен быть справа
            left = mid + 1
        else:
            # nums[mid] в "меньшей" отсортированной части
            # Минимум на mid или слева
            right = mid

    return nums[left]`
		},
		uz: {
			title: 'Aylantirilgan massivdagi minimum',
			description: `Aylantirilgan tartiblangan massivda minimal elementni toping.

**Masala:**

O'sish tartibida tartiblangan \`n\` uzunlikdagi massiv 1 dan n gacha marta **aylantirilgan** deb faraz qiling. Masalan, \`nums = [0, 1, 2, 4, 5, 6, 7]\` massiv quyidagicha bo'lishi mumkin:
- \`[4, 5, 6, 7, 0, 1, 2]\` agar 4 marta aylantirilsa
- \`[0, 1, 2, 4, 5, 6, 7]\` agar 7 marta aylantirilsa

**Noyob** elementlardan iborat aylantirilgan tartiblangan \`nums\` massivi berilgan, minimal elementni qaytaring.

O(log n) vaqtda ishlaydigan algoritm yozing.

**Misollar:**

\`\`\`
Kirish: nums = [3, 4, 5, 1, 2]
Chiqish: 1

Kirish: nums = [4, 5, 6, 7, 0, 1, 2]
Chiqish: 0

Kirish: nums = [11, 13, 15, 17]
Chiqish: 11 (massiv aylantirilmagan)
\`\`\`

**Asosiy tushuncha:**

\`nums[mid]\` ni \`nums[right]\` bilan solishtiring:
- Agar \`nums[mid] > nums[right]\`: minimum o'ng yarmida
- Aks holda: minimum chap yarmida (mid ni o'z ichiga oladi)

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Aylantirilgan tartiblangan massivda ikkita tartiblangan qism bor. mid qaysi qismda ekanligini aniqlash uchun nums[mid] ni nums[right] bilan solishtiring.`,
			hint2: `Agar nums[mid] > nums[right] bo'lsa, minimum o'ng yarmida (mid+1 dan right gacha). Aks holda minimum chap yarmida (left dan mid gacha, mid ni o'z ichiga oladi).`,
			whyItMatters: `Aylantirilgan massivda minimum topish oddiy tartibdan tashqari strukturaviy xususiyatlar bilan binar qidiruvni o'rgatadi.

**Bu nima uchun muhim:**

**1. Aylantirishni tushunish**

Ikkita tartiblangan qism bor: kattasi va kichigi. Minimum "uzilish" joyida.

**2. Solishtirish kaliti**

nums[right] bilan solishtirish mid qaysi qismda ekanligini aniqlaydi.

**3. Qo'llanishlar**

Log rotatsiyasi, halqasimon buferlar, o'rab oladigan vaqt qatorlari.`,
			solutionCode: `from typing import List

def find_min(nums: List[int]) -> int:
    """
    Aylantirilgan tartiblangan massivda minimal elementni topadi.
    """
    left, right = 0, len(nums) - 1

    # Agar aylantirilmagan yoki bitta element bo'lsa
    if nums[left] <= nums[right]:
        return nums[left]

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            # nums[mid] "katta" tartiblangan qismda
            # Minimum o'ngda bo'lishi kerak
            left = mid + 1
        else:
            # nums[mid] "kichik" tartiblangan qismda
            # Minimum mid da yoki chapda
            right = mid

    return nums[left]`
		}
	}
};

export default task;
