import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-find-peak-element',
	title: 'Find Peak Element',
	difficulty: 'medium',
	tags: ['python', 'searching', 'binary-search'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find a peak element in an array where a peak is greater than its neighbors.

**Problem:**

A peak element is an element that is strictly greater than its neighbors.

Given an integer array \`nums\`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any** of the peaks.

You may imagine that \`nums[-1] = nums[n] = -∞\`. This means elements at the boundaries only need to be greater than their one neighbor.

You must write an algorithm that runs in O(log n) time.

**Examples:**

\`\`\`
Input: nums = [1, 2, 3, 1]
Output: 2
Explanation: 3 is a peak element (index 2), greater than neighbors 2 and 1

Input: nums = [1, 2, 1, 3, 5, 6, 4]
Output: 5 (or 1)
Explanation: Both index 1 (value 2) and index 5 (value 6) are peaks

Input: nums = [1]
Output: 0
Explanation: Single element is always a peak
\`\`\`

**Key Insight:**

If \`nums[mid] < nums[mid + 1]\`, a peak must exist on the right side (since we're going "uphill"). Otherwise, a peak exists on the left side (including mid).

**Constraints:**
- 1 <= nums.length <= 1000
- -2^31 <= nums[i] <= 2^31 - 1
- nums[i] != nums[i + 1] for all valid i

**Time Complexity:** O(log n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    # TODO: Find and return index of any peak element

    return 0`,
	solutionCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    """
    Find index of any peak element using binary search.

    Args:
        nums: Array of integers (adjacent elements are different)

    Returns:
        Index of a peak element
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            # Going uphill to the right, peak must be on right
            left = mid + 1
        else:
            # Going downhill or at peak, peak is on left (including mid)
            right = mid

    # left == right, this is the peak
    return left


# Recursive version
def find_peak_recursive(nums: List[int]) -> int:
    """Recursive binary search for peak."""
    def search(left: int, right: int) -> int:
        if left == right:
            return left

        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            return search(mid + 1, right)
        else:
            return search(left, mid)

    return search(0, len(nums) - 1)


# Linear scan (O(n) - for comparison)
def find_peak_linear(nums: List[int]) -> int:
    """Simple linear scan to find peak."""
    for i in range(len(nums)):
        is_peak = True
        if i > 0 and nums[i] <= nums[i - 1]:
            is_peak = False
        if i < len(nums) - 1 and nums[i] <= nums[i + 1]:
            is_peak = False
        if is_peak:
            return i
    return 0`,
	testCode: `import pytest
from solution import find_peak_element

class TestFindPeakElement:
    def test_peak_in_middle(self):
        """Test peak in middle of array"""
        result = find_peak_element([1, 2, 3, 1])
        assert result == 2

    def test_multiple_peaks(self):
        """Test array with multiple peaks"""
        result = find_peak_element([1, 2, 1, 3, 5, 6, 4])
        # Either peak is valid
        assert result in [1, 5]

    def test_single_element(self):
        """Test single element array"""
        assert find_peak_element([1]) == 0

    def test_two_elements_first_peak(self):
        """Test two elements - first is peak"""
        assert find_peak_element([2, 1]) == 0

    def test_two_elements_second_peak(self):
        """Test two elements - second is peak"""
        assert find_peak_element([1, 2]) == 1

    def test_ascending(self):
        """Test ascending array - last element is peak"""
        result = find_peak_element([1, 2, 3, 4, 5])
        assert result == 4

    def test_descending(self):
        """Test descending array - first element is peak"""
        result = find_peak_element([5, 4, 3, 2, 1])
        assert result == 0

    def test_peak_at_start(self):
        """Test peak at array start"""
        result = find_peak_element([5, 1, 2, 3, 4])
        # 5 is a peak (greater than right neighbor, left is -inf)
        # 4 is also a peak (greater than left neighbor, right is -inf)
        nums = [5, 1, 2, 3, 4]
        peak_idx = result
        # Verify it's actually a peak
        is_peak = True
        if peak_idx > 0 and nums[peak_idx] <= nums[peak_idx - 1]:
            is_peak = False
        if peak_idx < len(nums) - 1 and nums[peak_idx] <= nums[peak_idx + 1]:
            is_peak = False
        assert is_peak

    def test_plateau_shape(self):
        """Test array with plateau in middle"""
        result = find_peak_element([1, 2, 3, 4, 3, 2, 1])
        assert result == 3  # Peak at index 3 (value 4)

    def test_negative_numbers(self):
        """Test with negative numbers"""
        result = find_peak_element([-5, -2, -1, -3, -10])
        # Peak should be at index 2 (value -1)
        nums = [-5, -2, -1, -3, -10]
        peak_idx = result
        is_peak = True
        if peak_idx > 0 and nums[peak_idx] <= nums[peak_idx - 1]:
            is_peak = False
        if peak_idx < len(nums) - 1 and nums[peak_idx] <= nums[peak_idx + 1]:
            is_peak = False
        assert is_peak`,
	hint1: `Think about what happens if nums[mid] < nums[mid + 1]. You're on an "upward slope" - there must be a peak somewhere to the right (or at the end).`,
	hint2: `Use binary search with left < right. If going uphill (nums[mid] < nums[mid + 1]), set left = mid + 1. Otherwise, set right = mid. When left == right, you've found the peak.`,
	whyItMatters: `Find Peak Element demonstrates binary search on non-sorted arrays by using a different invariant.

**Why This Matters:**

**1. Binary Search Beyond Sorting**

\`\`\`python
# Classic binary search: array is sorted
# Peak finding: array has a property (peak exists)

# The key insight: if going uphill, peak is ahead
# This gives us O(log n) even without sorting
\`\`\`

**2. The "Uphill/Downhill" Intuition**

\`\`\`python
# nums = [1, 2, 3, 1]
#         ↗  ↗  ↘
# At any point, go towards the higher side
# You'll eventually reach a peak
\`\`\`

**3. Applications**

- Finding local maxima in data
- Optimization problems (gradient ascent)
- Signal processing (finding peaks)
- Stock price analysis

**4. Template for "Any Peak" Problems**

\`\`\`python
while left < right:  # Note: < not <=
    mid = (left + right) // 2
    if going_up(mid):
        left = mid + 1
    else:
        right = mid  # mid could be the answer
return left
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Поиск пикового элемента',
			description: `Найдите пиковый элемент в массиве, где пик больше своих соседей.

**Задача:**

Пиковый элемент - элемент, строго больший своих соседей.

Дан массив целых чисел \`nums\`, найдите пиковый элемент и верните его индекс. Если массив содержит несколько пиков, верните индекс **любого** из них.

Считайте, что \`nums[-1] = nums[n] = -∞\`. Элементы на границах должны быть больше только одного соседа.

Алгоритм должен работать за O(log n).

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3, 1]
Выход: 2
Объяснение: 3 - пиковый элемент (индекс 2)

Вход: nums = [1, 2, 1, 3, 5, 6, 4]
Выход: 5 (или 1)
Объяснение: Оба индекса 1 и 5 - пики

Вход: nums = [1]
Выход: 0
\`\`\`

**Ключевая идея:**

Если \`nums[mid] < nums[mid + 1]\`, пик должен быть справа (мы идём "в гору"). Иначе пик слева (включая mid).

**Временная сложность:** O(log n)
**Пространственная сложность:** O(1)`,
			hint1: `Подумайте, что происходит если nums[mid] < nums[mid + 1]. Вы на "подъёме" - пик должен быть где-то справа.`,
			hint2: `Используйте бинарный поиск с left < right. Если идём в гору, left = mid + 1. Иначе right = mid. Когда left == right, это пик.`,
			whyItMatters: `Поиск пикового элемента демонстрирует бинарный поиск на неотсортированных массивах с другим инвариантом.

**Почему это важно:**

**1. Бинарный поиск без сортировки**

Классический бинарный поиск требует отсортированного массива. Поиск пика использует другое свойство.

**2. Интуиция "подъём/спуск"**

В любой точке идите к большей стороне - достигнете пика.

**3. Применения**

Поиск локальных максимумов, оптимизация, обработка сигналов, анализ цен акций.`,
			solutionCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    """
    Находит индекс любого пикового элемента бинарным поиском.
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            # Идём в гору вправо, пик справа
            left = mid + 1
        else:
            # Идём вниз или на пике, пик слева (включая mid)
            right = mid

    return left`
		},
		uz: {
			title: 'Cho\'qqi elementni topish',
			description: `Massivda qo'shnilaridan katta bo'lgan cho'qqi elementni toping.

**Masala:**

Cho'qqi element - qo'shnilaridan qat'iy katta element.

Butun sonlar massivi \`nums\` berilgan, cho'qqi elementni toping va uning indeksini qaytaring. Agar massivda bir nechta cho'qqi bo'lsa, **har qandayining** indeksini qaytaring.

\`nums[-1] = nums[n] = -∞\` deb tasavvur qiling. Chegaradagi elementlar faqat bitta qo'shnisidan katta bo'lishi kerak.

O(log n) vaqtda ishlaydigan algoritm yozing.

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3, 1]
Chiqish: 2
Tushuntirish: 3 cho'qqi element (indeks 2)

Kirish: nums = [1, 2, 1, 3, 5, 6, 4]
Chiqish: 5 (yoki 1)
Tushuntirish: Indekslar 1 va 5 ikkalasi ham cho'qqi

Kirish: nums = [1]
Chiqish: 0
\`\`\`

**Asosiy tushuncha:**

Agar \`nums[mid] < nums[mid + 1]\`, cho'qqi o'ngda bo'lishi kerak (biz "tepalikka" borayapmiz). Aks holda cho'qqi chapda (mid ni ham o'z ichiga oladi).

**Vaqt murakkabligi:** O(log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `nums[mid] < nums[mid + 1] bo'lganda nima bo'lishini o'ylang. Siz "ko'tarilishda"siz - cho'qqi o'ngda bo'lishi kerak.`,
			hint2: `left < right bilan binar qidiruv ishlating. Agar tepalikka borayotgan bo'lsak, left = mid + 1. Aks holda right = mid. left == right bo'lganda, bu cho'qqi.`,
			whyItMatters: `Cho'qqi elementni topish boshqa invariant yordamida tartiblanmagan massivlarda binar qidiruvni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Tartiblashsiz binar qidiruv**

Klassik binar qidiruv tartiblangan massiv talab qiladi. Cho'qqi qidirish boshqa xususiyatdan foydalanadi.

**2. "Ko'tarilish/tushish" intuitsiyasi**

Har qanday nuqtada katta tomonga boring - cho'qqiga yetasiz.

**3. Qo'llanishlar**

Lokal maksimumlarni topish, optimallashtirish, signal qayta ishlash, aksiya narxlari tahlili.`,
			solutionCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    """
    Binar qidiruv yordamida har qanday cho'qqi elementning indeksini topadi.
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            # O'ngga tepalikka borayapmiz, cho'qqi o'ngda
            left = mid + 1
        else:
            # Tushayapmiz yoki cho'qqidamiz, cho'qqi chapda (mid ni o'z ichiga oladi)
            right = mid

    return left`
		}
	}
};

export default task;
