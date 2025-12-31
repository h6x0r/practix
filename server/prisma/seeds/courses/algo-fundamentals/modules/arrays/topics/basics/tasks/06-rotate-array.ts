import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-rotate-array',
	title: 'Rotate Array',
	difficulty: 'medium',
	tags: ['python', 'arrays', 'in-place', 'reversal'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Rotate an array to the right by k steps.

**Problem:**

Given an integer array \`nums\` and integer \`k\`, rotate the array to the right by \`k\` steps.

**Examples:**

\`\`\`
Input: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
Output: [5, 6, 7, 1, 2, 3, 4]

Explanation:
Step 1: [7, 1, 2, 3, 4, 5, 6]
Step 2: [6, 7, 1, 2, 3, 4, 5]
Step 3: [5, 6, 7, 1, 2, 3, 4]

Input: nums = [-1, -100, 3, 99], k = 2
Output: [3, 99, -1, -100]
\`\`\`

**Reversal Algorithm (O(1) space):**

The trick is to use three reversals:
1. Reverse entire array
2. Reverse first k elements
3. Reverse remaining n-k elements

\`\`\`
Original:    [1, 2, 3, 4, 5, 6, 7], k = 3
Reverse all: [7, 6, 5, 4, 3, 2, 1]
Reverse 0-2: [5, 6, 7, 4, 3, 2, 1]
Reverse 3-6: [5, 6, 7, 1, 2, 3, 4]
\`\`\`

**Why it works:**

- Reversing all puts elements in "almost right" positions (just reversed)
- Reversing first k fixes the rotated portion
- Reversing rest fixes the remaining portion

**Time Complexity:** O(n)
**Space Complexity:** O(1) - in-place`,
	initialCode: `from typing import List

def rotate_array(nums: List[int], k: int) -> None:
    # TODO: Rotate the array to the right by k steps (in-place)
    pass


def reverse(nums: List[int], start: int, end: int) -> None:
    # TODO: Reverse elements in nums from index start to end (inclusive)
    pass`,
	solutionCode: `from typing import List

def rotate_array(nums: List[int], k: int) -> None:
    """
    Rotate the array to the right by k steps.
    Must modify in-place with O(1) extra space.

    Args:
        nums: List of integers to rotate
        k: Number of steps to rotate right
    """
    if not nums:
        return

    # Handle k larger than array length
    k = k % len(nums)
    if k == 0:
        return

    # Three reversals
    reverse(nums, 0, len(nums) - 1)  # Reverse all
    reverse(nums, 0, k - 1)           # Reverse first k
    reverse(nums, k, len(nums) - 1)   # Reverse rest

def reverse(nums: List[int], start: int, end: int) -> None:
    """
    Reverse elements in nums from index start to end (inclusive).

    Args:
        nums: List to modify
        start: Start index (inclusive)
        end: End index (inclusive)
    """
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1`,
	testCode: `import pytest
from solution import rotate_array, reverse

class TestRotateArray:
    def test_basic(self):
        """Test basic rotation"""
        nums = [1, 2, 3, 4, 5, 6, 7]
        rotate_array(nums, 3)
        assert nums == [5, 6, 7, 1, 2, 3, 4]

    def test_k_equals_length(self):
        """Test when k equals array length (no change)"""
        nums = [1, 2, 3]
        rotate_array(nums, 3)
        assert nums == [1, 2, 3]

    def test_k_greater_than_length(self):
        """Test when k is greater than array length"""
        nums = [1, 2, 3]
        rotate_array(nums, 5)
        assert nums == [2, 3, 1]

    def test_k_zero(self):
        """Test when k is zero (no change)"""
        nums = [1, 2, 3]
        rotate_array(nums, 0)
        assert nums == [1, 2, 3]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        nums = [-1, -100, 3, 99]
        rotate_array(nums, 2)
        assert nums == [3, 99, -1, -100]

    def test_single_element(self):
        """Test single element array"""
        nums = [1]
        rotate_array(nums, 5)
        assert nums == [1]

    def test_two_elements(self):
        """Test two element array"""
        nums = [1, 2]
        rotate_array(nums, 1)
        assert nums == [2, 1]

class TestReverse:
    def test_full_reverse(self):
        """Test reversing entire array"""
        nums = [1, 2, 3, 4, 5]
        reverse(nums, 0, 4)
        assert nums == [5, 4, 3, 2, 1]

    def test_partial_reverse(self):
        """Test reversing part of array"""
        nums = [1, 2, 3, 4, 5]
        reverse(nums, 1, 3)
        assert nums == [1, 4, 3, 2, 5]

    def test_large_rotation(self):
        """Test with large k value"""
        nums = [1, 2, 3, 4, 5]
        rotate_array(nums, 12)
        assert nums == [4, 5, 1, 2, 3]`,
	hint1: `First handle edge cases: k = k % len(nums). Then implement a helper function reverse(nums, start, end) that reverses elements between start and end indices using two pointers swapping nums[start] and nums[end].`,
	hint2: `Use three calls to reverse: reverse(nums, 0, len(nums)-1), then reverse(nums, 0, k-1), then reverse(nums, k, len(nums)-1). This gives you the rotated array.`,
	whyItMatters: `The reversal algorithm demonstrates elegant in-place array manipulation.

**Why This Matters:**

**1. Space Efficiency**

Naive approach uses O(n) extra space:
\`\`\`python
# O(n) space - creates new list
def rotate_naive(nums: List[int], k: int) -> List[int]:
    result = [0] * len(nums)
    for i, v in enumerate(nums):
        result[(i + k) % len(nums)] = v
    return result
\`\`\`

Reversal uses O(1) space - modifies in place!

**2. The Reversal Trick**

This pattern applies to many problems:
- Rotate string by k characters
- Reverse words in a string (reverse all, then reverse each word)
- Shuffle array segments

**3. In-Place Algorithm Design**

Key insight: transformations through intermediate "wrong" states
\`\`\`
Start:       [1, 2, 3, 4, 5]
After rev1:  [5, 4, 3, 2, 1]  <-- looks wrong
After rev2:  [4, 5, 3, 2, 1]  <-- still wrong
After rev3:  [4, 5, 1, 2, 3]  <-- correct!
\`\`\`

**4. Mathematical Insight**

Rotation by k = reverse(reverse(first part) + reverse(second part))
This is why the algorithm works mathematically.`,
	order: 6,
	translations: {
		ru: {
			title: 'Поворот массива',
			description: `Поверните массив вправо на k шагов.

**Задача:**

Дан массив целых чисел \`nums\` и целое число \`k\`, поверните массив вправо на \`k\` шагов.

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
Выход: [5, 6, 7, 1, 2, 3, 4]

Объяснение:
Шаг 1: [7, 1, 2, 3, 4, 5, 6]
Шаг 2: [6, 7, 1, 2, 3, 4, 5]
Шаг 3: [5, 6, 7, 1, 2, 3, 4]

Вход: nums = [-1, -100, 3, 99], k = 2
Выход: [3, 99, -1, -100]
\`\`\`

**Алгоритм разворота (O(1) памяти):**

Трюк в использовании трёх разворотов:
1. Развернуть весь массив
2. Развернуть первые k элементов
3. Развернуть оставшиеся n-k элементов

\`\`\`
Оригинал:        [1, 2, 3, 4, 5, 6, 7], k = 3
Развернуть всё:  [7, 6, 5, 4, 3, 2, 1]
Развернуть 0-2:  [5, 6, 7, 4, 3, 2, 1]
Развернуть 3-6:  [5, 6, 7, 1, 2, 3, 4]
\`\`\`

**Почему это работает:**

- Разворот всего ставит элементы в "почти правильные" позиции (только в обратном порядке)
- Разворот первых k исправляет повёрнутую часть
- Разворот остального исправляет оставшуюся часть

**Временная сложность:** O(n)
**Пространственная сложность:** O(1) - на месте`,
			hint1: `Сначала обработайте граничные случаи: k = k % len(nums). Затем реализуйте вспомогательную функцию reverse(nums, start, end), которая разворачивает элементы между индексами start и end, меняя местами nums[start] и nums[end].`,
			hint2: `Используйте три вызова reverse: reverse(nums, 0, len(nums)-1), затем reverse(nums, 0, k-1), затем reverse(nums, k, len(nums)-1). Это даст вам повёрнутый массив.`,
			whyItMatters: `Алгоритм разворота демонстрирует элегантную манипуляцию массивом на месте.

**Почему это важно:**

**1. Эффективность по памяти**

Наивный подход использует O(n) дополнительной памяти:
\`\`\`python
# O(n) памяти - создаёт новый список
def rotate_naive(nums: List[int], k: int) -> List[int]:
    result = [0] * len(nums)
    for i, v in enumerate(nums):
        result[(i + k) % len(nums)] = v
    return result
\`\`\`

Разворот использует O(1) памяти - модифицирует на месте!

**2. Трюк с разворотом**

Этот паттерн применяется ко многим задачам:
- Повернуть строку на k символов
- Развернуть слова в строке (развернуть всё, затем развернуть каждое слово)
- Перемешать сегменты массива

**3. Проектирование алгоритмов на месте**

Ключевая идея: преобразования через промежуточные "неправильные" состояния
\`\`\`
Начало:     [1, 2, 3, 4, 5]
После rev1: [5, 4, 3, 2, 1]  <-- выглядит неправильно
После rev2: [4, 5, 3, 2, 1]  <-- всё ещё неправильно
После rev3: [4, 5, 1, 2, 3]  <-- правильно!
\`\`\`

**4. Математическое понимание**

Поворот на k = reverse(reverse(первая часть) + reverse(вторая часть))
Вот почему алгоритм работает математически.`,
			solutionCode: `from typing import List

def rotate_array(nums: List[int], k: int) -> None:
    """
    Поворачивает массив вправо на k шагов.
    Должен модифицировать на месте с O(1) дополнительной памяти.

    Args:
        nums: Список целых чисел для поворота
        k: Количество шагов поворота вправо
    """
    if not nums:
        return

    # Обрабатываем k больше длины массива
    k = k % len(nums)
    if k == 0:
        return

    # Три разворота
    reverse(nums, 0, len(nums) - 1)  # Развернуть всё
    reverse(nums, 0, k - 1)           # Развернуть первые k
    reverse(nums, k, len(nums) - 1)   # Развернуть остальное

def reverse(nums: List[int], start: int, end: int) -> None:
    """
    Разворачивает элементы в nums от индекса start до end (включительно).

    Args:
        nums: Список для модификации
        start: Начальный индекс (включительно)
        end: Конечный индекс (включительно)
    """
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1`
		},
		uz: {
			title: 'Massivni aylantirish',
			description: `Massivni k qadam o'ngga aylantiring.

**Masala:**

Butun sonlar massivi \`nums\` va butun son \`k\` berilgan, massivni \`k\` qadam o'ngga aylantiring.

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3, 4, 5, 6, 7], k = 3
Chiqish: [5, 6, 7, 1, 2, 3, 4]

Tushuntirish:
Qadam 1: [7, 1, 2, 3, 4, 5, 6]
Qadam 2: [6, 7, 1, 2, 3, 4, 5]
Qadam 3: [5, 6, 7, 1, 2, 3, 4]

Kirish: nums = [-1, -100, 3, 99], k = 2
Chiqish: [3, 99, -1, -100]
\`\`\`

**Teskari aylantirish algoritmi (O(1) xotira):**

Hiyla uchta teskarilashni ishlatishda:
1. Butun massivni teskari aylantirish
2. Birinchi k elementni teskari aylantirish
3. Qolgan n-k elementni teskari aylantirish

\`\`\`
Asl:              [1, 2, 3, 4, 5, 6, 7], k = 3
Hammasini teskari: [7, 6, 5, 4, 3, 2, 1]
0-2 ni teskari:   [5, 6, 7, 4, 3, 2, 1]
3-6 ni teskari:   [5, 6, 7, 1, 2, 3, 4]
\`\`\`

**Nima uchun ishlaydi:**

- Hammasini teskari aylantirish elementlarni "deyarli to'g'ri" joylarga qo'yadi
- Birinchi k ni teskari aylantirish aylantirilgan qismni tuzatadi
- Qolganini teskari aylantirish qolgan qismni tuzatadi

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1) - joyida`,
			hint1: `Avval chegara holatlarini qayta ishlang: k = k % len(nums). Keyin start va end indekslari orasidagi elementlarni nums[start] va nums[end] ni almashtirish orqali teskari aylantiradigan reverse(nums, start, end) yordamchi funktsiyasini amalga oshiring.`,
			hint2: `Uchta reverse chaqiruvidan foydalaning: reverse(nums, 0, len(nums)-1), keyin reverse(nums, 0, k-1), keyin reverse(nums, k, len(nums)-1). Bu sizga aylantirilgan massivni beradi.`,
			whyItMatters: `Teskari aylantirish algoritmi nafis joyida massiv manipulyatsiyasini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Xotira samaradorligi**

Oddiy yondashuv O(n) qo'shimcha xotira ishlatadi:
\`\`\`python
# O(n) xotira - yangi ro'yxat yaratadi
def rotate_naive(nums: List[int], k: int) -> List[int]:
    result = [0] * len(nums)
    for i, v in enumerate(nums):
        result[(i + k) % len(nums)] = v
    return result
\`\`\`

Teskari aylantirish O(1) xotira ishlatadi - joyida o'zgartiradi!

**2. Teskari aylantirish hiylasi**

Bu pattern ko'plab masalalarga qo'llaniladi:
- Satrni k belgi aylantirish
- Satrdagi so'zlarni teskari aylantirish (hammasini teskari, keyin har bir so'zni teskari)
- Massiv segmentlarini aralashtirish

**3. Joyida algoritm dizayni**

Asosiy tushuncha: oraliq "noto'g'ri" holatlar orqali o'zgartirishlar
\`\`\`
Boshlanish:  [1, 2, 3, 4, 5]
rev1 keyin: [5, 4, 3, 2, 1]  <-- noto'g'ri ko'rinadi
rev2 keyin: [4, 5, 3, 2, 1]  <-- hali ham noto'g'ri
rev3 keyin: [4, 5, 1, 2, 3]  <-- to'g'ri!
\`\`\`

**4. Matematik tushuncha**

k ga aylantirish = reverse(reverse(birinchi qism) + reverse(ikkinchi qism))
Algoritm matematik jihatdan shuning uchun ishlaydi.`,
			solutionCode: `from typing import List

def rotate_array(nums: List[int], k: int) -> None:
    """
    Massivni k qadam o'ngga aylantiradi.
    O(1) qo'shimcha xotira bilan joyida o'zgartirishi kerak.

    Args:
        nums: Aylantirish uchun butun sonlar ro'yxati
        k: O'ngga aylantirish qadamlari soni
    """
    if not nums:
        return

    # k massiv uzunligidan katta bo'lsa qayta ishlaymiz
    k = k % len(nums)
    if k == 0:
        return

    # Uchta teskari aylantirish
    reverse(nums, 0, len(nums) - 1)  # Hammasini teskari
    reverse(nums, 0, k - 1)           # Birinchi k ni teskari
    reverse(nums, k, len(nums) - 1)   # Qolganini teskari

def reverse(nums: List[int], start: int, end: int) -> None:
    """
    nums dagi elementlarni start dan end gacha (kiritilgan) indekslarda teskari aylantiradi.

    Args:
        nums: O'zgartirish uchun ro'yxat
        start: Boshlang'ich indeks (kiritilgan)
        end: Tugash indeksi (kiritilgan)
    """
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1`
		}
	}
};

export default task;
