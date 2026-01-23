import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-two-sum',
	title: 'Two Sum',
	difficulty: 'easy',
	tags: ['python', 'arrays', 'hash-map', 'two-pointers'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find two numbers in an array that add up to a target value.

**Problem:**

Given an array of integers \`nums\` and an integer \`target\`, return the indices of the two numbers that add up to \`target\`.

**Constraints:**
- Each input has exactly one solution
- You may not use the same element twice
- Return indices in any order

**Examples:**

\`\`\`
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9

Input: nums = [3, 2, 4], target = 6
Output: [1, 2]
Explanation: nums[1] + nums[2] = 2 + 4 = 6

Input: nums = [3, 3], target = 6
Output: [0, 1]
\`\`\`

**Approach:**

Use a hash map (dictionary) to store numbers and their indices as you iterate:
1. For each number, calculate the complement (target - num)
2. Check if complement exists in the dictionary
3. If yes, return both indices
4. If no, add current number and index to dictionary

**Time Complexity:** O(n) - single pass through array
**Space Complexity:** O(n) - dictionary storage`,
	initialCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    # TODO: Find two numbers that add up to target, return their indices

    return [-1, -1]`,
	solutionCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find two numbers that add up to target.
    Returns their indices, or [-1, -1] if no solution exists.

    Args:
        nums: List of integers
        target: Target sum to find

    Returns:
        List of two indices whose values sum to target
    """
    # Dictionary: number -> index
    seen = {}

    for i, num in enumerate(nums):
        # Calculate what number we need to find
        complement = target - num

        # Check if complement was seen before
        if complement in seen:
            return [seen[complement], i]  # Found! Return both indices

        # Store current number and its index
        seen[num] = i

    return [-1, -1]  # No solution found`,
	testCode: `import pytest
from solution import two_sum

class TestTwoSum:
    def test_basic_case(self):
        """Test basic case with solution at start"""
        result = two_sum([2, 7, 11, 15], 9)
        assert sorted(result) == [0, 1]

    def test_middle_elements(self):
        """Test when solution is in middle of array"""
        result = two_sum([3, 2, 4], 6)
        assert sorted(result) == [1, 2]

    def test_same_numbers(self):
        """Test with duplicate numbers"""
        result = two_sum([3, 3], 6)
        assert sorted(result) == [0, 1]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        result = two_sum([-1, -2, -3, -4, -5], -8)
        assert sorted(result) == [2, 4]

    def test_large_array(self):
        """Test with larger array"""
        result = two_sum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 19)
        assert sorted(result) == [8, 9]

    def test_no_solution(self):
        """Test when no solution exists"""
        result = two_sum([1, 2, 3], 10)
        assert result == [-1, -1]

    def test_zero_target(self):
        """Test with zero as target"""
        result = two_sum([-1, 0, 1, 2, -1, -4], 0)
        # -1 + 1 = 0, indices [0, 2]
        assert sorted(result) == [0, 2]

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative numbers"""
        result = two_sum([5, -2, 3, -7, 8], 1)
        # -2 + 3 = 1 (indices [1, 2]) or -7 + 8 = 1 (indices [3, 4])
        assert sorted(result) in [[1, 2], [3, 4]]

    def test_solution_at_end(self):
        """Test when solution is at the end of array"""
        result = two_sum([1, 2, 3, 4, 5, 9, 10], 19)
        assert sorted(result) == [5, 6]

    def test_large_numbers(self):
        """Test with large numbers"""
        result = two_sum([1000000, 500000, 3000000, 1500000], 2500000)
        assert sorted(result) == [0, 3]`,
	hint1: `Create a dictionary to store each number as the key and its index as the value. For each number, check if (target - num) exists in the dictionary using the 'in' operator.`,
	hint2: `When you find the complement in the dictionary, return [seen[complement], i]. If not found after the loop, return [-1, -1].`,
	whyItMatters: `Two Sum is the quintessential hash map problem and appears in countless variations.

**Why This Pattern Matters:**

**1. Foundation for Many Problems**

The "complement lookup" pattern appears everywhere:
- Three Sum (sort + two pointers + complement)
- Four Sum (nested loops + two sum)
- Subarray Sum Equals K (prefix sum + complement)
- Two Sum variations (sorted array, BST, etc.)

**2. O(n^2) to O(n) Optimization**

\`\`\`python
# Brute force - O(n^2)
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]

# Hash map - O(n)
seen = {}
for i, num in enumerate(nums):
    if target - num in seen:
        return [seen[target - num], i]
    seen[num] = i
\`\`\`

**3. Space-Time Tradeoff**

This demonstrates the classic tradeoff:
- Use O(n) extra space to achieve O(n) time
- Trading memory for speed is often worth it

**4. Real-World Applications**

- Finding complementary items in e-commerce
- Matching transactions that sum to zero
- Finding pairs of elements meeting criteria`,
	order: 1,
	translations: {
		ru: {
			title: 'Сумма двух',
			description: `Найдите два числа в массиве, которые в сумме дают целевое значение.

**Задача:**

Дан массив целых чисел \`nums\` и целое число \`target\`, верните индексы двух чисел, которые в сумме дают \`target\`.

**Ограничения:**
- Каждый вход имеет ровно одно решение
- Нельзя использовать один и тот же элемент дважды
- Верните индексы в любом порядке

**Примеры:**

\`\`\`
Вход: nums = [2, 7, 11, 15], target = 9
Выход: [0, 1]
Объяснение: nums[0] + nums[1] = 2 + 7 = 9

Вход: nums = [3, 2, 4], target = 6
Выход: [1, 2]
Объяснение: nums[1] + nums[2] = 2 + 4 = 6

Вход: nums = [3, 3], target = 6
Выход: [0, 1]
\`\`\`

**Подход:**

Используйте словарь для хранения чисел и их индексов при итерации:
1. Для каждого числа вычислите дополнение (target - num)
2. Проверьте, есть ли дополнение в словаре
3. Если да, верните оба индекса
4. Если нет, добавьте текущее число и индекс в словарь

**Временная сложность:** O(n) - один проход по массиву
**Пространственная сложность:** O(n) - хранение в словаре`,
			hint1: `Создайте словарь для хранения каждого числа как ключа и его индекса как значения. Для каждого числа проверьте, существует ли (target - num) в словаре с помощью оператора 'in'.`,
			hint2: `Когда найдёте дополнение в словаре, верните [seen[complement], i]. Если не найдено после цикла, верните [-1, -1].`,
			whyItMatters: `Two Sum - это классическая задача на хеш-таблицы, которая встречается в бесчисленных вариациях.

**Почему этот паттерн важен:**

**1. Основа для многих задач**

Паттерн "поиск дополнения" появляется везде:
- Three Sum (сортировка + два указателя + дополнение)
- Four Sum (вложенные циклы + two sum)
- Subarray Sum Equals K (префиксная сумма + дополнение)

**2. Оптимизация от O(n^2) до O(n)**

\`\`\`python
# Перебор - O(n^2)
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]

# Словарь - O(n)
seen = {}
for i, num in enumerate(nums):
    if target - num in seen:
        return [seen[target - num], i]
    seen[num] = i
\`\`\`

**3. Компромисс память-время**

Это демонстрирует классический компромисс:
- Используем O(n) дополнительной памяти для достижения O(n) времени
- Обмен памяти на скорость часто оправдан

**4. Применение в реальном мире**

- Поиск дополняющих товаров в e-commerce
- Сопоставление транзакций с нулевой суммой
- Поиск пар элементов по критериям`,
			solutionCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Находит два числа, которые в сумме дают target.
    Возвращает их индексы, или [-1, -1] если решения нет.

    Args:
        nums: Список целых чисел
        target: Целевая сумма для поиска

    Returns:
        Список из двух индексов, значения которых дают target
    """
    # Словарь: число -> индекс
    seen = {}

    for i, num in enumerate(nums):
        # Вычисляем какое число нужно найти
        complement = target - num

        # Проверяем, видели ли дополнение раньше
        if complement in seen:
            return [seen[complement], i]  # Нашли! Возвращаем оба индекса

        # Сохраняем текущее число и его индекс
        seen[num] = i

    return [-1, -1]  # Решение не найдено`
		},
		uz: {
			title: 'Ikki yig\'indi',
			description: `Massivda maqsadli qiymatga teng yig'indini beradigan ikkita sonni toping.

**Masala:**

Butun sonlar massivi \`nums\` va butun son \`target\` berilgan, \`target\` ga teng yig'indini beradigan ikkita sonning indekslarini qaytaring.

**Cheklovlar:**
- Har bir kirishda aniq bitta yechim bor
- Bitta elementni ikki marta ishlatib bo'lmaydi
- Indekslarni istalgan tartibda qaytaring

**Misollar:**

\`\`\`
Kirish: nums = [2, 7, 11, 15], target = 9
Chiqish: [0, 1]
Tushuntirish: nums[0] + nums[1] = 2 + 7 = 9

Kirish: nums = [3, 2, 4], target = 6
Chiqish: [1, 2]
Tushuntirish: nums[1] + nums[2] = 2 + 4 = 6

Kirish: nums = [3, 3], target = 6
Chiqish: [0, 1]
\`\`\`

**Yondashuv:**

Iteratsiya qilishda sonlar va ularning indekslarini saqlash uchun lug'atdan foydalaning:
1. Har bir son uchun to'ldiruvchini hisoblang (target - num)
2. To'ldiruvchi lug'atda borligini tekshiring
3. Agar bor bo'lsa, ikkala indeksni qaytaring
4. Agar yo'q bo'lsa, joriy son va indeksni lug'atga qo'shing

**Vaqt murakkabligi:** O(n) - massiv bo'ylab bitta o'tish
**Xotira murakkabligi:** O(n) - lug'at uchun xotira`,
			hint1: `Har bir sonni kalit va uning indeksini qiymat sifatida saqlash uchun lug'at yarating. Har bir son uchun (target - num) lug'atda borligini 'in' operatori bilan tekshiring.`,
			hint2: `Lug'atda to'ldiruvchini topganingizda, [seen[complement], i] qaytaring. Tsikldan keyin topilmasa, [-1, -1] qaytaring.`,
			whyItMatters: `Two Sum - bu son-sanoqsiz variatsiyalarda uchraydigan klassik xesh-jadval masalasi.

**Bu pattern nima uchun muhim:**

**1. Ko'plab masalalar uchun asos**

"To'ldiruvchi qidirish" patterni hamma joyda uchraydi:
- Three Sum (saralash + ikki ko'rsatkich + to'ldiruvchi)
- Four Sum (ichma-ich tsikllar + two sum)
- Subarray Sum Equals K (prefiks yig'indi + to'ldiruvchi)

**2. O(n^2) dan O(n) ga optimallashtirish**

\`\`\`python
# To'liq qidirish - O(n^2)
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]

# Lug'at - O(n)
seen = {}
for i, num in enumerate(nums):
    if target - num in seen:
        return [seen[target - num], i]
    seen[num] = i
\`\`\`

**3. Xotira-vaqt kelishuvi**

Bu klassik kelishuvni ko'rsatadi:
- O(n) vaqtga erishish uchun O(n) qo'shimcha xotira ishlatamiz
- Tezlik uchun xotirani almashtirish ko'pincha foydali

**4. Haqiqiy dunyo qo'llanilishi**

- E-tijoratda to'ldiruvchi mahsulotlarni topish
- Nol yig'indili tranzaksiyalarni moslashtirish
- Mezonlarga mos juftliklarni topish`,
			solutionCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Maqsadga teng yig'indini beradigan ikkita sonni topadi.
    Ularning indekslarini qaytaradi, yechim bo'lmasa [-1, -1].

    Args:
        nums: Butun sonlar ro'yxati
        target: Qidirilayotgan maqsadli yig'indi

    Returns:
        Qiymatlari target ga teng bo'lgan ikki indeks ro'yxati
    """
    # Lug'at: son -> indeks
    seen = {}

    for i, num in enumerate(nums):
        # Qaysi sonni topish kerakligini hisoblaymiz
        complement = target - num

        # To'ldiruvchi oldin ko'rilganmi tekshiramiz
        if complement in seen:
            return [seen[complement], i]  # Topdik! Ikkala indeksni qaytaramiz

        # Joriy son va uning indeksini saqlaymiz
        seen[num] = i

    return [-1, -1]  # Yechim topilmadi`
		}
	}
};

export default task;
