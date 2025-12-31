import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-prefix-sum',
	title: 'Prefix Sum Array',
	difficulty: 'easy',
	tags: ['python', 'arrays', 'prefix-sum'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Build a prefix sum array and use it to answer range sum queries in O(1).

**Problem:**

Given an array of integers, implement two functions:
1. \`build_prefix_sum(nums)\` - Build a prefix sum array
2. \`range_sum(prefix, left, right)\` - Return sum of elements from index left to right (inclusive)

**Prefix Sum Array:**

\`prefix[i]\` = sum of nums[0] + nums[1] + ... + nums[i-1]
\`prefix[0]\` = 0 (empty sum)

**Examples:**

\`\`\`
Input: nums = [1, 2, 3, 4, 5]
Prefix: [0, 1, 3, 6, 10, 15]

range_sum(prefix, 1, 3) = prefix[4] - prefix[1] = 10 - 1 = 9
(sum of nums[1] + nums[2] + nums[3] = 2 + 3 + 4 = 9)

range_sum(prefix, 0, 4) = prefix[5] - prefix[0] = 15 - 0 = 15
(sum of entire array)
\`\`\`

**Formula:**

\`\`\`python
range_sum(left, right) = prefix[right+1] - prefix[left]
\`\`\`

**Time Complexity:**
- Build: O(n)
- Query: O(1)

**Space Complexity:** O(n) for prefix array`,
	initialCode: `from typing import List

def build_prefix_sum(nums: List[int]) -> List[int]:
    # TODO: Build a prefix sum array where prefix[i] = sum of nums[0] to nums[i-1]

    return []


def range_sum(prefix: List[int], left: int, right: int) -> int:
    # TODO: Return sum of elements from index left to right using prefix array

    return 0`,
	solutionCode: `from typing import List

def build_prefix_sum(nums: List[int]) -> List[int]:
    """
    Build a prefix sum array.
    prefix[i] = sum of nums[0] to nums[i-1]
    prefix[0] = 0 (empty prefix)

    Args:
        nums: List of integers

    Returns:
        Prefix sum array of length len(nums) + 1
    """
    # Create list with one extra element for empty prefix
    prefix = [0] * (len(nums) + 1)

    # Build prefix sums
    for i in range(1, len(nums) + 1):
        prefix[i] = prefix[i - 1] + nums[i - 1]

    return prefix

def range_sum(prefix: List[int], left: int, right: int) -> int:
    """
    Return sum of elements from index left to right (inclusive).
    Uses the prefix sum array for O(1) query.

    Args:
        prefix: Prefix sum array
        left: Start index (inclusive)
        right: End index (inclusive)

    Returns:
        Sum of elements in range [left, right]
    """
    # Sum of range [left, right] = prefix[right+1] - prefix[left]
    return prefix[right + 1] - prefix[left]`,
	testCode: `import pytest
from solution import build_prefix_sum, range_sum

class TestBuildPrefixSum:
    def test_basic(self):
        """Test basic case"""
        assert build_prefix_sum([1, 2, 3, 4, 5]) == [0, 1, 3, 6, 10, 15]

    def test_single_element(self):
        """Test single element array"""
        assert build_prefix_sum([5]) == [0, 5]

    def test_empty(self):
        """Test empty array"""
        assert build_prefix_sum([]) == [0]

    def test_negative_numbers(self):
        """Test with negative numbers"""
        assert build_prefix_sum([-1, 2, -3, 4]) == [0, -1, 1, -2, 2]

class TestRangeSum:
    def test_middle_range(self):
        """Test range in the middle"""
        prefix = build_prefix_sum([1, 2, 3, 4, 5])
        assert range_sum(prefix, 1, 3) == 9

    def test_full_range(self):
        """Test full array range"""
        prefix = build_prefix_sum([1, 2, 3, 4, 5])
        assert range_sum(prefix, 0, 4) == 15

    def test_single_element_range(self):
        """Test single element range"""
        prefix = build_prefix_sum([1, 2, 3, 4, 5])
        assert range_sum(prefix, 2, 2) == 3

    def test_start_range(self):
        """Test range from start"""
        prefix = build_prefix_sum([1, 2, 3, 4, 5])
        assert range_sum(prefix, 0, 2) == 6

    def test_end_range(self):
        """Test range at end"""
        prefix = build_prefix_sum([1, 2, 3, 4, 5])
        assert range_sum(prefix, 3, 4) == 9

    def test_negative_range(self):
        """Test range with negative numbers"""
        prefix = build_prefix_sum([-1, 2, -3, 4])
        assert range_sum(prefix, 1, 3) == 3`,
	hint1: `For build_prefix_sum, create a list of length len(nums)+1 initialized with zeros. Loop from i=1 to len(nums)+1, setting prefix[i] = prefix[i-1] + nums[i-1].`,
	hint2: `For range_sum, the formula is: prefix[right+1] - prefix[left]. This gives you the sum of elements from index left to right inclusive.`,
	whyItMatters: `Prefix sums enable O(1) range queries after O(n) preprocessing.

**Why This Matters:**

**1. Trade-off: Precomputation for Speed**

Without prefix sum - O(n) per query:
\`\`\`python
def range_sum_naive(nums: List[int], left: int, right: int) -> int:
    total = 0
    for i in range(left, right + 1):
        total += nums[i]
    return total  # O(n) each time
\`\`\`

With prefix sum - O(1) per query:
\`\`\`python
prefix = build_prefix_sum(nums)  # O(n) once
total = range_sum(prefix, left, right)  # O(1) always!
\`\`\`

**2. When to Use Prefix Sums**

- Multiple range sum queries on same array
- Subarray sum equals K (prefix sum + hash map)
- Count of subarrays with sum in range
- 2D prefix sums for matrix queries

**3. Common Interview Problems**

- Subarray Sum Equals K (combine with hash map)
- Product of Array Except Self (prefix products)
- Range Addition (difference array, inverse of prefix sum)
- 2D Range Sum Query

**4. Real-World Applications**

- Database query optimization (running totals)
- Image processing (integral images)
- Statistics (cumulative distribution)
- Financial calculations (running balances)`,
	order: 5,
	translations: {
		ru: {
			title: 'Массив префиксных сумм',
			description: `Постройте массив префиксных сумм и используйте его для ответа на запросы суммы диапазона за O(1).

**Задача:**

Дан массив целых чисел, реализуйте две функции:
1. \`build_prefix_sum(nums)\` - Построить массив префиксных сумм
2. \`range_sum(prefix, left, right)\` - Вернуть сумму элементов от индекса left до right (включительно)

**Массив префиксных сумм:**

\`prefix[i]\` = сумма nums[0] + nums[1] + ... + nums[i-1]
\`prefix[0]\` = 0 (пустая сумма)

**Примеры:**

\`\`\`
Вход: nums = [1, 2, 3, 4, 5]
Prefix: [0, 1, 3, 6, 10, 15]

range_sum(prefix, 1, 3) = prefix[4] - prefix[1] = 10 - 1 = 9
(сумма nums[1] + nums[2] + nums[3] = 2 + 3 + 4 = 9)

range_sum(prefix, 0, 4) = prefix[5] - prefix[0] = 15 - 0 = 15
(сумма всего массива)
\`\`\`

**Формула:**

\`\`\`python
range_sum(left, right) = prefix[right+1] - prefix[left]
\`\`\`

**Временная сложность:**
- Построение: O(n)
- Запрос: O(1)

**Пространственная сложность:** O(n) для массива префиксов`,
			hint1: `Для build_prefix_sum создайте список длины len(nums)+1, инициализированный нулями. Пройдите циклом от i=1 до len(nums)+1, устанавливая prefix[i] = prefix[i-1] + nums[i-1].`,
			hint2: `Для range_sum формула: prefix[right+1] - prefix[left]. Это даёт вам сумму элементов от индекса left до right включительно.`,
			whyItMatters: `Префиксные суммы позволяют делать запросы диапазона за O(1) после предобработки O(n).

**Почему это важно:**

**1. Компромисс: Предвычисление ради скорости**

Без префиксной суммы - O(n) на запрос:
\`\`\`python
def range_sum_naive(nums: List[int], left: int, right: int) -> int:
    total = 0
    for i in range(left, right + 1):
        total += nums[i]
    return total  # O(n) каждый раз
\`\`\`

С префиксной суммой - O(1) на запрос:
\`\`\`python
prefix = build_prefix_sum(nums)  # O(n) один раз
total = range_sum(prefix, left, right)  # O(1) всегда!
\`\`\`

**2. Когда использовать префиксные суммы**

- Множество запросов суммы диапазона на одном массиве
- Сумма подмассива равна K (префиксная сумма + хеш-таблица)
- Подсчёт подмассивов с суммой в диапазоне
- 2D префиксные суммы для запросов к матрице

**3. Распространённые задачи на интервью**

- Subarray Sum Equals K (комбинация с хеш-таблицей)
- Product of Array Except Self (префиксные произведения)
- Range Addition (массив разностей)
- 2D Range Sum Query

**4. Применения в реальном мире**

- Оптимизация запросов к базам данных (накопительные итоги)
- Обработка изображений (интегральные изображения)
- Статистика (кумулятивное распределение)
- Финансовые расчёты (текущие балансы)`,
			solutionCode: `from typing import List

def build_prefix_sum(nums: List[int]) -> List[int]:
    """
    Строит массив префиксных сумм.
    prefix[i] = сумма nums[0] до nums[i-1]
    prefix[0] = 0 (пустой префикс)

    Args:
        nums: Список целых чисел

    Returns:
        Массив префиксных сумм длины len(nums) + 1
    """
    # Создаём список с одним дополнительным элементом для пустого префикса
    prefix = [0] * (len(nums) + 1)

    # Строим префиксные суммы
    for i in range(1, len(nums) + 1):
        prefix[i] = prefix[i - 1] + nums[i - 1]

    return prefix

def range_sum(prefix: List[int], left: int, right: int) -> int:
    """
    Возвращает сумму элементов от индекса left до right (включительно).
    Использует массив префиксных сумм для запроса за O(1).

    Args:
        prefix: Массив префиксных сумм
        left: Начальный индекс (включительно)
        right: Конечный индекс (включительно)

    Returns:
        Сумма элементов в диапазоне [left, right]
    """
    # Сумма диапазона [left, right] = prefix[right+1] - prefix[left]
    return prefix[right + 1] - prefix[left]`
		},
		uz: {
			title: 'Prefiks yig\'indi massivi',
			description: `Prefiks yig'indi massivini yarating va undan O(1) da diapazon yig'indisi so'rovlariga javob berish uchun foydalaning.

**Masala:**

Butun sonlar massivi berilgan, ikkita funktsiyani amalga oshiring:
1. \`build_prefix_sum(nums)\` - Prefiks yig'indi massivini yaratish
2. \`range_sum(prefix, left, right)\` - left dan right gacha (kiritilgan) indekslardagi elementlar yig'indisini qaytarish

**Prefiks yig'indi massivi:**

\`prefix[i]\` = nums[0] + nums[1] + ... + nums[i-1] yig'indisi
\`prefix[0]\` = 0 (bo'sh yig'indi)

**Misollar:**

\`\`\`
Kirish: nums = [1, 2, 3, 4, 5]
Prefix: [0, 1, 3, 6, 10, 15]

range_sum(prefix, 1, 3) = prefix[4] - prefix[1] = 10 - 1 = 9
(nums[1] + nums[2] + nums[3] yig'indisi = 2 + 3 + 4 = 9)

range_sum(prefix, 0, 4) = prefix[5] - prefix[0] = 15 - 0 = 15
(butun massiv yig'indisi)
\`\`\`

**Formula:**

\`\`\`python
range_sum(left, right) = prefix[right+1] - prefix[left]
\`\`\`

**Vaqt murakkabligi:**
- Qurish: O(n)
- So'rov: O(1)

**Xotira murakkabligi:** O(n) prefiks massivi uchun`,
			hint1: `build_prefix_sum uchun len(nums)+1 uzunlikdagi ro'yxat yarating, nollar bilan to'ldiring. i=1 dan len(nums)+1 gacha tsikl qiling, prefix[i] = prefix[i-1] + nums[i-1] qilib o'rnating.`,
			hint2: `range_sum uchun formula: prefix[right+1] - prefix[left]. Bu sizga left dan right gacha indekslardagi elementlar yig'indisini beradi.`,
			whyItMatters: `Prefiks yig'indilari O(n) oldindan ishlov berishdan keyin O(1) diapazon so'rovlarini imkon qiladi.

**Bu nima uchun muhim:**

**1. Kelishuv: Tezlik uchun oldindan hisoblash**

Prefiks yig'indisiz - har bir so'rov uchun O(n):
\`\`\`python
def range_sum_naive(nums: List[int], left: int, right: int) -> int:
    total = 0
    for i in range(left, right + 1):
        total += nums[i]
    return total  # Har safar O(n)
\`\`\`

Prefiks yig'indi bilan - har bir so'rov uchun O(1):
\`\`\`python
prefix = build_prefix_sum(nums)  # Bir marta O(n)
total = range_sum(prefix, left, right)  # Har doim O(1)!
\`\`\`

**2. Prefiks yig'indilarini qachon ishlatish kerak**

- Bitta massivda bir nechta diapazon yig'indisi so'rovlari
- K ga teng qism massiv yig'indisi (prefiks yig'indi + xesh xarita)
- Diapazon ichidagi yig'indili qism massivlar soni
- Matritsa so'rovlari uchun 2D prefiks yig'indilari

**3. Intervyu uchun keng tarqalgan masalalar**

- Subarray Sum Equals K (xesh xarita bilan)
- Product of Array Except Self (prefiks ko'paytmalari)
- Range Addition (farq massivi)
- 2D Range Sum Query

**4. Haqiqiy dunyo qo'llanilishi**

- Ma'lumotlar bazasi so'rovlarini optimallashtirish (to'plam jami)
- Tasvir qayta ishlash (integral tasvirlar)
- Statistika (kumulyativ taqsimot)
- Moliyaviy hisob-kitoblar (joriy balanslar)`,
			solutionCode: `from typing import List

def build_prefix_sum(nums: List[int]) -> List[int]:
    """
    Prefiks yig'indi massivini yaratadi.
    prefix[i] = nums[0] dan nums[i-1] gacha yig'indi
    prefix[0] = 0 (bo'sh prefiks)

    Args:
        nums: Butun sonlar ro'yxati

    Returns:
        len(nums) + 1 uzunlikdagi prefiks yig'indi massivi
    """
    # Bo'sh prefiks uchun bitta qo'shimcha element bilan ro'yxat yaratamiz
    prefix = [0] * (len(nums) + 1)

    # Prefiks yig'indilarini quramiz
    for i in range(1, len(nums) + 1):
        prefix[i] = prefix[i - 1] + nums[i - 1]

    return prefix

def range_sum(prefix: List[int], left: int, right: int) -> int:
    """
    left dan right gacha (kiritilgan) indekslardagi elementlar yig'indisini qaytaradi.
    O(1) so'rov uchun prefiks yig'indi massividan foydalanadi.

    Args:
        prefix: Prefiks yig'indi massivi
        left: Boshlang'ich indeks (kiritilgan)
        right: Tugash indeksi (kiritilgan)

    Returns:
        [left, right] diapazoni elementlari yig'indisi
    """
    # [left, right] diapazoni yig'indisi = prefix[right+1] - prefix[left]
    return prefix[right + 1] - prefix[left]`
		}
	}
};

export default task;
