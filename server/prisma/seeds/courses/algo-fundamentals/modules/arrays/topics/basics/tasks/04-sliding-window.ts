import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-sliding-window-sum',
	title: 'Sliding Window Sum',
	difficulty: 'easy',
	tags: ['python', 'arrays', 'sliding-window'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the maximum sum of a subarray of size k.

**Problem:**

Given an array of integers \`nums\` and an integer \`k\`, find the maximum sum of any contiguous subarray of size \`k\`.

**Examples:**

\`\`\`
Input: nums = [2, 1, 5, 1, 3, 2], k = 3
Output: 9
Explanation: Subarray [5, 1, 3] has maximum sum = 9

Input: nums = [2, 3, 4, 1, 5], k = 2
Output: 7
Explanation: Subarray [3, 4] has maximum sum = 7

Input: nums = [1, 1, 1, 1, 1], k = 3
Output: 3
Explanation: Any subarray of size 3 has sum = 3
\`\`\`

**Sliding Window Approach:**

Instead of recalculating sum for each window:
1. Calculate sum of first window
2. Slide window: subtract outgoing element, add incoming element
3. Track maximum sum seen

\`\`\`python
# Initial: sum of first k elements
window_sum = sum(nums[:k])
# Slide: remove outgoing, add incoming
window_sum = window_sum - nums[i-k] + nums[i]
\`\`\`

**Time Complexity:** O(n) - single pass
**Space Complexity:** O(1) - only counters`,
	initialCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    # TODO: Find maximum sum of any contiguous subarray of size k

    return 0`,
	solutionCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    Find maximum sum of subarray with size k.

    Args:
        nums: List of integers
        k: Size of the sliding window

    Returns:
        Maximum sum of any contiguous subarray of size k
    """
    # Edge cases
    if len(nums) < k or k <= 0:
        return 0

    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide window through the rest of the array
    for i in range(k, len(nums)):
        # Remove element going out, add element coming in
        window_sum = window_sum - nums[i - k] + nums[i]

        # Update max if current window is larger
        max_sum = max(max_sum, window_sum)

    return max_sum`,
	testCode: `import pytest
from solution import max_sum_subarray

class TestMaxSumSubarray:
    def test_basic(self):
        """Test basic case with max in middle"""
        assert max_sum_subarray([2, 1, 5, 1, 3, 2], 3) == 9

    def test_start_max(self):
        """Test when max is at the start"""
        assert max_sum_subarray([5, 4, 1, 1, 1, 1], 2) == 9

    def test_end_max(self):
        """Test when max is at the end"""
        assert max_sum_subarray([1, 1, 1, 1, 4, 5], 2) == 9

    def test_equal_elements(self):
        """Test with all equal elements"""
        assert max_sum_subarray([1, 1, 1, 1, 1], 3) == 3

    def test_single_window(self):
        """Test when array size equals k"""
        assert max_sum_subarray([1, 2, 3], 3) == 6

    def test_k_equals_1(self):
        """Test with window size 1"""
        assert max_sum_subarray([1, 5, 3, 2], 1) == 5

    def test_negative_numbers(self):
        """Test with all negative numbers"""
        assert max_sum_subarray([-1, -2, -3, -4], 2) == -3

    def test_empty_array(self):
        """Test with empty array"""
        assert max_sum_subarray([], 3) == 0

    def test_k_larger_than_array(self):
        """Test when k is larger than array"""
        assert max_sum_subarray([1, 2], 5) == 0

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative numbers"""
        assert max_sum_subarray([3, -1, 4, -2, 5], 3) == 6`,
	hint1: `First calculate the sum of the first k elements using sum(nums[:k]). Then loop from index k to len(nums), updating the window sum by subtracting nums[i-k] and adding nums[i].`,
	hint2: `Remember to track the maximum sum seen so far. After each slide, use max_sum = max(max_sum, window_sum) to update if the current window is larger.`,
	whyItMatters: `The sliding window technique is essential for subarray problems.

**Why This Matters:**

**1. O(n) Instead of O(n*k)**

Naive approach recalculates sum for each window:
\`\`\`python
# O(n*k) - BAD
for i in range(len(nums) - k + 1):
    current_sum = sum(nums[i:i+k])
    max_sum = max(max_sum, current_sum)

# O(n) - GOOD (sliding window)
window_sum = sum(nums[:k])
for i in range(k, len(nums)):
    window_sum = window_sum - nums[i-k] + nums[i]
    max_sum = max(max_sum, window_sum)
\`\`\`

**2. Common Problem Patterns**

- Fixed-size window: maximum/minimum sum of k elements
- Variable-size window: smallest subarray with sum >= target
- Window with constraints: longest substring without repeating chars

**3. Real-World Applications**

- Network throughput monitoring (average over last N seconds)
- Stock price moving average
- Rate limiting (requests in sliding time window)
- Stream processing (aggregate over window)`,
	order: 4,
	translations: {
		ru: {
			title: 'Сумма скользящего окна',
			description: `Найдите максимальную сумму подмассива размера k.

**Задача:**

Дан массив целых чисел \`nums\` и целое число \`k\`, найдите максимальную сумму любого непрерывного подмассива размера \`k\`.

**Примеры:**

\`\`\`
Вход: nums = [2, 1, 5, 1, 3, 2], k = 3
Выход: 9
Объяснение: Подмассив [5, 1, 3] имеет максимальную сумму = 9

Вход: nums = [2, 3, 4, 1, 5], k = 2
Выход: 7
Объяснение: Подмассив [3, 4] имеет максимальную сумму = 7

Вход: nums = [1, 1, 1, 1, 1], k = 3
Выход: 3
Объяснение: Любой подмассив размера 3 имеет сумму = 3
\`\`\`

**Подход скользящего окна:**

Вместо пересчёта суммы для каждого окна:
1. Вычислите сумму первого окна
2. Сдвиньте окно: вычтите уходящий элемент, добавьте входящий
3. Отслеживайте максимальную виденную сумму

\`\`\`python
# Начальная: сумма первых k элементов
window_sum = sum(nums[:k])
# Сдвиг: убираем уходящий, добавляем входящий
window_sum = window_sum - nums[i-k] + nums[i]
\`\`\`

**Временная сложность:** O(n) - один проход
**Пространственная сложность:** O(1) - только счётчики`,
			hint1: `Сначала вычислите сумму первых k элементов с помощью sum(nums[:k]). Затем пройдите циклом от индекса k до len(nums), обновляя сумму окна вычитанием nums[i-k] и добавлением nums[i].`,
			hint2: `Не забудьте отслеживать максимальную виденную сумму. После каждого сдвига используйте max_sum = max(max_sum, window_sum) для обновления, если текущее окно больше.`,
			whyItMatters: `Техника скользящего окна необходима для задач с подмассивами.

**Почему это важно:**

**1. O(n) вместо O(n*k)**

Наивный подход пересчитывает сумму для каждого окна:
\`\`\`python
# O(n*k) - ПЛОХО
for i in range(len(nums) - k + 1):
    current_sum = sum(nums[i:i+k])
    max_sum = max(max_sum, current_sum)

# O(n) - ХОРОШО (скользящее окно)
window_sum = sum(nums[:k])
for i in range(k, len(nums)):
    window_sum = window_sum - nums[i-k] + nums[i]
    max_sum = max(max_sum, window_sum)
\`\`\`

**2. Общие паттерны задач**

- Фиксированный размер окна: максимальная/минимальная сумма k элементов
- Переменный размер окна: наименьший подмассив с суммой >= target
- Окно с ограничениями: самая длинная подстрока без повторяющихся символов

**3. Применения в реальном мире**

- Мониторинг пропускной способности сети (среднее за N секунд)
- Скользящее среднее цен акций
- Ограничение скорости запросов (requests в скользящем окне)
- Потоковая обработка (агрегация по окну)`,
			solutionCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    Находит максимальную сумму подмассива размера k.

    Args:
        nums: Список целых чисел
        k: Размер скользящего окна

    Returns:
        Максимальная сумма любого непрерывного подмассива размера k
    """
    # Граничные случаи
    if len(nums) < k or k <= 0:
        return 0

    # Вычисляем сумму первого окна
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Сдвигаем окно по остальному массиву
    for i in range(k, len(nums)):
        # Убираем уходящий элемент, добавляем входящий
        window_sum = window_sum - nums[i - k] + nums[i]

        # Обновляем максимум если текущее окно больше
        max_sum = max(max_sum, window_sum)

    return max_sum`
		},
		uz: {
			title: 'Sirg\'anuvchi oyna yig\'indisi',
			description: `k o'lchamli qism massivning maksimal yig'indisini toping.

**Masala:**

Butun sonlar massivi \`nums\` va butun son \`k\` berilgan, \`k\` o'lchamli har qanday uzluksiz qism massivning maksimal yig'indisini toping.

**Misollar:**

\`\`\`
Kirish: nums = [2, 1, 5, 1, 3, 2], k = 3
Chiqish: 9
Tushuntirish: [5, 1, 3] qism massivi maksimal yig'indiga ega = 9

Kirish: nums = [2, 3, 4, 1, 5], k = 2
Chiqish: 7
Tushuntirish: [3, 4] qism massivi maksimal yig'indiga ega = 7

Kirish: nums = [1, 1, 1, 1, 1], k = 3
Chiqish: 3
Tushuntirish: 3 o'lchamli har qanday qism massiv yig'indisi = 3
\`\`\`

**Sirg'anuvchi oyna yondashuvi:**

Har bir oyna uchun yig'indini qayta hisoblash o'rniga:
1. Birinchi oynaning yig'indisini hisoblang
2. Oynani siljiting: chiqayotgan elementni ayiring, kirayotganni qo'shing
3. Ko'rilgan maksimal yig'indini kuzating

\`\`\`python
# Boshlang'ich: birinchi k elementning yig'indisi
window_sum = sum(nums[:k])
# Siljitish: chiqayotganni olib tashlaymiz, kirayotganni qo'shamiz
window_sum = window_sum - nums[i-k] + nums[i]
\`\`\`

**Vaqt murakkabligi:** O(n) - bitta o'tish
**Xotira murakkabligi:** O(1) - faqat hisoblagichlar`,
			hint1: `Avval birinchi k elementning yig'indisini sum(nums[:k]) bilan hisoblang. Keyin k indeksdan len(nums) gacha tsikl qiling, oyna yig'indisini nums[i-k] ni ayirish va nums[i] ni qo'shish orqali yangilang.`,
			hint2: `Hozirgacha ko'rilgan maksimal yig'indini kuzatishni unutmang. Har bir siljishdan keyin max_sum = max(max_sum, window_sum) dan foydalaning.`,
			whyItMatters: `Sirg'anuvchi oyna texnikasi qism massiv masalalari uchun muhim.

**Bu nima uchun muhim:**

**1. O(n*k) o'rniga O(n)**

Oddiy yondashuv har bir oyna uchun yig'indini qayta hisoblaydi:
\`\`\`python
# O(n*k) - YOMON
for i in range(len(nums) - k + 1):
    current_sum = sum(nums[i:i+k])
    max_sum = max(max_sum, current_sum)

# O(n) - YAXSHI (sirg'anuvchi oyna)
window_sum = sum(nums[:k])
for i in range(k, len(nums)):
    window_sum = window_sum - nums[i-k] + nums[i]
    max_sum = max(max_sum, window_sum)
\`\`\`

**2. Umumiy masala patternlari**

- Belgilangan o'lchamli oyna: k elementning maksimal/minimal yig'indisi
- O'zgaruvchan o'lchamli oyna: yig'indisi >= target bo'lgan eng kichik qism massiv
- Cheklovli oyna: takrorlanmaydigan belgilar bilan eng uzun qism satr

**3. Haqiqiy dunyo qo'llanilishi**

- Tarmoq o'tkazuvchanligini monitoring qilish (oxirgi N soniya uchun o'rtacha)
- Aksiya narxlarining sirg'anuvchi o'rtachasi
- Tezlikni cheklash (sirg'anuvchi vaqt oynasidagi so'rovlar)
- Oqim qayta ishlash (oyna bo'ylab agregatsiya)`,
			solutionCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    k o'lchamli qism massivning maksimal yig'indisini topadi.

    Args:
        nums: Butun sonlar ro'yxati
        k: Sirg'anuvchi oyna o'lchami

    Returns:
        k o'lchamli har qanday uzluksiz qism massivning maksimal yig'indisi
    """
    # Chegara holatlari
    if len(nums) < k or k <= 0:
        return 0

    # Birinchi oynaning yig'indisini hisoblaymiz
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Oynani massivning qolgan qismi bo'ylab siljitamiz
    for i in range(k, len(nums)):
        # Chiqayotgan elementni olib tashlaymiz, kirayotganni qo'shamiz
        window_sum = window_sum - nums[i - k] + nums[i]

        # Joriy oyna kattaroq bo'lsa maksimumni yangilaymiz
        max_sum = max(max_sum, window_sum)

    return max_sum`
		}
	}
};

export default task;
