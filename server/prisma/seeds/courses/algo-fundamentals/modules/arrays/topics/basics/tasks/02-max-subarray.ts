import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-max-subarray',
	title: 'Maximum Subarray',
	difficulty: 'easy',
	tags: ['python', 'arrays', 'dynamic-programming', 'kadane'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the contiguous subarray with the largest sum.

**Problem:**

Given an integer array \`nums\`, find the subarray with the largest sum and return its sum.

**Examples:**

\`\`\`
Input: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Output: 6
Explanation: The subarray [4, -1, 2, 1] has the largest sum = 6

Input: nums = [1]
Output: 1
Explanation: Single element is the maximum subarray

Input: nums = [5, 4, -1, 7, 8]
Output: 23
Explanation: The entire array [5, 4, -1, 7, 8] has the largest sum
\`\`\`

**Kadane's Algorithm:**

The key insight is: at each position, decide whether to:
1. Extend the previous subarray (add current element)
2. Start a new subarray from current element

\`\`\`python
current_sum = max(nums[i], current_sum + nums[i])
max_sum = max(max_sum, current_sum)
\`\`\`

**Why it works:**

If \`current_sum + nums[i] < nums[i]\`, the previous subarray has negative contribution, so start fresh.

**Time Complexity:** O(n) - single pass
**Space Complexity:** O(1) - only two variables`,
	initialCode: `from typing import List

def max_subarray(nums: List[int]) -> int:
    # TODO: Find the contiguous subarray with the largest sum

    return 0`,
	solutionCode: `from typing import List

def max_subarray(nums: List[int]) -> int:
    """
    Find the contiguous subarray with the largest sum.
    Returns the maximum sum.

    Args:
        nums: List of integers

    Returns:
        Maximum sum of any contiguous subarray
    """
    if not nums:
        return 0

    # Initialize both to first element
    max_sum = nums[0]
    current_sum = nums[0]

    # Start from second element
    for i in range(1, len(nums)):
        # Decision: extend previous subarray or start new one?
        # If current_sum + nums[i] < nums[i], start fresh
        current_sum = max(nums[i], current_sum + nums[i])

        # Update global maximum
        max_sum = max(max_sum, current_sum)

    return max_sum`,
	testCode: `import pytest
from solution import max_subarray

class TestMaxSubarray:
    def test_mixed_numbers(self):
        """Test array with positive and negative numbers"""
        assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6

    def test_single_element(self):
        """Test single element array"""
        assert max_subarray([1]) == 1

    def test_all_positive(self):
        """Test array with all positive numbers"""
        assert max_subarray([5, 4, -1, 7, 8]) == 23

    def test_all_negative(self):
        """Test array with all negative numbers"""
        assert max_subarray([-3, -2, -5, -1]) == -1

    def test_two_elements(self):
        """Test array with two elements"""
        assert max_subarray([-2, 1]) == 1

    def test_large_sum(self):
        """Test array with increasing positive numbers"""
        assert max_subarray([1, 2, 3, 4, 5]) == 15

    def test_alternating(self):
        """Test alternating positive and negative"""
        assert max_subarray([2, -1, 2, -1, 2]) == 4

    def test_single_negative(self):
        """Test single negative element"""
        assert max_subarray([-5]) == -5

    def test_zero_in_array(self):
        """Test array containing zero"""
        assert max_subarray([-2, 0, -1]) == 0

    def test_large_negative_gap(self):
        """Test with large negative number in middle"""
        assert max_subarray([5, -10, 3, 4]) == 7`,
	hint1: `Initialize max_sum and current_sum to nums[0]. Then iterate from index 1, updating current_sum as max(nums[i], current_sum + nums[i]).`,
	hint2: `At each step, decide: is it better to extend the current subarray or start fresh? If current_sum becomes negative, starting fresh (nums[i] alone) is better.`,
	whyItMatters: `Kadane's Algorithm is a fundamental dynamic programming technique.

**Why This Matters:**

**1. Classic DP Pattern**

This problem teaches the core DP concept:
- **Optimal substructure**: max subarray ending at i depends on max subarray ending at i-1
- **Overlapping subproblems**: we build solution incrementally

\`\`\`python
# The recurrence relation:
# dp[i] = max(nums[i], dp[i-1] + nums[i])
#
# Space-optimized to O(1):
current_sum = max(nums[i], current_sum + nums[i])
\`\`\`

**2. Variations and Extensions**

Once you understand Kadane's, you can solve:
- Maximum Product Subarray (track min and max)
- Maximum Circular Subarray (Kadane + total sum trick)
- Maximum Subarray with at most K negatives
- 2D Maximum Subarray (Kadane per column)

**3. Real-World Applications**

- Stock price analysis (max profit in a period)
- Signal processing (peak detection)
- Financial analysis (best consecutive period)
- Image processing (brightest region)

**4. Interview Frequency**

This is one of the most common interview questions:
- Tests understanding of DP
- Simple to state, tricky to optimize
- Many follow-up variations possible`,
	order: 2,
	translations: {
		ru: {
			title: 'Максимальный подмассив',
			description: `Найдите непрерывный подмассив с наибольшей суммой.

**Задача:**

Дан массив целых чисел \`nums\`, найдите подмассив с наибольшей суммой и верните эту сумму.

**Примеры:**

\`\`\`
Вход: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Выход: 6
Объяснение: Подмассив [4, -1, 2, 1] имеет наибольшую сумму = 6

Вход: nums = [1]
Выход: 1
Объяснение: Единственный элемент является максимальным подмассивом

Вход: nums = [5, 4, -1, 7, 8]
Выход: 23
Объяснение: Весь массив [5, 4, -1, 7, 8] имеет наибольшую сумму
\`\`\`

**Алгоритм Кадане:**

Ключевая идея: на каждой позиции решите:
1. Расширить предыдущий подмассив (добавить текущий элемент)
2. Начать новый подмассив с текущего элемента

\`\`\`python
current_sum = max(nums[i], current_sum + nums[i])
max_sum = max(max_sum, current_sum)
\`\`\`

**Почему это работает:**

Если \`current_sum + nums[i] < nums[i]\`, предыдущий подмассив имеет отрицательный вклад, поэтому начинаем заново.

**Временная сложность:** O(n) - один проход
**Пространственная сложность:** O(1) - только две переменные`,
			hint1: `Инициализируйте max_sum и current_sum значением nums[0]. Затем итерируйте с индекса 1, обновляя current_sum как max(nums[i], current_sum + nums[i]).`,
			hint2: `На каждом шаге решите: лучше расширить текущий подмассив или начать заново? Если current_sum становится отрицательной, начать заново (только nums[i]) лучше.`,
			whyItMatters: `Алгоритм Кадане - это фундаментальная техника динамического программирования.

**Почему это важно:**

**1. Классический паттерн DP**

Эта задача учит основной концепции DP:
- **Оптимальная подструктура**: макс. подмассив, заканчивающийся на i, зависит от макс. подмассива, заканчивающегося на i-1
- **Перекрывающиеся подзадачи**: мы строим решение инкрементально

\`\`\`python
# Рекуррентное соотношение:
# dp[i] = max(nums[i], dp[i-1] + nums[i])
#
# Оптимизация памяти до O(1):
current_sum = max(nums[i], current_sum + nums[i])
\`\`\`

**2. Вариации и расширения**

Поняв Кадане, вы сможете решить:
- Maximum Product Subarray (отслеживать min и max)
- Maximum Circular Subarray (Кадане + трюк с общей суммой)
- 2D Maximum Subarray (Кадане по столбцам)

**3. Применение в реальном мире**

- Анализ цен акций (максимальная прибыль за период)
- Обработка сигналов (обнаружение пиков)
- Финансовый анализ (лучший последовательный период)`,
			solutionCode: `from typing import List

def max_subarray(nums: List[int]) -> int:
    """
    Находит непрерывный подмассив с наибольшей суммой.
    Возвращает максимальную сумму.

    Args:
        nums: Список целых чисел

    Returns:
        Максимальная сумма любого непрерывного подмассива
    """
    if not nums:
        return 0

    # Инициализируем оба первым элементом
    max_sum = nums[0]
    current_sum = nums[0]

    # Начинаем со второго элемента
    for i in range(1, len(nums)):
        # Решение: расширить предыдущий подмассив или начать новый?
        # Если current_sum + nums[i] < nums[i], начинаем заново
        current_sum = max(nums[i], current_sum + nums[i])

        # Обновляем глобальный максимум
        max_sum = max(max_sum, current_sum)

    return max_sum`
		},
		uz: {
			title: 'Maksimal qism massiv',
			description: `Eng katta yig'indiga ega bo'lgan uzluksiz qism massivni toping.

**Masala:**

Butun sonlar massivi \`nums\` berilgan, eng katta yig'indiga ega qism massivni toping va uning yig'indisini qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Chiqish: 6
Tushuntirish: [4, -1, 2, 1] qism massivi eng katta yig'indiga ega = 6

Kirish: nums = [1]
Chiqish: 1
Tushuntirish: Yagona element maksimal qism massiv

Kirish: nums = [5, 4, -1, 7, 8]
Chiqish: 23
Tushuntirish: Butun massiv [5, 4, -1, 7, 8] eng katta yig'indiga ega
\`\`\`

**Kadane algoritmi:**

Asosiy g'oya: har bir pozitsiyada qaror qiling:
1. Oldingi qism massivni kengaytirish (joriy elementni qo'shish)
2. Joriy elementdan yangi qism massiv boshlash

\`\`\`python
current_sum = max(nums[i], current_sum + nums[i])
max_sum = max(max_sum, current_sum)
\`\`\`

**Nima uchun ishlaydi:**

Agar \`current_sum + nums[i] < nums[i]\` bo'lsa, oldingi qism massiv manfiy hissa qo'shadi, shuning uchun yangi boshlaymiz.

**Vaqt murakkabligi:** O(n) - bitta o'tish
**Xotira murakkabligi:** O(1) - faqat ikkita o'zgaruvchi`,
			hint1: `max_sum va current_sum ni nums[0] bilan ishga tushiring. Keyin indeks 1 dan boshlab, current_sum ni max(nums[i], current_sum + nums[i]) sifatida yangilang.`,
			hint2: `Har bir qadamda qaror qiling: joriy qism massivni kengaytirish yaxshimi yoki yangi boshlash? Agar current_sum manfiy bo'lsa, yangi boshlash (faqat nums[i]) yaxshiroq.`,
			whyItMatters: `Kadane algoritmi asosiy dinamik dasturlash texnikasi.

**Bu nima uchun muhim:**

**1. Klassik DP pattern**

Bu masala asosiy DP kontseptsiyasini o'rgatadi:
- **Optimal tuzilma**: i da tugaydigan maks qism massiv i-1 da tugaydigan maks qism massivga bog'liq
- **Ustma-ust tushadigan kichik masalalar**: yechimni bosqichma-bosqich quramiz

\`\`\`python
# Rekurrent munosabat:
# dp[i] = max(nums[i], dp[i-1] + nums[i])
#
# Xotirani O(1) ga optimallashtirish:
current_sum = max(nums[i], current_sum + nums[i])
\`\`\`

**2. Variatsiyalar va kengaytmalar**

Kadane ni tushunganingizdan so'ng, quyidagilarni hal qilishingiz mumkin:
- Maximum Product Subarray (min va max ni kuzatish)
- Maximum Circular Subarray (Kadane + umumiy yig'indi hiylasi)
- 2D Maximum Subarray (har bir ustun uchun Kadane)

**3. Haqiqiy dunyo qo'llanilishi**

- Aksiya narxlarini tahlil qilish (davrdagi maksimal foyda)
- Signal qayta ishlash (cho'qqi aniqlash)
- Moliyaviy tahlil (eng yaxshi ketma-ket davr)`,
			solutionCode: `from typing import List

def max_subarray(nums: List[int]) -> int:
    """
    Eng katta yig'indiga ega uzluksiz qism massivni topadi.
    Maksimal yig'indini qaytaradi.

    Args:
        nums: Butun sonlar ro'yxati

    Returns:
        Har qanday uzluksiz qism massivning maksimal yig'indisi
    """
    if not nums:
        return 0

    # Ikkalasini ham birinchi element bilan ishga tushiramiz
    max_sum = nums[0]
    current_sum = nums[0]

    # Ikkinchi elementdan boshlaymiz
    for i in range(1, len(nums)):
        # Qaror: oldingi qism massivni kengaytirishmi yoki yangi boshlashmi?
        # Agar current_sum + nums[i] < nums[i] bo'lsa, yangi boshlaymiz
        current_sum = max(nums[i], current_sum + nums[i])

        # Global maksimumni yangilaymiz
        max_sum = max(max_sum, current_sum)

    return max_sum`
		}
	}
};

export default task;
