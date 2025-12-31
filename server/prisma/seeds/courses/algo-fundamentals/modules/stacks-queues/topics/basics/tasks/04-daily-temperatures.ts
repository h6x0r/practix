import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-daily-temperatures',
	title: 'Daily Temperatures',
	difficulty: 'medium',
	tags: ['python', 'stack', 'monotonic-stack'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find how many days until a warmer temperature.

**Problem:**

Given an array of integers \`temperatures\` representing daily temperatures, return an array \`answer\` such that \`answer[i]\` is the number of days you have to wait after day \`i\` for a warmer temperature.

If there is no future day with a warmer temperature, \`answer[i] = 0\`.

**Examples:**

\`\`\`
Input: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]

Explanation:
Day 0 (73): Day 1 is warmer (74), wait 1 day
Day 1 (74): Day 2 is warmer (75), wait 1 day
Day 2 (75): Day 6 is warmer (76), wait 4 days
...

Input: temperatures = [30, 40, 50, 60]
Output: [1, 1, 1, 0]

Input: temperatures = [30, 60, 90]
Output: [1, 1, 0]
\`\`\`

**Monotonic Stack Approach:**

Use a decreasing stack of indices:
1. For each temperature, pop from stack while current > stack top
2. Each pop finds its answer (current index - popped index)
3. Push current index to stack

**Time Complexity:** O(n)
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def daily_temperatures(temperatures: List[int]) -> List[int]:
    # TODO: Return days until warmer temperature for each day

    return []`,
	solutionCode: `from typing import List


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Return days until warmer temperature.

    Args:
        temperatures: List of daily temperatures

    Returns:
        List where each element is days to wait for warmer temp
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stack of indices

    for i in range(n):
        # Pop while current temperature is warmer
        while stack and temperatures[i] > temperatures[stack[-1]]:
            # Pop index
            j = stack.pop()

            # Calculate days to wait
            result[j] = i - j

        # Push current index
        stack.append(i)

    # Indices remaining in stack have no warmer day (result stays 0)
    return result`,
	testCode: `import pytest
from solution import daily_temperatures


class TestDailyTemperatures:
    def test_basic(self):
        """Test basic case"""
        temps = [73, 74, 75, 71, 69, 72, 76, 73]
        assert daily_temperatures(temps) == [1, 1, 4, 2, 1, 1, 0, 0]

    def test_increasing(self):
        """Test increasing temperatures"""
        assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]

    def test_short(self):
        """Test short array"""
        assert daily_temperatures([30, 60, 90]) == [1, 1, 0]

    def test_decreasing(self):
        """Test decreasing temperatures"""
        assert daily_temperatures([90, 80, 70, 60]) == [0, 0, 0, 0]

    def test_single(self):
        """Test single element"""
        assert daily_temperatures([50]) == [0]

    def test_same(self):
        """Test same temperatures"""
        assert daily_temperatures([50, 50]) == [0, 0]

    def test_all_same(self):
        """Test all same temperatures"""
        assert daily_temperatures([70, 70, 70, 70]) == [0, 0, 0, 0]

    def test_two_elements(self):
        """Test two elements"""
        assert daily_temperatures([60, 70]) == [1, 0]

    def test_alternating(self):
        """Test alternating temperatures"""
        assert daily_temperatures([50, 60, 40, 70]) == [1, 2, 1, 0]

    def test_large_gap(self):
        """Test large gap to warmer day"""
        assert daily_temperatures([30, 40, 50, 60, 100]) == [1, 1, 1, 1, 0]`,
	hint1: `Use a stack to store indices (not values). When you find a warmer day, you can calculate the distance using current index - stack index.`,
	hint2: `The stack maintains a decreasing sequence of temperatures. When you find a warmer temperature, pop all indices with smaller temperatures and record their answer.`,
	whyItMatters: `This problem introduces the monotonic stack pattern.

**Why This Matters:**

**1. Monotonic Stack Pattern**

A powerful technique for "next greater/smaller" problems:
\`\`\`python
# Decreasing stack: find next greater element
# Increasing stack: find next smaller element

# Template:
for i in range(len(arr)):
    while stack and arr[i] > arr[stack[-1]]:
        j = stack.pop()
        result[j] = i  # or some calculation
    stack.append(i)
\`\`\`

**2. Common Applications**

- Next Greater Element (I, II, III)
- Stock Span Problem
- Largest Rectangle in Histogram
- Trapping Rain Water
- Sum of Subarray Minimums

**3. Why It's O(n)**

Each element is pushed once and popped at most once:
\`\`\`
Total operations = n pushes + n pops = O(n)
\`\`\`

**4. Store Indices, Not Values**

We need indices to calculate distances:
\`\`\`python
# Store indices
result[j] = i - j  # Can calculate distance

# If we stored values, we'd need extra lookup
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Дневные температуры',
			description: `Найдите сколько дней ждать до более тёплой температуры.

**Задача:**

Дан массив целых чисел \`temperatures\`, представляющий дневные температуры, верните массив \`answer\`, где \`answer[i]\` - количество дней ожидания после дня \`i\` до более тёплой температуры.

Если нет будущего дня с более тёплой температурой, \`answer[i] = 0\`.

**Примеры:**

\`\`\`
Вход: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Выход: [1, 1, 4, 2, 1, 1, 0, 0]

Объяснение:
День 0 (73): День 1 теплее (74), ждать 1 день
День 1 (74): День 2 теплее (75), ждать 1 день
День 2 (75): День 6 теплее (76), ждать 4 дня
...
\`\`\`

**Подход с монотонным стеком:**

Используйте убывающий стек индексов:
1. Для каждой температуры извлекайте из стека пока текущая > вершины
2. Каждое извлечение находит свой ответ (текущий индекс - извлечённый)
3. Добавьте текущий индекс в стек

**Временная сложность:** O(n)
**Пространственная сложность:** O(n)`,
			hint1: `Используйте стек для хранения индексов (не значений). Когда находите более тёплый день, можете вычислить расстояние используя текущий индекс - индекс из стека.`,
			hint2: `Стек поддерживает убывающую последовательность температур. Когда находите более тёплую температуру, извлеките все индексы с меньшими температурами и запишите их ответ.`,
			whyItMatters: `Эта задача знакомит с паттерном монотонного стека.

**Почему это важно:**

**1. Паттерн монотонного стека**

Мощная техника для задач "следующий больший/меньший".

**2. Распространённые применения**

- Next Greater Element
- Stock Span Problem
- Largest Rectangle in Histogram`,
			solutionCode: `from typing import List


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Возвращает дни до более тёплой температуры.

    Args:
        temperatures: Список дневных температур

    Returns:
        Список, где каждый элемент - дней ожидания до более тёплой
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Стек индексов

    for i in range(n):
        # Извлекаем пока текущая температура теплее
        while stack and temperatures[i] > temperatures[stack[-1]]:
            # Извлекаем индекс
            j = stack.pop()

            # Вычисляем дни ожидания
            result[j] = i - j

        # Добавляем текущий индекс
        stack.append(i)

    # Индексы в стеке не имеют более тёплого дня (result остаётся 0)
    return result`
		},
		uz: {
			title: 'Kunlik haroratlar',
			description: `Iliqroq haroratgacha qancha kun kutish kerakligini toping.

**Masala:**

Kunlik haroratlarni ifodalovchi butun sonlar massivi \`temperatures\` berilgan, \`answer[i]\` \`i\` kundan keyin iliqroq haroratgacha kutish kunlari soni bo'lgan \`answer\` massivini qaytaring.

Agar kelajakda iliqroq kun bo'lmasa, \`answer[i] = 0\`.

**Misollar:**

\`\`\`
Kirish: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Chiqish: [1, 1, 4, 2, 1, 1, 0, 0]

Tushuntirish:
0-kun (73): 1-kun iliqroq (74), 1 kun kutish
1-kun (74): 2-kun iliqroq (75), 1 kun kutish
2-kun (75): 6-kun iliqroq (76), 4 kun kutish
...
\`\`\`

**Monoton stek yondashuvi:**

Indekslarning kamayuvchi stekidan foydalaning:
1. Har bir harorat uchun joriy > stek tepasi bo'lguncha stekdan oling
2. Har bir olish o'z javobini topadi (joriy indeks - olingan indeks)
3. Joriy indeksni stekga qo'shing

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Indekslarni (qiymatlarni emas) saqlash uchun stekdan foydalaning. Iliqroq kun topganingizda, joriy indeks - stek indeksi yordamida masofani hisoblashingiz mumkin.`,
			hint2: `Stek haroratlarning kamayuvchi ketma-ketligini saqlaydi. Iliqroq harorat topganingizda, kichikroq haroratli barcha indekslarni oling va ularning javobini yozing.`,
			whyItMatters: `Bu masala monoton stek patternini tanishtiradi.

**Bu nima uchun muhim:**

**1. Monoton stek patterni**

"Keyingi kattaroq/kichikroq" masalalari uchun kuchli texnika.

**2. Keng tarqalgan qo'llanilishlar**

- Next Greater Element
- Stock Span Problem
- Largest Rectangle in Histogram`,
			solutionCode: `from typing import List


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Iliqroq haroratgacha kunlarni qaytaradi.

    Args:
        temperatures: Kunlik haroratlar ro'yxati

    Returns:
        Har bir element iliqroq haroratgacha kutish kunlari ro'yxati
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Indekslar steki

    for i in range(n):
        # Joriy harorat iliqroq bo'lguncha olamiz
        while stack and temperatures[i] > temperatures[stack[-1]]:
            # Indeksni olamiz
            j = stack.pop()

            # Kutish kunlarini hisoblaymiz
            result[j] = i - j

        # Joriy indeksni qo'shamiz
        stack.append(i)

    # Stekda qolgan indekslarning iliqroq kuni yo'q (result 0 qoladi)
    return result`
		}
	}
};

export default task;
