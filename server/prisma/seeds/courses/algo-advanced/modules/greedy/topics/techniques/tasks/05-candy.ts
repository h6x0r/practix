import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-candy',
	title: 'Candy Distribution',
	difficulty: 'hard',
	tags: ['python', 'greedy', 'array', 'two-pass'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Distribute candies to children standing in a line.

**Problem:**

There are \`n\` children standing in a line. Each child is assigned a rating value given in the integer array \`ratings\`.

You are giving candies to these children with the following requirements:

1. Each child must have at least one candy
2. Children with a higher rating get more candies than their neighbors

Return the **minimum** number of candies you need to distribute.

**Examples:**

\`\`\`
Input: ratings = [1, 0, 2]
Output: 5

Explanation:
- Child 0: rating 1 -> 2 candies (more than neighbor with rating 0)
- Child 1: rating 0 -> 1 candy (minimum)
- Child 2: rating 2 -> 2 candies (more than neighbor with rating 0)
Total: 2 + 1 + 2 = 5

Input: ratings = [1, 2, 2]
Output: 4

Explanation:
- Child 0: 1 candy
- Child 1: 2 candies (higher rating than child 0)
- Child 2: 1 candy (same rating as child 1, no requirement)
Total: 1 + 2 + 1 = 4

Input: ratings = [1, 3, 2, 2, 1]
Output: 7

Explanation: [1, 2, 1, 2, 1] candies
\`\`\`

**Key Insight:**

Use two passes:
1. Left to right: ensure each child has more candies than left neighbor if rating is higher
2. Right to left: ensure each child has more candies than right neighbor if rating is higher

Take maximum of both passes at each position.

**Constraints:**
- n == ratings.length
- 1 <= n <= 2 * 10^4
- 0 <= ratings[i] <= 2 * 10^4

**Time Complexity:** O(n)
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def candy(ratings: List[int]) -> int:
    # TODO: Find minimum candies to distribute

    return 0`,
	solutionCode: `from typing import List

def candy(ratings: List[int]) -> int:
    """
    Find minimum candies to distribute.
    """
    n = len(ratings)
    if n == 0:
        return 0

    candies = [1] * n

    # Pass 1: Left to right
    # Ensure higher rating than left neighbor gets more candies
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Pass 2: Right to left
    # Ensure higher rating than right neighbor gets more candies
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)


# O(1) space solution (more complex)
def candy_constant_space(ratings: List[int]) -> int:
    """O(n) time, O(1) space solution."""
    n = len(ratings)
    if n <= 1:
        return n

    candies = 0
    up = 0
    down = 0
    old_slope = 0

    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            new_slope = 1
        elif ratings[i] < ratings[i - 1]:
            new_slope = -1
        else:
            new_slope = 0

        if (old_slope > 0 and new_slope == 0) or (old_slope < 0 and new_slope >= 0):
            candies += sum_sequence(up) + sum_sequence(down) + max(up, down)
            up = 0
            down = 0

        if new_slope > 0:
            up += 1
        elif new_slope < 0:
            down += 1
        else:
            candies += 1

        old_slope = new_slope

    candies += sum_sequence(up) + sum_sequence(down) + max(up, down) + 1
    return candies


def sum_sequence(n: int) -> int:
    """Sum of 1 + 2 + ... + n"""
    return n * (n + 1) // 2


# Visualization helper
def candy_with_distribution(ratings: List[int]) -> tuple:
    """Return total candies and distribution."""
    n = len(ratings)
    if n == 0:
        return 0, []

    candies = [1] * n

    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies), candies


# Three-pass solution (more intuitive)
def candy_three_pass(ratings: List[int]) -> int:
    """Three pass solution for clarity."""
    n = len(ratings)

    left = [1] * n
    right = [1] * n

    # Pass 1: Compare with left neighbor
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1

    # Pass 2: Compare with right neighbor
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            right[i] = right[i + 1] + 1

    # Pass 3: Take maximum at each position
    return sum(max(left[i], right[i]) for i in range(n))`,
	testCode: `import pytest
from solution import candy


class TestCandy:
    def test_basic_case(self):
        """Test basic case"""
        assert candy([1, 0, 2]) == 5

    def test_same_neighbors(self):
        """Test with same rating neighbors"""
        assert candy([1, 2, 2]) == 4

    def test_decreasing(self):
        """Test strictly decreasing ratings"""
        assert candy([3, 2, 1]) == 6  # [3, 2, 1]

    def test_increasing(self):
        """Test strictly increasing ratings"""
        assert candy([1, 2, 3]) == 6  # [1, 2, 3]

    def test_single_child(self):
        """Test single child"""
        assert candy([5]) == 1

    def test_two_children_increasing(self):
        """Test two children increasing"""
        assert candy([1, 2]) == 3

    def test_two_children_decreasing(self):
        """Test two children decreasing"""
        assert candy([2, 1]) == 3

    def test_same_ratings(self):
        """Test all same ratings"""
        assert candy([1, 1, 1, 1]) == 4

    def test_peak(self):
        """Test peak in middle"""
        assert candy([1, 2, 3, 2, 1]) == 9  # [1, 2, 3, 2, 1]

    def test_valley(self):
        """Test valley in middle"""
        assert candy([3, 2, 1, 2, 3]) == 9  # [3, 2, 1, 2, 3]

    def test_complex(self):
        """Test complex ratings"""
        assert candy([1, 3, 2, 2, 1]) == 7  # [1, 2, 1, 2, 1]

    def test_verify_constraints(self):
        """Verify candy distribution meets constraints"""
        ratings = [1, 3, 4, 5, 2]
        total, distribution = candy(ratings), None

        # Reconstruct distribution for verification
        n = len(ratings)
        candies = [1] * n
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)

        # Verify constraints
        for i in range(n):
            assert candies[i] >= 1
            if i > 0 and ratings[i] > ratings[i - 1]:
                assert candies[i] > candies[i - 1]
            if i < n - 1 and ratings[i] > ratings[i + 1]:
                assert candies[i] > candies[i + 1]`,
	hint1: `Think about handling left and right neighbors separately. First pass: ensure higher rating than left neighbor gets more candies. Second pass: ensure higher rating than right neighbor gets more candies.`,
	hint2: `In the second pass (right to left), use max(current, right_neighbor + 1) to preserve constraints from the first pass while adding new constraints.`,
	whyItMatters: `Candy Distribution demonstrates the two-pass greedy technique for handling bidirectional constraints. This pattern appears in many interview problems requiring optimization with neighbor relationships.

**Why This Matters:**

**1. Two-Pass Technique**

\`\`\`python
# Many problems with bidirectional constraints use two passes:
# Pass 1: Handle one direction
# Pass 2: Handle other direction, preserving pass 1 results

# Examples:
# - Trapping Rain Water
# - Product of Array Except Self
# - Best Time to Buy and Sell Stock variations
\`\`\`

**2. Why Two Passes Work**

\`\`\`python
# After pass 1 (left to right):
# Each child with higher rating than LEFT neighbor has more candies

# After pass 2 (right to left):
# Using max() ensures we keep pass 1 constraints
# While adding constraints for RIGHT neighbors

# The max() is key - it satisfies BOTH constraints
candies[i] = max(candies[i], candies[i + 1] + 1)
\`\`\`

**3. Space Optimization**

\`\`\`python
# Can reduce from O(n) to O(1) space by:
# - Tracking "up" and "down" slopes
# - Using arithmetic series sum formula
# - More complex but same time complexity
\`\`\`

**4. Similar Problems**

\`\`\`python
# Trapping Rain Water: Two passes for left/right max heights
# Product Except Self: Two passes for left/right products
# Stock problems: Track prefix/suffix min/max
# Queue Reconstruction: Sort + greedy insertion
\`\`\`

**5. Interview Insight**

\`\`\`python
# When you see constraints involving neighbors:
# 1. Can we handle one direction at a time?
# 2. Can we combine results with max/min/sum?
# 3. What's the invariant after each pass?
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Распределение конфет',
			description: `Раздайте конфеты детям, стоящим в очереди.

**Задача:**

\`n\` детей стоят в очереди. Каждому присвоен рейтинг из массива \`ratings\`.

Требования:
1. Каждый ребёнок должен получить минимум одну конфету
2. Дети с более высоким рейтингом получают больше конфет, чем соседи

Верните **минимальное** количество конфет.

**Примеры:**

\`\`\`
Вход: ratings = [1, 0, 2]
Выход: 5

Объяснение: 2 + 1 + 2 = 5 конфет

Вход: ratings = [1, 2, 2]
Выход: 4

Объяснение: 1 + 2 + 1 = 4 конфет
\`\`\`

**Ключевая идея:**

Два прохода:
1. Слева направо: больше конфет чем у левого соседа если рейтинг выше
2. Справа налево: больше конфет чем у правого соседа если рейтинг выше

Берём максимум на каждой позиции.

**Ограничения:**
- 1 <= n <= 2 * 10^4

**Временная сложность:** O(n)
**Пространственная сложность:** O(n)`,
			hint1: `Обрабатывайте левых и правых соседей отдельно. Первый проход: выше рейтинг чем у левого - больше конфет. Второй проход: выше рейтинг чем у правого - больше конфет.`,
			hint2: `Во втором проходе используйте max(текущее, правый_сосед + 1), чтобы сохранить ограничения из первого прохода.`,
			whyItMatters: `Candy Distribution демонстрирует технику двух проходов для двунаправленных ограничений.

**Почему это важно:**

**1. Техника двух проходов**

Многие задачи с двунаправленными ограничениями решаются двумя проходами.

**2. Почему работает**

max() обеспечивает выполнение ОБОИХ ограничений.

**3. Похожие задачи**

Trapping Rain Water, Product Except Self, задачи с акциями.`,
			solutionCode: `from typing import List

def candy(ratings: List[int]) -> int:
    """Находит минимальное количество конфет."""
    n = len(ratings)
    if n == 0:
        return 0

    candies = [1] * n

    # Проход 1: Слева направо
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Проход 2: Справа налево
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)`
		},
		uz: {
			title: 'Konfetlarni taqsimlash',
			description: `Navbatda turgan bolalarga konfetlarni tarqating.

**Masala:**

\`n\` bola navbatda turibdi. Har biriga \`ratings\` massividagi reyting berilgan.

Talablar:
1. Har bir bola kamida bitta konfet olishi kerak
2. Yuqori reytingli bolalar qo'shnilardan ko'proq konfet oladi

**Minimal** konfetlar sonini qaytaring.

**Misollar:**

\`\`\`
Kirish: ratings = [1, 0, 2]
Chiqish: 5

Izoh: 2 + 1 + 2 = 5 konfet

Kirish: ratings = [1, 2, 2]
Chiqish: 4

Izoh: 1 + 2 + 1 = 4 konfet
\`\`\`

**Asosiy tushuncha:**

Ikki o'tish:
1. Chapdan o'ngga: reyting yuqori bo'lsa chap qo'shnidan ko'proq konfet
2. O'ngdan chapga: reyting yuqori bo'lsa o'ng qo'shnidan ko'proq konfet

Har bir pozitsiyada maksimumni oling.

**Cheklovlar:**
- 1 <= n <= 2 * 10^4

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Chap va o'ng qo'shnilarni alohida ishlating. Birinchi o'tish: chap qo'shnidan yuqori reyting - ko'proq konfet. Ikkinchi o'tish: o'ng qo'shnidan yuqori reyting - ko'proq konfet.`,
			hint2: `Ikkinchi o'tishda max(joriy, o'ng_qo'shni + 1) ishlating, birinchi o'tishdagi cheklovlarni saqlash uchun.`,
			whyItMatters: `Candy Distribution ikki tomonlama cheklovlar uchun ikki o'tish texnikasini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Ikki o'tish texnikasi**

Ko'p ikki tomonlama cheklovli masalalar ikki o'tish bilan yechiladi.

**2. Nima uchun ishlaydi**

max() IKKALA cheklovni ham qondiradi.

**3. O'xshash masalalar**

Trapping Rain Water, Product Except Self, aktsiya masalalari.`,
			solutionCode: `from typing import List

def candy(ratings: List[int]) -> int:
    """Minimal konfetlar sonini topadi."""
    n = len(ratings)
    if n == 0:
        return 0

    candies = [1] * n

    # O'tish 1: Chapdan o'ngga
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # O'tish 2: O'ngdan chapga
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)`
		}
	}
};

export default task;
