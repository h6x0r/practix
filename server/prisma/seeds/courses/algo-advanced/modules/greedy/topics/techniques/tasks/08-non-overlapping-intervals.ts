import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-non-overlapping-intervals',
	title: 'Non-overlapping Intervals',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'sorting', 'intervals', 'dynamic-programming'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find minimum intervals to remove to make the rest non-overlapping.

**Problem:**

Given an array of intervals \`intervals\` where \`intervals[i] = [start, end]\`, return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

**Examples:**

\`\`\`
Input: intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
Output: 1

Explanation: Remove [1, 3] to make the rest non-overlapping.
Remaining: [1, 2], [2, 3], [3, 4] - no overlaps!

Input: intervals = [[1, 2], [1, 2], [1, 2]]
Output: 2

Explanation: Need to remove 2 intervals to have just 1 remaining.

Input: intervals = [[1, 2], [2, 3]]
Output: 0

Explanation: Already non-overlapping.
\`\`\`

**Key Insight:**

This is equivalent to: Keep maximum non-overlapping intervals (activity selection problem).

**Minimum to remove = Total intervals - Maximum non-overlapping**

Sort by end time, greedily select intervals that don't overlap with the last selected.

**Constraints:**
- 1 <= intervals.length <= 10^5
- intervals[i].length == 2
- -5 * 10^4 <= start < end <= 5 * 10^4

**Time Complexity:** O(n log n)
**Space Complexity:** O(1) or O(n) depending on sort`,
	initialCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    # TODO: Find minimum number of intervals to remove to make rest non-overlapping

    return 0`,
	solutionCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """
    Find minimum intervals to remove.
    """
    if not intervals:
        return 0

    # Sort by end time (key to greedy approach)
    intervals.sort(key=lambda x: x[1])

    count = 0  # Intervals to remove
    prev_end = float('-inf')

    for start, end in intervals:
        if start < prev_end:
            # Overlap detected - remove current interval
            count += 1
        else:
            # No overlap - keep this interval
            prev_end = end

    return count


# Alternative: Count kept, return total - kept
def erase_overlap_intervals_v2(intervals: List[List[int]]) -> int:
    """Count maximum non-overlapping, subtract from total."""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])

    kept = 1
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] >= prev_end:
            kept += 1
            prev_end = intervals[i][1]

    return len(intervals) - kept


# DP solution (O(n^2) - less efficient but demonstrates concept)
def erase_overlap_intervals_dp(intervals: List[List[int]]) -> int:
    """DP solution - similar to LIS."""
    if not intervals:
        return 0

    intervals.sort()
    n = len(intervals)

    # dp[i] = max non-overlapping intervals ending at or before i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if intervals[j][1] <= intervals[i][0]:
                dp[i] = max(dp[i], dp[j] + 1)

    return n - max(dp)


# Sort by start time approach
def erase_overlap_intervals_start_sort(intervals: List[List[int]]) -> int:
    """Alternative: sort by start, keep shorter when overlap."""
    if not intervals:
        return 0

    intervals.sort()

    count = 0
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        start, end = intervals[i]
        if start < prev_end:
            # Overlap - remove the one with later end
            count += 1
            prev_end = min(prev_end, end)
        else:
            prev_end = end

    return count


# Get which intervals to remove
def erase_overlap_intervals_detailed(intervals: List[List[int]]) -> tuple:
    """Return count and indices of removed intervals."""
    if not intervals:
        return 0, []

    indexed = [(start, end, i) for i, (start, end) in enumerate(intervals)]
    indexed.sort(key=lambda x: x[1])

    removed = []
    prev_end = float('-inf')

    for start, end, orig_idx in indexed:
        if start < prev_end:
            removed.append(orig_idx)
        else:
            prev_end = end

    return len(removed), removed`,
	testCode: `import pytest
from solution import erase_overlap_intervals


class TestNonOverlappingIntervals:
    def test_basic_case(self):
        """Test basic overlapping case"""
        assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1

    def test_duplicates(self):
        """Test duplicate intervals"""
        assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2

    def test_no_overlap(self):
        """Test already non-overlapping"""
        assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0

    def test_single_interval(self):
        """Test single interval"""
        assert erase_overlap_intervals([[1, 10]]) == 0

    def test_empty(self):
        """Test empty list"""
        assert erase_overlap_intervals([]) == 0

    def test_all_overlap(self):
        """Test all intervals overlap"""
        assert erase_overlap_intervals([[1, 10], [2, 9], [3, 8]]) == 2

    def test_chain(self):
        """Test chain of intervals"""
        assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [4, 5]]) == 0

    def test_nested(self):
        """Test nested intervals"""
        assert erase_overlap_intervals([[1, 10], [2, 3], [4, 5], [6, 7]]) == 1

    def test_negative_values(self):
        """Test negative values"""
        assert erase_overlap_intervals([[-10, -5], [-8, -2], [-4, 0]]) == 1

    def test_adjacent(self):
        """Test adjacent intervals (touching at endpoint)"""
        assert erase_overlap_intervals([[1, 3], [3, 5], [5, 7]]) == 0

    def test_verify_result(self):
        """Verify remaining intervals are non-overlapping"""
        intervals = [[1, 5], [2, 4], [3, 6], [5, 8]]
        removed = erase_overlap_intervals(intervals)

        # After removing 'removed' intervals, max kept = total - removed
        # Check that kept count is achievable
        intervals.sort(key=lambda x: x[1])
        kept = 0
        prev_end = float('-inf')
        for start, end in intervals:
            if start >= prev_end:
                kept += 1
                prev_end = end

        assert kept == len(intervals) - removed`,
	hint1: `This problem is the inverse of activity selection. Instead of counting maximum non-overlapping intervals, count minimum removals: total - maximum_kept.`,
	hint2: `Sort by end time. For each interval, if it overlaps with previous (start < prev_end), increment removal count. Otherwise, update prev_end and keep this interval.`,
	whyItMatters: `Non-overlapping Intervals demonstrates the connection between "maximize kept" and "minimize removed" problems. Understanding this duality helps solve many interval problems.

**Why This Matters:**

**1. Problem Duality**

\`\`\`python
# Two equivalent formulations:
# 1. Maximize non-overlapping intervals (Activity Selection)
# 2. Minimize intervals to remove

# Relationship:
minimum_removed = total - maximum_kept
\`\`\`

**2. Why Sort by End Time**

\`\`\`python
# Sorting by END time is key because:
# - Finishing early leaves more room for future intervals
# - If intervals overlap, remove the one that ends later
# - This greedy choice is provably optimal

# Example:
# [1, 10] vs [1, 2]: Keep [1, 2], more room for [3, 4], [5, 6], etc.
\`\`\`

**3. Comparison with DP**

\`\`\`python
# DP approach (similar to LIS):
# dp[i] = max non-overlapping ending at/before i
# O(n²) time

# Greedy approach:
# Sort by end, greedily keep non-overlapping
# O(n log n) time - much faster!

# Greedy works because of optimal substructure
\`\`\`

**4. Related Problems**

\`\`\`python
# Activity Selection: Keep maximum
# Merge Intervals: Combine overlapping
# Insert Interval: Insert and merge
# Meeting Rooms: Count overlaps
# Minimum Arrows to Burst Balloons: Same as this problem!
\`\`\`

**5. Minimum Arrows Problem**

\`\`\`python
# "Minimum arrows to burst balloons" is essentially:
# - How many groups of overlapping intervals?
# - Answer = Total intervals - Removals + Groups
# - Same greedy approach works!
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Непересекающиеся интервалы',
			description: `Найдите минимальное количество интервалов для удаления.

**Задача:**

Дан массив интервалов \`intervals[i] = [start, end]\`. Верните минимальное количество интервалов для удаления, чтобы остальные не пересекались.

**Примеры:**

\`\`\`
Вход: intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
Выход: 1

Объяснение: Удалите [1, 3], остальные не пересекаются.

Вход: intervals = [[1, 2], [1, 2], [1, 2]]
Выход: 2

Объяснение: Удалите 2 интервала, оставьте 1.
\`\`\`

**Ключевая идея:**

Эквивалентно: оставить максимум непересекающихся интервалов.

**Минимум удалений = Всего - Максимум оставленных**

Сортируйте по времени окончания, жадно выбирайте непересекающиеся.

**Ограничения:**
- 1 <= intervals.length <= 10^5

**Временная сложность:** O(n log n)
**Пространственная сложность:** O(1)`,
			hint1: `Эта задача - обратная к Activity Selection. Вместо подсчёта максимума непересекающихся, считаем минимум удалений: total - maximum_kept.`,
			hint2: `Сортируйте по концу. Если интервал пересекается с предыдущим (start < prev_end), увеличьте счётчик удалений. Иначе обновите prev_end.`,
			whyItMatters: `Non-overlapping Intervals демонстрирует связь между "максимизировать оставленные" и "минимизировать удалённые".

**Почему это важно:**

**1. Двойственность задач**

Минимум удалений = Всего - Максимум оставленных.

**2. Почему сортировка по концу**

Раннее окончание оставляет больше места для будущих интервалов.

**3. Связанные задачи**

Activity Selection, Merge Intervals, Minimum Arrows to Burst Balloons.`,
			solutionCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """Находит минимальное количество интервалов для удаления."""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = float('-inf')

    for start, end in intervals:
        if start < prev_end:
            count += 1
        else:
            prev_end = end

    return count`
		},
		uz: {
			title: 'Kesishmaydigan intervallar',
			description: `O'chirish kerak bo'lgan minimal intervallar sonini toping.

**Masala:**

\`intervals[i] = [start, end]\` intervallar massivi berilgan. Qolganlari kesishmasligi uchun minimal o'chirish kerak bo'lgan intervallar sonini qaytaring.

**Misollar:**

\`\`\`
Kirish: intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
Chiqish: 1

Izoh: [1, 3] ni o'chiring, qolganlari kesishmaydi.

Kirish: intervals = [[1, 2], [1, 2], [1, 2]]
Chiqish: 2

Izoh: 2 ta intervalni o'chiring, 1 ta qoldiring.
\`\`\`

**Asosiy tushuncha:**

Ekvivalent: maksimal kesishmaydigan intervallarni qoldirish.

**Minimal o'chirish = Jami - Maksimal qoldirilgan**

Tugash vaqti bo'yicha saralang, greedy usulda kesishmaydigan tanlab oling.

**Cheklovlar:**
- 1 <= intervals.length <= 10^5

**Vaqt murakkabligi:** O(n log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Bu masala Activity Selection ning teskari tomoni. Maksimal kesishmaydigan o'rniga minimal o'chirishni hisoblaymiz: jami - maksimal_qoldirilgan.`,
			hint2: `Tugash bo'yicha saralang. Agar interval oldingi bilan kesishsa (start < prev_end), o'chirish hisoblagichini oshiring. Aks holda prev_end ni yangilang.`,
			whyItMatters: `Non-overlapping Intervals "maksimal qoldirish" va "minimal o'chirish" masalalari orasidagi bog'liqlikni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Masalalar dualiteti**

Minimal o'chirish = Jami - Maksimal qoldirilgan.

**2. Tugash bo'yicha saralash sababi**

Erta tugash kelajakdagi intervallar uchun ko'proq joy qoldiradi.

**3. Bog'liq masalalar**

Activity Selection, Merge Intervals, Minimum Arrows to Burst Balloons.`,
			solutionCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """O'chirish kerak bo'lgan minimal intervallar sonini topadi."""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = float('-inf')

    for start, end in intervals:
        if start < prev_end:
            count += 1
        else:
            prev_end = end

    return count`
		}
	}
};

export default task;
