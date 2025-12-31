import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-activity-selection',
	title: 'Activity Selection',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'sorting', 'intervals'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Select the maximum number of non-overlapping activities.

**Problem:**

You are given \`n\` activities with their start and end times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on one activity at a time.

Activities are represented as pairs \`(start, end)\`.

**Examples:**

\`\`\`
Input: activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
Output: 4

Explanation: Selected activities: (1, 4), (5, 7), (8, 11), (12, 16)

Input: activities = [(1, 2), (3, 4), (0, 6), (5, 7)]
Output: 3

Explanation: Selected activities: (1, 2), (3, 4), (5, 7)

Input: activities = [(1, 3), (2, 4), (3, 5)]
Output: 2

Explanation: (1, 3) and (3, 5) or (2, 4) is not overlapping with either boundary
\`\`\`

**Key Insight:**

The greedy approach is to always pick the activity that finishes earliest. This leaves the maximum time for remaining activities.

**Algorithm:**
1. Sort activities by end time
2. Pick the first activity
3. For each subsequent activity, if its start time >= last selected end time, select it

**Constraints:**
- 1 <= activities.length <= 10^4
- 0 <= start < end <= 10^6

**Time Complexity:** O(n log n) for sorting
**Space Complexity:** O(1) extra space (excluding output)`,
	initialCode: `from typing import List, Tuple

def activity_selection(activities: List[Tuple[int, int]]) -> int:
    # TODO: Find maximum number of non-overlapping activities

    return 0`,
	solutionCode: `from typing import List, Tuple

def activity_selection(activities: List[Tuple[int, int]]) -> int:
    """
    Find maximum number of non-overlapping activities.
    """
    if not activities:
        return 0

    # Sort by end time (greedy: finish earliest first)
    activities = sorted(activities, key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count


def activity_selection_with_activities(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return the actual selected activities."""
    if not activities:
        return []

    # Sort by end time
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = [sorted_activities[0]]
    last_end = sorted_activities[0][1]

    for start, end in sorted_activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


# Weighted Activity Selection (needs DP)
def weighted_activity_selection(activities: List[Tuple[int, int, int]]) -> int:
    """
    Find maximum weight of non-overlapping activities.
    activities: List of (start, end, weight)
    """
    if not activities:
        return 0

    # Sort by end time
    activities = sorted(activities, key=lambda x: x[1])
    n = len(activities)

    # dp[i] = max weight considering activities 0..i
    dp = [0] * n
    dp[0] = activities[0][2]

    def binary_search_last_compatible(idx):
        """Find last activity that ends before activities[idx] starts."""
        target = activities[idx][0]
        lo, hi = 0, idx - 1
        result = -1

        while lo <= hi:
            mid = (lo + hi) // 2
            if activities[mid][1] <= target:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return result

    for i in range(1, n):
        # Option 1: Don't include activity i
        exclude = dp[i - 1]

        # Option 2: Include activity i
        include = activities[i][2]
        last_compatible = binary_search_last_compatible(i)
        if last_compatible != -1:
            include += dp[last_compatible]

        dp[i] = max(exclude, include)

    return dp[n - 1]`,
	testCode: `import pytest
from solution import activity_selection


class TestActivitySelection:
    def test_classic_example(self):
        """Test classic activity selection example"""
        activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
        assert activity_selection(activities) == 4

    def test_simple_case(self):
        """Test simple non-overlapping activities"""
        activities = [(1, 2), (3, 4), (5, 6)]
        assert activity_selection(activities) == 3

    def test_all_overlapping(self):
        """Test all activities overlap"""
        activities = [(1, 10), (2, 9), (3, 8), (4, 7)]
        assert activity_selection(activities) == 1

    def test_single_activity(self):
        """Test single activity"""
        activities = [(0, 5)]
        assert activity_selection(activities) == 1

    def test_empty_list(self):
        """Test empty activities list"""
        activities = []
        assert activity_selection(activities) == 0

    def test_adjacent_activities(self):
        """Test activities that end and start at same time"""
        activities = [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert activity_selection(activities) == 4

    def test_nested_intervals(self):
        """Test nested intervals"""
        activities = [(0, 10), (2, 3), (4, 5), (6, 7)]
        assert activity_selection(activities) == 3  # (2,3), (4,5), (6,7)

    def test_unsorted_input(self):
        """Test unsorted input activities"""
        activities = [(5, 9), (1, 2), (3, 4), (0, 6), (5, 7)]
        result = activity_selection(activities)
        assert result == 3  # (1,2), (3,4), (5,7)

    def test_same_end_time(self):
        """Test activities with same end time"""
        activities = [(0, 5), (1, 5), (2, 5), (3, 5)]
        assert activity_selection(activities) == 1

    def test_optimal_choice(self):
        """Test that greedy gives optimal result"""
        # Picking (0, 6) would only allow 2 activities
        # Picking (1, 2) and (3, 4) allows 3 activities
        activities = [(0, 6), (1, 2), (3, 4), (5, 7)]
        assert activity_selection(activities) == 3`,
	hint1: `Sort activities by their end time (finish time). This is the key to the greedy approach - always pick the activity that finishes earliest.`,
	hint2: `After sorting, iterate through activities. If current activity's start time >= last selected activity's end time, select it and update last_end.`,
	whyItMatters: `Activity Selection is the classic greedy algorithm problem that demonstrates the greedy choice property. It has wide applications in scheduling, resource allocation, and interval problems.

**Why This Matters:**

**1. Greedy Choice Property**

\`\`\`python
# Why sorting by END time works:
# If we pick the activity that finishes earliest,
# we leave maximum time for remaining activities.

# Proof by exchange argument:
# If optimal solution doesn't include earliest-finishing activity,
# we can swap it in without losing any activities.

activities.sort(key=lambda x: x[1])  # Sort by end time
\`\`\`

**2. Related Problems**

\`\`\`python
# Meeting Rooms II: Minimum rooms needed
# - Different approach: need to track overlaps

# Non-overlapping Intervals: Minimum removals
# - Same greedy strategy, count removals instead

# Merge Intervals: Combine overlapping intervals
# - Sort by start, merge greedily
\`\`\`

**3. Weighted Version (DP)**

\`\`\`python
# When activities have weights, greedy doesn't work
# Need dynamic programming + binary search

def weighted_activity(activities):
    # activities: (start, end, weight)
    activities.sort(key=lambda x: x[1])

    # dp[i] = max weight using activities 0..i
    # Binary search to find last compatible activity
\`\`\`

**4. Real-World Applications**

\`\`\`python
# Job scheduling on single machine
# Conference room booking
# CPU process scheduling
# Bandwidth allocation
# TV/Radio broadcasting schedules
\`\`\`

**5. Interval Scheduling Variations**

\`\`\`python
# Interval Partitioning: Minimum resources for all activities
# Weighted Job Scheduling: Maximize profit
# Interval Covering: Minimum points to cover all intervals
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Выбор активностей',
			description: `Выберите максимальное количество непересекающихся активностей.

**Задача:**

Дано \`n\` активностей с временем начала и окончания. Выберите максимальное количество активностей, которые может выполнить один человек.

**Примеры:**

\`\`\`
Вход: activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
Выход: 4

Объяснение: Выбранные активности: (1, 4), (5, 7), (8, 11), (12, 16)
\`\`\`

**Ключевая идея:**

Жадный подход - всегда выбирать активность, которая заканчивается раньше всех. Это оставляет максимум времени для оставшихся активностей.

**Ограничения:**
- 1 <= activities.length <= 10^4

**Временная сложность:** O(n log n)
**Пространственная сложность:** O(1)`,
			hint1: `Отсортируйте активности по времени окончания. Это ключ к жадному подходу - всегда выбирайте активность, которая заканчивается раньше.`,
			hint2: `После сортировки проходите по активностям. Если начало текущей >= конца последней выбранной, выберите её и обновите last_end.`,
			whyItMatters: `Activity Selection - классическая жадная задача, демонстрирующая свойство жадного выбора. Применяется в планировании и распределении ресурсов.

**Почему это важно:**

**1. Свойство жадного выбора**

Сортировка по времени окончания оставляет максимум времени для оставшихся активностей.

**2. Связанные задачи**

Meeting Rooms II, Non-overlapping Intervals, Merge Intervals.

**3. Взвешенная версия**

Когда активности имеют веса, нужно DP + бинарный поиск.`,
			solutionCode: `from typing import List, Tuple

def activity_selection(activities: List[Tuple[int, int]]) -> int:
    """Находит максимальное количество непересекающихся активностей."""
    if not activities:
        return 0

    # Сортировка по времени окончания
    activities = sorted(activities, key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count`
		},
		uz: {
			title: 'Faoliyatlarni tanlash',
			description: `Kesishmaydigan faoliyatlarning maksimal sonini tanlang.

**Masala:**

\`n\` ta faoliyat boshlanish va tugash vaqtlari bilan berilgan. Bir kishi bajarishi mumkin bo'lgan maksimal faoliyatlar sonini tanlang.

**Misollar:**

\`\`\`
Kirish: activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
Chiqish: 4

Izoh: Tanlangan faoliyatlar: (1, 4), (5, 7), (8, 11), (12, 16)
\`\`\`

**Asosiy tushuncha:**

Greedy yondashuv - har doim eng erta tugaydigan faoliyatni tanlash. Bu qolgan faoliyatlar uchun maksimal vaqt qoldiradi.

**Cheklovlar:**
- 1 <= activities.length <= 10^4

**Vaqt murakkabligi:** O(n log n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Faoliyatlarni tugash vaqti bo'yicha saralang. Bu greedy yondashuvning kaliti - har doim eng erta tugaydigan faoliyatni tanlang.`,
			hint2: `Saralashdan so'ng faoliyatlarni aylanib chiqing. Agar joriy boshlanish >= oxirgi tanlanganning tugashi, uni tanlang va last_end ni yangilang.`,
			whyItMatters: `Activity Selection - greedy tanlov xususiyatini ko'rsatadigan klassik greedy masala. Rejalashtirish va resurslarni taqsimlashda qo'llaniladi.

**Bu nima uchun muhim:**

**1. Greedy tanlov xususiyati**

Tugash vaqti bo'yicha saralash qolgan faoliyatlar uchun maksimal vaqt qoldiradi.

**2. Bog'liq masalalar**

Meeting Rooms II, Non-overlapping Intervals, Merge Intervals.

**3. Og'irlikli versiya**

Faoliyatlar og'irlikka ega bo'lganda DP + binary search kerak.`,
			solutionCode: `from typing import List, Tuple

def activity_selection(activities: List[Tuple[int, int]]) -> int:
    """Kesishmaydigan faoliyatlarning maksimal sonini topadi."""
    if not activities:
        return 0

    # Tugash vaqti bo'yicha saralash
    activities = sorted(activities, key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count`
		}
	}
};

export default task;
