import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-meeting-rooms',
	title: 'Meeting Rooms II',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'sorting', 'heap', 'intervals', 'two-pointers'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the minimum number of conference rooms required.

**Problem:**

Given an array of meeting time intervals \`intervals\` where \`intervals[i] = [start, end]\`, return the minimum number of conference rooms required.

**Examples:**

\`\`\`
Input: intervals = [[0, 30], [5, 10], [15, 20]]
Output: 2

Explanation:
- Room 1: Meeting [0, 30]
- Room 2: Meeting [5, 10], then Meeting [15, 20]

Input: intervals = [[7, 10], [2, 4]]
Output: 1

Explanation: Meetings don't overlap, one room is enough.

Input: intervals = [[1, 5], [2, 6], [3, 7], [4, 8]]
Output: 4

Explanation: All meetings overlap, need 4 rooms.
\`\`\`

**Visualization:**

\`\`\`
intervals = [[0, 30], [5, 10], [15, 20]]

Timeline:
0    5    10   15   20   25   30
|----+----+----+----+----+----+
[=========Meeting 1============]
     [M2]
               [M3]

At time 5: 2 meetings overlap -> need 2 rooms
Maximum overlap = 2
\`\`\`

**Key Insight:**

Track the maximum number of overlapping meetings at any point in time. This can be done by:
1. Sorting events (starts/ends) and sweeping
2. Using a min-heap to track ongoing meetings

**Constraints:**
- 1 <= intervals.length <= 10^4
- 0 <= start < end <= 10^6

**Time Complexity:** O(n log n) for sorting/heap
**Space Complexity:** O(n)`,
	initialCode: `from typing import List

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    # TODO: Find minimum number of conference rooms needed

    return 0`,
	solutionCode: `from typing import List
import heapq

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Find minimum conference rooms needed.
    """
    if not intervals:
        return 0

    # Approach 1: Two pointers (optimal)
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])

    rooms = 0
    end_ptr = 0

    for start in starts:
        if start < ends[end_ptr]:
            # Meeting starts before earliest end - need new room
            rooms += 1
        else:
            # Meeting starts after earliest end - reuse room
            end_ptr += 1

    return rooms


# Approach 2: Min-heap
def min_meeting_rooms_heap(intervals: List[List[int]]) -> int:
    """Use min-heap to track ongoing meetings."""
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min-heap of end times
    heap = []

    for start, end in intervals:
        # If current meeting starts after earliest ending meeting
        if heap and start >= heap[0]:
            heapq.heappop(heap)  # Reuse that room

        heapq.heappush(heap, end)

    return len(heap)


# Approach 3: Line sweep (event-based)
def min_meeting_rooms_sweep(intervals: List[List[int]]) -> int:
    """Line sweep algorithm."""
    events = []

    for start, end in intervals:
        events.append((start, 1))   # Meeting starts: +1 room
        events.append((end, -1))    # Meeting ends: -1 room

    # Sort by time, then by type (ends before starts at same time)
    events.sort(key=lambda x: (x[0], x[1]))

    rooms = 0
    max_rooms = 0

    for _, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)

    return max_rooms


# Get room assignments
def min_meeting_rooms_with_assignment(intervals: List[List[int]]) -> tuple:
    """Return room count and assignments."""
    if not intervals:
        return 0, []

    # Sort with original indices
    indexed = [(start, end, i) for i, (start, end) in enumerate(intervals)]
    indexed.sort()

    # Min-heap of (end_time, room_number)
    heap = []
    assignments = [0] * len(intervals)
    next_room = 0

    for start, end, orig_idx in indexed:
        if heap and start >= heap[0][0]:
            # Reuse room
            _, room = heapq.heappop(heap)
        else:
            # Need new room
            room = next_room
            next_room += 1

        assignments[orig_idx] = room
        heapq.heappush(heap, (end, room))

    return next_room, assignments`,
	testCode: `import pytest
from solution import min_meeting_rooms


class TestMeetingRooms:
    def test_basic_overlap(self):
        """Test basic overlapping meetings"""
        assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2

    def test_no_overlap(self):
        """Test non-overlapping meetings"""
        assert min_meeting_rooms([[7, 10], [2, 4]]) == 1

    def test_all_overlap(self):
        """Test all meetings overlap"""
        assert min_meeting_rooms([[1, 5], [2, 6], [3, 7], [4, 8]]) == 4

    def test_single_meeting(self):
        """Test single meeting"""
        assert min_meeting_rooms([[1, 10]]) == 1

    def test_empty(self):
        """Test empty intervals"""
        assert min_meeting_rooms([]) == 0

    def test_same_time(self):
        """Test meetings at same time"""
        assert min_meeting_rooms([[1, 5], [1, 5], [1, 5]]) == 3

    def test_adjacent_meetings(self):
        """Test adjacent meetings (end time = start time)"""
        assert min_meeting_rooms([[1, 2], [2, 3], [3, 4]]) == 1

    def test_nested_meetings(self):
        """Test nested meetings"""
        assert min_meeting_rooms([[1, 10], [2, 5], [6, 9]]) == 2

    def test_partial_overlap(self):
        """Test partial overlaps"""
        assert min_meeting_rooms([[1, 4], [2, 5], [4, 7]]) == 2

    def test_many_short_meetings(self):
        """Test many short meetings"""
        intervals = [[i, i + 1] for i in range(10)]
        assert min_meeting_rooms(intervals) == 1

    def test_complex_case(self):
        """Test complex scheduling"""
        intervals = [[0, 5], [1, 3], [2, 6], [4, 8], [5, 9]]
        assert min_meeting_rooms(intervals) == 3`,
	hint1: `Think about what happens at each point in time. Track when meetings start (need new room) and when they end (room freed). The maximum concurrent meetings = minimum rooms.`,
	hint2: `Two-pointer approach: Sort starts and ends separately. Iterate through starts. If current start < earliest end, need new room. Otherwise, one meeting ended, reuse that room.`,
	whyItMatters: `Meeting Rooms II demonstrates interval scheduling and the sweep line algorithm. These concepts are fundamental for calendar applications, resource allocation, and many real-world scheduling problems.

**Why This Matters:**

**1. Three Approaches**

\`\`\`python
# 1. Two Pointers (most efficient)
starts = sorted([i[0] for i in intervals])
ends = sorted([i[1] for i in intervals])
# Compare start with earliest end

# 2. Min-Heap (intuitive)
# Track end times of ongoing meetings
# Reuse room if new meeting starts after earliest end

# 3. Line Sweep (general purpose)
# Events: (time, +1 for start, -1 for end)
# Track running sum, max = answer
\`\`\`

**2. Related Problems**

\`\`\`python
# Meeting Rooms I: Can attend all? (just check overlap)
# Merge Intervals: Combine overlapping intervals
# Insert Interval: Insert and merge
# Employee Free Time: Find gaps across schedules
# Car Pooling: Track capacity over time
\`\`\`

**3. Why Two Pointers Work**

\`\`\`python
# Key insight: We don't need to know WHICH meeting ends
# We just need to know IF a meeting has ended

# At any start time:
# - If a meeting ended before this start -> reuse room
# - Otherwise -> need new room

# Sorting separately gives us this information efficiently
\`\`\`

**4. Real-World Applications**

\`\`\`python
# Conference room booking systems
# Server capacity planning
# Classroom scheduling
# Operating theater allocation
# Vehicle fleet management
\`\`\`

**5. Complexity Trade-offs**

\`\`\`python
# Two pointers: O(n log n) time, O(n) space
# Min-heap: O(n log n) time, O(n) space
# Line sweep: O(n log n) time, O(n) space

# Two pointers has better constants
# Heap is more intuitive
# Sweep is most general
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Конференц-залы II',
			description: `Найдите минимальное количество переговорных комнат.

**Задача:**

Дан массив интервалов встреч \`intervals[i] = [start, end]\`. Верните минимальное количество необходимых комнат.

**Примеры:**

\`\`\`
Вход: intervals = [[0, 30], [5, 10], [15, 20]]
Выход: 2

Объяснение:
- Комната 1: Встреча [0, 30]
- Комната 2: Встреча [5, 10], затем [15, 20]

Вход: intervals = [[7, 10], [2, 4]]
Выход: 1

Объяснение: Встречи не пересекаются, хватит одной комнаты.
\`\`\`

**Ключевая идея:**

Отслеживайте максимальное количество одновременных встреч. Можно использовать:
1. Сортировку событий и сканирование
2. Min-heap для отслеживания текущих встреч

**Ограничения:**
- 1 <= intervals.length <= 10^4

**Временная сложность:** O(n log n)
**Пространственная сложность:** O(n)`,
			hint1: `Подумайте, что происходит в каждый момент времени. Отслеживайте начала (нужна комната) и окончания (комната освобождается). Максимум одновременных = минимум комнат.`,
			hint2: `Два указателя: отсортируйте начала и концы отдельно. Если текущее начало < раннего конца - нужна новая комната. Иначе - используйте освободившуюся.`,
			whyItMatters: `Meeting Rooms II демонстрирует планирование интервалов и алгоритм сканирующей линии.

**Почему это важно:**

**1. Три подхода**

Два указателя (эффективный), Min-Heap (интуитивный), Line Sweep (универсальный).

**2. Связанные задачи**

Meeting Rooms I, Merge Intervals, Insert Interval, Employee Free Time.

**3. Применения**

Системы бронирования, планирование серверов, расписание классов.`,
			solutionCode: `from typing import List

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """Находит минимальное количество переговорных комнат."""
    if not intervals:
        return 0

    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])

    rooms = 0
    end_ptr = 0

    for start in starts:
        if start < ends[end_ptr]:
            rooms += 1
        else:
            end_ptr += 1

    return rooms`
		},
		uz: {
			title: 'Yig\'ilish xonalari II',
			description: `Minimal konferentsiya xonalari sonini toping.

**Masala:**

\`intervals[i] = [start, end]\` yig'ilish vaqtlari massivi berilgan. Kerakli minimal xonalar sonini qaytaring.

**Misollar:**

\`\`\`
Kirish: intervals = [[0, 30], [5, 10], [15, 20]]
Chiqish: 2

Izoh:
- Xona 1: Yig'ilish [0, 30]
- Xona 2: Yig'ilish [5, 10], keyin [15, 20]

Kirish: intervals = [[7, 10], [2, 4]]
Chiqish: 1

Izoh: Yig'ilishlar kesishmaydi, bitta xona yetarli.
\`\`\`

**Asosiy tushuncha:**

Bir vaqtning o'zida maksimal yig'ilishlar sonini kuzating:
1. Hodisalarni saralash va skanerlash
2. Joriy yig'ilishlarni kuzatish uchun min-heap

**Cheklovlar:**
- 1 <= intervals.length <= 10^4

**Vaqt murakkabligi:** O(n log n)
**Xotira murakkabligi:** O(n)`,
			hint1: `Har bir vaqt nuqtasida nima sodir bo'layotganini o'ylang. Boshlanishlar (xona kerak) va tugashlar (xona bo'shaydi) ni kuzating. Maksimal bir vaqtdagi = minimal xonalar.`,
			hint2: `Ikki ko'rsatkich: boshlanish va tugashlarni alohida saralang. Joriy boshlanish < erta tugashdan bo'lsa - yangi xona kerak. Aks holda - bo'shaganni ishlating.`,
			whyItMatters: `Meeting Rooms II interval rejalashtirish va line sweep algoritmini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Uchta yondashuv**

Ikki ko'rsatkich (samarali), Min-Heap (intuitiv), Line Sweep (universal).

**2. Bog'liq masalalar**

Meeting Rooms I, Merge Intervals, Insert Interval, Employee Free Time.

**3. Qo'llanishlar**

Bron qilish tizimlari, server rejalashtirish, sinf jadvallari.`,
			solutionCode: `from typing import List

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """Minimal konferentsiya xonalari sonini topadi."""
    if not intervals:
        return 0

    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])

    rooms = 0
    end_ptr = 0

    for start in starts:
        if start < ends[end_ptr]:
            rooms += 1
        else:
            end_ptr += 1

    return rooms`
		}
	}
};

export default task;
