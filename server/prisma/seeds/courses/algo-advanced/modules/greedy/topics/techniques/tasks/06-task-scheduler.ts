import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-task-scheduler',
	title: 'Task Scheduler',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'array', 'heap', 'scheduling'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find minimum time to finish all tasks with cooldown constraint.

**Problem:**

Given a characters array \`tasks\` representing CPU tasks and a non-negative integer \`n\` representing the cooldown period between two **same** tasks.

Each task takes one unit of time. During the cooldown, the CPU can either run a different task or stay idle.

Return the minimum number of units of time the CPU will take to finish all tasks.

**Examples:**

\`\`\`
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8

Explanation: A -> B -> idle -> A -> B -> idle -> A -> B
One possible sequence with minimum time.

Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6

Explanation: No cooldown needed, just run tasks sequentially.

Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16

Explanation:
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
\`\`\`

**Key Insight:**

The most frequent task determines the minimum time. Arrange tasks in "frames" of size (n + 1), with the most frequent task starting each frame.

**Formula:**
\`\`\`
min_time = max(len(tasks), (max_freq - 1) * (n + 1) + count_of_max_freq_tasks)
\`\`\`

**Constraints:**
- 1 <= tasks.length <= 10^4
- tasks[i] is uppercase English letter
- 0 <= n <= 100

**Time Complexity:** O(n) or O(n log 26) with heap
**Space Complexity:** O(1) - at most 26 unique tasks`,
	initialCode: `from typing import List

def least_interval(tasks: List[str], n: int) -> int:
    # TODO: Find minimum time to complete all tasks with cooldown

    return 0`,
	solutionCode: `from typing import List
from collections import Counter
import heapq

def least_interval(tasks: List[str], n: int) -> int:
    """
    Find minimum time to complete all tasks.
    """
    if n == 0:
        return len(tasks)

    # Count frequency of each task
    freq = Counter(tasks)

    # Find maximum frequency
    max_freq = max(freq.values())

    # Count tasks with maximum frequency
    max_count = sum(1 for f in freq.values() if f == max_freq)

    # Formula: (max_freq - 1) complete frames + final partial frame
    # Each frame has (n + 1) slots
    # Final frame has max_count tasks (all tasks with max frequency)
    result = (max_freq - 1) * (n + 1) + max_count

    # Result can't be less than total tasks (when no idle needed)
    return max(result, len(tasks))


# Heap-based simulation (more intuitive)
def least_interval_heap(tasks: List[str], n: int) -> int:
    """Simulate task scheduling with max heap."""
    freq = Counter(tasks)
    # Max heap (negate for max behavior)
    max_heap = [-f for f in freq.values()]
    heapq.heapify(max_heap)

    time = 0

    while max_heap:
        temp = []
        # Process up to (n + 1) tasks in this round
        for _ in range(n + 1):
            if max_heap:
                count = -heapq.heappop(max_heap)
                if count > 1:
                    temp.append(-(count - 1))

            time += 1

            # If all tasks done, stop
            if not max_heap and not temp:
                break

        # Put remaining tasks back
        for item in temp:
            heapq.heappush(max_heap, item)

    return time


# Simulation with queue (tracks cooldown)
from collections import deque

def least_interval_queue(tasks: List[str], n: int) -> int:
    """Simulate with explicit cooldown tracking."""
    freq = Counter(tasks)
    max_heap = [-f for f in freq.values()]
    heapq.heapify(max_heap)

    time = 0
    # Queue stores (count, available_time)
    cooldown = deque()

    while max_heap or cooldown:
        time += 1

        # Check if any task is ready
        if cooldown and cooldown[0][1] <= time:
            count, _ = cooldown.popleft()
            heapq.heappush(max_heap, count)

        if max_heap:
            count = heapq.heappop(max_heap)
            if count + 1 < 0:  # Still tasks remaining
                cooldown.append((count + 1, time + n + 1))

    return time


# Get actual schedule
def least_interval_with_schedule(tasks: List[str], n: int) -> tuple:
    """Return minimum time and one possible schedule."""
    freq = Counter(tasks)
    max_heap = [(-f, task) for task, f in freq.items()]
    heapq.heapify(max_heap)

    schedule = []
    time = 0

    while max_heap:
        temp = []
        for _ in range(n + 1):
            if max_heap:
                count, task = heapq.heappop(max_heap)
                schedule.append(task)
                if count + 1 < 0:
                    temp.append((count + 1, task))
            elif temp or max_heap:  # Need idle
                schedule.append('idle')

            time += 1

            if not max_heap and not temp:
                break

        for item in temp:
            heapq.heappush(max_heap, item)

    return time, schedule`,
	testCode: `import pytest
from solution import least_interval


class TestTaskScheduler:
    def test_basic_case(self):
        """Test basic case with cooldown"""
        assert least_interval(["A","A","A","B","B","B"], 2) == 8

    def test_no_cooldown(self):
        """Test with no cooldown"""
        assert least_interval(["A","A","A","B","B","B"], 0) == 6

    def test_high_frequency(self):
        """Test with one high frequency task"""
        result = least_interval(["A","A","A","A","A","A","B","C","D","E","F","G"], 2)
        assert result == 16

    def test_single_task(self):
        """Test single task"""
        assert least_interval(["A"], 2) == 1

    def test_same_tasks(self):
        """Test all same tasks"""
        assert least_interval(["A","A","A"], 2) == 7  # A _ _ A _ _ A

    def test_no_idle_needed(self):
        """Test when no idle time needed"""
        assert least_interval(["A","B","C","D","E","F"], 2) == 6

    def test_many_same_frequency(self):
        """Test many tasks with same frequency"""
        assert least_interval(["A","B","A","B"], 2) == 5  # A B _ A B

    def test_cooldown_larger_than_tasks(self):
        """Test cooldown larger than unique tasks"""
        assert least_interval(["A","A","B","B"], 3) == 6  # A B _ _ A B

    def test_equal_distribution(self):
        """Test equal distribution of tasks"""
        tasks = ["A","A","A","B","B","B","C","C","C"]
        assert least_interval(tasks, 2) == 9

    def test_minimum_is_task_count(self):
        """Test when answer is just task count"""
        tasks = ["A","B","C","D","E","F","A","B"]
        assert least_interval(tasks, 1) == 8`,
	hint1: `Focus on the most frequent task - it determines the minimum time. Imagine arranging tasks in frames of size (n + 1), where each frame starts with the most frequent task.`,
	hint2: `Formula: (max_freq - 1) * (n + 1) + count_of_max_freq_tasks. This creates frames with cooldown slots. If total tasks > formula result, no idle time is needed.`,
	whyItMatters: `Task Scheduler demonstrates greedy scheduling with constraints. Understanding this pattern helps with CPU scheduling, resource allocation, and similar optimization problems.

**Why This Matters:**

**1. Frame-Based Thinking**

\`\`\`python
# Think of time as frames of size (n + 1):
# Frame 1: [A _ _]  (3 slots if n=2)
# Frame 2: [A _ _]
# Frame 3: [A]      (partial, just remaining A's)

# Fill empty slots with other tasks to minimize idle time
\`\`\`

**2. The Formula Explained**

\`\`\`python
# (max_freq - 1) complete frames, each of size (n + 1)
# + final partial frame with all max_freq tasks

# Example: A appears 3 times, n = 2
# Frame 1: [A B C]
# Frame 2: [A D E]
# Frame 3: [A]
# Total = (3-1) * 3 + 1 = 7

# If multiple tasks have max_freq:
# A,B,C each appear 3 times, n = 2
# Frame 3: [A B C] (3 tasks, not just 1)
# Total = (3-1) * 3 + 3 = 9
\`\`\`

**3. Heap Approach**

\`\`\`python
# More intuitive simulation:
# 1. Always pick most frequent remaining task
# 2. Track cooldowns
# 3. Idle if no task available

# Good for understanding, same complexity
\`\`\`

**4. Real-World Applications**

\`\`\`python
# CPU process scheduling
# Network packet scheduling
# Job scheduling with dependencies
# Resource cooldown management (games, APIs)
\`\`\`

**5. Variations**

\`\`\`python
# Task Scheduler II: Different cooldowns per task
# Task Scheduler with deadlines
# Task Scheduler with priorities
# Multi-processor task scheduling
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Планировщик задач',
			description: `Найдите минимальное время для выполнения всех задач с учётом перерыва.

**Задача:**

Дан массив символов \`tasks\` (задачи CPU) и целое число \`n\` - период остывания между одинаковыми задачами.

Каждая задача занимает одну единицу времени. Во время остывания CPU может выполнять другую задачу или простаивать.

Верните минимальное время для выполнения всех задач.

**Примеры:**

\`\`\`
Вход: tasks = ["A","A","A","B","B","B"], n = 2
Выход: 8

Объяснение: A -> B -> idle -> A -> B -> idle -> A -> B

Вход: tasks = ["A","A","A","B","B","B"], n = 0
Выход: 6

Объяснение: Без остывания, выполняем последовательно.
\`\`\`

**Ключевая идея:**

Наиболее частая задача определяет минимальное время. Расположите задачи во "фреймах" размера (n + 1).

**Формула:** max(len(tasks), (max_freq - 1) * (n + 1) + count_max)

**Ограничения:**
- 1 <= tasks.length <= 10^4
- 0 <= n <= 100

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Фокус на самой частой задаче - она определяет минимальное время. Представьте задачи во фреймах размера (n + 1), каждый начинается с самой частой задачи.`,
			hint2: `Формула: (max_freq - 1) * (n + 1) + count_max. Если total tasks > результата, idle не нужен.`,
			whyItMatters: `Task Scheduler демонстрирует жадное планирование с ограничениями.

**Почему это важно:**

**1. Мышление фреймами**

Время делится на фреймы размера (n + 1), пустые слоты заполняются другими задачами.

**2. Объяснение формулы**

(max_freq - 1) полных фреймов + финальный частичный фрейм.

**3. Применения**

Планирование CPU, сетевых пакетов, задач с зависимостями.`,
			solutionCode: `from typing import List
from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    """Находит минимальное время выполнения всех задач."""
    if n == 0:
        return len(tasks)

    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for f in freq.values() if f == max_freq)

    result = (max_freq - 1) * (n + 1) + max_count
    return max(result, len(tasks))`
		},
		uz: {
			title: 'Vazifalar rejalashtiruvchisi',
			description: `Barcha vazifalarni sovutish vaqti bilan bajarish uchun minimal vaqtni toping.

**Masala:**

\`tasks\` belgilar massivi (CPU vazifalari) va \`n\` butun soni - bir xil vazifalar orasidagi sovutish davri berilgan.

Har bir vazifa bir vaqt birligi oladi. Sovutish vaqtida CPU boshqa vazifa bajarishi yoki bo'sh turishi mumkin.

Barcha vazifalarni bajarish uchun minimal vaqtni qaytaring.

**Misollar:**

\`\`\`
Kirish: tasks = ["A","A","A","B","B","B"], n = 2
Chiqish: 8

Izoh: A -> B -> idle -> A -> B -> idle -> A -> B

Kirish: tasks = ["A","A","A","B","B","B"], n = 0
Chiqish: 6

Izoh: Sovutishsiz, ketma-ket bajaramiz.
\`\`\`

**Asosiy tushuncha:**

Eng tez-tez uchragan vazifa minimal vaqtni belgilaydi. Vazifalarni (n + 1) o'lchamli "freymlar"da joylashtiring.

**Formula:** max(len(tasks), (max_freq - 1) * (n + 1) + count_max)

**Cheklovlar:**
- 1 <= tasks.length <= 10^4
- 0 <= n <= 100

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Eng tez-tez uchraydigan vazifaga e'tibor bering - u minimal vaqtni belgilaydi. Vazifalarni (n + 1) o'lchamli freymlarda tasavvur qiling.`,
			hint2: `Formula: (max_freq - 1) * (n + 1) + count_max. Agar total tasks > natijadan bo'lsa, idle kerak emas.`,
			whyItMatters: `Task Scheduler cheklovli greedy rejalashtirishni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Freymlar bilan fikrlash**

Vaqt (n + 1) o'lchamli freymlarga bo'linadi, bo'sh slotlar boshqa vazifalar bilan to'ldiriladi.

**2. Formulaning tushuntirishi**

(max_freq - 1) to'liq freym + yakuniy qisman freym.

**3. Qo'llanishlar**

CPU rejalashtirish, tarmoq paketlari, bog'liq vazifalar.`,
			solutionCode: `from typing import List
from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    """Barcha vazifalarni bajarish uchun minimal vaqtni topadi."""
    if n == 0:
        return len(tasks)

    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for f in freq.values() if f == max_freq)

    result = (max_freq - 1) * (n + 1) + max_count
    return max(result, len(tasks))`
		}
	}
};

export default task;
