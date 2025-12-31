import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-jump-game-ii',
	title: 'Jump Game II',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'array', 'dynamic-programming'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the minimum number of jumps to reach the last index.

**Problem:**

You are given a 0-indexed array of integers \`nums\` of length \`n\`. You are initially positioned at \`nums[0]\`.

Each element \`nums[i]\` represents the maximum length of a forward jump from index \`i\`.

Return the minimum number of jumps to reach \`nums[n - 1]\`. The test cases are generated such that you can reach \`nums[n - 1]\`.

**Examples:**

\`\`\`
Input: nums = [2, 3, 1, 1, 4]
Output: 2

Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Path: 0 -> 1 -> 4

Input: nums = [2, 3, 0, 1, 4]
Output: 2

Explanation: Jump 1 step from 0 to 1, then 3 steps to 4.

Input: nums = [1, 1, 1, 1]
Output: 3

Explanation: Must jump through each index.
\`\`\`

**Visualization:**

\`\`\`
nums = [2, 3, 1, 1, 4]

Jump 1: From index 0, can reach indices 1 or 2
        Best choice: index 1 (nums[1] = 3 gives farthest reach)

Jump 2: From index 1, can reach indices 2, 3, or 4
        Can reach last index!

Total jumps: 2

Window approach:
Index:     0   1   2   3   4
          [2] [3, 1] [1, 4]
           ^   current_end
           Jump when i == current_end
\`\`\`

**Key Insight:**

Use BFS-like level traversal. Track the farthest reachable position and jump count. Increment jumps when reaching the end of current "level".

**Constraints:**
- 1 <= nums.length <= 10^4
- 0 <= nums[i] <= 1000
- It's guaranteed that you can reach nums[n-1]

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def jump(nums: List[int]) -> int:
    # TODO: Find minimum number of jumps to reach the last index

    return 0`,
	solutionCode: `from typing import List

def jump(nums: List[int]) -> int:
    """
    Find minimum jumps to reach the last index.
    """
    n = len(nums)
    if n <= 1:
        return 0

    jumps = 0
    current_end = 0  # End of current jump's reach
    farthest = 0     # Farthest position we can reach

    # Iterate up to n-2 (we don't need to jump from last index)
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        # Reached the end of current jump's range
        if i == current_end:
            jumps += 1
            current_end = farthest

            # Early exit if we can already reach the end
            if current_end >= n - 1:
                break

    return jumps


# BFS solution (more intuitive but same complexity)
from collections import deque

def jump_bfs(nums: List[int]) -> int:
    """BFS solution treating positions as graph nodes."""
    n = len(nums)
    if n <= 1:
        return 0

    visited = [False] * n
    visited[0] = True
    queue = deque([(0, 0)])  # (position, jumps)

    while queue:
        pos, jumps = queue.popleft()

        for next_pos in range(pos + 1, min(pos + nums[pos] + 1, n)):
            if next_pos >= n - 1:
                return jumps + 1
            if not visited[next_pos]:
                visited[next_pos] = True
                queue.append((next_pos, jumps + 1))

    return -1  # Should never reach here given problem constraints


# DP solution (O(n^2) - less efficient)
def jump_dp(nums: List[int]) -> int:
    """Dynamic programming solution."""
    n = len(nums)
    # dp[i] = minimum jumps to reach position i
    dp = [float('inf')] * n
    dp[0] = 0

    for i in range(n):
        for j in range(i + 1, min(i + nums[i] + 1, n)):
            dp[j] = min(dp[j], dp[i] + 1)

    return dp[n - 1]


# Tracking the actual path
def jump_with_path(nums: List[int]) -> tuple:
    """Return minimum jumps and the path taken."""
    n = len(nums)
    if n <= 1:
        return 0, [0]

    # Track parent for path reconstruction
    parent = [-1] * n
    jumps = 0
    current_end = 0
    farthest = 0
    farthest_idx = 0

    for i in range(n - 1):
        if i + nums[i] > farthest:
            farthest = i + nums[i]
            farthest_idx = i

        if i == current_end:
            jumps += 1
            current_end = farthest

            # Mark the jump
            for j in range(current_end + 1):
                if parent[j] == -1 and j > i:
                    parent[j] = i

            if current_end >= n - 1:
                break

    # Reconstruct path
    path = []
    pos = n - 1
    while pos > 0:
        # Find where we jumped from
        for i in range(pos - 1, -1, -1):
            if i + nums[i] >= pos:
                path.append(i)
                pos = i
                break
    path.append(0)
    path.reverse()

    return jumps, path`,
	testCode: `import pytest
from solution import jump


class TestJumpGameII:
    def test_basic(self):
        """Test basic case"""
        assert jump([2, 3, 1, 1, 4]) == 2

    def test_with_zeros(self):
        """Test with zeros in array"""
        assert jump([2, 3, 0, 1, 4]) == 2

    def test_all_ones(self):
        """Test all ones"""
        assert jump([1, 1, 1, 1]) == 3

    def test_single_element(self):
        """Test single element"""
        assert jump([0]) == 0

    def test_two_elements(self):
        """Test two elements"""
        assert jump([1, 1]) == 1

    def test_large_first_jump(self):
        """Test when first jump reaches end"""
        assert jump([5, 1, 1, 1, 1]) == 1

    def test_optimal_path(self):
        """Test optimal path selection"""
        # Jumping to 1 then to 4 is optimal (2 jumps)
        # vs jumping to 2 (would need more jumps)
        assert jump([3, 4, 1, 1, 1, 1]) == 2

    def test_large_array(self):
        """Test larger array"""
        nums = [2] * 100
        result = jump(nums)
        assert result == 50  # Each jump covers 2 positions

    def test_decreasing_jumps(self):
        """Test decreasing jump values"""
        assert jump([4, 3, 2, 1, 0]) == 1  # Can reach directly

    def test_need_maximum_reach(self):
        """Test when need to use maximum reach"""
        assert jump([1, 2, 1, 1, 1]) == 3`,
	hint1: `Think of this as BFS on a graph where each index connects to positions it can jump to. Track the farthest reachable position at each "level" (number of jumps).`,
	hint2: `Use two pointers: current_end (end of current level) and farthest (maximum reach so far). When i reaches current_end, increment jumps and update current_end to farthest.`,
	whyItMatters: `Jump Game II demonstrates how to find minimum steps in reachability problems using a greedy BFS-like approach. It's more complex than Jump Game I and shows the power of "level traversal" thinking.

**Why This Matters:**

**1. BFS Level Traversal Insight**

\`\`\`python
# Think of positions as BFS levels:
# Level 0: [0]
# Level 1: all positions reachable in 1 jump
# Level 2: all positions reachable in 2 jumps
# ...

# current_end marks the end of current level
# farthest marks the end of next level
for i in range(n - 1):
    farthest = max(farthest, i + nums[i])
    if i == current_end:  # End of current level
        jumps += 1         # Go to next level
        current_end = farthest
\`\`\`

**2. Greedy Choice Property**

\`\`\`python
# Why greedy works:
# At each level, we explore ALL positions
# We track the farthest we can reach from any position
# This guarantees minimum jumps to reach any distance
\`\`\`

**3. Jump Game Variations**

\`\`\`python
# Jump Game I: Can reach end? (boolean)
# Jump Game II: Minimum jumps (count)
# Jump Game III: Can reach index with value 0? (bidirectional)
# Jump Game IV: Min jumps with same-value teleport
# Jump Game V: Max indices visited with constraints
\`\`\`

**4. Applications**

\`\`\`python
# Minimum bus/train transfers
# Minimum cable lengths in network
# Game level design (minimum moves)
# Resource delivery with limited range
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Игра в прыжки II',
			description: `Найдите минимальное количество прыжков до последнего индекса.

**Задача:**

Дан массив целых чисел \`nums\`. Вы начинаете с \`nums[0]\`. Каждый элемент \`nums[i]\` представляет максимальную длину прыжка вперёд.

Верните минимальное количество прыжков до \`nums[n - 1]\`. Гарантируется, что последний индекс достижим.

**Примеры:**

\`\`\`
Вход: nums = [2, 3, 1, 1, 4]
Выход: 2

Объяснение: Прыжок на 1 шаг от индекса 0 к 1, затем 3 шага к последнему.
Путь: 0 -> 1 -> 4

Вход: nums = [1, 1, 1, 1]
Выход: 3

Объяснение: Нужно прыгать через каждый индекс.
\`\`\`

**Ключевая идея:**

Используйте BFS-подобный обход по уровням. Отслеживайте самую дальнюю позицию и счётчик прыжков.

**Ограничения:**
- 1 <= nums.length <= 10^4

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Представьте это как BFS на графе, где каждый индекс соединён с позициями, куда можно прыгнуть. Отслеживайте farthest на каждом "уровне".`,
			hint2: `Используйте current_end (конец текущего уровня) и farthest (максимальная дальность). Когда i достигает current_end, увеличьте jumps и обновите current_end.`,
			whyItMatters: `Jump Game II демонстрирует поиск минимальных шагов в задачах достижимости с жадным BFS-подходом.

**Почему это важно:**

**1. BFS обход по уровням**

current_end отмечает конец текущего уровня, farthest - конец следующего.

**2. Свойство жадного выбора**

На каждом уровне исследуем ВСЕ позиции и отслеживаем максимальную дальность.

**3. Вариации Jump Game**

I - можно ли достичь, II - минимум прыжков, III - двунаправленные прыжки, IV - телепортация.`,
			solutionCode: `from typing import List

def jump(nums: List[int]) -> int:
    """Находит минимальное количество прыжков до последнего индекса."""
    n = len(nums)
    if n <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= n - 1:
                break

    return jumps`
		},
		uz: {
			title: "Sakrash o'yini II",
			description: `Oxirgi indeksga yetish uchun minimal sakrashlar sonini toping.

**Masala:**

\`nums\` butun sonlar massivi berilgan. Siz \`nums[0]\` dan boshlaysiz. Har bir element \`nums[i]\` maksimal oldinga sakrash uzunligini bildiradi.

\`nums[n - 1]\` ga yetish uchun minimal sakrashlar sonini qaytaring. Oxirgi indeksga yetib borish kafolatlanadi.

**Misollar:**

\`\`\`
Kirish: nums = [2, 3, 1, 1, 4]
Chiqish: 2

Izoh: 0-indeksdan 1-ga 1 qadam, keyin oxirgi indeksga 3 qadam.
Yo'l: 0 -> 1 -> 4

Kirish: nums = [1, 1, 1, 1]
Chiqish: 3

Izoh: Har bir indeks orqali sakrash kerak.
\`\`\`

**Asosiy tushuncha:**

BFS-ga o'xshash daraja bo'yicha aylanish ishlating. Eng uzoq pozitsiya va sakrashlar sonini kuzating.

**Cheklovlar:**
- 1 <= nums.length <= 10^4

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Buni graf bo'yicha BFS deb tasavvur qiling, har bir indeks sakrash mumkin bo'lgan pozitsiyalarga ulangan. Har bir "darajada" farthest ni kuzating.`,
			hint2: `current_end (joriy daraja oxiri) va farthest (maksimal masofa) ishlating. i current_end ga yetganda, jumps ni oshiring va current_end ni yangilang.`,
			whyItMatters: `Jump Game II greedy BFS yondashuvi bilan yetib borish masalalarida minimal qadamlarni topishni ko'rsatadi.

**Bu nima uchun muhim:**

**1. BFS daraja bo'yicha aylanish**

current_end joriy daraja oxirini, farthest keyingi daraja oxirini belgilaydi.

**2. Greedy tanlov xususiyati**

Har bir darajada BARCHA pozitsiyalarni tekshiramiz va maksimal masofani kuzatamiz.

**3. Jump Game variantlari**

I - yetib borish mumkinmi, II - minimal sakrashlar, III - ikki tomonlama sakrashlar, IV - teleportatsiya.`,
			solutionCode: `from typing import List

def jump(nums: List[int]) -> int:
    """Oxirgi indeksga yetish uchun minimal sakrashlar sonini topadi."""
    n = len(nums)
    if n <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

            if current_end >= n - 1:
                break

    return jumps`
		}
	}
};

export default task;
