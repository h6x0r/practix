import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-jump-game',
	title: 'Jump Game',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'array', 'dynamic-programming'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if you can reach the last index from the first.

**Problem:**

You are given an integer array \`nums\`. You are initially positioned at the first index, and each element represents your maximum jump length at that position.

Return \`True\` if you can reach the last index, or \`False\` otherwise.

**Examples:**

\`\`\`
Input: nums = [2, 3, 1, 1, 4]
Output: True

Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Input: nums = [3, 2, 1, 0, 4]
Output: False

Explanation: You will always arrive at index 3 no matter what.
Its maximum jump length is 0, which makes it impossible to reach the last index.

Input: nums = [0]
Output: True

Explanation: Already at the last index.
\`\`\`

**Visualization:**

\`\`\`
nums = [2, 3, 1, 1, 4]

Index:  0   1   2   3   4
Value:  2   3   1   1   4
        |   |   |   |   |
        +-->+   |   |   |  (from 0, can jump to 1 or 2)
            +---+---+-->|  (from 1, can jump to 2, 3, or 4)

Can reach index 4: True
\`\`\`

**Key Insight:**

Track the farthest position reachable. At each position, if it's reachable (i <= farthest), update farthest = max(farthest, i + nums[i]).

**Constraints:**
- 1 <= nums.length <= 10^4
- 0 <= nums[i] <= 10^5

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def can_jump(nums: List[int]) -> bool:
    # TODO: Determine if you can reach the last index

    return False`,
	solutionCode: `from typing import List

def can_jump(nums: List[int]) -> bool:
    """
    Determine if you can reach the last index.
    """
    n = len(nums)
    farthest = 0

    for i in range(n):
        if i > farthest:
            return False  # Can't reach position i

        farthest = max(farthest, i + nums[i])

        if farthest >= n - 1:
            return True

    return True


# Alternative: Work backwards
def can_jump_backward(nums: List[int]) -> bool:
    """Track last reachable position working backwards."""
    last_reachable = len(nums) - 1

    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= last_reachable:
            last_reachable = i

    return last_reachable == 0


# DP solution (less efficient but demonstrates concept)
def can_jump_dp(nums: List[int]) -> bool:
    """Dynamic programming solution."""
    n = len(nums)
    # dp[i] = True if we can reach position i
    dp = [False] * n
    dp[0] = True

    for i in range(n):
        if not dp[i]:
            continue
        # Mark all reachable positions
        for j in range(i + 1, min(i + nums[i] + 1, n)):
            dp[j] = True

    return dp[n - 1]


# BFS solution (less efficient)
from collections import deque

def can_jump_bfs(nums: List[int]) -> bool:
    """BFS solution treating array as a graph."""
    n = len(nums)
    if n <= 1:
        return True

    visited = set([0])
    queue = deque([0])

    while queue:
        pos = queue.popleft()
        for jump in range(1, nums[pos] + 1):
            next_pos = pos + jump
            if next_pos >= n - 1:
                return True
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append(next_pos)

    return False`,
	testCode: `import pytest
from solution import can_jump


class TestJumpGame:
    def test_reachable(self):
        """Test when last index is reachable"""
        assert can_jump([2, 3, 1, 1, 4]) == True

    def test_unreachable(self):
        """Test when last index is unreachable"""
        assert can_jump([3, 2, 1, 0, 4]) == False

    def test_single_element(self):
        """Test single element array"""
        assert can_jump([0]) == True

    def test_two_elements_reachable(self):
        """Test two elements reachable"""
        assert can_jump([1, 0]) == True

    def test_two_elements_unreachable(self):
        """Test two elements unreachable"""
        assert can_jump([0, 1]) == False

    def test_all_ones(self):
        """Test array of all ones"""
        assert can_jump([1, 1, 1, 1, 1]) == True

    def test_large_first_jump(self):
        """Test large jump from first position"""
        assert can_jump([5, 0, 0, 0, 0, 0]) == True

    def test_zero_in_middle(self):
        """Test zero in middle that can be skipped"""
        assert can_jump([2, 0, 1, 1, 1]) == True

    def test_multiple_paths(self):
        """Test multiple possible paths"""
        assert can_jump([2, 2, 2, 2, 2]) == True

    def test_barely_reachable(self):
        """Test barely reachable scenario"""
        assert can_jump([1, 1, 1, 1]) == True

    def test_stuck_at_zero(self):
        """Test stuck at zero"""
        assert can_jump([1, 0, 0, 0]) == False`,
	hint1: `Track the farthest position you can reach. At each index i, if i > farthest, you can't reach that position. Otherwise, update farthest = max(farthest, i + nums[i]).`,
	hint2: `You can return True early if farthest >= n - 1 at any point. The key insight is that if you can reach position i, you can potentially reach anywhere from i to i + nums[i].`,
	whyItMatters: `Jump Game demonstrates the power of greedy algorithms for reachability problems. It shows how to reduce a seemingly complex problem to tracking a single variable.

**Why This Matters:**

**1. Greedy vs DP Comparison**

\`\`\`python
# Greedy: O(n) time, O(1) space
farthest = 0
for i in range(n):
    if i > farthest:
        return False
    farthest = max(farthest, i + nums[i])

# DP: O(n²) time, O(n) space
dp = [False] * n
dp[0] = True
for i in range(n):
    if dp[i]:
        for j in range(i + 1, min(i + nums[i] + 1, n)):
            dp[j] = True
\`\`\`

**2. Forward vs Backward Greedy**

\`\`\`python
# Forward: Track farthest reachable
farthest = 0
for i in range(n):
    farthest = max(farthest, i + nums[i])

# Backward: Track last position needed to reach end
last_reachable = n - 1
for i in range(n - 2, -1, -1):
    if i + nums[i] >= last_reachable:
        last_reachable = i
return last_reachable == 0
\`\`\`

**3. Jump Game II (Minimum Jumps)**

\`\`\`python
def min_jumps(nums):
    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest

    return jumps
\`\`\`

**4. Applications**

\`\`\`python
# Video game level design (can player reach goal?)
# Network packet routing
# Resource allocation with limited reach
# Stepping stones problems
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Игра в прыжки',
			description: `Определите, можете ли вы достичь последнего индекса из первого.

**Задача:**

Дан массив целых чисел \`nums\`. Вы начинаете с первого индекса, каждый элемент представляет максимальную длину прыжка с этой позиции.

Верните \`True\`, если можете достичь последнего индекса.

**Примеры:**

\`\`\`
Вход: nums = [2, 3, 1, 1, 4]
Выход: True

Объяснение: Прыжок на 1 шаг от индекса 0 к 1, затем 3 шага к последнему индексу.

Вход: nums = [3, 2, 1, 0, 4]
Выход: False

Объяснение: Вы всегда придёте к индексу 3, откуда нельзя прыгнуть дальше.
\`\`\`

**Ключевая идея:**

Отслеживайте самую дальнюю достижимую позицию. На каждой позиции обновляйте farthest = max(farthest, i + nums[i]).

**Ограничения:**
- 1 <= nums.length <= 10^4

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Отслеживайте самую дальнюю достижимую позицию. Если i > farthest, позиция недостижима. Иначе обновите farthest = max(farthest, i + nums[i]).`,
			hint2: `Можно вернуть True раньше, если farthest >= n - 1. Если можете достичь позиции i, то можете достичь любой позиции от i до i + nums[i].`,
			whyItMatters: `Jump Game демонстрирует мощь жадных алгоритмов для задач достижимости. Показывает, как свести сложную задачу к отслеживанию одной переменной.

**Почему это важно:**

**1. Сравнение Greedy vs DP**

Greedy: O(n) времени, O(1) памяти. DP: O(n²) времени, O(n) памяти.

**2. Прямой vs обратный жадный подход**

Вперёд: отслеживаем farthest. Назад: отслеживаем last_reachable.

**3. Jump Game II**

Минимальное количество прыжков - немного другая жадная стратегия.`,
			solutionCode: `from typing import List

def can_jump(nums: List[int]) -> bool:
    """Определяет, можно ли достичь последнего индекса."""
    n = len(nums)
    farthest = 0

    for i in range(n):
        if i > farthest:
            return False

        farthest = max(farthest, i + nums[i])

        if farthest >= n - 1:
            return True

    return True`
		},
		uz: {
			title: "Sakrash o'yini",
			description: `Birinchi indeksdan oxirgi indeksga yetib bora olasizmi aniqlang.

**Masala:**

\`nums\` butun sonlar massivi berilgan. Siz birinchi indeksda turibsiz, har bir element o'sha pozitsiyadan maksimal sakrash uzunligini bildiradi.

Oxirgi indeksga yetib bora olsangiz \`True\` qaytaring.

**Misollar:**

\`\`\`
Kirish: nums = [2, 3, 1, 1, 4]
Chiqish: True

Izoh: 0-indeksdan 1-ga 1 qadam sakrash, keyin oxirgi indeksga 3 qadam.

Kirish: nums = [3, 2, 1, 0, 4]
Chiqish: False

Izoh: Har doim 3-indeksga kelasiz, u yerdan sakrab bo'lmaydi.
\`\`\`

**Asosiy tushuncha:**

Eng uzoq yetib boriladigan pozitsiyani kuzating. Har bir pozitsiyada farthest = max(farthest, i + nums[i]) yangilang.

**Cheklovlar:**
- 1 <= nums.length <= 10^4

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Eng uzoq yetib boriladigan pozitsiyani kuzating. Agar i > farthest bo'lsa, pozitsiya yetib bo'lmaydi. Aks holda farthest = max(farthest, i + nums[i]) yangilang.`,
			hint2: `Agar farthest >= n - 1 bo'lsa, erta True qaytarish mumkin. Agar i pozitsiyaga yetib borsangiz, i dan i + nums[i] gacha istalgan joyga yetib borishingiz mumkin.`,
			whyItMatters: `Jump Game yetib borish masalalari uchun greedy algoritmlar kuchini ko'rsatadi. Murakkab masalani bitta o'zgaruvchini kuzatishga qanday keltirishni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Greedy vs DP taqqoslash**

Greedy: O(n) vaqt, O(1) xotira. DP: O(n²) vaqt, O(n) xotira.

**2. Oldinga vs orqaga greedy**

Oldinga: farthest kuzatish. Orqaga: last_reachable kuzatish.

**3. Jump Game II**

Minimal sakrashlar soni - biroz boshqacha greedy strategiya.`,
			solutionCode: `from typing import List

def can_jump(nums: List[int]) -> bool:
    """Oxirgi indeksga yetib borish mumkinligini aniqlaydi."""
    n = len(nums)
    farthest = 0

    for i in range(n):
        if i > farthest:
            return False

        farthest = max(farthest, i + nums[i])

        if farthest >= n - 1:
            return True

    return True`
		}
	}
};

export default task;
