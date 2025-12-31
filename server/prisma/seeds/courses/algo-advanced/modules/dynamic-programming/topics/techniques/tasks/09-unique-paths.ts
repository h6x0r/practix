import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'dp-unique-paths',
	title: 'Unique Paths',
	difficulty: 'medium',
	tags: ['python', 'dynamic-programming', 'grid', 'combinatorics'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Count the number of unique paths in a grid from top-left to bottom-right.

**Problem:**

A robot is located at the top-left corner of an \`m x n\` grid. The robot can only move **right** or **down** at any point in time.

The robot is trying to reach the bottom-right corner of the grid. How many unique paths are there?

**Examples:**

\`\`\`
Input: m = 3, n = 7
Output: 28

Input: m = 3, n = 2
Output: 3
Explanation: From top-left to bottom-right, there are 3 paths:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

Input: m = 1, n = 1
Output: 1
\`\`\`

**Visualization (3x3 grid):**

\`\`\`
[S] [ ] [ ]
[ ] [ ] [ ]
[ ] [ ] [E]

Paths count at each cell:
[1] [1] [1]
[1] [2] [3]
[1] [3] [6]

Answer: 6 paths
\`\`\`

**Key Insight:**

To reach cell (i, j), you must come from either:
- Cell above: (i-1, j)
- Cell to the left: (i, j-1)

\`dp[i][j] = dp[i-1][j] + dp[i][j-1]\`

**Constraints:**
- 1 <= m, n <= 100

**Time Complexity:** O(m × n)
**Space Complexity:** O(n) with optimization`,
	initialCode: `def unique_paths(m: int, n: int) -> int:
    # TODO: Count unique paths from top-left to bottom-right (only right/down moves)

    return 0`,
	solutionCode: `def unique_paths(m: int, n: int) -> int:
    """
    Count unique paths from top-left to bottom-right.

    Args:
        m: Number of rows
        n: Number of columns

    Returns:
        Number of unique paths
    """
    # Space-optimized 1D DP
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]

    return dp[n - 1]


# Alternative: 2D DP for clarity
def unique_paths_2d(m: int, n: int) -> int:
    """Full 2D DP approach."""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]


# Mathematical approach: Combination
from math import comb

def unique_paths_math(m: int, n: int) -> int:
    """
    Mathematical solution using combinations.
    Total moves: (m-1) down + (n-1) right = m+n-2 moves
    Choose (m-1) positions for down moves from (m+n-2) positions
    """
    return comb(m + n - 2, m - 1)


# Unique Paths II: With obstacles
from typing import List

def unique_paths_with_obstacles(obstacle_grid: List[List[int]]) -> int:
    """Count paths with obstacles (1 = obstacle, 0 = free)."""
    m, n = len(obstacle_grid), len(obstacle_grid[0])

    if obstacle_grid[0][0] == 1 or obstacle_grid[m - 1][n - 1] == 1:
        return 0

    dp = [0] * n
    dp[0] = 1

    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]

    return dp[n - 1]`,
	testCode: `import pytest
from solution import unique_paths


class TestUniquePaths:
    def test_3x7_grid(self):
        """Test 3x7 grid -> 28"""
        assert unique_paths(3, 7) == 28

    def test_3x2_grid(self):
        """Test 3x2 grid -> 3"""
        assert unique_paths(3, 2) == 3

    def test_1x1_grid(self):
        """Test 1x1 grid -> 1"""
        assert unique_paths(1, 1) == 1

    def test_1xn_grid(self):
        """Test single row"""
        assert unique_paths(1, 5) == 1

    def test_mx1_grid(self):
        """Test single column"""
        assert unique_paths(5, 1) == 1

    def test_2x2_grid(self):
        """Test 2x2 grid -> 2"""
        assert unique_paths(2, 2) == 2

    def test_3x3_grid(self):
        """Test 3x3 grid -> 6"""
        assert unique_paths(3, 3) == 6

    def test_4x4_grid(self):
        """Test 4x4 grid -> 20"""
        assert unique_paths(4, 4) == 20

    def test_symmetric(self):
        """Test that m x n == n x m"""
        assert unique_paths(3, 7) == unique_paths(7, 3)

    def test_larger_grid(self):
        """Test 10x10 grid"""
        assert unique_paths(10, 10) == 48620`,
	hint1: `Each cell can only be reached from the cell above or to the left. First row and first column have only 1 path each.`,
	hint2: `dp[i][j] = dp[i-1][j] + dp[i][j-1]. For space optimization, use a single row and update in-place: dp[j] += dp[j-1].`,
	whyItMatters: `Unique Paths is a foundational grid DP problem. Understanding it unlocks many 2D DP problems and has a beautiful mathematical connection.

**Why This Matters:**

**1. Grid DP Pattern**

\`\`\`python
# Basic grid DP template
for i in range(m):
    for j in range(n):
        # dp[i][j] depends on neighbors
        dp[i][j] = f(dp[i-1][j], dp[i][j-1])

# Common variations:
# - Sum of paths (this problem)
# - Minimum path sum
# - Maximum path value
\`\`\`

**2. Space Optimization**

\`\`\`python
# 2D -> 1D: Only need previous row
# O(m*n) -> O(n) space

dp = [1] * n
for i in range(1, m):
    for j in range(1, n):
        dp[j] += dp[j-1]  # dp[j] is "from above", dp[j-1] is "from left"
\`\`\`

**3. Mathematical Solution**

\`\`\`python
# Total moves: (m-1) down + (n-1) right
# Choose positions for down moves
# C(m+n-2, m-1) = (m+n-2)! / ((m-1)! * (n-1)!)

from math import comb
def unique_paths(m, n):
    return comb(m + n - 2, m - 1)
\`\`\`

**4. Related FAANG Problems**

\`\`\`python
# Unique Paths II (with obstacles)
def unique_paths_obstacles(grid):
    if grid[0][0] == 1:
        return 0
    dp = [0] * len(grid[0])
    dp[0] = 1
    for row in grid:
        for j in range(len(row)):
            if row[j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[-1]

# Minimum Path Sum
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                grid[i][j] += grid[i][j-1]
            elif j == 0:
                grid[i][j] += grid[i-1][j]
            else:
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]

# Dungeon Game (reverse DP)
# Triangle (multiple paths from top)
\`\`\`

**5. Pascal's Triangle Connection**

\`\`\`
The grid values form Pascal's triangle rotated 45°:

Grid:       Pascal's:
1 1 1 1       1
1 2 3 4      1 1
1 3 6 10    1 2 1
           1 3 3 1
          1 4 6 4 1
\`\`\``,
	order: 9,
	translations: {
		ru: {
			title: 'Уникальные пути',
			description: `Подсчитайте количество уникальных путей в сетке от верхнего левого до нижнего правого угла.

**Задача:**

Робот находится в верхнем левом углу сетки \`m x n\`. Робот может двигаться только **вправо** или **вниз** в любой момент времени.

Робот пытается достичь нижнего правого угла сетки. Сколько существует уникальных путей?

**Примеры:**

\`\`\`
Вход: m = 3, n = 7
Выход: 28

Вход: m = 3, n = 2
Выход: 3
Объяснение: Есть 3 пути:
1. Вправо -> Вниз -> Вниз
2. Вниз -> Вниз -> Вправо
3. Вниз -> Вправо -> Вниз

Вход: m = 1, n = 1
Выход: 1
\`\`\`

**Ключевая идея:**

Чтобы достичь ячейки (i, j), нужно прийти из:
- Ячейки сверху: (i-1, j)
- Ячейки слева: (i, j-1)

\`dp[i][j] = dp[i-1][j] + dp[i][j-1]\`

**Ограничения:**
- 1 <= m, n <= 100

**Временная сложность:** O(m × n)
**Пространственная сложность:** O(n) с оптимизацией`,
			hint1: `Каждая ячейка достижима только сверху или слева. Первая строка и первый столбец имеют по 1 пути.`,
			hint2: `dp[i][j] = dp[i-1][j] + dp[i][j-1]. Для оптимизации памяти используйте одну строку: dp[j] += dp[j-1].`,
			whyItMatters: `Unique Paths - базовая задача grid DP. Понимание её открывает многие 2D DP задачи.

**Почему это важно:**

**1. Паттерн Grid DP**

Базовый шаблон для сеточного DP.

**2. Оптимизация памяти**

2D -> 1D: нужна только предыдущая строка.

**3. Математическое решение**

Всего ходов: (m-1) вниз + (n-1) вправо. C(m+n-2, m-1).

**4. Связанные задачи FAANG**

- Unique Paths II (с препятствиями)
- Minimum Path Sum
- Dungeon Game`,
			solutionCode: `def unique_paths(m: int, n: int) -> int:
    """
    Подсчитывает уникальные пути от верхнего левого до нижнего правого угла.

    Args:
        m: Количество строк
        n: Количество столбцов

    Returns:
        Количество уникальных путей
    """
    # Оптимизированный по памяти 1D DP
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]

    return dp[n - 1]`
		},
		uz: {
			title: 'Noyob yo\'llar',
			description: `To'rdagi yuqori chapdan pastki o'ngga bo'lgan noyob yo'llar sonini hisoblang.

**Masala:**

Robot \`m x n\` to'rning yuqori chap burchagida joylashgan. Robot istalgan vaqtda faqat **o'ngga** yoki **pastga** harakat qilishi mumkin.

Robot to'rning pastki o'ng burchagiga yetishga harakat qilmoqda. Nechta noyob yo'l bor?

**Misollar:**

\`\`\`
Kirish: m = 3, n = 7
Chiqish: 28

Kirish: m = 3, n = 2
Chiqish: 3
Izoh: 3 ta yo'l bor:
1. O'ng -> Past -> Past
2. Past -> Past -> O'ng
3. Past -> O'ng -> Past

Kirish: m = 1, n = 1
Chiqish: 1
\`\`\`

**Asosiy tushuncha:**

(i, j) katakka yetish uchun quyidagilardan biridan kelish kerak:
- Yuqoridagi katak: (i-1, j)
- Chapdagi katak: (i, j-1)

\`dp[i][j] = dp[i-1][j] + dp[i][j-1]\`

**Cheklovlar:**
- 1 <= m, n <= 100

**Vaqt murakkabligi:** O(m × n)
**Xotira murakkabligi:** O(n) optimallashtirish bilan`,
			hint1: `Har bir katakka faqat yuqoridan yoki chapdan yetish mumkin. Birinchi qator va birinchi ustunda har birida 1 ta yo'l bor.`,
			hint2: `dp[i][j] = dp[i-1][j] + dp[i][j-1]. Xotira optimallashtirish uchun bitta qator ishlating: dp[j] += dp[j-1].`,
			whyItMatters: `Unique Paths - asosiy grid DP masalasi. Uni tushunish ko'plab 2D DP masalalarini ochadi.

**Bu nima uchun muhim:**

**1. Grid DP patterni**

To'rli DP uchun asosiy shablon.

**2. Xotira optimallashtirish**

2D -> 1D: faqat oldingi qator kerak.

**3. Matematik yechim**

Jami harakatlar: (m-1) pastga + (n-1) o'ngga. C(m+n-2, m-1).

**4. Bog'liq FAANG masalalari**

- Unique Paths II (to'siqlar bilan)
- Minimum Path Sum
- Dungeon Game`,
			solutionCode: `def unique_paths(m: int, n: int) -> int:
    """
    Yuqori chapdan pastki o'ngga noyob yo'llarni hisoblaydi.

    Args:
        m: Qatorlar soni
        n: Ustunlar soni

    Returns:
        Noyob yo'llar soni
    """
    # Xotira bo'yicha optimallashtirilgan 1D DP
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]

    return dp[n - 1]`
		}
	}
};

export default task;
