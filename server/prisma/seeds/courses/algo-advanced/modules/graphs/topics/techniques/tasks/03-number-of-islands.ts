import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-number-of-islands',
	title: 'Number of Islands',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'dfs', 'bfs', 'matrix', 'flood-fill'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Count the number of islands in a 2D grid.

**Problem:**

Given an \`m x n\` 2D binary grid where \`'1'\` represents land and \`'0'\` represents water, count the number of islands.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are surrounded by water.

**Examples:**

\`\`\`
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Input: grid = [["1"]]
Output: 1

Input: grid = [["0"]]
Output: 0
\`\`\`

**Visualization:**

\`\`\`
Grid (3 islands):
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

Island 1: cells (0,0), (0,1), (1,0), (1,1)
Island 2: cell (2,2)
Island 3: cells (3,3), (3,4)

DFS Process:
- Start at (0,0), mark all connected '1's as visited
- Continue scanning, find (2,2), new island
- Continue scanning, find (3,3), new island
- Count = 3
\`\`\`

**Key Insight:**

For each unvisited land cell ('1'), perform DFS/BFS to mark all connected land cells as visited (this is one island). Count how many times you start a new DFS/BFS.

**Constraints:**
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 300
- grid[i][j] is '0' or '1'

**Time Complexity:** O(m × n)
**Space Complexity:** O(m × n) worst case for recursion stack`,
	initialCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    # TODO: Count the number of islands in the grid

    return 0`,
	solutionCode: `from typing import List
from collections import deque

def num_islands(grid: List[List[str]]) -> int:
    """
    Count the number of islands in the grid.

    Args:
        grid: 2D grid where '1' is land and '0' is water

    Returns:
        Number of islands
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(i: int, j: int) -> None:
        # Check bounds and if current cell is land
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] != '1':
            return

        # Mark as visited by changing to '0'
        grid[i][j] = '0'

        # Explore all 4 directions
        dfs(i + 1, j)  # down
        dfs(i - 1, j)  # up
        dfs(i, j + 1)  # right
        dfs(i, j - 1)  # left

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)

    return count


# BFS approach (iterative)
def num_islands_bfs(grid: List[List[str]]) -> int:
    """Count islands using BFS."""
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def bfs(start_i: int, start_j: int) -> None:
        queue = deque([(start_i, start_j)])
        grid[start_i][start_j] = '0'

        while queue:
            i, j = queue.popleft()
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == '1':
                    grid[ni][nj] = '0'
                    queue.append((ni, nj))

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                bfs(i, j)

    return count


# Union-Find approach
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = 0

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1


def num_islands_union_find(grid: List[List[str]]) -> int:
    """Count islands using Union-Find."""
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)

    # Count initial land cells
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                uf.count += 1

    # Union adjacent land cells
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                idx = i * cols + j
                # Check right and down neighbors
                if j + 1 < cols and grid[i][j + 1] == '1':
                    uf.union(idx, idx + 1)
                if i + 1 < rows and grid[i + 1][j] == '1':
                    uf.union(idx, idx + cols)

    return uf.count`,
	testCode: `import pytest
from solution import num_islands


class TestNumIslands:
    def test_single_island(self):
        """Test grid with one large island"""
        grid = [
            ["1","1","1","1","0"],
            ["1","1","0","1","0"],
            ["1","1","0","0","0"],
            ["0","0","0","0","0"]
        ]
        assert num_islands(grid) == 1

    def test_three_islands(self):
        """Test grid with three separate islands"""
        grid = [
            ["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]
        ]
        assert num_islands(grid) == 3

    def test_single_cell_island(self):
        """Test single cell with land"""
        grid = [["1"]]
        assert num_islands(grid) == 1

    def test_single_cell_water(self):
        """Test single cell with water"""
        grid = [["0"]]
        assert num_islands(grid) == 0

    def test_all_water(self):
        """Test grid with all water"""
        grid = [
            ["0","0","0"],
            ["0","0","0"],
            ["0","0","0"]
        ]
        assert num_islands(grid) == 0

    def test_all_land(self):
        """Test grid with all land (one island)"""
        grid = [
            ["1","1","1"],
            ["1","1","1"],
            ["1","1","1"]
        ]
        assert num_islands(grid) == 1

    def test_diagonal_not_connected(self):
        """Test that diagonal cells are not connected"""
        grid = [
            ["1","0","1"],
            ["0","0","0"],
            ["1","0","1"]
        ]
        assert num_islands(grid) == 4

    def test_single_row(self):
        """Test single row grid"""
        grid = [["1","0","1","0","1"]]
        assert num_islands(grid) == 3

    def test_single_column(self):
        """Test single column grid"""
        grid = [["1"],["0"],["1"],["0"],["1"]]
        assert num_islands(grid) == 3

    def test_complex_shape(self):
        """Test island with complex shape"""
        grid = [
            ["1","1","0","0","1"],
            ["1","0","0","1","1"],
            ["0","0","0","1","0"],
            ["1","1","0","0","0"]
        ]
        assert num_islands(grid) == 3`,
	hint1: `Treat the grid as a graph where each '1' cell is connected to adjacent '1' cells. Use DFS/BFS to explore all connected cells of an island.`,
	hint2: `Modify the grid in-place by changing visited '1' cells to '0'. This avoids using extra space for a visited set. Count how many times you start a new DFS.`,
	whyItMatters: `Number of Islands is a classic graph problem that introduces grid-based graph traversal. It's frequently asked in interviews and teaches flood-fill algorithms.

**Why This Matters:**

**1. Grid as Graph Pattern**

\`\`\`python
# Common pattern for grid problems
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def explore(grid, i, j):
    rows, cols = len(grid), len(grid[0])
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            # Process neighbor (ni, nj)
            pass
\`\`\`

**2. Flood Fill Algorithm**

\`\`\`python
def flood_fill(grid, i, j, old_color, new_color):
    """Fill all connected cells with same color."""
    if (i < 0 or i >= len(grid) or
        j < 0 or j >= len(grid[0]) or
        grid[i][j] != old_color):
        return

    grid[i][j] = new_color
    for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
        flood_fill(grid, i+di, j+dj, old_color, new_color)
\`\`\`

**3. Related FAANG Problems**

\`\`\`python
# Max Area of Island
def max_area_of_island(grid):
    def dfs(i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        return 1 + dfs(i+1,j) + dfs(i-1,j) + dfs(i,j+1) + dfs(i,j-1)

    return max((dfs(i,j) for i in range(len(grid))
                for j in range(len(grid[0]))), default=0)

# Surrounded Regions (capture 'O's not connected to border)
# Pacific Atlantic Water Flow
# Island Perimeter
\`\`\`

**4. Union-Find Alternative**

\`\`\`python
# Union-Find is useful when:
# - Need to handle dynamic connectivity
# - Query "are these connected?" multiple times
# - Can't modify the original grid

# Time: O(m*n * α(m*n)) ≈ O(m*n)
# Space: O(m*n)
\`\`\`

**5. Interview Tips**

\`\`\`
1. Ask about grid modification:
   - Can modify: change '1' to '0' as visited marker
   - Can't modify: use separate visited set

2. Consider edge cases:
   - Empty grid
   - All water / all land
   - Single cell

3. DFS vs BFS:
   - DFS: simpler code, may hit recursion limit
   - BFS: iterative, better for very large grids
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Количество островов',
			description: `Посчитайте количество островов в 2D сетке.

**Задача:**

Дана сетка \`m x n\`, где \`'1'\` - земля, а \`'0'\` - вода. Посчитайте количество островов.

**Остров** - это участок суши, окружённый водой и соединённый по горизонтали или вертикали с соседними участками суши.

**Примеры:**

\`\`\`
Вход: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Выход: 3

Остров 1: ячейки (0,0), (0,1), (1,0), (1,1)
Остров 2: ячейка (2,2)
Остров 3: ячейки (3,3), (3,4)
\`\`\`

**Ключевая идея:**

Для каждой непосещённой ячейки с землёй выполните DFS/BFS, чтобы пометить все связанные ячейки. Посчитайте, сколько раз вы запускали новый обход.

**Ограничения:**
- 1 <= m, n <= 300
- grid[i][j] равно '0' или '1'

**Временная сложность:** O(m × n)
**Пространственная сложность:** O(m × n)`,
			hint1: `Рассматривайте сетку как граф, где каждая ячейка '1' связана с соседними '1'. Используйте DFS/BFS для обхода всех связанных ячеек острова.`,
			hint2: `Изменяйте сетку на месте, меняя посещённые '1' на '0'. Это избавляет от дополнительного множества visited. Считайте запуски DFS.`,
			whyItMatters: `Количество островов - классическая задача на графы, которая учит обходу сеток. Часто встречается на собеседованиях.

**Почему это важно:**

**1. Паттерн сетки как графа**

Общий шаблон для задач на сетках с 4 направлениями.

**2. Алгоритм заливки (Flood Fill)**

Заполнение всех связанных ячеек одного цвета.

**3. Связанные задачи FAANG**

- Max Area of Island
- Surrounded Regions
- Pacific Atlantic Water Flow
- Island Perimeter`,
			solutionCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    Подсчитывает количество островов в сетке.

    Args:
        grid: 2D сетка где '1' - земля, '0' - вода

    Returns:
        Количество островов
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] != '1':
            return

        grid[i][j] = '0'  # Помечаем как посещённую

        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)

    return count`
		},
		uz: {
			title: 'Orollar soni',
			description: `2D to'rdagi orollar sonini hisoblang.

**Masala:**

\`m x n\` o'lchamli to'r berilgan, bu yerda \`'1'\` - quruqlik, \`'0'\` - suv. Orollar sonini hisoblang.

**Orol** - bu suv bilan o'ralgan va gorizontal yoki vertikal yo'nalishda qo'shni quruqlik bilan bog'langan quruqlik.

**Misollar:**

\`\`\`
Kirish: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Chiqish: 3

Orol 1: kataklar (0,0), (0,1), (1,0), (1,1)
Orol 2: katak (2,2)
Orol 3: kataklar (3,3), (3,4)
\`\`\`

**Asosiy tushuncha:**

Har bir tashrif buyurilmagan quruqlik katagi uchun barcha bog'langan kataklarni belgilash uchun DFS/BFS bajaring. Yangi aylanishni necha marta boshlaganingizni hisoblang.

**Cheklovlar:**
- 1 <= m, n <= 300
- grid[i][j] '0' yoki '1' ga teng

**Vaqt murakkabligi:** O(m × n)
**Xotira murakkabligi:** O(m × n)`,
			hint1: `To'rni graf sifatida ko'ring, bu yerda har bir '1' katagi qo'shni '1' kataklari bilan bog'langan. Orolning barcha bog'langan kataklarini aylanib chiqish uchun DFS/BFS ishlating.`,
			hint2: `Tashrif buyurilgan '1' kataklarni '0' ga o'zgartirib, to'rni joyida o'zgartiring. Bu qo'shimcha visited to'plamidan qochish imkonini beradi.`,
			whyItMatters: `Orollar soni - to'rli graf aylanishini o'rgatadigan klassik masala. Suhbatlarda tez-tez uchraydi.

**Bu nima uchun muhim:**

**1. To'rni graf sifatida ko'rish patterni**

4 yo'nalishli to'r masalalari uchun umumiy shablon.

**2. Flood Fill algoritmi**

Bir xil rangdagi barcha bog'langan kataklarni to'ldirish.

**3. Bog'liq FAANG masalalari**

- Max Area of Island
- Surrounded Regions
- Pacific Atlantic Water Flow
- Island Perimeter`,
			solutionCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    To'rdagi orollar sonini hisoblaydi.

    Args:
        grid: '1' quruqlik, '0' suv bo'lgan 2D to'r

    Returns:
        Orollar soni
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] != '1':
            return

        grid[i][j] = '0'  # Tashrif buyurilgan deb belgilash

        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)

    return count`
		}
	}
};

export default task;
