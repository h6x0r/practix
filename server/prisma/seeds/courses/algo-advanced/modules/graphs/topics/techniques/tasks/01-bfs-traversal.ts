import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-bfs-traversal',
	title: 'BFS Graph Traversal',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'bfs', 'queue', 'traversal'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement Breadth-First Search (BFS) traversal for a graph.

**Problem:**

Given a graph represented as an adjacency list and a starting node, perform BFS traversal and return the nodes in the order they were visited.

BFS explores all neighbors at the current depth before moving to nodes at the next depth level.

**Examples:**

\`\`\`
Input: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}, start = 0
Output: [0, 1, 2, 3, 4, 5]

Explanation:
Level 0: Visit node 0
Level 1: Visit neighbors of 0 → nodes 1, 2
Level 2: Visit neighbors of 1,2 → nodes 3, 4, 5

Input: graph = {0: [1], 1: [0, 2], 2: [1]}, start = 1
Output: [1, 0, 2]

Input: graph = {0: []}, start = 0
Output: [0]
\`\`\`

**Visualization:**

\`\`\`
Graph:
    0 --- 1 --- 3
    |     |
    2     4
    |
    5

BFS from 0:
Step 1: Visit 0, queue = [1, 2]
Step 2: Visit 1, queue = [2, 3, 4]
Step 3: Visit 2, queue = [3, 4, 5]
Step 4: Visit 3, queue = [4, 5]
Step 5: Visit 4, queue = [5]
Step 6: Visit 5, queue = []

Result: [0, 1, 2, 3, 4, 5]
\`\`\`

**Key Insight:**

BFS uses a **queue** (FIFO) to process nodes level by level:
1. Start with source node in queue
2. Dequeue a node, mark as visited
3. Enqueue all unvisited neighbors
4. Repeat until queue is empty

**Constraints:**
- 0 <= number of nodes <= 1000
- Graph may have cycles
- Graph may be disconnected (only traverse from start)

**Time Complexity:** O(V + E) where V = vertices, E = edges
**Space Complexity:** O(V) for visited set and queue`,
	initialCode: `from typing import Dict, List
from collections import deque

def bfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    # TODO: Perform BFS traversal and return visited nodes in order

    return []`,
	solutionCode: `from typing import Dict, List
from collections import deque

def bfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Perform BFS traversal starting from the given node.

    Args:
        graph: Adjacency list representation of the graph
        start: Starting node for traversal

    Returns:
        List of nodes in BFS traversal order
    """
    if start not in graph:
        return []

    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result.append(node)

            # Add all unvisited neighbors to queue
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

    return result


# Alternative: BFS with level tracking
def bfs_with_levels(graph: Dict[int, List[int]], start: int) -> List[List[int]]:
    """BFS that returns nodes grouped by level."""
    if start not in graph:
        return []

    visited = set([start])
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(current_level)

    return levels


# BFS for shortest path (unweighted graph)
def bfs_shortest_path(graph: Dict[int, List[int]], start: int, end: int) -> List[int]:
    """Find shortest path using BFS in unweighted graph."""
    if start not in graph or end not in graph:
        return []

    if start == end:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])  # (node, path)

    while queue:
        node, path = queue.popleft()

        for neighbor in graph.get(node, []):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []  # No path found`,
	testCode: `import pytest
from solution import bfs_traversal


class TestBFSTraversal:
    def test_simple_graph(self):
        """Test simple connected graph"""
        graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
        result = bfs_traversal(graph, 0)
        assert result[0] == 0
        assert set(result) == {0, 1, 2, 3}
        assert len(result) == 4

    def test_larger_graph(self):
        """Test larger graph with multiple levels"""
        graph = {
            0: [1, 2],
            1: [0, 3, 4],
            2: [0, 5],
            3: [1],
            4: [1],
            5: [2]
        }
        result = bfs_traversal(graph, 0)
        assert result[0] == 0
        # Level 1 nodes should come before level 2
        idx_1 = result.index(1)
        idx_2 = result.index(2)
        idx_3 = result.index(3)
        idx_5 = result.index(5)
        assert idx_1 < idx_3  # 1 is parent of 3
        assert idx_2 < idx_5  # 2 is parent of 5

    def test_single_node(self):
        """Test graph with single node"""
        graph = {0: []}
        assert bfs_traversal(graph, 0) == [0]

    def test_linear_graph(self):
        """Test linear graph (chain)"""
        graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        result = bfs_traversal(graph, 0)
        assert result == [0, 1, 2, 3]

    def test_start_from_middle(self):
        """Test starting from middle node"""
        graph = {0: [1], 1: [0, 2], 2: [1]}
        result = bfs_traversal(graph, 1)
        assert result[0] == 1
        assert set(result) == {0, 1, 2}

    def test_disconnected_graph(self):
        """Test that BFS only visits connected component"""
        graph = {0: [1], 1: [0], 2: [3], 3: [2]}
        result = bfs_traversal(graph, 0)
        assert set(result) == {0, 1}
        assert 2 not in result
        assert 3 not in result

    def test_graph_with_cycle(self):
        """Test graph with cycle"""
        graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = bfs_traversal(graph, 0)
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_star_graph(self):
        """Test star-shaped graph"""
        graph = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
        result = bfs_traversal(graph, 0)
        assert result[0] == 0
        assert set(result[1:]) == {1, 2, 3, 4}

    def test_empty_start_node(self):
        """Test when start node is not in graph"""
        graph = {0: [1], 1: [0]}
        result = bfs_traversal(graph, 5)
        assert result == []

    def test_complete_graph(self):
        """Test complete graph (every node connected to every other)"""
        graph = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2]
        }
        result = bfs_traversal(graph, 0)
        assert result[0] == 0
        assert len(result) == 4
        assert set(result) == {0, 1, 2, 3}`,
	hint1: `Use a queue (collections.deque) to process nodes in FIFO order. Add the start node first, then repeatedly dequeue and enqueue neighbors.`,
	hint2: `Use a visited set to avoid processing the same node twice. Check if a node is visited before adding it to the queue (not after dequeuing).`,
	whyItMatters: `BFS is one of the fundamental graph algorithms. It finds the shortest path in unweighted graphs and is the basis for many other algorithms.

**Why This Matters:**

**1. Shortest Path in Unweighted Graphs**

\`\`\`python
# BFS guarantees shortest path when all edges have equal weight
def shortest_path(graph, start, end):
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path  # Guaranteed shortest!

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []
\`\`\`

**2. Level-Order Processing**

\`\`\`python
# Process nodes by distance from source
def bfs_levels(graph, start):
    levels = []
    queue = deque([start])
    visited = set([start])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(level)
    return levels
\`\`\`

**3. Common BFS Applications**

\`\`\`python
# Social network: Find all friends within N degrees
# Web crawler: Explore links level by level
# GPS: Find nearest location of a type
# Puzzle solving: Find minimum moves

# Flood fill (connected components)
def flood_fill(grid, sr, sc, new_color):
    old_color = grid[sr][sc]
    if old_color == new_color:
        return grid

    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        if grid[r][c] == old_color:
            grid[r][c] = new_color
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                    queue.append((nr, nc))
    return grid
\`\`\`

**4. BFS vs DFS**

\`\`\`python
# BFS: Queue (FIFO), Level-by-level, Shortest path
# DFS: Stack/Recursion (LIFO), Deep exploration, Path finding

# When to use BFS:
# - Shortest path in unweighted graph
# - Level-order traversal needed
# - Finding nearest target

# When to use DFS:
# - Finding any path (not necessarily shortest)
# - Topological sorting
# - Detecting cycles
# - Exploring all possibilities
\`\`\`

**5. Time and Space Complexity**

\`\`\`
Time: O(V + E)
- Visit each vertex once: O(V)
- Examine each edge once: O(E)

Space: O(V)
- Queue can hold up to V nodes
- Visited set stores V nodes
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Обход графа в ширину (BFS)',
			description: `Реализуйте обход графа в ширину (BFS).

**Задача:**

Дан граф в виде списка смежности и начальная вершина. Выполните обход в ширину и верните вершины в порядке посещения.

BFS исследует всех соседей на текущей глубине перед переходом к вершинам на следующем уровне.

**Примеры:**

\`\`\`
Вход: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}, start = 0
Выход: [0, 1, 2, 3, 4, 5]

Объяснение:
Уровень 0: Посещаем вершину 0
Уровень 1: Посещаем соседей 0 → вершины 1, 2
Уровень 2: Посещаем соседей 1,2 → вершины 3, 4, 5
\`\`\`

**Ключевая идея:**

BFS использует **очередь** (FIFO) для обработки вершин уровень за уровнем.

**Ограничения:**
- 0 <= количество вершин <= 1000
- Граф может содержать циклы

**Временная сложность:** O(V + E)
**Пространственная сложность:** O(V)`,
			hint1: `Используйте очередь (collections.deque) для обработки вершин в порядке FIFO. Добавьте начальную вершину, затем извлекайте и добавляйте соседей.`,
			hint2: `Используйте множество visited для избежания повторной обработки вершин. Проверяйте посещение перед добавлением в очередь.`,
			whyItMatters: `BFS - один из фундаментальных алгоритмов на графах. Он находит кратчайший путь в невзвешенных графах.

**Почему это важно:**

**1. Кратчайший путь в невзвешенных графах**

BFS гарантирует кратчайший путь когда все рёбра равны.

**2. Обработка по уровням**

Обрабатывает вершины по расстоянию от источника.

**3. Применения BFS**

- Социальные сети: друзья в N степени
- Веб-краулеры: обход ссылок
- GPS: поиск ближайших мест
- Решение головоломок: минимум ходов`,
			solutionCode: `from typing import Dict, List
from collections import deque

def bfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Выполняет обход BFS начиная с заданной вершины.

    Args:
        graph: Граф в виде списка смежности
        start: Начальная вершина

    Returns:
        Список вершин в порядке обхода BFS
    """
    if start not in graph:
        return []

    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

    return result`
		},
		uz: {
			title: 'Grafni kenglik bo\'yicha aylanib chiqish (BFS)',
			description: `Grafni kenglik bo\'yicha aylanib chiqish (BFS) algoritmini amalga oshiring.

**Masala:**

Qo'shnilik ro'yxati ko'rinishidagi graf va boshlang'ich tugun berilgan. BFS aylanishini bajaring va tugunlarni tashrif buyurish tartibida qaytaring.

BFS keyingi darajadagi tugunlarga o'tishdan oldin joriy darajadagi barcha qo'shnilarni tekshiradi.

**Misollar:**

\`\`\`
Kirish: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}, start = 0
Chiqish: [0, 1, 2, 3, 4, 5]

Izoh:
Daraja 0: 0-tugunga tashrif
Daraja 1: 0 ning qo'shnilari → 1, 2 tugunlar
Daraja 2: 1,2 ning qo'shnilari → 3, 4, 5 tugunlar
\`\`\`

**Asosiy tushuncha:**

BFS tugunlarni daraja bo'yicha qayta ishlash uchun **navbat** (FIFO) ishlatadi.

**Cheklovlar:**
- 0 <= tugunlar soni <= 1000
- Grafda sikllar bo'lishi mumkin

**Vaqt murakkabligi:** O(V + E)
**Xotira murakkabligi:** O(V)`,
			hint1: `Tugunlarni FIFO tartibida qayta ishlash uchun navbat (collections.deque) ishlating. Boshlang'ich tugunni qo'shing, keyin qo'shnilarni chiqarib qo'shing.`,
			hint2: `Tugunlarni qayta ishlamaslik uchun visited to'plamini ishlating. Navbatga qo'shishdan oldin tashrif buyurilganligini tekshiring.`,
			whyItMatters: `BFS graf algoritmlarining asosiylaridan biri. U vaznlanmagan graflarda eng qisqa yo'lni topadi.

**Bu nima uchun muhim:**

**1. Vaznlanmagan graflarda eng qisqa yo'l**

Barcha qirralarda bir xil vazn bo'lganda BFS eng qisqa yo'lni kafolatlaydi.

**2. Darajalar bo'yicha qayta ishlash**

Tugunlarni manbadan masofaga qarab qayta ishlaydi.

**3. BFS qo'llanilishi**

- Ijtimoiy tarmoqlar: N darajadagi do'stlar
- Veb-kraulerlar: havolalarni aylanish
- GPS: yaqin joylarni topish
- Boshqotirmalarni yechish: minimal qadamlar`,
			solutionCode: `from typing import Dict, List
from collections import deque

def bfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Berilgan tugundan boshlab BFS aylanishini bajaradi.

    Args:
        graph: Qo'shnilik ro'yxati ko'rinishidagi graf
        start: Boshlang'ich tugun

    Returns:
        BFS aylanish tartibidagi tugunlar ro'yxati
    """
    if start not in graph:
        return []

    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

    return result`
		}
	}
};

export default task;
