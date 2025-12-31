import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-dfs-traversal',
	title: 'DFS Graph Traversal',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'dfs', 'stack', 'recursion', 'traversal'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement Depth-First Search (DFS) traversal for a graph.

**Problem:**

Given a graph represented as an adjacency list and a starting node, perform DFS traversal and return the nodes in the order they were visited.

DFS explores as far as possible along each branch before backtracking.

**Examples:**

\`\`\`
Input: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}, start = 0
Output: [0, 1, 3, 4, 2] (or other valid DFS order)

Explanation:
Start at 0 → go to first neighbor 1
From 1 → go to first unvisited neighbor 3
3 has no unvisited neighbors → backtrack to 1
From 1 → go to next unvisited neighbor 4
4 has no unvisited neighbors → backtrack to 1, then 0
From 0 → go to next unvisited neighbor 2

Input: graph = {0: [1], 1: [0, 2], 2: [1]}, start = 0
Output: [0, 1, 2]

Input: graph = {0: []}, start = 0
Output: [0]
\`\`\`

**Visualization:**

\`\`\`
Graph:
    0 --- 1 --- 3
    |     |
    2     4

DFS from 0 (following first neighbor):
Step 1: Visit 0, stack = [0]
Step 2: Visit 1 (neighbor of 0), stack = [0, 1]
Step 3: Visit 3 (neighbor of 1), stack = [0, 1, 3]
Step 4: Backtrack (3 has no unvisited neighbors)
Step 5: Visit 4 (neighbor of 1), stack = [0, 1, 4]
Step 6: Backtrack to 0
Step 7: Visit 2 (neighbor of 0)

Result: [0, 1, 3, 4, 2]
\`\`\`

**Key Insight:**

DFS can be implemented using:
1. **Recursion** (implicit stack)
2. **Explicit stack** (iterative)

Both approaches explore depth-first by processing the most recently discovered vertex.

**Constraints:**
- 0 <= number of nodes <= 1000
- Graph may have cycles
- Graph may be disconnected

**Time Complexity:** O(V + E) where V = vertices, E = edges
**Space Complexity:** O(V) for visited set and recursion stack`,
	initialCode: `from typing import Dict, List

def dfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    # TODO: Perform DFS traversal and return visited nodes in order

    return []`,
	solutionCode: `from typing import Dict, List

def dfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Perform DFS traversal starting from the given node.

    Args:
        graph: Adjacency list representation of the graph
        start: Starting node for traversal

    Returns:
        List of nodes in DFS traversal order
    """
    if start not in graph:
        return []

    visited = set()
    result = []

    def dfs(node: int) -> None:
        if node in visited:
            return

        visited.add(node)
        result.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result


# Iterative DFS using explicit stack
def dfs_iterative(graph: Dict[int, List[int]], start: int) -> List[int]:
    """DFS using explicit stack (gives reverse neighbor order)."""
    if start not in graph:
        return []

    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)
            result.append(node)

            # Add neighbors in reverse order to match recursive order
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


# DFS with path tracking
def dfs_with_path(graph: Dict[int, List[int]], start: int, end: int) -> List[int]:
    """Find a path from start to end using DFS."""
    if start not in graph or end not in graph:
        return []

    visited = set()

    def dfs(node: int, path: List[int]) -> List[int]:
        if node == end:
            return path

        visited.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = dfs(neighbor, path + [neighbor])
                if result:
                    return result

        return []

    return dfs(start, [start]) if start != end else [start]


# DFS for all connected components
def dfs_all_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """Find all connected components using DFS."""
    visited = set()
    components = []

    def dfs(node: int, component: List[int]) -> None:
        visited.add(node)
        component.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components`,
	testCode: `import pytest
from solution import dfs_traversal


class TestDFSTraversal:
    def test_simple_graph(self):
        """Test simple connected graph"""
        graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
        result = dfs_traversal(graph, 0)
        assert result[0] == 0
        assert set(result) == {0, 1, 2, 3}
        assert len(result) == 4

    def test_dfs_order_property(self):
        """Test that DFS visits children before siblings"""
        graph = {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}
        result = dfs_traversal(graph, 0)
        # If we go to 1 first, we should visit 1's children before 2
        if result.index(1) < result.index(2):
            # 3 and 4 should come before 2
            assert result.index(3) < result.index(2) or result.index(4) < result.index(2)

    def test_single_node(self):
        """Test graph with single node"""
        graph = {0: []}
        assert dfs_traversal(graph, 0) == [0]

    def test_linear_graph(self):
        """Test linear graph (chain)"""
        graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        result = dfs_traversal(graph, 0)
        assert result == [0, 1, 2, 3]

    def test_start_from_middle(self):
        """Test starting from middle node"""
        graph = {0: [1], 1: [0, 2], 2: [1]}
        result = dfs_traversal(graph, 1)
        assert result[0] == 1
        assert set(result) == {0, 1, 2}

    def test_disconnected_graph(self):
        """Test that DFS only visits connected component"""
        graph = {0: [1], 1: [0], 2: [3], 3: [2]}
        result = dfs_traversal(graph, 0)
        assert set(result) == {0, 1}
        assert 2 not in result

    def test_graph_with_cycle(self):
        """Test graph with cycle"""
        graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = dfs_traversal(graph, 0)
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_binary_tree_like(self):
        """Test binary tree-like structure"""
        graph = {
            0: [1, 2],
            1: [0, 3, 4],
            2: [0, 5, 6],
            3: [1], 4: [1],
            5: [2], 6: [2]
        }
        result = dfs_traversal(graph, 0)
        assert result[0] == 0
        assert len(result) == 7

    def test_empty_start_node(self):
        """Test when start node is not in graph"""
        graph = {0: [1], 1: [0]}
        result = dfs_traversal(graph, 5)
        assert result == []

    def test_deep_graph(self):
        """Test deep linear graph"""
        graph = {i: [i+1] if i < 9 else [] for i in range(10)}
        for i in range(1, 10):
            graph[i].append(i-1)
        result = dfs_traversal(graph, 0)
        assert len(result) == 10
        assert result[0] == 0`,
	hint1: `Use recursion: mark node as visited, add to result, then recursively visit all unvisited neighbors. Base case: node already visited.`,
	hint2: `For iterative approach, use a stack instead of queue. Pop from stack, mark visited, push unvisited neighbors. Reverse neighbors for same order as recursive.`,
	whyItMatters: `DFS is fundamental for exploring graph structures. It's the basis for many algorithms including topological sort, cycle detection, and strongly connected components.

**Why This Matters:**

**1. Topological Sort**

\`\`\`python
def topological_sort(graph):
    """Sort DAG nodes so all edges point forward."""
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)  # Add after all descendants

    for node in graph:
        if node not in visited:
            dfs(node)

    return result[::-1]  # Reverse for topological order
\`\`\`

**2. Cycle Detection**

\`\`\`python
def has_cycle(graph):
    """Detect cycle in directed graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY  # Currently exploring
        for neighbor in graph.get(node, []):
            if color[neighbor] == GRAY:
                return True  # Back edge = cycle!
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True
        color[node] = BLACK  # Done exploring
        return False

    return any(dfs(node) for node in graph if color[node] == WHITE)
\`\`\`

**3. Connected Components**

\`\`\`python
def count_components(graph):
    """Count connected components in undirected graph."""
    visited = set()
    count = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for node in graph:
        if node not in visited:
            dfs(node)
            count += 1

    return count
\`\`\`

**4. Path Finding**

\`\`\`python
def find_all_paths(graph, start, end, path=[]):
    """Find all paths from start to end."""
    path = path + [start]

    if start == end:
        return [path]

    paths = []
    for neighbor in graph.get(start, []):
        if neighbor not in path:  # Avoid cycles
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)

    return paths
\`\`\`

**5. DFS vs BFS Comparison**

\`\`\`
DFS:                          BFS:
- Uses stack/recursion        - Uses queue
- Goes deep first             - Goes wide first
- Memory: O(height)           - Memory: O(width)
- Not shortest path           - Shortest path (unweighted)

Use DFS for:                  Use BFS for:
- Topological sort            - Shortest path
- Cycle detection             - Level-order traversal
- Maze solving                - Nearest neighbor
- Backtracking problems       - Social network degrees
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Обход графа в глубину (DFS)',
			description: `Реализуйте обход графа в глубину (DFS).

**Задача:**

Дан граф в виде списка смежности и начальная вершина. Выполните обход в глубину и верните вершины в порядке посещения.

DFS исследует как можно глубже вдоль каждой ветви перед возвратом.

**Примеры:**

\`\`\`
Вход: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}, start = 0
Выход: [0, 1, 3, 4, 2]

Объяснение:
Начинаем с 0 → идём к первому соседу 1
Из 1 → идём к первому непосещённому соседу 3
У 3 нет непосещённых соседей → возвращаемся к 1
Из 1 → идём к следующему непосещённому соседу 4
И так далее...
\`\`\`

**Ключевая идея:**

DFS можно реализовать с помощью рекурсии (неявный стек) или явного стека.

**Ограничения:**
- 0 <= количество вершин <= 1000
- Граф может содержать циклы

**Временная сложность:** O(V + E)
**Пространственная сложность:** O(V)`,
			hint1: `Используйте рекурсию: пометьте вершину как посещённую, добавьте в результат, затем рекурсивно посетите всех непосещённых соседей.`,
			hint2: `Для итеративного подхода используйте стек вместо очереди. Извлекайте из стека, помечайте посещённым, добавляйте непосещённых соседей.`,
			whyItMatters: `DFS фундаментален для исследования структур графов. Это основа для многих алгоритмов.

**Почему это важно:**

**1. Топологическая сортировка**

Сортировка вершин DAG так, чтобы все рёбра шли вперёд.

**2. Обнаружение циклов**

Используя три цвета (белый, серый, чёрный) можно найти обратные рёбра.

**3. Компоненты связности**

Подсчёт и нахождение всех связных компонент.

**4. Поиск путей**

Нахождение всех путей между двумя вершинами.`,
			solutionCode: `from typing import Dict, List

def dfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Выполняет обход DFS начиная с заданной вершины.

    Args:
        graph: Граф в виде списка смежности
        start: Начальная вершина

    Returns:
        Список вершин в порядке обхода DFS
    """
    if start not in graph:
        return []

    visited = set()
    result = []

    def dfs(node: int) -> None:
        if node in visited:
            return

        visited.add(node)
        result.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result`
		},
		uz: {
			title: 'Grafni chuqurlik bo\'yicha aylanib chiqish (DFS)',
			description: `Grafni chuqurlik bo\'yicha aylanib chiqish (DFS) algoritmini amalga oshiring.

**Masala:**

Qo'shnilik ro'yxati ko'rinishidagi graf va boshlang'ich tugun berilgan. DFS aylanishini bajaring va tugunlarni tashrif buyurish tartibida qaytaring.

DFS orqaga qaytishdan oldin har bir tarmoq bo'ylab iloji boricha chuqurroq o'rganadi.

**Misollar:**

\`\`\`
Kirish: graph = {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}, start = 0
Chiqish: [0, 1, 3, 4, 2]

Izoh:
0 dan boshlaymiz → birinchi qo'shni 1 ga o'tamiz
1 dan → birinchi tashrif buyurilmagan qo'shni 3 ga
3 ning tashrif buyurilmagan qo'shnilari yo'q → 1 ga qaytamiz
Va hokazo...
\`\`\`

**Asosiy tushuncha:**

DFS ni rekursiya (yashirin stek) yoki aniq stek yordamida amalga oshirish mumkin.

**Cheklovlar:**
- 0 <= tugunlar soni <= 1000
- Grafda sikllar bo'lishi mumkin

**Vaqt murakkabligi:** O(V + E)
**Xotira murakkabligi:** O(V)`,
			hint1: `Rekursiyadan foydalaning: tugunni tashrif buyurilgan deb belgilang, natijaga qo'shing, keyin barcha tashrif buyurilmagan qo'shnilarni rekursiv ravishda aylanib chiqing.`,
			hint2: `Iterativ yondashuv uchun navbat o'rniga stek ishlating. Stekdan chiqaring, tashrif buyurilgan deb belgilang, tashrif buyurilmagan qo'shnilarni qo'shing.`,
			whyItMatters: `DFS graf strukturalarini o'rganish uchun asosiy algoritmdir. Ko'pgina algoritmlarning asosi.

**Bu nima uchun muhim:**

**1. Topologik saralash**

DAG tugunlarini barcha qirralar oldinga qaragan holda saralash.

**2. Sikl aniqlash**

Uch rang (oq, kulrang, qora) yordamida orqa qirralarni topish mumkin.

**3. Bog'lanish komponentlari**

Barcha bog'langan komponentlarni hisoblash va topish.

**4. Yo'l qidirish**

Ikki tugun o'rtasidagi barcha yo'llarni topish.`,
			solutionCode: `from typing import Dict, List

def dfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Berilgan tugundan boshlab DFS aylanishini bajaradi.

    Args:
        graph: Qo'shnilik ro'yxati ko'rinishidagi graf
        start: Boshlang'ich tugun

    Returns:
        DFS aylanish tartibidagi tugunlar ro'yxati
    """
    if start not in graph:
        return []

    visited = set()
    result = []

    def dfs(node: int) -> None:
        if node in visited:
            return

        visited.add(node)
        result.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result`
		}
	}
};

export default task;
