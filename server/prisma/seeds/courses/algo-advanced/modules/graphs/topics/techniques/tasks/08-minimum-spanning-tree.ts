import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-minimum-spanning-tree',
	title: 'Minimum Spanning Tree',
	difficulty: 'hard',
	tags: ['python', 'graphs', 'mst', 'kruskal', 'prim', 'union-find', 'heap'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the Minimum Spanning Tree (MST) of a weighted undirected graph.

**Problem:**

Given a connected, undirected graph with \`n\` vertices and weighted edges, find the MST - a subset of edges that:
1. Connects all vertices
2. Has minimum total edge weight
3. Contains no cycles (forms a tree)

Return the total weight of the MST.

**Examples:**

\`\`\`
Input: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,3]]
Output: 6

Explanation:
MST edges: [0,1,1], [1,2,2], [2,3,3]
Total weight: 1 + 2 + 3 = 6

    0---1---1
    |\\  |
   4| \\2|
    |  \\|
    2---3
      3

MST:
    0---1---1
        |
        2
        |
    2---3
      3

Input: n = 3, edges = [[0,1,1],[1,2,2],[0,2,3]]
Output: 3
Explanation: MST = [0,1,1] + [1,2,2] = 3 (skip [0,2,3])
\`\`\`

**Algorithms:**

\`\`\`
Kruskal's Algorithm:
1. Sort all edges by weight
2. For each edge (in order), add it if it doesn't create a cycle
3. Stop when we have n-1 edges

Prim's Algorithm:
1. Start from any vertex
2. Greedily add the minimum weight edge that connects a new vertex
3. Use a min-heap for efficiency
\`\`\`

**Constraints:**
- 1 <= n <= 1000
- 0 <= edges.length <= n * (n - 1) / 2
- edges[i] = [u, v, weight]
- 1 <= weight <= 10^6
- Graph is connected

**Time Complexity:** O(E log E) for both algorithms
**Space Complexity:** O(V + E)`,
	initialCode: `from typing import List
import heapq

def minimum_spanning_tree(n: int, edges: List[List[int]]) -> int:
    # TODO: Find the total weight of the Minimum Spanning Tree

    return 0`,
	solutionCode: `from typing import List
import heapq
from collections import defaultdict


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


def minimum_spanning_tree(n: int, edges: List[List[int]]) -> int:
    """
    Find the total weight of the Minimum Spanning Tree using Kruskal's algorithm.
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    total_weight = 0
    edges_used = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_weight


# Prim's Algorithm using min-heap
def minimum_spanning_tree_prim(n: int, edges: List[List[int]]) -> int:
    """
    Find MST using Prim's algorithm.
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((weight, v))
        graph[v].append((weight, u))

    visited = set()
    min_heap = [(0, 0)]  # (weight, vertex)
    total_weight = 0

    while min_heap and len(visited) < n:
        weight, u = heapq.heappop(min_heap)

        if u in visited:
            continue

        visited.add(u)
        total_weight += weight

        for edge_weight, v in graph[u]:
            if v not in visited:
                heapq.heappush(min_heap, (edge_weight, v))

    return total_weight


# Get MST edges (not just weight)
def get_mst_edges(n: int, edges: List[List[int]]) -> List[List[int]]:
    """Return the edges in the MST."""
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst_edges = []

    for u, v, weight in edges:
        if uf.union(u, v):
            mst_edges.append([u, v, weight])
            if len(mst_edges) == n - 1:
                break

    return mst_edges


# Min Cost to Connect All Points (LeetCode 1584)
def min_cost_connect_points(points: List[List[int]]) -> int:
    """MST where edge weight = Manhattan distance."""
    n = len(points)

    # Generate all edges with Manhattan distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append([i, j, dist])

    return minimum_spanning_tree(n, edges)`,
	testCode: `import pytest
from solution import minimum_spanning_tree


class TestMinimumSpanningTree:
    def test_basic_graph(self):
        """Test basic 4-node graph"""
        edges = [[0, 1, 1], [0, 2, 4], [1, 2, 2], [1, 3, 5], [2, 3, 3]]
        assert minimum_spanning_tree(4, edges) == 6

    def test_triangle(self):
        """Test triangle graph"""
        edges = [[0, 1, 1], [1, 2, 2], [0, 2, 3]]
        assert minimum_spanning_tree(3, edges) == 3

    def test_single_edge(self):
        """Test two vertices with one edge"""
        assert minimum_spanning_tree(2, [[0, 1, 5]]) == 5

    def test_single_vertex(self):
        """Test single vertex (no edges needed)"""
        assert minimum_spanning_tree(1, []) == 0

    def test_complete_graph(self):
        """Test complete graph - should pick minimum edges"""
        edges = [
            [0, 1, 1], [0, 2, 2], [0, 3, 3],
            [1, 2, 4], [1, 3, 5], [2, 3, 6]
        ]
        # MST: [0,1,1] + [0,2,2] + [0,3,3] = 6
        assert minimum_spanning_tree(4, edges) == 6

    def test_linear_graph(self):
        """Test linear graph (already a tree)"""
        edges = [[0, 1, 1], [1, 2, 2], [2, 3, 3]]
        assert minimum_spanning_tree(4, edges) == 6

    def test_star_graph(self):
        """Test star-shaped graph"""
        edges = [[0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4]]
        assert minimum_spanning_tree(5, edges) == 10

    def test_multiple_same_weight(self):
        """Test with multiple edges of same weight"""
        edges = [[0, 1, 1], [1, 2, 1], [2, 0, 1]]
        # Any 2 edges form MST with weight 2
        assert minimum_spanning_tree(3, edges) == 2

    def test_large_weights(self):
        """Test with large weights"""
        edges = [[0, 1, 1000000], [1, 2, 1000000]]
        assert minimum_spanning_tree(3, edges) == 2000000

    def test_skip_expensive_edge(self):
        """Test that expensive edge is skipped"""
        edges = [[0, 1, 1], [1, 2, 1], [0, 2, 100]]
        # Should use [0,1] + [1,2] = 2, not include [0,2]
        assert minimum_spanning_tree(3, edges) == 2`,
	hint1: `For Kruskal's: Sort edges by weight, use Union-Find. For each edge, add it if it connects different components (union returns true). Stop at n-1 edges.`,
	hint2: `For Prim's: Start from vertex 0, use min-heap with (weight, vertex). Pop minimum, add to MST if vertex not visited. Push all edges from new vertex to heap.`,
	whyItMatters: `Minimum Spanning Tree is fundamental in network design, clustering, and approximation algorithms. It appears in computer networks, road planning, and machine learning.

**Why This Matters:**

**1. Network Design Applications**

\`\`\`python
# Minimum cost to connect all computers in a network
# Minimum wiring to connect electrical components
# Optimal road network connecting cities

# Example: Connect N cities with minimum road cost
def min_road_network(cities, roads):
    # roads = [[city1, city2, cost], ...]
    return minimum_spanning_tree(len(cities), roads)
\`\`\`

**2. Kruskal vs Prim Comparison**

\`\`\`
Kruskal's Algorithm:
- Edge-centric: process edges in order of weight
- Best for sparse graphs (E ≈ V)
- Uses Union-Find
- Time: O(E log E)

Prim's Algorithm:
- Vertex-centric: grow tree from starting vertex
- Best for dense graphs (E ≈ V²)
- Uses Priority Queue
- Time: O(E log V) with binary heap
       O(E + V log V) with Fibonacci heap
\`\`\`

**3. MST Properties**

\`\`\`python
# Cut Property:
# For any cut of the graph, the minimum weight crossing edge
# is in some MST.

# Cycle Property:
# For any cycle, the maximum weight edge
# is NOT in any MST.

# Uniqueness:
# If all edge weights are distinct, MST is unique.
\`\`\`

**4. Related Problems**

\`\`\`python
# Min Cost to Connect All Points (LeetCode 1584)
# - MST where edge weight = Manhattan distance

# Connecting Cities With Minimum Cost (LeetCode 1135)
# - Direct MST application

# Critical and Pseudo-Critical Edges (LeetCode 1489)
# - Identify edges that must/may be in MST
\`\`\`

**5. Clustering with MST**

\`\`\`python
# K-means alternative: remove k-1 longest MST edges
# Creates k clusters with maximum inter-cluster distance

def mst_clustering(points, k):
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(points[i], points[j])
            edges.append([i, j, dist])

    mst_edges = get_mst_edges(n, edges)
    # Sort by weight descending, remove k-1 longest
    mst_edges.sort(key=lambda x: -x[2])

    # Remaining edges define k clusters
    return mst_edges[k-1:]  # Edges to keep
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Минимальное остовное дерево',
			description: `Найдите минимальное остовное дерево (MST) взвешенного неориентированного графа.

**Задача:**

Дан связный неориентированный граф с \`n\` вершинами и взвешенными рёбрами. Найдите MST - подмножество рёбер, которое:
1. Соединяет все вершины
2. Имеет минимальный суммарный вес
3. Не содержит циклов (образует дерево)

Верните суммарный вес MST.

**Примеры:**

\`\`\`
Вход: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,3]]
Выход: 6

Объяснение:
Рёбра MST: [0,1,1], [1,2,2], [2,3,3]
Суммарный вес: 1 + 2 + 3 = 6
\`\`\`

**Алгоритмы:**

\`\`\`
Алгоритм Краскала:
1. Отсортировать рёбра по весу
2. Добавлять ребро, если оно не создаёт цикл
3. Остановиться при n-1 рёбрах

Алгоритм Прима:
1. Начать с любой вершины
2. Жадно добавлять минимальное ребро к новой вершине
\`\`\`

**Ограничения:**
- 1 <= n <= 1000
- Граф связный

**Временная сложность:** O(E log E)
**Пространственная сложность:** O(V + E)`,
			hint1: `Для Краскала: отсортируйте рёбра по весу, используйте Union-Find. Добавляйте ребро если оно соединяет разные компоненты. Остановитесь при n-1 рёбрах.`,
			hint2: `Для Прима: начните с вершины 0, используйте min-heap с (вес, вершина). Извлекайте минимум, добавляйте в MST если вершина не посещена.`,
			whyItMatters: `MST фундаментально в проектировании сетей, кластеризации и приближённых алгоритмах.

**Почему это важно:**

**1. Применения в проектировании сетей**

Минимальная стоимость соединения всех компьютеров, городов и т.д.

**2. Сравнение Краскала и Прима**

Краскал лучше для разреженных графов, Прим - для плотных.

**3. Связанные задачи**

Min Cost to Connect All Points, Connecting Cities With Minimum Cost.`,
			solutionCode: `from typing import List
import heapq
from collections import defaultdict


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


def minimum_spanning_tree(n: int, edges: List[List[int]]) -> int:
    """Находит суммарный вес MST алгоритмом Краскала."""
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    total_weight = 0
    edges_used = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_weight`
		},
		uz: {
			title: 'Minimal qoplovchi daraxt',
			description: `Og'irlikli yo'naltirilmagan grafning Minimal Qoplovchi Daraxtini (MST) toping.

**Masala:**

\`n\` ta tugun va og'irlikli qirralar bilan bog'langan yo'naltirilmagan graf berilgan. MST ni toping - qirralar to'plami:
1. Barcha tugunlarni bog'laydi
2. Minimal umumiy og'irlikka ega
3. Sikl yo'q (daraxt hosil qiladi)

MST ning umumiy og'irligini qaytaring.

**Misollar:**

\`\`\`
Kirish: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,3]]
Chiqish: 6

Izoh:
MST qirralari: [0,1,1], [1,2,2], [2,3,3]
Umumiy og'irlik: 1 + 2 + 3 = 6
\`\`\`

**Algoritmlar:**

\`\`\`
Kruskal algoritmi:
1. Qirralarni og'irlik bo'yicha saralash
2. Sikl yaratmasa qirrani qo'shish
3. n-1 qirrada to'xtatish

Prim algoritmi:
1. Istalgan tugundan boshlash
2. Ochko'zlik bilan yangi tugunga minimal qirra qo'shish
\`\`\`

**Cheklovlar:**
- 1 <= n <= 1000
- Graf bog'langan

**Vaqt murakkabligi:** O(E log E)
**Xotira murakkabligi:** O(V + E)`,
			hint1: `Kruskal uchun: qirralarni og'irlik bo'yicha saralang, Union-Find ishlating. Turli komponentlarni bog'lasa qirra qo'shing. n-1 qirrada to'xtating.`,
			hint2: `Prim uchun: 0-tugundan boshlang, (og'irlik, tugun) bilan min-heap ishlating. Minimalni chiqaring, tashrif buyurilmagan bo'lsa MST ga qo'shing.`,
			whyItMatters: `MST tarmoq loyihalash, klasterlash va taxminiy algoritmlarda asosiy hisoblanadi.

**Bu nima uchun muhim:**

**1. Tarmoq loyihalashda qo'llanilishi**

Barcha kompyuterlar, shaharlar va hokazolarni ulashning minimal narxi.

**2. Kruskal va Prim taqqoslash**

Kruskal siyrak graflar uchun yaxshi, Prim zich graflar uchun.

**3. Bog'liq masalalar**

Min Cost to Connect All Points, Connecting Cities With Minimum Cost.`,
			solutionCode: `from typing import List
import heapq
from collections import defaultdict


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


def minimum_spanning_tree(n: int, edges: List[List[int]]) -> int:
    """Kruskal algoritmi bilan MST ning umumiy og'irligini topadi."""
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    total_weight = 0
    edges_used = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_weight`
		}
	}
};

export default task;
