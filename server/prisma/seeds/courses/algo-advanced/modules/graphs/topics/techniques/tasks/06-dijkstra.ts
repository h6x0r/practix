import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-dijkstra',
	title: "Dijkstra's Shortest Path",
	difficulty: 'hard',
	tags: ['python', 'graphs', 'dijkstra', 'heap', 'priority-queue', 'shortest-path'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement Dijkstra's algorithm to find the shortest path in a weighted graph.

**Problem:**

Given a weighted directed graph with non-negative edge weights and a source vertex, find the shortest distance from the source to all other vertices.

Return a list where result[i] is the shortest distance from source to vertex i, or -1 if unreachable.

**Examples:**

\`\`\`
Input: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]], source = 0
Output: [0, 1, 3, 4]

Explanation:
- Distance to 0: 0 (source)
- Distance to 1: 1 (direct edge 0→1)
- Distance to 2: 3 (path 0→1→2 = 1+2)
- Distance to 3: 4 (path 0→1→2→3 = 1+2+1)

Input: n = 3, edges = [[0,1,2],[1,2,3]], source = 0
Output: [0, 2, 5]

Input: n = 3, edges = [[0,1,1]], source = 0
Output: [0, 1, -1]
Explanation: Vertex 2 is unreachable from source 0.
\`\`\`

**Visualization:**

\`\`\`
Graph:
    0 --1--> 1
    |        |\\
   4|       2| \\5
    v        v  \\v
    2 <------   3
       \\----1--->/

Dijkstra from 0:
Step 1: dist = [0, ∞, ∞, ∞], visit 0
        Update: dist[1] = 1, dist[2] = 4
Step 2: dist = [0, 1, 4, ∞], visit 1 (smallest unvisited)
        Update: dist[2] = min(4, 1+2) = 3, dist[3] = 6
Step 3: dist = [0, 1, 3, 6], visit 2
        Update: dist[3] = min(6, 3+1) = 4
Step 4: dist = [0, 1, 3, 4], visit 3

Final: [0, 1, 3, 4]
\`\`\`

**Key Insight:**

Dijkstra's algorithm is a **greedy** algorithm that:
1. Always processes the unvisited vertex with the smallest distance
2. Uses a **min-heap (priority queue)** for efficient extraction
3. Relaxes edges: if dist[u] + weight < dist[v], update dist[v]

**Important:** Only works with **non-negative** edge weights!

**Constraints:**
- 1 <= n <= 1000
- 0 <= edges.length <= 10000
- edges[i] = [from, to, weight]
- 0 <= weight <= 10^6
- 0 <= source < n

**Time Complexity:** O((V + E) log V) with binary heap
**Space Complexity:** O(V + E)`,
	initialCode: `from typing import List
import heapq
from collections import defaultdict

def dijkstra(n: int, edges: List[List[int]], source: int) -> List[int]:
    # TODO: Find shortest distances from source to all vertices using Dijkstra's algorithm

    return [-1] * n`,
	solutionCode: `from typing import List
import heapq
from collections import defaultdict

def dijkstra(n: int, edges: List[List[int]], source: int) -> List[int]:
    """
    Find shortest distances from source to all vertices.

    Args:
        n: Number of vertices (0 to n-1)
        edges: List of [from, to, weight] edges
        source: Starting vertex

    Returns:
        List of shortest distances, -1 if unreachable
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    # Initialize distances
    dist = [float('inf')] * n
    dist[source] = 0

    # Min-heap: (distance, vertex)
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        # Skip if we've found a better path already
        if d > dist[u]:
            continue

        # Relax all neighbors
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    # Convert infinity to -1 for unreachable vertices
    return [d if d != float('inf') else -1 for d in dist]


# Dijkstra with path reconstruction
def dijkstra_with_path(n: int, edges: List[List[int]], source: int, target: int) -> tuple:
    """Return (distance, path) to target."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    parent = [-1] * n

    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        if u == target:
            # Reconstruct path
            path = []
            current = target
            while current != -1:
                path.append(current)
                current = parent[current]
            return dist[target], path[::-1]

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))

    return -1, []


# Network Delay Time (LeetCode 743)
def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """Time for signal to reach all nodes from k."""
    dist = dijkstra(n, [[u-1, v-1, w] for u, v, w in times], k - 1)
    max_time = max(dist)
    return max_time if max_time != -1 else -1`,
	testCode: `import pytest
from solution import dijkstra


class TestDijkstra:
    def test_basic_graph(self):
        """Test basic weighted graph"""
        edges = [[0, 1, 1], [0, 2, 4], [1, 2, 2], [1, 3, 5], [2, 3, 1]]
        assert dijkstra(4, edges, 0) == [0, 1, 3, 4]

    def test_linear_graph(self):
        """Test linear chain graph"""
        edges = [[0, 1, 2], [1, 2, 3]]
        assert dijkstra(3, edges, 0) == [0, 2, 5]

    def test_unreachable_vertex(self):
        """Test with unreachable vertex"""
        edges = [[0, 1, 1]]
        assert dijkstra(3, edges, 0) == [0, 1, -1]

    def test_single_vertex(self):
        """Test single vertex graph"""
        assert dijkstra(1, [], 0) == [0]

    def test_multiple_paths(self):
        """Test choosing shorter path"""
        edges = [[0, 1, 10], [0, 2, 1], [2, 1, 1]]
        # Direct: 0→1 = 10, Via 2: 0→2→1 = 2
        assert dijkstra(3, edges, 0) == [0, 2, 1]

    def test_disconnected_graph(self):
        """Test disconnected components"""
        edges = [[0, 1, 1], [2, 3, 1]]
        assert dijkstra(4, edges, 0) == [0, 1, -1, -1]

    def test_different_source(self):
        """Test starting from different source"""
        edges = [[0, 1, 1], [1, 2, 2], [0, 2, 5]]
        assert dijkstra(3, edges, 1) == [-1, 0, 2]

    def test_zero_weight_edge(self):
        """Test with zero weight edge"""
        edges = [[0, 1, 0], [1, 2, 1]]
        assert dijkstra(3, edges, 0) == [0, 0, 1]

    def test_parallel_edges(self):
        """Test with multiple edges between same nodes"""
        edges = [[0, 1, 5], [0, 1, 2], [0, 1, 3]]
        # Should find minimum (weight 2)
        result = dijkstra(2, edges, 0)
        assert result[1] == 2

    def test_larger_graph(self):
        """Test larger graph"""
        edges = [
            [0, 1, 4], [0, 2, 1], [2, 1, 2],
            [1, 3, 1], [2, 3, 5], [3, 4, 3]
        ]
        assert dijkstra(5, edges, 0) == [0, 3, 1, 4, 7]`,
	hint1: `Build an adjacency list with weights. Use a min-heap to always process the vertex with the smallest known distance. Initialize all distances to infinity except source = 0.`,
	hint2: `For each vertex popped from heap, check if we've already found a better path (skip if so). Otherwise, try to relax all outgoing edges: if dist[u] + weight < dist[v], update and push to heap.`,
	whyItMatters: `Dijkstra's algorithm is the foundation for shortest path problems in weighted graphs. It's used in GPS navigation, network routing, and countless optimization problems.

**Why This Matters:**

**1. Real-World Applications**

\`\`\`python
# GPS Navigation
# - Find fastest route between two locations
# - Edge weights = travel time or distance

# Network Routing (OSPF protocol)
# - Route packets through network
# - Edge weights = latency or bandwidth

# Social Networks
# - Find degrees of separation
# - Edge weights = relationship strength
\`\`\`

**2. Algorithm Variations**

\`\`\`python
# Bellman-Ford: Handles negative weights
# O(V * E), detects negative cycles
def bellman_ford(n, edges, source):
    dist = [float('inf')] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Detect negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # Negative cycle exists

    return dist

# A* Search: Heuristic-guided Dijkstra
# Uses f(n) = g(n) + h(n)
# g(n) = actual distance from start
# h(n) = estimated distance to goal
\`\`\`

**3. Common Interview Problems**

\`\`\`python
# Network Delay Time (LeetCode 743)
def networkDelayTime(times, n, k):
    dist = dijkstra(n, times, k-1)
    max_dist = max(dist)
    return max_dist if max_dist != float('inf') else -1

# Cheapest Flights Within K Stops (LeetCode 787)
# Modified Dijkstra with stop count

# Path with Minimum Effort (LeetCode 1631)
# Edge weight = max difference in path
\`\`\`

**4. Optimization with Fibonacci Heap**

\`\`\`
Binary Heap:    O((V + E) log V)
Fibonacci Heap: O(V log V + E)  - better for dense graphs

In practice, binary heap is usually faster due to:
- Lower constant factors
- Better cache locality
\`\`\`

**5. Bidirectional Dijkstra**

\`\`\`python
# For single source-target queries
# Run Dijkstra from both source and target
# Stop when frontiers meet
# Can be 2x faster in practice
\`\`\``,
	order: 6,
	translations: {
		ru: {
			title: 'Алгоритм Дейкстры',
			description: `Реализуйте алгоритм Дейкстры для поиска кратчайшего пути во взвешенном графе.

**Задача:**

Дан взвешенный ориентированный граф с неотрицательными весами рёбер и начальная вершина. Найдите кратчайшие расстояния от начальной вершины до всех остальных.

Верните список, где result[i] - кратчайшее расстояние до вершины i, или -1, если недостижима.

**Примеры:**

\`\`\`
Вход: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]], source = 0
Выход: [0, 1, 3, 4]

Объяснение:
- Расстояние до 0: 0 (источник)
- Расстояние до 1: 1 (прямое ребро 0→1)
- Расстояние до 2: 3 (путь 0→1→2 = 1+2)
- Расстояние до 3: 4 (путь 0→1→2→3 = 1+2+1)
\`\`\`

**Ключевая идея:**

Алгоритм Дейкстры - **жадный** алгоритм, который:
1. Всегда обрабатывает непосещённую вершину с минимальным расстоянием
2. Использует **min-heap** для эффективного извлечения
3. Релаксирует рёбра: если dist[u] + weight < dist[v], обновляем dist[v]

**Важно:** Работает только с **неотрицательными** весами!

**Ограничения:**
- 1 <= n <= 1000
- 0 <= weight <= 10^6

**Временная сложность:** O((V + E) log V)
**Пространственная сложность:** O(V + E)`,
			hint1: `Постройте список смежности с весами. Используйте min-heap для обработки вершины с минимальным расстоянием. Инициализируйте все расстояния бесконечностью, кроме source = 0.`,
			hint2: `Для каждой извлечённой вершины проверьте, нашли ли мы уже лучший путь (пропустите если да). Иначе релаксируйте все исходящие рёбра.`,
			whyItMatters: `Алгоритм Дейкстры - основа для задач кратчайшего пути во взвешенных графах. Используется в GPS-навигации, маршрутизации сетей и оптимизации.

**Почему это важно:**

**1. Реальные применения**

- GPS-навигация
- Сетевая маршрутизация
- Социальные сети

**2. Вариации алгоритма**

- Беллмана-Форда: отрицательные веса
- A*: эвристический поиск

**3. Частые задачи на собеседованиях**

Network Delay Time, Cheapest Flights, Path with Minimum Effort`,
			solutionCode: `from typing import List
import heapq
from collections import defaultdict

def dijkstra(n: int, edges: List[List[int]], source: int) -> List[int]:
    """
    Находит кратчайшие расстояния от источника до всех вершин.

    Args:
        n: Количество вершин (от 0 до n-1)
        edges: Список рёбер [от, до, вес]
        source: Начальная вершина

    Returns:
        Список кратчайших расстояний, -1 если недостижима
    """
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0

    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return [d if d != float('inf') else -1 for d in dist]`
		},
		uz: {
			title: 'Dijkstra algoritmi',
			description: `Og'irlikli grafda eng qisqa yo'lni topish uchun Dijkstra algoritmini amalga oshiring.

**Masala:**

Manfiy bo'lmagan qirra og'irliklari va boshlang'ich tugun bilan og'irlikli yo'naltirilgan graf berilgan. Boshlang'ich tugundan boshqa barcha tugunlarga eng qisqa masofalarni toping.

result[i] - i tugunga eng qisqa masofa yoki yetib bo'lmasa -1 bo'lgan ro'yxat qaytaring.

**Misollar:**

\`\`\`
Kirish: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]], source = 0
Chiqish: [0, 1, 3, 4]

Izoh:
- 0 ga masofa: 0 (manba)
- 1 ga masofa: 1 (to'g'ridan-to'g'ri qirra 0→1)
- 2 ga masofa: 3 (yo'l 0→1→2 = 1+2)
- 3 ga masofa: 4 (yo'l 0→1→2→3 = 1+2+1)
\`\`\`

**Asosiy tushuncha:**

Dijkstra algoritmi **ochko'z** algoritm:
1. Har doim eng kichik masofali tashrif buyurilmagan tugunni qayta ishlaydi
2. Samarali chiqarish uchun **min-heap** ishlatadi
3. Qirralarni bo'shatadi: agar dist[u] + weight < dist[v] bo'lsa, dist[v] ni yangilaydi

**Muhim:** Faqat **manfiy bo'lmagan** og'irliklarda ishlaydi!

**Cheklovlar:**
- 1 <= n <= 1000
- 0 <= weight <= 10^6

**Vaqt murakkabligi:** O((V + E) log V)
**Xotira murakkabligi:** O(V + E)`,
			hint1: `Og'irliklari bilan qo'shnilik ro'yxatini tuzing. Eng kichik ma'lum masofali tugunni qayta ishlash uchun min-heap ishlating. Barcha masofalarni cheksizlikka, source = 0 ga tenglashtiring.`,
			hint2: `Heapdan chiqarilgan har bir tugun uchun yaxshiroq yo'l topilganligini tekshiring (agar shunday bo'lsa o'tkazib yuboring). Aks holda barcha chiquvchi qirralarni bo'shating.`,
			whyItMatters: `Dijkstra algoritmi og'irlikli graflarda eng qisqa yo'l masalalari uchun asosdir. GPS navigatsiya, tarmoq marshrutlash va optimallashtirishda ishlatiladi.

**Bu nima uchun muhim:**

**1. Haqiqiy qo'llanishlar**

- GPS navigatsiya
- Tarmoq marshrutlash
- Ijtimoiy tarmoqlar

**2. Algoritm variantlari**

- Bellman-Ford: manfiy og'irliklar
- A*: evristik qidiruv

**3. Suhbatdagi tez-tez uchraydigan masalalar**

Network Delay Time, Cheapest Flights, Path with Minimum Effort`,
			solutionCode: `from typing import List
import heapq
from collections import defaultdict

def dijkstra(n: int, edges: List[List[int]], source: int) -> List[int]:
    """
    Manbadan barcha tugunlarga eng qisqa masofalarni topadi.

    Args:
        n: Tugunlar soni (0 dan n-1 gacha)
        edges: [dan, ga, og'irlik] qirralar ro'yxati
        source: Boshlang'ich tugun

    Returns:
        Eng qisqa masofalar ro'yxati, yetib bo'lmasa -1
    """
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0

    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return [d if d != float('inf') else -1 for d in dist]`
		}
	}
};

export default task;
