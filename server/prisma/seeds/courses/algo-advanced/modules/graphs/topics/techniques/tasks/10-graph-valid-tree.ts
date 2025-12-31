import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-valid-tree',
	title: 'Graph Valid Tree',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'dfs', 'bfs', 'union-find', 'tree'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if an undirected graph is a valid tree.

**Problem:**

Given \`n\` nodes labeled from \`0\` to \`n-1\` and a list of undirected edges, determine if these edges make up a valid tree.

A valid tree must satisfy:
1. **Connected**: All nodes are reachable from any other node
2. **Acyclic**: There are no cycles
3. **n-1 edges**: A tree with n nodes has exactly n-1 edges

**Examples:**

\`\`\`
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Explanation:
    0
   /|\\
  1 2 3
  |
  4
This is a valid tree.

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false

Explanation:
    0
    |
    1---4
   / \\
  2---3
Cycle exists: 1-2-3-1

Input: n = 4, edges = [[0,1],[2,3]]
Output: false
Explanation: Graph is not connected (two components).
\`\`\`

**Key Insight:**

A graph is a valid tree if and only if:
1. It has exactly **n-1 edges**, AND
2. It is **connected** (one component)

OR equivalently:
1. It has exactly **n-1 edges**, AND
2. It has **no cycles**

**Constraints:**
- 1 <= n <= 2000
- 0 <= edges.length <= 5000
- edges[i].length == 2
- 0 <= edges[i][0], edges[i][1] < n
- No duplicate edges

**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)`,
	initialCode: `from typing import List

def valid_tree(n: int, edges: List[List[int]]) -> bool:
    # TODO: Determine if the graph is a valid tree (connected + no cycles)

    return False`,
	solutionCode: `from typing import List
from collections import defaultdict, deque

def valid_tree(n: int, edges: List[List[int]]) -> bool:
    """
    Determine if the graph is a valid tree.
    """
    # Tree must have exactly n-1 edges
    if len(edges) != n - 1:
        return False

    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # DFS to check connectivity
    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)

    # All nodes must be reachable
    return len(visited) == n


# BFS approach
def valid_tree_bfs(n: int, edges: List[List[int]]) -> bool:
    """Check using BFS for connectivity."""
    if len(edges) != n - 1:
        return False

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set([0])
    queue = deque([0])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n


# Union-Find approach
def valid_tree_union_find(n: int, edges: List[List[int]]) -> bool:
    """Check using Union-Find for cycles."""
    if len(edges) != n - 1:
        return False

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> bool:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return False  # Cycle detected
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1
        return True

    for u, v in edges:
        if not union(u, v):
            return False

    return True


# DFS with cycle detection (without edge count check)
def valid_tree_full_check(n: int, edges: List[List[int]]) -> bool:
    """Full check with cycle detection during DFS."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node: int, parent: int) -> bool:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if not dfs(neighbor, node):
                    return False
            elif neighbor != parent:
                return False  # Cycle detected
        return True

    # Check no cycle and all connected
    if not dfs(0, -1):
        return False

    return len(visited) == n`,
	testCode: `import pytest
from solution import valid_tree


class TestValidTree:
    def test_simple_tree(self):
        """Test simple valid tree"""
        assert valid_tree(5, [[0, 1], [0, 2], [0, 3], [1, 4]]) == True

    def test_graph_with_cycle(self):
        """Test graph with cycle (not a tree)"""
        assert valid_tree(5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]) == False

    def test_disconnected_graph(self):
        """Test disconnected graph (not a tree)"""
        assert valid_tree(4, [[0, 1], [2, 3]]) == False

    def test_single_node(self):
        """Test single node (trivial tree)"""
        assert valid_tree(1, []) == True

    def test_two_nodes(self):
        """Test two connected nodes"""
        assert valid_tree(2, [[0, 1]]) == True

    def test_linear_tree(self):
        """Test linear tree (chain)"""
        assert valid_tree(4, [[0, 1], [1, 2], [2, 3]]) == True

    def test_star_tree(self):
        """Test star-shaped tree"""
        assert valid_tree(5, [[0, 1], [0, 2], [0, 3], [0, 4]]) == True

    def test_too_many_edges(self):
        """Test with more than n-1 edges"""
        assert valid_tree(3, [[0, 1], [1, 2], [0, 2]]) == False

    def test_too_few_edges(self):
        """Test with fewer than n-1 edges"""
        assert valid_tree(4, [[0, 1], [1, 2]]) == False

    def test_empty_edges_multiple_nodes(self):
        """Test multiple nodes with no edges"""
        assert valid_tree(3, []) == False

    def test_self_loop(self):
        """Test self-loop"""
        # Self loop would make it not a valid tree
        # but per constraints no self-loops exist
        pass`,
	hint1: `A tree with n nodes must have exactly n-1 edges. If edge count is wrong, return False immediately. Then check connectivity.`,
	hint2: `After checking n-1 edges, do DFS/BFS from node 0. If you can visit all n nodes, the graph is connected and therefore a valid tree (since n-1 edges + connected = tree).`,
	whyItMatters: `Understanding tree properties in graphs is fundamental. Trees are special graphs used in file systems, DOM, databases, and countless data structures.

**Why This Matters:**

**1. Tree Properties**

\`\`\`python
# A graph with n nodes is a tree if and only if:
# - Has n-1 edges AND is connected
# - Has n-1 edges AND has no cycles
# - Is connected AND has no cycles
# - Has unique path between every pair of nodes

# Any 2 of these 3 imply the third:
# - Connected
# - n-1 edges
# - Acyclic
\`\`\`

**2. Applications of Graph Trees**

\`\`\`python
# File systems: directories form a tree
# HTML/XML: DOM is a tree
# Databases: B-trees, B+ trees
# Version control: commit history
# Organizational charts
# Family trees (sort of)

# Tree traversals:
# - Pre-order: process node, then children
# - Post-order: process children, then node
# - Level-order: BFS
\`\`\`

**3. Related Problems**

\`\`\`python
# Redundant Connection (LeetCode 684)
# Find the edge that makes it not a tree
def find_redundant_connection(edges):
    parent = list(range(len(edges) + 1))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu == pv:
            return [u, v]  # This edge creates cycle
        parent[pu] = pv

    return []

# Number of Connected Components
# Minimum Height Trees (LeetCode 310)
\`\`\`

**4. Tree vs Forest**

\`\`\`python
# Forest: collection of trees (acyclic, possibly disconnected)
# Tree: connected forest (exactly one component)

def is_forest(n, edges):
    # Forest: no cycles (can be disconnected)
    if len(edges) > n - 1:
        return False  # Too many edges = cycle
    # Check no cycle using Union-Find
    return not has_cycle(n, edges)

def count_trees_in_forest(n, edges):
    # Number of connected components
    # = n - number_of_edges (if forest)
    if is_forest(n, edges):
        return n - len(edges)
    return -1  # Not a forest
\`\`\`

**5. Interview Tips**

\`\`\`
1. Know the equivalence:
   Tree = Connected + Acyclic = Connected + (n-1 edges) = Acyclic + (n-1 edges)

2. Fastest approach:
   - Check n-1 edges first (O(1))
   - Then check connectivity OR no cycles

3. Edge cases:
   - n=1, edges=[] → True (single node is a tree)
   - n=2, edges=[] → False (disconnected)
\`\`\``,
	order: 10,
	translations: {
		ru: {
			title: 'Проверка на дерево',
			description: `Определите, является ли неориентированный граф валидным деревом.

**Задача:**

Даны \`n\` вершин с метками от 0 до n-1 и список неориентированных рёбер. Определите, образуют ли эти рёбра валидное дерево.

Валидное дерево должно удовлетворять:
1. **Связность**: Все вершины достижимы
2. **Ациклничность**: Нет циклов
3. **n-1 рёбер**: Дерево с n вершинами имеет ровно n-1 рёбер

**Примеры:**

\`\`\`
Вход: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Выход: true
Объяснение: Это валидное дерево.

Вход: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Выход: false
Объяснение: Есть цикл: 1-2-3-1

Вход: n = 4, edges = [[0,1],[2,3]]
Выход: false
Объяснение: Граф не связный.
\`\`\`

**Ключевая идея:**

Граф является деревом тогда и только тогда, когда:
1. Имеет ровно **n-1 рёбер**, И
2. **Связный** (одна компонента)

**Ограничения:**
- 1 <= n <= 2000
- Без дубликатов рёбер

**Временная сложность:** O(V + E)
**Пространственная сложность:** O(V + E)`,
			hint1: `Дерево с n вершинами должно иметь ровно n-1 рёбер. Если количество рёбер неверное, сразу верните False. Затем проверьте связность.`,
			hint2: `После проверки n-1 рёбер, выполните DFS/BFS от вершины 0. Если можете посетить все n вершин, граф связный и является деревом.`,
			whyItMatters: `Понимание свойств деревьев в графах фундаментально. Деревья используются в файловых системах, DOM, базах данных и структурах данных.

**Почему это важно:**

**1. Свойства дерева**

Граф с n вершинами является деревом если: имеет n-1 рёбер И связный; или n-1 рёбер И ацикличный.

**2. Применения деревьев**

Файловые системы, DOM, B-деревья, история коммитов.

**3. Связанные задачи**

Redundant Connection, Number of Connected Components, Minimum Height Trees.`,
			solutionCode: `from typing import List
from collections import defaultdict

def valid_tree(n: int, edges: List[List[int]]) -> bool:
    """Определяет, является ли граф валидным деревом."""
    if len(edges) != n - 1:
        return False

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)

    return len(visited) == n`
		},
		uz: {
			title: 'Daraxt tekshiruvi',
			description: `Yo'naltirilmagan graf to'g'ri daraxt ekanligini aniqlang.

**Masala:**

0 dan n-1 gacha belgilangan \`n\` ta tugun va yo'naltirilmagan qirralar ro'yxati berilgan. Bu qirralar to'g'ri daraxt hosil qiladimi, aniqlang.

To'g'ri daraxt quyidagilarni qondirishi kerak:
1. **Bog'langan**: Barcha tugunlar yetishimli
2. **Asiklik**: Sikllar yo'q
3. **n-1 qirra**: n tugunli daraxtda aynan n-1 ta qirra bor

**Misollar:**

\`\`\`
Kirish: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Chiqish: true
Izoh: Bu to'g'ri daraxt.

Kirish: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Chiqish: false
Izoh: Sikl bor: 1-2-3-1

Kirish: n = 4, edges = [[0,1],[2,3]]
Chiqish: false
Izoh: Graf bog'lanmagan.
\`\`\`

**Asosiy tushuncha:**

Graf daraxt bo'lishi uchun:
1. Aynan **n-1 qirra** bo'lishi kerak, VA
2. **Bog'langan** bo'lishi kerak (bitta komponent)

**Cheklovlar:**
- 1 <= n <= 2000
- Takroriy qirralar yo'q

**Vaqt murakkabligi:** O(V + E)
**Xotira murakkabligi:** O(V + E)`,
			hint1: `n tugunli daraxtda aynan n-1 qirra bo'lishi kerak. Agar qirralar soni noto'g'ri bo'lsa, darhol False qaytaring. Keyin bog'lanishni tekshiring.`,
			hint2: `n-1 qirrani tekshirgandan so'ng, 0-tugundan DFS/BFS bajaring. Agar barcha n tugunga tashrif buyura olsangiz, graf bog'langan va daraxtdir.`,
			whyItMatters: `Graflardagi daraxt xususiyatlarini tushunish asosiy hisoblanadi. Daraxtlar fayl tizimlarida, DOM da, ma'lumotlar bazalarida ishlatiladi.

**Bu nima uchun muhim:**

**1. Daraxt xususiyatlari**

n tugunli graf daraxt bo'lsa: n-1 qirra VA bog'langan; yoki n-1 qirra VA asiklik.

**2. Daraxtlarning qo'llanilishi**

Fayl tizimlari, DOM, B-daraxtlar, commit tarixi.

**3. Bog'liq masalalar**

Redundant Connection, Number of Connected Components, Minimum Height Trees.`,
			solutionCode: `from typing import List
from collections import defaultdict

def valid_tree(n: int, edges: List[List[int]]) -> bool:
    """Graf to'g'ri daraxt ekanligini aniqlaydi."""
    if len(edges) != n - 1:
        return False

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)

    return len(visited) == n`
		}
	}
};

export default task;
