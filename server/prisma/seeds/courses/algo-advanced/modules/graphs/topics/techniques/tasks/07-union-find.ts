import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-union-find',
	title: 'Union-Find (Disjoint Set)',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'union-find', 'disjoint-set', 'cycle-detection'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement Union-Find data structure and use it to detect cycles in an undirected graph.

**Problem:**

Given an undirected graph with \`n\` vertices and a list of edges, determine if the graph contains a cycle using Union-Find (Disjoint Set Union).

**Examples:**

\`\`\`
Input: n = 4, edges = [[0,1],[1,2],[2,3]]
Output: false
Explanation: No cycle exists.
    0 -- 1 -- 2 -- 3

Input: n = 4, edges = [[0,1],[1,2],[2,3],[3,0]]
Output: true
Explanation: Cycle exists: 0 → 1 → 2 → 3 → 0

Input: n = 3, edges = [[0,1],[1,2],[0,2]]
Output: true
Explanation: Cycle: 0 → 1 → 2 → 0
\`\`\`

**Union-Find Operations:**

\`\`\`
Union-Find supports two main operations:
1. find(x): Find the root/representative of x's set
2. union(x, y): Merge the sets containing x and y

Optimizations:
- Path Compression: During find(), make each node point directly to root
- Union by Rank: Always attach smaller tree under larger tree
\`\`\`

**Cycle Detection Logic:**

\`\`\`
For each edge (u, v):
    If find(u) == find(v):
        # u and v are already connected
        # Adding this edge creates a cycle!
        return True
    Else:
        union(u, v)
return False
\`\`\`

**Constraints:**
- 1 <= n <= 1000
- 0 <= edges.length <= n * (n - 1) / 2
- edges[i].length == 2
- 0 <= edges[i][0], edges[i][1] < n
- No duplicate edges

**Time Complexity:** O(E × α(N)) ≈ O(E) where α is inverse Ackermann
**Space Complexity:** O(N)`,
	initialCode: `from typing import List

class UnionFind:
    def __init__(self, n: int):
        # TODO: Initialize n disjoint sets (0 to n-1)
        pass

    def find(self, x: int) -> int:
        # TODO: Find root with path compression
        pass

    def union(self, x: int, y: int) -> bool:
        # TODO: Merge sets, return True if merged, False if same set
        pass

def has_cycle(n: int, edges: List[List[int]]) -> bool:
    # TODO: Detect if the undirected graph has a cycle using Union-Find

    return False`,
	solutionCode: `from typing import List

class UnionFind:
    """
    Disjoint Set Union data structure with path compression
    and union by rank optimizations.
    """

    def __init__(self, n: int):
        """Initialize n disjoint sets (0 to n-1)."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union by rank. Returns True if merged, False if already same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank: attach smaller tree under larger
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True


def has_cycle(n: int, edges: List[List[int]]) -> bool:
    """
    Detect if the undirected graph has a cycle.

    Args:
        n: Number of vertices
        edges: List of undirected edges [u, v]

    Returns:
        True if cycle exists, False otherwise
    """
    uf = UnionFind(n)

    for u, v in edges:
        # If already connected, adding edge creates cycle
        if uf.find(u) == uf.find(v):
            return True
        uf.union(u, v)

    return False


# Count connected components
def count_components(n: int, edges: List[List[int]]) -> int:
    """Count number of connected components."""
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    # Count unique roots
    return len(set(uf.find(i) for i in range(n)))


# Graph Valid Tree (LeetCode 261)
def valid_tree(n: int, edges: List[List[int]]) -> bool:
    """
    Check if edges form a valid tree.
    Tree: connected graph with no cycles, n-1 edges.
    """
    # Tree must have exactly n-1 edges
    if len(edges) != n - 1:
        return False

    uf = UnionFind(n)

    for u, v in edges:
        # If union returns False, cycle exists
        if not uf.union(u, v):
            return False

    return True


# Redundant Connection (LeetCode 684)
def find_redundant_connection(edges: List[List[int]]) -> List[int]:
    """Find edge that creates cycle."""
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]

    return []`,
	testCode: `import pytest
from solution import UnionFind, has_cycle


class TestUnionFind:
    def test_initialization(self):
        """Test initial state"""
        uf = UnionFind(5)
        # Each element should be its own parent
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self):
        """Test basic union and find operations"""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)

        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_transitive_union(self):
        """Test that union is transitive"""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)

        # 0, 1, 2 should all be in same set
        assert uf.find(0) == uf.find(1) == uf.find(2)

    def test_union_returns_correct_value(self):
        """Test union return value"""
        uf = UnionFind(3)
        assert uf.union(0, 1) == True  # New union
        assert uf.union(0, 1) == False  # Already connected
        assert uf.union(1, 0) == False  # Same, different order


class TestHasCycle:
    def test_no_cycle_linear(self):
        """Test linear graph (no cycle)"""
        assert has_cycle(4, [[0, 1], [1, 2], [2, 3]]) == False

    def test_cycle_square(self):
        """Test square graph (has cycle)"""
        assert has_cycle(4, [[0, 1], [1, 2], [2, 3], [3, 0]]) == True

    def test_cycle_triangle(self):
        """Test triangle (has cycle)"""
        assert has_cycle(3, [[0, 1], [1, 2], [0, 2]]) == True

    def test_single_vertex(self):
        """Test single vertex (no edges)"""
        assert has_cycle(1, []) == False

    def test_two_vertices_one_edge(self):
        """Test simple edge (no cycle)"""
        assert has_cycle(2, [[0, 1]]) == False

    def test_disconnected_no_cycle(self):
        """Test disconnected graph without cycle"""
        assert has_cycle(4, [[0, 1], [2, 3]]) == False

    def test_tree_structure(self):
        """Test tree structure (no cycle)"""
        # Tree with root 0
        assert has_cycle(5, [[0, 1], [0, 2], [1, 3], [1, 4]]) == False

    def test_star_with_cycle(self):
        """Test star graph with added cycle"""
        edges = [[0, 1], [0, 2], [0, 3], [1, 2]]  # 0-1-2-0 is cycle
        assert has_cycle(4, edges) == True

    def test_empty_edges(self):
        """Test graph with no edges"""
        assert has_cycle(5, []) == False`,
	hint1: `Initialize parent[i] = i (each node is its own parent). For find(x), recursively find the root and apply path compression by setting parent[x] = root.`,
	hint2: `For cycle detection: before adding edge (u, v), check if find(u) == find(v). If true, they're already connected, so adding this edge creates a cycle.`,
	whyItMatters: `Union-Find is a powerful data structure for managing dynamic connectivity. It's used in Kruskal's MST algorithm, network connectivity, and image processing.

**Why This Matters:**

**1. Near-Constant Time Operations**

\`\`\`python
# With path compression + union by rank:
# - find(): O(α(n)) ≈ O(1) amortized
# - union(): O(α(n)) ≈ O(1) amortized
#
# α(n) = inverse Ackermann function
# For all practical n, α(n) ≤ 4
\`\`\`

**2. Kruskal's MST Algorithm**

\`\`\`python
def kruskal_mst(n, edges):
    """Find minimum spanning tree using Union-Find."""
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if uf.union(u, v):  # If not creating cycle
            mst.append((u, v, weight))
            total_weight += weight
            if len(mst) == n - 1:
                break

    return total_weight, mst
\`\`\`

**3. Dynamic Connectivity**

\`\`\`python
# Check if two nodes are connected
def is_connected(uf, x, y):
    return uf.find(x) == uf.find(y)

# Number of connected components
def num_components(uf, n):
    return len(set(uf.find(i) for i in range(n)))
\`\`\`

**4. Percolation Simulation**

\`\`\`python
# Monte Carlo simulation for percolation threshold
# Used in physics, material science

class Percolation:
    def __init__(self, n):
        self.n = n
        # n*n grid + 2 virtual nodes (top and bottom)
        self.uf = UnionFind(n * n + 2)
        self.top = n * n
        self.bottom = n * n + 1

    def open(self, row, col):
        # Connect to neighbors and virtual nodes
        pass

    def percolates(self):
        return self.uf.find(self.top) == self.uf.find(self.bottom)
\`\`\`

**5. Common Interview Problems**

\`\`\`python
# Graph Valid Tree: n-1 edges + no cycle
# Redundant Connection: find edge creating cycle
# Accounts Merge: merge accounts with same email
# Number of Provinces: count connected components
# Most Stones Removed: connected component counting
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Union-Find (Система непересекающихся множеств)',
			description: `Реализуйте структуру данных Union-Find и используйте её для обнаружения циклов в неориентированном графе.

**Задача:**

Дан неориентированный граф с \`n\` вершинами и список рёбер. Определите, содержит ли граф цикл, используя Union-Find.

**Примеры:**

\`\`\`
Вход: n = 4, edges = [[0,1],[1,2],[2,3]]
Выход: false
Объяснение: Цикла нет.

Вход: n = 4, edges = [[0,1],[1,2],[2,3],[3,0]]
Выход: true
Объяснение: Есть цикл: 0 → 1 → 2 → 3 → 0
\`\`\`

**Операции Union-Find:**

\`\`\`
1. find(x): Найти корень/представителя множества x
2. union(x, y): Объединить множества, содержащие x и y

Оптимизации:
- Сжатие путей: при find() направить узлы напрямую к корню
- Объединение по рангу: присоединять меньшее дерево к большему
\`\`\`

**Логика обнаружения цикла:**

Для каждого ребра (u, v): если find(u) == find(v), значит u и v уже связаны, и добавление ребра создаст цикл.

**Ограничения:**
- 1 <= n <= 1000

**Временная сложность:** O(E × α(N)) ≈ O(E)
**Пространственная сложность:** O(N)`,
			hint1: `Инициализируйте parent[i] = i. Для find(x) рекурсивно найдите корень и примените сжатие путей, установив parent[x] = root.`,
			hint2: `Для обнаружения цикла: перед добавлением ребра (u, v) проверьте find(u) == find(v). Если да - они уже связаны, ребро создаст цикл.`,
			whyItMatters: `Union-Find - мощная структура данных для управления динамической связностью. Используется в алгоритме Краскала, сетевой связности и обработке изображений.

**Почему это важно:**

**1. Почти константное время операций**

С сжатием путей + объединением по рангу: O(α(n)) ≈ O(1).

**2. Алгоритм Краскала для MST**

Сортируем рёбра по весу, добавляем если не создаёт цикл.

**3. Динамическая связность**

Проверка связи между узлами, подсчёт компонент связности.

**4. Частые задачи на собеседованиях**

Graph Valid Tree, Redundant Connection, Number of Provinces.`,
			solutionCode: `from typing import List

class UnionFind:
    """Система непересекающихся множеств со сжатием путей и объединением по рангу."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Найти корень со сжатием путей."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Объединение по рангу. True если объединено, False если уже в одном множестве."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True


def has_cycle(n: int, edges: List[List[int]]) -> bool:
    """Определяет, есть ли цикл в неориентированном графе."""
    uf = UnionFind(n)

    for u, v in edges:
        if uf.find(u) == uf.find(v):
            return True
        uf.union(u, v)

    return False`
		},
		uz: {
			title: 'Union-Find (Ajratilgan to\'plamlar)',
			description: `Union-Find ma'lumotlar strukturasini amalga oshiring va uni yo'naltirilmagan grafda siklni aniqlash uchun ishlating.

**Masala:**

\`n\` ta tugun va qirralar ro'yxati bilan yo'naltirilmagan graf berilgan. Union-Find yordamida grafda sikl borligini aniqlang.

**Misollar:**

\`\`\`
Kirish: n = 4, edges = [[0,1],[1,2],[2,3]]
Chiqish: false
Izoh: Sikl yo'q.

Kirish: n = 4, edges = [[0,1],[1,2],[2,3],[3,0]]
Chiqish: true
Izoh: Sikl bor: 0 → 1 → 2 → 3 → 0
\`\`\`

**Union-Find operatsiyalari:**

\`\`\`
1. find(x): x to'plamining ildizi/vakilini topish
2. union(x, y): x va y ni o'z ichiga olgan to'plamlarni birlashtirish

Optimallashtirishlar:
- Yo'lni siqish: find() da tugunlarni to'g'ridan-to'g'ri ildizga yo'naltirish
- Rang bo'yicha birlashtirish: kichik daraxtni katta ostiga biriktirish
\`\`\`

**Sikl aniqlash mantiqı:**

Har bir qirra (u, v) uchun: agar find(u) == find(v) bo'lsa, u va v allaqachon bog'langan, qirra qo'shish sikl yaratadi.

**Cheklovlar:**
- 1 <= n <= 1000

**Vaqt murakkabligi:** O(E × α(N)) ≈ O(E)
**Xotira murakkabligi:** O(N)`,
			hint1: `parent[i] = i ni ishga tushiring. find(x) uchun rekursiv ravishda ildizni toping va parent[x] = root qilib yo'lni siqing.`,
			hint2: `Sikl aniqlash uchun: qirra (u, v) qo'shishdan oldin find(u) == find(v) ni tekshiring. Ha bo'lsa - ular allaqachon bog'langan, qirra sikl yaratadi.`,
			whyItMatters: `Union-Find dinamik bog'lanishni boshqarish uchun kuchli ma'lumotlar strukturasi. Kruskal algoritmida, tarmoq bog'lanishida va tasvir qayta ishlashda ishlatiladi.

**Bu nima uchun muhim:**

**1. Deyarli o'zgarmas vaqt operatsiyalari**

Yo'lni siqish + rang bo'yicha birlashtirish bilan: O(α(n)) ≈ O(1).

**2. MST uchun Kruskal algoritmi**

Qirralarni og'irlik bo'yicha saralash, sikl yaratmasa qo'shish.

**3. Dinamik bog'lanish**

Tugunlar o'rtasidagi bog'lanishni tekshirish, komponentlar sonini hisoblash.

**4. Suhbatdagi tez-tez uchraydigan masalalar**

Graph Valid Tree, Redundant Connection, Number of Provinces.`,
			solutionCode: `from typing import List

class UnionFind:
    """Yo'lni siqish va rang bo'yicha birlashtirish bilan ajratilgan to'plamlar."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Yo'lni siqish bilan ildizni topish."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Rang bo'yicha birlashtirish. Birlashtirilsa True, bir xil to'plamda bo'lsa False."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True


def has_cycle(n: int, edges: List[List[int]]) -> bool:
    """Yo'naltirilmagan grafda sikl borligini aniqlaydi."""
    uf = UnionFind(n)

    for u, v in edges:
        if uf.find(u) == uf.find(v):
            return True
        uf.union(u, v)

    return False`
		}
	}
};

export default task;
