import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-clone-graph',
	title: 'Clone Graph',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'dfs', 'bfs', 'hash-map', 'deep-copy'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Create a deep copy of a connected undirected graph.

**Problem:**

Given a reference of a node in a connected undirected graph, return a **deep copy** (clone) of the graph.

Each node in the graph contains:
- A value (int)
- A list of its neighbors (List[Node])

**Examples:**

\`\`\`
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]

Explanation:
Node 1's neighbors are nodes 2 and 4.
Node 2's neighbors are nodes 1 and 3.
Node 3's neighbors are nodes 2 and 4.
Node 4's neighbors are nodes 1 and 3.

The cloned graph has the same structure but different object references.

Input: adjList = [[]]
Output: [[]]
Explanation: Graph has one node with no neighbors.

Input: adjList = []
Output: []
Explanation: Empty graph.
\`\`\`

**Visualization:**

\`\`\`
Original Graph:
    1 --- 2
    |     |
    4 --- 3

Clone Process (DFS):
1. Visit node 1, create clone1
2. Visit node 2 (neighbor of 1), create clone2
3. Visit node 3 (neighbor of 2), create clone3
4. Visit node 4 (neighbor of 3), create clone4
5. Node 1 already cloned, link clone4 to clone1
6. Continue linking all cloned neighbors

Result: New graph with same structure, different objects
\`\`\`

**Key Insight:**

Use a **hash map** to track already-cloned nodes:
- Key: original node
- Value: cloned node

This prevents infinite loops in cyclic graphs and ensures each node is cloned exactly once.

**Constraints:**
- 0 <= number of nodes <= 100
- 1 <= Node.val <= 100
- Node.val is unique for each node
- No self-loops or duplicate edges

**Time Complexity:** O(V + E)
**Space Complexity:** O(V) for the hash map`,
	initialCode: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: 'Node') -> 'Node':
    # TODO: Create a deep copy of the graph

    return None`,
	solutionCode: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: 'Node') -> 'Node':
    """
    Create a deep copy of the graph.

    Args:
        node: Reference to a node in the original graph

    Returns:
        Reference to the same node in the cloned graph
    """
    if not node:
        return None

    # Hash map: original node -> cloned node
    cloned = {}

    def dfs(original: 'Node') -> 'Node':
        # If already cloned, return the clone
        if original in cloned:
            return cloned[original]

        # Create clone
        copy = Node(original.val)
        cloned[original] = copy

        # Clone all neighbors
        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)


# BFS approach
from collections import deque

def clone_graph_bfs(node: 'Node') -> 'Node':
    """Clone graph using BFS."""
    if not node:
        return None

    cloned = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        current = queue.popleft()

        for neighbor in current.neighbors:
            if neighbor not in cloned:
                cloned[neighbor] = Node(neighbor.val)
                queue.append(neighbor)

            # Link the cloned neighbor
            cloned[current].neighbors.append(cloned[neighbor])

    return cloned[node]


# Iterative DFS with explicit stack
def clone_graph_iterative(node: 'Node') -> 'Node':
    """Clone graph using iterative DFS."""
    if not node:
        return None

    cloned = {node: Node(node.val)}
    stack = [node]

    while stack:
        current = stack.pop()

        for neighbor in current.neighbors:
            if neighbor not in cloned:
                cloned[neighbor] = Node(neighbor.val)
                stack.append(neighbor)

            cloned[current].neighbors.append(cloned[neighbor])

    return cloned[node]`,
	testCode: `import pytest


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# Import after Node class is defined
from solution import clone_graph


def build_graph(adj_list):
    """Build graph from adjacency list."""
    if not adj_list:
        return None

    nodes = [Node(i + 1) for i in range(len(adj_list))]

    for i, neighbors in enumerate(adj_list):
        nodes[i].neighbors = [nodes[j - 1] for j in neighbors]

    return nodes[0] if nodes else None


def graph_to_adj_list(node):
    """Convert graph to adjacency list for comparison."""
    if not node:
        return []

    visited = {}
    result = {}

    def dfs(n):
        if n.val in visited:
            return
        visited[n.val] = True
        result[n.val] = sorted([neighbor.val for neighbor in n.neighbors])
        for neighbor in n.neighbors:
            dfs(neighbor)

    dfs(node)
    return [result.get(i, []) for i in range(1, len(result) + 1)]


class TestCloneGraph:
    def test_square_graph(self):
        """Test 4-node square graph"""
        original = build_graph([[2, 4], [1, 3], [2, 4], [1, 3]])
        cloned = clone_graph(original)

        # Check structure is same
        assert graph_to_adj_list(cloned) == [[2, 4], [1, 3], [2, 4], [1, 3]]

        # Check it's a deep copy (different objects)
        assert cloned is not original

    def test_single_node(self):
        """Test single node with no neighbors"""
        original = Node(1)
        cloned = clone_graph(original)

        assert cloned.val == 1
        assert cloned.neighbors == []
        assert cloned is not original

    def test_empty_graph(self):
        """Test empty graph (None)"""
        assert clone_graph(None) is None

    def test_two_nodes(self):
        """Test two connected nodes"""
        node1 = Node(1)
        node2 = Node(2)
        node1.neighbors = [node2]
        node2.neighbors = [node1]

        cloned = clone_graph(node1)

        assert cloned.val == 1
        assert len(cloned.neighbors) == 1
        assert cloned.neighbors[0].val == 2
        assert cloned is not node1

    def test_linear_graph(self):
        """Test linear graph 1-2-3"""
        original = build_graph([[2], [1, 3], [2]])
        cloned = clone_graph(original)

        assert graph_to_adj_list(cloned) == [[2], [1, 3], [2]]

    def test_complete_graph(self):
        """Test complete graph (all nodes connected)"""
        original = build_graph([[2, 3], [1, 3], [1, 2]])
        cloned = clone_graph(original)

        assert graph_to_adj_list(cloned) == [[2, 3], [1, 3], [1, 2]]

    def test_deep_copy_verification(self):
        """Verify cloned graph has different object references"""
        node1 = Node(1)
        node2 = Node(2)
        node1.neighbors = [node2]
        node2.neighbors = [node1]

        cloned = clone_graph(node1)

        # Modify original
        node1.val = 100

        # Clone should be unchanged
        assert cloned.val == 1

    def test_star_graph(self):
        """Test star-shaped graph"""
        original = build_graph([[2, 3, 4], [1], [1], [1]])
        cloned = clone_graph(original)

        adj = graph_to_adj_list(cloned)
        assert 2 in adj[0] and 3 in adj[0] and 4 in adj[0]

    def test_cycle_graph(self):
        """Test graph with cycle"""
        original = build_graph([[2], [1, 3], [2, 4], [3, 1]])
        cloned = clone_graph(original)

        # Should handle cycle without infinite loop
        assert cloned is not None
        assert cloned.val == 1

    def test_self_loop_prevention(self):
        """Test that cloned graph neighbors are different objects"""
        node1 = Node(1)
        node2 = Node(2)
        node1.neighbors = [node2]
        node2.neighbors = [node1]

        cloned = clone_graph(node1)

        # Verify neighbor is also cloned
        assert cloned.neighbors[0] is not node2
        assert cloned.neighbors[0].val == 2
        assert cloned.neighbors[0].neighbors[0] is cloned`,
	hint1: `Use a hash map (dictionary) to store the mapping from original nodes to cloned nodes. This handles cycles and prevents infinite loops.`,
	hint2: `When cloning a node, first create the clone and add it to the hash map. Then recursively clone its neighbors. If a neighbor is already in the hash map, just use the existing clone.`,
	whyItMatters: `Clone Graph teaches deep copying of complex data structures with cycles. This pattern is essential for object serialization, state management, and working with graph structures.

**Why This Matters:**

**1. Deep Copy Pattern**

\`\`\`python
# Hash map prevents infinite loops in cyclic structures
# Key pattern: check hash map before creating new objects

def deep_copy_with_cycles(obj, visited=None):
    if visited is None:
        visited = {}

    if id(obj) in visited:
        return visited[id(obj)]

    copy = type(obj)()
    visited[id(obj)] = copy

    # Copy attributes, handling references
    for attr in obj.__dict__:
        setattr(copy, attr, deep_copy_with_cycles(getattr(obj, attr), visited))

    return copy
\`\`\`

**2. Graph Serialization**

\`\`\`python
# Serialize graph to JSON-compatible format
def serialize_graph(node):
    if not node:
        return []

    visited = {}
    def dfs(n, result):
        if n.val in visited:
            return
        visited[n.val] = len(result)
        result.append([neighbor.val for neighbor in n.neighbors])
        for neighbor in n.neighbors:
            dfs(neighbor, result)

    result = []
    dfs(node, result)
    return result

# Deserialize back to graph
def deserialize_graph(adj_list):
    if not adj_list:
        return None

    nodes = [Node(i + 1) for i in range(len(adj_list))]
    for i, neighbors in enumerate(adj_list):
        nodes[i].neighbors = [nodes[j - 1] for j in neighbors]
    return nodes[0]
\`\`\`

**3. State Management / Undo-Redo**

\`\`\`python
class GraphHistory:
    def __init__(self):
        self.history = []
        self.current = -1

    def save_state(self, graph_root):
        # Clone current state for undo
        clone = clone_graph(graph_root)
        self.current += 1
        self.history = self.history[:self.current]
        self.history.append(clone)

    def undo(self):
        if self.current > 0:
            self.current -= 1
            return clone_graph(self.history[self.current])
        return None
\`\`\`

**4. Related Problems**

\`\`\`python
# Copy List with Random Pointer (LeetCode 138)
# Similar idea: hash map to track cloned nodes

def copy_random_list(head):
    if not head:
        return None

    old_to_new = {}

    # First pass: create all nodes
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.val)
        curr = curr.next

    # Second pass: connect next and random pointers
    curr = head
    while curr:
        old_to_new[curr].next = old_to_new.get(curr.next)
        old_to_new[curr].random = old_to_new.get(curr.random)
        curr = curr.next

    return old_to_new[head]
\`\`\`

**5. Interview Tips**

\`\`\`
1. Clarify: What happens with None input?
2. Ask about self-loops and duplicate edges
3. Mention both DFS and BFS approaches
4. Emphasize the hash map for cycle handling
5. Verify it's a deep copy, not shallow
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Клонирование графа',
			description: `Создайте глубокую копию связного неориентированного графа.

**Задача:**

Дана ссылка на узел связного неориентированного графа. Верните **глубокую копию** (клон) графа.

Каждый узел графа содержит:
- Значение (int)
- Список соседей (List[Node])

**Примеры:**

\`\`\`
Вход: adjList = [[2,4],[1,3],[2,4],[1,3]]
Выход: [[2,4],[1,3],[2,4],[1,3]]

Объяснение:
Соседи узла 1 - узлы 2 и 4.
Соседи узла 2 - узлы 1 и 3.
И так далее...

Клонированный граф имеет ту же структуру, но другие ссылки на объекты.
\`\`\`

**Ключевая идея:**

Используйте **хеш-таблицу** для отслеживания уже клонированных узлов:
- Ключ: оригинальный узел
- Значение: клонированный узел

Это предотвращает бесконечные циклы и гарантирует, что каждый узел клонируется ровно один раз.

**Ограничения:**
- 0 <= количество узлов <= 100
- Node.val уникален для каждого узла

**Временная сложность:** O(V + E)
**Пространственная сложность:** O(V)`,
			hint1: `Используйте хеш-таблицу (словарь) для хранения соответствия оригинальных узлов клонированным. Это обрабатывает циклы и предотвращает бесконечные циклы.`,
			hint2: `При клонировании узла сначала создайте клон и добавьте в хеш-таблицу. Затем рекурсивно клонируйте соседей. Если сосед уже в таблице, используйте существующий клон.`,
			whyItMatters: `Клонирование графа учит глубокому копированию сложных структур данных с циклами. Этот паттерн важен для сериализации и управления состоянием.

**Почему это важно:**

**1. Паттерн глубокого копирования**

Хеш-таблица предотвращает бесконечные циклы в циклических структурах.

**2. Сериализация графов**

Преобразование графа в JSON-совместимый формат и обратно.

**3. Управление состоянием / Undo-Redo**

Сохранение состояний для отмены действий.

**4. Связанные задачи**

Copy List with Random Pointer - похожая идея с хеш-таблицей.`,
			solutionCode: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: 'Node') -> 'Node':
    """
    Создаёт глубокую копию графа.

    Args:
        node: Ссылка на узел оригинального графа

    Returns:
        Ссылка на тот же узел в клонированном графе
    """
    if not node:
        return None

    cloned = {}

    def dfs(original: 'Node') -> 'Node':
        if original in cloned:
            return cloned[original]

        copy = Node(original.val)
        cloned[original] = copy

        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)`
		},
		uz: {
			title: 'Grafni klonlash',
			description: `Bog'langan yo'naltirilmagan grafning chuqur nusxasini yarating.

**Masala:**

Bog'langan yo'naltirilmagan grafdagi tugunga havola berilgan. Grafning **chuqur nusxasini** (klonini) qaytaring.

Grafdagi har bir tugun quyidagilarni o'z ichiga oladi:
- Qiymat (int)
- Qo'shnilar ro'yxati (List[Node])

**Misollar:**

\`\`\`
Kirish: adjList = [[2,4],[1,3],[2,4],[1,3]]
Chiqish: [[2,4],[1,3],[2,4],[1,3]]

Izoh:
1-tugunning qo'shnilari 2 va 4 tugunlar.
2-tugunning qo'shnilari 1 va 3 tugunlar.
Va hokazo...

Klonlangan graf bir xil strukturaga ega, lekin boshqa obyekt havolalariga ega.
\`\`\`

**Asosiy tushuncha:**

Allaqachon klonlangan tugunlarni kuzatish uchun **xesh-jadval** ishlating:
- Kalit: asl tugun
- Qiymat: klonlangan tugun

Bu sikllarda cheksiz tsikllarni oldini oladi va har bir tugun faqat bir marta klonlanishini kafolatlaydi.

**Cheklovlar:**
- 0 <= tugunlar soni <= 100
- Node.val har bir tugun uchun noyob

**Vaqt murakkabligi:** O(V + E)
**Xotira murakkabligi:** O(V)`,
			hint1: `Asl tugunlardan klonlangan tugunlarga moslikni saqlash uchun xesh-jadval (lug'at) ishlating. Bu sikllarni qayta ishlaydi va cheksiz tsikllarni oldini oladi.`,
			hint2: `Tugunni klonlashda avval klonni yarating va xesh-jadvalga qo'shing. Keyin qo'shnilarni rekursiv klonlang. Agar qo'shni jadvalda bo'lsa, mavjud klonni ishlating.`,
			whyItMatters: `Grafni klonlash siklli murakkab ma'lumotlar strukturalarini chuqur nusxalashni o'rgatadi. Bu pattern serializatsiya va holat boshqarish uchun muhim.

**Bu nima uchun muhim:**

**1. Chuqur nusxalash patterni**

Xesh-jadval siklik strukturalarda cheksiz tsikllarni oldini oladi.

**2. Graflarni serializatsiya qilish**

Grafni JSON-mos formatga o'zgartirish va qaytarish.

**3. Holat boshqarish / Undo-Redo**

Amallarni bekor qilish uchun holatlarni saqlash.

**4. Bog'liq masalalar**

Copy List with Random Pointer - xesh-jadval bilan o'xshash g'oya.`,
			solutionCode: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: 'Node') -> 'Node':
    """
    Grafning chuqur nusxasini yaratadi.

    Args:
        node: Asl grafdagi tugunga havola

    Returns:
        Klonlangan grafdagi o'sha tugunga havola
    """
    if not node:
        return None

    cloned = {}

    def dfs(original: 'Node') -> 'Node':
        if original in cloned:
            return cloned[original]

        copy = Node(original.val)
        cloned[original] = copy

        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)`
		}
	}
};

export default task;
