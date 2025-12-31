import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-level-order',
	title: 'Level Order Traversal',
	difficulty: 'medium',
	tags: ['python', 'tree', 'bfs', 'queue'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Return level order traversal of a binary tree (BFS).

**Problem:**

Given the root of a binary tree, return the level order traversal of its nodes' values (from left to right, level by level).

**Examples:**

\`\`\`
Input:
    3
   / \\
  9  20
    /  \\
   15   7
Output: [[3], [9, 20], [15, 7]]

Input:
  1
Output: [[1]]

Input: None
Output: []
\`\`\`

**BFS Approach:**

1. Use a queue to process nodes level by level
2. For each level, process all nodes currently in queue
3. Add children to queue for next level

**Time Complexity:** O(n)
**Space Complexity:** O(n) for the queue`,
	initialCode: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    # TODO: Return level order traversal (BFS) of the tree

    return []`,
	solutionCode: `from typing import Optional, List
from collections import deque


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return level order traversal of binary tree.

    Args:
        root: Root of the binary tree

    Returns:
        List of levels, each level is a list of values
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result`,
	testCode: `import pytest
from solution import TreeNode, level_order


class TestLevelOrder:
    def test_normal(self):
        """Test normal tree"""
        root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)

        assert level_order(root) == [[3], [9, 20], [15, 7]]

    def test_single(self):
        """Test single node"""
        root = TreeNode(1)
        assert level_order(root) == [[1]]

    def test_empty(self):
        """Test empty tree"""
        assert level_order(None) == []

    def test_left_skewed(self):
        """Test left skewed tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.left.left = TreeNode(3)

        assert level_order(root) == [[1], [2], [3]]

    def test_right_skewed(self):
        """Test right skewed tree"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)

        assert level_order(root) == [[1], [2], [3]]

    def test_complete_tree(self):
        """Test complete binary tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        assert level_order(root) == [[1], [2, 3], [4, 5, 6, 7]]

    def test_unbalanced_tree(self):
        """Test unbalanced tree with varying depths"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.right.right = TreeNode(5)

        assert level_order(root) == [[1], [2, 3], [4, 5]]

    def test_only_left_children(self):
        """Test tree with only left children at each level"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.right.left = TreeNode(5)

        assert level_order(root) == [[1], [2, 3], [4, 5]]

    def test_only_right_children(self):
        """Test tree with only right children at deeper levels"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.right = TreeNode(5)

        assert level_order(root) == [[1], [2, 3], [4, 5]]

    def test_larger_tree(self):
        """Test larger tree with 4 levels"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.left.left.left = TreeNode(7)

        assert level_order(root) == [[1], [2, 3], [4, 5, 6], [7]]`,
	hint1: `Use collections.deque for efficient queue operations. The key is to process exactly level_size nodes in each iteration before moving to the next level.`,
	hint2: `Store level_size at the start of each level. This tells you how many nodes to process before the queue starts containing next level nodes.`,
	whyItMatters: `Level order traversal (BFS) is essential for many tree problems.

**Why This Matters:**

**1. BFS vs DFS**

\`\`\`python
# DFS: Goes deep (uses stack/recursion)
# BFS: Goes wide (uses queue)

# Choose based on problem:
# - Find shortest path: BFS
# - Explore all paths: DFS
# - Level-by-level processing: BFS
\`\`\`

**2. Common Applications**

- Find minimum depth
- Right/left side view
- Zigzag level order
- Connect nodes at same level
- Serialize/deserialize tree

**3. Level Size Trick**

The key insight for level-by-level processing:
\`\`\`python
level_size = len(queue)  # Capture BEFORE processing
for _ in range(level_size):
    # Process exactly this many nodes
    # New nodes added go to NEXT level
\`\`\`

**4. Queue in Python**

Use collections.deque for O(1) operations:
\`\`\`python
from collections import deque
queue = deque()
queue.append(item)    # Enqueue (right)
front = queue[0]      # Peek
item = queue.popleft() # Dequeue (left) - O(1)
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Обход по уровням',
			description: `Верните обход бинарного дерева по уровням (BFS).

**Задача:**

Дан корень бинарного дерева, верните обход значений его узлов по уровням (слева направо, уровень за уровнем).

**Примеры:**

\`\`\`
Вход:
    3
   / \\
  9  20
    /  \\
   15   7
Выход: [[3], [9, 20], [15, 7]]

Вход:
  1
Выход: [[1]]
\`\`\`

**Подход BFS:**

1. Используйте очередь для обработки узлов уровень за уровнем
2. Для каждого уровня обработайте все узлы в текущей очереди
3. Добавьте детей в очередь для следующего уровня

**Временная сложность:** O(n)
**Пространственная сложность:** O(n) для очереди`,
			hint1: `Используйте collections.deque для эффективных операций с очередью. Ключевое - обрабатывать ровно level_size узлов на каждой итерации перед переходом к следующему уровню.`,
			hint2: `Сохраните level_size в начале каждого уровня. Это говорит сколько узлов обработать прежде чем очередь начнёт содержать узлы следующего уровня.`,
			whyItMatters: `Обход по уровням (BFS) необходим для многих задач с деревьями.

**Почему это важно:**

**1. BFS vs DFS**

DFS: Идёт вглубь (использует стек/рекурсию)
BFS: Идёт вширь (использует очередь)

**2. Распространённые применения**

- Найти минимальную глубину
- Вид справа/слева
- Зигзагообразный обход по уровням`,
			solutionCode: `from typing import Optional, List
from collections import deque


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Возвращает обход бинарного дерева по уровням.

    Args:
        root: Корень бинарного дерева

    Returns:
        Список уровней, каждый уровень - список значений
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        # Обрабатываем все узлы на текущем уровне
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Добавляем детей для следующего уровня
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result`
		},
		uz: {
			title: 'Daraja bo\'yicha o\'tish',
			description: `Binar daraxtning daraja bo'yicha o'tishini (BFS) qaytaring.

**Masala:**

Binar daraxtning ildizi berilgan, uning tugunlari qiymatlarining daraja bo'yicha o'tishini qaytaring (chapdan o'ngga, daraja bo'yicha).

**Misollar:**

\`\`\`
Kirish:
    3
   / \\
  9  20
    /  \\
   15   7
Chiqish: [[3], [9, 20], [15, 7]]

Kirish:
  1
Chiqish: [[1]]
\`\`\`

**BFS yondashuvi:**

1. Tugunlarni daraja bo'yicha qayta ishlash uchun navbatdan foydalaning
2. Har bir daraja uchun navbatdagi barcha tugunlarni qayta ishlang
3. Keyingi daraja uchun bolalarni navbatga qo'shing

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(n) navbat uchun`,
			hint1: `Samarali navbat operatsiyalari uchun collections.deque dan foydalaning. Asosiysi - har bir iteratsiyada keyingi darajaga o'tishdan oldin aynan level_size tugunni qayta ishlash.`,
			hint2: `Har bir daraja boshida level_size ni saqlang. Bu navbat keyingi daraja tugunlarini o'z ichiga olishdan oldin qancha tugun qayta ishlash kerakligini aytadi.`,
			whyItMatters: `Daraja bo'yicha o'tish (BFS) ko'plab daraxt masalalari uchun muhim.

**Bu nima uchun muhim:**

**1. BFS vs DFS**

DFS: Chuqurlikka boradi (stek/rekursiya ishlatadi)
BFS: Kenglikka boradi (navbat ishlatadi)

**2. Keng tarqalgan qo'llanilishlar**

- Minimal chuqurlikni topish
- O'ng/chap tomondan ko'rinish
- Zigzag daraja bo'yicha o'tish`,
			solutionCode: `from typing import Optional, List
from collections import deque


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Binar daraxtning daraja bo'yicha o'tishini qaytaradi.

    Args:
        root: Binar daraxtning ildizi

    Returns:
        Darajalar ro'yxati, har bir daraja qiymatlar ro'yxati
    """
    if root is None:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        # Joriy darajadagi barcha tugunlarni qayta ishlaymiz
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Keyingi daraja uchun bolalarni qo'shamiz
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result`
		}
	}
};

export default task;
