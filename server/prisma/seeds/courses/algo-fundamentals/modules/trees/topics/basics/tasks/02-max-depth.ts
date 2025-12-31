import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-max-depth',
	title: 'Maximum Depth of Binary Tree',
	difficulty: 'easy',
	tags: ['python', 'tree', 'dfs', 'recursion'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the maximum depth of a binary tree.

**Problem:**

Given the root of a binary tree, return its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Examples:**

\`\`\`
Input:
    3
   / \\
  9  20
    /  \\
   15   7
Output: 3
Explanation: The tree has 3 levels

Input:
  1
   \\
    2
Output: 2

Input: nil
Output: 0
\`\`\`

**Recursive Approach:**

The depth of a tree = 1 + max(depth of left subtree, depth of right subtree)

Base case: empty tree has depth 0

**Time Complexity:** O(n)
**Space Complexity:** O(h) where h is height`,
	initialCode: `from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    # TODO: Return the maximum depth of the binary tree

    return 0`,
	solutionCode: `from typing import Optional


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Return maximum depth of binary tree.

    Args:
        root: Root of the binary tree

    Returns:
        Maximum depth of the tree
    """
    if root is None:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return 1 + max(left_depth, right_depth)`,
	testCode: `import pytest
from solution import TreeNode, max_depth


class TestMaxDepth:
    def test_normal(self):
        """Test normal tree with 3 levels"""
        root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)

        assert max_depth(root) == 3

    def test_skewed(self):
        """Test skewed tree"""
        root = TreeNode(1)
        root.right = TreeNode(2)

        assert max_depth(root) == 2

    def test_empty(self):
        """Test empty tree"""
        assert max_depth(None) == 0

    def test_single(self):
        """Test single node"""
        root = TreeNode(1)
        assert max_depth(root) == 1

    def test_left_heavy(self):
        """Test left-heavy tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.left.left = TreeNode(3)

        assert max_depth(root) == 3

    def test_right_only(self):
        """Test right-only tree with 4 levels"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)
        root.right.right.right = TreeNode(4)

        assert max_depth(root) == 4

    def test_complete_tree(self):
        """Test complete binary tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        assert max_depth(root) == 3

    def test_unbalanced_deeper_left(self):
        """Test unbalanced tree with deeper left subtree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.left.left = TreeNode(5)
        root.left.left.left.left = TreeNode(6)

        assert max_depth(root) == 5

    def test_unbalanced_deeper_right(self):
        """Test unbalanced tree with deeper right subtree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.right = TreeNode(4)
        root.right.right.right = TreeNode(5)

        assert max_depth(root) == 4

    def test_mixed_depth(self):
        """Test tree with mixed depths on both sides"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.right.right = TreeNode(5)
        root.right.right.right = TreeNode(6)

        assert max_depth(root) == 4`,
	hint1: `Think recursively: the depth of a tree is 1 (for the root) plus the maximum of the depths of its left and right subtrees.`,
	hint2: `Base case: a None node has depth 0. A single node (leaf) has depth 1.`,
	whyItMatters: `Tree depth calculation is a foundational recursive problem.

**Why This Matters:**

**1. Classic Recursion Pattern**

This is one of the purest examples of recursive thinking:
\`\`\`python
# The answer for the whole = combine answers from parts
max_depth(tree) = 1 + max(max_depth(left), max_depth(right))
\`\`\`

**2. Foundation for Other Problems**

Depth calculation is used in:
- Balanced tree checking
- Tree diameter calculation
- AVL tree balancing
- Red-black tree operations

**3. DFS Pattern**

This demonstrates depth-first search:
\`\`\`python
# DFS goes deep before going wide
# Process children completely before moving to siblings
\`\`\`

**4. BFS Alternative**

\`\`\`python
from collections import deque

def max_depth_bfs(root: TreeNode) -> int:
    if root is None:
        return 0

    queue = deque([root])
    depth = 0

    while queue:
        depth += 1
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return depth
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Максимальная глубина бинарного дерева',
			description: `Найдите максимальную глубину бинарного дерева.

**Задача:**

Дан корень бинарного дерева, верните его максимальную глубину.

Максимальная глубина - это количество узлов вдоль самого длинного пути от корневого узла до самого дальнего листового узла.

**Примеры:**

\`\`\`
Вход:
    3
   / \\
  9  20
    /  \\
   15   7
Выход: 3
Объяснение: Дерево имеет 3 уровня

Вход:
  1
   \\
    2
Выход: 2
\`\`\`

**Рекурсивный подход:**

Глубина дерева = 1 + max(глубина левого поддерева, глубина правого поддерева)

Базовый случай: пустое дерево имеет глубину 0

**Временная сложность:** O(n)
**Пространственная сложность:** O(h), где h - высота`,
			hint1: `Думайте рекурсивно: глубина дерева равна 1 (для корня) плюс максимум из глубин его левого и правого поддеревьев.`,
			hint2: `Базовый случай: None узел имеет глубину 0. Одиночный узел (лист) имеет глубину 1.`,
			whyItMatters: `Вычисление глубины дерева - фундаментальная рекурсивная задача.

**Почему это важно:**

**1. Классический паттерн рекурсии**

Это один из чистейших примеров рекурсивного мышления.

**2. Основа для других задач**

Вычисление глубины используется в:
- Проверке сбалансированности дерева
- Вычислении диаметра дерева
- Балансировке AVL дерева`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Возвращает максимальную глубину бинарного дерева.

    Args:
        root: Корень бинарного дерева

    Returns:
        Максимальная глубина дерева
    """
    if root is None:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return 1 + max(left_depth, right_depth)`
		},
		uz: {
			title: 'Binar daraxtning maksimal chuqurligi',
			description: `Binar daraxtning maksimal chuqurligini toping.

**Masala:**

Binar daraxtning ildizi berilgan, uning maksimal chuqurligini qaytaring.

Maksimal chuqurlik ildiz tugunidan eng uzoq barg tugunigacha bo'lgan eng uzun yo'l bo'ylab tugunlar soni.

**Misollar:**

\`\`\`
Kirish:
    3
   / \\
  9  20
    /  \\
   15   7
Chiqish: 3
Tushuntirish: Daraxt 3 darajaga ega

Kirish:
  1
   \\
    2
Chiqish: 2
\`\`\`

**Rekursiv yondashuv:**

Daraxt chuqurligi = 1 + max(chap pastki daraxt chuqurligi, o'ng pastki daraxt chuqurligi)

Asosiy holat: bo'sh daraxt 0 chuqurlikka ega

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(h), bu yerda h balandlik`,
			hint1: `Rekursiv o'ylang: daraxt chuqurligi 1 (ildiz uchun) va chap va o'ng pastki daraxtlarning chuqurliklarining maksimumiga teng.`,
			hint2: `Asosiy holat: None tugun 0 chuqurlikka ega. Yagona tugun (barg) 1 chuqurlikka ega.`,
			whyItMatters: `Daraxt chuqurligini hisoblash asosiy rekursiv masala.

**Bu nima uchun muhim:**

**1. Klassik rekursiya patterni**

Bu rekursiv fikrlashning eng toza misollaridan biri.

**2. Boshqa masalalar uchun asos**

Chuqurlik hisoblash quyidagilarda ishlatiladi:
- Daraxt muvozanatliligini tekshirish
- Daraxt diametrini hisoblash`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Binar daraxtning maksimal chuqurligini qaytaradi.

    Args:
        root: Binar daraxtning ildizi

    Returns:
        Daraxtning maksimal chuqurligi
    """
    if root is None:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return 1 + max(left_depth, right_depth)`
		}
	}
};

export default task;
