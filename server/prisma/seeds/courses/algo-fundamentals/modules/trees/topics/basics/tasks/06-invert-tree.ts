import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-invert-tree',
	title: 'Invert Binary Tree',
	difficulty: 'easy',
	tags: ['python', 'tree', 'dfs', 'recursion'],
	estimatedTime: '10m',
	isPremium: false,
	youtubeUrl: '',
	description: `Invert a binary tree (mirror image).

**Problem:**

Given the root of a binary tree, invert the tree, and return its root.

Inverting means swapping left and right children for every node.

**Examples:**

\`\`\`
Input:
     4
   /   \\
  2     7
 / \\   / \\
1   3 6   9

Output:
     4
   /   \\
  7     2
 / \\   / \\
9   6 3   1

Input:
  2
 / \\
1   3
Output:
  2
 / \\
3   1

Input: None
Output: None
\`\`\`

**Recursive Approach:**

1. If root is None, return None
2. Swap left and right children
3. Recursively invert left subtree
4. Recursively invert right subtree
5. Return root

**Time Complexity:** O(n)
**Space Complexity:** O(h)`,
	initialCode: `from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    # TODO: Invert the binary tree (swap left and right for all nodes)

    return None`,
	solutionCode: `from typing import Optional


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Invert a binary tree (creates mirror image).

    Args:
        root: Root of the binary tree

    Returns:
        Root of the inverted tree
    """
    if root is None:
        return None

    # Swap left and right children
    root.left, root.right = root.right, root.left

    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)

    return root`,
	testCode: `import pytest
from solution import TreeNode, invert_tree


class TestInvertTree:
    def test_normal(self):
        """Test normal tree inversion"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(7)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(9)

        result = invert_tree(root)

        # Check structure after inversion
        assert result.left.val == 7
        assert result.right.val == 2
        assert result.left.left.val == 9
        assert result.left.right.val == 6
        assert result.right.left.val == 3
        assert result.right.right.val == 1

    def test_simple(self):
        """Test simple tree"""
        root = TreeNode(2)
        root.left = TreeNode(1)
        root.right = TreeNode(3)

        result = invert_tree(root)

        assert result.left.val == 3
        assert result.right.val == 1

    def test_single(self):
        """Test single node"""
        root = TreeNode(1)
        result = invert_tree(root)

        assert result.val == 1
        assert result.left is None
        assert result.right is None

    def test_empty(self):
        """Test empty tree"""
        assert invert_tree(None) is None

    def test_left_only(self):
        """Test tree with only left children"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.left.left = TreeNode(3)

        result = invert_tree(root)

        assert result.right.val == 2
        assert result.right.right.val == 3
        assert result.left is None

    def test_right_only(self):
        """Test tree with only right children"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)

        result = invert_tree(root)

        assert result.left.val == 2
        assert result.left.left.val == 3
        assert result.right is None

    def test_complete_tree(self):
        """Test complete binary tree inversion"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        result = invert_tree(root)

        assert result.left.val == 3
        assert result.right.val == 2
        assert result.left.left.val == 7
        assert result.left.right.val == 6
        assert result.right.left.val == 5
        assert result.right.right.val == 4

    def test_unbalanced_tree(self):
        """Test unbalanced tree inversion"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.left.left = TreeNode(5)

        result = invert_tree(root)

        assert result.right.val == 2
        assert result.left.val == 3
        assert result.right.right.val == 4
        assert result.right.right.right.val == 5

    def test_partial_children(self):
        """Test tree with partial children"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.left = TreeNode(5)

        result = invert_tree(root)

        assert result.right.val == 2
        assert result.left.val == 3
        assert result.right.left.val == 4
        assert result.left.right.val == 5

    def test_double_invert(self):
        """Test double inversion returns to original"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)

        # First inversion
        inverted = invert_tree(root)
        assert inverted.left.val == 3
        assert inverted.right.val == 2

        # Second inversion should return to original structure
        double_inverted = invert_tree(inverted)
        assert double_inverted.left.val == 2
        assert double_inverted.right.val == 3`,
	hint1: `This is one of the simplest tree problems. Just swap left and right at each node, then recurse.`,
	hint2: `Use Python's multiple assignment: root.left, root.right = root.right, root.left`,
	whyItMatters: `Tree inversion is a classic interview question with a famous story.

**Why This Matters:**

**1. The Homebrew Story**

This problem became famous when Max Howell (creator of Homebrew) tweeted:
> "Google: 90% of our engineers use the software you wrote (Homebrew), but you can't invert a binary tree on a whiteboard so f*** off."

**2. Simplest Tree Transformation**

This is the most basic tree modification:
\`\`\`python
# The core operation is just swapping
root.left, root.right = root.right, root.left
\`\`\`

**3. Order of Operations**

You can swap before or after recursing - both work:
\`\`\`python
# Pre-order (swap first)
root.left, root.right = root.right, root.left
invert_tree(root.left)
invert_tree(root.right)

# Post-order (swap last)
invert_tree(root.left)
invert_tree(root.right)
root.left, root.right = root.right, root.left
\`\`\`

**4. Iterative Version**

\`\`\`python
from collections import deque

def invert_tree_iterative(root: TreeNode) -> TreeNode:
    if root is None:
        return None

    queue = deque([root])
    while queue:
        node = queue.popleft()
        node.left, node.right = node.right, node.left
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return root
\`\`\`

**5. Related Problems**

- Symmetric Tree (check if tree is mirror of itself)
- Same Tree (compare two trees)
- Mirror of N-ary tree`,
	order: 6,
	translations: {
		ru: {
			title: 'Инвертировать бинарное дерево',
			description: `Инвертируйте бинарное дерево (зеркальное отражение).

**Задача:**

Дан корень бинарного дерева, инвертируйте дерево и верните его корень.

Инвертирование означает обмен левого и правого потомков для каждого узла.

**Примеры:**

\`\`\`
Вход:
     4
   /   \\
  2     7
 / \\   / \\
1   3 6   9

Выход:
     4
   /   \\
  7     2
 / \\   / \\
9   6 3   1

Вход:
  2
 / \\
1   3
Выход:
  2
 / \\
3   1
\`\`\`

**Рекурсивный подход:**

1. Если корень None, вернуть None
2. Поменять местами левого и правого потомков
3. Рекурсивно инвертировать левое поддерево
4. Рекурсивно инвертировать правое поддерево
5. Вернуть корень

**Временная сложность:** O(n)
**Пространственная сложность:** O(h)`,
			hint1: `Это одна из простейших задач на деревья. Просто поменяйте местами left и right в каждом узле, затем рекурсия.`,
			hint2: `Используйте множественное присваивание Python: root.left, root.right = root.right, root.left`,
			whyItMatters: `Инвертирование дерева - классический вопрос на собеседовании с известной историей.

**Почему это важно:**

**1. История Homebrew**

Эта задача стала знаменитой после твита Макса Хауэлла (создателя Homebrew).

**2. Простейшая трансформация дерева**

Это самая базовая модификация дерева.

**3. Порядок операций**

Можно менять до или после рекурсии - оба варианта работают.`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Инвертирует бинарное дерево (создаёт зеркальное отражение).

    Args:
        root: Корень бинарного дерева

    Returns:
        Корень инвертированного дерева
    """
    if root is None:
        return None

    # Меняем местами левого и правого потомков
    root.left, root.right = root.right, root.left

    # Рекурсивно инвертируем поддеревья
    invert_tree(root.left)
    invert_tree(root.right)

    return root`
		},
		uz: {
			title: 'Binar daraxtni inversiya qilish',
			description: `Binar daraxtni inversiya qiling (ko'zgu tasviri).

**Masala:**

Binar daraxtning ildizi berilgan, daraxtni inversiya qiling va ildizini qaytaring.

Inversiya qilish har bir tugun uchun chap va o'ng bolalarni almashtirish degani.

**Misollar:**

\`\`\`
Kirish:
     4
   /   \\
  2     7
 / \\   / \\
1   3 6   9

Chiqish:
     4
   /   \\
  7     2
 / \\   / \\
9   6 3   1

Kirish:
  2
 / \\
1   3
Chiqish:
  2
 / \\
3   1
\`\`\`

**Rekursiv yondashuv:**

1. Agar ildiz None bo'lsa, None qaytaring
2. Chap va o'ng bolalarni almashtiring
3. Chap pastki daraxtni rekursiv inversiya qiling
4. O'ng pastki daraxtni rekursiv inversiya qiling
5. Ildizni qaytaring

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(h)`,
			hint1: `Bu eng oddiy daraxt masalalaridan biri. Shunchaki har bir tugunda left va rightni almashtiring, keyin rekursiya.`,
			hint2: `Python ning ko'p qiymat tayinlashidan foydalaning: root.left, root.right = root.right, root.left`,
			whyItMatters: `Daraxtni inversiya qilish mashhur hikoyaga ega klassik intervyu savoli.

**Bu nima uchun muhim:**

**1. Homebrew hikoyasi**

Bu masala Maks Xauell (Homebrew yaratuvchisi) ning tvitidan keyin mashhur bo'ldi.

**2. Eng oddiy daraxt transformatsiyasi**

Bu eng asosiy daraxt modifikatsiyasi.

**3. Operatsiyalar tartibi**

Rekursiyadan oldin yoki keyin almashtirishingiz mumkin - ikkisi ham ishlaydi.`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Binar daraxtni inversiya qiladi (ko'zgu tasviri yaratadi).

    Args:
        root: Binar daraxtning ildizi

    Returns:
        Inversiya qilingan daraxtning ildizi
    """
    if root is None:
        return None

    # Chap va o'ng bolalarni almashtiramiz
    root.left, root.right = root.right, root.left

    # Pastki daraxtlarni rekursiv inversiya qilamiz
    invert_tree(root.left)
    invert_tree(root.right)

    return root`
		}
	}
};

export default task;
