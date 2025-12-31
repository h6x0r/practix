import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-inorder-traversal',
	title: 'Inorder Traversal',
	difficulty: 'easy',
	tags: ['python', 'tree', 'dfs', 'traversal'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement inorder traversal of a binary tree.

**Problem:**

Given the root of a binary tree, return the inorder traversal of its nodes' values.

Inorder traversal visits nodes in: Left -> Root -> Right

**Examples:**

\`\`\`
Input:
    1
     \\
      2
     /
    3
Output: [1, 3, 2]

Input:
      4
     / \\
    2   6
   / \\ / \\
  1  3 5  7
Output: [1, 2, 3, 4, 5, 6, 7]
\`\`\`

**Note:** For a Binary Search Tree, inorder gives sorted order.

**Recursive Approach:**
1. Traverse left subtree
2. Visit root
3. Traverse right subtree

**Time Complexity:** O(n)
**Space Complexity:** O(h) where h is height`,
	initialCode: `from typing import Optional, List

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    # TODO: Return inorder traversal (Left -> Root -> Right)

    return []`,
	solutionCode: `from typing import Optional, List


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Return inorder traversal of binary tree.

    Args:
        root: Root of the binary tree

    Returns:
        List of values in inorder
    """
    result = []
    _inorder_helper(root, result)
    return result


def _inorder_helper(node: Optional[TreeNode], result: List[int]) -> None:
    """Helper function for recursive traversal."""
    if node is None:
        return

    # Left -> Root -> Right
    _inorder_helper(node.left, result)
    result.append(node.val)
    _inorder_helper(node.right, result)`,
	testCode: `import pytest
from solution import TreeNode, inorder_traversal


class TestInorderTraversal:
    def test_simple(self):
        """Test simple tree: 1 -> 2 -> 3 (right-left structure)"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.left = TreeNode(3)

        assert inorder_traversal(root) == [1, 3, 2]

    def test_bst(self):
        """Test BST gives sorted output"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)

        assert inorder_traversal(root) == [1, 2, 3, 4, 6]

    def test_empty(self):
        """Test empty tree"""
        assert inorder_traversal(None) == []

    def test_single(self):
        """Test single node"""
        root = TreeNode(5)
        assert inorder_traversal(root) == [5]

    def test_left_skewed(self):
        """Test left-skewed tree"""
        root = TreeNode(3)
        root.left = TreeNode(2)
        root.left.left = TreeNode(1)

        assert inorder_traversal(root) == [1, 2, 3]

    def test_right_skewed(self):
        """Test right-skewed tree"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)

        assert inorder_traversal(root) == [1, 2, 3]

    def test_complete_tree(self):
        """Test complete binary tree"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(5)
        root.right.right = TreeNode(7)

        assert inorder_traversal(root) == [1, 2, 3, 4, 5, 6, 7]

    def test_unbalanced_tree(self):
        """Test unbalanced tree with deeper left subtree"""
        root = TreeNode(5)
        root.left = TreeNode(3)
        root.right = TreeNode(6)
        root.left.left = TreeNode(2)
        root.left.left.left = TreeNode(1)

        assert inorder_traversal(root) == [1, 2, 3, 5, 6]

    def test_negative_values(self):
        """Test tree with negative values"""
        root = TreeNode(0)
        root.left = TreeNode(-2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(-3)
        root.right.right = TreeNode(3)

        assert inorder_traversal(root) == [-3, -2, 0, 2, 3]

    def test_duplicate_values(self):
        """Test tree with duplicate values"""
        root = TreeNode(2)
        root.left = TreeNode(2)
        root.right = TreeNode(2)
        root.left.left = TreeNode(1)

        assert inorder_traversal(root) == [1, 2, 2, 2]`,
	hint1: `Use a helper function that takes the result list as a parameter. This allows you to build up the result across recursive calls.`,
	hint2: `The order is: recurse left, append current value, recurse right. For BST, this gives sorted order.`,
	whyItMatters: `Tree traversals are fundamental operations in computer science.

**Why This Matters:**

**1. Three Traversal Orders**

\`\`\`python
# Inorder (L-Root-R): Used for BST to get sorted order
# Preorder (Root-L-R): Used for copying/serializing tree
# Postorder (L-R-Root): Used for deleting tree
\`\`\`

**2. BST Property**

Inorder on BST gives sorted sequence:
\`\`\`
    4
   / \\
  2   6
 / \\
1   3

Inorder: 1, 2, 3, 4, 6 (sorted!)
\`\`\`

**3. Applications**

- Expression trees (infix expression)
- BST operations (kth smallest element)
- Tree serialization
- Syntax tree evaluation

**4. Iterative Alternative**

\`\`\`python
def inorder_iterative(root: TreeNode) -> List[int]:
    result = []
    stack = []
    curr = root

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right

    return result
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Симметричный обход',
			description: `Реализуйте симметричный (inorder) обход бинарного дерева.

**Задача:**

Дан корень бинарного дерева, верните симметричный обход значений его узлов.

Симметричный обход посещает узлы в порядке: Левый -> Корень -> Правый

**Примеры:**

\`\`\`
Вход:
    1
     \\
      2
     /
    3
Выход: [1, 3, 2]

Вход:
      4
     / \\
    2   6
   / \\
  1   3
Выход: [1, 2, 3, 4, 6]
\`\`\`

**Примечание:** Для бинарного дерева поиска inorder даёт отсортированный порядок.

**Рекурсивный подход:**
1. Обойти левое поддерево
2. Посетить корень
3. Обойти правое поддерево

**Временная сложность:** O(n)
**Пространственная сложность:** O(h), где h - высота`,
			hint1: `Используйте вспомогательную функцию, которая принимает список результата как параметр. Это позволяет накапливать результат через рекурсивные вызовы.`,
			hint2: `Порядок: рекурсия влево, добавить текущее значение, рекурсия вправо. Для BST это даёт отсортированный порядок.`,
			whyItMatters: `Обходы деревьев - фундаментальные операции в информатике.

**Почему это важно:**

**1. Три порядка обхода**

- Inorder (Л-Корень-П): Для BST даёт отсортированный порядок
- Preorder (Корень-Л-П): Для копирования/сериализации дерева
- Postorder (Л-П-Корень): Для удаления дерева

**2. Свойство BST**

Inorder на BST даёт отсортированную последовательность.`,
			solutionCode: `from typing import Optional, List


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Возвращает симметричный обход бинарного дерева.

    Args:
        root: Корень бинарного дерева

    Returns:
        Список значений в порядке inorder
    """
    result = []
    _inorder_helper(root, result)
    return result


def _inorder_helper(node: Optional[TreeNode], result: List[int]) -> None:
    """Вспомогательная функция для рекурсивного обхода."""
    if node is None:
        return

    # Левый -> Корень -> Правый
    _inorder_helper(node.left, result)
    result.append(node.val)
    _inorder_helper(node.right, result)`
		},
		uz: {
			title: 'Inorder o\'tish',
			description: `Binar daraxtning inorder o'tishini amalga oshiring.

**Masala:**

Binar daraxtning ildizi berilgan, uning tugunlari qiymatlarining inorder o'tishini qaytaring.

Inorder o'tish tugunlarni quyidagi tartibda tashrif buyuradi: Chap -> Ildiz -> O'ng

**Misollar:**

\`\`\`
Kirish:
    1
     \\
      2
     /
    3
Chiqish: [1, 3, 2]

Kirish:
      4
     / \\
    2   6
   / \\
  1   3
Chiqish: [1, 2, 3, 4, 6]
\`\`\`

**Eslatma:** Binar qidiruv daraxti uchun inorder saralangan tartibni beradi.

**Rekursiv yondashuv:**
1. Chap pastki daraxtni o'ting
2. Ildizga tashrif buyuring
3. O'ng pastki daraxtni o'ting

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(h), bu yerda h balandlik`,
			hint1: `Natija ro'yxatini parametr sifatida qabul qiluvchi yordamchi funktsiyadan foydalaning. Bu rekursiv chaqiriqlar orqali natijani to'plash imkonini beradi.`,
			hint2: `Tartib: chapga rekursiya, joriy qiymatni qo'shish, o'ngga rekursiya. BST uchun bu saralangan tartibni beradi.`,
			whyItMatters: `Daraxt o'tishlari kompyuter fanida asosiy operatsiyalar.

**Bu nima uchun muhim:**

**1. Uchta o'tish tartibi**

- Inorder (Ch-Ildiz-O'): BST uchun saralangan tartibni beradi
- Preorder (Ildiz-Ch-O'): Daraxtni nusxalash/serializatsiya uchun
- Postorder (Ch-O'-Ildiz): Daraxtni o'chirish uchun

**2. BST xususiyati**

BST da inorder saralangan ketma-ketlikni beradi.`,
			solutionCode: `from typing import Optional, List


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    Binar daraxtning inorder o'tishini qaytaradi.

    Args:
        root: Binar daraxtning ildizi

    Returns:
        Inorder tartibidagi qiymatlar ro'yxati
    """
    result = []
    _inorder_helper(root, result)
    return result


def _inorder_helper(node: Optional[TreeNode], result: List[int]) -> None:
    """Rekursiv o'tish uchun yordamchi funktsiya."""
    if node is None:
        return

    # Chap -> Ildiz -> O'ng
    _inorder_helper(node.left, result)
    result.append(node.val)
    _inorder_helper(node.right, result)`
		}
	}
};

export default task;
