import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-validate-bst',
	title: 'Validate Binary Search Tree',
	difficulty: 'medium',
	tags: ['python', 'tree', 'bst', 'dfs', 'recursion'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if a binary tree is a valid BST.

**Problem:**

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as:
- The left subtree of a node contains only nodes with keys **less than** the node's key
- The right subtree of a node contains only nodes with keys **greater than** the node's key
- Both the left and right subtrees must also be binary search trees

**Examples:**

\`\`\`
Input:
    2
   / \\
  1   3
Output: true

Input:
    5
   / \\
  1   4
     / \\
    3   6
Output: false
Explanation: Root's right child is 4, but 4's left child is 3, which is less than 5

Input:
    5
   / \\
  4   6
     / \\
    3   7
Output: false
Explanation: 3 is in right subtree of 5, but 3 < 5
\`\`\`

**Key Insight:**

It's NOT enough to just check node > left and node < right. You must track the valid range for each node.

**Time Complexity:** O(n)
**Space Complexity:** O(h)`,
	initialCode: `from typing import Optional
import math

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    # TODO: Check if the tree is a valid binary search tree

    return False`,
	solutionCode: `from typing import Optional
import math


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Check if tree is a valid binary search tree.

    Args:
        root: Root of the binary tree

    Returns:
        True if valid BST, False otherwise
    """
    return _validate(root, -math.inf, math.inf)


def _validate(node: Optional[TreeNode], min_val: float, max_val: float) -> bool:
    """Helper function to validate BST with range constraints."""
    if node is None:
        return True

    # Check if current node violates range
    if node.val <= min_val or node.val >= max_val:
        return False

    # Left subtree: all values must be < node.val
    # Right subtree: all values must be > node.val
    return _validate(node.left, min_val, node.val) and \\
           _validate(node.right, node.val, max_val)`,
	testCode: `import pytest
from solution import TreeNode, is_valid_bst


class TestIsValidBST:
    def test_valid(self):
        """Test valid BST"""
        root = TreeNode(2)
        root.left = TreeNode(1)
        root.right = TreeNode(3)

        assert is_valid_bst(root) == True

    def test_invalid_right_subtree(self):
        """Test invalid - right subtree violation"""
        root = TreeNode(5)
        root.left = TreeNode(1)
        root.right = TreeNode(4)
        root.right.left = TreeNode(3)
        root.right.right = TreeNode(6)

        assert is_valid_bst(root) == False

    def test_invalid_deep(self):
        """Test invalid - deeper violation"""
        root = TreeNode(5)
        root.left = TreeNode(4)
        root.right = TreeNode(6)
        root.right.left = TreeNode(3)
        root.right.right = TreeNode(7)

        assert is_valid_bst(root) == False

    def test_single(self):
        """Test single node is valid BST"""
        root = TreeNode(1)
        assert is_valid_bst(root) == True

    def test_empty(self):
        """Test empty tree is valid BST"""
        assert is_valid_bst(None) == True

    def test_equal_values(self):
        """Test equal values are invalid"""
        root = TreeNode(2)
        root.left = TreeNode(2)

        assert is_valid_bst(root) == False

    def test_valid_complete_bst(self):
        """Test valid complete BST"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(5)
        root.right.right = TreeNode(7)

        assert is_valid_bst(root) == True

    def test_invalid_left_subtree(self):
        """Test invalid - left subtree violation"""
        root = TreeNode(5)
        root.left = TreeNode(1)
        root.right = TreeNode(7)
        root.left.right = TreeNode(6)

        assert is_valid_bst(root) == False

    def test_left_only_valid(self):
        """Test valid left-only BST"""
        root = TreeNode(3)
        root.left = TreeNode(2)
        root.left.left = TreeNode(1)

        assert is_valid_bst(root) == True

    def test_right_only_valid(self):
        """Test valid right-only BST"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)

        assert is_valid_bst(root) == True`,
	hint1: `Don't just compare with immediate children. A node in the right subtree must be greater than ALL ancestors on the path to it.`,
	hint2: `Pass a valid range (min, max) to each recursive call. For left child, update max to parent's value. For right child, update min to parent's value.`,
	whyItMatters: `BST validation teaches the concept of invariants and recursive constraints.

**Why This Matters:**

**1. Understanding BST Property**

The BST property is NOT just about parent-child:
\`\`\`python
# WRONG approach:
if node.left and node.left.val >= node.val:
    return False
# This misses violations deeper in the tree
\`\`\`

**2. Range Propagation Pattern**

This pattern appears in many problems:
\`\`\`python
# Each node inherits constraints from ancestors
_validate(node, min_val, max_val)
  _validate(left, min_val, node.val)   # tighter max
  _validate(right, node.val, max_val)  # tighter min
\`\`\`

**3. Alternative: Inorder Traversal**

\`\`\`python
# BST inorder gives sorted sequence
# Check if inorder is strictly increasing
def is_valid_bst_inorder(root: TreeNode) -> bool:
    prev = float('-inf')

    def inorder(node: TreeNode) -> bool:
        nonlocal prev
        if node is None:
            return True
        if not inorder(node.left):
            return False
        if node.val <= prev:
            return False
        prev = node.val
        return inorder(node.right)

    return inorder(root)
\`\`\`

**4. Applications**

- Database index validation
- Ensuring data structure integrity
- Interview favorite question`,
	order: 5,
	translations: {
		ru: {
			title: 'Проверка бинарного дерева поиска',
			description: `Определите, является ли бинарное дерево допустимым BST.

**Задача:**

Дан корень бинарного дерева, определите, является ли оно допустимым бинарным деревом поиска (BST).

Допустимое BST определяется как:
- Левое поддерево узла содержит только узлы с ключами **меньше** ключа узла
- Правое поддерево узла содержит только узлы с ключами **больше** ключа узла
- Оба поддерева также должны быть бинарными деревьями поиска

**Примеры:**

\`\`\`
Вход:
    2
   / \\
  1   3
Выход: true

Вход:
    5
   / \\
  1   4
     / \\
    3   6
Выход: false
Объяснение: Правый потомок корня - 4, но левый потомок 4 - это 3, что меньше 5
\`\`\`

**Ключевой момент:**

Недостаточно просто проверить node > left и node < right. Нужно отслеживать допустимый диапазон для каждого узла.

**Временная сложность:** O(n)
**Пространственная сложность:** O(h)`,
			hint1: `Не сравнивайте только с непосредственными потомками. Узел в правом поддереве должен быть больше ВСЕХ предков на пути к нему.`,
			hint2: `Передавайте допустимый диапазон (min, max) в каждый рекурсивный вызов. Для левого потомка обновите max значением родителя. Для правого - обновите min.`,
			whyItMatters: `Проверка BST учит концепции инвариантов и рекурсивных ограничений.

**Почему это важно:**

**1. Понимание свойства BST**

Свойство BST - это НЕ только про родителя и потомка.

**2. Паттерн распространения диапазона**

Этот паттерн встречается во многих задачах.

**3. Альтернатива: Inorder обход**

Inorder обход BST даёт отсортированную последовательность.`,
			solutionCode: `from typing import Optional
import math


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Проверяет, является ли дерево допустимым BST.

    Args:
        root: Корень бинарного дерева

    Returns:
        True если допустимое BST, False иначе
    """
    return _validate(root, -math.inf, math.inf)


def _validate(node: Optional[TreeNode], min_val: float, max_val: float) -> bool:
    """Вспомогательная функция для проверки BST с ограничениями диапазона."""
    if node is None:
        return True

    # Проверяем, нарушает ли текущий узел диапазон
    if node.val <= min_val or node.val >= max_val:
        return False

    # Левое поддерево: все значения должны быть < node.val
    # Правое поддерево: все значения должны быть > node.val
    return _validate(node.left, min_val, node.val) and \\
           _validate(node.right, node.val, max_val)`
		},
		uz: {
			title: 'Binar qidiruv daraxtini tekshirish',
			description: `Binar daraxt to'g'ri BST ekanligini aniqlang.

**Masala:**

Binar daraxtning ildizi berilgan, u to'g'ri binar qidiruv daraxti (BST) ekanligini aniqlang.

To'g'ri BST quyidagicha aniqlanadi:
- Tugunning chap pastki daraxti faqat tugun kalitidan **kichik** kalitli tugunlarni o'z ichiga oladi
- Tugunning o'ng pastki daraxti faqat tugun kalitidan **katta** kalitli tugunlarni o'z ichiga oladi
- Chap va o'ng pastki daraxtlarning ikkisi ham binar qidiruv daraxtlari bo'lishi kerak

**Misollar:**

\`\`\`
Kirish:
    2
   / \\
  1   3
Chiqish: true

Kirish:
    5
   / \\
  1   4
     / \\
    3   6
Chiqish: false
Tushuntirish: Ildizning o'ng bolasi 4, lekin 4ning chap bolasi 3, bu 5dan kichik
\`\`\`

**Asosiy tushuncha:**

Faqat node > left va node < right tekshirish yetarli EMAS. Har bir tugun uchun to'g'ri diapazonni kuzatish kerak.

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(h)`,
			hint1: `Faqat bevosita bolalar bilan solishtirib qolmang. O'ng pastki daraxtdagi tugun unga yo'ldagi BARCHA ajdodlardan katta bo'lishi kerak.`,
			hint2: `Har bir rekursiv chaqiriqqa to'g'ri diapazon (min, max) uzating. Chap bola uchun maxni ota qiymati bilan yangilang. O'ng bola uchun minni yangilang.`,
			whyItMatters: `BST tekshirish invariantlar va rekursiv cheklovlar konsepsiyasini o'rgatadi.

**Bu nima uchun muhim:**

**1. BST xususiyatini tushunish**

BST xususiyati faqat ota-bola haqida EMAS.

**2. Diapazon tarqatish patterni**

Bu pattern ko'plab masalalarda uchraydi.

**3. Alternativa: Inorder o'tish**

BST ning inorder o'tishi saralangan ketma-ketlikni beradi.`,
			solutionCode: `from typing import Optional
import math


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Daraxt to'g'ri BST ekanligini tekshiradi.

    Args:
        root: Binar daraxtning ildizi

    Returns:
        To'g'ri BST bo'lsa True, aks holda False
    """
    return _validate(root, -math.inf, math.inf)


def _validate(node: Optional[TreeNode], min_val: float, max_val: float) -> bool:
    """Diapazon cheklovlari bilan BST tekshirish uchun yordamchi funktsiya."""
    if node is None:
        return True

    # Joriy tugun diapazonni buzayotganini tekshiramiz
    if node.val <= min_val or node.val >= max_val:
        return False

    # Chap pastki daraxt: barcha qiymatlar node.val dan kichik bo'lishi kerak
    # O'ng pastki daraxt: barcha qiymatlar node.val dan katta bo'lishi kerak
    return _validate(node.left, min_val, node.val) and \\
           _validate(node.right, node.val, max_val)`
		}
	}
};

export default task;
