import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-same-tree',
	title: 'Same Tree',
	difficulty: 'easy',
	tags: ['python', 'tree', 'dfs', 'recursion'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Check if two binary trees are identical.

**Problem:**

Given the roots of two binary trees \`p\` and \`q\`, write a function to check if they are the same.

Two trees are considered the same if they are structurally identical and have the same node values.

**Examples:**

\`\`\`
Input:
  p:    1       q:    1
       / \\         / \\
      2   3       2   3
Output: true

Input:
  p:    1       q:    1
       /             \\
      2               2
Output: false

Input:
  p:    1       q:    1
       / \\         / \\
      2   1       1   2
Output: false
\`\`\`

**Recursive Approach:**

1. If both are None, return True
2. If one is None, return False
3. If values differ, return False
4. Recursively check left subtrees AND right subtrees

**Time Complexity:** O(n)
**Space Complexity:** O(h)`,
	initialCode: `from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # TODO: Check if two trees are structurally identical with same values

    return False`,
	solutionCode: `from typing import Optional


class TreeNode:
    """Represents a node in binary tree."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Check if two trees are identical.

    Args:
        p: Root of the first tree
        q: Root of the second tree

    Returns:
        True if trees are identical, False otherwise
    """
    # Both None - same (both empty)
    if p is None and q is None:
        return True

    # One None - different structure
    if p is None or q is None:
        return False

    # Values differ
    if p.val != q.val:
        return False

    # Recursively check both subtrees
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)`,
	testCode: `import pytest
from solution import TreeNode, is_same_tree


class TestIsSameTree:
    def test_same_trees(self):
        """Test identical trees"""
        p = TreeNode(1)
        p.left = TreeNode(2)
        p.right = TreeNode(3)

        q = TreeNode(1)
        q.left = TreeNode(2)
        q.right = TreeNode(3)

        assert is_same_tree(p, q) == True

    def test_different_structure(self):
        """Test trees with different structure"""
        p = TreeNode(1)
        p.left = TreeNode(2)

        q = TreeNode(1)
        q.right = TreeNode(2)

        assert is_same_tree(p, q) == False

    def test_different_values(self):
        """Test trees with different values"""
        p = TreeNode(1)
        p.left = TreeNode(2)
        p.right = TreeNode(1)

        q = TreeNode(1)
        q.left = TreeNode(1)
        q.right = TreeNode(2)

        assert is_same_tree(p, q) == False

    def test_both_none(self):
        """Test both trees are None"""
        assert is_same_tree(None, None) == True

    def test_one_none(self):
        """Test one tree is None"""
        p = TreeNode(1)
        assert is_same_tree(p, None) == False
        assert is_same_tree(None, p) == False

    def test_single_node_same(self):
        """Test single nodes with same value"""
        p = TreeNode(5)
        q = TreeNode(5)

        assert is_same_tree(p, q) == True

    def test_single_node_different(self):
        """Test single nodes with different values"""
        p = TreeNode(5)
        q = TreeNode(3)

        assert is_same_tree(p, q) == False

    def test_complete_trees_same(self):
        """Test complete trees that are identical"""
        p = TreeNode(1)
        p.left = TreeNode(2)
        p.right = TreeNode(3)
        p.left.left = TreeNode(4)
        p.left.right = TreeNode(5)

        q = TreeNode(1)
        q.left = TreeNode(2)
        q.right = TreeNode(3)
        q.left.left = TreeNode(4)
        q.left.right = TreeNode(5)

        assert is_same_tree(p, q) == True

    def test_different_depth(self):
        """Test trees with different depths"""
        p = TreeNode(1)
        p.left = TreeNode(2)
        p.left.left = TreeNode(3)

        q = TreeNode(1)
        q.left = TreeNode(2)

        assert is_same_tree(p, q) == False

    def test_left_only_same(self):
        """Test left-only trees that are identical"""
        p = TreeNode(1)
        p.left = TreeNode(2)
        p.left.left = TreeNode(3)

        q = TreeNode(1)
        q.left = TreeNode(2)
        q.left.left = TreeNode(3)

        assert is_same_tree(p, q) == True`,
	hint1: `Handle the base cases first: both None (return True), one None (return False). Then check if values are equal.`,
	hint2: `Use short-circuit evaluation: return the result of checking left subtrees AND right subtrees. If either is False, the whole expression is False.`,
	whyItMatters: `Tree comparison is fundamental to many tree operations.

**Why This Matters:**

**1. Foundation for Tree Operations**

This pattern is used in:
- Subtree checking
- Tree isomorphism
- Mirror/symmetric tree checking

**2. Structural Recursion**

This demonstrates comparing two recursive structures:
\`\`\`python
# Compare both value AND structure at each step
# Both must match for trees to be same
\`\`\`

**3. Short-Circuit Evaluation**

\`\`\`python
# 'and' stops at first False
return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
# If left subtrees differ, right subtrees not checked
\`\`\`

**4. Related Problems**

- Symmetric Tree (is tree mirror of itself)
- Subtree of Another Tree
- Merge Two Binary Trees`,
	order: 3,
	translations: {
		ru: {
			title: 'Одинаковые деревья',
			description: `Проверьте, идентичны ли два бинарных дерева.

**Задача:**

Даны корни двух бинарных деревьев \`p\` и \`q\`, напишите функцию для проверки их идентичности.

Два дерева считаются одинаковыми, если они структурно идентичны и имеют одинаковые значения узлов.

**Примеры:**

\`\`\`
Вход:
  p:    1       q:    1
       / \\         / \\
      2   3       2   3
Выход: true

Вход:
  p:    1       q:    1
       /             \\
      2               2
Выход: false
\`\`\`

**Рекурсивный подход:**

1. Если оба None, вернуть True
2. Если один None, вернуть False
3. Если значения различаются, вернуть False
4. Рекурсивно проверить левые поддеревья И правые поддеревья

**Временная сложность:** O(n)
**Пространственная сложность:** O(h)`,
			hint1: `Сначала обработайте базовые случаи: оба None (вернуть True), один None (вернуть False). Затем проверьте равенство значений.`,
			hint2: `Используйте ленивые вычисления: верните результат проверки левых поддеревьев И правых поддеревьев. Если любое False, всё выражение False.`,
			whyItMatters: `Сравнение деревьев фундаментально для многих операций с деревьями.

**Почему это важно:**

**1. Основа для операций с деревьями**

Этот паттерн используется в:
- Проверке поддерева
- Изоморфизме деревьев
- Проверке зеркальности/симметричности`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Представляет узел бинарного дерева."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Проверяет, идентичны ли два дерева.

    Args:
        p: Корень первого дерева
        q: Корень второго дерева

    Returns:
        True если деревья идентичны, False иначе
    """
    # Оба None - одинаковы (оба пустые)
    if p is None and q is None:
        return True

    # Один None - разная структура
    if p is None or q is None:
        return False

    # Значения различаются
    if p.val != q.val:
        return False

    # Рекурсивно проверяем оба поддерева
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)`
		},
		uz: {
			title: 'Bir xil daraxtlar',
			description: `Ikki binar daraxt bir xil ekanligini tekshiring.

**Masala:**

Ikki binar daraxtning ildizlari \`p\` va \`q\` berilgan, ular bir xil ekanligini tekshirish uchun funktsiya yozing.

Ikki daraxt tuzilmaviy bir xil bo'lsa va bir xil tugun qiymatlariga ega bo'lsa, bir xil hisoblanadi.

**Misollar:**

\`\`\`
Kirish:
  p:    1       q:    1
       / \\         / \\
      2   3       2   3
Chiqish: true

Kirish:
  p:    1       q:    1
       /             \\
      2               2
Chiqish: false
\`\`\`

**Rekursiv yondashuv:**

1. Agar ikkisi ham None, True qaytaring
2. Agar biri None, False qaytaring
3. Agar qiymatlar farq qilsa, False qaytaring
4. Chap pastki daraxtlarni VA o'ng pastki daraxtlarni rekursiv tekshiring

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(h)`,
			hint1: `Avval asosiy holatlarni qayta ishlang: ikkisi ham None (True qaytaring), biri None (False qaytaring). Keyin qiymatlar tengligini tekshiring.`,
			hint2: `Qisqa tutashuv hisoblashdan foydalaning: chap pastki daraxtlarni VA o'ng pastki daraxtlarni tekshirish natijasini qaytaring. Agar biri False bo'lsa, butun ifoda False.`,
			whyItMatters: `Daraxtlarni solishtirish ko'plab daraxt operatsiyalari uchun asosiy.

**Bu nima uchun muhim:**

**1. Daraxt operatsiyalari uchun asos**

Bu pattern quyidagilarda ishlatiladi:
- Pastki daraxtni tekshirish
- Daraxt izomorfizmi
- Ko'zgu/simmetrik daraxtni tekshirish`,
			solutionCode: `from typing import Optional


class TreeNode:
    """Binar daraxtdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Ikki daraxt bir xil ekanligini tekshiradi.

    Args:
        p: Birinchi daraxt ildizi
        q: Ikkinchi daraxt ildizi

    Returns:
        Daraxtlar bir xil bo'lsa True, aks holda False
    """
    # Ikkisi ham None - bir xil (ikkisi ham bo'sh)
    if p is None and q is None:
        return True

    # Biri None - farqli tuzilma
    if p is None or q is None:
        return False

    # Qiymatlar farq qiladi
    if p.val != q.val:
        return False

    # Ikkala pastki daraxtni rekursiv tekshiramiz
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)`
		}
	}
};

export default task;
