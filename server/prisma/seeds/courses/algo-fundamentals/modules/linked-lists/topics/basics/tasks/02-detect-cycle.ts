import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-detect-cycle',
	title: 'Linked List Cycle',
	difficulty: 'easy',
	tags: ['python', 'linked-list', 'two-pointers', 'floyd'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Detect if a linked list has a cycle.

**Problem:**

Given the head of a linked list, determine if the list has a cycle in it.

A cycle exists if some node can be reached again by continuously following the \`next\` pointer.

**Examples:**

\`\`\`
Input: 3 -> 2 -> 0 -> -4 -> (back to 2)
Output: true
Explanation: Node with value -4 connects back to node with value 2

Input: 1 -> 2 -> (back to 1)
Output: true

Input: 1 -> nil
Output: false
Explanation: No cycle exists
\`\`\`

**Floyd's Cycle Detection (Tortoise and Hare):**

1. Use two pointers: slow (moves 1 step) and fast (moves 2 steps)
2. If there's a cycle, fast will eventually meet slow
3. If fast reaches nil, no cycle exists

**Why it works:**

If there's a cycle, the fast pointer will "lap" the slow pointer inside the cycle.

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    # TODO: Detect if the linked list has a cycle

    return False`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Detect if linked list has a cycle.
    Uses Floyd's cycle detection algorithm.

    Args:
        head: Head of the linked list

    Returns:
        True if cycle exists, False otherwise
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next       # Move one step
        fast = fast.next.next  # Move two steps

        if slow == fast:
            return True  # Cycle detected

    return False  # No cycle`,
	testCode: `import pytest
from solution import ListNode, has_cycle


class TestHasCycle:
    def test_has_cycle(self):
        """Test list with cycle"""
        # Create: 3 -> 2 -> 0 -> -4 -> back to 2
        node4 = ListNode(-4)
        node3 = ListNode(0, node4)
        node2 = ListNode(2, node3)
        node1 = ListNode(3, node2)
        node4.next = node2  # Create cycle

        assert has_cycle(node1) == True

    def test_no_cycle(self):
        """Test list without cycle"""
        head = ListNode(1, ListNode(2, ListNode(3)))
        assert has_cycle(head) == False

    def test_single_no_cycle(self):
        """Test single node without cycle"""
        head = ListNode(1)
        assert has_cycle(head) == False

    def test_single_with_cycle(self):
        """Test single node with self-cycle"""
        head = ListNode(1)
        head.next = head
        assert has_cycle(head) == True

    def test_empty(self):
        """Test empty list"""
        assert has_cycle(None) == False

    def test_two_nodes_with_cycle(self):
        """Test two nodes with cycle"""
        # Create: 1 -> 2 -> back to 1
        node2 = ListNode(2)
        node1 = ListNode(1, node2)
        node2.next = node1
        assert has_cycle(node1) == True

    def test_two_nodes_no_cycle(self):
        """Test two nodes without cycle"""
        head = ListNode(1, ListNode(2))
        assert has_cycle(head) == False

    def test_cycle_at_tail(self):
        """Test cycle at tail (last node points to itself)"""
        node3 = ListNode(3)
        node2 = ListNode(2, node3)
        node1 = ListNode(1, node2)
        node3.next = node3
        assert has_cycle(node1) == True

    def test_cycle_at_beginning(self):
        """Test cycle where last node points to head"""
        node3 = ListNode(3)
        node2 = ListNode(2, node3)
        node1 = ListNode(1, node2)
        node3.next = node1
        assert has_cycle(node1) == True

    def test_long_list_no_cycle(self):
        """Test longer list without cycle"""
        head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6))))))
        assert has_cycle(head) == False`,
	hint1: `Use two pointers: slow moves one step at a time, fast moves two steps. Initialize both to head and check if they ever point to the same node.`,
	hint2: `The loop condition should be 'fast and fast.next' to safely access fast.next.next. If fast reaches None, there's no cycle.`,
	whyItMatters: `Floyd's cycle detection is a fundamental algorithm with many applications.

**Why This Matters:**

**1. Floyd's Algorithm**

This elegant O(1) space solution is a classic:
\`\`\`python
# The key insight: if there's a cycle,
# fast moves 2 steps while slow moves 1
# Inside the cycle, fast "catches up" by 1 step each iteration
# They MUST meet eventually
\`\`\`

**2. Extensions**

Once you detect a cycle, you can:
- Find cycle start (reset one pointer to head)
- Find cycle length (count steps after meeting)
- Remove the cycle (find node before cycle start)

**3. Applications**

- Detecting infinite loops in state machines
- Finding cycles in graphs
- Detecting resource leaks (circular references)
- Functional programming: detecting periodic sequences

**4. Two Pointer Technique**

This is a specific application of the two-pointer pattern:
- Fast and slow pointers
- Left and right pointers (binary search, two sum)
- Start and end pointers (sliding window)`,
	order: 2,
	translations: {
		ru: {
			title: 'Цикл в связном списке',
			description: `Определите, есть ли цикл в связном списке.

**Задача:**

Дана голова связного списка, определите, есть ли в списке цикл.

Цикл существует, если какой-то узел можно достичь снова, непрерывно следуя указателю \`next\`.

**Примеры:**

\`\`\`
Вход: 3 -> 2 -> 0 -> -4 -> (назад к 2)
Выход: true
Объяснение: Узел со значением -4 соединяется с узлом со значением 2

Вход: 1 -> 2 -> (назад к 1)
Выход: true

Вход: 1 -> nil
Выход: false
\`\`\`

**Алгоритм Флойда (Черепаха и Заяц):**

1. Используйте два указателя: slow (движется на 1 шаг) и fast (движется на 2 шага)
2. Если есть цикл, fast в конце концов встретит slow
3. Если fast достигнет nil, цикла нет

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Используйте два указателя: slow двигается на один шаг за раз, fast двигается на два шага. Инициализируйте оба на head и проверяйте, указывают ли они когда-нибудь на один узел.`,
			hint2: `Условие цикла должно быть 'fast and fast.next' для безопасного доступа к fast.next.next. Если fast достигает None, цикла нет.`,
			whyItMatters: `Алгоритм обнаружения цикла Флойда - фундаментальный алгоритм с множеством применений.

**Почему это важно:**

**1. Алгоритм Флойда**

Это элегантное решение с O(1) памяти - классика.

**2. Расширения**

Обнаружив цикл, вы можете:
- Найти начало цикла
- Найти длину цикла
- Удалить цикл`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Определяет, есть ли цикл в связном списке.
    Использует алгоритм обнаружения цикла Флойда.

    Args:
        head: Голова связного списка

    Returns:
        True если цикл существует, иначе False
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next       # Один шаг
        fast = fast.next.next  # Два шага

        if slow == fast:
            return True  # Цикл обнаружен

    return False  # Нет цикла`
		},
		uz: {
			title: 'Bog\'langan ro\'yxatda tsikl',
			description: `Bog'langan ro'yxatda tsikl bor yoki yo'qligini aniqlang.

**Masala:**

Bog'langan ro'yxatning boshi berilgan, ro'yxatda tsikl bor yoki yo'qligini aniqlang.

\`next\` ko'rsatkichiga doimiy ergashish orqali ba'zi tugunga qayta etish mumkin bo'lsa, tsikl mavjud.

**Misollar:**

\`\`\`
Kirish: 3 -> 2 -> 0 -> -4 -> (2 ga qaytish)
Chiqish: true
Tushuntirish: -4 qiymatli tugun 2 qiymatli tugunga ulanadi

Kirish: 1 -> nil
Chiqish: false
\`\`\`

**Floyd algoritmi (Toshbaqa va Quyon):**

1. Ikki ko'rsatkichdan foydalaning: slow (1 qadam) va fast (2 qadam)
2. Agar tsikl bo'lsa, fast oxir-oqibat slow ni uchratadi
3. Agar fast nil ga yetsa, tsikl yo'q

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikki ko'rsatkichdan foydalaning: slow bir vaqtda bir qadam, fast ikki qadam harakatlanadi. Ikkalasini ham head ga ishga tushiring va ular bir xil tugunga ko'rsatadimi tekshiring.`,
			hint2: `Tsikl sharti 'fast and fast.next' bo'lishi kerak fast.next.next ga xavfsiz kirish uchun. Agar fast None ga yetsa, tsikl yo'q.`,
			whyItMatters: `Floyd tsikl aniqlash algoritmi ko'p qo'llanmalarga ega asosiy algoritm.

**Bu nima uchun muhim:**

**1. Floyd algoritmi**

Bu O(1) xotirali nafis yechim klassik.

**2. Kengaytmalar**

Tsiklni aniqlagandan keyin siz:
- Tsikl boshini topishingiz
- Tsikl uzunligini topishingiz
- Tsiklni olib tashlashingiz mumkin`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Bog'langan ro'yxatda tsikl bor yoki yo'qligini aniqlaydi.
    Floyd tsikl aniqlash algoritmidan foydalanadi.

    Args:
        head: Bog'langan ro'yxatning boshi

    Returns:
        Agar tsikl mavjud bo'lsa True, aks holda False
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next       # Bir qadam
        fast = fast.next.next  # Ikki qadam

        if slow == fast:
            return True  # Tsikl aniqlandi

    return False  # Tsikl yo'q`
		}
	}
};

export default task;
