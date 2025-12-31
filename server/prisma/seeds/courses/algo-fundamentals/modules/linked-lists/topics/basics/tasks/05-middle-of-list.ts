import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-middle-of-list',
	title: 'Middle of Linked List',
	difficulty: 'easy',
	tags: ['python', 'linked-list', 'two-pointers'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the middle node of a linked list.

**Problem:**

Given the head of a singly linked list, return the middle node. If there are two middle nodes, return the second middle node.

**Examples:**

\`\`\`
Input: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Output: Node with value 3
Explanation: 3 is the middle node

Input: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> nil
Output: Node with value 4
Explanation: Two middles (3 and 4), return second (4)

Input: 1 -> nil
Output: Node with value 1
\`\`\`

**Fast and Slow Pointer Approach:**

1. Initialize slow and fast to head
2. Move slow 1 step, fast 2 steps
3. When fast reaches end, slow is at middle

**Why it works:**

Fast moves twice as fast, so when fast reaches the end, slow has traveled half the distance.

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    # TODO: Return the middle node of the linked list

    return None`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Return the middle node of linked list.
    If two middles exist, returns the second one.

    Args:
        head: Head of the linked list

    Returns:
        Middle node of the list
    """
    slow = head
    fast = head

    # Move fast twice as fast as slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # slow is now at middle
    return slow`,
	testCode: `import pytest
from typing import List, Optional
from solution import ListNode, middle_node


def create_list(vals: List[int]) -> Optional[ListNode]:
    """Helper to create linked list from list of values."""
    if not vals:
        return None
    head = ListNode(vals[0])
    current = head
    for val in vals[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


class TestMiddleNode:
    def test_odd(self):
        """Test odd length list"""
        head = create_list([1, 2, 3, 4, 5])
        result = middle_node(head)
        assert result is not None
        assert result.val == 3

    def test_even(self):
        """Test even length list (return second middle)"""
        head = create_list([1, 2, 3, 4, 5, 6])
        result = middle_node(head)
        assert result is not None
        assert result.val == 4

    def test_single(self):
        """Test single element"""
        head = create_list([1])
        result = middle_node(head)
        assert result is not None
        assert result.val == 1

    def test_two(self):
        """Test two elements"""
        head = create_list([1, 2])
        result = middle_node(head)
        assert result is not None
        assert result.val == 2

    def test_three(self):
        """Test three elements"""
        head = create_list([1, 2, 3])
        result = middle_node(head)
        assert result is not None
        assert result.val == 2

    def test_four(self):
        """Test four elements"""
        head = create_list([1, 2, 3, 4])
        result = middle_node(head)
        assert result is not None
        assert result.val == 3

    def test_seven(self):
        """Test seven elements (odd)"""
        head = create_list([1, 2, 3, 4, 5, 6, 7])
        result = middle_node(head)
        assert result is not None
        assert result.val == 4

    def test_eight(self):
        """Test eight elements (even)"""
        head = create_list([1, 2, 3, 4, 5, 6, 7, 8])
        result = middle_node(head)
        assert result is not None
        assert result.val == 5

    def test_nine(self):
        """Test nine elements (odd)"""
        head = create_list([10, 20, 30, 40, 50, 60, 70, 80, 90])
        result = middle_node(head)
        assert result is not None
        assert result.val == 50

    def test_ten(self):
        """Test ten elements (even)"""
        head = create_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = middle_node(head)
        assert result is not None
        assert result.val == 6`,
	hint1: `Use two pointers: slow moves one step at a time, fast moves two steps at a time. Start both at head.`,
	hint2: `The loop condition 'fast and fast.next' handles both odd and even length lists correctly. When fast can't move anymore, slow is at the middle.`,
	whyItMatters: `Finding the middle is a building block for many linked list algorithms.

**Why This Matters:**

**1. Foundation for Other Algorithms**

Finding middle is essential for:
- Merge sort (split list in half)
- Palindrome check (reverse second half)
- Reorder list (interleave halves)

\`\`\`python
def sort_list(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    mid = middle_node(head)
    # ... split and merge sort
\`\`\`

**2. Fast/Slow Pointer Pattern**

This is a key application of fast/slow pointers:
\`\`\`python
# Finding middle: slow = n/2 when fast = n
# Finding cycle: fast catches slow if cycle exists
# Finding nth from end: gap of n between pointers
\`\`\`

**3. Even vs Odd Length**

Understanding the behavior:
\`\`\`
Odd:  1 -> 2 -> 3 -> 4 -> 5
                ^
              middle (one middle)

Even: 1 -> 2 -> 3 -> 4 -> 5 -> 6
                     ^
               second middle (two middles)
\`\`\`

**4. Time/Space Trade-off**

\`\`\`python
# Two-pass: find length, then traverse to middle
# O(n) time, O(1) space, two traversals

# One-pass: fast/slow pointers
# O(n) time, O(1) space, single traversal
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Середина связного списка',
			description: `Найдите средний узел связного списка.

**Задача:**

Дана голова односвязного списка, верните средний узел. Если есть два средних узла, верните второй.

**Примеры:**

\`\`\`
Вход: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Выход: Узел со значением 3
Объяснение: 3 - средний узел

Вход: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> nil
Выход: Узел со значением 4
Объяснение: Два средних (3 и 4), вернуть второй (4)
\`\`\`

**Подход быстрого и медленного указателей:**

1. Инициализируйте slow и fast на head
2. Двигайте slow на 1 шаг, fast на 2 шага
3. Когда fast достигнет конца, slow будет в середине

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Используйте два указателя: slow двигается на один шаг за раз, fast на два шага. Начните оба с head.`,
			hint2: `Условие цикла 'fast and fast.next' правильно обрабатывает списки нечётной и чётной длины. Когда fast больше не может двигаться, slow в середине.`,
			whyItMatters: `Нахождение середины - строительный блок для многих алгоритмов связных списков.

**Почему это важно:**

**1. Основа для других алгоритмов**

Нахождение середины необходимо для:
- Merge sort (разделить список пополам)
- Проверка палиндрома (развернуть вторую половину)
- Переупорядочить список`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Возвращает средний узел связного списка.
    Если есть два средних, возвращает второй.

    Args:
        head: Голова связного списка

    Returns:
        Средний узел списка
    """
    slow = head
    fast = head

    # Двигаем fast вдвое быстрее slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # slow теперь в середине
    return slow`
		},
		uz: {
			title: 'Bog\'langan ro\'yxat o\'rtasi',
			description: `Bog'langan ro'yxatning o'rta tugunini toping.

**Masala:**

Bir tomonlama bog'langan ro'yxatning boshi berilgan, o'rta tugunni qaytaring. Agar ikkita o'rta tugun bo'lsa, ikkinchisini qaytaring.

**Misollar:**

\`\`\`
Kirish: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Chiqish: 3 qiymatli tugun
Tushuntirish: 3 o'rta tugun

Kirish: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> nil
Chiqish: 4 qiymatli tugun
Tushuntirish: Ikki o'rta (3 va 4), ikkinchisini qaytarish (4)
\`\`\`

**Tez va sekin ko'rsatkich yondashuvi:**

1. slow va fast ni head ga ishga tushiring
2. slow ni 1 qadam, fast ni 2 qadam siljiting
3. fast oxiriga yetganda, slow o'rtada bo'ladi

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikki ko'rsatkichdan foydalaning: slow bir vaqtda bir qadam, fast ikki qadam harakatlanadi. Ikkalasini ham head dan boshlang.`,
			hint2: `'fast and fast.next' tsikl sharti toq va juft uzunlikdagi ro'yxatlarni to'g'ri qayta ishlaydi. fast endi harakatlana olmasa, slow o'rtada.`,
			whyItMatters: `O'rtani topish ko'plab bog'langan ro'yxat algoritmlari uchun qurilish bloki.

**Bu nima uchun muhim:**

**1. Boshqa algoritmlar uchun asos**

O'rtani topish quyidagilar uchun muhim:
- Merge sort (ro'yxatni ikkiga bo'lish)
- Palindrom tekshirish
- Ro'yxatni qayta tartibga solish`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Bog'langan ro'yxatning o'rta tugunini qaytaradi.
    Agar ikkita o'rta bo'lsa, ikkinchisini qaytaradi.

    Args:
        head: Bog'langan ro'yxatning boshi

    Returns:
        Ro'yxatning o'rta tuguni
    """
    slow = head
    fast = head

    # fast ni slow dan ikki baravar tez siljitamiz
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # slow endi o'rtada
    return slow`
		}
	}
};

export default task;
