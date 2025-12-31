import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-remove-nth-from-end',
	title: 'Remove Nth Node From End',
	difficulty: 'medium',
	tags: ['python', 'linked-list', 'two-pointers'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Remove the nth node from the end of a linked list.

**Problem:**

Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Examples:**

\`\`\`
Input: head = 1 -> 2 -> 3 -> 4 -> 5, n = 2
Output: 1 -> 2 -> 3 -> 5
Explanation: Remove 4 (2nd from end)

Input: head = 1, n = 1
Output: nil
Explanation: Remove only node

Input: head = 1 -> 2, n = 1
Output: 1
Explanation: Remove 2 (last node)
\`\`\`

**Two Pointer Approach (One Pass):**

1. Create dummy node before head
2. Move fast pointer n+1 steps ahead
3. Move both pointers until fast reaches end
4. slow.Next = slow.Next.Next (skip target node)

**Why n+1:**
We need slow to point to the node BEFORE the one to remove.

**Time Complexity:** O(n) - single pass
**Space Complexity:** O(1)`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    # TODO: Remove the nth node from end and return new head

    return None`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Remove nth node from end of list.
    Returns the head of modified list.

    Args:
        head: Head of the linked list
        n: Position from end to remove

    Returns:
        Head of modified list
    """
    # Dummy node handles edge case of removing head
    dummy = ListNode(next=head)

    slow = dummy
    fast = dummy

    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    # Remove the nth node
    slow.next = slow.next.next

    return dummy.next`,
	testCode: `import pytest
from typing import List, Optional
from solution import ListNode, remove_nth_from_end


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


def list_to_array(head: Optional[ListNode]) -> List[int]:
    """Helper to convert linked list to list of values."""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


class TestRemoveNthFromEnd:
    def test_middle(self):
        """Test removing from middle"""
        head = create_list([1, 2, 3, 4, 5])
        result = remove_nth_from_end(head, 2)
        assert list_to_array(result) == [1, 2, 3, 5]

    def test_single(self):
        """Test removing only node"""
        head = create_list([1])
        result = remove_nth_from_end(head, 1)
        assert list_to_array(result) == []

    def test_last(self):
        """Test removing last node"""
        head = create_list([1, 2])
        result = remove_nth_from_end(head, 1)
        assert list_to_array(result) == [1]

    def test_first(self):
        """Test removing first node"""
        head = create_list([1, 2])
        result = remove_nth_from_end(head, 2)
        assert list_to_array(result) == [2]

    def test_second_of_three(self):
        """Test removing second of three"""
        head = create_list([1, 2, 3])
        result = remove_nth_from_end(head, 2)
        assert list_to_array(result) == [1, 3]

    def test_first_of_five(self):
        """Test removing first node of five"""
        head = create_list([1, 2, 3, 4, 5])
        result = remove_nth_from_end(head, 5)
        assert list_to_array(result) == [2, 3, 4, 5]

    def test_third_of_five(self):
        """Test removing third from end of five"""
        head = create_list([1, 2, 3, 4, 5])
        result = remove_nth_from_end(head, 3)
        assert list_to_array(result) == [1, 2, 4, 5]

    def test_last_of_five(self):
        """Test removing last node of five"""
        head = create_list([1, 2, 3, 4, 5])
        result = remove_nth_from_end(head, 1)
        assert list_to_array(result) == [1, 2, 3, 4]

    def test_first_of_two(self):
        """Test removing first of two nodes"""
        head = create_list([1, 2])
        result = remove_nth_from_end(head, 2)
        assert list_to_array(result) == [2]

    def test_middle_of_seven(self):
        """Test removing middle node of seven"""
        head = create_list([1, 2, 3, 4, 5, 6, 7])
        result = remove_nth_from_end(head, 4)
        assert list_to_array(result) == [1, 2, 3, 5, 6, 7]`,
	hint1: `Use a dummy node before head to handle the case where you need to remove the first node. Move the fast pointer n+1 steps ahead first.`,
	hint2: `When fast pointer reaches None, the slow pointer will be at the node BEFORE the one you need to remove. Then simply do slow.next = slow.next.next.`,
	whyItMatters: `This problem teaches the two-pointer gap technique.

**Why This Matters:**

**1. Gap Technique**

Creating a fixed gap between pointers is powerful:
\`\`\`python
# Gap of n nodes between fast and slow
# When fast reaches end, slow is n nodes behind

# Applications:
# - Find nth from end
# - Find middle (gap = half distance)
# - Rotate list by k
\`\`\`

**2. One-Pass Solution**

Two-pass solution is easier but slower:
\`\`\`python
# Two-pass: Find length, then find node
def remove_nth_two_pass(head: ListNode, n: int) -> ListNode:
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next
    # Now traverse to (length - n)th node

# One-pass: Use gap technique
# More elegant, same time complexity but single traversal
\`\`\`

**3. Dummy Node Pattern**

Without dummy, removing first node is a special case:
\`\`\`python
# Without dummy
if length_from_end == n:
    return head.next  # Special case

# With dummy: uniform handling
dummy = ListNode(next=head)
# ... normal logic
return dummy.next
\`\`\`

**4. Off-by-One Prevention**

This problem teaches careful counting:
- Why n+1 steps? So slow points BEFORE target
- Edge case: removing first node
- Edge case: removing last node`,
	order: 4,
	translations: {
		ru: {
			title: 'Удалить N-й узел с конца',
			description: `Удалите n-й узел с конца связного списка.

**Задача:**

Дана голова связного списка, удалите n-й узел с конца списка и верните его голову.

**Примеры:**

\`\`\`
Вход: head = 1 -> 2 -> 3 -> 4 -> 5, n = 2
Выход: 1 -> 2 -> 3 -> 5
Объяснение: Удалить 4 (2-й с конца)

Вход: head = 1, n = 1
Выход: nil
Объяснение: Удалить единственный узел
\`\`\`

**Подход двух указателей (один проход):**

1. Создайте фиктивный узел перед head
2. Продвиньте fast на n+1 шагов вперёд
3. Двигайте оба указателя пока fast не достигнет конца
4. slow.Next = slow.Next.Next (пропустить целевой узел)

**Временная сложность:** O(n) - один проход
**Пространственная сложность:** O(1)`,
			hint1: `Используйте фиктивный узел перед head для обработки случая, когда нужно удалить первый узел. Сначала продвиньте fast на n+1 шагов.`,
			hint2: `Когда fast достигает None, slow будет на узле ПЕРЕД тем, который нужно удалить. Затем просто сделайте slow.next = slow.next.next.`,
			whyItMatters: `Эта задача учит технике двух указателей с промежутком.

**Почему это важно:**

**1. Техника промежутка**

Создание фиксированного промежутка между указателями мощно.

**2. Решение за один проход**

Более элегантное, чем двухпроходное.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Удаляет n-й узел с конца списка.
    Возвращает голову изменённого списка.

    Args:
        head: Голова связного списка
        n: Позиция с конца для удаления

    Returns:
        Голова изменённого списка
    """
    # Фиктивный узел обрабатывает случай удаления головы
    dummy = ListNode(next=head)

    slow = dummy
    fast = dummy

    # Двигаем fast на n+1 шагов вперёд
    for _ in range(n + 1):
        fast = fast.next

    # Двигаем оба пока fast не достигнет конца
    while fast:
        slow = slow.next
        fast = fast.next

    # Удаляем n-й узел
    slow.next = slow.next.next

    return dummy.next`
		},
		uz: {
			title: 'Oxiridan N-chi tugunni olib tashlash',
			description: `Bog'langan ro'yxatning oxiridan n-chi tugunni olib tashlang.

**Masala:**

Bog'langan ro'yxatning boshi berilgan, ro'yxat oxiridan n-chi tugunni olib tashlang va boshini qaytaring.

**Misollar:**

\`\`\`
Kirish: head = 1 -> 2 -> 3 -> 4 -> 5, n = 2
Chiqish: 1 -> 2 -> 3 -> 5
Tushuntirish: 4 ni olib tashlash (oxiridan 2-chi)

Kirish: head = 1, n = 1
Chiqish: nil
Tushuntirish: Yagona tugunni olib tashlash
\`\`\`

**Ikki ko'rsatkich yondashuvi (bir o'tish):**

1. head oldidan dummy tugun yarating
2. fast ni n+1 qadam oldinga siljiting
3. fast oxiriga yetguncha ikkala ko'rsatkichni siljiting
4. slow.Next = slow.Next.Next (maqsad tugunni o'tkazib yuborish)

**Vaqt murakkabligi:** O(n) - bir o'tish
**Xotira murakkabligi:** O(1)`,
			hint1: `Birinchi tugunni olib tashlash kerak bo'lgan holatni qayta ishlash uchun head oldida dummy tugundan foydalaning. Avval fast ni n+1 qadam oldinga siljiting.`,
			hint2: `Fast ko'rsatkich None ga yetganda, slow ko'rsatkich olib tashlash kerak bo'lgan tugundan OLDINGI tugunga keladi. Keyin shunchaki slow.next = slow.next.next qiling.`,
			whyItMatters: `Bu masala oraliq texnikasi bilan ikki ko'rsatkich usulini o'rgatadi.

**Bu nima uchun muhim:**

**1. Oraliq texnikasi**

Ko'rsatkichlar orasida belgilangan oraliq yaratish kuchli.

**2. Bir o'tishli yechim**

Ikki o'tishli yechimdan ko'ra nafisroq.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Ro'yxat oxiridan n-chi tugunni olib tashlaydi.
    O'zgartirilgan ro'yxatning boshini qaytaradi.

    Args:
        head: Bog'langan ro'yxatning boshi
        n: Oxiridan olib tashlash pozitsiyasi

    Returns:
        O'zgartirilgan ro'yxatning boshi
    """
    # Dummy tugun boshni olib tashlash holatini qayta ishlaydi
    dummy = ListNode(next=head)

    slow = dummy
    fast = dummy

    # fast ni n+1 qadam oldinga siljitamiz
    for _ in range(n + 1):
        fast = fast.next

    # fast oxiriga yetguncha ikkalsini siljitamiz
    while fast:
        slow = slow.next
        fast = fast.next

    # n-chi tugunni olib tashlaymiz
    slow.next = slow.next.next

    return dummy.next`
		}
	}
};

export default task;
