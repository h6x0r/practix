import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-reverse-linked-list',
	title: 'Reverse Linked List',
	difficulty: 'easy',
	tags: ['python', 'linked-list', 'pointers'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Reverse a singly linked list.

**Problem:**

Given the head of a singly linked list, reverse the list and return the new head.

**Examples:**

\`\`\`
Input: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Output: 5 -> 4 -> 3 -> 2 -> 1 -> nil

Input: 1 -> 2 -> nil
Output: 2 -> 1 -> nil

Input: nil
Output: nil
\`\`\`

**Algorithm (Iterative):**

1. Track three pointers: prev, current, next
2. For each node:
   - Save next node
   - Reverse pointer (current.Next = prev)
   - Move prev and current forward
3. Return prev (new head)

\`\`\`
Initial:  nil   1 -> 2 -> 3 -> nil
          prev  curr

Step 1:   nil <- 1    2 -> 3 -> nil
          prev  curr  next

Step 2:   nil <- 1 <- 2    3 -> nil
               prev  curr  next
\`\`\`

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    # TODO: Reverse the linked list and return new head

    return None`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a singly linked list.

    Args:
        head: Head of the linked list

    Returns:
        New head of the reversed list
    """
    prev = None
    current = head

    while current is not None:
        # Save next node
        next_node = current.next

        # Reverse the pointer
        current.next = prev

        # Move prev and current forward
        prev = current
        current = next_node

    # prev is now the new head
    return prev`,
	testCode: `import pytest
from typing import List, Optional
from solution import ListNode, reverse_list


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


class TestReverseList:
    def test_basic(self):
        """Test basic reversal"""
        head = create_list([1, 2, 3, 4, 5])
        result = reverse_list(head)
        assert list_to_array(result) == [5, 4, 3, 2, 1]

    def test_two(self):
        """Test two-element list"""
        head = create_list([1, 2])
        result = reverse_list(head)
        assert list_to_array(result) == [2, 1]

    def test_single(self):
        """Test single element"""
        head = create_list([1])
        result = reverse_list(head)
        assert list_to_array(result) == [1]

    def test_empty(self):
        """Test empty list"""
        result = reverse_list(None)
        assert list_to_array(result) == []

    def test_three(self):
        """Test three-element list"""
        head = create_list([1, 2, 3])
        result = reverse_list(head)
        assert list_to_array(result) == [3, 2, 1]

    def test_even_length(self):
        """Test even length list"""
        head = create_list([10, 20, 30, 40])
        result = reverse_list(head)
        assert list_to_array(result) == [40, 30, 20, 10]

    def test_large_list(self):
        """Test larger list"""
        head = create_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = reverse_list(head)
        assert list_to_array(result) == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    def test_negative_values(self):
        """Test list with negative values"""
        head = create_list([-1, -2, -3, -4])
        result = reverse_list(head)
        assert list_to_array(result) == [-4, -3, -2, -1]

    def test_duplicate_values(self):
        """Test list with duplicate values"""
        head = create_list([5, 5, 5, 5])
        result = reverse_list(head)
        assert list_to_array(result) == [5, 5, 5, 5]

    def test_mixed_values(self):
        """Test list with mixed positive and negative values"""
        head = create_list([1, -2, 3, -4, 5])
        result = reverse_list(head)
        assert list_to_array(result) == [5, -4, 3, -2, 1]`,
	hint1: `Use three variables: prev (initially None), current (initially head), and next_node (temporary). In each iteration, save next_node, reverse the pointer, then advance prev and current.`,
	hint2: `The key insight is that after reversing, the original head becomes the tail (points to None), and the original tail becomes the new head (returned from the function).`,
	whyItMatters: `Reversing a linked list is a fundamental operation that appears in many problems.

**Why This Matters:**

**1. Foundation for Other Problems**

Many problems build on reversal:
- Reverse linked list II (reverse portion)
- Palindrome linked list (reverse half)
- Reorder list (reverse second half)
- Reverse nodes in k-group

**2. Reference Manipulation**

This teaches core reference skills:
\`\`\`python
# The three-variable dance
next_node = current.next  # Save
current.next = prev       # Reverse
prev = current            # Advance prev
current = next_node       # Advance current
\`\`\`

**3. Iterative vs Recursive**

\`\`\`python
# Iterative: O(1) space, shown above

# Recursive: O(n) space (stack)
def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
\`\`\`

**4. Interview Classic**

This is often the first linked list question asked because:
- Tests reference manipulation
- Simple to state, tricky to implement correctly
- Natural follow-ups available`,
	order: 1,
	translations: {
		ru: {
			title: 'Развернуть связный список',
			description: `Разверните односвязный список.

**Задача:**

Дана голова односвязного списка, разверните список и верните новую голову.

**Примеры:**

\`\`\`
Вход: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Выход: 5 -> 4 -> 3 -> 2 -> 1 -> nil

Вход: 1 -> 2 -> nil
Выход: 2 -> 1 -> nil
\`\`\`

**Алгоритм (итеративный):**

1. Отслеживайте три указателя: prev, current, next
2. Для каждого узла:
   - Сохраните следующий узел
   - Разверните указатель (current.Next = prev)
   - Продвиньте prev и current вперёд
3. Верните prev (новая голова)

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Используйте три переменные: prev (изначально None), current (изначально head) и next_node (временная). В каждой итерации сохраните next_node, разверните ссылку, затем продвиньте prev и current.`,
			hint2: `Ключевая идея: после разворота исходная голова становится хвостом (указывает на None), а исходный хвост становится новой головой (возвращается из функции).`,
			whyItMatters: `Разворот связного списка - фундаментальная операция, которая появляется во многих задачах.

**Почему это важно:**

**1. Основа для других задач**

Многие задачи строятся на развороте:
- Reverse linked list II
- Palindrome linked list
- Reorder list

**2. Манипуляция указателями**

Это учит основным навыкам работы с указателями.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Разворачивает односвязный список.

    Args:
        head: Голова связного списка

    Returns:
        Новая голова развёрнутого списка
    """
    prev = None
    current = head

    while current is not None:
        # Сохраняем следующий узел
        next_node = current.next

        # Разворачиваем ссылку
        current.next = prev

        # Двигаем prev и current вперёд
        prev = current
        current = next_node

    # prev теперь новая голова
    return prev`
		},
		uz: {
			title: 'Bog\'langan ro\'yxatni teskari aylantirish',
			description: `Bir tomonlama bog'langan ro'yxatni teskari aylantiring.

**Masala:**

Bir tomonlama bog'langan ro'yxatning boshi berilgan, ro'yxatni teskari aylantiring va yangi boshini qaytaring.

**Misollar:**

\`\`\`
Kirish: 1 -> 2 -> 3 -> 4 -> 5 -> nil
Chiqish: 5 -> 4 -> 3 -> 2 -> 1 -> nil

Kirish: 1 -> 2 -> nil
Chiqish: 2 -> 1 -> nil
\`\`\`

**Algoritm (iterativ):**

1. Uchta ko'rsatkichni kuzating: prev, current, next
2. Har bir tugun uchun:
   - Keyingi tugunni saqlang
   - Ko'rsatkichni teskari aylantiring (current.Next = prev)
   - prev va current ni oldinga siljiting
3. prev ni qaytaring (yangi bosh)

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Uchta o'zgaruvchidan foydalaning: prev (dastlab None), current (dastlab head) va next_node (vaqtinchalik). Har bir iteratsiyada next_node ni saqlang, havolani teskari aylantiring, keyin prev va current ni oldinga siljiting.`,
			hint2: `Asosiy tushuncha: teskari aylantirgandan keyin asl bosh dum bo'ladi (None ga ko'rsatadi), asl dum esa yangi bosh bo'ladi (funktsiyadan qaytariladi).`,
			whyItMatters: `Bog'langan ro'yxatni teskari aylantirish ko'p masalalarda uchraydigan asosiy operatsiya.

**Bu nima uchun muhim:**

**1. Boshqa masalalar uchun asos**

Ko'p masalalar teskari aylantirishga asoslanadi.

**2. Ko'rsatkich manipulyatsiyasi**

Bu asosiy ko'rsatkich ko'nikmalarini o'rgatadi.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Bir tomonlama bog'langan ro'yxatni teskari aylantiradi.

    Args:
        head: Bog'langan ro'yxatning boshi

    Returns:
        Teskari aylantirilgan ro'yxatning yangi boshi
    """
    prev = None
    current = head

    while current is not None:
        # Keyingi tugunni saqlaymiz
        next_node = current.next

        # Havolani teskari aylantiramiz
        current.next = prev

        # prev va current ni oldinga siljitamiz
        prev = current
        current = next_node

    # prev endi yangi bosh
    return prev`
		}
	}
};

export default task;
