import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-palindrome-list',
	title: 'Palindrome Linked List',
	difficulty: 'medium',
	tags: ['python', 'linked-list', 'two-pointers', 'reverse'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Check if a linked list is a palindrome.

**Problem:**

Given the head of a singly linked list, return \`true\` if it is a palindrome, or \`false\` otherwise.

**Examples:**

\`\`\`
Input: 1 -> 2 -> 2 -> 1 -> nil
Output: true

Input: 1 -> 2 -> nil
Output: false

Input: 1 -> 2 -> 3 -> 2 -> 1 -> nil
Output: true
\`\`\`

**Algorithm (O(1) space):**

1. Find the middle of the list
2. Reverse the second half
3. Compare first half with reversed second half
4. (Optional) Restore the list

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def is_palindrome(head: Optional[ListNode]) -> bool:
    # TODO: Check if the linked list is a palindrome

    return False`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Check if linked list is a palindrome.

    Args:
        head: Head of the linked list

    Returns:
        True if palindrome, False otherwise
    """
    if not head or not head.next:
        return True

    # Find middle of the list
    mid = find_middle(head)

    # Reverse second half
    second_half = reverse_list(mid)

    # Compare first half with reversed second half
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Return middle node (second middle for even length)."""
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse a linked list."""
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev`,
	testCode: `import pytest
from typing import List, Optional
from solution import ListNode, is_palindrome


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


class TestIsPalindrome:
    def test_even_true(self):
        """Test even length palindrome"""
        head = create_list([1, 2, 2, 1])
        assert is_palindrome(head) == True

    def test_odd_true(self):
        """Test odd length palindrome"""
        head = create_list([1, 2, 3, 2, 1])
        assert is_palindrome(head) == True

    def test_false(self):
        """Test not palindrome"""
        head = create_list([1, 2])
        assert is_palindrome(head) == False

    def test_single(self):
        """Test single element"""
        head = create_list([1])
        assert is_palindrome(head) == True

    def test_empty(self):
        """Test empty list"""
        assert is_palindrome(None) == True

    def test_three_true(self):
        """Test three element palindrome"""
        head = create_list([1, 2, 1])
        assert is_palindrome(head) == True

    def test_three_false(self):
        """Test three element not palindrome"""
        head = create_list([1, 2, 3])
        assert is_palindrome(head) == False

    def test_four_false(self):
        """Test four element not palindrome"""
        head = create_list([1, 2, 3, 4])
        assert is_palindrome(head) == False

    def test_two_same(self):
        """Test two element palindrome with same values"""
        head = create_list([5, 5])
        assert is_palindrome(head) == True

    def test_long_palindrome(self):
        """Test longer palindrome"""
        head = create_list([1, 2, 3, 4, 5, 4, 3, 2, 1])
        assert is_palindrome(head) == True`,
	hint1: `First find the middle of the list using slow/fast pointers. For even-length lists, this gives you the start of the second half.`,
	hint2: `Reverse the second half of the list starting from the middle. Then compare values from the start of the list and the start of the reversed second half.`,
	whyItMatters: `This problem combines multiple linked list techniques.

**Why This Matters:**

**1. Combining Techniques**

This problem requires:
- Finding middle (fast/slow pointers)
- Reversing a list
- Comparing lists

Each is useful on its own, and combining them shows mastery.

**2. Space Optimization**

\`\`\`python
# O(n) space: Copy to array and check
def is_palindrome_naive(head: ListNode) -> bool:
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    # Check if vals is palindrome
    return vals == vals[::-1]

# O(1) space: Reverse in place
def is_palindrome_optimal(head: ListNode) -> bool:
    mid = find_middle(head)
    second_half = reverse_list(mid)
    # Compare...
\`\`\`

**3. Trade-offs**

The O(1) space solution modifies the list:
\`\`\`python
# Before: 1 -> 2 -> 2 -> 1
# After:  1 -> 2 -> 1 -> 2
# The list structure is changed!

# If you need to preserve the list,
# reverse the second half again after checking
\`\`\`

**4. Real-World Application**

- Data validation (checking symmetric structures)
- Undo/redo systems (stack-based palindrome checking)
- DNA sequence analysis (palindromic sequences)`,
	order: 6,
	translations: {
		ru: {
			title: 'Палиндром в связном списке',
			description: `Проверьте, является ли связный список палиндромом.

**Задача:**

Дана голова односвязного списка, верните \`true\`, если он является палиндромом, иначе \`false\`.

**Примеры:**

\`\`\`
Вход: 1 -> 2 -> 2 -> 1 -> nil
Выход: true

Вход: 1 -> 2 -> nil
Выход: false

Вход: 1 -> 2 -> 3 -> 2 -> 1 -> nil
Выход: true
\`\`\`

**Алгоритм (O(1) память):**

1. Найдите середину списка
2. Разверните вторую половину
3. Сравните первую половину с развёрнутой второй
4. (Опционально) Восстановите список

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Сначала найдите середину списка используя slow/fast указатели. Для списков чётной длины это даст вам начало второй половины.`,
			hint2: `Разверните вторую половину списка начиная с середины. Затем сравните значения от начала списка и от начала развёрнутой второй половины.`,
			whyItMatters: `Эта задача комбинирует несколько техник работы со связными списками.

**Почему это важно:**

**1. Комбинирование техник**

Эта задача требует:
- Нахождение середины
- Разворот списка
- Сравнение списков

**2. Оптимизация памяти**

O(1) память вместо O(n) для копирования в массив.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Проверяет, является ли связный список палиндромом.

    Args:
        head: Голова связного списка

    Returns:
        True если палиндром, иначе False
    """
    if not head or not head.next:
        return True

    # Находим середину списка
    mid = find_middle(head)

    # Разворачиваем вторую половину
    second_half = reverse_list(mid)

    # Сравниваем первую половину с развёрнутой второй
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Возвращает средний узел."""
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Разворачивает связный список."""
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev`
		},
		uz: {
			title: 'Bog\'langan ro\'yxatda palindrom',
			description: `Bog'langan ro'yxat palindrom ekanligini tekshiring.

**Masala:**

Bir tomonlama bog'langan ro'yxatning boshi berilgan, u palindrom bo'lsa \`true\`, aks holda \`false\` qaytaring.

**Misollar:**

\`\`\`
Kirish: 1 -> 2 -> 2 -> 1 -> nil
Chiqish: true

Kirish: 1 -> 2 -> nil
Chiqish: false

Kirish: 1 -> 2 -> 3 -> 2 -> 1 -> nil
Chiqish: true
\`\`\`

**Algoritm (O(1) xotira):**

1. Ro'yxatning o'rtasini toping
2. Ikkinchi yarmini teskari aylantiring
3. Birinchi yarmini teskari aylantirilgan ikkinchi yarmi bilan solishtiring
4. (Ixtiyoriy) Ro'yxatni tiklang

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Avval slow/fast ko'rsatkichlar yordamida ro'yxatning o'rtasini toping. Juft uzunlikdagi ro'yxatlar uchun bu sizga ikkinchi yarmining boshini beradi.`,
			hint2: `Ro'yxatning ikkinchi yarmini o'rtadan boshlab teskari aylantiring. Keyin ro'yxat boshidan va teskari aylantirilgan ikkinchi yarmi boshidan qiymatlarni solishtiring.`,
			whyItMatters: `Bu masala bir nechta bog'langan ro'yxat texnikalarini birlashtiradi.

**Bu nima uchun muhim:**

**1. Texnikalarni birlashtirish**

Bu masala quyidagilarni talab qiladi:
- O'rtani topish
- Ro'yxatni teskari aylantirish
- Ro'yxatlarni solishtirish

**2. Xotira optimizatsiyasi**

Massivga nusxalash uchun O(n) o'rniga O(1) xotira.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Bog'langan ro'yxat palindrom ekanligini tekshiradi.

    Args:
        head: Bog'langan ro'yxatning boshi

    Returns:
        Palindrom bo'lsa True, aks holda False
    """
    if not head or not head.next:
        return True

    # Ro'yxatning o'rtasini topamiz
    mid = find_middle(head)

    # Ikkinchi yarmini teskari aylantiramiz
    second_half = reverse_list(mid)

    # Birinchi yarmini teskari aylantirilgan ikkinchi yarmi bilan solishtiramiz
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """O'rta tugunni qaytaradi."""
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Bog'langan ro'yxatni teskari aylantiradi."""
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev`
		}
	}
};

export default task;
