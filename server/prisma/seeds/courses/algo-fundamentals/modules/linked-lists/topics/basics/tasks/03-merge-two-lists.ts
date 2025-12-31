import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-merge-two-lists',
	title: 'Merge Two Sorted Lists',
	difficulty: 'easy',
	tags: ['python', 'linked-list', 'merge', 'recursion'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Merge two sorted linked lists into one sorted list.

**Problem:**

Given the heads of two sorted linked lists, merge them into one sorted list. The merged list should be made by splicing together the nodes of the input lists.

**Examples:**

\`\`\`
Input:
  list1: 1 -> 2 -> 4 -> nil
  list2: 1 -> 3 -> 4 -> nil
Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> nil

Input:
  list1: nil
  list2: 0 -> nil
Output: 0 -> nil
\`\`\`

**Algorithm:**

1. Create a dummy node as starting point
2. Compare heads of both lists
3. Attach smaller node to result, advance that list
4. Repeat until one list is exhausted
5. Attach remaining nodes

**Time Complexity:** O(n + m)
**Space Complexity:** O(1) - reusing existing nodes`,
	initialCode: `from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    # TODO: Merge two sorted linked lists into one sorted list

    return None`,
	solutionCode: `from typing import Optional

class ListNode:
    """Represents a node in linked list."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Merge two sorted linked lists.
    Returns head of merged sorted list.

    Args:
        list1: Head of first sorted list
        list2: Head of second sorted list

    Returns:
        Head of merged sorted list
    """
    # Dummy node simplifies edge cases
    dummy = ListNode()
    current = dummy

    # Merge while both lists have nodes
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # Attach remaining nodes
    current.next = list1 if list1 else list2

    return dummy.next`,
	testCode: `import pytest
from typing import List, Optional
from solution import ListNode, merge_two_lists


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


class TestMergeTwoLists:
    def test_basic(self):
        """Test basic merge"""
        list1 = create_list([1, 2, 4])
        list2 = create_list([1, 3, 4])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 1, 2, 3, 4, 4]

    def test_empty_first(self):
        """Test first list empty"""
        list1 = create_list([])
        list2 = create_list([0])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [0]

    def test_empty_second(self):
        """Test second list empty"""
        list1 = create_list([1, 2])
        list2 = create_list([])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 2]

    def test_both_empty(self):
        """Test both lists empty"""
        result = merge_two_lists(None, None)
        assert list_to_array(result) == []

    def test_single_each(self):
        """Test single element in each"""
        list1 = create_list([1])
        list2 = create_list([2])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 2]

    def test_interleaved(self):
        """Test interleaved values"""
        list1 = create_list([1, 3, 5])
        list2 = create_list([2, 4, 6])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 2, 3, 4, 5, 6]

    def test_different_lengths(self):
        """Test lists with different lengths"""
        list1 = create_list([1, 2, 3, 4, 5])
        list2 = create_list([6, 7])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 2, 3, 4, 5, 6, 7]

    def test_all_duplicates(self):
        """Test lists with all same values"""
        list1 = create_list([1, 1, 1])
        list2 = create_list([1, 1])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 1, 1, 1, 1]

    def test_negative_values(self):
        """Test merge with negative values"""
        list1 = create_list([-3, -1, 2])
        list2 = create_list([-2, 0, 4])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [-3, -2, -1, 0, 2, 4]

    def test_second_list_larger(self):
        """Test when second list has larger values"""
        list1 = create_list([1, 2])
        list2 = create_list([3, 4, 5, 6])
        result = merge_two_lists(list1, list2)
        assert list_to_array(result) == [1, 2, 3, 4, 5, 6]`,
	hint1: `Create a dummy node to simplify handling the head of the result list. Use a 'current' pointer that you advance as you build the merged list.`,
	hint2: `After the main loop, one list may still have nodes. Simply attach the remaining list with 'current.next = remaining_list' - no need to iterate.`,
	whyItMatters: `Merging sorted lists is a key building block for many algorithms.

**Why This Matters:**

**1. Merge Sort Foundation**

This is the merge step in merge sort for linked lists:
\`\`\`python
def merge_sort(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    mid = find_middle(head)
    left = merge_sort(head)
    right = merge_sort(mid)
    return merge_two_lists(left, right)
\`\`\`

**2. Dummy Node Technique**

The dummy node pattern simplifies linked list operations:
\`\`\`python
# Without dummy: must handle head specially
if list1.val < list2.val:
    head = list1
    list1 = list1.next
else:
    head = list2
    list2 = list2.next
current = head
# ... continue

# With dummy: clean, uniform code
dummy = ListNode()
current = dummy
# ... just build the list
return dummy.next
\`\`\`

**3. Merge K Lists**

This extends to merging K sorted lists:
- Pair-wise merge: O(kN)
- Heap-based merge: O(N log k)
- Divide and conquer: O(N log k)

**4. Database Operations**

Merge operations are used in:
- Sort-merge join
- External sorting
- Log-structured merge trees`,
	order: 3,
	translations: {
		ru: {
			title: 'Слияние двух отсортированных списков',
			description: `Объедините два отсортированных связных списка в один отсортированный.

**Задача:**

Даны головы двух отсортированных связных списков, объедините их в один отсортированный список. Объединённый список должен состоять из узлов входных списков.

**Примеры:**

\`\`\`
Вход:
  list1: 1 -> 2 -> 4 -> nil
  list2: 1 -> 3 -> 4 -> nil
Выход: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> nil

Вход:
  list1: nil
  list2: 0 -> nil
Выход: 0 -> nil
\`\`\`

**Алгоритм:**

1. Создайте фиктивный узел как начальную точку
2. Сравните головы обоих списков
3. Прикрепите меньший узел к результату, продвиньте тот список
4. Повторяйте пока один список не исчерпан
5. Прикрепите оставшиеся узлы

**Временная сложность:** O(n + m)
**Пространственная сложность:** O(1)`,
			hint1: `Создайте фиктивный узел для упрощения обработки головы результирующего списка. Используйте указатель 'current', который продвигаете по мере построения объединённого списка.`,
			hint2: `После основного цикла в одном списке ещё могут быть узлы. Просто прикрепите оставшийся список с 'current.next = remaining_list' - не нужно итерировать.`,
			whyItMatters: `Слияние отсортированных списков - ключевой строительный блок для многих алгоритмов.

**Почему это важно:**

**1. Основа Merge Sort**

Это шаг слияния в merge sort для связных списков.

**2. Техника фиктивного узла**

Паттерн фиктивного узла упрощает операции со связными списками.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Представляет узел связного списка."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Объединяет два отсортированных связных списка.
    Возвращает голову объединённого отсортированного списка.

    Args:
        list1: Голова первого отсортированного списка
        list2: Голова второго отсортированного списка

    Returns:
        Голова объединённого отсортированного списка
    """
    # Фиктивный узел упрощает граничные случаи
    dummy = ListNode()
    current = dummy

    # Слияние пока в обоих списках есть узлы
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # Прикрепляем оставшиеся узлы
    current.next = list1 if list1 else list2

    return dummy.next`
		},
		uz: {
			title: 'Ikki saralangan ro\'yxatni birlashtirish',
			description: `Ikki saralangan bog'langan ro'yxatni bitta saralangan ro'yxatga birlashtiring.

**Masala:**

Ikki saralangan bog'langan ro'yxatning boshlari berilgan, ularni bitta saralangan ro'yxatga birlashtiring. Birlashtirilgan ro'yxat kirish ro'yxatlarining tugunlaridan tashkil topishi kerak.

**Misollar:**

\`\`\`
Kirish:
  list1: 1 -> 2 -> 4 -> nil
  list2: 1 -> 3 -> 4 -> nil
Chiqish: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> nil
\`\`\`

**Algoritm:**

1. Boshlang'ich nuqta sifatida dummy tugun yarating
2. Ikkala ro'yxatning boshlarini solishtiring
3. Kichikroq tugunni natijaga biriktiring, o'sha ro'yxatni oldinga siljiting
4. Bitta ro'yxat tugaguncha takrorlang
5. Qolgan tugunlarni biriktiring

**Vaqt murakkabligi:** O(n + m)
**Xotira murakkabligi:** O(1)`,
			hint1: `Natija ro'yxatining boshini boshqarishni soddalashtirish uchun dummy tugun yarating. Birlashtirilgan ro'yxatni qurayotganingizda oldinga siljitadigan 'current' ko'rsatkichidan foydalaning.`,
			hint2: `Asosiy tsikldan keyin bitta ro'yxatda hali ham tugunlar bo'lishi mumkin. Shunchaki qolgan ro'yxatni 'current.next = remaining_list' bilan biriktiring - takrorlash shart emas.`,
			whyItMatters: `Saralangan ro'yxatlarni birlashtirish ko'plab algoritmlar uchun asosiy qurilish bloki.

**Bu nima uchun muhim:**

**1. Merge Sort asosi**

Bu bog'langan ro'yxatlar uchun merge sort dagi birlashtirish qadami.

**2. Dummy tugun texnikasi**

Dummy tugun patterni bog'langan ro'yxat operatsiyalarini soddalashtiradi.`,
			solutionCode: `from typing import Optional

class ListNode:
    """Bog'langan ro'yxatdagi tugunni ifodalaydi."""
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Ikki saralangan bog'langan ro'yxatni birlashtiradi.
    Birlashtirilgan saralangan ro'yxatning boshini qaytaradi.

    Args:
        list1: Birinchi saralangan ro'yxatning boshi
        list2: Ikkinchi saralangan ro'yxatning boshi

    Returns:
        Birlashtirilgan saralangan ro'yxatning boshi
    """
    # Dummy tugun chegara holatlarini soddalashtiradi
    dummy = ListNode()
    current = dummy

    # Ikkala ro'yxatda tugunlar bor ekan birlashtirish
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # Qolgan tugunlarni biriktiramiz
    current.next = list1 if list1 else list2

    return dummy.next`
		}
	}
};

export default task;
