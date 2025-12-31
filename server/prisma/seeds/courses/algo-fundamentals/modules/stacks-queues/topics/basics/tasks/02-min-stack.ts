import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-min-stack',
	title: 'Min Stack',
	difficulty: 'medium',
	tags: ['python', 'stack', 'design'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Design a stack that supports retrieving the minimum element in constant time.

**Problem:**

Implement a stack with the following operations, all in O(1) time:
- \`Push(val)\` - pushes element onto stack
- \`Pop()\` - removes top element
- \`Top()\` - gets top element
- \`GetMin()\` - retrieves minimum element

**Examples:**

\`\`\`
MinStack stack = new MinStack();
stack.Push(-2);
stack.Push(0);
stack.Push(-3);
stack.GetMin(); // Returns -3
stack.Pop();
stack.GetMin(); // Returns -2
stack.Top();    // Returns 0
\`\`\`

**Approach:**

Use two stacks:
1. Main stack for normal operations
2. Min stack that tracks minimum at each level

When pushing, also push to min stack if value <= current min.
When popping, also pop from min stack if value == current min.

**Time Complexity:** O(1) for all operations
**Space Complexity:** O(n)`,
	initialCode: `class MinStack:
    # TODO: Implement a stack that supports push, pop, top, and get_min in O(1)

    def __init__(self):
        pass

    def push(self, val: int) -> None:
        pass

    def pop(self) -> int:
        return 0

    def top(self) -> int:
        return 0

    def get_min(self) -> int:
        return 0`,
	solutionCode: `class MinStack:
    """
    Stack that supports push, pop, top, and get_min in O(1).
    """

    def __init__(self):
        """Initialize the MinStack."""
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        """Push element onto stack."""
        self.stack.append(val)

        # Push to min stack if empty or val <= current min
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> int:
        """Remove and return top element."""
        if not self.stack:
            return 0

        val = self.stack.pop()

        # Also pop from min stack if it was the minimum
        if val == self.min_stack[-1]:
            self.min_stack.pop()

        return val

    def top(self) -> int:
        """Return top element without removing."""
        if not self.stack:
            return 0
        return self.stack[-1]

    def get_min(self) -> int:
        """Return minimum element in stack."""
        if not self.min_stack:
            return 0
        return self.min_stack[-1]`,
	testCode: `import pytest
from solution import MinStack


class TestMinStack:
    def test_basic_operations(self):
        """Test basic push, pop, top, get_min"""
        stack = MinStack()

        stack.push(-2)
        stack.push(0)
        stack.push(-3)

        assert stack.get_min() == -3

        stack.pop()

        assert stack.get_min() == -2
        assert stack.top() == 0

    def test_duplicates(self):
        """Test handling duplicate minimum values"""
        stack = MinStack()

        stack.push(0)
        stack.push(1)
        stack.push(0)

        assert stack.get_min() == 0

        stack.pop()

        # After popping one 0, min should still be 0
        assert stack.get_min() == 0

    def test_single_element(self):
        """Test single element"""
        stack = MinStack()
        stack.push(5)

        assert stack.top() == 5
        assert stack.get_min() == 5

    def test_all_same(self):
        """Test all same values"""
        stack = MinStack()
        stack.push(3)
        stack.push(3)
        stack.push(3)

        assert stack.get_min() == 3
        stack.pop()
        assert stack.get_min() == 3

    def test_decreasing(self):
        """Test decreasing sequence"""
        stack = MinStack()
        stack.push(3)
        stack.push(2)
        stack.push(1)

        assert stack.get_min() == 1
        stack.pop()
        assert stack.get_min() == 2

    def test_increasing(self):
        """Test increasing sequence"""
        stack = MinStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)

        assert stack.get_min() == 1
        assert stack.top() == 3

    def test_negative_numbers(self):
        """Test with negative numbers"""
        stack = MinStack()
        stack.push(-1)
        stack.push(-5)
        stack.push(-3)

        assert stack.get_min() == -5
        stack.pop()
        assert stack.get_min() == -5

    def test_mixed_positive_negative(self):
        """Test mixed positive and negative values"""
        stack = MinStack()
        stack.push(5)
        stack.push(-2)
        stack.push(0)
        stack.push(-2)

        assert stack.get_min() == -2
        stack.pop()
        assert stack.get_min() == -2

    def test_large_numbers(self):
        """Test with large numbers"""
        stack = MinStack()
        stack.push(1000000)
        stack.push(999999)
        stack.push(1000001)

        assert stack.get_min() == 999999
        assert stack.top() == 1000001

    def test_alternating_min(self):
        """Test alternating minimum values"""
        stack = MinStack()
        stack.push(5)
        assert stack.get_min() == 5
        stack.push(3)
        assert stack.get_min() == 3
        stack.push(7)
        assert stack.get_min() == 3
        stack.push(1)
        assert stack.get_min() == 1
        stack.pop()
        assert stack.get_min() == 3`,
	hint1: `Use two stacks: one for the actual values and one to track the minimum at each level. The min stack only needs to store values when they are less than or equal to the current minimum.`,
	hint2: `When pushing, push to min stack if the value is <= top of min stack. When popping, pop from min stack only if the popped value equals the top of min stack.`,
	whyItMatters: `This problem teaches auxiliary data structures for O(1) operations.

**Why This Matters:**

**1. Auxiliary Data Structures**

The key insight is using extra space for faster queries:
\`\`\`python
# Without min stack: O(n) to find min
# With min stack: O(1) to find min, O(n) extra space
\`\`\`

**2. Synchronizing Data Structures**

Keeping two data structures in sync:
\`\`\`python
# Push: update both stacks
# Pop: conditionally update min stack
# The tricky part: handling duplicates
\`\`\`

**3. Space-Time Trade-off**

\`\`\`python
# Alternative: Store (val, min_so_far) tuples
class MinStack:
    def __init__(self):
        self.stack = []  # Each element: (val, min_so_far)

    def push(self, val: int) -> None:
        min_so_far = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, min_so_far))
# More space per element, but simpler logic
\`\`\`

**4. Design Pattern**

This pattern appears in:
- Max Stack (track maximum)
- Median Finder (two heaps)
- LRU Cache (hashmap + doubly linked list)`,
	order: 2,
	translations: {
		ru: {
			title: 'Стек с минимумом',
			description: `Спроектируйте стек, поддерживающий получение минимального элемента за константное время.

**Задача:**

Реализуйте стек со следующими операциями, все за O(1):
- \`Push(val)\` - добавляет элемент в стек
- \`Pop()\` - удаляет верхний элемент
- \`Top()\` - возвращает верхний элемент
- \`GetMin()\` - возвращает минимальный элемент

**Примеры:**

\`\`\`
stack.Push(-2);
stack.Push(0);
stack.Push(-3);
stack.GetMin(); // Возвращает -3
stack.Pop();
stack.GetMin(); // Возвращает -2
\`\`\`

**Подход:**

Используйте два стека:
1. Основной стек для обычных операций
2. Стек минимумов, отслеживающий минимум на каждом уровне

**Временная сложность:** O(1) для всех операций
**Пространственная сложность:** O(n)`,
			hint1: `Используйте два стека: один для значений, другой для отслеживания минимума на каждом уровне. Стек минимумов хранит значения только когда они меньше или равны текущему минимуму.`,
			hint2: `При push добавляйте в стек минимумов если значение <= вершине стека минимумов. При pop удаляйте из стека минимумов только если значение равно вершине.`,
			whyItMatters: `Эта задача учит использованию вспомогательных структур данных для операций O(1).

**Почему это важно:**

**1. Вспомогательные структуры данных**

Ключевая идея - использовать дополнительную память для быстрых запросов.

**2. Компромисс память-время**

Альтернатива: хранить пары (значение, минимумДоСих).`,
			solutionCode: `class MinStack:
    """
    Стек, поддерживающий push, pop, top и get_min за O(1).
    """

    def __init__(self):
        """Инициализирует MinStack."""
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        """Добавляет элемент в стек."""
        self.stack.append(val)

        # Добавляем в стек минимумов если пуст или val <= текущего минимума
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> int:
        """Удаляет и возвращает верхний элемент."""
        if not self.stack:
            return 0

        val = self.stack.pop()

        # Также удаляем из стека минимумов если это был минимум
        if val == self.min_stack[-1]:
            self.min_stack.pop()

        return val

    def top(self) -> int:
        """Возвращает верхний элемент без удаления."""
        if not self.stack:
            return 0
        return self.stack[-1]

    def get_min(self) -> int:
        """Возвращает минимальный элемент в стеке."""
        if not self.min_stack:
            return 0
        return self.min_stack[-1]`
		},
		uz: {
			title: 'Minimum bilan stek',
			description: `Minimal elementni doimiy vaqtda olishni qo'llab-quvvatlaydigan stek loyihalang.

**Masala:**

Quyidagi operatsiyalarni amalga oshiring, barchasi O(1) da:
- \`Push(val)\` - elementni stekga qo'shadi
- \`Pop()\` - yuqori elementni olib tashlaydi
- \`Top()\` - yuqori elementni qaytaradi
- \`GetMin()\` - minimal elementni qaytaradi

**Misollar:**

\`\`\`
stack.Push(-2);
stack.Push(0);
stack.Push(-3);
stack.GetMin(); // -3 qaytaradi
stack.Pop();
stack.GetMin(); // -2 qaytaradi
\`\`\`

**Yondashuv:**

Ikki stekdan foydalaning:
1. Oddiy operatsiyalar uchun asosiy stek
2. Har bir darajada minimumni kuzatuvchi min stek

**Vaqt murakkabligi:** Barcha operatsiyalar uchun O(1)
**Xotira murakkabligi:** O(n)`,
			hint1: `Ikki stekdan foydalaning: biri qiymatlar uchun, biri har bir darajada minimumni kuzatish uchun. Min stek faqat joriy minimumdan kichik yoki teng qiymatlarni saqlaydi.`,
			hint2: `Push da qiymat min stek tepasidan <= bo'lsa min stekga qo'shing. Pop da faqat qiymat min stek tepasiga teng bo'lsa min stekdan oling.`,
			whyItMatters: `Bu masala O(1) operatsiyalar uchun yordamchi ma'lumotlar tuzilmalarini o'rgatadi.

**Bu nima uchun muhim:**

**1. Yordamchi ma'lumotlar tuzilmalari**

Asosiy tushuncha - tez so'rovlar uchun qo'shimcha xotira ishlatish.

**2. Xotira-vaqt kelishtiruvi**

Alternativa: (qiymat, shuPaytgachaMin) juftliklarini saqlash.`,
			solutionCode: `class MinStack:
    """
    Push, pop, top va get_min ni O(1) da qo'llab-quvvatlaydigan stek.
    """

    def __init__(self):
        """MinStack ni ishga tushiradi."""
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        """Elementni stekga qo'shadi."""
        self.stack.append(val)

        # Min stekga qo'shamiz agar bo'sh yoki val <= joriy minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> int:
        """Yuqori elementni olib tashlaydi va qaytaradi."""
        if not self.stack:
            return 0

        val = self.stack.pop()

        # Agar bu minimum bo'lsa min stekdan ham olamiz
        if val == self.min_stack[-1]:
            self.min_stack.pop()

        return val

    def top(self) -> int:
        """Yuqori elementni olib tashlamay qaytaradi."""
        if not self.stack:
            return 0
        return self.stack[-1]

    def get_min(self) -> int:
        """Stekdagi minimal elementni qaytaradi."""
        if not self.min_stack:
            return 0
        return self.min_stack[-1]`
		}
	}
};

export default task;
