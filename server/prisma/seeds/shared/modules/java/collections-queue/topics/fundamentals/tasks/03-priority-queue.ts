import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-priority-queue',
    title: 'PriorityQueue and Heap Operations',
    difficulty: 'medium',
    tags: ['java', 'collections', 'priority-queue', 'heap', 'comparator'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn PriorityQueue and heap-based operations in Java.

**Requirements:**
1. Create a PriorityQueue with natural ordering (min-heap)
2. Add integers and demonstrate they're retrieved in sorted order
3. Create a PriorityQueue with custom Comparator (max-heap)
4. Create a Task class with priority field
5. Use PriorityQueue to schedule tasks by priority
6. Demonstrate peek() vs poll() behavior
7. Show that iteration order is not guaranteed to be sorted
8. Implement a simple task scheduler using PriorityQueue

PriorityQueue is backed by a binary heap and provides O(log n) insertion and removal, with O(1) access to the minimum element.`,
    initialCode: `import java.util.PriorityQueue;
import java.util.Comparator;

public class PriorityQueueDemo {
    static class Task {
        String name;
        int priority;

        // Add constructor and toString
    }

    public static void main(String[] args) {
        // Create min-heap with natural ordering

        // Create max-heap with custom Comparator

        // Create and schedule tasks by priority

        // Show peek vs poll

        // Show iteration is not sorted
    }
}`,
    solutionCode: `import java.util.PriorityQueue;
import java.util.Comparator;

public class PriorityQueueDemo {
    static class Task {
        String name;
        int priority; // Lower number = higher priority

        public Task(String name, int priority) {
            this.name = name;
            this.priority = priority;
        }

        @Override
        public String toString() {
            return name + "(P" + priority + ")";
        }
    }

    public static void main(String[] args) {
        // Create min-heap with natural ordering
        System.out.println("Min-Heap (natural ordering):");
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        minHeap.offer(1);
        minHeap.offer(10);

        System.out.println("Elements added: 5, 2, 8, 1, 10");
        System.out.println("Poll order (smallest first):");
        while (!minHeap.isEmpty()) {
            System.out.print(minHeap.poll() + " ");
        }
        System.out.println();

        // Create max-heap with custom Comparator
        System.out.println("\\nMax-Heap (reverse ordering):");
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(2);
        maxHeap.offer(8);
        maxHeap.offer(1);
        maxHeap.offer(10);

        System.out.println("Elements added: 5, 2, 8, 1, 10");
        System.out.println("Poll order (largest first):");
        while (!maxHeap.isEmpty()) {
            System.out.print(maxHeap.poll() + " ");
        }
        System.out.println();

        // Create and schedule tasks by priority
        System.out.println("\\nTask Scheduler (priority-based):");
        PriorityQueue<Task> taskQueue = new PriorityQueue<>(
            Comparator.comparingInt(t -> t.priority)
        );

        taskQueue.offer(new Task("Email", 3));
        taskQueue.offer(new Task("Critical-Bug", 1));
        taskQueue.offer(new Task("Meeting", 2));
        taskQueue.offer(new Task("Documentation", 5));
        taskQueue.offer(new Task("Code-Review", 2));

        System.out.println("Tasks scheduled: Email(P3), Critical-Bug(P1), Meeting(P2), Documentation(P5), Code-Review(P2)");
        System.out.println("\\nExecution order:");
        while (!taskQueue.isEmpty()) {
            System.out.println("  Executing: " + taskQueue.poll());
        }

        // Show peek vs poll
        System.out.println("\\nPeek vs Poll:");
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        queue.offer(30);
        queue.offer(10);
        queue.offer(20);

        System.out.println("Queue: " + queue);
        System.out.println("peek(): " + queue.peek() + " (queue unchanged)");
        System.out.println("Queue: " + queue);
        System.out.println("poll(): " + queue.poll() + " (removes element)");
        System.out.println("Queue: " + queue);

        // Show iteration is not sorted
        System.out.println("\\nIteration order (NOT guaranteed sorted):");
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(50);
        pq.offer(20);
        pq.offer(40);
        pq.offer(10);
        pq.offer(30);

        System.out.println("Iteration order:");
        for (Integer num : pq) {
            System.out.print(num + " ");
        }
        System.out.println("\\nPoll order (guaranteed sorted):");
        while (!pq.isEmpty()) {
            System.out.print(pq.poll() + " ");
        }
        System.out.println();
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.PriorityQueue;
import java.util.Comparator;

// Test1: Verify min-heap natural ordering
class Test1 {
    @Test
    public void test() {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        assertEquals(Integer.valueOf(2), minHeap.peek());
    }
}

// Test2: Verify min-heap poll order
class Test2 {
    @Test
    public void test() {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        assertEquals(Integer.valueOf(2), minHeap.poll());
        assertEquals(Integer.valueOf(5), minHeap.poll());
        assertEquals(Integer.valueOf(8), minHeap.poll());
    }
}

// Test3: Verify max-heap with reverseOrder
class Test3 {
    @Test
    public void test() {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(2);
        maxHeap.offer(8);
        assertEquals(Integer.valueOf(8), maxHeap.peek());
    }
}

// Test4: Verify max-heap poll order
class Test4 {
    @Test
    public void test() {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(2);
        maxHeap.offer(8);
        assertEquals(Integer.valueOf(8), maxHeap.poll());
        assertEquals(Integer.valueOf(5), maxHeap.poll());
        assertEquals(Integer.valueOf(2), maxHeap.poll());
    }
}

// Test5: Verify custom comparator with objects
class Test5 {
    static class Item {
        int priority;
        Item(int p) { priority = p; }
    }

    @Test
    public void test() {
        PriorityQueue<Item> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i.priority));
        pq.offer(new Item(3));
        pq.offer(new Item(1));
        pq.offer(new Item(2));
        assertEquals(1, pq.poll().priority);
    }
}

// Test6: Verify peek doesn't remove element
class Test6 {
    @Test
    public void test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(30);
        pq.offer(10);
        pq.offer(20);
        assertEquals(Integer.valueOf(10), pq.peek());
        assertEquals(3, pq.size());
    }
}

// Test7: Verify poll removes element
class Test7 {
    @Test
    public void test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(30);
        pq.offer(10);
        pq.offer(20);
        assertEquals(Integer.valueOf(10), pq.poll());
        assertEquals(2, pq.size());
    }
}

// Test8: Verify empty queue behavior
class Test8 {
    @Test
    public void test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        assertNull(pq.peek());
        assertNull(pq.poll());
        assertTrue(pq.isEmpty());
    }
}

// Test9: Verify size method
class Test9 {
    @Test
    public void test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        assertEquals(0, pq.size());
        pq.offer(1);
        pq.offer(2);
        assertEquals(2, pq.size());
        pq.poll();
        assertEquals(1, pq.size());
    }
}

// Test10: Verify priority queue maintains heap property
class Test10 {
    @Test
    public void test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(50);
        pq.offer(20);
        pq.offer(40);
        pq.offer(10);
        pq.offer(30);
        // Poll all elements and verify sorted order
        int prev = pq.poll();
        while (!pq.isEmpty()) {
            int curr = pq.poll();
            assertTrue(prev <= curr);
            prev = curr;
        }
    }
}
`,
    hint1: `PriorityQueue() creates a min-heap (smallest element first). Use Comparator.reverseOrder() for max-heap. For custom objects, use Comparator.comparingInt() or implement Comparable.`,
    hint2: `peek() returns the highest priority element without removal. poll() removes it. Iteration order is not sorted - only poll() guarantees sorted order. Use poll() in a loop for sorted processing.`,
    whyItMatters: `PriorityQueue is essential for scheduling, event processing, and algorithms like Dijkstra's. Understanding heap operations and custom ordering lets you implement efficient priority-based systems with O(log n) performance.

**Production Pattern:**
\`\`\`java
// Priority-based task scheduler
PriorityQueue<Task> taskScheduler = new PriorityQueue<>(
    Comparator.comparingInt(Task::getPriority));
taskScheduler.offer(new Task("critical-bug", 1));
taskScheduler.offer(new Task("feature", 3));
Task next = taskScheduler.poll();  // Get highest priority task

// Top-K elements (e.g., top-10 results)
PriorityQueue<Result> topK = new PriorityQueue<>(K,
    Comparator.comparingDouble(Result::getScore));
for (Result r : allResults) {
    topK.offer(r);
    if (topK.size() > K) topK.poll();  // Remove smallest
}
\`\`\`

**Practical Benefits:**
- Automatic task prioritization without manual sorting
- Efficient Top-K algorithm with O(n log k) complexity
- Perfect for schedulers and event processing systems`,
    order: 2,
    translations: {
        ru: {
            title: 'PriorityQueue и операции кучи',
            solutionCode: `import java.util.PriorityQueue;
import java.util.Comparator;

public class PriorityQueueDemo {
    static class Task {
        String name;
        int priority; // Меньшее число = выше приоритет

        public Task(String name, int priority) {
            this.name = name;
            this.priority = priority;
        }

        @Override
        public String toString() {
            return name + "(P" + priority + ")";
        }
    }

    public static void main(String[] args) {
        // Создаем min-heap с естественным порядком
        System.out.println("Min-Heap (естественный порядок):");
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        minHeap.offer(1);
        minHeap.offer(10);

        System.out.println("Добавлены элементы: 5, 2, 8, 1, 10");
        System.out.println("Порядок извлечения (сначала наименьший):");
        while (!minHeap.isEmpty()) {
            System.out.print(minHeap.poll() + " ");
        }
        System.out.println();

        // Создаем max-heap с пользовательским Comparator
        System.out.println("\\nMax-Heap (обратный порядок):");
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(2);
        maxHeap.offer(8);
        maxHeap.offer(1);
        maxHeap.offer(10);

        System.out.println("Добавлены элементы: 5, 2, 8, 1, 10");
        System.out.println("Порядок извлечения (сначала наибольший):");
        while (!maxHeap.isEmpty()) {
            System.out.print(maxHeap.poll() + " ");
        }
        System.out.println();

        // Создаем и планируем задачи по приоритету
        System.out.println("\\nПланировщик задач (на основе приоритета):");
        PriorityQueue<Task> taskQueue = new PriorityQueue<>(
            Comparator.comparingInt(t -> t.priority)
        );

        taskQueue.offer(new Task("Email", 3));
        taskQueue.offer(new Task("Critical-Bug", 1));
        taskQueue.offer(new Task("Meeting", 2));
        taskQueue.offer(new Task("Documentation", 5));
        taskQueue.offer(new Task("Code-Review", 2));

        System.out.println("Запланированы задачи: Email(P3), Critical-Bug(P1), Meeting(P2), Documentation(P5), Code-Review(P2)");
        System.out.println("\\nПорядок выполнения:");
        while (!taskQueue.isEmpty()) {
            System.out.println("  Выполняется: " + taskQueue.poll());
        }

        // Показываем peek vs poll
        System.out.println("\\nPeek vs Poll:");
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        queue.offer(30);
        queue.offer(10);
        queue.offer(20);

        System.out.println("Очередь: " + queue);
        System.out.println("peek(): " + queue.peek() + " (очередь не изменена)");
        System.out.println("Очередь: " + queue);
        System.out.println("poll(): " + queue.poll() + " (удаляет элемент)");
        System.out.println("Очередь: " + queue);

        // Показываем, что порядок итерации не отсортирован
        System.out.println("\\nПорядок итерации (НЕ гарантированно отсортирован):");
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(50);
        pq.offer(20);
        pq.offer(40);
        pq.offer(10);
        pq.offer(30);

        System.out.println("Порядок итерации:");
        for (Integer num : pq) {
            System.out.print(num + " ");
        }
        System.out.println("\\nПорядок poll (гарантированно отсортирован):");
        while (!pq.isEmpty()) {
            System.out.print(pq.poll() + " ");
        }
        System.out.println();
    }
}`,
            description: `Изучите PriorityQueue и операции на основе кучи в Java.

**Требования:**
1. Создайте PriorityQueue с естественным порядком (min-heap)
2. Добавьте целые числа и покажите, что они извлекаются в отсортированном порядке
3. Создайте PriorityQueue с пользовательским Comparator (max-heap)
4. Создайте класс Task с полем priority
5. Используйте PriorityQueue для планирования задач по приоритету
6. Продемонстрируйте поведение peek() vs poll()
7. Покажите, что порядок итерации не гарантированно отсортирован
8. Реализуйте простой планировщик задач используя PriorityQueue

PriorityQueue основан на двоичной куче и обеспечивает O(log n) вставку и удаление, с O(1) доступом к минимальному элементу.`,
            hint1: `PriorityQueue() создает min-heap (сначала наименьший элемент). Используйте Comparator.reverseOrder() для max-heap. Для пользовательских объектов используйте Comparator.comparingInt() или реализуйте Comparable.`,
            hint2: `peek() возвращает элемент с наивысшим приоритетом без удаления. poll() удаляет его. Порядок итерации не отсортирован - только poll() гарантирует отсортированный порядок. Используйте poll() в цикле для отсортированной обработки.`,
            whyItMatters: `PriorityQueue необходим для планирования, обработки событий и алгоритмов типа Дейкстры. Понимание операций кучи и пользовательского упорядочивания позволяет реализовывать эффективные системы на основе приоритетов с производительностью O(log n).

**Продакшен паттерн:**
\`\`\`java
// Планировщик задач по приоритету
PriorityQueue<Task> taskScheduler = new PriorityQueue<>(
    Comparator.comparingInt(Task::getPriority));
taskScheduler.offer(new Task("critical-bug", 1));
taskScheduler.offer(new Task("feature", 3));
Task next = taskScheduler.poll();  // Получаем задачу с наивысшим приоритетом

// Топ-K элементов (например, топ-10 результатов)
PriorityQueue<Result> topK = new PriorityQueue<>(K,
    Comparator.comparingDouble(Result::getScore));
for (Result r : allResults) {
    topK.offer(r);
    if (topK.size() > K) topK.poll();  // Удаляем наименьший
}
\`\`\`

**Практические преимущества:**
- Автоматическая приоритизация задач без ручной сортировки
- Эффективный алгоритм Top-K с O(n log k) сложностью
- Идеально для планировщиков и систем обработки событий`
        },
        uz: {
            title: 'PriorityQueue va Heap Operatsiyalari',
            solutionCode: `import java.util.PriorityQueue;
import java.util.Comparator;

public class PriorityQueueDemo {
    static class Task {
        String name;
        int priority; // Kichikroq son = yuqoriroq prioritet

        public Task(String name, int priority) {
            this.name = name;
            this.priority = priority;
        }

        @Override
        public String toString() {
            return name + "(P" + priority + ")";
        }
    }

    public static void main(String[] args) {
        // Tabiiy tartiblangan min-heap yaratamiz
        System.out.println("Min-Heap (tabiiy tartib):");
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(2);
        minHeap.offer(8);
        minHeap.offer(1);
        minHeap.offer(10);

        System.out.println("Qo'shilgan elementlar: 5, 2, 8, 1, 10");
        System.out.println("Olish tartibi (avval eng kichik):");
        while (!minHeap.isEmpty()) {
            System.out.print(minHeap.poll() + " ");
        }
        System.out.println();

        // Maxsus Comparator bilan max-heap yaratamiz
        System.out.println("\\nMax-Heap (teskari tartib):");
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(2);
        maxHeap.offer(8);
        maxHeap.offer(1);
        maxHeap.offer(10);

        System.out.println("Qo'shilgan elementlar: 5, 2, 8, 1, 10");
        System.out.println("Olish tartibi (avval eng katta):");
        while (!maxHeap.isEmpty()) {
            System.out.print(maxHeap.poll() + " ");
        }
        System.out.println();

        // Vazifalarni prioritet bo'yicha rejalashtirish
        System.out.println("\\nVazifa rejalashtiruvchi (prioritet asosida):");
        PriorityQueue<Task> taskQueue = new PriorityQueue<>(
            Comparator.comparingInt(t -> t.priority)
        );

        taskQueue.offer(new Task("Email", 3));
        taskQueue.offer(new Task("Critical-Bug", 1));
        taskQueue.offer(new Task("Meeting", 2));
        taskQueue.offer(new Task("Documentation", 5));
        taskQueue.offer(new Task("Code-Review", 2));

        System.out.println("Rejalashtirilgan vazifalar: Email(P3), Critical-Bug(P1), Meeting(P2), Documentation(P5), Code-Review(P2)");
        System.out.println("\\nBajarilish tartibi:");
        while (!taskQueue.isEmpty()) {
            System.out.println("  Bajarilmoqda: " + taskQueue.poll());
        }

        // peek vs poll ni ko'rsatamiz
        System.out.println("\\nPeek vs Poll:");
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        queue.offer(30);
        queue.offer(10);
        queue.offer(20);

        System.out.println("Navbat: " + queue);
        System.out.println("peek(): " + queue.peek() + " (navbat o'zgarmagan)");
        System.out.println("Navbat: " + queue);
        System.out.println("poll(): " + queue.poll() + " (elementni o'chiradi)");
        System.out.println("Navbat: " + queue);

        // Iteratsiya tartibi saralanmaganligini ko'rsatamiz
        System.out.println("\\nIteratsiya tartibi (saralanish kafolatlanmagan):");
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        pq.offer(50);
        pq.offer(20);
        pq.offer(40);
        pq.offer(10);
        pq.offer(30);

        System.out.println("Iteratsiya tartibi:");
        for (Integer num : pq) {
            System.out.print(num + " ");
        }
        System.out.println("\\nPoll tartibi (saralanish kafolatlangan):");
        while (!pq.isEmpty()) {
            System.out.print(pq.poll() + " ");
        }
        System.out.println();
    }
}`,
            description: `Java da PriorityQueue va heap asosidagi operatsiyalarni o'rganing.

**Talablar:**
1. Tabiiy tartiblangan PriorityQueue yarating (min-heap)
2. Butun sonlar qo'shing va ularning saralangan tartibda olinishini ko'rsating
3. Maxsus Comparator bilan PriorityQueue yarating (max-heap)
4. priority maydoni bilan Task klassi yarating
5. Vazifalarni prioritet bo'yicha rejalashtrish uchun PriorityQueue dan foydalaning
6. peek() va poll() xatti-harakatini ko'rsating
7. Iteratsiya tartibi saralanmaganligini ko'rsating
8. PriorityQueue yordamida oddiy vazifa rejalashtiruvchini amalga oshiring

PriorityQueue ikkilik heap ga asoslangan va O(log n) qo'shish va o'chirish, minimal elementga O(1) kirish imkonini beradi.`,
            hint1: `PriorityQueue() min-heap yaratadi (avval eng kichik element). max-heap uchun Comparator.reverseOrder() dan foydalaning. Maxsus obyektlar uchun Comparator.comparingInt() dan foydalaning yoki Comparable ni amalga oshiring.`,
            hint2: `peek() eng yuqori prioritetli elementni o'chirmasdan qaytaradi. poll() uni o'chiradi. Iteratsiya tartibi saralanmagan - faqat poll() saralangan tartibni kafolatlaydi. Saralangan qayta ishlash uchun poll() ni tsiklda ishlating.`,
            whyItMatters: `PriorityQueue rejalashtrish, hodisalarni qayta ishlash va Dijkstra kabi algoritmlar uchun zarur. Heap operatsiyalari va maxsus tartibni tushunish O(log n) samaradorlik bilan prioritet asosidagi samarali tizimlarni amalga oshirish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Prioritet bo'yicha vazifalar rejalashtiruvchisi
PriorityQueue<Task> taskScheduler = new PriorityQueue<>(
    Comparator.comparingInt(Task::getPriority));
taskScheduler.offer(new Task("critical-bug", 1));
taskScheduler.offer(new Task("feature", 3));
Task next = taskScheduler.poll();  // Eng yuqori prioritetli vazifani olamiz

// Top-K elementlar (masalan, top-10 natijalar)
PriorityQueue<Result> topK = new PriorityQueue<>(K,
    Comparator.comparingDouble(Result::getScore));
for (Result r : allResults) {
    topK.offer(r);
    if (topK.size() > K) topK.poll();  // Eng kichikni o'chiramiz
}
\`\`\`

**Amaliy foydalari:**
- Qo'lda saralashsiz vazifalarni avtomatik ustuvorlashtirish
- O(n log k) murakkablik bilan samarali Top-K algoritmi
- Rejalashtiruvchilar va hodisalarni qayta ishlash tizimlari uchun ideal`
        }
    }
};

export default task;
