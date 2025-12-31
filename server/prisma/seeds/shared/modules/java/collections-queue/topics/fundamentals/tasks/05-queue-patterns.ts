import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-queue-patterns',
    title: 'Queue Processing Patterns',
    difficulty: 'medium',
    tags: ['java', 'collections', 'queue', 'patterns', 'bfs'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn practical Queue processing patterns in Java.

**Requirements:**
1. Implement BFS (Breadth-First Search) using Queue
2. Create a simple Producer-Consumer pattern with Queue
3. Implement a task queue processor
4. Process events in order using Queue
5. Implement level-order traversal pattern
6. Show Queue-based batch processing
7. Demonstrate FIFO order guarantee
8. Handle queue overflow scenarios

Queue-based patterns are fundamental for BFS algorithms, event processing, task scheduling, and managing ordered workflows.`,
    initialCode: `import java.util.Queue;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

public class QueuePatterns {
    static class TreeNode {
        int value;
        TreeNode left, right;

        TreeNode(int value) {
            this.value = value;
        }
    }

    static class Task {
        String name;
        int id;

        Task(String name, int id) {
            this.name = name;
            this.id = id;
        }
    }

    // Implement BFS for tree

    // Implement task queue processor

    // Implement event processor

    public static void main(String[] args) {
        // Test BFS

        // Test task processing

        // Test event processing
    }
}`,
    solutionCode: `import java.util.Queue;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

public class QueuePatterns {
    static class TreeNode {
        int value;
        TreeNode left, right;

        TreeNode(int value) {
            this.value = value;
        }
    }

    static class Task {
        String name;
        int id;

        Task(String name, int id) {
            this.name = name;
            this.id = id;
        }

        @Override
        public String toString() {
            return "Task{" + name + ", id=" + id + "}";
        }
    }

    static class Event {
        String type;
        long timestamp;

        Event(String type) {
            this.type = type;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return "Event{" + type + "}";
        }
    }

    // BFS - Breadth-First Search using Queue
    public static List<Integer> bfsTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            result.add(current.value);

            // Add children to queue (left to right)
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }

        return result;
    }

    // Level-order traversal (BFS with level tracking)
    public static List<List<Integer>> levelOrderTraversal(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<>();
        if (root == null) return levels;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();

            // Process all nodes at current level
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.value);

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }

            levels.add(currentLevel);
        }

        return levels;
    }

    // Task Queue Processor
    public static void processTaskQueue(Queue<Task> taskQueue) {
        System.out.println("Processing tasks in FIFO order:");

        while (!taskQueue.isEmpty()) {
            Task task = taskQueue.poll();
            System.out.println("  Executing: " + task);

            // Simulate task execution
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println("All tasks completed!");
    }

    // Producer-Consumer Pattern (simplified)
    public static void producerConsumerDemo() {
        Queue<String> buffer = new LinkedList<>();
        int bufferCapacity = 5;

        System.out.println("Producer-Consumer Pattern:");

        // Producer adds items
        System.out.println("Producer adding items:");
        for (int i = 1; i <= 7; i++) {
            if (buffer.size() < bufferCapacity) {
                buffer.offer("Item-" + i);
                System.out.println("  Produced: Item-" + i);
            } else {
                System.out.println("  Buffer full! Cannot produce Item-" + i);
            }
        }

        // Consumer processes items
        System.out.println("\\nConsumer processing items:");
        while (!buffer.isEmpty()) {
            String item = buffer.poll();
            System.out.println("  Consumed: " + item);
        }
    }

    // Event Processing Queue
    public static void processEvents(Queue<Event> eventQueue) {
        System.out.println("Processing events in order:");

        int processed = 0;
        while (!eventQueue.isEmpty()) {
            Event event = eventQueue.poll();
            System.out.println("  Processing: " + event);
            processed++;
        }

        System.out.println("Total events processed: " + processed);
    }

    // Batch Processing with Queue
    public static void batchProcess(Queue<Integer> dataQueue, int batchSize) {
        System.out.println("Batch processing (batch size: " + batchSize + "):");

        int batchNum = 1;
        while (!dataQueue.isEmpty()) {
            List<Integer> batch = new ArrayList<>();

            // Collect batch
            for (int i = 0; i < batchSize && !dataQueue.isEmpty(); i++) {
                batch.add(dataQueue.poll());
            }

            // Process batch
            System.out.println("  Batch " + batchNum + ": " + batch);
            batchNum++;
        }
    }

    public static void main(String[] args) {
        // Test BFS
        System.out.println("=== BFS Traversal ===");
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);

        List<Integer> bfsResult = bfsTraversal(root);
        System.out.println("BFS order: " + bfsResult);

        List<List<Integer>> levelOrder = levelOrderTraversal(root);
        System.out.println("Level-order: " + levelOrder);

        // Test Task Queue Processing
        System.out.println("\\n=== Task Queue Processing ===");
        Queue<Task> taskQueue = new LinkedList<>();
        taskQueue.offer(new Task("Initialize", 1));
        taskQueue.offer(new Task("LoadData", 2));
        taskQueue.offer(new Task("Process", 3));
        taskQueue.offer(new Task("Cleanup", 4));
        processTaskQueue(taskQueue);

        // Test Producer-Consumer
        System.out.println("\\n=== Producer-Consumer ===");
        producerConsumerDemo();

        // Test Event Processing
        System.out.println("\\n=== Event Processing ===");
        Queue<Event> eventQueue = new LinkedList<>();
        eventQueue.offer(new Event("UserLogin"));
        eventQueue.offer(new Event("DataFetch"));
        eventQueue.offer(new Event("UpdateUI"));
        eventQueue.offer(new Event("UserLogout"));
        processEvents(eventQueue);

        // Test Batch Processing
        System.out.println("\\n=== Batch Processing ===");
        Queue<Integer> dataQueue = new LinkedList<>();
        for (int i = 1; i <= 10; i++) {
            dataQueue.offer(i * 10);
        }
        batchProcess(dataQueue, 3);

        // FIFO Order Guarantee
        System.out.println("\\n=== FIFO Order Guarantee ===");
        Queue<String> fifoQueue = new LinkedList<>();
        fifoQueue.offer("First");
        fifoQueue.offer("Second");
        fifoQueue.offer("Third");
        System.out.println("Added: First, Second, Third");
        System.out.println("Retrieved: " + fifoQueue.poll() + ", " +
                          fifoQueue.poll() + ", " + fifoQueue.poll());
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.Queue;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;

// Test1: Verify BFS traversal order
class Test1 {
    static class Node {
        int val;
        Node left, right;
        Node(int v) { val = v; }
    }

    @Test
    public void test() {
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        List<Integer> result = new ArrayList<>();

        while (!queue.isEmpty()) {
            Node curr = queue.poll();
            result.add(curr.val);
            if (curr.left != null) queue.offer(curr.left);
            if (curr.right != null) queue.offer(curr.right);
        }

        assertEquals(3, result.size());
        assertEquals(Integer.valueOf(1), result.get(0));
    }
}

// Test2: Verify level-order traversal groups
class Test2 {
    static class Node {
        int val;
        Node left, right;
        Node(int v) { val = v; }
    }

    @Test
    public void test() {
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        List<List<Integer>> levels = new ArrayList<>();

        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                Node node = queue.poll();
                level.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            levels.add(level);
        }

        assertEquals(2, levels.size());
    }
}

// Test3: Verify task queue FIFO processing
class Test3 {
    @Test
    public void test() {
        Queue<String> tasks = new LinkedList<>();
        tasks.offer("Task1");
        tasks.offer("Task2");
        tasks.offer("Task3");

        assertEquals("Task1", tasks.poll());
        assertEquals("Task2", tasks.poll());
        assertEquals("Task3", tasks.poll());
    }
}

// Test4: Verify producer-consumer buffer capacity
class Test4 {
    @Test
    public void test() {
        Queue<String> buffer = new LinkedList<>();
        int capacity = 3;

        for (int i = 1; i <= 5; i++) {
            if (buffer.size() < capacity) {
                buffer.offer("Item" + i);
            }
        }

        assertEquals(3, buffer.size());
    }
}

// Test5: Verify event processing order
class Test5 {
    @Test
    public void test() {
        Queue<String> events = new LinkedList<>();
        events.offer("Login");
        events.offer("Fetch");
        events.offer("Logout");

        List<String> processed = new ArrayList<>();
        while (!events.isEmpty()) {
            processed.add(events.poll());
        }

        assertEquals("Login", processed.get(0));
        assertEquals("Fetch", processed.get(1));
        assertEquals("Logout", processed.get(2));
    }
}

// Test6: Verify batch processing groups
class Test6 {
    @Test
    public void test() {
        Queue<Integer> data = new LinkedList<>();
        for (int i = 1; i <= 7; i++) {
            data.offer(i);
        }

        int batchSize = 3;
        int batches = 0;

        while (!data.isEmpty()) {
            int count = Math.min(batchSize, data.size());
            for (int i = 0; i < count; i++) {
                data.poll();
            }
            batches++;
        }

        assertEquals(3, batches); // 3, 3, 1
    }
}

// Test7: Verify FIFO order guarantee
class Test7 {
    @Test
    public void test() {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(1);
        queue.offer(2);
        queue.offer(3);

        assertEquals(Integer.valueOf(1), queue.poll());
        assertEquals(Integer.valueOf(2), queue.poll());
        assertEquals(Integer.valueOf(3), queue.poll());
    }
}

// Test8: Verify queue overflow handling
class Test8 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        int maxSize = 2;

        boolean added1 = queue.size() < maxSize && queue.offer("A");
        boolean added2 = queue.size() < maxSize && queue.offer("B");
        boolean added3 = queue.size() < maxSize && queue.offer("C");

        assertTrue(added1);
        assertTrue(added2);
        assertFalse(added3);
    }
}

// Test9: Verify empty queue processing
class Test9 {
    @Test
    public void test() {
        Queue<String> queue = new LinkedList<>();
        int processed = 0;

        while (!queue.isEmpty()) {
            queue.poll();
            processed++;
        }

        assertEquals(0, processed);
    }
}

// Test10: Verify queue-based workflow
class Test10 {
    @Test
    public void test() {
        Queue<String> workflow = new LinkedList<>();
        workflow.offer("Init");
        workflow.offer("Process");
        workflow.offer("Cleanup");

        List<String> steps = new ArrayList<>();
        while (!workflow.isEmpty()) {
            steps.add(workflow.poll());
        }

        assertEquals(3, steps.size());
        assertEquals("Init", steps.get(0));
        assertEquals("Cleanup", steps.get(2));
    }
}
`,
    hint1: `For BFS, use Queue to process nodes level by level. Add root, then poll and add children in a loop. For level-order, track queue size at each level to group nodes.`,
    hint2: `Task queues follow FIFO: poll() retrieves the oldest task. For producer-consumer, check capacity before offering. For batch processing, poll in groups of batch size.`,
    whyItMatters: `Queue patterns are essential for algorithms (BFS, topological sort), system design (task queues, event processing), and distributed systems (message queues). Mastering these patterns enables efficient ordered processing and resource management.

**Production Pattern:**
\`\`\`java
// Event processing with order guarantee
Queue<OrderEvent> eventQueue = new LinkedList<>();
eventQueue.offer(new OrderEvent("created", order));
eventQueue.offer(new OrderEvent("paid", order));
while (!eventQueue.isEmpty()) {
    OrderEvent event = eventQueue.poll();
    eventProcessor.process(event);  // FIFO processing
}

// BFS for shortest path finding
Queue<Node> queue = new LinkedList<>();
Set<Node> visited = new HashSet<>();
queue.offer(startNode);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    if (current.equals(target)) return getPath(current);
    for (Node neighbor : current.getNeighbors()) {
        if (!visited.contains(neighbor)) {
            queue.offer(neighbor);
            visited.add(neighbor);
        }
    }
}
\`\`\`

**Practical Benefits:**
- FIFO order guarantee for event processing systems
- Efficient BFS for graph algorithms
- Batch processing for performance optimization`,
    order: 4,
    translations: {
        ru: {
            title: 'Паттерны обработки очередей',
            solutionCode: `import java.util.Queue;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

public class QueuePatterns {
    static class TreeNode {
        int value;
        TreeNode left, right;

        TreeNode(int value) {
            this.value = value;
        }
    }

    static class Task {
        String name;
        int id;

        Task(String name, int id) {
            this.name = name;
            this.id = id;
        }

        @Override
        public String toString() {
            return "Task{" + name + ", id=" + id + "}";
        }
    }

    static class Event {
        String type;
        long timestamp;

        Event(String type) {
            this.type = type;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return "Event{" + type + "}";
        }
    }

    // BFS - Поиск в ширину используя Queue
    public static List<Integer> bfsTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            result.add(current.value);

            // Добавляем детей в очередь (слева направо)
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }

        return result;
    }

    // Обход по уровням (BFS с отслеживанием уровня)
    public static List<List<Integer>> levelOrderTraversal(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<>();
        if (root == null) return levels;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();

            // Обрабатываем все узлы текущего уровня
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.value);

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }

            levels.add(currentLevel);
        }

        return levels;
    }

    // Обработчик очереди задач
    public static void processTaskQueue(Queue<Task> taskQueue) {
        System.out.println("Обработка задач в порядке FIFO:");

        while (!taskQueue.isEmpty()) {
            Task task = taskQueue.poll();
            System.out.println("  Выполняется: " + task);

            // Имитируем выполнение задачи
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println("Все задачи выполнены!");
    }

    // Паттерн Производитель-Потребитель (упрощенный)
    public static void producerConsumerDemo() {
        Queue<String> buffer = new LinkedList<>();
        int bufferCapacity = 5;

        System.out.println("Паттерн Производитель-Потребитель:");

        // Производитель добавляет элементы
        System.out.println("Производитель добавляет элементы:");
        for (int i = 1; i <= 7; i++) {
            if (buffer.size() < bufferCapacity) {
                buffer.offer("Item-" + i);
                System.out.println("  Произведено: Item-" + i);
            } else {
                System.out.println("  Буфер полон! Невозможно произвести Item-" + i);
            }
        }

        // Потребитель обрабатывает элементы
        System.out.println("\\nПотребитель обрабатывает элементы:");
        while (!buffer.isEmpty()) {
            String item = buffer.poll();
            System.out.println("  Потреблено: " + item);
        }
    }

    // Очередь обработки событий
    public static void processEvents(Queue<Event> eventQueue) {
        System.out.println("Обработка событий по порядку:");

        int processed = 0;
        while (!eventQueue.isEmpty()) {
            Event event = eventQueue.poll();
            System.out.println("  Обрабатывается: " + event);
            processed++;
        }

        System.out.println("Всего обработано событий: " + processed);
    }

    // Пакетная обработка с Queue
    public static void batchProcess(Queue<Integer> dataQueue, int batchSize) {
        System.out.println("Пакетная обработка (размер пакета: " + batchSize + "):");

        int batchNum = 1;
        while (!dataQueue.isEmpty()) {
            List<Integer> batch = new ArrayList<>();

            // Собираем пакет
            for (int i = 0; i < batchSize && !dataQueue.isEmpty(); i++) {
                batch.add(dataQueue.poll());
            }

            // Обрабатываем пакет
            System.out.println("  Пакет " + batchNum + ": " + batch);
            batchNum++;
        }
    }

    public static void main(String[] args) {
        // Тестируем BFS
        System.out.println("=== Обход BFS ===");
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);

        List<Integer> bfsResult = bfsTraversal(root);
        System.out.println("Порядок BFS: " + bfsResult);

        List<List<Integer>> levelOrder = levelOrderTraversal(root);
        System.out.println("По уровням: " + levelOrder);

        // Тестируем обработку очереди задач
        System.out.println("\\n=== Обработка очереди задач ===");
        Queue<Task> taskQueue = new LinkedList<>();
        taskQueue.offer(new Task("Initialize", 1));
        taskQueue.offer(new Task("LoadData", 2));
        taskQueue.offer(new Task("Process", 3));
        taskQueue.offer(new Task("Cleanup", 4));
        processTaskQueue(taskQueue);

        // Тестируем Производитель-Потребитель
        System.out.println("\\n=== Производитель-Потребитель ===");
        producerConsumerDemo();

        // Тестируем обработку событий
        System.out.println("\\n=== Обработка событий ===");
        Queue<Event> eventQueue = new LinkedList<>();
        eventQueue.offer(new Event("UserLogin"));
        eventQueue.offer(new Event("DataFetch"));
        eventQueue.offer(new Event("UpdateUI"));
        eventQueue.offer(new Event("UserLogout"));
        processEvents(eventQueue);

        // Тестируем пакетную обработку
        System.out.println("\\n=== Пакетная обработка ===");
        Queue<Integer> dataQueue = new LinkedList<>();
        for (int i = 1; i <= 10; i++) {
            dataQueue.offer(i * 10);
        }
        batchProcess(dataQueue, 3);

        // Гарантия порядка FIFO
        System.out.println("\\n=== Гарантия порядка FIFO ===");
        Queue<String> fifoQueue = new LinkedList<>();
        fifoQueue.offer("First");
        fifoQueue.offer("Second");
        fifoQueue.offer("Third");
        System.out.println("Добавлено: First, Second, Third");
        System.out.println("Извлечено: " + fifoQueue.poll() + ", " +
                          fifoQueue.poll() + ", " + fifoQueue.poll());
    }
}`,
            description: `Изучите практические паттерны обработки очередей в Java.

**Требования:**
1. Реализуйте BFS (поиск в ширину) используя Queue
2. Создайте простой паттерн Производитель-Потребитель с Queue
3. Реализуйте обработчик очереди задач
4. Обрабатывайте события по порядку используя Queue
5. Реализуйте паттерн обхода по уровням
6. Покажите пакетную обработку на основе Queue
7. Продемонстрируйте гарантию порядка FIFO
8. Обработайте сценарии переполнения очереди

Паттерны на основе очереди фундаментальны для алгоритмов BFS, обработки событий, планирования задач и управления упорядоченными рабочими процессами.`,
            hint1: `Для BFS используйте Queue для обработки узлов уровень за уровнем. Добавьте корень, затем извлекайте и добавляйте детей в цикле. Для обхода по уровням отслеживайте размер очереди на каждом уровне для группировки узлов.`,
            hint2: `Очереди задач следуют FIFO: poll() извлекает самую старую задачу. Для производителя-потребителя проверяйте емкость перед offer. Для пакетной обработки извлекайте группами размера пакета.`,
            whyItMatters: `Паттерны очередей необходимы для алгоритмов (BFS, топологическая сортировка), проектирования систем (очереди задач, обработка событий) и распределенных систем (очереди сообщений). Владение этими паттернами обеспечивает эффективную упорядоченную обработку и управление ресурсами.

**Продакшен паттерн:**
\`\`\`java
// Обработка событий с гарантией порядка
Queue<OrderEvent> eventQueue = new LinkedList<>();
eventQueue.offer(new OrderEvent("created", order));
eventQueue.offer(new OrderEvent("paid", order));
while (!eventQueue.isEmpty()) {
    OrderEvent event = eventQueue.poll();
    eventProcessor.process(event);  // Обработка в порядке FIFO
}

// BFS для поиска кратчайшего пути
Queue<Node> queue = new LinkedList<>();
Set<Node> visited = new HashSet<>();
queue.offer(startNode);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    if (current.equals(target)) return getPath(current);
    for (Node neighbor : current.getNeighbors()) {
        if (!visited.contains(neighbor)) {
            queue.offer(neighbor);
            visited.add(neighbor);
        }
    }
}
\`\`\`

**Практические преимущества:**
- FIFO гарантия порядка для систем обработки событий
- Эффективный BFS для графовых алгоритмов
- Пакетная обработка для оптимизации производительности`
        },
        uz: {
            title: 'Navbatni Qayta Ishlash Naqshlari',
            solutionCode: `import java.util.Queue;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;

public class QueuePatterns {
    static class TreeNode {
        int value;
        TreeNode left, right;

        TreeNode(int value) {
            this.value = value;
        }
    }

    static class Task {
        String name;
        int id;

        Task(String name, int id) {
            this.name = name;
            this.id = id;
        }

        @Override
        public String toString() {
            return "Task{" + name + ", id=" + id + "}";
        }
    }

    static class Event {
        String type;
        long timestamp;

        Event(String type) {
            this.type = type;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return "Event{" + type + "}";
        }
    }

    // BFS - Queue yordamida kenglik bo'yicha qidiruv
    public static List<Integer> bfsTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            result.add(current.value);

            // Bolalarni navbatga qo'shamiz (chapdan o'ngga)
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }

        return result;
    }

    // Darajalar bo'yicha o'tish (BFS daraja kuzatuvi bilan)
    public static List<List<Integer>> levelOrderTraversal(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<>();
        if (root == null) return levels;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();

            // Joriy darajadagi barcha tugunlarni qayta ishlaymiz
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.value);

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }

            levels.add(currentLevel);
        }

        return levels;
    }

    // Vazifa navbati qayta ishlovchisi
    public static void processTaskQueue(Queue<Task> taskQueue) {
        System.out.println("Vazifalarni FIFO tartibida qayta ishlash:");

        while (!taskQueue.isEmpty()) {
            Task task = taskQueue.poll();
            System.out.println("  Bajarilmoqda: " + task);

            // Vazifa bajarilishini simulyatsiya qilamiz
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println("Barcha vazifalar bajarildi!");
    }

    // Ishlab chiqaruvchi-Iste'molchi naqshi (soddalashtirilgan)
    public static void producerConsumerDemo() {
        Queue<String> buffer = new LinkedList<>();
        int bufferCapacity = 5;

        System.out.println("Ishlab chiqaruvchi-Iste'molchi naqshi:");

        // Ishlab chiqaruvchi elementlar qo'shadi
        System.out.println("Ishlab chiqaruvchi elementlar qo'shmoqda:");
        for (int i = 1; i <= 7; i++) {
            if (buffer.size() < bufferCapacity) {
                buffer.offer("Item-" + i);
                System.out.println("  Ishlab chiqarildi: Item-" + i);
            } else {
                System.out.println("  Bufer to'lgan! Item-" + i + " ni ishlab chiqarib bo'lmaydi");
            }
        }

        // Iste'molchi elementlarni qayta ishlaydi
        System.out.println("\\nIste'molchi elementlarni qayta ishlamoqda:");
        while (!buffer.isEmpty()) {
            String item = buffer.poll();
            System.out.println("  Iste'mol qilindi: " + item);
        }
    }

    // Hodisalarni qayta ishlash navbati
    public static void processEvents(Queue<Event> eventQueue) {
        System.out.println("Hodisalarni tartib bilan qayta ishlash:");

        int processed = 0;
        while (!eventQueue.isEmpty()) {
            Event event = eventQueue.poll();
            System.out.println("  Qayta ishlanmoqda: " + event);
            processed++;
        }

        System.out.println("Jami qayta ishlangan hodisalar: " + processed);
    }

    // Queue bilan paketli qayta ishlash
    public static void batchProcess(Queue<Integer> dataQueue, int batchSize) {
        System.out.println("Paketli qayta ishlash (paket hajmi: " + batchSize + "):");

        int batchNum = 1;
        while (!dataQueue.isEmpty()) {
            List<Integer> batch = new ArrayList<>();

            // Paket yig'amiz
            for (int i = 0; i < batchSize && !dataQueue.isEmpty(); i++) {
                batch.add(dataQueue.poll());
            }

            // Paketni qayta ishlaymiz
            System.out.println("  Paket " + batchNum + ": " + batch);
            batchNum++;
        }
    }

    public static void main(String[] args) {
        // BFS ni sinab ko'ramiz
        System.out.println("=== BFS O'tish ===");
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(7);

        List<Integer> bfsResult = bfsTraversal(root);
        System.out.println("BFS tartibi: " + bfsResult);

        List<List<Integer>> levelOrder = levelOrderTraversal(root);
        System.out.println("Darajalar bo'yicha: " + levelOrder);

        // Vazifa navbati qayta ishlashni sinab ko'ramiz
        System.out.println("\\n=== Vazifa Navbati Qayta Ishlash ===");
        Queue<Task> taskQueue = new LinkedList<>();
        taskQueue.offer(new Task("Initialize", 1));
        taskQueue.offer(new Task("LoadData", 2));
        taskQueue.offer(new Task("Process", 3));
        taskQueue.offer(new Task("Cleanup", 4));
        processTaskQueue(taskQueue);

        // Ishlab chiqaruvchi-Iste'molchini sinab ko'ramiz
        System.out.println("\\n=== Ishlab chiqaruvchi-Iste'molchi ===");
        producerConsumerDemo();

        // Hodisalarni qayta ishlashni sinab ko'ramiz
        System.out.println("\\n=== Hodisalarni Qayta Ishlash ===");
        Queue<Event> eventQueue = new LinkedList<>();
        eventQueue.offer(new Event("UserLogin"));
        eventQueue.offer(new Event("DataFetch"));
        eventQueue.offer(new Event("UpdateUI"));
        eventQueue.offer(new Event("UserLogout"));
        processEvents(eventQueue);

        // Paketli qayta ishlashni sinab ko'ramiz
        System.out.println("\\n=== Paketli Qayta Ishlash ===");
        Queue<Integer> dataQueue = new LinkedList<>();
        for (int i = 1; i <= 10; i++) {
            dataQueue.offer(i * 10);
        }
        batchProcess(dataQueue, 3);

        // FIFO tartib kafolati
        System.out.println("\\n=== FIFO Tartib Kafolati ===");
        Queue<String> fifoQueue = new LinkedList<>();
        fifoQueue.offer("First");
        fifoQueue.offer("Second");
        fifoQueue.offer("Third");
        System.out.println("Qo'shildi: First, Second, Third");
        System.out.println("Olindi: " + fifoQueue.poll() + ", " +
                          fifoQueue.poll() + ", " + fifoQueue.poll());
    }
}`,
            description: `Java da amaliy navbatni qayta ishlash naqshlarini o'rganing.

**Talablar:**
1. Queue yordamida BFS (kenglik bo'yicha qidiruv) amalga oshiring
2. Queue bilan oddiy Ishlab chiqaruvchi-Iste'molchi naqshini yarating
3. Vazifa navbati qayta ishlovchisini amalga oshiring
4. Queue yordamida hodisalarni tartib bilan qayta ishlang
5. Darajalar bo'yicha o'tish naqshini amalga oshiring
6. Queue asosidagi paketli qayta ishlashni ko'rsating
7. FIFO tartib kafolatini ko'rsating
8. Navbat to'lishi stsenariylarini boshqaring

Navbat asosidagi naqshlar BFS algoritmlari, hodisalarni qayta ishlash, vazifalarni rejalashtrish va tartiblangan ish jarayonlarini boshqarish uchun asosiy hisoblanadi.`,
            hint1: `BFS uchun tugunlarni daraja bo'yicha qayta ishlash uchun Queue dan foydalaning. Ildizni qo'shing, keyin tsiklda bolalarni oling va qo'shing. Darajalar bo'yicha o'tish uchun tugunlarni guruhlash uchun har bir darajada navbat hajmini kuzating.`,
            hint2: `Vazifa navbatlari FIFO ga amal qiladi: poll() eng eski vazifani oladi. Ishlab chiqaruvchi-iste'molchi uchun offer dan oldin sig'imni tekshiring. Paketli qayta ishlash uchun paket hajmi guruhlari bo'yicha poll qiling.`,
            whyItMatters: `Navbat naqshlari algoritmlar (BFS, topologik saralash), tizim dizayni (vazifa navbatlari, hodisalarni qayta ishlash) va tarqatilgan tizimlar (xabarlar navbatlari) uchun zarur. Bu naqshlarni o'zlashtirish samarali tartiblangan qayta ishlash va resurslarni boshqarish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Tartib kafolati bilan hodisalarni qayta ishlash
Queue<OrderEvent> eventQueue = new LinkedList<>();
eventQueue.offer(new OrderEvent("created", order));
eventQueue.offer(new OrderEvent("paid", order));
while (!eventQueue.isEmpty()) {
    OrderEvent event = eventQueue.poll();
    eventProcessor.process(event);  // FIFO tartibida qayta ishlash
}

// Eng qisqa yo'lni topish uchun BFS
Queue<Node> queue = new LinkedList<>();
Set<Node> visited = new HashSet<>();
queue.offer(startNode);
while (!queue.isEmpty()) {
    Node current = queue.poll();
    if (current.equals(target)) return getPath(current);
    for (Node neighbor : current.getNeighbors()) {
        if (!visited.contains(neighbor)) {
            queue.offer(neighbor);
            visited.add(neighbor);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Hodisalarni qayta ishlash tizimlari uchun FIFO tartib kafolati
- Graf algoritmlari uchun samarali BFS
- Samaradorlikni optimallashtirish uchun paketli qayta ishlash`
        }
    }
};

export default task;
