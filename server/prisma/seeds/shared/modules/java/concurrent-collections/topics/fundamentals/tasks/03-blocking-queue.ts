import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-blocking-queue',
    title: 'BlockingQueue Basics',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'blocking-queue', 'producer-consumer'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn BlockingQueue for producer-consumer patterns.

**Requirements:**
1. Create a LinkedBlockingQueue with capacity 5
2. Implement a producer thread that puts items using put()
3. Implement a consumer thread that takes items using take()
4. Demonstrate blocking behavior when queue is full/empty
5. Use offer() with timeout for non-blocking insertion
6. Use poll() with timeout for non-blocking retrieval
7. Check queue size and remaining capacity
8. Handle interruption gracefully

BlockingQueue automatically handles thread coordination, blocking producers when full and consumers when empty.`,
    initialCode: `import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class BlockingQueueDemo {
    public static void main(String[] args) {
        // Create a LinkedBlockingQueue with capacity 5

        // Implement producer thread using put()

        // Implement consumer thread using take()

        // Demonstrate offer() and poll() with timeout

        // Check queue size and remaining capacity
    }
}`,
    solutionCode: `import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class BlockingQueueDemo {
    public static void main(String[] args) throws InterruptedException {
        // Create a LinkedBlockingQueue with capacity 5
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(5);
        System.out.println("Created BlockingQueue with capacity: 5");

        // Implement producer thread using put()
        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    String item = "Item-" + i;
                    System.out.println("[Producer] Putting: " + item);
                    queue.put(item); // Blocks if queue is full
                    System.out.println("[Producer] Put successful: " + item +
                                     " (queue size: " + queue.size() + ")");
                    Thread.sleep(500); // Simulate production time
                }
                System.out.println("[Producer] Finished producing");
            } catch (InterruptedException e) {
                System.out.println("[Producer] Interrupted");
                Thread.currentThread().interrupt();
            }
        }, "Producer");

        // Implement consumer thread using take()
        Thread consumer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    System.out.println("[Consumer] Waiting to take...");
                    String item = queue.take(); // Blocks if queue is empty
                    System.out.println("[Consumer] Took: " + item +
                                     " (queue size: " + queue.size() + ")");
                    Thread.sleep(1000); // Simulate consumption time (slower than producer)
                }
                System.out.println("[Consumer] Finished consuming");
            } catch (InterruptedException e) {
                System.out.println("[Consumer] Interrupted");
                Thread.currentThread().interrupt();
            }
        }, "Consumer");

        // Start threads
        producer.start();
        Thread.sleep(100); // Let producer start first
        consumer.start();

        // Wait for threads to complete
        producer.join();
        consumer.join();

        // Demonstrate offer() and poll() with timeout
        System.out.println("");
        System.out.println("=== Testing offer() and poll() with timeout ===");

        // offer() - non-blocking, with timeout
        boolean offered = queue.offer("Timeout-Item-1", 1, TimeUnit.SECONDS);
        System.out.println("offer() with 1s timeout: " + offered);

        // Add items for poll() test
        queue.put("Poll-Item-1");
        queue.put("Poll-Item-2");

        // poll() - non-blocking, with timeout
        String polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("poll() with 1s timeout: " + polled);

        // poll() on empty queue (should timeout)
        queue.clear();
        System.out.println("Cleared queue, attempting poll with 1s timeout...");
        polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("poll() result (should be null): " + polled);

        // Check queue size and remaining capacity
        System.out.println("");
        System.out.println("=== Queue Status ===");
        queue.put("Status-1");
        queue.put("Status-2");
        queue.put("Status-3");

        System.out.println("Current size: " + queue.size());
        System.out.println("Remaining capacity: " + queue.remainingCapacity());
        System.out.println("Total capacity: " + (queue.size() + queue.remainingCapacity()));

        // Demonstrate blocking behavior
        System.out.println("");
        System.out.println("=== Demonstrating Blocking Behavior ===");
        BlockingQueue<String> smallQueue = new LinkedBlockingQueue<>(2);

        // Fill the queue
        smallQueue.put("Full-1");
        smallQueue.put("Full-2");
        System.out.println("Queue is now full (capacity: 2)");

        // Try offer without timeout (should return false immediately)
        boolean offerResult = smallQueue.offer("Full-3");
        System.out.println("offer() on full queue (no timeout): " + offerResult);

        // Try offer with timeout (should timeout and return false)
        System.out.println("Attempting offer() with 2s timeout on full queue...");
        offerResult = smallQueue.offer("Full-3", 2, TimeUnit.SECONDS);
        System.out.println("offer() result after timeout: " + offerResult);

        System.out.println("");
        System.out.println("BlockingQueue demo completed!");
    }
}`,
    hint1: `put() blocks until space is available, take() blocks until an element is available. Use these for guaranteed delivery.`,
    hint2: `offer(timeout) and poll(timeout) provide timed waiting. They return false/null if the timeout expires, allowing you to handle timeouts.`,
    whyItMatters: `BlockingQueue is the foundation of producer-consumer patterns in Java, automatically handling thread synchronization and preventing race conditions without explicit locks.

**Production Pattern:**
\`\`\`java
// Task processing pool with producers and consumers
public class TaskProcessor {
    private final BlockingQueue<Task> taskQueue = new LinkedBlockingQueue<>(100);
    private final List<Thread> workers = new ArrayList<>();

    public void start(int workerCount) {
        // Launch consumer threads
        for (int i = 0; i < workerCount; i++) {
            Thread worker = new Thread(() -> {
                while (!Thread.interrupted()) {
                    try {
                        Task task = taskQueue.take(); // Blocks when empty
                        task.process();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            workers.add(worker);
            worker.start();
        }
    }

    public boolean submitTask(Task task, long timeout, TimeUnit unit) {
        try {
            return taskQueue.offer(task, timeout, unit); // Timeout when full
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }
}
\`\`\`

**Practical Benefits:**
- Automatic producer-consumer coordination without manual wait/notify
- Backpressure handling when queue is full
- Ideal for thread pools, task processors, message queues`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Verify BlockingQueue interface exists
class Test1 {
    @Test
    public void testBlockingQueueInterfaceExists() {
        assertNotNull(BlockingQueue.class);
    }
}

// Test2: Verify ArrayBlockingQueue class exists
class Test2 {
    @Test
    public void testArrayBlockingQueueExists() {
        assertNotNull(ArrayBlockingQueue.class);
    }
}

// Test3: Verify LinkedBlockingQueue class exists
class Test3 {
    @Test
    public void testLinkedBlockingQueueExists() {
        assertNotNull(LinkedBlockingQueue.class);
    }
}

// Test4: Verify basic put and take operations
class Test4 {
    @Test
    public void testPutAndTake() throws InterruptedException {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);

        queue.put("item1");
        queue.put("item2");

        assertEquals("item1", queue.take());
        assertEquals("item2", queue.take());
    }
}

// Test5: Verify offer method
class Test5 {
    @Test
    public void testOffer() {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(2);

        assertTrue(queue.offer("item1"));
        assertTrue(queue.offer("item2"));
        assertFalse(queue.offer("item3")); // Queue is full
    }
}

// Test6: Verify poll method
class Test6 {
    @Test
    public void testPoll() {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);
        queue.offer("item1");

        assertEquals("item1", queue.poll());
        assertNull(queue.poll()); // Queue is empty
    }
}

// Test7: Verify producer-consumer pattern
class Test7 {
    @Test
    public void testProducerConsumer() throws InterruptedException {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(10);
        int[] sum = {0};

        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 5; i++) {
                    queue.put(i);
                }
            } catch (InterruptedException e) {
            }
        });

        Thread consumer = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    sum[0] += queue.take();
                }
            } catch (InterruptedException e) {
            }
        });

        producer.start();
        consumer.start();
        producer.join();
        consumer.join();

        assertEquals(15, sum[0]); // 1+2+3+4+5 = 15
    }
}

// Test8: Verify size method
class Test8 {
    @Test
    public void testSize() {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);
        queue.offer("A");
        queue.offer("B");
        queue.offer("C");

        assertEquals(3, queue.size());
    }
}

// Test9: Verify remainingCapacity method
class Test9 {
    @Test
    public void testRemainingCapacity() {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(5);
        queue.offer("A");
        queue.offer("B");

        assertEquals(3, queue.remainingCapacity());
    }
}

// Test10: Verify poll with timeout
class Test10 {
    @Test
    public void testPollWithTimeout() throws InterruptedException {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);

        // Poll empty queue with timeout
        String result = queue.poll(100, TimeUnit.MILLISECONDS);
        assertNull(result);

        // Add item and poll
        queue.offer("item");
        result = queue.poll(100, TimeUnit.MILLISECONDS);
        assertEquals("item", result);
    }
}`,
    order: 2,
    translations: {
        ru: {
            title: 'Основы BlockingQueue',
            solutionCode: `import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class BlockingQueueDemo {
    public static void main(String[] args) throws InterruptedException {
        // Создаем LinkedBlockingQueue с вместимостью 5
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(5);
        System.out.println("Создана BlockingQueue с вместимостью: 5");

        // Реализуем поток-производитель используя put()
        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    String item = "Item-" + i;
                    System.out.println("[Производитель] Помещаем: " + item);
                    queue.put(item); // Блокируется если очередь полна
                    System.out.println("[Производитель] Помещение успешно: " + item +
                                     " (размер очереди: " + queue.size() + ")");
                    Thread.sleep(500); // Имитируем время производства
                }
                System.out.println("[Производитель] Закончил производство");
            } catch (InterruptedException e) {
                System.out.println("[Производитель] Прерван");
                Thread.currentThread().interrupt();
            }
        }, "Producer");

        // Реализуем поток-потребитель используя take()
        Thread consumer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    System.out.println("[Потребитель] Ожидание извлечения...");
                    String item = queue.take(); // Блокируется если очередь пуста
                    System.out.println("[Потребитель] Извлечен: " + item +
                                     " (размер очереди: " + queue.size() + ")");
                    Thread.sleep(1000); // Имитируем время потребления (медленнее производителя)
                }
                System.out.println("[Потребитель] Закончил потребление");
            } catch (InterruptedException e) {
                System.out.println("[Потребитель] Прерван");
                Thread.currentThread().interrupt();
            }
        }, "Consumer");

        // Запускаем потоки
        producer.start();
        Thread.sleep(100); // Даем производителю стартовать первым
        consumer.start();

        // Ждем завершения потоков
        producer.join();
        consumer.join();

        // Демонстрируем offer() и poll() с таймаутом
        System.out.println("");
        System.out.println("=== Тестирование offer() и poll() с таймаутом ===");

        // offer() - неблокирующий, с таймаутом
        boolean offered = queue.offer("Timeout-Item-1", 1, TimeUnit.SECONDS);
        System.out.println("offer() с таймаутом 1с: " + offered);

        // Добавляем элементы для теста poll()
        queue.put("Poll-Item-1");
        queue.put("Poll-Item-2");

        // poll() - неблокирующий, с таймаутом
        String polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("poll() с таймаутом 1с: " + polled);

        // poll() на пустой очереди (должен истечь таймаут)
        queue.clear();
        System.out.println("Очищена очередь, попытка poll с таймаутом 1с...");
        polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("Результат poll() (должен быть null): " + polled);

        // Проверяем размер очереди и оставшуюся вместимость
        System.out.println("");
        System.out.println("=== Статус очереди ===");
        queue.put("Status-1");
        queue.put("Status-2");
        queue.put("Status-3");

        System.out.println("Текущий размер: " + queue.size());
        System.out.println("Оставшаяся вместимость: " + queue.remainingCapacity());
        System.out.println("Общая вместимость: " + (queue.size() + queue.remainingCapacity()));

        // Демонстрируем блокирующее поведение
        System.out.println("");
        System.out.println("=== Демонстрация блокирующего поведения ===");
        BlockingQueue<String> smallQueue = new LinkedBlockingQueue<>(2);

        // Заполняем очередь
        smallQueue.put("Full-1");
        smallQueue.put("Full-2");
        System.out.println("Очередь теперь полна (вместимость: 2)");

        // Пробуем offer без таймаута (должен вернуть false сразу)
        boolean offerResult = smallQueue.offer("Full-3");
        System.out.println("offer() на полной очереди (без таймаута): " + offerResult);

        // Пробуем offer с таймаутом (должен истечь таймаут и вернуть false)
        System.out.println("Попытка offer() с таймаутом 2с на полной очереди...");
        offerResult = smallQueue.offer("Full-3", 2, TimeUnit.SECONDS);
        System.out.println("Результат offer() после таймаута: " + offerResult);

        System.out.println("");
        System.out.println("Демо BlockingQueue завершено!");
    }
}`,
            description: `Изучите BlockingQueue для паттернов производитель-потребитель.

**Требования:**
1. Создайте LinkedBlockingQueue с вместимостью 5
2. Реализуйте поток-производитель, который помещает элементы используя put()
3. Реализуйте поток-потребитель, который извлекает элементы используя take()
4. Продемонстрируйте блокирующее поведение когда очередь полна/пуста
5. Используйте offer() с таймаутом для неблокирующей вставки
6. Используйте poll() с таймаутом для неблокирующего извлечения
7. Проверьте размер очереди и оставшуюся вместимость
8. Обработайте прерывание корректно

BlockingQueue автоматически обрабатывает координацию потоков, блокируя производителей когда полна и потребителей когда пуста.`,
            hint1: `put() блокируется пока не появится место, take() блокируется пока не появится элемент. Используйте их для гарантированной доставки.`,
            hint2: `offer(timeout) и poll(timeout) обеспечивают ожидание с таймаутом. Они возвращают false/null если таймаут истекает, позволяя обрабатывать таймауты.`,
            whyItMatters: `BlockingQueue - основа паттернов производитель-потребитель в Java, автоматически обрабатывая синхронизацию потоков и предотвращая состояния гонки без явных блокировок.

**Продакшен паттерн:**
\`\`\`java
// Пул обработки задач с производителями и потребителями
public class TaskProcessor {
    private final BlockingQueue<Task> taskQueue = new LinkedBlockingQueue<>(100);
    private final List<Thread> workers = new ArrayList<>();

    public void start(int workerCount) {
        // Запускаем потоки-потребители
        for (int i = 0; i < workerCount; i++) {
            Thread worker = new Thread(() -> {
                while (!Thread.interrupted()) {
                    try {
                        Task task = taskQueue.take(); // Блокируется когда пусто
                        task.process();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            workers.add(worker);
            worker.start();
        }
    }

    public boolean submitTask(Task task, long timeout, TimeUnit unit) {
        try {
            return taskQueue.offer(task, timeout, unit); // Тайм-аут когда полно
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }
}
\`\`\`

**Практические преимущества:**
- Автоматическая координация потоков без явных блокировок wait/notify
- Встроенная обработка переполнения и опустошения очереди
- Идеален для пулов потоков, обработки задач, конвейеров данных`
        },
        uz: {
            title: 'BlockingQueue Asoslari',
            solutionCode: `import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class BlockingQueueDemo {
    public static void main(String[] args) throws InterruptedException {
        // 5 sig'imli LinkedBlockingQueue yaratamiz
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(5);
        System.out.println("Sig'imi 5 bo'lgan BlockingQueue yaratildi");

        // put() yordamida ishlab chiqaruvchi oqimni amalga oshiramiz
        Thread producer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    String item = "Item-" + i;
                    System.out.println("[Ishlab chiqaruvchi] Qo'yilmoqda: " + item);
                    queue.put(item); // Navbat to'lsa bloklaydi
                    System.out.println("[Ishlab chiqaruvchi] Muvaffaqiyatli qo'yildi: " + item +
                                     " (navbat hajmi: " + queue.size() + ")");
                    Thread.sleep(500); // Ishlab chiqarish vaqtini simulyatsiya qilamiz
                }
                System.out.println("[Ishlab chiqaruvchi] Ishlab chiqarishni tugatdi");
            } catch (InterruptedException e) {
                System.out.println("[Ishlab chiqaruvchi] To'xtatildi");
                Thread.currentThread().interrupt();
            }
        }, "Producer");

        // take() yordamida iste'molchi oqimni amalga oshiramiz
        Thread consumer = new Thread(() -> {
            try {
                for (int i = 1; i <= 10; i++) {
                    System.out.println("[Iste'molchi] Olishni kutmoqda...");
                    String item = queue.take(); // Navbat bo'sh bo'lsa bloklaydi
                    System.out.println("[Iste'molchi] Olindi: " + item +
                                     " (navbat hajmi: " + queue.size() + ")");
                    Thread.sleep(1000); // Iste'mol vaqtini simulyatsiya qilamiz (ishlab chiqaruvchidan sekinroq)
                }
                System.out.println("[Iste'molchi] Iste'molni tugatdi");
            } catch (InterruptedException e) {
                System.out.println("[Iste'molchi] To'xtatildi");
                Thread.currentThread().interrupt();
            }
        }, "Consumer");

        // Oqimlarni ishga tushiramiz
        producer.start();
        Thread.sleep(100); // Ishlab chiqaruvchini birinchi bo'lib boshlashga ruxsat beramiz
        consumer.start();

        // Oqimlar tugashini kutamiz
        producer.join();
        consumer.join();

        // offer() va poll() ni vaqt chegarasi bilan ko'rsatamiz
        System.out.println("");
        System.out.println("=== offer() va poll() ni vaqt chegarasi bilan sinash ===");

        // offer() - bloklamaydigan, vaqt chegarasi bilan
        boolean offered = queue.offer("Timeout-Item-1", 1, TimeUnit.SECONDS);
        System.out.println("1s vaqt chegarasi bilan offer(): " + offered);

        // poll() sinovi uchun elementlarni qo'shamiz
        queue.put("Poll-Item-1");
        queue.put("Poll-Item-2");

        // poll() - bloklamaydigan, vaqt chegarasi bilan
        String polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("1s vaqt chegarasi bilan poll(): " + polled);

        // Bo'sh navbatda poll() (vaqt tugashi kerak)
        queue.clear();
        System.out.println("Navbat tozalandi, 1s vaqt chegarasi bilan poll() urinishi...");
        polled = queue.poll(1, TimeUnit.SECONDS);
        System.out.println("poll() natijasi (null bo'lishi kerak): " + polled);

        // Navbat hajmi va qolgan sig'imni tekshiramiz
        System.out.println("");
        System.out.println("=== Navbat holati ===");
        queue.put("Status-1");
        queue.put("Status-2");
        queue.put("Status-3");

        System.out.println("Joriy hajm: " + queue.size());
        System.out.println("Qolgan sig'im: " + queue.remainingCapacity());
        System.out.println("Umumiy sig'im: " + (queue.size() + queue.remainingCapacity()));

        // Bloklash xatti-harakatini ko'rsatamiz
        System.out.println("");
        System.out.println("=== Bloklash xatti-harakatini ko'rsatish ===");
        BlockingQueue<String> smallQueue = new LinkedBlockingQueue<>(2);

        // Navbatni to'ldiramiz
        smallQueue.put("Full-1");
        smallQueue.put("Full-2");
        System.out.println("Navbat endi to'ldi (sig'im: 2)");

        // Vaqt chegarasisiz offer() ni sinab ko'ramiz (darhol false qaytarishi kerak)
        boolean offerResult = smallQueue.offer("Full-3");
        System.out.println("To'liq navbatda offer() (vaqt chegarasisiz): " + offerResult);

        // Vaqt chegarasi bilan offer() ni sinab ko'ramiz (vaqt tugashi va false qaytarishi kerak)
        System.out.println("To'liq navbatda 2s vaqt chegarasi bilan offer() urinishi...");
        offerResult = smallQueue.offer("Full-3", 2, TimeUnit.SECONDS);
        System.out.println("Vaqt tugagandan keyin offer() natijasi: " + offerResult);

        System.out.println("");
        System.out.println("BlockingQueue demosi tugadi!");
    }
}`,
            description: `Ishlab chiqaruvchi-iste'molchi naqshlari uchun BlockingQueue ni o'rganing.

**Talablar:**
1. 5 sig'imli LinkedBlockingQueue yarating
2. put() yordamida elementlarni qo'yadigan ishlab chiqaruvchi oqimni amalga oshiring
3. take() yordamida elementlarni oladigan iste'molchi oqimni amalga oshiring
4. Navbat to'liq/bo'sh bo'lganda bloklash xatti-harakatini ko'rsating
5. Bloklamaydigan qo'shish uchun vaqt chegarasi bilan offer() dan foydalaning
6. Bloklamaydigan olish uchun vaqt chegarasi bilan poll() dan foydalaning
7. Navbat hajmi va qolgan sig'imni tekshiring
8. To'xtatishni to'g'ri boshqaring

BlockingQueue oqimlar koordinatsiyasini avtomatik boshqaradi, to'liq bo'lganda ishlab chiqaruvchilarni va bo'sh bo'lganda iste'molchilarni bloklaydi.`,
            hint1: `put() joy bo'lguncha bloklaydi, take() element paydo bo'lguncha bloklaydi. Kafolatlangan yetkazib berish uchun ulardan foydalaning.`,
            hint2: `offer(timeout) va poll(timeout) vaqt chegarasi bilan kutishni ta'minlaydi. Vaqt tugasa false/null qaytaradi va vaqt tugashini boshqarish imkonini beradi.`,
            whyItMatters: `BlockingQueue Java da ishlab chiqaruvchi-iste'molchi naqshlarining asosi bo'lib, aniq qulflar isizsiz oqimlar sinxronizatsiyasini avtomatik boshqaradi va poyga holatlarini oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ishlab chiqaruvchilar va iste'molchilar bilan vazifalarni qayta ishlash puli
public class TaskProcessor {
    private final BlockingQueue<Task> taskQueue = new LinkedBlockingQueue<>(100);
    private final List<Thread> workers = new ArrayList<>();

    public void start(int workerCount) {
        // Iste'molchi oqimlarni ishga tushiramiz
        for (int i = 0; i < workerCount; i++) {
            Thread worker = new Thread(() -> {
                while (!Thread.interrupted()) {
                    try {
                        Task task = taskQueue.take(); // Bo'sh bo'lganda bloklaydi
                        task.process();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            workers.add(worker);
            worker.start();
        }
    }

    public boolean submitTask(Task task, long timeout, TimeUnit unit) {
        try {
            return taskQueue.offer(task, timeout, unit); // To'liq bo'lganda vaqt tugaydi
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Aniq wait/notify qulflarsiz avtomatik oqimlar koordinatsiyasi
- O'rnatilgan navbat to'lishi va bo'shatilishini boshqarish
- Oqim pullari, vazifalarni qayta ishlash, ma'lumotlar quvurlari uchun ideal`
        }
    }
};

export default task;
