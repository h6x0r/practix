import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-virtual-vs-platform',
    title: 'Virtual vs Platform Threads',
    difficulty: 'medium',
    tags: ['java', 'virtual-threads', 'concurrency', 'performance', 'comparison'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Virtual vs Platform Threads

Understanding the differences between virtual and platform threads is crucial for choosing the right threading model. Virtual threads are lightweight and cheap to create, while platform threads are heavyweight OS threads with limited scalability.

## Requirements:
1. Compare resource consumption:
   1. Memory footprint per thread
   2. Creation overhead
   3. Context switching cost

2. Demonstrate scalability differences:
   1. Create 10,000 virtual threads
   2. Show platform thread limitations
   3. Compare execution time

3. Show blocking behavior:
   1. Blocking I/O simulation
   2. CPU-intensive tasks
   3. Thread parking and unparking

4. Analyze thread pool patterns:
   1. Platform threads with fixed pool
   2. Virtual threads without pooling
   3. Performance metrics

## Example Output:
\`\`\`
=== Platform Thread Limitations ===
Creating 1000 platform threads...
Platform threads created in 450ms
Memory per platform thread: ~1MB

=== Virtual Thread Scalability ===
Creating 10000 virtual threads...
Virtual threads created in 85ms
Memory per virtual thread: ~1KB
All 10000 tasks completed successfully

=== Blocking I/O Comparison ===
Platform threads (pool of 10): 5000ms
Virtual threads (10000): 1200ms
Virtual threads are 4.2x faster!

=== CPU-Intensive Tasks ===
Platform threads: 2300ms
Virtual threads: 2350ms
Similar performance for CPU-bound work
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class VirtualVsPlatform {
    public static void main(String[] args) throws InterruptedException {
        // TODO: Compare platform thread creation

        // TODO: Compare virtual thread creation

        // TODO: Demonstrate blocking I/O with both types

        // TODO: Compare CPU-intensive tasks
    }
}`,
    solutionCode: `import java.time.Duration;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class VirtualVsPlatform {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Platform Thread Limitations ===");
        platformThreadTest();

        System.out.println("\\n=== Virtual Thread Scalability ===");
        virtualThreadTest();

        System.out.println("\\n=== Blocking I/O Comparison ===");
        blockingIOComparison();

        System.out.println("\\n=== CPU-Intensive Tasks ===");
        cpuIntensiveComparison();
    }

    static void platformThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 1000;

        // Platform threads have OS thread overhead (~1MB stack each)
        System.out.println("Creating " + threadCount + " platform threads...");

        Thread[] threads = new Thread[threadCount];
        for (int i = 0; i < threadCount; i++) {
            threads[i] = Thread.ofPlatform().unstarted(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        for (Thread t : threads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Platform threads created in " + creationTime + "ms");
        System.out.println("Memory per platform thread: ~1MB");

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static void virtualThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 10000; // 10x more threads!

        System.out.println("Creating " + threadCount + " virtual threads...");

        CountDownLatch latch = new CountDownLatch(threadCount);

        // Virtual threads are very lightweight (~1KB each)
        for (int i = 0; i < threadCount; i++) {
            Thread.startVirtualThread(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Virtual threads created in " + creationTime + "ms");
        System.out.println("Memory per virtual thread: ~1KB");

        try {
            latch.await();
            System.out.println("All " + threadCount + " tasks completed successfully");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    static void blockingIOComparison() throws Exception {
        int taskCount = 10000;

        // Platform threads with fixed pool (limited scalability)
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(10)) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    try {
                        // Simulate blocking I/O
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        platformLatch.countDown();
                    }
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads (pool of 10): " + platformTime + "ms");

        // Virtual threads (no pooling needed!)
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    try {
                        // Simulate blocking I/O
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        virtualLatch.countDown();
                    }
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads (" + taskCount + "): " + virtualTime + "ms");

        double speedup = (double) platformTime / virtualTime;
        System.out.printf("Virtual threads are %.1fx faster!%n", speedup);
    }

    static void cpuIntensiveComparison() throws Exception {
        int taskCount = 100;
        int iterations = 1000000;

        // Platform threads
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors())) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    // CPU-intensive work
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    platformLatch.countDown();
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads: " + platformTime + "ms");

        // Virtual threads
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    // CPU-intensive work
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    virtualLatch.countDown();
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads: " + virtualTime + "ms");
        System.out.println("Similar performance for CPU-bound work");
    }
}`,
    hint1: `Platform threads are mapped 1:1 to OS threads and have significant overhead (~1MB stack). Virtual threads are lightweight user-mode threads scheduled by the JVM.`,
    hint2: `Virtual threads excel at I/O-bound tasks where threads spend time waiting. For CPU-intensive tasks, the performance is similar since both need actual CPU cores.`,
    whyItMatters: `Understanding when to use virtual vs platform threads is critical for application performance. Virtual threads dramatically improve scalability for I/O-bound applications, allowing thousands or millions of concurrent operations without the resource overhead of platform threads. This knowledge helps you make informed architectural decisions.

**Production Pattern:**
\`\`\`java
// Choosing the right thread type based on task
public class ThreadStrategySelector {
    // For I/O operations - use virtual threads
    private final ExecutorService ioExecutor =
        Executors.newVirtualThreadPerTaskExecutor();

    // For CPU-intensive tasks - use platform threads
    private final ExecutorService cpuExecutor =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public CompletableFuture<Data> fetchData(String url) {
        return CompletableFuture.supplyAsync(() -> httpClient.get(url), ioExecutor);
    }

    public CompletableFuture<Result> computeHeavy(Data data) {
        return CompletableFuture.supplyAsync(() -> algorithm.process(data), cpuExecutor);
    }
}
\`\`\`

**Practical Benefits:**
- Virtual threads: up to 10x improvement for I/O operations
- Platform threads: optimal for computational tasks
- Memory consumption reduced by 99% for high-load services
- Simplified architecture without complex pool tuning`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Test platform thread creation
class Test1 {
    @Test
    public void test() throws Exception {
        Thread pThread = Thread.ofPlatform().start(() -> {});
        pThread.join();
        assertFalse(pThread.isVirtual());
    }
}

// Test2: Test virtual thread is virtual
class Test2 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().start(() -> {});
        vThread.join();
        assertTrue(vThread.isVirtual());
    }
}

// Test3: Test platform thread is not virtual
class Test3 {
    @Test
    public void test() throws Exception {
        Thread pThread = new Thread(() -> {});
        pThread.start();
        pThread.join();
        assertFalse(pThread.isVirtual());
    }
}

// Test4: Test virtual thread lightweight
class Test4 {
    @Test
    public void test() throws Exception {
        int count = 100;
        java.util.concurrent.atomic.AtomicInteger completed = new java.util.concurrent.atomic.AtomicInteger(0);
        Thread[] threads = new Thread[count];
        for (int i = 0; i < count; i++) {
            threads[i] = Thread.ofVirtual().start(() -> {
                try { Thread.sleep(10); } catch (InterruptedException e) {}
                completed.incrementAndGet();
            });
        }
        for (Thread t : threads) t.join();
        assertEquals(100, completed.get());
    }
}

// Test5: Test platform thread with name
class Test5 {
    @Test
    public void test() throws Exception {
        Thread pThread = Thread.ofPlatform().name("platform-thread").start(() -> {});
        pThread.join();
        assertEquals("platform-thread", pThread.getName());
    }
}

// Test6: Test virtual thread factory
class Test6 {
    @Test
    public void test() throws Exception {
        var factory = Thread.ofVirtual().factory();
        Thread t = factory.newThread(() -> {});
        t.start();
        t.join();
        assertTrue(t.isVirtual());
    }
}

// Test7: Test platform thread factory
class Test7 {
    @Test
    public void test() throws Exception {
        var factory = Thread.ofPlatform().factory();
        Thread t = factory.newThread(() -> {});
        t.start();
        t.join();
        assertFalse(t.isVirtual());
    }
}

// Test8: Test virtual thread with work
class Test8 {
    @Test
    public void test() throws Exception {
        final int[] sum = {0};
        Thread vThread = Thread.ofVirtual().start(() -> {
            for (int i = 1; i <= 10; i++) sum[0] += i;
        });
        vThread.join();
        assertEquals(55, sum[0]);
    }
}

// Test9: Test multiple platform threads
class Test9 {
    @Test
    public void test() throws Exception {
        Thread t1 = Thread.ofPlatform().start(() -> {});
        Thread t2 = Thread.ofPlatform().start(() -> {});
        t1.join();
        t2.join();
        assertFalse(t1.isVirtual() || t2.isVirtual());
    }
}

// Test10: Test thread type comparison
class Test10 {
    @Test
    public void test() throws Exception {
        Thread vThread = Thread.ofVirtual().start(() -> {});
        Thread pThread = Thread.ofPlatform().start(() -> {});
        vThread.join();
        pThread.join();
        assertTrue(vThread.isVirtual());
        assertFalse(pThread.isVirtual());
    }
}
`,
    translations: {
        ru: {
            title: 'Виртуальные vs Платформенные потоки',
            solutionCode: `import java.time.Duration;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class VirtualVsPlatform {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Ограничения платформенных потоков ===");
        platformThreadTest();

        System.out.println("\\n=== Масштабируемость виртуальных потоков ===");
        virtualThreadTest();

        System.out.println("\\n=== Сравнение блокирующего I/O ===");
        blockingIOComparison();

        System.out.println("\\n=== CPU-интенсивные задачи ===");
        cpuIntensiveComparison();
    }

    static void platformThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 1000;

        // Платформенные потоки имеют накладные расходы ОС (~1МБ стека каждый)
        System.out.println("Creating " + threadCount + " platform threads...");

        Thread[] threads = new Thread[threadCount];
        for (int i = 0; i < threadCount; i++) {
            threads[i] = Thread.ofPlatform().unstarted(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        for (Thread t : threads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Platform threads created in " + creationTime + "ms");
        System.out.println("Memory per platform thread: ~1MB");

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static void virtualThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 10000; // В 10 раз больше потоков!

        System.out.println("Creating " + threadCount + " virtual threads...");

        CountDownLatch latch = new CountDownLatch(threadCount);

        // Виртуальные потоки очень легковесны (~1КБ каждый)
        for (int i = 0; i < threadCount; i++) {
            Thread.startVirtualThread(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Virtual threads created in " + creationTime + "ms");
        System.out.println("Memory per virtual thread: ~1KB");

        try {
            latch.await();
            System.out.println("All " + threadCount + " tasks completed successfully");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    static void blockingIOComparison() throws Exception {
        int taskCount = 10000;

        // Платформенные потоки с фиксированным пулом (ограниченная масштабируемость)
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(10)) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    try {
                        // Имитация блокирующего I/O
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        platformLatch.countDown();
                    }
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads (pool of 10): " + platformTime + "ms");

        // Виртуальные потоки (пулинг не нужен!)
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    try {
                        // Имитация блокирующего I/O
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        virtualLatch.countDown();
                    }
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads (" + taskCount + "): " + virtualTime + "ms");

        double speedup = (double) platformTime / virtualTime;
        System.out.printf("Virtual threads are %.1fx faster!%n", speedup);
    }

    static void cpuIntensiveComparison() throws Exception {
        int taskCount = 100;
        int iterations = 1000000;

        // Платформенные потоки
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors())) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    // CPU-интенсивная работа
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    platformLatch.countDown();
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads: " + platformTime + "ms");

        // Виртуальные потоки
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    // CPU-интенсивная работа
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    virtualLatch.countDown();
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads: " + virtualTime + "ms");
        System.out.println("Similar performance for CPU-bound work");
    }
}`,
            description: `# Виртуальные vs Платформенные потоки

Понимание различий между виртуальными и платформенными потоками критически важно для выбора правильной модели потоков. Виртуальные потоки легковесны и дешевы в создании, в то время как платформенные потоки являются тяжеловесными потоками ОС с ограниченной масштабируемостью.

## Требования:
1. Сравните потребление ресурсов:
   1. Объем памяти на поток
   2. Накладные расходы на создание
   3. Стоимость переключения контекста

2. Продемонстрируйте различия в масштабируемости:
   1. Создайте 10,000 виртуальных потоков
   2. Покажите ограничения платформенных потоков
   3. Сравните время выполнения

3. Покажите поведение блокировки:
   1. Имитация блокирующего I/O
   2. CPU-интенсивные задачи
   3. Парковка и распарковка потоков

4. Проанализируйте паттерны пулов потоков:
   1. Платформенные потоки с фиксированным пулом
   2. Виртуальные потоки без пулинга
   3. Метрики производительности

## Пример вывода:
\`\`\`
=== Platform Thread Limitations ===
Creating 1000 platform threads...
Platform threads created in 450ms
Memory per platform thread: ~1MB

=== Virtual Thread Scalability ===
Creating 10000 virtual threads...
Virtual threads created in 85ms
Memory per virtual thread: ~1KB
All 10000 tasks completed successfully

=== Blocking I/O Comparison ===
Platform threads (pool of 10): 5000ms
Virtual threads (10000): 1200ms
Virtual threads are 4.2x faster!

=== CPU-Intensive Tasks ===
Platform threads: 2300ms
Virtual threads: 2350ms
Similar performance for CPU-bound work
\`\`\``,
            hint1: `Платформенные потоки отображаются 1:1 на потоки ОС и имеют значительные накладные расходы (~1МБ стека). Виртуальные потоки - это легковесные потоки пользовательского режима, планируемые JVM.`,
            hint2: `Виртуальные потоки превосходны для задач, связанных с I/O, где потоки тратят время на ожидание. Для CPU-интенсивных задач производительность схожа, так как обоим нужны реальные ядра CPU.`,
            whyItMatters: `Понимание, когда использовать виртуальные или платформенные потоки, критически важно для производительности приложения. Виртуальные потоки значительно улучшают масштабируемость для приложений с I/O-операциями, позволяя тысячи или миллионы параллельных операций без накладных расходов на ресурсы платформенных потоков. Эти знания помогают принимать обоснованные архитектурные решения.

**Продакшен паттерн:**
\`\`\`java
// Выбор правильного типа потоков в зависимости от задачи
public class ThreadStrategySelector {
    // Для I/O операций - используйте виртуальные потоки
    private final ExecutorService ioExecutor =
        Executors.newVirtualThreadPerTaskExecutor();

    // Для CPU-интенсивных задач - используйте платформенные потоки
    private final ExecutorService cpuExecutor =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public CompletableFuture<Data> fetchData(String url) {
        return CompletableFuture.supplyAsync(() -> httpClient.get(url), ioExecutor);
    }

    public CompletableFuture<Result> computeHeavy(Data data) {
        return CompletableFuture.supplyAsync(() -> algorithm.process(data), cpuExecutor);
    }
}
\`\`\`

**Практические преимущества:**
- Виртуальные потоки: до 10x улучшение для I/O-операций
- Платформенные потоки: оптимальны для вычислительных задач
- Снижение потребления памяти до 99% для высоконагруженных сервисов
- Упрощенная архитектура без сложной настройки пулов`
        },
        uz: {
            title: `Virtual vs Platforma oqimlari`,
            solutionCode: `import java.time.Duration;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class VirtualVsPlatform {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Platforma oqimlari cheklovlari ===");
        platformThreadTest();

        System.out.println("\\n=== Virtual oqimlar miqyosliligi ===");
        virtualThreadTest();

        System.out.println("\\n=== Bloklovchi I/O taqqoslash ===");
        blockingIOComparison();

        System.out.println("\\n=== CPU-intensiv vazifalar ===");
        cpuIntensiveComparison();
    }

    static void platformThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 1000;

        // Platforma oqimlari OS oqim xarajatiga ega (~1MB stek har biri)
        System.out.println("Creating " + threadCount + " platform threads...");

        Thread[] threads = new Thread[threadCount];
        for (int i = 0; i < threadCount; i++) {
            threads[i] = Thread.ofPlatform().unstarted(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        for (Thread t : threads) {
            t.start();
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Platform threads created in " + creationTime + "ms");
        System.out.println("Memory per platform thread: ~1MB");

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static void virtualThreadTest() {
        long startTime = System.currentTimeMillis();
        int threadCount = 10000; // 10 marta ko'p oqimlar!

        System.out.println("Creating " + threadCount + " virtual threads...");

        CountDownLatch latch = new CountDownLatch(threadCount);

        // Virtual oqimlar juda yengil (~1KB har biri)
        for (int i = 0; i < threadCount; i++) {
            Thread.startVirtualThread(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }

        long creationTime = System.currentTimeMillis() - startTime;
        System.out.println("Virtual threads created in " + creationTime + "ms");
        System.out.println("Memory per virtual thread: ~1KB");

        try {
            latch.await();
            System.out.println("All " + threadCount + " tasks completed successfully");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    static void blockingIOComparison() throws Exception {
        int taskCount = 10000;

        // Qat'iy hovuz bilan platforma oqimlari (cheklangan miqyoslilik)
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(10)) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    try {
                        // Bloklovchi I/O taqlid
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        platformLatch.countDown();
                    }
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads (pool of 10): " + platformTime + "ms");

        // Virtual oqimlar (hovuz kerak emas!)
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    try {
                        // Bloklovchi I/O taqlid
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        virtualLatch.countDown();
                    }
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads (" + taskCount + "): " + virtualTime + "ms");

        double speedup = (double) platformTime / virtualTime;
        System.out.printf("Virtual threads are %.1fx faster!%n", speedup);
    }

    static void cpuIntensiveComparison() throws Exception {
        int taskCount = 100;
        int iterations = 1000000;

        // Platforma oqimlari
        long platformStart = System.currentTimeMillis();
        try (ExecutorService platformPool = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors())) {
            CountDownLatch platformLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                platformPool.submit(() -> {
                    // CPU-intensiv ish
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    platformLatch.countDown();
                });
            }

            platformLatch.await();
        }
        long platformTime = System.currentTimeMillis() - platformStart;
        System.out.println("Platform threads: " + platformTime + "ms");

        // Virtual oqimlar
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            CountDownLatch virtualLatch = new CountDownLatch(taskCount);

            for (int i = 0; i < taskCount; i++) {
                virtualPool.submit(() -> {
                    // CPU-intensiv ish
                    long sum = 0;
                    for (int j = 0; j < iterations; j++) {
                        sum += j;
                    }
                    virtualLatch.countDown();
                });
            }

            virtualLatch.await();
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual threads: " + virtualTime + "ms");
        System.out.println("Similar performance for CPU-bound work");
    }
}`,
            description: `# Virtual vs Platforma oqimlari

Virtual va platforma oqimlari o'rtasidagi farqlarni tushunish to'g'ri oqim modelini tanlash uchun juda muhimdir. Virtual oqimlar yengil va arzon yaratiladi, platforma oqimlari esa cheklangan miqyoslilikka ega og'ir OS oqimlaridir.

## Talablar:
1. Resurs sarfini solishtiring:
   1. Oqim uchun xotira hajmi
   2. Yaratish xarajatlari
   3. Kontekstni almashtirish narxi

2. Miqyoslilik farqlarini namoyish eting:
   1. 10,000 ta virtual oqim yarating
   2. Platforma oqimlari cheklovlarini ko'rsating
   3. Bajarilish vaqtini solishtiring

3. Bloklash xatti-harakatini ko'rsating:
   1. Bloklovchi I/O taqlid
   2. CPU-intensiv vazifalar
   3. Oqimlarni to'xtatish va davom ettirish

4. Oqim hovuzlari naqshlarini tahlil qiling:
   1. Qat'iy hovuz bilan platforma oqimlari
   2. Hovuzsiz virtual oqimlar
   3. Unumdorlik ko'rsatkichlari

## Chiqish namunasi:
\`\`\`
=== Platform Thread Limitations ===
Creating 1000 platform threads...
Platform threads created in 450ms
Memory per platform thread: ~1MB

=== Virtual Thread Scalability ===
Creating 10000 virtual threads...
Virtual threads created in 85ms
Memory per virtual thread: ~1KB
All 10000 tasks completed successfully

=== Blocking I/O Comparison ===
Platform threads (pool of 10): 5000ms
Virtual threads (10000): 1200ms
Virtual threads are 4.2x faster!

=== CPU-Intensive Tasks ===
Platform threads: 2300ms
Virtual threads: 2350ms
Similar performance for CPU-bound work
\`\`\``,
            hint1: `Platforma oqimlari OS oqimlariga 1:1 moslanadi va sezilarli xarajatlarga ega (~1MB stek). Virtual oqimlar JVM tomonidan rejalashtirilgan yengil foydalanuvchi rejimi oqimlaridir.`,
            hint2: `Virtual oqimlar I/O bilan bog'liq vazifalar uchun zo'r, bu yerda oqimlar kutish vaqtini sarflaydi. CPU-intensiv vazifalar uchun unumdorlik o'xshash, chunki ikkalasiga ham haqiqiy CPU yadrolari kerak.`,
            whyItMatters: `Virtual yoki platforma oqimlaridan qachon foydalanishni tushunish dastur unumdorligi uchun juda muhimdir. Virtual oqimlar I/O-bog'liq dasturlar uchun miqyoslilikni sezilarli darajada yaxshilaydi, platforma oqimlarining resurs xarajatlari bo'lmagan holda minglab yoki millionlab parallel operatsiyalarga ruxsat beradi. Bu bilim asoslangan arxitektura qarorlarini qabul qilishga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Vazifaga qarab to'g'ri oqim turini tanlash
public class ThreadStrategySelector {
    // I/O operatsiyalar uchun - virtual oqimlardan foydalaning
    private final ExecutorService ioExecutor =
        Executors.newVirtualThreadPerTaskExecutor();

    // CPU-intensiv vazifalar uchun - platforma oqimlaridan foydalaning
    private final ExecutorService cpuExecutor =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public CompletableFuture<Data> fetchData(String url) {
        return CompletableFuture.supplyAsync(() -> httpClient.get(url), ioExecutor);
    }

    public CompletableFuture<Result> computeHeavy(Data data) {
        return CompletableFuture.supplyAsync(() -> algorithm.process(data), cpuExecutor);
    }
}
\`\`\`

**Amaliy foydalari:**
- Virtual oqimlar: I/O-operatsiyalar uchun 10x gacha yaxshilanish
- Platforma oqimlari: hisoblash vazifalari uchun optimal
- Yuqori yuklamali servislar uchun xotira sarfini 99% gacha kamaytirish
- Oqim hovuzlarini murakkab sozlashsiz soddalashtirilgan arxitektura`
        }
    }
};

export default task;
