import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-virtual-executors',
    title: 'Virtual Thread Executors',
    difficulty: 'medium',
    tags: ['java', 'virtual-threads', 'executors', 'concurrency', 'thread-pool'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Virtual Thread Executors

Executors.newVirtualThreadPerTaskExecutor() creates an executor that launches a new virtual thread for each task. This eliminates the need for thread pooling and simplifies concurrent code while maintaining excellent performance.

## Requirements:
1. Create and use virtual thread executor:
   1. Executors.newVirtualThreadPerTaskExecutor()
   2. Submit tasks and get Futures
   3. Handle task completion

2. Compare with traditional thread pools:
   1. Fixed thread pool
   2. Cached thread pool
   3. Virtual thread executor

3. Demonstrate executor patterns:
   1. Submit multiple tasks
   2. Collect results with Future
   3. Handle exceptions
   4. Proper shutdown

4. Show scalability benefits:
   1. Handle thousands of concurrent tasks
   2. No thread pool tuning needed
   3. Simplified resource management

## Example Output:
\`\`\`
=== Virtual Thread Executor ===
Submitting 1000 tasks to virtual thread executor...
Task 0 completed with result: Result-0
Task 1 completed with result: Result-1
...
All 1000 tasks completed in 250ms

=== Executor Comparison ===
Fixed pool (10 threads): 5000ms
Cached pool: 1200ms
Virtual thread executor: 280ms

=== Future Results ===
Collected 100 results:
[Result-0, Result-1, Result-2, ...]
All futures completed successfully

=== Exception Handling ===
Task failed: Operation failed
Caught ExecutionException: Task error
Executor shutdown gracefully
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class VirtualExecutors {
    public static void main(String[] args) throws Exception {
        // TODO: Create virtual thread executor

        // TODO: Submit tasks and collect results

        // TODO: Compare with traditional executors

        // TODO: Handle exceptions properly

        // TODO: Shutdown executor
    }
}`,
    solutionCode: `import java.util.*;
import java.util.concurrent.*;
import java.time.Duration;

public class VirtualExecutors {
    public static void main(String[] args) throws Exception {
        virtualThreadExecutorDemo();
        executorComparison();
        futureResultsDemo();
        exceptionHandlingDemo();
    }

    static void virtualThreadExecutorDemo() throws Exception {
        System.out.println("=== Virtual Thread Executor ===");

        // Create virtual thread executor - no need for thread pool tuning!
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 1000;
            System.out.println("Submitting " + taskCount + " tasks to virtual thread executor...");

            long startTime = System.currentTimeMillis();
            List<Future<String>> futures = new ArrayList<>();

            // Submit tasks
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                Future<String> future = executor.submit(() -> {
                    // Simulate some work
                    Thread.sleep(Duration.ofMillis(50));
                    return "Result-" + taskId;
                });
                futures.add(future);
            }

            // Show first few results
            for (int i = 0; i < Math.min(2, futures.size()); i++) {
                System.out.println("Task " + i + " completed with result: " +
                    futures.get(i).get());
            }
            System.out.println("...");

            // Wait for all to complete
            for (Future<String> future : futures) {
                future.get();
            }

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("All " + taskCount + " tasks completed in " + elapsed + "ms");
        }
    }

    static void executorComparison() throws Exception {
        System.out.println("\\n=== Executor Comparison ===");
        int taskCount = 1000;

        // Fixed thread pool - limited parallelism
        long fixedStart = System.currentTimeMillis();
        try (ExecutorService fixedPool = Executors.newFixedThreadPool(10)) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(fixedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long fixedTime = System.currentTimeMillis() - fixedStart;
        System.out.println("Fixed pool (10 threads): " + fixedTime + "ms");

        // Cached thread pool - creates platform threads as needed
        long cachedStart = System.currentTimeMillis();
        try (ExecutorService cachedPool = Executors.newCachedThreadPool()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(cachedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long cachedTime = System.currentTimeMillis() - cachedStart;
        System.out.println("Cached pool: " + cachedTime + "ms");

        // Virtual thread executor - optimal for I/O-bound tasks
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(virtualPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual thread executor: " + virtualTime + "ms");
    }

    static void futureResultsDemo() throws Exception {
        System.out.println("\\n=== Future Results ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 100;
            List<Future<String>> futures = new ArrayList<>();

            // Submit tasks
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                futures.add(executor.submit(() -> {
                    Thread.sleep(Duration.ofMillis(10));
                    return "Result-" + taskId;
                }));
            }

            // Collect all results
            List<String> results = new ArrayList<>();
            for (Future<String> future : futures) {
                results.add(future.get());
            }

            System.out.println("Collected " + results.size() + " results:");
            System.out.println("[" + String.join(", ",
                results.subList(0, Math.min(3, results.size()))) + ", ...]");
            System.out.println("All futures completed successfully");
        }
    }

    static void exceptionHandlingDemo() throws Exception {
        System.out.println("\\n=== Exception Handling ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            // Submit task that throws exception
            Future<String> future = executor.submit(() -> {
                Thread.sleep(Duration.ofMillis(100));
                throw new RuntimeException("Task error");
            });

            try {
                future.get();
            } catch (ExecutionException e) {
                System.out.println("Caught ExecutionException: " + e.getCause().getMessage());
            }

            // Submit successful task
            Future<String> successFuture = executor.submit(() -> {
                return "Success";
            });

            System.out.println("Successful task result: " + successFuture.get());
        }

        System.out.println("Executor shutdown gracefully");
    }
}`,
    hint1: `Use Executors.newVirtualThreadPerTaskExecutor() to create an executor that automatically creates a virtual thread for each task. No need to configure pool sizes!`,
    hint2: `Virtual thread executors are best for I/O-bound tasks. The executor automatically shuts down when used with try-with-resources.`,
    whyItMatters: `Virtual thread executors simplify concurrent programming by eliminating the need for thread pool tuning and management. They provide excellent scalability for I/O-bound workloads without the complexity of traditional thread pools. This makes it easier to write correct, performant concurrent code.

**Production Pattern:**
\`\`\`java
// Enterprise pattern with resource management and error handling
@Service
public class AsyncTaskService {
    private final ExecutorService executor;
    private final MetricsRegistry metrics;

    public AsyncTaskService(MetricsRegistry metrics) {
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.metrics = metrics;
    }

    public <T> CompletableFuture<T> executeAsync(Callable<T> task) {
        return CompletableFuture.supplyAsync(() -> {
            long start = System.nanoTime();
            try {
                return task.call();
            } catch (Exception e) {
                metrics.incrementErrors();
                throw new CompletionException(e);
            } finally {
                metrics.recordDuration(System.nanoTime() - start);
            }
        }, executor);
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
    }
}
\`\`\`

**Practical Benefits:**
- Automatic thread lifecycle management
- Built-in error handling and monitoring
- Zero configuration - works optimally out of the box
- Easy integration with Spring/Jakarta EE`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Test newVirtualThreadPerTaskExecutor
class Test1 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Integer> future = executor.submit(() -> 42);
            assertEquals(Integer.valueOf(42), future.get());
        }
    }
}

// Test2: Test virtual executor with multiple tasks
class Test2 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Integer> f1 = executor.submit(() -> 10);
            Future<Integer> f2 = executor.submit(() -> 20);
            assertEquals(Integer.valueOf(30), f1.get() + f2.get());
        }
    }
}

// Test3: Test virtual executor auto-close
class Test3 {
    @Test
    public void test() throws Exception {
        ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        executor.submit(() -> {}).get();
        executor.close();
        assertTrue(executor.isShutdown());
    }
}

// Test4: Test invokeAll with virtual executor
class Test4 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var tasks = java.util.List.of(
                (Callable<Integer>) () -> 1,
                (Callable<Integer>) () -> 2,
                (Callable<Integer>) () -> 3
            );
            var results = executor.invokeAll(tasks);
            assertEquals(3, results.size());
        }
    }
}

// Test5: Test virtual executor with runnable
class Test5 {
    @Test
    public void test() throws Exception {
        final boolean[] executed = {false};
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            executor.submit(() -> executed[0] = true).get();
            assertTrue(executed[0]);
        }
    }
}

// Test6: Test many virtual threads via executor
class Test6 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new java.util.ArrayList<Future<Integer>>();
            for (int i = 0; i < 100; i++) {
                final int value = i;
                futures.add(executor.submit(() -> value));
            }
            for (var f : futures) f.get();
            assertEquals(100, futures.size());
        }
    }
}

// Test7: Test virtual executor with exception
class Test7 {
    @Test
    public void test() {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Integer> future = executor.submit(() -> {
                throw new RuntimeException("Test error");
            });
            try {
                future.get();
                fail("Should throw exception");
            } catch (ExecutionException e) {
                assertTrue(e.getCause() instanceof RuntimeException);
            }
        } catch (Exception e) {
            fail("Unexpected exception");
        }
    }
}

// Test8: Test executor submit with timeout
class Test8 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<String> future = executor.submit(() -> {
                Thread.sleep(50);
                return "Done";
            });
            assertEquals("Done", future.get(200, TimeUnit.MILLISECONDS));
        }
    }
}

// Test9: Test virtual executor completion
class Test9 {
    @Test
    public void test() throws Exception {
        final int[] count = {0};
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 10; i++) {
                executor.submit(() -> count[0]++).get();
            }
            assertEquals(10, count[0]);
        }
    }
}

// Test10: Test structured task scope pattern
class Test10 {
    @Test
    public void test() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var f1 = executor.submit(() -> "Hello");
            var f2 = executor.submit(() -> "World");
            String result = f1.get() + " " + f2.get();
            assertEquals("Hello World", result);
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Исполнители виртуальных потоков',
            solutionCode: `import java.util.*;
import java.util.concurrent.*;
import java.time.Duration;

public class VirtualExecutors {
    public static void main(String[] args) throws Exception {
        virtualThreadExecutorDemo();
        executorComparison();
        futureResultsDemo();
        exceptionHandlingDemo();
    }

    static void virtualThreadExecutorDemo() throws Exception {
        System.out.println("=== Исполнитель виртуальных потоков ===");

        // Создать исполнитель виртуальных потоков - не нужна настройка пула потоков!
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 1000;
            System.out.println("Submitting " + taskCount + " tasks to virtual thread executor...");

            long startTime = System.currentTimeMillis();
            List<Future<String>> futures = new ArrayList<>();

            // Отправить задачи
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                Future<String> future = executor.submit(() -> {
                    // Имитация работы
                    Thread.sleep(Duration.ofMillis(50));
                    return "Result-" + taskId;
                });
                futures.add(future);
            }

            // Показать первые несколько результатов
            for (int i = 0; i < Math.min(2, futures.size()); i++) {
                System.out.println("Task " + i + " completed with result: " +
                    futures.get(i).get());
            }
            System.out.println("...");

            // Дождаться завершения всех
            for (Future<String> future : futures) {
                future.get();
            }

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("All " + taskCount + " tasks completed in " + elapsed + "ms");
        }
    }

    static void executorComparison() throws Exception {
        System.out.println("\\n=== Сравнение исполнителей ===");
        int taskCount = 1000;

        // Фиксированный пул потоков - ограниченный параллелизм
        long fixedStart = System.currentTimeMillis();
        try (ExecutorService fixedPool = Executors.newFixedThreadPool(10)) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(fixedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long fixedTime = System.currentTimeMillis() - fixedStart;
        System.out.println("Fixed pool (10 threads): " + fixedTime + "ms");

        // Кешированный пул потоков - создает платформенные потоки по необходимости
        long cachedStart = System.currentTimeMillis();
        try (ExecutorService cachedPool = Executors.newCachedThreadPool()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(cachedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long cachedTime = System.currentTimeMillis() - cachedStart;
        System.out.println("Cached pool: " + cachedTime + "ms");

        // Исполнитель виртуальных потоков - оптимален для I/O-задач
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(virtualPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual thread executor: " + virtualTime + "ms");
    }

    static void futureResultsDemo() throws Exception {
        System.out.println("\\n=== Результаты Future ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 100;
            List<Future<String>> futures = new ArrayList<>();

            // Отправить задачи
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                futures.add(executor.submit(() -> {
                    Thread.sleep(Duration.ofMillis(10));
                    return "Result-" + taskId;
                }));
            }

            // Собрать все результаты
            List<String> results = new ArrayList<>();
            for (Future<String> future : futures) {
                results.add(future.get());
            }

            System.out.println("Collected " + results.size() + " results:");
            System.out.println("[" + String.join(", ",
                results.subList(0, Math.min(3, results.size()))) + ", ...]");
            System.out.println("All futures completed successfully");
        }
    }

    static void exceptionHandlingDemo() throws Exception {
        System.out.println("\\n=== Обработка исключений ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            // Отправить задачу, которая выбрасывает исключение
            Future<String> future = executor.submit(() -> {
                Thread.sleep(Duration.ofMillis(100));
                throw new RuntimeException("Task error");
            });

            try {
                future.get();
            } catch (ExecutionException e) {
                System.out.println("Caught ExecutionException: " + e.getCause().getMessage());
            }

            // Отправить успешную задачу
            Future<String> successFuture = executor.submit(() -> {
                return "Success";
            });

            System.out.println("Successful task result: " + successFuture.get());
        }

        System.out.println("Executor shutdown gracefully");
    }
}`,
            description: `# Исполнители виртуальных потоков

Executors.newVirtualThreadPerTaskExecutor() создает исполнитель, который запускает новый виртуальный поток для каждой задачи. Это устраняет необходимость в пулинге потоков и упрощает параллельный код, сохраняя при этом отличную производительность.

## Требования:
1. Создайте и используйте исполнитель виртуальных потоков:
   1. Executors.newVirtualThreadPerTaskExecutor()
   2. Отправьте задачи и получите Future
   3. Обработайте завершение задач

2. Сравните с традиционными пулами потоков:
   1. Фиксированный пул потоков
   2. Кешированный пул потоков
   3. Исполнитель виртуальных потоков

3. Продемонстрируйте паттерны исполнителя:
   1. Отправка нескольких задач
   2. Сбор результатов с Future
   3. Обработка исключений
   4. Правильное завершение работы

4. Покажите преимущества масштабируемости:
   1. Обработка тысяч параллельных задач
   2. Не нужна настройка пула потоков
   3. Упрощенное управление ресурсами

## Пример вывода:
\`\`\`
=== Virtual Thread Executor ===
Submitting 1000 tasks to virtual thread executor...
Task 0 completed with result: Result-0
Task 1 completed with result: Result-1
...
All 1000 tasks completed in 250ms

=== Executor Comparison ===
Fixed pool (10 threads): 5000ms
Cached pool: 1200ms
Virtual thread executor: 280ms

=== Future Results ===
Collected 100 results:
[Result-0, Result-1, Result-2, ...]
All futures completed successfully

=== Exception Handling ===
Task failed: Operation failed
Caught ExecutionException: Task error
Executor shutdown gracefully
\`\`\``,
            hint1: `Используйте Executors.newVirtualThreadPerTaskExecutor() для создания исполнителя, который автоматически создает виртуальный поток для каждой задачи. Не нужно настраивать размеры пула!`,
            hint2: `Исполнители виртуальных потоков лучше всего подходят для I/O-задач. Исполнитель автоматически завершает работу при использовании с try-with-resources.`,
            whyItMatters: `Исполнители виртуальных потоков упрощают параллельное программирование, устраняя необходимость в настройке и управлении пулом потоков. Они обеспечивают отличную масштабируемость для рабочих нагрузок, связанных с I/O, без сложности традиционных пулов потоков. Это упрощает написание корректного, производительного параллельного кода.

**Продакшен паттерн:**
\`\`\`java
// Корпоративный паттерн с управлением ресурсами и обработкой ошибок
@Service
public class AsyncTaskService {
    private final ExecutorService executor;
    private final MetricsRegistry metrics;

    public AsyncTaskService(MetricsRegistry metrics) {
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.metrics = metrics;
    }

    public <T> CompletableFuture<T> executeAsync(Callable<T> task) {
        return CompletableFuture.supplyAsync(() -> {
            long start = System.nanoTime();
            try {
                return task.call();
            } catch (Exception e) {
                metrics.incrementErrors();
                throw new CompletionException(e);
            } finally {
                metrics.recordDuration(System.nanoTime() - start);
            }
        }, executor);
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
    }
}
\`\`\`

**Практические преимущества:**
- Автоматическое управление жизненным циклом потоков
- Встроенная обработка ошибок и мониторинг
- Нулевая настройка - работает оптимально из коробки
- Простая интеграция с Spring/Jakarta EE`
        },
        uz: {
            title: `Virtual oqim ijrochilar`,
            solutionCode: `import java.util.*;
import java.util.concurrent.*;
import java.time.Duration;

public class VirtualExecutors {
    public static void main(String[] args) throws Exception {
        virtualThreadExecutorDemo();
        executorComparison();
        futureResultsDemo();
        exceptionHandlingDemo();
    }

    static void virtualThreadExecutorDemo() throws Exception {
        System.out.println("=== Virtual oqim ijrochisi ===");

        // Virtual oqim ijrochisini yaratish - oqim hovuzini sozlash kerak emas!
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 1000;
            System.out.println("Submitting " + taskCount + " tasks to virtual thread executor...");

            long startTime = System.currentTimeMillis();
            List<Future<String>> futures = new ArrayList<>();

            // Vazifalarni yuborish
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                Future<String> future = executor.submit(() -> {
                    // Ishni taqlid qilish
                    Thread.sleep(Duration.ofMillis(50));
                    return "Result-" + taskId;
                });
                futures.add(future);
            }

            // Dastlabki bir nechta natijalarni ko'rsatish
            for (int i = 0; i < Math.min(2, futures.size()); i++) {
                System.out.println("Task " + i + " completed with result: " +
                    futures.get(i).get());
            }
            System.out.println("...");

            // Hammasining tugashini kutish
            for (Future<String> future : futures) {
                future.get();
            }

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("All " + taskCount + " tasks completed in " + elapsed + "ms");
        }
    }

    static void executorComparison() throws Exception {
        System.out.println("\\n=== Ijrochilarni taqqoslash ===");
        int taskCount = 1000;

        // Qat'iy oqim hovuzi - cheklangan parallellik
        long fixedStart = System.currentTimeMillis();
        try (ExecutorService fixedPool = Executors.newFixedThreadPool(10)) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(fixedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long fixedTime = System.currentTimeMillis() - fixedStart;
        System.out.println("Fixed pool (10 threads): " + fixedTime + "ms");

        // Keshli oqim hovuzi - kerak bo'lganda platforma oqimlarini yaratadi
        long cachedStart = System.currentTimeMillis();
        try (ExecutorService cachedPool = Executors.newCachedThreadPool()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(cachedPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long cachedTime = System.currentTimeMillis() - cachedStart;
        System.out.println("Cached pool: " + cachedTime + "ms");

        // Virtual oqim ijrochisi - I/O vazifalar uchun optimal
        long virtualStart = System.currentTimeMillis();
        try (ExecutorService virtualPool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < taskCount; i++) {
                futures.add(virtualPool.submit(() -> {
                    try {
                        Thread.sleep(Duration.ofMillis(50));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        }
        long virtualTime = System.currentTimeMillis() - virtualStart;
        System.out.println("Virtual thread executor: " + virtualTime + "ms");
    }

    static void futureResultsDemo() throws Exception {
        System.out.println("\\n=== Future natijalari ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            int taskCount = 100;
            List<Future<String>> futures = new ArrayList<>();

            // Vazifalarni yuborish
            for (int i = 0; i < taskCount; i++) {
                final int taskId = i;
                futures.add(executor.submit(() -> {
                    Thread.sleep(Duration.ofMillis(10));
                    return "Result-" + taskId;
                }));
            }

            // Barcha natijalarni yig'ish
            List<String> results = new ArrayList<>();
            for (Future<String> future : futures) {
                results.add(future.get());
            }

            System.out.println("Collected " + results.size() + " results:");
            System.out.println("[" + String.join(", ",
                results.subList(0, Math.min(3, results.size()))) + ", ...]");
            System.out.println("All futures completed successfully");
        }
    }

    static void exceptionHandlingDemo() throws Exception {
        System.out.println("\\n=== Istisno boshqarish ===");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            // Istisno chiqaradigan vazifani yuborish
            Future<String> future = executor.submit(() -> {
                Thread.sleep(Duration.ofMillis(100));
                throw new RuntimeException("Task error");
            });

            try {
                future.get();
            } catch (ExecutionException e) {
                System.out.println("Caught ExecutionException: " + e.getCause().getMessage());
            }

            // Muvaffaqiyatli vazifani yuborish
            Future<String> successFuture = executor.submit(() -> {
                return "Success";
            });

            System.out.println("Successful task result: " + successFuture.get());
        }

        System.out.println("Executor shutdown gracefully");
    }
}`,
            description: `# Virtual oqim ijrochilar

Executors.newVirtualThreadPerTaskExecutor() har bir vazifa uchun yangi virtual oqim ishga tushiradigan ijrochini yaratadi. Bu oqim hovuzini talab qilmaydi va ajoyib unumdorlikni saqlab qolgan holda parallel kodni soddalashtiradi.

## Talablar:
1. Virtual oqim ijrochisini yarating va foydalaning:
   1. Executors.newVirtualThreadPerTaskExecutor()
   2. Vazifalarni yuboring va Future oling
   3. Vazifa tugashini boshqaring

2. An'anaviy oqim hovuzlari bilan solishtiring:
   1. Qat'iy oqim hovuzi
   2. Keshli oqim hovuzi
   3. Virtual oqim ijrochisi

3. Ijrochi naqshlarini namoyish eting:
   1. Bir nechta vazifalarni yuborish
   2. Future bilan natijalarni yig'ish
   3. Istisnolarni boshqarish
   4. To'g'ri to'xtatish

4. Miqyoslilik afzalliklarini ko'rsating:
   1. Minglab parallel vazifalarni boshqarish
   2. Oqim hovuzini sozlash kerak emas
   3. Soddalashtirilgan resurs boshqaruvi

## Chiqish namunasi:
\`\`\`
=== Virtual Thread Executor ===
Submitting 1000 tasks to virtual thread executor...
Task 0 completed with result: Result-0
Task 1 completed with result: Result-1
...
All 1000 tasks completed in 250ms

=== Executor Comparison ===
Fixed pool (10 threads): 5000ms
Cached pool: 1200ms
Virtual thread executor: 280ms

=== Future Results ===
Collected 100 results:
[Result-0, Result-1, Result-2, ...]
All futures completed successfully

=== Exception Handling ===
Task failed: Operation failed
Caught ExecutionException: Task error
Executor shutdown gracefully
\`\`\``,
            hint1: `Har bir vazifa uchun avtomatik ravishda virtual oqim yaratadigan ijrochini yaratish uchun Executors.newVirtualThreadPerTaskExecutor() dan foydalaning. Hovuz o'lchamlarini sozlash kerak emas!`,
            hint2: `Virtual oqim ijrochilari I/O vazifalar uchun eng yaxshi. Try-with-resources bilan foydalanganda ijrochi avtomatik ravishda to'xtaydi.`,
            whyItMatters: `Virtual oqim ijrochilar oqim hovuzini sozlash va boshqarish zaruriyatini yo'q qilish orqali parallel dasturlashni soddalashtiradi. Ular an'anaviy oqim hovuzlarining murakkabligi bo'lmagan holda I/O bog'liq ish yuklari uchun ajoyib miqyoslilikni ta'minlaydi. Bu to'g'ri, samarali parallel kodni yozishni osonlashtiradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Resurslarni boshqarish va xatolarni qayta ishlash bilan korporativ pattern
@Service
public class AsyncTaskService {
    private final ExecutorService executor;
    private final MetricsRegistry metrics;

    public AsyncTaskService(MetricsRegistry metrics) {
        this.executor = Executors.newVirtualThreadPerTaskExecutor();
        this.metrics = metrics;
    }

    public <T> CompletableFuture<T> executeAsync(Callable<T> task) {
        return CompletableFuture.supplyAsync(() -> {
            long start = System.nanoTime();
            try {
                return task.call();
            } catch (Exception e) {
                metrics.incrementErrors();
                throw new CompletionException(e);
            } finally {
                metrics.recordDuration(System.nanoTime() - start);
            }
        }, executor);
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
    }
}
\`\`\`

**Amaliy foydalari:**
- Oqimlarning hayotiy tsiklini avtomatik boshqarish
- O'rnatilgan xato qayta ishlash va monitoring
- Nol sozlash - qutidan optimal ishlaydi
- Spring/Jakarta EE bilan oson integratsiya`
        }
    }
};

export default task;
