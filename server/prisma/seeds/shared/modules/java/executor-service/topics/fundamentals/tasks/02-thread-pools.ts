import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-thread-pools',
    title: 'Thread Pool Types',
    difficulty: 'easy',
    tags: ['java', 'concurrency', 'thread-pool', 'executors', 'java5'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Thread Pool Types

The Executors class provides factory methods for creating different types of thread pools, each optimized for specific use cases. Understanding when to use each type is crucial for building efficient concurrent applications.

## Requirements:
1. Create and demonstrate different thread pool types:
   1. **FixedThreadPool**: Fixed number of threads, bounded queue
   2. **CachedThreadPool**: Creates threads as needed, reuses idle threads
   3. **SingleThreadExecutor**: Single worker thread, sequential execution
   4. **ScheduledThreadPool**: For delayed and periodic task execution

2. For each pool type, demonstrate:
   1. How tasks are distributed among threads
   2. Thread creation and reuse behavior
   3. Appropriate use cases

3. Submit multiple tasks to each pool and observe behavior

4. Compare performance characteristics of different pools

## Example Output:
\`\`\`
=== Fixed Thread Pool (3 threads) ===
Task 1 on pool-1-thread-1
Task 2 on pool-1-thread-2
Task 3 on pool-1-thread-3
Task 4 on pool-1-thread-1 (reused)

=== Cached Thread Pool ===
Task 1 on pool-2-thread-1
Task 2 on pool-2-thread-2
Task 3 on pool-2-thread-3
(Creates new threads as needed)

=== Single Thread Executor ===
Task 1 on pool-3-thread-1
Task 2 on pool-3-thread-1 (sequential)
Task 3 on pool-3-thread-1 (sequential)

=== Scheduled Thread Pool ===
Scheduled task executed after delay
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class ThreadPools {
    public static void main(String[] args) {
        // TODO: Demonstrate fixed thread pool

        // TODO: Demonstrate cached thread pool

        // TODO: Demonstrate single thread executor

        // TODO: Demonstrate scheduled thread pool

        // TODO: Properly shutdown all executors
    }
}`,
    solutionCode: `import java.util.concurrent.*;

public class ThreadPools {
    public static void main(String[] args) throws InterruptedException {
        // Fixed Thread Pool - fixed number of threads
        System.out.println("=== Fixed Thread Pool (3 threads) ===");
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            fixedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(500);
            });
        }

        fixedPool.shutdown();
        fixedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Cached Thread Pool - creates threads as needed, reuses idle
        System.out.println("\\n=== Cached Thread Pool ===");
        ExecutorService cachedPool = Executors.newCachedThreadPool();

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            cachedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(100);
            });
        }

        cachedPool.shutdown();
        cachedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Single Thread Executor - only one thread, sequential execution
        System.out.println("\\n=== Single Thread Executor ===");
        ExecutorService singleExecutor = Executors.newSingleThreadExecutor();

        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            singleExecutor.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName() + " (sequential)");
                sleep(300);
            });
        }

        singleExecutor.shutdown();
        singleExecutor.awaitTermination(3, TimeUnit.SECONDS);

        // Scheduled Thread Pool - for delayed and periodic tasks
        System.out.println("\\n=== Scheduled Thread Pool ===");
        ScheduledExecutorService scheduledPool =
            Executors.newScheduledThreadPool(2);

        // Schedule task with delay
        scheduledPool.schedule(() -> {
            System.out.println("Delayed task executed on " +
                Thread.currentThread().getName());
        }, 1, TimeUnit.SECONDS);

        // Schedule periodic task
        scheduledPool.scheduleAtFixedRate(() -> {
            System.out.println("Periodic task on " +
                Thread.currentThread().getName());
        }, 0, 500, TimeUnit.MILLISECONDS);

        Thread.sleep(2000);
        scheduledPool.shutdown();

        System.out.println("\\n=== Pool Characteristics ===");
        System.out.println("FixedThreadPool: Best for known workload, bounded resources");
        System.out.println("CachedThreadPool: Best for many short-lived tasks");
        System.out.println("SingleThreadExecutor: Best for sequential execution");
        System.out.println("ScheduledThreadPool: Best for delayed/periodic tasks");
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}`,
    hint1: `Use newFixedThreadPool(n) for a bounded number of threads, newCachedThreadPool() for variable workloads, newSingleThreadExecutor() for sequential tasks, and newScheduledThreadPool(n) for delayed/periodic execution.`,
    hint2: `Fixed pools have bounded queues and reuse threads. Cached pools create threads on demand and terminate idle threads after 60 seconds. Single thread executors guarantee sequential execution order.`,
    whyItMatters: `Choosing the right thread pool type is critical for application performance and resource management. Fixed pools prevent resource exhaustion, cached pools optimize for throughput, single thread executors ensure ordering, and scheduled pools enable time-based execution. Understanding these trade-offs helps build efficient concurrent systems.

**Production pattern:**
\`\`\`java
@Configuration
public class ThreadPoolConfig {
    // FixedThreadPool for CPU-intensive tasks
    @Bean(name = "cpuBoundExecutor")
    public ExecutorService cpuBoundExecutor() {
        int cores = Runtime.getRuntime().availableProcessors();
        return new ThreadPoolExecutor(cores, cores,
            0L, TimeUnit.MILLISECONDS,
            new LinkedBlockingQueue<>(100),
            new ThreadFactoryBuilder().setNameFormat("cpu-%d").build(),
            new ThreadPoolExecutor.CallerRunsPolicy());
    }

    // CachedThreadPool for I/O operations
    @Bean(name = "ioBoundExecutor")
    public ExecutorService ioBoundExecutor() {
        return Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("io-%d").build());
    }

    // SingleThreadExecutor for ordered operations
    @Bean(name = "orderedExecutor")
    public ExecutorService orderedExecutor() {
        return Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setNameFormat("ordered-%d").build());
    }

    // ScheduledThreadPool for periodic tasks
    @Bean(name = "scheduledExecutor")
    public ScheduledExecutorService scheduledExecutor() {
        return Executors.newScheduledThreadPool(2,
            new ThreadFactoryBuilder().setNameFormat("scheduled-%d").build());
    }
}
\`\`\`

**Practical benefits:**
- Optimization for different workload types
- Customization of rejection policies and queues
- Monitoring and lifecycle management of pools`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Verify newFixedThreadPool method exists
class Test1 {
    @Test
    public void testNewFixedThreadPool() {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        assertNotNull(executor);
        executor.shutdown();
    }
}

// Test2: Verify newCachedThreadPool method exists
class Test2 {
    @Test
    public void testNewCachedThreadPool() {
        ExecutorService executor = Executors.newCachedThreadPool();
        assertNotNull(executor);
        executor.shutdown();
    }
}

// Test3: Verify newScheduledThreadPool method exists
class Test3 {
    @Test
    public void testNewScheduledThreadPool() {
        ScheduledExecutorService executor = Executors.newScheduledThreadPool(2);
        assertNotNull(executor);
        executor.shutdown();
    }
}

// Test4: Verify ThreadPoolExecutor class exists
class Test4 {
    @Test
    public void testThreadPoolExecutorClassExists() {
        assertNotNull(ThreadPoolExecutor.class);
    }
}

// Test5: Verify pool size with fixed thread pool
class Test5 {
    @Test
    public void testFixedThreadPoolSize() throws InterruptedException {
        int poolSize = 3;
        ExecutorService executor = Executors.newFixedThreadPool(poolSize);
        assertNotNull(executor);
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);
    }
}

// Test6: Verify getCorePoolSize method exists
class Test6 {
    @Test
    public void testGetCorePoolSizeMethodExists() throws NoSuchMethodException {
        assertNotNull(ThreadPoolExecutor.class.getMethod("getCorePoolSize"));
    }
}

// Test7: Verify getMaximumPoolSize method exists
class Test7 {
    @Test
    public void testGetMaximumPoolSizeMethodExists() throws NoSuchMethodException {
        assertNotNull(ThreadPoolExecutor.class.getMethod("getMaximumPoolSize"));
    }
}

// Test8: Verify getActiveCount method exists
class Test8 {
    @Test
    public void testGetActiveCountMethodExists() throws NoSuchMethodException {
        assertNotNull(ThreadPoolExecutor.class.getMethod("getActiveCount"));
    }
}

// Test9: Verify ThreadFactory interface exists
class Test9 {
    @Test
    public void testThreadFactoryInterfaceExists() {
        assertNotNull(ThreadFactory.class);
    }
}

// Test10: Verify multiple tasks execution with fixed pool
class Test10 {
    @Test
    public void testMultipleTasksExecution() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        int[] counter = {0};

        for (int i = 0; i < 5; i++) {
            executor.submit(() -> {
                synchronized(counter) {
                    counter[0]++;
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(2, TimeUnit.SECONDS);

        assertEquals(5, counter[0]);
    }
}`,
    order: 2,
    translations: {
        ru: {
            title: 'Типы пулов потоков',
            solutionCode: `import java.util.concurrent.*;

public class ThreadPools {
    public static void main(String[] args) throws InterruptedException {
        // Фиксированный пул потоков - фиксированное количество потоков
        System.out.println("=== Фиксированный пул потоков (3 потока) ===");
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            fixedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(500);
            });
        }

        fixedPool.shutdown();
        fixedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Кэшированный пул потоков - создает потоки по необходимости, переиспользует простаивающие
        System.out.println("\\n=== Кэшированный пул потоков ===");
        ExecutorService cachedPool = Executors.newCachedThreadPool();

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            cachedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(100);
            });
        }

        cachedPool.shutdown();
        cachedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Однопоточный исполнитель - только один поток, последовательное выполнение
        System.out.println("\\n=== Однопоточный исполнитель ===");
        ExecutorService singleExecutor = Executors.newSingleThreadExecutor();

        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            singleExecutor.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName() + " (sequential)");
                sleep(300);
            });
        }

        singleExecutor.shutdown();
        singleExecutor.awaitTermination(3, TimeUnit.SECONDS);

        // Запланированный пул потоков - для отложенных и периодических задач
        System.out.println("\\n=== Запланированный пул потоков ===");
        ScheduledExecutorService scheduledPool =
            Executors.newScheduledThreadPool(2);

        // Запланировать задачу с задержкой
        scheduledPool.schedule(() -> {
            System.out.println("Delayed task executed on " +
                Thread.currentThread().getName());
        }, 1, TimeUnit.SECONDS);

        // Запланировать периодическую задачу
        scheduledPool.scheduleAtFixedRate(() -> {
            System.out.println("Periodic task on " +
                Thread.currentThread().getName());
        }, 0, 500, TimeUnit.MILLISECONDS);

        Thread.sleep(2000);
        scheduledPool.shutdown();

        System.out.println("\\n=== Характеристики пулов ===");
        System.out.println("FixedThreadPool: Best for known workload, bounded resources");
        System.out.println("CachedThreadPool: Best for many short-lived tasks");
        System.out.println("SingleThreadExecutor: Best for sequential execution");
        System.out.println("ScheduledThreadPool: Best for delayed/periodic tasks");
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# Типы пулов потоков

Класс Executors предоставляет фабричные методы для создания различных типов пулов потоков, каждый из которых оптимизирован для конкретных случаев использования. Понимание того, когда использовать каждый тип, имеет решающее значение для создания эффективных параллельных приложений.

## Требования:
1. Создайте и продемонстрируйте различные типы пулов потоков:
   1. **FixedThreadPool**: Фиксированное количество потоков, ограниченная очередь
   2. **CachedThreadPool**: Создает потоки по мере необходимости, переиспользует простаивающие потоки
   3. **SingleThreadExecutor**: Один рабочий поток, последовательное выполнение
   4. **ScheduledThreadPool**: Для отложенного и периодического выполнения задач

2. Для каждого типа пула продемонстрируйте:
   1. Как задачи распределяются между потоками
   2. Поведение создания и переиспользования потоков
   3. Подходящие случаи использования

3. Отправьте несколько задач в каждый пул и наблюдайте за поведением

4. Сравните характеристики производительности различных пулов

## Пример вывода:
\`\`\`
=== Fixed Thread Pool (3 threads) ===
Task 1 on pool-1-thread-1
Task 2 on pool-1-thread-2
Task 3 on pool-1-thread-3
Task 4 on pool-1-thread-1 (reused)

=== Cached Thread Pool ===
Task 1 on pool-2-thread-1
Task 2 on pool-2-thread-2
Task 3 on pool-2-thread-3
(Creates new threads as needed)

=== Single Thread Executor ===
Task 1 on pool-3-thread-1
Task 2 on pool-3-thread-1 (sequential)
Task 3 on pool-3-thread-1 (sequential)

=== Scheduled Thread Pool ===
Scheduled task executed after delay
\`\`\``,
            hint1: `Используйте newFixedThreadPool(n) для ограниченного количества потоков, newCachedThreadPool() для переменных нагрузок, newSingleThreadExecutor() для последовательных задач и newScheduledThreadPool(n) для отложенного/периодического выполнения.`,
            hint2: `Фиксированные пулы имеют ограниченные очереди и переиспользуют потоки. Кэшированные пулы создают потоки по требованию и завершают простаивающие потоки через 60 секунд. Однопоточные исполнители гарантируют порядок последовательного выполнения.`,
            whyItMatters: `Выбор правильного типа пула потоков критически важен для производительности приложения и управления ресурсами. Фиксированные пулы предотвращают исчерпание ресурсов, кэшированные пулы оптимизируют пропускную способность, однопоточные исполнители обеспечивают упорядоченность, а запланированные пулы обеспечивают выполнение по времени. Понимание этих компромиссов помогает строить эффективные параллельные системы.

**Продакшен паттерн:**
\`\`\`java
@Configuration
public class ThreadPoolConfig {
    // FixedThreadPool для CPU-интенсивных задач
    @Bean(name = "cpuBoundExecutor")
    public ExecutorService cpuBoundExecutor() {
        int cores = Runtime.getRuntime().availableProcessors();
        return new ThreadPoolExecutor(cores, cores,
            0L, TimeUnit.MILLISECONDS,
            new LinkedBlockingQueue<>(100),
            new ThreadFactoryBuilder().setNameFormat("cpu-%d").build(),
            new ThreadPoolExecutor.CallerRunsPolicy());
    }

    // CachedThreadPool для I/O операций
    @Bean(name = "ioBoundExecutor")
    public ExecutorService ioBoundExecutor() {
        return Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("io-%d").build());
    }

    // SingleThreadExecutor для упорядоченных операций
    @Bean(name = "orderedExecutor")
    public ExecutorService orderedExecutor() {
        return Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setNameFormat("ordered-%d").build());
    }

    // ScheduledThreadPool для периодических задач
    @Bean(name = "scheduledExecutor")
    public ScheduledExecutorService scheduledExecutor() {
        return Executors.newScheduledThreadPool(2,
            new ThreadFactoryBuilder().setNameFormat("scheduled-%d").build());
    }
}
\`\`\`

**Практические преимущества:**
- Оптимизация для разных типов рабочих нагрузок
- Настройка политик отклонения и очередей
- Мониторинг и управление жизненным циклом пулов`
        },
        uz: {
            title: `Thread pool turlari`,
            solutionCode: `import java.util.concurrent.*;

public class ThreadPools {
    public static void main(String[] args) throws InterruptedException {
        // Fiksirlangan thread pool - fiksirlangan sondagi threadlar
        System.out.println("=== Fiksirlangan thread pool (3 ta thread) ===");
        ExecutorService fixedPool = Executors.newFixedThreadPool(3);

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            fixedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(500);
            });
        }

        fixedPool.shutdown();
        fixedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Keshirovchi thread pool - kerak bo'lganda threadlar yaratadi, bo'sh threadlarni qayta ishlatadi
        System.out.println("\\n=== Keshirovchi thread pool ===");
        ExecutorService cachedPool = Executors.newCachedThreadPool();

        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            cachedPool.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName());
                sleep(100);
            });
        }

        cachedPool.shutdown();
        cachedPool.awaitTermination(3, TimeUnit.SECONDS);

        // Bitta threadli executor - faqat bitta thread, ketma-ket bajarish
        System.out.println("\\n=== Bitta threadli executor ===");
        ExecutorService singleExecutor = Executors.newSingleThreadExecutor();

        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            singleExecutor.execute(() -> {
                System.out.println("Task " + taskId + " on " +
                    Thread.currentThread().getName() + " (sequential)");
                sleep(300);
            });
        }

        singleExecutor.shutdown();
        singleExecutor.awaitTermination(3, TimeUnit.SECONDS);

        // Rejalashtirilgan thread pool - kechiktirilgan va davriy vazifalar uchun
        System.out.println("\\n=== Rejalashtirilgan thread pool ===");
        ScheduledExecutorService scheduledPool =
            Executors.newScheduledThreadPool(2);

        // Kechikish bilan vazifani rejalashtirish
        scheduledPool.schedule(() -> {
            System.out.println("Delayed task executed on " +
                Thread.currentThread().getName());
        }, 1, TimeUnit.SECONDS);

        // Davriy vazifani rejalashtirish
        scheduledPool.scheduleAtFixedRate(() -> {
            System.out.println("Periodic task on " +
                Thread.currentThread().getName());
        }, 0, 500, TimeUnit.MILLISECONDS);

        Thread.sleep(2000);
        scheduledPool.shutdown();

        System.out.println("\\n=== Pool xususiyatlari ===");
        System.out.println("FixedThreadPool: Best for known workload, bounded resources");
        System.out.println("CachedThreadPool: Best for many short-lived tasks");
        System.out.println("SingleThreadExecutor: Best for sequential execution");
        System.out.println("ScheduledThreadPool: Best for delayed/periodic tasks");
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# Thread pool turlari

Executors klassi turli xil thread poollarni yaratish uchun fabrika metodlarini taqdim etadi, ularning har biri ma'lum holatlarda ishlatish uchun optimallashtirilgan. Har bir turni qachon ishlatishni tushunish samarali parallel dasturlarni yaratish uchun juda muhimdir.

## Talablar:
1. Turli thread pool turlarini yarating va ko'rsating:
   1. **FixedThreadPool**: Fiksirlangan sondagi threadlar, chegaralangan navbat
   2. **CachedThreadPool**: Kerak bo'lganda threadlar yaratadi, bo'sh threadlarni qayta ishlatadi
   3. **SingleThreadExecutor**: Bitta ishchi thread, ketma-ket bajarish
   4. **ScheduledThreadPool**: Kechiktirilgan va davriy vazifalarni bajarish uchun

2. Har bir pool turi uchun ko'rsating:
   1. Vazifalar threadlar o'rtasida qanday taqsimlanadi
   2. Threadlarni yaratish va qayta ishlatish xatti-harakati
   3. Mos foydalanish holatlari

3. Har bir poolga bir nechta vazifa yuboring va xatti-harakatni kuzating

4. Turli poollarning unumdorlik xususiyatlarini solishtiring

## Chiqish namunasi:
\`\`\`
=== Fixed Thread Pool (3 threads) ===
Task 1 on pool-1-thread-1
Task 2 on pool-1-thread-2
Task 3 on pool-1-thread-3
Task 4 on pool-1-thread-1 (reused)

=== Cached Thread Pool ===
Task 1 on pool-2-thread-1
Task 2 on pool-2-thread-2
Task 3 on pool-2-thread-3
(Creates new threads as needed)

=== Single Thread Executor ===
Task 1 on pool-3-thread-1
Task 2 on pool-3-thread-1 (sequential)
Task 3 on pool-3-thread-1 (sequential)

=== Scheduled Thread Pool ===
Scheduled task executed after delay
\`\`\``,
            hint1: `Chegaralangan sondagi threadlar uchun newFixedThreadPool(n), o'zgaruvchan yuklamalar uchun newCachedThreadPool(), ketma-ket vazifalar uchun newSingleThreadExecutor() va kechiktirilgan/davriy bajarish uchun newScheduledThreadPool(n) dan foydalaning.`,
            hint2: `Fiksirlangan poollar chegaralangan navbatlarga ega va threadlarni qayta ishlatadilar. Keshirovchi poollar talab bo'yicha threadlar yaratadi va 60 soniyadan keyin bo'sh threadlarni tugatadi. Bitta threadli executorlar ketma-ket bajarish tartibini kafolatlaydi.`,
            whyItMatters: `To'g'ri thread pool turini tanlash dastur unumdorligi va resurslarni boshqarish uchun juda muhimdir. Fiksirlangan poollar resurslarning tugashini oldini oladi, keshirovchi poollar o'tkazuvchanlikni optimallashtiradi, bitta threadli executorlar tartibni ta'minlaydi va rejalashtirilgan poollar vaqt bo'yicha bajarishni ta'minlaydi. Bu murosalarni tushunish samarali parallel tizimlar qurishga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Configuration
public class ThreadPoolConfig {
    // CPU-intensiv vazifalar uchun FixedThreadPool
    @Bean(name = "cpuBoundExecutor")
    public ExecutorService cpuBoundExecutor() {
        int cores = Runtime.getRuntime().availableProcessors();
        return new ThreadPoolExecutor(cores, cores,
            0L, TimeUnit.MILLISECONDS,
            new LinkedBlockingQueue<>(100),
            new ThreadFactoryBuilder().setNameFormat("cpu-%d").build(),
            new ThreadPoolExecutor.CallerRunsPolicy());
    }

    // I/O operatsiyalari uchun CachedThreadPool
    @Bean(name = "ioBoundExecutor")
    public ExecutorService ioBoundExecutor() {
        return Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("io-%d").build());
    }

    // Tartiblangan operatsiyalar uchun SingleThreadExecutor
    @Bean(name = "orderedExecutor")
    public ExecutorService orderedExecutor() {
        return Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setNameFormat("ordered-%d").build());
    }

    // Davriy vazifalar uchun ScheduledThreadPool
    @Bean(name = "scheduledExecutor")
    public ScheduledExecutorService scheduledExecutor() {
        return Executors.newScheduledThreadPool(2,
            new ThreadFactoryBuilder().setNameFormat("scheduled-%d").build());
    }
}
\`\`\`

**Amaliy foydalari:**
- Turli ish yuklari turlari uchun optimallashtirish
- Rad etish siyosatlari va navbatlarni sozlash
- Poollarning hayot tsiklini monitoring va boshqarish`
        }
    }
};

export default task;
