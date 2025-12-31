import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-executor-basics',
    title: 'ExecutorService Basics',
    difficulty: 'easy',
    tags: ['java', 'concurrency', 'executor', 'thread-pool', 'java5'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# ExecutorService Basics

ExecutorService is a high-level replacement for working directly with threads. It provides a framework for asynchronous task execution and manages a pool of threads automatically. The Executors factory class provides convenient methods to create different types of executor services.

## Requirements:
1. Create an ExecutorService using Executors factory methods:
   1. Use \`Executors.newFixedThreadPool(n)\` for a fixed number of threads
   2. Demonstrate task submission with \`execute()\` and \`submit()\`
   3. Show the difference between execute() and submit()

2. Submit multiple tasks to the executor:
   1. Create tasks that print thread information
   2. Show tasks executing concurrently
   3. Display task completion messages

3. Properly shutdown the executor:
   1. Call \`shutdown()\` to gracefully stop accepting new tasks
   2. Explain why shutdown is important

4. Compare manual thread creation vs ExecutorService

## Example Output:
\`\`\`
=== ExecutorService Basics ===
Task 1 executing on thread: pool-1-thread-1
Task 2 executing on thread: pool-1-thread-2
Task 3 executing on thread: pool-1-thread-3
Task 1 completed
Task 2 completed
Task 3 completed

=== Submit vs Execute ===
Execute: Task running on pool-1-thread-1
Submit: Task running on pool-1-thread-2
Submit returned Future: java.util.concurrent.FutureTask@a1b2c3d4

Executor shutdown initiated.
\`\`\``,
    initialCode: `// TODO: Import ExecutorService and Executors

public class ExecutorBasics {
    public static void main(String[] args) {
        // TODO: Create a fixed thread pool with 3 threads

        // TODO: Submit multiple tasks using execute()

        // TODO: Demonstrate submit() method

        // TODO: Show difference between execute() and submit()

        // TODO: Shutdown the executor properly
    }
}`,
    solutionCode: `import java.util.concurrent.*;

public class ExecutorBasics {
    public static void main(String[] args) {
        System.out.println("=== ExecutorService Basics ===");

        // Create a fixed thread pool with 3 threads
        ExecutorService executor = Executors.newFixedThreadPool(3);

        // Submit multiple tasks using execute()
        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            executor.execute(() -> {
                System.out.println("Task " + taskId + " executing on thread: " +
                    Thread.currentThread().getName());
                try {
                    Thread.sleep(1000); // Simulate work
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.println("Task " + taskId + " completed");
            });
        }

        // Wait for tasks to complete
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("\\n=== Submit vs Execute ===");

        // execute() - returns void, used for Runnable
        executor.execute(() -> {
            System.out.println("Execute: Task running on " +
                Thread.currentThread().getName());
        });

        // submit() - returns Future, can be used with Runnable or Callable
        Future<?> future = executor.submit(() -> {
            System.out.println("Submit: Task running on " +
                Thread.currentThread().getName());
        });

        System.out.println("Submit returned Future: " + future);

        // Properly shutdown the executor
        // shutdown() prevents new tasks from being submitted
        executor.shutdown();
        System.out.println("\\nExecutor shutdown initiated.");

        try {
            // Wait for existing tasks to complete
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}`,
    hint1: `Use Executors.newFixedThreadPool(n) to create an executor with a fixed number of threads. The execute() method takes a Runnable and returns void, while submit() returns a Future.`,
    hint2: `Always call shutdown() on an ExecutorService when done to release resources. The executor won't terminate the JVM automatically without shutdown.`,
    whyItMatters: `ExecutorService provides a higher-level abstraction for managing threads, eliminating the complexity of manual thread creation and management. It's essential for building scalable concurrent applications, managing thread pools efficiently, and avoiding resource leaks. Understanding ExecutorService is fundamental to Java concurrency.

**Production pattern:**
\`\`\`java
@Service
public class NotificationService {
    private final ExecutorService executor =
        Executors.newFixedThreadPool(10,
            new ThreadFactoryBuilder()
                .setNameFormat("notification-%d")
                .setDaemon(true)
                .build());

    public void sendNotifications(List<User> users) {
        users.forEach(user ->
            executor.submit(() -> {
                try {
                    emailService.send(user.getEmail(), "Notification");
                    metrics.incrementCounter("notifications.sent");
                } catch (Exception e) {
                    logger.error("Failed to send notification", e);
                    metrics.incrementCounter("notifications.failed");
                }
            })
        );
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
\`\`\`

**Practical benefits:**
- Thread pool management with configurable size
- Graceful shutdown with timeouts
- Metrics and error handling for reliability`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Verify ExecutorService interface exists
class Test1 {
    @Test
    public void testExecutorServiceInterfaceExists() {
        assertNotNull(ExecutorService.class);
    }
}

// Test2: Verify Executors class exists
class Test2 {
    @Test
    public void testExecutorsClassExists() {
        assertNotNull(Executors.class);
    }
}

// Test3: Verify newSingleThreadExecutor method exists
class Test3 {
    @Test
    public void testNewSingleThreadExecutor() {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        assertNotNull(executor);
        executor.shutdown();
    }
}

// Test4: Verify submit method exists
class Test4 {
    @Test
    public void testSubmitMethodExists() throws NoSuchMethodException {
        assertNotNull(ExecutorService.class.getMethod("submit", Runnable.class));
    }
}

// Test5: Verify shutdown method exists
class Test5 {
    @Test
    public void testShutdownMethodExists() throws NoSuchMethodException {
        assertNotNull(ExecutorService.class.getMethod("shutdown"));
    }
}

// Test6: Verify awaitTermination method exists
class Test6 {
    @Test
    public void testAwaitTerminationMethodExists() throws NoSuchMethodException {
        assertNotNull(ExecutorService.class.getMethod("awaitTermination", long.class, TimeUnit.class));
    }
}

// Test7: Verify task execution with single thread executor
class Test7 {
    @Test
    public void testTaskExecution() throws InterruptedException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        int[] counter = {0};

        executor.submit(() -> counter[0]++);
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);

        assertEquals(1, counter[0]);
    }
}

// Test8: Verify shutdownNow method exists
class Test8 {
    @Test
    public void testShutdownNowMethodExists() throws NoSuchMethodException {
        assertNotNull(ExecutorService.class.getMethod("shutdownNow"));
    }
}

// Test9: Verify isShutdown method
class Test9 {
    @Test
    public void testIsShutdown() {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        assertFalse(executor.isShutdown());

        executor.shutdown();
        assertTrue(executor.isShutdown());
    }
}

// Test10: Verify isTerminated method
class Test10 {
    @Test
    public void testIsTerminated() throws InterruptedException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {});
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.SECONDS);

        assertTrue(executor.isTerminated());
    }
}`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы ExecutorService',
            solutionCode: `import java.util.concurrent.*;

public class ExecutorBasics {
    public static void main(String[] args) {
        System.out.println("=== Основы ExecutorService ===");

        // Создаем фиксированный пул потоков с 3 потоками
        ExecutorService executor = Executors.newFixedThreadPool(3);

        // Отправляем несколько задач используя execute()
        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            executor.execute(() -> {
                System.out.println("Task " + taskId + " executing on thread: " +
                    Thread.currentThread().getName());
                try {
                    Thread.sleep(1000); // Имитируем работу
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.println("Task " + taskId + " completed");
            });
        }

        // Ждем завершения задач
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("\\n=== Submit vs Execute ===");

        // execute() - возвращает void, используется для Runnable
        executor.execute(() -> {
            System.out.println("Execute: Task running on " +
                Thread.currentThread().getName());
        });

        // submit() - возвращает Future, может использоваться с Runnable или Callable
        Future<?> future = executor.submit(() -> {
            System.out.println("Submit: Task running on " +
                Thread.currentThread().getName());
        });

        System.out.println("Submit returned Future: " + future);

        // Правильно завершаем работу executor
        // shutdown() предотвращает отправку новых задач
        executor.shutdown();
        System.out.println("\\nExecutor shutdown initiated.");

        try {
            // Ждем завершения существующих задач
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# Основы ExecutorService

ExecutorService - это высокоуровневая замена для работы напрямую с потоками. Он предоставляет фреймворк для асинхронного выполнения задач и автоматически управляет пулом потоков. Фабричный класс Executors предоставляет удобные методы для создания различных типов сервисов-исполнителей.

## Требования:
1. Создайте ExecutorService используя фабричные методы Executors:
   1. Используйте \`Executors.newFixedThreadPool(n)\` для фиксированного числа потоков
   2. Продемонстрируйте отправку задач с \`execute()\` и \`submit()\`
   3. Покажите разницу между execute() и submit()

2. Отправьте несколько задач в executor:
   1. Создайте задачи, которые выводят информацию о потоке
   2. Покажите параллельное выполнение задач
   3. Отобразите сообщения о завершении задач

3. Правильно завершите работу executor:
   1. Вызовите \`shutdown()\` для корректной остановки приема новых задач
   2. Объясните, почему shutdown важен

4. Сравните создание потоков вручную vs ExecutorService

## Пример вывода:
\`\`\`
=== ExecutorService Basics ===
Task 1 executing on thread: pool-1-thread-1
Task 2 executing on thread: pool-1-thread-2
Task 3 executing on thread: pool-1-thread-3
Task 1 completed
Task 2 completed
Task 3 completed

=== Submit vs Execute ===
Execute: Task running on pool-1-thread-1
Submit: Task running on pool-1-thread-2
Submit returned Future: java.util.concurrent.FutureTask@a1b2c3d4

Executor shutdown initiated.
\`\`\``,
            hint1: `Используйте Executors.newFixedThreadPool(n) для создания executor с фиксированным числом потоков. Метод execute() принимает Runnable и возвращает void, в то время как submit() возвращает Future.`,
            hint2: `Всегда вызывайте shutdown() на ExecutorService после завершения работы для освобождения ресурсов. Executor не завершит работу JVM автоматически без shutdown.`,
            whyItMatters: `ExecutorService предоставляет высокоуровневую абстракцию для управления потоками, устраняя сложность ручного создания и управления потоками. Это необходимо для построения масштабируемых параллельных приложений, эффективного управления пулами потоков и предотвращения утечек ресурсов. Понимание ExecutorService - основа параллелизма в Java.

**Продакшен паттерн:**
\`\`\`java
@Service
public class NotificationService {
    private final ExecutorService executor =
        Executors.newFixedThreadPool(10,
            new ThreadFactoryBuilder()
                .setNameFormat("notification-%d")
                .setDaemon(true)
                .build());

    public void sendNotifications(List<User> users) {
        users.forEach(user ->
            executor.submit(() -> {
                try {
                    emailService.send(user.getEmail(), "Notification");
                    metrics.incrementCounter("notifications.sent");
                } catch (Exception e) {
                    logger.error("Failed to send notification", e);
                    metrics.incrementCounter("notifications.failed");
                }
            })
        );
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
\`\`\`

**Практические преимущества:**
- Управление пулом потоков с настраиваемым размером
- Изящное завершение работы с таймаутами
- Метрики и обработка ошибок для надежности`
        },
        uz: {
            title: `ExecutorService asoslari`,
            solutionCode: `import java.util.concurrent.*;

public class ExecutorBasics {
    public static void main(String[] args) {
        System.out.println("=== ExecutorService asoslari ===");

        // 3 ta thread bilan fiksirlangan thread pool yaratamiz
        ExecutorService executor = Executors.newFixedThreadPool(3);

        // execute() yordamida bir nechta vazifalarni yuboramiz
        for (int i = 1; i <= 3; i++) {
            final int taskId = i;
            executor.execute(() -> {
                System.out.println("Task " + taskId + " executing on thread: " +
                    Thread.currentThread().getName());
                try {
                    Thread.sleep(1000); // Ishni simulyatsiya qilamiz
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.println("Task " + taskId + " completed");
            });
        }

        // Vazifalarning tugashini kutamiz
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("\\n=== Submit vs Execute ===");

        // execute() - void qaytaradi, Runnable uchun ishlatiladi
        executor.execute(() -> {
            System.out.println("Execute: Task running on " +
                Thread.currentThread().getName());
        });

        // submit() - Future qaytaradi, Runnable yoki Callable bilan ishlatiladi
        Future<?> future = executor.submit(() -> {
            System.out.println("Submit: Task running on " +
                Thread.currentThread().getName());
        });

        System.out.println("Submit returned Future: " + future);

        // Executorni to'g'ri yopamiz
        // shutdown() yangi vazifalar yuborishni oldini oladi
        executor.shutdown();
        System.out.println("\\nExecutor shutdown initiated.");

        try {
            // Mavjud vazifalarning tugashini kutamiz
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# ExecutorService asoslari

ExecutorService to'g'ridan-to'g'ri threadlar bilan ishlashning yuqori darajadagi o'rnini bosadi. U asinxron vazifalarni bajarish uchun freymvork taqdim etadi va threadlar poolini avtomatik ravishda boshqaradi. Executors fabrika klassi turli xil executor servicelarni yaratish uchun qulay metodlarni taqdim etadi.

## Talablar:
1. Executors fabrika metodlari yordamida ExecutorService yarating:
   1. Fiksirlangan sondagi threadlar uchun \`Executors.newFixedThreadPool(n)\` dan foydalaning
   2. \`execute()\` va \`submit()\` bilan vazifa yuborishni ko'rsating
   3. execute() va submit() o'rtasidagi farqni ko'rsating

2. Executorga bir nechta vazifa yuboring:
   1. Thread haqida ma'lumot chiqaradigan vazifalar yarating
   2. Vazifalarning parallel bajarilishini ko'rsating
   3. Vazifa tugash xabarlarini ko'rsating

3. Executorni to'g'ri yoping:
   1. Yangi vazifalarni qabul qilishni to'xtatish uchun \`shutdown()\` ni chaqiring
   2. Shutdown nima uchun muhimligini tushuntiring

4. Qo'lda thread yaratish vs ExecutorServiceni solishtiring

## Chiqish namunasi:
\`\`\`
=== ExecutorService Basics ===
Task 1 executing on thread: pool-1-thread-1
Task 2 executing on thread: pool-1-thread-2
Task 3 executing on thread: pool-1-thread-3
Task 1 completed
Task 2 completed
Task 3 completed

=== Submit vs Execute ===
Execute: Task running on pool-1-thread-1
Submit: Task running on pool-1-thread-2
Submit returned Future: java.util.concurrent.FutureTask@a1b2c3d4

Executor shutdown initiated.
\`\`\``,
            hint1: `Fiksirlangan sondagi threadlar bilan executor yaratish uchun Executors.newFixedThreadPool(n) dan foydalaning. execute() metodi Runnable qabul qiladi va void qaytaradi, submit() esa Future qaytaradi.`,
            hint2: `Resurslarni ozod qilish uchun ExecutorService da ishingiz tugagach doim shutdown() ni chaqiring. Executor shutdown qilinmasa JVM avtomatik tugamaydi.`,
            whyItMatters: `ExecutorService threadlarni boshqarish uchun yuqori darajadagi abstraktsiyani taqdim etadi, qo'lda thread yaratish va boshqarish murakkabligini bartaraf etadi. Bu masshtablanadigan parallel dasturlarni yaratish, thread poollarini samarali boshqarish va resurs oqishlarini oldini olish uchun zarurdir. ExecutorServiceni tushunish Java parallellik asosi hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class NotificationService {
    private final ExecutorService executor =
        Executors.newFixedThreadPool(10,
            new ThreadFactoryBuilder()
                .setNameFormat("notification-%d")
                .setDaemon(true)
                .build());

    public void sendNotifications(List<User> users) {
        users.forEach(user ->
            executor.submit(() -> {
                try {
                    emailService.send(user.getEmail(), "Bildirishnoma");
                    metrics.incrementCounter("notifications.sent");
                } catch (Exception e) {
                    logger.error("Bildirishnoma yuborib bo'lmadi", e);
                    metrics.incrementCounter("notifications.failed");
                }
            })
        );
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Sozlanuvchi o'lcham bilan thread poolini boshqarish
- Timeoutlar bilan chiroyli tugatish
- Ishonchlilik uchun metrikalar va xatolarni qayta ishlash`
        }
    }
};

export default task;
