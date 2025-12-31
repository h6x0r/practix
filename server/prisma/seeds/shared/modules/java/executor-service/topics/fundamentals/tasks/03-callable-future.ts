import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-callable-future',
    title: 'Callable and Future',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'callable', 'future', 'async', 'java5'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Callable and Future

Unlike Runnable which cannot return a result, Callable can return a value and throw checked exceptions. Future represents the result of an asynchronous computation, allowing you to retrieve results, check completion status, and cancel tasks.

## Requirements:
1. Create tasks using Callable<T> interface:
   1. Implement Callable that returns a computed result
   2. Handle exceptions in Callable tasks
   3. Compare Callable vs Runnable

2. Work with Future objects:
   1. Use \`Future.get()\` to retrieve results (blocking)
   2. Use \`Future.get(timeout, unit)\` with timeout
   3. Check task status with \`isDone()\` and \`isCancelled()\`
   4. Cancel tasks with \`cancel()\`

3. Submit multiple Callable tasks and collect results:
   1. Demonstrate parallel computation
   2. Handle timeouts and cancellations
   3. Process results as they complete

4. Show practical examples:
   1. Parallel mathematical calculations
   2. Asynchronous data fetching simulation
   3. Exception handling in concurrent tasks

## Example Output:
\`\`\`
=== Callable vs Runnable ===
Runnable: Cannot return value
Callable result: 42

=== Future Operations ===
Task submitted, computing...
Is done? false
Result: 5050 (sum of 1..100)
Is done? true

=== Multiple Callables ===
Task 1 result: 15
Task 2 result: 24
Task 3 result: 35
All tasks completed

=== Future with Timeout ===
Result retrieved within timeout: 100
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class CallableFuture {
    public static void main(String[] args) {
        // TODO: Create ExecutorService

        // TODO: Compare Runnable vs Callable

        // TODO: Submit Callable and get Future

        // TODO: Demonstrate Future operations (get, isDone, cancel)

        // TODO: Submit multiple Callables and collect results

        // TODO: Handle timeout with Future.get(timeout, unit)

        // TODO: Shutdown executor
    }
}`,
    solutionCode: `import java.util.concurrent.*;
import java.util.*;

public class CallableFuture {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(3);

        try {
            // Callable vs Runnable
            System.out.println("=== Callable vs Runnable ===");

            // Runnable cannot return a value
            Future<?> runnableFuture = executor.submit(() -> {
                System.out.println("Runnable: Cannot return value");
            });
            runnableFuture.get(); // Wait for completion

            // Callable can return a value
            Callable<Integer> callable = () -> {
                Thread.sleep(500);
                return 42;
            };
            Future<Integer> callableFuture = executor.submit(callable);
            System.out.println("Callable result: " + callableFuture.get());

            // Future operations
            System.out.println("\\n=== Future Operations ===");

            Future<Integer> sumFuture = executor.submit(() -> {
                System.out.println("Task submitted, computing...");
                Thread.sleep(1000);
                int sum = 0;
                for (int i = 1; i <= 100; i++) {
                    sum += i;
                }
                return sum;
            });

            // Check if done
            System.out.println("Is done? " + sumFuture.isDone());

            // Get result (blocks until complete)
            Integer result = sumFuture.get();
            System.out.println("Result: " + result + " (sum of 1..100)");
            System.out.println("Is done? " + sumFuture.isDone());

            // Multiple Callables
            System.out.println("\\n=== Multiple Callables ===");

            List<Callable<Integer>> tasks = Arrays.asList(
                () -> { Thread.sleep(500); return 5 * 3; },
                () -> { Thread.sleep(300); return 8 * 3; },
                () -> { Thread.sleep(400); return 7 * 5; }
            );

            List<Future<Integer>> futures = new ArrayList<>();
            for (Callable<Integer> task : tasks) {
                futures.add(executor.submit(task));
            }

            // Collect results
            for (int i = 0; i < futures.size(); i++) {
                System.out.println("Task " + (i + 1) + " result: " +
                    futures.get(i).get());
            }
            System.out.println("All tasks completed");

            // Future with timeout
            System.out.println("\\n=== Future with Timeout ===");

            Future<Integer> timeoutFuture = executor.submit(() -> {
                Thread.sleep(500);
                return 100;
            });

            try {
                // Wait up to 2 seconds
                Integer timeoutResult = timeoutFuture.get(2, TimeUnit.SECONDS);
                System.out.println("Result retrieved within timeout: " +
                    timeoutResult);
            } catch (TimeoutException e) {
                System.out.println("Task timed out!");
                timeoutFuture.cancel(true);
            }

            // Cancellation example
            System.out.println("\\n=== Future Cancellation ===");

            Future<Integer> cancelFuture = executor.submit(() -> {
                Thread.sleep(5000);
                return 999;
            });

            Thread.sleep(100);
            boolean cancelled = cancelFuture.cancel(true);
            System.out.println("Task cancelled: " + cancelled);
            System.out.println("Is cancelled? " + cancelFuture.isCancelled());

            // Exception handling
            System.out.println("\\n=== Exception Handling ===");

            Future<Integer> exceptionFuture = executor.submit(() -> {
                if (true) {
                    throw new Exception("Simulated error");
                }
                return 0;
            });

            try {
                exceptionFuture.get();
            } catch (ExecutionException e) {
                System.out.println("Task threw exception: " +
                    e.getCause().getMessage());
            }

        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
}`,
    hint1: `Callable<T> can return a value and throw checked exceptions: Callable<Integer> task = () -> { return 42; }. Use executor.submit(callable) to get a Future<T>.`,
    hint2: `Future.get() blocks until the result is available. Use get(timeout, unit) to avoid waiting indefinitely. Always check isDone() before get() if you want non-blocking behavior.`,
    whyItMatters: `Callable and Future are essential for asynchronous programming in Java. They enable you to start computations in background threads and retrieve results later, making it possible to build responsive applications. Understanding Future operations is crucial for handling timeouts, cancellations, and coordinating multiple concurrent tasks.

**Production pattern:**
\`\`\`java
@Service
public class DataAggregationService {
    private final ExecutorService executor = Executors.newFixedThreadPool(5);

    public AggregatedData fetchData(String userId) throws DataException {
        List<Callable<ServiceData>> tasks = Arrays.asList(
            () -> userService.getUserData(userId),
            () -> orderService.getOrders(userId),
            () -> paymentService.getPaymentHistory(userId)
        );

        try {
            // Execute all tasks with timeout
            List<Future<ServiceData>> futures = tasks.stream()
                .map(executor::submit)
                .collect(Collectors.toList());

            // Collect results with timeouts
            List<ServiceData> results = new ArrayList<>();
            for (Future<ServiceData> future : futures) {
                try {
                    results.add(future.get(5, TimeUnit.SECONDS));
                } catch (TimeoutException e) {
                    logger.warn("Service call timed out", e);
                    future.cancel(true);
                    metrics.incrementCounter("service.timeout");
                } catch (ExecutionException e) {
                    logger.error("Service call failed", e);
                    metrics.incrementCounter("service.error");
                }
            }

            return new AggregatedData(results);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DataException("Data fetch interrupted", e);
        }
    }
}
\`\`\`

**Practical benefits:**
- Parallel data aggregation from multiple sources
- Timeout handling for reliability
- Metrics for performance monitoring`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Verify Callable interface exists
class Test1 {
    @Test
    public void testCallableInterfaceExists() {
        assertNotNull(Callable.class);
    }
}

// Test2: Verify Future interface exists
class Test2 {
    @Test
    public void testFutureInterfaceExists() {
        assertNotNull(Future.class);
    }
}

// Test3: Verify Callable can return a value
class Test3 {
    @Test
    public void testCallableReturnsValue() throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Integer> task = () -> 42;

        Future<Integer> future = executor.submit(task);
        Integer result = future.get();

        assertEquals(Integer.valueOf(42), result);
        executor.shutdown();
    }
}

// Test4: Verify Future.get() method exists
class Test4 {
    @Test
    public void testFutureGetMethodExists() throws NoSuchMethodException {
        assertNotNull(Future.class.getMethod("get"));
    }
}

// Test5: Verify Future.get() with timeout exists
class Test5 {
    @Test
    public void testFutureGetWithTimeoutExists() throws NoSuchMethodException {
        assertNotNull(Future.class.getMethod("get", long.class, TimeUnit.class));
    }
}

// Test6: Verify Future.isDone() method exists
class Test6 {
    @Test
    public void testIsDoneMethodExists() throws NoSuchMethodException {
        assertNotNull(Future.class.getMethod("isDone"));
    }
}

// Test7: Verify Future.cancel() method exists
class Test7 {
    @Test
    public void testCancelMethodExists() throws NoSuchMethodException {
        assertNotNull(Future.class.getMethod("cancel", boolean.class));
    }
}

// Test8: Verify Future isDone returns true after completion
class Test8 {
    @Test
    public void testIsDoneAfterCompletion() throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(() -> "done");

        future.get();
        assertTrue(future.isDone());
        executor.shutdown();
    }
}

// Test9: Verify ExecutionException handling
class Test9 {
    @Test(expected = ExecutionException.class)
    public void testExecutionException() throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(() -> {
            throw new RuntimeException("Test exception");
        });

        try {
            future.get();
        } finally {
            executor.shutdown();
        }
    }
}

// Test10: Verify multiple Callables execution
class Test10 {
    @Test
    public void testMultipleCallables() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(3);

        Future<Integer> f1 = executor.submit(() -> 1);
        Future<Integer> f2 = executor.submit(() -> 2);
        Future<Integer> f3 = executor.submit(() -> 3);

        int sum = f1.get() + f2.get() + f3.get();

        assertEquals(6, sum);
        executor.shutdown();
    }
}`,
    order: 3,
    translations: {
        ru: {
            title: 'Callable и Future',
            solutionCode: `import java.util.concurrent.*;
import java.util.*;

public class CallableFuture {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(3);

        try {
            // Callable vs Runnable
            System.out.println("=== Callable vs Runnable ===");

            // Runnable не может вернуть значение
            Future<?> runnableFuture = executor.submit(() -> {
                System.out.println("Runnable: Cannot return value");
            });
            runnableFuture.get(); // Ждем завершения

            // Callable может вернуть значение
            Callable<Integer> callable = () -> {
                Thread.sleep(500);
                return 42;
            };
            Future<Integer> callableFuture = executor.submit(callable);
            System.out.println("Callable result: " + callableFuture.get());

            // Операции с Future
            System.out.println("\\n=== Операции с Future ===");

            Future<Integer> sumFuture = executor.submit(() -> {
                System.out.println("Task submitted, computing...");
                Thread.sleep(1000);
                int sum = 0;
                for (int i = 1; i <= 100; i++) {
                    sum += i;
                }
                return sum;
            });

            // Проверяем, завершена ли задача
            System.out.println("Is done? " + sumFuture.isDone());

            // Получаем результат (блокируется до завершения)
            Integer result = sumFuture.get();
            System.out.println("Result: " + result + " (sum of 1..100)");
            System.out.println("Is done? " + sumFuture.isDone());

            // Несколько Callable
            System.out.println("\\n=== Несколько Callable ===");

            List<Callable<Integer>> tasks = Arrays.asList(
                () -> { Thread.sleep(500); return 5 * 3; },
                () -> { Thread.sleep(300); return 8 * 3; },
                () -> { Thread.sleep(400); return 7 * 5; }
            );

            List<Future<Integer>> futures = new ArrayList<>();
            for (Callable<Integer> task : tasks) {
                futures.add(executor.submit(task));
            }

            // Собираем результаты
            for (int i = 0; i < futures.size(); i++) {
                System.out.println("Task " + (i + 1) + " result: " +
                    futures.get(i).get());
            }
            System.out.println("All tasks completed");

            // Future с таймаутом
            System.out.println("\\n=== Future с таймаутом ===");

            Future<Integer> timeoutFuture = executor.submit(() -> {
                Thread.sleep(500);
                return 100;
            });

            try {
                // Ждем до 2 секунд
                Integer timeoutResult = timeoutFuture.get(2, TimeUnit.SECONDS);
                System.out.println("Result retrieved within timeout: " +
                    timeoutResult);
            } catch (TimeoutException e) {
                System.out.println("Task timed out!");
                timeoutFuture.cancel(true);
            }

            // Пример отмены
            System.out.println("\\n=== Отмена Future ===");

            Future<Integer> cancelFuture = executor.submit(() -> {
                Thread.sleep(5000);
                return 999;
            });

            Thread.sleep(100);
            boolean cancelled = cancelFuture.cancel(true);
            System.out.println("Task cancelled: " + cancelled);
            System.out.println("Is cancelled? " + cancelFuture.isCancelled());

            // Обработка исключений
            System.out.println("\\n=== Обработка исключений ===");

            Future<Integer> exceptionFuture = executor.submit(() -> {
                if (true) {
                    throw new Exception("Simulated error");
                }
                return 0;
            });

            try {
                exceptionFuture.get();
            } catch (ExecutionException e) {
                System.out.println("Task threw exception: " +
                    e.getCause().getMessage());
            }

        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
}`,
            description: `# Callable и Future

В отличие от Runnable, который не может вернуть результат, Callable может вернуть значение и выбросить проверяемые исключения. Future представляет результат асинхронного вычисления, позволяя вам получить результаты, проверить статус завершения и отменить задачи.

## Требования:
1. Создайте задачи используя интерфейс Callable<T>:
   1. Реализуйте Callable, который возвращает вычисленный результат
   2. Обработайте исключения в задачах Callable
   3. Сравните Callable vs Runnable

2. Работайте с объектами Future:
   1. Используйте \`Future.get()\` для получения результатов (блокирующий)
   2. Используйте \`Future.get(timeout, unit)\` с таймаутом
   3. Проверяйте статус задачи с \`isDone()\` и \`isCancelled()\`
   4. Отменяйте задачи с \`cancel()\`

3. Отправьте несколько задач Callable и соберите результаты:
   1. Продемонстрируйте параллельное вычисление
   2. Обработайте таймауты и отмены
   3. Обработайте результаты по мере их завершения

4. Покажите практические примеры:
   1. Параллельные математические вычисления
   2. Симуляция асинхронной загрузки данных
   3. Обработка исключений в параллельных задачах

## Пример вывода:
\`\`\`
=== Callable vs Runnable ===
Runnable: Cannot return value
Callable result: 42

=== Future Operations ===
Task submitted, computing...
Is done? false
Result: 5050 (sum of 1..100)
Is done? true

=== Multiple Callables ===
Task 1 result: 15
Task 2 result: 24
Task 3 result: 35
All tasks completed

=== Future with Timeout ===
Result retrieved within timeout: 100
\`\`\``,
            hint1: `Callable<T> может вернуть значение и выбросить проверяемые исключения: Callable<Integer> task = () -> { return 42; }. Используйте executor.submit(callable) для получения Future<T>.`,
            hint2: `Future.get() блокируется до тех пор, пока результат не станет доступен. Используйте get(timeout, unit) чтобы избежать бесконечного ожидания. Всегда проверяйте isDone() перед get(), если хотите неблокирующее поведение.`,
            whyItMatters: `Callable и Future необходимы для асинхронного программирования в Java. Они позволяют запускать вычисления в фоновых потоках и получать результаты позже, что делает возможным создание отзывчивых приложений. Понимание операций Future критически важно для обработки таймаутов, отмен и координации нескольких параллельных задач.

**Продакшен паттерн:**
\`\`\`java
@Service
public class DataAggregationService {
    private final ExecutorService executor = Executors.newFixedThreadPool(5);

    public AggregatedData fetchData(String userId) throws DataException {
        List<Callable<ServiceData>> tasks = Arrays.asList(
            () -> userService.getUserData(userId),
            () -> orderService.getOrders(userId),
            () -> paymentService.getPaymentHistory(userId)
        );

        try {
            // Выполняем все задачи с таймаутом
            List<Future<ServiceData>> futures = tasks.stream()
                .map(executor::submit)
                .collect(Collectors.toList());

            // Собираем результаты с таймаутами
            List<ServiceData> results = new ArrayList<>();
            for (Future<ServiceData> future : futures) {
                try {
                    results.add(future.get(5, TimeUnit.SECONDS));
                } catch (TimeoutException e) {
                    logger.warn("Service call timed out", e);
                    future.cancel(true);
                    metrics.incrementCounter("service.timeout");
                } catch (ExecutionException e) {
                    logger.error("Service call failed", e);
                    metrics.incrementCounter("service.error");
                }
            }

            return new AggregatedData(results);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DataException("Data fetch interrupted", e);
        }
    }
}
\`\`\`

**Практические преимущества:**
- Параллельная агрегация данных из нескольких источников
- Обработка таймаутов для надежности
- Метрики для мониторинга производительности`
        },
        uz: {
            title: `Callable va Future`,
            solutionCode: `import java.util.concurrent.*;
import java.util.*;

public class CallableFuture {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(3);

        try {
            // Callable vs Runnable
            System.out.println("=== Callable vs Runnable ===");

            // Runnable qiymat qaytara olmaydi
            Future<?> runnableFuture = executor.submit(() -> {
                System.out.println("Runnable: Cannot return value");
            });
            runnableFuture.get(); // Tugashni kutamiz

            // Callable qiymat qaytarishi mumkin
            Callable<Integer> callable = () -> {
                Thread.sleep(500);
                return 42;
            };
            Future<Integer> callableFuture = executor.submit(callable);
            System.out.println("Callable result: " + callableFuture.get());

            // Future operatsiyalari
            System.out.println("\\n=== Future operatsiyalari ===");

            Future<Integer> sumFuture = executor.submit(() -> {
                System.out.println("Task submitted, computing...");
                Thread.sleep(1000);
                int sum = 0;
                for (int i = 1; i <= 100; i++) {
                    sum += i;
                }
                return sum;
            });

            // Tugaganligini tekshiramiz
            System.out.println("Is done? " + sumFuture.isDone());

            // Natijani olamiz (tugagunga qadar bloklaydi)
            Integer result = sumFuture.get();
            System.out.println("Result: " + result + " (sum of 1..100)");
            System.out.println("Is done? " + sumFuture.isDone());

            // Bir nechta Callable
            System.out.println("\\n=== Bir nechta Callable ===");

            List<Callable<Integer>> tasks = Arrays.asList(
                () -> { Thread.sleep(500); return 5 * 3; },
                () -> { Thread.sleep(300); return 8 * 3; },
                () -> { Thread.sleep(400); return 7 * 5; }
            );

            List<Future<Integer>> futures = new ArrayList<>();
            for (Callable<Integer> task : tasks) {
                futures.add(executor.submit(task));
            }

            // Natijalarni yig'amiz
            for (int i = 0; i < futures.size(); i++) {
                System.out.println("Task " + (i + 1) + " result: " +
                    futures.get(i).get());
            }
            System.out.println("All tasks completed");

            // Timeout bilan Future
            System.out.println("\\n=== Timeout bilan Future ===");

            Future<Integer> timeoutFuture = executor.submit(() -> {
                Thread.sleep(500);
                return 100;
            });

            try {
                // 2 soniyagacha kutamiz
                Integer timeoutResult = timeoutFuture.get(2, TimeUnit.SECONDS);
                System.out.println("Result retrieved within timeout: " +
                    timeoutResult);
            } catch (TimeoutException e) {
                System.out.println("Task timed out!");
                timeoutFuture.cancel(true);
            }

            // Bekor qilish misoli
            System.out.println("\\n=== Future bekor qilish ===");

            Future<Integer> cancelFuture = executor.submit(() -> {
                Thread.sleep(5000);
                return 999;
            });

            Thread.sleep(100);
            boolean cancelled = cancelFuture.cancel(true);
            System.out.println("Task cancelled: " + cancelled);
            System.out.println("Is cancelled? " + cancelFuture.isCancelled());

            // Xatolarni boshqarish
            System.out.println("\\n=== Xatolarni boshqarish ===");

            Future<Integer> exceptionFuture = executor.submit(() -> {
                if (true) {
                    throw new Exception("Simulated error");
                }
                return 0;
            });

            try {
                exceptionFuture.get();
            } catch (ExecutionException e) {
                System.out.println("Task threw exception: " +
                    e.getCause().getMessage());
            }

        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
}`,
            description: `# Callable va Future

Natija qaytara olmaydigan Runnable dan farqli o'laroq, Callable qiymat qaytarishi va tekshiriladigan xatolarni tashlashi mumkin. Future asinxron hisoblash natijasini ifodalaydi, natijalarni olish, tugash holatini tekshirish va vazifalarni bekor qilish imkonini beradi.

## Talablar:
1. Callable<T> interfeysi yordamida vazifalar yarating:
   1. Hisoblangan natijani qaytaradigan Callable ni amalga oshiring
   2. Callable vazifalarida xatolarni boshqaring
   3. Callable vs Runnable ni solishtiring

2. Future obyektlari bilan ishlang:
   1. Natijalarni olish uchun \`Future.get()\` dan foydalaning (bloklash)
   2. Timeout bilan \`Future.get(timeout, unit)\` dan foydalaning
   3. \`isDone()\` va \`isCancelled()\` bilan vazifa holatini tekshiring
   4. \`cancel()\` bilan vazifalarni bekor qiling

3. Bir nechta Callable vazifalarini yuboring va natijalarni yig'ing:
   1. Parallel hisoblashni ko'rsating
   2. Timeoutlar va bekor qilishlarni boshqaring
   3. Natijalar tugaganda ularni qayta ishlang

4. Amaliy misollarni ko'rsating:
   1. Parallel matematik hisoblashlar
   2. Asinxron ma'lumot yuklash simulyatsiyasi
   3. Parallel vazifalarida xatolarni boshqarish

## Chiqish namunasi:
\`\`\`
=== Callable vs Runnable ===
Runnable: Cannot return value
Callable result: 42

=== Future Operations ===
Task submitted, computing...
Is done? false
Result: 5050 (sum of 1..100)
Is done? true

=== Multiple Callables ===
Task 1 result: 15
Task 2 result: 24
Task 3 result: 35
All tasks completed

=== Future with Timeout ===
Result retrieved within timeout: 100
\`\`\``,
            hint1: `Callable<T> qiymat qaytarishi va tekshiriladigan xatolarni tashlashi mumkin: Callable<Integer> task = () -> { return 42; }. Future<T> ni olish uchun executor.submit(callable) dan foydalaning.`,
            hint2: `Future.get() natija mavjud bo'lgunga qadar bloklaydi. Cheksiz kutishdan qochish uchun get(timeout, unit) dan foydalaning. Bloklashsiz xatti-harakatni xohlasangiz, get() dan oldin doim isDone() ni tekshiring.`,
            whyItMatters: `Callable va Future Java da asinxron dasturlash uchun zarurdir. Ular hisoblashlarni orqa fon threadlarida boshlash va natijalarni keyinroq olish imkonini beradi, bu esa sezgir dasturlar yaratishni mumkin qiladi. Future operatsiyalarini tushunish timeoutlar, bekor qilishlar va bir nechta parallel vazifalarni koordinatsiya qilish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class DataAggregationService {
    private final ExecutorService executor = Executors.newFixedThreadPool(5);

    public AggregatedData fetchData(String userId) throws DataException {
        List<Callable<ServiceData>> tasks = Arrays.asList(
            () -> userService.getUserData(userId),
            () -> orderService.getOrders(userId),
            () -> paymentService.getPaymentHistory(userId)
        );

        try {
            // Barcha vazifalarni timeout bilan bajaramiz
            List<Future<ServiceData>> futures = tasks.stream()
                .map(executor::submit)
                .collect(Collectors.toList());

            // Timeoutlar bilan natijalarni yig'amiz
            List<ServiceData> results = new ArrayList<>();
            for (Future<ServiceData> future : futures) {
                try {
                    results.add(future.get(5, TimeUnit.SECONDS));
                } catch (TimeoutException e) {
                    logger.warn("Servis chaqiruvi vaqti tugadi", e);
                    future.cancel(true);
                    metrics.incrementCounter("service.timeout");
                } catch (ExecutionException e) {
                    logger.error("Servis chaqiruvi muvaffaqiyatsiz", e);
                    metrics.incrementCounter("service.error");
                }
            }

            return new AggregatedData(results);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DataException("Ma'lumot olish to'xtatildi", e);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Bir nechta manbalardan ma'lumotlarni parallel agregatsiya qilish
- Ishonchlilik uchun timeoutlarni qayta ishlash
- Unumdorlikni monitoring qilish uchun metrikalar`
        }
    }
};

export default task;
