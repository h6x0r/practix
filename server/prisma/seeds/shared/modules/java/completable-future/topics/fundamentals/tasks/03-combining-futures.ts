import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-combining-futures',
    title: 'Combining Futures (thenCombine, thenCompose, allOf, anyOf)',
    difficulty: 'medium',
    tags: ['java', 'completablefuture', 'async', 'composition', 'parallel'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Combining Futures

CompletableFuture provides powerful methods for combining multiple async operations. You can combine two futures, compose dependent operations, or coordinate multiple independent futures. These methods are essential for building complex async workflows.

## Requirements:
1. Use thenCombine() to combine two independent futures:
   1. Both futures run in parallel
   2. Combine results when both complete
   3. Takes BiFunction to merge results

2. Use thenCompose() for dependent futures:
   1. Second future depends on result of first
   2. Flatten nested CompletableFutures
   3. Sequential async operations

3. Use allOf() to wait for multiple futures:
   1. Wait for all futures to complete
   2. Returns CompletableFuture<Void>
   3. Collect results from all futures

4. Use anyOf() to get first completed future:
   1. Returns when any future completes
   2. Useful for racing multiple operations
   3. Get fastest result

## Example Output:
\`\`\`
=== Combining Futures ===

--- thenCombine: Parallel Execution ---
Starting user fetch...
Starting order fetch...
Combined result: User: Alice, Orders: 5

--- thenCompose: Sequential Composition ---
Step 1: Getting user ID...
Step 2: Fetching user details for ID: 123
Result: User Details: Alice (ID: 123)

--- allOf: Wait for All ---
Fetching from API 1...
Fetching from API 2...
Fetching from API 3...
All APIs responded!
API 1: Data from API 1
API 2: Data from API 2
API 3: Data from API 3

--- anyOf: First to Complete ---
Starting query 1 (3000ms)...
Starting query 2 (1000ms)...
Starting query 3 (2000ms)...
First result: Query 2 result
\`\`\``,
    initialCode: `import java.util.concurrent.*;
import java.util.*;

public class CombiningFutures {
    public static void main(String[] args) {
        // TODO: Demonstrate thenCombine for parallel execution

        // TODO: Demonstrate thenCompose for sequential composition

        // TODO: Demonstrate allOf to wait for multiple futures

        // TODO: Demonstrate anyOf to get first completed future
    }
}`,
    solutionCode: `import java.util.concurrent.*;
import java.util.*;
import java.util.stream.*;

public class CombiningFutures {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Combining Futures ===");

        // thenCombine: Combine two independent futures (run in parallel)
        System.out.println("\\n--- thenCombine: Parallel Execution ---");
        CompletableFuture<String> userFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting user fetch...");
            sleep(1000);
            return "Alice";
        });

        CompletableFuture<Integer> ordersFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting order fetch...");
            sleep(1500);
            return 5;
        });

        // Combine results when both complete
        CompletableFuture<String> combined = userFuture.thenCombine(
            ordersFuture,
            (user, orders) -> "User: " + user + ", Orders: " + orders
        );

        System.out.println("Combined result: " + combined.join());

        // thenCompose: Chain dependent futures (sequential)
        System.out.println("\\n--- thenCompose: Sequential Composition ---");
        CompletableFuture<String> composedResult = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Getting user ID...");
                sleep(500);
                return 123;
            })
            .thenCompose(userId -> {
                // Second future depends on first result
                System.out.println("Step 2: Fetching user details for ID: " + userId);
                return CompletableFuture.supplyAsync(() -> {
                    sleep(500);
                    return "User Details: Alice (ID: " + userId + ")";
                });
            });

        System.out.println("Result: " + composedResult.join());

        // allOf: Wait for all futures to complete
        System.out.println("\\n--- allOf: Wait for All ---");
        CompletableFuture<String> api1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 1...");
            sleep(1000);
            return "Data from API 1";
        });

        CompletableFuture<String> api2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 2...");
            sleep(800);
            return "Data from API 2";
        });

        CompletableFuture<String> api3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 3...");
            sleep(1200);
            return "Data from API 3";
        });

        // Wait for all to complete
        CompletableFuture<Void> allDone = CompletableFuture.allOf(api1, api2, api3);
        allDone.join(); // Blocks until all complete

        System.out.println("All APIs responded!");
        System.out.println("API 1: " + api1.join());
        System.out.println("API 2: " + api2.join());
        System.out.println("API 3: " + api3.join());

        // anyOf: Get result from first completed future
        System.out.println("\\n--- anyOf: First to Complete ---");
        CompletableFuture<String> query1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 1 (3000ms)...");
            sleep(3000);
            return "Query 1 result";
        });

        CompletableFuture<String> query2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 2 (1000ms)...");
            sleep(1000);
            return "Query 2 result";
        });

        CompletableFuture<String> query3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 3 (2000ms)...");
            sleep(2000);
            return "Query 3 result";
        });

        // Get first completed result
        CompletableFuture<Object> firstDone = CompletableFuture.anyOf(query1, query2, query3);
        System.out.println("First result: " + firstDone.join());
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}`,
    hint1: `thenCombine() runs two futures in parallel and combines results. thenCompose() chains futures where the second depends on the first's result. Use compose to flatten nested futures.`,
    hint2: `allOf() waits for all futures to complete - useful for parallel batch operations. anyOf() returns the first completed future - useful for racing multiple sources or implementing timeouts.`,
    whyItMatters: `Combining futures is essential for building real-world async applications. You'll often need to fetch data from multiple sources in parallel (thenCombine, allOf), chain dependent operations (thenCompose), or implement fallback mechanisms (anyOf). These patterns are fundamental to microservices, API aggregation, and high-performance applications.

**Production Pattern:**
\`\`\`java
// Parallel data aggregation from microservices
CompletableFuture<User> userService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/users/" + userId));

CompletableFuture<List<Order>> orderService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/orders?userId=" + userId));

CompletableFuture<Preferences> prefService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/preferences/" + userId));

// Wait for all and collect results
CompletableFuture.allOf(userService, orderService, prefService)
    .thenApply(v -> new Dashboard(
        userService.join(),
        orderService.join(),
        prefService.join()
    ));

// Or get the fastest response (failover)
CompletableFuture.anyOf(primaryDB, secondaryDB, cacheDB)
    .thenAccept(result -> processData(result));
\`\`\`

**Practical Benefits:**
- Parallel requests to multiple services save time
- thenCompose prevents nested CompletableFuture<CompletableFuture<T>>
- anyOf enables failover and selecting the fastest source
- allOf simplifies coordination of multiple async tasks`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Test thenCombine combines two futures
class Test1 {
    @Test
    public void test() {
        CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 10);
        CompletableFuture<Integer> f2 = CompletableFuture.supplyAsync(() -> 20);
        CompletableFuture<Integer> result = f1.thenCombine(f2, (a, b) -> a + b);
        assertEquals(Integer.valueOf(30), result.join());
    }
}

// Test2: Test thenCompose chains dependent futures
class Test2 {
    @Test
    public void test() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 5)
            .thenCompose(x -> CompletableFuture.supplyAsync(() -> x * 2));
        assertEquals(Integer.valueOf(10), future.join());
    }
}

// Test3: Test allOf waits for all futures
class Test3 {
    @Test
    public void test() {
        CompletableFuture<String> f1 = CompletableFuture.supplyAsync(() -> "A");
        CompletableFuture<String> f2 = CompletableFuture.supplyAsync(() -> "B");
        CompletableFuture<String> f3 = CompletableFuture.supplyAsync(() -> "C");
        CompletableFuture<Void> all = CompletableFuture.allOf(f1, f2, f3);
        all.join();
        assertTrue(f1.isDone() && f2.isDone() && f3.isDone());
    }
}

// Test4: Test anyOf returns first completed
class Test4 {
    @Test
    public void test() throws Exception {
        CompletableFuture<String> f1 = CompletableFuture.supplyAsync(() -> {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
            return "Slow";
        });
        CompletableFuture<String> f2 = CompletableFuture.supplyAsync(() -> "Fast");
        CompletableFuture<Object> result = CompletableFuture.anyOf(f1, f2);
        assertEquals("Fast", result.get());
    }
}

// Test5: Test thenCombine with strings
class Test5 {
    @Test
    public void test() {
        CompletableFuture<String> f1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> f2 = CompletableFuture.supplyAsync(() -> "World");
        String result = f1.thenCombine(f2, (a, b) -> a + " " + b).join();
        assertEquals("Hello World", result);
    }
}

// Test6: Test thenCompose with transformation
class Test6 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "test")
            .thenCompose(s -> CompletableFuture.supplyAsync(() -> s.toUpperCase()));
        assertEquals("TEST", future.join());
    }
}

// Test7: Test allOf with result collection
class Test7 {
    @Test
    public void test() {
        CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 1);
        CompletableFuture<Integer> f2 = CompletableFuture.supplyAsync(() -> 2);
        CompletableFuture<Integer> f3 = CompletableFuture.supplyAsync(() -> 3);
        CompletableFuture.allOf(f1, f2, f3).join();
        assertEquals(Integer.valueOf(6), f1.join() + f2.join() + f3.join());
    }
}

// Test8: Test anyOf with multiple futures
class Test8 {
    @Test
    public void test() {
        CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 10);
        CompletableFuture<Integer> f2 = CompletableFuture.supplyAsync(() -> 20);
        CompletableFuture<Integer> f3 = CompletableFuture.supplyAsync(() -> 30);
        CompletableFuture<Object> result = CompletableFuture.anyOf(f1, f2, f3);
        assertNotNull(result.join());
    }
}

// Test9: Test thenCombine with BiFunction
class Test9 {
    @Test
    public void test() {
        CompletableFuture<Integer> result = CompletableFuture.supplyAsync(() -> 5)
            .thenCombine(CompletableFuture.supplyAsync(() -> 3), Integer::max);
        assertEquals(Integer.valueOf(5), result.join());
    }
}

// Test10: Test nested thenCompose operations
class Test10 {
    @Test
    public void test() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 2)
            .thenCompose(x -> CompletableFuture.supplyAsync(() -> x * x))
            .thenCompose(x -> CompletableFuture.supplyAsync(() -> x + 1));
        assertEquals(Integer.valueOf(5), future.join());
    }
}
`,
    translations: {
        ru: {
            title: 'Комбинирование Future (thenCombine, thenCompose, allOf, anyOf)',
            solutionCode: `import java.util.concurrent.*;
import java.util.*;
import java.util.stream.*;

public class CombiningFutures {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Комбинирование Futures ===");

        // thenCombine: Объединение двух независимых futures (параллельное выполнение)
        System.out.println("\\n--- thenCombine: Параллельное выполнение ---");
        CompletableFuture<String> userFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting user fetch...");
            sleep(1000);
            return "Alice";
        });

        CompletableFuture<Integer> ordersFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting order fetch...");
            sleep(1500);
            return 5;
        });

        // Объединение результатов когда оба завершены
        CompletableFuture<String> combined = userFuture.thenCombine(
            ordersFuture,
            (user, orders) -> "User: " + user + ", Orders: " + orders
        );

        System.out.println("Combined result: " + combined.join());

        // thenCompose: Цепочка зависимых futures (последовательное выполнение)
        System.out.println("\\n--- thenCompose: Последовательная композиция ---");
        CompletableFuture<String> composedResult = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Getting user ID...");
                sleep(500);
                return 123;
            })
            .thenCompose(userId -> {
                // Второй future зависит от результата первого
                System.out.println("Step 2: Fetching user details for ID: " + userId);
                return CompletableFuture.supplyAsync(() -> {
                    sleep(500);
                    return "User Details: Alice (ID: " + userId + ")";
                });
            });

        System.out.println("Result: " + composedResult.join());

        // allOf: Ожидание завершения всех futures
        System.out.println("\\n--- allOf: Ожидание всех ---");
        CompletableFuture<String> api1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 1...");
            sleep(1000);
            return "Data from API 1";
        });

        CompletableFuture<String> api2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 2...");
            sleep(800);
            return "Data from API 2";
        });

        CompletableFuture<String> api3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 3...");
            sleep(1200);
            return "Data from API 3";
        });

        // Ожидание завершения всех
        CompletableFuture<Void> allDone = CompletableFuture.allOf(api1, api2, api3);
        allDone.join(); // Блокирует до завершения всех

        System.out.println("All APIs responded!");
        System.out.println("API 1: " + api1.join());
        System.out.println("API 2: " + api2.join());
        System.out.println("API 3: " + api3.join());

        // anyOf: Получение результата от первого завершенного future
        System.out.println("\\n--- anyOf: Первый завершенный ---");
        CompletableFuture<String> query1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 1 (3000ms)...");
            sleep(3000);
            return "Query 1 result";
        });

        CompletableFuture<String> query2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 2 (1000ms)...");
            sleep(1000);
            return "Query 2 result";
        });

        CompletableFuture<String> query3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 3 (2000ms)...");
            sleep(2000);
            return "Query 3 result";
        });

        // Получение результата от первого завершенного
        CompletableFuture<Object> firstDone = CompletableFuture.anyOf(query1, query2, query3);
        System.out.println("First result: " + firstDone.join());
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}`,
            description: `# Комбинирование Futures

CompletableFuture предоставляет мощные методы для комбинирования нескольких асинхронных операций. Вы можете объединять два futures, компоновать зависимые операции или координировать несколько независимых futures. Эти методы необходимы для построения сложных асинхронных рабочих процессов.

## Требования:
1. Используйте thenCombine() для объединения двух независимых futures:
   1. Оба futures выполняются параллельно
   2. Объединяйте результаты когда оба завершены
   3. Принимает BiFunction для слияния результатов

2. Используйте thenCompose() для зависимых futures:
   1. Второй future зависит от результата первого
   2. Выравнивайте вложенные CompletableFutures
   3. Последовательные асинхронные операции

3. Используйте allOf() для ожидания нескольких futures:
   1. Ожидайте завершения всех futures
   2. Возвращает CompletableFuture<Void>
   3. Собирайте результаты со всех futures

4. Используйте anyOf() для получения первого завершенного future:
   1. Возвращается когда любой future завершен
   2. Полезно для соревнования нескольких операций
   3. Получите самый быстрый результат

## Пример вывода:
\`\`\`
=== Combining Futures ===

--- thenCombine: Parallel Execution ---
Starting user fetch...
Starting order fetch...
Combined result: User: Alice, Orders: 5

--- thenCompose: Sequential Composition ---
Step 1: Getting user ID...
Step 2: Fetching user details for ID: 123
Result: User Details: Alice (ID: 123)

--- allOf: Wait for All ---
Fetching from API 1...
Fetching from API 2...
Fetching from API 3...
All APIs responded!
API 1: Data from API 1
API 2: Data from API 2
API 3: Data from API 3

--- anyOf: First to Complete ---
Starting query 1 (3000ms)...
Starting query 2 (1000ms)...
Starting query 3 (2000ms)...
First result: Query 2 result
\`\`\``,
            hint1: `thenCombine() выполняет два futures параллельно и объединяет результаты. thenCompose() объединяет futures в цепочку, где второй зависит от результата первого. Используйте compose для выравнивания вложенных futures.`,
            hint2: `allOf() ожидает завершения всех futures - полезно для параллельных пакетных операций. anyOf() возвращает первый завершенный future - полезно для соревнования нескольких источников или реализации таймаутов.`,
            whyItMatters: `Комбинирование futures необходимо для построения реальных асинхронных приложений. Вам часто нужно получать данные из нескольких источников параллельно (thenCombine, allOf), объединять зависимые операции в цепочку (thenCompose) или реализовывать механизмы резервирования (anyOf). Эти паттерны фундаментальны для микросервисов, агрегации API и высокопроизводительных приложений.

**Продакшен паттерн:**
\`\`\`java
// Параллельная агрегация данных из микросервисов
CompletableFuture<User> userService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/users/" + userId));

CompletableFuture<List<Order>> orderService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/orders?userId=" + userId));

CompletableFuture<Preferences> prefService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/preferences/" + userId));

// Дождаться всех и собрать результаты
CompletableFuture.allOf(userService, orderService, prefService)
    .thenApply(v -> new Dashboard(
        userService.join(),
        orderService.join(),
        prefService.join()
    ));

// Или получить самый быстрый ответ (резервирование)
CompletableFuture.anyOf(primaryDB, secondaryDB, cacheDB)
    .thenAccept(result -> processData(result));
\`\`\`

**Практические преимущества:**
- Параллельные запросы к нескольким сервисам экономят время
- thenCompose предотвращает вложенные CompletableFuture<CompletableFuture<T>>
- anyOf позволяет реализовать failover и выбор самого быстрого источника
- allOf упрощает координацию множественных асинхронных задач`
        },
        uz: {
            title: `Future larni birlashtirish (thenCombine, thenCompose, allOf, anyOf)`,
            solutionCode: `import java.util.concurrent.*;
import java.util.*;
import java.util.stream.*;

public class CombiningFutures {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Future larni birlashtirish ===");

        // thenCombine: Ikkita mustaqil future larni birlashtirish (parallel bajarilish)
        System.out.println("\\n--- thenCombine: Parallel bajarilish ---");
        CompletableFuture<String> userFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting user fetch...");
            sleep(1000);
            return "Alice";
        });

        CompletableFuture<Integer> ordersFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting order fetch...");
            sleep(1500);
            return 5;
        });

        // Ikkalasi ham yakunlanganda natijalarni birlashtirish
        CompletableFuture<String> combined = userFuture.thenCombine(
            ordersFuture,
            (user, orders) -> "User: " + user + ", Orders: " + orders
        );

        System.out.println("Combined result: " + combined.join());

        // thenCompose: Bog'liq future lar zanjiri (ketma-ket bajarilish)
        System.out.println("\\n--- thenCompose: Ketma-ket kompozitsiya ---");
        CompletableFuture<String> composedResult = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Getting user ID...");
                sleep(500);
                return 123;
            })
            .thenCompose(userId -> {
                // Ikkinchi future birinchi natijaga bog'liq
                System.out.println("Step 2: Fetching user details for ID: " + userId);
                return CompletableFuture.supplyAsync(() -> {
                    sleep(500);
                    return "User Details: Alice (ID: " + userId + ")";
                });
            });

        System.out.println("Result: " + composedResult.join());

        // allOf: Barcha future larning yakunlanishini kutish
        System.out.println("\\n--- allOf: Barchasini kutish ---");
        CompletableFuture<String> api1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 1...");
            sleep(1000);
            return "Data from API 1";
        });

        CompletableFuture<String> api2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 2...");
            sleep(800);
            return "Data from API 2";
        });

        CompletableFuture<String> api3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Fetching from API 3...");
            sleep(1200);
            return "Data from API 3";
        });

        // Barchasining yakunlanishini kutish
        CompletableFuture<Void> allDone = CompletableFuture.allOf(api1, api2, api3);
        allDone.join(); // Barchasi yakunlanguncha blokirovka qiladi

        System.out.println("All APIs responded!");
        System.out.println("API 1: " + api1.join());
        System.out.println("API 2: " + api2.join());
        System.out.println("API 3: " + api3.join());

        // anyOf: Birinchi yakunlangan future dan natija olish
        System.out.println("\\n--- anyOf: Birinchi yakunlangan ---");
        CompletableFuture<String> query1 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 1 (3000ms)...");
            sleep(3000);
            return "Query 1 result";
        });

        CompletableFuture<String> query2 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 2 (1000ms)...");
            sleep(1000);
            return "Query 2 result";
        });

        CompletableFuture<String> query3 = CompletableFuture.supplyAsync(() -> {
            System.out.println("Starting query 3 (2000ms)...");
            sleep(2000);
            return "Query 3 result";
        });

        // Birinchi yakunlangandan natija olish
        CompletableFuture<Object> firstDone = CompletableFuture.anyOf(query1, query2, query3);
        System.out.println("First result: " + firstDone.join());
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}`,
            description: `# Future larni birlashtirish

CompletableFuture bir nechta asinxron operatsiyalarni birlashtirish uchun kuchli usullarni taqdim etadi. Siz ikkita future ni birlashtirish, bog'liq operatsiyalarni kompozitsiya qilish yoki bir nechta mustaqil future larni muvofiqlashtirish mumkin. Bu usullar murakkab asinxron ish oqimlarini qurish uchun zarurdir.

## Talablar:
1. Ikkita mustaqil future ni birlashtirish uchun thenCombine() dan foydalaning:
   1. Ikkala future ham parallel bajariladi
   2. Ikkalasi ham yakunlanganda natijalarni birlashtiring
   3. Natijalarni birlashtirish uchun BiFunction qabul qiladi

2. Bog'liq future lar uchun thenCompose() dan foydalaning:
   1. Ikkinchi future birinchisining natijasiga bog'liq
   2. Ichki CompletableFuture larni tekislang
   3. Ketma-ket asinxron operatsiyalar

3. Bir nechta future larni kutish uchun allOf() dan foydalaning:
   1. Barcha future larning yakunlanishini kuting
   2. CompletableFuture<Void> qaytaradi
   3. Barcha future lardan natijalarni to'plang

4. Birinchi yakunlangan future ni olish uchun anyOf() dan foydalaning:
   1. Har qanday future yakunlanganda qaytadi
   2. Bir nechta operatsiyalarni musobaqalash uchun foydali
   3. Eng tez natijani oling

## Chiqish namunasi:
\`\`\`
=== Combining Futures ===

--- thenCombine: Parallel Execution ---
Starting user fetch...
Starting order fetch...
Combined result: User: Alice, Orders: 5

--- thenCompose: Sequential Composition ---
Step 1: Getting user ID...
Step 2: Fetching user details for ID: 123
Result: User Details: Alice (ID: 123)

--- allOf: Wait for All ---
Fetching from API 1...
Fetching from API 2...
Fetching from API 3...
All APIs responded!
API 1: Data from API 1
API 2: Data from API 2
API 3: Data from API 3

--- anyOf: First to Complete ---
Starting query 1 (3000ms)...
Starting query 2 (1000ms)...
Starting query 3 (2000ms)...
First result: Query 2 result
\`\`\``,
            hint1: `thenCombine() ikkita future ni parallel bajaradi va natijalarni birlashtiradi. thenCompose() future larni zanjirlaydi, ikkinchisi birinchisining natijasiga bog'liq. Ichki future larni tekislash uchun compose dan foydalaning.`,
            hint2: `allOf() barcha future larning yakunlanishini kutadi - parallel paket operatsiyalari uchun foydali. anyOf() birinchi yakunlangan future ni qaytaradi - bir nechta manbalarni musobaqalash yoki vaqt tugashini amalga oshirish uchun foydali.`,
            whyItMatters: `Future larni birlashtirish real asinxron ilovalarni qurish uchun zarurdir. Ko'pincha siz bir nechta manbalardan parallel ravishda ma'lumotlarni olishingiz (thenCombine, allOf), bog'liq operatsiyalarni zanjirlashingiz (thenCompose) yoki zaxira mexanizmlarini amalga oshirishingiz (anyOf) kerak bo'ladi. Bu naqshlar mikroservislar, API agregatsiyasi va yuqori samarali ilovalar uchun asosiy hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Mikroservislardan parallel ma'lumotlar agregatsiyasi
CompletableFuture<User> userService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/users/" + userId));

CompletableFuture<List<Order>> orderService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/orders?userId=" + userId));

CompletableFuture<Preferences> prefService = CompletableFuture.supplyAsync(() ->
    httpClient.get("/api/preferences/" + userId));

// Barchasini kutish va natijalarni yig'ish
CompletableFuture.allOf(userService, orderService, prefService)
    .thenApply(v -> new Dashboard(
        userService.join(),
        orderService.join(),
        prefService.join()
    ));

// Yoki eng tez javobni olish (zaxira)
CompletableFuture.anyOf(primaryDB, secondaryDB, cacheDB)
    .thenAccept(result -> processData(result));
\`\`\`

**Amaliy foydalari:**
- Bir nechta xizmatlarga parallel so'rovlar vaqtni tejaydi
- thenCompose ichki CompletableFuture<CompletableFuture<T>> dan qochadi
- anyOf failover va eng tez manbani tanlashni amalga oshirish imkon beradi
- allOf ko'plab asinxron vazifalarni muvofiqlashtirish ni soddalashtiradi`
        }
    }
};

export default task;
