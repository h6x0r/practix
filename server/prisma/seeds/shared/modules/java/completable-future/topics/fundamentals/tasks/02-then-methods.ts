import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-then-methods',
    title: 'Then Methods (thenApply, thenAccept, thenRun)',
    difficulty: 'medium',
    tags: ['java', 'completablefuture', 'async', 'chaining', 'composition'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Then Methods

CompletableFuture provides "then" methods for chaining asynchronous operations. These methods allow you to transform results, consume values, or perform actions after a future completes. Understanding when to use each method is crucial for building async pipelines.

## Requirements:
1. Use thenApply() to transform results:
   1. Takes Function<T, U> and returns CompletableFuture<U>
   2. Transform the result of previous stage
   3. Chain multiple transformations

2. Use thenAccept() to consume results:
   1. Takes Consumer<T> and returns CompletableFuture<Void>
   2. Consume the result without returning new value
   3. Useful for side effects (logging, saving, etc.)

3. Use thenRun() to execute actions:
   1. Takes Runnable and returns CompletableFuture<Void>
   2. Execute code after completion
   3. No access to previous result

4. Demonstrate method chaining:
   1. Combine multiple then methods
   2. Show execution order
   3. Build complete async pipeline

## Example Output:
\`\`\`
=== Then Methods ===

--- thenApply: Transform Results ---
Original value: 10
After doubling: 20
After adding 5: 25
Final result: 25

--- thenAccept: Consume Results ---
Processing user: John Doe
User data saved to database
Notification sent to John Doe

--- thenRun: Execute Actions ---
Task completed successfully
Cleanup operations started...
Cleanup completed

--- Method Chaining Pipeline ---
Step 1: Fetching data...
Step 2: Processing: RAW DATA
Step 3: Formatted: raw data
Step 4: Saving: raw data
Pipeline completed!
\`\`\``,
    initialCode: `import java.util.concurrent.*;

public class ThenMethods {
    public static void main(String[] args) {
        // TODO: Demonstrate thenApply for transformations

        // TODO: Demonstrate thenAccept for consuming results

        // TODO: Demonstrate thenRun for running actions

        // TODO: Create a complete chaining pipeline
    }
}`,
    solutionCode: `import java.util.concurrent.*;

public class ThenMethods {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Then Methods ===");

        // thenApply: Transform the result of CompletableFuture
        System.out.println("\\n--- thenApply: Transform Results ---");
        CompletableFuture<Integer> applyFuture = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Original value: 10");
                return 10;
            })
            .thenApply(value -> {
                int doubled = value * 2;
                System.out.println("After doubling: " + doubled);
                return doubled;
            })
            .thenApply(value -> {
                int added = value + 5;
                System.out.println("After adding 5: " + added);
                return added;
            });

        System.out.println("Final result: " + applyFuture.join());

        // thenAccept: Consume the result without returning new value
        System.out.println("\\n--- thenAccept: Consume Results ---");
        CompletableFuture<Void> acceptFuture = CompletableFuture
            .supplyAsync(() -> "John Doe")
            .thenAccept(name -> {
                System.out.println("Processing user: " + name);
                // Simulate database save
                System.out.println("User data saved to database");
            })
            .thenAccept(v -> {
                // No access to previous value, v is null
                System.out.println("Notification sent to John Doe");
            });

        acceptFuture.join(); // Wait for completion

        // thenRun: Execute code after completion (no access to result)
        System.out.println("\\n--- thenRun: Execute Actions ---");
        CompletableFuture<Void> runFuture = CompletableFuture
            .supplyAsync(() -> {
                // Simulate some task
                return "Task data";
            })
            .thenRun(() -> {
                // No access to "Task data"
                System.out.println("Task completed successfully");
            })
            .thenRun(() -> {
                System.out.println("Cleanup operations started...");
                // Perform cleanup
                System.out.println("Cleanup completed");
            });

        runFuture.join();

        // Complete chaining pipeline demonstrating all methods
        System.out.println("\\n--- Method Chaining Pipeline ---");
        CompletableFuture<Void> pipeline = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Fetching data...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                return "RAW DATA";
            })
            .thenApply(data -> {
                System.out.println("Step 2: Processing: " + data);
                return data.toLowerCase();
            })
            .thenApply(data -> {
                System.out.println("Step 3: Formatted: " + data);
                return data;
            })
            .thenAccept(data -> {
                System.out.println("Step 4: Saving: " + data);
                // Save to database
            })
            .thenRun(() -> {
                System.out.println("Pipeline completed!");
            });

        pipeline.join();
    }
}`,
    hint1: `thenApply() transforms the result and returns a new value. thenAccept() consumes the result for side effects. thenRun() just executes code after completion.`,
    hint2: `Chain methods together to create async pipelines. Each "then" method is executed after the previous stage completes. The execution happens asynchronously.`,
    whyItMatters: `Then methods are fundamental to CompletableFuture's composability. They enable building complex async workflows in a readable, sequential manner. Understanding when to use thenApply (transform), thenAccept (consume), or thenRun (action) is essential for effective async programming and avoiding common mistakes like blocking unnecessarily.

**Production Pattern:**
\`\`\`java
// API order processing pipeline
CompletableFuture.supplyAsync(() -> orderRepository.findById(orderId))
    .thenApply(order -> {
        // Transformation: validation and data enrichment
        return validator.validate(order);
    })
    .thenApply(validatedOrder -> {
        // Transformation: calculate total amount
        return priceCalculator.calculate(validatedOrder);
    })
    .thenAccept(finalOrder -> {
        // Consumption: save to database
        orderRepository.save(finalOrder);
        logger.info("Order saved: {}", finalOrder.getId());
    })
    .thenRun(() -> {
        // Action: clear cache
        cache.invalidate("orders");
    });
\`\`\`

**Practical Benefits:**
- Creates readable async pipelines without nested callbacks
- Each stage executes asynchronously, maximizing parallelism
- Easy to add new processing stages
- Clear separation of transformation, consumption, and side effects`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Test thenApply transforms value
class Test1 {
    @Test
    public void test() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 10)
            .thenApply(x -> x * 2);
        assertEquals(Integer.valueOf(20), future.join());
    }
}

// Test2: Test chaining multiple thenApply
class Test2 {
    @Test
    public void test() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 5)
            .thenApply(x -> x + 10)
            .thenApply(x -> x * 2);
        assertEquals(Integer.valueOf(30), future.join());
    }
}

// Test3: Test thenAccept consumes value
class Test3 {
    @Test
    public void test() {
        final int[] result = {0};
        CompletableFuture<Void> future = CompletableFuture.supplyAsync(() -> 42)
            .thenAccept(x -> result[0] = x);
        future.join();
        assertEquals(42, result[0]);
    }
}

// Test4: Test thenRun executes without value
class Test4 {
    @Test
    public void test() {
        final boolean[] executed = {false};
        CompletableFuture<Void> future = CompletableFuture.supplyAsync(() -> "data")
            .thenRun(() -> executed[0] = true);
        future.join();
        assertTrue(executed[0]);
    }
}

// Test5: Test thenApply with string transformation
class Test5 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "hello")
            .thenApply(String::toUpperCase);
        assertEquals("HELLO", future.join());
    }
}

// Test6: Test combination of thenApply and thenAccept
class Test6 {
    @Test
    public void test() {
        final String[] result = {null};
        CompletableFuture<Void> future = CompletableFuture.supplyAsync(() -> 100)
            .thenApply(x -> "Value: " + x)
            .thenAccept(s -> result[0] = s);
        future.join();
        assertEquals("Value: 100", result[0]);
    }
}

// Test7: Test thenAccept returns Void
class Test7 {
    @Test
    public void test() {
        CompletableFuture<Void> future = CompletableFuture.supplyAsync(() -> "test")
            .thenAccept(s -> System.out.println(s));
        assertNull(future.join());
    }
}

// Test8: Test thenRun after thenApply
class Test8 {
    @Test
    public void test() {
        final boolean[] flag = {false};
        CompletableFuture.supplyAsync(() -> 42)
            .thenApply(x -> x * 2)
            .thenRun(() -> flag[0] = true)
            .join();
        assertTrue(flag[0]);
    }
}

// Test9: Test thenApply with null handling
class Test9 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "data")
            .thenApply(s -> s + " processed");
        assertEquals("data processed", future.join());
    }
}

// Test10: Test complete pipeline with all then methods
class Test10 {
    @Test
    public void test() {
        final int[] count = {0};
        CompletableFuture.supplyAsync(() -> 10)
            .thenApply(x -> x + 5)
            .thenApply(x -> x * 2)
            .thenAccept(x -> count[0] = x)
            .thenRun(() -> count[0] += 1)
            .join();
        assertEquals(31, count[0]);
    }
}
`,
    translations: {
        ru: {
            title: 'Методы Then (thenApply, thenAccept, thenRun)',
            solutionCode: `import java.util.concurrent.*;

public class ThenMethods {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Методы Then ===");

        // thenApply: Трансформация результата CompletableFuture
        System.out.println("\\n--- thenApply: Трансформация результатов ---");
        CompletableFuture<Integer> applyFuture = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Original value: 10");
                return 10;
            })
            .thenApply(value -> {
                int doubled = value * 2;
                System.out.println("After doubling: " + doubled);
                return doubled;
            })
            .thenApply(value -> {
                int added = value + 5;
                System.out.println("After adding 5: " + added);
                return added;
            });

        System.out.println("Final result: " + applyFuture.join());

        // thenAccept: Потребление результата без возврата нового значения
        System.out.println("\\n--- thenAccept: Потребление результатов ---");
        CompletableFuture<Void> acceptFuture = CompletableFuture
            .supplyAsync(() -> "John Doe")
            .thenAccept(name -> {
                System.out.println("Processing user: " + name);
                // Симуляция сохранения в БД
                System.out.println("User data saved to database");
            })
            .thenAccept(v -> {
                // Нет доступа к предыдущему значению, v равно null
                System.out.println("Notification sent to John Doe");
            });

        acceptFuture.join(); // Ожидание завершения

        // thenRun: Выполнение кода после завершения (нет доступа к результату)
        System.out.println("\\n--- thenRun: Выполнение действий ---");
        CompletableFuture<Void> runFuture = CompletableFuture
            .supplyAsync(() -> {
                // Симуляция задачи
                return "Task data";
            })
            .thenRun(() -> {
                // Нет доступа к "Task data"
                System.out.println("Task completed successfully");
            })
            .thenRun(() -> {
                System.out.println("Cleanup operations started...");
                // Выполнение очистки
                System.out.println("Cleanup completed");
            });

        runFuture.join();

        // Полный конвейер цепочки, демонстрирующий все методы
        System.out.println("\\n--- Конвейер цепочки методов ---");
        CompletableFuture<Void> pipeline = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Fetching data...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                return "RAW DATA";
            })
            .thenApply(data -> {
                System.out.println("Step 2: Processing: " + data);
                return data.toLowerCase();
            })
            .thenApply(data -> {
                System.out.println("Step 3: Formatted: " + data);
                return data;
            })
            .thenAccept(data -> {
                System.out.println("Step 4: Saving: " + data);
                // Сохранение в БД
            })
            .thenRun(() -> {
                System.out.println("Pipeline completed!");
            });

        pipeline.join();
    }
}`,
            description: `# Методы Then

CompletableFuture предоставляет методы "then" для цепочки асинхронных операций. Эти методы позволяют трансформировать результаты, потреблять значения или выполнять действия после завершения future. Понимание того, когда использовать каждый метод, критически важно для построения асинхронных конвейеров.

## Требования:
1. Используйте thenApply() для трансформации результатов:
   1. Принимает Function<T, U> и возвращает CompletableFuture<U>
   2. Трансформируйте результат предыдущего этапа
   3. Цепочка нескольких трансформаций

2. Используйте thenAccept() для потребления результатов:
   1. Принимает Consumer<T> и возвращает CompletableFuture<Void>
   2. Потребляйте результат без возврата нового значения
   3. Полезно для побочных эффектов (логирование, сохранение и т.д.)

3. Используйте thenRun() для выполнения действий:
   1. Принимает Runnable и возвращает CompletableFuture<Void>
   2. Выполняйте код после завершения
   3. Нет доступа к предыдущему результату

4. Продемонстрируйте цепочку методов:
   1. Объедините несколько методов then
   2. Покажите порядок выполнения
   3. Постройте полный асинхронный конвейер

## Пример вывода:
\`\`\`
=== Then Methods ===

--- thenApply: Transform Results ---
Original value: 10
After doubling: 20
After adding 5: 25
Final result: 25

--- thenAccept: Consume Results ---
Processing user: John Doe
User data saved to database
Notification sent to John Doe

--- thenRun: Execute Actions ---
Task completed successfully
Cleanup operations started...
Cleanup completed

--- Method Chaining Pipeline ---
Step 1: Fetching data...
Step 2: Processing: RAW DATA
Step 3: Formatted: raw data
Step 4: Saving: raw data
Pipeline completed!
\`\`\``,
            hint1: `thenApply() трансформирует результат и возвращает новое значение. thenAccept() потребляет результат для побочных эффектов. thenRun() просто выполняет код после завершения.`,
            hint2: `Объединяйте методы в цепочку для создания асинхронных конвейеров. Каждый метод "then" выполняется после завершения предыдущего этапа. Выполнение происходит асинхронно.`,
            whyItMatters: `Методы Then являются основой композируемости CompletableFuture. Они позволяют строить сложные асинхронные рабочие процессы в читаемой, последовательной манере. Понимание того, когда использовать thenApply (трансформация), thenAccept (потребление) или thenRun (действие), необходимо для эффективного асинхронного программирования и избежания распространенных ошибок, таких как ненужная блокировка.

**Продакшен паттерн:**
\`\`\`java
// API конвейер обработки заказа
CompletableFuture.supplyAsync(() -> orderRepository.findById(orderId))
    .thenApply(order -> {
        // Трансформация: валидация и обогащение данных
        return validator.validate(order);
    })
    .thenApply(validatedOrder -> {
        // Трансформация: расчет итоговой суммы
        return priceCalculator.calculate(validatedOrder);
    })
    .thenAccept(finalOrder -> {
        // Потребление: сохранение в БД
        orderRepository.save(finalOrder);
        logger.info("Order saved: {}", finalOrder.getId());
    })
    .thenRun(() -> {
        // Действие: очистка кеша
        cache.invalidate("orders");
    });
\`\`\`

**Практические преимущества:**
- Создает читаемые асинхронные конвейеры без вложенных callback
- Каждый этап выполняется асинхронно, максимизируя параллелизм
- Легко добавлять новые этапы обработки
- Явное разделение трансформации, потребления и побочных эффектов`
        },
        uz: {
            title: `Then usullari (thenApply, thenAccept, thenRun)`,
            solutionCode: `import java.util.concurrent.*;

public class ThenMethods {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Then usullari ===");

        // thenApply: CompletableFuture natijasini o'zgartirish
        System.out.println("\\n--- thenApply: Natijalarni o'zgartirish ---");
        CompletableFuture<Integer> applyFuture = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Original value: 10");
                return 10;
            })
            .thenApply(value -> {
                int doubled = value * 2;
                System.out.println("After doubling: " + doubled);
                return doubled;
            })
            .thenApply(value -> {
                int added = value + 5;
                System.out.println("After adding 5: " + added);
                return added;
            });

        System.out.println("Final result: " + applyFuture.join());

        // thenAccept: Yangi qiymat qaytarmasdan natijani iste'mol qilish
        System.out.println("\\n--- thenAccept: Natijalarni iste'mol qilish ---");
        CompletableFuture<Void> acceptFuture = CompletableFuture
            .supplyAsync(() -> "John Doe")
            .thenAccept(name -> {
                System.out.println("Processing user: " + name);
                // Ma'lumotlar bazasiga saqlashni simulyatsiya qilish
                System.out.println("User data saved to database");
            })
            .thenAccept(v -> {
                // Oldingi qiymatga kirish yo'q, v null
                System.out.println("Notification sent to John Doe");
            });

        acceptFuture.join(); // Yakunlanishini kutish

        // thenRun: Yakunlangandan keyin kodni bajarish (natijaga kirish yo'q)
        System.out.println("\\n--- thenRun: Harakatlarni bajarish ---");
        CompletableFuture<Void> runFuture = CompletableFuture
            .supplyAsync(() -> {
                // Vazifani simulyatsiya qilish
                return "Task data";
            })
            .thenRun(() -> {
                // "Task data" ga kirish yo'q
                System.out.println("Task completed successfully");
            })
            .thenRun(() -> {
                System.out.println("Cleanup operations started...");
                // Tozalashni amalga oshirish
                System.out.println("Cleanup completed");
            });

        runFuture.join();

        // Barcha usullarni namoyish etuvchi to'liq zanjir konveyeri
        System.out.println("\\n--- Usullar zanjiri konveyeri ---");
        CompletableFuture<Void> pipeline = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Fetching data...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                return "RAW DATA";
            })
            .thenApply(data -> {
                System.out.println("Step 2: Processing: " + data);
                return data.toLowerCase();
            })
            .thenApply(data -> {
                System.out.println("Step 3: Formatted: " + data);
                return data;
            })
            .thenAccept(data -> {
                System.out.println("Step 4: Saving: " + data);
                // Ma'lumotlar bazasiga saqlash
            })
            .thenRun(() -> {
                System.out.println("Pipeline completed!");
            });

        pipeline.join();
    }
}`,
            description: `# Then usullari

CompletableFuture asinxron operatsiyalarni zanjirga bog'lash uchun "then" usullarini taqdim etadi. Bu usullar natijalarni o'zgartirish, qiymatlarni iste'mol qilish yoki future yakunlangandan keyin harakatlarni bajarishga imkon beradi. Har bir usuldan qachon foydalanishni tushunish asinxron konveyerlarni qurish uchun juda muhimdir.

## Talablar:
1. Natijalarni o'zgartirish uchun thenApply() dan foydalaning:
   1. Function<T, U> qabul qiladi va CompletableFuture<U> qaytaradi
   2. Oldingi bosqich natijasini o'zgartiring
   3. Bir nechta o'zgartirishlarni zanjirlang

2. Natijalarni iste'mol qilish uchun thenAccept() dan foydalaning:
   1. Consumer<T> qabul qiladi va CompletableFuture<Void> qaytaradi
   2. Yangi qiymat qaytarmasdan natijani iste'mol qiling
   3. Yon ta'sirlar uchun foydali (jurnal yuritish, saqlash va boshqalar)

3. Harakatlarni bajarish uchun thenRun() dan foydalaning:
   1. Runnable qabul qiladi va CompletableFuture<Void> qaytaradi
   2. Yakunlangandan keyin kodni bajaring
   3. Oldingi natijaga kirish yo'q

4. Usullar zanjirini namoyish eting:
   1. Bir nechta then usullarini birlashtiring
   2. Bajarilish tartibini ko'rsating
   3. To'liq asinxron konveyerni quring

## Chiqish namunasi:
\`\`\`
=== Then Methods ===

--- thenApply: Transform Results ---
Original value: 10
After doubling: 20
After adding 5: 25
Final result: 25

--- thenAccept: Consume Results ---
Processing user: John Doe
User data saved to database
Notification sent to John Doe

--- thenRun: Execute Actions ---
Task completed successfully
Cleanup operations started...
Cleanup completed

--- Method Chaining Pipeline ---
Step 1: Fetching data...
Step 2: Processing: RAW DATA
Step 3: Formatted: raw data
Step 4: Saving: raw data
Pipeline completed!
\`\`\``,
            hint1: `thenApply() natijani o'zgartiradi va yangi qiymat qaytaradi. thenAccept() yon ta'sirlar uchun natijani iste'mol qiladi. thenRun() faqat yakunlangandan keyin kodni bajaradi.`,
            hint2: `Asinxron konveyerlarni yaratish uchun usullarni zanjirlang. Har bir "then" usuli oldingi bosqich yakunlangandan keyin bajariladi. Bajarilish asinxron amalga oshiriladi.`,
            whyItMatters: `Then usullari CompletableFuture ning kompozitsiyasining asosidir. Ular murakkab asinxron ish oqimlarini o'qilishi oson, ketma-ket tarzda qurishga imkon beradi. thenApply (o'zgartirish), thenAccept (iste'mol qilish) yoki thenRun (harakat) dan qachon foydalanishni tushunish samarali asinxron dasturlash va keraksiz blokirovka qilish kabi keng tarqalgan xatolardan qochish uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Buyurtmani qayta ishlash API konveyeri
CompletableFuture.supplyAsync(() -> orderRepository.findById(orderId))
    .thenApply(order -> {
        // Transformatsiya: ma'lumotlarni tekshirish va boyitish
        return validator.validate(order);
    })
    .thenApply(validatedOrder -> {
        // Transformatsiya: jami summani hisoblash
        return priceCalculator.calculate(validatedOrder);
    })
    .thenAccept(finalOrder -> {
        // Iste'mol qilish: ma'lumotlar bazasiga saqlash
        orderRepository.save(finalOrder);
        logger.info("Order saved: {}", finalOrder.getId());
    })
    .thenRun(() -> {
        // Harakat: keshni tozalash
        cache.invalidate("orders");
    });
\`\`\`

**Amaliy foydalari:**
- Ichki callback larsiz o'qilishi oson asinxron konveyerlar yaratadi
- Har bir bosqich asinxron bajariladi, parallellikni maksimallashtiradi
- Yangi qayta ishlash bosqichlarini qo'shish oson
- Transformatsiya, iste'mol qilish va yon ta'sirlarning aniq ajratilishi`
        }
    }
};

export default task;
