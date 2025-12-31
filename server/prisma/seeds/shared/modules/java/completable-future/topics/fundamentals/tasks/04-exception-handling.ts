import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-exception-handling',
    title: 'Exception Handling (exceptionally, handle, whenComplete)',
    difficulty: 'medium',
    tags: ['java', 'completablefuture', 'exceptions', 'error-handling', 'async'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Exception Handling in CompletableFuture

Proper exception handling is crucial in asynchronous programming. CompletableFuture provides three main methods for handling exceptions: exceptionally() for recovery, handle() for handling both success and failure, and whenComplete() for performing cleanup or logging.

## Requirements:
1. Use exceptionally() to recover from exceptions:
   1. Provide fallback value on error
   2. Only called if exception occurs
   3. Returns CompletableFuture with recovery value

2. Use handle() to handle both success and failure:
   1. Called in both success and error cases
   2. Receives result and exception (one is null)
   3. Transform result or recover from error

3. Use whenComplete() for side effects:
   1. Called after completion (success or failure)
   2. Cannot modify result or exception
   3. Useful for logging, cleanup, metrics

4. Demonstrate exception propagation in chains:
   1. Show how exceptions flow through pipeline
   2. Handle exceptions at different stages
   3. Recover gracefully from failures

## Example Output:
\`\`\`
=== Exception Handling ===

--- exceptionally: Recover from Errors ---
Error case: java.lang.RuntimeException: Database unavailable
Recovered with: Default User

--- handle: Handle Both Success and Failure ---
Success case: Result: DATA (processed successfully)
Error case: Error occurred: Simulated failure, using fallback

--- whenComplete: Side Effects ---
Processing order...
Completed with result: Order #12345
Logging: Operation completed

Handling error case...
Completed with exception: java.lang.RuntimeException: Order failed
Logging: Operation failed

--- Exception Propagation ---
Step 1: Starting...
Step 2: Error occurred!
Final result (recovered): Operation failed, using backup data
\`\`\``,
    initialCode: `import java.util.concurrent.*;

public class ExceptionHandling {
    public static void main(String[] args) {
        // TODO: Demonstrate exceptionally for recovery

        // TODO: Demonstrate handle for both cases

        // TODO: Demonstrate whenComplete for side effects

        // TODO: Show exception propagation in chains
    }
}`,
    solutionCode: `import java.util.concurrent.*;

public class ExceptionHandling {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Exception Handling ===");

        // exceptionally: Recover from exceptions with fallback value
        System.out.println("\\n--- exceptionally: Recover from Errors ---");
        CompletableFuture<String> withError = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Database unavailable");
        }).exceptionally(ex -> {
            // Only called if exception occurs
            System.out.println("Error case: " + ex);
            return "Default User"; // Fallback value
        });

        System.out.println("Recovered with: " + withError.join());

        // handle: Handle both success and failure cases
        System.out.println("\\n--- handle: Handle Both Success and Failure ---");

        // Success case
        CompletableFuture<String> successCase = CompletableFuture
            .supplyAsync(() -> "DATA")
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error: " + ex.getMessage();
                } else {
                    return "Result: " + result + " (processed successfully)";
                }
            });
        System.out.println("Success case: " + successCase.join());

        // Error case
        CompletableFuture<String> errorCase = CompletableFuture
            .supplyAsync(() -> {
                throw new RuntimeException("Simulated failure");
            })
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error occurred: " + ex.getCause().getMessage() + ", using fallback";
                } else {
                    return "Result: " + result;
                }
            });
        System.out.println("Error case: " + errorCase.join());

        // whenComplete: Perform side effects (logging, cleanup)
        System.out.println("\\n--- whenComplete: Side Effects ---");

        // Success case with side effects
        CompletableFuture<String> completeSuccess = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Processing order...");
                return "Order #12345";
            })
            .whenComplete((result, ex) -> {
                // Called in both success and error cases
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex);
                } else {
                    System.out.println("Completed with result: " + result);
                }
                System.out.println("Logging: Operation completed");
            });
        completeSuccess.join();

        // Error case with side effects
        CompletableFuture<String> completeError = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("\\nHandling error case...");
                throw new RuntimeException("Order failed");
            })
            .whenComplete((result, ex) -> {
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex.getCause());
                    System.out.println("Logging: Operation failed");
                } else {
                    System.out.println("Completed with result: " + result);
                }
            })
            .exceptionally(ex -> "Error handled"); // Recover to prevent exception

        completeError.join();

        // Exception propagation in chains
        System.out.println("\\n--- Exception Propagation ---");
        CompletableFuture<String> chain = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Starting...");
                return "initial";
            })
            .thenApply(s -> {
                System.out.println("Step 2: Error occurred!");
                throw new RuntimeException("Processing failed");
            })
            .thenApply(s -> {
                // This won't execute because of exception
                System.out.println("Step 3: This won't print");
                return s.toUpperCase();
            })
            .exceptionally(ex -> {
                // Exception propagates here
                return "Operation failed, using backup data";
            });

        System.out.println("Final result (recovered): " + chain.join());
    }
}`,
    hint1: `exceptionally() is like a catch block - it only runs on error and provides a fallback value. handle() is like finally - it always runs and receives both result and exception.`,
    hint2: `whenComplete() is for side effects only (logging, metrics) - it cannot change the result. Exceptions propagate through the chain until handled by exceptionally() or handle().`,
    whyItMatters: `Robust exception handling is critical in production async code. Without proper error handling, failures can go unnoticed, cause cascading errors, or crash the application. Understanding exceptionally() for recovery, handle() for transformation, and whenComplete() for cleanup ensures your async code is resilient and maintainable. These patterns are essential for building fault-tolerant distributed systems.

**Production Pattern:**
\`\`\`java
// Fault-tolerant external API call with fallback data
CompletableFuture.supplyAsync(() -> {
    return externalAPI.fetchUserData(userId);
})
.exceptionally(ex -> {
    // Log error and return cached data
    logger.error("API failed, using cache", ex);
    return cache.get(userId);
})
.handle((result, ex) -> {
    if (ex != null) {
        // Even cache failed - use default values
        metrics.incrementFailureCount();
        return UserData.defaultUser();
    }
    metrics.incrementSuccessCount();
    return result;
})
.whenComplete((result, ex) -> {
    // Always executes: update metrics and log
    logger.info("Request completed for user: {}", userId);
    requestTracker.finish(userId);
});
\`\`\`

**Practical Benefits:**
- exceptionally provides graceful degradation on failures
- handle enables complex recovery logic
- whenComplete guarantees resource cleanup execution
- Prevents silent failures in async chains`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Test exceptionally recovers from error
class Test1 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Error");
        }).exceptionally(ex -> "Recovered");
        assertEquals("Recovered", future.join());
    }
}

// Test2: Test handle with successful completion
class Test2 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Success")
            .handle((result, ex) -> ex != null ? "Error" : result);
        assertEquals("Success", future.join());
    }
}

// Test3: Test handle with exception
class Test3 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Failed");
        }).handle((result, ex) -> ex != null ? "Handled" : result);
        assertEquals("Handled", future.join());
    }
}

// Test4: Test whenComplete doesn't change result
class Test4 {
    @Test
    public void test() {
        final boolean[] called = {false};
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Value")
            .whenComplete((result, ex) -> called[0] = true);
        assertEquals("Value", future.join());
        assertTrue(called[0]);
    }
}

// Test5: Test exceptionally not called on success
class Test5 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "OK")
            .exceptionally(ex -> "Fallback");
        assertEquals("OK", future.join());
    }
}

// Test6: Test exception propagation in chain
class Test6 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Start")
            .thenApply(s -> { throw new RuntimeException("Error"); })
            .exceptionally(ex -> "Recovered");
        assertEquals("Recovered", future.join());
    }
}

// Test7: Test whenComplete with exception
class Test7 {
    @Test
    public void test() {
        final boolean[] exceptionDetected = {false};
        try {
            CompletableFuture.supplyAsync(() -> {
                throw new RuntimeException("Test");
            }).whenComplete((result, ex) -> {
                exceptionDetected[0] = (ex != null);
            }).join();
        } catch (Exception e) {
            assertTrue(exceptionDetected[0]);
        }
    }
}

// Test8: Test handle transforms both success and error
class Test8 {
    @Test
    public void test() {
        CompletableFuture<Integer> success = CompletableFuture.supplyAsync(() -> 42)
            .handle((result, ex) -> ex != null ? 0 : result);
        assertEquals(Integer.valueOf(42), success.join());
    }
}

// Test9: Test multiple exception handlers
class Test9 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Error");
        }).exceptionally(ex -> "First")
          .exceptionally(ex -> "Second");
        assertEquals("First", future.join());
    }
}

// Test10: Test whenComplete followed by exceptionally
class Test10 {
    @Test
    public void test() {
        final int[] count = {0};
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Error");
        }).whenComplete((result, ex) -> count[0]++)
          .exceptionally(ex -> "Handled");
        assertEquals("Handled", future.join());
        assertEquals(1, count[0]);
    }
}
`,
    translations: {
        ru: {
            title: 'Обработка исключений (exceptionally, handle, whenComplete)',
            solutionCode: `import java.util.concurrent.*;

public class ExceptionHandling {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Обработка исключений ===");

        // exceptionally: Восстановление после исключений с резервным значением
        System.out.println("\\n--- exceptionally: Восстановление после ошибок ---");
        CompletableFuture<String> withError = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Database unavailable");
        }).exceptionally(ex -> {
            // Вызывается только при возникновении исключения
            System.out.println("Error case: " + ex);
            return "Default User"; // Резервное значение
        });

        System.out.println("Recovered with: " + withError.join());

        // handle: Обработка случаев успеха и неудачи
        System.out.println("\\n--- handle: Обработка успеха и неудачи ---");

        // Случай успеха
        CompletableFuture<String> successCase = CompletableFuture
            .supplyAsync(() -> "DATA")
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error: " + ex.getMessage();
                } else {
                    return "Result: " + result + " (processed successfully)";
                }
            });
        System.out.println("Success case: " + successCase.join());

        // Случай ошибки
        CompletableFuture<String> errorCase = CompletableFuture
            .supplyAsync(() -> {
                throw new RuntimeException("Simulated failure");
            })
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error occurred: " + ex.getCause().getMessage() + ", using fallback";
                } else {
                    return "Result: " + result;
                }
            });
        System.out.println("Error case: " + errorCase.join());

        // whenComplete: Выполнение побочных эффектов (логирование, очистка)
        System.out.println("\\n--- whenComplete: Побочные эффекты ---");

        // Случай успеха с побочными эффектами
        CompletableFuture<String> completeSuccess = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Processing order...");
                return "Order #12345";
            })
            .whenComplete((result, ex) -> {
                // Вызывается в обоих случаях: успех и ошибка
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex);
                } else {
                    System.out.println("Completed with result: " + result);
                }
                System.out.println("Logging: Operation completed");
            });
        completeSuccess.join();

        // Случай ошибки с побочными эффектами
        CompletableFuture<String> completeError = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("\\nHandling error case...");
                throw new RuntimeException("Order failed");
            })
            .whenComplete((result, ex) -> {
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex.getCause());
                    System.out.println("Logging: Operation failed");
                } else {
                    System.out.println("Completed with result: " + result);
                }
            })
            .exceptionally(ex -> "Error handled"); // Восстановление для предотвращения исключения

        completeError.join();

        // Распространение исключений в цепочках
        System.out.println("\\n--- Распространение исключений ---");
        CompletableFuture<String> chain = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Starting...");
                return "initial";
            })
            .thenApply(s -> {
                System.out.println("Step 2: Error occurred!");
                throw new RuntimeException("Processing failed");
            })
            .thenApply(s -> {
                // Это не выполнится из-за исключения
                System.out.println("Step 3: This won't print");
                return s.toUpperCase();
            })
            .exceptionally(ex -> {
                // Исключение распространяется сюда
                return "Operation failed, using backup data";
            });

        System.out.println("Final result (recovered): " + chain.join());
    }
}`,
            description: `# Обработка исключений в CompletableFuture

Правильная обработка исключений критически важна в асинхронном программировании. CompletableFuture предоставляет три основных метода для обработки исключений: exceptionally() для восстановления, handle() для обработки как успеха, так и неудачи, и whenComplete() для выполнения очистки или логирования.

## Требования:
1. Используйте exceptionally() для восстановления после исключений:
   1. Предоставьте резервное значение при ошибке
   2. Вызывается только при возникновении исключения
   3. Возвращает CompletableFuture с восстановленным значением

2. Используйте handle() для обработки успеха и неудачи:
   1. Вызывается в обоих случаях: успех и ошибка
   2. Получает результат и исключение (одно из них null)
   3. Трансформируйте результат или восстановитесь после ошибки

3. Используйте whenComplete() для побочных эффектов:
   1. Вызывается после завершения (успех или неудача)
   2. Не может изменить результат или исключение
   3. Полезно для логирования, очистки, метрик

4. Продемонстрируйте распространение исключений в цепочках:
   1. Покажите как исключения проходят через конвейер
   2. Обрабатывайте исключения на разных этапах
   3. Грациозно восстанавливайтесь после сбоев

## Пример вывода:
\`\`\`
=== Exception Handling ===

--- exceptionally: Recover from Errors ---
Error case: java.lang.RuntimeException: Database unavailable
Recovered with: Default User

--- handle: Handle Both Success and Failure ---
Success case: Result: DATA (processed successfully)
Error case: Error occurred: Simulated failure, using fallback

--- whenComplete: Side Effects ---
Processing order...
Completed with result: Order #12345
Logging: Operation completed

Handling error case...
Completed with exception: java.lang.RuntimeException: Order failed
Logging: Operation failed

--- Exception Propagation ---
Step 1: Starting...
Step 2: Error occurred!
Final result (recovered): Operation failed, using backup data
\`\`\``,
            hint1: `exceptionally() работает как блок catch - он выполняется только при ошибке и предоставляет резервное значение. handle() работает как finally - всегда выполняется и получает как результат, так и исключение.`,
            hint2: `whenComplete() предназначен только для побочных эффектов (логирование, метрики) - он не может изменить результат. Исключения распространяются по цепочке до обработки exceptionally() или handle().`,
            whyItMatters: `Надежная обработка исключений критична в продакшн асинхронном коде. Без правильной обработки ошибок сбои могут остаться незамеченными, вызвать каскадные ошибки или привести к падению приложения. Понимание exceptionally() для восстановления, handle() для трансформации и whenComplete() для очистки обеспечивает устойчивость и поддерживаемость вашего асинхронного кода. Эти паттерны необходимы для построения отказоустойчивых распределенных систем.

**Продакшен паттерн:**
\`\`\`java
// Отказоустойчивый вызов внешнего API с резервными данными
CompletableFuture.supplyAsync(() -> {
    return externalAPI.fetchUserData(userId);
})
.exceptionally(ex -> {
    // Логирование ошибки и возврат кешированных данных
    logger.error("API failed, using cache", ex);
    return cache.get(userId);
})
.handle((result, ex) -> {
    if (ex != null) {
        // Даже кеш не сработал - используем значения по умолчанию
        metrics.incrementFailureCount();
        return UserData.defaultUser();
    }
    metrics.incrementSuccessCount();
    return result;
})
.whenComplete((result, ex) -> {
    // Всегда выполняется: обновление метрик и логирование
    logger.info("Request completed for user: {}", userId);
    requestTracker.finish(userId);
});
\`\`\`

**Практические преимущества:**
- exceptionally обеспечивает graceful degradation при сбоях
- handle позволяет реализовать сложную логику восстановления
- whenComplete гарантирует выполнение очистки ресурсов
- Предотвращает молчаливые сбои в асинхронных цепочках`
        },
        uz: {
            title: `Istisnolarni boshqarish (exceptionally, handle, whenComplete)`,
            solutionCode: `import java.util.concurrent.*;

public class ExceptionHandling {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Istisnolarni boshqarish ===");

        // exceptionally: Zaxira qiymati bilan istisnolardan tiklanish
        System.out.println("\\n--- exceptionally: Xatolardan tiklanish ---");
        CompletableFuture<String> withError = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("Database unavailable");
        }).exceptionally(ex -> {
            // Faqat istisno yuz berganda chaqiriladi
            System.out.println("Error case: " + ex);
            return "Default User"; // Zaxira qiymati
        });

        System.out.println("Recovered with: " + withError.join());

        // handle: Muvaffaqiyat va muvaffaqiyatsizlik holatlarini boshqarish
        System.out.println("\\n--- handle: Muvaffaqiyat va xatolarni boshqarish ---");

        // Muvaffaqiyat holati
        CompletableFuture<String> successCase = CompletableFuture
            .supplyAsync(() -> "DATA")
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error: " + ex.getMessage();
                } else {
                    return "Result: " + result + " (processed successfully)";
                }
            });
        System.out.println("Success case: " + successCase.join());

        // Xato holati
        CompletableFuture<String> errorCase = CompletableFuture
            .supplyAsync(() -> {
                throw new RuntimeException("Simulated failure");
            })
            .handle((result, ex) -> {
                if (ex != null) {
                    return "Error occurred: " + ex.getCause().getMessage() + ", using fallback";
                } else {
                    return "Result: " + result;
                }
            });
        System.out.println("Error case: " + errorCase.join());

        // whenComplete: Yon ta'sirlarni bajarish (jurnal yuritish, tozalash)
        System.out.println("\\n--- whenComplete: Yon ta'sirlar ---");

        // Yon ta'sirlar bilan muvaffaqiyat holati
        CompletableFuture<String> completeSuccess = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Processing order...");
                return "Order #12345";
            })
            .whenComplete((result, ex) -> {
                // Ikkala holatda ham chaqiriladi: muvaffaqiyat va xato
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex);
                } else {
                    System.out.println("Completed with result: " + result);
                }
                System.out.println("Logging: Operation completed");
            });
        completeSuccess.join();

        // Yon ta'sirlar bilan xato holati
        CompletableFuture<String> completeError = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("\\nHandling error case...");
                throw new RuntimeException("Order failed");
            })
            .whenComplete((result, ex) -> {
                if (ex != null) {
                    System.out.println("Completed with exception: " + ex.getCause());
                    System.out.println("Logging: Operation failed");
                } else {
                    System.out.println("Completed with result: " + result);
                }
            })
            .exceptionally(ex -> "Error handled"); // Istisnodan oldini olish uchun tiklanish

        completeError.join();

        // Zanjirlarda istisnolarning tarqalishi
        System.out.println("\\n--- Istisnolarning tarqalishi ---");
        CompletableFuture<String> chain = CompletableFuture
            .supplyAsync(() -> {
                System.out.println("Step 1: Starting...");
                return "initial";
            })
            .thenApply(s -> {
                System.out.println("Step 2: Error occurred!");
                throw new RuntimeException("Processing failed");
            })
            .thenApply(s -> {
                // Bu istisno tufayli bajarilmaydi
                System.out.println("Step 3: This won't print");
                return s.toUpperCase();
            })
            .exceptionally(ex -> {
                // Istisno bu yerga tarqaladi
                return "Operation failed, using backup data";
            });

        System.out.println("Final result (recovered): " + chain.join());
    }
}`,
            description: `# CompletableFuture da istisnolarni boshqarish

To'g'ri istisnolarni boshqarish asinxron dasturlashda juda muhimdir. CompletableFuture istisnolarni boshqarish uchun uchta asosiy usulni taqdim etadi: tiklanish uchun exceptionally(), muvaffaqiyat va muvaffaqiyatsizlikni boshqarish uchun handle() va tozalash yoki jurnal yuritish uchun whenComplete().

## Talablar:
1. Istisnolardan tiklanish uchun exceptionally() dan foydalaning:
   1. Xato yuz berganda zaxira qiymatini taqdim eting
   2. Faqat istisno yuz berganda chaqiriladi
   3. Tiklangan qiymat bilan CompletableFuture qaytaradi

2. Muvaffaqiyat va muvaffaqiyatsizlikni boshqarish uchun handle() dan foydalaning:
   1. Ikkala holatda ham chaqiriladi: muvaffaqiyat va xato
   2. Natija va istisno oladi (biri null)
   3. Natijani o'zgartiring yoki xatodan tiklaning

3. Yon ta'sirlar uchun whenComplete() dan foydalaning:
   1. Yakunlangandan keyin chaqiriladi (muvaffaqiyat yoki xato)
   2. Natija yoki istisnoni o'zgartira olmaydi
   3. Jurnal yuritish, tozalash, ko'rsatkichlar uchun foydali

4. Zanjirlarda istisnolarning tarqalishini namoyish eting:
   1. Istisnolar konveyer orqali qanday o'tishini ko'rsating
   2. Turli bosqichlarda istisnolarni boshqaring
   3. Muvaffaqiyatsizliklardan yumshoq tiklaning

## Chiqish namunasi:
\`\`\`
=== Exception Handling ===

--- exceptionally: Recover from Errors ---
Error case: java.lang.RuntimeException: Database unavailable
Recovered with: Default User

--- handle: Handle Both Success and Failure ---
Success case: Result: DATA (processed successfully)
Error case: Error occurred: Simulated failure, using fallback

--- whenComplete: Side Effects ---
Processing order...
Completed with result: Order #12345
Logging: Operation completed

Handling error case...
Completed with exception: java.lang.RuntimeException: Order failed
Logging: Operation failed

--- Exception Propagation ---
Step 1: Starting...
Step 2: Error occurred!
Final result (recovered): Operation failed, using backup data
\`\`\``,
            hint1: `exceptionally() catch bloki kabi ishlaydi - u faqat xato yuz berganda ishlaydi va zaxira qiymatini taqdim etadi. handle() finally kabi ishlaydi - har doim ishlaydi va natija va istisno oladi.`,
            hint2: `whenComplete() faqat yon ta'sirlar uchun (jurnal yuritish, ko'rsatkichlar) - u natijani o'zgartira olmaydi. Istisnolar exceptionally() yoki handle() tomonidan boshqarilguncha zanjir orqali tarqaladi.`,
            whyItMatters: `Ishlab chiqarish asinxron kodida mustahkam istisnolarni boshqarish juda muhimdir. To'g'ri xatolarni boshqarishsiz, nosozliklar sezilmay qolishi, kaskadli xatolarga olib kelishi yoki ilovani ishdan chiqarishi mumkin. Tiklanish uchun exceptionally(), transformatsiya uchun handle() va tozalash uchun whenComplete() ni tushunish asinxron kodingiz bardoshli va saqlash mumkin bo'lishini ta'minlaydi. Bu naqshlar nosozliklarga chidamli taqsimlangan tizimlarni qurish uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Zaxira ma'lumotlar bilan tashqi API ga nosozliklarga chidamli chaqiruv
CompletableFuture.supplyAsync(() -> {
    return externalAPI.fetchUserData(userId);
})
.exceptionally(ex -> {
    // Xatoni jurnal yozish va keshlangan ma'lumotlarni qaytarish
    logger.error("API failed, using cache", ex);
    return cache.get(userId);
})
.handle((result, ex) -> {
    if (ex != null) {
        // Hatto kesh ham ishlamadi - standart qiymatlardan foydalanish
        metrics.incrementFailureCount();
        return UserData.defaultUser();
    }
    metrics.incrementSuccessCount();
    return result;
})
.whenComplete((result, ex) -> {
    // Har doim bajariladi: ko'rsatkichlarni yangilash va jurnal yozish
    logger.info("Request completed for user: {}", userId);
    requestTracker.finish(userId);
});
\`\`\`

**Amaliy foydalari:**
- exceptionally nosozliklar vaqtida yumshoq pasayishni ta'minlaydi
- handle murakkab tiklanish mantiqini amalga oshirish imkon beradi
- whenComplete resurslarni tozalashning bajarilishini kafolatlaydi
- Asinxron zanjirlarda jim nosozliklardan qochadi`
        }
    }
};

export default task;
