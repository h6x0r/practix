import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-completable-basics',
    title: 'CompletableFuture Basics',
    difficulty: 'easy',
    tags: ['java', 'completablefuture', 'async', 'java8', 'concurrency'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# CompletableFuture Basics

CompletableFuture is a powerful class introduced in Java 8 for asynchronous programming. It represents a future result of an asynchronous computation. You can create CompletableFutures using supplyAsync (returns a value) or runAsync (runs without returning a value).

## Requirements:
1. Create CompletableFuture with completedFuture():
   1. Return an already completed future with a value
   2. Useful for testing or when value is immediately available

2. Use supplyAsync() for async computations:
   1. Execute computation in ForkJoinPool
   2. Return result asynchronously
   3. Simulate database query or API call

3. Use runAsync() for async operations without return:
   1. Execute side effects asynchronously
   2. No return value needed
   3. Useful for logging, notifications, etc.

4. Demonstrate get() and join() methods:
   1. get(): Blocks and throws checked exception
   2. join(): Blocks and throws unchecked exception
   3. Show blocking behavior

## Example Output:
\`\`\`
=== CompletableFuture Basics ===
Completed future result: Hello, CompletableFuture!

Async computation started...
Async computation result: Data from database
Computation took: ~2000ms

Async operation started...
Async operation completed
Operation took: ~1000ms

=== get() vs join() ===
Result using get(): 42
Result using join(): 42
\`\`\``,
    initialCode: `import java.util.concurrent.*;

public class CompletableBasics {
    public static void main(String[] args) {
        // TODO: Create a completed future with a value

        // TODO: Create async computation with supplyAsync

        // TODO: Create async operation with runAsync

        // TODO: Demonstrate get() vs join()
    }
}`,
    solutionCode: `import java.util.concurrent.*;

public class CompletableBasics {
    public static void main(String[] args) throws Exception {
        System.out.println("=== CompletableFuture Basics ===");

        // Create a completed future with a value
        CompletableFuture<String> completedFuture =
            CompletableFuture.completedFuture("Hello, CompletableFuture!");
        System.out.println("Completed future result: " + completedFuture.join());

        // Use supplyAsync for async computation that returns a value
        System.out.println("\\nAsync computation started...");
        long start1 = System.currentTimeMillis();

        CompletableFuture<String> asyncSupply = CompletableFuture.supplyAsync(() -> {
            try {
                // Simulate database query or expensive computation
                Thread.sleep(2000);
                return "Data from database";
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        // Block and wait for result
        String result = asyncSupply.join();
        long elapsed1 = System.currentTimeMillis() - start1;
        System.out.println("Async computation result: " + result);
        System.out.println("Computation took: ~" + elapsed1 + "ms");

        // Use runAsync for async operation without return value
        System.out.println("\\nAsync operation started...");
        long start2 = System.currentTimeMillis();

        CompletableFuture<Void> asyncRun = CompletableFuture.runAsync(() -> {
            try {
                // Simulate some async work (logging, sending notification, etc.)
                Thread.sleep(1000);
                System.out.println("Async operation completed");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        asyncRun.join(); // Wait for completion
        long elapsed2 = System.currentTimeMillis() - start2;
        System.out.println("Operation took: ~" + elapsed2 + "ms");

        System.out.println("\\n=== get() vs join() ===");

        // get() - throws checked exception (ExecutionException, InterruptedException)
        CompletableFuture<Integer> future1 = CompletableFuture.supplyAsync(() -> 42);
        try {
            Integer value = future1.get(); // Throws checked exceptions
            System.out.println("Result using get(): " + value);
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Exception from get(): " + e.getMessage());
        }

        // join() - throws unchecked exception (CompletionException)
        CompletableFuture<Integer> future2 = CompletableFuture.supplyAsync(() -> 42);
        Integer value = future2.join(); // Throws unchecked CompletionException
        System.out.println("Result using join(): " + value);
    }
}`,
    hint1: `CompletableFuture.supplyAsync() takes a Supplier<T> and returns CompletableFuture<T>. CompletableFuture.runAsync() takes a Runnable and returns CompletableFuture<Void>.`,
    hint2: `Use join() to get the result without handling checked exceptions. Use get() when you need to handle InterruptedException or ExecutionException explicitly.`,
    whyItMatters: `CompletableFuture enables non-blocking, asynchronous programming in Java. Understanding the basics of creating and completing futures is essential for building responsive applications, handling I/O operations efficiently, and utilizing modern Java concurrency features. It's the foundation for reactive and scalable Java applications.

**Production Pattern:**
\`\`\`java
// Asynchronous HTTP request processing
CompletableFuture<UserData> userFuture = CompletableFuture.supplyAsync(() -> {
    // Database query doesn't block the thread
    return database.getUser(userId);
});

// While first request executes, launch others
CompletableFuture<Orders> ordersFuture = CompletableFuture.supplyAsync(() -> {
    return orderService.getUserOrders(userId);
});

// Combine results when both are ready
return userFuture.thenCombine(ordersFuture, (user, orders) -> {
    return new UserProfile(user, orders);
}).join();
\`\`\`

**Practical Benefits:**
- Doesn't block threads waiting for I/O operations
- Allows handling thousands of requests with limited thread pool
- Simplifies composition of async operations
- Improves application responsiveness and scalability`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Test completedFuture returns correct value
class Test1 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.completedFuture("Test Value");
        assertEquals("Test Value", future.join());
    }
}

// Test2: Test supplyAsync executes asynchronously
class Test2 {
    @Test
    public void test() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 42);
        assertEquals(Integer.valueOf(42), future.join());
    }
}

// Test3: Test runAsync executes without return value
class Test3 {
    @Test
    public void test() throws Exception {
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            System.out.println("Running async task");
        });
        assertNull(future.join());
    }
}

// Test4: Test get() method blocks and returns result
class Test4 {
    @Test
    public void test() throws Exception {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Result");
        assertEquals("Result", future.get());
    }
}

// Test5: Test join() method blocks and returns result
class Test5 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "Result");
        assertEquals("Result", future.join());
    }
}

// Test6: Test supplyAsync with delay
class Test6 {
    @Test
    public void test() {
        long start = System.currentTimeMillis();
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
            return "Delayed";
        });
        assertEquals("Delayed", future.join());
        assertTrue(System.currentTimeMillis() - start >= 100);
    }
}

// Test7: Test multiple completedFutures
class Test7 {
    @Test
    public void test() {
        CompletableFuture<Integer> f1 = CompletableFuture.completedFuture(1);
        CompletableFuture<Integer> f2 = CompletableFuture.completedFuture(2);
        assertEquals(Integer.valueOf(1), f1.join());
        assertEquals(Integer.valueOf(2), f2.join());
    }
}

// Test8: Test supplyAsync with exception handling
class Test8 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            return "Success";
        });
        assertTrue(future.join().equals("Success"));
    }
}

// Test9: Test runAsync completion
class Test9 {
    @Test
    public void test() {
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {});
        future.join();
        assertTrue(future.isDone());
    }
}

// Test10: Test isDone after completion
class Test10 {
    @Test
    public void test() {
        CompletableFuture<String> future = CompletableFuture.completedFuture("Done");
        assertTrue(future.isDone());
        assertFalse(future.isCancelled());
        assertFalse(future.isCompletedExceptionally());
    }
}
`,
    translations: {
        ru: {
            title: 'Основы CompletableFuture',
            solutionCode: `import java.util.concurrent.*;

public class CompletableBasics {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Основы CompletableFuture ===");

        // Создание завершенного future со значением
        CompletableFuture<String> completedFuture =
            CompletableFuture.completedFuture("Hello, CompletableFuture!");
        System.out.println("Completed future result: " + completedFuture.join());

        // Использование supplyAsync для асинхронных вычислений с возвратом значения
        System.out.println("\\nAsync computation started...");
        long start1 = System.currentTimeMillis();

        CompletableFuture<String> asyncSupply = CompletableFuture.supplyAsync(() -> {
            try {
                // Симуляция запроса к БД или дорогостоящих вычислений
                Thread.sleep(2000);
                return "Data from database";
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        // Блокировка и ожидание результата
        String result = asyncSupply.join();
        long elapsed1 = System.currentTimeMillis() - start1;
        System.out.println("Async computation result: " + result);
        System.out.println("Computation took: ~" + elapsed1 + "ms");

        // Использование runAsync для асинхронных операций без возвращаемого значения
        System.out.println("\\nAsync operation started...");
        long start2 = System.currentTimeMillis();

        CompletableFuture<Void> asyncRun = CompletableFuture.runAsync(() -> {
            try {
                // Симуляция асинхронной работы (логирование, отправка уведомлений и т.д.)
                Thread.sleep(1000);
                System.out.println("Async operation completed");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        asyncRun.join(); // Ожидание завершения
        long elapsed2 = System.currentTimeMillis() - start2;
        System.out.println("Operation took: ~" + elapsed2 + "ms");

        System.out.println("\\n=== get() vs join() ===");

        // get() - выбрасывает проверяемое исключение (ExecutionException, InterruptedException)
        CompletableFuture<Integer> future1 = CompletableFuture.supplyAsync(() -> 42);
        try {
            Integer value = future1.get(); // Выбрасывает проверяемые исключения
            System.out.println("Result using get(): " + value);
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Exception from get(): " + e.getMessage());
        }

        // join() - выбрасывает непроверяемое исключение (CompletionException)
        CompletableFuture<Integer> future2 = CompletableFuture.supplyAsync(() -> 42);
        Integer value = future2.join(); // Выбрасывает непроверяемое CompletionException
        System.out.println("Result using join(): " + value);
    }
}`,
            description: `# Основы CompletableFuture

CompletableFuture - это мощный класс, представленный в Java 8 для асинхронного программирования. Он представляет будущий результат асинхронных вычислений. Вы можете создавать CompletableFuture используя supplyAsync (возвращает значение) или runAsync (выполняется без возврата значения).

## Требования:
1. Создайте CompletableFuture с completedFuture():
   1. Верните уже завершенный future со значением
   2. Полезно для тестирования или когда значение сразу доступно

2. Используйте supplyAsync() для асинхронных вычислений:
   1. Выполните вычисление в ForkJoinPool
   2. Верните результат асинхронно
   3. Симулируйте запрос к БД или API

3. Используйте runAsync() для асинхронных операций без возврата:
   1. Выполните побочные эффекты асинхронно
   2. Не требуется возвращаемое значение
   3. Полезно для логирования, уведомлений и т.д.

4. Продемонстрируйте методы get() и join():
   1. get(): Блокирует и выбрасывает проверяемое исключение
   2. join(): Блокирует и выбрасывает непроверяемое исключение
   3. Покажите поведение блокировки

## Пример вывода:
\`\`\`
=== CompletableFuture Basics ===
Completed future result: Hello, CompletableFuture!

Async computation started...
Async computation result: Data from database
Computation took: ~2000ms

Async operation started...
Async operation completed
Operation took: ~1000ms

=== get() vs join() ===
Result using get(): 42
Result using join(): 42
\`\`\``,
            hint1: `CompletableFuture.supplyAsync() принимает Supplier<T> и возвращает CompletableFuture<T>. CompletableFuture.runAsync() принимает Runnable и возвращает CompletableFuture<Void>.`,
            hint2: `Используйте join() для получения результата без обработки проверяемых исключений. Используйте get(), когда нужно явно обрабатывать InterruptedException или ExecutionException.`,
            whyItMatters: `CompletableFuture обеспечивает неблокирующее асинхронное программирование в Java. Понимание основ создания и завершения futures необходимо для построения отзывчивых приложений, эффективной обработки I/O операций и использования современных возможностей параллелизма Java. Это основа для реактивных и масштабируемых Java-приложений.

**Продакшен паттерн:**
\`\`\`java
// Асинхронная обработка HTTP запроса
CompletableFuture<UserData> userFuture = CompletableFuture.supplyAsync(() -> {
    // Запрос к базе данных не блокирует поток
    return database.getUser(userId);
});

// Пока выполняется первый запрос, можно запустить другие
CompletableFuture<Orders> ordersFuture = CompletableFuture.supplyAsync(() -> {
    return orderService.getUserOrders(userId);
});

// Объединить результаты когда оба готовы
return userFuture.thenCombine(ordersFuture, (user, orders) -> {
    return new UserProfile(user, orders);
}).join();
\`\`\`

**Практические преимущества:**
- Не блокирует потоки при ожидании I/O операций
- Позволяет обрабатывать тысячи запросов с ограниченным пулом потоков
- Упрощает композицию асинхронных операций
- Улучшает отзывчивость и масштабируемость приложения`
        },
        uz: {
            title: `CompletableFuture asoslari`,
            solutionCode: `import java.util.concurrent.*;

public class CompletableBasics {
    public static void main(String[] args) throws Exception {
        System.out.println("=== CompletableFuture asoslari ===");

        // Qiymat bilan yakunlangan future yaratish
        CompletableFuture<String> completedFuture =
            CompletableFuture.completedFuture("Hello, CompletableFuture!");
        System.out.println("Completed future result: " + completedFuture.join());

        // Qiymat qaytaradigan asinxron hisoblashlar uchun supplyAsync ishlatish
        System.out.println("\\nAsync computation started...");
        long start1 = System.currentTimeMillis();

        CompletableFuture<String> asyncSupply = CompletableFuture.supplyAsync(() -> {
            try {
                // Ma'lumotlar bazasi so'rovi yoki qimmat hisoblashlarni simulyatsiya qilish
                Thread.sleep(2000);
                return "Data from database";
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        // Blokirovka qilish va natijani kutish
        String result = asyncSupply.join();
        long elapsed1 = System.currentTimeMillis() - start1;
        System.out.println("Async computation result: " + result);
        System.out.println("Computation took: ~" + elapsed1 + "ms");

        // Qaytariladigan qiymatsiz asinxron operatsiyalar uchun runAsync ishlatish
        System.out.println("\\nAsync operation started...");
        long start2 = System.currentTimeMillis();

        CompletableFuture<Void> asyncRun = CompletableFuture.runAsync(() -> {
            try {
                // Asinxron ishni simulyatsiya qilish (jurnal yuritish, xabarnomalar yuborish va boshqalar)
                Thread.sleep(1000);
                System.out.println("Async operation completed");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        asyncRun.join(); // Yakunlanishini kutish
        long elapsed2 = System.currentTimeMillis() - start2;
        System.out.println("Operation took: ~" + elapsed2 + "ms");

        System.out.println("\\n=== get() vs join() ===");

        // get() - tekshiriladigan istisno tashlaydi (ExecutionException, InterruptedException)
        CompletableFuture<Integer> future1 = CompletableFuture.supplyAsync(() -> 42);
        try {
            Integer value = future1.get(); // Tekshiriladigan istisnolarni tashlaydi
            System.out.println("Result using get(): " + value);
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Exception from get(): " + e.getMessage());
        }

        // join() - tekshirilmaydigan istisno tashlaydi (CompletionException)
        CompletableFuture<Integer> future2 = CompletableFuture.supplyAsync(() -> 42);
        Integer value = future2.join(); // Tekshirilmaydigan CompletionException ni tashlaydi
        System.out.println("Result using join(): " + value);
    }
}`,
            description: `# CompletableFuture asoslari

CompletableFuture - bu Java 8 da asinxron dasturlash uchun kiritilgan kuchli klass. U asinxron hisoblashning kelajak natijasini ifodalaydi. Siz supplyAsync (qiymat qaytaradi) yoki runAsync (qiymat qaytarmasdan ishlaydi) yordamida CompletableFuture yaratishingiz mumkin.

## Talablar:
1. completedFuture() bilan CompletableFuture yarating:
   1. Qiymat bilan allaqachon yakunlangan future ni qaytaring
   2. Sinov uchun yoki qiymat darhol mavjud bo'lganda foydali

2. Asinxron hisoblashlar uchun supplyAsync() dan foydalaning:
   1. ForkJoinPool da hisoblashni bajaring
   2. Natijani asinxron qaytaring
   3. Ma'lumotlar bazasi so'rovi yoki API chaqiruvini simulyatsiya qiling

3. Qaytarishsiz asinxron operatsiyalar uchun runAsync() dan foydalaning:
   1. Yon ta'sirlarni asinxron bajaring
   2. Qaytariladigan qiymat kerak emas
   3. Jurnal yuritish, xabarnomalar uchun foydali

4. get() va join() usullarini namoyish eting:
   1. get(): Bloklaydi va tekshiriladigan istisno tashlaydi
   2. join(): Bloklaydi va tekshirilmaydigan istisno tashlaydi
   3. Bloklash xatti-harakatini ko'rsating

## Chiqish namunasi:
\`\`\`
=== CompletableFuture Basics ===
Completed future result: Hello, CompletableFuture!

Async computation started...
Async computation result: Data from database
Computation took: ~2000ms

Async operation started...
Async operation completed
Operation took: ~1000ms

=== get() vs join() ===
Result using get(): 42
Result using join(): 42
\`\`\``,
            hint1: `CompletableFuture.supplyAsync() Supplier<T> qabul qiladi va CompletableFuture<T> qaytaradi. CompletableFuture.runAsync() Runnable qabul qiladi va CompletableFuture<Void> qaytaradi.`,
            hint2: `Tekshiriladigan istisnolarni boshqarmasdan natijani olish uchun join() dan foydalaning. InterruptedException yoki ExecutionException ni aniq boshqarish kerak bo'lganda get() dan foydalaning.`,
            whyItMatters: `CompletableFuture Java da bloklanmaydigan asinxron dasturlashni ta'minlaydi. Future yaratish va yakunlash asoslarini tushunish javob beradigan ilovalarni qurish, I/O operatsiyalarini samarali boshqarish va zamonaviy Java parallellik imkoniyatlaridan foydalanish uchun zarurdir. Bu reaktiv va miqyoslanadigan Java ilovalari uchun asosdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// HTTP so'rovini asinxron qayta ishlash
CompletableFuture<UserData> userFuture = CompletableFuture.supplyAsync(() -> {
    // Ma'lumotlar bazasi so'rovi oqimni bloklamaydi
    return database.getUser(userId);
});

// Birinchi so'rov bajarilayotganda boshqalarni ishga tushirish mumkin
CompletableFuture<Orders> ordersFuture = CompletableFuture.supplyAsync(() -> {
    return orderService.getUserOrders(userId);
});

// Ikkalasi ham tayyor bo'lganda natijalarni birlashtirish
return userFuture.thenCombine(ordersFuture, (user, orders) -> {
    return new UserProfile(user, orders);
}).join();
\`\`\`

**Amaliy foydalari:**
- I/O operatsiyalarini kutishda oqimlarni bloklamaydi
- Cheklangan oqim hovuzi bilan minglab so'rovlarni qayta ishlashga imkon beradi
- Asinxron operatsiyalarni kompozitsiya qilishni soddalashtiradi
- Ilova javobgarligini va miqyoslanishini yaxshilaydi`
        }
    }
};

export default task;
