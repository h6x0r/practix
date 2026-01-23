import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-mdc-context',
    title: 'MDC for Context Tracking',
    difficulty: 'medium',
    tags: ['java', 'logging', 'slf4j', 'mdc', 'context'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master MDC (Mapped Diagnostic Context) for tracking request context across logs.

**Requirements:**
1. Import MDC from SLF4J
2. Add request ID to MDC at the start of a request
3. Add user ID to MDC when user is authenticated
4. Log messages that will include MDC values
5. Clear MDC at the end of the request
6. Simulate multiple threads with different contexts
7. Demonstrate that MDC is thread-local

MDC is essential for tracking requests in multi-threaded applications, especially in web applications where you need to correlate all logs for a single request.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import java.util.UUID;

public class MdcContext {
    private static final Logger logger = LoggerFactory.getLogger(MdcContext.class);

    public static void main(String[] args) {
        // Simulate multiple concurrent requests
        simulateRequest("user123");
        simulateRequest("user456");
    }

    private static void simulateRequest(String userId) {
        // Add request ID to MDC

        // Add user ID to MDC

        // Log with context

        // Process the request

        // Clear MDC
    }

    private static void processOrder(String orderId) {
        // Logs will automatically include MDC context
    }
}`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import java.util.UUID;

public class MdcContext {
    private static final Logger logger = LoggerFactory.getLogger(MdcContext.class);

    public static void main(String[] args) throws InterruptedException {
        System.out.println("MDC Demo - Thread-local context tracking\\n");

        // Simulate multiple concurrent requests
        Thread thread1 = new Thread(() -> simulateRequest("user123"));
        Thread thread2 = new Thread(() -> simulateRequest("user456"));

        thread1.start();
        thread2.start();

        thread1.join();
        thread2.join();

        System.out.println("\\nNote: Each thread has its own MDC context");
    }

    private static void simulateRequest(String userId) {
        try {
            // Add request ID to MDC - this will be included in all logs in this thread
            String requestId = UUID.randomUUID().toString().substring(0, 8);
            MDC.put("requestId", requestId);
            logger.info("Request started");

            // Add user ID to MDC after authentication
            MDC.put("userId", userId);
            logger.info("User authenticated");

            // Process the request - all logs will include MDC context
            processOrder("ORDER-" + requestId);

            // Simulate some processing time
            Thread.sleep(100);

            logger.info("Request completed successfully");

        } catch (Exception e) {
            logger.error("Request failed", e);
        } finally {
            // IMPORTANT: Always clear MDC to prevent memory leaks
            MDC.clear();
            logger.info("MDC cleared");
        }
    }

    private static void processOrder(String orderId) {
        // These logs will automatically include requestId and userId from MDC
        logger.info("Processing order: {}", orderId);

        validateOrder(orderId);

        logger.info("Order validated");

        // Add order-specific context
        MDC.put("orderId", orderId);
        logger.info("Order context added to MDC");

        saveOrder(orderId);

        // Remove order context when done
        MDC.remove("orderId");
    }

    private static void validateOrder(String orderId) {
        // This log will include requestId and userId from MDC
        logger.debug("Validating order: {}", orderId);
    }

    private static void saveOrder(String orderId) {
        // This log will include requestId, userId, and orderId from MDC
        logger.debug("Saving order: {}", orderId);
    }
}`,
    hint1: `Use MDC.put(key, value) to add context, MDC.get(key) to retrieve, and MDC.clear() to remove all context. MDC is thread-local.`,
    hint2: `Always clear MDC in a finally block to prevent memory leaks. In web applications, use filters or interceptors to manage MDC lifecycle.`,
    whyItMatters: `MDC is crucial for debugging production issues in multi-threaded applications. It allows you to track all logs related to a specific request, user, or transaction, making it much easier to diagnose problems in complex systems.

**Production Pattern:**
\`\`\`java
// Spring Boot Filter for automatic MDC
@Component
public class MdcFilter implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        try {
            MDC.put("requestId", UUID.randomUUID().toString());
            MDC.put("userId", extractUserId(request));
            MDC.put("ip", request.getRemoteAddr());
            chain.doFilter(request, response);
        } finally {
            MDC.clear(); // CRITICAL: always clear
        }
    }
}
\`\`\`

**Practical Benefits:**
- Automatic context added to all request logs
- Easy search for all logs by requestId in logging systems
- Log correlation in microservice architecture`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: class can be instantiated
class Test1 {
    @Test
    public void test() {
        MdcContext obj = new MdcContext();
        assertNotNull("MdcContext instance should be created", obj);
    }
}

// Test2: main runs without exceptions
class Test2 {
    @Test
    public void test() {
        boolean completed = false;
        try {
            MdcContext.main(new String[]{});
            completed = true;
        } catch (Exception e) {
            fail("Main should not throw: " + e.getMessage());
        }
        assertTrue("Main should complete without exceptions", completed);
    }
}

// Test3: static final logger field exists
class Test3 {
    @Test
    public void test() {
        try {
            java.lang.reflect.Field loggerField = MdcContext.class.getDeclaredField("logger");
            assertTrue("Logger should be static",
                java.lang.reflect.Modifier.isStatic(loggerField.getModifiers()));
            assertTrue("Logger should be final",
                java.lang.reflect.Modifier.isFinal(loggerField.getModifiers()));
        } catch (NoSuchFieldException e) {
            fail("Should have a 'logger' field");
        }
    }
}

// Test4: output contains MDC demo header
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Exception caught = null;
        try {
            MdcContext.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        if (caught != null) {
            fail("Should not throw exception: " + caught.getMessage());
        }
        String output = out.toString();
        assertTrue("Should print MDC demo header",
            output.contains("MDC") || output.contains("context") ||
            output.contains("Demo") || output.contains("Демо") ||
            output.contains("kontekst"));
    }
}

// Test5: output mentions thread-local
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Exception caught = null;
        try {
            MdcContext.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        if (caught != null) {
            fail("Should not throw exception: " + caught.getMessage());
        }
        String output = out.toString();
        assertTrue("Should mention thread-local concept",
            output.contains("thread") || output.contains("Thread") ||
            output.contains("поток") || output.contains("Поток") ||
            output.contains("Note") || output.contains("Примечание") ||
            output.contains("Eslatma"));
    }
}

// Test6: produces some output
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        PrintStream oldErr = System.err;
        System.setOut(new PrintStream(out));
        System.setErr(new PrintStream(err));
        Exception caught = null;
        try {
            MdcContext.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        System.setErr(oldErr);
        if (caught != null) {
            fail("Should not throw exception: " + caught.getMessage());
        }
        String allOutput = out.toString() + err.toString();
        assertTrue("Should produce some output", allOutput.length() > 0);
    }
}

// Test7: logger is private
class Test7 {
    @Test
    public void test() {
        try {
            java.lang.reflect.Field loggerField = MdcContext.class.getDeclaredField("logger");
            assertTrue("Logger should be private",
                java.lang.reflect.Modifier.isPrivate(loggerField.getModifiers()));
        } catch (NoSuchFieldException e) {
            fail("Should have a 'logger' field");
        }
    }
}

// Test8: multiple calls work
class Test8 {
    @Test
    public void test() {
        int callCount = 0;
        try {
            MdcContext.main(new String[]{});
            callCount++;
            MdcContext.main(new String[]{});
            callCount++;
        } catch (Exception e) {
            fail("Multiple calls should not throw: " + e.getMessage());
        }
        assertEquals("Both calls should complete", 2, callCount);
    }
}

// Test9: no NullPointerException
class Test9 {
    @Test
    public void test() {
        boolean noNPE = true;
        try {
            MdcContext.main(new String[]{});
        } catch (NullPointerException e) {
            noNPE = false;
            fail("Should not have null pointer exceptions");
        } catch (Exception e) {
            // Other exceptions like InterruptedException are acceptable for MDC threads
            // but NPE check still passed since we didn't catch NPE
        }
        assertTrue("Should not throw NullPointerException", noNPE);
    }
}

// Test10: threads complete successfully
class Test10 {
    @Test
    public void test() {
        try {
            long start = System.currentTimeMillis();
            MdcContext.main(new String[]{});
            long duration = System.currentTimeMillis() - start;
            assertTrue("Should complete within reasonable time", duration < 5000);
        } catch (Exception e) {
            fail("Threads should complete: " + e.getMessage());
        }
    }
}
`,
    translations: {
        ru: {
            title: 'MDC для отслеживания контекста',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import java.util.UUID;

public class MdcContext {
    private static final Logger logger = LoggerFactory.getLogger(MdcContext.class);

    public static void main(String[] args) throws InterruptedException {
        System.out.println("Демо MDC - Отслеживание локального контекста потока\\n");

        // Симулируем несколько одновременных запросов
        Thread thread1 = new Thread(() -> simulateRequest("user123"));
        Thread thread2 = new Thread(() -> simulateRequest("user456"));

        thread1.start();
        thread2.start();

        thread1.join();
        thread2.join();

        System.out.println("\\nПримечание: Каждый поток имеет свой собственный контекст MDC");
    }

    private static void simulateRequest(String userId) {
        try {
            // Добавляем ID запроса в MDC - это будет включено во все логи в этом потоке
            String requestId = UUID.randomUUID().toString().substring(0, 8);
            MDC.put("requestId", requestId);
            logger.info("Запрос начат");

            // Добавляем ID пользователя в MDC после аутентификации
            MDC.put("userId", userId);
            logger.info("Пользователь аутентифицирован");

            // Обрабатываем запрос - все логи будут включать контекст MDC
            processOrder("ORDER-" + requestId);

            // Симулируем время обработки
            Thread.sleep(100);

            logger.info("Запрос успешно завершен");

        } catch (Exception e) {
            logger.error("Запрос не выполнен", e);
        } finally {
            // ВАЖНО: Всегда очищайте MDC чтобы предотвратить утечки памяти
            MDC.clear();
            logger.info("MDC очищен");
        }
    }

    private static void processOrder(String orderId) {
        // Эти логи автоматически включат requestId и userId из MDC
        logger.info("Обработка заказа: {}", orderId);

        validateOrder(orderId);

        logger.info("Заказ проверен");

        // Добавляем контекст специфичный для заказа
        MDC.put("orderId", orderId);
        logger.info("Контекст заказа добавлен в MDC");

        saveOrder(orderId);

        // Удаляем контекст заказа когда закончили
        MDC.remove("orderId");
    }

    private static void validateOrder(String orderId) {
        // Этот лог будет включать requestId и userId из MDC
        logger.debug("Проверка заказа: {}", orderId);
    }

    private static void saveOrder(String orderId) {
        // Этот лог будет включать requestId, userId и orderId из MDC
        logger.debug("Сохранение заказа: {}", orderId);
    }
}`,
            description: `Освойте MDC (Mapped Diagnostic Context) для отслеживания контекста запросов в логах.

**Требования:**
1. Импортируйте MDC из SLF4J
2. Добавьте ID запроса в MDC в начале запроса
3. Добавьте ID пользователя в MDC когда пользователь аутентифицирован
4. Запишите сообщения которые будут включать значения MDC
5. Очистите MDC в конце запроса
6. Симулируйте несколько потоков с разными контекстами
7. Продемонстрируйте что MDC локален для потока

MDC необходим для отслеживания запросов в многопоточных приложениях, особенно в веб-приложениях где нужно коррелировать все логи для одного запроса.`,
            hint1: `Используйте MDC.put(key, value) для добавления контекста, MDC.get(key) для получения, и MDC.clear() для удаления всего контекста. MDC локален для потока.`,
            hint2: `Всегда очищайте MDC в блоке finally для предотвращения утечек памяти. В веб-приложениях используйте фильтры или перехватчики для управления жизненным циклом MDC.`,
            whyItMatters: `MDC критически важен для отладки проблем в продакшене в многопоточных приложениях. Он позволяет отслеживать все логи связанные с конкретным запросом, пользователем или транзакцией, значительно упрощая диагностику проблем в сложных системах.

**Продакшен паттерн:**
\`\`\`java
// Spring Boot Filter для автоматического MDC
@Component
public class MdcFilter implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        try {
            MDC.put("requestId", UUID.randomUUID().toString());
            MDC.put("userId", extractUserId(request));
            MDC.put("ip", request.getRemoteAddr());
            chain.doFilter(request, response);
        } finally {
            MDC.clear(); // КРИТИЧНО: всегда очищать
        }
    }
}
\`\`\`

**Практические преимущества:**
- Автоматическое добавление контекста ко всем логам запроса
- Легкий поиск всех логов по requestId в системах логирования
- Корреляция логов в микросервисной архитектуре`
        },
        uz: {
            title: 'Kontekstni Kuzatish uchun MDC',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import java.util.UUID;

public class MdcContext {
    private static final Logger logger = LoggerFactory.getLogger(MdcContext.class);

    public static void main(String[] args) throws InterruptedException {
        System.out.println("MDC Demo - Thread-local kontekstni kuzatish\\n");

        // Bir nechta parallel so'rovlarni simulyatsiya qilamiz
        Thread thread1 = new Thread(() -> simulateRequest("user123"));
        Thread thread2 = new Thread(() -> simulateRequest("user456"));

        thread1.start();
        thread2.start();

        thread1.join();
        thread2.join();

        System.out.println("\\nEslatma: Har bir thread o'zining MDC kontekstiga ega");
    }

    private static void simulateRequest(String userId) {
        try {
            // So'rov ID sini MDC ga qo'shamiz - bu thread dagi barcha loglarga kiritiladi
            String requestId = UUID.randomUUID().toString().substring(0, 8);
            MDC.put("requestId", requestId);
            logger.info("So'rov boshlandi");

            // Autentifikatsiyadan keyin foydalanuvchi ID sini MDC ga qo'shamiz
            MDC.put("userId", userId);
            logger.info("Foydalanuvchi autentifikatsiya qilindi");

            // So'rovni qayta ishlaymiz - barcha loglar MDC kontekstini o'z ichiga oladi
            processOrder("ORDER-" + requestId);

            // Qayta ishlash vaqtini simulyatsiya qilamiz
            Thread.sleep(100);

            logger.info("So'rov muvaffaqiyatli yakunlandi");

        } catch (Exception e) {
            logger.error("So'rov amalga oshmadi", e);
        } finally {
            // MUHIM: Xotira oqishini oldini olish uchun doimo MDC ni tozalang
            MDC.clear();
            logger.info("MDC tozalandi");
        }
    }

    private static void processOrder(String orderId) {
        // Bu loglar avtomatik ravishda MDC dan requestId va userId ni o'z ichiga oladi
        logger.info("Buyurtma qayta ishlanmoqda: {}", orderId);

        validateOrder(orderId);

        logger.info("Buyurtma tekshirildi");

        // Buyurtmaga xos kontekstni qo'shamiz
        MDC.put("orderId", orderId);
        logger.info("Buyurtma konteksti MDC ga qo'shildi");

        saveOrder(orderId);

        // Tugagach buyurtma kontekstini o'chiramiz
        MDC.remove("orderId");
    }

    private static void validateOrder(String orderId) {
        // Bu log MDC dan requestId va userId ni o'z ichiga oladi
        logger.debug("Buyurtma tekshirilmoqda: {}", orderId);
    }

    private static void saveOrder(String orderId) {
        // Bu log MDC dan requestId, userId va orderId ni o'z ichiga oladi
        logger.debug("Buyurtma saqlanmoqda: {}", orderId);
    }
}`,
            description: `Loglar bo'ylab so'rov kontekstini kuzatish uchun MDC (Mapped Diagnostic Context) ni o'rganing.

**Talablar:**
1. SLF4J dan MDC ni import qiling
2. So'rov boshida MDC ga so'rov ID sini qo'shing
3. Foydalanuvchi autentifikatsiya qilinganda MDC ga foydalanuvchi ID sini qo'shing
4. MDC qiymatlarini o'z ichiga oladigan xabarlarni yozing
5. So'rov oxirida MDC ni tozalang
6. Turli kontekstlarga ega bir nechta threadlarni simulyatsiya qiling
7. MDC thread-local ekanligini ko'rsating

MDC ko'p threadli ilovalarda so'rovlarni kuzatish uchun zarur, ayniqsa bitta so'rov uchun barcha loglarni bog'lash kerak bo'lgan veb-ilovalarda.`,
            hint1: `Kontekst qo'shish uchun MDC.put(key, value), olish uchun MDC.get(key) va barcha kontekstni o'chirish uchun MDC.clear() dan foydalaning. MDC thread-local.`,
            hint2: `Xotira oqishini oldini olish uchun doimo finally blokida MDC ni tozalang. Veb-ilovalarda MDC hayotiy siklini boshqarish uchun filtrlar yoki interceptorlardan foydalaning.`,
            whyItMatters: `MDC ko'p threadli ilovalarda ishlab chiqarish muammolarini tuzatish uchun juda muhim. U ma'lum bir so'rov, foydalanuvchi yoki tranzaksiya bilan bog'liq barcha loglarni kuzatish imkonini beradi va murakkab tizimlardagi muammolarni tashxis qilishni ancha osonlashtiradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Avtomatik MDC uchun Spring Boot Filter
@Component
public class MdcFilter implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        try {
            MDC.put("requestId", UUID.randomUUID().toString());
            MDC.put("userId", extractUserId(request));
            MDC.put("ip", request.getRemoteAddr());
            chain.doFilter(request, response);
        } finally {
            MDC.clear(); // MUHIM: har doim tozalash
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- So'rovning barcha loglariga avtomatik kontekst qo'shish
- Logging tizimlarida requestId bo'yicha osongina qidirish
- Mikroservis arxitekturasida loglarni korrelyatsiya qilish`
        }
    }
};

export default task;
