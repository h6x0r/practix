import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-distribution-summary',
    title: 'Distribution Summary',
    difficulty: 'medium',
    tags: ['java', 'metrics', 'micrometer', 'distribution'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master DistributionSummary for tracking size metrics.

**Requirements:**
1. Create a MeterRegistry
2. Create a DistributionSummary named "request.size"
3. Record various request sizes: 150, 200, 175, 300, 225
4. Print summary statistics: count, total, max, mean
5. Create a summary for "response.size"
6. Simulate 10 responses with random sizes (100-500 bytes)
7. Print response size statistics
8. Create summaries with percentile configuration
9. Configure percentiles at 0.5 (median), 0.95, and 0.99
10. Print percentile values for both request and response sizes

DistributionSummary tracks the distribution of values, like request sizes, response payloads, or batch sizes. Unlike Timer which measures time, it measures arbitrary numeric values.`,
    initialCode: `import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class DistributionSummaryMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry

        // Create a DistributionSummary

        // Record various sizes

        // Print statistics

        // Configure percentiles
    }
}`,
    solutionCode: `import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class DistributionSummaryMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Create a DistributionSummary named "request.size"
        DistributionSummary requestSize = registry.summary("request.size");

        // Record various request sizes: 150, 200, 175, 300, 225
        System.out.println("Recording request sizes...");
        requestSize.record(150);
        requestSize.record(200);
        requestSize.record(175);
        requestSize.record(300);
        requestSize.record(225);

        // Print summary statistics: count, total, max, mean
        System.out.println("\\n=== Request Size Statistics ===");
        System.out.println("Count: " + requestSize.count());
        System.out.println("Total: " + String.format("%.0f", requestSize.totalAmount()) + " bytes");
        System.out.println("Max: " + String.format("%.0f", requestSize.max()) + " bytes");
        System.out.println("Mean: " + String.format("%.2f", requestSize.mean()) + " bytes");

        // Create a summary for "response.size"
        DistributionSummary responseSize = registry.summary("response.size");

        // Simulate 10 responses with random sizes (100-500 bytes)
        System.out.println("\\n=== Recording response sizes ===");
        for (int i = 0; i < 10; i++) {
            double size = 100 + Math.random() * 400;
            responseSize.record(size);
            System.out.println("Response " + (i + 1) + ": " + String.format("%.0f", size) + " bytes");
        }

        // Print response size statistics
        System.out.println("\\n=== Response Size Statistics ===");
        System.out.println("Count: " + responseSize.count());
        System.out.println("Total: " + String.format("%.0f", responseSize.totalAmount()) + " bytes");
        System.out.println("Max: " + String.format("%.0f", responseSize.max()) + " bytes");
        System.out.println("Mean: " + String.format("%.2f", responseSize.mean()) + " bytes");

        // Create summaries with percentile configuration
        // Configure percentiles at 0.5 (median), 0.95, and 0.99
        DistributionSummary payloadSize = DistributionSummary.builder("payload.size")
            .description("Distribution of payload sizes")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);

        // Record sample data
        System.out.println("\\n=== Recording payload sizes with percentiles ===");
        for (int i = 0; i < 100; i++) {
            double size = 50 + Math.random() * 450; // 50-500 bytes
            payloadSize.record(size);
        }

        // Print percentile values
        System.out.println("\\n=== Payload Size Percentiles ===");
        System.out.println("Count: " + payloadSize.count());
        System.out.println("Mean: " + String.format("%.2f", payloadSize.mean()) + " bytes");
        System.out.println("Max: " + String.format("%.0f", payloadSize.max()) + " bytes");

        // Note: SimpleMeterRegistry doesn't support percentile querying
        // In production with Prometheus/Grafana, you would query:
        // - 50th percentile (median): Half of values are below this
        // - 95th percentile: 95% of values are below this
        // - 99th percentile: 99% of values are below this
        System.out.println("\\nPercentiles are published to monitoring systems.");
        System.out.println("In Prometheus: payload_size{quantile=\\\"0.5\\\"}");
        System.out.println("In Prometheus: payload_size{quantile=\\\"0.95\\\"}");
        System.out.println("In Prometheus: payload_size{quantile=\\\"0.99\\\"}");

        // Summary comparison
        System.out.println("\\n=== Summary Comparison ===");
        System.out.println("Request sizes - Mean: " + String.format("%.2f", requestSize.mean()) + " bytes");
        System.out.println("Response sizes - Mean: " + String.format("%.2f", responseSize.mean()) + " bytes");
        System.out.println("Payload sizes - Mean: " + String.format("%.2f", payloadSize.mean()) + " bytes");
    }
}`,
    hint1: `Use DistributionSummary.builder() to configure percentiles. The record(double) method adds a value to the distribution.`,
    hint2: `DistributionSummary provides count(), totalAmount(), max(), and mean(). Percentiles require backend support (like Prometheus) to query.`,
    whyItMatters: `DistributionSummary is essential for understanding the distribution of values in your system. It helps identify outliers, understand typical sizes, and monitor data volume trends.

**Production Pattern:**
\`\`\`java
@RestController
public class FileUploadController {
    private final DistributionSummary uploadSize;

    public FileUploadController(MeterRegistry registry) {
        this.uploadSize = DistributionSummary.builder("upload.file.size")
            .description("Uploaded file sizes")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .minimumExpectedValue(1024.0)
            .maximumExpectedValue(10485760.0) // 10MB
            .register(registry);
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam MultipartFile file) {
        uploadSize.record(file.getSize());
        fileService.save(file);
        return ResponseEntity.ok().build();
    }
}
\`\`\`

**Practical Benefits:**
- Analyze file size distribution
- Set optimal upload limits
- Plan storage capacity`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

// Test 1: MeterRegistry is created
class Test1 {
    @Test
    void testMeterRegistryCreated() {
        MeterRegistry registry = new SimpleMeterRegistry();
        assertNotNull(registry);
    }
}

// Test 2: DistributionSummary is created with name
class Test2 {
    @Test
    void testDistributionSummaryCreated() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = registry.summary("request.size");
        assertNotNull(summary);
        assertEquals("request.size", summary.getId().getName());
    }
}

// Test 3: Record adds values to summary
class Test3 {
    @Test
    void testRecordAddsValues() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = registry.summary("request.size");
        summary.record(150);
        summary.record(200);
        assertEquals(2, summary.count());
    }
}

// Test 4: TotalAmount sums all values
class Test4 {
    @Test
    void testTotalAmountSumsValues() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = registry.summary("request.size");
        summary.record(150);
        summary.record(200);
        summary.record(175);
        assertEquals(525.0, summary.totalAmount(), 0.01);
    }
}

// Test 5: Max returns maximum value
class Test5 {
    @Test
    void testMaxReturnsMaximum() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = registry.summary("request.size");
        summary.record(150);
        summary.record(300);
        summary.record(200);
        assertEquals(300.0, summary.max(), 0.01);
    }
}

// Test 6: Mean calculates average
class Test6 {
    @Test
    void testMeanCalculatesAverage() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = registry.summary("request.size");
        summary.record(100);
        summary.record(200);
        summary.record(300);
        assertEquals(200.0, summary.mean(), 0.01);
    }
}

// Test 7: Builder creates summary with description
class Test7 {
    @Test
    void testBuilderWithDescription() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = DistributionSummary.builder("payload.size")
            .description("Payload sizes")
            .baseUnit("bytes")
            .register(registry);
        assertNotNull(summary);
    }
}

// Test 8: Builder with percentiles creates summary
class Test8 {
    @Test
    void testBuilderWithPercentiles() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary summary = DistributionSummary.builder("payload.size")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);
        assertNotNull(summary);
        summary.record(100);
        assertEquals(1, summary.count());
    }
}

// Test 9: Multiple summaries are independent
class Test9 {
    @Test
    void testMultipleSummariesIndependent() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary s1 = registry.summary("request.size");
        DistributionSummary s2 = registry.summary("response.size");
        s1.record(100);
        s1.record(200);
        s2.record(50);
        assertEquals(2, s1.count());
        assertEquals(1, s2.count());
    }
}

// Test 10: Same name returns same summary
class Test10 {
    @Test
    void testSameNameReturnsSameSummary() {
        MeterRegistry registry = new SimpleMeterRegistry();
        DistributionSummary s1 = registry.summary("my.summary");
        DistributionSummary s2 = registry.summary("my.summary");
        s1.record(100);
        assertEquals(1, s2.count());
    }
}`,
    translations: {
        ru: {
            title: 'Сводка распределения',
            solutionCode: `import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class DistributionSummaryMetrics {
    public static void main(String[] args) {
        // Создаем MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Создаем DistributionSummary с именем "request.size"
        DistributionSummary requestSize = registry.summary("request.size");

        // Записываем различные размеры запросов: 150, 200, 175, 300, 225
        System.out.println("Запись размеров запросов...");
        requestSize.record(150);
        requestSize.record(200);
        requestSize.record(175);
        requestSize.record(300);
        requestSize.record(225);

        // Выводим сводную статистику: count, total, max, mean
        System.out.println("\\n=== Статистика размеров запросов ===");
        System.out.println("Количество: " + requestSize.count());
        System.out.println("Всего: " + String.format("%.0f", requestSize.totalAmount()) + " байт");
        System.out.println("Максимум: " + String.format("%.0f", requestSize.max()) + " байт");
        System.out.println("Среднее: " + String.format("%.2f", requestSize.mean()) + " байт");

        // Создаем сводку для "response.size"
        DistributionSummary responseSize = registry.summary("response.size");

        // Симулируем 10 ответов со случайными размерами (100-500 байт)
        System.out.println("\\n=== Запись размеров ответов ===");
        for (int i = 0; i < 10; i++) {
            double size = 100 + Math.random() * 400;
            responseSize.record(size);
            System.out.println("Ответ " + (i + 1) + ": " + String.format("%.0f", size) + " байт");
        }

        // Выводим статистику размеров ответов
        System.out.println("\\n=== Статистика размеров ответов ===");
        System.out.println("Количество: " + responseSize.count());
        System.out.println("Всего: " + String.format("%.0f", responseSize.totalAmount()) + " байт");
        System.out.println("Максимум: " + String.format("%.0f", responseSize.max()) + " байт");
        System.out.println("Среднее: " + String.format("%.2f", responseSize.mean()) + " байт");

        // Создаем сводки с конфигурацией процентилей
        // Конфигурируем процентили на 0.5 (медиана), 0.95 и 0.99
        DistributionSummary payloadSize = DistributionSummary.builder("payload.size")
            .description("Распределение размеров полезной нагрузки")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);

        // Записываем примерные данные
        System.out.println("\\n=== Запись размеров полезной нагрузки с процентилями ===");
        for (int i = 0; i < 100; i++) {
            double size = 50 + Math.random() * 450; // 50-500 байт
            payloadSize.record(size);
        }

        // Выводим значения процентилей
        System.out.println("\\n=== Процентили размеров полезной нагрузки ===");
        System.out.println("Количество: " + payloadSize.count());
        System.out.println("Среднее: " + String.format("%.2f", payloadSize.mean()) + " байт");
        System.out.println("Максимум: " + String.format("%.0f", payloadSize.max()) + " байт");

        // Примечание: SimpleMeterRegistry не поддерживает запрос процентилей
        // В продакшене с Prometheus/Grafana вы бы запрашивали:
        // - 50-й процентиль (медиана): Половина значений ниже этого
        // - 95-й процентиль: 95% значений ниже этого
        // - 99-й процентиль: 99% значений ниже этого
        System.out.println("\\nПроцентили публикуются в системы мониторинга.");
        System.out.println("В Prometheus: payload_size{quantile=\\\"0.5\\\"}");
        System.out.println("В Prometheus: payload_size{quantile=\\\"0.95\\\"}");
        System.out.println("В Prometheus: payload_size{quantile=\\\"0.99\\\"}");

        // Сравнение сводок
        System.out.println("\\n=== Сравнение сводок ===");
        System.out.println("Размеры запросов - Среднее: " + String.format("%.2f", requestSize.mean()) + " байт");
        System.out.println("Размеры ответов - Среднее: " + String.format("%.2f", responseSize.mean()) + " байт");
        System.out.println("Размеры полезной нагрузки - Среднее: " + String.format("%.2f", payloadSize.mean()) + " байт");
    }
}`,
            description: `Освойте DistributionSummary для отслеживания метрик размера.

**Требования:**
1. Создайте MeterRegistry
2. Создайте DistributionSummary с именем "request.size"
3. Запишите различные размеры запросов: 150, 200, 175, 300, 225
4. Выведите сводную статистику: count, total, max, mean
5. Создайте сводку для "response.size"
6. Симулируйте 10 ответов со случайными размерами (100-500 байт)
7. Выведите статистику размеров ответов
8. Создайте сводки с конфигурацией процентилей
9. Настройте процентили на 0.5 (медиана), 0.95 и 0.99
10. Выведите значения процентилей для размеров запросов и ответов

DistributionSummary отслеживает распределение значений, таких как размеры запросов, полезная нагрузка ответов или размеры пакетов. В отличие от Timer, который измеряет время, он измеряет произвольные числовые значения.`,
            hint1: `Используйте DistributionSummary.builder() для конфигурации процентилей. Метод record(double) добавляет значение в распределение.`,
            hint2: `DistributionSummary предоставляет count(), totalAmount(), max() и mean(). Процентили требуют поддержки бэкенда (например Prometheus) для запроса.`,
            whyItMatters: `DistributionSummary необходим для понимания распределения значений в вашей системе. Он помогает выявлять выбросы, понимать типичные размеры и отслеживать тенденции объема данных.

**Продакшен паттерн:**
\`\`\`java
@RestController
public class FileUploadController {
    private final DistributionSummary uploadSize;

    public FileUploadController(MeterRegistry registry) {
        this.uploadSize = DistributionSummary.builder("upload.file.size")
            .description("Uploaded file sizes")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .minimumExpectedValue(1024.0)
            .maximumExpectedValue(10485760.0) // 10MB
            .register(registry);
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam MultipartFile file) {
        uploadSize.record(file.getSize());
        fileService.save(file);
        return ResponseEntity.ok().build();
    }
}
\`\`\`

**Практические преимущества:**
- Анализ распределения размеров файлов
- Установка оптимальных лимитов загрузки
- Планирование емкости хранилища`
        },
        uz: {
            title: 'Taqsimlash xulosasi',
            solutionCode: `import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class DistributionSummaryMetrics {
    public static void main(String[] args) {
        // MeterRegistry yaratamiz
        MeterRegistry registry = new SimpleMeterRegistry();

        // "request.size" nomi bilan DistributionSummary yaratamiz
        DistributionSummary requestSize = registry.summary("request.size");

        // Turli so'rov o'lchamlarini yozamiz: 150, 200, 175, 300, 225
        System.out.println("So'rov o'lchamlari yozilmoqda...");
        requestSize.record(150);
        requestSize.record(200);
        requestSize.record(175);
        requestSize.record(300);
        requestSize.record(225);

        // Umumiy statistikani chiqaramiz: count, total, max, mean
        System.out.println("\\n=== So'rov o'lchami statistikasi ===");
        System.out.println("Soni: " + requestSize.count());
        System.out.println("Jami: " + String.format("%.0f", requestSize.totalAmount()) + " bayt");
        System.out.println("Maksimum: " + String.format("%.0f", requestSize.max()) + " bayt");
        System.out.println("O'rtacha: " + String.format("%.2f", requestSize.mean()) + " bayt");

        // "response.size" uchun xulosa yaratamiz
        DistributionSummary responseSize = registry.summary("response.size");

        // Tasodifiy o'lchamda 10 ta javobni simulyatsiya qilamiz (100-500 bayt)
        System.out.println("\\n=== Javob o'lchamlari yozilmoqda ===");
        for (int i = 0; i < 10; i++) {
            double size = 100 + Math.random() * 400;
            responseSize.record(size);
            System.out.println("Javob " + (i + 1) + ": " + String.format("%.0f", size) + " bayt");
        }

        // Javob o'lchami statistikasini chiqaramiz
        System.out.println("\\n=== Javob o'lchami statistikasi ===");
        System.out.println("Soni: " + responseSize.count());
        System.out.println("Jami: " + String.format("%.0f", responseSize.totalAmount()) + " bayt");
        System.out.println("Maksimum: " + String.format("%.0f", responseSize.max()) + " bayt");
        System.out.println("O'rtacha: " + String.format("%.2f", responseSize.mean()) + " bayt");

        // Percentil konfiguratsiyasi bilan xulosalar yaratamiz
        // 0.5 (median), 0.95 va 0.99 da percentillarni sozlaymiz
        DistributionSummary payloadSize = DistributionSummary.builder("payload.size")
            .description("Foydali yuk o'lchamlari taqsimoti")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);

        // Namuna ma'lumotlarini yozamiz
        System.out.println("\\n=== Percentillar bilan foydali yuk o'lchamlari yozilmoqda ===");
        for (int i = 0; i < 100; i++) {
            double size = 50 + Math.random() * 450; // 50-500 bayt
            payloadSize.record(size);
        }

        // Percentil qiymatlarini chiqaramiz
        System.out.println("\\n=== Foydali yuk o'lchami percentillari ===");
        System.out.println("Soni: " + payloadSize.count());
        System.out.println("O'rtacha: " + String.format("%.2f", payloadSize.mean()) + " bayt");
        System.out.println("Maksimum: " + String.format("%.0f", payloadSize.max()) + " bayt");

        // Eslatma: SimpleMeterRegistry percentil so'rovini qo'llab-quvvatlamaydi
        // Prometheus/Grafana bilan ishlab chiqarishda siz so'rar edingiz:
        // - 50-percentil (median): Qiymatlarning yarmi bundan past
        // - 95-percentil: Qiymatlarning 95% bundan past
        // - 99-percentil: Qiymatlarning 99% bundan past
        System.out.println("\\nPercentillar monitoring tizimlariga nashr qilinadi.");
        System.out.println("Prometheus da: payload_size{quantile=\\\"0.5\\\"}");
        System.out.println("Prometheus da: payload_size{quantile=\\\"0.95\\\"}");
        System.out.println("Prometheus da: payload_size{quantile=\\\"0.99\\\"}");

        // Xulosalar solishtiruvi
        System.out.println("\\n=== Xulosalar solishtiruvi ===");
        System.out.println("So'rov o'lchamlari - O'rtacha: " + String.format("%.2f", requestSize.mean()) + " bayt");
        System.out.println("Javob o'lchamlari - O'rtacha: " + String.format("%.2f", responseSize.mean()) + " bayt");
        System.out.println("Foydali yuk o'lchamlari - O'rtacha: " + String.format("%.2f", payloadSize.mean()) + " bayt");
    }
}`,
            description: `O'lchov metrikalarini kuzatish uchun DistributionSummary ni o'zlashtirig.

**Talablar:**
1. MeterRegistry yarating
2. "request.size" nomi bilan DistributionSummary yarating
3. Turli so'rov o'lchamlarini yozing: 150, 200, 175, 300, 225
4. Umumiy statistikani chiqaring: count, total, max, mean
5. "response.size" uchun xulosa yarating
6. Tasodifiy o'lchamda 10 ta javobni simulyatsiya qiling (100-500 bayt)
7. Javob o'lchami statistikasini chiqaring
8. Percentil konfiguratsiyasi bilan xulosalar yarating
9. 0.5 (median), 0.95 va 0.99 da percentillarni sozlang
10. So'rov va javob o'lchamlari uchun percentil qiymatlarini chiqaring

DistributionSummary so'rov o'lchamlari, javob foydali yuklari yoki to'plam o'lchamlari kabi qiymatlar taqsimotini kuzatadi. Vaqtni o'lchaydigan Timer dan farqli o'laroq, u ixtiyoriy raqamli qiymatlarni o'lchaydi.`,
            hint1: `Percentillarni sozlash uchun DistributionSummary.builder() dan foydalaning. record(double) metodi taqsimotga qiymat qo'shadi.`,
            hint2: `DistributionSummary count(), totalAmount(), max() va mean() ni taqdim etadi. Percentillar so'rov uchun backend qo'llab-quvvatlashni (masalan Prometheus) talab qiladi.`,
            whyItMatters: `DistributionSummary tizimingizdagi qiymatlar taqsimotini tushunish uchun zarur. U chetga chiqishlarni aniqlashga, odatiy o'lchamlarni tushunishga va ma'lumotlar hajmi tendentsiyalarini kuzatishga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@RestController
public class FileUploadController {
    private final DistributionSummary uploadSize;

    public FileUploadController(MeterRegistry registry) {
        this.uploadSize = DistributionSummary.builder("upload.file.size")
            .description("Yuklangan fayl hajmlari")
            .baseUnit("bytes")
            .publishPercentiles(0.5, 0.95, 0.99)
            .minimumExpectedValue(1024.0)
            .maximumExpectedValue(10485760.0) // 10MB
            .register(registry);
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam MultipartFile file) {
        uploadSize.record(file.getSize());
        fileService.save(file);
        return ResponseEntity.ok().build();
    }
}
\`\`\`

**Amaliy foydalari:**
- Fayl hajmlari taqsimotini tahlil qilish
- Optimal yuklash limitlarini belgilash
- Saqlash sig'imini rejalashtirish`
        }
    }
};

export default task;
