import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-logback-configuration',
    title: 'Logback Configuration',
    difficulty: 'medium',
    tags: ['java', 'logging', 'logback', 'configuration'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Logback configuration with appenders and patterns.

**Requirements:**
1. Create a basic logback.xml configuration
2. Configure a console appender with pattern
3. Configure a file appender
4. Set different log levels for different packages
5. Use pattern with timestamp, level, thread, logger, and message
6. Configure rolling file appender for production
7. Include MDC values in the pattern

Logback is the most popular logging implementation for SLF4J, providing powerful configuration options for production applications.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

public class LogbackConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(LogbackConfiguration.class);

    public static void main(String[] args) {
        // Test different log levels
        logger.trace("This is a TRACE message");
        logger.debug("This is a DEBUG message");
        logger.info("This is an INFO message");
        logger.warn("This is a WARN message");
        logger.error("This is an ERROR message");

        // Test with MDC
        MDC.put("userId", "user123");
        MDC.put("requestId", "req-456");
        logger.info("Message with MDC context");
        MDC.clear();
    }
}

/*
Create logback.xml in src/main/resources:

<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <!-- Add console appender -->

    <!-- Add file appender -->

    <!-- Configure root logger -->

    <!-- Configure package-specific loggers -->
</configuration>
*/`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

public class LogbackConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(LogbackConfiguration.class);

    public static void main(String[] args) {
        logger.info("=== Logback Configuration Demo ===");

        // Test different log levels
        logger.trace("This is a TRACE message - very detailed");
        logger.debug("This is a DEBUG message - debugging info");
        logger.info("This is an INFO message - general information");
        logger.warn("This is a WARN message - warning");
        logger.error("This is an ERROR message - error occurred");

        // Test with MDC context
        MDC.put("userId", "user123");
        MDC.put("requestId", "req-456");
        logger.info("Message with MDC context - check logs for userId and requestId");
        MDC.clear();

        // Test exception logging
        try {
            throw new RuntimeException("Sample exception for logging");
        } catch (Exception e) {
            logger.error("Exception occurred during processing", e);
        }

        logger.info("Check application.log file for file output");
    }
}

/*
Complete logback.xml configuration (place in src/main/resources):

<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <!-- Console Appender - for development -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- File Appender - basic file logging -->
    <appender name="FILE" class="ch.qos.logback.core.FileAppender">
        <file>application.log</file>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Rolling File Appender - for production -->
    <appender name="ROLLING" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <!-- Daily rollover -->
            <fileNamePattern>logs/application.%d{yyyy-MM-dd}.log</fileNamePattern>
            <!-- Keep 30 days of history -->
            <maxHistory>30</maxHistory>
            <!-- Total size cap of 3GB -->
            <totalSizeCap>3GB</totalSizeCap>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Set default log level to INFO -->
    <root level="INFO">
        <appender-ref ref="CONSOLE" />
        <appender-ref ref="FILE" />
    </root>

    <!-- Set DEBUG level for specific package -->
    <logger name="com.example.service" level="DEBUG" />

    <!-- Turn off verbose libraries -->
    <logger name="org.springframework" level="WARN" />
    <logger name="org.hibernate" level="WARN" />

</configuration>

Pattern elements:
- %d{HH:mm:ss.SSS} - timestamp
- %thread - thread name
- %-5level - log level (padded to 5 chars)
- %logger{36} - logger name (max 36 chars)
- %msg - log message
- %X{userId} - MDC value for 'userId'
- %n - newline
*/`,
    hint1: `Logback.xml should be in src/main/resources. Use <appender> for output destinations and <logger> for package-specific configuration.`,
    hint2: `Pattern format: %d for date, %level for log level, %logger for class name, %msg for message, %X{key} for MDC values, %n for newline.`,
    whyItMatters: `Proper Logback configuration is essential for production applications. It allows you to control where logs go, their format, rotation policies, and log levels for different parts of your application, making debugging and monitoring much easier.

**Production Pattern:**
\`\`\`xml
<!-- Production logback.xml with async appenders -->
<configuration>
    <appender name="ASYNC_FILE" class="ch.qos.logback.classic.AsyncAppender">
        <queueSize>512</queueSize>
        <discardingThreshold>0</discardingThreshold>
        <appender-ref ref="ROLLING_FILE"/>
    </appender>

    <appender name="ROLLING_FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/app.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
            <fileNamePattern>logs/app-%d{yyyy-MM-dd}.%i.log</fileNamePattern>
            <maxFileSize>100MB</maxFileSize>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{ISO8601} [%thread] %-5level %logger{36} - %msg %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <root level="INFO">
        <appender-ref ref="ASYNC_FILE"/>
    </root>
</configuration>
\`\`\`

**Practical Benefits:**
- Async appender prevents thread blocking during logging
- Size and time-based rotation saves disk space
- JSON format easily parsed by monitoring systems`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify LogbackConfiguration class instantiation
class Test1 {
    @Test
    public void test() {
        LogbackConfiguration obj = new LogbackConfiguration();
        assertNotNull("LogbackConfiguration instance should be created", obj);
    }
}

// Test2: Verify main method executes successfully
class Test2 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Main method should execute successfully", true);
        } catch (Exception e) {
            fail("Main method should not throw exceptions: " + e.getMessage());
        }
    }
}

// Test3: Verify all log levels are logged
class Test3 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("All log levels should work", true);
        } catch (Exception e) {
            fail("All log levels should work: " + e.getMessage());
        }
    }
}

// Test4: Verify MDC context in logs
class Test4 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("MDC context in logs should work", true);
        } catch (Exception e) {
            fail("MDC context should work: " + e.getMessage());
        }
    }
}

// Test5: Verify exception logging
class Test5 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Exception logging should work", true);
        } catch (Exception e) {
            fail("Exception logging should work: " + e.getMessage());
        }
    }
}

// Test6: Verify console appender configuration
class Test6 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Console appender should work", true);
        } catch (Exception e) {
            fail("Console appender should work");
        }
    }
}

// Test7: Verify file appender configuration
class Test7 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("File appender should work", true);
        } catch (Exception e) {
            fail("File appender should work");
        }
    }
}

// Test8: Verify pattern formatting
class Test8 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Pattern formatting should work", true);
        } catch (Exception e) {
            fail("Pattern formatting should work");
        }
    }
}

// Test9: Verify log output generation
class Test9 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Log output should be generated", true);
        } catch (Exception e) {
            fail("Log output generation should work");
        }
    }
}

// Test10: Verify complete logging configuration
class Test10 {
    @Test
    public void test() {
        try {
            LogbackConfiguration.main(new String[]{});
            assertTrue("Complete logging configuration should work", true);
        } catch (Exception e) {
            fail("Complete configuration should work: " + e.getMessage());
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Конфигурация Logback',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

public class LogbackConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(LogbackConfiguration.class);

    public static void main(String[] args) {
        logger.info("=== Демо конфигурации Logback ===");

        // Тестируем разные уровни логирования
        logger.trace("Это TRACE сообщение - очень детальное");
        logger.debug("Это DEBUG сообщение - отладочная информация");
        logger.info("Это INFO сообщение - общая информация");
        logger.warn("Это WARN сообщение - предупреждение");
        logger.error("Это ERROR сообщение - произошла ошибка");

        // Тестируем с контекстом MDC
        MDC.put("userId", "user123");
        MDC.put("requestId", "req-456");
        logger.info("Сообщение с контекстом MDC - проверьте логи на userId и requestId");
        MDC.clear();

        // Тестируем логирование исключений
        try {
            throw new RuntimeException("Пример исключения для логирования");
        } catch (Exception e) {
            logger.error("Произошло исключение во время обработки", e);
        }

        logger.info("Проверьте файл application.log для вывода в файл");
    }
}

/*
Полная конфигурация logback.xml (поместите в src/main/resources):

<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <!-- Консольный Appender - для разработки -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Файловый Appender - базовое логирование в файл -->
    <appender name="FILE" class="ch.qos.logback.core.FileAppender">
        <file>application.log</file>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Rolling File Appender - для продакшена -->
    <appender name="ROLLING" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <!-- Ежедневная ротация -->
            <fileNamePattern>logs/application.%d{yyyy-MM-dd}.log</fileNamePattern>
            <!-- Хранить 30 дней истории -->
            <maxHistory>30</maxHistory>
            <!-- Общий лимит размера 3ГБ -->
            <totalSizeCap>3GB</totalSizeCap>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Установить уровень логирования по умолчанию на INFO -->
    <root level="INFO">
        <appender-ref ref="CONSOLE" />
        <appender-ref ref="FILE" />
    </root>

    <!-- Установить уровень DEBUG для конкретного пакета -->
    <logger name="com.example.service" level="DEBUG" />

    <!-- Выключить подробные библиотеки -->
    <logger name="org.springframework" level="WARN" />
    <logger name="org.hibernate" level="WARN" />

</configuration>

Элементы паттерна:
- %d{HH:mm:ss.SSS} - временная метка
- %thread - имя потока
- %-5level - уровень логирования (выровнен до 5 символов)
- %logger{36} - имя логгера (макс 36 символов)
- %msg - сообщение лога
- %X{userId} - значение MDC для 'userId'
- %n - новая строка
*/`,
            description: `Освойте конфигурацию Logback с appenders и паттернами.

**Требования:**
1. Создайте базовую конфигурацию logback.xml
2. Настройте консольный appender с паттерном
3. Настройте файловый appender
4. Установите разные уровни логирования для разных пакетов
5. Используйте паттерн с временной меткой, уровнем, потоком, логгером и сообщением
6. Настройте rolling file appender для продакшена
7. Включите значения MDC в паттерн

Logback - самая популярная реализация логирования для SLF4J, предоставляющая мощные опции конфигурации для продакшн приложений.`,
            hint1: `Logback.xml должен быть в src/main/resources. Используйте <appender> для выходных направлений и <logger> для конфигурации конкретных пакетов.`,
            hint2: `Формат паттерна: %d для даты, %level для уровня лога, %logger для имени класса, %msg для сообщения, %X{key} для значений MDC, %n для новой строки.`,
            whyItMatters: `Правильная конфигурация Logback необходима для продакшн приложений. Она позволяет контролировать куда идут логи, их формат, политики ротации и уровни логирования для разных частей приложения, значительно упрощая отладку и мониторинг.

**Продакшен паттерн:**
\`\`\`xml
<!-- Production logback.xml с async appenders -->
<configuration>
    <appender name="ASYNC_FILE" class="ch.qos.logback.classic.AsyncAppender">
        <queueSize>512</queueSize>
        <discardingThreshold>0</discardingThreshold>
        <appender-ref ref="ROLLING_FILE"/>
    </appender>

    <appender name="ROLLING_FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/app.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
            <fileNamePattern>logs/app-%d{yyyy-MM-dd}.%i.log</fileNamePattern>
            <maxFileSize>100MB</maxFileSize>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{ISO8601} [%thread] %-5level %logger{36} - %msg %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <root level="INFO">
        <appender-ref ref="ASYNC_FILE"/>
    </root>
</configuration>
\`\`\`

**Практические преимущества:**
- Async appender предотвращает блокировку потоков при логировании
- Ротация по размеру и времени экономит дисковое пространство
- JSON формат легко парсится системами мониторинга`
        },
        uz: {
            title: 'Logback Konfiguratsiyasi',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

public class LogbackConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(LogbackConfiguration.class);

    public static void main(String[] args) {
        logger.info("=== Logback Konfiguratsiyasi Demo ===");

        // Turli log darajalarini sinab ko'ramiz
        logger.trace("Bu TRACE xabari - juda batafsil");
        logger.debug("Bu DEBUG xabari - debug ma'lumoti");
        logger.info("Bu INFO xabari - umumiy ma'lumot");
        logger.warn("Bu WARN xabari - ogohlantirish");
        logger.error("Bu ERROR xabari - xato yuz berdi");

        // MDC konteksti bilan sinab ko'ramiz
        MDC.put("userId", "user123");
        MDC.put("requestId", "req-456");
        logger.info("MDC konteksti bilan xabar - userId va requestId uchun loglarni tekshiring");
        MDC.clear();

        // Exception loggingni sinab ko'ramiz
        try {
            throw new RuntimeException("Logging uchun namuna exception");
        } catch (Exception e) {
            logger.error("Qayta ishlash paytida exception yuz berdi", e);
        }

        logger.info("Fayl chiqishi uchun application.log faylini tekshiring");
    }
}

/*
To'liq logback.xml konfiguratsiyasi (src/main/resources ga joylashtiring):

<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <!-- Console Appender - ishlab chiqish uchun -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- File Appender - oddiy faylga logging -->
    <appender name="FILE" class="ch.qos.logback.core.FileAppender">
        <file>application.log</file>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Rolling File Appender - ishlab chiqarish uchun -->
    <appender name="ROLLING" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <!-- Kunlik rotatsiya -->
            <fileNamePattern>logs/application.%d{yyyy-MM-dd}.log</fileNamePattern>
            <!-- 30 kunlik tarixni saqlash -->
            <maxHistory>30</maxHistory>
            <!-- Jami hajm limiti 3GB -->
            <totalSizeCap>3GB</totalSizeCap>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg %X{userId} %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <!-- Standart log darajasini INFO ga o'rnatish -->
    <root level="INFO">
        <appender-ref ref="CONSOLE" />
        <appender-ref ref="FILE" />
    </root>

    <!-- Ma'lum paket uchun DEBUG darajasini o'rnatish -->
    <logger name="com.example.service" level="DEBUG" />

    <!-- Batafsil kutubxonalarni o'chirish -->
    <logger name="org.springframework" level="WARN" />
    <logger name="org.hibernate" level="WARN" />

</configuration>

Pattern elementlari:
- %d{HH:mm:ss.SSS} - vaqt belgisi
- %thread - thread nomi
- %-5level - log darajasi (5 belgiga to'ldirilgan)
- %logger{36} - logger nomi (maks 36 belgi)
- %msg - log xabari
- %X{userId} - 'userId' uchun MDC qiymati
- %n - yangi qator
*/`,
            description: `Appenderlar va patternlar bilan Logback konfiguratsiyasini o'rganing.

**Talablar:**
1. Asosiy logback.xml konfiguratsiyasini yarating
2. Pattern bilan console appender ni sozlang
3. File appender ni sozlang
4. Turli paketlar uchun turli log darajalarini o'rnating
5. Vaqt belgisi, daraja, thread, logger va xabar bilan pattern ishlatilang
6. Ishlab chiqarish uchun rolling file appender ni sozlang
7. Pattern ga MDC qiymatlarini kiriting

Logback - SLF4J uchun eng mashhur logging implementatsiyasi bo'lib, ishlab chiqarish ilovalari uchun kuchli konfiguratsiya imkoniyatlarini taqdim etadi.`,
            hint1: `Logback.xml src/main/resources da bo'lishi kerak. Chiqish yo'nalishlari uchun <appender> va paketga xos konfiguratsiya uchun <logger> dan foydalaning.`,
            hint2: `Pattern formati: %d sana uchun, %level log darajasi uchun, %logger klass nomi uchun, %msg xabar uchun, %X{key} MDC qiymatlari uchun, %n yangi qator uchun.`,
            whyItMatters: `To'g'ri Logback konfiguratsiyasi ishlab chiqarish ilovalari uchun zarur. U loglar qayerga borishini, ularning formatini, rotatsiya siyosatlarini va ilovangizning turli qismlari uchun log darajalarini boshqarish imkonini beradi, bu debug va monitoringni ancha osonlashtiradi.

**Ishlab chiqarish patterni:**
\`\`\`xml
<!-- Async appenders bilan production logback.xml -->
<configuration>
    <appender name="ASYNC_FILE" class="ch.qos.logback.classic.AsyncAppender">
        <queueSize>512</queueSize>
        <discardingThreshold>0</discardingThreshold>
        <appender-ref ref="ROLLING_FILE"/>
    </appender>

    <appender name="ROLLING_FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/app.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
            <fileNamePattern>logs/app-%d{yyyy-MM-dd}.%i.log</fileNamePattern>
            <maxFileSize>100MB</maxFileSize>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{ISO8601} [%thread] %-5level %logger{36} - %msg %X{requestId}%n</pattern>
        </encoder>
    </appender>

    <root level="INFO">
        <appender-ref ref="ASYNC_FILE"/>
    </root>
</configuration>
\`\`\`

**Amaliy foydalari:**
- Async appender loggingda threadlar blokirovka qilinishini oldini oladi
- Hajm va vaqt bo'yicha rotatsiya disk joyini tejaydi
- JSON format monitoring tizimlar tomonidan osongina parse qilinadi`
        }
    }
};

export default task;
