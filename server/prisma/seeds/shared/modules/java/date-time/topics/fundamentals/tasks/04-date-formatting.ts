import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-date-formatting',
    title: 'DateTimeFormatter - Parsing and Formatting',
    difficulty: 'easy',
    tags: ['java', 'date-time', 'formatting', 'parsing', 'datetimeformatter'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# DateTimeFormatter - Parsing and Formatting

DateTimeFormatter is used to parse and format date-time objects. It provides predefined formatters (ISO_DATE, ISO_TIME, etc.) and allows creating custom patterns for any date-time format.

## Requirements:
1. Use predefined formatters:
   1. ISO_DATE, ISO_TIME, ISO_DATE_TIME
   2. ISO_LOCAL_DATE, ISO_LOCAL_TIME
   3. BASIC_ISO_DATE

2. Create custom formatters:
   1. Pattern-based formatting (yyyy-MM-dd, dd/MM/yyyy)
   2. Time patterns (HH:mm:ss, hh:mm a)
   3. Combined date-time patterns

3. Parse strings to date-time objects:
   1. Parse with predefined formatters
   2. Parse with custom patterns
   3. Handle different input formats

4. Format date-time objects to strings:
   1. Format with different patterns
   2. Localized formatting
   3. Format for display and storage

## Example Output:
\`\`\`
=== Predefined Formatters ===
ISO_DATE: 2024-12-10
ISO_TIME: 14:30:45.123
ISO_DATE_TIME: 2024-12-10T14:30:45.123
BASIC_ISO_DATE: 20241210

=== Custom Formatters ===
yyyy-MM-dd: 2024-12-10
dd/MM/yyyy: 10/12/2024
MMMM dd, yyyy: December 10, 2024
HH:mm:ss: 14:30:45
hh:mm a: 02:30 PM

=== Parsing Examples ===
Parsed date: 2024-12-10
Parsed time: 14:30:45
Parsed date-time: 2024-12-10T14:30:45
From custom format: 2024-12-10

=== Formatting Examples ===
Short format: 12/10/2024
Long format: Tuesday, December 10, 2024
Time: 02:30:45 PM
ISO format: 2024-12-10T14:30:45
\`\`\``,
    initialCode: `// TODO: Import java.time classes

public class DateTimeFormatting {
    public static void main(String[] args) {
        // TODO: Use predefined formatters

        // TODO: Create custom formatters

        // TODO: Parse dates from strings

        // TODO: Format dates to strings
    }
}`,
    solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;

public class DateTimeFormatting {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45, 123000000);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Predefined Formatters ===");

        // ISO formatters
        System.out.println("ISO_DATE: " +
            date.format(DateTimeFormatter.ISO_DATE));
        System.out.println("ISO_TIME: " +
            time.format(DateTimeFormatter.ISO_TIME));
        System.out.println("ISO_DATE_TIME: " +
            dateTime.format(DateTimeFormatter.ISO_DATE_TIME));
        System.out.println("BASIC_ISO_DATE: " +
            date.format(DateTimeFormatter.BASIC_ISO_DATE));

        System.out.println("\\n=== Custom Formatters ===");

        // Create custom formatters
        DateTimeFormatter standardFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        System.out.println("yyyy-MM-dd: " + date.format(standardFormat));

        DateTimeFormatter euroFormat = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        System.out.println("dd/MM/yyyy: " + date.format(euroFormat));

        DateTimeFormatter longFormat = DateTimeFormatter.ofPattern("MMMM dd, yyyy");
        System.out.println("MMMM dd, yyyy: " + date.format(longFormat));

        DateTimeFormatter time24 = DateTimeFormatter.ofPattern("HH:mm:ss");
        System.out.println("HH:mm:ss: " + time.format(time24));

        DateTimeFormatter time12 = DateTimeFormatter.ofPattern("hh:mm a");
        System.out.println("hh:mm a: " + time.format(time12));

        DateTimeFormatter fullFormat = DateTimeFormatter
            .ofPattern("EEEE, MMMM dd, yyyy 'at' hh:mm a");
        System.out.println("Full format: " + dateTime.format(fullFormat));

        System.out.println("\\n=== Parsing Examples ===");

        // Parse dates from strings
        LocalDate parsedDate = LocalDate.parse("2024-12-10");
        System.out.println("Parsed date: " + parsedDate);

        LocalTime parsedTime = LocalTime.parse("14:30:45");
        System.out.println("Parsed time: " + parsedTime);

        LocalDateTime parsedDateTime = LocalDateTime.parse("2024-12-10T14:30:45");
        System.out.println("Parsed date-time: " + parsedDateTime);

        // Parse with custom formatter
        DateTimeFormatter customParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        LocalDate customParsed = LocalDate.parse("10/12/2024", customParser);
        System.out.println("From custom format: " + customParsed);

        System.out.println("\\n=== Formatting Examples ===");

        // Different format styles
        DateTimeFormatter shortStyle = DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT);
        System.out.println("Short format: " + date.format(shortStyle));

        DateTimeFormatter longStyle = DateTimeFormatter
            .ofLocalizedDate(FormatStyle.FULL);
        System.out.println("Long format: " + date.format(longStyle));

        DateTimeFormatter timeStyle = DateTimeFormatter
            .ofLocalizedTime(FormatStyle.MEDIUM);
        System.out.println("Time: " + time.format(timeStyle));

        // Custom patterns for different use cases
        DateTimeFormatter dbFormat = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
        System.out.println("ISO format: " + dateTime.format(dbFormat));

        DateTimeFormatter displayFormat = DateTimeFormatter
            .ofPattern("MMM dd, yyyy HH:mm");
        System.out.println("Display format: " + dateTime.format(displayFormat));

        // Localized formatting (example with French)
        DateTimeFormatter frenchFormat = DateTimeFormatter
            .ofPattern("d MMMM yyyy", Locale.FRENCH);
        System.out.println("French format: " + date.format(frenchFormat));
    }
}`,
    hint1: `Use DateTimeFormatter.ofPattern() to create custom formatters. Common patterns: yyyy (year), MM (month), dd (day), HH (hour 24), hh (hour 12), mm (minute), ss (second).`,
    hint2: `Use parse() to convert strings to date-time objects and format() to convert date-time objects to strings. Both methods use DateTimeFormatter.`,
    whyItMatters: `Date-time formatting and parsing are essential for displaying dates to users and storing them in databases. DateTimeFormatter provides a thread-safe, flexible way to handle various date formats across different locales. Proper formatting ensures dates are displayed correctly for users worldwide, while proper parsing prevents errors when reading date strings from external sources.

**Production Pattern:**
\`\`\`java
// API responses
DateTimeFormatter isoFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
String apiResponse = currentDateTime.format(isoFormatter);

// User interface
DateTimeFormatter uiFormatter = DateTimeFormatter.ofPattern("MMM dd, yyyy");
String displayDate = date.format(uiFormatter);

// Parsing user input
DateTimeFormatter inputParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
LocalDate userDate = LocalDate.parse(userInput, inputParser);
\`\`\`

**Practical Benefits:**
- Consistent API format between systems
- Localized dates for users
- User input validation and parsing`,
    order: 4,
    testCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: ISO_DATE formats correctly
class Test1 {
    @Test
    void testIsoDate() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        String formatted = date.format(DateTimeFormatter.ISO_DATE);
        assertEquals("2024-12-10", formatted);
    }
}

// Test2: Custom pattern yyyy-MM-dd
class Test2 {
    @Test
    void testCustomPattern() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        assertEquals("10/12/2024", date.format(formatter));
    }
}

// Test3: Time formatting HH:mm:ss
class Test3 {
    @Test
    void testTimeFormatting() {
        LocalTime time = LocalTime.of(14, 30, 45);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        assertEquals("14:30:45", time.format(formatter));
    }
}

// Test4: Parsing date from string
class Test4 {
    @Test
    void testParsing() {
        LocalDate date = LocalDate.parse("2024-12-10");
        assertEquals(2024, date.getYear());
        assertEquals(12, date.getMonthValue());
    }
}

// Test5: Parsing with custom formatter
class Test5 {
    @Test
    void testCustomParsing() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        LocalDate date = LocalDate.parse("10/12/2024", formatter);
        assertEquals(LocalDate.of(2024, 12, 10), date);
    }
}

// Test6: BASIC_ISO_DATE format
class Test6 {
    @Test
    void testBasicIsoDate() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        String formatted = date.format(DateTimeFormatter.BASIC_ISO_DATE);
        assertEquals("20241210", formatted);
    }
}

// Test7: 12-hour format with AM/PM
class Test7 {
    @Test
    void test12HourFormat() {
        LocalTime time = LocalTime.of(14, 30);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("hh:mm a");
        String formatted = time.format(formatter);
        assertTrue(formatted.contains("PM") || formatted.contains("02:30"));
    }
}

// Test8: ISO_DATE_TIME format
class Test8 {
    @Test
    void testIsoDateTime() {
        LocalDateTime dt = LocalDateTime.of(2024, 12, 10, 14, 30, 45);
        String formatted = dt.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        assertEquals("2024-12-10T14:30:45", formatted);
    }
}

// Test9: Parsing LocalDateTime from ISO string
class Test9 {
    @Test
    void testParseDateTime() {
        LocalDateTime dt = LocalDateTime.parse("2024-12-10T14:30:45");
        assertEquals(2024, dt.getYear());
        assertEquals(14, dt.getHour());
    }
}

// Test10: Format with month name
class Test10 {
    @Test
    void testMonthName() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("MMMM dd, yyyy");
        String formatted = date.format(formatter);
        assertTrue(formatted.contains("December") || formatted.contains("10") || formatted.contains("2024"));
    }
}
`,
    translations: {
        ru: {
            title: 'DateTimeFormatter - Парсинг и форматирование',
            solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;

public class DateTimeFormatting {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45, 123000000);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Предопределенные форматеры ===");

        // ISO форматеры
        System.out.println("ISO_DATE: " +
            date.format(DateTimeFormatter.ISO_DATE));
        System.out.println("ISO_TIME: " +
            time.format(DateTimeFormatter.ISO_TIME));
        System.out.println("ISO_DATE_TIME: " +
            dateTime.format(DateTimeFormatter.ISO_DATE_TIME));
        System.out.println("BASIC_ISO_DATE: " +
            date.format(DateTimeFormatter.BASIC_ISO_DATE));

        System.out.println("\\n=== Пользовательские форматеры ===");

        // Создать пользовательские форматеры
        DateTimeFormatter standardFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        System.out.println("yyyy-MM-dd: " + date.format(standardFormat));

        DateTimeFormatter euroFormat = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        System.out.println("dd/MM/yyyy: " + date.format(euroFormat));

        DateTimeFormatter longFormat = DateTimeFormatter.ofPattern("MMMM dd, yyyy");
        System.out.println("MMMM dd, yyyy: " + date.format(longFormat));

        DateTimeFormatter time24 = DateTimeFormatter.ofPattern("HH:mm:ss");
        System.out.println("HH:mm:ss: " + time.format(time24));

        DateTimeFormatter time12 = DateTimeFormatter.ofPattern("hh:mm a");
        System.out.println("hh:mm a: " + time.format(time12));

        DateTimeFormatter fullFormat = DateTimeFormatter
            .ofPattern("EEEE, MMMM dd, yyyy 'at' hh:mm a");
        System.out.println("Full format: " + dateTime.format(fullFormat));

        System.out.println("\\n=== Примеры парсинга ===");

        // Разобрать даты из строк
        LocalDate parsedDate = LocalDate.parse("2024-12-10");
        System.out.println("Parsed date: " + parsedDate);

        LocalTime parsedTime = LocalTime.parse("14:30:45");
        System.out.println("Parsed time: " + parsedTime);

        LocalDateTime parsedDateTime = LocalDateTime.parse("2024-12-10T14:30:45");
        System.out.println("Parsed date-time: " + parsedDateTime);

        // Разобрать с пользовательским форматером
        DateTimeFormatter customParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        LocalDate customParsed = LocalDate.parse("10/12/2024", customParser);
        System.out.println("From custom format: " + customParsed);

        System.out.println("\\n=== Примеры форматирования ===");

        // Различные стили формата
        DateTimeFormatter shortStyle = DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT);
        System.out.println("Short format: " + date.format(shortStyle));

        DateTimeFormatter longStyle = DateTimeFormatter
            .ofLocalizedDate(FormatStyle.FULL);
        System.out.println("Long format: " + date.format(longStyle));

        DateTimeFormatter timeStyle = DateTimeFormatter
            .ofLocalizedTime(FormatStyle.MEDIUM);
        System.out.println("Time: " + time.format(timeStyle));

        // Пользовательские шаблоны для различных случаев использования
        DateTimeFormatter dbFormat = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
        System.out.println("ISO format: " + dateTime.format(dbFormat));

        DateTimeFormatter displayFormat = DateTimeFormatter
            .ofPattern("MMM dd, yyyy HH:mm");
        System.out.println("Display format: " + dateTime.format(displayFormat));

        // Локализованное форматирование (пример с французским)
        DateTimeFormatter frenchFormat = DateTimeFormatter
            .ofPattern("d MMMM yyyy", Locale.FRENCH);
        System.out.println("French format: " + date.format(frenchFormat));
    }
}`,
            description: `# DateTimeFormatter - Парсинг и форматирование

DateTimeFormatter используется для парсинга и форматирования объектов даты-времени. Он предоставляет предопределенные форматеры (ISO_DATE, ISO_TIME и т.д.) и позволяет создавать пользовательские шаблоны для любого формата даты-времени.

## Требования:
1. Использовать предопределенные форматеры:
   1. ISO_DATE, ISO_TIME, ISO_DATE_TIME
   2. ISO_LOCAL_DATE, ISO_LOCAL_TIME
   3. BASIC_ISO_DATE

2. Создать пользовательские форматеры:
   1. Форматирование на основе шаблонов (yyyy-MM-dd, dd/MM/yyyy)
   2. Шаблоны времени (HH:mm:ss, hh:mm a)
   3. Комбинированные шаблоны даты-времени

3. Разбирать строки в объекты даты-времени:
   1. Разбор с предопределенными форматерами
   2. Разбор с пользовательскими шаблонами
   3. Обработка различных входных форматов

4. Форматировать объекты даты-времени в строки:
   1. Форматирование с различными шаблонами
   2. Локализованное форматирование
   3. Форматирование для отображения и хранения

## Пример вывода:
\`\`\`
=== Predefined Formatters ===
ISO_DATE: 2024-12-10
ISO_TIME: 14:30:45.123
ISO_DATE_TIME: 2024-12-10T14:30:45.123
BASIC_ISO_DATE: 20241210

=== Custom Formatters ===
yyyy-MM-dd: 2024-12-10
dd/MM/yyyy: 10/12/2024
MMMM dd, yyyy: December 10, 2024
HH:mm:ss: 14:30:45
hh:mm a: 02:30 PM

=== Parsing Examples ===
Parsed date: 2024-12-10
Parsed time: 14:30:45
Parsed date-time: 2024-12-10T14:30:45
From custom format: 2024-12-10

=== Formatting Examples ===
Short format: 12/10/2024
Long format: Tuesday, December 10, 2024
Time: 02:30:45 PM
ISO format: 2024-12-10T14:30:45
\`\`\``,
            hint1: `Используйте DateTimeFormatter.ofPattern() для создания пользовательских форматеров. Общие шаблоны: yyyy (год), MM (месяц), dd (день), HH (час 24), hh (час 12), mm (минута), ss (секунда).`,
            hint2: `Используйте parse() для преобразования строк в объекты даты-времени и format() для преобразования объектов даты-времени в строки. Оба метода используют DateTimeFormatter.`,
            whyItMatters: `Форматирование и парсинг даты-времени необходимы для отображения дат пользователям и их хранения в базах данных. DateTimeFormatter предоставляет потокобезопасный, гибкий способ обработки различных форматов дат в разных локалях. Правильное форматирование гарантирует, что даты отображаются правильно для пользователей по всему миру, а правильный парсинг предотвращает ошибки при чтении строк дат из внешних источников.

**Продакшен паттерн:**
\`\`\`java
// API ответы
DateTimeFormatter isoFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
String apiResponse = currentDateTime.format(isoFormatter);

// Пользовательский интерфейс
DateTimeFormatter uiFormatter = DateTimeFormatter.ofPattern("MMM dd, yyyy");
String displayDate = date.format(uiFormatter);

// Парсинг пользовательского ввода
DateTimeFormatter inputParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
LocalDate userDate = LocalDate.parse(userInput, inputParser);
\`\`\`

**Практические преимущества:**
- Единообразный формат API между системами
- Локализованные даты для пользователей
- Валидация и парсинг пользовательского ввода`
        },
        uz: {
            title: `DateTimeFormatter - Tahlil va formatlash`,
            solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;

public class DateTimeFormatting {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45, 123000000);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Oldindan belgilangan formatlovchilar ===");

        // ISO formatlovchilar
        System.out.println("ISO_DATE: " +
            date.format(DateTimeFormatter.ISO_DATE));
        System.out.println("ISO_TIME: " +
            time.format(DateTimeFormatter.ISO_TIME));
        System.out.println("ISO_DATE_TIME: " +
            dateTime.format(DateTimeFormatter.ISO_DATE_TIME));
        System.out.println("BASIC_ISO_DATE: " +
            date.format(DateTimeFormatter.BASIC_ISO_DATE));

        System.out.println("\\n=== Maxsus formatlovchilar ===");

        // Maxsus formatlovchilarni yaratish
        DateTimeFormatter standardFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        System.out.println("yyyy-MM-dd: " + date.format(standardFormat));

        DateTimeFormatter euroFormat = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        System.out.println("dd/MM/yyyy: " + date.format(euroFormat));

        DateTimeFormatter longFormat = DateTimeFormatter.ofPattern("MMMM dd, yyyy");
        System.out.println("MMMM dd, yyyy: " + date.format(longFormat));

        DateTimeFormatter time24 = DateTimeFormatter.ofPattern("HH:mm:ss");
        System.out.println("HH:mm:ss: " + time.format(time24));

        DateTimeFormatter time12 = DateTimeFormatter.ofPattern("hh:mm a");
        System.out.println("hh:mm a: " + time.format(time12));

        DateTimeFormatter fullFormat = DateTimeFormatter
            .ofPattern("EEEE, MMMM dd, yyyy 'at' hh:mm a");
        System.out.println("Full format: " + dateTime.format(fullFormat));

        System.out.println("\\n=== Tahlil qilish namunalari ===");

        // Satrlardan sanalarni tahlil qilish
        LocalDate parsedDate = LocalDate.parse("2024-12-10");
        System.out.println("Parsed date: " + parsedDate);

        LocalTime parsedTime = LocalTime.parse("14:30:45");
        System.out.println("Parsed time: " + parsedTime);

        LocalDateTime parsedDateTime = LocalDateTime.parse("2024-12-10T14:30:45");
        System.out.println("Parsed date-time: " + parsedDateTime);

        // Maxsus formatlovchi bilan tahlil qilish
        DateTimeFormatter customParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
        LocalDate customParsed = LocalDate.parse("10/12/2024", customParser);
        System.out.println("From custom format: " + customParsed);

        System.out.println("\\n=== Formatlash namunalari ===");

        // Turli format uslublari
        DateTimeFormatter shortStyle = DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT);
        System.out.println("Short format: " + date.format(shortStyle));

        DateTimeFormatter longStyle = DateTimeFormatter
            .ofLocalizedDate(FormatStyle.FULL);
        System.out.println("Long format: " + date.format(longStyle));

        DateTimeFormatter timeStyle = DateTimeFormatter
            .ofLocalizedTime(FormatStyle.MEDIUM);
        System.out.println("Time: " + time.format(timeStyle));

        // Turli foydalanish holatlari uchun maxsus naqshlar
        DateTimeFormatter dbFormat = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
        System.out.println("ISO format: " + dateTime.format(dbFormat));

        DateTimeFormatter displayFormat = DateTimeFormatter
            .ofPattern("MMM dd, yyyy HH:mm");
        System.out.println("Display format: " + dateTime.format(displayFormat));

        // Mahalliylashtirgan formatlash (frantsuzcha misol)
        DateTimeFormatter frenchFormat = DateTimeFormatter
            .ofPattern("d MMMM yyyy", Locale.FRENCH);
        System.out.println("French format: " + date.format(frenchFormat));
    }
}`,
            description: `# DateTimeFormatter - Tahlil va formatlash

DateTimeFormatter sana-vaqt obyektlarini tahlil qilish va formatlash uchun ishlatiladi. U oldindan belgilangan formatlovchilarni (ISO_DATE, ISO_TIME va boshqalar) taqdim etadi va har qanday sana-vaqt formati uchun maxsus naqshlar yaratishga imkon beradi.

## Talablar:
1. Oldindan belgilangan formatlovchilardan foydalaning:
   1. ISO_DATE, ISO_TIME, ISO_DATE_TIME
   2. ISO_LOCAL_DATE, ISO_LOCAL_TIME
   3. BASIC_ISO_DATE

2. Maxsus formatlovchilarni yarating:
   1. Naqshga asoslangan formatlash (yyyy-MM-dd, dd/MM/yyyy)
   2. Vaqt naqshlari (HH:mm:ss, hh:mm a)
   3. Birlashgan sana-vaqt naqshlari

3. Satrlarni sana-vaqt obyektlariga tahlil qiling:
   1. Oldindan belgilangan formatlovchilar bilan tahlil
   2. Maxsus naqshlar bilan tahlil
   3. Turli kirish formatlarini boshqarish

4. Sana-vaqt obyektlarini satrlarga formatlang:
   1. Turli naqshlar bilan formatlash
   2. Mahalliylashtirgan formatlash
   3. Ko'rsatish va saqlash uchun formatlash

## Chiqish namunasi:
\`\`\`
=== Predefined Formatters ===
ISO_DATE: 2024-12-10
ISO_TIME: 14:30:45.123
ISO_DATE_TIME: 2024-12-10T14:30:45.123
BASIC_ISO_DATE: 20241210

=== Custom Formatters ===
yyyy-MM-dd: 2024-12-10
dd/MM/yyyy: 10/12/2024
MMMM dd, yyyy: December 10, 2024
HH:mm:ss: 14:30:45
hh:mm a: 02:30 PM

=== Parsing Examples ===
Parsed date: 2024-12-10
Parsed time: 14:30:45
Parsed date-time: 2024-12-10T14:30:45
From custom format: 2024-12-10

=== Formatting Examples ===
Short format: 12/10/2024
Long format: Tuesday, December 10, 2024
Time: 02:30:45 PM
ISO format: 2024-12-10T14:30:45
\`\`\``,
            hint1: `Maxsus formatlovchilarni yaratish uchun DateTimeFormatter.ofPattern() dan foydalaning. Umumiy naqshlar: yyyy (yil), MM (oy), dd (kun), HH (soat 24), hh (soat 12), mm (daqiqa), ss (soniya).`,
            hint2: `Satrlarni sana-vaqt obyektlariga aylantirish uchun parse() va sana-vaqt obyektlarini satrlarga aylantirish uchun format() dan foydalaning. Ikkala usul ham DateTimeFormatter dan foydalanadi.`,
            whyItMatters: `Sana-vaqtni formatlash va tahlil qilish foydalanuvchilarga sanalarni ko'rsatish va ma'lumotlar bazalarida saqlash uchun zarurdir. DateTimeFormatter turli joylar bo'ylab turli sana formatlarini boshqarishning thread-safe, moslashuvchan usulini taqdim etadi. To'g'ri formatlash dunyo bo'ylab foydalanuvchilar uchun sanalarning to'g'ri ko'rsatilishini ta'minlaydi, to'g'ri tahlil esa tashqi manbalardan sana satrlarini o'qishda xatolarning oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// API javoblari
DateTimeFormatter isoFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
String apiResponse = currentDateTime.format(isoFormatter);

// Foydalanuvchi interfeysi
DateTimeFormatter uiFormatter = DateTimeFormatter.ofPattern("MMM dd, yyyy");
String displayDate = date.format(uiFormatter);

// Foydalanuvchi kiritmasini tahlil qilish
DateTimeFormatter inputParser = DateTimeFormatter.ofPattern("dd/MM/yyyy");
LocalDate userDate = LocalDate.parse(userInput, inputParser);
\`\`\`

**Amaliy foydalari:**
- Tizimlar o'rtasida izchil API formati
- Foydalanuvchilar uchun mahalliylashtirgan sanalar
- Foydalanuvchi kiritmasini validatsiya va tahlil qilish`
        }
    }
};

export default task;
