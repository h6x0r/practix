import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-zoned-datetime',
    title: 'ZonedDateTime, ZoneId, OffsetDateTime',
    difficulty: 'medium',
    tags: ['java', 'date-time', 'timezone', 'zoned-datetime', 'offset'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# ZonedDateTime, ZoneId, OffsetDateTime

ZonedDateTime represents a date-time with a time zone, ZoneId represents a time zone identifier, and OffsetDateTime represents a date-time with an offset from UTC. These classes are essential for handling dates and times across different time zones.

## Requirements:
1. Work with ZoneId:
   1. Get available zone IDs
   2. Get system default zone
   3. Create specific zone IDs
   4. Display zone rules and offsets

2. Use ZonedDateTime:
   1. Get current date-time in different zones
   2. Create zoned date-time from local date-time
   3. Convert between time zones
   4. Handle daylight saving time

3. Work with OffsetDateTime:
   1. Create offset date-time
   2. Convert to/from ZonedDateTime
   3. Compare with different offsets

4. Demonstrate practical scenarios (meeting times, flight schedules)

## Example Output:
\`\`\`
=== ZoneId Examples ===
System default zone: America/New_York
Tokyo zone: Asia/Tokyo
Zone offset: +09:00

=== ZonedDateTime Examples ===
Current in New York: 2024-12-10T09:30:45-05:00[America/New_York]
Current in Tokyo: 2024-12-10T23:30:45+09:00[Asia/Tokyo]
Current in London: 2024-12-10T14:30:45Z[Europe/London]
Converted to UTC: 2024-12-10T14:30:45Z[UTC]

=== OffsetDateTime Examples ===
UTC: 2024-12-10T14:30:45Z
UTC+5: 2024-12-10T19:30:45+05:00
From ZonedDateTime: 2024-12-10T09:30:45-05:00

=== Meeting Time Across Zones ===
Meeting in New York: 2024-12-15T14:00-05:00[America/New_York]
Same time in Tokyo: 2024-12-16T04:00+09:00[Asia/Tokyo]
Same time in London: 2024-12-15T19:00Z[Europe/London]
\`\`\``,
    initialCode: `// TODO: Import java.time classes

public class ZonedDateTimeExample {
    public static void main(String[] args) {
        // TODO: Work with ZoneId

        // TODO: Work with ZonedDateTime

        // TODO: Work with OffsetDateTime

        // TODO: Demonstrate time zone conversions
    }
}`,
    solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Set;

public class ZonedDateTimeExample {
    public static void main(String[] args) {
        System.out.println("=== ZoneId Examples ===");

        // Get system default zone
        ZoneId systemZone = ZoneId.systemDefault();
        System.out.println("System default zone: " + systemZone);

        // Create specific zone IDs
        ZoneId tokyoZone = ZoneId.of("Asia/Tokyo");
        System.out.println("Tokyo zone: " + tokyoZone);

        // Get zone offset
        ZoneOffset tokyoOffset = tokyoZone.getRules()
            .getOffset(Instant.now());
        System.out.println("Zone offset: " + tokyoOffset);

        // Get available zone IDs (show first 5)
        Set<String> zones = ZoneId.getAvailableZoneIds();
        System.out.println("Total available zones: " + zones.size());

        System.out.println("\\n=== ZonedDateTime Examples ===");

        // Get current time in different zones
        ZonedDateTime nowNY = ZonedDateTime.now(ZoneId.of("America/New_York"));
        System.out.println("Current in New York: " + nowNY);

        ZonedDateTime nowTokyo = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        System.out.println("Current in Tokyo: " + nowTokyo);

        ZonedDateTime nowLondon = ZonedDateTime.now(ZoneId.of("Europe/London"));
        System.out.println("Current in London: " + nowLondon);

        // Create ZonedDateTime from LocalDateTime
        LocalDateTime localDT = LocalDateTime.of(2024, 12, 10, 14, 30, 45);
        ZonedDateTime zonedDT = localDT.atZone(ZoneId.of("America/New_York"));
        System.out.println("From LocalDateTime: " + zonedDT);

        // Convert between time zones
        ZonedDateTime convertedToUTC = zonedDT.withZoneSameInstant(ZoneId.of("UTC"));
        System.out.println("Converted to UTC: " + convertedToUTC);

        System.out.println("\\n=== OffsetDateTime Examples ===");

        // Create OffsetDateTime
        OffsetDateTime offsetDT = OffsetDateTime.now(ZoneOffset.UTC);
        System.out.println("UTC: " + offsetDT);

        OffsetDateTime offsetPlus5 = OffsetDateTime.now(ZoneOffset.ofHours(5));
        System.out.println("UTC+5: " + offsetPlus5);

        // Convert ZonedDateTime to OffsetDateTime
        OffsetDateTime offsetFromZoned = zonedDT.toOffsetDateTime();
        System.out.println("From ZonedDateTime: " + offsetFromZoned);

        System.out.println("\\n=== Meeting Time Across Zones ===");

        // Schedule a meeting in New York
        ZonedDateTime meetingNY = ZonedDateTime.of(
            2024, 12, 15, 14, 0, 0, 0,
            ZoneId.of("America/New_York")
        );
        System.out.println("Meeting in New York: " + meetingNY);

        // Convert to other time zones
        ZonedDateTime meetingTokyo = meetingNY
            .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
        System.out.println("Same time in Tokyo: " + meetingTokyo);

        ZonedDateTime meetingLondon = meetingNY
            .withZoneSameInstant(ZoneId.of("Europe/London"));
        System.out.println("Same time in London: " + meetingLondon);

        // Format with custom formatter
        DateTimeFormatter formatter = DateTimeFormatter
            .ofPattern("yyyy-MM-dd HH:mm z");
        System.out.println("\\nFormatted meeting time:");
        System.out.println("NY: " + meetingNY.format(formatter));
        System.out.println("Tokyo: " + meetingTokyo.format(formatter));
        System.out.println("London: " + meetingLondon.format(formatter));
    }
}`,
    hint1: `Use ZoneId.of("Zone/Name") to create time zones. ZonedDateTime.now(zoneId) gives you the current time in that zone.`,
    hint2: `Use withZoneSameInstant() to convert a ZonedDateTime to a different time zone while keeping the same moment in time.`,
    whyItMatters: `Working with time zones is critical for global applications. ZonedDateTime handles the complexity of time zones, including daylight saving time transitions. This is essential for scheduling systems, international communication, flight bookings, and any application serving users across multiple time zones. Proper time zone handling prevents bugs and ensures accurate time-based operations.

**Production Pattern:**
\`\`\`java
// Global meeting scheduling
ZonedDateTime meetingNY = ZonedDateTime.of(
    2024, 12, 15, 14, 0, 0, 0,
    ZoneId.of("America/New_York")
);
// Convert for participants in different zones
ZonedDateTime meetingTokyo = meetingNY
    .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
ZonedDateTime meetingLondon = meetingNY
    .withZoneSameInstant(ZoneId.of("Europe/London"));
\`\`\`

**Practical Benefits:**
- Automatic daylight saving time handling
- Accurate conversion between time zones
- Prevention of errors in global coordination`,
    order: 3,
    testCode: `import java.time.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: ZoneId.systemDefault() returns non-null zone
class Test1 {
    @Test
    void testSystemDefault() {
        ZoneId zone = ZoneId.systemDefault();
        assertNotNull(zone);
    }
}

// Test2: ZoneId.of() creates specific zone
class Test2 {
    @Test
    void testZoneIdOf() {
        ZoneId tokyo = ZoneId.of("Asia/Tokyo");
        assertEquals("Asia/Tokyo", tokyo.getId());
    }
}

// Test3: ZonedDateTime.now() with zone
class Test3 {
    @Test
    void testZonedDateTimeNow() {
        ZonedDateTime nowUTC = ZonedDateTime.now(ZoneId.of("UTC"));
        assertNotNull(nowUTC);
        assertEquals("UTC", nowUTC.getZone().getId());
    }
}

// Test4: withZoneSameInstant() converts zones
class Test4 {
    @Test
    void testZoneConversion() {
        ZonedDateTime nyTime = ZonedDateTime.of(2024, 12, 10, 12, 0, 0, 0, ZoneId.of("America/New_York"));
        ZonedDateTime utcTime = nyTime.withZoneSameInstant(ZoneId.of("UTC"));
        assertEquals(17, utcTime.getHour());
    }
}

// Test5: OffsetDateTime with UTC
class Test5 {
    @Test
    void testOffsetDateTime() {
        OffsetDateTime utc = OffsetDateTime.now(ZoneOffset.UTC);
        assertEquals(ZoneOffset.UTC, utc.getOffset());
    }
}

// Test6: toOffsetDateTime() converts correctly
class Test6 {
    @Test
    void testToOffsetDateTime() {
        ZonedDateTime zdt = ZonedDateTime.now(ZoneId.of("UTC"));
        OffsetDateTime odt = zdt.toOffsetDateTime();
        assertEquals(zdt.toInstant(), odt.toInstant());
    }
}

// Test7: ZoneOffset.ofHours() creates offset
class Test7 {
    @Test
    void testZoneOffset() {
        ZoneOffset offset = ZoneOffset.ofHours(5);
        assertEquals("+05:00", offset.getId());
    }
}

// Test8: Same instant in different zones represents same moment
class Test8 {
    @Test
    void testSameInstant() {
        ZonedDateTime zdt1 = ZonedDateTime.now(ZoneId.of("UTC"));
        ZonedDateTime zdt2 = zdt1.withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
        assertEquals(zdt1.toInstant(), zdt2.toInstant());
    }
}

// Test9: ZoneId.getAvailableZoneIds() returns zones
class Test9 {
    @Test
    void testAvailableZones() {
        var zones = ZoneId.getAvailableZoneIds();
        assertTrue(zones.size() > 100);
        assertTrue(zones.contains("UTC"));
    }
}

// Test10: LocalDateTime.atZone() creates ZonedDateTime
class Test10 {
    @Test
    void testAtZone() {
        LocalDateTime ldt = LocalDateTime.of(2024, 12, 10, 12, 0);
        ZonedDateTime zdt = ldt.atZone(ZoneId.of("UTC"));
        assertEquals(12, zdt.getHour());
        assertEquals("UTC", zdt.getZone().getId());
    }
}
`,
    translations: {
        ru: {
            title: 'ZonedDateTime, ZoneId, OffsetDateTime',
            solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Set;

public class ZonedDateTimeExample {
    public static void main(String[] args) {
        System.out.println("=== Примеры ZoneId ===");

        // Получить системный часовой пояс по умолчанию
        ZoneId systemZone = ZoneId.systemDefault();
        System.out.println("System default zone: " + systemZone);

        // Создать конкретные идентификаторы зон
        ZoneId tokyoZone = ZoneId.of("Asia/Tokyo");
        System.out.println("Tokyo zone: " + tokyoZone);

        // Получить смещение зоны
        ZoneOffset tokyoOffset = tokyoZone.getRules()
            .getOffset(Instant.now());
        System.out.println("Zone offset: " + tokyoOffset);

        // Получить доступные идентификаторы зон (показать первые 5)
        Set<String> zones = ZoneId.getAvailableZoneIds();
        System.out.println("Total available zones: " + zones.size());

        System.out.println("\\n=== Примеры ZonedDateTime ===");

        // Получить текущее время в разных зонах
        ZonedDateTime nowNY = ZonedDateTime.now(ZoneId.of("America/New_York"));
        System.out.println("Current in New York: " + nowNY);

        ZonedDateTime nowTokyo = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        System.out.println("Current in Tokyo: " + nowTokyo);

        ZonedDateTime nowLondon = ZonedDateTime.now(ZoneId.of("Europe/London"));
        System.out.println("Current in London: " + nowLondon);

        // Создать ZonedDateTime из LocalDateTime
        LocalDateTime localDT = LocalDateTime.of(2024, 12, 10, 14, 30, 45);
        ZonedDateTime zonedDT = localDT.atZone(ZoneId.of("America/New_York"));
        System.out.println("From LocalDateTime: " + zonedDT);

        // Конвертировать между часовыми поясами
        ZonedDateTime convertedToUTC = zonedDT.withZoneSameInstant(ZoneId.of("UTC"));
        System.out.println("Converted to UTC: " + convertedToUTC);

        System.out.println("\\n=== Примеры OffsetDateTime ===");

        // Создать OffsetDateTime
        OffsetDateTime offsetDT = OffsetDateTime.now(ZoneOffset.UTC);
        System.out.println("UTC: " + offsetDT);

        OffsetDateTime offsetPlus5 = OffsetDateTime.now(ZoneOffset.ofHours(5));
        System.out.println("UTC+5: " + offsetPlus5);

        // Конвертировать ZonedDateTime в OffsetDateTime
        OffsetDateTime offsetFromZoned = zonedDT.toOffsetDateTime();
        System.out.println("From ZonedDateTime: " + offsetFromZoned);

        System.out.println("\\n=== Время встречи в разных зонах ===");

        // Запланировать встречу в Нью-Йорке
        ZonedDateTime meetingNY = ZonedDateTime.of(
            2024, 12, 15, 14, 0, 0, 0,
            ZoneId.of("America/New_York")
        );
        System.out.println("Meeting in New York: " + meetingNY);

        // Конвертировать в другие часовые пояса
        ZonedDateTime meetingTokyo = meetingNY
            .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
        System.out.println("Same time in Tokyo: " + meetingTokyo);

        ZonedDateTime meetingLondon = meetingNY
            .withZoneSameInstant(ZoneId.of("Europe/London"));
        System.out.println("Same time in London: " + meetingLondon);

        // Форматировать с пользовательским форматером
        DateTimeFormatter formatter = DateTimeFormatter
            .ofPattern("yyyy-MM-dd HH:mm z");
        System.out.println("\\nFormatted meeting time:");
        System.out.println("NY: " + meetingNY.format(formatter));
        System.out.println("Tokyo: " + meetingTokyo.format(formatter));
        System.out.println("London: " + meetingLondon.format(formatter));
    }
}`,
            description: `# ZonedDateTime, ZoneId, OffsetDateTime

ZonedDateTime представляет дату-время с часовым поясом, ZoneId представляет идентификатор часового пояса, а OffsetDateTime представляет дату-время со смещением от UTC. Эти классы необходимы для обработки дат и времени в разных часовых поясах.

## Требования:
1. Работа с ZoneId:
   1. Получить доступные идентификаторы зон
   2. Получить системную зону по умолчанию
   3. Создать конкретные идентификаторы зон
   4. Отобразить правила зон и смещения

2. Использовать ZonedDateTime:
   1. Получить текущую дату-время в разных зонах
   2. Создать зонированную дату-время из локальной даты-времени
   3. Конвертировать между часовыми поясами
   4. Обработать переход на летнее время

3. Работа с OffsetDateTime:
   1. Создать дату-время со смещением
   2. Конвертировать в/из ZonedDateTime
   3. Сравнить с разными смещениями

4. Продемонстрировать практические сценарии (время встреч, расписание рейсов)

## Пример вывода:
\`\`\`
=== ZoneId Examples ===
System default zone: America/New_York
Tokyo zone: Asia/Tokyo
Zone offset: +09:00

=== ZonedDateTime Examples ===
Current in New York: 2024-12-10T09:30:45-05:00[America/New_York]
Current in Tokyo: 2024-12-10T23:30:45+09:00[Asia/Tokyo]
Current in London: 2024-12-10T14:30:45Z[Europe/London]
Converted to UTC: 2024-12-10T14:30:45Z[UTC]

=== OffsetDateTime Examples ===
UTC: 2024-12-10T14:30:45Z
UTC+5: 2024-12-10T19:30:45+05:00
From ZonedDateTime: 2024-12-10T09:30:45-05:00

=== Meeting Time Across Zones ===
Meeting in New York: 2024-12-15T14:00-05:00[America/New_York]
Same time in Tokyo: 2024-12-16T04:00+09:00[Asia/Tokyo]
Same time in London: 2024-12-15T19:00Z[Europe/London]
\`\`\``,
            hint1: `Используйте ZoneId.of("Zone/Name") для создания часовых поясов. ZonedDateTime.now(zoneId) дает вам текущее время в этой зоне.`,
            hint2: `Используйте withZoneSameInstant() для конвертации ZonedDateTime в другой часовой пояс, сохраняя тот же момент времени.`,
            whyItMatters: `Работа с часовыми поясами критически важна для глобальных приложений. ZonedDateTime обрабатывает сложность часовых поясов, включая переходы на летнее время. Это необходимо для систем планирования, международного общения, бронирования авиабилетов и любого приложения, обслуживающего пользователей в нескольких часовых поясах. Правильная обработка часовых поясов предотвращает ошибки и обеспечивает точные операции на основе времени.

**Продакшен паттерн:**
\`\`\`java
// Глобальное планирование встреч
ZonedDateTime meetingNY = ZonedDateTime.of(
    2024, 12, 15, 14, 0, 0, 0,
    ZoneId.of("America/New_York")
);
// Конвертировать для участников в разных зонах
ZonedDateTime meetingTokyo = meetingNY
    .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
ZonedDateTime meetingLondon = meetingNY
    .withZoneSameInstant(ZoneId.of("Europe/London"));
\`\`\`

**Практические преимущества:**
- Автоматическая обработка летнего времени
- Точная конвертация между часовыми поясами
- Предотвращение ошибок при глобальной координации`
        },
        uz: {
            title: `ZonedDateTime, ZoneId, OffsetDateTime`,
            solutionCode: `import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.Set;

public class ZonedDateTimeExample {
    public static void main(String[] args) {
        System.out.println("=== ZoneId namunalari ===");

        // Tizim standart vaqt mintaqasini olish
        ZoneId systemZone = ZoneId.systemDefault();
        System.out.println("System default zone: " + systemZone);

        // Aniq mintaqa identifikatorlarini yaratish
        ZoneId tokyoZone = ZoneId.of("Asia/Tokyo");
        System.out.println("Tokyo zone: " + tokyoZone);

        // Mintaqa siljishini olish
        ZoneOffset tokyoOffset = tokyoZone.getRules()
            .getOffset(Instant.now());
        System.out.println("Zone offset: " + tokyoOffset);

        // Mavjud mintaqa identifikatorlarini olish (birinchi 5 tani ko'rsatish)
        Set<String> zones = ZoneId.getAvailableZoneIds();
        System.out.println("Total available zones: " + zones.size());

        System.out.println("\\n=== ZonedDateTime namunalari ===");

        // Turli mintaqalarda joriy vaqtni olish
        ZonedDateTime nowNY = ZonedDateTime.now(ZoneId.of("America/New_York"));
        System.out.println("Current in New York: " + nowNY);

        ZonedDateTime nowTokyo = ZonedDateTime.now(ZoneId.of("Asia/Tokyo"));
        System.out.println("Current in Tokyo: " + nowTokyo);

        ZonedDateTime nowLondon = ZonedDateTime.now(ZoneId.of("Europe/London"));
        System.out.println("Current in London: " + nowLondon);

        // LocalDateTime dan ZonedDateTime yaratish
        LocalDateTime localDT = LocalDateTime.of(2024, 12, 10, 14, 30, 45);
        ZonedDateTime zonedDT = localDT.atZone(ZoneId.of("America/New_York"));
        System.out.println("From LocalDateTime: " + zonedDT);

        // Vaqt mintaqalari o'rtasida konvertatsiya qilish
        ZonedDateTime convertedToUTC = zonedDT.withZoneSameInstant(ZoneId.of("UTC"));
        System.out.println("Converted to UTC: " + convertedToUTC);

        System.out.println("\\n=== OffsetDateTime namunalari ===");

        // OffsetDateTime yaratish
        OffsetDateTime offsetDT = OffsetDateTime.now(ZoneOffset.UTC);
        System.out.println("UTC: " + offsetDT);

        OffsetDateTime offsetPlus5 = OffsetDateTime.now(ZoneOffset.ofHours(5));
        System.out.println("UTC+5: " + offsetPlus5);

        // ZonedDateTime ni OffsetDateTime ga konvertatsiya qilish
        OffsetDateTime offsetFromZoned = zonedDT.toOffsetDateTime();
        System.out.println("From ZonedDateTime: " + offsetFromZoned);

        System.out.println("\\n=== Turli mintaqalarda uchrashuv vaqti ===");

        // Nyu-Yorkda uchrashuv rejalashtirish
        ZonedDateTime meetingNY = ZonedDateTime.of(
            2024, 12, 15, 14, 0, 0, 0,
            ZoneId.of("America/New_York")
        );
        System.out.println("Meeting in New York: " + meetingNY);

        // Boshqa vaqt mintaqalariga konvertatsiya qilish
        ZonedDateTime meetingTokyo = meetingNY
            .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
        System.out.println("Same time in Tokyo: " + meetingTokyo);

        ZonedDateTime meetingLondon = meetingNY
            .withZoneSameInstant(ZoneId.of("Europe/London"));
        System.out.println("Same time in London: " + meetingLondon);

        // Maxsus formatlash vositasi bilan formatlash
        DateTimeFormatter formatter = DateTimeFormatter
            .ofPattern("yyyy-MM-dd HH:mm z");
        System.out.println("\\nFormatted meeting time:");
        System.out.println("NY: " + meetingNY.format(formatter));
        System.out.println("Tokyo: " + meetingTokyo.format(formatter));
        System.out.println("London: " + meetingLondon.format(formatter));
    }
}`,
            description: `# ZonedDateTime, ZoneId, OffsetDateTime

ZonedDateTime vaqt mintaqasi bilan sana-vaqtni, ZoneId vaqt mintaqasi identifikatorini va OffsetDateTime UTC dan siljish bilan sana-vaqtni ifodalaydi. Bu klasslar turli vaqt mintaqalarida sana va vaqtlarni boshqarish uchun zarur.

## Talablar:
1. ZoneId bilan ishlash:
   1. Mavjud mintaqa identifikatorlarini olish
   2. Tizim standart mintaqasini olish
   3. Aniq mintaqa identifikatorlarini yaratish
   4. Mintaqa qoidalari va siljishlarini ko'rsatish

2. ZonedDateTime dan foydalanish:
   1. Turli mintaqalarda joriy sana-vaqtni olish
   2. Mahalliy sana-vaqtdan mintaqali sana-vaqt yaratish
   3. Vaqt mintaqalari o'rtasida konvertatsiya qilish
   4. Yozgi vaqtga o'tishni boshqarish

3. OffsetDateTime bilan ishlash:
   1. Siljish bilan sana-vaqt yaratish
   2. ZonedDateTime ga/dan konvertatsiya qilish
   3. Turli siljishlar bilan taqqoslash

4. Amaliy stsenariylarni ko'rsatish (uchrashuv vaqtlari, parvoz jadvallari)

## Chiqish namunasi:
\`\`\`
=== ZoneId Examples ===
System default zone: America/New_York
Tokyo zone: Asia/Tokyo
Zone offset: +09:00

=== ZonedDateTime Examples ===
Current in New York: 2024-12-10T09:30:45-05:00[America/New_York]
Current in Tokyo: 2024-12-10T23:30:45+09:00[Asia/Tokyo]
Current in London: 2024-12-10T14:30:45Z[Europe/London]
Converted to UTC: 2024-12-10T14:30:45Z[UTC]

=== OffsetDateTime Examples ===
UTC: 2024-12-10T14:30:45Z
UTC+5: 2024-12-10T19:30:45+05:00
From ZonedDateTime: 2024-12-10T09:30:45-05:00

=== Meeting Time Across Zones ===
Meeting in New York: 2024-12-15T14:00-05:00[America/New_York]
Same time in Tokyo: 2024-12-16T04:00+09:00[Asia/Tokyo]
Same time in London: 2024-12-15T19:00Z[Europe/London]
\`\`\``,
            hint1: `Vaqt mintaqalarini yaratish uchun ZoneId.of("Zone/Name") dan foydalaning. ZonedDateTime.now(zoneId) sizga ushbu mintaqadagi joriy vaqtni beradi.`,
            hint2: `ZonedDateTime ni boshqa vaqt mintaqasiga konvertatsiya qilish uchun withZoneSameInstant() dan foydalaning, bir vaqtning o'zida bir xil lahzani saqlang.`,
            whyItMatters: `Vaqt mintaqalari bilan ishlash global ilovalar uchun muhimdir. ZonedDateTime vaqt mintaqalarining murakkabligini, shu jumladan yozgi vaqtga o'tishlarni boshqaradi. Bu rejalashtirish tizimlari, xalqaro aloqa, parvoz bron qilish va bir nechta vaqt mintaqalarida foydalanuvchilarga xizmat ko'rsatadigan har qanday ilova uchun zarurdir. To'g'ri vaqt mintaqasi bilan ishlash xatolarning oldini oladi va vaqtga asoslangan aniq operatsiyalarni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Global uchrashuvlarni rejalashtirish
ZonedDateTime meetingNY = ZonedDateTime.of(
    2024, 12, 15, 14, 0, 0, 0,
    ZoneId.of("America/New_York")
);
// Turli zonalardagi ishtirokchilar uchun konvertatsiya qilish
ZonedDateTime meetingTokyo = meetingNY
    .withZoneSameInstant(ZoneId.of("Asia/Tokyo"));
ZonedDateTime meetingLondon = meetingNY
    .withZoneSameInstant(ZoneId.of("Europe/London"));
\`\`\`

**Amaliy foydalari:**
- Yozgi vaqtni avtomatik boshqarish
- Vaqt mintaqalari o'rtasida aniq konvertatsiya
- Global koordinatsiyada xatolarning oldini olish`
        }
    }
};

export default task;
