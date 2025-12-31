import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-rule-based-ner',
	title: 'Rule-Based NER',
	difficulty: 'easy',
	tags: ['nlp', 'ner', 'regex', 'rules'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Rule-Based NER

Extract named entities using pattern matching rules.

## Task

Implement rule-based entity extraction:
- Extract email addresses
- Extract phone numbers
- Extract dates
- Extract monetary values

## Example

\`\`\`java
RuleBasedNER ner = new RuleBasedNER();
List<Entity> entities = ner.extract("Call me at 555-1234");
// Result: [Entity("555-1234", "PHONE")]
\`\`\``,

	initialCode: `import java.util.*;
import java.util.regex.*;

public class RuleBasedNER {

    /**
     */
    public List<String> extractEmails(String text) {
        return null;
    }

    /**
     */
    public List<String> extractPhones(String text) {
        return null;
    }

    /**
     */
    public List<String> extractDates(String text) {
        return null;
    }

    /**
     */
    public List<Entity> extractAll(String text) {
        return null;
    }

    public static class Entity {
        public String text;
        public String type;

        public Entity(String text, String type) {
            this.text = text;
            this.type = type;
        }
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class RuleBasedNER {

    private static final Pattern EMAIL = Pattern.compile(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}"
    );
    private static final Pattern PHONE = Pattern.compile(
        "(?:\\\\+?1[-.\\\\s]?)?\\\\(?[0-9]{3}\\\\)?[-.\\\\s]?[0-9]{3}[-.\\\\s]?[0-9]{4}"
    );
    private static final Pattern DATE = Pattern.compile(
        "(?:\\\\d{1,2}[/\\\\-.]\\\\d{1,2}[/\\\\-.]\\\\d{2,4})|" +
        "(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\\\s+\\\\d{1,2},?\\\\s+\\\\d{4})"
    );
    private static final Pattern MONEY = Pattern.compile(
        "\\\\$[0-9,]+(?:\\\\.[0-9]{2})?"
    );
    private static final Pattern URL = Pattern.compile(
        "https?://[\\\\w\\\\-._~:/?#\\\\[\\\\]@!$&'()*+,;=%]+"
    );

    /**
     * Extract email addresses.
     */
    public List<String> extractEmails(String text) {
        return extractPattern(text, EMAIL);
    }

    /**
     * Extract phone numbers.
     */
    public List<String> extractPhones(String text) {
        return extractPattern(text, PHONE);
    }

    /**
     * Extract dates.
     */
    public List<String> extractDates(String text) {
        return extractPattern(text, DATE);
    }

    /**
     * Extract monetary values.
     */
    public List<String> extractMoney(String text) {
        return extractPattern(text, MONEY);
    }

    /**
     * Extract URLs.
     */
    public List<String> extractUrls(String text) {
        return extractPattern(text, URL);
    }

    private List<String> extractPattern(String text, Pattern pattern) {
        List<String> matches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            matches.add(matcher.group());
        }
        return matches;
    }

    /**
     * Extract all entities with types.
     */
    public List<Entity> extractAll(String text) {
        List<Entity> entities = new ArrayList<>();

        for (String email : extractEmails(text)) {
            entities.add(new Entity(email, "EMAIL"));
        }
        for (String phone : extractPhones(text)) {
            entities.add(new Entity(phone, "PHONE"));
        }
        for (String date : extractDates(text)) {
            entities.add(new Entity(date, "DATE"));
        }
        for (String money : extractMoney(text)) {
            entities.add(new Entity(money, "MONEY"));
        }
        for (String url : extractUrls(text)) {
            entities.add(new Entity(url, "URL"));
        }

        return entities;
    }

    /**
     * Entity with position information.
     */
    public List<EntityWithPosition> extractWithPositions(String text) {
        List<EntityWithPosition> entities = new ArrayList<>();
        Pattern[] patterns = {EMAIL, PHONE, DATE, MONEY, URL};
        String[] types = {"EMAIL", "PHONE", "DATE", "MONEY", "URL"};

        for (int i = 0; i < patterns.length; i++) {
            Matcher m = patterns[i].matcher(text);
            while (m.find()) {
                entities.add(new EntityWithPosition(
                    m.group(), types[i], m.start(), m.end()
                ));
            }
        }
        return entities;
    }

    public static class Entity {
        public String text;
        public String type;

        public Entity(String text, String type) {
            this.text = text;
            this.type = type;
        }
    }

    public static class EntityWithPosition extends Entity {
        public int start;
        public int end;

        public EntityWithPosition(String text, String type, int start, int end) {
            super(text, type);
            this.start = start;
            this.end = end;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class RuleBasedNERTest {

    @Test
    void testExtractEmails() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> emails = ner.extractEmails("Contact test@example.com for info");

        assertEquals(1, emails.size());
        assertEquals("test@example.com", emails.get(0));
    }

    @Test
    void testExtractPhones() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> phones = ner.extractPhones("Call me at 555-123-4567");

        assertEquals(1, phones.size());
    }

    @Test
    void testExtractDates() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> dates = ner.extractDates("Meeting on 12/25/2024");

        assertEquals(1, dates.size());
    }

    @Test
    void testExtractAll() {
        RuleBasedNER ner = new RuleBasedNER();
        String text = "Email test@example.com or call 555-1234";
        List<RuleBasedNER.Entity> entities = ner.extractAll(text);

        assertTrue(entities.size() >= 2);
    }

    @Test
    void testExtractMoney() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> money = ner.extractMoney("Price is $99.99 today");
        assertEquals(1, money.size());
        assertEquals("$99.99", money.get(0));
    }

    @Test
    void testExtractUrls() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> urls = ner.extractUrls("Visit https://example.com for more");
        assertEquals(1, urls.size());
    }

    @Test
    void testExtractWithPositions() {
        RuleBasedNER ner = new RuleBasedNER();
        List<RuleBasedNER.EntityWithPosition> entities = ner.extractWithPositions("test@example.com");
        assertFalse(entities.isEmpty());
        assertTrue(entities.get(0).start >= 0);
    }

    @Test
    void testEntityClass() {
        RuleBasedNER.Entity entity = new RuleBasedNER.Entity("test@example.com", "EMAIL");
        assertEquals("test@example.com", entity.text);
        assertEquals("EMAIL", entity.type);
    }

    @Test
    void testExtractEmptyText() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> emails = ner.extractEmails("");
        assertTrue(emails.isEmpty());
    }

    @Test
    void testMultipleEmails() {
        RuleBasedNER ner = new RuleBasedNER();
        List<String> emails = ner.extractEmails("Contact a@test.com or b@test.com");
        assertEquals(2, emails.size());
    }
}`,

	hint1: 'Use regex patterns with capturing groups for entity extraction',
	hint2: 'Store both the matched text and entity type',

	whyItMatters: `Rule-based NER is fast and precise:

- **No training**: Works immediately with good patterns
- **Interpretable**: Easy to understand and debug
- **High precision**: Exact matches when patterns are correct
- **Customizable**: Easy to add domain-specific patterns

Good for structured entities like emails, phones, dates.`,

	translations: {
		ru: {
			title: 'NER на основе правил',
			description: `# NER на основе правил

Извлекайте именованные сущности используя правила сопоставления паттернов.

## Задача

Реализуйте извлечение сущностей на правилах:
- Извлечение email адресов
- Извлечение телефонных номеров
- Извлечение дат
- Извлечение денежных значений

## Пример

\`\`\`java
RuleBasedNER ner = new RuleBasedNER();
List<Entity> entities = ner.extract("Call me at 555-1234");
// Result: [Entity("555-1234", "PHONE")]
\`\`\``,
			hint1: 'Используйте regex паттерны с группами захвата для извлечения сущностей',
			hint2: 'Сохраняйте и извлеченный текст и тип сущности',
			whyItMatters: `NER на правилах быстрый и точный:

- **Без обучения**: Работает сразу с хорошими паттернами
- **Интерпретируемый**: Легко понять и отладить
- **Высокая точность**: Точные совпадения при правильных паттернах
- **Настраиваемый**: Легко добавить доменные паттерны`,
		},
		uz: {
			title: "Qoidalarga asoslangan NER",
			description: `# Qoidalarga asoslangan NER

Pattern moslashtirish qoidalari yordamida nomlangan ob'ektlarni ajratib oling.

## Topshiriq

Qoidalarga asoslangan ob'ekt ajratishni amalga oshiring:
- Email manzillarini ajratib olish
- Telefon raqamlarini ajratib olish
- Sanalarni ajratib olish
- Pul qiymatlarini ajratib olish

## Misol

\`\`\`java
RuleBasedNER ner = new RuleBasedNER();
List<Entity> entities = ner.extract("Call me at 555-1234");
// Result: [Entity("555-1234", "PHONE")]
\`\`\``,
			hint1: "Ob'ekt ajratish uchun tutish guruhlari bilan regex patternlardan foydalaning",
			hint2: "Mos kelgan matnni ham, ob'ekt turini ham saqlang",
			whyItMatters: `Qoidalarga asoslangan NER tez va aniq:

- **O'qitish kerak emas**: Yaxshi patternlar bilan darhol ishlaydi
- **Tushunarli**: Tushunish va debugging oson
- **Yuqori aniqlik**: Patternlar to'g'ri bo'lganda aniq mosliklar
- **Sozlanishi mumkin**: Domenga xos patternlarni qo'shish oson`,
		},
	},
};

export default task;
