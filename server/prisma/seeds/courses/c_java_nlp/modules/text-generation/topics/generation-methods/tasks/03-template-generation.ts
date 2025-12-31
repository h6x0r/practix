import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-template-generation',
	title: 'Template-Based Generation',
	difficulty: 'easy',
	tags: ['nlp', 'text-generation', 'templates', 'nlg'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Template-Based Generation

Implement template-based Natural Language Generation (NLG).

## Task

Build a template generator that:
- Defines templates with placeholders
- Fills placeholders with data
- Supports conditional sections
- Handles pluralization

## Example

\`\`\`java
TemplateGenerator gen = new TemplateGenerator();
gen.addTemplate("greeting", "Hello, {name}! You have {count} messages.");
String result = gen.generate("greeting", Map.of("name", "Alice", "count", 5));
\`\`\``,

	initialCode: `import java.util.*;

public class TemplateGenerator {

    /**
     * Add a template with given name.
     */
    public void addTemplate(String name, String template) {
    }

    /**
     * Generate text from template with data.
     */
    public String generate(String templateName, Map<String, Object> data) {
        return null;
    }

    /**
     * Fill placeholders in template string.
     */
    public String fillTemplate(String template, Map<String, Object> data) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class TemplateGenerator {

    private Map<String, String> templates;
    private Map<String, List<String>> synonyms;

    public TemplateGenerator() {
        this.templates = new HashMap<>();
        this.synonyms = new HashMap<>();
    }

    /**
     * Add a template with given name.
     */
    public void addTemplate(String name, String template) {
        templates.put(name, template);
    }

    /**
     * Add synonyms for variety.
     */
    public void addSynonyms(String key, List<String> options) {
        synonyms.put(key, new ArrayList<>(options));
    }

    /**
     * Generate text from template with data.
     */
    public String generate(String templateName, Map<String, Object> data) {
        String template = templates.get(templateName);
        if (template == null) {
            throw new IllegalArgumentException("Template not found: " + templateName);
        }
        return fillTemplate(template, data);
    }

    /**
     * Fill placeholders in template string.
     */
    public String fillTemplate(String template, Map<String, Object> data) {
        String result = template;

        // Handle conditionals: {?condition}text{/condition}
        Pattern condPattern = Pattern.compile("\\\\{\\\\?(\\\\w+)\\\\}(.*?)\\\\{/\\\\1\\\\}");
        Matcher condMatcher = condPattern.matcher(result);
        StringBuffer condBuffer = new StringBuffer();

        while (condMatcher.find()) {
            String condKey = condMatcher.group(1);
            String condContent = condMatcher.group(2);
            Object value = data.get(condKey);

            boolean show = value != null && !"false".equals(value.toString()) &&
                          !"0".equals(value.toString()) && !"".equals(value.toString());

            condMatcher.appendReplacement(condBuffer, show ? condContent : "");
        }
        condMatcher.appendTail(condBuffer);
        result = condBuffer.toString();

        // Handle pluralization: {count:singular|plural}
        Pattern pluralPattern = Pattern.compile("\\\\{(\\\\w+):([^|]+)\\\\|([^}]+)\\\\}");
        Matcher pluralMatcher = pluralPattern.matcher(result);
        StringBuffer pluralBuffer = new StringBuffer();

        while (pluralMatcher.find()) {
            String key = pluralMatcher.group(1);
            String singular = pluralMatcher.group(2);
            String plural = pluralMatcher.group(3);

            Object value = data.get(key);
            int count = 1;
            if (value instanceof Number) {
                count = ((Number) value).intValue();
            } else if (value != null) {
                try {
                    count = Integer.parseInt(value.toString());
                } catch (NumberFormatException e) {
                    // keep default
                }
            }

            String replacement = count == 1 ? singular : plural;
            pluralMatcher.appendReplacement(pluralBuffer, Matcher.quoteReplacement(replacement));
        }
        pluralMatcher.appendTail(pluralBuffer);
        result = pluralBuffer.toString();

        // Handle synonym selection: {~synonymKey}
        Pattern synPattern = Pattern.compile("\\\\{~(\\\\w+)\\\\}");
        Matcher synMatcher = synPattern.matcher(result);
        StringBuffer synBuffer = new StringBuffer();
        Random random = new Random();

        while (synMatcher.find()) {
            String synKey = synMatcher.group(1);
            List<String> options = synonyms.get(synKey);
            String replacement = (options != null && !options.isEmpty())
                ? options.get(random.nextInt(options.size()))
                : synKey;
            synMatcher.appendReplacement(synBuffer, Matcher.quoteReplacement(replacement));
        }
        synMatcher.appendTail(synBuffer);
        result = synBuffer.toString();

        // Handle simple placeholders: {key}
        Pattern simplePattern = Pattern.compile("\\\\{(\\\\w+)\\\\}");
        Matcher simpleMatcher = simplePattern.matcher(result);
        StringBuffer simpleBuffer = new StringBuffer();

        while (simpleMatcher.find()) {
            String key = simpleMatcher.group(1);
            Object value = data.get(key);
            String replacement = value != null ? value.toString() : "";
            simpleMatcher.appendReplacement(simpleBuffer, Matcher.quoteReplacement(replacement));
        }
        simpleMatcher.appendTail(simpleBuffer);
        result = simpleBuffer.toString();

        return result.trim().replaceAll("\\\\s+", " ");
    }

    /**
     * Generate with random variation.
     */
    public String generateWithVariation(String templateName, Map<String, Object> data,
                                        String... alternatives) {
        if (alternatives.length == 0) {
            return generate(templateName, data);
        }

        Random random = new Random();
        String selected = alternatives[random.nextInt(alternatives.length)];
        return fillTemplate(selected, data);
    }

    /**
     * List all templates.
     */
    public Set<String> listTemplates() {
        return new HashSet<>(templates.keySet());
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class TemplateGeneratorTest {

    @Test
    void testSimplePlaceholder() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addTemplate("greeting", "Hello, {name}!");

        String result = gen.generate("greeting", Map.of("name", "Alice"));
        assertEquals("Hello, Alice!", result);
    }

    @Test
    void testMultiplePlaceholders() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addTemplate("info", "{name} has {count} items.");

        String result = gen.generate("info", Map.of("name", "Bob", "count", 5));
        assertEquals("Bob has 5 items.", result);
    }

    @Test
    void testPluralization() {
        TemplateGenerator gen = new TemplateGenerator();

        String singular = gen.fillTemplate("You have {count} {count:message|messages}.",
            Map.of("count", 1));
        assertEquals("You have 1 message.", singular);

        String plural = gen.fillTemplate("You have {count} {count:message|messages}.",
            Map.of("count", 5));
        assertEquals("You have 5 messages.", plural);
    }

    @Test
    void testConditional() {
        TemplateGenerator gen = new TemplateGenerator();

        String with = gen.fillTemplate("Hello{?premium}, Premium Member{/premium}!",
            Map.of("premium", true));
        assertEquals("Hello, Premium Member!", with);

        String without = gen.fillTemplate("Hello{?premium}, Premium Member{/premium}!",
            Map.of("premium", false));
        assertEquals("Hello!", without);
    }

    @Test
    void testMissingPlaceholder() {
        TemplateGenerator gen = new TemplateGenerator();
        String result = gen.fillTemplate("Hello, {name}!", Map.of());
        assertEquals("Hello, !", result);
    }

    @Test
    void testListTemplates() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addTemplate("greet", "Hello");
        gen.addTemplate("bye", "Goodbye");
        Set<String> templates = gen.listTemplates();
        assertEquals(2, templates.size());
        assertTrue(templates.contains("greet"));
    }

    @Test
    void testAddSynonyms() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addSynonyms("greeting", Arrays.asList("Hi", "Hello", "Hey"));
        String result = gen.fillTemplate("{~greeting} there!", Map.of());
        assertTrue(result.contains("there!"));
    }

    @Test
    void testTemplateNotFound() {
        TemplateGenerator gen = new TemplateGenerator();
        assertThrows(IllegalArgumentException.class, () ->
            gen.generate("nonexistent", Map.of()));
    }

    @Test
    void testEmptyData() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addTemplate("simple", "No placeholders here.");
        String result = gen.generate("simple", Map.of());
        assertEquals("No placeholders here.", result);
    }

    @Test
    void testGenerateWithVariation() {
        TemplateGenerator gen = new TemplateGenerator();
        gen.addTemplate("main", "Default: {name}");
        String result = gen.generateWithVariation("main", Map.of("name", "Test"),
            "Option 1: {name}", "Option 2: {name}");
        assertTrue(result.contains("Test"));
    }
}`,

	hint1: 'Use regex to find placeholders like {name} and replace with values',
	hint2: 'Handle pluralization by checking if count equals 1',

	whyItMatters: `Template-based generation is practical for production systems:

- **Control**: Exact control over output format
- **Consistency**: Predictable, grammatically correct output
- **Localization**: Easy to translate templates
- **Efficiency**: No ML model needed

Templates are used in chatbots, notifications, reports, and data-to-text systems.`,

	translations: {
		ru: {
			title: 'Генерация на шаблонах',
			description: `# Генерация на шаблонах

Реализуйте Natural Language Generation (NLG) на основе шаблонов.

## Задача

Создайте генератор шаблонов:
- Определение шаблонов с плейсхолдерами
- Заполнение плейсхолдеров данными
- Поддержка условных секций
- Обработка склонений

## Пример

\`\`\`java
TemplateGenerator gen = new TemplateGenerator();
gen.addTemplate("greeting", "Hello, {name}! You have {count} messages.");
String result = gen.generate("greeting", Map.of("name", "Alice", "count", 5));
\`\`\``,
			hint1: 'Используйте regex для поиска плейсхолдеров вида {name} и замены значениями',
			hint2: 'Обрабатывайте множественное число проверяя равно ли количество 1',
			whyItMatters: `Генерация на шаблонах практична для production систем:

- **Контроль**: Точный контроль над форматом вывода
- **Консистентность**: Предсказуемый, грамматически правильный вывод
- **Локализация**: Легко переводить шаблоны
- **Эффективность**: Не нужна ML модель`,
		},
		uz: {
			title: 'Shablonga asoslangan generatsiya',
			description: `# Shablonga asoslangan generatsiya

Shablonga asoslangan Natural Language Generation (NLG) ni amalga oshiring.

## Topshiriq

Shablon generatorini yarating:
- Placeholderlar bilan shablonlarni aniqlash
- Placeholderlarni ma'lumotlar bilan to'ldirish
- Shartli bo'limlarni qo'llab-quvvatlash
- Ko'plik shakllarini qayta ishlash

## Misol

\`\`\`java
TemplateGenerator gen = new TemplateGenerator();
gen.addTemplate("greeting", "Hello, {name}! You have {count} messages.");
String result = gen.generate("greeting", Map.of("name", "Alice", "count", 5));
\`\`\``,
			hint1: "{name} kabi placeholderlarni topish va qiymatlar bilan almashtirish uchun regex dan foydalaning",
			hint2: "Ko'plik shaklini qayta ishlash uchun sanoq 1 ga teng yoki yo'qligini tekshiring",
			whyItMatters: `Shablonga asoslangan generatsiya ishlab chiqarish tizimlari uchun amaliy:

- **Nazorat**: Chiqish formati ustidan aniq nazorat
- **Izchillik**: Bashorat qilinadigan, grammatik jihatdan to'g'ri chiqish
- **Lokalizatsiya**: Shablonlarni tarjima qilish oson
- **Samaradorlik**: ML model kerak emas`,
		},
	},
};

export default task;
