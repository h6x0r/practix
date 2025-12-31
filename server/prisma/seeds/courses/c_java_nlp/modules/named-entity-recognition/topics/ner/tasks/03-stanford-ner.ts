import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-stanford-ner',
	title: 'Stanford CoreNLP NER',
	difficulty: 'medium',
	tags: ['nlp', 'ner', 'stanford-nlp', 'corenlp'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Stanford CoreNLP NER

Use Stanford CoreNLP for named entity recognition.

## Task

Implement Stanford NER wrapper:
- Configure NER pipeline
- Process text for entities
- Handle different entity types

## Example

\`\`\`java
StanfordNER ner = new StanfordNER();
List<Entity> entities = ner.extract("Barack Obama visited Paris");
// Result: [Entity("Barack Obama", "PERSON"), Entity("Paris", "LOCATION")]
\`\`\``,

	initialCode: `import java.util.*;

public class StanfordNERWrapper {

    /**
     */
    public StanfordNERWrapper(String... entityTypes) {
    }

    /**
     */
    public List<Entity> extract(String text) {
        return null;
    }

    /**
     */
    public List<String> extractByType(String text, String entityType) {
        return null;
    }

    /**
     */
    public Map<String, Integer> getEntityStats(String text) {
        return null;
    }

    public static class Entity {
        public String text;
        public String type;
        public int start;
        public int end;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;

public class StanfordNERWrapper {

    private Set<String> enabledTypes;
    private Map<String, Pattern> typePatterns;

    /**
     * Initialize NER with entity types.
     */
    public StanfordNERWrapper(String... entityTypes) {
        this.enabledTypes = new HashSet<>(Arrays.asList(entityTypes));
        if (enabledTypes.isEmpty()) {
            enabledTypes.addAll(Arrays.asList("PERSON", "LOCATION", "ORGANIZATION", "DATE"));
        }

        // Simple pattern-based simulation (real impl would use CoreNLP)
        typePatterns = new HashMap<>();
        typePatterns.put("PERSON", Pattern.compile(
            "(?:Mr\\\\.|Mrs\\\\.|Ms\\\\.|Dr\\\\.)\\\\s*[A-Z][a-z]+(?:\\\\s+[A-Z][a-z]+)*|" +
            "[A-Z][a-z]+\\\\s+[A-Z][a-z]+"
        ));
        typePatterns.put("LOCATION", Pattern.compile(
            "(?:New\\\\s+York|Los\\\\s+Angeles|San\\\\s+Francisco|Washington|Paris|London|" +
            "Berlin|Tokyo|Beijing|Moscow|Sydney)"
        ));
        typePatterns.put("ORGANIZATION", Pattern.compile(
            "(?:Google|Microsoft|Apple|Amazon|Facebook|Twitter|IBM|Intel|" +
            "[A-Z][a-z]+(?:\\\\s+[A-Z][a-z]+)*\\\\s+(?:Inc|Corp|Ltd|LLC|Co))"
        ));
        typePatterns.put("DATE", Pattern.compile(
            "(?:January|February|March|April|May|June|July|August|September|" +
            "October|November|December)\\\\s+\\\\d{1,2},?\\\\s+\\\\d{4}|" +
            "\\\\d{1,2}[/-]\\\\d{1,2}[/-]\\\\d{2,4}"
        ));
    }

    /**
     * Extract entities from text.
     */
    public List<Entity> extract(String text) {
        List<Entity> entities = new ArrayList<>();

        for (String type : enabledTypes) {
            Pattern pattern = typePatterns.get(type);
            if (pattern != null) {
                Matcher m = pattern.matcher(text);
                while (m.find()) {
                    Entity e = new Entity();
                    e.text = m.group();
                    e.type = type;
                    e.start = m.start();
                    e.end = m.end();
                    entities.add(e);
                }
            }
        }

        // Sort by position
        entities.sort(Comparator.comparingInt(e -> e.start));
        return entities;
    }

    /**
     * Extract specific entity type.
     */
    public List<String> extractByType(String text, String entityType) {
        return extract(text).stream()
            .filter(e -> e.type.equals(entityType))
            .map(e -> e.text)
            .collect(Collectors.toList());
    }

    /**
     * Get entity statistics.
     */
    public Map<String, Integer> getEntityStats(String text) {
        List<Entity> entities = extract(text);
        Map<String, Integer> stats = new HashMap<>();

        for (Entity e : entities) {
            stats.merge(e.type, 1, Integer::sum);
        }
        return stats;
    }

    /**
     * Extract entities with context.
     */
    public List<EntityWithContext> extractWithContext(String text, int contextWindow) {
        List<Entity> entities = extract(text);
        List<EntityWithContext> result = new ArrayList<>();

        for (Entity e : entities) {
            EntityWithContext ewc = new EntityWithContext();
            ewc.entity = e;

            int start = Math.max(0, e.start - contextWindow);
            int end = Math.min(text.length(), e.end + contextWindow);
            ewc.context = text.substring(start, end);

            result.add(ewc);
        }
        return result;
    }

    /**
     * Get most common entities.
     */
    public List<Map.Entry<String, Integer>> getMostCommon(String text, int n) {
        List<Entity> entities = extract(text);
        Map<String, Integer> counts = new HashMap<>();

        for (Entity e : entities) {
            counts.merge(e.text, 1, Integer::sum);
        }

        return counts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(n)
            .collect(Collectors.toList());
    }

    public static class Entity {
        public String text;
        public String type;
        public int start;
        public int end;
    }

    public static class EntityWithContext {
        public Entity entity;
        public String context;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class StanfordNERWrapperTest {

    @Test
    void testExtract() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<StanfordNERWrapper.Entity> entities = ner.extract(
            "Mr. John Smith visited Paris on January 1, 2024"
        );

        assertTrue(entities.size() >= 2);
    }

    @Test
    void testExtractByType() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<String> locations = ner.extractByType(
            "I visited Paris and London", "LOCATION"
        );

        assertEquals(2, locations.size());
        assertTrue(locations.contains("Paris"));
        assertTrue(locations.contains("London"));
    }

    @Test
    void testGetEntityStats() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        Map<String, Integer> stats = ner.getEntityStats(
            "Google and Microsoft are in New York"
        );

        assertTrue(stats.containsKey("ORGANIZATION") || stats.containsKey("LOCATION"));
    }

    @Test
    void testWithSpecificTypes() {
        StanfordNERWrapper ner = new StanfordNERWrapper("LOCATION");
        List<StanfordNERWrapper.Entity> entities = ner.extract(
            "John visited Paris"
        );

        assertTrue(entities.stream().allMatch(e -> e.type.equals("LOCATION")));
    }

    @Test
    void testExtractWithContext() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<StanfordNERWrapper.EntityWithContext> entities = ner.extractWithContext(
            "I went to Paris yesterday", 5
        );
        assertFalse(entities.isEmpty());
        assertNotNull(entities.get(0).context);
    }

    @Test
    void testGetMostCommon() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<Map.Entry<String, Integer>> common = ner.getMostCommon(
            "Paris Paris London", 2
        );
        assertFalse(common.isEmpty());
    }

    @Test
    void testExtractEmpty() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<StanfordNERWrapper.Entity> entities = ner.extract("");
        assertTrue(entities.isEmpty());
    }

    @Test
    void testEntityPositions() {
        StanfordNERWrapper ner = new StanfordNERWrapper("LOCATION");
        List<StanfordNERWrapper.Entity> entities = ner.extract("I visited Paris");
        if (!entities.isEmpty()) {
            assertTrue(entities.get(0).start >= 0);
            assertTrue(entities.get(0).end > entities.get(0).start);
        }
    }

    @Test
    void testExtractDates() {
        StanfordNERWrapper ner = new StanfordNERWrapper("DATE");
        List<StanfordNERWrapper.Entity> entities = ner.extract(
            "Meeting on January 15, 2024"
        );
        assertFalse(entities.isEmpty());
    }

    @Test
    void testExtractByTypeNoMatch() {
        StanfordNERWrapper ner = new StanfordNERWrapper();
        List<String> persons = ner.extractByType("hello world", "PERSON");
        assertTrue(persons.isEmpty());
    }
}`,

	hint1: 'Stanford CoreNLP provides pre-trained NER models for common entity types',
	hint2: 'Pipeline configuration determines which annotators are applied',

	whyItMatters: `Stanford CoreNLP is a production-grade NER system:

- **Pre-trained models**: Works out of the box for common entities
- **High accuracy**: State-of-the-art performance on benchmarks
- **Multiple languages**: Support for various languages
- **Integration**: Easily integrates with Java applications

Industry standard for Java NLP applications.`,

	translations: {
		ru: {
			title: 'Stanford CoreNLP NER',
			description: `# Stanford CoreNLP NER

Используйте Stanford CoreNLP для распознавания именованных сущностей.

## Задача

Реализуйте обертку Stanford NER:
- Настройка NER пайплайна
- Обработка текста для извлечения сущностей
- Обработка разных типов сущностей

## Пример

\`\`\`java
StanfordNER ner = new StanfordNER();
List<Entity> entities = ner.extract("Barack Obama visited Paris");
// Result: [Entity("Barack Obama", "PERSON"), Entity("Paris", "LOCATION")]
\`\`\``,
			hint1: 'Stanford CoreNLP предоставляет предобученные NER модели для распространенных типов сущностей',
			hint2: 'Конфигурация пайплайна определяет какие аннотаторы применяются',
			whyItMatters: `Stanford CoreNLP - production-grade NER система:

- **Предобученные модели**: Работает из коробки для распространенных сущностей
- **Высокая точность**: State-of-the-art производительность на бенчмарках
- **Множество языков**: Поддержка различных языков
- **Интеграция**: Легко интегрируется в Java приложения`,
		},
		uz: {
			title: 'Stanford CoreNLP NER',
			description: `# Stanford CoreNLP NER

Nomlangan ob'ektlarni aniqlash uchun Stanford CoreNLP dan foydalaning.

## Topshiriq

Stanford NER wrapper ni amalga oshiring:
- NER pipeline ni sozlash
- Ob'ektlar uchun matnni qayta ishlash
- Turli ob'ekt turlarini boshqarish

## Misol

\`\`\`java
StanfordNER ner = new StanfordNER();
List<Entity> entities = ner.extract("Barack Obama visited Paris");
// Result: [Entity("Barack Obama", "PERSON"), Entity("Paris", "LOCATION")]
\`\`\``,
			hint1: "Stanford CoreNLP umumiy ob'ekt turlari uchun oldindan o'qitilgan NER modellarini taqdim etadi",
			hint2: "Pipeline konfiguratsiyasi qaysi annotatorlar qo'llanilishini belgilaydi",
			whyItMatters: `Stanford CoreNLP production-grade NER tizimi:

- **Oldindan o'qitilgan modellar**: Umumiy ob'ektlar uchun darhol ishlaydi
- **Yuqori aniqlik**: Benchmarklarda state-of-the-art samaradorlik
- **Ko'p tillar**: Turli tillar uchun qo'llab-quvvatlash
- **Integratsiya**: Java ilovalariga oson integratsiyalanadi`,
		},
	},
};

export default task;
