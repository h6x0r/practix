import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-entity-linking',
	title: 'Entity Linking',
	difficulty: 'hard',
	tags: ['nlp', 'ner', 'entity-linking', 'knowledge-base'],
	estimatedTime: '30m',
	isPremium: true,
	order: 4,
	description: `# Entity Linking

Link named entities to knowledge base entries.

## Task

Implement entity linking that:
- Extracts named entities from text
- Matches entities to knowledge base entries
- Handles ambiguous entity mentions
- Returns linked entities with confidence scores

## Example

\`\`\`java
EntityLinker linker = new EntityLinker(knowledgeBase);
List<LinkedEntity> entities = linker.link("Apple released the iPhone in California.");
// [Apple -> Apple_Inc (company), iPhone -> iPhone (product), California -> California (state)]
\`\`\``,

	initialCode: `import java.util.*;

public class EntityLinker {

    /**
     * Add entry to knowledge base.
     */
    public void addKBEntry(String id, String name, String type, List<String> aliases) {
    }

    /**
     * Link entities in text to knowledge base.
     */
    public List<LinkedEntity> link(String text) {
        return null;
    }

    /**
     * Find best matching KB entry for entity mention.
     */
    public String findBestMatch(String mention, String context) {
        return null;
    }

    public static class LinkedEntity {
        public String mention;
        public String kbId;
        public String type;
        public double confidence;

        public LinkedEntity(String mention, String kbId, String type, double confidence) {
            this.mention = mention;
            this.kbId = kbId;
            this.type = type;
            this.confidence = confidence;
        }
    }
}`,

	solutionCode: `import java.util.*;

public class EntityLinker {

    private Map<String, KBEntry> knowledgeBase;
    private Map<String, List<String>> aliasToIds;

    public EntityLinker() {
        this.knowledgeBase = new HashMap<>();
        this.aliasToIds = new HashMap<>();
        initializeDefaultKB();
    }

    private void initializeDefaultKB() {
        // Companies
        addKBEntry("apple_inc", "Apple", "COMPANY",
            Arrays.asList("Apple Inc", "Apple Computer", "Apple Inc."));
        addKBEntry("microsoft", "Microsoft", "COMPANY",
            Arrays.asList("Microsoft Corporation", "MS", "MSFT"));
        addKBEntry("google", "Google", "COMPANY",
            Arrays.asList("Google LLC", "Alphabet", "GOOG"));

        // Products
        addKBEntry("iphone", "iPhone", "PRODUCT",
            Arrays.asList("Apple iPhone", "iPhone device"));
        addKBEntry("windows", "Windows", "PRODUCT",
            Arrays.asList("Microsoft Windows", "Windows OS"));

        // Locations
        addKBEntry("california", "California", "LOCATION",
            Arrays.asList("CA", "Calif", "State of California"));
        addKBEntry("new_york", "New York", "LOCATION",
            Arrays.asList("NY", "New York City", "NYC"));

        // People
        addKBEntry("tim_cook", "Tim Cook", "PERSON",
            Arrays.asList("Timothy Cook", "Apple CEO"));
        addKBEntry("bill_gates", "Bill Gates", "PERSON",
            Arrays.asList("William Gates", "Gates"));

        // Fruit (for disambiguation)
        addKBEntry("apple_fruit", "apple", "FOOD",
            Arrays.asList("apples", "apple fruit"));
    }

    /**
     * Add entry to knowledge base.
     */
    public void addKBEntry(String id, String name, String type, List<String> aliases) {
        KBEntry entry = new KBEntry(id, name, type, aliases);
        knowledgeBase.put(id, entry);

        // Index by name and aliases
        String lowerName = name.toLowerCase();
        aliasToIds.computeIfAbsent(lowerName, k -> new ArrayList<>()).add(id);

        for (String alias : aliases) {
            String lowerAlias = alias.toLowerCase();
            aliasToIds.computeIfAbsent(lowerAlias, k -> new ArrayList<>()).add(id);
        }
    }

    /**
     * Extract entity mentions from text (simplified).
     */
    private List<String> extractMentions(String text) {
        List<String> mentions = new ArrayList<>();
        String[] words = text.split("\\\\s+");

        StringBuilder current = new StringBuilder();
        for (String word : words) {
            String cleaned = word.replaceAll("[^a-zA-Z]", "");
            if (cleaned.isEmpty()) continue;

            if (Character.isUpperCase(cleaned.charAt(0))) {
                if (current.length() > 0) current.append(" ");
                current.append(cleaned);
            } else {
                if (current.length() > 0) {
                    mentions.add(current.toString());
                    current = new StringBuilder();
                }
            }
        }
        if (current.length() > 0) {
            mentions.add(current.toString());
        }

        return mentions;
    }

    /**
     * Get context words around mention.
     */
    private Set<String> getContextWords(String text, String mention) {
        Set<String> context = new HashSet<>();
        String lower = text.toLowerCase();
        String[] words = lower.split("\\\\W+");

        for (String word : words) {
            if (word.length() > 3 && !mention.toLowerCase().contains(word)) {
                context.add(word);
            }
        }
        return context;
    }

    /**
     * Find best matching KB entry for entity mention.
     */
    public String findBestMatch(String mention, String context) {
        String lowerMention = mention.toLowerCase();
        List<String> candidates = aliasToIds.get(lowerMention);

        if (candidates == null || candidates.isEmpty()) {
            // Try partial match
            for (String alias : aliasToIds.keySet()) {
                if (alias.contains(lowerMention) || lowerMention.contains(alias)) {
                    candidates = aliasToIds.get(alias);
                    break;
                }
            }
        }

        if (candidates == null || candidates.isEmpty()) {
            return null;
        }

        if (candidates.size() == 1) {
            return candidates.get(0);
        }

        // Disambiguate using context
        Set<String> contextWords = getContextWords(context, mention);
        String bestId = null;
        double bestScore = -1;

        for (String candidateId : candidates) {
            KBEntry entry = knowledgeBase.get(candidateId);
            double score = 0;

            // Context word matching
            if (contextWords.contains("company") || contextWords.contains("inc") ||
                contextWords.contains("corporation") || contextWords.contains("released") ||
                contextWords.contains("announced")) {
                if ("COMPANY".equals(entry.type)) score += 2;
            }
            if (contextWords.contains("city") || contextWords.contains("state") ||
                contextWords.contains("country") || contextWords.contains("located")) {
                if ("LOCATION".equals(entry.type)) score += 2;
            }
            if (contextWords.contains("ceo") || contextWords.contains("founder") ||
                contextWords.contains("said")) {
                if ("PERSON".equals(entry.type)) score += 2;
            }
            if (contextWords.contains("product") || contextWords.contains("device") ||
                contextWords.contains("phone") || contextWords.contains("computer")) {
                if ("PRODUCT".equals(entry.type)) score += 2;
            }
            if (contextWords.contains("fruit") || contextWords.contains("food") ||
                contextWords.contains("eat")) {
                if ("FOOD".equals(entry.type)) score += 2;
            }

            // Prefer exact name match
            if (entry.name.equalsIgnoreCase(mention)) {
                score += 1;
            }

            if (score > bestScore) {
                bestScore = score;
                bestId = candidateId;
            }
        }

        return bestId != null ? bestId : candidates.get(0);
    }

    /**
     * Link entities in text to knowledge base.
     */
    public List<LinkedEntity> link(String text) {
        List<LinkedEntity> linked = new ArrayList<>();
        List<String> mentions = extractMentions(text);

        for (String mention : mentions) {
            String kbId = findBestMatch(mention, text);

            if (kbId != null) {
                KBEntry entry = knowledgeBase.get(kbId);
                double confidence = entry.name.equalsIgnoreCase(mention) ? 0.9 : 0.7;
                linked.add(new LinkedEntity(mention, kbId, entry.type, confidence));
            }
        }

        return linked;
    }

    private static class KBEntry {
        String id;
        String name;
        String type;
        List<String> aliases;

        KBEntry(String id, String name, String type, List<String> aliases) {
            this.id = id;
            this.name = name;
            this.type = type;
            this.aliases = aliases;
        }
    }

    public static class LinkedEntity {
        public String mention;
        public String kbId;
        public String type;
        public double confidence;

        public LinkedEntity(String mention, String kbId, String type, double confidence) {
            this.mention = mention;
            this.kbId = kbId;
            this.type = type;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return mention + " -> " + kbId + " (" + type + ", " +
                   String.format("%.2f", confidence) + ")";
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class EntityLinkerTest {

    @Test
    void testLink() {
        EntityLinker linker = new EntityLinker();
        List<EntityLinker.LinkedEntity> entities = linker.link(
            "Apple released the iPhone in California."
        );

        assertTrue(entities.size() >= 2);
    }

    @Test
    void testDisambiguation() {
        EntityLinker linker = new EntityLinker();

        // Company context
        List<EntityLinker.LinkedEntity> company = linker.link(
            "Apple Inc announced new products."
        );
        assertTrue(company.stream().anyMatch(e ->
            e.kbId.equals("apple_inc") && e.type.equals("COMPANY")));

        // Fruit context
        List<EntityLinker.LinkedEntity> fruit = linker.link(
            "I eat an apple fruit every day."
        );
        assertTrue(fruit.stream().anyMatch(e ->
            e.type.equals("FOOD") || e.mention.toLowerCase().contains("apple")));
    }

    @Test
    void testFindBestMatch() {
        EntityLinker linker = new EntityLinker();

        String match = linker.findBestMatch("Apple", "The company released new products");
        assertEquals("apple_inc", match);
    }

    @Test
    void testAddKBEntry() {
        EntityLinker linker = new EntityLinker();
        linker.addKBEntry("test_entity", "Test", "TEST", Arrays.asList("Testing"));

        String match = linker.findBestMatch("Test", "context");
        assertEquals("test_entity", match);
    }

    @Test
    void testLinkedEntityToString() {
        EntityLinker.LinkedEntity entity = new EntityLinker.LinkedEntity(
            "Apple", "apple_inc", "COMPANY", 0.9
        );
        String str = entity.toString();
        assertTrue(str.contains("Apple"));
        assertTrue(str.contains("apple_inc"));
    }

    @Test
    void testLinkEmpty() {
        EntityLinker linker = new EntityLinker();
        List<EntityLinker.LinkedEntity> entities = linker.link("");
        assertTrue(entities.isEmpty());
    }

    @Test
    void testFindBestMatchNoMatch() {
        EntityLinker linker = new EntityLinker();
        String match = linker.findBestMatch("UnknownEntity123", "some context");
        assertNull(match);
    }

    @Test
    void testConfidenceRange() {
        EntityLinker linker = new EntityLinker();
        List<EntityLinker.LinkedEntity> entities = linker.link("Microsoft announced");
        if (!entities.isEmpty()) {
            assertTrue(entities.get(0).confidence >= 0 && entities.get(0).confidence <= 1);
        }
    }

    @Test
    void testLinkLocation() {
        EntityLinker linker = new EntityLinker();
        List<EntityLinker.LinkedEntity> entities = linker.link("We visited California");
        assertTrue(entities.stream().anyMatch(e -> e.type.equals("LOCATION")));
    }

    @Test
    void testLinkPerson() {
        EntityLinker linker = new EntityLinker();
        List<EntityLinker.LinkedEntity> entities = linker.link("Tim Cook is the CEO");
        assertFalse(entities.isEmpty());
    }
}`,

	hint1: 'Index entities by name and all aliases for fast lookup',
	hint2: 'Use context words to disambiguate between multiple candidate matches',

	whyItMatters: `Entity linking connects NLP to knowledge graphs:

- **Disambiguation**: "Apple" -> company or fruit based on context
- **Knowledge enrichment**: Add structured data to extracted entities
- **Question answering**: Link entities to retrieve facts from KB
- **Information integration**: Connect mentions across documents

Entity linking powers knowledge-based AI systems.`,

	translations: {
		ru: {
			title: 'Связывание сущностей',
			description: `# Связывание сущностей

Связывание именованных сущностей с записями базы знаний.

## Задача

Реализуйте связывание сущностей:
- Извлечение именованных сущностей из текста
- Сопоставление сущностей с записями базы знаний
- Обработка неоднозначных упоминаний
- Возврат связанных сущностей с оценками уверенности

## Пример

\`\`\`java
EntityLinker linker = new EntityLinker(knowledgeBase);
List<LinkedEntity> entities = linker.link("Apple released the iPhone in California.");
// [Apple -> Apple_Inc (company), iPhone -> iPhone (product), California -> California (state)]
\`\`\``,
			hint1: 'Индексируйте сущности по имени и всем псевдонимам для быстрого поиска',
			hint2: 'Используйте контекстные слова для устранения неоднозначности между кандидатами',
			whyItMatters: `Связывание сущностей соединяет NLP с графами знаний:

- **Устранение неоднозначности**: "Apple" -> компания или фрукт по контексту
- **Обогащение знаний**: Добавление структурированных данных к сущностям
- **Ответы на вопросы**: Связывание для получения фактов из БЗ
- **Интеграция информации**: Связывание упоминаний между документами`,
		},
		uz: {
			title: "Ob'ektlarni bog'lash",
			description: `# Ob'ektlarni bog'lash

Nomlangan ob'ektlarni bilimlar bazasi yozuvlariga bog'lash.

## Topshiriq

Ob'ektlarni bog'lashni amalga oshiring:
- Matndan nomlangan ob'ektlarni ajratib olish
- Ob'ektlarni bilimlar bazasi yozuvlariga moslashtirish
- Noaniq eslatmalarni qayta ishlash
- Ishonch ballari bilan bog'langan ob'ektlarni qaytarish

## Misol

\`\`\`java
EntityLinker linker = new EntityLinker(knowledgeBase);
List<LinkedEntity> entities = linker.link("Apple released the iPhone in California.");
// [Apple -> Apple_Inc (company), iPhone -> iPhone (product), California -> California (state)]
\`\`\``,
			hint1: "Tez qidirish uchun ob'ektlarni nom va barcha taxalluslar bo'yicha indekslang",
			hint2: "Bir nechta kandidatlar orasida aniqlashtirish uchun kontekst so'zlaridan foydalaning",
			whyItMatters: `Ob'ektlarni bog'lash NLPni bilim graflariga ulaydi:

- **Aniqlashtirish**: "Apple" -> kontekstga qarab kompaniya yoki meva
- **Bilimni boyitish**: Ajratilgan ob'ektlarga strukturalangan ma'lumot qo'shish
- **Savollarga javob**: BB dan faktlarni olish uchun bog'lash
- **Ma'lumot integratsiyasi**: Hujjatlar bo'ylab eslatmalarni bog'lash`,
		},
	},
};

export default task;
