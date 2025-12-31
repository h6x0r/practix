import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-ml-ner',
	title: 'ML-Based NER',
	difficulty: 'medium',
	tags: ['nlp', 'ner', 'classification', 'sequence-labeling'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# ML-Based NER

Use machine learning for named entity recognition.

## Task

Implement ML-based NER:
- Feature extraction for words
- BIO tagging scheme
- Sequence labeling

## Example

\`\`\`java
MLBasedNER ner = new MLBasedNER();
ner.train(trainData);
List<Entity> entities = ner.predict("John works at Google");
// Result: [Entity("John", "PERSON"), Entity("Google", "ORG")]
\`\`\``,

	initialCode: `import java.util.*;

public class MLBasedNER {

    /**
     * Extract features for a word.
     */
    public Map<String, Object> extractFeatures(String word, String prevWord, String nextWord) {
        return null;
    }

    /**
     * Convert BIO tags to entities.
     */
    public List<Entity> bioToEntities(List<String> words, List<String> tags) {
        return null;
    }

    /**
     * Check if word is likely a name (capitalized).
     */
    public boolean isCapitalized(String word) {
        return false;
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

public class MLBasedNER {

    private Set<String> personTitles;
    private Set<String> orgSuffixes;
    private Set<String> locationWords;

    public MLBasedNER() {
        personTitles = new HashSet<>(Arrays.asList(
            "mr", "mrs", "ms", "dr", "prof", "sir", "madam"
        ));
        orgSuffixes = new HashSet<>(Arrays.asList(
            "inc", "corp", "ltd", "llc", "co", "company", "corporation"
        ));
        locationWords = new HashSet<>(Arrays.asList(
            "city", "state", "country", "street", "avenue", "road"
        ));
    }

    /**
     * Extract features for a word.
     */
    public Map<String, Object> extractFeatures(String word, String prevWord, String nextWord) {
        Map<String, Object> features = new HashMap<>();

        // Word features
        features.put("word", word.toLowerCase());
        features.put("isCapitalized", isCapitalized(word));
        features.put("isAllCaps", word.equals(word.toUpperCase()) && word.length() > 1);
        features.put("length", word.length());
        features.put("hasDigit", word.matches(".*\\\\d.*"));
        features.put("isDigit", word.matches("\\\\d+"));

        // Prefix/suffix
        if (word.length() >= 3) {
            features.put("prefix3", word.substring(0, 3).toLowerCase());
            features.put("suffix3", word.substring(word.length() - 3).toLowerCase());
        }

        // Context features
        if (prevWord != null) {
            features.put("prevWord", prevWord.toLowerCase());
            features.put("prevIsTitle", personTitles.contains(prevWord.toLowerCase()));
        }
        if (nextWord != null) {
            features.put("nextWord", nextWord.toLowerCase());
            features.put("nextIsOrgSuffix", orgSuffixes.contains(nextWord.toLowerCase()));
        }

        // Gazeteer features
        features.put("isOrgSuffix", orgSuffixes.contains(word.toLowerCase()));
        features.put("isLocationWord", locationWords.contains(word.toLowerCase()));

        return features;
    }

    /**
     * Convert BIO tags to entities.
     */
    public List<Entity> bioToEntities(List<String> words, List<String> tags) {
        List<Entity> entities = new ArrayList<>();
        StringBuilder currentEntity = new StringBuilder();
        String currentType = null;

        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            String tag = tags.get(i);

            if (tag.startsWith("B-")) {
                // Save previous entity if exists
                if (currentEntity.length() > 0) {
                    entities.add(new Entity(currentEntity.toString().trim(), currentType));
                }
                // Start new entity
                currentEntity = new StringBuilder(word);
                currentType = tag.substring(2);
            } else if (tag.startsWith("I-") && currentType != null) {
                // Continue entity
                currentEntity.append(" ").append(word);
            } else {
                // O tag - save and reset
                if (currentEntity.length() > 0) {
                    entities.add(new Entity(currentEntity.toString().trim(), currentType));
                    currentEntity = new StringBuilder();
                    currentType = null;
                }
            }
        }

        // Don't forget last entity
        if (currentEntity.length() > 0) {
            entities.add(new Entity(currentEntity.toString().trim(), currentType));
        }

        return entities;
    }

    /**
     * Check if word is capitalized.
     */
    public boolean isCapitalized(String word) {
        if (word == null || word.isEmpty()) return false;
        return Character.isUpperCase(word.charAt(0));
    }

    /**
     * Simple heuristic prediction.
     */
    public List<Entity> predict(String text) {
        String[] words = text.split("\\\\s+");
        List<String> wordList = Arrays.asList(words);
        List<String> tags = new ArrayList<>();

        for (int i = 0; i < words.length; i++) {
            String prev = i > 0 ? words[i-1] : null;
            String next = i < words.length - 1 ? words[i+1] : null;
            Map<String, Object> features = extractFeatures(words[i], prev, next);

            // Simple heuristic classification
            String tag = "O";
            if ((Boolean) features.get("isCapitalized") &&
                !(Boolean) features.get("hasDigit")) {

                if (features.get("prevIsTitle") != null &&
                    (Boolean) features.get("prevIsTitle")) {
                    tag = "B-PERSON";
                } else if (features.get("nextIsOrgSuffix") != null &&
                           (Boolean) features.get("nextIsOrgSuffix")) {
                    tag = "B-ORG";
                } else {
                    tag = "B-ENTITY";
                }
            }
            tags.add(tag);
        }

        return bioToEntities(wordList, tags);
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

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class MLBasedNERTest {

    @Test
    void testExtractFeatures() {
        MLBasedNER ner = new MLBasedNER();
        Map<String, Object> features = ner.extractFeatures("John", "Mr", "Smith");

        assertTrue((Boolean) features.get("isCapitalized"));
        assertTrue((Boolean) features.get("prevIsTitle"));
    }

    @Test
    void testBioToEntities() {
        MLBasedNER ner = new MLBasedNER();
        List<String> words = Arrays.asList("John", "Smith", "works", "at", "Google");
        List<String> tags = Arrays.asList("B-PERSON", "I-PERSON", "O", "O", "B-ORG");

        List<MLBasedNER.Entity> entities = ner.bioToEntities(words, tags);

        assertEquals(2, entities.size());
        assertEquals("John Smith", entities.get(0).text);
        assertEquals("PERSON", entities.get(0).type);
    }

    @Test
    void testIsCapitalized() {
        MLBasedNER ner = new MLBasedNER();
        assertTrue(ner.isCapitalized("John"));
        assertFalse(ner.isCapitalized("john"));
    }

    @Test
    void testPredict() {
        MLBasedNER ner = new MLBasedNER();
        List<MLBasedNER.Entity> entities = ner.predict("Mr John works at Google Inc");

        assertTrue(entities.size() >= 1);
    }

    @Test
    void testIsCapitalizedEmpty() {
        MLBasedNER ner = new MLBasedNER();
        assertFalse(ner.isCapitalized(""));
        assertFalse(ner.isCapitalized(null));
    }

    @Test
    void testExtractFeaturesAllCaps() {
        MLBasedNER ner = new MLBasedNER();
        Map<String, Object> features = ner.extractFeatures("NASA", null, null);
        assertTrue((Boolean) features.get("isAllCaps"));
    }

    @Test
    void testBioToEntitiesSingleEntity() {
        MLBasedNER ner = new MLBasedNER();
        List<String> words = Arrays.asList("John");
        List<String> tags = Arrays.asList("B-PERSON");
        List<MLBasedNER.Entity> entities = ner.bioToEntities(words, tags);
        assertEquals(1, entities.size());
    }

    @Test
    void testBioToEntitiesAllO() {
        MLBasedNER ner = new MLBasedNER();
        List<String> words = Arrays.asList("the", "quick", "brown", "fox");
        List<String> tags = Arrays.asList("O", "O", "O", "O");
        List<MLBasedNER.Entity> entities = ner.bioToEntities(words, tags);
        assertTrue(entities.isEmpty());
    }

    @Test
    void testExtractFeaturesWithDigit() {
        MLBasedNER ner = new MLBasedNER();
        Map<String, Object> features = ner.extractFeatures("123", null, null);
        assertTrue((Boolean) features.get("isDigit"));
    }

    @Test
    void testEntityClass() {
        MLBasedNER.Entity entity = new MLBasedNER.Entity("Google", "ORG");
        assertEquals("Google", entity.text);
        assertEquals("ORG", entity.type);
    }
}`,

	hint1: 'BIO: B-TYPE for entity start, I-TYPE for continuation, O for outside',
	hint2: 'Features like capitalization, context words are important for NER',

	whyItMatters: `ML-based NER handles complex entities:

- **Context-aware**: Uses surrounding words for prediction
- **Flexible**: Learns from examples, adapts to domain
- **Multi-word entities**: Handles "New York City" as one entity
- **Industry standard**: Production NER systems use ML

Foundation for information extraction pipelines.`,

	translations: {
		ru: {
			title: 'ML NER',
			description: `# ML NER

Используйте машинное обучение для распознавания именованных сущностей.

## Задача

Реализуйте ML NER:
- Извлечение признаков для слов
- Схема разметки BIO
- Последовательная разметка

## Пример

\`\`\`java
MLBasedNER ner = new MLBasedNER();
ner.train(trainData);
List<Entity> entities = ner.predict("John works at Google");
// Result: [Entity("John", "PERSON"), Entity("Google", "ORG")]
\`\`\``,
			hint1: 'BIO: B-TYPE для начала сущности, I-TYPE для продолжения, O для внешнего',
			hint2: 'Признаки заглавных букв и контекстных слов важны для NER',
			whyItMatters: `ML NER обрабатывает сложные сущности:

- **Учет контекста**: Использует окружающие слова для предсказания
- **Гибкий**: Учится на примерах, адаптируется к домену
- **Многословные сущности**: Обрабатывает "New York City" как одну сущность
- **Промышленный стандарт**: Production NER системы используют ML`,
		},
		uz: {
			title: 'ML asosli NER',
			description: `# ML asosli NER

Nomlangan ob'ektlarni aniqlash uchun mashina o'rganishdan foydalaning.

## Topshiriq

ML NER ni amalga oshiring:
- So'zlar uchun xususiyat ajratish
- BIO belgilash sxemasi
- Ketma-ket belgilash

## Misol

\`\`\`java
MLBasedNER ner = new MLBasedNER();
ner.train(trainData);
List<Entity> entities = ner.predict("John works at Google");
// Result: [Entity("John", "PERSON"), Entity("Google", "ORG")]
\`\`\``,
			hint1: "BIO: ob'ekt boshi uchun B-TYPE, davomi uchun I-TYPE, tashqari uchun O",
			hint2: "Bosh harf va kontekst so'zlar kabi xususiyatlar NER uchun muhim",
			whyItMatters: `ML NER murakkab ob'ektlarni boshqaradi:

- **Kontekstni hisobga oladi**: Bashorat qilish uchun atrofdagi so'zlardan foydalanadi
- **Moslashuvchan**: Misollardan o'rganadi, domenga moslashadi
- **Ko'p so'zli ob'ektlar**: "New York City" ni bitta ob'ekt sifatida boshqaradi
- **Sanoat standarti**: Production NER tizimlari ML dan foydalanadi`,
		},
	},
};

export default task;
