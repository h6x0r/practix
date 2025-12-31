import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-rule-based-pos',
	title: 'Rule-Based POS Tagging',
	difficulty: 'medium',
	tags: ['nlp', 'pos-tagging', 'rules', 'text-processing'],
	estimatedTime: '25m',
	isPremium: false,
	order: 1,
	description: `# Rule-Based POS Tagging

Implement a simple rule-based Part-of-Speech tagger using suffix patterns.

## Task

Create a POS tagger using suffix rules:
- Words ending in "-ing" are likely verbs (VBG)
- Words ending in "-ly" are likely adverbs (RB)
- Words ending in "-ed" are likely past tense verbs (VBD)
- Words ending in "-tion" are likely nouns (NN)

## Example

\`\`\`java
RuleBasedPOS tagger = new RuleBasedPOS();
String tag = tagger.tagWord("running"); // "VBG"
Map<String, String> tags = tagger.tagSentence("quickly running");
\`\`\``,

	initialCode: `import java.util.*;

public class RuleBasedPOS {

    /**
     * Tag a single word based on suffix rules.
     */
    public String tagWord(String word) {
        return null;
    }

    /**
     * Tag all words in a sentence.
     */
    public Map<String, String> tagSentence(String sentence) {
        return null;
    }

    /**
     * Add a custom suffix rule.
     */
    public void addRule(String suffix, String tag) {
    }
}`,

	solutionCode: `import java.util.*;

public class RuleBasedPOS {

    private Map<String, String> suffixRules;
    private Map<String, String> wordLookup;

    public RuleBasedPOS() {
        this.suffixRules = new LinkedHashMap<>();
        this.wordLookup = new HashMap<>();

        // Initialize default suffix rules (order matters)
        suffixRules.put("ing", "VBG");   // Present participle
        suffixRules.put("ly", "RB");     // Adverb
        suffixRules.put("ed", "VBD");    // Past tense
        suffixRules.put("tion", "NN");   // Noun
        suffixRules.put("ness", "NN");   // Noun
        suffixRules.put("ment", "NN");   // Noun
        suffixRules.put("able", "JJ");   // Adjective
        suffixRules.put("ible", "JJ");   // Adjective
        suffixRules.put("ful", "JJ");    // Adjective
        suffixRules.put("less", "JJ");   // Adjective
        suffixRules.put("ous", "JJ");    // Adjective
        suffixRules.put("ive", "JJ");    // Adjective
        suffixRules.put("er", "NN");     // Noun (comparative adj also)
        suffixRules.put("est", "JJS");   // Superlative

        // Common word lookup
        wordLookup.put("the", "DT");
        wordLookup.put("a", "DT");
        wordLookup.put("an", "DT");
        wordLookup.put("is", "VBZ");
        wordLookup.put("are", "VBP");
        wordLookup.put("was", "VBD");
        wordLookup.put("were", "VBD");
        wordLookup.put("and", "CC");
        wordLookup.put("or", "CC");
        wordLookup.put("but", "CC");
    }

    /**
     * Tag a single word based on suffix rules.
     */
    public String tagWord(String word) {
        if (word == null || word.isEmpty()) {
            return "NN";
        }

        String lower = word.toLowerCase();

        // Check word lookup first
        if (wordLookup.containsKey(lower)) {
            return wordLookup.get(lower);
        }

        // Check suffix rules
        for (Map.Entry<String, String> rule : suffixRules.entrySet()) {
            if (lower.endsWith(rule.getKey())) {
                return rule.getValue();
            }
        }

        // Default: assume noun
        return "NN";
    }

    /**
     * Tag all words in a sentence.
     */
    public Map<String, String> tagSentence(String sentence) {
        Map<String, String> result = new LinkedHashMap<>();

        if (sentence == null || sentence.isEmpty()) {
            return result;
        }

        String[] words = sentence.split("\\\\s+");
        for (String word : words) {
            // Remove punctuation for tagging
            String cleanWord = word.replaceAll("[^a-zA-Z]", "");
            if (!cleanWord.isEmpty()) {
                result.put(word, tagWord(cleanWord));
            }
        }

        return result;
    }

    /**
     * Add a custom suffix rule.
     */
    public void addRule(String suffix, String tag) {
        if (suffix != null && tag != null) {
            suffixRules.put(suffix.toLowerCase(), tag.toUpperCase());
        }
    }

    /**
     * Add word to lookup dictionary.
     */
    public void addWordLookup(String word, String tag) {
        if (word != null && tag != null) {
            wordLookup.put(word.toLowerCase(), tag.toUpperCase());
        }
    }

    /**
     * Get tag description.
     */
    public static String getTagDescription(String tag) {
        Map<String, String> descriptions = new HashMap<>();
        descriptions.put("NN", "Noun, singular");
        descriptions.put("NNS", "Noun, plural");
        descriptions.put("VB", "Verb, base form");
        descriptions.put("VBG", "Verb, gerund/present participle");
        descriptions.put("VBD", "Verb, past tense");
        descriptions.put("VBZ", "Verb, 3rd person singular present");
        descriptions.put("VBP", "Verb, non-3rd person singular present");
        descriptions.put("JJ", "Adjective");
        descriptions.put("JJS", "Adjective, superlative");
        descriptions.put("RB", "Adverb");
        descriptions.put("DT", "Determiner");
        descriptions.put("CC", "Coordinating conjunction");

        return descriptions.getOrDefault(tag, "Unknown");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class RuleBasedPOSTest {

    @Test
    void testTagWordIng() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        assertEquals("VBG", tagger.tagWord("running"));
        assertEquals("VBG", tagger.tagWord("walking"));
    }

    @Test
    void testTagWordLy() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        assertEquals("RB", tagger.tagWord("quickly"));
        assertEquals("RB", tagger.tagWord("slowly"));
    }

    @Test
    void testTagWordEd() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        assertEquals("VBD", tagger.tagWord("walked"));
        assertEquals("VBD", tagger.tagWord("jumped"));
    }

    @Test
    void testTagSentence() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        Map<String, String> tags = tagger.tagSentence("quickly running");

        assertEquals(2, tags.size());
        assertEquals("RB", tags.get("quickly"));
        assertEquals("VBG", tags.get("running"));
    }

    @Test
    void testAddRule() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        tagger.addRule("ize", "VB");
        assertEquals("VB", tagger.tagWord("optimize"));
    }

    @Test
    void testTagWordDeterminer() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        assertEquals("DT", tagger.tagWord("the"));
        assertEquals("DT", tagger.tagWord("a"));
    }

    @Test
    void testAddWordLookup() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        tagger.addWordLookup("hello", "UH");
        assertEquals("UH", tagger.tagWord("hello"));
    }

    @Test
    void testGetTagDescription() {
        String desc = RuleBasedPOS.getTagDescription("NN");
        assertEquals("Noun, singular", desc);
    }

    @Test
    void testTagWordEmpty() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        assertEquals("NN", tagger.tagWord(""));
    }

    @Test
    void testTagSentenceEmpty() {
        RuleBasedPOS tagger = new RuleBasedPOS();
        Map<String, String> tags = tagger.tagSentence("");
        assertTrue(tags.isEmpty());
    }
}`,

	hint1: 'Use a Map to store suffix-to-tag mappings and iterate through them',
	hint2: 'Check word lookup dictionary before applying suffix rules for common words',

	whyItMatters: `Rule-based POS tagging is foundational in NLP:

- **Interpretable**: Easy to understand and debug tagging decisions
- **Fast**: No model loading or training required
- **Baseline**: Provides a comparison point for ML-based taggers
- **Domain adaptation**: Easy to add domain-specific rules

Understanding suffix patterns helps you appreciate how language morphology works.`,

	translations: {
		ru: {
			title: 'POS-теггинг на правилах',
			description: `# POS-теггинг на правилах

Реализуйте простой POS-теггер на основе правил с использованием суффиксных паттернов.

## Задача

Создайте POS-теггер с правилами суффиксов:
- Слова на "-ing" - вероятно глаголы (VBG)
- Слова на "-ly" - вероятно наречия (RB)
- Слова на "-ed" - вероятно глаголы прошедшего времени (VBD)
- Слова на "-tion" - вероятно существительные (NN)

## Пример

\`\`\`java
RuleBasedPOS tagger = new RuleBasedPOS();
String tag = tagger.tagWord("running"); // "VBG"
Map<String, String> tags = tagger.tagSentence("quickly running");
\`\`\``,
			hint1: 'Используйте Map для хранения соответствий суффикс-тег и перебирайте их',
			hint2: 'Проверьте словарь слов перед применением правил суффиксов',
			whyItMatters: `POS-теггинг на правилах - основа NLP:

- **Интерпретируемость**: Легко понять и отладить решения о тегах
- **Быстрота**: Не требуется загрузка или обучение модели
- **Базовый уровень**: Точка сравнения для ML-теггеров
- **Адаптация к домену**: Легко добавлять специфичные правила`,
		},
		uz: {
			title: "Qoidalarga asoslangan POS teglash",
			description: `# Qoidalarga asoslangan POS teglash

Suffiks patternlaridan foydalanib oddiy qoidalarga asoslangan POS taggerni amalga oshiring.

## Topshiriq

Suffiks qoidalari bilan POS tagger yarating:
- "-ing" bilan tugaydigan so'zlar - ehtimol fe'llar (VBG)
- "-ly" bilan tugaydigan so'zlar - ehtimol ravishlar (RB)
- "-ed" bilan tugaydigan so'zlar - ehtimol o'tgan zamon fe'llari (VBD)
- "-tion" bilan tugaydigan so'zlar - ehtimol otlar (NN)

## Misol

\`\`\`java
RuleBasedPOS tagger = new RuleBasedPOS();
String tag = tagger.tagWord("running"); // "VBG"
Map<String, String> tags = tagger.tagSentence("quickly running");
\`\`\``,
			hint1: "Suffiks-teg mosliklarini saqlash uchun Map ishlating va ular bo'ylab iteratsiya qiling",
			hint2: "Suffiks qoidalarini qo'llashdan oldin so'z lug'atini tekshiring",
			whyItMatters: `Qoidalarga asoslangan POS teglash NLPda asos:

- **Tushunarli**: Teg qarorlarini tushunish va debug qilish oson
- **Tez**: Model yuklash yoki o'qitish kerak emas
- **Bazaviy daraja**: ML-taggerlar uchun taqqoslash nuqtasi
- **Domen moslashuvi**: Domenga xos qoidalarni qo'shish oson`,
		},
	},
};

export default task;
