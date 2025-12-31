import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-stanford-pos',
	title: 'Stanford POS Tagger',
	difficulty: 'medium',
	tags: ['nlp', 'pos-tagging', 'stanford-nlp', 'ml'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# Stanford POS Tagger

Use Stanford CoreNLP for accurate Part-of-Speech tagging.

## Task

Implement POS tagging using Stanford CoreNLP:
- Initialize the tagger with appropriate model
- Tag words with Penn Treebank tags
- Extract tag statistics from text

## Example

\`\`\`java
StanfordPOSTagger tagger = new StanfordPOSTagger();
List<TaggedWord> result = tagger.tag("The quick brown fox jumps over the lazy dog");
Map<String, Integer> stats = tagger.getTagDistribution(text);
\`\`\``,

	initialCode: `import java.util.*;

public class StanfordPOSTagger {

    /**
     * Tag words in a sentence.
     */
    public List<TaggedWord> tag(String sentence) {
        return null;
    }

    /**
     * Get distribution of POS tags in text.
     */
    public Map<String, Integer> getTagDistribution(String text) {
        return null;
    }

    /**
     * Extract words by specific tag.
     */
    public List<String> getWordsByTag(String text, String targetTag) {
        return null;
    }

    public static class TaggedWord {
        public String word;
        public String tag;

        public TaggedWord(String word, String tag) {
            this.word = word;
            this.tag = tag;
        }
    }
}`,

	solutionCode: `import java.util.*;

public class StanfordPOSTagger {

    // Simulated Stanford tagger (real impl uses edu.stanford.nlp)
    private Map<String, String> commonWords;
    private Map<String, String> suffixRules;

    public StanfordPOSTagger() {
        initializeRules();
    }

    private void initializeRules() {
        commonWords = new HashMap<>();
        // Determiners
        commonWords.put("the", "DT");
        commonWords.put("a", "DT");
        commonWords.put("an", "DT");
        commonWords.put("this", "DT");
        commonWords.put("that", "DT");

        // Pronouns
        commonWords.put("i", "PRP");
        commonWords.put("you", "PRP");
        commonWords.put("he", "PRP");
        commonWords.put("she", "PRP");
        commonWords.put("it", "PRP");

        // Verbs
        commonWords.put("is", "VBZ");
        commonWords.put("are", "VBP");
        commonWords.put("was", "VBD");
        commonWords.put("were", "VBD");
        commonWords.put("be", "VB");
        commonWords.put("have", "VBP");
        commonWords.put("has", "VBZ");
        commonWords.put("had", "VBD");

        // Prepositions
        commonWords.put("in", "IN");
        commonWords.put("on", "IN");
        commonWords.put("at", "IN");
        commonWords.put("to", "TO");
        commonWords.put("for", "IN");
        commonWords.put("with", "IN");
        commonWords.put("over", "IN");

        // Adjectives
        commonWords.put("quick", "JJ");
        commonWords.put("brown", "JJ");
        commonWords.put("lazy", "JJ");

        // Nouns
        commonWords.put("fox", "NN");
        commonWords.put("dog", "NN");

        // Verbs
        commonWords.put("jumps", "VBZ");
        commonWords.put("runs", "VBZ");

        suffixRules = new LinkedHashMap<>();
        suffixRules.put("ing", "VBG");
        suffixRules.put("ed", "VBD");
        suffixRules.put("ly", "RB");
        suffixRules.put("tion", "NN");
        suffixRules.put("ness", "NN");
        suffixRules.put("ment", "NN");
        suffixRules.put("able", "JJ");
        suffixRules.put("ful", "JJ");
        suffixRules.put("less", "JJ");
        suffixRules.put("ous", "JJ");
    }

    /**
     * Tag words in a sentence.
     */
    public List<TaggedWord> tag(String sentence) {
        List<TaggedWord> result = new ArrayList<>();

        if (sentence == null || sentence.isEmpty()) {
            return result;
        }

        String[] tokens = sentence.split("\\\\s+");
        for (String token : tokens) {
            String cleanToken = token.replaceAll("[^a-zA-Z]", "");
            if (!cleanToken.isEmpty()) {
                String tag = predictTag(cleanToken);
                result.add(new TaggedWord(cleanToken, tag));
            }
        }

        return result;
    }

    private String predictTag(String word) {
        String lower = word.toLowerCase();

        // Check common words
        if (commonWords.containsKey(lower)) {
            return commonWords.get(lower);
        }

        // Check suffix rules
        for (Map.Entry<String, String> rule : suffixRules.entrySet()) {
            if (lower.endsWith(rule.getKey())) {
                return rule.getValue();
            }
        }

        // Capitalized words are likely proper nouns
        if (Character.isUpperCase(word.charAt(0))) {
            return "NNP";
        }

        // Default to noun
        return "NN";
    }

    /**
     * Get distribution of POS tags in text.
     */
    public Map<String, Integer> getTagDistribution(String text) {
        Map<String, Integer> distribution = new HashMap<>();

        List<TaggedWord> tagged = tag(text);
        for (TaggedWord tw : tagged) {
            distribution.merge(tw.tag, 1, Integer::sum);
        }

        return distribution;
    }

    /**
     * Extract words by specific tag.
     */
    public List<String> getWordsByTag(String text, String targetTag) {
        List<String> words = new ArrayList<>();

        List<TaggedWord> tagged = tag(text);
        for (TaggedWord tw : tagged) {
            if (tw.tag.equals(targetTag)) {
                words.add(tw.word);
            }
        }

        return words;
    }

    /**
     * Get noun phrases (simplified).
     */
    public List<String> getNounPhrases(String text) {
        List<String> phrases = new ArrayList<>();
        List<TaggedWord> tagged = tag(text);

        StringBuilder phrase = new StringBuilder();
        for (TaggedWord tw : tagged) {
            if (tw.tag.startsWith("NN") || tw.tag.equals("JJ") || tw.tag.equals("DT")) {
                if (phrase.length() > 0) phrase.append(" ");
                phrase.append(tw.word);
            } else {
                if (phrase.length() > 0 && phrase.toString().split("\\\\s+").length > 1) {
                    phrases.add(phrase.toString().trim());
                }
                phrase = new StringBuilder();
            }
        }

        if (phrase.length() > 0 && phrase.toString().split("\\\\s+").length > 1) {
            phrases.add(phrase.toString().trim());
        }

        return phrases;
    }

    public static class TaggedWord {
        public String word;
        public String tag;

        public TaggedWord(String word, String tag) {
            this.word = word;
            this.tag = tag;
        }

        @Override
        public String toString() {
            return word + "/" + tag;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class StanfordPOSTaggerTest {

    @Test
    void testTagSentence() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<StanfordPOSTagger.TaggedWord> result = tagger.tag("The quick brown fox");

        assertEquals(4, result.size());
        assertEquals("DT", result.get(0).tag);  // The
        assertEquals("JJ", result.get(1).tag);  // quick
    }

    @Test
    void testGetTagDistribution() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        Map<String, Integer> dist = tagger.getTagDistribution("The quick brown fox jumps over the lazy dog");

        assertTrue(dist.containsKey("DT"));
        assertTrue(dist.containsKey("JJ"));
        assertEquals(2, dist.get("DT"));  // the, the
    }

    @Test
    void testGetWordsByTag() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<String> adjectives = tagger.getWordsByTag("The quick brown fox", "JJ");

        assertEquals(2, adjectives.size());
        assertTrue(adjectives.contains("quick"));
        assertTrue(adjectives.contains("brown"));
    }

    @Test
    void testEmptyInput() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<StanfordPOSTagger.TaggedWord> result = tagger.tag("");
        assertTrue(result.isEmpty());
    }

    @Test
    void testGetNounPhrases() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<String> phrases = tagger.getNounPhrases("The quick brown fox jumps");
        assertFalse(phrases.isEmpty());
    }

    @Test
    void testTaggedWordToString() {
        StanfordPOSTagger.TaggedWord tw = new StanfordPOSTagger.TaggedWord("running", "VBG");
        assertEquals("running/VBG", tw.toString());
    }

    @Test
    void testProperNounDetection() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<StanfordPOSTagger.TaggedWord> result = tagger.tag("John went to Paris");
        assertEquals("NNP", result.get(0).tag);  // John
    }

    @Test
    void testSuffixRulesIng() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<StanfordPOSTagger.TaggedWord> result = tagger.tag("running quickly");
        assertEquals("VBG", result.get(0).tag);
        assertEquals("RB", result.get(1).tag);
    }

    @Test
    void testGetWordsByTagNoMatch() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<String> words = tagger.getWordsByTag("The quick fox", "VBZ");
        assertTrue(words.isEmpty());
    }

    @Test
    void testNullInput() {
        StanfordPOSTagger tagger = new StanfordPOSTagger();
        List<StanfordPOSTagger.TaggedWord> result = tagger.tag(null);
        assertTrue(result.isEmpty());
    }
}`,

	hint1: 'Use a dictionary for common words and fallback to suffix rules',
	hint2: 'Penn Treebank tagset includes NN (noun), VB (verb), JJ (adjective), etc.',

	whyItMatters: `Stanford POS tagger provides production-quality tagging:

- **Accuracy**: 97%+ on standard benchmarks
- **Penn Treebank**: Industry-standard tagset with 45 tags
- **Integration**: Works with full NLP pipeline
- **Models**: Multiple language models available

POS tagging is essential for downstream tasks like parsing and NER.`,

	translations: {
		ru: {
			title: 'Stanford POS Tagger',
			description: `# Stanford POS Tagger

Используйте Stanford CoreNLP для точного POS-теггинга.

## Задача

Реализуйте POS-теггинг с Stanford CoreNLP:
- Инициализация теггера с подходящей моделью
- Разметка слов тегами Penn Treebank
- Извлечение статистики тегов из текста

## Пример

\`\`\`java
StanfordPOSTagger tagger = new StanfordPOSTagger();
List<TaggedWord> result = tagger.tag("The quick brown fox jumps over the lazy dog");
Map<String, Integer> stats = tagger.getTagDistribution(text);
\`\`\``,
			hint1: 'Используйте словарь для частых слов и запасные правила суффиксов',
			hint2: 'Набор тегов Penn Treebank включает NN (существительное), VB (глагол), JJ (прилагательное) и др.',
			whyItMatters: `Stanford POS tagger обеспечивает качество продакшена:

- **Точность**: 97%+ на стандартных бенчмарках
- **Penn Treebank**: Отраслевой стандарт с 45 тегами
- **Интеграция**: Работает с полным NLP пайплайном
- **Модели**: Доступны модели для многих языков`,
		},
		uz: {
			title: 'Stanford POS Tagger',
			description: `# Stanford POS Tagger

Aniq POS teglash uchun Stanford CoreNLP dan foydalaning.

## Topshiriq

Stanford CoreNLP bilan POS teglashni amalga oshiring:
- Taggerni mos model bilan ishga tushirish
- So'zlarni Penn Treebank teglari bilan belgilash
- Matndan teg statistikasini ajratib olish

## Misol

\`\`\`java
StanfordPOSTagger tagger = new StanfordPOSTagger();
List<TaggedWord> result = tagger.tag("The quick brown fox jumps over the lazy dog");
Map<String, Integer> stats = tagger.getTagDistribution(text);
\`\`\``,
			hint1: "Tez-tez ishlatiladigan so'zlar uchun lug'at va zaxira suffiks qoidalaridan foydalaning",
			hint2: "Penn Treebank teglar to'plami NN (ot), VB (fe'l), JJ (sifat) va boshqalarni o'z ichiga oladi",
			whyItMatters: `Stanford POS tagger ishlab chiqarish sifatini ta'minlaydi:

- **Aniqlik**: Standart benchmarklarda 97%+
- **Penn Treebank**: 45 tegdan iborat sanoat standarti
- **Integratsiya**: To'liq NLP pipeline bilan ishlaydi
- **Modellar**: Ko'p tillar uchun modellar mavjud`,
		},
	},
};

export default task;
