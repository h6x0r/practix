import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-sentence-tokenization',
	title: 'Sentence Tokenization',
	difficulty: 'easy',
	tags: ['nlp', 'tokenization', 'sentences'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Sentence Tokenization

Split text into individual sentences.

## Task

Implement sentence detection:
- Handle common abbreviations
- Multiple punctuation marks
- Edge cases (Mr., Dr., etc.)

## Example

\`\`\`java
SentenceTokenizer tokenizer = new SentenceTokenizer();
String[] sentences = tokenizer.tokenize("Hello. How are you? I'm fine!");
// Result: ["Hello.", "How are you?", "I'm fine!"]
\`\`\``,

	initialCode: `import java.util.*;

public class SentenceTokenizer {

    private Set<String> abbreviations;

    /**
     */
    public SentenceTokenizer() {
    }

    /**
     */
    public List<String> tokenize(String text) {
        return null;
    }

    /**
     */
    private boolean isAbbreviation(String word) {
        return false;
    }

    /**
     */
    public int countSentences(String text) {
        return 0;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class SentenceTokenizer {

    private Set<String> abbreviations;
    private static final Pattern SENTENCE_END = Pattern.compile(
        "(?<=[.!?])\\\\s+(?=[A-Z])"
    );

    /**
     * Initialize with common abbreviations.
     */
    public SentenceTokenizer() {
        abbreviations = new HashSet<>(Arrays.asList(
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "vs", "etc", "inc", "ltd", "corp",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec",
            "st", "rd", "th", "ave", "blvd"
        ));
    }

    /**
     * Split text into sentences.
     */
    public List<String> tokenize(String text) {
        List<String> sentences = new ArrayList<>();
        if (text == null || text.isEmpty()) return sentences;

        // Simple approach: split on sentence-ending punctuation
        StringBuilder current = new StringBuilder();
        String[] words = text.split("\\\\s+");

        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            current.append(word);

            if (endsWithSentencePunctuation(word) && !isAbbreviation(word)) {
                sentences.add(current.toString().trim());
                current = new StringBuilder();
            } else if (i < words.length - 1) {
                current.append(" ");
            }
        }

        // Add remaining text as last sentence
        if (current.length() > 0) {
            sentences.add(current.toString().trim());
        }

        return sentences;
    }

    private boolean endsWithSentencePunctuation(String word) {
        return word.endsWith(".") || word.endsWith("!") || word.endsWith("?");
    }

    /**
     * Check if period is end of abbreviation.
     */
    private boolean isAbbreviation(String word) {
        if (word == null) return false;
        String clean = word.replaceAll("[^a-zA-Z]", "").toLowerCase();
        return abbreviations.contains(clean);
    }

    /**
     * Count sentences in text.
     */
    public int countSentences(String text) {
        return tokenize(text).size();
    }

    /**
     * Add custom abbreviation.
     */
    public void addAbbreviation(String abbr) {
        abbreviations.add(abbr.toLowerCase());
    }

    /**
     * Get average sentence length.
     */
    public double averageSentenceLength(String text) {
        List<String> sentences = tokenize(text);
        if (sentences.isEmpty()) return 0;

        int totalWords = 0;
        for (String sentence : sentences) {
            totalWords += sentence.split("\\\\s+").length;
        }
        return (double) totalWords / sentences.size();
    }

    /**
     * Get longest sentence.
     */
    public String getLongestSentence(String text) {
        List<String> sentences = tokenize(text);
        return sentences.stream()
            .max(Comparator.comparingInt(String::length))
            .orElse("");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SentenceTokenizerTest {

    @Test
    void testTokenize() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        List<String> sentences = tokenizer.tokenize("Hello. How are you? I'm fine!");

        assertEquals(3, sentences.size());
        assertEquals("Hello.", sentences.get(0));
    }

    @Test
    void testAbbreviations() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        List<String> sentences = tokenizer.tokenize("Dr. Smith went to the store. He bought milk.");

        assertEquals(2, sentences.size());
        assertTrue(sentences.get(0).contains("Dr."));
    }

    @Test
    void testCountSentences() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        int count = tokenizer.countSentences("First. Second. Third.");
        assertEquals(3, count);
    }

    @Test
    void testAverageSentenceLength() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        double avg = tokenizer.averageSentenceLength("Hello world. This is a test.");
        assertTrue(avg > 0);
    }

    @Test
    void testGetLongestSentence() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        String longest = tokenizer.getLongestSentence("Short. This is a much longer sentence.");
        assertTrue(longest.length() > 10);
    }

    @Test
    void testTokenizeNull() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        List<String> sentences = tokenizer.tokenize(null);
        assertTrue(sentences.isEmpty());
    }

    @Test
    void testTokenizeEmpty() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        List<String> sentences = tokenizer.tokenize("");
        assertTrue(sentences.isEmpty());
    }

    @Test
    void testAddAbbreviation() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        tokenizer.addAbbreviation("pkg");
        // Custom abbreviation should not break sentence
        List<String> sentences = tokenizer.tokenize("Use pkg. Then import.");
        assertEquals(2, sentences.size());
    }

    @Test
    void testMultiplePunctuation() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        List<String> sentences = tokenizer.tokenize("Really?! Yes!");
        assertEquals(2, sentences.size());
    }

    @Test
    void testAverageSentenceLengthEmpty() {
        SentenceTokenizer tokenizer = new SentenceTokenizer();
        assertEquals(0, tokenizer.averageSentenceLength(""), 0.001);
    }
}`,

	hint1: 'Track common abbreviations to avoid false sentence breaks',
	hint2: 'Look for capital letter after punctuation as sentence start signal',

	whyItMatters: `Sentence tokenization enables document-level NLP:

- **Document structure**: Understand paragraph and sentence boundaries
- **Summarization**: Work with sentences as units
- **Translation**: Align sentences between languages
- **Readability**: Analyze sentence complexity

Essential for document-level NLP tasks.`,

	translations: {
		ru: {
			title: 'Токенизация предложений',
			description: `# Токенизация предложений

Разбейте текст на отдельные предложения.

## Задача

Реализуйте определение предложений:
- Обработка распространенных сокращений
- Множественные знаки пунктуации
- Краевые случаи (Mr., Dr. и т.д.)

## Пример

\`\`\`java
SentenceTokenizer tokenizer = new SentenceTokenizer();
String[] sentences = tokenizer.tokenize("Hello. How are you? I'm fine!");
// Result: ["Hello.", "How are you?", "I'm fine!"]
\`\`\``,
			hint1: 'Отслеживайте распространенные сокращения для избежания ложных разрывов',
			hint2: 'Ищите заглавную букву после пунктуации как сигнал начала предложения',
			whyItMatters: `Токенизация предложений позволяет NLP на уровне документа:

- **Структура документа**: Понимание границ абзацев и предложений
- **Суммаризация**: Работа с предложениями как единицами
- **Перевод**: Выравнивание предложений между языками
- **Читаемость**: Анализ сложности предложений`,
		},
		uz: {
			title: 'Gap tokenizatsiyasi',
			description: `# Gap tokenizatsiyasi

Matnni alohida gaplarga bo'ling.

## Topshiriq

Gap aniqlashni amalga oshiring:
- Umumiy qisqartmalarni boshqarish
- Ko'p tinish belgilari
- Chekka holatlar (Mr., Dr. va h.k.)

## Misol

\`\`\`java
SentenceTokenizer tokenizer = new SentenceTokenizer();
String[] sentences = tokenizer.tokenize("Hello. How are you? I'm fine!");
// Result: ["Hello.", "How are you?", "I'm fine!"]
\`\`\``,
			hint1: "Yolg'on gap uzilishlaridan qochish uchun umumiy qisqartmalarni kuzating",
			hint2: "Tinish belgisidan keyin bosh harfni gap boshlanishi signali sifatida qidiring",
			whyItMatters: `Gap tokenizatsiyasi hujjat darajasidagi NLP ni yoqadi:

- **Hujjat tuzilmasi**: Paragraf va gap chegaralarini tushunish
- **Umumlashtirish**: Gaplar bilan birlik sifatida ishlash
- **Tarjima**: Tillar o'rtasida gaplarni tekislash
- **O'qilish qobiliyati**: Gap murakkabligini tahlil qilish`,
		},
	},
};

export default task;
