import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-word-tokenization',
	title: 'Word Tokenization',
	difficulty: 'easy',
	tags: ['nlp', 'tokenization', 'opennlp'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Word Tokenization

Split text into individual words (tokens).

## Task

Implement word tokenization:
- Simple whitespace tokenization
- Handle punctuation properly
- Use OpenNLP tokenizer

## Example

\`\`\`java
Tokenizer tokenizer = new WordTokenizer();
String[] tokens = tokenizer.tokenize("Hello, world!");
// Result: ["Hello", ",", "world", "!"]
\`\`\``,

	initialCode: `import java.util.*;

public class WordTokenizer {

    /**
     * Simple whitespace tokenization.
     */
    public static String[] tokenizeWhitespace(String text) {
        return null;
    }

    /**
     * Tokenize keeping punctuation separate.
     */
    public static List<String> tokenizeWithPunctuation(String text) {
        return null;
    }

    /**
     * Count tokens in text.
     */
    public static int countTokens(String text) {
        return 0;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class WordTokenizer {

    private static final Pattern WORD_PATTERN = Pattern.compile(
        "\\\\w+|[^\\\\w\\\\s]"
    );

    /**
     * Simple whitespace tokenization.
     */
    public static String[] tokenizeWhitespace(String text) {
        if (text == null || text.isEmpty()) return new String[0];
        return text.trim().split("\\\\s+");
    }

    /**
     * Tokenize keeping punctuation separate.
     */
    public static List<String> tokenizeWithPunctuation(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null) return tokens;

        Matcher matcher = WORD_PATTERN.matcher(text);
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    /**
     * Count tokens in text.
     */
    public static int countTokens(String text) {
        return tokenizeWithPunctuation(text).size();
    }

    /**
     * Tokenize into n-grams.
     */
    public static List<String> nGrams(String text, int n) {
        List<String> tokens = tokenizeWithPunctuation(text);
        List<String> ngrams = new ArrayList<>();

        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append(" ");
                sb.append(tokens.get(i + j));
            }
            ngrams.add(sb.toString());
        }
        return ngrams;
    }

    /**
     * Get token frequencies.
     */
    public static Map<String, Integer> tokenFrequencies(String text) {
        List<String> tokens = tokenizeWithPunctuation(text);
        Map<String, Integer> freq = new HashMap<>();

        for (String token : tokens) {
            freq.merge(token.toLowerCase(), 1, Integer::sum);
        }
        return freq;
    }

    /**
     * Get vocabulary (unique tokens).
     */
    public static Set<String> getVocabulary(String text) {
        List<String> tokens = tokenizeWithPunctuation(text);
        Set<String> vocab = new HashSet<>();
        for (String token : tokens) {
            vocab.add(token.toLowerCase());
        }
        return vocab;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class WordTokenizerTest {

    @Test
    void testTokenizeWhitespace() {
        String[] tokens = WordTokenizer.tokenizeWhitespace("Hello world test");
        assertEquals(3, tokens.length);
        assertEquals("Hello", tokens[0]);
    }

    @Test
    void testTokenizeWithPunctuation() {
        List<String> tokens = WordTokenizer.tokenizeWithPunctuation("Hello, world!");
        assertEquals(4, tokens.size());
        assertTrue(tokens.contains(","));
        assertTrue(tokens.contains("!"));
    }

    @Test
    void testCountTokens() {
        int count = WordTokenizer.countTokens("Hello, world!");
        assertEquals(4, count);
    }

    @Test
    void testNGrams() {
        List<String> bigrams = WordTokenizer.nGrams("I love NLP", 2);
        assertEquals(2, bigrams.size());
        assertEquals("I love", bigrams.get(0));
    }

    @Test
    void testTokenFrequencies() {
        Map<String, Integer> freq = WordTokenizer.tokenFrequencies("hello hello world");
        assertEquals(2, freq.get("hello").intValue());
        assertEquals(1, freq.get("world").intValue());
    }

    @Test
    void testGetVocabulary() {
        Set<String> vocab = WordTokenizer.getVocabulary("hello Hello world");
        assertEquals(2, vocab.size());
        assertTrue(vocab.contains("hello"));
    }

    @Test
    void testTokenizeWhitespaceNull() {
        String[] tokens = WordTokenizer.tokenizeWhitespace(null);
        assertEquals(0, tokens.length);
    }

    @Test
    void testTokenizeWhitespaceEmpty() {
        String[] tokens = WordTokenizer.tokenizeWhitespace("");
        assertEquals(0, tokens.length);
    }

    @Test
    void testTokenizeWithPunctuationNull() {
        List<String> tokens = WordTokenizer.tokenizeWithPunctuation(null);
        assertTrue(tokens.isEmpty());
    }

    @Test
    void testNGramsTrigrams() {
        List<String> trigrams = WordTokenizer.nGrams("I love NLP very much", 3);
        assertTrue(trigrams.size() >= 3);
    }
}`,

	hint1: 'Use regex \\\\w+ to match word characters',
	hint2: 'Handle edge cases like empty strings and null',

	whyItMatters: `Tokenization is the first step in NLP:

- **Text to data**: Convert unstructured text to processable units
- **Vocabulary building**: Create word lists for models
- **Feature extraction**: Tokens become features for ML
- **Language agnostic**: Basic tokenization works across languages

All NLP pipelines start with tokenization.`,

	translations: {
		ru: {
			title: 'Токенизация слов',
			description: `# Токенизация слов

Разбейте текст на отдельные слова (токены).

## Задача

Реализуйте токенизацию слов:
- Простая токенизация по пробелам
- Правильная обработка пунктуации
- Использование OpenNLP токенизатора

## Пример

\`\`\`java
Tokenizer tokenizer = new WordTokenizer();
String[] tokens = tokenizer.tokenize("Hello, world!");
// Result: ["Hello", ",", "world", "!"]
\`\`\``,
			hint1: 'Используйте regex \\\\w+ для захвата символов слова',
			hint2: 'Обрабатывайте краевые случаи пустых строк и null',
			whyItMatters: `Токенизация - первый шаг в NLP:

- **Текст в данные**: Преобразование неструктурированного текста в обрабатываемые единицы
- **Построение словаря**: Создание списков слов для моделей
- **Извлечение признаков**: Токены становятся признаками для ML
- **Языконезависимость**: Базовая токенизация работает на разных языках`,
		},
		uz: {
			title: "So'z tokenizatsiyasi",
			description: `# So'z tokenizatsiyasi

Matnni alohida so'zlarga (tokenlarga) bo'ling.

## Topshiriq

So'z tokenizatsiyasini amalga oshiring:
- Oddiy bo'shliq bo'yicha tokenizatsiya
- Tinish belgilarini to'g'ri boshqarish
- OpenNLP tokenizatoridan foydalanish

## Misol

\`\`\`java
Tokenizer tokenizer = new WordTokenizer();
String[] tokens = tokenizer.tokenize("Hello, world!");
// Result: ["Hello", ",", "world", "!"]
\`\`\``,
			hint1: "So'z belgilarini olish uchun regex \\\\w+ dan foydalaning",
			hint2: "Bo'sh satrlar va null kabi chekka holatlarni boshqaring",
			whyItMatters: `Tokenizatsiya NLP da birinchi qadam:

- **Matndan ma'lumotga**: Tuzilmagan matnni qayta ishlanadigan birliklarga aylantirish
- **Lug'at qurish**: Modellar uchun so'zlar ro'yxatini yaratish
- **Xususiyat ajratib olish**: Tokenlar ML uchun xususiyatlarga aylanadi
- **Tilga bog'liq emas**: Asosiy tokenizatsiya turli tillarda ishlaydi`,
		},
	},
};

export default task;
