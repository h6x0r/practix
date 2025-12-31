import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-stemming',
	title: 'Stemming',
	difficulty: 'medium',
	tags: ['nlp', 'preprocessing', 'stemming', 'porter'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Stemming

Reduce words to their root form using stemming algorithms.

## Task

Implement stemming operations:
- Porter Stemmer algorithm
- Stem word lists
- Handle edge cases

## Example

\`\`\`java
PorterStemmer stemmer = new PorterStemmer();
String stem = stemmer.stem("running");
// Result: "run"
\`\`\``,

	initialCode: `import java.util.*;

public class SimpleStemmer {

    /**
     */
    public String stem(String word) {
        return null;
    }

    /**
     */
    public List<String> stemAll(List<String> words) {
        return null;
    }

    /**
     */
    private boolean endsWith(String word, String suffix) {
        return false;
    }

    /**
     */
    private String replaceSuffix(String word, String suffix, String replacement) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class SimpleStemmer {

    private static final String[][] SUFFIX_RULES = {
        {"ational", "ate"},
        {"tional", "tion"},
        {"enci", "ence"},
        {"anci", "ance"},
        {"izer", "ize"},
        {"isation", "ize"},
        {"ization", "ize"},
        {"ation", "ate"},
        {"ator", "ate"},
        {"alism", "al"},
        {"iveness", "ive"},
        {"fulness", "ful"},
        {"ousness", "ous"},
        {"aliti", "al"},
        {"iviti", "ive"},
        {"biliti", "ble"},
        {"ness", ""},
        {"ment", ""},
        {"ing", ""},
        {"ies", "y"},
        {"es", "e"},
        {"ed", ""},
        {"s", ""}
    };

    /**
     * Apply basic suffix stripping.
     */
    public String stem(String word) {
        if (word == null || word.length() < 3) return word;

        String result = word.toLowerCase();

        for (String[] rule : SUFFIX_RULES) {
            if (endsWith(result, rule[0])) {
                String stemmed = replaceSuffix(result, rule[0], rule[1]);
                if (stemmed.length() >= 2) {
                    return stemmed;
                }
            }
        }

        return result;
    }

    /**
     * Stem a list of words.
     */
    public List<String> stemAll(List<String> words) {
        if (words == null) return Collections.emptyList();
        return words.stream()
            .map(this::stem)
            .collect(Collectors.toList());
    }

    /**
     * Check if word ends with suffix.
     */
    private boolean endsWith(String word, String suffix) {
        return word != null && word.endsWith(suffix);
    }

    /**
     * Replace suffix with replacement.
     */
    private String replaceSuffix(String word, String suffix, String replacement) {
        if (word == null) return null;
        int index = word.length() - suffix.length();
        return word.substring(0, index) + replacement;
    }

    /**
     * Check if character is a consonant.
     */
    public static boolean isConsonant(String word, int index) {
        char c = word.charAt(index);
        if ("aeiou".indexOf(c) >= 0) return false;
        if (c == 'y') {
            return index == 0 || !isConsonant(word, index - 1);
        }
        return true;
    }

    /**
     * Count consonant-vowel sequences.
     */
    public static int measureWord(String word) {
        int m = 0;
        int n = word.length();
        int i = 0;

        while (i < n && isConsonant(word, i)) i++;

        while (i < n) {
            while (i < n && !isConsonant(word, i)) i++;
            while (i < n && isConsonant(word, i)) i++;
            m++;
        }

        return m;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleStemmerTest {

    @Test
    void testStemBasic() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertEquals("run", stemmer.stem("running"));
        assertEquals("play", stemmer.stem("played"));
        assertEquals("happ", stemmer.stem("happiness"));
    }

    @Test
    void testStemPlural() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertEquals("cat", stemmer.stem("cats"));
        assertEquals("box", stemmer.stem("boxes"));
    }

    @Test
    void testStemAll() {
        SimpleStemmer stemmer = new SimpleStemmer();
        List<String> words = Arrays.asList("running", "plays", "cats");
        List<String> stemmed = stemmer.stemAll(words);

        assertEquals(3, stemmed.size());
        assertTrue(stemmed.contains("run"));
    }

    @Test
    void testShortWords() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertEquals("a", stemmer.stem("a"));
        assertEquals("is", stemmer.stem("is"));
    }

    @Test
    void testIsConsonant() {
        assertTrue(SimpleStemmer.isConsonant("hello", 0)); // h
        assertFalse(SimpleStemmer.isConsonant("hello", 1)); // e
    }

    @Test
    void testStemNull() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertNull(stemmer.stem(null));
    }

    @Test
    void testStemAllNull() {
        SimpleStemmer stemmer = new SimpleStemmer();
        List<String> result = stemmer.stemAll(null);
        assertTrue(result.isEmpty());
    }

    @Test
    void testMeasureWord() {
        assertTrue(SimpleStemmer.measureWord("troubles") > 0);
    }

    @Test
    void testStemIng() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertEquals("walk", stemmer.stem("walking"));
        assertEquals("jump", stemmer.stem("jumping"));
    }

    @Test
    void testStemIes() {
        SimpleStemmer stemmer = new SimpleStemmer();
        assertEquals("happy", stemmer.stem("happies"));
        assertEquals("story", stemmer.stem("stories"));
    }
}`,

	hint1: 'Apply rules in order of suffix length (longest first)',
	hint2: 'Ensure minimum stem length after stripping',

	whyItMatters: `Stemming normalizes word variations:

- **Vocabulary reduction**: "run", "running", "runs" become "run"
- **Better matching**: Find related documents regardless of form
- **Faster search**: Index fewer unique terms
- **Simple algorithm**: Fast and language-agnostic rules

Foundation for search engines and information retrieval.`,

	translations: {
		ru: {
			title: 'Стемминг',
			description: `# Стемминг

Приводите слова к корневой форме используя алгоритмы стемминга.

## Задача

Реализуйте операции стемминга:
- Алгоритм стеммера Портера
- Стемминг списков слов
- Обработка краевых случаев

## Пример

\`\`\`java
PorterStemmer stemmer = new PorterStemmer();
String stem = stemmer.stem("running");
// Result: "run"
\`\`\``,
			hint1: 'Применяйте правила в порядке длины суффикса (сначала длинные)',
			hint2: 'Обеспечьте минимальную длину основы после удаления',
			whyItMatters: `Стемминг нормализует вариации слов:

- **Сокращение словаря**: "run", "running", "runs" становятся "run"
- **Лучшее сопоставление**: Находите связанные документы независимо от формы
- **Быстрый поиск**: Индексируйте меньше уникальных терминов
- **Простой алгоритм**: Быстрые и языконезависимые правила`,
		},
		uz: {
			title: 'Stemming',
			description: `# Stemming

Stemming algoritmlari yordamida so'zlarni ildiz shakliga keltiring.

## Topshiriq

Stemming operatsiyalarini amalga oshiring:
- Porter Stemmer algoritmi
- So'zlar ro'yxatini stem qilish
- Chekka holatlarni boshqarish

## Misol

\`\`\`java
PorterStemmer stemmer = new PorterStemmer();
String stem = stemmer.stem("running");
// Result: "run"
\`\`\``,
			hint1: "Qoidalarni suffiks uzunligi tartibida qo'llang (avval uzunlari)",
			hint2: "Olib tashlagandan keyin minimal stem uzunligini ta'minlang",
			whyItMatters: `Stemming so'z variatsiyalarini normallashtiradi:

- **Lug'at qisqartirish**: "run", "running", "runs" "run" bo'ladi
- **Yaxshiroq moslik**: Shaklidan qat'i nazar tegishli hujjatlarni toping
- **Tezroq qidiruv**: Kamroq noyob atamalarni indekslang
- **Oddiy algoritm**: Tez va tilga bog'liq bo'lmagan qoidalar`,
		},
	},
};

export default task;
