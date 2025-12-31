import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-lemmatization',
	title: 'Lemmatization',
	difficulty: 'medium',
	tags: ['nlp', 'preprocessing', 'lemmatization', 'stanford-nlp'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Lemmatization

Convert words to their dictionary base form (lemma).

## Task

Implement lemmatization:
- Use Stanford CoreNLP for lemmatization
- Handle different parts of speech
- Compare with stemming

## Example

\`\`\`java
Lemmatizer lemmatizer = new Lemmatizer();
String lemma = lemmatizer.lemmatize("better");
// Result: "good"
\`\`\``,

	initialCode: `import java.util.*;

public class SimpleLemmatizer {

    private Map<String, String> irregularForms;

    /**
     */
    public SimpleLemmatizer() {
    }

    /**
     */
    public String lemmatize(String word) {
        return null;
    }

    /**
     */
    public String lemmatize(String word, String pos) {
        return null;
    }

    /**
     */
    public List<String> lemmatizeAll(List<String> words) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;
import java.util.stream.Collectors;

public class SimpleLemmatizer {

    private Map<String, String> irregularForms;
    private Map<String, Map<String, String>> posForms;

    /**
     * Initialize with common irregular forms.
     */
    public SimpleLemmatizer() {
        irregularForms = new HashMap<>();
        // Irregular verbs
        irregularForms.put("was", "be");
        irregularForms.put("were", "be");
        irregularForms.put("been", "be");
        irregularForms.put("am", "be");
        irregularForms.put("is", "be");
        irregularForms.put("are", "be");
        irregularForms.put("had", "have");
        irregularForms.put("has", "have");
        irregularForms.put("did", "do");
        irregularForms.put("does", "do");
        irregularForms.put("went", "go");
        irregularForms.put("gone", "go");
        irregularForms.put("came", "come");
        irregularForms.put("ran", "run");
        irregularForms.put("saw", "see");
        irregularForms.put("seen", "see");
        irregularForms.put("took", "take");
        irregularForms.put("taken", "take");

        // Irregular adjectives
        irregularForms.put("better", "good");
        irregularForms.put("best", "good");
        irregularForms.put("worse", "bad");
        irregularForms.put("worst", "bad");

        // Irregular nouns
        irregularForms.put("children", "child");
        irregularForms.put("men", "man");
        irregularForms.put("women", "woman");
        irregularForms.put("feet", "foot");
        irregularForms.put("teeth", "tooth");
        irregularForms.put("mice", "mouse");

        posForms = new HashMap<>();
    }

    /**
     * Get lemma for a word.
     */
    public String lemmatize(String word) {
        if (word == null) return null;
        String lower = word.toLowerCase();

        // Check irregular forms first
        if (irregularForms.containsKey(lower)) {
            return irregularForms.get(lower);
        }

        // Apply regular rules
        return applyRules(lower);
    }

    private String applyRules(String word) {
        // Verb endings
        if (word.endsWith("ing")) {
            String base = word.substring(0, word.length() - 3);
            if (base.length() > 2) {
                // running -> run (double consonant)
                if (base.length() > 1 &&
                    base.charAt(base.length() - 1) == base.charAt(base.length() - 2)) {
                    return base.substring(0, base.length() - 1);
                }
                return base;
            }
        }

        if (word.endsWith("ed")) {
            String base = word.substring(0, word.length() - 2);
            if (base.length() >= 2) {
                return base;
            }
        }

        if (word.endsWith("ies")) {
            return word.substring(0, word.length() - 3) + "y";
        }

        if (word.endsWith("es")) {
            return word.substring(0, word.length() - 2);
        }

        if (word.endsWith("s") && !word.endsWith("ss")) {
            return word.substring(0, word.length() - 1);
        }

        return word;
    }

    /**
     * Lemmatize with part of speech hint.
     */
    public String lemmatize(String word, String pos) {
        // POS can help disambiguate (e.g., "saw" as noun vs verb)
        return lemmatize(word);
    }

    /**
     * Lemmatize all words in a list.
     */
    public List<String> lemmatizeAll(List<String> words) {
        if (words == null) return Collections.emptyList();
        return words.stream()
            .map(this::lemmatize)
            .collect(Collectors.toList());
    }

    /**
     * Add custom irregular form.
     */
    public void addIrregularForm(String word, String lemma) {
        irregularForms.put(word.toLowerCase(), lemma.toLowerCase());
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleLemmatizerTest {

    @Test
    void testLemmatizeIrregular() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("be", lemmatizer.lemmatize("was"));
        assertEquals("be", lemmatizer.lemmatize("were"));
        assertEquals("go", lemmatizer.lemmatize("went"));
        assertEquals("good", lemmatizer.lemmatize("better"));
    }

    @Test
    void testLemmatizeRegular() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("play", lemmatizer.lemmatize("played"));
        assertEquals("cat", lemmatizer.lemmatize("cats"));
    }

    @Test
    void testLemmatizeAll() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        List<String> words = Arrays.asList("was", "running", "better");
        List<String> lemmas = lemmatizer.lemmatizeAll(words);

        assertEquals("be", lemmas.get(0));
        assertEquals("good", lemmas.get(2));
    }

    @Test
    void testAddCustomForm() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        lemmatizer.addIrregularForm("customword", "base");
        assertEquals("base", lemmatizer.lemmatize("customword"));
    }

    @Test
    void testLemmatizeNull() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertNull(lemmatizer.lemmatize(null));
    }

    @Test
    void testLemmatizeAllNull() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        List<String> result = lemmatizer.lemmatizeAll(null);
        assertTrue(result.isEmpty());
    }

    @Test
    void testLemmatizeWithPos() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("be", lemmatizer.lemmatize("was", "VB"));
    }

    @Test
    void testIrregularNouns() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("child", lemmatizer.lemmatize("children"));
        assertEquals("man", lemmatizer.lemmatize("men"));
    }

    @Test
    void testIrregularVerbs() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("see", lemmatizer.lemmatize("saw"));
        assertEquals("take", lemmatizer.lemmatize("taken"));
    }

    @Test
    void testIrregularAdjectives() {
        SimpleLemmatizer lemmatizer = new SimpleLemmatizer();
        assertEquals("bad", lemmatizer.lemmatize("worse"));
        assertEquals("bad", lemmatizer.lemmatize("worst"));
    }
}`,

	hint1: 'Check irregular forms dictionary before applying rules',
	hint2: 'Lemmatization is more accurate than stemming but requires a dictionary',

	whyItMatters: `Lemmatization provides accurate base forms:

- **Dictionary-based**: "better" becomes "good" (not "bett")
- **Context-aware**: Can use POS tags for disambiguation
- **More accurate**: Produces real words unlike stemming
- **Better semantics**: Preserves word meaning

Essential for sentiment analysis and semantic understanding.`,

	translations: {
		ru: {
			title: 'Лемматизация',
			description: `# Лемматизация

Преобразуйте слова в их словарную базовую форму (лемму).

## Задача

Реализуйте лемматизацию:
- Использование Stanford CoreNLP для лемматизации
- Обработка разных частей речи
- Сравнение со стеммингом

## Пример

\`\`\`java
Lemmatizer lemmatizer = new Lemmatizer();
String lemma = lemmatizer.lemmatize("better");
// Result: "good"
\`\`\``,
			hint1: 'Проверяйте словарь неправильных форм перед применением правил',
			hint2: 'Лемматизация точнее стемминга, но требует словарь',
			whyItMatters: `Лемматизация дает точные базовые формы:

- **На основе словаря**: "better" становится "good" (не "bett")
- **Учет контекста**: Может использовать POS теги для разрешения неоднозначности
- **Более точная**: Производит реальные слова в отличие от стемминга
- **Лучшая семантика**: Сохраняет значение слова`,
		},
		uz: {
			title: 'Lemmatizatsiya',
			description: `# Lemmatizatsiya

So'zlarni lug'atdagi asosiy shakliga (lemma) aylantiring.

## Topshiriq

Lemmatizatsiyani amalga oshiring:
- Lemmatizatsiya uchun Stanford CoreNLP dan foydalanish
- Turli nutq qismlarini boshqarish
- Stemming bilan taqqoslash

## Misol

\`\`\`java
Lemmatizer lemmatizer = new Lemmatizer();
String lemma = lemmatizer.lemmatize("better");
// Result: "good"
\`\`\``,
			hint1: "Qoidalarni qo'llashdan oldin noregular shakllar lug'atini tekshiring",
			hint2: "Lemmatizatsiya stemmingdan aniqroq, lekin lug'at talab qiladi",
			whyItMatters: `Lemmatizatsiya aniq asosiy shakllarni beradi:

- **Lug'atga asoslangan**: "better" "good" bo'ladi ("bett" emas)
- **Kontekstni hisobga oladi**: Noaniqlikni hal qilish uchun POS teglaridan foydalanishi mumkin
- **Aniqroq**: Stemmingdan farqli ravishda haqiqiy so'zlarni ishlab chiqaradi
- **Yaxshiroq semantika**: So'z ma'nosini saqlaydi`,
		},
	},
};

export default task;
