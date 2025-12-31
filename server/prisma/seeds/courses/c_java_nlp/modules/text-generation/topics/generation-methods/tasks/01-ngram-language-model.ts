import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-ngram-language-model',
	title: 'N-gram Language Model',
	difficulty: 'medium',
	tags: ['nlp', 'text-generation', 'ngram', 'language-model'],
	estimatedTime: '30m',
	isPremium: true,
	order: 1,
	description: `# N-gram Language Model

Implement an n-gram language model for text generation.

## Task

Build an n-gram model that:
- Counts n-gram frequencies from training text
- Calculates n-gram probabilities
- Generates text using probabilistic sampling
- Supports different n values (bigram, trigram)

## Example

\`\`\`java
NgramModel model = new NgramModel(3); // trigram
model.train(corpus);
String generated = model.generate(20); // generate 20 words
\`\`\``,

	initialCode: `import java.util.*;

public class NgramModel {

    private int n;

    public NgramModel(int n) {
        this.n = n;
    }

    /**
     */
    public void train(String corpus) {
    }

    /**
     */
    public String generate(int numWords) {
        return null;
    }

    /**
     */
    public double getProbability(String context, String word) {
        return 0.0;
    }

    /**
     */
    public List<String> getNextWordCandidates(String context) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class NgramModel {

    private int n;
    private Map<String, Map<String, Integer>> ngramCounts;
    private Map<String, Integer> contextCounts;
    private List<String> vocabulary;
    private Random random;

    public NgramModel(int n) {
        this.n = n;
        this.ngramCounts = new HashMap<>();
        this.contextCounts = new HashMap<>();
        this.vocabulary = new ArrayList<>();
        this.random = new Random(42);
    }

    /**
     * Tokenize text into words.
     */
    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        // Add start tokens
        for (int i = 0; i < n - 1; i++) {
            tokens.add("<START>");
        }

        String[] words = text.toLowerCase().split("\\\\s+");
        for (String word : words) {
            String cleaned = word.replaceAll("[^a-zA-Z0-9]", "");
            if (!cleaned.isEmpty()) {
                tokens.add(cleaned);
                if (!vocabulary.contains(cleaned)) {
                    vocabulary.add(cleaned);
                }
            }
        }

        tokens.add("<END>");
        return tokens;
    }

    /**
     * Train the model on text corpus.
     */
    public void train(String corpus) {
        List<String> tokens = tokenize(corpus);

        for (int i = 0; i <= tokens.size() - n; i++) {
            // Build context (n-1 tokens)
            StringBuilder contextBuilder = new StringBuilder();
            for (int j = 0; j < n - 1; j++) {
                if (j > 0) contextBuilder.append(" ");
                contextBuilder.append(tokens.get(i + j));
            }
            String context = contextBuilder.toString();

            // Get next word
            String nextWord = tokens.get(i + n - 1);

            // Update counts
            ngramCounts.computeIfAbsent(context, k -> new HashMap<>())
                .merge(nextWord, 1, Integer::sum);
            contextCounts.merge(context, 1, Integer::sum);
        }
    }

    /**
     * Get probability of word given context.
     */
    public double getProbability(String context, String word) {
        int contextCount = contextCounts.getOrDefault(context, 0);
        if (contextCount == 0) {
            return 1.0 / vocabulary.size();  // Uniform fallback
        }

        Map<String, Integer> nextWords = ngramCounts.get(context);
        if (nextWords == null) {
            return 1.0 / vocabulary.size();
        }

        int wordCount = nextWords.getOrDefault(word, 0);
        return (double) wordCount / contextCount;
    }

    /**
     * Get next word predictions.
     */
    public List<String> getNextWordCandidates(String context) {
        List<String> candidates = new ArrayList<>();
        Map<String, Integer> nextWords = ngramCounts.get(context);

        if (nextWords != null) {
            candidates.addAll(nextWords.keySet());
            candidates.sort((a, b) -> nextWords.get(b) - nextWords.get(a));
        }

        return candidates;
    }

    /**
     * Sample next word based on probability distribution.
     */
    private String sampleNextWord(String context) {
        Map<String, Integer> nextWords = ngramCounts.get(context);

        if (nextWords == null || nextWords.isEmpty()) {
            // Random word from vocabulary
            return vocabulary.get(random.nextInt(vocabulary.size()));
        }

        int totalCount = contextCounts.get(context);
        double rand = random.nextDouble() * totalCount;
        double cumulative = 0;

        for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
            cumulative += entry.getValue();
            if (cumulative >= rand) {
                return entry.getKey();
            }
        }

        return vocabulary.get(0);
    }

    /**
     * Generate text with given number of words.
     */
    public String generate(int numWords) {
        List<String> generated = new ArrayList<>();

        // Initialize with start tokens
        List<String> context = new ArrayList<>();
        for (int i = 0; i < n - 1; i++) {
            context.add("<START>");
        }

        for (int i = 0; i < numWords; i++) {
            String contextStr = String.join(" ", context);
            String nextWord = sampleNextWord(contextStr);

            if ("<END>".equals(nextWord)) {
                break;
            }

            generated.add(nextWord);

            // Update context
            context.remove(0);
            context.add(nextWord);
        }

        return String.join(" ", generated);
    }

    /**
     * Calculate perplexity on test text.
     */
    public double perplexity(String testText) {
        List<String> tokens = tokenize(testText);
        double logProb = 0;
        int count = 0;

        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder contextBuilder = new StringBuilder();
            for (int j = 0; j < n - 1; j++) {
                if (j > 0) contextBuilder.append(" ");
                contextBuilder.append(tokens.get(i + j));
            }
            String context = contextBuilder.toString();
            String nextWord = tokens.get(i + n - 1);

            double prob = getProbability(context, nextWord);
            if (prob > 0) {
                logProb += Math.log(prob);
                count++;
            }
        }

        return Math.exp(-logProb / count);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class NgramModelTest {

    @Test
    void testTrain() {
        NgramModel model = new NgramModel(2);  // bigram
        model.train("the cat sat on the mat the cat");

        double prob = model.getProbability("the", "cat");
        assertTrue(prob > 0);
    }

    @Test
    void testGenerate() {
        NgramModel model = new NgramModel(2);
        model.train("hello world hello there hello friend");

        String generated = model.generate(5);
        assertNotNull(generated);
        assertFalse(generated.isEmpty());
    }

    @Test
    void testGetNextWordCandidates() {
        NgramModel model = new NgramModel(2);
        model.train("the cat the dog the bird");

        List<String> candidates = model.getNextWordCandidates("the");
        assertTrue(candidates.size() >= 1);
    }

    @Test
    void testTrigramModel() {
        NgramModel model = new NgramModel(3);
        model.train("I love machine learning and I love deep learning");

        String generated = model.generate(10);
        assertNotNull(generated);
    }

    @Test
    void testPerplexity() {
        NgramModel model = new NgramModel(2);
        model.train("the cat sat on the mat");

        double perplexity = model.perplexity("the cat sat");
        assertTrue(perplexity > 0);
    }

    @Test
    void testGetProbabilityUnseenContext() {
        NgramModel model = new NgramModel(2);
        model.train("hello world");
        double prob = model.getProbability("unseen", "word");
        assertTrue(prob > 0);
    }

    @Test
    void testGetProbabilityRange() {
        NgramModel model = new NgramModel(2);
        model.train("the cat the dog");
        double prob = model.getProbability("the", "cat");
        assertTrue(prob >= 0 && prob <= 1);
    }

    @Test
    void testEmptyGeneration() {
        NgramModel model = new NgramModel(2);
        model.train("hello");
        String generated = model.generate(0);
        assertTrue(generated.isEmpty() || generated.equals(""));
    }

    @Test
    void testGetNextWordCandidatesEmpty() {
        NgramModel model = new NgramModel(2);
        model.train("hello world");
        List<String> candidates = model.getNextWordCandidates("unseen");
        assertTrue(candidates.isEmpty());
    }

    @Test
    void testUnigramModel() {
        NgramModel model = new NgramModel(1);
        model.train("the cat sat");
        String generated = model.generate(3);
        assertNotNull(generated);
    }
}`,

	hint1: 'Store n-gram counts as Map<context, Map<next_word, count>>',
	hint2: 'Sample next word by accumulating probabilities until random threshold is reached',

	whyItMatters: `N-gram models are foundational for NLP:

- **Simplicity**: Easy to understand and implement
- **Interpretability**: Clear probability calculations
- **Baseline**: Benchmark for neural language models
- **Applications**: Spell checking, predictive text, speech recognition

Understanding n-grams helps appreciate neural LM improvements.`,

	translations: {
		ru: {
			title: 'N-граммная языковая модель',
			description: `# N-граммная языковая модель

Реализуйте n-граммную языковую модель для генерации текста.

## Задача

Создайте n-граммную модель:
- Подсчет частот n-грамм из обучающего текста
- Вычисление вероятностей n-грамм
- Генерация текста с вероятностной выборкой
- Поддержка разных значений n (биграмм, триграмм)

## Пример

\`\`\`java
NgramModel model = new NgramModel(3); // trigram
model.train(corpus);
String generated = model.generate(20); // generate 20 words
\`\`\``,
			hint1: 'Храните счетчики n-грамм как Map<контекст, Map<следующее_слово, счет>>',
			hint2: 'Выбирайте следующее слово накапливая вероятности до случайного порога',
			whyItMatters: `N-граммные модели - основа NLP:

- **Простота**: Легко понять и реализовать
- **Интерпретируемость**: Понятные вычисления вероятностей
- **Baseline**: Бенчмарк для нейронных языковых моделей
- **Применения**: Проверка орфографии, предиктивный ввод, распознавание речи`,
		},
		uz: {
			title: 'N-gram til modeli',
			description: `# N-gram til modeli

Matn generatsiyasi uchun n-gram til modelini amalga oshiring.

## Topshiriq

N-gram modelini yarating:
- O'qitish matnidan n-gram chastotalarini hisoblash
- N-gram ehtimolliklarini hisoblash
- Ehtimoliy sampling bilan matn generatsiyasi
- Turli n qiymatlarini qo'llab-quvvatlash (bigram, trigram)

## Misol

\`\`\`java
NgramModel model = new NgramModel(3); // trigram
model.train(corpus);
String generated = model.generate(20); // generate 20 words
\`\`\``,
			hint1: "N-gram sanashlarni Map<kontekst, Map<keyingi_so'z, sanoq>> sifatida saqlang",
			hint2: "Tasodifiy chegaraga yetguncha ehtimolliklarni to'plab keyingi so'zni tanlang",
			whyItMatters: `N-gram modellar NLP uchun asos:

- **Soddalik**: Tushunish va amalga oshirish oson
- **Tushunarlilik**: Aniq ehtimollik hisoblashlari
- **Baseline**: Neyron til modellari uchun benchmark
- **Qo'llanilishi**: Imlo tekshirish, predictive text, nutq tanish`,
		},
	},
};

export default task;
