import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-lexicon-sentiment',
	title: 'Lexicon-Based Sentiment',
	difficulty: 'easy',
	tags: ['nlp', 'sentiment', 'lexicon'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Lexicon-Based Sentiment Analysis

Use word sentiment lexicons to analyze text sentiment.

## Task

Implement lexicon-based sentiment:
- Build sentiment lexicon
- Calculate sentiment scores
- Handle negation

## Example

\`\`\`java
SentimentAnalyzer analyzer = new SentimentAnalyzer();
double score = analyzer.analyze("This movie is great!");
// Result: positive score (> 0)
\`\`\``,

	initialCode: `import java.util.*;

public class LexiconSentiment {

    private Map<String, Double> lexicon;

    /**
     */
    public LexiconSentiment() {
    }

    /**
     */
    public double getWordScore(String word) {
        return 0.0;
    }

    /**
     */
    public double analyze(String text) {
        return 0.0;
    }

    /**
     */
    public String classify(String text) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class LexiconSentiment {

    private Map<String, Double> lexicon;
    private Set<String> negations;

    /**
     * Initialize with sentiment words.
     */
    public LexiconSentiment() {
        lexicon = new HashMap<>();

        // Positive words
        lexicon.put("good", 0.7);
        lexicon.put("great", 0.9);
        lexicon.put("excellent", 1.0);
        lexicon.put("amazing", 0.95);
        lexicon.put("wonderful", 0.9);
        lexicon.put("fantastic", 0.95);
        lexicon.put("love", 0.8);
        lexicon.put("like", 0.5);
        lexicon.put("happy", 0.8);
        lexicon.put("best", 0.9);
        lexicon.put("beautiful", 0.7);

        // Negative words
        lexicon.put("bad", -0.7);
        lexicon.put("terrible", -0.9);
        lexicon.put("awful", -0.95);
        lexicon.put("horrible", -0.9);
        lexicon.put("hate", -0.8);
        lexicon.put("dislike", -0.5);
        lexicon.put("sad", -0.6);
        lexicon.put("worst", -0.9);
        lexicon.put("ugly", -0.6);
        lexicon.put("boring", -0.5);

        // Negation words
        negations = new HashSet<>(Arrays.asList(
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "couldn't", "shouldn't"
        ));
    }

    /**
     * Get sentiment score for a word.
     */
    public double getWordScore(String word) {
        return lexicon.getOrDefault(word.toLowerCase(), 0.0);
    }

    /**
     * Analyze sentiment of text.
     */
    public double analyze(String text) {
        String[] words = text.toLowerCase()
            .replaceAll("[^a-zA-Z'\\\\s]", "")
            .split("\\\\s+");

        double score = 0.0;
        boolean negated = false;

        for (int i = 0; i < words.length; i++) {
            String word = words[i];

            if (negations.contains(word)) {
                negated = true;
                continue;
            }

            double wordScore = getWordScore(word);
            if (negated && wordScore != 0) {
                wordScore = -wordScore * 0.5; // Flip and reduce intensity
                negated = false;
            }

            score += wordScore;
        }

        return score;
    }

    /**
     * Classify as positive, negative, or neutral.
     */
    public String classify(String text) {
        double score = analyze(text);
        if (score > 0.1) return "positive";
        if (score < -0.1) return "negative";
        return "neutral";
    }

    /**
     * Get normalized score between -1 and 1.
     */
    public double normalizedScore(String text) {
        double score = analyze(text);
        return Math.max(-1, Math.min(1, score / 3));
    }

    /**
     * Add custom word to lexicon.
     */
    public void addWord(String word, double score) {
        lexicon.put(word.toLowerCase(), score);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LexiconSentimentTest {

    @Test
    void testPositiveSentiment() {
        LexiconSentiment analyzer = new LexiconSentiment();
        double score = analyzer.analyze("This movie is great and amazing!");
        assertTrue(score > 0);
    }

    @Test
    void testNegativeSentiment() {
        LexiconSentiment analyzer = new LexiconSentiment();
        double score = analyzer.analyze("This movie is terrible and awful.");
        assertTrue(score < 0);
    }

    @Test
    void testNegation() {
        LexiconSentiment analyzer = new LexiconSentiment();
        double positive = analyzer.analyze("This is good");
        double negated = analyzer.analyze("This is not good");

        assertTrue(positive > 0);
        assertTrue(negated < positive);
    }

    @Test
    void testClassify() {
        LexiconSentiment analyzer = new LexiconSentiment();
        assertEquals("positive", analyzer.classify("great movie"));
        assertEquals("negative", analyzer.classify("terrible movie"));
    }

    @Test
    void testGetWordScore() {
        LexiconSentiment analyzer = new LexiconSentiment();
        assertTrue(analyzer.getWordScore("great") > 0);
        assertTrue(analyzer.getWordScore("terrible") < 0);
        assertEquals(0.0, analyzer.getWordScore("neutral"), 0.001);
    }

    @Test
    void testAddWord() {
        LexiconSentiment analyzer = new LexiconSentiment();
        analyzer.addWord("superb", 0.95);
        assertTrue(analyzer.getWordScore("superb") > 0.9);
    }

    @Test
    void testNormalizedScore() {
        LexiconSentiment analyzer = new LexiconSentiment();
        double score = analyzer.normalizedScore("great");
        assertTrue(score >= -1 && score <= 1);
    }

    @Test
    void testClassifyNeutral() {
        LexiconSentiment analyzer = new LexiconSentiment();
        assertEquals("neutral", analyzer.classify("the"));
    }

    @Test
    void testEmptyText() {
        LexiconSentiment analyzer = new LexiconSentiment();
        assertEquals(0.0, analyzer.analyze(""), 0.001);
    }

    @Test
    void testMixedSentiment() {
        LexiconSentiment analyzer = new LexiconSentiment();
        double mixed = analyzer.analyze("great but also terrible");
        // Result depends on word weights
        assertNotNull(mixed);
    }
}`,

	hint1: 'Build a dictionary mapping words to sentiment scores (-1 to 1)',
	hint2: 'Handle negation by flipping the sign of the following sentiment word',

	whyItMatters: `Lexicon-based sentiment is fast and interpretable:

- **No training needed**: Works immediately with a good lexicon
- **Interpretable**: Can explain why a text is positive/negative
- **Domain adaptable**: Customize lexicon for specific domains
- **Baseline**: Good baseline before trying ML approaches

Still used in production for simple sentiment tasks.`,

	translations: {
		ru: {
			title: 'Лексиконный анализ тональности',
			description: `# Лексиконный анализ тональности

Используйте лексиконы тональности слов для анализа тональности текста.

## Задача

Реализуйте лексиконный анализ:
- Создание лексикона тональности
- Вычисление оценок тональности
- Обработка отрицаний

## Пример

\`\`\`java
SentimentAnalyzer analyzer = new SentimentAnalyzer();
double score = analyzer.analyze("This movie is great!");
// Result: positive score (> 0)
\`\`\``,
			hint1: 'Создайте словарь сопоставляющий слова с оценками тональности (-1 до 1)',
			hint2: 'Обрабатывайте отрицания меняя знак следующего тонального слова',
			whyItMatters: `Лексиконный анализ быстрый и интерпретируемый:

- **Не требует обучения**: Работает сразу с хорошим лексиконом
- **Интерпретируемый**: Можно объяснить почему текст положительный/отрицательный
- **Адаптируемый**: Настройка лексикона для конкретных доменов
- **Baseline**: Хорошая база перед попыткой ML подходов`,
		},
		uz: {
			title: 'Leksikonga asoslangan sentiment',
			description: `# Leksikonga asoslangan sentiment tahlili

Matn sentimentini tahlil qilish uchun so'z sentiment leksikonlaridan foydalaning.

## Topshiriq

Leksikonga asoslangan sentimentni amalga oshiring:
- Sentiment leksikonini qurish
- Sentiment ballarini hisoblash
- Inkorni boshqarish

## Misol

\`\`\`java
SentimentAnalyzer analyzer = new SentimentAnalyzer();
double score = analyzer.analyze("This movie is great!");
// Result: positive score (> 0)
\`\`\``,
			hint1: "So'zlarni sentiment ballariga (-1 dan 1 gacha) moslashtiruvchi lug'at yarating",
			hint2: "Inkorni keyingi sentiment so'zining belgisini o'zgartirish orqali boshqaring",
			whyItMatters: `Leksikonga asoslangan sentiment tez va tushunarli:

- **O'qitish kerak emas**: Yaxshi leksikon bilan darhol ishlaydi
- **Tushunarli**: Matn nima uchun ijobiy/salbiy ekanligini tushuntirish mumkin
- **Domenga moslashuvchan**: Muayyan domenlar uchun leksikonni sozlash
- **Baseline**: ML yondashuvlaridan oldin yaxshi asos`,
		},
	},
};

export default task;
