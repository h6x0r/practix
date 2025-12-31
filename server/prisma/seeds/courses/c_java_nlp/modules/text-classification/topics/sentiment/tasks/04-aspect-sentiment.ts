import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-aspect-sentiment',
	title: 'Aspect-Based Sentiment',
	difficulty: 'hard',
	tags: ['nlp', 'sentiment', 'aspect-based', 'fine-grained'],
	estimatedTime: '30m',
	isPremium: true,
	order: 4,
	description: `# Aspect-Based Sentiment Analysis

Implement sentiment analysis for specific aspects of a product or service.

## Task

Build an aspect-based sentiment analyzer that:
- Extracts aspect terms from reviews
- Determines sentiment for each aspect
- Handles multiple aspects per sentence
- Returns aspect-sentiment pairs

## Example

\`\`\`java
AspectSentiment analyzer = new AspectSentiment();
List<AspectOpinion> opinions = analyzer.analyze(
    "The food was excellent but the service was slow."
);
// [food: positive, service: negative]
\`\`\``,

	initialCode: `import java.util.*;

public class AspectSentiment {

    /**
     * Analyze aspect-based sentiment.
     */
    public List<AspectOpinion> analyze(String text) {
        return null;
    }

    /**
     * Extract aspects from text.
     */
    public List<String> extractAspects(String text) {
        return null;
    }

    /**
     * Get sentiment for specific aspect.
     */
    public String getAspectSentiment(String text, String aspect) {
        return null;
    }

    public static class AspectOpinion {
        public String aspect;
        public String sentiment;
        public double confidence;

        public AspectOpinion(String aspect, String sentiment, double confidence) {
            this.aspect = aspect;
            this.sentiment = sentiment;
            this.confidence = confidence;
        }
    }
}`,

	solutionCode: `import java.util.*;
import java.util.regex.*;

public class AspectSentiment {

    private Set<String> aspectKeywords;
    private Map<String, Double> sentimentLexicon;
    private Set<String> negations;
    private Set<String> intensifiers;

    public AspectSentiment() {
        initializeAspects();
        initializeLexicon();
    }

    private void initializeAspects() {
        aspectKeywords = new HashSet<>(Arrays.asList(
            // Restaurant aspects
            "food", "service", "price", "ambiance", "atmosphere",
            "staff", "waiter", "waitress", "menu", "drinks",
            "dessert", "appetizer", "portion", "taste", "quality",
            // Product aspects
            "battery", "screen", "camera", "performance", "design",
            "build", "value", "features", "sound", "display",
            // Hotel aspects
            "room", "location", "breakfast", "pool", "wifi",
            "cleanliness", "comfort", "view", "bed", "bathroom"
        ));
    }

    private void initializeLexicon() {
        sentimentLexicon = new HashMap<>();

        // Positive words
        String[] positive = {"excellent", "great", "good", "amazing", "wonderful",
            "fantastic", "perfect", "delicious", "friendly", "fast",
            "beautiful", "comfortable", "clean", "fresh", "nice",
            "love", "loved", "best", "recommend", "impressed"};
        for (String word : positive) {
            sentimentLexicon.put(word, 1.0);
        }

        // Negative words
        String[] negative = {"bad", "terrible", "awful", "horrible", "poor",
            "slow", "rude", "dirty", "cold", "expensive",
            "disappointing", "worst", "hate", "hated", "overpriced",
            "small", "noisy", "uncomfortable", "stale", "mediocre"};
        for (String word : negative) {
            sentimentLexicon.put(word, -1.0);
        }

        negations = new HashSet<>(Arrays.asList(
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "hardly", "barely", "scarcely", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "couldn't", "shouldn't"
        ));

        intensifiers = new HashSet<>(Arrays.asList(
            "very", "really", "extremely", "absolutely", "quite",
            "so", "too", "incredibly", "exceptionally", "highly"
        ));
    }

    /**
     * Extract aspects from text.
     */
    public List<String> extractAspects(String text) {
        List<String> aspects = new ArrayList<>();
        String[] words = text.toLowerCase().split("\\\\W+");

        for (String word : words) {
            if (aspectKeywords.contains(word)) {
                aspects.add(word);
            }
        }

        return aspects;
    }

    /**
     * Get sentiment for specific aspect.
     */
    public String getAspectSentiment(String text, String aspect) {
        // Find the clause containing the aspect
        String[] clauses = text.split("[,;]");
        String relevantClause = text;

        for (String clause : clauses) {
            if (clause.toLowerCase().contains(aspect.toLowerCase())) {
                relevantClause = clause;
                break;
            }
        }

        // Calculate sentiment score for the clause
        String[] words = relevantClause.toLowerCase().split("\\\\W+");
        double score = 0;
        boolean negated = false;
        double intensifier = 1.0;

        for (int i = 0; i < words.length; i++) {
            String word = words[i];

            if (negations.contains(word)) {
                negated = true;
                continue;
            }

            if (intensifiers.contains(word)) {
                intensifier = 1.5;
                continue;
            }

            if (sentimentLexicon.containsKey(word)) {
                double wordScore = sentimentLexicon.get(word) * intensifier;
                if (negated) {
                    wordScore = -wordScore;
                }
                score += wordScore;
                negated = false;
                intensifier = 1.0;
            }
        }

        if (score > 0.3) return "positive";
        if (score < -0.3) return "negative";
        return "neutral";
    }

    /**
     * Analyze aspect-based sentiment.
     */
    public List<AspectOpinion> analyze(String text) {
        List<AspectOpinion> opinions = new ArrayList<>();
        List<String> aspects = extractAspects(text);

        for (String aspect : aspects) {
            String sentiment = getAspectSentiment(text, aspect);

            // Calculate confidence based on sentiment word presence
            double confidence = calculateConfidence(text, aspect);

            opinions.add(new AspectOpinion(aspect, sentiment, confidence));
        }

        return opinions;
    }

    private double calculateConfidence(String text, String aspect) {
        // Find relevant clause
        String[] clauses = text.split("[,;.]");
        for (String clause : clauses) {
            if (clause.toLowerCase().contains(aspect.toLowerCase())) {
                String[] words = clause.toLowerCase().split("\\\\W+");
                int sentimentWords = 0;
                for (String word : words) {
                    if (sentimentLexicon.containsKey(word)) {
                        sentimentWords++;
                    }
                }
                return Math.min(0.5 + sentimentWords * 0.15, 0.95);
            }
        }
        return 0.5;
    }

    /**
     * Get aspect summary from multiple reviews.
     */
    public Map<String, Map<String, Integer>> summarizeAspects(List<String> reviews) {
        Map<String, Map<String, Integer>> summary = new HashMap<>();

        for (String review : reviews) {
            List<AspectOpinion> opinions = analyze(review);
            for (AspectOpinion opinion : opinions) {
                summary.computeIfAbsent(opinion.aspect, k -> new HashMap<>())
                    .merge(opinion.sentiment, 1, Integer::sum);
            }
        }

        return summary;
    }

    public static class AspectOpinion {
        public String aspect;
        public String sentiment;
        public double confidence;

        public AspectOpinion(String aspect, String sentiment, double confidence) {
            this.aspect = aspect;
            this.sentiment = sentiment;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return aspect + ": " + sentiment + " (" + String.format("%.2f", confidence) + ")";
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class AspectSentimentTest {

    @Test
    void testAnalyze() {
        AspectSentiment analyzer = new AspectSentiment();
        List<AspectSentiment.AspectOpinion> opinions = analyzer.analyze(
            "The food was excellent but the service was slow."
        );

        assertEquals(2, opinions.size());
    }

    @Test
    void testExtractAspects() {
        AspectSentiment analyzer = new AspectSentiment();
        List<String> aspects = analyzer.extractAspects(
            "Great food, friendly staff, but expensive prices."
        );

        assertTrue(aspects.contains("food"));
        assertTrue(aspects.contains("staff"));
    }

    @Test
    void testGetAspectSentiment() {
        AspectSentiment analyzer = new AspectSentiment();

        String positive = analyzer.getAspectSentiment("The food was excellent.", "food");
        assertEquals("positive", positive);

        String negative = analyzer.getAspectSentiment("The service was terrible.", "service");
        assertEquals("negative", negative);
    }

    @Test
    void testNegation() {
        AspectSentiment analyzer = new AspectSentiment();
        String sentiment = analyzer.getAspectSentiment(
            "The food was not good.", "food"
        );
        assertEquals("negative", sentiment);
    }

    @Test
    void testSummarizeAspects() {
        AspectSentiment analyzer = new AspectSentiment();
        List<String> reviews = Arrays.asList(
            "Great food!",
            "The food was excellent.",
            "Terrible service."
        );

        Map<String, Map<String, Integer>> summary = analyzer.summarizeAspects(reviews);
        assertTrue(summary.containsKey("food"));
    }

    @Test
    void testAnalyzeEmptyText() {
        AspectSentiment analyzer = new AspectSentiment();
        List<AspectSentiment.AspectOpinion> opinions = analyzer.analyze("");
        assertTrue(opinions.isEmpty());
    }

    @Test
    void testExtractAspectsNoMatch() {
        AspectSentiment analyzer = new AspectSentiment();
        List<String> aspects = analyzer.extractAspects("Hello world");
        assertTrue(aspects.isEmpty());
    }

    @Test
    void testGetAspectSentimentNeutral() {
        AspectSentiment analyzer = new AspectSentiment();
        String sentiment = analyzer.getAspectSentiment("The food was okay.", "food");
        assertEquals("neutral", sentiment);
    }

    @Test
    void testAspectOpinionToString() {
        AspectSentiment.AspectOpinion opinion = new AspectSentiment.AspectOpinion(
            "food", "positive", 0.85
        );
        String str = opinion.toString();
        assertTrue(str.contains("food"));
        assertTrue(str.contains("positive"));
    }

    @Test
    void testConfidenceRange() {
        AspectSentiment analyzer = new AspectSentiment();
        List<AspectSentiment.AspectOpinion> opinions = analyzer.analyze(
            "The food was excellent and delicious."
        );
        assertFalse(opinions.isEmpty());
        assertTrue(opinions.get(0).confidence >= 0.5);
        assertTrue(opinions.get(0).confidence <= 0.95);
    }
}`,

	hint1: 'Split text into clauses and analyze sentiment only for the clause containing the aspect',
	hint2: 'Handle negation words that flip sentiment polarity',

	whyItMatters: `Aspect-based sentiment provides actionable insights:

- **Fine-grained**: Know exactly what customers like/dislike
- **Actionable**: Identify specific areas for improvement
- **Competitive**: Compare aspect ratings vs. competitors
- **Product development**: Prioritize features based on sentiment

ABSA is essential for review analysis and customer feedback systems.`,

	translations: {
		ru: {
			title: 'Аспектный анализ тональности',
			description: `# Аспектный анализ тональности

Реализуйте анализ тональности для конкретных аспектов продукта или услуги.

## Задача

Создайте аспектный анализатор тональности:
- Извлечение аспектных термов из отзывов
- Определение тональности для каждого аспекта
- Обработка нескольких аспектов в одном предложении
- Возврат пар аспект-тональность

## Пример

\`\`\`java
AspectSentiment analyzer = new AspectSentiment();
List<AspectOpinion> opinions = analyzer.analyze(
    "The food was excellent but the service was slow."
);
// [food: positive, service: negative]
\`\`\``,
			hint1: 'Разделите текст на клаузы и анализируйте тональность только для клаузы с аспектом',
			hint2: 'Обрабатывайте слова отрицания, которые инвертируют полярность',
			whyItMatters: `Аспектный анализ тональности дает actionable инсайты:

- **Детальность**: Точно знать, что нравится/не нравится клиентам
- **Действенность**: Определение конкретных областей для улучшения
- **Конкурентность**: Сравнение рейтингов аспектов с конкурентами
- **Разработка продукта**: Приоритизация функций на основе тональности`,
		},
		uz: {
			title: "Aspektga asoslangan sentiment tahlili",
			description: `# Aspektga asoslangan sentiment tahlili

Mahsulot yoki xizmatning muayyan aspektlari uchun sentiment tahlilini amalga oshiring.

## Topshiriq

Aspektga asoslangan sentiment analizatorini yarating:
- Sharhlardan aspekt so'zlarini ajratib olish
- Har bir aspekt uchun sentimentni aniqlash
- Bir gapda bir nechta aspektlarni qayta ishlash
- Aspekt-sentiment juftlarini qaytarish

## Misol

\`\`\`java
AspectSentiment analyzer = new AspectSentiment();
List<AspectOpinion> opinions = analyzer.analyze(
    "The food was excellent but the service was slow."
);
// [food: positive, service: negative]
\`\`\``,
			hint1: "Matnni klauzalarga ajrating va faqat aspekt mavjud klauza uchun sentimentni tahlil qiling",
			hint2: "Sentiment qutbini o'zgartiruvchi inkor so'zlarini qayta ishlang",
			whyItMatters: `Aspektga asoslangan sentiment amaliy tushunchalarni beradi:

- **Nozik donali**: Mijozlarga nima yoqishi/yoqmasligini aniq bilish
- **Amaliy**: Yaxshilash uchun aniq sohalarni aniqlash
- **Raqobatbardosh**: Aspekt reytinglarini raqiblar bilan solishtirish
- **Mahsulot ishlab chiqish**: Sentiment asosida funksiyalarni birinchi o'ringa qo'yish`,
		},
	},
};

export default task;
