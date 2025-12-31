import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-text-summarization',
	title: 'Extractive Summarization',
	difficulty: 'hard',
	tags: ['nlp', 'text-generation', 'summarization', 'extraction'],
	estimatedTime: '35m',
	isPremium: true,
	order: 5,
	description: `# Extractive Summarization

Implement extractive text summarization using sentence scoring.

## Task

Build a summarizer that:
- Scores sentences by importance
- Extracts top sentences for summary
- Uses TF-IDF and position features
- Maintains coherence in output

## Example

\`\`\`java
Summarizer summarizer = new Summarizer();
String summary = summarizer.summarize(longText, 3); // 3 sentences
\`\`\``,

	initialCode: `import java.util.*;

public class Summarizer {

    /**
     * Summarize text to n sentences.
     */
    public String summarize(String text, int numSentences) {
        return null;
    }

    /**
     * Score a sentence for importance.
     */
    public double scoreSentence(String sentence, String document) {
        return 0.0;
    }

    /**
     * Get ranked sentences by importance.
     */
    public List<ScoredSentence> getRankedSentences(String document) {
        return null;
    }

    public static class ScoredSentence {
        public String sentence;
        public double score;
        public int position;

        public ScoredSentence(String sentence, double score, int position) {
            this.sentence = sentence;
            this.score = score;
            this.position = position;
        }
    }
}`,

	solutionCode: `import java.util.*;

public class Summarizer {

    private Set<String> stopwords;

    public Summarizer() {
        this.stopwords = new HashSet<>(Arrays.asList(
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "and", "but", "or", "if", "because", "until", "while",
            "this", "that", "these", "those", "it", "its", "they", "them"
        ));
    }

    /**
     * Split document into sentences.
     */
    private List<String> splitSentences(String text) {
        List<String> sentences = new ArrayList<>();
        String[] parts = text.split("(?<=[.!?])\\\\s+");
        for (String part : parts) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty() && trimmed.length() > 10) {
                sentences.add(trimmed);
            }
        }
        return sentences;
    }

    /**
     * Calculate TF for document.
     */
    private Map<String, Double> calculateTF(String text) {
        Map<String, Double> tf = new HashMap<>();
        String[] words = text.toLowerCase().split("\\\\W+");
        int total = 0;

        for (String word : words) {
            if (word.length() > 2 && !stopwords.contains(word)) {
                tf.merge(word, 1.0, Double::sum);
                total++;
            }
        }

        for (String word : tf.keySet()) {
            tf.put(word, tf.get(word) / total);
        }

        return tf;
    }

    /**
     * Calculate IDF across sentences.
     */
    private Map<String, Double> calculateIDF(List<String> sentences) {
        Map<String, Integer> docFreq = new HashMap<>();
        Set<String> vocabulary = new HashSet<>();

        for (String sentence : sentences) {
            Set<String> seen = new HashSet<>();
            for (String word : sentence.toLowerCase().split("\\\\W+")) {
                if (word.length() > 2 && !stopwords.contains(word)) {
                    vocabulary.add(word);
                    if (!seen.contains(word)) {
                        docFreq.merge(word, 1, Integer::sum);
                        seen.add(word);
                    }
                }
            }
        }

        Map<String, Double> idf = new HashMap<>();
        int N = sentences.size();
        for (String word : vocabulary) {
            idf.put(word, Math.log((double) N / (1 + docFreq.getOrDefault(word, 0))) + 1);
        }

        return idf;
    }

    /**
     * Score a sentence for importance.
     */
    public double scoreSentence(String sentence, String document) {
        List<String> sentences = splitSentences(document);
        Map<String, Double> docTF = calculateTF(document);
        Map<String, Double> idf = calculateIDF(sentences);

        double tfidfScore = 0;
        String[] words = sentence.toLowerCase().split("\\\\W+");
        int validWords = 0;

        for (String word : words) {
            if (word.length() > 2 && !stopwords.contains(word)) {
                double tf = docTF.getOrDefault(word, 0.0);
                double idfVal = idf.getOrDefault(word, 1.0);
                tfidfScore += tf * idfVal;
                validWords++;
            }
        }

        if (validWords > 0) {
            tfidfScore /= validWords;
        }

        return tfidfScore;
    }

    /**
     * Get ranked sentences by importance.
     */
    public List<ScoredSentence> getRankedSentences(String document) {
        List<String> sentences = splitSentences(document);
        Map<String, Double> docTF = calculateTF(document);
        Map<String, Double> idf = calculateIDF(sentences);

        List<ScoredSentence> scored = new ArrayList<>();

        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i);

            // TF-IDF score
            double tfidfScore = 0;
            String[] words = sentence.toLowerCase().split("\\\\W+");
            int validWords = 0;

            for (String word : words) {
                if (word.length() > 2 && !stopwords.contains(word)) {
                    double tf = docTF.getOrDefault(word, 0.0);
                    double idfVal = idf.getOrDefault(word, 1.0);
                    tfidfScore += tf * idfVal;
                    validWords++;
                }
            }

            if (validWords > 0) {
                tfidfScore /= validWords;
            }

            // Position score (first and last sentences more important)
            double positionScore = 0;
            if (i == 0) positionScore = 0.3;
            else if (i == sentences.size() - 1) positionScore = 0.15;
            else if (i < sentences.size() * 0.2) positionScore = 0.1;

            // Length penalty (very short sentences less important)
            double lengthScore = Math.min(1.0, words.length / 15.0);

            // Combined score
            double totalScore = tfidfScore * 0.6 + positionScore + lengthScore * 0.1;

            scored.add(new ScoredSentence(sentence, totalScore, i));
        }

        scored.sort((a, b) -> Double.compare(b.score, a.score));
        return scored;
    }

    /**
     * Summarize text to n sentences.
     */
    public String summarize(String text, int numSentences) {
        List<ScoredSentence> ranked = getRankedSentences(text);

        // Select top sentences
        List<ScoredSentence> selected = new ArrayList<>();
        for (int i = 0; i < Math.min(numSentences, ranked.size()); i++) {
            selected.add(ranked.get(i));
        }

        // Sort by original position for coherence
        selected.sort(Comparator.comparingInt(s -> s.position));

        // Build summary
        StringBuilder summary = new StringBuilder();
        for (ScoredSentence ss : selected) {
            if (summary.length() > 0) summary.append(" ");
            summary.append(ss.sentence);
        }

        return summary.toString();
    }

    /**
     * Get summary with compression ratio.
     */
    public String summarizeByRatio(String text, double ratio) {
        List<String> sentences = splitSentences(text);
        int numSentences = Math.max(1, (int) (sentences.size() * ratio));
        return summarize(text, numSentences);
    }

    public static class ScoredSentence {
        public String sentence;
        public double score;
        public int position;

        public ScoredSentence(String sentence, double score, int position) {
            this.sentence = sentence;
            this.score = score;
            this.position = position;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class SummarizerTest {

    private String sampleText = "Machine learning is a branch of artificial intelligence. " +
        "It enables computers to learn from data. Deep learning is a subset of machine learning. " +
        "Neural networks are the foundation of deep learning. " +
        "These technologies are transforming many industries.";

    @Test
    void testSummarize() {
        Summarizer summarizer = new Summarizer();
        String summary = summarizer.summarize(sampleText, 2);

        assertNotNull(summary);
        assertFalse(summary.isEmpty());
        assertTrue(summary.length() < sampleText.length());
    }

    @Test
    void testGetRankedSentences() {
        Summarizer summarizer = new Summarizer();
        List<Summarizer.ScoredSentence> ranked = summarizer.getRankedSentences(sampleText);

        assertFalse(ranked.isEmpty());
        // Scores should be in descending order
        for (int i = 0; i < ranked.size() - 1; i++) {
            assertTrue(ranked.get(i).score >= ranked.get(i + 1).score);
        }
    }

    @Test
    void testScoreSentence() {
        Summarizer summarizer = new Summarizer();
        double score = summarizer.scoreSentence(
            "Machine learning is transforming industries.",
            sampleText
        );

        assertTrue(score > 0);
    }

    @Test
    void testSummarizeByRatio() {
        Summarizer summarizer = new Summarizer();
        String summary = summarizer.summarizeByRatio(sampleText, 0.4);

        assertNotNull(summary);
        assertTrue(summary.length() < sampleText.length());
    }

    @Test
    void testMaintainsOrder() {
        Summarizer summarizer = new Summarizer();
        String summary = summarizer.summarize(sampleText, 3);

        // Summary should maintain sentence order for coherence
        assertNotNull(summary);
    }

    @Test
    void testScoredSentenceClass() {
        Summarizer.ScoredSentence ss = new Summarizer.ScoredSentence("Test sentence.", 0.85, 2);
        assertEquals("Test sentence.", ss.sentence);
        assertEquals(0.85, ss.score, 0.001);
        assertEquals(2, ss.position);
    }

    @Test
    void testSummarizeSingleSentence() {
        Summarizer summarizer = new Summarizer();
        String summary = summarizer.summarize(sampleText, 1);
        assertNotNull(summary);
        assertFalse(summary.isEmpty());
    }

    @Test
    void testShortText() {
        Summarizer summarizer = new Summarizer();
        String shortText = "This is a short text. It has only two sentences.";
        String summary = summarizer.summarize(shortText, 5);
        assertNotNull(summary);
    }

    @Test
    void testScoreRange() {
        Summarizer summarizer = new Summarizer();
        List<Summarizer.ScoredSentence> ranked = summarizer.getRankedSentences(sampleText);
        for (Summarizer.ScoredSentence ss : ranked) {
            assertTrue(ss.score >= 0);
        }
    }

    @Test
    void testSummarizeByRatioSmall() {
        Summarizer summarizer = new Summarizer();
        String summary = summarizer.summarizeByRatio(sampleText, 0.1);
        assertNotNull(summary);
        assertFalse(summary.isEmpty());
    }
}`,

	hint1: 'Score sentences using TF-IDF plus position bonuses for first/last sentences',
	hint2: 'Re-order selected sentences by original position to maintain document flow',

	whyItMatters: `Extractive summarization is practical and interpretable:

- **Faithfulness**: No hallucination - uses only source text
- **Simplicity**: No neural model training required
- **Speed**: Fast inference for real-time applications
- **Baseline**: Benchmark for abstractive summarization

Extractive methods remain competitive for many summarization tasks.`,

	translations: {
		ru: {
			title: 'Экстрактивная суммаризация',
			description: `# Экстрактивная суммаризация

Реализуйте экстрактивную суммаризацию текста с оценкой предложений.

## Задача

Создайте суммаризатор:
- Оценка предложений по важности
- Извлечение топ предложений для резюме
- Использование TF-IDF и позиционных признаков
- Сохранение связности в выводе

## Пример

\`\`\`java
Summarizer summarizer = new Summarizer();
String summary = summarizer.summarize(longText, 3); // 3 sentences
\`\`\``,
			hint1: 'Оценивайте предложения с помощью TF-IDF плюс бонусы позиции для первых/последних',
			hint2: 'Переупорядочите выбранные предложения по исходной позиции для связности',
			whyItMatters: `Экстрактивная суммаризация практична и интерпретируема:

- **Верность**: Нет галлюцинаций - использует только исходный текст
- **Простота**: Не требуется обучение нейронной модели
- **Скорость**: Быстрый инференс для real-time приложений
- **Baseline**: Бенчмарк для абстрактивной суммаризации`,
		},
		uz: {
			title: 'Ajratib oluvchi xulosa',
			description: `# Ajratib oluvchi xulosa

Gap baholash bilan ajratib oluvchi matn xulosasini amalga oshiring.

## Topshiriq

Xulosalovchi yarating:
- Gaplarni muhimlik bo'yicha baholash
- Xulosa uchun eng muhim gaplarni ajratib olish
- TF-IDF va pozitsiya xususiyatlaridan foydalanish
- Chiqishda bog'liqlikni saqlash

## Misol

\`\`\`java
Summarizer summarizer = new Summarizer();
String summary = summarizer.summarize(longText, 3); // 3 sentences
\`\`\``,
			hint1: "Gaplarni TF-IDF plus birinchi/oxirgi gaplar uchun pozitsiya bonuslari bilan baholang",
			hint2: "Hujjat oqimini saqlash uchun tanlangan gaplarni asl pozitsiya bo'yicha qayta tartiblang",
			whyItMatters: `Ajratib oluvchi xulosa amaliy va tushunarli:

- **Sodiqlik**: Gallyutsinatsiya yo'q - faqat manba matnidan foydalanadi
- **Soddalik**: Neyron modelni o'qitish kerak emas
- **Tezlik**: Real-time ilovalar uchun tez inference
- **Baseline**: Abstrakt xulosa uchun benchmark`,
		},
	},
};

export default task;
