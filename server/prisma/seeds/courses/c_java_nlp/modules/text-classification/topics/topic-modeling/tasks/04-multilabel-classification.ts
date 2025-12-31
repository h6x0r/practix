import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-multilabel-classification',
	title: 'Multi-Label Classification',
	difficulty: 'hard',
	tags: ['nlp', 'classification', 'multi-label', 'tagging'],
	estimatedTime: '30m',
	isPremium: true,
	order: 4,
	description: `# Multi-Label Classification

Implement multi-label text classification for document tagging.

## Task

Build a multi-label classifier that:
- Assigns multiple labels to a single document
- Uses binary relevance approach
- Calculates per-label confidence scores
- Handles label dependencies

## Example

\`\`\`java
MultiLabelClassifier classifier = new MultiLabelClassifier();
classifier.train(documents, labelSets);
Set<String> labels = classifier.classify("AI news about deep learning in healthcare");
// Returns: {"technology", "health", "ai"}
\`\`\``,

	initialCode: `import java.util.*;

public class MultiLabelClassifier {

    /**
     */
    public void train(List<String> documents, List<Set<String>> labelSets) {
    }

    /**
     */
    public Set<String> classify(String document) {
        return null;
    }

    /**
     */
    public Map<String, Double> getLabelProbabilities(String document) {
        return null;
    }

    /**
     */
    public Set<String> classifyWithThreshold(String document, double threshold) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class MultiLabelClassifier {

    private Set<String> allLabels;
    private Map<String, Map<String, Double>> labelWordScores;
    private Map<String, Double> labelPriors;
    private Set<String> stopwords;

    public MultiLabelClassifier() {
        this.allLabels = new HashSet<>();
        this.labelWordScores = new HashMap<>();
        this.labelPriors = new HashMap<>();
        this.stopwords = new HashSet<>(Arrays.asList(
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "and", "or", "but"
        ));
        initializeDefaultLabels();
    }

    private void initializeDefaultLabels() {
        // Technology keywords
        Map<String, Double> tech = new HashMap<>();
        for (String word : new String[]{"technology", "tech", "software", "hardware",
            "computer", "digital", "app", "programming", "code", "developer"}) {
            tech.put(word, 1.0);
        }
        labelWordScores.put("technology", tech);

        // Health keywords
        Map<String, Double> health = new HashMap<>();
        for (String word : new String[]{"health", "medical", "healthcare", "hospital",
            "doctor", "patient", "disease", "treatment", "medicine", "clinical"}) {
            health.put(word, 1.0);
        }
        labelWordScores.put("health", health);

        // AI keywords
        Map<String, Double> ai = new HashMap<>();
        for (String word : new String[]{"ai", "artificial", "intelligence", "machine",
            "learning", "neural", "deep", "model", "algorithm", "prediction"}) {
            ai.put(word, 1.0);
        }
        labelWordScores.put("ai", ai);

        // Business keywords
        Map<String, Double> business = new HashMap<>();
        for (String word : new String[]{"business", "company", "market", "sales",
            "revenue", "profit", "startup", "investment", "enterprise", "corporate"}) {
            business.put(word, 1.0);
        }
        labelWordScores.put("business", business);

        // Science keywords
        Map<String, Double> science = new HashMap<>();
        for (String word : new String[]{"science", "research", "study", "experiment",
            "scientific", "discovery", "theory", "hypothesis", "analysis", "data"}) {
            science.put(word, 1.0);
        }
        labelWordScores.put("science", science);

        allLabels.addAll(labelWordScores.keySet());
        for (String label : allLabels) {
            labelPriors.put(label, 0.2);  // Default prior
        }
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] words = text.toLowerCase().split("\\\\W+");
        for (String word : words) {
            if (word.length() > 2 && !stopwords.contains(word)) {
                tokens.add(word);
            }
        }
        return tokens;
    }

    /**
     * Train the classifier.
     */
    public void train(List<String> documents, List<Set<String>> labelSets) {
        // Count label occurrences
        Map<String, Integer> labelCounts = new HashMap<>();
        Map<String, Map<String, Integer>> labelWordCounts = new HashMap<>();

        for (int i = 0; i < documents.size(); i++) {
            List<String> tokens = tokenize(documents.get(i));
            Set<String> labels = labelSets.get(i);

            for (String label : labels) {
                allLabels.add(label);
                labelCounts.merge(label, 1, Integer::sum);

                labelWordCounts.computeIfAbsent(label, k -> new HashMap<>());
                for (String token : tokens) {
                    labelWordCounts.get(label).merge(token, 1, Integer::sum);
                }
            }
        }

        // Calculate priors and word scores
        int totalDocs = documents.size();
        for (String label : allLabels) {
            int count = labelCounts.getOrDefault(label, 0);
            labelPriors.put(label, (double) count / totalDocs);

            Map<String, Integer> wordCounts = labelWordCounts.getOrDefault(label, new HashMap<>());
            int totalWords = wordCounts.values().stream().mapToInt(Integer::intValue).sum();

            if (totalWords > 0) {
                Map<String, Double> scores = new HashMap<>();
                for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
                    scores.put(entry.getKey(), (double) entry.getValue() / totalWords);
                }
                labelWordScores.put(label, scores);
            }
        }
    }

    /**
     * Get label probabilities.
     */
    public Map<String, Double> getLabelProbabilities(String document) {
        List<String> tokens = tokenize(document);
        Map<String, Double> probabilities = new HashMap<>();

        for (String label : allLabels) {
            double score = labelPriors.getOrDefault(label, 0.1);
            Map<String, Double> wordScores = labelWordScores.get(label);

            if (wordScores != null) {
                for (String token : tokens) {
                    if (wordScores.containsKey(token)) {
                        score += wordScores.get(token);
                    }
                }
            }

            probabilities.put(label, score);
        }

        // Normalize to probabilities
        double maxScore = probabilities.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        for (String label : probabilities.keySet()) {
            probabilities.put(label, probabilities.get(label) / (maxScore + 1));
        }

        return probabilities;
    }

    /**
     * Classify document with multiple labels.
     */
    public Set<String> classify(String document) {
        return classifyWithThreshold(document, 0.3);
    }

    /**
     * Classify with threshold.
     */
    public Set<String> classifyWithThreshold(String document, double threshold) {
        Map<String, Double> probs = getLabelProbabilities(document);
        Set<String> labels = new HashSet<>();

        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            if (entry.getValue() >= threshold) {
                labels.add(entry.getKey());
            }
        }

        // If no labels pass threshold, return the highest scoring one
        if (labels.isEmpty() && !probs.isEmpty()) {
            String best = probs.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
            if (best != null) {
                labels.add(best);
            }
        }

        return labels;
    }

    /**
     * Get top N labels with scores.
     */
    public List<LabelScore> getTopLabels(String document, int n) {
        Map<String, Double> probs = getLabelProbabilities(document);
        List<LabelScore> sorted = new ArrayList<>();

        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            sorted.add(new LabelScore(entry.getKey(), entry.getValue()));
        }

        sorted.sort((a, b) -> Double.compare(b.score, a.score));
        return sorted.subList(0, Math.min(n, sorted.size()));
    }

    public static class LabelScore {
        public String label;
        public double score;

        public LabelScore(String label, double score) {
            this.label = label;
            this.score = score;
        }
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class MultiLabelClassifierTest {

    @Test
    void testClassify() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Set<String> labels = classifier.classify("AI news about deep learning in healthcare");

        assertTrue(labels.contains("ai") || labels.contains("health") || labels.contains("technology"));
    }

    @Test
    void testMultipleLabels() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Set<String> labels = classifier.classify(
            "Machine learning technology is transforming healthcare and medical research"
        );

        assertTrue(labels.size() >= 2);
    }

    @Test
    void testGetLabelProbabilities() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Map<String, Double> probs = classifier.getLabelProbabilities("technology news");

        assertTrue(probs.containsKey("technology"));
        assertTrue(probs.get("technology") > 0);
    }

    @Test
    void testClassifyWithThreshold() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();

        Set<String> low = classifier.classifyWithThreshold("AI technology health", 0.1);
        Set<String> high = classifier.classifyWithThreshold("AI technology health", 0.9);

        assertTrue(low.size() >= high.size());
    }

    @Test
    void testGetTopLabels() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        List<MultiLabelClassifier.LabelScore> top = classifier.getTopLabels("AI machine learning research", 3);

        assertEquals(3, top.size());
        assertTrue(top.get(0).score >= top.get(1).score);
    }

    @Test
    void testTrainAndClassify() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        List<String> docs = Arrays.asList("sports news", "tech gadgets");
        List<Set<String>> labels = Arrays.asList(
            new HashSet<>(Arrays.asList("sports")),
            new HashSet<>(Arrays.asList("technology"))
        );
        classifier.train(docs, labels);
        assertNotNull(classifier.classify("technology news"));
    }

    @Test
    void testEmptyDocument() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Set<String> labels = classifier.classify("");
        assertNotNull(labels);
    }

    @Test
    void testLabelScoreClass() {
        MultiLabelClassifier.LabelScore score = new MultiLabelClassifier.LabelScore("test", 0.75);
        assertEquals("test", score.label);
        assertEquals(0.75, score.score, 0.001);
    }

    @Test
    void testProbabilitiesAreNormalized() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Map<String, Double> probs = classifier.getLabelProbabilities("technology");
        for (double prob : probs.values()) {
            assertTrue(prob >= 0 && prob <= 1);
        }
    }

    @Test
    void testClassifyReturnsAtLeastOneLabel() {
        MultiLabelClassifier classifier = new MultiLabelClassifier();
        Set<String> labels = classifier.classify("random text here");
        assertFalse(labels.isEmpty());
    }
}`,

	hint1: 'Use binary relevance - treat each label as an independent binary classification problem',
	hint2: 'Apply threshold on label probabilities to select final labels',

	whyItMatters: `Multi-label classification handles real-world complexity:

- **Document tagging**: Articles belong to multiple categories
- **Product categorization**: Items have multiple attributes
- **Content moderation**: Multiple policy violations possible
- **Skill extraction**: Resumes contain multiple skills

Multi-label is more realistic than single-label for most applications.`,

	translations: {
		ru: {
			title: 'Мультилейбл классификация',
			description: `# Мультилейбл классификация

Реализуйте мультилейбл классификацию текста для тегирования документов.

## Задача

Создайте мультилейбл классификатор:
- Присвоение нескольких меток одному документу
- Использование подхода binary relevance
- Расчет оценок уверенности для каждой метки
- Обработка зависимостей между метками

## Пример

\`\`\`java
MultiLabelClassifier classifier = new MultiLabelClassifier();
classifier.train(documents, labelSets);
Set<String> labels = classifier.classify("AI news about deep learning in healthcare");
// Returns: {"technology", "health", "ai"}
\`\`\``,
			hint1: 'Используйте binary relevance - рассматривайте каждую метку как независимую бинарную классификацию',
			hint2: 'Применяйте порог к вероятностям меток для выбора финальных меток',
			whyItMatters: `Мультилейбл классификация обрабатывает реальную сложность:

- **Тегирование документов**: Статьи принадлежат нескольким категориям
- **Категоризация товаров**: Товары имеют множество атрибутов
- **Модерация контента**: Возможны множественные нарушения политики
- **Извлечение навыков**: Резюме содержат множество навыков`,
		},
		uz: {
			title: 'Ko\'p teglik klassifikatsiya',
			description: `# Ko'p teglik klassifikatsiya

Hujjatlarni teglash uchun ko'p teglik matn klassifikatsiyasini amalga oshiring.

## Topshiriq

Ko'p teglik klassifikatorini yarating:
- Bitta hujjatga bir nechta teglarni tayinlash
- Binary relevance yondashuvidan foydalanish
- Har bir teg uchun ishonch balllarini hisoblash
- Teglar orasidagi bog'liqliklarni qayta ishlash

## Misol

\`\`\`java
MultiLabelClassifier classifier = new MultiLabelClassifier();
classifier.train(documents, labelSets);
Set<String> labels = classifier.classify("AI news about deep learning in healthcare");
// Returns: {"technology", "health", "ai"}
\`\`\``,
			hint1: "Binary relevance dan foydalaning - har bir tegni mustaqil ikkilik klassifikatsiya muammosi sifatida ko'ring",
			hint2: "Yakuniy teglarni tanlash uchun teg ehtimolliklariga chegara qo'llang",
			whyItMatters: `Ko'p teglik klassifikatsiya haqiqiy murakkablikni qayta ishlaydi:

- **Hujjatlarni teglash**: Maqolalar bir nechta kategoriyalarga tegishli
- **Mahsulot kategoriyalashi**: Elementlar bir nechta atributlarga ega
- **Kontent moderatsiyasi**: Bir nechta siyosat buzilishlari mumkin
- **Ko'nikma ajratib olish**: Rezyumelarda bir nechta ko'nikmalar mavjud`,
		},
	},
};

export default task;
