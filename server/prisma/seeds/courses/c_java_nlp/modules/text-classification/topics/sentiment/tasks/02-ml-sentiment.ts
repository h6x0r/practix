import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-ml-sentiment',
	title: 'ML-Based Sentiment',
	difficulty: 'medium',
	tags: ['nlp', 'sentiment', 'classification', 'tribuo'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# ML-Based Sentiment Analysis

Train a machine learning classifier for sentiment analysis.

## Task

Implement ML sentiment classifier:
- Prepare text features
- Train classifier
- Evaluate model performance

## Example

\`\`\`java
SentimentClassifier classifier = new SentimentClassifier();
classifier.train(trainData);
String sentiment = classifier.predict("Great product!");
\`\`\``,

	initialCode: `import java.util.*;

public class MLSentimentClassifier {

    private Map<String, Integer> vocabulary;
    private double[] weights;

    /**
     */
    public double[] extractFeatures(String text) {
        return null;
    }

    /**
     */
    public void train(List<String> texts, List<Integer> labels) {
    }

    /**
     */
    public int predict(String text) {
        return 0;
    }

    /**
     */
    public double evaluate(List<String> texts, List<Integer> labels) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;

public class MLSentimentClassifier {

    private Map<String, Integer> vocabulary;
    private double[] weights;
    private double bias;
    private double learningRate = 0.01;
    private int epochs = 100;

    public MLSentimentClassifier() {
        this.vocabulary = new LinkedHashMap<>();
    }

    /**
     * Build vocabulary from training data.
     */
    private void buildVocabulary(List<String> texts) {
        vocabulary.clear();
        int index = 0;
        for (String text : texts) {
            for (String word : tokenize(text)) {
                if (!vocabulary.containsKey(word)) {
                    vocabulary.put(word, index++);
                }
            }
        }
        weights = new double[vocabulary.size()];
        bias = 0.0;
    }

    /**
     * Extract features from text.
     */
    public double[] extractFeatures(String text) {
        double[] features = new double[vocabulary.size()];
        for (String word : tokenize(text)) {
            Integer idx = vocabulary.get(word);
            if (idx != null) {
                features[idx]++;
            }
        }
        return features;
    }

    /**
     * Sigmoid activation.
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Train on labeled data using logistic regression.
     */
    public void train(List<String> texts, List<Integer> labels) {
        buildVocabulary(texts);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < texts.size(); i++) {
                double[] features = extractFeatures(texts.get(i));
                int label = labels.get(i);

                // Forward pass
                double z = bias;
                for (int j = 0; j < features.length; j++) {
                    z += weights[j] * features[j];
                }
                double prediction = sigmoid(z);

                // Backward pass
                double error = prediction - label;
                for (int j = 0; j < features.length; j++) {
                    weights[j] -= learningRate * error * features[j];
                }
                bias -= learningRate * error;
            }
        }
    }

    /**
     * Predict sentiment (0 = negative, 1 = positive).
     */
    public int predict(String text) {
        double[] features = extractFeatures(text);
        double z = bias;
        for (int j = 0; j < features.length; j++) {
            z += weights[j] * features[j];
        }
        return sigmoid(z) >= 0.5 ? 1 : 0;
    }

    /**
     * Get prediction probability.
     */
    public double predictProba(String text) {
        double[] features = extractFeatures(text);
        double z = bias;
        for (int j = 0; j < features.length; j++) {
            z += weights[j] * features[j];
        }
        return sigmoid(z);
    }

    /**
     * Calculate accuracy on test data.
     */
    public double evaluate(List<String> texts, List<Integer> labels) {
        int correct = 0;
        for (int i = 0; i < texts.size(); i++) {
            if (predict(texts.get(i)) == labels.get(i)) {
                correct++;
            }
        }
        return (double) correct / texts.size();
    }

    private String[] tokenize(String text) {
        return text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class MLSentimentClassifierTest {

    @Test
    void testTrainAndPredict() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();

        List<String> texts = Arrays.asList(
            "great movie loved it",
            "amazing product excellent",
            "terrible waste of money",
            "awful horrible experience"
        );
        List<Integer> labels = Arrays.asList(1, 1, 0, 0);

        classifier.train(texts, labels);

        assertEquals(1, classifier.predict("great product"));
        assertEquals(0, classifier.predict("terrible movie"));
    }

    @Test
    void testExtractFeatures() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("hello world"), Arrays.asList(1));

        double[] features = classifier.extractFeatures("hello world");
        assertTrue(features.length > 0);
    }

    @Test
    void testEvaluate() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();

        List<String> train = Arrays.asList("good", "great", "bad", "awful");
        List<Integer> labels = Arrays.asList(1, 1, 0, 0);
        classifier.train(train, labels);

        double accuracy = classifier.evaluate(train, labels);
        assertTrue(accuracy >= 0.5);
    }

    @Test
    void testPredictProba() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("good", "bad"), Arrays.asList(1, 0));
        double proba = classifier.predictProba("good");
        assertTrue(proba >= 0 && proba <= 1);
    }

    @Test
    void testPredictReturnsValidLabel() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("positive", "negative"), Arrays.asList(1, 0));
        int prediction = classifier.predict("positive");
        assertTrue(prediction == 0 || prediction == 1);
    }

    @Test
    void testExtractFeaturesUnknownWord() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("known word"), Arrays.asList(1));
        double[] features = classifier.extractFeatures("unknown");
        assertEquals(2, features.length);
    }

    @Test
    void testEvaluatePerfect() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        List<String> texts = Arrays.asList("excellent", "terrible");
        List<Integer> labels = Arrays.asList(1, 0);
        classifier.train(texts, labels);
        double accuracy = classifier.evaluate(texts, labels);
        assertTrue(accuracy >= 0);
    }

    @Test
    void testTrainMultipleTexts() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        List<String> texts = Arrays.asList("a", "b", "c", "d");
        List<Integer> labels = Arrays.asList(1, 1, 0, 0);
        classifier.train(texts, labels);
        assertNotNull(classifier);
    }

    @Test
    void testExtractFeaturesEmpty() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("test"), Arrays.asList(1));
        double[] features = classifier.extractFeatures("");
        assertNotNull(features);
    }

    @Test
    void testPredictAfterTraining() {
        MLSentimentClassifier classifier = new MLSentimentClassifier();
        classifier.train(Arrays.asList("love", "hate"), Arrays.asList(1, 0));
        int pred1 = classifier.predict("love");
        int pred2 = classifier.predict("hate");
        assertTrue(pred1 == 0 || pred1 == 1);
        assertTrue(pred2 == 0 || pred2 == 1);
    }
}`,

	hint1: 'Use bag-of-words features for simple text classification',
	hint2: 'Logistic regression is a good baseline for binary classification',

	whyItMatters: `ML-based sentiment learns from data:

- **Data-driven**: Learns patterns from labeled examples
- **Adaptable**: Works on any domain with training data
- **Scalable**: Can handle large datasets
- **Production-ready**: Industry standard approach

ML classifiers are the foundation of modern text classification.`,

	translations: {
		ru: {
			title: 'ML анализ тональности',
			description: `# ML анализ тональности

Обучите классификатор машинного обучения для анализа тональности.

## Задача

Реализуйте ML классификатор тональности:
- Подготовка текстовых признаков
- Обучение классификатора
- Оценка производительности модели

## Пример

\`\`\`java
SentimentClassifier classifier = new SentimentClassifier();
classifier.train(trainData);
String sentiment = classifier.predict("Great product!");
\`\`\``,
			hint1: 'Используйте bag-of-words признаки для простой классификации текста',
			hint2: 'Логистическая регрессия - хороший baseline для бинарной классификации',
			whyItMatters: `ML анализ тональности учится на данных:

- **Управляемый данными**: Учится паттернам из размеченных примеров
- **Адаптивный**: Работает на любом домене с обучающими данными
- **Масштабируемый**: Обрабатывает большие датасеты
- **Production-ready**: Промышленный стандартный подход`,
		},
		uz: {
			title: 'ML sentiment tahlili',
			description: `# ML sentiment tahlili

Sentiment tahlili uchun mashina o'rganish klassifikatorini o'qiting.

## Topshiriq

ML sentiment klassifikatorini amalga oshiring:
- Matn xususiyatlarini tayyorlash
- Klassifikatorni o'qitish
- Model samaradorligini baholash

## Misol

\`\`\`java
SentimentClassifier classifier = new SentimentClassifier();
classifier.train(trainData);
String sentiment = classifier.predict("Great product!");
\`\`\``,
			hint1: "Oddiy matn klassifikatsiyasi uchun bag-of-words xususiyatlaridan foydalaning",
			hint2: "Logistik regressiya ikkilik klassifikatsiya uchun yaxshi baseline",
			whyItMatters: `ML sentiment tahlili ma'lumotlardan o'rganadi:

- **Ma'lumotlarga asoslangan**: Belgilangan misollardan patternlarni o'rganadi
- **Moslashuvchan**: O'qitish ma'lumotlari bilan har qanday domenda ishlaydi
- **Masshtabli**: Katta ma'lumotlar to'plamlarini boshqarishi mumkin
- **Production-ready**: Sanoat standart yondashuvi`,
		},
	},
};

export default task;
