import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-deep-sentiment',
	title: 'Deep Learning Sentiment',
	difficulty: 'hard',
	tags: ['nlp', 'sentiment', 'lstm', 'dl4j'],
	estimatedTime: '30m',
	isPremium: true,
	order: 3,
	description: `# Deep Learning Sentiment Analysis

Use LSTM networks for sentiment classification with DL4J.

## Task

Implement deep learning sentiment:
- Build LSTM network
- Prepare sequence data
- Train and evaluate model

## Example

\`\`\`java
MultiLayerNetwork model = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new LSTM.Builder().nIn(100).nOut(64).build())
    .layer(new RnnOutputLayer.Builder().nIn(64).nOut(2).build())
    .build();
\`\`\``,

	initialCode: `import java.util.*;

public class DeepSentiment {

    private int vocabSize;
    private int embeddingDim;
    private int sequenceLength;

    /**
     */
    public DeepSentiment(int vocabSize, int embeddingDim, int seqLength) {
    }

    /**
     */
    public int[] padSequence(int[] indices, int maxLen) {
        return null;
    }

    /**
     */
    public int[] textToSequence(String text, Map<String, Integer> vocab) {
        return null;
    }

    /**
     */
    public double[][] createEmbedding(int[] sequence, double[][] embeddings) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class DeepSentiment {

    private int vocabSize;
    private int embeddingDim;
    private int sequenceLength;

    /**
     * Initialize network parameters.
     */
    public DeepSentiment(int vocabSize, int embeddingDim, int seqLength) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.sequenceLength = seqLength;
    }

    /**
     * Pad or truncate sequence to fixed length.
     */
    public int[] padSequence(int[] indices, int maxLen) {
        int[] padded = new int[maxLen];

        if (indices.length >= maxLen) {
            // Truncate
            System.arraycopy(indices, 0, padded, 0, maxLen);
        } else {
            // Pad with zeros (left padding)
            int offset = maxLen - indices.length;
            System.arraycopy(indices, 0, padded, offset, indices.length);
        }

        return padded;
    }

    /**
     * Convert text to sequence of word indices.
     */
    public int[] textToSequence(String text, Map<String, Integer> vocab) {
        String[] words = text.toLowerCase()
            .replaceAll("[^a-zA-Z\\\\s]", "")
            .split("\\\\s+");

        List<Integer> indices = new ArrayList<>();
        for (String word : words) {
            int idx = vocab.getOrDefault(word, 0); // 0 = unknown
            indices.add(idx);
        }

        return indices.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Create embedding lookup.
     */
    public double[][] createEmbedding(int[] sequence, double[][] embeddings) {
        double[][] embedded = new double[sequence.length][embeddingDim];

        for (int i = 0; i < sequence.length; i++) {
            int wordIdx = sequence[i];
            if (wordIdx < embeddings.length) {
                embedded[i] = embeddings[wordIdx];
            }
        }

        return embedded;
    }

    /**
     * Simple LSTM cell forward pass (educational).
     */
    public double[] lstmForward(double[][] embeddings,
                                  double[][] Wf, double[][] Wi,
                                  double[][] Wc, double[][] Wo) {
        int hiddenSize = Wf[0].length;
        double[] h = new double[hiddenSize];
        double[] c = new double[hiddenSize];

        for (double[] x : embeddings) {
            // Forget gate
            double[] f = sigmoid(add(matmul(x, Wf), h));
            // Input gate
            double[] i = sigmoid(add(matmul(x, Wi), h));
            // Candidate
            double[] cCandidate = tanh(add(matmul(x, Wc), h));
            // Cell state
            c = add(multiply(f, c), multiply(i, cCandidate));
            // Output gate
            double[] o = sigmoid(add(matmul(x, Wo), h));
            // Hidden state
            h = multiply(o, tanh(c));
        }

        return h;
    }

    // Helper functions
    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        }
        return result;
    }

    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return result;
    }

    private double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + (i < b.length ? b[i] : 0);
        }
        return result;
    }

    private double[] multiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    private double[] matmul(double[] x, double[][] W) {
        double[] result = new double[W[0].length];
        for (int j = 0; j < W[0].length; j++) {
            for (int i = 0; i < x.length && i < W.length; i++) {
                result[j] += x[i] * W[i][j];
            }
        }
        return result;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class DeepSentimentTest {

    @Test
    void testPadSequence() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        int[] seq = {1, 2, 3};
        int[] padded = ds.padSequence(seq, 5);

        assertEquals(5, padded.length);
        assertEquals(3, padded[4]); // Last element
    }

    @Test
    void testTextToSequence() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("hello", 1);
        vocab.put("world", 2);

        int[] sequence = ds.textToSequence("hello world", vocab);
        assertEquals(2, sequence.length);
        assertEquals(1, sequence[0]);
        assertEquals(2, sequence[1]);
    }

    @Test
    void testCreateEmbedding() {
        DeepSentiment ds = new DeepSentiment(3, 2, 10);
        int[] sequence = {1, 2};
        double[][] embeddings = {{0, 0}, {0.1, 0.2}, {0.3, 0.4}};

        double[][] embedded = ds.createEmbedding(sequence, embeddings);
        assertEquals(2, embedded.length);
        assertEquals(0.1, embedded[0][0], 0.001);
    }

    @Test
    void testPadSequenceTruncate() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        int[] seq = {1, 2, 3, 4, 5};
        int[] padded = ds.padSequence(seq, 3);
        assertEquals(3, padded.length);
    }

    @Test
    void testTextToSequenceUnknown() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("hello", 1);
        int[] sequence = ds.textToSequence("hello unknown", vocab);
        assertEquals(2, sequence.length);
        assertEquals(0, sequence[1]); // unknown = 0
    }

    @Test
    void testPadSequenceExactLength() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        int[] seq = {1, 2, 3};
        int[] padded = ds.padSequence(seq, 3);
        assertEquals(3, padded.length);
    }

    @Test
    void testCreateEmbeddingUnknownIndex() {
        DeepSentiment ds = new DeepSentiment(3, 2, 10);
        int[] sequence = {0}; // index 0 (unknown)
        double[][] embeddings = {{0, 0}, {0.1, 0.2}};
        double[][] embedded = ds.createEmbedding(sequence, embeddings);
        assertEquals(0.0, embedded[0][0], 0.001);
    }

    @Test
    void testTextToSequenceEmpty() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        Map<String, Integer> vocab = new HashMap<>();
        int[] sequence = ds.textToSequence("", vocab);
        assertNotNull(sequence);
    }

    @Test
    void testLstmForward() {
        DeepSentiment ds = new DeepSentiment(100, 2, 5);
        double[][] embeddings = {{0.1, 0.2}, {0.3, 0.4}};
        double[][] W = {{0.5, 0.5}, {0.5, 0.5}};
        double[] h = ds.lstmForward(embeddings, W, W, W, W);
        assertNotNull(h);
        assertEquals(2, h.length);
    }

    @Test
    void testPadSequenceEmpty() {
        DeepSentiment ds = new DeepSentiment(1000, 50, 20);
        int[] seq = {};
        int[] padded = ds.padSequence(seq, 5);
        assertEquals(5, padded.length);
        assertEquals(0, padded[0]);
    }
}`,

	hint1: 'Convert text to fixed-length sequences using padding',
	hint2: 'LSTM processes sequences one step at a time, maintaining hidden state',

	whyItMatters: `Deep learning captures complex patterns:

- **Context understanding**: LSTM models word order and context
- **Feature learning**: Learns features automatically from data
- **State-of-the-art**: Deep models achieve best results on many benchmarks
- **Transfer learning**: Pre-trained models can be fine-tuned

Foundation for transformer models like BERT.`,

	translations: {
		ru: {
			title: 'Глубокое обучение для тональности',
			description: `# Глубокое обучение для анализа тональности

Используйте LSTM сети для классификации тональности с DL4J.

## Задача

Реализуйте глубокое обучение для тональности:
- Построение LSTM сети
- Подготовка последовательных данных
- Обучение и оценка модели

## Пример

\`\`\`java
MultiLayerNetwork model = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new LSTM.Builder().nIn(100).nOut(64).build())
    .layer(new RnnOutputLayer.Builder().nIn(64).nOut(2).build())
    .build();
\`\`\``,
			hint1: 'Преобразуйте текст в последовательности фиксированной длины с помощью паддинга',
			hint2: 'LSTM обрабатывает последовательности пошагово, сохраняя скрытое состояние',
			whyItMatters: `Глубокое обучение захватывает сложные паттерны:

- **Понимание контекста**: LSTM моделирует порядок слов и контекст
- **Обучение признаков**: Автоматически учит признаки из данных
- **State-of-the-art**: Глубокие модели достигают лучших результатов
- **Transfer learning**: Предобученные модели можно дообучить`,
		},
		uz: {
			title: 'Chuqur sentiment tahlili',
			description: `# Chuqur o'rganish sentiment tahlili

DL4J bilan sentiment klassifikatsiyasi uchun LSTM tarmoqlaridan foydalaning.

## Topshiriq

Chuqur o'rganish sentimentini amalga oshiring:
- LSTM tarmog'ini qurish
- Ketma-ket ma'lumotlarni tayyorlash
- Modelni o'qitish va baholash

## Misol

\`\`\`java
MultiLayerNetwork model = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new LSTM.Builder().nIn(100).nOut(64).build())
    .layer(new RnnOutputLayer.Builder().nIn(64).nOut(2).build())
    .build();
\`\`\``,
			hint1: "Matnni padding yordamida belgilangan uzunlikdagi ketma-ketliklarga aylantiring",
			hint2: "LSTM ketma-ketliklarni birma-bir qayta ishlaydi, yashirin holatni saqlaydi",
			whyItMatters: `Chuqur o'rganish murakkab patternlarni oladi:

- **Kontekstni tushunish**: LSTM so'z tartibini va kontekstni modellaydi
- **Xususiyat o'rganish**: Ma'lumotlardan avtomatik xususiyatlarni o'rganadi
- **State-of-the-art**: Chuqur modellar ko'p benchmarklarda eng yaxshi natijalarni oladi
- **Transfer learning**: Oldindan o'qitilgan modellarni fine-tune qilish mumkin`,
		},
	},
};

export default task;
