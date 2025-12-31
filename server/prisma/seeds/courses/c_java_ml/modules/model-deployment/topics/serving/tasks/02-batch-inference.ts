import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-batch-inference',
	title: 'Batch Inference',
	difficulty: 'medium',
	tags: ['inference', 'batch', 'performance'],
	estimatedTime: '25m',
	isPremium: true,
	order: 2,
	description: `# Batch Inference

Implement efficient batch predictions for high throughput.

## Task

Build batch inference:
- Accept multiple samples per request
- Process efficiently in batches
- Return all predictions at once

## Example

\`\`\`java
@PostMapping("/batch")
public BatchResponse batchPredict(@RequestBody BatchRequest request) {
    INDArray batch = createBatch(request.getSamples());
    INDArray predictions = model.output(batch);
    return new BatchResponse(predictions);
}
\`\`\``,

	initialCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.List;
import java.util.ArrayList;

class BatchRequest {
    private List<double[]> samples;
    public List<double[]> getSamples() { return samples; }
}

class BatchResponse {
    private List<PredictionResult> results;
    public List<PredictionResult> getResults() { return results; }
}

class PredictionResult {
    private int predictedClass;
    private double confidence;
}

public class BatchInferenceService {

    private MultiLayerNetwork model;

    /**
     * Convert list of samples to batch tensor.
     */
    public INDArray createBatch(List<double[]> samples) {
        return null;
    }

    /**
     * Run batch inference.
     */
    public List<PredictionResult> batchPredict(List<double[]> samples) {
        return null;
    }

    /**
     * Optimal batch size for performance.
     */
    public int getOptimalBatchSize() {
        return 32;
    }
}`,

	solutionCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.List;
import java.util.ArrayList;

class BatchRequest {
    private List<double[]> samples;
    public List<double[]> getSamples() { return samples; }
    public void setSamples(List<double[]> samples) { this.samples = samples; }
}

class BatchResponse {
    private List<PredictionResult> results;
    public BatchResponse(List<PredictionResult> results) { this.results = results; }
    public List<PredictionResult> getResults() { return results; }
}

class PredictionResult {
    private int predictedClass;
    private double confidence;

    public PredictionResult(int predictedClass, double confidence) {
        this.predictedClass = predictedClass;
        this.confidence = confidence;
    }

    public int getPredictedClass() { return predictedClass; }
    public double getConfidence() { return confidence; }
}

public class BatchInferenceService {

    private MultiLayerNetwork model;
    private static final int DEFAULT_BATCH_SIZE = 32;

    public BatchInferenceService(MultiLayerNetwork model) {
        this.model = model;
    }

    /**
     * Convert list of samples to batch tensor.
     */
    public INDArray createBatch(List<double[]> samples) {
        int numSamples = samples.size();
        int featureSize = samples.get(0).length;

        double[][] batchData = new double[numSamples][featureSize];
        for (int i = 0; i < numSamples; i++) {
            batchData[i] = samples.get(i);
        }

        return Nd4j.create(batchData);
    }

    /**
     * Run batch inference.
     */
    public List<PredictionResult> batchPredict(List<double[]> samples) {
        if (samples.isEmpty()) {
            return new ArrayList<>();
        }

        // Create batch tensor
        INDArray batch = createBatch(samples);

        // Run inference
        INDArray output = model.output(batch);

        // Parse results
        List<PredictionResult> results = new ArrayList<>();
        for (int i = 0; i < samples.size(); i++) {
            INDArray row = output.getRow(i);
            int predictedClass = Nd4j.argMax(row).getInt(0);
            double confidence = row.getDouble(predictedClass);
            results.add(new PredictionResult(predictedClass, confidence));
        }

        return results;
    }

    /**
     * Optimal batch size for performance.
     */
    public int getOptimalBatchSize() {
        // Based on model size and available memory
        // Typically 32-128 works well for most models
        return DEFAULT_BATCH_SIZE;
    }

    /**
     * Process large dataset in optimal batches.
     */
    public List<PredictionResult> processLargeDataset(List<double[]> allSamples) {
        List<PredictionResult> allResults = new ArrayList<>();
        int batchSize = getOptimalBatchSize();

        for (int i = 0; i < allSamples.size(); i += batchSize) {
            int end = Math.min(i + batchSize, allSamples.size());
            List<double[]> batch = allSamples.subList(i, end);
            allResults.addAll(batchPredict(batch));
        }

        return allResults;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.ArrayList;
import static org.junit.jupiter.api.Assertions.*;

public class BatchInferenceServiceTest {

    @Test
    void testBatchRequest() {
        BatchRequest request = new BatchRequest();
        List<double[]> samples = new ArrayList<>();
        samples.add(new double[]{1.0, 2.0});
        samples.add(new double[]{3.0, 4.0});
        request.setSamples(samples);

        assertEquals(2, request.getSamples().size());
    }

    @Test
    void testPredictionResult() {
        PredictionResult result = new PredictionResult(1, 0.95);
        assertEquals(1, result.getPredictedClass());
        assertEquals(0.95, result.getConfidence(), 0.001);
    }

    @Test
    void testBatchResponse() {
        List<PredictionResult> results = new ArrayList<>();
        results.add(new PredictionResult(0, 0.8));
        results.add(new PredictionResult(1, 0.9));

        BatchResponse response = new BatchResponse(results);
        assertEquals(2, response.getResults().size());
    }

    @Test
    void testPredictionResultClass() {
        PredictionResult result = new PredictionResult(2, 0.85);
        assertEquals(2, result.getPredictedClass());
    }

    @Test
    void testPredictionResultConfidence() {
        PredictionResult result = new PredictionResult(0, 0.99);
        assertTrue(result.getConfidence() > 0.9);
    }

    @Test
    void testBatchRequestEmpty() {
        BatchRequest request = new BatchRequest();
        request.setSamples(new ArrayList<>());
        assertTrue(request.getSamples().isEmpty());
    }

    @Test
    void testBatchResponseNotNull() {
        List<PredictionResult> results = new ArrayList<>();
        BatchResponse response = new BatchResponse(results);
        assertNotNull(response.getResults());
    }

    @Test
    void testMultipleSamples() {
        BatchRequest request = new BatchRequest();
        List<double[]> samples = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            samples.add(new double[]{i, i + 1});
        }
        request.setSamples(samples);
        assertEquals(10, request.getSamples().size());
    }

    @Test
    void testResultConfidenceRange() {
        PredictionResult result = new PredictionResult(1, 0.75);
        assertTrue(result.getConfidence() >= 0 && result.getConfidence() <= 1);
    }

    @Test
    void testBatchResponseMultiple() {
        List<PredictionResult> results = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            results.add(new PredictionResult(i % 3, 0.8 + i * 0.02));
        }
        BatchResponse response = new BatchResponse(results);
        assertEquals(5, response.getResults().size());
    }
}`,

	hint1: 'Use Nd4j.create(double[][]) to create batch tensor from 2D array',
	hint2: 'Process large datasets in chunks of optimal batch size',

	whyItMatters: `Batch inference maximizes throughput:

- **Efficiency**: GPU/CPU parallelism for multiple samples
- **Latency**: Amortize overhead across batch
- **Cost**: Reduce per-prediction compute cost
- **Throughput**: Process more requests per second

Batch processing is essential for production ML systems.`,

	translations: {
		ru: {
			title: 'Пакетный инференс',
			description: `# Пакетный инференс

Реализуйте эффективные пакетные предсказания для высокой пропускной способности.

## Задача

Создайте пакетный инференс:
- Принимайте несколько образцов за запрос
- Обрабатывайте эффективно пакетами
- Возвращайте все предсказания сразу

## Пример

\`\`\`java
@PostMapping("/batch")
public BatchResponse batchPredict(@RequestBody BatchRequest request) {
    INDArray batch = createBatch(request.getSamples());
    INDArray predictions = model.output(batch);
    return new BatchResponse(predictions);
}
\`\`\``,
			hint1: 'Используйте Nd4j.create(double[][]) для создания batch тензора из 2D массива',
			hint2: 'Обрабатывайте большие датасеты порциями оптимального размера',
			whyItMatters: `Пакетный инференс максимизирует пропускную способность:

- **Эффективность**: GPU/CPU параллелизм для нескольких образцов
- **Латентность**: Амортизация накладных расходов по пакету
- **Стоимость**: Снижение вычислительных затрат на предсказание
- **Пропускная способность**: Обработка больше запросов в секунду`,
		},
		uz: {
			title: 'Batch inference',
			description: `# Batch inference

Yuqori o'tkazuvchanlik uchun samarali batch bashoratlarni amalga oshiring.

## Topshiriq

Batch inference yarating:
- Har bir so'rovda bir nechta namunalarni qabul qiling
- Batchlarda samarali qayta ishlang
- Barcha bashoratlarni bir vaqtda qaytaring

## Misol

\`\`\`java
@PostMapping("/batch")
public BatchResponse batchPredict(@RequestBody BatchRequest request) {
    INDArray batch = createBatch(request.getSamples());
    INDArray predictions = model.output(batch);
    return new BatchResponse(predictions);
}
\`\`\``,
			hint1: "2D massivdan batch tensor yaratish uchun Nd4j.create(double[][]) dan foydalaning",
			hint2: "Katta datasetlarni optimal batch hajmida qayta ishlang",
			whyItMatters: `Batch inference o'tkazuvchanlikni maksimal darajaga ko'taradi:

- **Samaradorlik**: Bir nechta namunalar uchun GPU/CPU parallelligi
- **Kechikish**: Batch bo'ylab qo'shimcha xarajatlarni amortizatsiya qilish
- **Narx**: Har bir bashorat uchun hisoblash xarajatlarini kamaytirish
- **O'tkazuvchanlik**: Sekundiga ko'proq so'rovlarni qayta ishlash`,
		},
	},
};

export default task;
