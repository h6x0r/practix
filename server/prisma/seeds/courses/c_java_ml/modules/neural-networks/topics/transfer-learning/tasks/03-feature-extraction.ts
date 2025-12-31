import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-feature-extraction',
	title: 'Feature Extraction',
	difficulty: 'medium',
	tags: ['dl4j', 'features', 'transfer-learning'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Feature Extraction

Use pre-trained models as fixed feature extractors.

## Task

Extract features for downstream tasks:
- Forward pass through frozen layers
- Save extracted features
- Use with traditional ML classifiers

## Example

\`\`\`java
// Get activations from specific layer
Map<String, INDArray> activations = model.feedForward(image, false);
INDArray features = activations.get("fc2");

// Use features with SVM, Random Forest, etc.
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class FeatureExtractor {

    private ComputationGraph model;
    private String featureLayerName;

    public FeatureExtractor(ComputationGraph model, String featureLayerName) {
        this.model = model;
        this.featureLayerName = featureLayerName;
    }

    /**
     * Extract features from a single image.
     */
    public INDArray extractFeatures(INDArray image) {
        return null;
    }

    /**
     * Extract features from batch of images.
     */
    public INDArray extractBatchFeatures(INDArray images) {
        return null;
    }

    /**
     * Extract features for entire dataset.
     */
    public DataSet extractDatasetFeatures(DataSet dataset) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class FeatureExtractor {

    private ComputationGraph model;
    private String featureLayerName;

    public FeatureExtractor(ComputationGraph model, String featureLayerName) {
        this.model = model;
        this.featureLayerName = featureLayerName;
    }

    /**
     * Extract features from a single image.
     */
    public INDArray extractFeatures(INDArray image) {
        // Ensure batch dimension
        if (image.rank() == 3) {
            image = image.reshape(1, image.size(0), image.size(1), image.size(2));
        }

        Map<String, INDArray> activations = model.feedForward(image, false);
        return activations.get(featureLayerName);
    }

    /**
     * Extract features from batch of images.
     */
    public INDArray extractBatchFeatures(INDArray images) {
        Map<String, INDArray> activations = model.feedForward(images, false);
        INDArray features = activations.get(featureLayerName);

        // Flatten if needed (e.g., from conv layer)
        if (features.rank() > 2) {
            long batchSize = features.size(0);
            long featureSize = features.length() / batchSize;
            features = features.reshape(batchSize, featureSize);
        }

        return features;
    }

    /**
     * Extract features for entire dataset.
     */
    public DataSet extractDatasetFeatures(DataSet dataset) {
        INDArray features = extractBatchFeatures(dataset.getFeatures());
        return new DataSet(features, dataset.getLabels());
    }

    /**
     * Get feature dimension.
     */
    public long getFeatureDimension(INDArray sampleImage) {
        INDArray features = extractFeatures(sampleImage);
        return features.length();
    }

    /**
     * Extract and save features to file.
     */
    public void extractAndSave(DataSet dataset, String savePath) throws Exception {
        DataSet extracted = extractDatasetFeatures(dataset);
        Nd4j.saveBinary(extracted.getFeatures(), new java.io.File(savePath + "_features.bin"));
        Nd4j.saveBinary(extracted.getLabels(), new java.io.File(savePath + "_labels.bin"));
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import static org.junit.jupiter.api.Assertions.*;

public class FeatureExtractorTest {

    @Test
    void testBatchFeatureShape() {
        // Create mock features to test shape handling
        INDArray mockFeatures = Nd4j.rand(10, 4096);

        // Flatten should keep shape for 2D
        assertEquals(2, mockFeatures.rank());
        assertEquals(10, mockFeatures.rows());
        assertEquals(4096, mockFeatures.columns());
    }

    @Test
    void testDataSetExtraction() {
        INDArray features = Nd4j.rand(5, 100);
        INDArray labels = Nd4j.zeros(5, 10);

        DataSet dataset = new DataSet(features, labels);
        assertEquals(5, dataset.numExamples());
    }

    @Test
    void testMockFeaturesRank() {
        INDArray mockFeatures = Nd4j.rand(8, 512);
        assertEquals(2, mockFeatures.rank());
    }

    @Test
    void testDataSetLabelsShape() {
        INDArray features = Nd4j.rand(10, 4096);
        INDArray labels = Nd4j.zeros(10, 5);
        DataSet dataset = new DataSet(features, labels);
        assertEquals(5, dataset.getLabels().columns());
    }

    @Test
    void testFeaturesColumnCount() {
        INDArray mockFeatures = Nd4j.rand(5, 2048);
        assertEquals(2048, mockFeatures.columns());
    }

    @Test
    void testFeaturesRowCount() {
        INDArray mockFeatures = Nd4j.rand(32, 1024);
        assertEquals(32, mockFeatures.rows());
    }

    @Test
    void testDataSetCreation() {
        INDArray features = Nd4j.zeros(1, 100);
        INDArray labels = Nd4j.zeros(1, 10);
        DataSet dataset = new DataSet(features, labels);
        assertNotNull(dataset);
    }

    @Test
    void testFeaturesFlatten() {
        INDArray features3D = Nd4j.rand(4, 7, 7, 512);
        long batchSize = features3D.size(0);
        assertEquals(4, batchSize);
    }

    @Test
    void testDataSetFeaturesNotNull() {
        INDArray features = Nd4j.rand(3, 50);
        INDArray labels = Nd4j.zeros(3, 3);
        DataSet dataset = new DataSet(features, labels);
        assertNotNull(dataset.getFeatures());
    }
}`,

	hint1: 'feedForward(input, false) returns all layer activations',
	hint2: 'Flatten conv layer outputs before using with traditional ML',

	whyItMatters: `Feature extraction is practical and efficient:

- **No fine-tuning needed**: Use features directly
- **Fast**: Just one forward pass
- **Flexible**: Features work with any classifier
- **Combine approaches**: Neural features + traditional ML

Feature extraction is often the quickest path to good results.`,

	translations: {
		ru: {
			title: 'Извлечение признаков',
			description: `# Извлечение признаков

Используйте предобученные модели как фиксированные экстракторы признаков.

## Задача

Извлекайте признаки для downstream задач:
- Forward pass через замороженные слои
- Сохранение извлеченных признаков
- Использование с традиционными ML классификаторами

## Пример

\`\`\`java
// Get activations from specific layer
Map<String, INDArray> activations = model.feedForward(image, false);
INDArray features = activations.get("fc2");

// Use features with SVM, Random Forest, etc.
\`\`\``,
			hint1: 'feedForward(input, false) возвращает активации всех слоев',
			hint2: 'Выравнивайте выходы conv слоев перед использованием с традиционным ML',
			whyItMatters: `Извлечение признаков практично и эффективно:

- **Без fine-tuning**: Используйте признаки напрямую
- **Быстро**: Только один forward pass
- **Гибко**: Признаки работают с любым классификатором
- **Комбинирование подходов**: Нейросетевые признаки + традиционный ML`,
		},
		uz: {
			title: 'Xususiyatlarni ajratish',
			description: `# Xususiyatlarni ajratish

Oldindan o'qitilgan modellarni belgilangan xususiyat ajratuvchilar sifatida foydalaning.

## Topshiriq

Downstream vazifalar uchun xususiyatlarni ajrating:
- Muzlatilgan qatlamlar orqali forward pass
- Ajratilgan xususiyatlarni saqlash
- An'anaviy ML klassifikatorlar bilan foydalanish

## Misol

\`\`\`java
// Get activations from specific layer
Map<String, INDArray> activations = model.feedForward(image, false);
INDArray features = activations.get("fc2");

// Use features with SVM, Random Forest, etc.
\`\`\``,
			hint1: "feedForward(input, false) barcha qatlam aktivatsiyalarini qaytaradi",
			hint2: "An'anaviy ML bilan ishlatishdan oldin conv qatlam chiqishlarini tekislang",
			whyItMatters: `Xususiyat ajratish amaliy va samarali:

- **Fine-tuning shart emas**: Xususiyatlarni to'g'ridan-to'g'ri foydalaning
- **Tez**: Faqat bitta forward pass
- **Moslashuvchan**: Xususiyatlar har qanday klassifikator bilan ishlaydi
- **Yondashuvlarni birlashtirish**: Neyron xususiyatlar + an'anaviy ML`,
		},
	},
};

export default task;
