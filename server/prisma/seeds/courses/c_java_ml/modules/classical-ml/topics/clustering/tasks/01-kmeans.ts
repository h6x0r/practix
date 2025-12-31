import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-kmeans',
	title: 'K-Means Clustering',
	difficulty: 'medium',
	tags: ['tribuo', 'clustering', 'kmeans'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# K-Means Clustering

Implement K-Means clustering for unsupervised learning.

## Task

Build K-Means clusterer:
- Configure number of clusters (k)
- Set initialization method
- Evaluate cluster quality

## Example

\`\`\`java
KMeansTrainer trainer = new KMeansTrainer(
    5,        // number of clusters
    100,      // max iterations
    Distance.L2,
    KMeansTrainer.Initialisation.PLUSPLUS,
    1         // num threads
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.kmeans.*;
import org.tribuo.math.distance.*;

public class KMeansClusterer {

    /**
     * @param k Number of clusters
     * @param maxIterations Maximum iterations
     */
    public static KMeansTrainer createTrainer(int k, int maxIterations) {
        return null;
    }

    /**
     */
    public static KMeansModel train(
            KMeansTrainer trainer, Dataset<ClusterID> data) {
        return null;
    }

    /**
     */
    public static double[][] getCentroids(KMeansModel model) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.evaluation.*;
import org.tribuo.clustering.kmeans.*;
import org.tribuo.math.distance.*;
import org.nd4j.linalg.api.ndarray.INDArray;

public class KMeansClusterer {

    /**
     * Create K-Means trainer.
     * @param k Number of clusters
     * @param maxIterations Maximum iterations
     */
    public static KMeansTrainer createTrainer(int k, int maxIterations) {
        return new KMeansTrainer(
            k,
            maxIterations,
            DistanceType.L2,
            KMeansTrainer.Initialisation.PLUSPLUS,
            Runtime.getRuntime().availableProcessors(),
            Trainer.DEFAULT_SEED
        );
    }

    /**
     * Train clustering model.
     */
    public static KMeansModel train(
            KMeansTrainer trainer, Dataset<ClusterID> data) {
        return (KMeansModel) trainer.train(data);
    }

    /**
     * Get cluster centroids.
     */
    public static double[][] getCentroids(KMeansModel model) {
        INDArray centroids = model.getCentroids();
        int numClusters = (int) centroids.rows();
        int numFeatures = (int) centroids.columns();

        double[][] result = new double[numClusters][numFeatures];
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < numFeatures; j++) {
                result[i][j] = centroids.getDouble(i, j);
            }
        }
        return result;
    }

    /**
     * Create trainer with custom initialization.
     */
    public static KMeansTrainer createCustomTrainer(
            int k, int maxIterations, KMeansTrainer.Initialisation init) {
        return new KMeansTrainer(
            k,
            maxIterations,
            DistanceType.L2,
            init,
            Runtime.getRuntime().availableProcessors(),
            Trainer.DEFAULT_SEED
        );
    }

    /**
     * Predict cluster for new data.
     */
    public static int predictCluster(KMeansModel model, Example<ClusterID> example) {
        Prediction<ClusterID> pred = model.predict(example);
        return pred.getOutput().getID();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.kmeans.*;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class KMeansClustererTest {

    @Test
    void testCreateTrainer() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(3, 100);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(2, 50);
        MutableDataset<ClusterID> data = createTestDataset();

        KMeansModel model = KMeansClusterer.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testGetCentroids() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(3, 50);
        MutableDataset<ClusterID> data = createTestDataset();

        KMeansModel model = KMeansClusterer.train(trainer, data);
        double[][] centroids = KMeansClusterer.getCentroids(model);

        assertEquals(3, centroids.length);
    }

    private MutableDataset<ClusterID> createTestDataset() {
        ClusteringFactory factory = new ClusteringFactory();
        MutableDataset<ClusterID> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        // Create 3 clusters of points
        for (int i = 0; i < 30; i++) {
            double[] features = {
                (i % 3) * 5 + Math.random(),
                (i % 3) * 5 + Math.random()
            };
            dataset.add(new ArrayExample<>(new ClusterID(i % 3),
                new String[]{"x", "y"}, features));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerReturnsType() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(5, 100);
        assertInstanceOf(KMeansTrainer.class, trainer);
    }

    @Test
    void testTrainReturnsModel() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(2, 20);
        MutableDataset<ClusterID> data = createTestDataset();
        KMeansModel model = KMeansClusterer.train(trainer, data);
        assertInstanceOf(KMeansModel.class, model);
    }

    @Test
    void testDifferentK() {
        KMeansTrainer trainer1 = KMeansClusterer.createTrainer(2, 50);
        KMeansTrainer trainer2 = KMeansClusterer.createTrainer(5, 50);
        assertNotNull(trainer1);
        assertNotNull(trainer2);
    }

    @Test
    void testCentroidsNotNull() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(2, 30);
        MutableDataset<ClusterID> data = createTestDataset();
        KMeansModel model = KMeansClusterer.train(trainer, data);
        double[][] centroids = KMeansClusterer.getCentroids(model);
        assertNotNull(centroids);
    }

    @Test
    void testCentroidsHasFeatures() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(2, 30);
        MutableDataset<ClusterID> data = createTestDataset();
        KMeansModel model = KMeansClusterer.train(trainer, data);
        double[][] centroids = KMeansClusterer.getCentroids(model);
        assertEquals(2, centroids[0].length);
    }

    @Test
    void testModelNotNull() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(3, 50);
        MutableDataset<ClusterID> data = createTestDataset();
        KMeansModel model = KMeansClusterer.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testHighIterations() {
        KMeansTrainer trainer = KMeansClusterer.createTrainer(3, 200);
        assertNotNull(trainer);
    }
}`,

	hint1: 'Use KMeansTrainer with DistanceType.L2 for Euclidean distance',
	hint2: 'PLUSPLUS initialization is recommended for better convergence',

	whyItMatters: `K-Means is the most widely used clustering algorithm:

- **Scalable**: Efficient on large datasets
- **Simple**: Easy to understand and implement
- **Versatile**: Works for many clustering tasks
- **Foundation**: Basis for more advanced methods

K-Means is essential for customer segmentation and data exploration.`,

	translations: {
		ru: {
			title: 'Кластеризация K-Means',
			description: `# Кластеризация K-Means

Реализуйте кластеризацию K-Means для обучения без учителя.

## Задача

Создайте кластеризатор K-Means:
- Настройте количество кластеров (k)
- Установите метод инициализации
- Оцените качество кластеров

## Пример

\`\`\`java
KMeansTrainer trainer = new KMeansTrainer(
    5,        // number of clusters
    100,      // max iterations
    Distance.L2,
    KMeansTrainer.Initialisation.PLUSPLUS,
    1         // num threads
);
\`\`\``,
			hint1: 'Используйте KMeansTrainer с DistanceType.L2 для евклидова расстояния',
			hint2: 'Инициализация PLUSPLUS рекомендуется для лучшей сходимости',
			whyItMatters: `K-Means - самый используемый алгоритм кластеризации:

- **Масштабируемость**: Эффективен на больших датасетах
- **Простота**: Легко понять и реализовать
- **Универсальность**: Работает для многих задач кластеризации
- **Основа**: База для более продвинутых методов`,
		},
		uz: {
			title: 'K-Means klasterlash',
			description: `# K-Means klasterlash

O'qitishsiz o'rganish uchun K-Means klasterlashni amalga oshiring.

## Topshiriq

K-Means klasterlovchisini yarating:
- Klasterlar sonini (k) sozlang
- Initsializatsiya metodini o'rnating
- Klaster sifatini baholang

## Misol

\`\`\`java
KMeansTrainer trainer = new KMeansTrainer(
    5,        // number of clusters
    100,      // max iterations
    Distance.L2,
    KMeansTrainer.Initialisation.PLUSPLUS,
    1         // num threads
);
\`\`\``,
			hint1: "Evklid masofasi uchun DistanceType.L2 bilan KMeansTrainer dan foydalaning",
			hint2: "Yaxshiroq konvergensiya uchun PLUSPLUS initsializatsiyasi tavsiya etiladi",
			whyItMatters: `K-Means eng ko'p ishlatiladigan klasterlash algoritmi:

- **Masshtablanadigan**: Katta datasetlarda samarali
- **Oddiy**: Tushunish va amalga oshirish oson
- **Ko'p qirrali**: Ko'p klasterlash vazifalari uchun ishlaydi
- **Asos**: Murakkabroq metodlar uchun asos`,
		},
	},
};

export default task;
