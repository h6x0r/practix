import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-dbscan',
	title: 'DBSCAN Clustering',
	difficulty: 'medium',
	tags: ['tribuo', 'clustering', 'dbscan'],
	estimatedTime: '20m',
	isPremium: true,
	order: 2,
	description: `# DBSCAN Clustering

Implement density-based clustering that finds arbitrary-shaped clusters.

## Task

Build DBSCAN clusterer:
- Configure epsilon (neighborhood radius)
- Set minimum points for core samples
- Handle noise points

## Example

\`\`\`java
DBSCANTrainer trainer = new DBSCANTrainer(
    0.5,    // epsilon (neighborhood radius)
    5,      // minimum points
    Distance.L2
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.hdbscan.*;
import org.tribuo.math.distance.*;

public class DBSCANClusterer {

    /**
     * @param epsilon Neighborhood radius
     * @param minPoints Minimum points to form cluster
     */
    public static HdbscanTrainer createTrainer(double epsilon, int minPoints) {
        return null;
    }

    /**
     */
    public static Model<ClusterID> train(
            HdbscanTrainer trainer, Dataset<ClusterID> data) {
        return null;
    }

    /**
     */
    public static boolean isNoise(Prediction<ClusterID> prediction) {
        return false;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.hdbscan.*;
import org.tribuo.math.distance.*;

public class DBSCANClusterer {

    /**
     * Create HDBSCAN trainer (improved DBSCAN).
     * @param minClusterSize Minimum cluster size
     * @param minPoints Minimum points for core sample
     */
    public static HdbscanTrainer createTrainer(int minClusterSize, int minPoints) {
        return new HdbscanTrainer(
            minClusterSize,
            minPoints,
            DistanceType.L2,
            Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Train clustering model.
     */
    public static HdbscanModel train(
            HdbscanTrainer trainer, Dataset<ClusterID> data) {
        return (HdbscanModel) trainer.train(data);
    }

    /**
     * Check if point is noise (cluster -1).
     */
    public static boolean isNoise(Prediction<ClusterID> prediction) {
        return prediction.getOutput().getID() == -1;
    }

    /**
     * Get number of clusters (excluding noise).
     */
    public static int getNumClusters(HdbscanModel model, Dataset<ClusterID> data) {
        java.util.Set<Integer> clusters = new java.util.HashSet<>();
        for (Example<ClusterID> example : data) {
            Prediction<ClusterID> pred = model.predict(example);
            int clusterId = pred.getOutput().getID();
            if (clusterId >= 0) {
                clusters.add(clusterId);
            }
        }
        return clusters.size();
    }

    /**
     * Count noise points.
     */
    public static int countNoisePoints(HdbscanModel model, Dataset<ClusterID> data) {
        int noiseCount = 0;
        for (Example<ClusterID> example : data) {
            if (isNoise(model.predict(example))) {
                noiseCount++;
            }
        }
        return noiseCount;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.clustering.*;
import org.tribuo.clustering.hdbscan.*;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class DBSCANClustererTest {

    @Test
    void testCreateTrainer() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(5, 3);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(3, 2);
        MutableDataset<ClusterID> data = createTestDataset();

        HdbscanModel model = DBSCANClusterer.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testIsNoise() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(5, 3);
        MutableDataset<ClusterID> data = createTestDataset();
        HdbscanModel model = DBSCANClusterer.train(trainer, data);

        // Check that some predictions are not noise
        boolean hasNonNoise = false;
        for (Example<ClusterID> ex : data) {
            if (!DBSCANClusterer.isNoise(model.predict(ex))) {
                hasNonNoise = true;
                break;
            }
        }
        assertTrue(hasNonNoise);
    }

    private MutableDataset<ClusterID> createTestDataset() {
        ClusteringFactory factory = new ClusteringFactory();
        MutableDataset<ClusterID> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        // Dense cluster 1
        for (int i = 0; i < 20; i++) {
            dataset.add(new ArrayExample<>(new ClusterID(0),
                new String[]{"x", "y"},
                new double[]{Math.random(), Math.random()}));
        }
        // Dense cluster 2
        for (int i = 0; i < 20; i++) {
            dataset.add(new ArrayExample<>(new ClusterID(1),
                new String[]{"x", "y"},
                new double[]{5 + Math.random(), 5 + Math.random()}));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerWithDifferentParams() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(10, 5);
        assertNotNull(trainer);
    }

    @Test
    void testCreateTrainerSmallMinCluster() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(2, 2);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetSize() {
        MutableDataset<ClusterID> data = createTestDataset();
        assertEquals(40, data.size());
    }

    @Test
    void testTrainReturnsModel() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(5, 3);
        MutableDataset<ClusterID> data = createTestDataset();
        HdbscanModel model = DBSCANClusterer.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testModelCanPredict() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(5, 3);
        MutableDataset<ClusterID> data = createTestDataset();
        HdbscanModel model = DBSCANClusterer.train(trainer, data);
        for (Example<ClusterID> ex : data) {
            Prediction<ClusterID> pred = model.predict(ex);
            assertNotNull(pred);
            break;
        }
    }

    @Test
    void testCreateTrainerLargeMinPoints() {
        HdbscanTrainer trainer = DBSCANClusterer.createTrainer(8, 10);
        assertNotNull(trainer);
    }

    @Test
    void testMultipleTrainersCanBeCreated() {
        HdbscanTrainer t1 = DBSCANClusterer.createTrainer(3, 2);
        HdbscanTrainer t2 = DBSCANClusterer.createTrainer(5, 4);
        assertNotNull(t1);
        assertNotNull(t2);
    }
}`,

	hint1: 'Use HdbscanTrainer as improved DBSCAN in Tribuo',
	hint2: 'Noise points have cluster ID of -1',

	whyItMatters: `DBSCAN offers unique advantages over K-Means:

- **Arbitrary shapes**: Finds non-spherical clusters
- **Automatic k**: No need to specify cluster count
- **Outlier detection**: Naturally identifies noise points
- **Robust**: Handles varying densities with HDBSCAN

DBSCAN is essential for spatial data and anomaly detection.`,

	translations: {
		ru: {
			title: 'Кластеризация DBSCAN',
			description: `# Кластеризация DBSCAN

Реализуйте кластеризацию на основе плотности для кластеров произвольной формы.

## Задача

Создайте кластеризатор DBSCAN:
- Настройте epsilon (радиус соседства)
- Установите минимум точек для ядра
- Обработайте шумовые точки

## Пример

\`\`\`java
DBSCANTrainer trainer = new DBSCANTrainer(
    0.5,    // epsilon (neighborhood radius)
    5,      // minimum points
    Distance.L2
);
\`\`\``,
			hint1: 'Используйте HdbscanTrainer как улучшенный DBSCAN в Tribuo',
			hint2: 'Шумовые точки имеют ID кластера -1',
			whyItMatters: `DBSCAN имеет уникальные преимущества перед K-Means:

- **Произвольные формы**: Находит несферические кластеры
- **Автоматический k**: Не нужно указывать число кластеров
- **Обнаружение выбросов**: Естественно определяет шумовые точки
- **Устойчивость**: Обрабатывает разные плотности с HDBSCAN`,
		},
		uz: {
			title: 'DBSCAN klasterlash',
			description: `# DBSCAN klasterlash

Ixtiyoriy shakldagi klasterlarni topadigan zichlikka asoslangan klasterlashni amalga oshiring.

## Topshiriq

DBSCAN klasterlovchisini yarating:
- Epsilon (qo'shnilik radiusi) ni sozlang
- Yadro namunalari uchun minimal nuqtalarni o'rnating
- Shovqin nuqtalarini boshqaring

## Misol

\`\`\`java
DBSCANTrainer trainer = new DBSCANTrainer(
    0.5,    // epsilon (neighborhood radius)
    5,      // minimum points
    Distance.L2
);
\`\`\``,
			hint1: "Tribuo da yaxshilangan DBSCAN sifatida HdbscanTrainer dan foydalaning",
			hint2: "Shovqin nuqtalarining klaster IDsi -1",
			whyItMatters: `DBSCAN K-Means ga nisbatan noyob afzalliklarga ega:

- **Ixtiyoriy shakllar**: Nosferikal klasterlarni topadi
- **Avtomatik k**: Klaster sonini ko'rsatish shart emas
- **Outlier aniqlash**: Shovqin nuqtalarini tabiiy ravishda aniqlaydi
- **Barqaror**: HDBSCAN bilan turli zichliklarni boshqaradi`,
		},
	},
};

export default task;
