import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-hierarchical-clustering',
	title: 'Hierarchical Clustering',
	difficulty: 'medium',
	tags: ['clustering', 'hierarchical', 'dendrogram'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Hierarchical Clustering

Implement hierarchical (agglomerative) clustering algorithm.

## Task

Build hierarchical clustering:
- Agglomerative clustering approach
- Different linkage methods
- Create dendrograms

## Example

\`\`\`java
HierarchicalClusterer clusterer = new HierarchicalClusterer();
clusterer.setNumClusters(3);
clusterer.setLinkType(LinkType.COMPLETE);
\`\`\``,

	initialCode: `import java.util.*;

public class HierarchicalClusterer {

    public enum LinkType { SINGLE, COMPLETE, AVERAGE }

    private int numClusters = 2;
    private LinkType linkType = LinkType.COMPLETE;

    /**
     */
    public void setNumClusters(int n) {
    }

    /**
     */
    public void setLinkType(LinkType type) {
    }

    /**
     */
    public static double euclideanDistance(double[] a, double[] b) {
        return 0.0;
    }

    /**
     */
    public double clusterDistance(List<double[]> cluster1, List<double[]> cluster2) {
        return 0.0;
    }
}`,

	solutionCode: `import java.util.*;

public class HierarchicalClusterer {

    public enum LinkType { SINGLE, COMPLETE, AVERAGE }

    private int numClusters = 2;
    private LinkType linkType = LinkType.COMPLETE;

    /**
     * Set number of clusters.
     */
    public void setNumClusters(int n) {
        this.numClusters = n;
    }

    /**
     * Set linkage type.
     */
    public void setLinkType(LinkType type) {
        this.linkType = type;
    }

    /**
     * Calculate distance between two points.
     */
    public static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Calculate cluster distance based on linkage type.
     */
    public double clusterDistance(List<double[]> cluster1, List<double[]> cluster2) {
        switch (linkType) {
            case SINGLE:
                return singleLinkage(cluster1, cluster2);
            case COMPLETE:
                return completeLinkage(cluster1, cluster2);
            case AVERAGE:
                return averageLinkage(cluster1, cluster2);
            default:
                return completeLinkage(cluster1, cluster2);
        }
    }

    private double singleLinkage(List<double[]> c1, List<double[]> c2) {
        double minDist = Double.MAX_VALUE;
        for (double[] p1 : c1) {
            for (double[] p2 : c2) {
                double dist = euclideanDistance(p1, p2);
                if (dist < minDist) minDist = dist;
            }
        }
        return minDist;
    }

    private double completeLinkage(List<double[]> c1, List<double[]> c2) {
        double maxDist = 0.0;
        for (double[] p1 : c1) {
            for (double[] p2 : c2) {
                double dist = euclideanDistance(p1, p2);
                if (dist > maxDist) maxDist = dist;
            }
        }
        return maxDist;
    }

    private double averageLinkage(List<double[]> c1, List<double[]> c2) {
        double sumDist = 0.0;
        int count = 0;
        for (double[] p1 : c1) {
            for (double[] p2 : c2) {
                sumDist += euclideanDistance(p1, p2);
                count++;
            }
        }
        return sumDist / count;
    }

    /**
     * Perform hierarchical clustering.
     */
    public List<List<double[]>> cluster(double[][] data) {
        List<List<double[]>> clusters = new ArrayList<>();
        for (double[] point : data) {
            List<double[]> cluster = new ArrayList<>();
            cluster.add(point);
            clusters.add(cluster);
        }

        while (clusters.size() > numClusters) {
            int[] closestPair = findClosestClusters(clusters);
            mergeClusters(clusters, closestPair[0], closestPair[1]);
        }

        return clusters;
    }

    private int[] findClosestClusters(List<List<double[]>> clusters) {
        double minDist = Double.MAX_VALUE;
        int[] pair = new int[2];

        for (int i = 0; i < clusters.size(); i++) {
            for (int j = i + 1; j < clusters.size(); j++) {
                double dist = clusterDistance(clusters.get(i), clusters.get(j));
                if (dist < minDist) {
                    minDist = dist;
                    pair[0] = i;
                    pair[1] = j;
                }
            }
        }

        return pair;
    }

    private void mergeClusters(List<List<double[]>> clusters, int i, int j) {
        clusters.get(i).addAll(clusters.get(j));
        clusters.remove(j);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

public class HierarchicalClustererTest {

    @Test
    void testEuclideanDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        assertEquals(5.0, HierarchicalClusterer.euclideanDistance(a, b), 0.001);
    }

    @Test
    void testSetNumClusters() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setNumClusters(3);
        // Implicitly tested through clustering
    }

    @Test
    void testClusterDistance() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setLinkType(HierarchicalClusterer.LinkType.SINGLE);

        List<double[]> c1 = Arrays.asList(new double[]{0, 0}, new double[]{1, 0});
        List<double[]> c2 = Arrays.asList(new double[]{3, 0}, new double[]{4, 0});

        double dist = clusterer.clusterDistance(c1, c2);
        assertEquals(2.0, dist, 0.001);
    }

    @Test
    void testClustering() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setNumClusters(2);

        double[][] data = {{0, 0}, {1, 0}, {10, 0}, {11, 0}};
        List<List<double[]>> result = clusterer.cluster(data);

        assertEquals(2, result.size());
    }

    @Test
    void testEuclideanDistanceZero() {
        double[] a = {5.0, 5.0};
        double[] b = {5.0, 5.0};
        assertEquals(0.0, HierarchicalClusterer.euclideanDistance(a, b), 0.001);
    }

    @Test
    void testEuclideanDistance3D() {
        double[] a = {0.0, 0.0, 0.0};
        double[] b = {1.0, 2.0, 2.0};
        assertEquals(3.0, HierarchicalClusterer.euclideanDistance(a, b), 0.001);
    }

    @Test
    void testCompleteLinkage() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setLinkType(HierarchicalClusterer.LinkType.COMPLETE);

        List<double[]> c1 = Arrays.asList(new double[]{0, 0}, new double[]{1, 0});
        List<double[]> c2 = Arrays.asList(new double[]{3, 0}, new double[]{4, 0});

        double dist = clusterer.clusterDistance(c1, c2);
        assertEquals(4.0, dist, 0.001);
    }

    @Test
    void testAverageLinkage() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setLinkType(HierarchicalClusterer.LinkType.AVERAGE);
        assertNotNull(clusterer);
    }

    @Test
    void testClusteringThreeClusters() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setNumClusters(3);

        double[][] data = {{0, 0}, {1, 0}, {5, 0}, {6, 0}, {10, 0}, {11, 0}};
        List<List<double[]>> result = clusterer.cluster(data);

        assertEquals(3, result.size());
    }

    @Test
    void testClusteringOneClusters() {
        HierarchicalClusterer clusterer = new HierarchicalClusterer();
        clusterer.setNumClusters(1);

        double[][] data = {{0, 0}, {1, 0}, {2, 0}};
        List<List<double[]>> result = clusterer.cluster(data);

        assertEquals(1, result.size());
    }
}`,

	hint1: 'Single linkage uses minimum distance between clusters',
	hint2: 'Complete linkage uses maximum distance, making tighter clusters',

	whyItMatters: `Hierarchical clustering provides insight into data structure:

- **No k needed**: Explore different numbers of clusters via dendrograms
- **Linkage choice**: Different linkages suit different data shapes
- **Interpretability**: Dendrograms show cluster relationships
- **Deterministic**: Same data always gives same result

Useful when exploring data or when cluster hierarchy matters.`,

	translations: {
		ru: {
			title: 'Иерархическая кластеризация',
			description: `# Иерархическая кластеризация

Реализуйте алгоритм иерархической (агломеративной) кластеризации.

## Задача

Создайте иерархическую кластеризацию:
- Агломеративный подход
- Разные методы связывания
- Создание дендрограмм

## Пример

\`\`\`java
HierarchicalClusterer clusterer = new HierarchicalClusterer();
clusterer.setNumClusters(3);
clusterer.setLinkType(LinkType.COMPLETE);
\`\`\``,
			hint1: 'Single linkage использует минимальное расстояние между кластерами',
			hint2: 'Complete linkage использует максимальное расстояние, создавая плотные кластеры',
			whyItMatters: `Иерархическая кластеризация дает понимание структуры данных:

- **Не нужно k**: Исследуйте разное число кластеров через дендрограммы
- **Выбор связывания**: Разные методы подходят разным формам данных
- **Интерпретируемость**: Дендрограммы показывают отношения кластеров
- **Детерминированность**: Те же данные всегда дают тот же результат`,
		},
		uz: {
			title: 'Ierarxik klasterlash',
			description: `# Ierarxik klasterlash

Ierarxik (aglomerativ) klasterlash algoritmini amalga oshiring.

## Topshiriq

Ierarxik klasterlashni yarating:
- Aglomerativ yondashuv
- Turli bog'lash usullari
- Dendrogrammalar yaratish

## Misol

\`\`\`java
HierarchicalClusterer clusterer = new HierarchicalClusterer();
clusterer.setNumClusters(3);
clusterer.setLinkType(LinkType.COMPLETE);
\`\`\``,
			hint1: "Single linkage klasterlar orasidagi minimal masofadan foydalanadi",
			hint2: "Complete linkage maksimal masofadan foydalanadi, qattiqroq klasterlar yaratadi",
			whyItMatters: `Ierarxik klasterlash ma'lumot tuzilmasi haqida tushuncha beradi:

- **k kerak emas**: Dendrogrammalar orqali turli klaster sonlarini o'rganing
- **Bog'lash tanlovi**: Turli bog'lashlar turli ma'lumot shakllariga mos
- **Tushunarlilik**: Dendrogrammalar klaster munosabatlarini ko'rsatadi
- **Deterministik**: Bir xil ma'lumotlar doimo bir xil natija beradi`,
		},
	},
};

export default task;
