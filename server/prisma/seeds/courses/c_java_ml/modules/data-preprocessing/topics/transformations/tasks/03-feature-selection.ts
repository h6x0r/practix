import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-feature-selection',
	title: 'Feature Selection',
	difficulty: 'medium',
	tags: ['datavec', 'features', 'selection'],
	estimatedTime: '20m',
	isPremium: true,
	order: 3,
	description: `# Feature Selection

Select most relevant features for model training.

## Task

Implement feature selection:
- Remove columns by name
- Select subset of features
- Filter based on variance

## Example

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .removeColumns("id", "timestamp")
    .selectColumns("feature1", "feature2", "target")
    .build();
\`\`\``,

	initialCode: `import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;

public class FeatureSelector {

    /**
     */
    public static TransformProcess removeColumns(Schema schema,
                                                   String... columnsToRemove) {
        return null;
    }

    /**
     */
    public static TransformProcess selectColumns(Schema schema,
                                                   String... columnsToKeep) {
        return null;
    }

    /**
     */
    public static double[] calculateVariances(INDArray features) {
        return null;
    }

    /**
     */
    public static List<Integer> getHighVarianceFeatures(double[] variances,
                                                          double threshold) {
        return null;
    }
}`,

	solutionCode: `import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.util.List;
import java.util.ArrayList;

public class FeatureSelector {

    /**
     * Remove specified columns.
     */
    public static TransformProcess removeColumns(Schema schema,
                                                   String... columnsToRemove) {
        return new TransformProcess.Builder(schema)
            .removeColumns(columnsToRemove)
            .build();
    }

    /**
     * Select only specified columns.
     */
    public static TransformProcess selectColumns(Schema schema,
                                                   String... columnsToKeep) {
        // Get all columns
        List<String> allColumns = schema.getColumnNames();
        List<String> toRemove = new ArrayList<>();

        for (String col : allColumns) {
            boolean keep = false;
            for (String keepCol : columnsToKeep) {
                if (col.equals(keepCol)) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                toRemove.add(col);
            }
        }

        if (toRemove.isEmpty()) {
            return new TransformProcess.Builder(schema).build();
        }

        return new TransformProcess.Builder(schema)
            .removeColumns(toRemove.toArray(new String[0]))
            .build();
    }

    /**
     * Calculate variance for each feature column.
     */
    public static double[] calculateVariances(INDArray features) {
        int numFeatures = (int) features.columns();
        double[] variances = new double[numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            INDArray column = features.getColumn(i);
            double mean = column.meanNumber().doubleValue();
            INDArray diff = column.sub(mean);
            INDArray squared = Transforms.pow(diff, 2);
            variances[i] = squared.meanNumber().doubleValue();
        }

        return variances;
    }

    /**
     * Get indices of features with variance above threshold.
     */
    public static List<Integer> getHighVarianceFeatures(double[] variances,
                                                          double threshold) {
        List<Integer> highVariance = new ArrayList<>();

        for (int i = 0; i < variances.length; i++) {
            if (variances[i] > threshold) {
                highVariance.add(i);
            }
        }

        return highVariance;
    }

    /**
     * Filter columns by indices.
     */
    public static INDArray selectFeaturesByIndex(INDArray features,
                                                   List<Integer> indices) {
        int[] indicesArray = indices.stream().mapToInt(i -> i).toArray();
        return features.getColumns(indicesArray);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.List;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

public class FeatureSelectorTest {

    @Test
    void testCalculateVariances() {
        // Create features with known variance
        INDArray features = Nd4j.create(new double[][]{
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0}
        });

        double[] variances = FeatureSelector.calculateVariances(features);

        assertEquals(2, variances.length);
        assertTrue(variances[1] > variances[0]); // Second column has higher variance
    }

    @Test
    void testGetHighVarianceFeatures() {
        double[] variances = {0.1, 2.0, 0.05, 1.5};
        List<Integer> high = FeatureSelector.getHighVarianceFeatures(variances, 0.5);

        assertEquals(2, high.size());
        assertTrue(high.contains(1));
        assertTrue(high.contains(3));
    }

    @Test
    void testSelectFeaturesByIndex() {
        INDArray features = Nd4j.create(new double[][]{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0}
        });

        List<Integer> indices = Arrays.asList(0, 2);
        INDArray selected = FeatureSelector.selectFeaturesByIndex(features, indices);

        assertEquals(2, selected.columns());
    }

    @Test
    void testVariancesLength() {
        INDArray features = Nd4j.create(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });
        double[] variances = FeatureSelector.calculateVariances(features);
        assertEquals(3, variances.length);
    }

    @Test
    void testHighVarianceEmpty() {
        double[] variances = {0.01, 0.02, 0.03};
        List<Integer> high = FeatureSelector.getHighVarianceFeatures(variances, 1.0);
        assertEquals(0, high.size());
    }

    @Test
    void testHighVarianceAll() {
        double[] variances = {2.0, 3.0, 4.0};
        List<Integer> high = FeatureSelector.getHighVarianceFeatures(variances, 0.5);
        assertEquals(3, high.size());
    }

    @Test
    void testSelectSingleFeature() {
        INDArray features = Nd4j.create(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        });
        List<Integer> indices = Arrays.asList(1);
        INDArray selected = FeatureSelector.selectFeaturesByIndex(features, indices);
        assertEquals(1, selected.columns());
    }

    @Test
    void testVarianceZeroColumn() {
        INDArray features = Nd4j.create(new double[][]{
            {5.0, 1.0},
            {5.0, 2.0},
            {5.0, 3.0}
        });
        double[] variances = FeatureSelector.calculateVariances(features);
        assertEquals(0.0, variances[0], 0.001);
    }

    @Test
    void testHighVarianceBoundary() {
        double[] variances = {0.5, 0.5, 0.6};
        List<Integer> high = FeatureSelector.getHighVarianceFeatures(variances, 0.5);
        assertEquals(1, high.size());
        assertTrue(high.contains(2));
    }

    @Test
    void testSelectAllFeatures() {
        INDArray features = Nd4j.create(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0}
        });
        List<Integer> indices = Arrays.asList(0, 1);
        INDArray selected = FeatureSelector.selectFeaturesByIndex(features, indices);
        assertEquals(2, selected.columns());
        assertEquals(2, selected.rows());
    }
}`,

	hint1: 'Use removeColumns() from TransformProcess.Builder',
	hint2: 'Variance filter removes low-information features',

	whyItMatters: `Feature selection improves models:

- **Reduce overfitting**: Fewer features, less noise
- **Speed up training**: Less computation needed
- **Improve accuracy**: Remove irrelevant features
- **Interpretability**: Easier to understand fewer features

Good feature selection is often more important than algorithm choice.`,

	translations: {
		ru: {
			title: 'Отбор признаков',
			description: `# Отбор признаков

Выберите наиболее релевантные признаки для обучения модели.

## Задача

Реализуйте отбор признаков:
- Удаление столбцов по имени
- Выбор подмножества признаков
- Фильтрация по дисперсии

## Пример

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .removeColumns("id", "timestamp")
    .selectColumns("feature1", "feature2", "target")
    .build();
\`\`\``,
			hint1: 'Используйте removeColumns() из TransformProcess.Builder',
			hint2: 'Фильтр дисперсии удаляет малоинформативные признаки',
			whyItMatters: `Отбор признаков улучшает модели:

- **Уменьшение переобучения**: Меньше признаков, меньше шума
- **Ускорение обучения**: Меньше вычислений
- **Улучшение точности**: Удаление нерелевантных признаков
- **Интерпретируемость**: Легче понять меньше признаков`,
		},
		uz: {
			title: 'Xususiyat tanlash',
			description: `# Xususiyat tanlash

Model o'qitish uchun eng tegishli xususiyatlarni tanlang.

## Topshiriq

Xususiyat tanlashni amalga oshiring:
- Ustunlarni nom bo'yicha olib tashlash
- Xususiyatlar to'plamini tanlash
- Dispersiya bo'yicha filtrlash

## Misol

\`\`\`java
TransformProcess tp = new TransformProcess.Builder(schema)
    .removeColumns("id", "timestamp")
    .selectColumns("feature1", "feature2", "target")
    .build();
\`\`\``,
			hint1: "TransformProcess.Builder dan removeColumns() dan foydalaning",
			hint2: "Dispersiya filtri past ma'lumotli xususiyatlarni olib tashlaydi",
			whyItMatters: `Xususiyat tanlash modellarni yaxshilaydi:

- **Overfittingni kamaytirish**: Kamroq xususiyatlar, kamroq shovqin
- **O'qitishni tezlashtirish**: Kamroq hisoblash kerak
- **Aniqlikni yaxshilash**: Tegishsiz xususiyatlarni olib tashlash
- **Interpretatsiya**: Kamroq xususiyatlarni tushunish osonroq`,
		},
	},
};

export default task;
