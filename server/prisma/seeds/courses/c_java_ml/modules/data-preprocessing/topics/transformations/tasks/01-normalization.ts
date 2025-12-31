import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-normalization',
	title: 'Feature Normalization',
	difficulty: 'easy',
	tags: ['datavec', 'normalization', 'preprocessing'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Feature Normalization

Normalize features for better model training.

## Task

Implement normalization:
- MinMax scaling (0-1 range)
- StandardScaler (zero mean, unit variance)
- Apply to datasets

## Example

\`\`\`java
NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
normalizer.fit(trainData);
normalizer.transform(trainData);
normalizer.transform(testData);
\`\`\``,

	initialCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.*;

public class FeatureNormalizer {

    /**
     * Create and fit MinMax scaler.
     */
    public static NormalizerMinMaxScaler createMinMaxScaler(DataSet data) {
        return null;
    }

    /**
     * Create and fit StandardScaler.
     */
    public static NormalizerStandardize createStandardScaler(DataSet data) {
        return null;
    }

    /**
     * Apply normalizer to dataset.
     */
    public static void applyNormalizer(DataNormalization normalizer, DataSet data) {
    }
}`,

	solutionCode: `import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.*;

public class FeatureNormalizer {

    /**
     * Create and fit MinMax scaler.
     */
    public static NormalizerMinMaxScaler createMinMaxScaler(DataSet data) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(data);
        return normalizer;
    }

    /**
     * Create and fit StandardScaler.
     */
    public static NormalizerStandardize createStandardScaler(DataSet data) {
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(data);
        return normalizer;
    }

    /**
     * Apply normalizer to dataset.
     */
    public static void applyNormalizer(DataNormalization normalizer, DataSet data) {
        normalizer.transform(data);
    }

    /**
     * Create MinMax scaler with custom range.
     */
    public static NormalizerMinMaxScaler createMinMaxScaler(DataSet data,
                                                              double min, double max) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(min, max);
        normalizer.fit(data);
        return normalizer;
    }

    /**
     * Revert normalization.
     */
    public static void revertNormalization(DataNormalization normalizer, DataSet data) {
        normalizer.revert(data);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.*;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class FeatureNormalizerTest {

    @Test
    void testCreateMinMaxScaler() {
        DataSet data = createTestDataSet();
        NormalizerMinMaxScaler scaler = FeatureNormalizer.createMinMaxScaler(data);
        assertNotNull(scaler);
    }

    @Test
    void testCreateStandardScaler() {
        DataSet data = createTestDataSet();
        NormalizerStandardize scaler = FeatureNormalizer.createStandardScaler(data);
        assertNotNull(scaler);
    }

    @Test
    void testApplyNormalizer() {
        DataSet data = createTestDataSet();
        NormalizerMinMaxScaler scaler = FeatureNormalizer.createMinMaxScaler(data);

        DataSet testData = createTestDataSet();
        FeatureNormalizer.applyNormalizer(scaler, testData);

        // Values should be in [0,1] range after MinMax
        double max = testData.getFeatures().maxNumber().doubleValue();
        double min = testData.getFeatures().minNumber().doubleValue();
        assertTrue(max <= 1.0);
        assertTrue(min >= 0.0);
    }

    private DataSet createTestDataSet() {
        return new DataSet(
            Nd4j.rand(10, 5).mul(100),  // Random features [0-100]
            Nd4j.rand(10, 2)
        );
    }

    @Test
    void testMinMaxScalerFit() {
        DataSet data = createTestDataSet();
        NormalizerMinMaxScaler scaler = FeatureNormalizer.createMinMaxScaler(data);
        assertNotNull(scaler);
    }

    @Test
    void testStandardScalerFit() {
        DataSet data = createTestDataSet();
        NormalizerStandardize scaler = FeatureNormalizer.createStandardScaler(data);
        assertNotNull(scaler);
    }

    @Test
    void testRevertNormalization() {
        DataSet data = createTestDataSet();
        double originalMax = data.getFeatures().maxNumber().doubleValue();

        NormalizerMinMaxScaler scaler = FeatureNormalizer.createMinMaxScaler(data);
        FeatureNormalizer.applyNormalizer(scaler, data);
        FeatureNormalizer.revertNormalization(scaler, data);

        double revertedMax = data.getFeatures().maxNumber().doubleValue();
        assertTrue(Math.abs(originalMax - revertedMax) < 1.0);
    }

    @Test
    void testDataSetNotNull() {
        DataSet data = createTestDataSet();
        assertNotNull(data);
    }

    @Test
    void testDataSetFeatures() {
        DataSet data = createTestDataSet();
        assertEquals(10, data.getFeatures().rows());
        assertEquals(5, data.getFeatures().columns());
    }

    @Test
    void testStandardScalerTransform() {
        DataSet data = createTestDataSet();
        NormalizerStandardize scaler = FeatureNormalizer.createStandardScaler(data);
        FeatureNormalizer.applyNormalizer(scaler, data);
        assertNotNull(data.getFeatures());
    }

    @Test
    void testMultipleScalers() {
        DataSet data1 = createTestDataSet();
        DataSet data2 = createTestDataSet();
        NormalizerMinMaxScaler scaler1 = FeatureNormalizer.createMinMaxScaler(data1);
        NormalizerStandardize scaler2 = FeatureNormalizer.createStandardScaler(data2);
        assertNotNull(scaler1);
        assertNotNull(scaler2);
    }
}`,

	hint1: 'Call fit() before transform() to compute statistics',
	hint2: 'MinMaxScaler scales to [0,1], StandardScaler to mean=0, std=1',

	whyItMatters: `Normalization is critical for neural networks:

- **Convergence**: Faster and more stable training
- **Gradient flow**: Prevents vanishing/exploding gradients
- **Equal treatment**: All features contribute equally
- **Algorithm requirements**: Many algorithms assume normalized data

Always normalize before training neural networks.`,

	translations: {
		ru: {
			title: 'Нормализация признаков',
			description: `# Нормализация признаков

Нормализуйте признаки для лучшего обучения моделей.

## Задача

Реализуйте нормализацию:
- MinMax масштабирование (диапазон 0-1)
- StandardScaler (среднее ноль, дисперсия единица)
- Применение к датасетам

## Пример

\`\`\`java
NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
normalizer.fit(trainData);
normalizer.transform(trainData);
normalizer.transform(testData);
\`\`\``,
			hint1: 'Вызовите fit() перед transform() для вычисления статистик',
			hint2: 'MinMaxScaler масштабирует в [0,1], StandardScaler в mean=0, std=1',
			whyItMatters: `Нормализация критична для нейронных сетей:

- **Сходимость**: Более быстрое и стабильное обучение
- **Поток градиента**: Предотвращает затухание/взрыв градиентов
- **Равное обращение**: Все признаки вносят равный вклад
- **Требования алгоритмов**: Многие алгоритмы предполагают нормализованные данные`,
		},
		uz: {
			title: 'Xususiyat normalizatsiyasi',
			description: `# Xususiyat normalizatsiyasi

Yaxshiroq model o'qitish uchun xususiyatlarni normalizatsiya qiling.

## Topshiriq

Normalizatsiyani amalga oshiring:
- MinMax masshtablash (0-1 diapazoni)
- StandardScaler (nol o'rtacha, birlik dispersiya)
- Datasetlarga qo'llash

## Misol

\`\`\`java
NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
normalizer.fit(trainData);
normalizer.transform(trainData);
normalizer.transform(testData);
\`\`\``,
			hint1: "Statistikani hisoblash uchun transform() dan oldin fit() ni chaqiring",
			hint2: "MinMaxScaler [0,1] ga masshtablaydi, StandardScaler mean=0, std=1 ga",
			whyItMatters: `Normalizatsiya neyron tarmoqlar uchun muhim:

- **Konvergensiya**: Tezroq va barqarorroq o'qitish
- **Gradient oqimi**: Yo'qoladigan/portlaydigan gradientlarni oldini oladi
- **Teng muomala**: Barcha xususiyatlar teng hissa qo'shadi
- **Algoritm talablari**: Ko'p algoritmlar normallashtirilgan ma'lumotlarni taxmin qiladi`,
		},
	},
};

export default task;
