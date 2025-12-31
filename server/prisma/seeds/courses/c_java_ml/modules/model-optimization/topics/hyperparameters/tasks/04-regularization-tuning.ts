import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-regularization-tuning',
	title: 'Regularization Tuning',
	difficulty: 'medium',
	tags: ['regularization', 'l1', 'l2', 'hyperparameters'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Regularization Tuning

Tune L1 and L2 regularization parameters to prevent overfitting.

## Task

Implement regularization tuning:
- Configure L1 (Lasso) regularization
- Configure L2 (Ridge) regularization
- Find optimal regularization strength

## Example

\`\`\`java
NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .l1(0.00001)
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RegularizationTuner {

    /**
     * Create network with L2 regularization.
     */
    public static NeuralNetConfiguration.ListBuilder withL2(double l2Value) {
        return null;
    }

    /**
     * Create network with L1 regularization.
     */
    public static NeuralNetConfiguration.ListBuilder withL1(double l1Value) {
        return null;
    }

    /**
     * Create network with combined L1 and L2 (Elastic Net).
     */
    public static NeuralNetConfiguration.ListBuilder withElasticNet(
            double l1Value, double l2Value) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RegularizationTuner {

    /**
     * Create network with L2 regularization.
     */
    public static NeuralNetConfiguration.ListBuilder withL2(double l2Value) {
        return new NeuralNetConfiguration.Builder()
            .l2(l2Value)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(256)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build());
    }

    /**
     * Create network with L1 regularization.
     */
    public static NeuralNetConfiguration.ListBuilder withL1(double l1Value) {
        return new NeuralNetConfiguration.Builder()
            .l1(l1Value)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(256)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build());
    }

    /**
     * Create network with combined L1 and L2 (Elastic Net).
     */
    public static NeuralNetConfiguration.ListBuilder withElasticNet(
            double l1Value, double l2Value) {
        return new NeuralNetConfiguration.Builder()
            .l1(l1Value)
            .l2(l2Value)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(256)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build());
    }

    /**
     * Common regularization values to try.
     */
    public static double[] getRegularizationGrid() {
        return new double[] {0.0001, 0.0005, 0.001, 0.005, 0.01};
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import static org.junit.jupiter.api.Assertions.*;

public class RegularizationTunerTest {

    @Test
    void testWithL2() {
        NeuralNetConfiguration.ListBuilder builder = RegularizationTuner.withL2(0.001);
        assertNotNull(builder);
    }

    @Test
    void testWithL1() {
        NeuralNetConfiguration.ListBuilder builder = RegularizationTuner.withL1(0.0001);
        assertNotNull(builder);
    }

    @Test
    void testWithElasticNet() {
        NeuralNetConfiguration.ListBuilder builder =
            RegularizationTuner.withElasticNet(0.0001, 0.001);
        assertNotNull(builder);
    }

    @Test
    void testRegularizationGrid() {
        double[] grid = RegularizationTuner.getRegularizationGrid();
        assertEquals(5, grid.length);
        assertEquals(0.0001, grid[0], 0.00001);
    }

    @Test
    void testL2BuilderNotNull() {
        NeuralNetConfiguration.ListBuilder builder = RegularizationTuner.withL2(0.01);
        assertNotNull(builder);
    }

    @Test
    void testL1BuilderNotNull() {
        NeuralNetConfiguration.ListBuilder builder = RegularizationTuner.withL1(0.001);
        assertNotNull(builder);
    }

    @Test
    void testElasticNetBuilderNotNull() {
        NeuralNetConfiguration.ListBuilder builder =
            RegularizationTuner.withElasticNet(0.001, 0.01);
        assertNotNull(builder);
    }

    @Test
    void testGridContainsExpectedValues() {
        double[] grid = RegularizationTuner.getRegularizationGrid();
        assertTrue(grid.length > 0);
        assertEquals(0.01, grid[4], 0.001);
    }

    @Test
    void testGridIsOrdered() {
        double[] grid = RegularizationTuner.getRegularizationGrid();
        for (int i = 1; i < grid.length; i++) {
            assertTrue(grid[i] > grid[i-1]);
        }
    }

    @Test
    void testDifferentL2Values() {
        NeuralNetConfiguration.ListBuilder builder1 = RegularizationTuner.withL2(0.0001);
        NeuralNetConfiguration.ListBuilder builder2 = RegularizationTuner.withL2(0.01);
        assertNotNull(builder1);
        assertNotNull(builder2);
    }
}`,

	hint1: 'L2 regularization adds penalty proportional to squared weights',
	hint2: 'Start with small values like 0.0001 and increase gradually',

	whyItMatters: `Regularization is key to preventing overfitting:

- **Generalization**: Models perform better on unseen data
- **Weight control**: Prevents weights from growing too large
- **L1 vs L2**: L1 creates sparsity, L2 distributes weight penalty
- **Elastic Net**: Combines benefits of both approaches

Proper regularization is essential for production ML models.`,

	translations: {
		ru: {
			title: 'Настройка регуляризации',
			description: `# Настройка регуляризации

Настройте параметры L1 и L2 регуляризации для предотвращения переобучения.

## Задача

Реализуйте настройку регуляризации:
- Настройка L1 (Lasso) регуляризации
- Настройка L2 (Ridge) регуляризации
- Поиск оптимальной силы регуляризации

## Пример

\`\`\`java
NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .l1(0.00001)
    .build();
\`\`\``,
			hint1: 'L2 регуляризация добавляет штраф пропорциональный квадрату весов',
			hint2: 'Начните с малых значений 0.0001 и постепенно увеличивайте',
			whyItMatters: `Регуляризация ключевая для предотвращения переобучения:

- **Обобщение**: Модели лучше работают на новых данных
- **Контроль весов**: Предотвращает чрезмерный рост весов
- **L1 vs L2**: L1 создает разреженность, L2 распределяет штраф
- **Elastic Net**: Объединяет преимущества обоих подходов`,
		},
		uz: {
			title: 'Regularizatsiyani sozlash',
			description: `# Regularizatsiyani sozlash

Overfitting ni oldini olish uchun L1 va L2 regularizatsiya parametrlarini sozlang.

## Topshiriq

Regularizatsiya sozlashni amalga oshiring:
- L1 (Lasso) regularizatsiyani sozlash
- L2 (Ridge) regularizatsiyani sozlash
- Optimal regularizatsiya kuchini topish

## Misol

\`\`\`java
NeuralNetConfiguration config = new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .l1(0.00001)
    .build();
\`\`\``,
			hint1: "L2 regularizatsiya og'irliklar kvadratiga proportsional jarima qo'shadi",
			hint2: "0.0001 kabi kichik qiymatlardan boshlang va asta-sekin oshiring",
			whyItMatters: `Regularizatsiya overfitting ni oldini olish uchun kalit:

- **Umumlashtirish**: Modellar ko'rilmagan ma'lumotlarda yaxshiroq ishlaydi
- **Og'irlik nazorati**: Og'irliklarning haddan tashqari o'sishini oldini oladi
- **L1 vs L2**: L1 siyraklik yaratadi, L2 jarimani taqsimlaydi
- **Elastic Net**: Ikkala yondashuvning afzalliklarini birlashtiradi`,
		},
	},
};

export default task;
