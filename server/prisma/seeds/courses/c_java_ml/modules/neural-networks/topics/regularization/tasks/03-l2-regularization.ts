import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-l2-regularization',
	title: 'L2 Weight Regularization',
	difficulty: 'easy',
	tags: ['dl4j', 'l2', 'weight-decay'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# L2 Weight Regularization

Apply L2 regularization (weight decay) to prevent overfitting.

## Task

Configure weight regularization:
- Set global L2 coefficient
- Apply per-layer regularization
- Combine with other regularization techniques

## Example

\`\`\`java
new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .list()
    // ... layers
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class L2RegularizedNetwork {

    /**
     * Build network with global L2 regularization.
     * @param l2Coefficient L2 regularization strength
     */
    public static MultiLayerConfiguration buildWithL2(double l2Coefficient) {
        return null;
    }

    /**
     * Build network with both L2 and dropout.
     */
    public static MultiLayerConfiguration buildWithL2AndDropout(
            double l2Coefficient, double dropoutRate) {
        return null;
    }

    /**
     * Build network with L1 and L2 regularization (elastic net).
     */
    public static MultiLayerConfiguration buildWithElasticNet(
            double l1Coefficient, double l2Coefficient) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class L2RegularizedNetwork {

    /**
     * Build network with global L2 regularization.
     * @param l2Coefficient L2 regularization strength
     */
    public static MultiLayerConfiguration buildWithL2(double l2Coefficient) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .l2(l2Coefficient)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }

    /**
     * Build network with both L2 and dropout.
     */
    public static MultiLayerConfiguration buildWithL2AndDropout(
            double l2Coefficient, double dropoutRate) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .l2(l2Coefficient)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .dropOut(dropoutRate)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .dropOut(dropoutRate)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }

    /**
     * Build network with L1 and L2 regularization (elastic net).
     */
    public static MultiLayerConfiguration buildWithElasticNet(
            double l1Coefficient, double l2Coefficient) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .l1(l1Coefficient)
            .l2(l2Coefficient)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import static org.junit.jupiter.api.Assertions.*;

public class L2RegularizedNetworkTest {

    @Test
    void testBuildWithL2() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2(0.0001);
        assertNotNull(conf);
        assertEquals(3, conf.getConfs().size());
    }

    @Test
    void testBuildWithL2AndDropout() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2AndDropout(0.0001, 0.5);
        assertNotNull(conf);
        assertEquals(3, conf.getConfs().size());
    }

    @Test
    void testBuildWithElasticNet() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithElasticNet(0.00001, 0.0001);
        assertNotNull(conf);
        assertEquals(3, conf.getConfs().size());
    }

    @Test
    void testL2ReturnsConfig() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2(0.001);
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testL2AndDropoutReturnsConfig() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2AndDropout(0.001, 0.3);
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testElasticNetReturnsConfig() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithElasticNet(0.0001, 0.001);
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testL2ZeroCoefficient() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2(0.0);
        assertNotNull(conf);
    }

    @Test
    void testL2LargeCoefficient() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2(0.1);
        assertNotNull(conf);
    }

    @Test
    void testL2AndDropoutZeroDropout() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithL2AndDropout(0.0001, 0.0);
        assertNotNull(conf);
    }

    @Test
    void testElasticNetLayerCount() {
        MultiLayerConfiguration conf = L2RegularizedNetwork.buildWithElasticNet(0.0001, 0.0001);
        assertTrue(conf.getConfs().size() > 0);
    }
}`,

	hint1: 'Use .l2(coefficient) on NeuralNetConfiguration.Builder',
	hint2: 'Combine .l1() and .l2() for elastic net regularization',

	whyItMatters: `L2 regularization is fundamental for generalization:

- **Weight penalty**: Discourages large weights
- **Smooth solutions**: Prevents extreme parameter values
- **Proven technique**: Used since early neural networks
- **Easy to tune**: Single hyperparameter to adjust

L2 is often combined with dropout for best results.`,

	translations: {
		ru: {
			title: 'L2 регуляризация весов',
			description: `# L2 регуляризация весов

Применяйте L2 регуляризацию (weight decay) для предотвращения переобучения.

## Задача

Настройте регуляризацию весов:
- Установите глобальный коэффициент L2
- Примените регуляризацию для каждого слоя
- Комбинируйте с другими техниками регуляризации

## Пример

\`\`\`java
new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .list()
    // ... layers
    .build();
\`\`\``,
			hint1: 'Используйте .l2(coefficient) на NeuralNetConfiguration.Builder',
			hint2: 'Комбинируйте .l1() и .l2() для elastic net регуляризации',
			whyItMatters: `L2 регуляризация фундаментальна для обобщения:

- **Штраф весов**: Препятствует большим весам
- **Гладкие решения**: Предотвращает экстремальные значения параметров
- **Проверенная техника**: Используется с ранних нейронных сетей
- **Легко настроить**: Один гиперпараметр для настройки`,
		},
		uz: {
			title: "L2 vazn regularizatsiyasi",
			description: `# L2 vazn regularizatsiyasi

Overfittingni oldini olish uchun L2 regularizatsiyasini (weight decay) qo'llang.

## Topshiriq

Vazn regularizatsiyasini sozlang:
- Global L2 koeffitsientini o'rnating
- Har bir qatlam uchun regularizatsiyani qo'llang
- Boshqa regularizatsiya texnikalari bilan birlashtiring

## Misol

\`\`\`java
new NeuralNetConfiguration.Builder()
    .l2(0.0001)
    .list()
    // ... layers
    .build();
\`\`\``,
			hint1: "NeuralNetConfiguration.Builder da .l2(coefficient) dan foydalaning",
			hint2: "Elastic net regularizatsiyasi uchun .l1() va .l2() ni birlashtiring",
			whyItMatters: `L2 regularizatsiya umumlashtirish uchun fundamental:

- **Vazn jarimalari**: Katta vaznlarni to'sadi
- **Silliq yechimlar**: Ekstremal parametr qiymatlarini oldini oladi
- **Tasdiqlangan texnika**: Dastlabki neyron tarmoqlardan beri ishlatiladi
- **Sozlash oson**: Sozlash uchun bitta giperparametr`,
		},
	},
};

export default task;
