import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-dropout',
	title: 'Dropout Regularization',
	difficulty: 'easy',
	tags: ['dl4j', 'dropout', 'regularization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Dropout Regularization

Implement dropout to prevent overfitting in neural networks.

## Task

Add dropout layers to your network:
- Configure dropout probability
- Apply after dense layers
- Understand train vs inference behavior

## Example

\`\`\`java
new DenseLayer.Builder()
    .nOut(256)
    .activation(Activation.RELU)
    .dropOut(0.5)
    .build()
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DropoutNetwork {

    /**
     * Build a network with dropout after each hidden layer.
     * @param dropoutRate Probability of dropping a neuron (0.0-1.0)
     */
    public static MultiLayerConfiguration buildWithDropout(double dropoutRate) {
        // Your code here
        return null;
    }

    /**
     * Build network with varying dropout rates per layer.
     */
    public static MultiLayerConfiguration buildWithVariableDropout(
            double[] dropoutRates) {
        // Your code here
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

public class DropoutNetwork {

    /**
     * Build a network with dropout after each hidden layer.
     * @param dropoutRate Probability of dropping a neuron (0.0-1.0)
     */
    public static MultiLayerConfiguration buildWithDropout(double dropoutRate) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
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
            .layer(2, new DenseLayer.Builder()
                .nOut(64)
                .activation(Activation.RELU)
                .dropOut(dropoutRate)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }

    /**
     * Build network with varying dropout rates per layer.
     */
    public static MultiLayerConfiguration buildWithVariableDropout(
            double[] dropoutRates) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(512)
                .activation(Activation.RELU)
                .dropOut(dropoutRates[0])
                .build())
            .layer(1, new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .dropOut(dropoutRates[1])
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .dropOut(dropoutRates[2])
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import static org.junit.jupiter.api.Assertions.*;

public class DropoutNetworkTest {

    @Test
    void testBuildWithDropout() {
        MultiLayerConfiguration conf = DropoutNetwork.buildWithDropout(0.5);
        assertNotNull(conf);
        assertEquals(4, conf.getConfs().size());
    }

    @Test
    void testBuildWithVariableDropout() {
        double[] rates = {0.2, 0.3, 0.4};
        MultiLayerConfiguration conf = DropoutNetwork.buildWithVariableDropout(rates);
        assertNotNull(conf);
        assertEquals(4, conf.getConfs().size());
    }

    @Test
    void testDropoutRateRange() {
        // Should work with valid dropout rates
        assertDoesNotThrow(() -> DropoutNetwork.buildWithDropout(0.0));
        assertDoesNotThrow(() -> DropoutNetwork.buildWithDropout(0.5));
        assertDoesNotThrow(() -> DropoutNetwork.buildWithDropout(0.9));
    }

    @Test
    void testBuildWithDropoutReturnsConfig() {
        MultiLayerConfiguration conf = DropoutNetwork.buildWithDropout(0.3);
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testBuildWithVariableDropoutReturnsConfig() {
        double[] rates = {0.1, 0.2, 0.3};
        MultiLayerConfiguration conf = DropoutNetwork.buildWithVariableDropout(rates);
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testDropoutZero() {
        MultiLayerConfiguration conf = DropoutNetwork.buildWithDropout(0.0);
        assertNotNull(conf);
    }

    @Test
    void testDropoutLayerCount() {
        MultiLayerConfiguration conf = DropoutNetwork.buildWithDropout(0.4);
        assertTrue(conf.getConfs().size() > 0);
    }

    @Test
    void testVariableDropoutLayerCount() {
        double[] rates = {0.2, 0.3, 0.5};
        MultiLayerConfiguration conf = DropoutNetwork.buildWithVariableDropout(rates);
        assertTrue(conf.getConfs().size() > 0);
    }

    @Test
    void testVariableDropoutNotNull() {
        double[] rates = {0.1, 0.1, 0.1};
        MultiLayerConfiguration conf = DropoutNetwork.buildWithVariableDropout(rates);
        assertNotNull(conf);
    }

    @Test
    void testHighDropoutRate() {
        MultiLayerConfiguration conf = DropoutNetwork.buildWithDropout(0.8);
        assertNotNull(conf);
        assertEquals(4, conf.getConfs().size());
    }
}`,

	hint1: 'Use .dropOut(rate) method on layer builder',
	hint2: 'Dropout is typically not applied to output layer',

	whyItMatters: `Dropout is a powerful regularization technique:

- **Prevents co-adaptation**: Neurons cannot rely on specific other neurons
- **Ensemble effect**: Training many sub-networks implicitly
- **Simple to implement**: Just add dropout rate to layers
- **Proven effective**: Used in most modern architectures

Standard dropout rates are 0.2-0.5 depending on layer size.`,

	translations: {
		ru: {
			title: 'Регуляризация Dropout',
			description: `# Регуляризация Dropout

Реализуйте dropout для предотвращения переобучения в нейронных сетях.

## Задача

Добавьте слои dropout в сеть:
- Настройте вероятность dropout
- Применяйте после dense слоев
- Поймите различие между обучением и инференсом

## Пример

\`\`\`java
new DenseLayer.Builder()
    .nOut(256)
    .activation(Activation.RELU)
    .dropOut(0.5)
    .build()
\`\`\``,
			hint1: 'Используйте метод .dropOut(rate) на builder слоя',
			hint2: 'Dropout обычно не применяется к выходному слою',
			whyItMatters: `Dropout - мощная техника регуляризации:

- **Предотвращает ко-адаптацию**: Нейроны не могут полагаться на конкретные другие нейроны
- **Эффект ансамбля**: Неявное обучение многих подсетей
- **Простота реализации**: Просто добавьте dropout rate к слоям
- **Доказанная эффективность**: Используется в большинстве современных архитектур`,
		},
		uz: {
			title: 'Dropout regularizatsiyasi',
			description: `# Dropout regularizatsiyasi

Neyron tarmoqlarda overfittingni oldini olish uchun dropout ni amalga oshiring.

## Topshiriq

Tarmoqqa dropout qatlamlarini qo'shing:
- Dropout ehtimolligini sozlang
- Dense qatlamlardan keyin qo'llang
- O'qitish va inference orasidagi farqni tushuning

## Misol

\`\`\`java
new DenseLayer.Builder()
    .nOut(256)
    .activation(Activation.RELU)
    .dropOut(0.5)
    .build()
\`\`\``,
			hint1: "Qatlam builderda .dropOut(rate) metodidan foydalaning",
			hint2: "Dropout odatda chiqish qatlamiga qo'llanilmaydi",
			whyItMatters: `Dropout kuchli regularizatsiya texnikasi:

- **Ko-adaptatsiyani oldini oladi**: Neyronlar boshqa ma'lum neyronlarga tayanolmaydi
- **Ansambl effekti**: Ko'p sub-tarmoqlarni bilvosita o'qitish
- **Amalga oshirish oson**: Qatlamlarga dropout rate qo'shish kifoya
- **Isbotlangan samaradorlik**: Ko'p zamonaviy arxitekturalarda ishlatiladi`,
		},
	},
};

export default task;
