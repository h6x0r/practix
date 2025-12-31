import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-batch-normalization',
	title: 'Batch Normalization',
	difficulty: 'medium',
	tags: ['dl4j', 'batch-norm', 'normalization'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Batch Normalization

Apply batch normalization to stabilize and accelerate training.

## Task

Add batch normalization layers:
- After activation functions
- Configure momentum and epsilon
- Understand gamma and beta parameters

## Example

\`\`\`java
new BatchNormalization.Builder()
    .nOut(256)
    .decay(0.9)
    .eps(1e-5)
    .build()
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BatchNormNetwork {

    /**
     * Build network with batch normalization after each dense layer.
     */
    public static MultiLayerConfiguration buildWithBatchNorm() {
        // Your code here
        return null;
    }

    /**
     * Build CNN with batch norm after conv layers.
     */
    public static MultiLayerConfiguration buildCNNWithBatchNorm() {
        // Your code here
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BatchNormNetwork {

    /**
     * Build network with batch normalization after each dense layer.
     */
    public static MultiLayerConfiguration buildWithBatchNorm() {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(1, new BatchNormalization.Builder()
                .nOut(256)
                .decay(0.9)
                .eps(1e-5)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(3, new BatchNormalization.Builder()
                .nOut(128)
                .decay(0.9)
                .eps(1e-5)
                .build())
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
    }

    /**
     * Build CNN with batch norm after conv layers.
     */
    public static MultiLayerConfiguration buildCNNWithBatchNorm() {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .nOut(32)
                .stride(1, 1)
                .activation(Activation.IDENTITY)
                .build())
            .layer(1, new BatchNormalization.Builder()
                .nOut(32)
                .build())
            .layer(2, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.IDENTITY)
                .build())
            .layer(5, new BatchNormalization.Builder()
                .nOut(64)
                .build())
            .layer(6, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build())
            .layer(7, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import static org.junit.jupiter.api.Assertions.*;

public class BatchNormNetworkTest {

    @Test
    void testBuildWithBatchNorm() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildWithBatchNorm();
        assertNotNull(conf);
        assertTrue(conf.getConfs().size() >= 4);
    }

    @Test
    void testBuildCNNWithBatchNorm() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildCNNWithBatchNorm();
        assertNotNull(conf);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        assertTrue(model.numParams() > 0);
    }

    @Test
    void testBatchNormReturnsConfig() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildWithBatchNorm();
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testCNNBatchNormReturnsConfig() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildCNNWithBatchNorm();
        assertInstanceOf(MultiLayerConfiguration.class, conf);
    }

    @Test
    void testBatchNormLayerCount() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildWithBatchNorm();
        assertEquals(5, conf.getConfs().size());
    }

    @Test
    void testCNNBatchNormLayerCount() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildCNNWithBatchNorm();
        assertTrue(conf.getConfs().size() > 5);
    }

    @Test
    void testBatchNormModelInit() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildWithBatchNorm();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        assertNotNull(model);
    }

    @Test
    void testBatchNormModelParams() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildWithBatchNorm();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        assertTrue(model.numParams() > 0);
    }

    @Test
    void testCNNBatchNormModelNotNull() {
        MultiLayerConfiguration conf = BatchNormNetwork.buildCNNWithBatchNorm();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        assertNotNull(model.params());
    }
}`,

	hint1: 'BatchNormalization.Builder() creates a batch norm layer',
	hint2: 'Place batch norm before activation or after linear layer',

	whyItMatters: `Batch normalization revolutionized deep learning:

- **Faster training**: Allows higher learning rates
- **Reduces internal covariate shift**: Stabilizes layer inputs
- **Slight regularization**: Acts as noise during training
- **Deeper networks**: Enables training very deep models

BatchNorm is used in almost all modern architectures.`,

	translations: {
		ru: {
			title: 'Batch нормализация',
			description: `# Batch нормализация

Применяйте batch нормализацию для стабилизации и ускорения обучения.

## Задача

Добавьте слои batch нормализации:
- После функций активации
- Настройте momentum и epsilon
- Поймите параметры gamma и beta

## Пример

\`\`\`java
new BatchNormalization.Builder()
    .nOut(256)
    .decay(0.9)
    .eps(1e-5)
    .build()
\`\`\``,
			hint1: 'BatchNormalization.Builder() создает слой batch norm',
			hint2: 'Размещайте batch norm перед активацией или после линейного слоя',
			whyItMatters: `Batch нормализация революционизировала глубокое обучение:

- **Быстрое обучение**: Позволяет использовать более высокие learning rate
- **Уменьшает internal covariate shift**: Стабилизирует входы слоев
- **Легкая регуляризация**: Действует как шум при обучении
- **Глубокие сети**: Позволяет обучать очень глубокие модели`,
		},
		uz: {
			title: 'Batch normalizatsiyasi',
			description: `# Batch normalizatsiyasi

O'qitishni barqarorlashtirish va tezlashtirish uchun batch normalizatsiyasini qo'llang.

## Topshiriq

Batch normalizatsiya qatlamlarini qo'shing:
- Aktivatsiya funksiyalaridan keyin
- Momentum va epsilonni sozlang
- Gamma va beta parametrlarini tushuning

## Misol

\`\`\`java
new BatchNormalization.Builder()
    .nOut(256)
    .decay(0.9)
    .eps(1e-5)
    .build()
\`\`\``,
			hint1: "BatchNormalization.Builder() batch norm qatlamini yaratadi",
			hint2: "Batch normni aktivatsiyadan oldin yoki lineer qatlamdan keyin joylashtiring",
			whyItMatters: `Batch normalizatsiya deep learningni inqilob qildi:

- **Tez o'qitish**: Yuqoriroq learning rate lardan foydalanish imkonini beradi
- **Internal covariate shiftni kamaytiradi**: Qatlam kirishlarini barqarorlashtiradi
- **Yengil regularizatsiya**: O'qitish paytida shovqin sifatida ishlaydi
- **Chuqur tarmoqlar**: Juda chuqur modellarni o'qitish imkonini beradi`,
		},
	},
};

export default task;
