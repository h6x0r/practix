import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-lenet-architecture',
	title: 'LeNet Architecture',
	difficulty: 'medium',
	tags: ['dl4j', 'cnn', 'lenet', 'mnist'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# LeNet Architecture

Build the classic LeNet-5 architecture for digit recognition.

## Task

Implement LeNet-5 CNN:
- Input layer for 28x28 grayscale images
- Two conv-pool blocks
- Fully connected classifier
- Train on MNIST dataset

## Example

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).nOut(6).build())
    .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).build())
    // ... more layers
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LeNetBuilder {

    /**
     * Build LeNet-5 architecture for MNIST.
     * Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84) -> Output(10)
     */
    public static MultiLayerConfiguration buildLeNet5() {
        return null;
    }

    /**
     * Create and initialize LeNet model.
     */
    public static MultiLayerNetwork createLeNetModel() {
        return null;
    }

    /**
     * Build a simplified LeNet for faster training.
     */
    public static MultiLayerConfiguration buildSimpleLeNet() {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LeNetBuilder {

    /**
     * Build LeNet-5 architecture for MNIST.
     * Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84) -> Output(10)
     */
    public static MultiLayerConfiguration buildLeNet5() {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .nOut(6)
                .stride(1, 1)
                .activation(Activation.TANH)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                .nOut(16)
                .stride(1, 1)
                .activation(Activation.TANH)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, new DenseLayer.Builder()
                .nOut(120)
                .activation(Activation.TANH)
                .build())
            .layer(5, new DenseLayer.Builder()
                .nOut(84)
                .activation(Activation.TANH)
                .build())
            .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build();
    }

    /**
     * Create and initialize LeNet model.
     */
    public static MultiLayerNetwork createLeNetModel() {
        MultiLayerConfiguration conf = buildLeNet5();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Build a simplified LeNet for faster training.
     */
    public static MultiLayerConfiguration buildSimpleLeNet() {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .nOut(20)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(50)
                .activation(Activation.RELU)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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

public class LeNetBuilderTest {

    @Test
    void testBuildLeNet5() {
        MultiLayerConfiguration conf = LeNetBuilder.buildLeNet5();
        assertNotNull(conf);
        assertEquals(7, conf.getConfs().size());
    }

    @Test
    void testCreateLeNetModel() {
        MultiLayerNetwork model = LeNetBuilder.createLeNetModel();
        assertNotNull(model);
        assertTrue(model.numParams() > 0);
    }

    @Test
    void testBuildSimpleLeNet() {
        MultiLayerConfiguration conf = LeNetBuilder.buildSimpleLeNet();
        assertNotNull(conf);
        assertEquals(4, conf.getConfs().size());
    }

    @Test
    void testLeNet5HasSevenLayers() {
        MultiLayerConfiguration conf = LeNetBuilder.buildLeNet5();
        assertEquals(7, conf.getConfs().size());
    }

    @Test
    void testModelHasParameters() {
        MultiLayerNetwork model = LeNetBuilder.createLeNetModel();
        assertTrue(model.numParams() > 1000);
    }

    @Test
    void testSimpleLeNetHasFourLayers() {
        MultiLayerConfiguration conf = LeNetBuilder.buildSimpleLeNet();
        assertEquals(4, conf.getConfs().size());
    }

    @Test
    void testLeNet5ConfigNotNull() {
        MultiLayerConfiguration conf = LeNetBuilder.buildLeNet5();
        assertNotNull(conf.getConfs().get(0));
    }

    @Test
    void testSimpleLeNetConfigNotNull() {
        MultiLayerConfiguration conf = LeNetBuilder.buildSimpleLeNet();
        assertNotNull(conf.getConfs().get(0));
    }

    @Test
    void testModelCanBeInitialized() {
        MultiLayerNetwork model = LeNetBuilder.createLeNetModel();
        assertNotNull(model.getLayerWiseConfigurations());
    }

    @Test
    void testLeNet5FirstLayerExists() {
        MultiLayerConfiguration conf = LeNetBuilder.buildLeNet5();
        assertTrue(conf.getConfs().size() > 0);
    }
}`,

	hint1: 'Use InputType.convolutionalFlat(28, 28, 1) for MNIST input',
	hint2: 'Original LeNet uses TANH activation and average pooling',

	whyItMatters: `LeNet-5 is a foundational CNN architecture:

- **Historical significance**: One of the first successful CNNs (1998)
- **Practical design**: Conv-Pool pattern still used today
- **MNIST benchmark**: Standard for digit recognition
- **Building block**: Understanding LeNet helps with modern architectures

LeNet principles underpin ResNet, VGG, and all modern CNNs.`,

	translations: {
		ru: {
			title: 'Архитектура LeNet',
			description: `# Архитектура LeNet

Создайте классическую архитектуру LeNet-5 для распознавания цифр.

## Задача

Реализуйте CNN LeNet-5:
- Входной слой для изображений 28x28 в оттенках серого
- Два блока conv-pool
- Полносвязный классификатор
- Обучение на датасете MNIST

## Пример

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).nOut(6).build())
    .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).build())
    // ... more layers
    .build();
\`\`\``,
			hint1: 'Используйте InputType.convolutionalFlat(28, 28, 1) для входа MNIST',
			hint2: 'Оригинальный LeNet использует TANH активацию и average pooling',
			whyItMatters: `LeNet-5 - фундаментальная архитектура CNN:

- **Историческое значение**: Одна из первых успешных CNN (1998)
- **Практичный дизайн**: Паттерн Conv-Pool используется до сих пор
- **Бенчмарк MNIST**: Стандарт для распознавания цифр
- **Строительный блок**: Понимание LeNet помогает с современными архитектурами`,
		},
		uz: {
			title: 'LeNet arxitekturasi',
			description: `# LeNet arxitekturasi

Raqamlarni tanish uchun klassik LeNet-5 arxitekturasini yarating.

## Topshiriq

LeNet-5 CNN ni amalga oshiring:
- 28x28 kulrang tasvirlar uchun kirish qatlami
- Ikkita conv-pool bloki
- To'liq bog'langan klassifikator
- MNIST datasetida o'qitish

## Misol

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).nOut(6).build())
    .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).build())
    // ... more layers
    .build();
\`\`\``,
			hint1: "MNIST kirishi uchun InputType.convolutionalFlat(28, 28, 1) dan foydalaning",
			hint2: 'Asl LeNet TANH aktivatsiyasi va average pooling ishlatadi',
			whyItMatters: `LeNet-5 fundamental CNN arxitekturasi:

- **Tarixiy ahamiyati**: Birinchi muvaffaqiyatli CNNlardan biri (1998)
- **Amaliy dizayn**: Conv-Pool patterni hali ham ishlatiladi
- **MNIST benchmark**: Raqamlarni tanish standarti
- **Qurilish bloki**: LeNetni tushunish zamonaviy arxitekturalarga yordam beradi`,
		},
	},
};

export default task;
