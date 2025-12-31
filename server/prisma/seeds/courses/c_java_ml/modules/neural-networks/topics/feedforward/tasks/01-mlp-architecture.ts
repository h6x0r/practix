import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-mlp-architecture',
	title: 'MLP Architecture',
	difficulty: 'medium',
	tags: ['dl4j', 'mlp', 'neural-network'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# MLP Architecture

Build a Multi-Layer Perceptron (MLP) using DL4J's configuration API.

## Task

Implement methods to:
- Configure network architecture
- Set up layers with activations
- Initialize the model

## Example

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new DenseLayer.Builder()
        .nIn(784)
        .nOut(256)
        .activation(Activation.RELU)
        .build())
    .layer(new OutputLayer.Builder()
        .nIn(256)
        .nOut(10)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MLPBuilder {

    /**
     * Build a simple 2-layer MLP for classification.
     */
    public static MultiLayerNetwork buildClassifier(int inputSize, int hiddenSize,
                                                     int numClasses) {
        return null;
    }

    /**
     * Build an MLP for regression.
     */
    public static MultiLayerNetwork buildRegressor(int inputSize, int hiddenSize,
                                                    int outputSize) {
        return null;
    }

    /**
     * Build a deep MLP with multiple hidden layers.
     */
    public static MultiLayerNetwork buildDeepMLP(int inputSize, int[] hiddenSizes,
                                                  int numClasses) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MLPBuilder {

    /**
     * Build a simple 2-layer MLP for classification.
     */
    public static MultiLayerNetwork buildClassifier(int inputSize, int hiddenSize,
                                                     int numClasses) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(hiddenSize)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder()
                .nIn(hiddenSize)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Build an MLP for regression.
     */
    public static MultiLayerNetwork buildRegressor(int inputSize, int hiddenSize,
                                                    int outputSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(hiddenSize)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder()
                .nIn(hiddenSize)
                .nOut(outputSize)
                .activation(Activation.IDENTITY)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Build a deep MLP with multiple hidden layers.
     */
    public static MultiLayerNetwork buildDeepMLP(int inputSize, int[] hiddenSizes,
                                                  int numClasses) {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .list();

        int prevSize = inputSize;
        for (int i = 0; i < hiddenSizes.length; i++) {
            builder.layer(new DenseLayer.Builder()
                .nIn(prevSize)
                .nOut(hiddenSizes[i])
                .activation(Activation.RELU)
                .build());
            prevSize = hiddenSizes[i];
        }

        builder.layer(new OutputLayer.Builder()
            .nIn(prevSize)
            .nOut(numClasses)
            .activation(Activation.SOFTMAX)
            .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .build());

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();
        return model;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import static org.junit.jupiter.api.Assertions.*;

public class MLPBuilderTest {

    @Test
    void testBuildClassifier() {
        MultiLayerNetwork model = MLPBuilder.buildClassifier(784, 256, 10);
        assertNotNull(model);
        assertEquals(2, model.getnLayers());
    }

    @Test
    void testBuildRegressor() {
        MultiLayerNetwork model = MLPBuilder.buildRegressor(10, 32, 1);
        assertNotNull(model);
        assertEquals(2, model.getnLayers());
    }

    @Test
    void testBuildDeepMLP() {
        int[] hiddenSizes = {256, 128, 64};
        MultiLayerNetwork model = MLPBuilder.buildDeepMLP(784, hiddenSizes, 10);
        assertNotNull(model);
        assertEquals(4, model.getnLayers()); // 3 hidden + 1 output
    }

    @Test
    void testModelInitialized() {
        MultiLayerNetwork model = MLPBuilder.buildClassifier(100, 50, 5);
        // Model should be initialized and ready for training
        assertNotNull(model.params());
    }

    @Test
    void testClassifierReturnsNetwork() {
        MultiLayerNetwork model = MLPBuilder.buildClassifier(50, 25, 3);
        assertInstanceOf(MultiLayerNetwork.class, model);
    }

    @Test
    void testRegressorReturnsNetwork() {
        MultiLayerNetwork model = MLPBuilder.buildRegressor(20, 10, 1);
        assertInstanceOf(MultiLayerNetwork.class, model);
    }

    @Test
    void testDeepMLPReturnsNetwork() {
        int[] hidden = {64, 32};
        MultiLayerNetwork model = MLPBuilder.buildDeepMLP(100, hidden, 5);
        assertInstanceOf(MultiLayerNetwork.class, model);
    }

    @Test
    void testDeepMLPLayerCount() {
        int[] hidden = {128, 64, 32, 16};
        MultiLayerNetwork model = MLPBuilder.buildDeepMLP(200, hidden, 10);
        assertEquals(5, model.getnLayers());
    }

    @Test
    void testClassifierHasParams() {
        MultiLayerNetwork model = MLPBuilder.buildClassifier(64, 32, 4);
        assertTrue(model.params().length() > 0);
    }

    @Test
    void testRegressorHasParams() {
        MultiLayerNetwork model = MLPBuilder.buildRegressor(32, 16, 2);
        assertTrue(model.params().length() > 0);
    }
}`,

	hint1: 'Use NeuralNetConfiguration.Builder() to start configuration',
	hint2: 'Call model.init() after creating MultiLayerNetwork',

	whyItMatters: `MLPs are the foundation of deep learning:

- **Universal approximators**: Can learn any function
- **Classification**: Softmax + cross-entropy for multi-class
- **Regression**: Identity activation + MSE loss
- **Building block**: More complex architectures build on this

Essential knowledge for all neural network work.`,

	translations: {
		ru: {
			title: 'Архитектура MLP',
			description: `# Архитектура MLP

Создайте многослойный перцептрон (MLP) с помощью API конфигурации DL4J.

## Задача

Реализуйте методы для:
- Конфигурации архитектуры сети
- Настройки слоев с активациями
- Инициализации модели

## Пример

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new DenseLayer.Builder()
        .nIn(784)
        .nOut(256)
        .activation(Activation.RELU)
        .build())
    .layer(new OutputLayer.Builder()
        .nIn(256)
        .nOut(10)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
    .build();
\`\`\``,
			hint1: 'Используйте NeuralNetConfiguration.Builder() для начала конфигурации',
			hint2: 'Вызовите model.init() после создания MultiLayerNetwork',
			whyItMatters: `MLP - основа глубокого обучения:

- **Универсальные аппроксиматоры**: Могут выучить любую функцию
- **Классификация**: Softmax + cross-entropy для многоклассовой
- **Регрессия**: Identity активация + MSE loss
- **Строительный блок**: Более сложные архитектуры строятся на этом`,
		},
		uz: {
			title: 'MLP arxitekturasi',
			description: `# MLP arxitekturasi

DL4J ning konfiguratsiya API si yordamida ko'p qatlamli pertseptron (MLP) yarating.

## Topshiriq

Metodlarni amalga oshiring:
- Tarmoq arxitekturasini sozlash
- Aktivatsiyalar bilan qatlamlarni sozlash
- Modelni ishga tushirish

## Misol

\`\`\`java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new DenseLayer.Builder()
        .nIn(784)
        .nOut(256)
        .activation(Activation.RELU)
        .build())
    .layer(new OutputLayer.Builder()
        .nIn(256)
        .nOut(10)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
    .build();
\`\`\``,
			hint1: "Konfiguratsiyani boshlash uchun NeuralNetConfiguration.Builder() dan foydalaning",
			hint2: "MultiLayerNetwork yaratilgandan keyin model.init() ni chaqiring",
			whyItMatters: `MLP chuqur o'rganishning asosi:

- **Universal approksimatorlar**: Har qanday funksiyani o'rganishi mumkin
- **Klassifikatsiya**: Ko'p sinfli uchun Softmax + cross-entropy
- **Regessiya**: Identity aktivatsiya + MSE loss
- **Qurilish bloki**: Murakkab arxitekturalar shu asosda quriladi`,
		},
	},
};

export default task;
