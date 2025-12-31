import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-fine-tuning',
	title: 'Fine-tuning Models',
	difficulty: 'hard',
	tags: ['dl4j', 'fine-tuning', 'transfer-learning'],
	estimatedTime: '30m',
	isPremium: true,
	order: 2,
	description: `# Fine-tuning Models

Adapt pre-trained models to new classification tasks.

## Task

Implement fine-tuning:
- Replace output layer for new classes
- Set different learning rates per layer
- Train on custom dataset

## Example

\`\`\`java
FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
    .updater(new Adam(0.001))
    .seed(123)
    .build();

ComputationGraph model = new TransferLearning.GraphBuilder(vgg16)
    .fineTuneConfiguration(fineTuneConf)
    .setFeatureExtractor("fc2")
    .removeVertexAndConnections("predictions")
    .addLayer("predictions", new OutputLayer.Builder()
        .nIn(4096).nOut(numClasses).build(), "fc2")
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.activations.Activation;

public class ModelFineTuner {

    /**
     * Create fine-tuned model for new number of classes.
     * @param baseModel Pre-trained model
     * @param numClasses Number of output classes
     * @param featureLayer Name of layer to use as features
     */
    public static ComputationGraph fineTune(ComputationGraph baseModel,
                                              int numClasses,
                                              String featureLayer) {
        return null;
    }

    /**
     * Create fine-tune configuration.
     */
    public static FineTuneConfiguration createFineTuneConfig(double learningRate) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.activations.Activation;

public class ModelFineTuner {

    /**
     * Create fine-tuned model for new number of classes.
     * @param baseModel Pre-trained model
     * @param numClasses Number of output classes
     * @param featureLayer Name of layer to use as features
     */
    public static ComputationGraph fineTune(ComputationGraph baseModel,
                                              int numClasses,
                                              String featureLayer) {
        FineTuneConfiguration fineTuneConf = createFineTuneConfig(0.001);

        // Get the number of inputs to the output layer
        int nIn = (int) baseModel.getLayer(featureLayer)
            .getParam("W").size(1);

        return new TransferLearning.GraphBuilder(baseModel)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureLayer)
            .removeVertexAndConnections("predictions")
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(nIn)
                    .nOut(numClasses)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build(),
                featureLayer)
            .build();
    }

    /**
     * Create fine-tune configuration.
     */
    public static FineTuneConfiguration createFineTuneConfig(double learningRate) {
        return new FineTuneConfiguration.Builder()
            .updater(new Adam(learningRate))
            .seed(123)
            .build();
    }

    /**
     * Fine-tune with frozen feature extractor.
     */
    public static ComputationGraph fineTuneWithFrozenFeatures(
            ComputationGraph baseModel,
            int numClasses,
            String lastFrozenLayer) {

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Adam(0.0001))
            .seed(123)
            .build();

        return new TransferLearning.GraphBuilder(baseModel)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(lastFrozenLayer)
            .build();
    }

    /**
     * Set different learning rates for different layers.
     */
    public static ComputationGraph fineTuneWithLayerLRs(
            ComputationGraph baseModel,
            int numClasses,
            double baseLR,
            double newLayerLR) {

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Adam(baseLR))
            .seed(123)
            .build();

        return new TransferLearning.GraphBuilder(baseModel)
            .fineTuneConfiguration(fineTuneConf)
            .removeVertexAndConnections("predictions")
            .addLayer("predictions",
                new OutputLayer.Builder()
                    .nOut(numClasses)
                    .activation(Activation.SOFTMAX)
                    .updater(new Adam(newLayerLR))
                    .build(),
                "fc2")
            .build();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.transferlearning.*;
import static org.junit.jupiter.api.Assertions.*;

public class ModelFineTunerTest {

    @Test
    void testCreateFineTuneConfig() {
        FineTuneConfiguration config = ModelFineTuner.createFineTuneConfig(0.001);
        assertNotNull(config);
    }

    @Test
    void testFineTuneConfigLearningRate() {
        FineTuneConfiguration config = ModelFineTuner.createFineTuneConfig(0.0001);
        assertNotNull(config);
        // Learning rate is set in the configuration
    }

    @Test
    void testCreateFineTuneConfigReturnsType() {
        FineTuneConfiguration config = ModelFineTuner.createFineTuneConfig(0.01);
        assertInstanceOf(FineTuneConfiguration.class, config);
    }

    @Test
    void testCreateFineTuneConfigLowLR() {
        FineTuneConfiguration config = ModelFineTuner.createFineTuneConfig(0.00001);
        assertNotNull(config);
    }

    @Test
    void testCreateFineTuneConfigHighLR() {
        FineTuneConfiguration config = ModelFineTuner.createFineTuneConfig(0.1);
        assertNotNull(config);
    }

    @Test
    void testFineTuneConfigBuilder() {
        var config = new FineTuneConfiguration.Builder()
            .updater(new org.nd4j.linalg.learning.config.Adam(0.001))
            .seed(42)
            .build();
        assertNotNull(config);
    }

    @Test
    void testFineTuneConfigDifferentSeeds() {
        FineTuneConfiguration config1 = ModelFineTuner.createFineTuneConfig(0.001);
        FineTuneConfiguration config2 = ModelFineTuner.createFineTuneConfig(0.001);
        assertNotNull(config1);
        assertNotNull(config2);
    }

    @Test
    void testFineTuneConfigBuilderNotNull() {
        var builder = new FineTuneConfiguration.Builder();
        assertNotNull(builder);
    }

    @Test
    void testCreateFineTuneConfigWithSGD() {
        var config = new FineTuneConfiguration.Builder()
            .updater(new org.nd4j.linalg.learning.config.Sgd(0.01))
            .build();
        assertNotNull(config);
    }

    @Test
    void testFineTuneConfigWithNesterovs() {
        var config = new FineTuneConfiguration.Builder()
            .updater(new org.nd4j.linalg.learning.config.Nesterovs(0.01))
            .build();
        assertNotNull(config);
    }
}`,

	hint1: 'Use TransferLearning.GraphBuilder to modify pre-trained models',
	hint2: 'setFeatureExtractor() freezes layers up to the specified layer',

	whyItMatters: `Fine-tuning adapts models to new domains:

- **Domain adaptation**: Apply ImageNet features to medical images
- **Limited data**: Works with small datasets
- **Best practice**: Standard approach in modern CV
- **Layer selection**: Choose which layers to freeze vs train

Fine-tuning is the practical application of transfer learning.`,

	translations: {
		ru: {
			title: 'Fine-tuning моделей',
			description: `# Fine-tuning моделей

Адаптируйте предобученные модели к новым задачам классификации.

## Задача

Реализуйте fine-tuning:
- Замените выходной слой для новых классов
- Установите разные learning rate для разных слоев
- Обучите на кастомном датасете

## Пример

\`\`\`java
FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
    .updater(new Adam(0.001))
    .seed(123)
    .build();

ComputationGraph model = new TransferLearning.GraphBuilder(vgg16)
    .fineTuneConfiguration(fineTuneConf)
    .setFeatureExtractor("fc2")
    .removeVertexAndConnections("predictions")
    .addLayer("predictions", new OutputLayer.Builder()
        .nIn(4096).nOut(numClasses).build(), "fc2")
    .build();
\`\`\``,
			hint1: 'Используйте TransferLearning.GraphBuilder для модификации предобученных моделей',
			hint2: 'setFeatureExtractor() замораживает слои до указанного слоя',
			whyItMatters: `Fine-tuning адаптирует модели к новым доменам:

- **Адаптация домена**: Применение ImageNet признаков к медицинским изображениям
- **Ограниченные данные**: Работает с маленькими датасетами
- **Лучшая практика**: Стандартный подход в современном CV
- **Выбор слоев**: Выберите какие слои замораживать vs обучать`,
		},
		uz: {
			title: 'Modellarni fine-tuning qilish',
			description: `# Modellarni fine-tuning qilish

Oldindan o'qitilgan modellarni yangi klassifikatsiya vazifalariga moslashtiring.

## Topshiriq

Fine-tuning ni amalga oshiring:
- Yangi sinflar uchun chiqish qatlamini almashtiring
- Har xil qatlamlar uchun turli learning rate larni o'rnating
- Maxsus datasetda o'rgating

## Misol

\`\`\`java
FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
    .updater(new Adam(0.001))
    .seed(123)
    .build();

ComputationGraph model = new TransferLearning.GraphBuilder(vgg16)
    .fineTuneConfiguration(fineTuneConf)
    .setFeatureExtractor("fc2")
    .removeVertexAndConnections("predictions")
    .addLayer("predictions", new OutputLayer.Builder()
        .nIn(4096).nOut(numClasses).build(), "fc2")
    .build();
\`\`\``,
			hint1: "Oldindan o'qitilgan modellarni o'zgartirish uchun TransferLearning.GraphBuilder dan foydalaning",
			hint2: "setFeatureExtractor() ko'rsatilgan qatlamgacha qatlamlarni muzlatadi",
			whyItMatters: `Fine-tuning modellarni yangi domenlarga moslashtiradi:

- **Domen moslashuvi**: ImageNet xususiyatlarini tibbiy tasvirlarga qo'llash
- **Cheklangan ma'lumotlar**: Kichik datasetlar bilan ishlaydi
- **Eng yaxshi amaliyot**: Zamonaviy CV da standart yondashuv
- **Qatlam tanlash**: Qaysi qatlamlarni muzlatish yoki o'qitishni tanlang`,
		},
	},
};

export default task;
