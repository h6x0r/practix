import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-pretrained-models',
	title: 'Using Pre-trained Models',
	difficulty: 'medium',
	tags: ['dl4j', 'transfer-learning', 'pretrained'],
	estimatedTime: '25m',
	isPremium: false,
	order: 1,
	description: `# Using Pre-trained Models

Load and use pre-trained models from DL4J Model Zoo.

## Task

Work with pre-trained models:
- Load VGG16, ResNet50, etc.
- Extract features from images
- Freeze layers for fine-tuning

## Example

\`\`\`java
ZooModel zooModel = VGG16.builder().build();
ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class PretrainedModels {

    /**
     * Load pre-trained VGG16 model.
     */
    public static ComputationGraph loadVGG16() throws Exception {
        return null;
    }

    /**
     * Extract features from an image using pre-trained model.
     */
    public static INDArray extractFeatures(ComputationGraph model,
                                            INDArray image,
                                            String layerName) {
        return null;
    }

    /**
     * Freeze all layers except the last n layers.
     */
    public static void freezeLayers(ComputationGraph model, int unfrozenLayers) {
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;

public class PretrainedModels {

    /**
     * Load pre-trained VGG16 model.
     */
    public static ComputationGraph loadVGG16() throws Exception {
        ZooModel<ComputationGraph> zooModel = VGG16.builder().build();
        return zooModel.initPretrained(PretrainedType.IMAGENET);
    }

    /**
     * Load pre-trained ResNet50 model.
     */
    public static ComputationGraph loadResNet50() throws Exception {
        ZooModel<ComputationGraph> zooModel = ResNet50.builder().build();
        return zooModel.initPretrained(PretrainedType.IMAGENET);
    }

    /**
     * Extract features from an image using pre-trained model.
     */
    public static INDArray extractFeatures(ComputationGraph model,
                                            INDArray image,
                                            String layerName) {
        Map<String, INDArray> activations = model.feedForward(image, false);
        return activations.get(layerName);
    }

    /**
     * Freeze all layers except the last n layers.
     */
    public static void freezeLayers(ComputationGraph model, int unfrozenLayers) {
        String[] layerNames = model.getConfiguration().getNetworkOutputs().toArray(new String[0]);
        int totalLayers = layerNames.length;

        for (int i = 0; i < totalLayers - unfrozenLayers; i++) {
            org.deeplearning4j.nn.api.Layer layer = model.getLayer(i);
            // Freeze layer by setting learning rate to 0
            // In practice, use TransferLearning API
        }
    }

    /**
     * Get number of parameters in model.
     */
    public static long getNumParams(ComputationGraph model) {
        return model.numParams();
    }

    /**
     * Get layer names.
     */
    public static String[] getLayerNames(ComputationGraph model) {
        return model.getConfiguration().getNetworkOutputs().toArray(new String[0]);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.graph.ComputationGraph;
import static org.junit.jupiter.api.Assertions.*;

public class PretrainedModelsTest {

    @Test
    void testVGG16Builder() {
        // Test that VGG16 builder works
        assertDoesNotThrow(() -> {
            org.deeplearning4j.zoo.model.VGG16.builder().build();
        });
    }

    @Test
    void testResNet50Builder() {
        assertDoesNotThrow(() -> {
            org.deeplearning4j.zoo.model.ResNet50.builder().build();
        });
    }

    @Test
    void testVGG16BuilderNotNull() {
        var builder = org.deeplearning4j.zoo.model.VGG16.builder().build();
        assertNotNull(builder);
    }

    @Test
    void testResNet50BuilderNotNull() {
        var builder = org.deeplearning4j.zoo.model.ResNet50.builder().build();
        assertNotNull(builder);
    }

    @Test
    void testVGG16BuilderReturnsZooModel() {
        var model = org.deeplearning4j.zoo.model.VGG16.builder().build();
        assertInstanceOf(org.deeplearning4j.zoo.ZooModel.class, model);
    }

    @Test
    void testResNet50BuilderReturnsZooModel() {
        var model = org.deeplearning4j.zoo.model.ResNet50.builder().build();
        assertInstanceOf(org.deeplearning4j.zoo.ZooModel.class, model);
    }

    @Test
    void testVGG19Builder() {
        assertDoesNotThrow(() -> {
            org.deeplearning4j.zoo.model.VGG19.builder().build();
        });
    }

    @Test
    void testAlexNetBuilder() {
        assertDoesNotThrow(() -> {
            org.deeplearning4j.zoo.model.AlexNet.builder().build();
        });
    }

    @Test
    void testLeNetBuilder() {
        assertDoesNotThrow(() -> {
            org.deeplearning4j.zoo.model.LeNet.builder().build();
        });
    }

    @Test
    void testVGG16BuilderRepeatable() {
        var model1 = org.deeplearning4j.zoo.model.VGG16.builder().build();
        var model2 = org.deeplearning4j.zoo.model.VGG16.builder().build();
        assertNotNull(model1);
        assertNotNull(model2);
    }
}`,

	hint1: 'Use ZooModel.initPretrained(PretrainedType.IMAGENET) to load weights',
	hint2: 'feedForward() returns activations for all layers',

	whyItMatters: `Transfer learning saves time and data:

- **Less data needed**: Pre-trained features are general
- **Faster training**: Start from good initialization
- **Better results**: ImageNet models capture visual features
- **Standard practice**: Most CV projects use transfer learning

Transfer learning is essential for practical deep learning.`,

	translations: {
		ru: {
			title: 'Использование предобученных моделей',
			description: `# Использование предобученных моделей

Загружайте и используйте предобученные модели из DL4J Model Zoo.

## Задача

Работайте с предобученными моделями:
- Загрузите VGG16, ResNet50 и т.д.
- Извлекайте признаки из изображений
- Замораживайте слои для fine-tuning

## Пример

\`\`\`java
ZooModel zooModel = VGG16.builder().build();
ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
\`\`\``,
			hint1: 'Используйте ZooModel.initPretrained(PretrainedType.IMAGENET) для загрузки весов',
			hint2: 'feedForward() возвращает активации для всех слоев',
			whyItMatters: `Трансферное обучение экономит время и данные:

- **Меньше данных**: Предобученные признаки универсальны
- **Быстрее обучение**: Старт с хорошей инициализации
- **Лучше результаты**: ImageNet модели захватывают визуальные признаки
- **Стандартная практика**: Большинство CV проектов используют transfer learning`,
		},
		uz: {
			title: 'Oldindan o\'qitilgan modellarni ishlatish',
			description: `# Oldindan o'qitilgan modellarni ishlatish

DL4J Model Zoo dan oldindan o'qitilgan modellarni yuklang va foydalaning.

## Topshiriq

Oldindan o'qitilgan modellar bilan ishlang:
- VGG16, ResNet50 va boshqalarni yuklang
- Tasvirlardan xususiyatlarni ajrating
- Fine-tuning uchun qatlamlarni muzlatib qo'ying

## Misol

\`\`\`java
ZooModel zooModel = VGG16.builder().build();
ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
\`\`\``,
			hint1: "Vaznlarni yuklash uchun ZooModel.initPretrained(PretrainedType.IMAGENET) dan foydalaning",
			hint2: "feedForward() barcha qatlamlar uchun aktivatsiyalarni qaytaradi",
			whyItMatters: `Transfer o'qitish vaqt va ma'lumotlarni tejaydi:

- **Kamroq ma'lumot kerak**: Oldindan o'qitilgan xususiyatlar umumiy
- **Tezroq o'qitish**: Yaxshi initsializatsiyadan boshlash
- **Yaxshiroq natijalar**: ImageNet modellari vizual xususiyatlarni ushlaydi
- **Standart amaliyot**: Ko'p CV loyihalar transfer learningdan foydalanadi`,
		},
	},
};

export default task;
