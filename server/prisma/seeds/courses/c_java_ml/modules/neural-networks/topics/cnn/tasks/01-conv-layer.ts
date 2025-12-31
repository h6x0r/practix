import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-conv-layer',
	title: 'Convolutional Layers',
	difficulty: 'medium',
	tags: ['dl4j', 'cnn', 'convolution'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Convolutional Layers

Learn to create convolutional layers for feature extraction in DL4J.

## Task

Implement CNN layer configuration:
- Create 2D convolutional layers with filters
- Configure kernel size, stride, and padding
- Stack multiple conv layers

## Example

\`\`\`java
ConvolutionLayer convLayer = new ConvolutionLayer.Builder(5, 5)
    .nIn(1)
    .nOut(20)
    .stride(1, 1)
    .activation(Activation.RELU)
    .build();
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.activations.Activation;

public class ConvLayerBuilder {

    /**
     * @param nIn Number of input channels
     * @param nOut Number of output filters
     */
    public static ConvolutionLayer createConvLayer(int nIn, int nOut, int kernelSize) {
        return null;
    }

    /**
     * @param poolSize Size of the pooling window
     * @param stride Stride of the pooling
     */
    public static SubsamplingLayer createMaxPoolLayer(int poolSize, int stride) {
        return null;
    }

    /**
     */
    public static ConvolutionLayer createConvLayerWithPadding(
            int nIn, int nOut, int kernelSize, int stride, int padding) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.activations.Activation;

public class ConvLayerBuilder {

    /**
     * Create a convolutional layer for feature extraction.
     * @param nIn Number of input channels
     * @param nOut Number of output filters
     * @param kernelSize Size of the convolution kernel
     */
    public static ConvolutionLayer createConvLayer(int nIn, int nOut, int kernelSize) {
        return new ConvolutionLayer.Builder(kernelSize, kernelSize)
            .nIn(nIn)
            .nOut(nOut)
            .stride(1, 1)
            .activation(Activation.RELU)
            .build();
    }

    /**
     * Create a max pooling layer.
     * @param poolSize Size of the pooling window
     * @param stride Stride of the pooling
     */
    public static SubsamplingLayer createMaxPoolLayer(int poolSize, int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(poolSize, poolSize)
            .stride(stride, stride)
            .build();
    }

    /**
     * Create a conv layer with custom stride and padding.
     */
    public static ConvolutionLayer createConvLayerWithPadding(
            int nIn, int nOut, int kernelSize, int stride, int padding) {
        return new ConvolutionLayer.Builder(kernelSize, kernelSize)
            .nIn(nIn)
            .nOut(nOut)
            .stride(stride, stride)
            .padding(padding, padding)
            .activation(Activation.RELU)
            .build();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import static org.junit.jupiter.api.Assertions.*;

public class ConvLayerBuilderTest {

    @Test
    void testCreateConvLayer() {
        ConvolutionLayer layer = ConvLayerBuilder.createConvLayer(1, 32, 5);
        assertNotNull(layer);
        assertEquals(1, layer.getNIn());
        assertEquals(32, layer.getNOut());
    }

    @Test
    void testCreateMaxPoolLayer() {
        SubsamplingLayer layer = ConvLayerBuilder.createMaxPoolLayer(2, 2);
        assertNotNull(layer);
        assertEquals(SubsamplingLayer.PoolingType.MAX, layer.getPoolingType());
    }

    @Test
    void testCreateConvLayerWithPadding() {
        ConvolutionLayer layer = ConvLayerBuilder.createConvLayerWithPadding(3, 64, 3, 1, 1);
        assertNotNull(layer);
        assertEquals(3, layer.getNIn());
        assertEquals(64, layer.getNOut());
    }

    @Test
    void testConvLayerNOut() {
        ConvolutionLayer layer = ConvLayerBuilder.createConvLayer(1, 16, 3);
        assertEquals(16, layer.getNOut());
    }

    @Test
    void testMaxPoolLayerNotNull() {
        SubsamplingLayer layer = ConvLayerBuilder.createMaxPoolLayer(3, 3);
        assertNotNull(layer);
    }

    @Test
    void testConvLayerDifferentKernelSizes() {
        ConvolutionLayer layer1 = ConvLayerBuilder.createConvLayer(1, 32, 3);
        ConvolutionLayer layer2 = ConvLayerBuilder.createConvLayer(1, 32, 5);
        assertNotNull(layer1);
        assertNotNull(layer2);
    }

    @Test
    void testConvLayerMultipleChannels() {
        ConvolutionLayer layer = ConvLayerBuilder.createConvLayer(3, 64, 3);
        assertEquals(3, layer.getNIn());
    }

    @Test
    void testMaxPoolDifferentSizes() {
        SubsamplingLayer layer = ConvLayerBuilder.createMaxPoolLayer(4, 4);
        assertNotNull(layer);
    }

    @Test
    void testConvWithPaddingLargeKernel() {
        ConvolutionLayer layer = ConvLayerBuilder.createConvLayerWithPadding(1, 128, 7, 2, 3);
        assertNotNull(layer);
        assertEquals(128, layer.getNOut());
    }
}`,

	hint1: 'Use ConvolutionLayer.Builder with kernel size tuple',
	hint2: 'SubsamplingLayer.Builder takes PoolingType as parameter',

	whyItMatters: `Convolutional layers are the foundation of computer vision:

- **Feature extraction**: Automatically learn visual patterns
- **Translation invariance**: Detect features regardless of position
- **Parameter sharing**: Efficient compared to fully connected layers
- **Hierarchical learning**: Stack layers for complex features

CNNs power image classification, object detection, and more.`,

	translations: {
		ru: {
			title: 'Сверточные слои',
			description: `# Сверточные слои

Научитесь создавать сверточные слои для извлечения признаков в DL4J.

## Задача

Реализуйте конфигурацию слоев CNN:
- Создайте 2D сверточные слои с фильтрами
- Настройте размер ядра, шаг и паддинг
- Объедините несколько сверточных слоев

## Пример

\`\`\`java
ConvolutionLayer convLayer = new ConvolutionLayer.Builder(5, 5)
    .nIn(1)
    .nOut(20)
    .stride(1, 1)
    .activation(Activation.RELU)
    .build();
\`\`\``,
			hint1: 'Используйте ConvolutionLayer.Builder с размером ядра',
			hint2: 'SubsamplingLayer.Builder принимает PoolingType как параметр',
			whyItMatters: `Сверточные слои - основа компьютерного зрения:

- **Извлечение признаков**: Автоматическое обучение визуальным паттернам
- **Трансляционная инвариантность**: Обнаружение признаков независимо от позиции
- **Разделение параметров**: Эффективнее полносвязных слоев
- **Иерархическое обучение**: Сложные признаки через стек слоев`,
		},
		uz: {
			title: 'Konvolyutsion qatlamlar',
			description: `# Konvolyutsion qatlamlar

DL4J da xususiyatlarni ajratib olish uchun konvolyutsion qatlamlarni yaratishni o'rganing.

## Topshiriq

CNN qatlam konfiguratsiyasini amalga oshiring:
- Filtrlar bilan 2D konvolyutsion qatlamlarni yarating
- Yadro o'lchami, qadam va paddingni sozlang
- Bir nechta conv qatlamlarini birlashtiring

## Misol

\`\`\`java
ConvolutionLayer convLayer = new ConvolutionLayer.Builder(5, 5)
    .nIn(1)
    .nOut(20)
    .stride(1, 1)
    .activation(Activation.RELU)
    .build();
\`\`\``,
			hint1: "Yadro o'lchami bilan ConvolutionLayer.Builder dan foydalaning",
			hint2: 'SubsamplingLayer.Builder PoolingType ni parametr sifatida qabul qiladi',
			whyItMatters: `Konvolyutsion qatlamlar kompyuter ko'rish asosi:

- **Xususiyatlarni ajratish**: Vizual patternlarni avtomatik o'rganish
- **Translyatsion invariantlik**: Pozitsiyadan qat'i nazar xususiyatlarni aniqlash
- **Parametrlarni ulashish**: To'liq bog'langan qatlamlarga qaraganda samaraliroq
- **Ierarxik o'rganish**: Qatlamlar stacki orqali murakkab xususiyatlar`,
		},
	},
};

export default task;
