import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-image-augmentation',
	title: 'Image Augmentation',
	difficulty: 'medium',
	tags: ['dl4j', 'augmentation', 'preprocessing'],
	estimatedTime: '20m',
	isPremium: true,
	order: 3,
	description: `# Image Augmentation

Apply data augmentation to improve CNN generalization.

## Task

Implement image transformations:
- Random rotation
- Horizontal/vertical flips
- Random cropping
- Color jittering

## Example

\`\`\`java
ImageTransform flipTransform = new FlipImageTransform(1);
ImageTransform rotateTransform = new RotateImageTransform(15);

List<ImageTransform> pipeline = Arrays.asList(
    flipTransform,
    rotateTransform
);
\`\`\``,

	initialCode: `import org.datavec.image.transform.*;
import org.nd4j.linalg.primitives.Pair;
import java.util.List;
import java.util.Arrays;
import java.util.Random;

public class ImageAugmentor {

    /**
     * Create a basic augmentation pipeline.
     */
    public static List<Pair<ImageTransform, Double>> createBasicPipeline() {
        return null;
    }

    /**
     * Create rotation transform with random angle.
     * @param maxAngle Maximum rotation angle in degrees
     */
    public static ImageTransform createRotationTransform(float maxAngle) {
        return null;
    }

    /**
     * Create a full augmentation pipeline with probability.
     */
    public static PipelineImageTransform createFullPipeline(Random random) {
        return null;
    }
}`,

	solutionCode: `import org.datavec.image.transform.*;
import org.nd4j.linalg.primitives.Pair;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Random;

public class ImageAugmentor {

    /**
     * Create a basic augmentation pipeline.
     */
    public static List<Pair<ImageTransform, Double>> createBasicPipeline() {
        List<Pair<ImageTransform, Double>> pipeline = new ArrayList<>();

        // 50% chance of horizontal flip
        pipeline.add(new Pair<>(new FlipImageTransform(1), 0.5));

        // 30% chance of rotation
        pipeline.add(new Pair<>(new RotateImageTransform(15), 0.3));

        // 40% chance of scaling
        pipeline.add(new Pair<>(new ScaleImageTransform(0.9f), 0.4));

        return pipeline;
    }

    /**
     * Create rotation transform with random angle.
     * @param maxAngle Maximum rotation angle in degrees
     */
    public static ImageTransform createRotationTransform(float maxAngle) {
        return new RotateImageTransform(maxAngle);
    }

    /**
     * Create a full augmentation pipeline with probability.
     */
    public static PipelineImageTransform createFullPipeline(Random random) {
        List<Pair<ImageTransform, Double>> transforms = new ArrayList<>();

        // Horizontal flip
        transforms.add(new Pair<>(new FlipImageTransform(1), 0.5));

        // Vertical flip
        transforms.add(new Pair<>(new FlipImageTransform(0), 0.2));

        // Random rotation up to 20 degrees
        transforms.add(new Pair<>(new RotateImageTransform(random, 20), 0.4));

        // Random scale 0.8-1.2
        transforms.add(new Pair<>(new ScaleImageTransform(random, 0.8f), 0.3));

        // Warp transform for slight distortion
        transforms.add(new Pair<>(new WarpImageTransform(random, 2), 0.2));

        return new PipelineImageTransform(transforms, false);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.datavec.image.transform.*;
import org.nd4j.linalg.primitives.Pair;
import java.util.List;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

public class ImageAugmentorTest {

    @Test
    void testCreateBasicPipeline() {
        List<Pair<ImageTransform, Double>> pipeline = ImageAugmentor.createBasicPipeline();
        assertNotNull(pipeline);
        assertFalse(pipeline.isEmpty());
        assertTrue(pipeline.size() >= 2);
    }

    @Test
    void testCreateRotationTransform() {
        ImageTransform transform = ImageAugmentor.createRotationTransform(30);
        assertNotNull(transform);
        assertTrue(transform instanceof RotateImageTransform);
    }

    @Test
    void testCreateFullPipeline() {
        Random random = new Random(42);
        PipelineImageTransform pipeline = ImageAugmentor.createFullPipeline(random);
        assertNotNull(pipeline);
    }

    @Test
    void testBasicPipelineHasTransforms() {
        List<Pair<ImageTransform, Double>> pipeline = ImageAugmentor.createBasicPipeline();
        assertTrue(pipeline.size() >= 3);
    }

    @Test
    void testRotationTransformWithZeroAngle() {
        ImageTransform transform = ImageAugmentor.createRotationTransform(0);
        assertNotNull(transform);
    }

    @Test
    void testRotationTransformWithLargeAngle() {
        ImageTransform transform = ImageAugmentor.createRotationTransform(180);
        assertNotNull(transform);
    }

    @Test
    void testFullPipelineWithDifferentSeed() {
        Random random1 = new Random(1);
        Random random2 = new Random(2);
        PipelineImageTransform pipeline1 = ImageAugmentor.createFullPipeline(random1);
        PipelineImageTransform pipeline2 = ImageAugmentor.createFullPipeline(random2);
        assertNotNull(pipeline1);
        assertNotNull(pipeline2);
    }

    @Test
    void testBasicPipelineProbabilities() {
        List<Pair<ImageTransform, Double>> pipeline = ImageAugmentor.createBasicPipeline();
        for (Pair<ImageTransform, Double> pair : pipeline) {
            assertTrue(pair.getSecond() >= 0 && pair.getSecond() <= 1);
        }
    }

    @Test
    void testBasicPipelineTransformsNotNull() {
        List<Pair<ImageTransform, Double>> pipeline = ImageAugmentor.createBasicPipeline();
        for (Pair<ImageTransform, Double> pair : pipeline) {
            assertNotNull(pair.getFirst());
        }
    }

    @Test
    void testRotationTransformType() {
        ImageTransform transform = ImageAugmentor.createRotationTransform(45);
        assertTrue(transform instanceof RotateImageTransform);
    }
}`,

	hint1: 'Use Pair<ImageTransform, Double> for transform with probability',
	hint2: 'PipelineImageTransform chains multiple transforms',

	whyItMatters: `Data augmentation is critical for CNN training:

- **Prevent overfitting**: Increase effective dataset size
- **Improve generalization**: Model learns invariant features
- **Real-world robustness**: Handle variations in input
- **Cost-effective**: Better results without more data

Augmentation is standard practice in all production CV systems.`,

	translations: {
		ru: {
			title: 'Аугментация изображений',
			description: `# Аугментация изображений

Применяйте аугментацию данных для улучшения обобщения CNN.

## Задача

Реализуйте трансформации изображений:
- Случайный поворот
- Горизонтальные/вертикальные отражения
- Случайная обрезка
- Изменение цвета

## Пример

\`\`\`java
ImageTransform flipTransform = new FlipImageTransform(1);
ImageTransform rotateTransform = new RotateImageTransform(15);

List<ImageTransform> pipeline = Arrays.asList(
    flipTransform,
    rotateTransform
);
\`\`\``,
			hint1: 'Используйте Pair<ImageTransform, Double> для трансформации с вероятностью',
			hint2: 'PipelineImageTransform объединяет несколько трансформаций',
			whyItMatters: `Аугментация данных критична для обучения CNN:

- **Предотвращение переобучения**: Увеличение эффективного размера датасета
- **Улучшение обобщения**: Модель учится инвариантным признакам
- **Устойчивость в реальном мире**: Обработка вариаций во входных данных
- **Экономичность**: Лучшие результаты без дополнительных данных`,
		},
		uz: {
			title: 'Tasvir augmentatsiyasi',
			description: `# Tasvir augmentatsiyasi

CNN umumlashtirishni yaxshilash uchun data augmentatsiyasini qo'llang.

## Topshiriq

Tasvir transformatsiyalarini amalga oshiring:
- Tasodifiy aylantirish
- Gorizontal/vertikal aylantirish
- Tasodifiy kesish
- Rang o'zgarishi

## Misol

\`\`\`java
ImageTransform flipTransform = new FlipImageTransform(1);
ImageTransform rotateTransform = new RotateImageTransform(15);

List<ImageTransform> pipeline = Arrays.asList(
    flipTransform,
    rotateTransform
);
\`\`\``,
			hint1: "Ehtimollik bilan transformatsiya uchun Pair<ImageTransform, Double> dan foydalaning",
			hint2: "PipelineImageTransform bir nechta transformatsiyalarni bog'laydi",
			whyItMatters: `Data augmentatsiyasi CNN o'qitish uchun juda muhim:

- **Overfittingni oldini olish**: Samarali dataset hajmini oshirish
- **Umumlashtirishni yaxshilash**: Model invariant xususiyatlarni o'rganadi
- **Haqiqiy dunyo barqarorligi**: Kirishdagi o'zgarishlarni boshqarish
- **Tejamkorlik**: Ko'proq datasiz yaxshiroq natijalar`,
		},
	},
};

export default task;
