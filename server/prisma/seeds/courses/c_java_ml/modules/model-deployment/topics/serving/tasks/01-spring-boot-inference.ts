import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-spring-boot-inference',
	title: 'Spring Boot Inference Service',
	difficulty: 'medium',
	tags: ['spring-boot', 'rest-api', 'inference'],
	estimatedTime: '30m',
	isPremium: false,
	order: 1,
	description: `# Spring Boot Inference Service

Build a REST API for ML model inference with Spring Boot.

## Task

Create an inference service:
- Load model on startup
- Expose prediction endpoint
- Handle request/response DTOs

## Example

\`\`\`java
@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    @PostMapping
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        // Run inference
        return new PredictionResponse(prediction);
    }
}
\`\`\``,

	initialCode: `import org.springframework.web.bind.annotation.*;
import org.springframework.stereotype.Service;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import javax.annotation.PostConstruct;

class PredictionRequest {
    private double[] features;

    public double[] getFeatures() { return features; }
    public void setFeatures(double[] features) { this.features = features; }
}

class PredictionResponse {
    private int predictedClass;
    private double[] probabilities;

}

@Service
public class ModelService {

    private MultiLayerNetwork model;

    /**
     */
    @PostConstruct
    public void loadModel() {
    }

    /**
     */
    public PredictionResponse predict(double[] features) {
        return null;
    }
}

@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    /**
     */
    @PostMapping
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        return null;
    }
}`,

	solutionCode: `import org.springframework.web.bind.annotation.*;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import javax.annotation.PostConstruct;
import java.io.File;

// Request DTO
class PredictionRequest {
    private double[] features;

    public double[] getFeatures() { return features; }
    public void setFeatures(double[] features) { this.features = features; }
}

// Response DTO
class PredictionResponse {
    private int predictedClass;
    private double[] probabilities;

    public PredictionResponse(int predictedClass, double[] probabilities) {
        this.predictedClass = predictedClass;
        this.probabilities = probabilities;
    }

    public int getPredictedClass() { return predictedClass; }
    public double[] getProbabilities() { return probabilities; }
}

@Service
public class ModelService {

    private MultiLayerNetwork model;

    @Value("\${model.path:model.zip}")
    private String modelPath;

    /**
     * Load model on service startup.
     */
    @PostConstruct
    public void loadModel() {
        try {
            this.model = ModelSerializer.restoreMultiLayerNetwork(
                new File(modelPath)
            );
            System.out.println("Model loaded successfully from: " + modelPath);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }

    /**
     * Run prediction on input features.
     */
    public PredictionResponse predict(double[] features) {
        // Create input tensor
        INDArray input = Nd4j.create(features).reshape(1, features.length);

        // Run inference
        INDArray output = model.output(input);

        // Get predicted class
        int predictedClass = Nd4j.argMax(output, 1).getInt(0);

        // Get probabilities
        double[] probabilities = output.toDoubleVector();

        return new PredictionResponse(predictedClass, probabilities);
    }

    /**
     * Check if model is loaded.
     */
    public boolean isModelLoaded() {
        return model != null;
    }
}

@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    @Autowired
    private ModelService modelService;

    /**
     * Prediction endpoint.
     */
    @PostMapping
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        if (request.getFeatures() == null || request.getFeatures().length == 0) {
            throw new IllegalArgumentException("Features cannot be empty");
        }
        return modelService.predict(request.getFeatures());
    }

    /**
     * Health check endpoint.
     */
    @GetMapping("/health")
    public String health() {
        return modelService.isModelLoaded() ? "OK" : "Model not loaded";
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class PredictionServiceTest {

    @Test
    void testPredictionRequest() {
        PredictionRequest request = new PredictionRequest();
        double[] features = {1.0, 2.0, 3.0};
        request.setFeatures(features);

        assertArrayEquals(features, request.getFeatures());
    }

    @Test
    void testPredictionResponse() {
        double[] probs = {0.1, 0.9};
        PredictionResponse response = new PredictionResponse(1, probs);

        assertEquals(1, response.getPredictedClass());
        assertArrayEquals(probs, response.getProbabilities());
    }

    @Test
    void testEmptyFeaturesValidation() {
        PredictionRequest request = new PredictionRequest();
        request.setFeatures(new double[]{});

        // Controller should validate empty features
        assertTrue(request.getFeatures().length == 0);
    }

    @Test
    void testPredictionResponseClass() {
        PredictionResponse response = new PredictionResponse(0, new double[]{0.8, 0.2});
        assertEquals(0, response.getPredictedClass());
    }

    @Test
    void testPredictionResponseProbabilities() {
        double[] probs = {0.3, 0.7};
        PredictionResponse response = new PredictionResponse(1, probs);
        assertEquals(2, response.getProbabilities().length);
    }

    @Test
    void testPredictionRequestFeatures() {
        PredictionRequest request = new PredictionRequest();
        request.setFeatures(new double[]{0.5, 0.5, 0.5});
        assertEquals(3, request.getFeatures().length);
    }

    @Test
    void testMultipleFeatures() {
        PredictionRequest request = new PredictionRequest();
        double[] features = {1.0, 2.0, 3.0, 4.0, 5.0};
        request.setFeatures(features);
        assertEquals(5, request.getFeatures().length);
    }

    @Test
    void testPredictionResponseNotNull() {
        PredictionResponse response = new PredictionResponse(0, new double[]{1.0});
        assertNotNull(response.getProbabilities());
    }

    @Test
    void testPredictionRequestSetGet() {
        PredictionRequest request = new PredictionRequest();
        double[] input = {0.1, 0.2};
        request.setFeatures(input);
        assertArrayEquals(input, request.getFeatures());
    }

    @Test
    void testResponseWithMultipleClasses() {
        double[] probs = {0.1, 0.2, 0.3, 0.4};
        PredictionResponse response = new PredictionResponse(3, probs);
        assertEquals(4, response.getProbabilities().length);
        assertEquals(3, response.getPredictedClass());
    }
}`,

	hint1: 'Use @PostConstruct to load model when service starts',
	hint2: 'Nd4j.create() converts double array to INDArray for inference',

	whyItMatters: `REST APIs are the standard for ML serving:

- **Accessibility**: Any client can call the API
- **Scalability**: Horizontal scaling with load balancing
- **Decoupling**: Separate model from application
- **Standardization**: Consistent interface for predictions

Spring Boot is the most popular Java framework for APIs.`,

	translations: {
		ru: {
			title: 'Spring Boot сервис инференса',
			description: `# Spring Boot сервис инференса

Создайте REST API для инференса ML моделей с Spring Boot.

## Задача

Создайте сервис инференса:
- Загрузите модель при старте
- Предоставьте endpoint для предсказаний
- Обработайте request/response DTO

## Пример

\`\`\`java
@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    @PostMapping
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        // Run inference
        return new PredictionResponse(prediction);
    }
}
\`\`\``,
			hint1: 'Используйте @PostConstruct для загрузки модели при старте сервиса',
			hint2: 'Nd4j.create() конвертирует double массив в INDArray для инференса',
			whyItMatters: `REST API - стандарт для обслуживания ML:

- **Доступность**: Любой клиент может вызвать API
- **Масштабируемость**: Горизонтальное масштабирование с балансировкой
- **Декаплинг**: Отделение модели от приложения
- **Стандартизация**: Единый интерфейс для предсказаний`,
		},
		uz: {
			title: "Spring Boot inference xizmati",
			description: `# Spring Boot inference xizmati

Spring Boot bilan ML model inference uchun REST API yarating.

## Topshiriq

Inference xizmatini yarating:
- Ishga tushirishda modelni yuklang
- Bashorat endpointini expose qiling
- Request/response DTOlarni boshqaring

## Misol

\`\`\`java
@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    @PostMapping
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        // Run inference
        return new PredictionResponse(prediction);
    }
}
\`\`\``,
			hint1: "Xizmat ishga tushganda modelni yuklash uchun @PostConstruct dan foydalaning",
			hint2: "Nd4j.create() double massivni inference uchun INDArray ga aylantiradi",
			whyItMatters: `REST APIlar ML xizmat qilish uchun standart:

- **Foydalanish imkoniyati**: Har qanday klient APIni chaqirishi mumkin
- **Masshtablanish**: Load balancing bilan gorizontal masshtablash
- **Ajratish**: Modelni ilovadan ajratish
- **Standartlashtirish**: Bashoratlar uchun izchil interfeys`,
		},
	},
};

export default task;
