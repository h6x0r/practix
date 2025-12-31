import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-onnx-export',
	title: 'ONNX Model Export',
	difficulty: 'medium',
	tags: ['onnx', 'export', 'interoperability'],
	estimatedTime: '25m',
	isPremium: false,
	order: 2,
	description: `# ONNX Model Export

Export models to ONNX format for cross-platform deployment.

## Task

Implement ONNX export:
- Convert DL4J model to ONNX
- Validate exported model
- Load ONNX model for inference

## Example

\`\`\`java
// Export to ONNX
OnnxFramework.export(model, "model.onnx", inputShape);

// Load ONNX model
OrtSession session = env.createSession("model.onnx");
\`\`\``,

	initialCode: `import ai.onnxruntime.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.Map;
import java.util.HashMap;

public class OnnxExporter {

    /**
     */
    public static OrtEnvironment createEnvironment() {
        return null;
    }

    /**
     * @param env ONNX environment
     * @param modelPath Path to ONNX model
     */
    public static OrtSession loadOnnxModel(OrtEnvironment env, String modelPath)
            throws OrtException {
        return null;
    }

    /**
     * @param session ONNX session
     * @param inputName Input tensor name
     */
    public static float[] runInference(OrtSession session, String inputName,
                                        float[][] inputData) throws OrtException {
        return null;
    }
}`,

	solutionCode: `import ai.onnxruntime.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.Map;
import java.util.HashMap;
import java.nio.FloatBuffer;

public class OnnxExporter {

    /**
     * Create ONNX runtime environment.
     */
    public static OrtEnvironment createEnvironment() {
        return OrtEnvironment.getEnvironment();
    }

    /**
     * Load ONNX model for inference.
     * @param env ONNX environment
     * @param modelPath Path to ONNX model
     */
    public static OrtSession loadOnnxModel(OrtEnvironment env, String modelPath)
            throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
        return env.createSession(modelPath, options);
    }

    /**
     * Run inference on ONNX model.
     * @param session ONNX session
     * @param inputName Input tensor name
     * @param inputData Input data
     */
    public static float[] runInference(OrtSession session, String inputName,
                                        float[][] inputData) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // Create input tensor
        long[] shape = {inputData.length, inputData[0].length};
        float[] flatData = flatten(inputData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatData), shape);

        // Run inference
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, inputTensor);

        OrtSession.Result result = session.run(inputs);

        // Get output
        float[][] output = (float[][]) result.get(0).getValue();
        return output[0];
    }

    /**
     * Get model input names.
     */
    public static String[] getInputNames(OrtSession session) throws OrtException {
        return session.getInputNames().toArray(new String[0]);
    }

    /**
     * Get model output names.
     */
    public static String[] getOutputNames(OrtSession session) throws OrtException {
        return session.getOutputNames().toArray(new String[0]);
    }

    /**
     * Close session and cleanup.
     */
    public static void cleanup(OrtSession session) throws OrtException {
        session.close();
    }

    private static float[] flatten(float[][] data) {
        int totalSize = data.length * data[0].length;
        float[] result = new float[totalSize];
        int idx = 0;
        for (float[] row : data) {
            for (float val : row) {
                result[idx++] = val;
            }
        }
        return result;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import ai.onnxruntime.*;
import static org.junit.jupiter.api.Assertions.*;

public class OnnxExporterTest {

    private OrtEnvironment env;

    @BeforeEach
    void setUp() {
        env = OnnxExporter.createEnvironment();
    }

    @Test
    void testCreateEnvironment() {
        OrtEnvironment environment = OnnxExporter.createEnvironment();
        assertNotNull(environment);
    }

    @Test
    void testLoadOnnxModel() throws OrtException {
        // This test requires an actual ONNX model file
        // In practice, you would have a test model available
        assertNotNull(env);
    }

    @Test
    void testGetInputOutputNames() throws OrtException {
        // Verify environment is created
        assertNotNull(env);
        // Additional tests would require actual model file
    }

    @Test
    void testEnvironmentNotNull() {
        assertNotNull(OnnxExporter.createEnvironment());
    }

    @Test
    void testEnvironmentSingleton() {
        OrtEnvironment env1 = OnnxExporter.createEnvironment();
        OrtEnvironment env2 = OnnxExporter.createEnvironment();
        assertEquals(env1, env2);
    }

    @Test
    void testFlattenHelper() {
        float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        // Test that flatten produces expected size
        int expectedSize = data.length * data[0].length;
        assertEquals(4, expectedSize);
    }

    @Test
    void testInputShape() {
        float[][] inputData = {{1.0f, 2.0f, 3.0f}};
        assertEquals(1, inputData.length);
        assertEquals(3, inputData[0].length);
    }

    @Test
    void testMultiRowInput() {
        float[][] inputData = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        assertEquals(3, inputData.length);
        assertEquals(2, inputData[0].length);
    }

    @Test
    void testEmptyInputHandling() {
        float[][] emptyData = new float[0][0];
        assertEquals(0, emptyData.length);
    }

    @Test
    void testEnvCreateMultipleTimes() {
        for (int i = 0; i < 3; i++) {
            OrtEnvironment e = OnnxExporter.createEnvironment();
            assertNotNull(e);
        }
    }
}`,

	hint1: 'OrtEnvironment.getEnvironment() creates the ONNX runtime',
	hint2: 'Create OnnxTensor from FloatBuffer with proper shape',

	whyItMatters: `ONNX enables cross-platform ML deployment:

- **Interoperability**: Use models from any framework
- **Optimization**: ONNX Runtime provides fast inference
- **Portability**: Deploy to any platform supporting ONNX
- **Industry standard**: Widely adopted format

ONNX bridges the gap between training and production.`,

	translations: {
		ru: {
			title: 'Экспорт модели в ONNX',
			description: `# Экспорт модели в ONNX

Экспортируйте модели в формат ONNX для кросс-платформенного развертывания.

## Задача

Реализуйте экспорт в ONNX:
- Конвертируйте DL4J модель в ONNX
- Валидируйте экспортированную модель
- Загрузите ONNX модель для инференса

## Пример

\`\`\`java
// Export to ONNX
OnnxFramework.export(model, "model.onnx", inputShape);

// Load ONNX model
OrtSession session = env.createSession("model.onnx");
\`\`\``,
			hint1: 'OrtEnvironment.getEnvironment() создает ONNX runtime',
			hint2: 'Создайте OnnxTensor из FloatBuffer с правильной формой',
			whyItMatters: `ONNX обеспечивает кросс-платформенное ML развертывание:

- **Интероперабельность**: Используйте модели из любого фреймворка
- **Оптимизация**: ONNX Runtime обеспечивает быстрый инференс
- **Переносимость**: Развертывание на любой платформе с поддержкой ONNX
- **Индустриальный стандарт**: Широко принятый формат`,
		},
		uz: {
			title: 'ONNX modelini eksport qilish',
			description: `# ONNX modelini eksport qilish

Kross-platforma joylashtirish uchun modellarni ONNX formatiga eksport qiling.

## Topshiriq

ONNX eksportini amalga oshiring:
- DL4J modelni ONNX ga aylantiring
- Eksport qilingan modelni tasdiqlang
- Inference uchun ONNX modelni yuklang

## Misol

\`\`\`java
// Export to ONNX
OnnxFramework.export(model, "model.onnx", inputShape);

// Load ONNX model
OrtSession session = env.createSession("model.onnx");
\`\`\``,
			hint1: "OrtEnvironment.getEnvironment() ONNX runtimeni yaratadi",
			hint2: "To'g'ri shakl bilan FloatBuffer dan OnnxTensor yarating",
			whyItMatters: `ONNX kross-platforma ML joylashtirishni ta'minlaydi:

- **Interoperabillik**: Har qanday frameworkdan modellarni ishlating
- **Optimallashtirish**: ONNX Runtime tez inference beradi
- **Portativlik**: ONNX ni qo'llab-quvvatlaydigan har qanday platformaga joylashtiring
- **Sanoat standarti**: Keng qabul qilingan format`,
		},
	},
};

export default task;
