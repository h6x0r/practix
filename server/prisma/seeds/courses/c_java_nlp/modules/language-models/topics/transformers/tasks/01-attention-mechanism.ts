import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jnlp-attention-mechanism',
	title: 'Attention Mechanism',
	difficulty: 'hard',
	tags: ['nlp', 'transformers', 'attention', 'deep-learning'],
	estimatedTime: '30m',
	isPremium: true,
	order: 1,
	description: `# Attention Mechanism

Implement the core attention mechanism used in transformers.

## Task

Build attention mechanism:
- Query, Key, Value projections
- Scaled dot-product attention
- Multi-head attention

## Example

\`\`\`java
Attention attn = new Attention(dim=64, numHeads=8);
double[][] output = attn.forward(query, key, value);
\`\`\``,

	initialCode: `import java.util.*;

public class Attention {

    private int dim;
    private int numHeads;

    /**
     */
    public Attention(int dim, int numHeads) {
    }

    /**
     */
    public double[][] scaledDotProduct(double[][] Q, double[][] K, double[][] V) {
        return null;
    }

    /**
     */
    public double[] softmax(double[] x) {
        return null;
    }

    /**
     */
    public double[][] matmul(double[][] A, double[][] B) {
        return null;
    }
}`,

	solutionCode: `import java.util.*;

public class Attention {

    private int dim;
    private int numHeads;
    private int headDim;

    /**
     * Initialize attention with dimension and heads.
     */
    public Attention(int dim, int numHeads) {
        this.dim = dim;
        this.numHeads = numHeads;
        this.headDim = dim / numHeads;
    }

    /**
     * Compute scaled dot-product attention.
     */
    public double[][] scaledDotProduct(double[][] Q, double[][] K, double[][] V) {
        int seqLen = Q.length;
        double scale = Math.sqrt(headDim);

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        double[][] scores = new double[seqLen][seqLen];
        double[][] Kt = transpose(K);

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double sum = 0;
                for (int k = 0; k < Q[i].length; k++) {
                    sum += Q[i][k] * Kt[k][j];
                }
                scores[i][j] = sum / scale;
            }
        }

        // Apply softmax
        double[][] attnWeights = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            attnWeights[i] = softmax(scores[i]);
        }

        // Multiply by V
        return matmul(attnWeights, V);
    }

    /**
     * Apply softmax to attention scores.
     */
    public double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(0);
        double[] exp = new double[x.length];
        double sum = 0;

        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }

        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = exp[i] / sum;
        }
        return result;
    }

    /**
     * Matrix multiplication.
     */
    public double[][] matmul(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int k = B.length;
        double[][] C = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int p = 0; p < k; p++) {
                    C[i][j] += A[i][p] * B[p][j];
                }
            }
        }
        return C;
    }

    /**
     * Transpose matrix.
     */
    public double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] T = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }

    /**
     * Self-attention (Q=K=V from same input).
     */
    public double[][] selfAttention(double[][] X) {
        return scaledDotProduct(X, X, X);
    }

    /**
     * Get attention weights for visualization.
     */
    public double[][] getAttentionWeights(double[][] Q, double[][] K) {
        int seqLen = Q.length;
        double scale = Math.sqrt(headDim);
        double[][] Kt = transpose(K);

        double[][] scores = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double sum = 0;
                for (int k = 0; k < Q[i].length; k++) {
                    sum += Q[i][k] * Kt[k][j];
                }
                scores[i][j] = sum / scale;
            }
            scores[i] = softmax(scores[i]);
        }
        return scores;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AttentionTest {

    @Test
    void testSoftmax() {
        Attention attn = new Attention(64, 8);
        double[] x = {1.0, 2.0, 3.0};
        double[] result = attn.softmax(x);

        double sum = 0;
        for (double r : result) sum += r;
        assertEquals(1.0, sum, 0.001);

        assertTrue(result[2] > result[1]);
        assertTrue(result[1] > result[0]);
    }

    @Test
    void testMatmul() {
        Attention attn = new Attention(64, 8);
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] C = attn.matmul(A, B);

        assertEquals(19, C[0][0], 0.001);
        assertEquals(22, C[0][1], 0.001);
    }

    @Test
    void testTranspose() {
        Attention attn = new Attention(64, 8);
        double[][] A = {{1, 2, 3}, {4, 5, 6}};
        double[][] T = attn.transpose(A);

        assertEquals(2, T.length);
        assertEquals(3, T.length);
        assertEquals(4, T[0][1], 0.001);
    }

    @Test
    void testScaledDotProduct() {
        Attention attn = new Attention(4, 1);
        double[][] Q = {{1, 0, 0, 0}, {0, 1, 0, 0}};
        double[][] K = {{1, 0, 0, 0}, {0, 1, 0, 0}};
        double[][] V = {{1, 1, 1, 1}, {2, 2, 2, 2}};

        double[][] output = attn.scaledDotProduct(Q, K, V);
        assertEquals(2, output.length);
    }

    @Test
    void testSelfAttention() {
        Attention attn = new Attention(4, 1);
        double[][] X = {{1, 0, 0, 0}, {0, 1, 0, 0}};
        double[][] output = attn.selfAttention(X);
        assertEquals(2, output.length);
        assertEquals(4, output[0].length);
    }

    @Test
    void testGetAttentionWeights() {
        Attention attn = new Attention(4, 1);
        double[][] Q = {{1, 0, 0, 0}};
        double[][] K = {{1, 0, 0, 0}};
        double[][] weights = attn.getAttentionWeights(Q, K);
        assertEquals(1, weights.length);
    }

    @Test
    void testSoftmaxSumsToOne() {
        Attention attn = new Attention(64, 8);
        double[] x = {0.5, 1.5, 2.5, 3.5};
        double[] result = attn.softmax(x);
        double sum = 0;
        for (double r : result) sum += r;
        assertEquals(1.0, sum, 0.001);
    }

    @Test
    void testMatmulDimensions() {
        Attention attn = new Attention(64, 8);
        double[][] A = {{1, 2, 3}};
        double[][] B = {{1}, {2}, {3}};
        double[][] C = attn.matmul(A, B);
        assertEquals(1, C.length);
        assertEquals(1, C[0].length);
    }

    @Test
    void testTransposeDimensions() {
        Attention attn = new Attention(64, 8);
        double[][] A = {{1, 2, 3, 4}};
        double[][] T = attn.transpose(A);
        assertEquals(4, T.length);
        assertEquals(1, T[0].length);
    }

    @Test
    void testSoftmaxNegativeValues() {
        Attention attn = new Attention(64, 8);
        double[] x = {-1.0, -2.0, -3.0};
        double[] result = attn.softmax(x);
        double sum = 0;
        for (double r : result) sum += r;
        assertEquals(1.0, sum, 0.001);
        assertTrue(result[0] > result[1]);
    }
}`,

	hint1: 'Attention = softmax(QK^T / sqrt(d_k)) * V',
	hint2: 'Apply softmax row-wise to get attention weights that sum to 1',

	whyItMatters: `Attention is the core of modern NLP:

- **Transformers**: BERT, GPT, T5 all use attention
- **Long-range dependencies**: Connects any two positions in sequence
- **Parallelizable**: Unlike RNNs, all positions computed at once
- **Interpretable**: Attention weights show what model focuses on

Understanding attention is essential for modern NLP.`,

	translations: {
		ru: {
			title: 'Механизм внимания',
			description: `# Механизм внимания

Реализуйте основной механизм внимания используемый в трансформерах.

## Задача

Создайте механизм внимания:
- Query, Key, Value проекции
- Масштабированное скалярное произведение
- Multi-head внимание

## Пример

\`\`\`java
Attention attn = new Attention(dim=64, numHeads=8);
double[][] output = attn.forward(query, key, value);
\`\`\``,
			hint1: 'Attention = softmax(QK^T / sqrt(d_k)) * V',
			hint2: 'Применяйте softmax построчно чтобы веса внимания суммировались в 1',
			whyItMatters: `Внимание - основа современного NLP:

- **Трансформеры**: BERT, GPT, T5 все используют внимание
- **Дальние зависимости**: Соединяет любые две позиции в последовательности
- **Параллелизуемый**: В отличие от RNN, все позиции вычисляются сразу
- **Интерпретируемый**: Веса внимания показывают на что модель обращает внимание`,
		},
		uz: {
			title: 'Attention mexanizmi',
			description: `# Attention mexanizmi

Transformerlarda ishlatiladigan asosiy attention mexanizmini amalga oshiring.

## Topshiriq

Attention mexanizmini yarating:
- Query, Key, Value proektsiyalari
- Masshtablangan nuqtali ko'paytma
- Multi-head attention

## Misol

\`\`\`java
Attention attn = new Attention(dim=64, numHeads=8);
double[][] output = attn.forward(query, key, value);
\`\`\``,
			hint1: 'Attention = softmax(QK^T / sqrt(d_k)) * V',
			hint2: "Attention vaznlari 1 ga yig'ilishi uchun softmax ni qatorlar bo'yicha qo'llang",
			whyItMatters: `Attention zamonaviy NLP ning asosi:

- **Transformerlar**: BERT, GPT, T5 hammasi attention dan foydalanadi
- **Uzoq masofali bog'liqliklar**: Ketma-ketlikdagi har qanday ikkita pozitsiyani bog'laydi
- **Parallellashtiriladi**: RNN lardan farqli, barcha pozitsiyalar bir vaqtda hisoblanadi
- **Tushunarli**: Attention vaznlari model nimaga e'tibor qaratayotganini ko'rsatadi`,
		},
	},
};

export default task;
