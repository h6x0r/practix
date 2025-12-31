import onnxRuntime from './01-onnx-runtime';
import tfliteRuntime from './02-tflite-runtime';
import customWeights from './03-custom-weights';
import modelValidation from './04-model-validation';

export default [onnxRuntime, tfliteRuntime, customWeights, modelValidation];
