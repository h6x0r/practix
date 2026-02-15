/**
 * All Courses Index
 */

// Go Courses (4 courses + Design Patterns)
import { goBasicsCourse } from "./go-basics";
import { goConcurrencyCourse } from "./go-concurrency";
import { goWebApisCourse } from "./go-web-apis";
import { goProductionCourse } from "./go-production";
import { goDesignPatternsCourse } from "./go-design-patterns";

// Java Courses (3 courses + Design Patterns)
import { javaCoreCourse } from "./java-core";
import { javaModernCourse } from "./java-modern";
import { javaAdvancedCourse } from "./java-advanced";
import { javaDesignPatternsCourse } from "./java-design-patterns";

// Python Courses (AI/ML)
import pythonMlFundamentals from "./c_python_ml_fundamentals";
import pythonDeepLearning from "./c_python_deep_learning";
import pythonLlm from "./c_python_llm";

// Java AI/ML Courses
import javaMl from "./c_java_ml";
import javaNlp from "./c_java_nlp";

// Go AI/ML Courses
import goMlInference from "./c_go_ml_inference";

// Computer Science Courses
import { softwareEngineeringCourse } from "./software-engineering";
import { algoFundamentalsCourse } from "./algo-fundamentals";
import { algoAdvancedCourse } from "./algo-advanced";

// Prompt Engineering Course
import { promptEngineeringCourse } from "./c_prompt_engineering";

// Python Fundamentals Course
import { pythonFundamentalsCourse } from "./c_python_fundamentals";

// Math for Data Science Course
import { mathForDsCourse } from "./c_math_for_ds";

// Application Security Course
import { appSecurityCourse } from "./c_app_security";

export const ALL_COURSES = [
  // Go Courses
  goBasicsCourse,
  goConcurrencyCourse,
  goWebApisCourse,
  goProductionCourse,
  goDesignPatternsCourse,

  // Java Courses
  javaCoreCourse,
  javaModernCourse,
  javaAdvancedCourse,
  javaDesignPatternsCourse,

  // Python Courses (AI/ML)
  pythonMlFundamentals,
  pythonDeepLearning,
  pythonLlm,

  // Java AI/ML Courses
  javaMl,
  javaNlp,

  // Go AI/ML Courses
  goMlInference,

  // Computer Science
  softwareEngineeringCourse,
  algoFundamentalsCourse,
  algoAdvancedCourse,

  // Prompt Engineering
  promptEngineeringCourse,

  // Python Fundamentals
  pythonFundamentalsCourse,

  // Math for Data Science
  mathForDsCourse,

  // Application Security
  appSecurityCourse,
];
