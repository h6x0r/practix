# PRACTIX Platform Roadmap

> Last updated: 2026-02-14

---

## Quick Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Platform Core** | DONE | Courses, Auth, Progress tracking |
| **7 Go + Java Courses** | DONE | 423 tasks, 3 languages (EN/RU/UZ) |
| **Piston Code Execution** | DONE | 8 languages, BullMQ queue, Redis caching |
| **Playground (Web IDE)** | DONE | At /playground with snippets library |
| **GoF Design Patterns (Go)** | DONE | 3 modules, 30 tasks |
| **GoF Design Patterns (Java)** | DONE | 3 modules, 30 tasks |
| **Software Engineering** | DONE | 6 modules, 78 tasks |
| **Algo Fundamentals (Python)** | DONE | 7 modules, 56 tasks |
| **Algo Advanced (Python)** | DONE | 6 modules, 63 tasks |
| **Python ML Fundamentals** | DONE | 5 modules, 106 tasks |
| **Python Deep Learning** | DONE | 6 modules, 78 tasks |
| **Python LLM** | DONE | 5 modules, 43 tasks |
| **Java ML** | DONE | 7 modules, 71 tasks |
| **Java NLP** | DONE | 8 modules, 59 tasks |
| **Go ML Inference** | DONE | 7 modules, 52 tasks |
| **Prompt Engineering** | DONE | 8 modules, 57 tasks |
| **Python Fundamentals** | DONE | 5 modules, 61 tasks |
| **Math for Data Science** | DONE | 4 modules, 35 tasks |
| **Application Security** | DONE | 7 modules, 59 tasks |
| **System Design** | Planned | Phase 4 |

---

## Table of Contents

### Platform
1. [Platform Features](#platform-features) - **DONE**
2. [Code Execution Engine](#code-execution-engine) - **DONE**
3. [Playground (Web IDE)](#playground-web-ide) - **DONE**

### Courses
4. [Course Structure](#course-structure) - **DONE** (22 Courses)
5. [Go Courses](#go-courses) - **Production Ready** (4 Courses)
6. [Java Courses](#java-courses) - **Production Ready** (3 Courses)
7. [Design Patterns Courses](#design-patterns-courses) - **DONE** (2 Courses)
8. [Software Engineering](#software-engineering-course) - **DONE**
9. [Algorithms Courses](#algorithms-courses) - **DONE** (2 Courses)
10. [AI/ML Courses](#aiml-courses) - **DONE** (6 Courses)
11. [New Courses (2026)](#new-courses-2026) - **DONE** (4 Courses)
12. [Future Courses](#future-courses) - Planned

### Implementation
13. [Implementation Priorities](#implementation-priorities)
14. [Task File Template](#task-file-template)

---

# Platform Features

## Current State - ALL DONE
- [x] Course system with modules, topics, tasks
- [x] Multi-language support (EN, RU, UZ)
- [x] Code editor with Monaco (syntax highlighting)
- [x] Solution verification
- [x] Progress tracking (from passed submissions)
- [x] Module progress bars
- [x] Shared module architecture (reusable across courses)
- [x] **Piston Code Execution** - 8 languages
- [x] **BullMQ Queue** - Redis-based job queue
- [x] **Execution Caching** - 30min TTL, ~60% load reduction
- [x] **Rate Limiting** - Per-endpoint throttling
- [x] **Playground (Web IDE)** - at /playground
- [x] User authentication (JWT)
- [x] Premium/Free tier system
- [x] AI Tutor integration (Gemini 2.0 Flash)
- [x] Submission history (real API data)

---

# Code Execution Engine

## Status: DONE

Production-grade code execution with Piston + BullMQ + Redis:
- **8 Languages:** Go, Java, JavaScript, TypeScript, Python, Rust, C++, C
- **Resource Limits:** CPU time, memory, process limits
- **Queue Management:** BullMQ with Redis persistence
- **Caching:** SHA-256 hash-based, 30min TTL
- **Rate Limiting:** @nestjs/throttler integration

### Supported Languages

| Language | Time Limit | Memory Limit |
|----------|------------|--------------|
| Go | 5s | 256MB |
| Java | 10s | 512MB |
| JavaScript | 5s | 256MB |
| TypeScript | 10s | 256MB |
| Python | 10s | 256MB |
| Rust | 10s | 256MB |
| C++ | 5s | 256MB |
| C | 5s | 256MB |

### Key Files
```
server/src/judge0/judge0.service.ts       # Judge0 API client
server/src/queue/code-execution.service.ts # BullMQ queue API
server/src/queue/code-execution.processor.ts # Queue worker
server/src/cache/cache.service.ts         # Redis caching
server/src/submissions/                   # Submission handling
```

### Usage
```bash
# Start full stack
make start-docker

# Check status
curl http://localhost:8080/submissions/judge/status
```

> **Note:** Piston works natively on ARM64/Apple Silicon.

---

# Playground (Web IDE)

## Status: DONE

Standalone coding environment at **`/playground`** - no login required.

### Features
- [x] Multi-language support (8 languages)
- [x] Monaco editor with syntax highlighting
- [x] Input (stdin) panel
- [x] Output (stdout/stderr/compile errors) panel
- [x] Code templates for each language
- [x] Language selector dropdown
- [x] Execution status indicator
- [x] Reset to template button
- [ ] Save and share code snippets (future)
- [ ] Multiple files support (future)

### Key Files
```
src/features/playground/
├── api/playgroundService.ts    # API calls to /submissions/run
└── ui/PlaygroundPage.tsx       # Main playground component
```

---

# Course Structure

## Architecture: DONE

```
server/prisma/seeds/
├── types.ts                       # Type definitions
├── courses/
│   ├── index.ts                   # ALL_COURSES export (22 courses)
│   ├── go-basics/                 # Go Basics Course
│   ├── go-concurrency/            # Go Concurrency Course
│   ├── go-web-apis/               # Go Web & APIs Course
│   ├── go-production/             # Go Production Course
│   ├── java-core/                 # Java Core Course
│   ├── java-modern/               # Java Modern Course
│   ├── java-advanced/             # Java Advanced Course
│   ├── go-design-patterns/        # GoF Patterns in Go
│   ├── java-design-patterns/      # GoF Patterns in Java
│   ├── software-engineering/      # Software Engineering
│   ├── algo-fundamentals/         # Algorithms Fundamentals (Python)
│   ├── algo-advanced/             # Algorithms Advanced (Python)
│   ├── c_python_ml_fundamentals/  # Python ML Fundamentals
│   ├── c_python_deep_learning/    # Python Deep Learning
│   ├── c_python_llm/              # Python LLM
│   ├── c_java_ml/                 # Java ML (DJL + Tribuo)
│   ├── c_java_nlp/                # Java NLP
│   ├── c_go_ml_inference/         # Go ML Inference
│   ├── c_prompt_engineering/      # Prompt Engineering
│   ├── c_python_fundamentals/     # Python Fundamentals
│   ├── c_math_for_ds/             # Math for Data Science
│   └── c_app_security/            # Application Security
└── shared/
    └── modules/
        ├── go/                    # 25 reusable Go modules (209 tasks)
        └── java/                  # 35 reusable Java modules (214 tasks)
```

## Summary

| Course | Modules | Tasks | Status |
|--------|---------|-------|--------|
| **Go Basics** | 9 | 63 | DONE |
| **Go Concurrency** | 4 | 42 | DONE |
| **Go Web & APIs** | 6 | 71 | DONE |
| **Go Production** | 6 | 33 | DONE |
| **Java Core** | 7 | 54 | DONE |
| **Java Modern** | 9 | 56 | DONE |
| **Java Advanced** | 19 | 104 | DONE |
| **Go Design Patterns** | 3 | 30 | DONE |
| **Java Design Patterns** | 3 | 30 | DONE |
| **Software Engineering** | 6 | 78 | DONE |
| **Algo Fundamentals** | 7 | 56 | DONE |
| **Algo Advanced** | 6 | 63 | DONE |
| **Python ML Fundamentals** | 5 | 106 | DONE |
| **Python Deep Learning** | 6 | 78 | DONE |
| **Python LLM** | 5 | 43 | DONE |
| **Java ML** | 7 | 71 | DONE |
| **Java NLP** | 8 | 59 | DONE |
| **Go ML Inference** | 7 | 52 | DONE |
| **Prompt Engineering** | 8 | 57 | DONE |
| **Python Fundamentals** | 5 | 61 | DONE |
| **Math for Data Science** | 4 | 35 | DONE |
| **Application Security** | 7 | 59 | DONE |
| **Total** | **146** | **~1301** | **Production** |

---

# Go Courses

## Go Basics
**Modules:** 9 | **Tasks:** 63 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| fundamentals | constructors, data-structures, io-interfaces | 18 |
| error-handling | fundamentals | 7 |
| pointersx | fundamentals | 5 |
| datastructsx | operations | 5 |
| encodingx | json-validation | 5 |
| generics | fundamentals | 10 |
| io-interfaces | implementation | 5 |
| constructor-patterns | implementation | 4 |
| panic-recovery | implementation | 4 |

## Go Concurrency
**Modules:** 4 | **Tasks:** 42 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| concurrency-patterns | context, pipeline, worker-pool | 30 |
| channels | implementation | 4 |
| goroutines | implementation | 4 |
| synchronization | implementation | 4 |

## Go Web & APIs
**Modules:** 6 | **Tasks:** 71 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| http-middleware | fundamentals, advanced | 18 |
| grpc-interceptors | interceptors | 5 |
| database | sql-basics, transactions, connection-pool | 20 |
| logging | implementation | 4 |
| config-management | implementation | 4 |
| testing | unit-tests, table-driven, mocking, benchmarks | 20 |

## Go Production
**Modules:** 6 | **Tasks:** 33 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| circuit-breaker | implementation | 7 |
| rate-limiting | implementation | 6 |
| caching | implementation | 6 |
| retry-patterns | implementation | 5 |
| metrics | implementation | 5 |
| profiling | implementation | 4 |

---

# Java Courses

## Java Core
**Modules:** 7 | **Tasks:** 54 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| syntax-basics | fundamentals | 7 |
| oop-core | fundamentals | 8 |
| interfaces | fundamentals | 8 |
| exception-handling | fundamentals | 8 |
| collections-list | fundamentals | 8 |
| collections-set-map | fundamentals | 8 |
| collections-queue | fundamentals | 7 |

## Java Modern
**Modules:** 9 | **Tasks:** 56 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| generics | fundamentals | 9 |
| lambda-expressions | fundamentals | 9 |
| stream-api | fundamentals | 6 |
| optional | fundamentals | 6 |
| date-time | fundamentals | 7 |
| records | fundamentals | 5 |
| sealed-classes | fundamentals | 5 |
| pattern-matching | fundamentals | 4 |
| virtual-threads | fundamentals | 5 |

## Java Advanced
**Modules:** 19 | **Tasks:** 104 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| threads-basics | fundamentals | 8 |
| executor-service | fundamentals | 5 |
| concurrent-collections | fundamentals | 5 |
| completable-future | fundamentals | 6 |
| locks-advanced | fundamentals | 5 |
| atomic-operations | fundamentals | 6 |
| io-streams | fundamentals | 7 |
| nio | fundamentals | 7 |
| jdbc | fundamentals | 5 |
| connection-pooling | fundamentals | 4 |
| design-patterns | fundamentals | 5 |
| testing | fundamentals | 5 |
| logging | fundamentals | 8 |
| metrics | fundamentals | 7 |
| config-management | fundamentals | 4 |
| error-handling | fundamentals | 4 |
| retry-resilience | fundamentals | 5 |
| caching | fundamentals | 4 |
| http-clients | fundamentals | 4 |

---

# Design Patterns Courses

## Go Design Patterns (GoF)
**Modules:** 3 | **Tasks:** 23 | **Status:** DONE

| Module | Patterns | Tasks |
|--------|----------|-------|
| creational | Singleton, Factory, Abstract Factory, Builder, Prototype | 5 |
| structural | Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy | 7 |
| behavioral | Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor, Interpreter | 11 |

## Java Design Patterns (GoF)
**Modules:** 3 | **Tasks:** 23 | **Status:** DONE

| Module | Patterns | Tasks |
|--------|----------|-------|
| creational | Singleton, Factory, Abstract Factory, Builder, Prototype | 5 |
| structural | Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy | 7 |
| behavioral | Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor, Interpreter | 11 |

---

# Software Engineering Course

**Modules:** 6 | **Tasks:** ~65 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| solid | Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion | 10 |
| clean-code | Naming, Functions, Comments, Formatting, Error Handling, Classes | 12 |
| refactoring | Extract Method, Replace Conditional, Composition over Inheritance, Move Method, Inline Method, Parameter Object | 12 |
| grasp | Information Expert, Creator, Controller, Low Coupling, High Cohesion, Polymorphism, Pure Fabrication, Indirection, Protected Variations | 9 |
| anti-patterns | God Object, Spaghetti Code, Copy-Paste, Magic Numbers, Golden Hammer, Premature Optimization, Cargo Cult, Lava Flow, Dead Code, Feature Envy, Primitive Obsession, Shotgun Surgery | 12 |
| api-design | RESTful Design, Versioning, Error Handling, Pagination, Rate Limiting, Documentation, Consistency, Authentication, Idempotency, HATEOAS | 10 |

---

# Algorithms Courses

## Algo Fundamentals (Python)
**Modules:** 7 | **Tasks:** 56 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| arrays | Array Basics | 6 |
| strings | String Basics | 6 |
| linked-lists | Linked List Basics | 6 |
| stacks-queues | Stack and Queue Basics | 6 |
| trees | Binary Tree Basics | 6 |
| sorting | Sorting Algorithms | 5 |
| searching | Binary Search & Variations | 6 |

## Algo Advanced (Python)
**Modules:** 6 | **Tasks:** 63 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| dynamic-programming | DP Techniques | 10 |
| graphs | Graph Algorithms | 10 |
| backtracking | Backtracking Techniques | 8 |
| greedy | Greedy Algorithms | 8 |
| divide-conquer | Divide & Conquer | 6 |
| bit-manipulation | Bit Operations | 8 |

---

# AI/ML Courses

## Python ML Fundamentals
**Modules:** 5 | **Tasks:** 106 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| numpy-essentials | Array Basics, Operations, Linear Algebra | 15 |
| pandas-mastery | DataFrame Basics, Manipulation, Aggregation | 15 |
| data-visualization | Matplotlib, Seaborn, Plotly | 6 |
| classical-ml | Supervised, Unsupervised, Evaluation | 10 |
| gradient-boosting | XGBoost, LightGBM, CatBoost | 5 |

## Python Deep Learning
**Modules:** 6 | **Tasks:** 78 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| nn-basics | Neural Network Fundamentals | 10 |
| pytorch-fundamentals | PyTorch Tensors, Autograd | 12 |
| convolutional-networks | CNN Architectures, Image Processing | 12 |
| recurrent-networks | RNN, LSTM, Sequence Models | 10 |
| transfer-learning | Fine-tuning, Feature Extraction | 10 |
| model-deployment | ONNX, TorchServe, Optimization | 7 |

## Python LLM
**Modules:** 5 | **Tasks:** 43 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| transformer-architecture | Attention, Transformers | 10 |
| huggingface-transformers | Models, Tokenizers, Pipelines | 12 |
| fine-tuning | LoRA, PEFT, QLoRA | 10 |
| langchain-basics | Chains, Agents, Memory | 10 |
| rag-systems | Retrieval, Vector Stores | 9 |

## Java ML (DJL + Tribuo)
**Modules:** 7 | **Tasks:** 71 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| djl-fundamentals | NDArray, Models, Inference | 12 |
| tribuo-ml | Classification, Regression | 12 |
| model-inference | ONNX Runtime, Optimization | 10 |
| feature-engineering | Preprocessing, Transformation | 10 |
| ml-pipelines | Training, Evaluation Pipelines | 7 |

## Java NLP
**Modules:** 8 | **Tasks:** 59 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| text-preprocessing | Tokenization, Cleaning | 10 |
| nlp-fundamentals | Embeddings, Word2Vec | 12 |
| text-classification | Sentiment, Topic Modeling | 10 |
| named-entity-recognition | NER, Entity Extraction | 10 |
| language-models | Transformers, Fine-tuning | 9 |

## Go ML Inference
**Modules:** 7 | **Tasks:** 52 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| onnx-inference | ONNX Runtime in Go | 12 |
| tensorflow-go | TensorFlow Serving | 10 |
| ml-api-design | REST APIs for ML | 10 |
| model-optimization | Quantization, Pruning | 10 |
| production-ml | Monitoring, Logging | 9 |

---

# New Courses (2026)

## Prompt Engineering
**Modules:** 8 | **Tasks:** 57 | **Status:** DONE
**Special Feature:** 100 AI Tutor requests/day for this course

| Module | Topics | Tasks |
|--------|--------|-------|
| fundamentals | Prompt structure, Context, Instructions | ~7 |
| zero-few-shot | Zero-shot, Few-shot examples, Formatting | ~8 |
| chain-of-thought | Step-by-step reasoning, Self-consistency | ~7 |
| structured-output | JSON mode, Schema definition, Parsing | ~7 |
| role-playing | System prompts, Personas, Constraints | ~7 |
| rag-basics | Context injection, Chunking strategies | ~7 |
| prompt-security | Injection attacks, Jailbreaks, Defense | ~7 |
| multimodal | Image prompts, Vision analysis | ~7 |

## Python Fundamentals
**Modules:** 5 | **Tasks:** 61 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| basics | Variables, Types, Control Flow | ~12 |
| data-structures | Lists, Dicts, Sets, Tuples | ~12 |
| functions | Functions, Closures, Decorators | ~12 |
| oop | Classes, Inheritance, Polymorphism | ~12 |
| advanced | Generators, Context Managers, Itertools | ~13 |

## Math for Data Science
**Modules:** 4 | **Tasks:** 35 | **Status:** DONE
**Language:** Python (NumPy, SciPy, SymPy)

| Module | Topics | Tasks |
|--------|--------|-------|
| linear-algebra | Vectors, Matrices, Eigenvalues, SVD | ~9 |
| calculus | Derivatives, Integrals, Optimization | ~9 |
| probability-statistics | Distributions, Hypothesis Testing | ~9 |
| numerical-methods | Root Finding, Integration, ODEs | ~8 |

## Application Security
**Modules:** 7 | **Tasks:** 59 | **Status:** DONE

| Module | Topics | Tasks |
|--------|--------|-------|
| owasp-top-10 | Injection, XSS, CSRF, SSRF | ~9 |
| authentication | JWT, OAuth, Session Management | ~9 |
| authorization | RBAC, ABAC, Permission Models | ~8 |
| cryptography | Hashing, Encryption, Key Management | ~8 |
| secure-coding | Input Validation, Output Encoding | ~9 |
| api-security | Rate Limiting, CORS, API Keys | ~8 |
| security-testing | SAST, DAST, Penetration Testing | ~8 |

---

# Future Courses

## System Design
**Status:** Planned | **Tasks:** 30+ | **Priority:** Medium

```
system-design/
├── fundamentals/            # CAP, ACID, BASE, Scalability
├── building-blocks/         # Load Balancer, CDN, Cache, DB
├── case-studies/            # URL Shortener, Twitter, Uber
└── interviews/              # Common questions, Frameworks
```

## Data Engineering (Python)
**Status:** Planned | **Tasks:** 50+ | **Priority:** Medium
**Language:** Python (with SQLite for SQL tasks)

```
data-engineering/
├── sql-advanced/            # Window functions, CTEs, Query optimization (8-10 tasks)
├── pandas-basics/           # DataFrames, Series, Indexing, Selection (8-10 tasks)
├── data-cleaning/           # Missing values, Duplicates, Type conversion (6-8 tasks)
├── etl-patterns/            # Extract, Transform, Load pipelines (6-8 tasks)
├── file-formats/            # CSV, JSON, Parquet, Avro reading/writing (5-6 tasks)
├── data-validation/         # Schema validation, Data quality checks (5-6 tasks)
├── batch-processing/        # Chunking, Memory optimization, Parallelization (5-6 tasks)
└── data-modeling/           # Star schema, Normalization, Denormalization (5-6 tasks)
```

---

# Implementation Priorities

## Phase 1: Platform - DONE
- [x] Course restructuring (7 courses)
- [x] Shared module architecture
- [x] All translations (EN, RU, UZ)
- [x] Piston Code Execution (8 languages)
- [x] BullMQ Queue + Redis Caching
- [x] Rate Limiting
- [x] Playground (Web IDE)

## Phase 2: Content Expansion - DONE
- [x] GoF Design Patterns (Go + Java) - 60 tasks
- [x] Software Engineering Principles - 78 tasks
- [x] Algo Fundamentals + Advanced (Python) - 119 tasks
- [x] AI/ML Courses (6 courses) - 409 tasks
- [x] Real API data for submissions
- [x] Progress tracking cleanup

## Phase 3: New Courses - DONE
- [x] Prompt Engineering - 57 tasks
- [x] Python Fundamentals - 61 tasks
- [x] Math for Data Science - 35 tasks
- [x] Application Security - 59 tasks

## Phase 4: Planned
- [ ] System Design - 30+ tasks
- [ ] Data Engineering (Python) - 50+ tasks

## Future: Performance Optimization
- [ ] WebAssembly browser execution for playground (Pyodide, TinyGo WASM)
- [ ] Kubernetes migration for horizontal scaling

---

# Task File Template

```typescript
import { Task } from '../../../../../../../types';

export const task: Task = {
    slug: 'course-module-task-name',
    title: 'Task Title',
    difficulty: 'easy' | 'medium' | 'hard',
    tags: ['language', 'topic1', 'topic2'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Task description with requirements...`,
    initialCode: `// Initial code template`,
    solutionCode: `// Solution with English comments`,
    hint1: `First hint...`,
    hint2: `Second hint...`,
    whyItMatters: `Why this concept is important...`,
    order: 0,
    translations: {
        ru: {
            title: 'Russian title',
            solutionCode: `// Solution with Russian comments`,
            description: `Russian description...`,
            hint1: `Russian hint 1...`,
            hint2: `Russian hint 2...`,
            whyItMatters: `Russian explanation...`
        },
        uz: {
            title: `Uzbek title`,
            solutionCode: `// Solution with Uzbek comments`,
            description: `Uzbek description...`,
            hint1: `Uzbek hint 1...`,
            hint2: `Uzbek hint 2...`,
            whyItMatters: `Uzbek explanation...`
        }
    }
};
```

---

## Notes

1. **Translation Requirements:** All tasks must have complete ru/uz translations including solutionCode with translated comments
2. **Task Quality:** Each task should teach one specific concept with clear requirements
3. **Difficulty Progression:** Topics should progress from easy to hard within each module
4. **Real-World Focus:** Tasks should reflect production patterns and best practices
5. **Code Comments:** English for main solutionCode, Russian for ru.solutionCode, Uzbek for uz.solutionCode

---

## Statistics

### Current State (Production) — 22 Courses, ~1301 Tasks

| Category | Courses | Modules | Tasks |
|----------|---------|---------|-------|
| Go Courses | 4 | 25 | 209 |
| Java Courses | 3 | 35 | 214 |
| GoF Design Patterns | 2 | 6 | 60 |
| Software Engineering | 1 | 6 | 78 |
| Algorithms (Python) | 2 | 13 | 119 |
| AI/ML Courses | 6 | 38 | 409 |
| Prompt Engineering | 1 | 8 | 57 |
| Python Fundamentals | 1 | 5 | 61 |
| Math for Data Science | 1 | 4 | 35 |
| Application Security | 1 | 7 | 59 |
| **Total** | **22** | **147** | **~1301** |

### Planned (Future Courses)

| Course | Modules | Tasks | Priority |
|--------|---------|-------|----------|
| System Design | 4 | 30+ | Medium |
| Data Engineering | 8 | 50+ | Medium |

---

*Last updated: 2026-02-14*
