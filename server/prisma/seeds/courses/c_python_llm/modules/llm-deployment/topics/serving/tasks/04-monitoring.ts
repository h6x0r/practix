import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-llm-monitoring',
	title: 'LLM Monitoring',
	difficulty: 'medium',
	tags: ['monitoring', 'metrics', 'production'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# LLM Monitoring

Implement monitoring for production LLM systems.

## Key Metrics

**Performance:**
- Latency (p50, p95, p99)
- Throughput (requests/second, tokens/second)
- GPU utilization

**Quality:**
- Token usage
- Error rates
- Output length distribution

**Cost:**
- Cost per request
- Token costs over time

## Example

\`\`\`python
from prometheus_client import Counter, Histogram

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds'
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated'
)
\`\`\``,

	initialCode: `import time
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import statistics

@dataclass
class RequestMetrics:
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    success: bool
    timestamp: float

class LLMMonitor:
    """Monitor LLM performance and usage."""

    def __init__(self, window_size: int = 1000):
        # Your code here
        pass

    def record_request(self, metrics: RequestMetrics):
        """Record a request's metrics."""
        # Your code here
        pass

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency percentiles (p50, p95, p99)."""
        # Your code here
        pass

    def get_throughput(self) -> Dict[str, float]:
        """Get throughput metrics."""
        # Your code here
        pass

    def get_error_rate(self) -> float:
        """Get error rate."""
        # Your code here
        pass

    def get_summary(self) -> Dict:
        """Get complete monitoring summary."""
        # Your code here
        pass
`,

	solutionCode: `import time
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import statistics

@dataclass
class RequestMetrics:
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    success: bool
    timestamp: float

class LLMMonitor:
    """Monitor LLM performance and usage."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.requests: deque = deque(maxlen=window_size)
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()

    def record_request(self, metrics: RequestMetrics):
        """Record a request's metrics."""
        self.requests.append(metrics)
        self.total_requests += 1
        if metrics.success:
            self.total_tokens += metrics.generated_tokens

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency percentiles (p50, p95, p99)."""
        if not self.requests:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

        latencies = [r.latency_ms for r in self.requests if r.success]

        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
            "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            "mean": statistics.mean(latencies)
        }

    def get_throughput(self) -> Dict[str, float]:
        """Get throughput metrics."""
        elapsed = time.time() - self.start_time
        elapsed = max(elapsed, 0.001)

        recent_tokens = sum(r.generated_tokens for r in self.requests if r.success)
        recent_requests = sum(1 for r in self.requests if r.success)

        # Estimate time window for recent requests
        if len(self.requests) >= 2:
            recent_elapsed = self.requests[-1].timestamp - self.requests[0].timestamp
            recent_elapsed = max(recent_elapsed, 0.001)
        else:
            recent_elapsed = elapsed

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "requests_per_second": recent_requests / recent_elapsed,
            "tokens_per_second": recent_tokens / recent_elapsed
        }

    def get_error_rate(self) -> float:
        """Get error rate."""
        if not self.requests:
            return 0.0

        errors = sum(1 for r in self.requests if not r.success)
        return errors / len(self.requests)

    def get_summary(self) -> Dict:
        """Get complete monitoring summary."""
        latency = self.get_latency_stats()
        throughput = self.get_throughput()

        avg_tokens = 0
        if self.requests:
            successful = [r for r in self.requests if r.success]
            if successful:
                avg_tokens = statistics.mean(r.generated_tokens for r in successful)

        return {
            "latency": latency,
            "throughput": throughput,
            "error_rate": self.get_error_rate(),
            "avg_tokens_per_request": avg_tokens,
            "window_size": len(self.requests),
            "uptime_seconds": time.time() - self.start_time
        }
`,

	testCode: `import unittest
import time

class TestLLMMonitor(unittest.TestCase):
    def test_record_request(self):
        monitor = LLMMonitor()
        metrics = RequestMetrics(
            prompt_tokens=10,
            generated_tokens=50,
            latency_ms=100,
            success=True,
            timestamp=time.time()
        )
        monitor.record_request(metrics)
        self.assertEqual(monitor.total_requests, 1)
        self.assertEqual(monitor.total_tokens, 50)

    def test_get_latency_stats(self):
        monitor = LLMMonitor()
        for i in range(100):
            metrics = RequestMetrics(10, 50, 100 + i, True, time.time())
            monitor.record_request(metrics)

        stats = monitor.get_latency_stats()
        self.assertIn("p50", stats)
        self.assertIn("p95", stats)
        self.assertIn("p99", stats)
        self.assertGreater(stats["p95"], stats["p50"])

    def test_get_error_rate(self):
        monitor = LLMMonitor()
        for i in range(10):
            success = i < 8  # 80% success
            metrics = RequestMetrics(10, 50, 100, success, time.time())
            monitor.record_request(metrics)

        error_rate = monitor.get_error_rate()
        self.assertAlmostEqual(error_rate, 0.2, places=2)

    def test_get_summary(self):
        monitor = LLMMonitor()
        metrics = RequestMetrics(10, 50, 100, True, time.time())
        monitor.record_request(metrics)

        summary = monitor.get_summary()
        self.assertIn("latency", summary)
        self.assertIn("throughput", summary)
        self.assertIn("error_rate", summary)

    def test_monitor_init(self):
        monitor = LLMMonitor(window_size=500)
        self.assertEqual(monitor.window_size, 500)

    def test_empty_monitor_latency(self):
        monitor = LLMMonitor()
        stats = monitor.get_latency_stats()
        self.assertEqual(stats["p50"], 0)

    def test_empty_monitor_error_rate(self):
        monitor = LLMMonitor()
        error_rate = monitor.get_error_rate()
        self.assertEqual(error_rate, 0.0)

    def test_get_throughput(self):
        monitor = LLMMonitor()
        metrics = RequestMetrics(10, 50, 100, True, time.time())
        monitor.record_request(metrics)
        throughput = monitor.get_throughput()
        self.assertIn("total_requests", throughput)
        self.assertIn("total_tokens", throughput)

    def test_summary_has_uptime(self):
        monitor = LLMMonitor()
        summary = monitor.get_summary()
        self.assertIn("uptime_seconds", summary)

    def test_request_metrics_dataclass(self):
        metrics = RequestMetrics(5, 25, 50.0, True, 1000.0)
        self.assertEqual(metrics.prompt_tokens, 5)
        self.assertEqual(metrics.generated_tokens, 25)
        self.assertEqual(metrics.latency_ms, 50.0)
`,

	hint1: 'Use deque with maxlen for sliding window metrics',
	hint2: 'Calculate percentiles from sorted latency values',

	whyItMatters: `Monitoring is essential for production LLMs:

- **SLAs**: Ensure latency requirements are met
- **Cost control**: Track token usage and costs
- **Debugging**: Identify slow or failing requests
- **Capacity planning**: Understand usage patterns

Without monitoring, you're flying blind.`,

	translations: {
		ru: {
			title: 'Мониторинг LLM',
			description: `# Мониторинг LLM

Реализуйте мониторинг для продакшен LLM систем.

## Ключевые метрики

**Производительность:**
- Latency (p50, p95, p99)
- Throughput (requests/second, tokens/second)
- Использование GPU

**Качество:**
- Использование токенов
- Частота ошибок
- Распределение длины вывода

## Пример

\`\`\`python
from prometheus_client import Counter, Histogram

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds'
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated'
)
\`\`\``,
			hint1: 'Используйте deque с maxlen для метрик скользящего окна',
			hint2: 'Вычисляйте перцентили из отсортированных значений latency',
			whyItMatters: `Мониторинг необходим для продакшен LLM:

- **SLA**: Обеспечение требований к latency
- **Контроль стоимости**: Отслеживание использования токенов
- **Отладка**: Выявление медленных или падающих запросов
- **Планирование мощностей**: Понимание паттернов использования`,
		},
		uz: {
			title: 'LLM monitoring',
			description: `# LLM monitoring

Ishlab chiqarish LLM tizimlari uchun monitoring ni amalga oshiring.

## Asosiy metrikalar

**Samaradorlik:**
- Latency (p50, p95, p99)
- Throughput (requests/second, tokens/second)
- GPU foydalanishi

**Sifat:**
- Token foydalanishi
- Xato darajasi
- Chiqish uzunligi taqsimoti

## Misol

\`\`\`python
from prometheus_client import Counter, Histogram

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds'
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated'
)
\`\`\``,
			hint1: "Sirpanuvchi oyna metrikalari uchun maxlen bilan deque dan foydalaning",
			hint2: "Saralangan latency qiymatlaridan persentillarni hisoblang",
			whyItMatters: `Monitoring ishlab chiqarish LLM lari uchun muhim:

- **SLA**: Latency talablarini bajarilishini ta'minlash
- **Xarajat nazorati**: Token foydalanishi va xarajatlarni kuzatish
- **Debugging**: Sekin yoki muvaffaqiyatsiz so'rovlarni aniqlash
- **Quvvat rejalashtirish**: Foydalanish naqshlarini tushunish`,
		},
	},
};

export default task;
