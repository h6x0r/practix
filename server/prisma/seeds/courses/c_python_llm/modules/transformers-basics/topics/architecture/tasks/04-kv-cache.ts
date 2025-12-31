import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pyllm-kv-cache-optimization',
	title: 'KV Cache for Inference',
	difficulty: 'medium',
	tags: ['pytorch', 'transformer', 'optimization', 'inference'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# KV Cache for Inference

Implement Key-Value caching to speed up autoregressive generation.

## Problem

During generation, each new token requires computing attention over all previous tokens.
Without caching, this is O(n²) for sequence length n.

## Solution

Cache the Key and Value projections from previous steps:
- Only compute K, V for new tokens
- Reuse cached K, V from previous steps

## Example

\`\`\`python
model = CachedDecoder(vocab_size=50000, d_model=512)

# First token - no cache
output, cache = model(tokens[:, :1], cache=None)

# Next tokens - use cache
output, cache = model(tokens[:, 1:2], cache=cache)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CachedAttention(nn.Module):
    """Attention with KV caching."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Project Q/K/V, concatenate with cache, apply attention, return (output, new_cache)
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CachedAttention(nn.Module):
    """Attention with KV caching."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V for new tokens
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_new = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_new = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Concatenate with cache if available
        if cache is not None:
            K_cached, V_cached = cache
            K = torch.cat([K_cached, K_new], dim=2)
            V = torch.cat([V_cached, V_new], dim=2)
        else:
            K = K_new
            V = V_new

        # Update cache
        new_cache = (K, V)

        # Attention (Q only attends to current position onward with full K, V)
        total_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask - only mask future positions relative to Q's positions
        # Q starts at position (total_len - seq_len)
        start_pos = total_len - seq_len
        mask = torch.ones(seq_len, total_len, device=x.device)
        for i in range(seq_len):
            mask[i, start_pos + i + 1:] = 0

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context), new_cache
`,

	testCode: `import torch
import unittest

class TestCachedAttention(unittest.TestCase):
    def test_no_cache(self):
        attn = CachedAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)
        output, cache = attn(x, cache=None)
        self.assertEqual(output.shape, (2, 10, 128))
        self.assertEqual(cache[0].shape, (2, 4, 10, 32))  # K cache

    def test_with_cache(self):
        attn = CachedAttention(d_model=64, num_heads=2)
        # First pass
        x1 = torch.randn(1, 5, 64)
        out1, cache = attn(x1, cache=None)
        # Second pass
        x2 = torch.randn(1, 1, 64)
        out2, new_cache = attn(x2, cache=cache)
        self.assertEqual(out2.shape, (1, 1, 64))
        self.assertEqual(new_cache[0].shape, (1, 2, 6, 32))  # K grows

    def test_cache_accumulation(self):
        attn = CachedAttention(d_model=64, num_heads=2)
        cache = None
        for i in range(5):
            x = torch.randn(1, 1, 64)
            out, cache = attn(x, cache=cache)
        self.assertEqual(cache[0].shape[2], 5)  # 5 tokens cached

    def test_is_nn_module(self):
        attn = CachedAttention(d_model=128, num_heads=4)
        self.assertIsInstance(attn, torch.nn.Module)

    def test_has_projections(self):
        attn = CachedAttention(d_model=128, num_heads=4)
        self.assertTrue(hasattr(attn, 'W_q'))
        self.assertTrue(hasattr(attn, 'W_k'))
        self.assertTrue(hasattr(attn, 'W_v'))
        self.assertTrue(hasattr(attn, 'W_o'))

    def test_cache_is_tuple(self):
        attn = CachedAttention(d_model=64, num_heads=2)
        x = torch.randn(1, 5, 64)
        _, cache = attn(x, cache=None)
        self.assertIsInstance(cache, tuple)
        self.assertEqual(len(cache), 2)

    def test_output_not_nan(self):
        attn = CachedAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 8, 128)
        output, _ = attn(x, cache=None)
        self.assertFalse(torch.isnan(output).any())

    def test_d_k_calculation(self):
        attn = CachedAttention(d_model=256, num_heads=8)
        self.assertEqual(attn.d_k, 32)

    def test_single_token_input(self):
        attn = CachedAttention(d_model=64, num_heads=2)
        x = torch.randn(1, 1, 64)
        output, cache = attn(x, cache=None)
        self.assertEqual(output.shape, (1, 1, 64))
        self.assertEqual(cache[0].shape, (1, 2, 1, 32))
`,

	hint1: 'Cache K and V after projection, concatenate with new K, V',
	hint2: 'Adjust the causal mask based on the starting position in the sequence',

	whyItMatters: `KV caching is essential for fast LLM inference:

- **Linear complexity**: O(n) instead of O(n²) per new token
- **Memory tradeoff**: Store K, V at cost of memory
- **Production requirement**: All deployed LLMs use KV cache
- **vLLM/TGI**: Advanced caching with paged attention

This optimization makes real-time generation possible.`,

	translations: {
		ru: {
			title: 'KV Cache для инференса',
			description: `# KV Cache для инференса

Реализуйте кэширование Key-Value для ускорения авторегрессивной генерации.

## Проблема

При генерации каждый новый токен требует вычисления attention по всем предыдущим.
Без кэширования это O(n²) для длины последовательности n.

## Решение

Кэшируйте проекции Key и Value из предыдущих шагов:
- Вычисляйте K, V только для новых токенов
- Используйте закэшированные K, V из предыдущих шагов

## Пример

\`\`\`python
model = CachedDecoder(vocab_size=50000, d_model=512)

# First token - no cache
output, cache = model(tokens[:, :1], cache=None)

# Next tokens - use cache
output, cache = model(tokens[:, 1:2], cache=cache)
\`\`\``,
			hint1: 'Кэшируйте K и V после проекции, конкатенируйте с новыми K, V',
			hint2: 'Корректируйте каузальную маску на основе начальной позиции в последовательности',
			whyItMatters: `KV кэширование необходимо для быстрого инференса LLM:

- **Линейная сложность**: O(n) вместо O(n²) на новый токен
- **Компромисс памяти**: Хранение K, V за счет памяти
- **Требование продакшена**: Все развернутые LLM используют KV cache
- **vLLM/TGI**: Продвинутое кэширование с paged attention`,
		},
		uz: {
			title: 'Inference uchun KV Cache',
			description: `# Inference uchun KV Cache

Avtoregessiv generatsiyani tezlashtirish uchun Key-Value keshlashni amalga oshiring.

## Muammo

Generatsiya vaqtida har bir yangi token barcha oldingi tokenlar bo'yicha attention hisoblashni talab qiladi.
Keshlashsiz bu ketma-ketlik uzunligi n uchun O(n²).

## Yechim

Oldingi qadamlardan Key va Value proyeksiyalarini keshlang:
- K, V ni faqat yangi tokenlar uchun hisoblang
- Oldingi qadamlardan keshlangan K, V dan foydalaning

## Misol

\`\`\`python
model = CachedDecoder(vocab_size=50000, d_model=512)

# First token - no cache
output, cache = model(tokens[:, :1], cache=None)

# Next tokens - use cache
output, cache = model(tokens[:, 1:2], cache=cache)
\`\`\``,
			hint1: "Proyeksiyadan keyin K va V ni keshlang, yangi K, V bilan birlashtiring",
			hint2: "Kauzal niqobni ketma-ketlikdagi boshlang'ich pozitsiyaga qarab moslang",
			whyItMatters: `KV keshlash tez LLM inference uchun zarur:

- **Chiziqli murakkablik**: Har bir yangi token uchun O(n²) o'rniga O(n)
- **Xotira kelishuvi**: K, V ni xotira hisobiga saqlash
- **Ishlab chiqarish talabi**: Barcha deploy qilingan LLM lar KV cache dan foydalanadi
- **vLLM/TGI**: Paged attention bilan ilg'or keshlash`,
		},
	},
};

export default task;
