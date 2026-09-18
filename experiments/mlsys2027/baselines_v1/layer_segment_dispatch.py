"""Opt-in Mistral layer segments; append and planning remain outside capture."""
import torch
from segment_graph_dispatch import SegmentGraph
from mistral_dense_segments import BeforeAttention, AfterAttention


def install(model, driver, context, steps, validate=True):
    originals, records = [], []
    for index, layer in enumerate(model.model.layers):
        before, after = BeforeAttention(layer).eval(), AfterAttention(layer).eval()
        example = torch.zeros(driver.batch_size, 1, model.config.hidden_size,
                              device=next(layer.parameters()).device, dtype=torch.float16)
        pre = SegmentGraph(before, (example,))
        post = SegmentGraph(after, (example, example))
        stats = {'calls': 0, 'pre_checks': 0, 'post_checks': 0}
        records.append(stats)

        def forward(hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    index=index, pre=pre, post=post, before=before, after=after,
                    stats=stats, **kwargs):
            if output_attentions or not use_cache or past_key_value is None or hidden_states.shape[1] != 1:
                raise ValueError('Only unpadded recurrent decode supported')
            actual_driver, position = past_key_value
            if actual_driver is not driver or driver.logical_lengths[index] != position:
                raise ValueError('Wrong driver or logical position')
            if not context <= position < context+steps:
                raise ValueError('Outside declared recurrence')
            if index == 0:
                driver.plan(position+1)
            query, key, value = pre(hidden_states)
            check = validate and position-context in (0, 15, 16, 767, 768, steps-1)
            if check:
                for actual, expected in zip((query, key, value), before(hidden_states)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                stats['pre_checks'] += 1
            batch = hidden_states.shape[0]
            driver.append(index, query.view(batch, model.config.num_attention_heads, 128),
                          key.view(batch, model.config.num_key_value_heads, 128),
                          value.view(batch, model.config.num_key_value_heads, 128), position)
            attended = driver.eager_attention(index).reshape(batch, 1, model.config.hidden_size)
            result = post(attended, hidden_states)[0]
            if check:
                torch.testing.assert_close(result, after(attended, hidden_states)[0], rtol=0, atol=0)
                stats['post_checks'] += 1
            driver.logical_lengths[index] += 1
            stats['calls'] += 1
            return result, (driver, position+1)

        originals.append((layer, layer.forward))
        layer.forward = forward
    return originals, records


def verify(records, steps, validate):
    expected = 6 if validate else 0
    if len(records) != 32 or any(r != {'calls': steps, 'pre_checks': expected, 'post_checks': expected} for r in records):
        raise ValueError('Incomplete layer-segment execution')
