"""Explicit in-memory development amendment; never edits production source."""
import inspect
import textwrap
from merge_center_candidate import merge_center


def install(decoder_class):
    original = decoder_class.eager_attention
    source = textwrap.dedent(inspect.getsource(original))
    begin = source.index('    flashinfer.merge_state_in_place(')
    end = source.index('    return self.attention_output[layer]', begin)
    old = source[begin:end]
    if 'self.attention_output[layer].add_(self.cache.output_center[layer])' not in old:
        raise ValueError('Unexpected production merge source')
    replacement = '''    if self.center_restore == "attention_add":
        merge_center(self.attention_output[layer], self.exact_output[layer],
                     self.old_lse[layer], self.exact_lse[layer],
                     self.cache.output_center[layer], self.attention_output[layer])
    else:
        flashinfer.merge_state_in_place(self.attention_output[layer],
            self.old_lse[layer], self.exact_output[layer], self.exact_lse[layer])
'''
    namespace = dict(original.__globals__, merge_center=merge_center)
    exec(compile(source[:begin]+replacement+source[end:], __file__, 'exec'), namespace)
    decoder_class.eager_attention = namespace['eager_attention']
