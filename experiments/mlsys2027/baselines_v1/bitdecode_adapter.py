"""BitDecoding kernels in a shared native-stack Mistral harness, not upstream model support."""
import sys

# Reuse the already compiled extension without changing either environment.
sys.path.append('/home/anonymous/pagegauge_baselines/bitdecode_sm120_env/lib/python3.12/site-packages')
from kivi_adapter import imports,configure_decode as configure_skeleton
from bitdecode_cache import BitDecodeCache
native,_=imports()


class BitDecodeMistralAttention(native.MistralAttention_KIVI):
    def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,**kwargs):
        batch,length,_=hidden_states.shape
        if length!=1 or past_key_value is None or output_attentions or not use_cache:
            raise ValueError('Only unpadded recurrent decode after shared FP16 prefill is supported')
        cache,old_length=past_key_value
        if cache.length!=old_length:raise RuntimeError('Layer cache length mismatch')
        q=self.q_proj(hidden_states).view(batch,1,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(hidden_states).view(batch,1,self.num_key_value_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(hidden_states).view(batch,1,self.num_key_value_heads,self.head_dim).transpose(1,2)
        cos,sin=self.rotary_emb(v,seq_len=old_length+1)
        q,k=native.apply_rotary_pos_emb(q,k,cos,sin,position_ids)
        value=cache.step(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
        value=self.o_proj(value.reshape(batch,1,self.hidden_size))
        return value,None,(cache,cache.length)


def configure_decode(model,bits):
    if bits!=4 or model.config.sliding_window is not None:
        raise ValueError('SM120 integration currently supports INT4/full context only')
    configure_skeleton(model,4)
    for layer in model.model.layers:layer.self_attn.__class__=BitDecodeMistralAttention
    return model


def pack_initial_cache(legacy_cache,bits):
    if bits!=4:raise ValueError('INT2 packing exceeds this SM120 port shared-memory capacity')
    result=[]
    for k,v in legacy_cache:
        cache=BitDecodeCache(k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous())
        result.append((cache,cache.length))
    return tuple(result)
