"""FI/PageGauge attention in the same ungraphed Mistral body as native KIVI.

No packed projection replacement, model graph capture, or weight transformation.
This common-engine contrast is separate from the optimized production decoder.
"""
import math
import os
from kivi_adapter import imports,configure_decode as configure_skeleton
native,_=imports()


class PagedMistralAttention(native.MistralAttention_KIVI):
    def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,**kwargs):
        batch,length,_=hidden_states.shape
        if length!=1 or past_key_value is None or output_attentions or not use_cache:
            raise ValueError('Only unpadded recurrent decode is supported')
        driver,position=past_key_value
        index=self._pagegauge_common_layer_index
        if driver.logical_lengths[index]!=position:raise RuntimeError('Cache position mismatch')
        if index==0:driver.plan(position+1)
        q=self.q_proj(hidden_states).view(batch,self.num_heads,self.head_dim)
        k=self.k_proj(hidden_states).view(batch,self.num_key_value_heads,self.head_dim)
        v=self.v_proj(hidden_states).view(batch,self.num_key_value_heads,self.head_dim)
        driver.append(index,q,k,v,position)
        attended=driver.eager_attention(index)
        result=self.o_proj(attended.reshape(batch,1,self.hidden_size))
        driver.logical_lengths[index]+=1
        return result,None,(driver,position+1)


def configure_and_pack(model,initial,backend,max_context,exact_split_pages=None):
    import torch
    import flashinfer
    import benchmark_pg19_external_quality as quality
    pg=quality.PG
    if backend not in ('flashinfer_fp16','page_gauge') or model.config.sliding_window is not None:
        raise ValueError('Only full-context FI/PG supported')
    if os.environ.get('PAGEGAUGE_VALUE_CONDITIONING','none')!='none':
        raise ValueError('Common-engine contrast requires unchanged weights')
    layers=len(initial);batch,heads,context,dim=initial[0][0].shape
    if dim!=128 or context%16 or context<4096:raise ValueError('Unsupported fixture')
    maximum_pages=math.ceil(max_context/16)
    cache=quality.allocate_baseline_cache(layers,maximum_pages,batch,heads)
    for i,(k,v) in enumerate(initial):
        for request in range(batch):
            begin=request*maximum_pages
            cache.key[i,begin:begin+context//16].copy_(k[request].transpose(0,1).reshape(context//16,16,heads,dim))
            cache.value[i,begin:begin+context//16].copy_(v[request].transpose(0,1).reshape(context//16,16,heads,dim))
    if backend=='page_gauge':
        cache=quality.build_gauge_cache_from_baseline(cache,layers,maximum_pages,context//16,48,batch,heads,4,128)
    cos,sin=model.model.layers[0].self_attn.rotary_emb(initial[0][1],seq_len=max_context)
    cos,sin=cos.half().contiguous(),sin.half().contiguous()
    driver=pg.TransformerDecoder(model,flashinfer,pg.RUNTIME.load_append_extension(),backend,cache,max_context,
        768,256,128,cos,sin,'attention_add',tail_attention='flashinfer_merge',batch_size=batch,
        exact_sink_pages=4 if backend=='page_gauge' else 0,
        exact_static_suffix_pages=128 if backend=='page_gauge' else 0,
        initial_context_pages=context//16 if backend=='page_gauge' else None)
    driver.logical_lengths=[context]*layers
    driver.exact_split_override=exact_split_pages
    if exact_split_pages is not None:
        if backend!='page_gauge' or exact_split_pages not in (32,64,128):raise ValueError('Invalid exact-region split override')
        original_plan=driver.exact_wrapper.plan
        def plan_exact(physical_indices,logical_tokens,last_page_len,split_pages,page_table_epoch=None):
            return original_plan(physical_indices,logical_tokens,last_page_len,exact_split_pages,page_table_epoch)
        driver.exact_wrapper.plan=plan_exact
    configure_skeleton(model,4)
    for i,layer in enumerate(model.model.layers):
        layer.self_attn.__class__=PagedMistralAttention
        layer.self_attn._pagegauge_common_layer_index=i
    return tuple((driver,context) for _ in initial)


def cache_accounting(past):
    import torch
    driver=past[0][0]
    tensors=[t for t in vars(driver.cache).values() if isinstance(t,torch.Tensor)]
    stores={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
    return {'logical_tensor_bytes':sum(t.numel()*t.element_size() for t in tensors),
        'unique_storage_bytes':sum(stores.values()),'logical_lengths':driver.logical_lengths,
        'exact_split_override':driver.exact_split_override,
        'scope':'Served preallocated paged cache only; excludes workspace, weights and retained reference fixture'}
