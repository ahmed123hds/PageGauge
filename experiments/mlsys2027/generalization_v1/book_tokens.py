"""One fixed prefix/continuation per verified PG19 TRAIN development book."""
import json
from pathlib import Path


def window(encoded, context, steps, family, bos_token_id):
    if context <= 0 or steps <= 0 or family not in ('mistral', 'qwen3'):
        raise ValueError('Unsupported model/window')
    bos = family == 'mistral'
    if bos and bos_token_id is None:
        raise ValueError('Mistral requires its native BOS')
    needed = context+steps+1-int(bos)
    if len(encoded) < needed:
        raise ValueError(f'Infeasible book: need {needed} corpus tokens, have {len(encoded)}; do not replace by quality')
    ids = ([bos_token_id] if bos else [])+encoded[:needed]
    return ids, {'bos_prepended_per_request': bos, 'corpus_tokens_per_request': needed,
                 'corpus_label_start_offset': context+1-int(bos),
                 'corpus_label_end_offset_exclusive': context+1-int(bos)+steps}


def build_tokens(cohort, index, model, context, steps, quality):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    cohort = Path(cohort).resolve()
    selection = json.loads((cohort/'selection.json').read_text())
    downloaded = json.loads((cohort/'analysis.json').read_text())
    if not downloaded['download_complete'] or downloaded['split'] != 'train' or selection['split'] != 'train':
        raise ValueError('Only verified training development books allowed')
    digest = quality.sha256_file(cohort/'selection.json')
    if downloaded['selection_sha256'] != digest or len(downloaded['files']) != 8 or not 0 <= index < 8:
        raise ValueError('Invalid/changed book selection')
    record = downloaded['files'][index]
    selected = selection['selected'][index]
    if record['object_name'] != selected['name'] or not record['object_name'].startswith('train/'):
        raise ValueError('Book order/split mismatch')
    path = Path(record['path']).resolve()
    if path.parent != cohort:
        raise ValueError('Book path escaped selected corpus')
    payload = path.read_bytes()
    if len(payload) != record['size'] or quality.sha256_bytes(payload) != record['sha256']:
        raise ValueError('Book bytes changed')
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    encoded = tokenizer(payload.decode('utf-8'), add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    row, policy = window(encoded, context, steps, config.model_type, tokenizer.bos_token_id)
    tokens = torch.tensor([row], dtype=torch.long)
    if int(tokens.min()) < 0 or int(tokens.max()) >= config.vocab_size:
        raise ValueError('Token outside vocabulary')
    provenance = {'kind': 'pg19', 'dataset': 'PG-19', 'split': 'train',
        'purpose': 'Metadata-selected development book; not final TEST or unseen model-pretraining data',
        'book_index': index, 'object_name': record['object_name'],
        'archive_member': record['object_name'], 'archive_member_sha256': record['sha256'],
        'document_path': str(path), 'document_sha256': record['sha256'],
        'selection_path': str(cohort/'selection.json'), 'selection_sha256': digest,
        'tokenizer': quality.tokenizer_artifact_provenance(tokenizer, model),
        'model_family': config.model_type, 'bos_token_id': tokenizer.bos_token_id,
        'add_special_tokens': False, 'chat_template_applied': False, **policy,
        'corpus_window_start_offsets': [0], 'corpus_window_end_offsets_exclusive': [policy['corpus_tokens_per_request']],
        'corpus_windows_disjoint': True, 'available_corpus_token_count': len(encoded),
        'shape': list(tokens.shape), 'token_ids_sha256': quality.canonical_json_sha256([row]),
        'teacher_forced_decode_ids_step_major': tokens[:, context:context+steps].T.tolist(),
        'decode_label_ids_step_major': tokens[:, context+1:context+steps+1].T.tolist(),
        'metric_scope': 'Model-token NLL/PPL and paired ratio to own HF, not full-corpus word-normalized PG19 benchmark score.'}
    return tokens, provenance


def request_windows(provenance, context, steps, quality):
    rows = quality.request_window_metadata(provenance, context, steps, 1)
    start = context+1-int(provenance['bos_prepended_per_request'])
    rows[0].update(corpus_label_start_offset=start, corpus_label_end_offset_exclusive=start+steps,
        cluster_unit_id=f"pg19-train-{provenance['object_name'].split('/')[-1]}-{provenance['document_sha256'][:16]}-{start}-{start+steps}")
    return rows
