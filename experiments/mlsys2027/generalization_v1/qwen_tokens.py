"""Raw-text Qwen corpus windows: no synthetic BOS, EOS or chat template."""
import zipfile


def raw_windows(encoded, context, steps, offset, batch=1, stride=23600):
    needed = context+steps+1
    if min(context, steps, batch) <= 0 or offset < 0 or stride < needed:
        raise ValueError('Invalid/disjoint-window configuration')
    starts = [offset+i*stride for i in range(batch)]
    if starts[-1]+needed > len(encoded):
        raise ValueError('Insufficient corpus tokens')
    ids = [encoded[start:start+needed] for start in starts]
    return ids, starts, [start+needed for start in starts]


def build_tokens(args, quality, vocab_size):
    import torch
    from transformers import AutoTokenizer
    if args.token_source != 'wikitext2' or args.wikitext_member != 'wikitext-2-raw/wiki.train.raw':
        raise ValueError('Qwen development loader is restricted to WikiText TRAIN')
    archive_sha = quality.sha256_file(args.wikitext_zip)
    if archive_sha != quality.WIKITEXT2_RAW_V1_SHA256:
        raise ValueError('Corpus archive drift')
    with zipfile.ZipFile(args.wikitext_zip) as archive:
        info = archive.getinfo(args.wikitext_member)
        member = archive.read(info)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer(member.decode('utf-8'), add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    ids, starts, ends = raw_windows(encoded, args.context, args.decode_steps,
        args.token_offset, args.batch_size, args.token_stride)
    tokens = torch.tensor(ids, dtype=torch.long)
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError('Token outside model vocabulary')
    provenance = {'kind': 'wikitext2', 'dataset': 'WikiText-2 raw', 'split': 'train',
        'archive_path': str(args.wikitext_zip.resolve()), 'archive_sha256': archive_sha,
        'archive_sha256_verified': True, 'archive_member': args.wikitext_member,
        'archive_member_sha256': quality.sha256_bytes(member),
        'archive_member_crc32': f'{info.CRC:08x}',
        'tokenizer': quality.tokenizer_artifact_provenance(tokenizer, args.model),
        'add_special_tokens': False, 'bos_prepended_per_request': False,
        'bos_token_id': tokenizer.bos_token_id, 'chat_template_applied': False,
        'corpus_window_start_offsets': starts, 'corpus_window_end_offsets_exclusive': ends,
        'corpus_window_stride': args.token_stride, 'corpus_windows_disjoint': True,
        'corpus_tokens_per_request': args.context+args.decode_steps+1,
        'available_corpus_token_count': len(encoded), 'shape': list(tokens.shape),
        'token_ids_sha256': quality.canonical_json_sha256(ids),
        'layout': 'request-major [batch,context_plus_decode_plus_label]',
        'label_alignment': 'No prepended BOS: model position p maps to corpus offset start+p; logit p predicts corpus offset start+p+1.',
        'teacher_forced_decode_ids_step_major': tokens[:, args.context:args.context+args.decode_steps].T.tolist(),
        'decode_label_ids_step_major': tokens[:, args.context+1:args.context+args.decode_steps+1].T.tolist()}
    return tokens, provenance


def request_windows(provenance, context, steps, quality):
    rows = quality.request_window_metadata(provenance, context, steps, provenance['shape'][0])
    if provenance.get('bos_prepended_per_request') is False:
        for row in rows:
            start = row['corpus_window_start_offset']+context+1
            row['corpus_label_start_offset'] = start
            row['corpus_label_end_offset_exclusive'] = start+steps
            row['cluster_unit_id'] = f"wikitext2-train-{provenance['archive_member_sha256'][:16]}-{start}-{start+steps}"
    return rows
