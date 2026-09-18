"""Freeze and fetch eight PG-19 TRAIN books, never validation or TEST.

Selection uses object metadata only, before book contents or model outcomes.
This is a development corpus, not an independent final evaluation claim.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
API = 'https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o'
SEED = 'PageGauge-MLSys2027-PG19-TRAIN-dev-v1'


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2*(attempt+1))


def select(items):
    eligible = []
    names = set()
    for item in items:
        name = item['name']
        if name in names:
            raise ValueError('Duplicate object listing')
        names.add(name)
        if not re.fullmatch(r'train/\d+\.txt', name):
            continue
        if not 200000 <= int(item['size']) <= 2000000:
            continue
        if not item.get('md5Hash') or not item.get('generation', '').isdigit():
            raise ValueError('Missing pinned object generation/digest')
        eligible.append(item)
    ranked = sorted(eligible, key=lambda row: (hashlib.sha256((SEED+'|'+row['name']).encode()).hexdigest(), row['name']))
    if len(ranked) < 8:
        raise ValueError('Insufficient eligible training books')
    return ranked[:8], len(ranked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Resume only this already-frozen TRAIN selection')
    args = parser.parse_args()
    if hasattr(os, 'nice'):
        os.nice(10)
    if args.resume:
        out = args.resume.resolve()
        selection = json.loads((out/'selection.json').read_text())
        listing = json.loads((out/'listing.json').read_text())
        selected, count = select(listing['items'])
        if selection['split'] != 'train' or selection['selected'] != selected or selection['eligible_count'] != count:
            raise ValueError('Changed development selection')
        if selection['listing_sha256'] != base.sha256_file(out/'listing.json'):
            raise ValueError('Listing changed')
    else:
        out = ROOT/'data/mlsys2027_pg19_train_v1'/('cohort_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True)
        print('PG19 TRAIN development output: '+str(out), flush=True)
        items, pages, token = [], [], None
        while True:
            query = {'prefix': 'train/', 'maxResults': 1000,
                     'fields': 'nextPageToken,items(name,size,generation,md5Hash)'}
            if token:
                query['pageToken'] = token
            payload = fetch(API+'?'+urllib.parse.urlencode(query))
            result = json.loads(payload)
            pages.append({'page': len(pages), 'sha256': hashlib.sha256(payload).hexdigest(),
                          'objects': len(result.get('items', []))})
            items.extend(result.get('items', []))
            token = result.get('nextPageToken')
            if not token:
                break
        selected, count = select(items)
        base.atomic_json(out/'listing.json', {'items': items, 'pages': pages, 'prefix': 'train/'})
        selection = {'schema_version': 1, 'dataset': 'PG-19', 'split': 'train',
            'purpose': 'Cross-domain development only; no final TEST claim',
            'official_homepage': 'https://github.com/google-deepmind/pg19',
            'bucket': 'gs://deepmind-gutenberg', 'metadata_api': API,
            'seed': SEED, 'selection_rule': 'First eight by SHA256(seed|object_name), tie by name, among TRAIN digit.txt objects with metadata size in [200000,2000000] bytes. No contents or model outcomes used.',
            'listing_sha256': base.sha256_file(out/'listing.json'), 'listed_count': len(items),
            'eligible_count': count, 'selected': selected, 'frozen_before_book_download': True,
            'source_sha256': base.sha256_file(Path(__file__)), 'frozen_at': datetime.now(timezone.utc).isoformat(),
            'planned_window': {'context': 20480, 'decode_steps': 1536, 'corpus_token_start': 0,
                'one_window_per_book': True, 'insufficient_tokens_policy': 'Record infeasible book for each tokenizer; do not choose replacements by quality.'},
            'metric_scope': 'Model-token NLL and paired PPL ratio to own FP16 reference; not the official full-corpus word-normalized PG19 benchmark score.'}
        base.atomic_json(out/'selection.json', selection)
    frozen = base.sha256_file(out/'selection.json')
    print('Frozen TRAIN selection SHA256 '+frozen, flush=True)
    files = []
    for item in selection['selected']:
        name = item['name']
        if not re.fullmatch(r'train/\d+\.txt', name):
            raise ValueError('Only training objects may be fetched')
        path = out/Path(name).name
        url = API+'/'+urllib.parse.quote(name, safe='')+'?'+urllib.parse.urlencode({'alt': 'media', 'generation': item['generation']})
        payload = path.read_bytes() if path.exists() else fetch(url)
        if len(payload) != int(item['size']) or base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode() != item['md5Hash']:
            raise ValueError('Downloaded bytes do not match frozen object metadata: '+name)
        payload.decode('utf-8')
        if not path.exists():
            path.write_bytes(payload)
        files.append({'object_name': name, 'generation': item['generation'], 'path': str(path),
                      'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'md5_verified': True})
        print('Verified training book '+name+' ('+str(len(payload))+' bytes)', flush=True)
    if frozen != base.sha256_file(out/'selection.json'):
        raise ValueError('Selection changed during download')
    base.atomic_json(out/'analysis.json', {'download_complete': True, 'split': 'train',
        'selection_sha256': frozen, 'files': files, 'total_bytes': sum(row['size'] for row in files),
        'scope': 'Eight metadata-selected PG19 training books only. No tokenizer/model execution or final TEST access.'})
    print('PG19 development download complete: '+str(out), flush=True)


if __name__ == '__main__':
    main()
