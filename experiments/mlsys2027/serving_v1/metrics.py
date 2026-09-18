"""Arrival-to-completion serving metrics; queueing and failures stay visible."""
import math
import statistics


def percentile(values, probability):
    if not values: return None
    ordered = sorted(values)
    location = (len(ordered)-1)*probability
    lower, upper = math.floor(location), math.ceil(location)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(location-lower)


def summarize(records, *, measurement_end_s, ttft_slo_s, tpot_slo_s):
    if not records or not all(math.isfinite(x) and x > 0 for x in (ttft_slo_s,tpot_slo_s)):
        raise ValueError('Nonempty declared request cohort and positive SLOs required')
    ids = [r['request_id'] for r in records]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate request ID')
    arrivals = [r['arrival_s'] for r in records]
    if not all(math.isfinite(x) for x in arrivals+[measurement_end_s]): raise ValueError('Nonfinite clock')
    duration = measurement_end_s-min(arrivals)
    if duration <= 0: raise ValueError('Empty measurement interval')
    latencies, ttfts, tpots, gaps = [],[],[],[]
    tokens = good = failures = 0
    request_rows = []
    for record in records:
        arrival, finish = record['arrival_s'],record['completed_s']
        times = record['output_token_times_s']
        if type(record['success']) is not bool or type(record['output_tokens']) is not int or record['output_tokens'] < 0:
            raise ValueError('Invalid request outcome')
        if len(times) != record['output_tokens'] or not all(math.isfinite(x) for x in times+[finish]):
            raise ValueError('Missing token-level timestamps or nonfinite completion')
        if not arrival <= finish <= measurement_end_s or times != sorted(times) or any(not arrival <= t <= finish for t in times):
            raise ValueError('Invalid arrival/token/completion ordering')
        if not record['success']:
            failures += 1
            request_rows.append({'request_id':record['request_id'],'success':False,'latency_s':finish-arrival,
                'partial_output_tokens':record['output_tokens'],'slo_passed':False})
            continue
        if not times: raise ValueError('Successful empty output needs an explicit separate protocol')
        latency, ttft = finish-arrival,times[0]-arrival
        tpot = (times[-1]-times[0])/(len(times)-1) if len(times)>1 else None
        passed = ttft <= ttft_slo_s and (tpot is None or tpot <= tpot_slo_s)
        latencies.append(latency);ttfts.append(ttft)
        if tpot is not None: tpots.append(tpot)
        gaps.extend(b-a for a,b in zip(times,times[1:]))
        tokens += len(times);good += int(passed)
        request_rows.append({'request_id':record['request_id'],'success':True,'latency_s':latency,'ttft_s':ttft,
            'mean_tpot_s':tpot,'slo_passed':passed})
    def summary(values):
        return {'count':len(values),'mean':statistics.fmean(values) if values else None,
            'p50':percentile(values,.5),'p95':percentile(values,.95)}
    return {'requests':len(records),'successful_requests':len(records)-failures,'failed_requests':failures,
        'measurement_seconds':duration,'successful_output_tokens':tokens,
        'output_tokens_per_second':tokens/duration,'slo_goodput_requests_per_second':good/duration,
        'slo_pass_fraction_all_offered_requests':good/len(records),'latency_s_successful':summary(latencies),
        'ttft_s_successful':summary(ttfts),'mean_tpot_s_successful_multitoken':summary(tpots),
        'intertoken_gap_s_successful':summary(gaps),'request_rows':request_rows,
        'scope':'All offered requests retained. Arrival clock is scheduled arrival, not admission/first GPU execution. Throughput uses the entire declared measurement interval; failed partial outputs are not successful token throughput. TPOT excludes the first token and one-token responses have no TPOT. No extrapolated server result.'}
