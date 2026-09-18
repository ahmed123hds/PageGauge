# Shift-invariant error analysis for the MLSys manuscript

This analysis does not change PageGauge's representation or attention algebra.
It explains what exact factorization does, and does not, guarantee about lossy
quantization. Treat these as supporting bounds, not a new algorithmic novelty
claim or a substitute for task evaluation.

## Contract and notation

Consider one query and head, with the same finite, nonempty set of unmasked
positions in both executions. Keys are already in the coordinates consumed by
attention (including RoPE where applicable). Let

\[
 x_j=\alpha q^\top k_j,
 \quad \widehat x_j=x_j+\delta_j,
 \quad \delta_j=\alpha q^\top(\widehat k_j-k_j),
 \quad p=\operatorname{softmax}(x),\quad
 \widehat p=\operatorname{softmax}(\widehat x).
\]

Let \(\Delta=\max_j\delta_j-\min_j\delta_j\), let
\(\epsilon_V=\max_j\|\widehat v_j-v_j\|\), and let
\(D_V=\max_{i,j}\|v_i-v_j\|\) for any norm. With outputs
\(o=\sum_jp_jv_j\) and \(\widehat o=\sum_j\widehat p_j\widehat v_j\),

\[
 \boxed{\|\widehat o-o\|
 \leq \sum_j\widehat p_j\|\widehat v_j-v_j\|
       +D_V\tanh(\Delta/4)
 \leq \epsilon_V+D_V\tanh(\Delta/4).}
\]

This is a real-arithmetic bound against the original unquantized cache. It is
separate from the identity proving that PageGauge evaluates the reconstructed
mixed cache. Finite-precision execution error must be added as a separate term
if comparing actual floating-point outputs with this ideal reconstructed output.

## Proof

Write \(R_j=e^{\delta_j}\), \(Z=\sum_jp_jR_j\),
\(a=\min_jR_j\), and \(b=\max_jR_j\). Then
\(\widehat p_j=p_jR_j/Z\). If \(a=b\), both distributions are equal.
Otherwise, convexity of absolute value bounds it by the chord joining its
values at \(a,b\). Taking expectation under \(p\) gives

\[
 \operatorname{TV}(p,\widehat p)
 =\frac{\mathbb E_p|R-Z|}{2Z}
 \leq\frac{(b-Z)(Z-a)}{(b-a)Z}
 \leq\frac{\sqrt b-\sqrt a}{\sqrt b+\sqrt a}
 =\tanh(\Delta/4).
\]

For the second inequality, maximize
\(((a+b)-Z-ab/Z)/(b-a)\) over \(Z\in[a,b]\); its maximum occurs at
\(Z=\sqrt{ab}\). The positive and negative parts of
\(\widehat p-p\) each have mass TV. After normalization their weighted value
averages are in the convex hull of the original values, whose diameter is at
most \(D_V\). Therefore
\(\|\sum_j(\widehat p_j-p_j)v_j\|\leq D_V\operatorname{TV}\).
Apply the triangle inequality to

\[
 \widehat o-o=\sum_j\widehat p_j(\widehat v_j-v_j)
             +\sum_j(\widehat p_j-p_j)v_j.
\]

The TV constant is attainable with two positions: take
\(x=(\Delta/2,0)\), \(\delta=(-\Delta/2,\Delta/2)\),
and scalar values \((0,D_V)\). The original and perturbed distributions swap,
and output distance is exactly \(D_V\tanh(\Delta/4)\).

## Consequences and limitations

- A common key error vector adds the same \(\delta_j\) at every position;
  \(\Delta=0\), so it does not perturb attention probabilities. This is the
  same shift invariance underlying PageGauge's shared-key-center cancellation.
  A per-page center mismatch is generally not common across all positions and
  cannot be discarded without its page-level logit correction.
- Exact residual regions have zero representation error at their positions.
  They can reduce the weighted value-error term or the range of key-logit
  error, but retaining them does not guarantee zero error on the other rows.
- With the norm dual to the chosen key norm,
  \(\Delta\leq\alpha\|q\|_*\max_{i,j}\|e_i^K-e_j^K\|\).
  The error differences, not a common offset, are the relevant key geometry.
- INT8 alone is not a fidelity guarantee: scale choice, outliers, query
  directions and value geometry determine these terms. Conversely, a low
  cosine in a representation or logits does not by itself prove task failure.
- The statement is for one attention operation. Residual connections, later
  layers, generation feedback, finite precision and model changes prevent
  inferring a complete-model quality result from this bound alone. It supplies
  no speed guarantee and no universal quality threshold.

The key-error expression above fixes the query. When comparing executions whose
queries also differ, write \(\widehat q=q+e_q\) and use the total logit error
\(\delta_j=\alpha(q^\top e_j^K+e_q^\top k_j+e_q^\top e_j^K)\) instead.
The same TV/output proof applies to its range. Omitting the query-error terms
would incorrectly turn a fixed-query cache analysis into a network-level bound.

## Why top-1 and distribution quality are different

For arbitrary final model logits \(z\) and perturbed logits \(z+e\), define
\(\Delta_z=\max e-\min e\). If the original top-1 margin is strictly larger
than \(\Delta_z\), the top-1 token is preserved. This is sufficient, not
necessary; equal top-1 on measured tokens does not bound NLL or future tokens.
For any target token \(y\), the change in its negative log likelihood is

\[
 \widehat{\operatorname{NLL}}(y)-\operatorname{NLL}(y)
 =\log\mathbb E_{p}e^{e_j}-e_y,
 \qquad |\widehat{\operatorname{NLL}}(y)-\operatorname{NLL}(y)|\leq\Delta_z.
\]

The logarithm lies between the minimum and maximum logit perturbations. This
justifies reporting both task/top-1 behavior and distributional metrics; it does
not justify retroactively changing a historical acceptance gate. For a future
manuscript, any prospective non-inferiority margin must be justified separately
in task units, not chosen after viewing final evaluation.
