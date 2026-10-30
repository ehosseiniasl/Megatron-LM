# Runtime BOS-to-EOS boundary rewriting

## Proposed behavior

`drop_bos` is an opt-in runtime transformation over samples produced by the
existing BOS-leading epoch indexes. It replaces BOS inputs at indexed document
starts with EOS but does not rebuild indexes or change `sizes` and `doc_ids`.

For a sample starting partway through a document, the intended stream is:

```text
x:     [0,  1]   [EOS, 2, 3]    [EOS]
y:     [1, EOS]  [2,   3, EOS]  [4]
sizes: [2, 3, 1]
```

The bracketed sizes stay exactly as recorded by the epoch index. This is safe
for the current experiment only because the adapter requires complementary
configuration:

```text
interdocument_masking=True  => drop_bos=False
interdocument_masking=False => drop_bos=True
```

With interdocument masking disabled, `sizes` are logging metadata and do not
control attention boundaries or position-ID resets. The assertion should remain
until every consumer can handle both boundary conventions deliberately.

EOS-to-next-token loss masking is not implemented in the Mistral dataloader.
Megatron already provides that behavior through `--eod-mask-loss` (internally
`eod_mask_loss`), which zeros loss wherever the input token is EOD/EOS.

## Comparison with the standard Megatron dataset path

`BlendedDataset` selects a constituent dataset and sample; it does not rewrite
document boundaries. `GPTDataset` fetches a flattened token stream and, with
`add_extra_token_to_sequence=True`, returns `text[:-1]` as `tokens` and
`text[1:]` as `labels`. `eod_mask_loss` optionally masks positions whose input
token is EOD.

Standard Megatron therefore gets canonical BOS-free behavior when indexed
documents themselves contain content followed by EOD:

```text
indexed: [0, 1, EOS] [2, 3, EOS] [4, ...]
x:       [0, 1, EOS] [2, 3, EOS]
y:       [1, EOS]    [2, 3, EOS] [4]
```

Runtime replacement can make the visible boundary tokens match this stream in
many samples, but it cannot change which token window the BOS-leading epoch
index selected.

## Why this design is dangerous

- It is not token-for-token equivalent to index-time BOS removal. If a sample
  starts on BOS, runtime replacement creates a leading EOS; with
  `--eod-mask-loss`, the following target is dropped instead of selecting the
  later input/target pair that canonical indexing would have included.
- `sizes` and `doc_ids` deliberately retain the old indexed segmentation. They
  are suitable for current logging, but enabling THD/interdocument masking with
  these values would reset attention and position IDs using the wrong boundary
  convention. The adapter assertion prevents that configuration for now.
- Epoch-index files and their human-readable metadata still show BOS-leading
  tokens, while runtime batches contain EOS at those positions. This makes
  debugging a sampled sequence against its cached entry less direct.
- `drop_bos` is absent from epoch-index cache identity and saved datasource
  state. The same cached index and resume position can produce different input
  tokens after the flag changes without a cache/state mismatch.
- Image token offsets remain unchanged, which avoids moving image tensors, but
  a BOS immediately before an image becomes EOS. Both the image target mask and
  Megatron's EOD loss mask can then mask that transition; useful loss must come
  from a later non-image token.
- Replacement trusts `sizes` to identify every internal document start. Stale
  or malformed metadata can rewrite the wrong input or fail only at load time.
- Fixed-overlap and perplexity paths that do not use this adapter retain their
  existing boundary behavior, so training and evaluation can disagree.

`drop_bos` has no schema default: every config must explicitly select its
boundary convention. This runtime mode is appropriate for a constrained
experiment that must reuse existing epoch indexes. Prefer removing BOS while
building indexes when canonical Megatron semantics, cache identity, or
consistent training and evaluation streams matter.
