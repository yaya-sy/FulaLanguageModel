[x] Masking with indexing (see Whisper implementation)
[ ] Use more lisible matrix operations : einsop or einsum
[ ] Use gradient accumulation for large batch size
[ ] Use autocast (see amp pytorch package) to use float with less resolution
when possible (this helps to accelarate operations). Since the transformers architecture uses many linear functions, we can use float with less precision.