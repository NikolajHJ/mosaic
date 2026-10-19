"""Optional fixed-pose Boltz-2 affinity backend for Mosaic probability graphs.

The native worker uses PyTorch; the optimizer and chemistry remain JAX. Import
the bridge explicitly so native preparation never imports JAX or a decoder.
"""
