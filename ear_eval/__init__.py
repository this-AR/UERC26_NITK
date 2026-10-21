"""ear_eval: shared library for the ear-recognition matching evaluation.

    config      paths and the model selection (DT2_MODEL / DT2_CLAHE)
    datasets    image discovery for AWE, ITW, UBEAR and NITJ
    models      model registry (pruned TinyViT, Swin-V2-B) and input preprocessing
    embeddings  embedding extraction, L2 / mean-centring, on-disk cache
    metrics     verification (EER, VER, AUC, DI) and identification (rank-1 / CRR) metrics
    plotting    shared palette, matplotlib styling and the gnuplot runner
"""
