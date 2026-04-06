"""BetaEarth: Emulating Earth Observation Foundation Model Embeddings.

Usage:
    from betaearth import BetaEarth

    model = BetaEarth.from_pretrained("asterisk-labs/betaearth-segformer-film")
    embedding = model.predict(
        s2_l2a=s2_l2a_arr,   # (9, H, W) uint16 DN
        s2_l1c=s2_l1c_arr,   # (9, H, W) uint16 DN
        s1=s1_arr,            # (2, H, W) float32 linear power
        dem=dem_arr,          # (1, H, W) float32 meters
        doy=182,
    )
    # embedding: (H, W, 64) numpy array, L2-normalised per pixel
"""

from .modeling_betaearth import BetaEarth

__version__ = "0.1.0"
__all__ = ["BetaEarth"]
