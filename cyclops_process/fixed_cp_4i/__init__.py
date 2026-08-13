"""Unified fixed-cell imaging pipeline (cell painting + 4i).

Cell painting and 4i are the same pipeline shape — N imaging rounds ("parts" for
CP, "rounds" for 4i), each with a nuclei channel, stitched/segmented, registered
to the live-cell phenotyping store, warped into the v3 store, then linked to
pheno + ISS. The only differences are settings: channel definitions, unit naming,
register-YAML names, and unit count. Those live in ``modality_config.py``; every
step script here is universal and takes ``--modality {cp,4i}``.
"""
