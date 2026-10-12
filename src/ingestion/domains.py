"""Domain taxonomy for the corpus and the manual document -> domain assignment.

With only 12 documents, domains are assigned by hand (reviewed against the
inventory) instead of using an automatic classifier. Domains are defined by
application area, not by technique: almost every paper is applied ML, so an
"AI" domain would not filter anything.
"""

DOMAINS: dict[str, str] = {
    "healthcare": (
        "Lung cancer research: CT-based nodule/tumor detection, segmentation and "
        "classification with deep learning, plus NSCLC subtyping from lung "
        "microbiome data."
    ),
    "remote_sensing": (
        "Satellite imagery analysis: land use / land cover classification, NDVI "
        "change detection, Sentinel-2, satellite image time series (deep learning "
        "and classical ML)."
    ),
    "ai_security": (
        "Security of LLM applications: prompt injection, context manipulation, "
        "authenticated prompts and context."
    ),
}

# Keys are content-hash doc_ids from data/processed/inventory.csv.
DOC_DOMAINS: dict[str, str] = {
    # healthcare
    "a731e118b66b": "healthcare",      # Capsule Network hybrid model, lung nodule detection
    "6fb414ce0903": "healthcare",      # YOLOv11 + scale-aware curriculum learning
    "2d2127229350": "healthcare",      # Object detection for early lung cancer detection
    "b85ac180014f": "healthcare",      # VGG16 vs CNN, CT scan classification
    "93c4d3a53ca1": "healthcare",      # Deep learning lung cancer diagnosis from CT (Shatnawi)
    "9508bda7b433": "healthcare",      # Revisiting lung tumor segmentation from CT
    "cf2ee32231c8": "healthcare",      # NSCLC subtypes from lung microbiome
    # remote_sensing
    "36ecde91c4ff": "remote_sensing",  # Review: DL for satellite image time series
    "564f8799c7ef": "remote_sensing",  # ML land cover classification, mountainous landscape
    "3d40606d4ed2": "remote_sensing",  # DL land-cover classification, high-res imagery
    "9a346178fcd5": "remote_sensing",  # Sentinel-2 LULC and NDVI change, Gelephu (Bhutan)
    # ai_security
    "c9d5afe5bfb8": "ai_security",     # Authenticated prompts and context for LLM apps
}


def get_domain(doc_id: str) -> str:
    """Return the domain of a document; fail loudly for unlabeled documents."""
    try:
        return DOC_DOMAINS[doc_id]
    except KeyError:
        raise KeyError(
            f"Document {doc_id} has no domain. Add it to DOC_DOMAINS in {__name__}."
        ) from None


assert set(DOC_DOMAINS.values()) <= DOMAINS.keys(), "DOC_DOMAINS uses an undefined domain"
