#!/usr/bin/env python
"""Step 2b: Embed all documents and queries with Specter2.

Similar to 02_embed.py but uses the Specter2 model (allenai/specter2_base)
with [CLS] pooling. Produces specter2_macro.npz, specter2_meso.npz,
specter2_micro.npz, and specter2_queries.npz in data/embeddings/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.embed import embed_all_specter2


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")
    embed_all_specter2(config, project_root=project_root)


if __name__ == "__main__":
    main()
