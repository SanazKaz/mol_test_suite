#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDKit: straighter 2D depictions + BIG legends + reference first

This script:
- Forces CoordGen-based 2D layout (straighter chains)
- Tries to "straighten" depictions if available (RDKit >= ~2022.09)
- Places the reference molecule first
- Renders large legends using PIL so font size actually changes on all RDKit builds
- Saves molecular_comparison_grid.png in the current folder
"""

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap
import sys

# ---------- depiction helpers ----------

def compute_straight_2d(mol):
    """
    Generate 2D coordinates with a strong bias towards straighter layouts.
    Tries (in order):
      1) CoordGen-based 2D coords (rdDepictor.SetPreferCoordGen)
      2) Canonical orientation
      3) StraightenDepiction (if available on your RDKit)
    """
    if mol is None:
        return None

    # Prefer CoordGen (often straighter than legacy)
    try:
        rdDepictor.SetPreferCoordGen(True)
    except Exception:
        pass

    # Compute coords (canonical orientation if supported)
    try:
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    except TypeError:
        rdDepictor.Compute2DCoords(mol)

    # Try to "straighten" (newer RDKit)
    try:
        rdDepictor.StraightenDepiction(mol)
    except AttributeError:
        pass

    return mol

# ---------- legend rendering (PIL) ----------

def _get_font(size):
    """
    Try a sensible TTF font; fall back to PIL default if not found.
    """
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

def grid_with_big_legends(
    mols,
    legends,
    subImgSize=(550, 550),
    molsPerRow=3,
    font_size=72,
    line_width=28,
    legend_pad_px=20,
    bg=(255, 255, 255, 255),
    text_color=(0, 0, 0, 255),
):
    """
    Draw molecules in a grid (no RDKit legends), then add large legends via PIL.
    Works regardless of RDKit legendFontSize support.
    """
    if legends is None:
        legends = [""] * len(mols)

    # 1) draw molecules-only grid using RDKit
    png = Draw.MolsToGridImage(
        mols,
        legends=None,  # we draw our own legends
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        useSVG=False,
        returnPNG=True,
    )
    png_bytes = png if isinstance(png, (bytes, bytearray)) else getattr(png, "data", bytes(png))
    base = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    cell_w, cell_h = subImgSize
    rows = (len(mols) + molsPerRow - 1) // molsPerRow

    # 2) Estimate legend block height (multi-line friendly)
    font = _get_font(font_size)
    # rough multi-line height: 1.25 line spacing * max 3 lines typical
    line_height = font_size + int(0.25 * font_size)

    # Allow for wrapped multi-line legends; reserve up to ~4 lines
    max_lines = 4
    legend_block_h = max( int(line_height * max_lines + legend_pad_px), int(1.6 * font_size) )

    # 3) create taller canvas
    canvas_h = base.height + rows * legend_block_h
    canvas = Image.new("RGBA", (base.width, canvas_h), bg)
    canvas.paste(base, (0, 0))

    # 4) draw wrapped legends centered under each cell
    draw = ImageDraw.Draw(canvas)
    for idx, text in enumerate(legends):
        row = idx // molsPerRow
        col = idx % molsPerRow

        # Wrap text by characters (simple + robust across fonts)
        wrapped = textwrap.fill(text, width=line_width)

        # Measure.
        try:
            # Pillow 8.0+: precise bounding box
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            # Fallback estimate
            lines = wrapped.splitlines() or [""]
            text_w = max(font.getsize(l)[0] for l in lines)
            text_h = line_height * len(lines)

        # Position centered beneath its cell
        x0 = col * cell_w
        y0 = row * (cell_h + legend_block_h) + cell_h
        cx = x0 + cell_w // 2
        tx = cx - text_w // 2
        ty = y0 + (legend_block_h - text_h) // 2

        draw.multiline_text((tx, ty), wrapped, font=font, fill=text_color, align="center", spacing=int(0.15*font_size))

    return canvas

# ---------- main ----------

def main():
    # Input molecules (reference FIRST as requested)
    m_max_fcfp4 = "[H]c1c([H])c(OC([H])([H])C(=O)N([H])c2c([H])c([H])c(C([H])([H])N3C([H])([H])C([H])([H])C([H])([H])C([H])([H])C3([H])[H])c([H])c2[H])c(Cl)c([H])c1Cl"
    m_max_ecfp4 = "[H]c1c([H])c(OC([H])([H])C([H])([H])C([H])([H])C([H])([H])N2C([H])([H])C([H])([H])C([H])([H])C([H])([H])C2([H])[H])c2c(c1[H])C(=O)N([H])C([H])([H])C2([H])[H]"
    ref_smiles   = "C1CC(=O)NC2=C1C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl"  # Aripiprazole

    mol_fcfp4 = Chem.MolFromSmiles(m_max_fcfp4)
    mol_ecfp4 = Chem.MolFromSmiles(m_max_ecfp4)
    mol_ref   = Chem.MolFromSmiles(ref_smiles)

    # Straighter 2D
    mols = [compute_straight_2d(m) for m in [mol_ref, mol_ecfp4, mol_fcfp4]]

    legends = [
        "Aripiprazole\nReference Compound\nPDB: 7e2z",
        "Maximum ECFP4\nSimilarity: 0.354\nPDB: 4tk0",
        "Maximum FCFP4\nSimilarity: 0.517\nPDB: 5foq",
    ]

    # Build image with BIG legends
    img = grid_with_big_legends(
        mols,
        legends,
        subImgSize=(550, 550),  # tile size for each molecule
        molsPerRow=3,
        font_size=28,       # <<< make this bigger/smaller as you like
        line_width=20,         # wrapping width
        legend_pad_px=24,      # extra padding inside legend area
    )

    out_name = "molecular_comparison_grid.png"
    img.convert("RGB").save(out_name)
    print(f"Saved: {out_name}")

    # Optional preview (may open an image viewer depending on OS)
    try:
        img.show()
    except Exception:
        pass

if __name__ == "__main__":
    sys.exit(main())
