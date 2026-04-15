#!/usr/bin/env python3
"""
Chemistry Compound Analyzer — Web Application
Run with:  uvicorn app:app --host 0.0.0.0 --port 8000
Then open: http://localhost:8000
"""

import sys, os, json, re, base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ── Optional library imports ──────────────────────────────────────────────────
try:
    import ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    import pubchempy as pcp
    PUBCHEM_OK = True
except ImportError:
    PUBCHEM_OK = False

try:
    from molmass import Formula as MolFormula
    MOLMASS_OK = True
except ImportError:
    MOLMASS_OK = False

try:
    import periodictable as pt_lib
    PERIODIC_OK = True
except ImportError:
    PERIODIC_OK = False

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Draw
    RDLogger.DisableLog('rdApp.*')
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

try:
    import py2opsin
    PY2OPSIN_OK = True
except ImportError:
    PY2OPSIN_OK = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ase.io import read as ase_read
    from ase.visualize.plot import plot_atoms as ase_plot_atoms
    ASE_OK = True
except ImportError:
    ASE_OK = False


# ── Configuration ──────────────────────────────────────────────────────────────
MODEL = "qwen2.5:14b"
MP_API_KEY = os.environ.get("MP_API_KEY", "pNalliqeGlGGpCD5h02vfxwXI5uyrM1Z")

# ── System prompts ─────────────────────────────────────────────────────────────
ISOMER_SYSTEM = """You are an expert organic chemist. Given a molecular formula:

Validate it first:
- If it contains no carbon (inorganic, e.g. H2O, NaCl), reply with: INORGANIC: <one line why>
- If it violates valence rules (degree of unsaturation < 0), reply with: INVALID: <one line why>
- If it is not a chemical formula at all, reply with: NOT A FORMULA: <one line why>
- If chemically valid but no known stable organic compound exists, reply with: UNKNOWN: <one line why>

If valid and known, list every distinct structural (constitutional) isomer. Rules:

CRITICAL — avoid these mistakes:
- One compound = ONE entry only. Different names for the same structure (e.g. ethyne / acetylene,
  ethanol / ethyl alcohol) must NOT appear as separate numbered entries. Pick the IUPAC name.
- Always show ALL atoms in the structural formula, including every hydrogen.
  WRONG: C≡C      RIGHT: H-C≡C-H
- Only list isomers that are genuinely structurally different (different connectivity).

Format each entry like this:

1. <IUPAC name>  (<common name if different>)
   <full line-notation structural formula showing every atom>

Use plain text only. No markdown."""

ANALYSIS_SYSTEM = """You are an expert chemist (organic and inorganic). \
You will receive verified data collected from chemistry libraries and then provide a full analysis.

CRITICAL: Analyze ONLY the exact compound specified. Never substitute a similar or more common compound.
Example: if asked for CuS, analyze CuS — NOT Cu2S or CuO or any other formula.
The compound name and formula in the verified data are authoritative — follow them exactly.

RING CLOSURE NOTATION:
  When two atoms in a structural formula are marked with * they are bonded together, closing a ring.
  Examples:
    CH*-CH2-CH2-CH2-CH2-CH*   -> cyclohexane (6-membered ring)
    CH*=CH-CH=CH-CH=CH*        -> benzene (aromatic, 6-membered ring)
  Always interpret the * notation before analyzing.

Use the verified data provided. Do NOT contradict any verified values.
Be concise — 2-4 sentences per section maximum.

NAME             Common name(s) and IUPAC name.
CHEMICAL FAMILY  Class of compound and why (one sentence).
STRUCTURE        Geometry, key bond types, hybridization of main atoms.
FUNCTIONAL GROUPS  Each group present; write "none" if absent.
KEY REACTIONS    2-3 most important reactions with conditions.
PROPERTIES       State, boiling/melting point, solubility, odor.
USES             Main industrial, biological, or everyday applications.
SYNTHESIS        How this compound can be made in a lab or industrially.
  - Write the main reaction equation and conditions (temperature, catalyst, solvent).
  - If multiple practical routes exist, mention the simplest one first.
  - End with 1-2 specific safety precautions for that synthesis.
  - If the compound is a controlled substance, explosive, strong poison, or nerve agent,
    write only: "Synthesis not covered — hazardous or regulated compound."

Use plain text only — no markdown, no LaTeX, no backslashes of any kind.
Write chemical equations like this:  C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O
Never use \\text{}, \\rightarrow, \\frac{}, subscript/superscript notation, or any LaTeX symbols."""

MIXTURE_SYSTEM = """You are an expert chemist specializing in chemical mixtures and solutions.
Analyze the given chemical mixture or reagent solution.

Use plain text only — no markdown, no LaTeX, no backslashes.

Cover these topics naturally in flowing paragraphs (no rigid headers):
- What the mixture is, its composition and typical preparation ratios
- The key chemical reactions or interactions that occur between components
- Why this specific combination produces unique properties not found in individual components
- Practical applications, industries, and laboratory uses
- Hazards, safety precautions, and proper handling
- Any notable historical or industrial significance

Be concise and accurate. Do not invent data."""

POLYMER_SYSTEM = """You are an expert polymer chemist.
Analyze the given polymer, focusing on its repeating unit and chain properties.

Use plain text only — no markdown, no LaTeX, no backslashes.

Cover these topics naturally in flowing paragraphs (no rigid headers):
- What the polymer is and its repeating unit structure
- Polymerization mechanism (addition, condensation, ring-opening, etc.)
- Physical and mechanical properties (flexibility, tensile strength, crystallinity)
- Thermal properties (glass transition Tg, melting point Tm if applicable)
- Common applications and industries it is used in
- Environmental impact, recyclability, or biodegradability

Be concise and accurate. Do not invent data."""

CRYSTAL_SYSTEM = """You are an expert materials scientist and crystallographer.
Analyze the given crystal, mineral, or solid-state material.

If verified crystallographic data is provided (crystal system, space group, Bravais lattice),
use it exactly — do not contradict or substitute different values.

Use plain text only — no markdown, no LaTeX, no backslashes.

Cover these topics naturally in flowing paragraphs (no rigid headers):
- What it is and its crystal system / lattice type (use verified data if provided)
- Bonding and structure (ionic, covalent network, metallic, etc.)
- Key physical properties (hardness, melting point, optical, electrical)
- Where it occurs naturally or how it is made
- Main uses

Be concise and clear. Do not invent data you are unsure about."""

REACTION_SYSTEM = """You are an expert chemist (organic and inorganic).

STRICT FORMATTING RULES — follow these before anything else:
- Plain text ONLY. Zero LaTeX. Zero markdown.
- Never write \\[, \\], \\text{}, \\frac{}, \\rightarrow, or any backslash symbol.
- Never use superscript/subscript notation.
- Write ALL chemical equations like this:  2 H2 + O2 -> 2 H2O
- Charges: write Mn2+ not Mn^{2+}, write OH- not OH^{-}
- If you find yourself writing a backslash, stop and rewrite without it.

Given reactants and optional conditions or reagents, predict the major product(s) and explain the
reaction mechanism in detail.

Format your response with these sections:

PREDICTED EQUATION
  Write the balanced chemical equation in plain text.
  Example: 2 H2 + O2 -> 2 H2O

REACTION TYPE
  Name the specific reaction class (combustion, SN2, E2, Diels-Alder, acid-base neutralization,
  addition, elimination, substitution, redox, etc.)

MECHANISM
  Step-by-step explanation of how the reaction proceeds at the electron/bond level.
  Label each step (Step 1: ..., Step 2: ...).
  Write every equation in plain text using ->.
  Identify any key intermediates (carbocation, carbanion, radical, transition state, etc.).

DRIVING FORCE
  Explain why this reaction is thermodynamically or kinetically favored.
  Mention bond energies, stability of products, entropy, or other relevant factors.

CONDITIONS (only if the user did not provide any)
  If no conditions or reagents were given, suggest the typical conditions needed
  (temperature, catalyst, solvent, pressure, etc.) and explain briefly why each is required.
  Skip this section entirely if conditions were already provided.

SIDE REACTIONS
  List any common competing reactions or notable by-products."""

LEWIS_SYSTEM = """You are an expert chemist. Draw the Lewis structure of the given compound in ASCII art.

VALENCE ELECTRON RULES (follow exactly):
  H : 1 bond, 0 lone pairs
  C : 4 bonds
  N : 3 bonds, 1 lone pair  (:N)
  O : 2 bonds, 2 lone pairs  (e.g. H-:O:-H  not just :O:)
  F : 1 bond, 3 lone pairs  (:F:)
  Cl: 1 bond, 3 lone pairs  (:Cl:)
  S : 2 bonds, 2 lone pairs
  P : 3 bonds, 1 lone pair

BOND RULES:
  - Use  -  for single,  =  for double,  ≡  for triple bond
  - ONLY use triple bonds for compounds that genuinely have them (N2, CO, HC≡CH, HCN, etc.)
  - Common single-bond compounds: H2O, HF, HCl, NH3, CH4, OF2

LONE PAIR RULES:
  - Show every lone pair as  :  next to the atom — never omit on O, N, F, Cl
  - H2O correct:   H - :O: - H
  - OF2 correct:   :F: - :O: - :F:  (single bonds only)

Show ALL atoms explicitly including every H atom.
Output ONLY the ASCII art — no words, no explanation."""


# ── Element symbols ────────────────────────────────────────────────────────────
ELEMENTS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
    'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
    'Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd',
    'Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm',
    'Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os',
    'Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa',
    'U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
}


# ── Chemistry functions ────────────────────────────────────────────────────────
def looks_like_formula(s: str) -> bool:
    if " " in s or not s:
        return False
    if s[0].islower():
        return False
    if re.match(r'^[A-Z][a-z]?$', s):
        return True
    has_digit  = bool(re.search(r'\d', s))
    multi_caps = len(re.findall(r'[A-Z]', s)) > 1
    return has_digit or multi_caps


def to_hill_formula(formula: str) -> str:
    """Convert structural notation (e.g. C2H5OH) to Hill notation (C2H6O)."""
    if not MOLMASS_OK:
        return formula
    try:
        f = MolFormula(formula)
        counts: dict = {}
        for row in f.composition():
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                counts[str(row[0])] = int(row[1])
        if not counts:
            return formula
        hill = ""
        for el in (["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))):
            if el in counts:
                n = counts[el]
                hill += el + (str(n) if n > 1 else "")
        return hill or formula
    except Exception:
        return formula


def has_carbon(formula: str) -> bool:
    return bool(re.search(r'C(?![a-z])', formula))


def normalize_formula(s: str) -> str:
    """Fix capitalisation for user input (e.g. 'c6h12o6' → 'C6H12O6').
    Two adjacent UPPERCASE letters are never merged into a two-letter symbol
    so that 'CH3COONa' stays intact instead of becoming 'CH3CoONa'."""
    result = ""
    i = 0
    while i < len(s):
        c = s[i]
        if not c.isalpha():
            result += c
            i += 1
            continue
        next_c = s[i + 1] if i + 1 < len(s) else ""
        if next_c.isalpha():
            two = c.upper() + next_c.lower()
            # Only merge into a two-letter element when the pair is NOT two
            # uppercase letters (e.g. Na ✓, na ✓, but CO or NA → don't merge)
            if two in ELEMENTS and not (c.isupper() and next_c.isupper()):
                result += two
                i += 2
                continue
        result += c.upper()
        i += 1
    return result


def gather_molmass(formula: str) -> dict:
    if not MOLMASS_OK:
        return {}
    try:
        f = MolFormula(formula)
        result = {"molecular_weight": f"{f.mass:.4f} g/mol"}
        try:
            result["monoisotopic_mass"] = f"{f.isotope.mass:.4f} g/mol"
        except Exception:
            pass
        try:
            comp = f.composition()
            parts = []
            for row in comp:
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    sym, cnt, _, frac = row[0], row[1], row[2], row[3]
                    parts.append(f"{sym}: {cnt} atoms ({frac * 100:.1f}%)")
                elif hasattr(row, "symbol"):
                    parts.append(f"{row.symbol}: {row.count}")
            if parts:
                result["composition"] = ", ".join(parts)
        except Exception:
            pass
        return result
    except Exception as e:
        return {"error": str(e)}


def gather_elements(formula: str) -> list:
    if not PERIODIC_OK:
        return []
    seen = set()
    lines = []
    for sym, _ in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if not sym or sym in seen:
            continue
        seen.add(sym)
        try:
            el = getattr(pt_lib, sym)
            lines.append({"name": el.name, "symbol": sym,
                          "number": el.number, "mass": round(el.mass, 3)})
        except AttributeError:
            pass
    return lines


def gather_pubchem(query: str, by_formula: bool) -> dict:
    if not PUBCHEM_OK:
        return {}
    try:
        namespace = "formula" if by_formula else "name"
        compounds = pcp.get_compounds(query, namespace, listkey_count=1)
        if not compounds:
            return {"found": False}
        c = compounds[0]
        d = {"found": True, "cid": c.cid}
        for attr, key in [("iupac_name", "iupac_name"), ("molecular_formula", "formula")]:
            val = getattr(c, attr, None)
            if val:
                d[key] = str(val)
        if c.synonyms:
            d["common_name"] = c.synonyms[0]
        for smiles_attr in ("smiles", "isomeric_smiles", "canonical_smiles"):
            val = getattr(c, smiles_attr, None)
            if val:
                d["smiles"] = val
                break
        if getattr(c, "molecular_weight", None):
            d["mw"] = f"{c.molecular_weight} g/mol"
        for attr, key in [
            ("xlogp", "logp"), ("tpsa", "tpsa"),
            ("h_bond_donor_count", "hbd"), ("h_bond_acceptor_count", "hba"),
        ]:
            val = getattr(c, attr, None)
            if val is not None:
                d[key] = val
        return d
    except Exception as e:
        return {"error": str(e)}


def gather_rdkit(smiles: str) -> dict:
    if not RDKIT_OK or not smiles:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "exact_mw":          f"{Descriptors.ExactMolWt(mol):.4f} g/mol",
            "logp":              f"{Descriptors.MolLogP(mol):.3f}",
            "tpsa":              f"{Descriptors.TPSA(mol):.2f} Ų",
            "h_bond_donors":     rdMolDescriptors.CalcNumHBD(mol),
            "h_bond_acceptors":  rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds":   rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings":             rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings":    rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    except Exception:
        return {}


# ── Chemical mixture database ─────────────────────────────────────────────────
# Each entry: name → {full_name, type, components: [(name, formula, ratio)], description}
MIXTURE_DB: dict = {
    # Acid mixtures
    "aqua regia": {
        "full_name": "Aqua Regia",
        "type": "Oxidizing Acid Mixture",
        "components": [("Hydrochloric acid", "HCl", "3 parts"), ("Nitric acid", "HNO₃", "1 part")],
        "ratio": "HCl : HNO₃ = 3 : 1 (by volume)",
        "description": "Fuming corrosive mixture capable of dissolving noble metals (gold, platinum, palladium)",
    },
    "nitrating mixture": {
        "full_name": "Nitrating Mixture",
        "type": "Electrophilic Nitration Reagent",
        "components": [("Concentrated sulfuric acid", "H₂SO₄", "1 part"), ("Concentrated nitric acid", "HNO₃", "1 part")],
        "ratio": "H₂SO₄ : HNO₃ = 1 : 1 (by volume)",
        "description": "Used for electrophilic aromatic nitration; H₂SO₄ generates the nitronium ion NO₂⁺",
    },
    "mixed acid": {
        "full_name": "Mixed Acid (Nitrating Mixture)",
        "type": "Electrophilic Nitration Reagent",
        "components": [("Concentrated sulfuric acid", "H₂SO₄", "1 part"), ("Concentrated nitric acid", "HNO₃", "1 part")],
        "ratio": "H₂SO₄ : HNO₃ = 1 : 1 (by volume)",
        "description": "Industrial nitration agent; same composition as nitrating mixture",
    },
    "piranha solution": {
        "full_name": "Piranha Solution (Caro's Acid Mixture)",
        "type": "Strong Oxidizing Cleaner",
        "components": [("Concentrated sulfuric acid", "H₂SO₄", "3 parts"), ("Hydrogen peroxide", "H₂O₂", "1 part")],
        "ratio": "H₂SO₄ : H₂O₂ = 3 : 1 (by volume)",
        "description": "Extremely aggressive oxidizer used to clean organic residues from substrates in semiconductor fabrication",
    },
    "piranha":  {  # alias
        "full_name": "Piranha Solution",
        "type": "Strong Oxidizing Cleaner",
        "components": [("Concentrated sulfuric acid", "H₂SO₄", "3 parts"), ("Hydrogen peroxide", "H₂O₂", "1 part")],
        "ratio": "H₂SO₄ : H₂O₂ = 3 : 1 (by volume)",
        "description": "Extremely aggressive oxidizer for removing organic residues; extremely exothermic upon mixing",
    },
    # Metallographic etchants
    "nital": {
        "full_name": "Nital",
        "type": "Metallographic Etchant",
        "components": [("Nitric acid", "HNO₃", "2–4%"), ("Ethanol", "C₂H₅OH", "96–98%")],
        "ratio": "HNO₃ : Ethanol = 2–4 : 96–98 (by volume)",
        "description": "Standard etchant for revealing microstructure of iron, steel, and carbon steels",
    },
    "keller's reagent": {
        "full_name": "Keller's Reagent",
        "type": "Metallographic Etchant (Aluminium)",
        "components": [("Hydrofluoric acid", "HF", "2 mL"), ("Hydrochloric acid", "HCl", "3 mL"),
                       ("Nitric acid", "HNO₃", "5 mL"), ("Water", "H₂O", "190 mL")],
        "ratio": "HF : HCl : HNO₃ : H₂O = 2 : 3 : 5 : 190",
        "description": "Standard etchant for revealing grain structure and second-phase particles in aluminium alloys",
    },
    "fry's reagent": {
        "full_name": "Fry's Reagent",
        "type": "Metallographic Etchant (Copper alloys)",
        "components": [("Cupric chloride", "CuCl₂", "40 g"), ("Hydrochloric acid", "HCl", "20 mL"),
                       ("Water", "H₂O", "25 mL"), ("Ethanol", "C₂H₅OH", "25 mL")],
        "ratio": "CuCl₂(40g) : HCl(20mL) : H₂O(25mL) : Ethanol(25mL)",
        "description": "Etchant for cold-worked copper and brass; reveals deformation bands and grain boundaries",
    },
    # Analytical / laboratory reagents
    "fehling's solution": {
        "full_name": "Fehling's Solution",
        "type": "Analytical Reagent (Reducing Sugar Test)",
        "components": [("Copper(II) sulfate solution", "CuSO₄·5H₂O", "Solution A"), ("Sodium potassium tartrate + NaOH", "KNaC₄H₄O₆", "Solution B")],
        "ratio": "Equal volumes of Solution A and Solution B mixed before use",
        "description": "Classic test for reducing sugars; Cu²⁺ is reduced to brick-red Cu₂O precipitate by aldehydes",
    },
    "benedict's reagent": {
        "full_name": "Benedict's Reagent",
        "type": "Analytical Reagent (Reducing Sugar Test)",
        "components": [("Copper(II) sulfate", "CuSO₄", "17.3 g/L"), ("Sodium citrate", "Na₃C₆H₅O₇", "173 g/L"),
                       ("Sodium carbonate", "Na₂CO₃", "100 g/L")],
        "ratio": "Single solution: CuSO₄ + sodium citrate + Na₂CO₃ in water",
        "description": "More stable than Fehling's; same Cu²⁺ reduction principle; brick-red precipitate indicates reducing sugars",
    },
    "tollens' reagent": {
        "full_name": "Tollens' Reagent",
        "type": "Analytical Reagent (Silver Mirror Test)",
        "components": [("Silver nitrate", "AgNO₃", "aqueous"), ("Ammonia solution", "NH₃", "excess")],
        "ratio": "AgNO₃ aq. + NH₃ until precipitate redissolves → [Ag(NH₃)₂]⁺",
        "description": "Silver mirror test for aldehydes; Ag⁺ is reduced to Ag⁰ producing a reflective silver mirror",
    },
    "lugol's solution": {
        "full_name": "Lugol's Iodine Solution",
        "type": "Staining Reagent",
        "components": [("Iodine", "I₂", "1 part"), ("Potassium iodide", "KI", "2 parts"), ("Water", "H₂O", "solvent")],
        "ratio": "I₂ : KI : H₂O = 1 : 2 : ~30 (by mass, typical 5% w/v)",
        "description": "Staining reagent; turns blue-black with starch; used in biology, microbiology, and as a disinfectant",
    },
    "millon's reagent": {
        "full_name": "Millon's Reagent",
        "type": "Analytical Reagent (Protein/Tyrosine Test)",
        "components": [("Mercury", "Hg", "1 part"), ("Concentrated nitric acid", "HNO₃", "2 parts")],
        "ratio": "Hg dissolved in HNO₃; final solution contains Hg⁺ and Hg²⁺ nitrates",
        "description": "Test for phenolic amino acids (tyrosine); produces brick-red precipitate with proteins",
    },
    "biuret reagent": {
        "full_name": "Biuret Reagent",
        "type": "Analytical Reagent (Protein Test)",
        "components": [("Sodium hydroxide", "NaOH", "6 M, 1 mL"), ("Copper(II) sulfate", "CuSO₄", "1% aq., 2 drops")],
        "ratio": "NaOH added first, then dilute CuSO₄",
        "description": "Detects peptide bonds in proteins; Cu²⁺ forms violet complex with ≥2 peptide bonds in alkaline solution",
    },
    # Buffer solutions
    "phosphate buffered saline": {
        "full_name": "Phosphate Buffered Saline (PBS)",
        "type": "Buffer Solution",
        "components": [("Sodium chloride", "NaCl", "137 mM"), ("Potassium chloride", "KCl", "2.7 mM"),
                       ("Disodium phosphate", "Na₂HPO₄", "10 mM"), ("Potassium phosphate", "KH₂PO₄", "1.8 mM")],
        "ratio": "pH 7.4 at standard 1× concentration",
        "description": "Isotonic buffer ubiquitous in biology; matches physiological osmolarity and pH of human blood",
    },
    "pbs": {
        "full_name": "Phosphate Buffered Saline (PBS)",
        "type": "Buffer Solution",
        "components": [("Sodium chloride", "NaCl", "137 mM"), ("Potassium chloride", "KCl", "2.7 mM"),
                       ("Disodium phosphate", "Na₂HPO₄", "10 mM"), ("Potassium phosphate", "KH₂PO₄", "1.8 mM")],
        "ratio": "pH 7.4 at 1× concentration",
        "description": "Standard isotonic physiological buffer used in cell biology and biochemistry",
    },
    # Alloys / metallurgical
    "amalgam": {
        "full_name": "Amalgam",
        "type": "Mercury Alloy Mixture",
        "components": [("Mercury", "Hg", "variable"), ("Other metal (Ag, Au, Sn, Cu…)", "variable", "variable")],
        "ratio": "Mercury content varies; dental amalgam: ~50% Hg + Ag/Sn/Cu alloy powder",
        "description": "Alloy of mercury with another metal; used historically in dentistry, gold extraction, and mirror silvering",
    },
    # Fuel / combustion
    "syngas": {
        "full_name": "Synthesis Gas (Syngas)",
        "type": "Industrial Fuel/Feedstock Gas Mixture",
        "components": [("Carbon monoxide", "CO", "variable"), ("Hydrogen", "H₂", "variable")],
        "ratio": "H₂ : CO ratio varies by process (steam reforming: ~3:1; partial oxidation: ~2:1)",
        "description": "Key industrial feedstock for producing ammonia, methanol, and liquid fuels via Fischer-Tropsch",
    },
    "producer gas": {
        "full_name": "Producer Gas",
        "type": "Fuel Gas Mixture",
        "components": [("Carbon monoxide", "CO", "~25%"), ("Hydrogen", "H₂", "~15%"),
                       ("Nitrogen", "N₂", "~55%"), ("Carbon dioxide", "CO₂", "~5%")],
        "ratio": "Approximate: N₂(55%) + CO(25%) + H₂(15%) + CO₂(5%)",
        "description": "Low-calorific fuel gas produced by partial combustion of coal/coke with air; historical industrial use",
    },
    "water gas": {
        "full_name": "Water Gas",
        "type": "Fuel Gas Mixture",
        "components": [("Carbon monoxide", "CO", "~50%"), ("Hydrogen", "H₂", "~50%")],
        "ratio": "CO : H₂ ≈ 1 : 1",
        "description": "Produced by passing steam over hot coke (C + H₂O → CO + H₂); historical fuel and H₂ source",
    },
    # Industrial / chemical processes
    "brine": {
        "full_name": "Brine",
        "type": "Saline Solution",
        "components": [("Sodium chloride", "NaCl", "high concentration"), ("Water", "H₂O", "solvent")],
        "ratio": "Typically 3.5% NaCl (seawater) to 26% (saturated at 25°C)",
        "description": "Salt water solution; used in food preservation, chlor-alkali industry, and desalination",
    },
    "oleum": {
        "full_name": "Oleum (Fuming Sulfuric Acid)",
        "type": "Superacid / Industrial Sulfonating Agent",
        "components": [("Sulfuric acid", "H₂SO₄", "base"), ("Sulfur trioxide", "SO₃", "dissolved, 20–65%")],
        "ratio": "Free SO₃ content: 20–65% by mass (common grades: 20%, 30%, 65%)",
        "description": "More reactive than concentrated H₂SO₄; industrial sulfonation of aromatics and explosive manufacture",
    },
    "formalin": {
        "full_name": "Formalin",
        "type": "Fixative Solution",
        "components": [("Formaldehyde", "CH₂O / HCHO", "37–40%"), ("Water", "H₂O", "60–63%"), ("Methanol", "CH₃OH", "~10–15% stabilizer")],
        "ratio": "Formaldehyde: 37–40% w/v in water; methanol added to prevent polymerization",
        "description": "Biological tissue fixative and disinfectant; 10% formalin (= 4% formaldehyde) used in histology",
    },
}

def get_mixture(name: str) -> dict | None:
    """Return mixture info if name matches MIXTURE_DB, else None."""
    return MIXTURE_DB.get(name.strip().lower())

# ── Polymer repeating-unit data ───────────────────────────────────────────────
# Keys: lowercase compound name  Values: (display_notation, repeat_unit_SMILES, repeat_formula)
# SMILES use [*] for chain attachment points
POLYMER_UNITS: dict = {
    # polyolefins
    "polyethylene":              ("−(CH₂−CH₂)ₙ−",        "[*]CC[*]",                      "C₂H₄"),
    "polypropylene":             ("−(CH₂−CH(CH₃))ₙ−",    "[*]CC([*])C",                   "C₃H₆"),
    "polyisobutylene":           ("−(CH₂−C(CH₃)₂)ₙ−",   "[*]CC([*])(C)C",                "C₄H₈"),
    "polystyrene":               ("−(CH₂−CH(Ph))ₙ−",     "[*]CC([*])c1ccccc1",            "C₈H₈"),
    # halogenated
    "polyvinyl chloride":        ("−(CH₂−CHCl)ₙ−",       "[*]CC([*])Cl",                  "C₂H₃Cl"),
    "pvc":                       ("−(CH₂−CHCl)ₙ−",       "[*]CC([*])Cl",                  "C₂H₃Cl"),
    "polyvinylidene fluoride":   ("−(CH₂−CF₂)ₙ−",        "[*]CC([*])(F)F",                "C₂H₂F₂"),
    "pvdf":                      ("−(CH₂−CF₂)ₙ−",        "[*]CC([*])(F)F",                "C₂H₂F₂"),
    "polytetrafluoroethylene":   ("−(CF₂−CF₂)ₙ−",        "[*]C(F)(F)C([*])(F)F",          "C₂F₄"),
    "ptfe":                      ("−(CF₂−CF₂)ₙ−",        "[*]C(F)(F)C([*])(F)F",          "C₂F₄"),
    "teflon":                    ("−(CF₂−CF₂)ₙ−",        "[*]C(F)(F)C([*])(F)F",          "C₂F₄"),
    # acrylics / vinyl
    "polymethyl methacrylate":   ("−(C(CH₃)(COOCH₃))ₙ−","[*]CC([*])(C)C(=O)OC",          "C₅H₈O₂"),
    "pmma":                      ("−(C(CH₃)(COOCH₃))ₙ−","[*]CC([*])(C)C(=O)OC",          "C₅H₈O₂"),
    "acrylic":                   ("−(C(CH₃)(COOCH₃))ₙ−","[*]CC([*])(C)C(=O)OC",          "C₅H₈O₂"),
    "polyacrylic acid":          ("−(CH₂−CH(COOH))ₙ−",   "[*]CC([*])C(=O)O",              "C₃H₄O₂"),
    "polyacrylamide":            ("−(CH₂−CH(CONH₂))ₙ−",  "[*]CC([*])C(=O)N",              "C₃H₅NO"),
    "polyacrylonitrile":         ("−(CH₂−CH(CN))ₙ−",     "[*]CC([*])C#N",                 "C₃H₃N"),
    "polyvinyl alcohol":         ("−(CH₂−CHOH)ₙ−",       "[*]CC([*])O",                   "C₂H₄O"),
    "polyvinyl acetate":         ("−(CH₂−CH(OCOCH₃))ₙ−", "[*]CC([*])OC(C)=O",             "C₄H₆O₂"),
    # polyesters / polyethers
    "polyethylene terephthalate":("−(OCH₂CH₂O−CO−Ph−CO)ₙ−","[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]","C₁₀H₈O₄"),
    "pet":                       ("−(OCH₂CH₂O−CO−Ph−CO)ₙ−","[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]","C₁₀H₈O₄"),
    "polylactic acid":           ("−(O−CH(CH₃)−CO)ₙ−",   "[*]OC(C)C(=O)[*]",              "C₃H₄O₂"),
    "pla":                       ("−(O−CH(CH₃)−CO)ₙ−",   "[*]OC(C)C(=O)[*]",              "C₃H₄O₂"),
    "polycaprolactone":          ("−(O−(CH₂)₅−CO)ₙ−",    "[*]OCCCCC C(=O)[*]",            "C₆H₁₀O₂"),
    "pcl":                       ("−(O−(CH₂)₅−CO)ₙ−",    "[*]OCCCCC C(=O)[*]",            "C₆H₁₀O₂"),
    "polyethylene glycol":       ("−(CH₂−CH₂−O)ₙ−",      "[*]CCO[*]",                     "C₂H₄O"),
    "peg":                       ("−(CH₂−CH₂−O)ₙ−",      "[*]CCO[*]",                     "C₂H₄O"),
    "polyoxymethylene":          ("−(CH₂−O)ₙ−",           "[*]CO[*]",                      "CH₂O"),
    "pom":                       ("−(CH₂−O)ₙ−",           "[*]CO[*]",                      "CH₂O"),
    # polyamides
    "nylon":                     ("−(NH−(CH₂)₆−CO)ₙ−",   "[*]NCCCCCC(=O)[*]",             "C₆H₁₁NO"),
    "nylon-6":                   ("−(NH−(CH₂)₅−CO)ₙ−",   "[*]NCCCCC(=O)[*]",              "C₆H₁₁NO"),
    "nylon-6,6":                 ("−(NH(CH₂)₆NH−CO(CH₂)₄CO)ₙ−","[*]NCCCCCCNC(=O)CCCCC(=O)[*]","C₁₂H₂₂N₂O₂"),
    "nylon-6-6":                 ("−(NH(CH₂)₆NH−CO(CH₂)₄CO)ₙ−","[*]NCCCCCCNC(=O)CCCCC(=O)[*]","C₁₂H₂₂N₂O₂"),
    # polyamides (more)
    "polyamide":                 ("−(NH−(CH₂)₅−CO)ₙ−",    "[*]NCCCCC(=O)[*]",              "C₆H₁₁NO"),
    "nomex":                     ("−(NH−Ph−NH−CO−Ph−CO)ₙ−","[*]Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)[*]", "C₁₄H₁₀N₂O₂"),
    "kevlar":                    ("−(NH−Ph−NH−CO−Ph−CO)ₙ−","[*]Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)[*]", "C₁₄H₁₀N₂O₂"),
    "poly-para-phenylene terephthalamide": ("−(NH−Ph−NH−CO−Ph−CO)ₙ−","[*]Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)[*]","C₁₄H₁₀N₂O₂"),
    # polycarbonates
    "polycarbonate":             ("−(O−BPA−O−CO)ₙ−",       "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)O[*]", "C₁₆H₁₄O₃"),
    "pc":                        ("−(O−BPA−O−CO)ₙ−",       "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)O[*]", "C₁₆H₁₄O₃"),
    # silicones
    "silicone":                  ("−(Si(CH₃)₂−O)ₙ−",      "[*][Si](C)(C)O[*]",             "C₂H₆OSi"),
    "pdms":                      ("−(Si(CH₃)₂−O)ₙ−",      "[*][Si](C)(C)O[*]",             "C₂H₆OSi"),
    "polydimethylsiloxane":      ("−(Si(CH₃)₂−O)ₙ−",      "[*][Si](C)(C)O[*]",             "C₂H₆OSi"),
    # thermosets / phenolic resins
    "bakelite":                  ("−(C₆H₃(OH)−CH₂)ₙ−",   "[*]Cc1cc(O)cc(C[*])c1",         "C₇H₆O"),
    "phenol-formaldehyde":       ("−(C₆H₃(OH)−CH₂)ₙ−",   "[*]Cc1cc(O)cc(C[*])c1",         "C₇H₆O"),
    "phenolic resin":            ("−(C₆H₃(OH)−CH₂)ₙ−",   "[*]Cc1cc(O)cc(C[*])c1",         "C₇H₆O"),
    # polyurethanes
    "polyurethane":              ("−(NH−CO−O−(CH₂)₄−O)ₙ−","[*]NC(=O)OCCCCO[*]",            "C₅H₉NO₃"),
    "pu":                        ("−(NH−CO−O−(CH₂)₄−O)ₙ−","[*]NC(=O)OCCCCO[*]",            "C₅H₉NO₃"),
    # polyimides
    "polyimide":                 ("−(N−CO−Ph−CO)ₙ−",       "[*]N1C(=O)c2ccccc2C1=O",        "C₁₀H₅NO₂"),
    "pi":                        ("−(N−CO−Ph−CO)ₙ−",       "[*]N1C(=O)c2ccccc2C1=O",        "C₁₀H₅NO₂"),
    "kapton":                    ("−(N−CO−Ph−CO)ₙ−",       "[*]N1C(=O)c2ccccc2C1=O",        "C₁₀H₅NO₂"),
    # other engineering polymers
    "abs":                       ("−(CH₂−CH(Ph)·CH₂−CH(CN)·CH₂−CH=CH₂)ₙ−","[*]CC([*])c1ccccc1", "C₁₅H₁₇N"),
    "polyphenylene oxide":       ("−(C₆H₂(CH₃)₂−O)ₙ−",   "[*]Oc1cc(C)cc(C)c1[*]",         "C₈H₈O"),
    "ppo":                       ("−(C₆H₂(CH₃)₂−O)ₙ−",   "[*]Oc1cc(C)cc(C)c1[*]",         "C₈H₈O"),
    "polysulfone":               ("−(Ph−SO₂−Ph−O)ₙ−",     "[*]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)O[*]","C₁₂H₈O₃S"),
    "polyetherimide":            ("−(N−CO−Ph−CO−O−Ph)ₙ−",  "[*]N1C(=O)c2ccccc2C1=O",        "C₁₄H₇NO₃"),
    "pei":                       ("−(N−CO−Ph−CO−O−Ph)ₙ−",  "[*]N1C(=O)c2ccccc2C1=O",        "C₁₄H₇NO₃"),
    "peek":                      ("−(Ph−O−Ph−O−Ph−CO)ₙ−",  "[*]Oc1ccc(cc1)Oc1ccc(cc1)C(=O)[*]","C₁₉H₁₂O₃"),
    "polyether ether ketone":    ("−(Ph−O−Ph−O−Ph−CO)ₙ−",  "[*]Oc1ccc(cc1)Oc1ccc(cc1)C(=O)[*]","C₁₉H₁₂O₃"),
    "polyphenylene sulfide":     ("−(C₆H₄−S)ₙ−",          "[*]c1ccc(cc1)S[*]",             "C₆H₄S"),
    "pps":                       ("−(C₆H₄−S)ₙ−",          "[*]c1ccc(cc1)S[*]",             "C₆H₄S"),
    # elastomers
    "neoprene":                  ("−(CH₂−CCl=CH−CH₂)ₙ−",  "[*]C/C=C(\\Cl)C[*]",            "C₄H₅Cl"),
    "polychloroprene":           ("−(CH₂−CCl=CH−CH₂)ₙ−",  "[*]C/C=C(\\Cl)C[*]",            "C₄H₅Cl"),
    "styrene-butadiene rubber":  ("−(CH₂−CH=CH−CH₂·CH₂−CH(Ph))ₙ−","[*]C=CC[*]",           "C₁₂H₁₄"),
    "sbr":                       ("−(CH₂−CH=CH−CH₂·CH₂−CH(Ph))ₙ−","[*]C=CC[*]",           "C₁₂H₁₄"),
    "polybutadiene":             ("−(CH₂−CH=CH−CH₂)ₙ−",   "[*]CC=CC[*]",                   "C₄H₆"),
    "nitrile rubber":            ("−(CH₂−CH=CH−CH₂·CH₂−CH(CN))ₙ−","[*]CC([*])C#N",        "C₇H₉N"),
    "nbr":                       ("−(CH₂−CH=CH−CH₂·CH₂−CH(CN))ₙ−","[*]CC([*])C#N",        "C₇H₉N"),
    # natural / biological polymers
    "natural rubber":            ("−(CH₂−C(CH₃)=CH−CH₂)ₙ−","[*]C/C=C(\\C)C[*]",          "C₅H₈"),
    "rubber":                    ("−(CH₂−C(CH₃)=CH−CH₂)ₙ−","[*]C/C=C(\\C)C[*]",          "C₅H₈"),
    "polyisoprene":              ("−(CH₂−C(CH₃)=CH−CH₂)ₙ−","[*]C/C=C(\\C)C[*]",          "C₅H₈"),
    "cellulose":                 ("−(C₆H₁₀O₅)ₙ−",         "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C₆H₁₀O₅"),
    "starch":                    ("−(C₆H₁₀O₅)ₙ−",         "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C₆H₁₀O₅"),
    "chitin":                    ("−(C₈H₁₃NO₅)ₙ−",        "CC(=O)N[C@@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O", "C₈H₁₃NO₅"),
}

def get_polymer_unit(name: str) -> dict | None:
    """Return polymer unit info if the compound name matches a known polymer, else None."""
    key = name.strip().lower()
    info = POLYMER_UNITS.get(key)
    # Fuzzy prefix match — only when query is at least 4 chars to avoid
    # short element symbols (Na, Fe, Cu...) hitting polymer names by accident
    if not info and len(key) >= 4:
        for k, v in POLYMER_UNITS.items():
            if key == k or key.startswith(k) or k.startswith(key):
                info = v
                break
    if not info:
        return None
    notation, smiles, formula = info
    img_b64 = ""
    if RDKIT_OK and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                from rdkit.Chem import Draw as _Draw
                img_pil = _Draw.MolToImage(mol, size=(400, 300))
                buf = BytesIO()
                img_pil.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass
    return {"notation": notation, "formula": formula, "img_b64": img_b64}


def mol_to_b64_png(smiles: str, size=(400, 300)) -> str:
    if not RDKIT_OK or not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def ollama_sse(system: str, user_content: str):
    """Synchronous generator — yields SSE-formatted lines from Ollama streaming."""
    try:
        for chunk in ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            stream=True,
        ):
            text = chunk["message"]["content"]
            yield f"data: {json.dumps({'text': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Chemistry Compound Analyzer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Autocomplete (PubChem live) ────────────────────────────────────────────────
@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(...)):
    if len(q) < 2:
        return {"suggestions": []}
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete"
                   f"/compound/{q}/JSON?limit=12")
            resp = await client.get(url)
            data = resp.json()
            terms = data.get("dictionary_terms", {}).get("compound", [])
            return {"suggestions": terms}
    except Exception:
        return {"suggestions": []}


# ── Gather library data ────────────────────────────────────────────────────────
class GatherRequest(BaseModel):
    query: str

async def wikipedia_image_b64(query: str) -> str:
    """Fetch the main Wikipedia thumbnail for a search term, return as base64 PNG/JPEG."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
                headers={"User-Agent": "ChemistryAnalyzer/1.0"},
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            thumb_url = (data.get("thumbnail") or {}).get("source", "")
            if not thumb_url:
                return ""
            img_r = await client.get(thumb_url)
            if img_r.status_code != 200:
                return ""
            return base64.b64encode(img_r.content).decode()
    except Exception:
        return ""

@app.post("/api/gather")
async def gather_data(body: GatherRequest):
    query = body.query.strip()

    # Auto-correct capitalization only when lowercase letters present
    if looks_like_formula(query) and any(c.islower() for c in query if c.isalpha()):
        query = normalize_formula(query)

    by_formula = looks_like_formula(query)

    # Detect structural notation: same element symbol appears more than once
    # e.g. C2H5OH has H twice → structural; C2H6O has each element once → Hill
    is_structural = False
    search_query  = query          # used for PubChem / element lookups
    if by_formula:
        syms = re.findall(r'[A-Z][a-z]?', query)
        if len(syms) != len(set(syms)):
            is_structural = True
            hill = to_hill_formula(query)
            if hill and hill != query:
                search_query = hill  # normalised Hill for library searches

    mm       = gather_molmass(query)        if by_formula else {}   # molmass handles structural notation fine
    elements = gather_elements(search_query) if by_formula else []
    # Structural notation (CH3CHO, C2H5OH): search PubChem by name so we get
    # the exact compound, not PubChem's first-match for the Hill formula.
    pc       = gather_pubchem(query, by_formula=False) if is_structural else \
               gather_pubchem(search_query, by_formula=by_formula)
    smiles   = pc.get("smiles", "")

    # For name searches, derive molmass + elements from PubChem's returned formula
    if not by_formula and pc.get("formula"):
        if not mm:
            mm = gather_molmass(pc["formula"])
        if not elements:
            elements = gather_elements(pc["formula"])
    rd       = gather_rdkit(smiles)
    organic  = has_carbon(query) if by_formula else True
    # Skip image only for generic organic formula searches that show the isomer grid.
    # Structural notation (C2H5OH) and name searches go direct, so generate the image.
    skip_img = organic and by_formula and not is_structural
    img_b64  = mol_to_b64_png(smiles) if not skip_img else ""

    # Fallback: no molecular image → try Wikipedia thumbnail (crystals, allotropes, etc.)
    if not img_b64 and not by_formula and not smiles:
        img_b64 = await wikipedia_image_b64(query)

    polymer = get_polymer_unit(query) if not by_formula else None
    mixture = get_mixture(query)      if not by_formula else None

    return {
        "query":        query,
        "by_formula":   by_formula,
        "is_structural": is_structural,
        "organic":      organic,
        "smiles":     smiles,
        "img_b64":    img_b64,
        "molmass":    mm,
        "elements":   elements,
        "pubchem":    pc,
        "rdkit":      rd,
        "polymer":    polymer,
        "mixture":    mixture,
    }


# ── Stream: full analysis ──────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    compound: str
    verified_text: str = ""
    is_crystal: bool = False
    is_polymer: bool = False
    is_mixture: bool = False

@app.post("/api/stream/analyze")
def stream_analyze(body: AnalyzeRequest):
    if body.is_crystal:
        system       = CRYSTAL_SYSTEM
        user_content = f"Analyze this crystal / solid-state material: {body.compound}"
        if body.verified_text:
            user_content += f"\n\nVerified crystallographic data:\n{body.verified_text}"
    elif body.is_polymer:
        system       = POLYMER_SYSTEM
        user_content = f"Analyze this polymer: {body.compound}"
    elif body.is_mixture:
        system       = MIXTURE_SYSTEM
        user_content = f"Analyze this chemical mixture or reagent: {body.compound}"
    else:
        system       = ANALYSIS_SYSTEM
        user_content = (
            f"Verified data from chemistry libraries:\n{body.verified_text}\n\n"
            f"Compound to analyze: {body.compound}\n"
            f"IMPORTANT: Analyze EXACTLY '{body.compound}' — do not analyze any other formula or isomer."
        )
    return StreamingResponse(
        ollama_sse(system, user_content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Stream: isomers ────────────────────────────────────────────────────────────
class IsomerRequest(BaseModel):
    formula: str

@app.post("/api/stream/isomers")
def stream_isomers(body: IsomerRequest):
    return StreamingResponse(
        ollama_sse(ISOMER_SYSTEM, f"Molecular formula: {body.formula}"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Full data for one selected isomer (by CID + SMILES) ───────────────────────
class GatherIsomerRequest(BaseModel):
    cid: int
    smiles: str

@app.post("/api/gather-isomer")
async def gather_isomer(body: GatherIsomerRequest):
    """Return PubChem + RDKit + full-res image for the compound the user picked."""
    pc: dict = {}
    if PUBCHEM_OK and body.cid:
        try:
            compounds = pcp.get_compounds(body.cid, "cid")
            if compounds:
                c = compounds[0]
                pc = {"found": True, "cid": c.cid}
                for attr, key in [("iupac_name", "iupac_name"), ("molecular_formula", "formula")]:
                    val = getattr(c, attr, None)
                    if val:
                        pc[key] = str(val)
                if getattr(c, "synonyms", None):
                    pc["common_name"] = c.synonyms[0]
                for smiles_attr in ("isomeric_smiles", "canonical_smiles"):
                    val = getattr(c, smiles_attr, None)
                    if val:
                        pc["smiles"] = val
                        break
                if getattr(c, "molecular_weight", None):
                    pc["mw"] = f"{c.molecular_weight} g/mol"
                for attr, key in [
                    ("xlogp", "logp"), ("tpsa", "tpsa"),
                    ("h_bond_donor_count", "hbd"), ("h_bond_acceptor_count", "hba"),
                ]:
                    val = getattr(c, attr, None)
                    if val is not None:
                        pc[key] = val
        except Exception as e:
            pc = {"error": str(e)}

    rd      = gather_rdkit(body.smiles)
    img_b64 = mol_to_b64_png(body.smiles)          # full 400×300
    return {"pubchem": pc, "rdkit": rd, "img_b64": img_b64}


# ── Isomer visual grid (PubChem + RDKit images) ───────────────────────────────
class IsomerGridRequest(BaseModel):
    formula: str

@app.post("/api/isomers/grid")
async def isomers_grid(body: IsomerGridRequest):
    formula = body.formula.strip()
    if not PUBCHEM_OK:
        return {"error": "PubChem library not installed", "isomers": []}
    try:
        compounds = pcp.get_compounds(formula, "formula", listkey_count=24)
    except Exception as e:
        return {"error": str(e), "isomers": []}
    if not compounds:
        return {"error": f"No compounds found for formula {formula}", "isomers": []}
    result = []
    for c in compounds[:24]:
        smiles = None
        for attr in ("isomeric_smiles", "canonical_smiles"):
            val = getattr(c, attr, None)
            if val:
                smiles = val
                break
        name = getattr(c, "iupac_name", None) or ""
        if not name and getattr(c, "synonyms", None):
            name = c.synonyms[0]
        if not name:
            name = f"CID {c.cid}"
        img_b64 = mol_to_b64_png(smiles, size=(200, 150)) if smiles else ""
        result.append({"cid": c.cid, "name": name, "smiles": smiles or "", "img_b64": img_b64})
    return {"isomers": result, "total": len(compounds), "all_shown": len(compounds) < 24}


# ── Name / SMILES lookup (for manual isomer input) ───────────────────────────
CRYSTAL_IMG_SKIP = ["logo", "icon", "commons", "wikiquote", "wikibooks",
                    "wiktionary", "protection", "oojs", "symbol", "cscr",
                    "wikproject", "wikiproject",
                    # 3D molecular ball-and-stick renders
                    "3d-", "-3d", "_3d", "3d_", "-balls", "_balls",
                    "ball-and-stick", "ball_and_stick"]
# Words that suggest gem photos or 3D molecular renders rather than crystal structure diagrams
CRYSTAL_IMG_PHOTO_WORDS = ["rose", "tiger", "red", "blue", "pink", "green",
                            "purple", "eye", "amethyst", "citrine", "carnelian",
                            "jasper", "agate", "onyx", "cabochon", "jewel",
                            "jewelry", "gem", "rough", "specimen", "mine",
                            "quarry", "outcrop", "inclusion", "macro", "jug",
                            "ewer", "necklace", "ring", "museum", "natural",
                            # 3D molecular renders (ball-and-stick, space-fill, etc.)
                            "3d", "ball", "stick", "vdw", "spacefill", "space-fill",
                            "space_fill", "balls", "sticks", "model", "render",
                            # synthetic / lab / industrial photos
                            "synthetic", "hpht", "cvd", "lab", "industrial",
                            "polished", "faceted", "cut", "jewel", "jewelry"]
import re as _re
_SPACE_GROUP = _re.compile(r'(?<![a-zA-Z])(?:[a-z]\d{1,3}|fd3|im3|fm3|r3|p3|p4|p6|c2|i4|p-1)', _re.I)

CRYSTAL_SUMMARY_KEYWORDS = [
    "crystal", "crystalline", "lattice", "unit cell", "space group",
    "cubic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic",
    "mineral", "allotrope", "ionic compound", "ionic solid",
    "covalent network", "metallic", "face-centered", "body-centered",
    # Elements — Wikipedia describes them as metals/elements, not "crystals"
    "chemical element", "transition metal", "alkali metal", "alkaline earth",
    "noble metal", "post-transition metal", "metalloid", "semimetal",
    "native metal", "refractory metal", "precious metal",
    "atomic number", "periodic table", "solid at room temperature",
    "lustrous", "ductile", "malleable", "conductor",
]

import asyncio as _asyncio
import tempfile as _tempfile
import io as _io_mod

def _render_cif_sync(cif_text: str) -> str:
    """Blocking: parse CIF with ASE and render to base64 PNG."""
    if not ASE_OK:
        return ""
    tmp = None
    try:
        with _tempfile.NamedTemporaryFile(mode='w', suffix='.cif',
                                          delete=False, encoding='utf-8') as f:
            f.write(cif_text)
            tmp = f.name
        atoms = ase_read(tmp, format='cif')

        # If too many atoms (supercell), try to get a smaller conventional cell
        if len(atoms) > 80:
            try:
                from ase.build import make_supercell
                import numpy as np
                # Try to find a primitive-like cell by reducing repetition
                from pymatgen.core import Structure as _PMGStr
                from pymatgen.io.ase import AseAtomsAdaptor
                pmg_struct = AseAtomsAdaptor.get_structure(atoms)
                pmg_prim = pmg_struct.get_primitive_structure()
                atoms = AseAtomsAdaptor.get_atoms(pmg_prim)
            except Exception:
                pass  # keep original if reduction fails

        # Pick rotation based on crystal system from cell angles
        cell = atoms.get_cell()
        cell_params = atoms.cell.cellpar()  # [a,b,c,α,β,γ]
        alpha, beta, gamma = cell_params[3], cell_params[4], cell_params[5]
        if abs(gamma - 120) < 5:            # hexagonal / trigonal
            rotation = '-70x,20y,0z'
        elif abs(alpha - 90) < 2 and abs(beta - 90) < 2 and abs(gamma - 90) < 2:
            rotation = '15x,-20y,5z'        # cubic / orthorhombic → isometric
        else:
            rotation = '10x,-15y,5z'        # general

        # Render three views (front, iso, top) side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='#f8f9fa')
        rotations = [rotation, '0x,0y,0z', '90x,0y,0z']
        labels    = ['Perspective', 'Front', 'Top']
        for ax, rot, lbl in zip(axes, rotations, labels):
            ax.set_facecolor('#f8f9fa')
            ase_plot_atoms(atoms, ax, radii=0.55, rotation=rot, show_unit_cell=2)
            for line in ax.get_lines():
                line.set_color('#2d3748')
                line.set_linewidth(1.5)
            ax.set_axis_off()
            ax.set_title(lbl, fontsize=9, color='#4a5568', pad=4)

        # Element legend at the bottom
        from ase.data import chemical_symbols, covalent_radii
        from ase.data.colors import jmol_colors
        symbols = sorted(set(atoms.get_chemical_symbols()))
        legend_x = 0.02
        for sym in symbols:
            z = atoms[atoms.get_chemical_symbols().index(sym) if sym in atoms.get_chemical_symbols() else 0].number
            color = tuple(jmol_colors[z])
            fig.text(legend_x, 0.02, f'● {sym}', color=color,
                     fontsize=9, fontweight='bold', transform=fig.transFigure)
            legend_x += 0.06

        # Force a fixed output size — ASE can distort figure dimensions
        buf = _io_mod.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='#f8f9fa')
        plt.close(fig)
        # Always resize to max 1200px wide with PIL for web-friendly size
        try:
            from PIL import Image as _PILImg
            buf.seek(0)
            pil_img = _PILImg.open(buf)
            max_w = 1200
            w, h = pil_img.size
            if w > max_w:
                pil_img = pil_img.resize(
                    (max_w, int(h * max_w / w)), _PILImg.LANCZOS)
            out = _io_mod.BytesIO()
            pil_img.save(out, format='JPEG', quality=82, optimize=True)
            return base64.b64encode(out.getvalue()).decode()
        except Exception:
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass

def _render_cif_text_sync(cif_text: str, title: str = "") -> str:
    """Render a CIF string using pymatgen + matplotlib. Returns base64 PNG or ''."""
    if not ASE_OK:
        return ""
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser
        import io as _sio
        parser = CifParser.from_str(cif_text)
        structure = parser.parse_structures(primitive=False)[0]

        # Write to temporary CIF for ASE
        tmp_cif = _tempfile.NamedTemporaryFile(mode='w', suffix='.cif',
                                               delete=False, encoding='utf-8')
        structure.to(fmt="cif", filename=tmp_cif.name)
        tmp_cif.close()
        result = _render_cif_sync(open(tmp_cif.name).read())
        os.unlink(tmp_cif.name)
        return result
    except Exception:
        return _render_cif_sync(cif_text)   # fall back to direct ASE parse


# Known polymorph spacegroup numbers: ensures correct structure for ambiguous formulas
# Keys are lowercase mineral/compound names OR chemical formulas
_MP_SPACEGROUP_HINTS = {
    # Named minerals
    "diamond": 227, "graphite": 194, "lonsdaleite": 194,
    "quartz": 152,  "cristobalite": 227, "tridymite": 20,
    "calcite": 167, "aragonite": 62,
    "ice": 194,     "ice ih": 194,
    "pyrite": 205,  "marcasite": 58,
    "sphalerite": 216, "wurtzite": 186,
    "galena": 225,  "cinnabar": 152,
    "corundum": 167, "rutile": 136, "anatase": 141,
    "halite": 225,  "rocksalt": 225, "salt": 225,
    "fluorite": 225, "perovskite": 221,
    "zincite": 186, "periclase": 225,
    "covellite": 194,
    # Common metals (elemental) — prevents MP returning metastable phases
    "iron":       229,  "fe":   229,  # alpha-Fe BCC (Im-3m)
    "copper":     225,  "cu":   225,  # FCC (Fm-3m)
    "aluminum":   225,  "al":   225,  # FCC
    "aluminium":  225,
    "nickel":     225,  "ni":   225,  # FCC
    "gold":       225,  "au":   225,  # FCC
    "silver":     225,  "ag":   225,  # FCC
    "platinum":   225,  "pt":   225,  # FCC
    "palladium":  225,  "pd":   225,  # FCC
    "lead":       225,  "pb":   225,  # FCC
    "titanium":   194,  "ti":   194,  # HCP (P63/mmc)
    "magnesium":  194,  "mg":   194,  # HCP
    "zinc":       194,  "zn":   194,  # HCP
    "cobalt":     194,  "co":   194,  # HCP at RT
    "cadmium":    194,  "cd":   194,  # HCP
    "tungsten":   229,  "w":    229,  # BCC
    "molybdenum": 229,  "mo":   229,  # BCC (pure Mo)
    "chromium":   229,  "cr":   229,  # BCC
    "vanadium":   229,  "v":    229,  # BCC
    "sodium":     229,  "na":   229,  # BCC
    "potassium":  229,  "k":    229,  # BCC
    "silicon":    227,  "si":   227,  # diamond cubic (Fd-3m)
    "germanium":  227,  "ge":   227,  # diamond cubic
    "diamond":    227,  "carbon": 227, "c": 227,  # diamond cubic carbon
    "tin":        141,  "sn":   141,  # beta-Sn (I41/amd)
    "beryllium":  194,  "be":   194,  # HCP
    "osmium":     194,  "os":   194,  # HCP
    "ruthenium":  194,  "ru":   194,  # HCP
    "iridium":    225,  "ir":   225,  # FCC
    "rhodium":    225,  "rh":   225,  # FCC
    # Common chemical formulas (users often search by formula)
    "zns":   216,  # sphalerite (cubic ZnS)
    "cus":   194,  # covellite
    "fes2":  205,  # pyrite
    "fes":   194,  # troilite/pyrrhotite
    "sio2":  152,  # alpha-quartz
    "tio2":  136,  # rutile
    "al2o3": 167,  # corundum
    "fe2o3": 167,  # hematite
    "fe3o4": 227,  # magnetite
    "nacl":  225,  # halite
    "naf":   225,  # villaumite
    "kcl":   225,  # sylvite
    "mgo":   225,  # periclase
    "cao":   225,  # lime
    "bao":   225,  # barite-like (rocksalt)
    "zno":   186,  # zincite (wurtzite)
    "cdo":   225,  # monteponite
    "mns":   225,  # alabandite
    "pbs":   225,  # galena
    "cds":   216,  # hawleyite
    "cu2s":  167,  # chalcocite (high-temp)
    "cuo":    15,  # tenorite
    "cu2o":  224,  # cuprite
    "mos2":  194,  # molybdenite
    "ws2":   194,  # tungstenite
    "sno2":  136,  # cassiterite
    "wo3":    14,  # tungstite
    "caco3": 167,  # calcite
    "mgco3": 167,  # magnesite
    "feco3": 167,  # siderite
    "baso4":  62,  # barite
    "srso4":  62,  # celestite
    "caso4":  63,  # anhydrite
    "bi2s3":  62,  # bismuthinite
    "sb2s3":  62,  # stibnite
    "as2s3":  15,  # orpiment
    "cufes2": 122, # chalcopyrite
}

# Mineral varieties → (parent formula, spacegroup hint or None)
# These are color/impurity varieties of parent compounds, not findable by name in PubChem/MP
_MINERAL_ALIASES = {
    # Quartz varieties (SiO2, trigonal spacegroup 152)
    "amethyst":     ("SiO2", 152),
    "citrine":      ("SiO2", 152),
    "rose quartz":  ("SiO2", 152),
    "smoky quartz": ("SiO2", 152),
    "milky quartz": ("SiO2", 152),
    "chalcedony":   ("SiO2", 152),
    "agate":        ("SiO2", 152),
    "onyx":         ("SiO2", 152),
    "jasper":       ("SiO2", 152),
    "flint":        ("SiO2", 152),
    "chert":        ("SiO2", 152),
    "opal":         ("SiO2", 152),
    "obsidian":     ("SiO2", 152),
    # Corundum varieties (Al2O3, trigonal spacegroup 167)
    "ruby":         ("Al2O3", 167),
    "sapphire":     ("Al2O3", 167),
    "padparadscha": ("Al2O3", 167),
    # Beryl varieties (Be3Al2Si6O18, hexagonal)
    "emerald":      ("Be3Al2Si6O18", None),
    "aquamarine":   ("Be3Al2Si6O18", None),
    "morganite":    ("Be3Al2Si6O18", None),
    "heliodor":     ("Be3Al2Si6O18", None),
    # Garnet varieties
    "almandine":    ("Fe3Al2Si3O12", 230),
    "pyrope":       ("Mg3Al2Si3O12", 230),
    "spessartine":  ("Mn3Al2Si3O12", 230),
    # Feldspar varieties
    "moonstone":    ("KAlSi3O8", None),
    "amazonite":    ("KAlSi3O8", None),
    "labradorite":  ("CaAl2Si2O8", None),
    # Lazurite / Lapis lazuli (sodalite framework — S-free form is in MP as spacegroup 218)
    "lapis lazuli": ("Na8Al6Si6O24", 218),
    "lazurite":     ("Na8Al6Si6O24", 218),
    "lapis":        ("Na8Al6Si6O24", 218),
    # Mica group (phyllosilicates) — "mica" → muscovite as most common variety
    "mica":         ("KAl3Si3O12H2", 13),   # muscovite (monoclinic P2/c)
    "muscovite":    ("KAl3Si3O12H2", 13),
    "biotite":      ("KMg3AlSi3O12H2", 1),  # phlogopite approximation
    "phlogopite":   ("KMg3AlSi3O12H2", 1),
    "lepidolite":   ("KAl3Si3O12H2", 13),   # Li-mica, same framework as muscovite
    # Silicate minerals
    "talc":         ("Mg3H2Si4O12", 2),     # pyrophyllite-group
    "serpentine":   ("Mg3Si2O5H4", None),
    "albite":       ("NaAlSi3O8", 2),
    "sanidine":     ("KAlSi3O8", 2),
    "orthoclase":   ("KAlSi3O8", 2),
    "anorthite":    ("CaAl2Si2O8", None),
    "olivine":      ("Mg2SiO4", 62),        # forsterite end-member
    "forsterite":   ("Mg2SiO4", 62),
    "fayalite":     ("Fe2SiO4", 62),
    "enstatite":    ("MgSiO3", 61),
    "diopside":     ("CaMgSi2O6", 15),
    "augite":       ("CaMgSi2O6", 15),
    "hornblende":   ("Ca2Mg5Si8O22H2", None),
    "actinolite":   ("Ca2Mg5Si8O22H2", None),
    "wollastonite": ("CaSiO3", 14),
    "kyanite":      ("Al2SiO5", 2),
    "sillimanite":  ("Al2SiO5", 58),
    "andalusite":   ("Al2SiO5", 58),
    "topaz":        ("Al2SiO4F2", 58),
    "zircon":       ("ZrSiO4", 141),
    "titanite":     ("CaTiSiO5", 14),
    # Carbonate / sulfate minerals
    "malachite":    ("Cu2CO3H2O4", None),
    "azurite":      ("Cu3C2H2O8", None),
    "rhodonite":    ("CaMn4Si5O15", None),
    "dolomite":     ("CaMgC2O6", 148),
    "magnesite":    ("MgCO3", 167),
    "siderite":     ("FeCO3", 167),
    "barite":       ("BaSO4", 62),
    "celestite":    ("SrSO4", 62),
    "gypsum":       ("CaSO4H4O2", 15),
    "anhydrite":    ("CaSO4", 63),
    # Oxide / hydroxide minerals
    "hematite":     ("Fe2O3", 167),
    "magnetite":    ("Fe3O4", 227),
    "chromite":     ("FeCr2O4", 227),
    "spinel":       ("MgAl2O4", 227),
    "cassiterite":  ("SnO2", 136),
    "ilmenite":     ("FeTiO3", 148),
    "wolframite":   ("FeWO4", 13),
    "scheelite":    ("CaWO4", 88),
    # Phosphate minerals
    "apatite":      ("Ca5P3O12F", 176),     # fluorapatite
    "fluorapatite": ("Ca5P3O12F", 176),
    "monazite":     ("LaPO4", 14),
    "xenotime":     ("YPO4", 141),
    # Sulfide minerals
    "chalcopyrite": ("CuFeS2", 122),
    "arsenopyrite": ("FeAsS", 14),
    "stibnite":     ("Sb2S3", 62),
    "molybdenite":  ("MoS2", 194),
    "bornite":      ("Cu5FeS4", 216),
    "chalcocite":   ("Cu2S", 167),
}

def _crystal_system(sg_number: int) -> str:
    if sg_number <= 2:   return "Triclinic"
    if sg_number <= 15:  return "Monoclinic"
    if sg_number <= 74:  return "Orthorhombic"
    if sg_number <= 142: return "Tetragonal"
    if sg_number <= 167: return "Trigonal"
    if sg_number <= 194: return "Hexagonal"
    return "Cubic"

def _bravais_label(sg_symbol: str, sg_number: int) -> str:
    """Return common Bravais lattice label (BCC, FCC, HCP, etc.)."""
    if not sg_symbol or not sg_number:
        return ""
    letter = sg_symbol[0].upper()
    system = _crystal_system(sg_number)
    if system == "Cubic":
        # Special structures with FCC Bravais but distinct topology
        if sg_number == 227: return "Diamond cubic"        # Fd-3m  (C, Si, Ge)
        if sg_number == 216: return "Zinc blende (FCC)"    # F-43m  (ZnS, GaAs)
        return {"F": "Face-centered cubic (FCC)",
                "I": "Body-centered cubic (BCC)",
                "P": "Simple cubic (SC)"}.get(letter, f"Cubic ({letter})")
    if system == "Hexagonal":
        if sg_number == 194: return "Hexagonal close-packed (HCP)"
        return "Hexagonal"
    if system == "Trigonal":
        return "Rhombohedral" if letter == "R" else "Trigonal"
    if system == "Tetragonal":
        return "Body-centered tetragonal (BCT)" if letter == "I" else "Simple tetragonal"
    if system == "Orthorhombic":
        return {"F": "Face-centered orthorhombic",
                "I": "Body-centered orthorhombic",
                "C": "Base-centered orthorhombic",
                "A": "Base-centered orthorhombic"}.get(letter, "Simple orthorhombic")
    if system == "Monoclinic":
        return "Base-centered monoclinic" if letter in ("C","A","B") else "Simple monoclinic"
    return system  # Triclinic


_LATTICE_DIAGRAM_CACHE: dict = {}

# Compounds that use the rock salt (NaCl-type) structure — sg 225 but two species
_ROCK_SALT_COMPOUNDS = {
    "nacl", "sodium chloride", "salt", "halite",
    "mgo", "magnesium oxide", "periclase",
    "lif", "lithium fluoride",
    "kcl", "potassium chloride", "sylvite",
    "kbr", "potassium bromide",
    "naf", "sodium fluoride",
    "cao", "calcium oxide", "lime",
    "feo", "iron(ii) oxide", "wustite",
    "nio", "nickel oxide",
    "tio", "titanium oxide",
    "mns", "manganese sulfide",
    "pbte", "lead telluride",
    "pbse", "lead selenide",
    "pbs", "lead sulfide", "galena",
}

def _gen_lattice_diagram_sync(bravais_key: str, hint: str = "") -> str:
    """Generate a ball-and-stick unit cell diagram (base64 PNG). Cached."""
    # Override bravais for rock salt compounds
    if hint in _ROCK_SALT_COMPOUNDS or bravais_key.lower() == "rock salt (fcc)":
        bravais_key = "Rock salt (FCC)"
    cache_key = bravais_key.lower() + (f"|{hint}" if hint in _ROCK_SALT_COMPOUNDS else "")
    if cache_key in _LATTICE_DIAGRAM_CACHE:
        return _LATTICE_DIAGRAM_CACHE[cache_key]
    result = ""
    try:
        import math
        from PIL import Image as _PIL, ImageDraw as _IDraw
        from io import BytesIO

        bk = cache_key
        is_hcp = 'hcp' in bk

        W, H = 420, 420

        # ── Projection helpers ───────────────────────────────────────────────
        def _make_proj(az_deg, el_deg, scale, ox, oy):
            az = math.radians(az_deg)
            el = math.radians(el_deg)
            def _p(x, y, z):
                x1 =  x*math.cos(az) + y*math.sin(az)
                y1 = -x*math.sin(az) + y*math.cos(az)
                x2 = x1
                z2 = y1*math.sin(el) + z*math.cos(el)
                return (ox + scale*x2, oy - scale*z2)
            def _d(x, y, z):
                y1 = -x*math.sin(az) + y*math.cos(az)
                return y1*math.cos(el) - z*math.sin(el)
            return _p, _d

        # ── Draw helpers ─────────────────────────────────────────────────────
        BOND_C  = (120, 120, 120)
        FRAME_C = (50,  50,  50)

        def _draw_bond(draw, p1, p2, width=4):
            draw.line([p1, p2], fill=BOND_C, width=width)

        def _draw_frame(draw, p1, p2, width=2):
            draw.line([p1, p2], fill=FRAME_C, width=width)

        def _draw_sphere(draw, px, py, r, rgb):
            px, py = int(px), int(py)
            draw.ellipse([px-r, py-r, px+r, py+r], fill=rgb,
                         outline=(int(rgb[0]*0.5), int(rgb[1]*0.5), int(rgb[2]*0.5)), width=1)
            hr = max(2, int(r*0.38))
            hx = int(px - r*0.28); hy = int(py - r*0.28)
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=(255, 255, 255))

        C_CORNER = (68,  119, 187)
        C_CENTER = (200,  55,  55)
        C_FACE   = ( 45, 158,  75)
        C_INNER  = (145,  65, 200)

        # ── HCP ─────────────────────────────────────────────────────────────
        if is_hcp:
            p, d = _make_proj(20, 25, 85, 210, 220)
            R_hex, c = 1.0, 1.633
            angs = [math.pi/6 + math.pi/3*i for i in range(6)]
            bott = [(R_hex*math.cos(a), R_hex*math.sin(a), 0.0) for a in angs]
            topp = [(x, y, c) for x, y, _ in bott]
            mid_pts = [(R_hex*math.cos(math.pi/6+math.pi/3*(2*i+1))*0.577,
                        R_hex*math.sin(math.pi/6+math.pi/3*(2*i+1))*0.577, c/2)
                       for i in range(3)]

            img  = _PIL.new('RGB', (W, H), (252, 252, 252))
            draw = _IDraw.Draw(img)

            # Prism frame
            for i in range(6):
                j = (i+1) % 6
                _draw_frame(draw, p(*bott[i]), p(*bott[j]))
                _draw_frame(draw, p(*topp[i]), p(*topp[j]))
                _draw_frame(draw, p(*bott[i]), p(*topp[i]), width=1)

            # Bonds: each interstitial atom bonds to 3 bottom + 3 top neighbors
            ctr_pts_b = bott; ctr_pts_t = topp
            for mp in mid_pts:
                # find 3 nearest bottom and 3 nearest top atoms
                dists_b = sorted(range(6), key=lambda i: (mp[0]-bott[i][0])**2+(mp[1]-bott[i][1])**2)
                dists_t = sorted(range(6), key=lambda i: (mp[0]-topp[i][0])**2+(mp[1]-topp[i][1])**2)
                for idx in dists_b[:3]:
                    _draw_bond(draw, p(*mp), p(*bott[idx]))
                for idx in dists_t[:3]:
                    _draw_bond(draw, p(*mp), p(*topp[idx]))

            # Atoms sorted back-to-front
            atoms = [(d(*pt), pt, C_CORNER, 13) for pt in bott+topp+[(0,0,0),(0,0,c)]]
            atoms += [(d(*pt), pt, C_CENTER, 16) for pt in mid_pts]
            atoms.sort(key=lambda a: -a[0])
            for _, pt, col, r in atoms:
                _draw_sphere(draw, *p(*pt), r, col)
            lbl = 'Hexagonal close-packed (HCP)'

        # ── Cubic / tetragonal ───────────────────────────────────────────────
        else:
            cz    = 1.5 if 'tetragonal' in bk else 1.0
            scale = 110 if cz == 1.0 else 88
            p, d  = _make_proj(35, 22, scale, 205, 215)

            corners = [(x,y,z) for x in [0,1] for y in [0,1] for z in [0,cz]]
            EDGE_IDX = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
                        (4,5),(4,6),(5,7),(6,7)]

            img  = _PIL.new('RGB', (W, H), (252, 252, 252))
            draw = _IDraw.Draw(img)

            # Cube wireframe
            for i, j in EDGE_IDX:
                _draw_frame(draw, p(*corners[i]), p(*corners[j]))

            atoms = [(d(*pt), pt, C_CORNER, 14) for pt in corners]
            bonds = []

            if 'body-centered cubic' in bk or bk == 'bcc':
                ctr = (0.5, 0.5, 0.5)
                atoms.append((d(*ctr), ctr, C_CENTER, 18))
                # bonds: center → all 8 corners
                bonds = [(ctr, c) for c in corners]
                lbl = 'Body-centered cubic (BCC)'

            elif 'face-centered cubic' in bk or bk == 'fcc':
                fc_pts = [(.5,.5,0),(.5,0,.5),(0,.5,.5),
                          (.5,.5,1),(.5,1,.5),(1,.5,.5)]
                atoms += [(d(*pt), pt, C_FACE, 15) for pt in fc_pts]
                # bonds: each face-center → its 4 corner neighbors
                for fx,fy,fz in fc_pts:
                    for cx,cy,cz2 in corners:
                        dist2 = (fx-cx)**2+(fy-cy)**2+(fz-cz2)**2
                        if abs(dist2 - 0.5) < 0.01:   # distance = a/√2 ≈ 0.707
                            bonds.append(((fx,fy,fz),(cx,cy,cz2)))
                lbl = 'Face-centered cubic (FCC)'

            elif 'diamond' in bk:
                fc_pts  = [(.5,.5,0),(.5,0,.5),(0,.5,.5),
                           (.5,.5,1),(.5,1,.5),(1,.5,.5)]
                inn_pts = [(.25,.25,.25),(.75,.75,.25),
                           (.75,.25,.75),(.25,.75,.75)]
                atoms += [(d(*pt), pt, C_FACE,  14) for pt in fc_pts]
                atoms += [(d(*pt), pt, C_INNER, 14) for pt in inn_pts]
                all_pts = corners + fc_pts + inn_pts
                # bonds: each inner atom → 4 nearest neighbors (dist ≈ √3/4 ≈ 0.433)
                for ip in inn_pts:
                    nbrs = sorted(all_pts,
                                  key=lambda q: (ip[0]-q[0])**2+(ip[1]-q[1])**2+(ip[2]-q[2])**2)
                    for nb in nbrs[1:5]:
                        bonds.append((ip, nb))
                lbl = 'Diamond cubic'

            elif 'body-centered tetragonal' in bk or 'bct' in bk:
                ctr = (0.5, 0.5, cz/2)
                atoms.append((d(*ctr), ctr, C_CENTER, 18))
                bonds = [(ctr, c) for c in corners]
                lbl = 'Body-centered tetragonal (BCT)'

            elif 'rock salt' in bk:
                # Na positions: corners + face-centers (FCC sublattice)
                na_pts = corners + [(.5,.5,0),(.5,0,.5),(0,.5,.5),
                                    (.5,.5,1),(.5,1,.5),(1,.5,.5)]
                # Cl positions: edge-midpoints + body-center (offset FCC sublattice)
                cl_pts = [(.5,0,0),(0,.5,0),(0,0,.5),
                          (1,.5,0),(1,0,.5),(.5,1,0),
                          (0,1,.5),(.5,0,1),(0,.5,1),
                          (1,.5,1),(1,1,.5),(.5,1,1),
                          (.5,.5,.5)]
                C_NA = (68, 119, 187)   # blue for Na
                C_CL = (220, 160, 40)   # amber for Cl
                atoms  = [(d(*pt), pt, C_NA, 14) for pt in na_pts]
                atoms += [(d(*pt), pt, C_CL, 16) for pt in cl_pts]
                # Bonds: each Cl to its 6 nearest Na (distance = 0.5)
                for cp in cl_pts:
                    for np_ in na_pts:
                        dist2 = sum((cp[i]-np_[i])**2 for i in range(3))
                        if abs(dist2 - 0.25) < 0.01:   # distance = 0.5
                            bonds.append((cp, np_))
                lbl = 'Rock salt (NaCl-type)'

            else:
                # No proper diagram for this structure type —
                # return empty so the main crystal image shows instead
                _LATTICE_DIAGRAM_CACHE[cache_key] = ""
                return ""

            # Draw bonds BEFORE atoms (bonds are behind)
            for b1, b2 in bonds:
                _draw_bond(draw, p(*b1), p(*b2))

            # Draw atoms back-to-front
            atoms.sort(key=lambda a: -a[0])
            for _, pt, col, r in atoms:
                _draw_sphere(draw, *p(*pt), r, col)

        # ── Label ────────────────────────────────────────────────────────────
        try:
            from PIL import ImageFont as _IFont
            font = _IFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
        except Exception:
            font = None
        tw_bbox = draw.textbbox((0, 0), lbl, font=font)
        tw = tw_bbox[2] - tw_bbox[0]
        draw.text(((W-tw)//2, H-28), lbl, fill=(20, 20, 20), font=font)

        out = BytesIO()
        img.save(out, format='PNG', optimize=True)
        result = base64.b64encode(out.getvalue()).decode()
    except Exception:
        result = ""
    _LATTICE_DIAGRAM_CACHE[cache_key] = result
    return result


async def fetch_mp_image(query: str, client: httpx.AsyncClient) -> str:
    """Search Materials Project by name/formula, fetch CIF, render to PNG."""
    if not MP_API_KEY:
        return ""
    try:
        headers = {"X-API-KEY": MP_API_KEY, "accept": "application/json"}
        search_url = "https://api.materialsproject.org/materials/summary/"

        query_lower = query.lower().strip()

        # 0. Check mineral alias table first — these won't be found in PubChem/MP by name
        formula = ""
        sg_hint = None
        alias = _MINERAL_ALIASES.get(query_lower)
        if alias:
            formula, sg_hint = alias
        else:
            sg_hint = _MP_SPACEGROUP_HINTS.get(query_lower)

            # If query looks like a chemical formula (e.g. CuS, TiO2, Fe2O3),
            # use it directly — don't let PubChem substitute a different compound
            _is_formula = bool(_re.match(r'^[A-Z][a-zA-Z0-9]*$', query.strip()))
            if _is_formula:
                formula = query.strip()
            else:
                # 1. Get formula from PubChem for named compounds
                r = await client.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}"
                    "/property/MolecularFormula/JSON"
                )
                if r.status_code == 200:
                    props = r.json().get("PropertyTable", {}).get("Properties", [{}])
                    formula = props[0].get("MolecularFormula", "")
                if not formula:
                    formula = query

        # 2a. If we have a spacegroup hint, search directly — guaranteed correct polymorph
        mat_id = None
        _sym_info: dict = {}
        if sg_hint:
            params = {
                "formula": formula, "spacegroup_number": str(sg_hint),
                "_fields": "material_id,symmetry,energy_above_hull", "_limit": "1",
            }
            sr = await client.get(search_url, params=params, headers=headers)
            hits = sr.json().get("data", []) if sr.status_code == 200 else []
            if hits:
                mat_id = hits[0]["material_id"]
                _sym_info = hits[0].get("symmetry") or {}

        # 2b. No hint — take most stable structure, filtered to reasonable hull energy
        if not mat_id:
            params = {
                "formula": formula,
                "energy_above_hull_max": "0.1",
                "_fields": "material_id,symmetry,energy_above_hull",
                "_limit": "10",
            }
            sr = await client.get(search_url, params=params, headers=headers)
            hits = sr.json().get("data", []) if sr.status_code == 200 else []
            if not hits:
                params.pop("energy_above_hull_max")
                params["_limit"] = "5"
                sr = await client.get(search_url, params=params, headers=headers)
                hits = sr.json().get("data", []) if sr.status_code == 200 else []
            if not hits:
                return ""
            hits.sort(key=lambda h: h.get("energy_above_hull") or 999)
            mat_id = hits[0]["material_id"]
            _sym_info = hits[0].get("symmetry") or {}

        # 3. Fetch structure via core endpoint (returns pymatgen Structure dict)
        core_r = await client.get(
            "https://api.materialsproject.org/materials/core/",
            params={"material_ids": mat_id, "_fields": "material_id,structure", "_limit": "1"},
            headers=headers,
        )
        if core_r.status_code != 200:
            return ""
        core_data = (core_r.json().get("data") or [{}])[0]
        struct_dict = core_data.get("structure")
        if not struct_dict:
            return ""

        # 4. Convert pymatgen Structure → CIF text
        try:
            from pymatgen.core import Structure as _PMGStructure
            structure = _PMGStructure.from_dict(struct_dict)
            tmp_cif = _tempfile.NamedTemporaryFile(
                mode='w', suffix='.cif', delete=False, encoding='utf-8')
            structure.to(fmt="cif", filename=tmp_cif.name)
            tmp_cif.close()
            cif_text = open(tmp_cif.name, encoding='utf-8').read()
            os.unlink(tmp_cif.name)
        except Exception:
            return ""

        # 5. Render CIF to PNG
        loop = _asyncio.get_event_loop()
        img_b64 = await loop.run_in_executor(None, _render_cif_sync, cif_text)
        if not img_b64:
            return ""
        sg_num = _sym_info.get("number") or 0
        sg_sym = _sym_info.get("symbol") or ""
        return {
            "img_b64":        img_b64,
            "spacegroup_num": sg_num,
            "spacegroup_sym": sg_sym,
            "crystal_system": _crystal_system(sg_num) if sg_num else "",
            "bravais":        _bravais_label(sg_sym, sg_num),
        }
    except Exception:
        return ""


async def fetch_cod_rendered_image(query: str, client: httpx.AsyncClient) -> str:
    """Fetch CIF from COD OPTIMADE and render to PNG. Returns base64 or ''."""
    try:
        # Determine element set: try PubChem first, then parse query directly
        formula = ""
        r = await client.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}"
            "/property/MolecularFormula/JSON"
        )
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [{}])
            formula = props[0].get("MolecularFormula", "")

        # If PubChem didn't return a formula, treat query itself as a formula
        if not formula:
            formula = query

        elements = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))
        n = len(elements)
        if n == 0:
            return ""

        el_clauses = " AND ".join(f'elements HAS "{e}"' for e in elements)
        # Try exact nelements match first, then broader
        for fil in [
            f'nelements={n} AND {el_clauses}',
            el_clauses,
        ]:
            cod_r = await client.get(
                "https://www.crystallography.net/cod/optimade/v1/structures",
                params={"filter": fil, "page_limit": 5},
            )
            if cod_r.status_code != 200:
                continue
            items = cod_r.json().get("data", [])
            for item in items:
                cod_id = item["id"]
                cif_r = await client.get(
                    f"https://www.crystallography.net/cod/{cod_id}.cif"
                )
                if cif_r.status_code != 200:
                    continue
                loop = _asyncio.get_event_loop()
                img = await loop.run_in_executor(None, _render_cif_sync, cif_r.text)
                if img:
                    return img
    except Exception:
        pass
    return ""

async def fetch_crystal_structure_image(title: str, client: httpx.AsyncClient) -> str:
    """Search the Wikipedia article's image list for a crystal structure diagram."""
    try:
        # Get all images listed in the article
        api_url = "https://en.wikipedia.org/w/api.php"
        r = await client.get(api_url, params={
            "action": "query", "titles": title,
            "prop": "images", "imlimit": "50", "format": "json",
        })
        pages = r.json().get("query", {}).get("pages", {})
        images = []
        for page in pages.values():
            images = page.get("images", [])

        title_lower = title.lower()

        def score(fname):
            fl = fname.lower()
            if any(skip in fl for skip in CRYSTAL_IMG_SKIP):
                return -1
            s = 0
            # Space group notation in filename → very likely a crystal structure diagram
            if _SPACE_GROUP.search(fl):
                s += 4
            # Explicit structure keywords
            for kw in ("crystal structure", "crystal_structure", "unit cell", "unit_cell",
                       "structure", "lattice", "conventional", "bravais", "crystalline"):
                if kw in fl:
                    s += 2
                    break
            # PNG = diagram, not a photo
            if fl.endswith(".png"):
                s += 1
            # Compound name in filename (minor boost)
            if title_lower in fl:
                s += 1
            # Penalise photo/specimen/gem words
            if any(w in fl for w in CRYSTAL_IMG_PHOTO_WORDS):
                s -= 3
            return s

        images_scored = sorted(
            [(score(img["title"]), img["title"]) for img in images],
            reverse=True,
        )

        # Only try images that scored above 0
        for s, fname in images_scored:
            if s <= 0:
                break
            # Resolve to actual URL
            r2 = await client.get(api_url, params={
                "action": "query", "titles": fname,
                "prop": "imageinfo", "iiprop": "url", "format": "json",
            })
            pages2 = r2.json().get("query", {}).get("pages", {})
            for p2 in pages2.values():
                url = (p2.get("imageinfo") or [{}])[0].get("url", "")
                if not url:
                    continue
                # Determine MIME type from extension
                ext = url.rsplit(".", 1)[-1].lower()
                if ext in ("tif", "tiff", "webm"):
                    continue   # skip non-web formats
                img_r = await client.get(url)
                if img_r.status_code == 200:
                    mime_map = {"svg": "image/svg+xml", "gif": "image/gif",
                                "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
                    mime = mime_map.get(ext, "image/png")
                    return base64.b64encode(img_r.content).decode(), mime
    except Exception:
        pass
    return "", "image/png"

class CrystalRequest(BaseModel):
    query: str

@app.post("/api/crystal-3d")
async def crystal_3d_data(body: CrystalRequest):
    """Fetch CIF crystal structure from COD via PubChem formula lookup."""
    query = body.query.strip()
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "ChemistryAnalyzer/1.0"}) as client:
            # Step 1: get molecular formula from PubChem
            r = await client.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}"
                "/property/MolecularFormula/JSON"
            )
            if r.status_code != 200:
                return {"found": False}
            props = r.json().get("PropertyTable", {}).get("Properties", [{}])
            formula = props[0].get("MolecularFormula", "")
            if not formula:
                return {"found": False}

            # Step 2: extract element symbols and build COD OPTIMADE filter
            elements = sorted(set(re.findall(r"[A-Z][a-z]?", formula)))
            n = len(elements)
            el_clauses = " AND ".join(f'elements HAS "{e}"' for e in elements)
            fil = f"nelements={n} AND {el_clauses}"

            # Step 3: search COD OPTIMADE
            cod_r = await client.get(
                "https://www.crystallography.net/cod/optimade/v1/structures",
                params={"filter": fil, "page_limit": 1},
            )
            if cod_r.status_code != 200:
                return {"found": False}
            items = cod_r.json().get("data", [])
            if not items:
                return {"found": False}

            # Step 4: fetch CIF file
            cod_id = items[0]["id"]
            cif_r = await client.get(f"https://www.crystallography.net/cod/{cod_id}.cif")
            if cif_r.status_code != 200:
                return {"found": False}
            return {"found": True, "cif": cif_r.text}
    except Exception as e:
        return {"found": False, "error": str(e)}

@app.post("/api/crystal")
async def crystal_data(body: CrystalRequest):
    query = body.query.strip()
    if not query:
        return {"found": False}
    try:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "ChemistryAnalyzer/1.0"}) as client:
            # Get summary to confirm it exists and check crystal keywords
            r = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
            )
            if r.status_code != 200:
                return {"found": False, "error": "Not found on Wikipedia"}
            data  = r.json()
            title = data.get("title", query)
            summary_lower = data.get("extract", "").lower()
            is_crystal = any(kw in summary_lower for kw in CRYSTAL_SUMMARY_KEYWORDS)
            img_mime = "image/png"
            structure_note = ""
            # Check if query is a mineral alias — set note for UI
            _alias = _MINERAL_ALIASES.get(query.lower().strip())
            if _alias:
                _alias_formula, _ = _alias
                structure_note = (
                    f"{title} is a mineral variety — showing crystal structure of "
                    f"its parent compound ({_alias_formula})"
                )
            # 1st priority: Materials Project API (accurate, name-aware, 150k+ structures)
            mp_result = await fetch_mp_image(query, client)
            # If MP returned a structure, it's definitely a crystal — override Wikipedia check
            if isinstance(mp_result, dict) and mp_result.get("img_b64"):
                is_crystal = True
            spacegroup_num = 0; spacegroup_sym = ""; crystal_system = ""; bravais = ""
            if isinstance(mp_result, dict):
                img_b64       = mp_result.get("img_b64", "")
                spacegroup_num = mp_result.get("spacegroup_num", 0)
                spacegroup_sym = mp_result.get("spacegroup_sym", "")
                crystal_system = mp_result.get("crystal_system", "")
                bravais        = mp_result.get("bravais", "")
            else:
                img_b64 = mp_result or ""
            # 2nd priority: Wikipedia article image scoring (fallback if no MP key)
            if not img_b64:
                img_b64, img_mime = await fetch_crystal_structure_image(title, client)
            # 3rd priority: COD CIF rendered with ASE
            if not img_b64:
                img_b64 = await fetch_cod_rendered_image(query, client)
            # 4th priority: Wikipedia thumbnail (last resort for inorganics)
            # Skip for mineral aliases — Wikipedia thumbnail will always be a gem/rock photo
            if not img_b64 and not _alias:
                thumb_url = (data.get("thumbnail") or {}).get("source", "")
                if thumb_url:
                    thumb_url = _re.sub(r'/\d+px-', '/400px-', thumb_url)
                    tr = await client.get(thumb_url)
                    if tr.status_code == 200:
                        ext = thumb_url.rsplit(".", 1)[-1].lower()
                        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                                    "png": "image/png", "gif": "image/gif"}
                        img_b64  = base64.b64encode(tr.content).decode()
                        img_mime = mime_map.get(ext, "image/jpeg")
            # 5th priority: for organics/polymers with no crystal structure,
            # render molecular structure of the repeating unit via RDKit
            if not img_b64 and RDKIT_OK:
                try:
                    pu_r = await client.get(
                        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}"
                        "/property/IsomericSMILES,MolecularFormula/JSON"
                    )
                    if pu_r.status_code == 200:
                        pu = pu_r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                        smiles  = pu.get("IsomericSMILES", "")
                        formula = pu.get("MolecularFormula", "")
                        if smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                from rdkit.Chem import Draw as _Draw
                                img_pil = _Draw.MolToImage(mol, size=(500, 380))
                                buf = BytesIO()
                                img_pil.save(buf, format="PNG")
                                img_b64 = base64.b64encode(buf.getvalue()).decode()
                                img_mime = "image/png"
                                structure_note = (
                                    f"No crystal structure found in database — "
                                    f"showing molecular structure of repeating unit ({formula})"
                                )
                except Exception:
                    pass
            # Generate unit cell diagram (BCC/FCC/HCP/etc.) if bravais type known
            lattice_diagram = ""
            if bravais:
                loop = _asyncio.get_event_loop()
                lattice_diagram = await loop.run_in_executor(
                    None, _gen_lattice_diagram_sync, bravais, query.lower().strip()
                )
            return {
                "found":           True,
                "is_crystal":      is_crystal,
                "img_mime":        img_mime,
                "title":           title,
                "img_b64":         img_b64,
                "structure_note":  structure_note,
                "spacegroup_num":  spacegroup_num,
                "spacegroup_sym":  spacegroup_sym,
                "crystal_system":  crystal_system,
                "bravais":         bravais,
                "lattice_diagram": lattice_diagram,
            }
    except Exception as e:
        return {"found": False, "error": str(e)}

@app.post("/api/polymer")
async def polymer_endpoint(body: CrystalRequest):
    query = body.query.strip()
    if not query:
        return {"found": False}
    unit = get_polymer_unit(query)
    if not unit:
        return {"found": False}
    return {"found": True, "title": query, **unit}


class LookupNameRequest(BaseModel):
    query: str          # compound name or SMILES string
    formula: str = ""  # expected Hill formula for validation

@app.post("/api/lookup-name")
async def lookup_name(body: LookupNameRequest):
    q = body.query.strip()
    if not q:
        return {"found": False}
    # Detect SMILES (contains typical SMILES-only characters)
    is_smiles = any(c in q for c in ['=', '#', '[', '@', '/', '\\'])
    img_b64 = ""
    formula = ""
    name = q
    if is_smiles:
        img_b64 = mol_to_b64_png(q, size=(300, 220))
        if not img_b64:
            return {"found": False, "error": "Invalid SMILES string"}
        if RDKIT_OK:
            try:
                from rdkit.Chem import rdMolDescriptors
                mol = Chem.MolFromSmiles(q)
                if mol:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
            except Exception:
                pass
        return {"found": True, "name": q, "formula": formula, "img_b64": img_b64}
    else:
        if not PUBCHEM_OK:
            return {"found": False, "error": "PubChem not available"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property"
                    "/MolecularFormula,IsomericSMILES,IUPACName,Title/JSON"
                )
            if r.status_code != 200:
                return {"found": False}
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if not props:
                return {"found": False}
            p = props[0]
            smiles  = p.get("IsomericSMILES") or p.get("SMILES") or ""
            formula = p.get("MolecularFormula", "")
            pname   = p.get("IUPACName") or p.get("Title") or q
            cid     = p.get("CID")
        except Exception as e:
            return {"found": False, "error": str(e)}
        img_b64 = mol_to_b64_png(smiles, size=(300, 220)) if smiles else ""
        return {"found": True, "name": pname, "formula": formula,
                "img_b64": img_b64, "smiles": smiles, "cid": cid}


# ── Stream: analyze a specific isomer ─────────────────────────────────────────
class IsomerAnalyzeRequest(BaseModel):
    formula: str
    isomers_text: str
    choice: int
    verified_text: str

@app.post("/api/stream/analyze-isomer")
def stream_analyze_isomer(body: IsomerAnalyzeRequest):
    user_content = (
        f"Verified data from chemistry libraries:\n{body.verified_text}\n\n"
        f"From these structural isomers for {body.formula}:\n\n"
        f"{body.isomers_text}\n\n"
        f"Provide a full analysis of isomer number {body.choice}."
    )
    return StreamingResponse(
        ollama_sse(ANALYSIS_SYSTEM, user_content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Stream: reaction prediction ────────────────────────────────────────────────
class ReactionRequest(BaseModel):
    reactants: str
    conditions: str = ""

@app.post("/api/stream/reaction")
def stream_reaction(body: ReactionRequest):
    cond_part = f"\nConditions / Reagents: {body.conditions}" if body.conditions else ""
    user_content = (
        f"Reactants: {body.reactants}{cond_part}\n\n"
        f"Predict the major product(s) and explain the full reaction mechanism."
    )
    return StreamingResponse(
        ollama_sse(REACTION_SYSTEM, user_content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    libs = {
        "ollama": OLLAMA_OK, "pubchem": PUBCHEM_OK, "rdkit": RDKIT_OK,
        "molmass": MOLMASS_OK, "periodictable": PERIODIC_OK,
    }
    return {"model": MODEL, "libs": libs}
