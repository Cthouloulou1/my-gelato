#!/usr/bin/env python3
"""
Visualisation ChimeraX des acides amines reconstruits par les modeles de
flow, superposes a la vraie structure et a la carte de densite, sur une
portion de proteine. Aucune inference : les positions viennent des fichiers
<ensemble>_predictions.npz de evaluate_flow.py.

Pour une entree (PDB) d'un ensemble, ecrit dans --out-dir/<ensemble>_<PDB>_<chaine><debut>-<fin>/ :
    vrai.cif             residus de l'entree (positions deposees)
    <modele>.cif         memes residus, positions predites ; B-factor = RMSD du
                         residu (A), pour colorer par erreur
    residus.csv          RMSD de chaque residu de la portion, par modele
    view.cxc             session ChimeraX : structures en batonnets, carte en
                         maillage autour de la portion, etiquettes, camera
Ouvrir avec :  chimerax view.cxc   (ou, dans ChimeraX : open /chemin/view.cxc)

Choix de la portion (residus consecutifs d'une meme chaine) :
    --residues A:50-61   portion imposee
    --pick median        (defaut) portion dont l'erreur moyenne est la mediane
                         des portions : un exemple representatif
    --pick best | worst  la mieux / la moins bien reconstruite
    --pick aromatic      la plus riche en PHE/TYR/TRP/HIS (cycles)
L'erreur d'une portion est celle du premier modele (--ref pour en choisir un
autre). --pdb-id impose l'entree ; sinon celle qui a le plus de residus.
--list affiche les entrees de l'ensemble avec leur RMSD moyen par modele.

Carte : <data-dir>/<PDB>/emd_*.map (download_datasets.py), par defaut le
dossier de splits.json. Seuil en ecarts-types (--sd-level).

Usage :
    python make_chimerax_view.py --atom14 /data/eyerly/splits_atom14.npz \\
        --splits /data/eyerly/pdb_cryoem_splits/splits.json \\
        --eval-dir runs/evaluation_flow_e100 --set test_1_2A --pick aromatic
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from flow_common import ATOM14, CA_SLOT, STANDARD
from train_utils import CATEGORICAL, TEXT_SECONDARY

VARIANT_ORDER = ["joint", "atome", "backbone", "hybride"]
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
TRUE_COLOR = TEXT_SECONDARY
MAP_COLOR = "#9ec5f4"


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

def load_atom14(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("pos", "mask", "pdb_ids", "chains", "seqids", "resnames", "ca_xyz")}


def load_predictions(path, a14, set_ids):
    """pred (N,14,3), rmsd (N,), idx (N,) indices dans atom14.npz.
    Anciens fichiers sans 'idx' : ordre de evaluate_flow (residus de
    l'ensemble dans l'ordre de atom14.npz), verifie sur les vraies positions."""
    d = np.load(path)
    if "idx" in d.files:
        idx = d["idx"]
    else:
        ids = set(set_ids)
        idx = np.array([i for i, p in enumerate(a14["pdb_ids"]) if str(p) in ids])
    if len(idx) != len(d["gt"]) or not np.allclose(d["gt"], a14["pos"][idx], atol=1e-3):
        raise SystemExit(f"{path} : impossible de retrouver les residus (fichier d'un autre "
                         f"ensemble, ou atom14.npz different de celui de l'evaluation ?)")
    return d["pred"], d["rmsd"], idx


def windows(rows, a14, size):
    """Portions de `size` residus consecutifs (meme chaine, numeros qui se
    suivent). rows : indices (dans les tableaux de prediction) d'une entree."""
    by_chain = {}
    for r in rows:
        by_chain.setdefault(str(a14["chains"][r[1]]), []).append(r)
    out = []
    for chain, rs in by_chain.items():
        rs.sort(key=lambda r: int(a14["seqids"][r[1]]))
        seq = [int(a14["seqids"][r[1]]) for r in rs]
        for s in range(len(rs) - size + 1):
            if seq[s + size - 1] - seq[s] == size - 1:
                out.append((chain, rs[s:s + size]))
    return out


# ---------------------------------------------------------------------------
# Ecriture mmCIF (chaines a plusieurs caracteres acceptees, contrairement au PDB)
# ---------------------------------------------------------------------------

def write_cif(path, name, residues):
    """residues : liste de (chaine, numero, nom, positions absolues (14,3),
    masque (14,), bfactor)."""
    lines = [f"data_{name}", "#", "loop_"] + [f"_atom_site.{c}" for c in (
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id",
        "label_entity_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
        "pdbx_PDB_model_num", "label_alt_id", "pdbx_PDB_ins_code")]
    serial = 1
    for chain, seqid, resname, xyz, m, b in residues:
        for s, atom in enumerate(ATOM14[resname]):
            if not m[s]:
                continue
            x, y, z = xyz[s]
            lines.append(f"ATOM {serial} {atom[0]} {atom} {resname} {chain} 1 {seqid} "
                         f"{x:.3f} {y:.3f} {z:.3f} 1.00 {b:.2f} {seqid} {resname} {chain} "
                         f"{atom} 1 . ?")
            serial += 1
    lines.append("#")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Session ChimeraX
# ---------------------------------------------------------------------------

def chimerax_script(out_dir, models, map_path, chain, first, last, sd_level):
    """models : liste de (nom, chemin .cif, couleur) ; le premier est le vrai."""
    n = len(models)
    allm = f"#1-{n}"
    sel = f"/{chain}:{first}-{last}"
    L = ["# Session generee par make_chimerax_view.py",
         "close session",
         "set bgColor white",
         "lighting soft",
         "graphics silhouettes true width 1.5"]
    for i, (name, path, _) in enumerate(models, start=1):
        L.append(f'open "{os.path.abspath(path)}" name {name}')
    L += ["cartoon hide", "hide atoms",
          f"show {allm}{sel} atoms",
          f"style {allm} stick",
          f"size {allm} stickRadius 0.15"]
    for i, (name, _, color) in enumerate(models, start=1):
        L += [f"# #{i} = {name}", f"color #{i} {color} target a"]
    L += [f"color byhetero {allm}",
          "# verite plus epaisse",
          "size #1 stickRadius 0.25"]
    if map_path:
        m = n + 1
        L += [f'open "{os.path.abspath(map_path)}"',
              f"volume #{m} style mesh sdLevel {sd_level} color {MAP_COLOR}",
              f"volume zone #{m} nearAtoms #1{sel} range 2.5"]
    L += [f"label #1{sel} residues height 0.6 color {TRUE_COLOR}",
          f"view #1{sel}",
          "",
          "# --- utile ---",
          "# afficher / masquer un modele : show #3 models  /  hide #3 models",
          f"# colorer un modele par l'erreur du residu (B-factor = RMSD, A) :",
          f"#   color bfactor #2{sel} palette 0,#2a78d6:1,#fcfcfb:2,#e34948",
          "# seuil de la carte : volume #N sdLevel 2   (N = numero de la carte)",
          f"# image : save {os.path.abspath(os.path.join(out_dir, 'view.png'))} "
          "width 1600 supersample 3"]
    with open(os.path.join(out_dir, "view.cxc"), "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--set", default="test_1_2A", help="val, test_1_2A, test_2_3A, test_3_4A")
    ap.add_argument("--eval-dir", default=None,
                    help="dossier de evaluate_flow.py : tous ses modeles")
    ap.add_argument("--pred", action="append", default=[],
                    help="nom=dossier d'un modele (contenant <ensemble>_predictions.npz)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="ne garder que ces modeles (ex. joint hybride)")
    ap.add_argument("--ref", default=None, help="modele servant a choisir la portion")
    ap.add_argument("--pdb-id", default=None)
    ap.add_argument("--residues", default=None, help="portion imposee, ex. A:50-61")
    ap.add_argument("--window", type=int, default=12, help="longueur de la portion")
    ap.add_argument("--pick", choices=["median", "best", "worst", "aromatic"], default="median")
    ap.add_argument("--data-dir", default=None,
                    help="dossier <PDB>/emd_*.map (defaut : celui de splits.json)")
    ap.add_argument("--sd-level", type=float, default=1.5)
    ap.add_argument("--list", action="store_true", help="lister les entrees et sortir")
    ap.add_argument("--out-dir", default="runs/chimerax")
    args = ap.parse_args()

    with open(args.splits) as f:
        set_ids = json.load(f)[args.set]
    a14 = load_atom14(args.atom14)

    specs = []
    if args.eval_dir:
        subs = [os.path.basename(os.path.dirname(p)) for p in
                glob.glob(os.path.join(args.eval_dir, "*", f"{args.set}_predictions.npz"))]
        subs.sort(key=lambda g: (VARIANT_ORDER.index(g) if g in VARIANT_ORDER else 99, g))
        specs += [(s, os.path.join(args.eval_dir, s)) for s in subs]
    specs += [tuple(p.split("=", 1)) for p in args.pred]
    if args.models:
        specs = [s for s in specs if s[0] in args.models]
    if not specs:
        raise SystemExit(f"aucun {args.set}_predictions.npz trouve (--eval-dir / --pred)")
    if len(specs) > len(CATEGORICAL):
        raise SystemExit(f"au plus {len(CATEGORICAL)} modeles (--models)")

    preds = {}
    for name, d in specs:
        pred, rmsd, idx = load_predictions(os.path.join(d, f"{args.set}_predictions.npz"),
                                           a14, set_ids)
        preds[name] = (pred, rmsd, idx)
    names = list(preds)
    idx0 = preds[names[0]][2]
    for n in names[1:]:
        if not np.array_equal(preds[n][2], idx0):
            raise SystemExit(f"{n} : residus differents de {names[0]}")
    ref = args.ref or names[0]
    rmsd_ref = preds[ref][1]
    entry_of = np.array([str(a14["pdb_ids"][i]) for i in idx0])

    if args.list:
        print(f"{args.set} : {len(set(entry_of))} entrees")
        print(f"{'PDB':>8s} {'n':>6s}  " + "  ".join(f"{n:>9s}" for n in names))
        for pid in sorted(set(entry_of), key=lambda p: -(entry_of == p).sum()):
            m = entry_of == pid
            print(f"{pid:>8s} {m.sum():6d}  " + "  ".join(
                f"{np.nanmean(preds[n][1][m]):9.3f}" for n in names))
        return

    pid = args.pdb_id or max(set(entry_of), key=lambda p: (entry_of == p).sum())
    rows = [(r, idx0[r]) for r in np.where(entry_of == pid)[0]]
    if not rows:
        raise SystemExit(f"{pid} absent de {args.set}")

    if args.residues:
        chain, rng = args.residues.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        win = [r for r in rows if str(a14["chains"][r[1]]) == chain
               and lo <= int(a14["seqids"][r[1]]) <= hi]
        if not win:
            raise SystemExit(f"aucun residu {args.residues} dans {pid}")
    else:
        cands = windows(rows, a14, args.window)
        if not cands:
            raise SystemExit(f"{pid} : pas de portion de {args.window} residus consecutifs "
                             f"(residus incomplets exclus) ; essayer --window plus petit")
        score = np.array([np.nanmean(rmsd_ref[[r[0] for r in w]]) for _, w in cands])
        if args.pick == "best":
            k = int(np.argmin(score))
        elif args.pick == "worst":
            k = int(np.argmax(score))
        elif args.pick == "aromatic":
            arom = np.array([sum(str(a14["resnames"][r[1]]) in AROMATIC for r in w)
                             for _, w in cands])
            best = np.where(arom == arom.max())[0]
            k = int(best[np.argmin(np.abs(score[best] - np.median(score)))])
        else:
            k = int(np.argmin(np.abs(score - np.median(score))))
        chain, win = cands[k]
    seqs = [int(a14["seqids"][r[1]]) for r in win]
    first, last = min(seqs), max(seqs)

    out = os.path.join(args.out_dir, f"{args.set}_{pid}_{chain}{first}-{last}")
    os.makedirs(out, exist_ok=True)

    def residues(coords_of, bfac_of):
        res = []
        for r, i in rows:
            ca = a14["ca_xyz"][i]
            xyz = ca + coords_of(r, i)
            xyz[CA_SLOT] = ca
            res.append((str(a14["chains"][i]), int(a14["seqids"][i]), str(a14["resnames"][i]),
                        xyz, a14["mask"][i], bfac_of(r)))
        return res

    models = [("vrai", os.path.join(out, "vrai.cif"), TRUE_COLOR)]
    write_cif(models[0][1], "vrai", residues(lambda r, i: a14["pos"][i], lambda r: 0.0))
    for k, n in enumerate(names):
        pred, rmsd, _ = preds[n]
        path = os.path.join(out, f"{n}.cif")
        write_cif(path, n, residues(lambda r, i: pred[r],
                                    lambda r: float(np.nan_to_num(rmsd[r]))))
        models.append((n, path, CATEGORICAL[k]))

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.splits))
    maps = sorted(glob.glob(os.path.join(data_dir, pid, "emd_*.map")))
    if not maps:
        print(f"Attention : pas de carte {data_dir}/{pid}/emd_*.map (--data-dir) ; "
              f"session sans densite")
    chimerax_script(out, models, maps[0] if maps else None, chain, first, last, args.sd_level)

    with open(os.path.join(out, "residus.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chaine", "numero", "residu"] + [f"rmsd_{n}" for n in names])
        for r, i in sorted(win, key=lambda r: int(a14["seqids"][r[1]])):
            w.writerow([chain, int(a14["seqids"][i]), str(a14["resnames"][i])]
                       + [f"{preds[n][1][r]:.3f}" for n in names])

    print(f"{args.set} / {pid} / chaine {chain} {first}-{last} ({len(win)} residus, "
          f"choix : {'impose' if args.residues else args.pick})")
    print(f"{'residu':>10s}  " + "  ".join(f"{n:>9s}" for n in names))
    for r, i in sorted(win, key=lambda r: int(a14["seqids"][r[1]])):
        print(f"{str(a14['resnames'][i]):>6s}{int(a14['seqids'][i]):4d}  "
              + "  ".join(f"{preds[n][1][r]:9.3f}" for n in names))
    print(f"{'moyenne':>10s}  " + "  ".join(
        f"{np.nanmean(preds[n][1][[r for r, _ in win]]):9.3f}" for n in names))
    print(f"\nEcrit dans {out}\nOuvrir : chimerax {os.path.join(out, 'view.cxc')}")


if __name__ == "__main__":
    main()
