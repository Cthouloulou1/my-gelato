#!/usr/bin/env python3
"""
Exemples ChimeraX de chaque type d'erreur de analyze_flow_geometry.py :
sens de chaine, orientation, rotamere, decalage. Aucune inference : les
positions viennent des fichiers <ensemble>_predictions.npz de evaluate_flow.py.

Le premier --pred sert a classer les erreurs ; pour chaque type, on prend
--n residus typiques (RMSD proche de la mediane du type, chaine laterale assez
longue pour etre lisible, voisins i-1 et i+1 presents, entrees differentes).
Les autres --pred sont affiches sur le meme residu (par ex. le modele avec
contexte, pour voir s'il corrige l'erreur).

Criteres de choix, en plus du type :
    sens de chaine : residu autre que GLY (on voit le CB du mauvais cote)
    orientation    : repere du squelette tourne de 45 a 120 degres (au-dela, proche
                     d'un sens de chaine inverse)
    rotamere       : squelette bon (< 15 deg) et un chi faux de plus de 90 deg
    decalage       : aucun critere de plus

Ecrit dans --out-dir/<ensemble>/<n°>_<type>_<PDB>_<chaine><numero><residu>/ :
    vrai.cif       vraie structure, residu et ses voisins (+/- --flank)
    <modele>.cif   residu predit par chaque modele (B-factor = RMSD, A)
    view.cxc       session : densite autour du residu, atomes N/C/CB etiquetes,
                   distance de l'atome le plus faux, legende de l'erreur
et --out-dir/<ensemble>/exemples.csv (mesures de chaque exemple).

Usage (contexte : hybride classe, hybrid_chain_pep affiche en plus) :
    python make_error_examples.py --atom14 /data/eyerly/splits_atom14.npz \\
        --splits /data/eyerly/pdb_cryoem_splits/splits.json --set test_2_3A \\
        --pred hybride=runs/evaluation_flow_context_e100/hybride \\
        --pred hybrid_chain_pep=runs/evaluation_flow_context_e100/hybrid_chain_pep
Puis :  chimerax runs/chimerax_erreurs/test_2_3A/1_orientation_*/view.cxc
"""

import argparse
import csv
import glob
import json
import os
from types import SimpleNamespace

import numpy as np

import analyze_flow_geometry as geo
from flow_common import ATOM14, CA_SLOT, STANDARD
from make_chimerax_view import MAP_COLOR, TRUE_COLOR, load_atom14, load_predictions, write_cif
from train_utils import CATEGORICAL

KINDS = ["sens de chaine", "orientation", "rotamere", "decalage"]
SLUG = {"sens de chaine": "sens_chaine", "orientation": "orientation",
        "rotamere": "rotamere", "decalage": "decalage"}
NEIGHBOUR_COLOR = "#c9ccd1"


def describe(kind, o, r, rt):
    """Legende courte de l'erreur du residu r."""
    if kind == "sens de chaine":
        return "Sens de chaine : N et C echanges"
    if kind == "orientation":
        return f"Orientation : repere du squelette tourne de {o['frame_err'][r]:.0f} deg"
    if kind == "rotamere":
        d = o["dchi"][r]
        bad = [f"chi{j + 1} {d[j]:.0f} deg" for j in range(4) if np.isfinite(d[j]) and d[j] > 40]
        return "Rotamere : " + ", ".join(bad) + f" (squelette a {o['frame_err'][r]:.0f} deg)"
    return (f"Decalage : bon squelette et bon rotamere, atomes deplaces "
            f"(forme {o['shape_rmsd'][r]:.2f} A apres superposition)")


def pick(kind, o, rt, n_atoms, has_nb, entry, n):
    """n indices typiques du type `kind` (entrees differentes)."""
    sel = (o["fail_kind"] == kind) & has_nb
    if kind == "sens de chaine":
        sel &= rt != STANDARD.index("GLY")
    else:
        sel &= n_atoms >= 7                    # au moins 2 atomes apres le CB
    if kind == "orientation":
        sel &= (o["frame_err"] > 45) & (o["frame_err"] < 120)
    if kind == "rotamere":
        sel &= (o["frame_err"] < 15) & (np.nan_to_num(o["dchi"], nan=0.0).max(1) > 90)
    cand = np.where(sel)[0]
    if len(cand) == 0:
        return []
    med = np.median(o["rmsd"][cand])
    out, seen = [], set()
    for r in cand[np.argsort(np.abs(o["rmsd"][cand] - med))]:
        if entry[r] in seen:
            continue
        out.append(int(r))
        seen.add(entry[r])
        if len(out) == n:
            break
    return out


def session(path, kind_text, models, map_path, chain, seq, flank, worst, sd_level):
    """models : [(nom, chemin, couleur)], le premier est le vrai."""
    n = len(models)
    res = f"/{chain}:{seq}"
    L = ["# Session generee par make_error_examples.py",
         "close session", "set bgColor white", "lighting soft",
         "graphics silhouettes true width 1.5"]
    for name, p, _ in models:
        L.append(f'open "{os.path.abspath(p)}" name {name}')
    L += ["cartoon hide", "hide atoms", f"show #1-{n} atoms", f"style #1-{n} stick",
          f"size #1-{n} stickRadius 0.15",
          "# voisins (vrai) : fins et gris",
          f"color #1 {NEIGHBOUR_COLOR} target a", f"size #1 stickRadius 0.08"]
    for i, (name, _, color) in enumerate(models, start=1):
        L.append(f"# #{i} = {name}")
        if i == 1:
            L += [f"color #1{res} {color} target a", f"size #1{res} stickRadius 0.25"]
        else:
            L.append(f"color #{i} {color} target a")
    L.append(f"color byhetero #1{res}")
    for i in range(2, n + 1):
        L.append(f"color byhetero #{i}")
    if map_path:
        m = n + 1
        L += [f'open "{os.path.abspath(map_path)}"',
              f"volume #{m} style mesh sdLevel {sd_level} color {MAP_COLOR}",
              f"volume zone #{m} nearAtoms #1{res} range 3"]
    L += [f"label #1{res} residues height 0.7 color black",
          f"label #1-2{res}@N,C,CB atoms height 0.4",
          "# atome le plus faux du premier modele : distance a sa vraie position",
          f"distance #1{res}@{worst} #2{res}@{worst}",
          f'2dlabels text "{kind_text}" xpos 0.02 ypos 0.94 size 18 color black',
          f'2dlabels text "gris epais : vrai , couleur : {", ".join(m[0] for m in models[1:])}'
          f' , fins : voisins +/-{flank}" xpos 0.02 ypos 0.89 size 13 color dimgray',
          f"view #1{res}", "",
          "# --- utile ---",
          "# masquer un modele : hide #3 models   (et show #3 models)",
          "# seuil de la carte : volume #N sdLevel 2",
          f"# image : save {os.path.abspath(os.path.join(os.path.dirname(path), 'view.png'))} "
          "width 1600 supersample 3"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--set", default="test_2_3A", help="val, test_1_2A, test_2_3A, test_3_4A")
    ap.add_argument("--pred", action="append", required=True,
                    help="nom=dossier d'un modele (contenant <ensemble>_predictions.npz) ; "
                         "le premier sert a classer les erreurs")
    ap.add_argument("--n", type=int, default=3, help="exemples par type d'erreur")
    ap.add_argument("--flank", type=int, default=3, help="voisins affiches de chaque cote")
    ap.add_argument("--data-dir", default=None,
                    help="dossier <PDB>/emd_*.map (defaut : celui de splits.json)")
    ap.add_argument("--sd-level", type=float, default=1.5)
    ap.add_argument("--chi-tol", type=float, default=40.0)
    ap.add_argument("--frame-tol", type=float, default=30.0)
    ap.add_argument("--fail-rmsd", type=float, default=1.0)
    ap.add_argument("--out-dir", default="runs/chimerax_erreurs")
    args = ap.parse_args()
    if len(args.pred) > len(CATEGORICAL):
        raise SystemExit(f"au plus {len(CATEGORICAL)} modeles")

    with open(args.splits) as f:
        set_ids = json.load(f)[args.set]
    a14 = load_atom14(args.atom14)

    # modele de reference : classement des erreurs
    specs = [tuple(p.split("=", 1)) for p in args.pred]
    files = [os.path.join(d, f"{args.set}_predictions.npz") for _, d in specs]
    pred, gt, mask, rt, rmsd = geo.load(files[0])
    _, _, idx = load_predictions(files[0], a14, set_ids)
    o = geo.residue_analysis(pred, gt, mask, rt, rmsd, SimpleNamespace(
        chi_tol=args.chi_tol, frame_tol=args.frame_tol, fail_rmsd=args.fail_rmsd))
    others = []
    for (name, _), f in zip(specs[1:], files[1:]):
        p2, r2, i2 = load_predictions(f, a14, set_ids)
        where = {int(i): k for k, i in enumerate(i2)}
        others.append((name, p2, r2, where))

    # voisins dans atom14 (meme entree, meme chaine, numeros consecutifs)
    key = {(str(a14["pdb_ids"][i]), str(a14["chains"][i]), int(a14["seqids"][i])): i
           for i in range(len(a14["pdb_ids"]))}
    entry = np.array([str(a14["pdb_ids"][i]) for i in idx])

    def nb(i, d):
        return key.get((str(a14["pdb_ids"][i]), str(a14["chains"][i]), int(a14["seqids"][i]) + d))

    has_nb = np.array([nb(i, -1) is not None and nb(i, 1) is not None for i in idx])
    n_atoms = mask.sum(1)

    out_root = os.path.join(args.out_dir, args.set)
    os.makedirs(out_root, exist_ok=True)
    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.splits))
    rows = []
    print(f"{args.set} : {len(rt)} residus, classes selon {specs[0][0]}")
    for k, kind in enumerate(KINDS, start=1):
        chosen = pick(kind, o, rt, n_atoms, has_nb, entry, args.n)
        n_kind = int((o["fail_kind"] == kind).sum())
        print(f"\n{kind} : {n_kind} residus ({n_kind / len(rt):.1%}), {len(chosen)} exemple(s)")
        for j, r in enumerate(chosen, start=1):
            i = int(idx[r])
            pid, chain, seq = str(a14["pdb_ids"][i]), str(a14["chains"][i]), int(a14["seqids"][i])
            resname = str(a14["resnames"][i])
            ca = a14["ca_xyz"][i]
            d = os.path.join(out_root, f"{k}_{SLUG[kind]}_{j}_{pid}_{chain}{seq}{resname}")
            os.makedirs(d, exist_ok=True)

            # vrai : voisins (positions deposees) + residu (numerotation symetrique
            # la plus proche de la prediction, comme le RMSD)
            true_res = []
            for dd in range(-args.flank, args.flank + 1):
                ii = i if dd == 0 else nb(i, dd)
                if ii is None:
                    continue
                c = a14["ca_xyz"][ii]
                xyz = c + (gt[r] if dd == 0 else a14["pos"][ii])
                xyz[CA_SLOT] = c
                true_res.append((chain, int(a14["seqids"][ii]), str(a14["resnames"][ii]), xyz,
                                 a14["mask"][ii], 0.0))
            models = [("vrai", os.path.join(d, "vrai.cif"), TRUE_COLOR)]
            write_cif(models[0][1], "vrai", true_res)

            def one(name, p, err, color):
                xyz = ca + p
                xyz[CA_SLOT] = ca
                path = os.path.join(d, f"{name}.cif")
                write_cif(path, name, [(chain, seq, resname, xyz, a14["mask"][i], float(err))])
                models.append((name, path, color))

            one(specs[0][0], pred[r], rmsd[r], CATEGORICAL[0])
            extra = []
            for c, (name, p2, r2, where) in enumerate(others, start=1):
                if i in where:
                    one(name, p2[where[i]], r2[where[i]], CATEGORICAL[c])
                    extra.append(f"{name} {r2[where[i]]:.2f} A")

            err = np.linalg.norm(pred[r] - gt[r], axis=-1)
            err[~mask[r]] = -1
            err[CA_SLOT] = -1
            worst = ATOM14[resname][int(np.argmax(err))]
            text = describe(kind, o, r, rt)
            maps = sorted(glob.glob(os.path.join(data_dir, pid, "emd_*.map")))
            if not maps:
                print(f"  attention : pas de carte {data_dir}/{pid}/emd_*.map")
            session(os.path.join(d, "view.cxc"), text, models, maps[0] if maps else None,
                    chain, seq, args.flank, worst, args.sd_level)

            dchi = o["dchi"][r]
            rows.append({"type": kind, "dossier": os.path.basename(d), "pdb": pid,
                         "chaine": chain, "numero": seq, "residu": resname,
                         "rmsd": f"{rmsd[r]:.3f}", "repere_deg": f"{o['frame_err'][r]:.1f}",
                         "forme": f"{o['shape_rmsd'][r]:.3f}",
                         **{f"dchi{q + 1}": ("" if not np.isfinite(dchi[q]) else f"{dchi[q]:.0f}")
                            for q in range(4)},
                         "atome_pire": worst, "erreur_pire": f"{err.max():.2f}",
                         "autres_modeles": "; ".join(extra)})
            print(f"  {pid} {chain}{seq} {resname:3s}  RMSD {rmsd[r]:.2f} A  "
                  f"repere {o['frame_err'][r]:5.1f} deg  pire {worst} {err.max():.2f} A"
                  + (f"  |  {', '.join(extra)}" if extra else ""))

    if rows:
        with open(os.path.join(out_root, "exemples.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nEcrit dans {out_root}  (un dossier par exemple, ouvrir view.cxc dans ChimeraX)")


if __name__ == "__main__":
    main()
