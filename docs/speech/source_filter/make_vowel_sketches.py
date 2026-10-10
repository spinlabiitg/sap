"""Vowel-gallery mouth sketches for ../source_filter.html: a schematic side view of the head (facing left) with the air
space of the vocal tract shaded, plus a small front view of the lips.

    python make_vowel_sketches.py      ->  fig/07_gallery_sketch_{i,ae,a,u}.png

Schematic, not anatomical tracings: tongue positions follow the standard descriptions (high/low, front/back) and
the vowel chart; the jaw opens more for low vowels; lips spread for /i/ and round for /u/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch  # noqa: E402
from scipy.interpolate import splprep, splev  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)

SKIN, SKIN_EDGE = "#F3DCC8", "#9C7B63"
AIR, AIR_EDGE = "#CFE6F7", "#7FB2D9"
TONGUE, TONGUE_EDGE = "#E58C8C", "#B04A4A"
BONE, BONE_EDGE = "#EFEAE2", "#9A9183"
TOOTH = "#FFFFFF"
INK, MUTED, RED = "#10233F", "#5C6B80", "#C1272D"


def smooth(pts, n=200, closed=False, s=0.0):
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, per=closed, k=3)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


def rot(pts, ang_deg, centre):
    a = np.deg2rad(ang_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(pts, float) - centre) @ R.T + centre


TMJ = np.array([7.4, 6.0])           # jaw hinge (approximately at the ear)

# vowel settings: jaw opening (deg), tongue dorsum peak, tongue-root x, lip opening (front view), rounding, spreading
V = {
    "i":  dict(jaw=3,  peak=(3.55, 5.15), root=6.55, lip_h=0.28, lip_w=1.95, rnd=0.0, spread=1.0, ipa="/i/"),
    "ae": dict(jaw=12, peak=(3.55, 3.85), root=6.65, lip_h=0.95, lip_w=1.85, rnd=0.0, spread=0.6, ipa="/æ/"),
    "a":  dict(jaw=15, peak=(5.25, 3.55), root=7.15, lip_h=1.15, lip_w=1.45, rnd=0.0, spread=0.2, ipa="/ɑ/"),
    "u":  dict(jaw=4,  peak=(5.55, 5.05), root=6.60, lip_h=0.42, lip_w=0.48, rnd=1.0, spread=0.0, ipa="/u/"),
}


def draw(key, labels=False):
    p = V[key]
    fig = plt.figure(figsize=(3.6, 3.6), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.6, 9.4); ax.set_ylim(-3.6, 9.4); ax.set_aspect("equal"); ax.axis("off")
    jaw = p["jaw"]
    J = lambda pts: rot(pts, jaw, TMJ)                     # noqa: E731  lower-jaw parts rotate open (clockwise in this view)

    # ---- the head (upper, fixed) and neck, as one skin silhouette behind everything
    lip_fwd = -0.45 * p["rnd"] + 0.15 * p["spread"]
    upper = [(9.8, 9.3), (3.0, 9.3), (1.6, 8.6), (1.0, 7.6), (-0.15, 6.0), (0.2, 5.55), (0.95, 5.3),
             (0.75, 4.75), (0.15 + lip_fwd, 4.45), (0.35 + lip_fwd, 4.05), (1.5, 4.05)]
    lower_face = J([(1.5, 3.95), (0.4 + lip_fwd, 3.85), (0.25 + lip_fwd, 3.35), (0.75, 2.95), (0.65, 2.1),
                    (1.0, 1.15), (2.4, 0.75), (4.6, 0.85), (6.2, 1.2)])
    neck = [(6.0, 0.6), (5.6, -1.2), (5.5, -3.6), (9.8, -3.6)]
    sil = np.vstack([smooth(upper, 120, s=0.0), lower_face, neck])
    ax.add_patch(Polygon(sil, closed=True, fc=SKIN, ec="none", zorder=1))
    prof = np.vstack([smooth(upper, 120, s=0.0)[3:], lower_face, neck[:3]])   # outline only along the face and neck
    ax.plot(prof[:, 0], prof[:, 1], color=SKIN_EDGE, lw=1.4, zorder=1.5)

    # ---- nasal cavity (air, but closed off by the raised velum for these oral vowels)
    nasal = smooth([(1.1, 5.55), (2.6, 6.25), (5.0, 6.45), (7.2, 6.35), (8.1, 5.9), (7.6, 5.55), (5.2, 5.75), (2.6, 5.7)],
                   120, closed=True)
    ax.add_patch(Polygon(nasal, closed=True, fc="#E6EEF5", ec=AIR_EDGE, lw=0.8, ls="--", zorder=2))

    # ---- the vocal-tract air space: mouth + pharynx, down to the glottis
    pal = smooth([(1.75, 4.35), (2.3, 4.85), (3.5, 5.45), (4.9, 5.6), (6.0, 5.45)], 80)           # hard palate (underside)
    velum = smooth([(6.0, 5.45), (6.7, 5.25), (7.25, 4.75), (7.35, 4.35)], 40)                  # soft palate, raised
    back_wall = smooth([(8.05, 5.6), (8.25, 4.0), (8.25, 1.5), (8.05, -0.8), (7.75, -2.3)], 80)
    air = np.vstack([[(0.9, 4.1)], pal, velum, [(7.45, 5.45), (8.05, 5.6)], back_wall,
                     [(7.0, -2.45), (6.5, -1.0), (6.3, 0.6)], J([(5.0, 1.4), (2.6, 1.5), (1.4, 2.9), (0.9, 3.9)])])
    ax.add_patch(Polygon(air, closed=True, fc=AIR, ec=AIR_EDGE, lw=1.0, zorder=3))

    # ---- palate bone, velum tissue
    pal_bone = np.vstack([pal, smooth([(6.0, 5.45), (4.9, 5.85), (3.4, 5.7), (2.2, 5.15), (1.75, 4.35)], 60)])
    ax.add_patch(Polygon(pal_bone, closed=True, fc=BONE, ec=BONE_EDGE, lw=1.2, zorder=6))
    vel = np.vstack([velum, smooth([(7.35, 4.35), (7.55, 4.9), (7.45, 5.45), (6.6, 5.7), (6.0, 5.45)], 40)])
    ax.add_patch(Polygon(vel, closed=True, fc=SKIN, ec=SKIN_EDGE, lw=1.2, zorder=6))

    # ---- teeth
    ax.add_patch(Polygon([(1.55, 4.55), (1.95, 4.55), (1.85, 3.95), (1.6, 3.9)], closed=True, fc=TOOTH, ec=BONE_EDGE, lw=1, zorder=7))
    ax.add_patch(Polygon(J([(1.55, 3.85), (1.8, 3.9), (1.95, 3.25), (1.6, 3.25)]), closed=True, fc=TOOTH, ec=BONE_EDGE, lw=1, zorder=7))

    # ---- tongue: tip behind the lower teeth, dorsum peak, root in front of the pharynx wall, floor on the jaw
    tip = J([(2.05, 3.3)])[0]
    px, py = p["peak"]
    front = ((tip[0] + px) / 2 - 0.15, (tip[1] + py) / 2 + 0.35)
    back = (min(p["root"] + 0.35, 7.6), py - 0.9 if key != "a" else 2.4)
    tongue_top = [tip, front, (px, py), (px + 1.0, py - 0.25), back, (p["root"] + 0.05, 1.3), (p["root"] - 0.55, 0.55)]
    floor = J([(4.6, 1.25), (3.0, 1.35), (2.0, 2.4)])     # below the air floor, so no air shows under the tongue
    tongue = np.vstack([smooth(tongue_top, 160), floor, [tip]])
    ax.add_patch(Polygon(tongue, closed=True, fc=TONGUE, ec=TONGUE_EDGE, lw=1.4, zorder=5))

    # ---- lips (side view): spread = thin and drawn back; rounded = pushed forward
    ul = smooth([(1.5, 4.55), (0.6 + lip_fwd, 4.6), (0.1 + lip_fwd, 4.3), (0.55 + lip_fwd, 4.05), (1.55, 4.1)], 50)
    ll = J(smooth([(1.55, 3.95), (0.5 + lip_fwd, 3.95), (0.2 + lip_fwd, 3.55), (0.7 + lip_fwd, 3.25), (1.55, 3.3)], 50))
    for L in (ul, ll):
        ax.add_patch(Polygon(L, closed=True, fc="#D9867B", ec="#9E5148", lw=1.1, zorder=8))

    # ---- glottis (vocal folds) at the bottom of the tube
    ax.plot([7.0, 7.45], [-2.35, -2.25], color=INK, lw=2.2, zorder=8, solid_capstyle="round")
    ax.plot([7.75, 7.5], [-2.2, -2.25], color=INK, lw=2.2, zorder=8, solid_capstyle="round")

    # ---- mark the narrowest point between tongue and palate / pharynx wall
    roof = np.vstack([pal, velum, back_wall])
    tt = smooth(tongue_top, 400)
    dmin, best = 1e9, None
    for q in tt:
        d = np.min(np.hypot(roof[:, 0] - q[0], roof[:, 1] - q[1]))
        if d < dmin and q[0] > 2.3:
            dmin, best = d, q
    k = np.argmin(np.hypot(roof[:, 0] - best[0], roof[:, 1] - best[1]))
    mid = (best + roof[k]) / 2
    ax.add_patch(Ellipse(mid, 0.75, 0.75, fc="none", ec=RED, lw=1.8, zorder=9))

    # ---- front view of the lips (inset, top left)
    cx, cy = 0.0, 8.3
    ax.add_patch(Ellipse((cx, cy), 2.5, 1.55, fc=SKIN, ec=SKIN_EDGE, lw=1, zorder=10))
    w, h = p["lip_w"], p["lip_h"]
    ax.add_patch(Ellipse((cx, cy), w + 0.45, h + 0.5, fc="#D9867B", ec="#9E5148", lw=1, zorder=11))
    ax.add_patch(Ellipse((cx, cy), w, h, fc="#4A2A2A", ec="none", zorder=12))
    ax.text(cx, cy - 1.05, "lips, front", ha="center", va="top", fontsize=6.5, color=MUTED, zorder=12)

    if labels:
        lab = dict(fontsize=7, color=INK, zorder=13)
        ax.text(5.6, 6.85, "nasal cavity", ha="center", **lab, alpha=0.8)
        ax.text(4.9, 6.02, "palate", ha="center", **lab)
        ax.annotate("velum (raised)", (7.0, 5.0), xytext=(8.0, 7.4), ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), **lab)
        ax.text(4.3, 2.35, "tongue", ha="center", **lab)
        ax.annotate("pharynx\n(throat)", (7.7, 0.8), xytext=(9.2, 1.6), ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), **lab)
        ax.annotate("vocal folds", (7.35, -2.3), xytext=(6.3, -3.25), ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), **lab)
        ax.annotate("narrowest point", tuple(mid + [-0.25, 0.3]), xytext=(2.7, 7.75), ha="center", fontsize=6.8, color=RED, zorder=13, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    fig.savefig(os.path.join(OUT, f"07_gallery_sketch_{key}.png"), facecolor="white")
    plt.close(fig)
    print("sketch", key, "narrowest gap %.2f" % dmin)


if __name__ == "__main__":
    for k in V:
        draw(k, labels=(k == "i"))
