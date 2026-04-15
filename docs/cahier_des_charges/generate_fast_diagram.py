from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def add_box(ax, x, y, w, h, text, fontsize=11, facecolor="#f5f5f5"):
    rect = Rectangle((x, y), w, h, facecolor=facecolor, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def add_arrow(ax, start, end, text=None, text_offset=(0, 0)):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.2, color="black")
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=10)


fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

box_w = 2.2
box_h = 0.9
y_main = 4.2

main_functions = [
    (0.7, "Répondre au besoin\nde numérisation"),
    (3.3, "Scanner une\npièce"),
    (5.9, "Mesurer la\ngéométrie"),
    (8.5, "Traiter les\ndonnées"),
    (11.1, "Générer un\nmodèle 3D"),
]

for x, label in main_functions:
    add_box(ax, x, y_main, box_w, box_h, label)

for idx in range(len(main_functions) - 1):
    x1 = main_functions[idx][0] + box_w
    x2 = main_functions[idx + 1][0]
    add_arrow(ax, (x1, y_main + box_h / 2), (x2, y_main + box_h / 2))

ax.text(7.6, 5.45, "Comment ?", ha="center", va="center", fontsize=11)
ax.text(1.8, 5.45, "Pourquoi ?", ha="center", va="center", fontsize=11)

support_boxes = [
    (5.9, 2.4, "Capturer les images\navec une caméra"),
    (8.9, 2.4, "Filtrer et reconstruire\nle nuage de points"),
    (11.5, 2.4, "Exporter en\nSTL ou OBJ"),
    (3.3, 1.0, "Automatiser le cycle\net piloter le plateau"),
    (7.6, 1.0, "Informer l'utilisateur\nsur l'état du scan"),
]

for x, y, label in support_boxes:
    add_box(ax, x, y, box_w, box_h, label, fontsize=10, facecolor="#ffffff")
    add_arrow(ax, (x + box_w / 2, y + box_h), (x + box_w / 2, y_main), text_offset=(0, 0))

constraint_text = (
    "Contraintes : budget <= 300 CHF, sécurité laser, alimentation 230 V,\n"
    "encombrement compatible MakeLab, objets simples jusqu'à 150 x 150 x 150 mm"
)
add_box(ax, 2.1, 6.5, 9.8, 0.95, constraint_text, fontsize=10, facecolor="#eaf2f8")

ax.text(7.0, 7.75, "Diagramme FAST du système de scanner 3D", ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.savefig("fast_diagram.pdf", bbox_inches="tight")
