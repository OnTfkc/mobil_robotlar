import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import math

# ============================================================
# RAW PATH (KNN-RRT ÇIKIŞI)
# ============================================================
# KNN-RRT algoritması sonucunda elde edildiği varsayılan
# düzensiz (raw) yol noktaları
raw_path = np.array([
    [0.5, 0.5],
    [1.2, 1.0],
    [1.8, 1.9],
    [2.5, 2.4],
    [3.0, 3.3],
    [3.1, 4.2],
    [3.5, 5.1],
    [4.2, 5.8],
    [5.0, 6.6],
    [6.0, 7.4],
    [7.0, 8.2],
    [8.2, 8.9],
    [9.5, 9.5]
])

# ============================================================
# BEZIER CURVE SMOOTHING
# ============================================================
# Bezier eğrisi kullanarak tüm yolu tek bir düzgün eğri
# haline getiren fonksiyon
def bezier_curve(points, n=200):
    # Parametre (0-1 arası)
    t = np.linspace(0, 1, n)

    # Eğri noktalarının tutulacağı dizi
    curve = np.zeros((n, 2))

    # Bezier eğrisinin derecesi
    deg = len(points) - 1

    # Bernstein polinomları kullanılarak Bezier hesabı
    for i, p in enumerate(points):
        curve += (
            math.comb(deg, i)
            * (t**i)[:, None]
            * ((1 - t)**(deg - i))[:, None]
            * p
        )
    return curve

# Bezier ile yumuşatılmış yol
bezier_path = bezier_curve(raw_path)

# ============================================================
# OBSTACLE REPULSION (ENGELDEN İTME)
# ============================================================
# B-spline uygulanmadan önce yol noktalarının
# engellere yaklaşması durumunda dışarı doğru itilmesi
def repel_from_obstacles(path, obstacles, safe_dist=1.2, strength=0.4):
    new_path = path.copy()

    for i, p in enumerate(new_path):
        for ox, oy, r in obstacles:
            # Engel merkezinden noktaya vektör
            vec = p - np.array([ox, oy])

            # Engel merkezine uzaklık
            dist = np.linalg.norm(vec)

            # Güvenli mesafeden daha yakınsa itme uygula
            if dist < r + safe_dist:
                direction = vec / (dist + 1e-6)  # normalize yön
                new_path[i] += strength * (r + safe_dist - dist) * direction

    return new_path

# ============================================================
# B-SPLINE SMOOTHING
# ============================================================
# Ortamda bulunan dairesel engeller (x, y, yarıçap)
OBSTACLES = [
    (4, 4, 1.0),
    (6, 6, 1.0),
    (5, 2, 0.8)
]

# Engel farkındalıklı (repulsion uygulanmış) yol
safe_path = repel_from_obstacles(raw_path, OBSTACLES)

# B-spline için x ve y koordinatları
x, y = safe_path[:, 0], safe_path[:, 1]

# splprep ile parametrik B-spline oluşturma
# s parametresi: yumuşatma miktarı
tck, _ = splprep([x, y], s=0.3)

# Parametre aralığı
u = np.linspace(0, 1, 200)

# B-spline eğrisinden noktaları elde etme
bx, by = splev(u, tck)
bspline_path = np.vstack((bx, by)).T

# ============================================================
# PLOTTING (GÖRSELLEŞTİRME)
# ============================================================
plt.figure(figsize=(7, 7))
plt.title("KNN-RRT Yol Planlama ve Smooth Path Karşılaştırması")

# Engelleri çiz (gri yarı saydam daireler)
for ox, oy, r in OBSTACLES:
    plt.gca().add_patch(
        plt.Circle((ox, oy), r, color="gray", alpha=0.4)
    )

# KNN-RRT (Raw) yol
plt.plot(
    raw_path[:, 0],
    raw_path[:, 1],
    "o--",
    color="tab:blue",
    label="KNN-RRT (Raw)",
    linewidth=2,
    markersize=4
)

# Bezier ile yumuşatılmış yol
plt.plot(
    bezier_path[:, 0],
    bezier_path[:, 1],
    color="tab:orange",
    label="Bezier Smooth",
    linewidth=2
)

# Engel farkındalıklı B-spline yol
plt.plot(
    bspline_path[:, 0],
    bspline_path[:, 1],
    color="tab:green",
    label="B-Spline Smooth",
    linewidth=2
)

# Başlangıç noktası
plt.scatter(
    raw_path[0, 0],
    raw_path[0, 1],
    color="green",
    s=60,
    label="Start",
    zorder=5
)

# Hedef noktası
plt.scatter(
    raw_path[-1, 0],
    raw_path[-1, 1],
    color="red",
    s=60,
    label="Goal",
    zorder=5
)

# Grafik ayarları
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.axis("equal")

plt.show()
