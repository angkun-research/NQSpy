import matplotlib.pyplot as plt
import numpy as np

def generate_sites(nx, ny, a=1.0):
    A = []  # sublattice A positions: (a/2, 0) offset
    B = []  # sublattice B positions: (0, a/2) offset
    for i in range(nx):
        for j in range(ny):
            A.append((i * a + a/2, j * a + 0.0))
            B.append((i * a + 0.0, j * a + a/2))
    return np.array(A), np.array(B)

def plot_checkerboard(nx=6, ny=6, a=1.0, show_grid=True, savepath=None):
    A, B = generate_sites(nx, ny, a)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    # dashed square lattice (primitive vectors grid)
    if show_grid:
        xs = np.arange(0, (nx+1) * a, a)
        ys = np.arange(0, (ny+1) * a, a)
        for x in xs:
            ax.plot([x, x], [0, ny * a], color='0.3', linestyle='--', linewidth=0.8, alpha=0.6)
        for y in ys:
            ax.plot([0, nx * a], [y, y], color='0.3', linestyle='--', linewidth=0.8, alpha=0.6)

    # plot sublattices
    ax.scatter(A[:,0], A[:,1], c='C1', marker='o', s=70, label='A sublattice', zorder=3)
    ax.scatter(B[:,0], B[:,1], c='C0', marker='s', s=70, label='B sublattice', zorder=3)

    # solid bonds: within each unit cell the four sites are fully connected
    # four sites for cell (i,j): pA=(i+a/2, j), pB=(i, j+a/2), pA_up=(i+a/2, j+a), pB_right=(i+a, j+a/2)+(a,0)=(i+a, j+a/2)
    for i in range(nx):
        for j in range(ny):
            pA = np.array((i*a + a/2, j*a))
            pB = np.array((i*a, j*a + a/2))
            pA_up = np.array((i*a + a/2, j*a + a))
            pB_right = np.array((i*a + a, j*a + a/2))
            quartet = [pA, pB, pA_up, pB_right]
            # connect all pairs among the quartet (solid lines)
            for m in range(4):
                for n in range(m+1, 4):
                    xs = [quartet[m][0], quartet[n][0]]
                    ys = [quartet[m][1], quartet[n][1]]
                    ax.plot(xs, ys, color='k', linestyle='-', linewidth=1.2, zorder=1)

    ax.set_xlim(-0.5*a, nx*a + 0.5*a)
    ax.set_ylim(-0.5*a, ny*a + 0.5*a)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right')
    #ax.set_title('Checkerboard lattice — A (C1) and B (C0). Grid dashed, bonds solid')

    for spine in ax.spines.values():
        spine.set_visible(False)
        
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # simple demo
    plot_checkerboard(nx=6, ny=6, a=1.0, savepath='/Users/angkunwu/Desktop/checkerboard_lattice.pdf')