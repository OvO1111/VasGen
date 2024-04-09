import os, sys
sys.path.append(os.path.join(os.environ.get("nas"), "code/vessel.da/"))

import numpy as np
from numpy.random import randn
from numpy.linalg import norm, inv

import matplotlib.pyplot as plt
from scipy import ndimage
import SimpleITK as sitk
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import torch

 
class AutoRegression(torch.nn.Module):
    def __init__(self, n, r, alpha) -> None:
        super(AutoRegression, self).__init__()
        self.Y = torch.nn.Parameter(torch.randn(n, 3, dtype=torch.float32), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(n, 1, dtype=torch.float32), requires_grad=False) 
        self.gamma = torch.nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        self.I = torch.eye(n, n)
        self.r = torch.from_numpy(r).to(torch.float32)
        self.alpha = torch.from_numpy(alpha).to(torch.float32)
                
    def forward(self):
        # Y = self.Y * self.gamma
        X = (self.beta * self.gamma @ self.alpha.T + self.Y @ (torch.eye(3, 3) - self.alpha @ self.alpha.T)).T
        r_sq = self.r ** 2
        r_inv = r_sq.T / torch.linalg.norm(r_sq.flatten()) ** 2
        L = (R ** 2 * self.alpha @ r_inv + X @ (self.I - r_sq @ r_inv))
        L_norm = torch.linalg.norm(L, dim=0)
        return L, (L_norm - 1).abs().max()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self.zorder = kwargs.get("zorder", -1)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        return np.min(zs) if self.zorder == -1 else self.zorder


def get_murray_child_r(r_parent, n_child, tau=3, equals=True, props=None, **_):
    if not equals:
        r_children = np.abs(randn(n_child))
    else:
        r_children = np.ones(n_child, dtype=np.float32)
    if props is not None:
        r_children = np.array(props)
    r_children = (r_children * r_parent ** tau / r_children.sum()) ** (1 / tau)
    return r_children[:, None]


def make_orthogonal(dim, fixed_start=None):
    if fixed_start is not None:
        start = fixed_start
    else:
        start = randn(dim, dim)
        
    betas = []
    for ui in range(dim):
        alpha = start[:, ui: ui + 1].copy()
        for beta in betas:
            alpha -= start[:, ui: ui + 1].T.dot(beta)[0, 0] / beta.T.dot(beta)[0, 0] * beta
        betas.append(alpha)
    
    betas = np.hstack(betas)
    ortho = betas / norm(betas, axis=0)
    return ortho


def get_children_bif(alpha, r, n, gamma=1, **kwargs):
    beta = kwargs.get("beta", np.ones((n, 1)))
    Y = kwargs.get("Y", randn(n, 3) * gamma)
    X = (beta @ alpha.T + Y @ (np.eye(3, 3) - alpha @ alpha.T)).T
    r_sq = r ** 2
    r_inv = r_sq.T / norm(r_sq.flatten()) ** 2
    L = (R ** 2 * alpha @ r_inv + X @ (np.eye(n, n) - r_sq @ r_inv))
    IL = np.diag(norm(L, axis=0))
    L = L @ inv(IL)
    return r, L


def solve_children_bif(alpha, r, n, **kwargs):
    itr, lr, loss_min, max_step = 0, 3e-4, 1, 1e5
    current_best = None
    
    model = AutoRegression(n=n, r=r, alpha=alpha)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    while True:
        L, loss = model()
        if loss_min < 1e-8 or itr > max_step: 
            break
        if torch.isnan(loss).item():
            break
        current_best = current_best if loss >= loss_min else L
        loss_min = min(loss.item(), loss_min)
        optim.zero_grad()
        loss.backward()
        optim.step()
        itr += 1
        print(f"itr{itr} current_loss {loss.item():.8f} best loss {loss_min:.8f}", end="\r")
        
        for param_group in optim.param_groups:
            param_group["lr"] = lr * (1 - itr / max_step) ** 0.5
            
    print(f"best loss: {loss_min:.8f}")
    if loss_min > 1e-6:
        raise RuntimeError("loss did not converge")

    L = current_best.cpu().data.numpy()
    IL = np.diag(norm(L, axis=0))
    L = L @ inv(IL)
    if kwargs.get("out", None) is not None:
        kwargs["out"][tuple(r[:, 0].tolist())] = L
    return L


def _draw_different_nbranch(ax, r, L):
    
    scale = 30
    origin = np.array([30, 70, 30]) + alpha.flatten() * scale
    for v in range(len(r)):
        a = Arrow3D([origin[0], origin[0] + scale * L[:, v][0]], 
                    [origin[1], origin[1] + scale * L[:, v][1]],
                    [origin[2], origin[2] + scale * L[:, v][2]], 
                    mutation_scale=10, lw=r[v], arrowstyle="-|>", color="r")
        ax.add_artist(a)
    """for v in range(n):
        ax.quiver(*origin, *(scale * L[:, v]), color='r', lw=r[v])"""
        
    ax.scatter(*origin, s=.5, c='k')
    ax.plot(*list(zip(origin, origin - scale * alpha.flatten())),
            color="k", lw=R)
    ax.plot(*list(zip(origin, origin + scale * alpha.flatten())),
            linestyle="--", color="k")
            
    ax.set_xlim((30, 70))
    ax.set_ylim((30, 70))
    ax.set_zlim((30, 70))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 0))
    ax.yaxis.set_pane_color((0.9, 0.9, 0.9, 0))
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 0))
    
    thetas = np.round(np.arccos(L.T @ alpha) * 180 / np.pi, 1)
    gammas = np.round(np.arccos(L.T @ L) * 180 / np.pi, 1)
    print("thetas:\n", thetas)
    print("gammas:\n", gammas)


def _draw_different_rratio(xs, ys, ax):
    colors = ['b', 'r']
    labels = [r"$\theta_0$", r"$\gamma_{01}$"]
    save_arr = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        ix = np.argsort(x)
        x_ = x[ix]
        y_ = y[ix]
        save_arr.append(np.array([x_, y_]))
        ax.scatter(x_, y_, c=colors[i])
        ax.plot(x_, y_, c=colors[i], lw=1)
        ax.plot([0, min(x)], [90, max(y)], c=colors[i], ls="--", lw=1, label=labels[i])
        
    np.savez("v3/imgs/rratio.npz", theta=save_arr[0], gamma=save_arr[1])
    ax.set_xlabel("ratio between radii")
    ax.set_ylabel("angle")
    ax.legend()
    
    
def _draw_different_rratio_3d(xs, ys, ax):
    colors = ['b', 'r']
    labels = [r"$\theta_0$", r"$\gamma_{01}$"]
    save_arr = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        ix = np.argsort(x, axis=1)
        x_ = x[ix]
        y_ = y[0, ix]
        save_arr.append(np.array([x_, y_]))
        ax.plot_surface(x_[0], x_[1], y_, c=colors[i], antialiased=True)
        # ax.plot([0, min(x)], [90, max(y)], c=colors[i], ls="--", lw=1, label=labels[i])
        
    np.savez("v3/imgs/rratio_3d.npz", theta=save_arr[0], gamma=save_arr[1])
    ax.set_xlabel("ratio between radii")
    ax.set_ylabel("angle")
    ax.legend()
    

def draw_different_nbranch(alpha, R, ns, **rs_kwargs):
    import multiprocessing as mp
    
    fig = plt.figure(figsize=(15, 8))
    for iin, n in enumerate(ns):
        ax = fig.add_subplot(2, 2, iin+1, projection="3d")
        r = get_murray_child_r(R, n, **rs_kwargs)
        L = solve_children_bif(alpha, r, n)
        _draw_different_nbranch(ax, r, L)
            
    plt.draw()
    plt.savefig("nbranch.png", dpi=300, bbox_inches='tight', transparent=True)
    
    
def draw_different_rratio(alpha, R):
    import multiprocessing as mp
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    out = mp.Manager().dict()
    
    nproc = 32
    pool = []
    xys = [[], []]
    r2s = np.linspace(0, 1, nproc) ** 3
    
    for proc in range(nproc):
        r = get_murray_child_r(R, 2, props=(1, r2s[proc]))
        pool.append(mp.Process(target=solve_children_bif, args=(alpha, r, 2), kwargs=dict(out=out)))
        pool[-1].start()
        
    for proc in pool:
        proc.join()
        
    ratios = np.array(list(_[1] / _[0] for _ in out.keys()))
    Ls = np.array(list(np.arccos(_.T @ _)[1, 0] * 180 / np.pi for _ in out.values()))
    xys[0].append(ratios)
    xys[1].append(Ls)
    
    pool = []
    out = mp.Manager().dict()
    for proc in range(nproc):
        r = get_murray_child_r(R, 2, props=(1 - r2s[proc], r2s[proc]))
        pool.append(mp.Process(target=solve_children_bif, args=(alpha, r, 2), kwargs=dict(out=out)))
        pool[-1].start()
        
    for proc in pool:
        proc.join()
    
    ratios = np.array(list(_[1] / R for _ in out.keys()))
    Ls = np.array(list(np.arccos(_.T @ alpha)[1, 0] * 180 / np.pi for _ in out.values()))
    xys[0].append(ratios)
    xys[1].append(Ls)
    _draw_different_rratio(xys[0], xys[1], ax)
    ax.set_ylim((0, 90))
    ax.set_xlim((0, 1))
        
    plt.draw()
    plt.savefig("rratio.png", dpi=300, bbox_inches='tight')
    
    
def draw_different_rratio_3d(alpha, R):
    import multiprocessing as mp
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    out = mp.Manager().dict()
    
    nproc = 8
    pool = []
    xys = [[], []]
    r2s = np.linspace(0, 1, nproc) ** 3
    r3s = r2s
    
    for proc in range(nproc):
        for qroc in range(nproc):
            r = get_murray_child_r(R, 3, props=(1, r2s[proc], r3s[qroc]))
            pool.append(mp.Process(target=solve_children_bif, args=(alpha, r, 3), kwargs=dict(out=out)))
            pool[-1].start()
        
    for proc in pool:
        proc.join()
        
    ratios = np.array([list(_[1] / _[0] for _ in out.keys()),
                       list(_[2] / _[0] for _ in out.keys())])
    Ls = np.array([list(np.arccos(_.T @ alpha)[1, 0] * 180 / np.pi for _ in out.values()),
                   list(np.arccos(_.T @ alpha)[2, 0] * 180 / np.pi for _ in out.values())])
    xys[0].append(ratios)
    xys[1].append(Ls)
    
    _draw_different_rratio_3d(xys[0], xys[1], ax)
    ax.set_zlim((0, 90))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
        
    plt.draw()
    plt.savefig("v3/imgs/rratio_3d.png", dpi=300, bbox_inches='tight')
            

if __name__ == "__main__":
    R = 5
    ns = [2, 3, 4, 5]
    
    torch.manual_seed(1024)
    torch.use_deterministic_algorithms(True)
    np.random.seed(1024)
    alpha = np.array([1,-1,1], dtype=np.float32)[:, None]
    alpha /= norm(alpha)

    # draw_different_rratio(alpha, R)
    draw_different_nbranch(alpha, R, ns, equals=True)

    