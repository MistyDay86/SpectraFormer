import os
import math
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def smooth_lorentzian(E, y, fwhm_eV):

    dE = np.mean(np.diff(E))
    gamma = fwhm_eV / 2   # half width at half maximum
    half_width = int(np.ceil(10 * gamma / dE))  # wider tail
    xk = np.arange(-half_width, half_width + 1) * dE
    kernel = gamma**2 / (xk**2 + gamma**2)
    kernel /= kernel.sum()
    y_pad = np.pad(y, half_width, mode="constant", constant_values=0.0)
    y_conv = np.convolve(y_pad, kernel, mode="same")

    return y_conv[half_width:-half_width]

def smooth_gaussian(f_sorted,w_sorted,sigma):
    # Frequency grid for the smooth curve
    f_min, f_max = f_sorted.min(), f_sorted.max()
    f_grid = np.linspace(f_min - 3*sigma, f_max + 3*sigma, 2000)

    # Build Gaussian kernels centered at each frequency
    # shape: (n_grid, n_peaks)
    diff = f_grid[:, None] - f_sorted[None, :]
    kernel = np.exp(-0.5 * (diff / sigma)**2)
    kernel /= np.sqrt(2*np.pi) * sigma  # normalize each Gaussian

    # Weighted sum over all peaks
    smoothed = kernel @ w_sorted  # shape: (n_grid,)

    # Optional: normalize so that area under curve = sum of weights
    df = f_grid[1] - f_grid[0]
    smoothed *= (np.sum(w_sorted) / np.trapezoid(smoothed, f_grid))
    return f_grid, smoothed

def read_xyz_trajectory(path):
    """
    Read a multi-frame XYZ (anime) file.

    Returns
    -------
    frames : list of dict
        Each dict:
          - 'symbols': list of element strings
          - 'coords' : (N,3) float array
          - 'comment': string
    """
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                break
            n = int(line)
            comment = f.readline().rstrip("\n")
            symbols = []
            coords = np.zeros((n, 3), float)
            for i in range(n):
                parts = f.readline().split()
                symbols.append(parts[0])
                coords[i] = list(map(float, parts[1:4]))
            frames.append({"symbols": symbols, "coords": coords, "comment": comment})
    return frames

#compute mode displacement with respect to a reference frame (e.g., initial frame)
def mode_disp(frames, frame0=0, frame1=-1):
    return frames[frame1]["coords"] - frames[frame0]["coords"]

#select buffer layer atoms based on z coordinate
def select_BL_mask(coords, z_threshold=17.0):
    return coords[:, 2] > z_threshold

def find_bonds(coords, dmin=1.2, dmax=1.7, subset_idx=None):
    """
    Find bonds as pairs of atoms with distances in [dmin, dmax].
    If subset_idx is given, only those atoms are used.
    """
    if subset_idx is None:
        idx = np.arange(len(coords), dtype=int)
        sub = coords
    else:
        idx = np.array(subset_idx, dtype=int)
        sub = coords[idx]

    n = len(idx)
    bonds = []
    for a in range(n):
        diff = sub[a+1:] - sub[a]
        dist = np.linalg.norm(diff, axis=1)
        partners = np.where((dist >= dmin) & (dist <= dmax))[0]
        for p in partners:
            bonds.append((idx[a], idx[a+1+p]))
    return np.array(bonds, int)

def bipartite_coloring(nnodes, edges):
    """
    2-color an undirected graph; returns (ok, colors),
    colors = 0/1 for each node.
    """
    adj = [[] for _ in range(nnodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    color = [None] * nnodes
    ok = True
    for start in range(nnodes):
        if color[start] is not None:
            continue
        color[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if color[v] is None:
                    color[v] = 1 - color[u]
                    dq.append(v)
                elif color[v] == color[u]:
                    ok = False
    return ok, np.array(color, int)

#classify graphene sublattices from mode displacements
def classify_graphene_sublattices_from_mode(disp_gr):
    norms = np.linalg.norm(disp_gr, axis=1) # compute norms of displacement vectors
    ref_index = int(np.argmax(norms)) # index of atom with largest displacement
    v0 = disp_gr[ref_index] # reference displacement vector
    dots = disp_gr @ v0 # dot products with reference vector
    labels_gr = (dots < 0).astype(int)  # 0 = aligned, 1 = anti-aligned
    e0 = disp_gr[labels_gr == 0].mean(axis=0) 
    e1 = disp_gr[labels_gr == 1].mean(axis=0)
    return labels_gr, e0, e1, ref_index

def load_frequencies(freq_file):
    """
    Parse phonopy-style Gamma-point file:

      # k point ...
         Mode, Frequency
            1   0.0000 cm^-1  ( 0.0000 THz )
            2   0.0000 cm^-1  ( 0.0000 THz )
            ...

    Returns dict {mode_index_1based: freq_cm^-1}.
    """
    freq_dict = {}
    pat = re.compile(r'^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?)\s+cm\^-1')
    with open(freq_file) as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            mode_idx = int(m.group(1))   # 1-based
            freq_cm  = float(m.group(2))
            freq_dict[mode_idx] = freq_cm
    return freq_dict

def get_mode_index_from_filename(fname, pattern=r"SiC\.anime(\d+)\.xyz"):
    m = re.search(pattern, fname)
    if not m:
        raise ValueError(f"Cannot extract mode index from {fname}")
    return int(m.group(1))   # keep this 1-based to match freq file

def compute_local_cosines(u_BL, e_ref):
    """
    Compute local cos(theta) between BL displacements and reference pattern.

    Parameters
    ----------
    u_BL : (N_BL, 3) array
        Displacements of BL atoms for one mode.
    e_ref : (N_BL, 3) array
        Reference/template displacement assigned to each BL atom
        (e.g. graphene-unitcell pattern per sublattice).

    Returns
    -------
    local_cos : (N_BL,) array
        cos(theta_i) for each BL atom. If either vector has zero norm,
        the cosine is set to 0.
    """
    u_BL  = np.asarray(u_BL, float)
    e_ref = np.asarray(e_ref, float)

    u_norm = np.linalg.norm(u_BL, axis=1)   # |u_i|
    e_norm = np.linalg.norm(e_ref, axis=1)  # |e_i|

    num = np.sum(u_BL * e_ref, axis=1)      # u_i · e_i

    with np.errstate(divide="ignore", invalid="ignore"):
        local_cos = np.where(
            (u_norm > 0) & (e_norm > 0),
            num / (u_norm * e_norm),
            0.0,
        )

    return local_cos

def read_group_indices(filename, one_based=True):
    """
    Read a 1-column text file of atom indices.

    Parameters
    ----------
    filename : str
        Path to text file.
    one_based : bool
        If True, convert from 1-based to 0-based indices.

    Returns
    -------
    idx : (M,) int array (0-based)
    """
    idx = np.loadtxt(filename, dtype=int)
    idx = np.atleast_1d(idx).ravel()
    if one_based:
        idx = idx - 1
    return idx

def sum_local_cosines_over_global_groups(res_mode, groups_global):
    """
    Sum / average local cosines over selected groups of atoms, defined
    in terms of *global* indices on the full SiC+BL system.

    Parameters
    ----------
    res_mode : dict
        Output of project_BL_mode_on_graphene_unitcell_bipartite.
        Must contain:
          'bl_indices'   : (N_BL,) global indices of BL atoms
          'local_cosines': (N_BL,) cos(theta_i) for BL atoms
    groups_global : dict
        Mapping name -> array of global indices (0-based).

    Returns
    -------
    out : dict
        name -> {
            'sum_cos'    : sum of local cos over BL atoms in group
            'mean_cos'   : average local cos (over BL atoms in group)
            'n_in_BL'    : number of atoms of the group that belong to BL
            'local_idx'  : BL-local indices used
        }
    """
    bl_indices   = res_mode["bl_indices"]      # global indices of BL atoms
    local_cos_BL = res_mode["local_cosines"]   # (N_BL,)

    # map global -> local BL index
    global_to_local = {g: i for i, g in enumerate(bl_indices)}

    out = {}
    for name, g_glob in groups_global.items():
        g_glob = np.asarray(g_glob, int)
        loc = [global_to_local[g] for g in g_glob if g in global_to_local]
        loc = np.array(loc, int)

        if len(loc) == 0:
            sum_cos  = 0.0
            mean_cos = 0.0
        else:
            vals = local_cos_BL[loc]
            sum_cos  = float(np.sum(vals))
            mean_cos = float(np.mean(vals))

        out[name] = dict(
            sum_cos=sum_cos,
            mean_cos=mean_cos,
            n_in_BL=len(loc),
            local_idx=loc,
        )
    return out

def decompose_overlap2_by_global_groups(res_mode, groups_global):
    """
    For ONE mode, decompose global overlap^2 into partial contributions
    per group using:
        d_i   = u_i · e_i
        N_g   = sum_{i in group} d_i
        I_g   = (N_g / D)^2
    where D^2 = sum_i |u_i|^2 * sum_i |e_i|^2.

    Parameters
    ----------
    res_mode : dict
        Output of project_BL_mode_on_graphene_unitcell_bipartite for one mode.
        Needs keys:
          'bl_indices' : (N_BL,) global indices of BL atoms
          'u_BL'       : (N_BL,3)
          'e_ref'      : (N_BL,3)
          'overlap2'   : float
    groups_global : dict
        name -> array of global indices (0-based).

    Returns
    -------
    out : dict
        name -> {
           'N_g'       : float, sum_i d_i over group
           'frac_num'  : N_g / N_tot
           'I_group'   : (N_g / D)^2  (partial overlap^2)
           'n_in_BL'   : number of atoms of this group that are in BL
        }
    """
    bl_indices = res_mode["bl_indices"]
    u_BL = res_mode["u_BL"]
    e_ref = res_mode["e_ref"]

    # per-atom dot products d_i
    pair_dot = np.sum(u_BL * e_ref, axis=1)

    U2 = np.sum(u_BL**2)
    E2 = np.sum(e_ref**2)
    D2 = U2 * E2
    D  = math.sqrt(D2) if D2 > 0 else 0.0

    N_tot = pair_dot.sum()

    # global->local map
    global_to_local = {g: i for i, g in enumerate(bl_indices)}

    out = {}
    for name, g_glob in groups_global.items():
        g_glob = np.asarray(g_glob, int)
        loc = [global_to_local[g] for g in g_glob if g in global_to_local]
        loc = np.array(loc, int)

        if len(loc) == 0 or D == 0.0:
            N_g = 0.0
            frac = 0.0 if N_tot != 0 else 0.0
            I_g = 0.0
        else:
            N_g = float(pair_dot[loc].sum())
            frac = N_g / N_tot if N_tot != 0 else 0.0
            I_g = (N_g / D)**2

        out[name] = dict(
            N_g=N_g,
            frac_num=frac,
            I_group=I_g,
            n_in_BL=len(loc),
        )

    return out

def project_BL_mode_on_graphene_unitcell_bipartite(
    sic_frames,
    gr_frames,
    z_threshold=17.0,
    frame0=0,
    frame1=-1,
    bond_dmin=1.3,
    bond_dmax=1.6,
):
    # --- BL bipartite splitting ---
    coords0 = sic_frames[frame0]["coords"]
    bl_mask = select_BL_mask(coords0, z_threshold=z_threshold)
    bl_indices = np.where(bl_mask)[0]

    bonds_BL_global = find_bonds(coords0, dmin=bond_dmin, dmax=bond_dmax, subset_idx=bl_indices)

    # build local index graph
    idx_to_local = {idx: i for i, idx in enumerate(bl_indices)}
    edges_local  = np.array([[idx_to_local[i], idx_to_local[j]] for i, j in bonds_BL_global], int)

    ok, colors_local = bipartite_coloring(len(bl_indices), edges_local)
    if not ok:
        raise RuntimeError("BL graph is not bipartite with given bond cutoffs.")

    # --- graphene template (from eigenmode) ---
    disp_gr = mode_disp(gr_frames, frame0, frame1)
    labels_gr, e0, e1, ref_idx = classify_graphene_sublattices_from_mode(disp_gr)
    # (we only need e0, e1)

    # --- BL displacements ---
    disp_sic = mode_disp(sic_frames, frame0, frame1)
    u_BL     = disp_sic[bl_indices]          # (N_BL,3)

    # build reference template per BL atom based on its color
    e_ref = np.zeros_like(u_BL)
    for i, c in enumerate(colors_local):
        e_ref[i] = e0 if c == 0 else e1

    # global normalized overlap
    num = np.sum(u_BL * e_ref)
    den = np.sqrt(np.sum(u_BL**2) * np.sum(e_ref**2))
    overlap = num / den if den > 0 else 0.0

    # local cosines
    u_norm = np.linalg.norm(u_BL, axis=1)
    e_norm = np.linalg.norm(e_ref, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_cos = np.where(
            (u_norm > 0) & (e_norm > 0),
            np.sum(u_BL * e_ref, axis=1) / (u_norm * e_norm),
            0.0,
        )

    return {
        "overlap": overlap,
        "overlap2": overlap**2,
        "bl_indices": bl_indices,
        "colors_local": colors_local,
        "bonds_BL_global": bonds_BL_global,
        "template_e0": e0,
        "template_e1": e1,
        "local_cosines": local_cos,
        # ADDED for group decomposition:
        "u_BL": u_BL,
        "e_ref": e_ref,
    }

def mode_atom_raman_contrib(res_mode, inplane_only=True, normalize=True):
    """
    Compute per-atom Raman-like contribution for ONE mode,
    according to the paper's I_{t',s'}(ω) idea.

    Parameters
    ----------
    res_mode : dict
        Output of project_BL_mode_on_graphene_unitcell_bipartite.
        Must contain:
          'u_BL'     : (N_BL, 3) BL displacements for this mode
          'overlap2' : scalar I_nu (global E2g-like intensity)
    inplane_only : bool
        If True, only use x,y components for atom weights (c' = x,y).
    normalize : bool
        If True, rescale so that sum_p I_{nu,p} = I_nu exactly.

    Returns
    -------
    I_atom : (N_BL,) array
        Per-BL-atom intensity I_{nu,p}.
        BL atoms are in the same local order as res_mode['bl_indices'].
    """
    u_BL = np.asarray(res_mode["u_BL"], float)   # (N_BL,3)
    I_mode = float(res_mode["overlap2"])         # global I_nu

    # in-plane components for c' = x,y
    if inplane_only:
        disp = u_BL[:, :2]   # (N_BL,2)
    else:
        disp = u_BL

    # W_{nu,p} = sum_{c'} |u_{p,c'}|^2
    W_atom = np.sum(disp**2, axis=1)   # (N_BL,)

    if not normalize:
        # raw product (careful: scales like amplitude^4 if anime amplitude changes)
        return I_mode * W_atom

    W_tot = float(W_atom.sum())
    if W_tot <= 0.0 or I_mode == 0.0:
        return np.zeros_like(W_atom)

    # I_{nu,p} = I_nu * W_p / sum_q W_q
    I_atom = I_mode * W_atom / W_tot
    return I_atom

def plot_BL_atom_scalar(coords0, indices, bl_indices, values,
                        title="", cmap="viridis",
                        vmin=None, vmax=None,
                        logscale=False):
    """
    Color-code a per-BL-atom scalar on an (x,y) scatter of the BL.

    Parameters
    ----------
    coords0 : (N_tot, 3) array
        Equilibrium coordinates of the full SiC+BL system.
    bl_indices : (N_BL,) array
        Global indices of BL atoms.
    values : (N_BL,) array
        Per-BL-atom scalar to plot (e.g. I_B_atom or I_L_atom).
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color scale limits. If None, use min/max of values.
    logscale : bool
        If True, plot log10(values) (useful if the range spans many orders).
    """
    coords_BL = coords0[indices]  # (N_BL,3)
    full_bl = coords0[bl_indices] 
    x = coords_BL[:, 0]
    y = coords_BL[:, 1]

    vals = np.asarray(values, float)
    if logscale:
        # avoid log of zero
        eps = vals[vals > 0].min() * 1e-3 if np.any(vals > 0) else 1e-12
        vals_plot = np.log10(np.clip(vals, eps, None))
        cbar_label = r"local contribution ($\log_{10}$)"
    else:
        vals_plot = vals
        cbar_label = "value"

    if vmin is None:
        vmin = np.min(vals_plot)
    if vmax is None:
        vmax = np.max(vals_plot)

    fig, ax = plt.subplots(figsize=(6,4))
    sc = ax.scatter(x, y,
                    c=vals_plot,
                    s=80,
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    edgecolors="none")
    
    plt.scatter(
        full_bl[:,0],
        full_bl[:,1],
        c='tab:grey',
        cmap="bwr",
        s=70,
        # edgecolors="k",
        linewidths=0.3,
        alpha=0.3
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    
def subset_BL_for_group(bl_indices, values, ids_group):
    """
    Restrict BL data to a subset of atoms given by global indices.

    Parameters
    ----------
    bl_indices : (N_BL,) array
        Global indices of BL atoms (res["bl_indices"]).
    values : (N_BL,) array
        Per-BL-atom values (e.g. I_B_atom, I_L_atom).
    ids_group : 1D array-like of int
        Global indices of atoms belonging to the group (0-based!).

    Returns
    -------
    bl_indices_sub : (N_sub,) array
        Global indices of BL atoms in the group.
    values_sub : (N_sub,) array
        Values for those atoms, same order.
    """
    bl_indices = np.asarray(bl_indices, int)
    values     = np.asarray(values, float)
    ids_group  = np.asarray(ids_group, int)

    # global -> local BL index
    global_to_local = {g: i for i, g in enumerate(bl_indices)}

    loc = [global_to_local[g] for g in ids_group if g in global_to_local]
    if not loc:
        return np.array([], int), np.array([], float)

    loc = np.array(loc, int)
    return bl_indices[loc], values[loc]

def build_base_xy(coords, cell):
    """
    Take PDB xy coords + cell, convert to fractional, wrap into [0,1),
    and return base_xy (atoms in one unit cell) and lattice vectors a_vec, b_vec.
    """
    a = cell["a"]
    b = cell["b"]
    gamma_deg = cell["gamma"]

    if a is None or b is None or gamma_deg is None:
        raise RuntimeError("CRYST1 record not found or incomplete in PDB.")

    gamma_rad = np.deg2rad(gamma_deg)
    a_vec = np.array([a, 0.0])
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad)])

    # Matrix whose columns are a_vec and b_vec
    M = np.column_stack((a_vec, b_vec))  # shape (2,2)

    # coords^T = M * frac^T  =>  frac^T = M^{-1} coords^T
    frac = np.linalg.solve(M, coords.T).T  # (N,2)
    frac_wrapped = frac - np.floor(frac)   # wrap into [0,1)

    # Back to Cartesian: atoms inside a single cell anchored at origin
    base_xy = frac_wrapped @ M.T
    return base_xy, a_vec, b_vec

def tile_xy(base_xy, a_vec, b_vec, nrep_a, nrep_b):
    """
    Tile base_xy by integer combinations of a_vec and b_vec.

    Returns
    -------
    all_xy : (N_tiles*N_atoms, 2)
    """
    all_xy = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            shift = ia * a_vec + ib * b_vec
            all_xy.append(base_xy + shift)
    return np.vstack(all_xy)

def read_pdb_xyz_cell(pdb_file):
    """
    Read x,y,z coordinates from a PDB file and the CRYST1 cell.

    Returns
    -------
    coords : (N, 3) array
        x, y, z positions of atoms
    cell : dict
        {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
    """
    a = b = c = alpha = beta = gamma = None
    coords = []

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("CRYST1"):
                a     = float(line[6:15])
                b     = float(line[15:24])
                c     = float(line[24:33])
                alpha = float(line[33:40])
                beta  = float(line[40:47])
                gamma = float(line[47:54])
            elif line.startswith(("ATOM  ", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    cell = dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    return np.array(coords, dtype=float), cell


def build_base_xyz(coords, cell):
    """
    Wrap x,y into [0,1) fractional cell (using a,b,gamma), keep z unchanged.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")

    a = cell["a"]; b = cell["b"]; gamma_deg = cell["gamma"]
    if a is None or b is None or gamma_deg is None:
        raise RuntimeError("CRYST1 record not found or incomplete in PDB.")

    gamma_rad = np.deg2rad(gamma_deg)
    a_vec = np.array([a, 0.0])
    b_vec = np.array([b*np.cos(gamma_rad), b*np.sin(gamma_rad)])
    M = np.column_stack((a_vec, b_vec))  # (2,2)

    xy = coords[:, :2]
    z  = coords[:, 2:3]

    frac = np.linalg.solve(M, xy.T).T
    frac_wrapped = frac - np.floor(frac)
    base_xy = frac_wrapped @ M.T

    base_xyz = np.hstack([base_xy, z])
    return base_xyz, a_vec, b_vec


def tile_xyz(base_xyz, a_vec, b_vec, nrep_a, nrep_b):
    """
    Replicate only in x,y; z stays the same.
    """
    base_xyz = np.asarray(base_xyz, dtype=float)
    all_xyz = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            shift_xy = ia * a_vec + ib * b_vec
            tmp = base_xyz.copy()
            tmp[:, :2] += shift_xy
            all_xyz.append(tmp)
    return np.vstack(all_xyz)

def read_pdb_xy_cell(pdb_file):
    """
    Read x,y coordinates from a PDB file and the CRYST1 cell.

    Returns
    -------
    coords : (N, 2) array
        x, y positions of atoms
    cell : dict
        {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
    """
    a = b = c = alpha = beta = gamma = None
    coords = []

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("CRYST1"):
                a     = float(line[6:15])
                b     = float(line[15:24])
                c     = float(line[24:33])
                alpha = float(line[33:40])
                beta  = float(line[40:47])
                gamma = float(line[47:54])
            elif line.startswith(("ATOM  ", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                coords.append([x, y])

    cell = dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    return np.array(coords), cell


def bond_segments_xy(all_xyz, r_cut=1.9, r_min=0.5, xlim=None, ylim=None, pad=0.0):
    """
    Find all pairs within r_cut (3D distance) and return xy-segments + lengths.
    Optionally keep only bonds whose endpoints fall inside (xlim, ylim) (with padding).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("This function needs scipy (scipy.spatial.cKDTree).") from e

    xyz = np.asarray(all_xyz, dtype=float)
    tree = cKDTree(xyz)

    pairs = np.array(list(tree.query_pairs(r_cut)), dtype=int)
    if pairs.size == 0:
        return np.empty((0, 2, 2)), np.empty((0,))

    p0 = xyz[pairs[:, 0]]
    p1 = xyz[pairs[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)

    mask = lengths >= r_min

    if xlim is not None:
        xmin, xmax = xlim
        mask &= (
            (p0[:, 0] >= xmin - pad) & (p0[:, 0] <= xmax + pad) &
            (p1[:, 0] >= xmin - pad) & (p1[:, 0] <= xmax + pad)
        )
    if ylim is not None:
        ymin, ymax = ylim
        mask &= (
            (p0[:, 1] >= ymin - pad) & (p0[:, 1] <= ymax + pad) &
            (p1[:, 1] >= ymin - pad) & (p1[:, 1] <= ymax + pad)
        )

    p0 = p0[mask]
    p1 = p1[mask]
    lengths = lengths[mask]

    segments = np.stack([p0[:, :2], p1[:, :2]], axis=1)  # (Nbonds, 2, 2)
    return segments, lengths

def read_poscar(path):
    with open(path, "r") as f:
        lines = [l.rstrip() for l in f if l.strip()]

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], float) * scale

    def is_number(s):
        try:
            float(s); return True
        except:
            return False

    # VASP5: symbols then counts; VASP4: counts directly
    tokens = lines[5].split()
    if all(is_number(t) for t in tokens):
        symbols = None
        counts = [int(float(t)) for t in tokens]
        idx = 6
    else:
        symbols = tokens
        counts = [int(float(t)) for t in lines[6].split()]
        idx = 7

    if lines[idx].lower().startswith("s"):  # Selective dynamics
        idx += 1

    coord_type = lines[idx].lower()
    direct = coord_type.startswith("d")
    cart = coord_type.startswith(("c", "k"))
    if not (direct or cart):
        raise ValueError(f"Unknown coordinate type: {lines[idx]}")
    idx += 1

    nat = sum(counts)
    raw = np.array([[float(x) for x in lines[idx+i].split()[:3]] for i in range(nat)], float)

    if direct:
        frac = raw
        cart_coords = frac @ lattice
    else:
        cart_coords = raw * scale  # VASP scale also applies to Cartesian coords
        frac = cart_coords @ np.linalg.inv(lattice)

    if symbols is None:
        # fallback names
        syms = []
        for i, c in enumerate(counts):
            syms += [f"X{i+1}"] * c
    else:
        syms = []
        for s, c in zip(symbols, counts):
            syms += [s] * c

    return lattice, np.array(syms), frac, cart_coords


def mic_deltas(dcart, lattice):
    inv_lat = np.linalg.inv(lattice)
    df = dcart @ inv_lat
    df -= np.round(df)          # wrap to [-0.5,0.5)
    return df @ lattice


def pair_distances_mic(posA, posB, lattice):
    d = posA[:, None, :] - posB[None, :, :]
    dmic = mic_deltas(d.reshape(-1, 3), lattice).reshape(d.shape)
    return np.linalg.norm(dmic, axis=2)


def wrap_xy_keep_z(cart_coords, lattice):
    inv_lat = np.linalg.inv(lattice)
    frac = cart_coords @ inv_lat
    frac[:, 0] -= np.floor(frac[:, 0])
    frac[:, 1] -= np.floor(frac[:, 1])
    # keep frac[:,2] unchanged
    return frac @ lattice


def tile_positions(base_cart, a_vec, b_vec, nrep_a, nrep_b):
    all_pos = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            shift = ia * a_vec + ib * b_vec
            all_pos.append(base_cart + shift)
    return np.vstack(all_pos)

def gaussian(x, center, fwhm):
    """Gaussian with unit area."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def lorentzian(x, center, fwhm):
    """Lorentzian with unit area."""
    gamma = fwhm / 2.0
    return (gamma / np.pi) / ((x - center)**2 + gamma**2)

def pseudo_voigt(x, center, fwhm, area, eta):
    """
    Pseudo-Voigt with given area.
    eta = gamma_ratio: fraction of Lorentzian component (0..1).
    """
    g = gaussian(x, center, fwhm)
    l = lorentzian(x, center, fwhm)
    pv_unit = eta * l + (1.0 - eta) * g   # still unit area
    return area * pv_unit                 # scale to requested area

def tile_positions_3d(base_xyz, a_vec, b_vec, nrep_a, nrep_b):
    """
    Tile base_xyz (N,3) by integer shifts of a_vec and b_vec.
    Robust to base_xyz being a tuple/list like (base_xyz, frac_xyz).
    """
    if isinstance(base_xyz, (tuple, list)):
        base_xyz = base_xyz[0]

    base_xyz = np.asarray(base_xyz, float)
    if base_xyz.ndim != 2 or base_xyz.shape[1] != 3:
        raise ValueError(f"base_xyz must be (N,3). Got {base_xyz.shape}")

    shifts = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            shifts.append(ia * a_vec + ib * b_vec)
    shifts = np.asarray(shifts, float)  # (Ntile,3)

    tiled = (base_xyz[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    return tiled, shifts.shape[0]

def multi_gaussian_bg(x, *params):
    """
    params = [A1, x1, s1, A2, x2, s2, ..., AN, xN, sN, bg]
    """
    n_peaks = (len(params) - 1) // 3
    bg = params[-1]
    y = np.full_like(x, bg, dtype=float)
    for i in range(n_peaks):
        A, x0, sigma = params[3*i:3*(i+1)]
        y += A * np.exp(- (x - x0)**2 / (2.0 * sigma**2))
    return y

def make_initial_guess(x, y, n_peaks):
    # Find candidate maxima
    # distance is just a heuristic to avoid too-close peaks
    distance = max(len(x) // (n_peaks * 5), 1)
    idx_peaks, _ = find_peaks(y, distance=distance)

    width_guess = (x.max() - x.min()) / (4.0 * n_peaks)
    bg = np.min(y)

    if len(idx_peaks) == 0:
        # Fallback: equally spaced centers
        centers = np.linspace(x.min(), x.max(), n_peaks + 2)[1:-1]
        amps = np.full(n_peaks, y.max())
    else:
        # Sort peaks by height (descending)
        idx_sorted = idx_peaks[np.argsort(y[idx_peaks])][::-1]

        if len(idx_sorted) >= n_peaks:
            # Use the n highest peaks
            idx_use = idx_sorted[:n_peaks]
            centers = x[idx_use]
            amps = y[idx_use]
        else:
            # Fewer peaks than requested: pad with rough guesses
            m = len(idx_sorted)
            centers_list = list(x[idx_sorted])
            amps_list = list(y[idx_sorted])

            extra = n_peaks - m
            extra_centers = np.linspace(x.min(), x.max(), extra + 2)[1:-1]
            centers_list.extend(extra_centers)
            amps_list.extend([y.max()] * extra)

            centers = np.array(centers_list)
            amps = np.array(amps_list)

    sigmas = np.full(n_peaks, width_guess)

    # Now all three have length n_peaks
    p0 = np.concatenate(
        [np.column_stack([amps, centers, sigmas]).ravel(), [bg]]
    )
    return p0

def fit_peaks(x, y, n_peaks):
    p0 = make_initial_guess(x, y, n_peaks)
    popt, pcov = curve_fit(multi_gaussian_bg, x, y, p0=p0,maxfev=10000)
    return popt, pcov

def split_components(x, params):
    n_peaks = (len(params) - 1) // 3
    bg = params[-1]
    comps = []
    for i in range(n_peaks):
        A, x0, sigma = params[3*i:3*(i+1)]
        comps.append(A * np.exp(- (x - x0)**2 / (2.0 * sigma**2)))
    return np.array(comps), bg


def read_anime_max_displacement(xyz_path):
    """
    Read an ALAMODE anime .xyz file and return the
    maximum |displacement| per atom for that mode.

    Returns
    -------
    max_disp : (N_atoms,) array
        Max displacement amplitude for each atom.
    mean_coords : (N_atoms, 3) array
        Time-averaged coordinates (equilibrium structure).
    """
    frames = []

    with open(xyz_path, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                break

            nat = int(line)              # number of atoms
            _comment = fh.readline()     # skip comment

            coords = []
            for _ in range(nat):
                parts = fh.readline().split()
                # ALAMODE: symbol, x, y, z
                x, y, z = map(float, parts[1:4])
                coords.append([x, y, z])

            frames.append(coords)

    frames = np.array(frames)           # (n_frames, N_atoms, 3)
    mean_coords = frames.mean(axis=0)   # (N_atoms, 3)

    # displacement of each frame from mean
    disps = frames - mean_coords        # (n_frames, N_atoms, 3)
    # norm of displacement for each atom & frame
    norms = np.linalg.norm(disps[:,:,0:2], axis=2)  # (n_frames, N_atoms)
    # max amplitude per atom over the pseudo-time
    max_disp = norms.max(axis=0)        # (N_atoms,)

    return max_disp, mean_coords

def mode_indices_in_window(freq_dict, fmin, fmax):
    """Return list of mode indices with frequency in [fmin, fmax]."""
    return [m for m, f in freq_dict.items() if fmin <= f <= fmax]


def build_weighted_displacement(
    pattern_dir,
    freq_dict,
    dos,
    fmin,
    fmax,
    file_prefix="SiC.anime",
    use_squared=False,
):
    """
    For a given frequency window [fmin, fmax], compute a DOS-weighted
    displacement contribution for each atom.

    Parameters
    ----------
    pattern_dir : str
        Directory containing SiC.animeXXXX.xyz files.
    freq_dict : dict[int, float]
        {mode_index: frequency_in_cm-1}.
    dos : (N_dos, 2) array
        dos[:,0] = frequency grid, dos[:,1] = DOS values.
    fmin, fmax : float
        Frequency window (cm^-1).
    file_prefix : str
        Prefix of anime files, default "SiC.anime".
    use_squared : bool
        If True, weight by |u|^2 instead of |u|.

    Returns
    -------
    contrib : (N_atoms,) array
        Frequency-window DOS-weighted displacement per atom, normalized to [0,1].
    """
    dos_freq = dos[:, 0]
    dos_vals = dos[:, 1].copy()
    dos_vals /= dos_vals.sum()  # normalized DOS like in your plot

    # First get one pattern to know N_atoms
    test_mode = next(iter(freq_dict.keys()))
    test_file = os.path.join(pattern_dir, f"{file_prefix}{test_mode:04d}.xyz")
    max_disp_test, _ = read_anime_max_displacement(test_file)
    nat = max_disp_test.size

    contrib = np.zeros(nat, dtype=float)

    modes = mode_indices_in_window(freq_dict, fmin, fmax)
    print(f"Using {len(modes)} modes between {fmin} and {fmax} cm^-1")

    for m in modes:
        freq = freq_dict[m]

        # DOS weight at this frequency (normalized)
        weight = np.interp(freq, dos_freq, dos_vals, left=0.0, right=0.0)
        if weight == 0.0:
            continue

        xyz_path = os.path.join(pattern_dir, f"{file_prefix}{m:04d}.xyz")
        max_disp, _ = read_anime_max_displacement(xyz_path)

        if use_squared:
            max_disp = max_disp**2

        contrib += weight * max_disp

    # Normalize to [0,1] for plotting
    if contrib.max() > 0:
        contrib /= contrib.max()

    return contrib

def get_bl_mask(pattern_dir, freq_dict, z_thresh=17.0, file_prefix="SiC.anime"):
    """
    Build a boolean mask selecting buffer-layer atoms (z > z_thresh)
    using the equilibrium coords extracted from one anime file.
    """
    # use the lowest mode index, but any existing mode is fine
    ref_mode = min(freq_dict.keys())
    ref_xyz = os.path.join(pattern_dir, f"{file_prefix}{ref_mode:04d}.xyz")
    _, mean_coords = read_anime_max_displacement(ref_xyz)  # shape (N_tot, 3)

    z = mean_coords[:, 2]
    mask_bl = z > z_thresh
    print("Total atoms:", mean_coords.shape[0])
    print("BL atoms (z > %.1f):" % z_thresh, mask_bl.sum())
    return mask_bl


def tile_values(base_values, nrep_a, nrep_b):
    """
    Repeat base_values for each tiling of the cell, in the same ordering
    as tile_xy.
    """
    tiled = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            tiled.append(base_values)
    return np.concatenate(tiled)


def extract_centers(popt):
    """
    popt = [A1, x1, s1, A2, x2, s2, ..., AN, xN, sN, bg]
    returns array of [x1, x2, ..., xN]
    """
    n_peaks = (len(popt) - 1) // 3
    centers = []
    for i in range(n_peaks):
        A, x0, sigma = popt[3*i:3*(i+1)]
        centers.append(x0)
    return np.array(centers)


# ---- helper to find nearest mode in the dict ------------------------------
def nearest_mode(freq_dict, target_freq, ignore_zero=True):
    """
    Returns (mode_index, mode_freq, delta) where delta = mode_freq - target_freq
    """
    items = freq_dict.items()
    if ignore_zero:
        items = ((k, v) for k, v in items if v != 0.0)

    mode_idx, mode_freq = min(items, key=lambda kv: abs(kv[1] - target_freq))
    delta = mode_freq - target_freq
    return mode_idx, mode_freq, delta

def get_disp_hist(coords0, displacements, n_bins=50):
    z = coords0[:, 2]

    # define bin edges
    z_min, z_max = z.min(), z.max()
    bins = np.linspace(z_min, z_max, n_bins + 1)

    # count atoms per bin
    counts, edges = np.histogram(z, bins=bins)

    disp_profile_x = displacements[:,0]
    disp_profile_y = displacements[:,1]
    disp_profile_z = displacements[:,2]
    disp_profile_sq = disp_profile_x**2 + disp_profile_y**2 + disp_profile_z**2

    sum_disp, _ = np.histogram(z, bins=bins, weights=disp_profile_x)
    avg_disp_x = np.divide(sum_disp, counts, out=np.zeros_like(sum_disp), where=counts>0)
    centers_x = 0.5 * (edges[:-1] + edges[1:])

    sum_disp, _ = np.histogram(z, bins=bins, weights=disp_profile_y)
    avg_disp_y = np.divide(sum_disp, counts, out=np.zeros_like(sum_disp), where=counts>0)
    centers_y = 0.5 * (edges[:-1] + edges[1:])

    sum_disp, _ = np.histogram(z, bins=bins, weights=disp_profile_z)
    avg_disp_z = np.divide(sum_disp, counts, out=np.zeros_like(sum_disp), where=counts>0)
    centers_z = 0.5 * (edges[:-1] + edges[1:])

    return counts, edges, avg_disp_x, centers_x, avg_disp_y, centers_y, avg_disp_z, centers_z

def get_disp_arrows(coords0, cell, displacements, z_cut=17, arrow_scale=1.0):
    """
    Prepare data for plotting displacement arrows in a 2D slice.

    Parameters
    ----------
    coords0 : (N, 3) array
        Original atomic coordinates
    displacements : (N, 3) array
        Atomic displacements
    z_cut : float
        z-coordinate threshold to select the layer
    arrow_scale : float
        Scaling factor for displacement arrows

    Returns
    -------
    all_xy : (M, 2) array
        x, y positions of atoms in the selected layer (tiled)
    all_disp_plot : (M, 2) array
        x, y components of displacements (scaled for plotting)
    """
    # --- select layer ---
    mask = coords0[:, 2] > z_cut    # same slice you used before
    xyz = coords0[mask]
    disp = displacements[mask]

    xy = xyz[:, :2]
    disp_xy = disp[:, :2]

    a = cell["a"]
    b = cell["b"]
    gamma_deg = cell["gamma"]

    # --- 2D lattice vectors ---
    gamma_rad = np.deg2rad(gamma_deg)
    a_vec = np.array([a, 0.0])
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad)])

    # matrix with columns a_vec, b_vec
    M = np.column_stack((a_vec, b_vec))    # shape (2,2)

    # fractional coords, wrap into [0,1)
    frac = np.linalg.solve(M, xy.T).T
    frac_wrapped = frac - np.floor(frac)
    base_xy = frac_wrapped @ M.T          # atoms inside one cell at origin

    nrep_a = 3
    nrep_b = 3

    # --- tile positions and displacements ---
    all_xy = []
    all_disp = []
    for ia in range(-nrep_a, nrep_a + 1):
        for ib in range(-nrep_b, nrep_b + 1):
            shift = ia * a_vec + ib * b_vec
            all_xy.append(base_xy + shift)
            all_disp.append(disp_xy)

    all_xy = np.vstack(all_xy)
    all_disp = np.vstack(all_disp)

    # optionally rescale displacements for visibility
    all_disp_plot = all_disp * arrow_scale

    cell_poly = np.array([
        [0.0, 0.0],
        a_vec,
        a_vec + b_vec,
        b_vec,
        [0.0, 0.0],
    ])

    return all_xy, all_disp_plot, cell_poly
