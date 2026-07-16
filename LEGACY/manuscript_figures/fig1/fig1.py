import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, rc
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import copy
from pathlib import Path

from spectraformer.input_pipeline import (
    neg_val_removal_fn,
    maxnorm_fn,
    modified_z_score,
    whitaker_hayes_modified_z_score,
    whitaker_hayes_spectrum,
    whitaker_hayes,
)

# ---------------------------
# styling
# ---------------------------
rcParams['font.size'] = 16
rc('font', **{'family':'sans-serif','sans-serif':['Arial','DejaVu Sans']})

# ---------------------------
# peaks
# ---------------------------
peaks = [
    (1360, 'D', 'm', 0.7),
    (1606, 'G', 'green', 0.7),
    (2735, '2D', 'purple', 0.7)
]

# ---------------------------
# helpers
# ---------------------------
def mean_std_over_spatial(da):
    spatial_dims = [d for d in da.dims if d != "wave_number"]
    if len(spatial_dims) == 0:
        mean = da.values
        std = np.zeros_like(mean)
    else:
        mean = da.mean(dim=spatial_dims).values
        std  = da.std(dim=spatial_dims).values
    wn = da["wave_number"].values
    return wn, mean, std

def plot_mean_with_std(ax, wn, mean, std, label, color,
                       offset=0.0, lw=2.2, shade_alpha=0.6):
    """Plot mean line with ±1σ shading and a vertical offset added to the mean."""
    m = mean + offset
    ax.plot(wn, m, lw=lw, color=color, label=label)
    ax.fill_between(wn, m - std, m + std, color=color, alpha=shade_alpha, linewidth=0)

# ---------------------------
# paths
# ---------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sic_da = xr.load_dataarray(
    PROJECT_ROOT / 'data/parsed_data_spatial/SiC-high-f/6H_spectra_20250423/5s_5p/6H_spectra_20250423_15x15_5s_5p_1.nc'
)
buffer_da = xr.load_dataarray(
    PROJECT_ROOT / 'data/parsed_data_spatial/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1/RUN3REC2_Buffer_20241011_1_15x15_pt4.nc'
)
graphene_da = xr.load_dataarray(
    PROJECT_ROOT / 'data/parsed_data_spatial/buffer+graphene/main/main_8x8_Mixed_8x8.nc'
)

# ---------------------------
# preprocessing
# ---------------------------
sic_da      = maxnorm_fn(
    neg_val_removal_fn(
        whitaker_hayes(sic_da)
        )
    )
buffer_da   = maxnorm_fn(
    neg_val_removal_fn(
        whitaker_hayes(buffer_da)
        )
    )
graphene_da = maxnorm_fn(
    neg_val_removal_fn(
        whitaker_hayes(graphene_da)
        )
    )

# ---------------------------
# compute mean ± std across spatial dims
# ---------------------------
wn_sic, mean_sic, std_sic = mean_std_over_spatial(sic_da)
wn_buf, mean_buf, std_buf = mean_std_over_spatial(buffer_da)
wn_g,   mean_g,   std_g   = mean_std_over_spatial(graphene_da)

# ---------------------------
# vertical offsets (small positive offsets to separate traces)
# ---------------------------
offset_sic   = 0.00
offset_buf   = 0.30
offset_graph = 2*offset_buf

# ---------------------------
# zoom windows
# ---------------------------
zoom_left  = (1440, 1720)
zoom_right = (2650, 2810)

# ---------------------------
# figure layout: main on top spanning two columns, two zoom subplots below
# ---------------------------
fig = plt.figure(figsize=(14*1.382, 14/1.618))

gs = gridspec.GridSpec(
    2, 3,
    width_ratios=[1, 1.618, 1],   # левая = как zoom-панель
    height_ratios=[1.618, 1],
    wspace=0.09,
    hspace=0.26
)
gs_a = gs[:, 0].subgridspec(
    3, 1,
    height_ratios=[0.2, 1, 1],
    hspace=0.15,
)
ax_label = fig.add_subplot(gs_a[0])
ax_label.axis("off")
ax_a_top    = fig.add_subplot(gs_a[1])
ax_a_bottom = fig.add_subplot(gs_a[2])
for ax in (ax_a_top, ax_a_bottom):
    ax.axis("off")
import matplotlib.image as mpimg

img_side = mpimg.imread(SCRIPT_DIR / "temp/cropped-high-res-figures/side_view_with_MLG-ZLG-SiC_96dpi.png")
img_top  = mpimg.imread(SCRIPT_DIR / "temp/cropped-high-res-figures/SiC_top-view.png")

ax_a_top.imshow(img_side).set_rasterized(True)
ax_a_bottom.imshow(img_top).set_rasterized(True)


ax_main   = fig.add_subplot(gs[0, 1:])
ax_zleft  = fig.add_subplot(gs[1, 1])
ax_zright = fig.add_subplot(gs[1, 2])

# ---------------------------
# plot main mean ± std (with offsets)
# ---------------------------
plot_mean_with_std(ax_main, wn_sic, mean_sic, std_sic, 'SiC', color='C1', offset=offset_sic, lw=2.0, shade_alpha=0.45)
plot_mean_with_std(ax_main, wn_buf, mean_buf, std_buf, 'ZLG/SiC', color='C0', offset=offset_buf, lw=2.0, shade_alpha=0.45)
plot_mean_with_std(ax_main, wn_g,   mean_g,   std_g,   'MLG/ZLG/SiC', color='C2', offset=offset_graph, lw=2.0, shade_alpha=0.45)

# vertical peak markers + labels on main
for x, label, _, y in peaks:
    ax_main.axvline(x=x, color='black', linestyle='--', alpha=0.6)
    ax_main.text(
        x=x,
        y=y,
        s=f"{label}\n{x}",
        transform=ax_main.get_xaxis_transform(),
        ha='center',
        va='bottom',
        color='black',
        bbox=dict(boxstyle='round', fc='white', alpha=0.85),
        fontsize='x-small'
    )

ax_main.set_xlim(np.min(wn_sic) - 5, np.max(wn_sic) + 5)
ax_main.set_ylim(-0.05, 1.7)
ax_main.set_xlabel('Raman shift, cm$^{-1}$', fontsize='large')
ax_main.set_ylabel('Intensity, a.u.', fontsize='large')
ax_main.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(50))
# ax_main.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
ax_main.set_yticks([])  # hide y-ticks
ax_main.tick_params(axis='both', which='major', labelsize='large')
ax_main.legend(loc='upper center', fontsize='small')

# ---------------------------
# zoom panels: plot same mean ± std with offsets (thinner lines)
# ---------------------------
plot_mean_with_std(ax_zleft, wn_sic, mean_sic, std_sic, 'SiC', color='C1', offset=offset_sic, lw=1.5, shade_alpha=0.45)
plot_mean_with_std(ax_zleft, wn_buf, mean_buf, std_buf, 'ZLG/SiC', color='C0', offset=offset_buf, lw=1.5, shade_alpha=0.45)
plot_mean_with_std(ax_zleft, wn_g,   mean_g,   std_g,   'MLG/ZLG/SiC', color='C2', offset=offset_graph, lw=1.5, shade_alpha=0.45)

for x, *_ in peaks:
    ax_zleft.axvline(x=x, color='black', linestyle='--', alpha=0.6)

ax_zleft.set_xlim(*zoom_left)
# choose an appropriate ylim for zoom left that reflects offsets + shading
ax_zleft.set_ylim(-0.025, 1.65)
ax_zleft.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax_zleft.xaxis.set_minor_locator(ticker.MultipleLocator(10))
# ax_zleft.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
ax_zleft.set_yticks([])  # hide y-ticks
ax_zleft.tick_params(axis='both', which='major', labelsize='large')
ax_zleft.set_xlabel('Raman shift, cm$^{-1}$', fontsize='large')
ax_zleft.set_ylabel('Intensity, a.u.', fontsize='large')

plot_mean_with_std(ax_zright, wn_sic, mean_sic, std_sic, 'SiC', color='C1', offset=offset_sic, lw=1.5, shade_alpha=0.45)
plot_mean_with_std(ax_zright, wn_buf, mean_buf, std_buf, 'ZLG/SiC', color='C0', offset=offset_buf, lw=1.5, shade_alpha=0.45)
plot_mean_with_std(ax_zright, wn_g,   mean_g,   std_g,   'MLG/ZLG/SiC', color='C2', offset=offset_graph, lw=1.5, shade_alpha=0.45)

for x, *_ in peaks:
    ax_zright.axvline(x=x, color='black', linestyle='--', alpha=0.6)

ax_zright.set_xlim(*zoom_right)
ax_zright.set_ylim(-0.025, 0.93)
ax_zright.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax_zright.xaxis.set_minor_locator(ticker.MultipleLocator(10))
# ax_zright.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
ax_zright.set_yticks([])  # hide y-ticks
ax_zright.tick_params(axis='both', which='major', labelsize='large')
ax_zright.set_xlabel('Raman shift, cm$^{-1}$', fontsize='large')
ax_zright.set_ylabel('Intensity, a.u.', fontsize='large')



# ---------------------------
# Use mark_inset to draw rectangle on main and connectors to the subplot axes.
# mark_inset draws a rectangle on the parent axes and connector lines to the inset axes.
# It requires the inset axes to already exist (they do — ax_zleft and ax_zright).
# ---------------------------

# IMPORTANT: set zoom axes limits before calling mark_inset (we already did),
# so mark_inset can compute connector endpoints correctly.

# left zoom: use loc1=2, loc2=4 (corners on the rectangle) — these choices give diagonal connectors
rect_left, con1_left, con2_left = mark_inset(ax_main, ax_zleft, loc1=1, loc2=2, fc="none", ec="crimson", lw=1.0, ls='-')
for spine in ax_zleft.spines.values():
    spine.set_edgecolor("crimson")
    spine.set_linewidth(2.5)
con1_left.set_linestyle('')
con2_left.set_linestyle('')
# raise z-order so rectangle and connectors are visible over plotted lines
for artist in (rect_left, con1_left, con2_left):
    try:
        artist.set_zorder(15)
    except Exception:
        pass

# right zoom: use loc1=1, loc2=3
rect_right, con1_right, con2_right = mark_inset(ax_main, ax_zright, loc1=1, loc2=2, fc="none", ec="blue", lw=1.0)
for spine in ax_zright.spines.values():
    spine.set_edgecolor("blue")
    spine.set_linewidth(2.5)
con1_right.set_linestyle('')
con2_right.set_linestyle('')
for artist in (rect_right, con1_right, con2_right):
    try:
        artist.set_zorder(15)
    except Exception:
        pass

# Small visual tweak: make rectangle lines a bit thicker and semi-transparent white background for texts already set
rect_left.set_linewidth(2.5)
rect_right.set_linewidth(2.5)

# ---------------------------
# final polish & show
# ---------------------------

annotations_letters = ['(b)', '(c)', '(d)']
annotations = []
for ax, label in zip([ax_main, ax_zleft, ax_zright], annotations_letters):
    annotation = ax.annotate(label, xy=(0.01, 0.97), xycoords='axes fraction', ha='left', va='top', fontsize='large', fontweight='bold')
    annotations.append(annotation)

ax_label.annotate('(a)', xy=(0, 0.5), xycoords='axes fraction', ha='left', va='top', fontsize='large', fontweight='bold')

ax_label.annotate('— C', xy=(0.35, 0.5), xycoords='axes fraction', ha='center', va='top', fontsize='large', color='#9b7761ff', fontweight='bold')
ax_label.annotate('— Si', xy=(0.55, 0.5), xycoords='axes fraction', ha='center', va='top', fontsize='large', color='#6b77bfff', fontweight='bold')

plt.show()

# ---------------------------
# save
# ---------------------------
outdir = SCRIPT_DIR / "temp/fig1-output"
outdir.mkdir(parents=True, exist_ok=True)

filename = "fig1_with_MLG-ZLG-SiC"
# fig.savefig(outdir / f"{filename}.eps", transparent=True, bbox_inches='tight')
fig.savefig(outdir / f"{filename}.svg", transparent=True, bbox_inches='tight', dpi=300)
fig.savefig(outdir / f"{filename}.png", transparent=True, dpi=300, bbox_inches='tight')
# fig.savefig(outdir / f"{filename}.pdf", transparent=True, bbox_inches='tight')