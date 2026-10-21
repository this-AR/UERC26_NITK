"""Shared figure styling and the gnuplot runner.

Colours are the validated default categorical palette, in fixed order (never cycled);
anything beyond 8 categories is drawn in GREY.
"""
import os
import subprocess

from .config import BASH, GNUPLOT_BIN, TOOLS_BIN

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK2, GREY = '#fcfcfb', '#0b0b0b', '#52514e', '#c9c8c3'


def clean_axes(ax, title=None, grid=True):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, color=INK, loc='left')


def run_bash(script, cwd=None):
    """Run a bash snippet (matching scripts, gnuplot heredocs) with tools/bin and gnuplot on PATH."""
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([str(TOOLS_BIN), str(GNUPLOT_BIN), env['PATH']])
    subprocess.run([str(BASH), '-c', script], cwd=cwd, env=env, check=True)
