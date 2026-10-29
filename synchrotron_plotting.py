"""Plotting for Basics.ipynb; all physical equations and constants live in the notebook.

Inputs are precomputed NumPy arrays. Frequencies are Hz, times seconds, fluxes mJy.
No model is calculated, notebook edited, or figure created on import.
"""
from dataclasses import dataclass
from pathlib import Path
import base64
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': '#243547', 'axes.edgecolor': '#8090a0',
    'xtick.color': '#405267', 'ytick.color': '#405267',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
COLORS = ('#087f8c', '#c66b16', '#8054a6')
LINESTYLES = ('-', '--', '-.')


def _finish(fig, savepath=None):
    if savepath is not None:
        path = Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    return fig


def plot_linear_log(x, y, xlabel, ylabel, title=None, savepath=None):
    """The same curve on linear and log-log axes (no global x/y variables)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        for ax, log in zip(axes, (False, True)):
            ax.plot(x, y, color='#2167b0', lw=2.3)
            if log:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set(xlabel=xlabel, ylabel=ylabel, title='Log-log' if log else 'Linear')
            ax.grid(which='major', color='#e1e7ed')
        if title:
            fig.suptitle(title, fontsize=15, fontweight='bold')
        return _finish(fig, savepath)


def plot_energy_budget(fields, particles, magnetic, reference_field, savepath=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        ax.loglog(fields, particles, color='#c66b16', label='Particles', lw=2)
        ax.loglog(fields, magnetic, color='#2167b0', label='Magnetic field', lw=2)
        ax.loglog(fields, np.asarray(particles) + magnetic, color='#087f8c',
                  label='Total energy', lw=2.6)
        ax.axvline(reference_field, color='#718096', ls=':', label='Minimum-energy field')
        ax.set(xlabel='Magnetic field, B (gauss)', ylabel='Energy (erg)',
               title='Energy as a function of magnetic field')
        ax.grid(which='major', color='#e1e7ed')
        ax.legend()
        return _finish(fig, savepath)


def plot_spectra(frequency_hz, fluxes, labels, title='Evolving synchrotron spectrum',
                 savepath=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        for values, label, color, style in zip(fluxes, labels, COLORS, LINESTYLES):
            ax.loglog(np.asarray(frequency_hz) / 1e9, values, color=color,
                      ls=style, lw=2.2, label=label)
        ax.set(xlabel='Frequency (GHz)', ylabel='Flux density (mJy)', title=title)
        ax.grid(which='major', color='#e1e7ed')
        ax.legend()
        return _finish(fig, savepath)


def _lightcurve_axes(axes, times, fluxes, labels, *, draw_curves, views=("linear", "loglog")):
    """Configure the requested views of exactly the same arrays."""
    insets = []
    settings = {"linear": ("Linear-linear", False, False),
                "loglog": ("Log-log", True, True)}
    for ax, view in zip(axes, views):
        title, logx, logy = settings[view]
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set(xlabel='Time (s)', ylabel='Flux density (mJy)', title=title)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim((fluxes.min() * 0.5 if logy else 0), fluxes.max() * 1.6)
        ax.grid(which='major', color='#e1e7ed', lw=0.8)
        for i, (label, color, style) in enumerate(zip(labels, COLORS, LINESTYLES)):
            ax.plot(times if draw_curves else [], fluxes[:, i] if draw_curves else [],
                    color=color, ls=style, lw=1.9, label=label)
        ax.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white',
                  framealpha=0.9, edgecolor='none')
        if view == "linear":
            inset = ax.inset_axes([.48, .31, .48, .35])
            end = times[0] + .16*(times[-1]-times[0])
            inset.set_xlim(times[0], end)
            inset.set_ylim(0, fluxes.max()*1.15)
            inset.set_title('Early time', fontsize=7, pad=2)
            inset.tick_params(labelsize=6)
            inset.grid(color='#e1e7ed', lw=.5)
            for i, (color, style) in enumerate(zip(COLORS, LINESTYLES)):
                inset.plot(times if draw_curves else [],
                           fluxes[:, i] if draw_curves else [],
                           color=color, ls=style, lw=1)
            insets.append(inset)
    return insets



def _check_lightcurves(times, fluxes, labels):
    times, fluxes = np.asarray(times), np.asarray(fluxes)
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(times)) or times[0] <= 0):
        raise ValueError('Use at least two finite, increasing, positive times.')
    if fluxes.shape != (len(times), 3) or len(labels) != 3:
        raise ValueError('Provide three labelled light curves with shape (n_times, 3).')
    if not np.all(np.isfinite(fluxes)) or np.any(fluxes <= 0):
        raise ValueError('Flux densities must be finite and positive for logarithmic plots.')
    return times, fluxes


def plot_lightcurves(times, fluxes, labels, savepath=None):
    """Static linear-linear and log-log light curves, retained for notes."""
    times, fluxes = _check_lightcurves(times, fluxes, labels)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        _lightcurve_axes(axes, times, fluxes, labels, draw_curves=True)
        fig.suptitle('Radio light curves at fixed observing frequencies',
                     fontsize=15, fontweight='bold')
        return _finish(fig, savepath)


@dataclass
class EvolutionAnimation:
    """Own the figure and artists so updates and exports can be checked explicitly."""
    figure: object
    movie: object
    update: object
    frame_indices: np.ndarray
    spectrum_line: object
    lightcurve_lines: list
    cursors: list
    markers: list
    fps: int
    inset_lines: list
    gif_path: Path | None = None
    html_path: Path | None = None
    preview_path: Path | None = None

    def save(self, directory='output/graphs', stem='evolving_synchrotron'):
        """Save a compact GIF, a PNG still, and an independent controlled HTML player."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.gif_path = directory / f'{stem}.gif'
        self.html_path = directory / f'{stem}.html'
        self.preview_path = directory / f'{stem}_still.png'
        self.movie.save(self.gif_path, writer=PillowWriter(fps=self.fps), dpi=72)
        with plt.rc_context({'animation.embed_limit': 40}):
            html = self.movie.to_jshtml(fps=self.fps, default_mode='once')
        symbols = {'fa-minus': '−', 'fa-fast-backward': '|◀', 'fa-step-backward': '◀',
                   'fa-play': '▶', 'fa-pause': 'Ⅱ', 'fa-step-forward': '▶',
                   'fa-fast-forward': '▶|', 'fa-plus': '+'}
        html = re.sub(r'<link[^>]*font-awesome[^>]*>', '', html)
        html = re.sub(r'<i class="fa (fa-[a-z-]+)([^"]*)"></i>',
                      lambda m: '◀' if 'flip-horizontal' in m[2] else symbols.get(m[1], '?'), html)
        css = ('<style>body{margin:16px;font-family:sans-serif;background:white}'
               '.animation{width:100%;max-width:1000px}'
               '.animation img{max-width:100%;height:auto;display:block}'
               '.animation .anim-controls{width:100%!important;box-sizing:border-box}'
               '.anim-buttons{display:flex;justify-content:center;flex-wrap:wrap}</style>')
        self.html_path.write_text(
            '<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Evolving synchrotron spectrum and light curves</title>' + css
            + '</head><body>' + html + '</body></html>', encoding='utf-8')
        self.update(0)
        self.figure.savefig(self.preview_path, dpi=110, facecolor='white')
        plt.close(self.figure)
        return self

    def display(self):
        """No executable JS in notebook output. PNG fallback for restricted renderers."""
        from IPython.display import display
        if self.gif_path is None:
            raise RuntimeError('Call .save() before .display().')
        gif = base64.b64encode(self.gif_path.read_bytes()).decode('ascii')
        png = base64.b64encode(self.preview_path.read_bytes()).decode('ascii')
        display({'text/html': '<img alt="Animated spectrum with linear and log-log light curves" '
                 'style="max-width:100%;height:auto" src="data:image/gif;base64,' + gif + '">',
                 'image/png': png,
                 'text/plain': 'Synchrotron animation; open the GIF or HTML player if this viewer only shows the still.'}, raw=True)


def animate_evolution(times, frequency_hz, spectra, observing_hz, lightcurves,
                      radii, fields, nu1_hz, *, p, labels,
                      frame_indices=None, fps=10):
    """Draw the spectrum alongside two progressively traced light-curve views.

    Every physical array is supplied by the notebook; this function only plots it.
    frame_indices selects playback samples without changing the underlying curves.
    """
    times, lightcurves = _check_lightcurves(times, lightcurves, labels)
    frequency_hz, spectra = np.asarray(frequency_hz), np.asarray(spectra)
    if (frequency_hz.ndim != 1 or len(frequency_hz) < 2
            or np.any(frequency_hz <= 0) or np.any(np.diff(frequency_hz) <= 0)
            or not np.all(np.isfinite(frequency_hz))
            or spectra.shape != (len(times), len(frequency_hz))
            or np.any(spectra <= 0) or not np.all(np.isfinite(spectra))):
        raise ValueError('Provide positive spectra with shape (n_times, n_frequencies) and increasing frequencies.')
    observing_hz = np.asarray(observing_hz)
    if observing_hz.shape != (3,) or np.any(observing_hz <= 0) or not np.all(np.isfinite(observing_hz)):
        raise ValueError('Provide three positive observing frequencies.')
    for values in (radii, fields, nu1_hz):
        if np.shape(values) != times.shape or not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError('Radii, fields and nu1 must be positive arrays matching times.')
    frames = np.arange(len(times)) if frame_indices is None else np.asarray(frame_indices)
    if (frames.ndim != 1 or len(frames) < 2 or frames.dtype.kind not in 'iu'
            or frames.min() < 0 or frames.max() >= len(times) or np.any(np.diff(frames) <= 0)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Use increasing valid frame indices and positive fps.')

    with plt.rc_context(STYLE):
        fig, all_axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=72)
        fig.subplots_adjust(left=.065, right=.98, top=.75, bottom=.18, wspace=.30)
        spectrum_ax = all_axes[0]
        axes = list(all_axes[1:])
        fig.suptitle('How an expanding spectrum produces radio light curves',
                     x=0.075, ha='left', y=0.97, fontsize=16, fontweight='bold')
        status = fig.text(0.075, 0.91, '', color='#405267')
        spectrum_ax.set(xscale='log', yscale='log', xlabel='Frequency (GHz)',
                        ylabel='Flux density (mJy)', title='Spectrum at the current time')
        spectrum_ax.set_xlim(frequency_hz[[0, -1]] / 1e9)
        viewed = spectra[frames]
        spectrum_ax.set_ylim(min(viewed.max() * 1e-7, lightcurves.min() * .5), viewed.max() * 2)
        spectrum_ax.grid(which='major', color='#e1e7ed', lw=.8)
        spectrum_ax.plot(frequency_hz / 1e9, spectra[frames[0]], ls=':',
                         lw=1.2, color='#a3adba', label='First animation frame')
        spectrum_line, = spectrum_ax.plot([], [], color='#2167b0', lw=2.3, label='Current spectrum')
        nu1_line = spectrum_ax.axvline(nu1_hz[frames[0]] / 1e9, color='#718096', ls='--', lw=1)
        band_dots = []
        for frequency, label, color, style in zip(observing_hz, labels, COLORS, LINESTYLES):
            spectrum_ax.axvline(frequency / 1e9, color=color, ls=style, lw=.9, alpha=.5)
            dot, = spectrum_ax.plot([], [], 'o', color=color, ms=5, label=label)
            band_dots.append(dot)
        spectrum_ax.legend(loc='lower left', fontsize=7, ncol=2, framealpha=.9)
        inset_axes = _lightcurve_axes(axes, times, lightcurves, labels, draw_curves=False)
        inset_lines = [list(ax.lines) for ax in inset_axes]
        curve_lines = [list(ax.lines) for ax in axes]
        cursors = [ax.axvline(times[frames[0]], color='#718096', lw=1) for ax in axes]
        time_dots = [[ax.plot([], [], 'o', color=color, ms=4)[0] for color in COLORS] for ax in axes]
        inset_cursors = [ax.axvline(times[frames[0]], color='#718096', lw=.7) for ax in inset_axes]
        fig.text(.065, .045, 'Colour = observing band. Grey dashed spectral line: optical depth = 1. '
                 'Each light curve grows up to the current time.', fontsize=8, color='#718096')

        def update(frame):
            index = int(frames[frame])
            spectrum_line.set_data(frequency_hz / 1e9, spectra[index])
            nu1_line.set_xdata([nu1_hz[index] / 1e9] * 2)
            status.set_text(f't = {times[index]:6.1f} s     R = {radii[index]:.2e} cm     '
                            f'B = {fields[index]:.2e} G     p = {p:g}')
            for i, dot in enumerate(band_dots):
                dot.set_data([observing_hz[i] / 1e9], [lightcurves[index, i]])
            for lines, cursor, dots in zip(curve_lines, cursors, time_dots):
                cursor.set_xdata([times[index]] * 2)
                for i, (line, dot) in enumerate(zip(lines, dots)):
                    line.set_data(times[:index + 1], lightcurves[:index + 1, i])
                    dot.set_data([times[index]], [lightcurves[index, i]])
            for lines, cursor in zip(inset_lines, inset_cursors):
                cursor.set_xdata([times[index]] * 2)
                for i, line in enumerate(lines):
                    line.set_data(times[:index+1], lightcurves[:index+1, i])
            return (spectrum_line, nu1_line, status, *band_dots, *cursors,
                    *(line for group in curve_lines for line in group),
                    *(dot for group in time_dots for dot in group),
                    *(line for group in inset_lines for line in group), *inset_cursors)

        update(0)
        movie = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps,
                               blit=False, repeat=False)
        return EvolutionAnimation(fig, movie, update, frames, spectrum_line,
                                   curve_lines, cursors, time_dots, fps,
                                   inset_lines)


def show_figure(figure):
    """Display one static output without Matplotlib's duplicate automatic display."""
    from IPython.display import display
    display(figure)
    plt.close(figure)



def plot_plasma_evolution(radius_factor, values, labels, *, times=None, savepath=None):
    """Separate, normalised plasma diagnostics; no physical calculations here."""
    x, values = np.asarray(radius_factor), np.asarray(values)
    if (x.ndim != 1 or len(x) < 2 or np.any(x <= 0) or np.any(np.diff(x) <= 0)
            or not np.all(np.isfinite(x)) or values.shape != (len(x), len(labels))
            or len(labels) != 4 or np.any(values <= 0) or not np.all(np.isfinite(values))):
        raise ValueError('Provide increasing radius factors and four positive diagnostic series.')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        for i, (label, color, style) in enumerate(zip(labels,
                ['#2167b0','#c66b16','#8054a6','#718096'], ['-','--',':','-.'])):
            ax.loglog(x, values[:, i], color=color, ls=style, lw=2.2, label=label)
        ax.set(xlabel=r'Expansion factor, $R/R_0$',
               ylabel='Value / value at first model time',
               title='Physical evolution of the expanding plasma')
        ax.set_xlim(x[0], x[-1])
        ax.grid(which='major', color='#e1e7ed', lw=.8)
        ax.legend(loc='lower left', fontsize=10, framealpha=.95)
        if times is not None:
            times = np.asarray(times)
            if times.shape != x.shape or np.any(np.diff(times) <= 0):
                raise ValueError('Times must increase and match the radius array.')
            # This only maps the already supplied axes; the radius law stays in the notebook.
            top = ax.secondary_xaxis('top', functions=(
                lambda r: np.interp(r, x, times), lambda t: np.interp(t, times, x)))
            top.set_xlabel('Time since expansion began (s)')
            ticks = [t for t in [1,10,100,500] if times[0] <= t <= times[-1]]
            if len(ticks) < 2: ticks = [times[0], times[-1]]
            top.set_xticks(ticks)
            top.set_xticklabels([f'{t:g}' for t in ticks])
            ax.set_ylabel(f'Value / value at t = {times[0]:g} s')
        return _finish(fig, savepath)
