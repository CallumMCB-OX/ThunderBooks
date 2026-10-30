"""Labelled S1 spectra, with consistent p=2.5 coefficients and sizefromt input.
Run with ../.venv/bin/python plot_synchrotron_spectra.py from this directory,
or .venv/bin/python scripts/plot_synchrotron_spectra.py from the repo.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parents[1]/'output'/'graphs'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
 'axes.labelsize':14,'axes.titlesize':20,'axes.titleweight':'bold',
 'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#8090a0','axes.labelcolor':'#243547',
 'xtick.color':'#405267','ytick.color':'#405267',
 'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
# Parameters from S1. Use its requested cs expansion speed and p=2.5 table values.
p=2.5
B=1.9
c=3e10
r=1800*(c/np.sqrt(3))*0.07
D=3*3.1e21
N0=0.0045
c1,c5,c6=6.27e18,0.97e-23,8.10e-41
omega=np.pi*(r/D)**2
nu1=2*c1*(r*c6*N0)**(2/(p+4))*B**((p+2)/(p+4))
SF=(c5/c6)*B**(-0.5)*(nu1/(2*c1))**2.5
scale=omega*SF*1e26

def flux(nu):
 z=np.asarray(nu)/nu1
 return scale*z**2.5*(-np.expm1(-z**(-(p+4)/2)))

# At a spectral maximum: 5/2 = [(p+4)/2] tau/(exp(tau)-1).
a=(p+4)/2
taupeak=brentq(lambda t:a*t/np.expm1(t)-2.5,1e-8,10)
nupeak=nu1*taupeak**(-1/a)
fpeak=float(flux(nupeak))
f1=float(flux(nu1))
nu=np.geomspace(nu1/10,nu1*30,4000)
f=flux(nu)
assert np.all(np.isfinite(f)) and np.all(f>0)
assert abs(flux(nupeak*1.00001)-fpeak)/fpeak<1e-8
assert np.isclose(f1/scale,1-np.exp(-1))

BLUE='#2167b0'; TEAL='#087f8c'; ORANGE='#c66b16'; GRAY='#718096'

def draw(ax,log=False):
 ax.plot(nu/1e9,f,color=BLUE,lw=3,zorder=4)
 ax.set_xlabel(r'Observing frequency, $\nu$ (GHz)',labelpad=12)
 ax.set_ylabel(r'Observed flux density, $S_\nu$ (mJy)',labelpad=10)
 ax.set_title('Synchrotron spectrum',loc='left',pad=30)
 ax.text(0,1.035,'Log–log axes' if log else 'Linear axes',transform=ax.transAxes,
         fontsize=12,color=GRAY)
 ax.axvline(nu1/1e9,color=GRAY,ls=(0,(4,4)),lw=1.4,zorder=1)
 ax.scatter([nupeak/1e9],[fpeak],s=52,color=BLUE,edgecolor='white',zorder=6)
 ax.scatter([nu1/1e9],[f1],s=45,color='white',edgecolor=GRAY,zorder=5)
 ax.annotate('Spectral peak\n'+rf'$\nu_{{\rm peak}}={nupeak/1e9:.1f}$ GHz',
  xy=(nupeak/1e9,fpeak),xycoords='data',xytext=((0.57,0.94) if log else (0.47,0.84)),
  textcoords='axes fraction',fontsize=12,color=BLUE,
  arrowprops={'arrowstyle':'-','color':BLUE,'connectionstyle':'arc3,rad=0.12'})
 if log:
  ax.set_xscale('log');ax.set_yscale('log')
  ax.set_xlim(nu[0]/1e9,nu[-1]/1e9)
  ax.set_ylim(f.min()*0.55,fpeak*3)
  ax.grid(which='major',color='#e1e7ed',lw=0.9)
  ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*0.1))
  ax.xaxis.set_minor_formatter(NullFormatter())
  ax.annotate('Optically thick\n'+r'$\tau_\nu\gg1$'+ '\n'+r'$S_\nu\propto\nu^{5/2}$',
   xy=(nu1*.23/1e9,float(flux(nu1*.23))),xytext=(0.04,0.63),
   textcoords='axes fraction',color=TEAL,fontsize=13,
   arrowprops={'arrowstyle':'-','color':TEAL})
  ax.annotate('Optically thin\n'+r'$\tau_\nu\ll1$'+'\n'+r'$S_\nu\propto\nu^{-0.75}$',
   xy=(nu1*9/1e9,float(flux(nu1*9))),xytext=(0.69,0.35),
   textcoords='axes fraction',color=ORANGE,fontsize=13,
   arrowprops={'arrowstyle':'-','color':ORANGE})
  ax.text(nu1/1e9,0.025,r'$\nu_1$: $\tau_\nu=1$',
   transform=ax.get_xaxis_transform(),rotation=90,va='bottom',ha='right',
   color=GRAY,fontsize=11,bbox={'facecolor':'white','edgecolor':'none','pad':2})
 else:
  ax.set_xlim(0,nu[-1]/1e9);ax.set_ylim(0,fpeak*1.3)
  ax.grid(axis='y',color='#e1e7ed',lw=.9)
  ax.annotate('Self-absorbed rise\n'+r'$S_\nu\propto\nu^{5/2}$',
   xy=(nu1*.7/1e9,float(flux(nu1*.7))),xytext=(.20,.54),
   textcoords='axes fraction',color=TEAL,fontsize=13,
   arrowprops={'arrowstyle':'-','color':TEAL})
  ax.annotate('Optically thin decline\n'+r'$S_\nu\propto\nu^{-0.75}$',
   xy=(nu1*16/1e9,float(flux(nu1*16))),xytext=(.58,.38),
   textcoords='axes fraction',color=ORANGE,fontsize=13,
   arrowprops={'arrowstyle':'-','color':ORANGE})
  ax.text(.18,.08,rf'$\nu_1={nu1/1e9:.1f}$ GHz: $\tau_\nu=1$',
   transform=ax.transAxes,fontsize=11,color=GRAY)
 ax.set_axisbelow(True)

footer=(r'Uniform synchrotron source • $p=2.5$ • $B=1.9$ G • $D=3$ kpc'
        '\nSame spectrum in both views; the spectral peak is not at '+r'$\tau_\nu=1$.')
for log,name in [(False,'synchrotron_linear'),(True,'synchrotron_loglog')]:
 fig,ax=plt.subplots(figsize=(10.8,7.1))
 fig.subplots_adjust(left=.12,right=.96,bottom=.19,top=.84)
 draw(ax,log)
 fig.text(.12,.055,footer,fontsize=10,color=GRAY,linespacing=1.7)
 for ext in ['png','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
# A convenient two-panel export for pasting both plots together.
fig,axes=plt.subplots(1,2,figsize=(18,6.8))
fig.subplots_adjust(left=.065,right=.98,bottom=.19,top=.83,wspace=.24)
for ax,log in zip(axes,[False,True]):draw(ax,log)
fig.text(.065,.045,footer,fontsize=11,color=GRAY,linespacing=1.6)
for ext in ['png','svg']:fig.savefig(OUT/f'synchrotron_spectra_pair.{ext}',dpi=300,facecolor='white')
plt.close(fig)
print(f'nu1={nu1/1e9:.4f} GHz, nu_peak={nupeak/1e9:.4f} GHz, F_peak={fpeak:.4f} mJy, tau_peak={taupeak:.4f}')
print('Saved PNG and SVG exports in',OUT)
