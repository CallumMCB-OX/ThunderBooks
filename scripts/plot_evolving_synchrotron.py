"""Annotated van der Laan spectra using the corrected S1 example parameters."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parents[1]/'output'/'graphs'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
 'axes.labelsize':14,'axes.titlesize':21,'axes.titleweight':'bold',
 'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#8090a0','axes.labelcolor':'#243547',
 'xtick.color':'#405267','ytick.color':'#405267',
 'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})
p=2.5
c1,c5,c6=6.27e18,0.97e-23,8.10e-41
Rref=1800*(3e10/np.sqrt(3))*0.07
Bref=1.9
Nref=.0045
D=3*3.1e21
q=(p+4)/2
taupeak=brentq(lambda t:q*t/np.expm1(t)-2.5,1e-8,10)


def spectrum(nu,expansion):
 R=Rref*expansion
 B=Bref*expansion**(-2)
 N0=Nref*expansion**(-(p+2))
 omega=np.pi*(R/D)**2
 nu1=2*c1*(R*c6*N0)**(2/(p+4))*B**((p+2)/(p+4))
 sf=(c5/c6)*B**(-.5)*(nu1/(2*c1))**2.5
 z=np.asarray(nu)/nu1
 f=omega*sf*z**2.5*(-np.expm1(-z**(-q)))*1e26
 return f,nu1

nu=np.geomspace(.3e9,400e9,5000)
colors=['#c66b16','#2167b0','#087f8c']
sizes=[.5,1,2]
peaks=[]
fig,ax=plt.subplots(figsize=(12,8.4))
fig.subplots_adjust(left=.11,right=.96,bottom=.245,top=.84)
for size,color in zip(sizes,colors):
 f,nu1=spectrum(nu,size)
 nup=nu1*taupeak**(-1/q)
 fp=float(spectrum(nup,size)[0])
 assert np.all(np.isfinite(f)) and np.all(f>0)
 assert abs(float(spectrum(nup*1.00001,size)[0])/fp-1)<1e-8
 peaks.append((nup/1e9,fp))
 ax.plot(nu/1e9,f,color=color,lw=3,zorder=3)
 ax.scatter([nup/1e9],[fp],s=65,color=color,edgecolor='white',linewidth=1.3,zorder=6)
 print(f'R/R_ref={size}: peak frequency={nup/1e9:.4f} GHz, peak flux={fp:.4f} mJy')

ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlim(.3,400);ax.set_ylim(1e-3,700)
ax.set_xlabel(r'Observing frequency, $\nu$ (GHz)',labelpad=12)
ax.set_ylabel(r'Observed flux density, $S_\nu$ (mJy)',labelpad=10)
ax.set_title('Evolving synchrotron spectrum',loc='left',pad=31)
ax.text(0,1.035,'van der Laan expansion • log–log axes',transform=ax.transAxes,
        color='#718096',fontsize=12)
ax.grid(which='major',color='#e1e7ed',lw=.85)
ax.set_axisbelow(True)

# Direct labels identify radius and chronology; leaders identify the exact peaks.
positions=[(105,310),(5,65),(1.15,7)]
labels=[r'$R=0.5R_{\rm ref}$ — earlier',r'$R=R_{\rm ref}$ — reference',r'$R=2R_{\rm ref}$ — later']
for peak,pos,label,color in zip(peaks,positions,labels,colors):
 ax.annotate(label+'\n'+rf'$\nu_{{\rm peak}}={peak[0]:.1f}$ GHz',xy=peak,xytext=pos,
  fontsize=12,color=color,ha='left',va='bottom',
  arrowprops={'arrowstyle':'-','color':color,'lw':1.15})

# Track the peaks to show motion to lower frequency and lower peak flux.
px,py=np.array(peaks).T
ax.plot(px,py,color='#7c8795',lw=1.3,ls=(0,(3,4)),zorder=2)
ax.annotate('',xy=peaks[2],xytext=(peaks[1][0]*.9,peaks[1][1]*.82),
 arrowprops={'arrowstyle':'-|>','color':'#7c8795','lw':1.4,'mutation_scale':15},zorder=5)
ax.text(.035,.93,'As the source expands:\nthe peak moves to lower frequency\nand lower flux density.',
 transform=ax.transAxes,color='#405267',fontsize=12,va='top',linespacing=1.5,
 bbox={'facecolor':'white','edgecolor':'none','pad':3})

ax.annotate('Optically thick\n'+r'$S_\nu\propto\nu^{5/2}$',
 xy=(.8,float(spectrum(.8e9,1)[0])),xytext=(.035,.45),textcoords='axes fraction',
 color='#405267',fontsize=13,
 arrowprops={'arrowstyle':'-','color':'#718096','lw':1.1})
ax.annotate('Optically thin\n'+r'$S_\nu\propto\nu^{-0.75}$',
 xy=(110,float(spectrum(110e9,2)[0])),xytext=(.76,.25),textcoords='axes fraction',
 color='#405267',fontsize=13,
 arrowprops={'arrowstyle':'-','color':'#718096','lw':1.1})

fig.text(.11,.137,
 r'Expansion model:  $B\propto R^{-2}$,   $N_0\propto R^{-(p+2)}$;   $p=2.5$ stays fixed.',
 fontsize=11,color='#405267')
fig.text(.11,.092,
 r'At fixed frequency: thick emission rises as $R^3$; thin emission fades as $R^{-5}$.',
 fontsize=11,color='#405267')
fig.text(.11,.045,
 r'Reference: $B_{\rm ref}=1.9$ G, $D=3$ kpc. Markers show spectral peaks, not $\tau_\nu=1$.',
 fontsize=10,color='#718096')

# Independent expansion checks: frequency and peak flux follow exact power laws.
for i,size in enumerate(sizes):
 assert np.isclose(peaks[i][0]/peaks[1][0],size**(-2*(2*p+3)/(p+4)))
 assert np.isclose(peaks[i][1]/peaks[1][1],size**(-(7*p+3)/(p+4)))
for ext in ['png','svg']:
 fig.savefig(OUT/f'evolving_synchrotron_spectrum.{ext}',dpi=300,facecolor='white')
plt.close(fig)
print('Saved annotated PNG and SVG; peak evolution checks passed.')
