"""Readable XY density preview, using only an observation and five parameters."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .gaussian import GaussianBelief


def save_preview(observation, parameters, path):
    belief = GaussianBelief.from_parameters(parameters)
    ellipse = belief.ellipse(.99)
    observed = np.array(observation.box.center[:2])
    lower = np.minimum(ellipse.min(axis=0), observed)-.003
    upper = np.maximum(ellipse.max(axis=0), observed)+.003
    center = (lower+upper)/2
    radius = max((upper-lower).max()/2, .01)
    x = np.linspace(center[0]-radius, center[0]+radius, 180)
    y = np.linspace(center[1]-radius, center[1]+radius, 180)
    gx, gy = np.meshgrid(x, y)
    density = belief.density(np.column_stack((gx.ravel(), gy.ravel()))).reshape(gx.shape)
    fig = Figure(figsize=(6.4, 4.2), dpi=110, layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    image = ax.pcolormesh(y*100, x*100, density.T/10000, cmap='viridis', shading='auto')
    boundary = belief.ellipse(.9)
    ax.plot(boundary[:,1]*100,boundary[:,0]*100,color='cyan',lw=2,label='90% probability boundary')
    ax.plot(observed[1]*100, observed[0]*100,'x',color='orange',ms=9,mew=2,label='Observed OBB center')
    ax.plot(belief.mean[1]*100,belief.mean[0]*100,'o',color='red',ms=5,label='Estimated mean')
    ax.set_aspect('equal')
    ax.set_xlabel('World Y (cm)');ax.set_ylabel('World X (cm)')
    ax.set_title(f'Observation ID {observation.observation_id} | XY Gaussian (zoomed)')
    ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.03,1.))
    fig.colorbar(image,ax=ax,shrink=.65,label='Density (1/cm²)')
    fig.savefig(path, bbox_inches="tight", pad_inches=.1)
    fig.clear()
