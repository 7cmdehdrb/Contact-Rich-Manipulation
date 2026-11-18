"""Small in-memory density image; no pyplot, PNG files or disk IO per reset."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .gaussian import GaussianBelief


@lru_cache(maxsize=1)
def _font():
    try:
        return ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        return ImageFont.load_default()


def _colors(values):
    anchors = np.array([[20,12,65],[35,60,155],[0,155,185],[100,215,110],[255,235,45]])
    x = np.clip(values,0,1)*4
    i = np.minimum(x.astype(int),3)
    return ((1-(x-i))[...,None]*anchors[i]+(x-i)[...,None]*anchors[i+1]).astype(np.uint8)


def preview_rgba(observation, parameters):
    belief = GaussianBelief.from_parameters(parameters)
    ellipse = belief.ellipse(.99)
    observed = np.array(observation.box.center[:2])
    lower = np.minimum(ellipse.min(0),observed)-.003
    upper = np.maximum(ellipse.max(0),observed)+.003
    center = (lower+upper)/2
    radius = max(float((upper-lower).max()/2),.01)
    lower,upper = center-radius,center+radius
    # image column = world Y; image row = decreasing world X
    n=256
    y=np.linspace(lower[1],upper[1],n)
    x=np.linspace(upper[0],lower[0],n)
    xx,yy=np.meshgrid(x,y,indexing='ij')
    dx,dy=xx-belief.mean[0],yy-belief.mean[1]
    inv=np.linalg.inv(belief.covariance)
    density=np.exp(-.5*(inv[0,0]*dx*dx+2*inv[0,1]*dx*dy+inv[1,1]*dy*dy))
    peak=1/(2*np.pi*np.sqrt(np.linalg.det(belief.covariance)))/10000
    image=Image.new('RGBA',(720,500),'white')
    heat=Image.fromarray(_colors(density)).resize((400,400),Image.Resampling.BILINEAR)
    image.paste(heat,(75,45))
    draw=ImageDraw.Draw(image);font=_font()
    def point(xy):
        return (75+(xy[1]-lower[1])/(2*radius)*400,45+(upper[0]-xy[0])/(2*radius)*400)
    draw.line([point(p) for p in belief.ellipse(.9)],fill='cyan',width=2)
    ox,oy=point(observed);draw.line((ox-5,oy-5,ox+5,oy+5),fill='orange',width=2)
    draw.line((ox-5,oy+5,ox+5,oy-5),fill='orange',width=2)
    mx,my=point(belief.mean);draw.ellipse((mx-3,my-3,mx+3,my+3),fill='red')
    draw.rectangle((75,45,475,445),outline='black')
    for t in np.linspace(0,1,5):
        draw.text((75+t*400,451),f'{(lower[1]+2*radius*t)*100:.1f}',font=font,fill='black',anchor='mt')
        draw.text((68,45+t*400),f'{(upper[0]-2*radius*t)*100:.1f}',font=font,fill='black',anchor='rm')
    draw.text((75,12),f'ID {observation.observation_id}: Gaussian XY (zoomed)',font=font,fill='black')
    draw.text((220,478),'World Y (cm)',font=font,fill='black')
    draw.text((7,25),'X (cm)',font=font,fill='black')
    bar_colors = _colors(np.linspace(1,0,240))
    for row in range(240):
        color=tuple(bar_colors[row].tolist())
        draw.line((510,145+row,535,145+row),fill=color,width=1)
    for t in (0,.5,1):
        draw.text((542,145+(1-t)*239),f'{peak*t:.3g}',font=font,fill='black',anchor='lm')
    draw.text((500,110),'Density (1/cm^2)',font=font,fill='black')
    draw.text((495,45),'Cyan: 90% boundary',font=font,fill='black')
    draw.text((495,65),'Red: estimated mean',font=font,fill='black')
    draw.text((495,85),'Orange X: observed',font=font,fill='black')
    return np.asarray(image).copy()
