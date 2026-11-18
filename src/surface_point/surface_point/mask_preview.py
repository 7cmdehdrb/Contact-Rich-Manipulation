"""Diagnostic mask view from ray ownership, not a rendered RGB sensor."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def preview(frame):
    palette=np.array([[81,161,217],[232,157,74],[90,193,125],[170,112,220],
                      [205,120,160],[80,190,190],[190,190,70],[150,170,220]],dtype=np.uint8)
    rgb=np.full((*frame.owner.shape,3),245,dtype=np.uint8)
    rgb[frame.owner==-1]=[140,140,140]
    mask=frame.owner>=0
    rgb[mask]=palette[frame.owner[mask]%len(palette)]
    scale=min(900/frame.owner.shape[1],360/frame.owner.shape[0])
    width,height=max(1,int(frame.owner.shape[1]*scale)),max(1,int(frame.owner.shape[0]*scale))
    image=Image.new('RGBA',(max(width,600),height+60),'white')
    image.paste(Image.fromarray(rgb[::-1]).resize((width,height),Image.Resampling.NEAREST),(0,30))
    draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    draw.text((8,5),'Geometric visible mask | red: chosen point | outline: component bbox',fill='black',font=font)
    h,w=frame.owner.shape
    for item in frame.pixels:
        r,c=item['pixel_rc'];r0,c0,r1,c1=item['bbox_rc']
        x,y=(c+.5)*width/w,30+(h-r-.5)*height/h
        draw.rectangle((c0*width/w,30+(h-r1)*height/h,c1*width/w,30+(h-r0)*height/h),outline='black',width=1)
        draw.ellipse((x-3,y-3,x+3,y+3),fill='red')
        draw.text((x+4,y+4),str(item['observation_id']),fill='black',font=font)
    draw.text((8,height+35),f'Grid {w}x{h} | spacing {frame.spacing*1000:g} mm | fixed +X -> -X',fill='black',font=font)
    return np.asarray(image).copy()
