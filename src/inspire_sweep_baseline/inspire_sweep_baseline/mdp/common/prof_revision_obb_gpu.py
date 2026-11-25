"""Batched initial frontal OBB calculation; native scalar geometry per scene.

At most seven active sources (target + six obstacles). Unlike raster masks,
interval partition includes exact projected edge intersections and uses doubles.
The CPU implementation in prof_revision_obb is the independent reference.
"""
import torch
import warp as wp


_OBSERVER_CODE = r'''
const int row = task;
double polygon[7][4][2], box[7][7];
int source[7], n = 0;
for (int slot=0; slot<17; ++slot) {
    if (!wp::index(active, row,slot)) continue;
    if (n >= 7) { wp::index(status, row)=3; return; }
    source[n]=slot;
    for(int k=0;k<7;++k) box[n][k]=(double)wp::index(boxes, row,slot,k);
    const double c=cos(box[n][6]),s=sin(box[n][6]);
    for(int k=0;k<4;++k) {
        double x=((k==1||k==2)?0.5:-0.5)*box[n][3];
        double y=((k>=2)?0.5:-0.5)*box[n][4];
        polygon[n][k][0]=box[n][1]+s*x+c*y;
        polygon[n][k][1]=-(box[n][0]+c*x-s*y);
    }
    ++n;
}
if(n==0) { wp::index(status, row)=1; return; }
double cuts[406]; int nc=0;
for(int i=0;i<n;++i) for(int k=0;k<4;++k) cuts[nc++]=polygon[i][k][0];
// Exact crossing cuts; physical placement normally makes this loop a no-op.
for(int e=0;e<n*4;++e) {
    int i=e/4,k=e%4,j=(k+1)%4;
    double ax=polygon[i][k][0],ay=polygon[i][k][1];
    double dx=polygon[i][j][0]-ax,dy=polygon[i][j][1]-ay;
    for(int f=e+1;f<n*4;++f) {
        int u=f/4,v=f%4,w=(v+1)%4;
        double cx=polygon[u][v][0],cy=polygon[u][v][1];
        double ox=polygon[u][w][0]-cx,oy=polygon[u][w][1]-cy;
        double det=dx*oy-dy*ox;
        if(det==0.0) continue;
        double t=((cx-ax)*oy-(cy-ay)*ox)/det;
        double a=((cx-ax)*dy-(cy-ay)*dx)/det;
        if(t>0.0&&t<1.0&&a>0.0&&a<1.0) cuts[nc++]=ax+t*dx;
    }
}
for(int i=1;i<nc;++i) { double v=cuts[i];int j=i-1;while(j>=0&&cuts[j]>v){cuts[j+1]=cuts[j];--j;}cuts[j+1]=v; }
int unique=0;
for(int i=0;i<nc;++i) if(unique==0||cuts[i]!=cuts[unique-1]) cuts[unique++]=cuts[i];
int owners[405],np=0; double lows[405],highs[405];
for(int i=0;i+1<unique;++i) {
    double lo=cuts[i],hi=cuts[i+1],middle=(lo+hi)/2.0;
    int owner=-1;double depth=1.0e300;
    for(int a=0;a<n;++a) {
        double front=1.0e300;
        for(int k=0;k<4;++k) {
            int j=(k+1)%4;double x=polygon[a][k][0],y=polygon[a][j][0];
            if(x!=y&&middle>=fmin(x,y)&&middle<=fmax(x,y))
                front=fmin(front,polygon[a][k][1]+(middle-x)*(polygon[a][j][1]-polygon[a][k][1])/(y-x));
        }
        if(front<depth){depth=front;owner=a;}
    }
    if(owner<0) continue;
    if(np>0&&owners[np-1]==owner&&highs[np-1]==lo) highs[np-1]=hi;
    else {owners[np]=owner;lows[np]=lo;highs[np]=hi;++np;}
}
int obstacle_count=0; double best_width=-1.0;
for(int a=0;a<n;++a) {
    double full_lo=1.0e300,full_hi=-1.0e300;
    for(int k=0;k<4;++k){full_lo=fmin(full_lo,polygon[a][k][0]);full_hi=fmax(full_hi,polygon[a][k][0]);}
    double threshold=fmax((double)min_width,(double)min_ratio*(full_hi-full_lo));
    for(int part=0;part<np;++part) {
        if(owners[part]!=a)continue;
        double lo=lows[part],hi=highs[part],width=hi-lo;
        double rounding=8.881784197001252e-16*fmax(fmax(fabs(lo),fabs(hi)),threshold);
        if(width<threshold-rounding)continue;
        double result[7];for(int k=0;k<7;++k)result[k]=box[a][k];
        if(lo!=full_lo||hi!=full_hi) {
            double px[8],py[8],tx[8],ty[8];int count=4;
            for(int k=0;k<4;++k){px[k]=polygon[a][k][0];py[k]=polygon[a][k][1];}
            for(int clipping=0;clipping<2;++clipping) {
                double bound=clipping==0?lo:hi;int total=0;
                for(int k=0;k<count;++k) {
                    int j=(k+1)%count;
                    bool ai=clipping==0?px[k]>=bound:px[k]<=bound;
                    bool bi=clipping==0?px[j]>=bound:px[j]<=bound;
                    if(ai){tx[total]=px[k];ty[total]=py[k];++total;}
                    if(ai!=bi){double t=(bound-px[k])/(px[j]-px[k]);tx[total]=bound;ty[total]=py[k]+t*(py[j]-py[k]);++total;}
                }
                count=total;for(int k=0;k<count;++k){px[k]=tx[k];py[k]=ty[k];}
            }
            // Invert projection: shelf XY = (-depth,lateral).
            double mx=0.0,my=0.0;
            for(int k=0;k<count;++k){tx[k]=-py[k];ty[k]=px[k];mx+=tx[k];my+=ty[k];}
            mx/=count;my/=count;
            for(int k=0;k<count;++k){tx[k]-=mx;ty[k]-=my;}
            double area=1.0e300,half_pi=1.57079632679489661923;
            for(int k=0;k<count;++k) {
                int j=(k+1)%count;double dx=tx[j]-tx[k],dy=ty[j]-ty[k];
                if(dx==0.0&&dy==0.0)continue;
                double angle=fmod(atan2(dy,dx),half_pi);if(angle<0.0)angle+=half_pi;
                if(fmin(angle,half_pi-angle)<1.0e-12)angle=0.0;
                double c=cos(angle),s=sin(angle),xmin=1.0e300,xmax=-1.0e300,ymin=1.0e300,ymax=-1.0e300;
                for(int v=0;v<count;++v){double x=c*tx[v]+s*ty[v],y=-s*tx[v]+c*ty[v];xmin=fmin(xmin,x);xmax=fmax(xmax,x);ymin=fmin(ymin,y);ymax=fmax(ymax,y);}
                double current=(xmax-xmin)*(ymax-ymin);
                if(current<area-1.0e-12||(fabs(current-area)<=1.0e-12&&angle<result[6])){area=current;double x=(xmin+xmax)/2.0,y=(ymin+ymax)/2.0;
                    result[0]=c*x-s*y+mx;result[1]=s*x+c*y+my;
                    result[3]=xmax-xmin;result[4]=ymax-ymin;result[6]=angle;}
            }
        }
        if(source[a]==0) {
            if(width>best_width){best_width=width;for(int k=0;k<7;++k)wp::index(target, row,k)=(float)result[k];}
        } else {
            if(obstacle_count<16){for(int k=0;k<7;++k)wp::index(obstacles, row,obstacle_count,k)=(float)result[k];
                wp::index(valid, row,obstacle_count)=true;wp::index(sources, row,obstacle_count)=source[a];}
            ++obstacle_count;
        }
    }
}
if(best_width<0.0)wp::index(status, row)=1;
else if(obstacle_count>16)wp::index(status, row)=2;
'''


@wp.func_native(_OBSERVER_CODE)
def _observe_scene(task: int, boxes: wp.array3d(dtype=float), active: wp.array2d(dtype=bool),
                   min_width: float, min_ratio: float, target: wp.array2d(dtype=float),
                   obstacles: wp.array3d(dtype=float), valid: wp.array2d(dtype=bool),
                   sources: wp.array2d(dtype=wp.int64), status: wp.array(dtype=int)):
    ...


@wp.kernel(enable_backward=False)
def _observe(boxes: wp.array3d(dtype=float), active: wp.array2d(dtype=bool),
             min_width: float, min_ratio: float, target: wp.array2d(dtype=float),
             obstacles: wp.array3d(dtype=float), valid: wp.array2d(dtype=bool),
             sources: wp.array2d(dtype=wp.int64), status: wp.array(dtype=int)):
    _observe_scene(wp.tid(), boxes, active, min_width, min_ratio, target, obstacles, valid, sources, status)


_observe.module.options['fuse_fp'] = False


def observe_batch(boxes, active, minimum_width=.02, minimum_ratio=.20):
    """[B,17,7] shelf-local bounds -> target, obstacles, validity, source, status.

    Status: 0=valid, 1=missing target, 2=slot overflow, 3=roster violation.
    """
    if boxes.shape[1:] != (17, 7) or active.shape != boxes.shape[:2]:
        raise ValueError('Expected boxes[B,17,7] and active[B,17]')
    wp.init()
    n, device = len(boxes), boxes.device
    target = boxes.new_zeros(n, 7)
    obstacles = boxes.new_zeros(n, 16, 7)
    valid = torch.zeros((n, 16), device=device, dtype=torch.bool)
    sources = torch.full((n, 16), -1, device=device, dtype=torch.long)
    status = torch.zeros(n, device=device, dtype=torch.int32)
    args = [wp.from_torch(boxes.contiguous()), wp.from_torch(active.contiguous()), minimum_width, minimum_ratio,
            wp.from_torch(target), wp.from_torch(obstacles), wp.from_torch(valid), wp.from_torch(sources), wp.from_torch(status)]
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if boxes.is_cuda else None
    wp.launch(_observe, dim=n, inputs=args, device=str(device), stream=stream)
    return target, obstacles, valid, sources, status


# Eleven sources need 44 edge cuts plus C(44,2) crossings (990).
# Keep the widest 16 obstacle fragments on overflow; physical bodies stay active.
_HARD_OBSERVER_CODE = (_OBSERVER_CODE
    .replace('[7][4][2]', '[11][4][2]').replace('box[7][7]', 'box[11][7]')
    .replace('source[7]', 'source[11]').replace('n >= 7', 'n >= 11')
    .replace('[406]', '[990]').replace('[405]', '[989]')
    .replace('int obstacle_count=0;', 'double widths[16]; int obstacle_count=0;')
    .replace(r'''if(obstacle_count<16){for(int k=0;k<7;++k)wp::index(obstacles, row,obstacle_count,k)=(float)result[k];
                wp::index(valid, row,obstacle_count)=true;wp::index(sources, row,obstacle_count)=source[a];}''', r'''int index=obstacle_count;
            if(index>=16){index=0;for(int k=1;k<16;++k)if(widths[k]<widths[index])index=k;
                if(width<=widths[index]){++obstacle_count;continue;}}
            widths[index]=width;
            for(int k=0;k<7;++k)wp::index(obstacles, row,index,k)=(float)result[k];
            wp::index(valid, row,index)=true;wp::index(sources, row,index)=source[a];'''))


@wp.func_native(_HARD_OBSERVER_CODE)
def _observe_hard_scene(task: int, boxes: wp.array3d(dtype=float), active: wp.array2d(dtype=bool),
                   min_width: float, min_ratio: float, target: wp.array2d(dtype=float),
                   obstacles: wp.array3d(dtype=float), valid: wp.array2d(dtype=bool),
                   sources: wp.array2d(dtype=wp.int64), status: wp.array(dtype=int)):
    ...


@wp.kernel(enable_backward=False)
def _observe_hard(boxes: wp.array3d(dtype=float), active: wp.array2d(dtype=bool),
             min_width: float, min_ratio: float, target: wp.array2d(dtype=float),
             obstacles: wp.array3d(dtype=float), valid: wp.array2d(dtype=bool),
             sources: wp.array2d(dtype=wp.int64), status: wp.array(dtype=int)):
    _observe_hard_scene(wp.tid(), boxes, active, min_width, min_ratio, target, obstacles, valid, sources, status)


_observe_hard.module.options['fuse_fp'] = False


def observe_hard_batch(boxes, active, minimum_width=.02, minimum_ratio=.20):
    """[B,17,7] shelf-local bounds -> target, obstacles, validity, source, status.

    Status: 0=valid, 1=missing target, 2=slot overflow, 3=roster violation.
    """
    if boxes.shape[1:] != (17, 7) or active.shape != boxes.shape[:2]:
        raise ValueError('Expected boxes[B,17,7] and active[B,17]')
    wp.init()
    n, device = len(boxes), boxes.device
    target = boxes.new_zeros(n, 7)
    obstacles = boxes.new_zeros(n, 16, 7)
    valid = torch.zeros((n, 16), device=device, dtype=torch.bool)
    sources = torch.full((n, 16), -1, device=device, dtype=torch.long)
    status = torch.zeros(n, device=device, dtype=torch.int32)
    args = [wp.from_torch(boxes.contiguous()), wp.from_torch(active.contiguous()), minimum_width, minimum_ratio,
            wp.from_torch(target), wp.from_torch(obstacles), wp.from_torch(valid), wp.from_torch(sources), wp.from_torch(status)]
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if boxes.is_cuda else None
    wp.launch(_observe_hard, dim=n, inputs=args, device=str(device), stream=stream)
    return target, obstacles, valid, sources, status
