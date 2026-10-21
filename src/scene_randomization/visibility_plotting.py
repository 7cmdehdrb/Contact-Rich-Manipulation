"""Headless views of geometric recognition; plotting is outside timed decisions."""

import numpy as np

SEEN = "#168574"
UNSEEN = "#ce583c"
TARGET = "#71459c"


def object_names(count):
    return ["T", *(f"O{i}" for i in range(1, count))]


def draw_visibility(ax, batch, result, index, *, rays=True):
    from matplotlib.patches import Circle

    c = batch.config
    centers = np.concatenate((batch.target[index:index+1], batch.obstacles[index]), axis=0)
    names = object_names(len(centers))
    x0, x1, y0, y1 = c.board
    ax.set_facecolor("#f8fafc")
    ax.plot([x0, x0, x1, x1], [y0, y1, y1, y0], color="#334155", lw=1.5)
    ax.plot([x0, x1], [y0, y0], "--", color="#64748b", lw=1)
    if rays:
        edges, owners = result.edges[index], result.owners[index]
        for left, right, owner in zip(edges[:-1], edges[1:], owners):
            if owner < 0:
                continue
            x = (left + right) / 2
            cx, cy = centers[owner]
            near_y = cy - np.sqrt(max(c.radius**2 - (x-cx)**2, 0))
            ax.plot([x, x], [y0-1, near_y], color="#64748b", alpha=.25, lw=.7, zorder=1)
    path = batch.approach[index]
    ax.plot(path[:,0], path[:,1], "--", lw=.7, color="#94a3b8", zorder=1)
    goal = batch.goal[index]
    ax.add_patch(Circle(goal, c.radius, fill=False, ec=TARGET, ls="--", lw=1))
    ax.annotate("", xy=goal, xytext=centers[0], arrowprops=dict(arrowstyle="->",color=TARGET,lw=1))
    for j, (x,y) in enumerate(centers):
        seen = result.visible[index,j]
        ax.add_patch(Circle((x,y),c.radius,fc=SEEN if seen else UNSEEN,
                            ec=TARGET if j==0 else "#334155",lw=2 if j==0 else .8,zorder=3))
        ax.text(x,y,f"{names[j]}\n{100*result.ratio[index,j]:.1f}%",ha="center",va="center",
                color="white",fontsize=7.5,zorder=4)
    ax.text((x0+x1)/2,y0-3,"FRONT: view direction +y  ↑",ha="center",va="center",fontsize=8,color="#475569")
    ax.set(xlim=(x0-2,x1+2),ylim=(y0-5,y1+2),aspect="equal")
    ax.tick_params(labelsize=7)
    hidden = int(result.not_visible[index].sum())
    ax.set_title(f"Scene {index} | seen {len(centers)-hidden}, not seen {hidden}",fontsize=10)


def save_visibility_gallery(batch, result, path, count, timing_text):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    count = min(count,len(batch))
    order = np.argsort(result.not_visible.sum(axis=1),kind="stable")
    indices = order[np.linspace(0,len(batch)-1,count).astype(int)]
    cols = min(3,count); rows = int(np.ceil(count/cols))
    fig, axes = plt.subplots(rows,cols,figsize=(6.2*cols,3.7*rows+1.1),squeeze=False)
    for ax,i in zip(axes.flat,indices):
        draw_visibility(ax,batch,result,int(i))
    for ax in list(axes.flat)[count:]:ax.set_visible(False)
    fig.suptitle(f"Front visibility >= {100*result.threshold:g}% | orthographic, equal height\n{timing_text}",fontsize=12)
    fig.legend(handles=[Patch(fc=SEEN,label="Seen >= threshold"),Patch(fc=UNSEEN,label="Not seen < threshold"),
                        Patch(fc="none",ec=TARGET,label="T: target; dashed circle: goal (not an occluder)")],
               loc="lower center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93),h_pad=1.5)
    fig.savefig(path,dpi=160);plt.close(fig)
    return indices.tolist()


def save_visibility_detail(batch,result,path,index,timing_text):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = result.ratio.shape[1]; c=batch.config
    fig = plt.figure(figsize=(15,7.6))
    grid = fig.add_gridspec(2,2,width_ratios=(3.2,1.35),height_ratios=(3,1.1))
    top=fig.add_subplot(grid[0,0]);draw_visibility(top,batch,result,index)
    top.set(xlabel="x (cm)",ylabel="y (cm)")
    bars=fig.add_subplot(grid[0,1])
    bars.barh(object_names(n),result.ratio[index]*100,
              color=[SEEN if x else UNSEEN for x in result.visible[index]])
    bars.axvline(result.threshold*100,ls="--",color="#334155",label=f"Threshold {100*result.threshold:g}%")
    for j,value in enumerate(result.ratio[index]*100):
        bars.text(min(value+1,101),j,f"{value:.1f}%",va="center",fontsize=9)
    bars.set(xlim=(0,113),xlabel="Visible projected area (%)",title="Geometric recognition")
    bars.invert_yaxis();bars.legend(fontsize=8,loc="lower right")
    front=fig.add_subplot(grid[1,0])
    for left,right,owner in zip(result.edges[index,:-1],result.edges[index,1:],result.owners[index]):
        if owner<0:continue
        front.add_patch(Rectangle((left,0),right-left,c.height,
                                 fc=SEEN if result.visible[index,owner] else UNSEEN,
                                 ec="white",lw=.4))
        if right-left>1:
            front.text((left+right)/2,c.height/2,object_names(n)[owner],color="white",ha="center",va="center",fontsize=8)
    front.set(xlim=c.board[:2],ylim=(0,c.height),xlabel="x (cm)",ylabel="height (cm)",
              title="Front projection: only first-hit silhouettes are drawn")
    info=fig.add_subplot(grid[1,1]);info.axis("off")
    info.text(0,.95,timing_text.replace(" | ","\n"),va="top",fontsize=9)
    fig.suptitle(f"Scene {index} | T=target, O1..O{n-1}=obstacles | goal does not occlude",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(path,dpi=180);plt.close(fig)
