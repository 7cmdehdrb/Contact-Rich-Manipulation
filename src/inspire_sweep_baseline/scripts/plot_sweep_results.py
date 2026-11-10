"""Replot existing episode CSVs using explicit GT groups and minimal labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from extract_sweep_results import SECTORS

QUADRANTS = ('Q1', 'Q2', 'Q3', 'Q4')
QUADRANT_COLORS = dict(Q1='#B8DDA8', Q2='#FFD0A1', Q3='#A8DCD5', Q4='#CEC0EB')


def goal_region_distance(x, y, inner=6., outer=12., deadzone_degrees=30.):
    """Exact distance (cm) to the union of the two allowed annular sectors."""
    angle = np.arctan2(y, x)
    dead = np.deg2rad(deadzone_degrees)
    distance = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.inf)
    for low, high in ((dead, np.pi - dead), (-np.pi + dead, -dead)):
        nearest_angle = np.clip(angle, low, high)
        projection = x * np.cos(nearest_angle) + y * np.sin(nearest_angle)
        radius = np.clip(projection, inner, outer)
        distance = np.minimum(distance, np.sqrt(np.maximum(
            0., x*x + y*y + radius*radius - 2*radius*projection)))
    return distance


def plot_goal_region(output):
    """Configuration-derived support, including the 2 cm spawn disk; no episodes."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'Times New Roman', 'font.size': 12}):
        fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        for group, sx, sy in (('Q1', 1, 1), ('Q2', -1, 1), ('Q3', -1, -1), ('Q4', 1, -1)):
            xs = np.linspace(0 if sx > 0 else -15, 15 if sx > 0 else 0, 751)
            ys = np.linspace(0 if sy > 0 else -15, 15 if sy > 0 else 0, 751)
            x, y = np.meshgrid(xs, ys)
            field = goal_region_distance(x, y) - 2.
            axis.contourf(x, y, field, levels=[-100, 0], colors=[QUADRANT_COLORS[group]], alpha=.65)
            axis.contour(x, y, field, levels=[0], colors=['#555555'], linewidths=1.1)
            axis.text(sx * 6.5, sy * 8, group, ha='center', va='center')
        axis.axhline(0, color='#888888', lw=.7)
        axis.axvline(0, color='#888888', lw=.7)
        axis.set(xlim=(-15, 15), ylim=(-15, 15), aspect='equal',
                 xlabel='GT goal X (cm)', ylabel='GT goal Y (cm)',
                 xticks=np.arange(-15, 16, 5), yticks=np.arange(-15, 16, 5))
        axis.grid(alpha=.15)
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'goal_generation_region.{suffix}', dpi=200)
        plt.close(fig)
    metadata = dict(task='Inspire-Sweep-Simple-Finger-v0', units='cm',
        min_distance=6, max_distance=12, spawn_disk_radius=2, excluded_x_half_angle_degrees=30,
        definition='union of allowed displacement annular sectors Minkowski-summed with spawn disk',
        quadrant_origin='shelf origin; absolute GT coordinate signs', colors=QUADRANT_COLORS,
        source='sweep_simple.py event config and wrench_refactor_layout.py sample_commands',
        numerical_boundary_grid_cm=.02, empirical_data_used=False,
        shelf_edge='inactive: maximum |goal X| <= 12*cos(30 deg)+2 = 12.393 cm; safe X >= 13.6 cm')
    (output / 'goal_generation_region.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return output


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def goal_quadrant(x, y):
    """Shelf X/Y signs; positive axes belong to Q1/Q2/Q4 deterministically."""
    if not np.isfinite([x, y]).all() or (x == 0 and y == 0):
        return 'undefined'
    if y >= 0:
        return 'Q1' if x >= 0 else 'Q2'
    return 'Q4' if x >= 0 else 'Q3'


def grouped_error_statistics(rows, order):
    result = []
    for group in ('all', *order):
        selected = rows if group == 'all' else [r for r in rows if r['plot_group'] == group]
        values = np.array([float(r['error_xy_m']) * 100 for r in selected if r['error_xy_m']], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) or group == 'all':
            result.append(dict(group=group, n=len(values),
                median_cm=float(np.median(values)) if len(values) else None,
                mean_cm=float(np.mean(values)) if len(values) else None,
                min_cm=float(values.min()) if len(values) else None,
                max_cm=float(values.max()) if len(values) else None))
    return result


def error_box_statistics(rows, order):
    """Quartile boxes; whiskers span the actual min/max, not the 1.5-IQR rule."""
    result = []
    for group in ('all', *order):
        selected = rows if group == 'all' else [r for r in rows if r['plot_group'] == group]
        values = np.array([float(r['error_xy_m']) * 100 for r in selected if r['error_xy_m']])
        values = values[np.isfinite(values)]
        if len(values):
            result.append(dict(label=group, n=len(values), med=float(np.median(values)),
                q1=float(np.percentile(values, 25)), q3=float(np.percentile(values, 75)),
                mean=float(np.mean(values)), whislo=float(values.min()), whishi=float(values.max())))
    return result


def filter_large_errors(rows, multiplier=1.5):
    """One pooled upper Tukey fence keeps group/overall populations additive."""
    values = np.array([float(r['error_xy_m']) * 100 for r in rows if r['error_xy_m']])
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75]) if len(values) else (0., 0.)
    upper = float(q3 + multiplier * (q3 - q1))
    kept, excluded = [], []
    for row in rows:
        value = float(row['error_xy_m']) * 100 if row['error_xy_m'] else np.nan
        (kept if np.isfinite(value) and value <= upper else excluded).append(row)
    return kept, excluded, dict(method='pooled upper Q3 + multiplier*IQR; low errors retained',
        multiplier=multiplier, q1_cm=float(q1), q3_cm=float(q3), upper_fence_cm=upper,
        source_n=len(rows), retained_n=len(kept), excluded_n=len(excluded))


def box_axis_ranges(values, main_max=10.):
    """Only break when retained data exceed main_max; gap contains no samples."""
    high = [float(v) for v in values if np.isfinite(v) and v > main_max]
    return dict(main=[0., main_max], upper=[(main_max + min(high)) / 2,
                max(high) + max(.2, .05 * (max(high) - main_max))] if high else None)


def restyle_error_boxplot(output, axis_max=7., font='Times New Roman', step=.72):
    """Redraw only the existing error chart; preserve other images and all CSVs."""
    from matplotlib import font_manager

    if not np.isfinite(axis_max) or axis_max <= 0 or not np.isfinite(step) or step <= .5:
        raise ValueError('axis_max must be positive and box step must exceed .5')
    font_path = font_manager.findfont(font, fallback_to_default=False)
    output = Path(output).resolve()
    rows = read_csv(output / 'error_boxplot.csv')
    boxes = [{key: (row[key] if key == 'label' else int(row[key]) if key == 'n' else float(row[key]))
              for key in row} for row in rows if row['label'] != 'all']
    colors = QUADRANT_COLORS
    ranges = box_axis_ranges([r['whishi'] for r in boxes], axis_max)
    with plt.rc_context({'font.family': font, 'font.size': 12}):
        if ranges['upper'] is None:
            fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
            axis.set_ylim(ranges['main'])
            axes = [axis]
        else:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6), constrained_layout=True,
                                    gridspec_kw=dict(height_ratios=[1, 4], hspace=.06))
            axes[0].set_ylim(ranges['upper'])
            axes[1].set_ylim(ranges['main'])
            axes[0].spines['bottom'].set_visible(False)
            axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
            wave = np.linspace(-1, 1, 80)
            for axis, edge in ((axes[0], 0), (axes[1], 1)):
                for x in (0, 1):
                    axis.plot(x + wave * .018, edge + .009 * np.sin(wave * 2 * np.pi),
                              transform=axis.transAxes, color='#222222', lw=1, clip_on=False)
        positions = np.arange(len(boxes)) * step
        for axis in axes:
            artists = axis.bxp(boxes, positions=positions, widths=.5, patch_artist=True,
                showmeans=True, showfliers=False, medianprops=dict(color='#222222', linewidth=1.5),
                meanprops=dict(marker='D', markerfacecolor='#222222', markeredgecolor='#222222', markersize=5))
            for patch, row in zip(artists['boxes'], boxes):
                patch.set_facecolor(colors.get(row['label'], '#44827B'))
                patch.set_alpha(.85)
            axis.set_xlim(-.48, positions[-1] + .48 if len(positions) else .48)
            axis.spines[['top', 'right']].set_visible(False)
        axes[-1].set_ylabel('Final XY error (cm)')
        if boxes:
            axes[0].legend([artists['medians'][0], artists['means'][0]], ['Median', 'Mean'], frameon=False)
        for name in ('error_boxplot.png', 'error_statistics.png'):
            fig.savefig(output / name, dpi=200)
        plt.close(fig)
    style = dict(font=font, font_path=font_path, colors=colors, displayed_groups=[r['label'] for r in boxes],
                 axis_ranges=ranges, positions=positions.tolist(), box_width=.5,
                 gap=step - .5, aggregate_all='omitted; per-quadrant mean markers retained',
                 source='existing error_boxplot.csv; no recomputation or recollection')
    (output / 'error_boxplot_style.json').write_text(json.dumps(style, indent=2) + '\n', encoding='utf-8')
    return output


def restyle_timeout_scatter(source, output, per_group=200, alpha=.65, seed=42):
    """Equal-probability sampling without replacement in each GT quadrant."""
    source, output = Path(source).resolve(), Path(output).resolve()
    rows = [r for r in read_csv(source / 'episodes.csv') if r['completed'] == '1'
            and r['time_out'] == '1' and r['horizon_reached'] == '1'
            and np.isfinite([float(r['error_x_m']), float(r['error_y_m'])]).all()]
    rng = np.random.default_rng(seed)
    samples, counts = {}, {}
    for group in QUADRANTS:
        candidates = [r for r in rows if goal_quadrant(float(r['goal_x_m']), float(r['goal_y_m'])) == group]
        indices = rng.choice(len(candidates), min(per_group, len(candidates)), replace=False)
        samples[group] = [candidates[i] for i in sorted(indices)]
        counts[group] = dict(source=len(candidates), sampled=len(indices), visible=sum(
            abs(float(r['error_x_m'])) <= .05 and abs(float(r['error_y_m'])) <= .05
            for r in samples[group]))
    radius = json.loads((source / 'summary.json').read_text(encoding='utf-8')).get('goal_radius_m', .03) * 100

    def draw(axis, group):
        for name in QUADRANTS if group == 'all' else (group,):
            selected = samples[name]
            axis.scatter([float(r['error_x_m']) * 100 for r in selected],
                         [float(r['error_y_m']) * 100 for r in selected],
                         s=15, alpha=alpha, color=QUADRANT_COLORS[name], linewidths=0, label=name)
        axis.axhline(0, color='#888888', lw=.65)
        axis.axvline(0, color='#888888', lw=.65)
        axis.add_patch(Circle((0, 0), radius, fill=False, color='#333333', ls='--', lw=1))
        axis.set(xlim=(-5, 5), ylim=(-5, 5), aspect='equal', title=group,
                 xlabel='Shelf X error (cm)', ylabel='Shelf Y error (cm)')
        axis.grid(alpha=.13)
        if group == 'all':
            axis.legend(frameon=False, markerscale=1.5)

    with plt.rc_context({'font.family': 'Times New Roman', 'font.size': 12}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        for axis, group in zip(axes.flat, ('all', *QUADRANTS)):
            draw(axis, group)
        axes.flat[-1].axis('off')
        fig.savefig(output / 'timeout_scatter_zoom.png', dpi=200)
        plt.close(fig)
        folder = output / 'individual' / 'timeout_scatter_zoom'
        folder.mkdir(parents=True, exist_ok=True)
        for group in ('all', *QUADRANTS):
            fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
            draw(axis, group)
            fig.savefig(folder / f'{group}.png', dpi=200)
            plt.close(fig)
    metadata = dict(method='uniform random without replacement per GT quadrant; before viewport clipping',
        seed=seed, requested_per_quadrant=per_group, alpha=alpha, colors=QUADRANT_COLORS,
        counts=counts, axis_limits_cm=[-5, 5], outlier_filter=False,
        selected_episode_ids={g: [[r['env_id'], r['episode_id']] for r in v] for g, v in samples.items()})
    (output / 'timeout_scatter_zoom_style.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return output


def plot(input_dir, output_dir=None, clean_timeouts_only=False, scatter_limit_cm=None,
         individual=False, group_by='goal-quadrant', error_outlier_iqr=1.5, error_axis_max_cm=10.):
    if not np.isfinite(error_outlier_iqr) or error_outlier_iqr <= 0:
        raise ValueError('error_outlier_iqr must be finite and positive')
    if not np.isfinite(error_axis_max_cm) or error_axis_max_cm <= 0:
        raise ValueError('error_axis_max_cm must be finite and positive')
    if scatter_limit_cm is not None and (not np.isfinite(scatter_limit_cm) or scatter_limit_cm <= 0):
        raise ValueError('scatter_limit_cm must be finite and positive')
    if group_by not in ('goal-quadrant', 'direction'):
        raise ValueError('group_by must be goal-quadrant or direction')
    source = Path(input_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else source / 'plots'
    raw = source.parent / 'raw'
    if output == source or output == raw or raw in output.parents:
        raise ValueError('Plots must be separate from analysis CSVs and raw data')
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    episodes = [r for r in read_csv(source / 'episodes.csv') if r['completed'] == '1']
    for row in episodes:
        row['plot_group'] = (goal_quadrant(float(row['goal_x_m']), float(row['goal_y_m']))
                             if group_by == 'goal-quadrant' else row['sector'])
    timeouts = [r for r in episodes if r['time_out'] == '1' and r['horizon_reached'] == '1']
    if clean_timeouts_only:
        timeouts = [r for r in timeouts if r['clean_timeout'] == '1']
    order = (*QUADRANTS, 'undefined') if group_by == 'goal-quadrant' else (*SECTORS, 'undefined')
    colors = dict(zip(SECTORS, ['#246A9A', '#BD651B', '#78812C', '#A65385', '#6D629A', '#44827B']))
    colors.update(QUADRANT_COLORS, undefined='#666666', all='#333333')
    plt.rcParams['font.family'] = 'Times New Roman'
    output.mkdir(parents=True, exist_ok=True)
    zoom_counts, displayed_panels, individual_files = {}, {}, []

    def groups_for(rows):
        observed = {r['plot_group'] for r in rows}
        return ('all', *(g for g in order if g in observed))

    def new_grid(count, width=4.8, height=4.4):
        columns = min(3, max(count, 1))
        rows = int(np.ceil(max(count, 1) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                                 constrained_layout=True, squeeze=False)
        for axis in list(axes.flat)[count:]:
            axis.axis('off')
        return fig, list(axes.flat)

    def save(fig, path):
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def individual_path(family, group):
        folder = output / 'individual' / family
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (group + '.png')
        individual_files.append(path.relative_to(output).as_posix())
        return path

    def scatter(family, rows, first, second, xlabel, ylabel, goal_centered=False, limit_cm=None, unit_direction=False):
        panels = groups_for(rows)
        displayed_panels[family] = list(panels)
        valid = [r for r in rows if r[first] not in ('', None) and r[second] not in ('', None)
                 and np.isfinite([float(r[first]), float(r[second])]).all()]
        factor = 1 if unit_direction else 100
        values = [abs(float(r[key])) * factor for r in valid for key in (first, second)]
        radius = summary.get('goal_radius_m')
        limit = (limit_cm if limit_cm is not None else
                 1.1 if unit_direction else max(values + [radius * 100 if goal_centered and radius else 0, 1]) * 1.1)

        def draw(axis, group):
            selected = valid if group == 'all' else [r for r in valid if r['plot_group'] == group]
            for name in order:
                subset = [r for r in selected if r['plot_group'] == name]
                if subset:
                    axis.scatter([float(r[first]) * factor for r in subset],
                                 [float(r[second]) * factor for r in subset],
                                 s=13, alpha=.65, color=colors[name], linewidths=0, label=name)
            axis.axhline(0, color='#888888', lw=.65)
            axis.axvline(0, color='#888888', lw=.65)
            if goal_centered and radius:
                axis.add_patch(Circle((0, 0), radius * 100, fill=False, color='#333333', ls='--', lw=1))
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
                     xlabel=xlabel, ylabel=ylabel, title=f'{group} (n={len(selected)})')
            axis.grid(alpha=.13)
            if group == 'all' and selected:
                axis.legend(loc='upper right', fontsize=8, frameon=False)
            if limit_cm is not None:
                visible = sum(abs(float(r[first])) * 100 <= limit and abs(float(r[second])) * 100 <= limit
                              for r in selected)
                zoom_counts[group] = dict(total=len(selected), visible=visible, outside=len(selected) - visible)
            if not selected:
                axis.text(.5, .5, 'No data', ha='center', transform=axis.transAxes)

        fig, axes = new_grid(len(panels))
        for axis, group in zip(axes, panels):
            draw(axis, group)
        save(fig, output / (family + '.png'))
        if individual:
            for group in panels:
                fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
                draw(axis, group)
                save(fig, individual_path(family, group))

    scatter('timeout_scatter', timeouts, 'error_x_m', 'error_y_m',
            'Shelf X error (cm)', 'Shelf Y error (cm)', goal_centered=True)
    if scatter_limit_cm is not None:
        scatter('timeout_scatter_zoom', timeouts, 'error_x_m', 'error_y_m',
                'Shelf X error (cm)', 'Shelf Y error (cm)', goal_centered=True, limit_cm=scatter_limit_cm)
    scatter('generated_goals', episodes, 'goal_x_m', 'goal_y_m', 'GT goal X (cm)', 'GT goal Y (cm)')
    directions = []
    for row in episodes:
        entry = dict(row)
        vector = np.array([float(row['goal_x_m']), float(row['goal_y_m'])])
        length = np.linalg.norm(vector)
        entry['gt_direction_x'], entry['gt_direction_y'] = vector / length if np.isfinite(length) and length > 0 else (np.nan, np.nan)
        directions.append(entry)
    scatter('goal_directions', directions, 'gt_direction_x', 'gt_direction_y',
            'GT direction X', 'GT direction Y', unit_direction=True)

    contact = []
    for group in groups_for(episodes):
        selected = episodes if group == 'all' else [r for r in episodes if r['plot_group'] == group]
        moving = sum(int(r['moving_steps']) for r in selected)
        accepted = sum(int(r['moving_contact_steps']) for r in selected)
        contact.append(dict(group=group, n=len(selected), moving_steps=moving,
                            accepted_steps=accepted, ratio=accepted / moving if moving else None))
    displayed_panels['moving_contact_ratio'] = [r['group'] for r in contact]

    def contact_chart(rows, path):
        fig, axis = plt.subplots(figsize=(max(5, 1.45 * len(rows)), 5), constrained_layout=True)
        for i, row in enumerate(rows):
            rate = row['ratio']
            if rate is not None:
                axis.bar(i, rate * 100, color=colors[row['group']], width=.6)
                axis.text(i, rate * 100 + 1.5, f'{rate * 100:.1f}', ha='center', fontsize=10)
            else:
                axis.text(i, 2, 'N/A', ha='center')
        axis.set_xticks(range(len(rows)), [r['group'] for r in rows])
        axis.set(ylim=(0, 108), ylabel=f"Contact (v >= {summary['speed_threshold_m_s']:g} m/s, %)")
        axis.set_yticks(range(0, 101, 20))
        axis.spines[['top', 'right']].set_visible(False)
        save(fig, path)

    contact_chart(contact, output / 'moving_contact_ratio.png')
    if individual:
        for row in contact:
            contact_chart([row], individual_path('moving_contact_ratio', row['group']))

    error_rows, excluded_errors, error_filter = filter_large_errors(timeouts, error_outlier_iqr)
    statistics = grouped_error_statistics(error_rows, order)
    boxes = error_box_statistics(error_rows, order)
    axis_ranges = box_axis_ranges([float(r['error_xy_m']) * 100 for r in error_rows], error_axis_max_cm)
    displayed_panels['error_statistics'] = [r['group'] for r in statistics]

    def box_chart(selected):
        if axis_ranges['upper'] is not None:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(max(5, len(selected) * 1.5), 6.5),
                gridspec_kw=dict(height_ratios=[1, 4], hspace=.06), constrained_layout=True)
            axes[0].set_ylim(axis_ranges['upper'])
            axes[1].set_ylim(axis_ranges['main'])
            axes[0].spines['bottom'].set_visible(False)
            axes[1].spines['top'].set_visible(False)
            axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
            # Wavy breaks at both sides; all retained observations stay in range.
            for axis, edge in ((axes[0], 0), (axes[1], 1)):
                wave = np.linspace(-1, 1, 80)
                for x in (0, 1):
                    axis.plot(x + wave * .018, edge + .009 * np.sin(wave * 2 * np.pi),
                              transform=axis.transAxes, color='#222222', lw=1, clip_on=False)
        else:
            fig, axis = plt.subplots(figsize=(max(5, len(selected) * 1.5), 5.5), constrained_layout=True)
            axis.set_ylim(axis_ranges['main'])
            axes = [axis]
        for axis in axes:
            artists = axis.bxp(selected, showmeans=True, showfliers=False, patch_artist=True, widths=.5,
                medianprops=dict(color='#222222', linewidth=1.5),
                meanprops=dict(marker='D', markerfacecolor='#222222', markeredgecolor='#222222', markersize=5))
            for patch, row in zip(artists['boxes'], selected):
                patch.set_facecolor(colors[row['label']])
                patch.set_alpha(.5)
            axis.spines[['top', 'right']].set_visible(False)
        axes[-1].set_ylabel('Final XY error (cm)')
        if selected:
            axes[0].legend([artists['medians'][0], artists['means'][0]], ['Median', 'Mean'], frameon=False)
        return fig

    fig = box_chart(boxes)
    fig.savefig(output / 'error_boxplot.png', dpi=200)
    save(fig, output / 'error_statistics.png')
    if individual:
        for row in boxes:
            save(box_chart([row]), individual_path('error_statistics', row['label']))
    with (output / 'error_boxplot.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('label', 'n', 'med', 'q1', 'q3', 'mean', 'whislo', 'whishi'))
        writer.writeheader()
        writer.writerows(boxes)
    with (output / 'error_outliers.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ('env_id', 'episode_id', 'plot_group', 'error_xy_m')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(excluded_errors)
    for filename, rows in [('error_statistics.csv', statistics), ('contact_statistics.csv', contact)]:
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'plot_episode_groups.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ('env_id', 'episode_id', 'sector', 'goal_x_m', 'goal_y_m', 'plot_group')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(episodes)
    metadata = dict(source=str(source), clean_timeouts_only=clean_timeouts_only, timeout_episodes=len(timeouts),
        scatter_limit_cm=scatter_limit_cm, zoom_counts=zoom_counts, group_by=group_by,
        quadrant_definition=dict(Q1='X>=0,Y>=0', Q2='X<0,Y>=0', Q3='X<0,Y<0', Q4='X>=0,Y<0',
                                 origin='shelf origin; exact origin/nonfinite -> undefined'),
        axes='horizontal shelf X, vertical shelf Y; cm except unit GT direction',
        goal_directions='normalized GT goal relative to shelf origin, not goal-initial displacement',
        statistics_cohort='timeout episodes within pooled upper IQR fence; final XY root-to-goal error',
        error_outlier_filter=error_filter, error_axis_ranges=axis_ranges,
        error_boxplot='box=25th to 75th percentiles; median=line; mean=diamond; whiskers=actual min/max; no 1.5-IQR cutoff',
        contact_cohort='all completed episodes; pooled moving-step counts',
        captions='only axes, units, group/sample labels, essential legends and contact values',
        empty_directions='omitted; never represented as 0% observations',
        displayed_panels=displayed_panels, individual_files=individual_files)
    (output / 'plot_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    if individual:
        lines = ['# Individual plots', '']
        for family, groups in displayed_panels.items():
            lines += ['## ' + family, ''] + [f'- [{g}]({family}/{g}.png)' for g in groups] + ['']
        (output / 'individual/index.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--clean-timeouts-only', action='store_true')
    parser.add_argument('--scatter-limit-cm', type=float, help='Additional centered zoom window in cm')
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--group-by', choices=('goal-quadrant', 'direction'), default='goal-quadrant')
    parser.add_argument('--error-outlier-iqr', type=float, default=1.5)
    parser.add_argument('--error-axis-max-cm', type=float)
    parser.add_argument('--error-only', action='store_true', help='Restyle only existing combined error boxplot images')
    parser.add_argument('--goal-region-only', action='store_true', help='Draw theoretical Simple Finger-v0 goal support only')
    parser.add_argument('--scatter-only', action='store_true', help='Restyle zoom scatter with 200 uniform samples per quadrant')
    parser.add_argument('--box-font', default='Times New Roman')
    parser.add_argument('--box-step', type=float, default=.72)
    args = parser.parse_args()
    if args.scatter_only:
        print(restyle_timeout_scatter(args.input_dir, args.output_dir or args.input_dir / 'plots'))
        return
    if args.goal_region_only:
        print(plot_goal_region(args.output_dir or args.input_dir / 'plots'))
        return
    if args.error_only:
        print(restyle_error_boxplot(args.output_dir or args.input_dir / 'plots',
                                   args.error_axis_max_cm if args.error_axis_max_cm is not None else 7.,
                                   args.box_font, args.box_step))
        return
    print(plot(args.input_dir, args.output_dir, args.clean_timeouts_only,
               args.scatter_limit_cm, args.individual, args.group_by, args.error_outlier_iqr,
               args.error_axis_max_cm if args.error_axis_max_cm is not None else 10.))


if __name__ == '__main__':
    main()
