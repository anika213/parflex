import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

def parflex_stage2_with_discontinuity(C_global, tiled_result, show_fig=True):
    """
    Stage 2 (full matching) with discontinuity penalty.
    Decision at each block (i,j) chooses predecessor p in {up, left, diag} minimizing:
        (D[p] + C_chunk[i,j] + penalty) / (L[p] + L_chunk[i,j])
    where penalty = (C_chunk[i,j] / L_chunk[i,j]) * magn_discontinuity(p -> (i,j))
    Stores unnormalized totals:
        D_chunks[i,j] = numerator chosen (D[p] + C_chunk + penalty)
        L_chunks[i,j] = denominator chosen (L[p] + L_chunk)
    B_chunks stores predecessor coordinates (pi,pj) or (-1,-1) for origin.

    Inputs: tiled_result from stage1 (must contain 'blocks' with 'bi','bj','wp_global','rows','cols' and 'C_shape').
    Returns: dict containing C_chunk, L_chunk, D_chunks, L_chunks, B_chunks, ordered_blocks, avg_cost, stitched_wp, ...
    """
    blocks = tiled_result['blocks']
    if not blocks:
        raise ValueError("Stage 2: no blocks in tiled_result['blocks'].")

    n_row = max(b['bi'] for b in blocks) + 1
    n_col = max(b['bj'] for b in blocks) + 1
    have  = {(b['bi'], b['bj']) for b in blocks}
    by_ij = {(b['bi'], b['bj']): b for b in blocks}

    INF = 1e18
    C_chunk = np.full((n_row, n_col), np.nan, dtype=float)  
    L_chunk = np.full((n_row, n_col), np.nan, dtype=float)  

    for b in blocks:
        bi, bj = b['bi'], b['bj']

        raw = float(b['raw_cost'])

        plen = int(b['path_len'])

        C_chunk[bi, bj] = raw
        L_chunk[bi, bj] = max(plen, 1.0)

    if np.isnan(C_chunk).any():
        finite = C_chunk[np.isfinite(C_chunk)]
        if finite.size == 0:
            raise ValueError("Stage 2: all block costs are missing.")
        penalty_val = np.nanpercentile(C_chunk, 95) + 6*(np.nanmedian(np.abs(finite - np.nanmedian(finite))) + 1e-6)
        C_chunk = np.where(np.isfinite(C_chunk), C_chunk, penalty_val)
    if np.isnan(L_chunk).any():
        L_chunk = np.where(np.isfinite(L_chunk), L_chunk, 1.0)

    for b in blocks:
        wp = np.asarray(b['wp_global'])
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise ValueError("wp_global must be (N,2) for each block.")
        s_idx = int(np.argmin(wp[:, 0] + wp[:, 1]))
        e_idx = int(np.argmax(wp[:, 0] + wp[:, 1]))
        b['start'] = (int(wp[s_idx, 0]), int(wp[s_idx, 1]))
        b['end']   = (int(wp[e_idx, 0]), int(wp[e_idx, 1]))

    def magn_discontinuity(prev_cell, cur_cell): 
        """
        Manhattan distance between prev.end and cur.start (global coords).
        If either block missing, return large value (so path avoids it if possible).
        """
        if prev_cell not in by_ij or cur_cell not in by_ij:
            return float('inf')
        ie_prev, je_prev = by_ij[prev_cell]['end']
        is_cur, js_cur = by_ij[cur_cell]['start']
        return float(abs(is_cur - ie_prev) + abs(js_cur - je_prev))

    if (0, 0) not in have or (n_row-1, n_col-1) not in have:
        raise ValueError("Stage 2 full-matching requires blocks at (0,0) and (n_row-1,n_col-1).")

    D_chunks = np.full((n_row, n_col), INF, dtype=float)   
    L_chunks = np.full((n_row, n_col), 0.0, dtype=float)   
    B_chunks = np.full((n_row, n_col, 2), -1, dtype=int)   

    D_chunks[0, 0] = C_chunk[0, 0]   
    L_chunks[0, 0] = L_chunk[0, 0]
    B_chunks[0, 0] = [-1, -1]

    for i in range(n_row):
        for j in range(n_col):
            if i == 0 and j == 0:
                continue
            if (i, j) not in have:
                continue

            best_ratio = float('inf')
            best_choice = None  

            for (pi, pj) in ((i-1, j), (i, j-1), (i-1, j-1)):
                if pi < 0 or pj < 0:
                    continue
                if (pi, pj) not in have:
                    continue
                if not np.isfinite(D_chunks[pi, pj]) or D_chunks[pi, pj] >= INF/2:
                    continue

                base_ratio_factor = C_chunk[i, j] / max(L_chunk[i, j], 1e-12)
                mag = magn_discontinuity((pi, pj), (i, j))
                penalty = base_ratio_factor * mag

                num = D_chunks[pi, pj] + C_chunk[i, j] + penalty
                den = L_chunks[pi, pj] + L_chunk[i, j]
                ratio = num / max(den, 1.0)

                if ratio < best_ratio:
                    best_ratio = ratio
                    best_choice = (pi, pj, num, den)

            if best_choice is None:

                continue

            pi, pj, num, den = best_choice
            D_chunks[i, j] = num
            L_chunks[i, j] = den
            B_chunks[i, j] = [pi, pj]

    ei, ej = n_row - 1, n_col - 1

    ordered_blocks = []
    i, j = ei, ej
    while True:
        ordered_blocks.append((i, j))
        pi, pj = int(B_chunks[i, j, 0]), int(B_chunks[i, j, 1])
        if pi == -1 and pj == -1:
            break
        i, j = pi, pj
    ordered_blocks = ordered_blocks[::-1]  
    ordered_linear = [bi * n_col + bj for (bi, bj) in ordered_blocks]
    avg_cost = float(D_chunks[ei, ej] / max(L_chunks[ei, ej], 1.0))

    stitched_parts = []
    last = None
    for (bi, bj) in ordered_blocks:
        wp = np.asarray(by_ij[(bi, bj)]['wp_global'], dtype=float)
        if wp.size == 0:
            continue
        if last is None:
            stitched_parts.append(wp)
            last = (wp[-1, 0], wp[-1, 1])
        else:

            if np.allclose(stitched_parts[-1][-1], wp[0], atol=1e-8):
                stitched_parts.append(wp[1:])
            else:
                stitched_parts.append(wp)
            last = (wp[-1, 0], wp[-1, 1])

    stitched_wp = np.vstack(stitched_parts) if stitched_parts else np.zeros((0, 2), dtype=float)

    total_normalized_from_stitched = float('inf')
    stitched_raw_cost = None
    if stitched_wp.size:

        C_global = tiled_result.get('C', None)
        if C_global is not None:
            stitched_raw_cost = float(np.sum([C_global[int(i), int(j)] for (i, j) in stitched_wp.astype(int)]))
            si, sj = stitched_wp[0].astype(int)
            ei_g, ej_g = stitched_wp[-1].astype(int)
            mdist = abs(ei_g - si) + abs(ej_g - sj)
            total_normalized_from_stitched = stitched_raw_cost / mdist if mdist > 0 else float('inf')

    if show_fig:
        L1, L2 = tiled_result['C_shape']
        fig = go.Figure()

        for b in blocks:
            (r0, r1), (c0, c1) = b['rows'], b['cols']
            fig.add_shape(
                type="rect",
                x0=c0, x1=c1, y0=r0, y1=r1,
                line=dict(color="rgba(140,140,140,0.6)", width=1, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="below"
            )

        palette = pc.qualitative.Plotly
        for idx, b in enumerate(blocks):
            wp = np.asarray(b['wp_global'])
            if wp.size:
                fig.add_trace(go.Scatter(
                    x=wp[:, 1], y=wp[:, 0],
                    mode="lines",
                    line=dict(width=2, color=palette[idx % len(palette)]),
                    name=f"blk({b['bi']},{b['bj']}) path",
                    showlegend=False,
                    opacity=0.45
                ))

        centers_x, centers_y, labels = [], [], []
        for t, (bi, bj) in enumerate(ordered_blocks, start=1):
            b = by_ij[(bi, bj)]
            (r0, r1), (c0, c1) = b['rows'], b['cols']
            centers_x.append((c0 + c1) / 2.0)
            centers_y.append((r0 + r1) / 2.0)
            labels.append(str(t))
        if centers_x:
            fig.add_trace(go.Scatter(
                x=centers_x, y=centers_y,
                mode="markers+text",
                marker=dict(size=18, symbol="circle-open-dot"),
                text=labels, textposition="middle center",
                name=f"chunk order (avg={avg_cost:.3f})",
                hovertemplate="order=%{text}<br>center (j=%{x:.0f}, i=%{y:.0f})<extra></extra>"
            ))

        start_x, start_y, end_x, end_y = [], [], [], []
        for (bi, bj) in ordered_blocks:
            b = by_ij[(bi, bj)]
            (is_, js_), (ie, je) = b['start'], b['end']
            start_x.append(js_); start_y.append(is_)
            end_x.append(je);   end_y.append(ie)
        if start_x:
            fig.add_trace(go.Scatter(
                x=start_x, y=start_y, mode="markers",
                marker=dict(size=9, symbol="triangle-up", color="green"),
                name="block start (global)"
            ))
            fig.add_trace(go.Scatter(
                x=end_x, y=end_y, mode="markers",
                marker=dict(size=9, symbol="diamond", color="red"),
                name="block end (global)"
            ))

        if stitched_wp.size:
            seg_x, seg_y = [], []
            for k in range(len(ordered_blocks) - 1):
                b_cur = by_ij[ordered_blocks[k]]
                b_next = by_ij[ordered_blocks[k+1]]

                wp_cur = np.asarray(b_cur['wp_global'])
                if wp_cur.size == 0:
                    continue

                seg_x.extend(wp_cur[:, 1])
                seg_y.extend(wp_cur[:, 0])

                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y,
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name="Stitched DTW segment" if k == 0 else None,
                    showlegend=(k == 0)
                ))
                seg_x, seg_y = [], []

            b_last = by_ij[ordered_blocks[-1]]
            wp_last = np.asarray(b_last['wp_global'])
            if wp_last.size:
                fig.add_trace(go.Scatter(
                    x=wp_last[:, 1], y=wp_last[:, 0],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name=None,
                    showlegend=False
                ))

        if isinstance(tiled_result.get('full_global'), dict):
            fg = tiled_result['full_global']
            wp_full = fg.get('wp', None)
            best_full = fg.get('best_cost', None)
            if wp_full is not None:
                wp_tmp = wp_full
                if wp_tmp.ndim == 2 and wp_tmp.shape[0] == 2:
                    wp_tmp = wp_tmp.T
                step = max(1, len(wp_tmp) // 5000)
                fig.add_trace(go.Scatter(
                    x=wp_tmp[::step, 1], y=wp_tmp[::step, 0],
                    mode="lines",
                    line=dict(width=2, color="rgba(0,0,0,0.25)"),
                    name=f"Global DTW" + (f" (cost={best_full:.3f})" if best_full is not None else "")
                ))

        fig.update_layout(
            title="Stage 2: chunk DP with discontinuity penalty + stitched path",
            xaxis_title="F2 frame j (global)",
            yaxis_title="F1 frame i (global)",
            template="plotly_white", width=900, height=700, legend=dict(orientation="h")
        )
        fig.update_xaxes(range=[0, tiled_result['C_shape'][1]], showgrid=False)
        fig.update_yaxes(range=[0, tiled_result['C_shape'][0]], showgrid=False, scaleanchor="x", scaleratio=1)
        fig.show()

    return {
        'C_chunk': C_chunk,
        'C_global': C_global,
        'L_chunk': L_chunk,
        'D_chunks': D_chunks,
        'L_chunks': L_chunks,
        'B_chunks': B_chunks,
        'ordered_blocks': ordered_blocks,
        'ordered_linear': ordered_linear,
        'avg_cost': float(avg_cost),
        'stitched_wp': stitched_wp,
        'stitched_raw_cost': stitched_raw_cost,
        'stitched_normalized': float(total_normalized_from_stitched) if np.isfinite(total_normalized_from_stitched) else None,
        'n_row': n_row, 'n_col': n_col
    }
