import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

def endpoint_distance_algorithm(
    tiled_result,
    allow_diag=True,
    neighbor_radius=1,
    normalize=True,          # unused in this greedy version (kept for API)
    start_policy="auto",
    end_policy="auto",       # unused in greedy (kept for API)
    show_fig=True,
    show_full_path=True,
    stage="greedy"           # "greedy" (original) or "chunk_dp" (Stage 2 DP)
):
    """
    stage:
      - "greedy" : original greedy neighbor-based block ordering
      - "chunk_dp": chunk-level Dynamic Programming (D_chunks / B_chunks),
                    backtrace and stitch per-chunk optimal local paths.
    """

    blocks = tiled_result['blocks']
    if not blocks:
        raise ValueError("No blocks found in tiled_result['blocks'].")

    # ---- Block grid shape ----
    max_bi = max(b['bi'] for b in blocks)
    max_bj = max(b['bj'] for b in blocks)
    n_row, n_col = max_bi + 1, max_bj + 1

    have = {(b['bi'], b['bj']) for b in blocks}
    by_idx = {(b['bi'], b['bj']): b for b in blocks}

    # ---- Ensure start/end endpoints per block (global coords) ----
    for b in blocks:
        wp = b['wp_global']
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise ValueError("wp_global must be (N,2)")
        s_idx = int(np.argmin(wp[:, 0] + wp[:, 1]))
        e_idx = int(np.argmax(wp[:, 0] + wp[:, 1]))
        b['start'] = (int(wp[s_idx, 0]), int(wp[s_idx, 1]))  # (i_start, j_start)
        b['end']   = (int(wp[e_idx, 0]), int(wp[e_idx, 1]))  # (i_end,   j_end)

    # ---- helper: magn_discontinuity between two chunk cells ----
    def magn_discontinuity(prev_cell, cur_cell):
        """
        Magnitude of discontinuity between blocks (prev -> cur).
        We compute Euclidean distance between prev.end and cur.start in global coords.
        """
        if prev_cell not in by_idx or cur_cell not in by_idx:
            return np.inf
        ie_prev, je_prev = by_idx[prev_cell]['end']
        is_cur, js_cur = by_idx[cur_cell]['start']
        di = is_cur - ie_prev
        dj = js_cur - je_prev
        return float((di * di + dj * dj) ** 0.5)

    # ---- neighbors function (used by greedy stage) ----
    def neighbors(bi, bj):
        nbrs = []
        for di in range(0, neighbor_radius + 1):
            for dj in range(0, neighbor_radius + 1):
                if di == 0 and dj == 0:
                    continue
                if not allow_diag and (di != 0 and dj != 0):
                    continue
                bii, bjj = bi + di, bj + dj
                if (bii, bjj) in have:
                    nbrs.append((bii, bjj))
        return nbrs

    # -----------------
    # Stage: chunk-level DP (Stage 2) OR original greedy
    # -----------------
    if stage == "chunk_dp":
        # Build D_chunks and B_chunks (shape n_row x n_col)
        INF = float("inf")
        D_chunks = np.full((n_row, n_col), INF, dtype=float)
        B_chunks = np.full((n_row, n_col, 2), -1, dtype=int)  # store predecessor coords

        # For chunk local cost use block['best_cost'] if present, else 0
        def chunk_local_cost(cell):
            return float(by_idx[cell].get('best_cost', 0.0))

        # DP iteration row-major (0..max_bi) x (0..max_bj)
        for i in range(n_row):
            for j in range(n_col):
                cell = (i, j)
                if cell not in have:
                    continue  # unreachable block cell
                local_cost = chunk_local_cost(cell)
                if i == 0 and j == 0:
                    # origin cell initialization
                    D_chunks[i, j] = local_cost
                    B_chunks[i, j] = [-1, -1]
                    continue

                # consider three predecessors: up (i-1,j), left (i,j-1), diag (i-1,j-1)
                candidates = []
                if i - 1 >= 0 and (i - 1, j) in have and np.isfinite(D_chunks[i - 1, j]):
                    mag = magn_discontinuity((i - 1, j), cell)
                    candidates.append(((i - 1, j), D_chunks[i - 1, j] + mag + local_cost))
                if j - 1 >= 0 and (i, j - 1) in have and np.isfinite(D_chunks[i, j - 1]):
                    mag = magn_discontinuity((i, j - 1), cell)
                    candidates.append(((i, j - 1), D_chunks[i, j - 1] + mag + local_cost))
                if i - 1 >= 0 and j - 1 >= 0 and (i - 1, j - 1) in have and np.isfinite(D_chunks[i - 1, j - 1]):
                    mag = magn_discontinuity((i - 1, j - 1), cell)
                    candidates.append(((i - 1, j - 1), D_chunks[i - 1, j - 1] + mag + local_cost))

                if not candidates:
                    # no valid predecessors: leave INF (unreachable)
                    continue

                # pick min
                pred_cell, best_val = min(candidates, key=lambda x: x[1])
                D_chunks[i, j] = best_val
                B_chunks[i, j] = [pred_cell[0], pred_cell[1]]

        # After DP: backtrace from (max_bi, max_bj)
        goal = (max_bi, max_bj)
        if goal not in have or not np.isfinite(D_chunks[goal]):
            # If the strict (0,0)->(max_bi,max_bj) path isn't available, attempt to find
            # the best reachable cell on the bottom or right edges (flexible fallback).
            # But since Stage 2 "only handles full matching" - we will raise with diagnostic
            reachable_goal = None
            # try fallback: any cell on bottom row OR rightmost column with finite D
            best_fall_val = INF
            for (i, j) in have:
                if (i == max_bi or j == max_bj) and np.isfinite(D_chunks[i, j]) and D_chunks[i, j] < best_fall_val:
                    best_fall_val = D_chunks[i, j]; reachable_goal = (i, j)
            if reachable_goal is None:
                raise ValueError(f"No DP chunk-level path found to ({max_bi},{max_bj}) and no fallback reachable edge cells.")
            else:
                goal = reachable_goal  # use fallback reachable endpoint

        # Backtrace chunk path
        ordered_blocks = []
        cur = goal
        while True:
            ordered_blocks.append(cur)
            pi, pj = B_chunks[cur]
            if pi == -1 and pj == -1:
                break
            cur = (int(pi), int(pj))
        ordered_blocks = ordered_blocks[::-1]  # reverse to go origin->goal

        # Build ordered_linear and path_nodes
        ordered_linear = [bi * n_col + bj for (bi, bj) in ordered_blocks]
        path_nodes = np.array(ordered_blocks, dtype=int)
        total_cost = float(D_chunks[goal])

        # ---- Stitch per-chunk local paths (wp_global) along the chunk sequence ----
        stitched_wp_list = []
        for idx, (bi, bj) in enumerate(ordered_blocks):
            b = by_idx[(bi, bj)]
            wp = np.asarray(b['wp_global'], dtype=float)
            if wp.size == 0:
                continue
            if len(stitched_wp_list) == 0:
                stitched_wp_list.append(wp)
            else:
                # ensure we don't duplicate the joining point if identical (or extremely close)
                prev_wp = stitched_wp_list[-1]
                if np.allclose(prev_wp[-1], wp[0], atol=1e-8):
                    stitched_wp_list.append(wp[1:])   # drop duplicate first row
                else:
                    # If there is a gap, we still append the new chunk's path.
                    # (Optionally could insert interpolated connector points.)
                    stitched_wp_list.append(wp)

        if stitched_wp_list:
            stitched_wp = np.vstack(stitched_wp_list)
        else:
            stitched_wp = np.zeros((0, 2), dtype=float)

        # Optionally attach stitched_wp to tiled_result for plotting or later use
        tiled_result = dict(tiled_result)  # shallow copy to avoid mutating original
        tiled_result['stitched_chunk_wp'] = stitched_wp

        # Plotting similar to original (but show chunk DP order)
        if show_fig:
            L1, L2 = tiled_result['C_shape']
            fig = go.Figure()

            # 1) All tile boxes (thin dashed gray)
            for b in blocks:
                (r0, r1), (c0, c1) = b['rows'], b['cols']
                fig.add_shape(
                    type="rect",
                    x0=c0, x1=c1, y0=r0, y1=r1,
                    line=dict(color="rgba(140,140,140,0.6)", width=1, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below"
                )

            # 2) Overlay ALL blocks' local paths (global coords)
            palette = pc.qualitative.Plotly
            for idx, b in enumerate(blocks):
                wp = b['wp_global']
                fig.add_trace(go.Scatter(
                    x=wp[:, 1], y=wp[:, 0],
                    mode="lines",
                    line=dict(width=2, color=palette[idx % len(palette)]),
                    name=f"blk({b['bi']},{b['bj']}) path",
                    showlegend=False,
                    hovertemplate=f"blk({b['bi']},{b['bj']}) path<extra></extra>",
                    opacity=0.5
                ))

            # 3) Numbered markers (order 1..K) at block centers from DP order
            centers_x, centers_y, labels = [], [], []
            for t, (bi, bj) in enumerate(ordered_blocks, start=1):
                b = by_idx[(bi, bj)]
                (r0, r1), (c0, c1) = b['rows'], b['cols']
                centers_x.append((c0 + c1) / 2.0)
                centers_y.append((r0 + r1) / 2.0)
                labels.append(str(t))
            fig.add_trace(go.Scatter(
                x=centers_x, y=centers_y,
                mode="markers+text",
                marker=dict(size=18, symbol="circle-open-dot"),
                text=labels, textposition="middle center",
                name=f"chunk order (DP cost={total_cost:.3f})",
                hovertemplate="order=%{text}<br>center (j=%{x:.0f}, i=%{y:.0f})<extra></extra>"
            ))

            # 4) Highlight endpoints (no connecting lines)
            start_x, start_y, end_x, end_y = [], [], [], []
            for (bi, bj) in ordered_blocks:
                b = by_idx[(bi, bj)]
                (is_, js_), (ie, je) = b['start'], b['end']
                start_x.append(js_); start_y.append(is_)
                end_x.append(je);   end_y.append(ie)

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

            # 5) Stitched chunk-level path overlay (thick black)
            if stitched_wp.size:
                fig.add_trace(go.Scatter(
                    x=stitched_wp[:, 1], y=stitched_wp[:, 0],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name=f"Stitched chunk DTW path (DP cost={total_cost:.3f})"
                ))

            # 6) Optional global DTW overlay (if provided)
            if show_full_path and isinstance(tiled_result.get('full_global'), dict):
                wp = tiled_result['full_global'].get('wp', None)
                best = tiled_result['full_global'].get('best_cost', None)
                if wp is not None:
                    step = max(1, len(wp) // 5000)
                    fig.add_trace(go.Scatter(
                        x=wp[::step, 1], y=wp[::step, 0],
                        mode="lines",
                        line=dict(width=2, color="rgba(0,0,0,0.3)"),
                        name=f"Global DTW" + (f" (cost={best:.3f})" if best is not None else "")
                    ))

            fig.update_layout(
                title="Chunk-level DP order + local paths + stitched chunk path",
                xaxis_title="F2 frame j (global)",
                yaxis_title="F1 frame i (global)",
                template="plotly_white",
                width=900, height=700,
                legend=dict(orientation="h")
            )
            fig.update_xaxes(range=[0, tiled_result['C_shape'][1]], showgrid=False)
            fig.update_yaxes(range=[0, tiled_result['C_shape'][0]], showgrid=False, scaleanchor="x", scaleratio=1)
            fig.show()

        return {
            'ordered_blocks': ordered_blocks,
            'ordered_linear': ordered_linear,
            'path_nodes': path_nodes,
            'dp_cost': total_cost,
            'D_chunks': D_chunks,
            'B_chunks': B_chunks,
            'stitched_wp': stitched_wp,
            'n_row': n_row, 'n_col': n_col
        }

    else:
        # -----------------
        # Original greedy behaviour (preserved; mostly unchanged)
        # -----------------
        if isinstance(start_policy, tuple) and start_policy[0] == "fixed" and start_policy[1] in have:
            cur = start_policy[1]
        else:
            # valid start blocks = bottom edge (bi == max_bi) OR left edge (bj == 0)
            valid_starts = [b for b in blocks if (b['bi'] == max_bi or b['bj'] == 0)]

            if not valid_starts:
                raise ValueError("No valid start blocks on bottom or left edge!")

            # among them, pick block with minimum normalized cost
            # (use cost per element of block, to avoid size bias)
            def norm_cost(b):
                R = b['rows'][1] - b['rows'][0]
                Cw = b['cols'][1] - b['cols'][0]
                return b['best_cost'] / max(R * Cw, 1)

            pick = min(valid_starts, key=norm_cost)
            cur = (pick['bi'], pick['bj'])

        ordered_blocks = []
        visited = set()
        total_dist = 0.0

        while True:
            # append current matrix to ordered blocks+set of visited
            ordered_blocks.append(cur)
            visited.add(cur)
            
            # 2. get possible neighbers
            cand = [v for v in neighbors(*cur) if v not in visited]
            if not cand:
                break

            # helper to get distance between end points
            ie, je = by_idx[cur]['end']  # current block end (i,j)

            def dist_to(v):
                is_, js_ = by_idx[v]['start']  # candidate start (i,j)
                # move type in block grid
                di_block = v[0] - cur[0]
                dj_block = v[1] - cur[1]

                if di_block == 0 and dj_block > 0:
                    # RIGHT move: prioritize continuity along j (x-axis)
                    return abs(js_ - je)
                elif di_block > 0 and dj_block == 0:
                    # DOWN move: prioritize continuity along i (y-axis)
                    return abs(is_ - ie)
                else:
                    # DIAGONAL (or larger jumps if neighbor_radius>1): use Euclidean
                    di = is_ - ie
                    dj = js_ - je
                    return float((di*di + dj*dj) ** 0.5)
            # 4. pick nearest neighbor
            nxt = min(cand, key=dist_to)
            
            d = dist_to(nxt)
            if not np.isfinite(d):         # stuck
                break

            total_dist += d
            cur = nxt

        ordered_linear = [bi * n_col + bj for (bi, bj) in ordered_blocks]
        total_cost = total_dist  # naming for display/return

        # ---- Plotly: all local paths + numbered block order + endpoints ----
        if show_fig:
            L1, L2 = tiled_result['C_shape']
            fig = go.Figure()

            # 1) All tile boxes (thin dashed gray)
            for b in blocks:
                (r0, r1), (c0, c1) = b['rows'], b['cols']
                fig.add_shape(
                    type="rect",
                    x0=c0, x1=c1, y0=r0, y1=r1,
                    line=dict(color="rgba(140,140,140,0.6)", width=1, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below"
                )

            # 2) Overlay ALL blocks' local paths (global coords)
            palette = pc.qualitative.Plotly
            for idx, b in enumerate(blocks):
                wp = b['wp_global']
                fig.add_trace(go.Scatter(
                    x=wp[:, 1], y=wp[:, 0],
                    mode="lines",
                    line=dict(width=2, color=palette[idx % len(palette)]),
                    name=f"blk({b['bi']},{b['bj']}) path",
                    showlegend=False,
                    hovertemplate=f"blk({b['bi']},{b['bj']}) path<extra></extra>",
                    opacity=0.6
                ))

            # 3) Numbered markers (order 1..K) at block centers
            centers_x, centers_y, labels = [], [], []
            for t, (bi, bj) in enumerate(ordered_blocks, start=1):
                b = by_idx[(bi, bj)]
                (r0, r1), (c0, c1) = b['rows'], b['cols']
                centers_x.append((c0 + c1) / 2.0)
                centers_y.append((r0 + r1) / 2.0)
                labels.append(str(t))
            fig.add_trace(go.Scatter(
                x=centers_x, y=centers_y,
                mode="markers+text",
                marker=dict(size=18, symbol="circle-open-dot"),
                text=labels, textposition="middle center",
                name=f"block order (greedy dist={total_cost:.3f})",
                hovertemplate="order=%{text}<br>center (j=%{x:.0f}, i=%{y:.0f})<extra></extra>"
            ))

            # 4) Highlight endpoints (no connecting lines)
            start_x, start_y, end_x, end_y = [], [], [], []
            for (bi, bj) in ordered_blocks:
                b = by_idx[(bi, bj)]
                (is_, js_), (ie, je) = b['start'], b['end']
                start_x.append(js_); start_y.append(is_)
                end_x.append(je);   end_y.append(ie)

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

            # 5) Optional global DTW overlay
            if show_full_path and isinstance(tiled_result.get('full_global'), dict):
                wp = tiled_result['full_global'].get('wp', None)
                best = tiled_result['full_global'].get('best_cost', None)
                if wp is not None:
                    step = max(1, len(wp) // 5000)
                    fig.add_trace(go.Scatter(
                        x=wp[::step, 1], y=wp[::step, 0],
                        mode="lines",
                        line=dict(width=3, color="black"),
                        name=f"Global DTW" + (f" (cost={best:.3f})" if best is not None else "")
                    ))

            fig.update_layout(
                title="Block order (numbered) + local paths + endpoint highlights",
                xaxis_title="F2 frame j (global)",
                yaxis_title="F1 frame i (global)",
                template="plotly_white",
                width=900, height=700,
                legend=dict(orientation="h")
            )
            fig.update_xaxes(range=[0, L2], showgrid=False)
            fig.update_yaxes(range=[0, L1], showgrid=False, scaleanchor="x", scaleratio=1)
            fig.show()

        return {
            'ordered_blocks': ordered_blocks,
            'ordered_linear': ordered_linear,
            'path_nodes': np.array(ordered_blocks, dtype=int),
            'dp_cost': float(total_cost),      # greedy distance
            'n_row': n_row, 'n_col': n_col
        }
