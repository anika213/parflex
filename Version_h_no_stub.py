import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

import numpy as np
import pickle
import plotly.colors as pc
import plotly.graph_objects as go

def align_system(
    system,
    F1,
    F2,
    outfile=None,
    L_block='auto',         # tile size along both axes (frames). Use 'auto' or None to choose 2-5 blocks
    min_block=5,            # skip tiny ragged tiles smaller than this on any side
    plot_overlay=True,      # make a single global overlay figure
    show_block_boxes=True,  # draw dashed rectangles for each tile
    overlay_full=True       # overlay full-matrix FlexDTW path
):
    """
    Tile the FULL cost matrix C into L_block × L_block blocks (ragged edges handled),
    run FlexDTW on EACH block, store GLOBAL (i,j) paths, and visualize in Plotly.

    Returns:
      C, stage1_result
    where stage1_result contains the standardized fields required by Stage 2:
      - 'C_shape': (L1, L2) - Global cost matrix dimensions
      - 'L_block': L_block - Chunk size (LxL)
      - 'blocks': List of chunk results with:
          * 'bi', 'bj': Chunk grid position
          * 'rows': (r0, r1) - Global row range
          * 'cols': (c0, c1) - Global col range
          * 'best_cost': float - Unnormalized total path cost
          * 'wp_global': np.array - Optimal FlexDTW path in global coords (N, 2)
          * 'path_length': int - Manhattan distance of the path
          * 'start_local': (i, j) - Path start in LOCAL chunk coordinates
          * 'end_local': (i, j) - Path end in LOCAL chunk coordinates
          * 'path_cache': dict - Maps end positions to {cost, length, path} for stubs
      - 'full_global': {...} - Optional reference global DTW
      - Additional fields: 'C', 'auto_info', standardized chunk arrays
    """
    assert system == 'flexdtw', "This implementation targets 'flexdtw' only."

    # ---------- Build cost matrix once ----------
    L1 = F1.shape[1]
    L2 = F2.shape[1]
    if L1 == 0 or L2 == 0:
        raise ValueError("Empty features: F1 or F2 has zero length.")

    # --- Auto-select L_block to avoid tiny slivers if requested ---
    if L_block in (None, 'auto'):
        best = None  # (metric, n_row, n_col, chosen_block)
        for n_row_try in range(2, 6):      # try 2..5 blocks along rows
            for n_col_try in range(2, 6):  # try 2..5 blocks along cols
                r_size = int(np.ceil(L1 / n_row_try))
                c_size = int(np.ceil(L2 / n_col_try))
                pad_rows = r_size * n_row_try - L1
                pad_cols = c_size * n_col_try - L2
                total_pad = pad_rows + pad_cols
                smallest_tile = min(r_size, c_size)
                metric = (total_pad, -smallest_tile, n_row_try * n_col_try, max(r_size, c_size))
                chosen_block = max(r_size, c_size)
                if best is None or metric < best[0]:
                    best = (metric, n_row_try, n_col_try, chosen_block)

        if best is not None:
            _, n_row_best, n_col_best, L_block_chosen = best
            L_block = max(int(L_block_chosen), int(min_block))
            auto_info = {
                'auto_n_row': n_row_best,
                'auto_n_col': n_col_best,
                'auto_L_block': L_block
            }
        else:
            L_block = max(4000, min_block)
            auto_info = {'auto_fallback': True}
    else:
        auto_info = None

    F1n = FlexDTW.L2norm(F1)      # (D, L1)
    F2n = FlexDTW.L2norm(F2)      # (D, L2)
    C = 1.0 - F1n.T @ F2n         # (L1, L2), cosine distance

    # ---------- Optional full-matrix FlexDTW (for overlay) ----------
    full_global = {'best_cost': None, 'wp': None}
    if overlay_full:
        beta_full = other_params['flexdtw']['beta']
        buffer_full = min(L1, L2) * (1 - (1 - beta_full) * min(L1, L2) / max(L1, L2))
        best_cost_full, wp_full, debug_full = FlexDTW.flexdtw(
            C, steps=steps['flexdtw'], weights=weights['flexdtw'], buffer=buffer_full
        )
        if wp_full is not None and wp_full.ndim == 2 and wp_full.shape[0] == 2:
            wp_full = wp_full.T
        full_global = {'best_cost': float(best_cost_full) if best_cost_full is not None else None,
                       'wp': wp_full}

    # ---------- Tile C into blocks ----------
    n_row = (L1 + L_block - 1) // L_block   # ceil
    n_col = (L2 + L_block - 1) // L_block   # ceil

    # prepare standardized per-chunk containers
    C_chunk = np.full((n_row, n_col), np.inf, dtype=float)
    pathLength_chunk = np.zeros((n_row, n_col), dtype=float)
    pathStart_chunk = np.full((n_row, n_col, 2), np.nan, dtype=float)
    pathEnd_chunk = np.full((n_row, n_col, 2), np.nan, dtype=float)
    wp_chunk = {}  # dict keyed by (bi,bj) -> (N,2) np.array

    blocks = []
    for bi in range(n_row):
        r0 = bi * L_block
        r1 = min((bi + 1) * L_block, L1)
        for bj in range(n_col):
            c0 = bj * L_block
            c1 = min((bj + 1) * L_block, L2)

            block = C[r0:r1, c0:c1]
            R, Cw = block.shape
            if R < min_block or Cw < min_block:
                # Skip tiny ragged tiles that tend to produce trivial 2-point paths
                continue

            # Per-block buffer (scale to local sizes)
            beta = other_params['flexdtw']['beta']
            buffer_blk = min(R, Cw) * (1 - (1 - beta) * min(R, Cw) / max(R, Cw))

            # Run FlexDTW on the block
            best_cost_blk, wp_local, debug_blk = FlexDTW.flexdtw(
                block, steps=steps['flexdtw'], weights=weights['flexdtw'], buffer=buffer_blk
            )
            if wp_local is not None and wp_local.ndim == 2 and wp_local.shape[0] == 2:
                wp_local = wp_local.T

            # raw unnormalized cost by summing the cost entries at path indices
            raw_cost_blk = float(block[wp_local[:, 0], wp_local[:, 1]].sum())
            # manhattan path length: sum over (abs di + abs dj)
            diffs = np.abs(np.diff(wp_local, axis=0))
            path_len_blk = int(diffs.sum())

            # Map local (i,j) -> GLOBAL indices
            wp_global = np.column_stack([wp_local[:, 0] + r0, wp_local[:, 1] + c0])

            # Extract start/end in LOCAL coordinates
            start_local = (int(wp_local[0, 0]), int(wp_local[0, 1]))
            end_local = (int(wp_local[-1, 0]), int(wp_local[-1, 1]))

            # Build path_cache for stub lookups
            # For now, store the full path with its endpoint as the key
            # Stage 2 can use this to find paths ending at specific positions
            path_cache = {
                end_local: {
                    'cost': raw_cost_blk,
                    'length': path_len_blk,
                    'path': wp_local.copy()  # Store local coordinates
                }
            }

            # Create block dictionary with Stage 2 compatible format
            block_dict = {
                'bi': bi,
                'bj': bj,
                'rows': (r0, r1),
                'cols': (c0, c1),
                'best_cost': raw_cost_blk,
                'wp_global': wp_global,
                'path_length': path_len_blk,
                'start_local': start_local,
                'end_local': end_local,
                'path_cache': path_cache
            }
            blocks.append(block_dict)

            # fill standardized per-chunk containers
            C_chunk[bi, bj] = raw_cost_blk
            pathLength_chunk[bi, bj] = float(path_len_blk) if path_len_blk > 0 else 0.0
            pathStart_chunk[bi, bj, :] = wp_global[0, :]
            pathEnd_chunk[bi, bj, :] = wp_global[-1, :]
            wp_chunk[(bi, bj)] = wp_global

    # ---------- Plotly overlay ----------
    if plot_overlay:
        fig = go.Figure()

        # (Optional) draw dashed rectangles for each tile
        if show_block_boxes:
            for b in blocks:
                (r0, r1), (c0, c1) = b['rows'], b['cols']
                fig.add_shape(
                    type="rect",
                    x0=c0, x1=c1, y0=r0, y1=r1,
                    line=dict(color="rgba(120,120,120,0.8)", width=1, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below"
                )

        # Per-block paths
        palette = pc.qualitative.Plotly  # good default categorical palette
        for idx, b in enumerate(blocks):
            wp = b['wp_global']
            fig.add_trace(go.Scatter(
                x=wp[:, 1], y=wp[:, 0],
                mode="lines",
                name=f"blk ({b['bi']},{b['bj']})",
                line=dict(width=2, color=palette[idx % len(palette)]),
                hovertemplate=("blk (%{customdata[0]},%{customdata[1]})<br>"
                               "j=%{x}, i=%{y}<extra></extra>"),
                customdata=np.tile([b['bi'], b['bj']], (wp.shape[0], 1))
            ))

        # Overlay full-matrix path (bold black)
        if overlay_full and full_global['wp'] is not None:
            wp = full_global['wp']
            step = max(1, len(wp) // 5000)
            fig.add_trace(go.Scatter(
                x=wp[::step, 1], y=wp[::step, 0],
                mode="lines",
                name=f"Global DTW (cost={full_global['best_cost']:.3f})",
                line=dict(color="black", width=3),
                opacity=0.95
            ))

        # optionally annotate auto info
        if auto_info is not None:
            ann_text = f"auto L_block={auto_info.get('auto_L_block', '?')}"
            if 'auto_n_row' in auto_info:
                ann_text += f", n_row={auto_info['auto_n_row']}, n_col={auto_info['auto_n_col']}"
            fig.add_annotation(x=0.99 * L2, y=0.01 * L1,
                               text=ann_text, showarrow=False, xanchor="right", yanchor="bottom",
                               bgcolor="rgba(255,255,255,0.7)")

        fig.update_layout(
            title="All block DTW paths (global coords)" + (
                " + Global path" if overlay_full and full_global['wp'] is not None else ""
            ),
            xaxis_title="F2 frame j (global)",
            yaxis_title="F1 frame i (global)",
            width=900, height=700,
            template="plotly_white",
            legend=dict(orientation="h")
        )
        fig.update_xaxes(range=[0, L2], showgrid=False)
        fig.update_yaxes(range=[0, L1], showgrid=False, scaleanchor="x", scaleratio=1)
        fig.show()

    # ---------- Persist & standardized stage1 result ----------
    stage1_result = {
        'C_shape': (L1, L2),
        'L_block': L_block,
        'blocks': blocks,
        'full_global': full_global,
        'C': C,
        'auto_info': auto_info,
        # standardized outputs for Stage 2
        'C_chunk': C_chunk,
        'pathLength_chunk': pathLength_chunk,
        'pathStart_chunk': pathStart_chunk,
        'pathEnd_chunk': pathEnd_chunk,
        'wp_chunk': wp_chunk,
        'n_row': n_row,
        'n_col': n_col
    }

    result = stage1_result  # keep same name for backward compatibility
    if outfile:
        pickle.dump(result, open(outfile, 'wb'))

    return C, result

def endpoint_distance_algorithm_v2(
    tiled_result,
    allow_diag=True,
    show_fig=True,
    show_full_path=True,
):
    """
    Stage 2: Chunk-level Dynamic Programming with normalized scoring.

    Now uses bottom-left as (0,0) origin.

    Path must start from bottom edge (bi=0) or left edge (bj=0).
    Path must end at top edge (bi=max_bi) or right edge (bj=max_bj).

    Uses normalized cost (total_cost / total_length) for DP decisions.
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

    # ---- Ensure required fields in each block ----
    for b in blocks:
        if 'best_cost' not in b:
            raise ValueError(f"Block ({b['bi']},{b['bj']}) missing 'best_cost'")
        if 'path_length' not in b:
            raise ValueError(f"Block ({b['bi']},{b['bj']}) missing 'path_length'")
        if 'wp_global' not in b:
            raise ValueError(f"Block ({b['bi']},{b['bj']}) missing 'wp_global'")

        wp = np.asarray(b['wp_global'])
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise ValueError(f"Block ({b['bi']},{b['bj']}) wp_global must be (N,2)")

        b['start_global'] = tuple(wp[0].astype(int))
        b['end_global'] = tuple(wp[-1].astype(int))

    # ---- Helper: magnitude of discontinuity ----
    def magn_discontinuity(prev_cell, cur_cell):
        if prev_cell not in by_idx or cur_cell not in by_idx:
            return np.inf
        ie_prev, je_prev = by_idx[prev_cell]['end_global']
        is_cur, js_cur = by_idx[cur_cell]['start_global']
        di = is_cur - ie_prev
        dj = js_cur - je_prev
        return float((di * di + dj * dj) ** 0.5)

    # ---- Adjusted start/end chunk rules ----
    def is_valid_start_chunk(bi, bj):
        """Valid start: on left edge (bj=0) or bottom edge (bi=0)"""
        return (bj == 0) or (bi == 0)

    def is_valid_end_chunk(bi, bj):
        """Valid end: on top edge (bi=max_bi) or right edge (bj=max_bj)"""
        return (bi == max_bi) or (bj == max_bj)

    def is_on_start_boundary(i_global, j_global):
        """Check if position is on bottom (i=0) or left (j=0) edge of global matrix"""
        return (i_global == 0) or (j_global == 0)

    # ---- DP Initialization ----
    INF = float('inf')
    D_chunks = np.full((n_row, n_col), INF)
    L_chunks = np.full((n_row, n_col), INF)
    B_chunks = np.full((n_row, n_col, 2), -1, dtype=int)

    # ---- DP Iteration (bottom-up) ----
    for i in range(n_row):
        for j in range(n_col):
            cell = (i, j)
            if cell not in have:
                continue
            b = by_idx[cell]
            C_curr = b['best_cost']
            L_curr = b['path_length']
            is_start_chunk = is_valid_start_chunk(i, j)
            candidates = []

            # From below (i-1, j)
            if i > 0 and (i-1, j) in have and np.isfinite(D_chunks[i-1, j]):
                prev_cell = (i-1, j)
                mag = magn_discontinuity(prev_cell, cell)
                penalty = (C_curr / L_curr) * mag
                unnorm_cost = D_chunks[i-1, j] + C_curr + penalty
                path_len = L_chunks[i-1, j] + L_curr
                norm_score = unnorm_cost / path_len
                candidates.append((prev_cell, norm_score, unnorm_cost, path_len))

            # From left (i, j-1)
            if j > 0 and (i, j-1) in have and np.isfinite(D_chunks[i, j-1]):
                prev_cell = (i, j-1)
                mag = magn_discontinuity(prev_cell, cell)
                penalty = (C_curr / L_curr) * mag
                unnorm_cost = D_chunks[i, j-1] + C_curr + penalty
                path_len = L_chunks[i, j-1] + L_curr
                norm_score = unnorm_cost / path_len
                candidates.append((prev_cell, norm_score, unnorm_cost, path_len))

            # Diagonal (i-1, j-1)
            if allow_diag and not is_start_chunk:
                if i > 0 and j > 0 and (i-1, j-1) in have and np.isfinite(D_chunks[i-1, j-1]):
                    prev_cell = (i-1, j-1)
                    mag = magn_discontinuity(prev_cell, cell)
                    penalty = (C_curr / L_curr) * mag
                    unnorm_cost = D_chunks[i-1, j-1] + C_curr + penalty
                    path_len = L_chunks[i-1, j-1] + L_curr
                    norm_score = unnorm_cost / path_len
                    candidates.append((prev_cell, norm_score, unnorm_cost, path_len))

            # Start chunk
            if is_start_chunk:
                i_start, j_start = b['start_global']
                if is_on_start_boundary(i_start, j_start):
                    start_penalty = 0.0
                    stub_length = 0.0
                else:
                    stub_mag = 0.0
                    start_penalty = (C_curr / L_curr) * stub_mag
                    stub_length = stub_mag
                unnorm_cost = C_curr + start_penalty
                path_len = L_curr + stub_length
                norm_score = unnorm_cost / path_len if path_len > 0 else INF
                candidates.append((None, norm_score, unnorm_cost, path_len))

            if not candidates:
                continue
            best_prev, best_norm, best_unnorm, best_len = min(candidates, key=lambda x: x[1])
            D_chunks[i, j] = best_unnorm
            L_chunks[i, j] = best_len
            B_chunks[i, j] = [-1, -1] if best_prev is None else [best_prev[0], best_prev[1]]

    # ---- Find best endpoint ----
    valid_endpoints = [(i, j) for (i, j) in have if is_valid_end_chunk(i, j)]
    if not valid_endpoints:
        raise ValueError("No valid endpoint chunks found!")

    best_goal, best_goal_score = None, INF
    for cell in valid_endpoints:
        if np.isfinite(D_chunks[cell]):
            norm_score = D_chunks[cell] / L_chunks[cell]
            if norm_score < best_goal_score:
                best_goal_score = norm_score
                best_goal = cell

    if best_goal is None:
        raise ValueError("No reachable path!")

    # ---- Backtrace ----
    ordered_blocks = []
    cur = best_goal
    while True:
        ordered_blocks.append(cur)
        pi, pj = B_chunks[cur]
        if pi == -1 and pj == -1:
            break
        cur = (int(pi), int(pj))
    ordered_blocks = ordered_blocks[::-1]

    # ---- Stitch paths together ----
    stitched_wp_list = []
    for idx, (bi, bj) in enumerate(ordered_blocks):
        b = by_idx[(bi, bj)]
        wp = np.asarray(b['wp_global'], dtype=float)
        if wp.size == 0:
            continue
        
        if len(stitched_wp_list) == 0:
            stitched_wp_list.append(wp)
        else:
            # Check for duplicate/overlapping points at junction
            prev_wp = stitched_wp_list[-1]
            if np.allclose(prev_wp[-1], wp[0], atol=1e-8):
                stitched_wp_list.append(wp[1:])  # Skip duplicate first point
            else:
                # Gap exists - still append
                stitched_wp_list.append(wp)
    
    stitched_wp = np.vstack(stitched_wp_list) if stitched_wp_list else np.zeros((0, 2))

    # Store stitched path in result
    tiled_result = dict(tiled_result)
    tiled_result['stitched_chunk_wp'] = stitched_wp

    # Final metrics
    total_unnorm_cost = D_chunks[best_goal]
    total_path_length = L_chunks[best_goal]
    normalized_cost = total_unnorm_cost / total_path_length

    if show_fig:
        L1, L2 = tiled_result['C_shape']
        
        # ============ FIGURE 1: DP Matrices Heatmaps ============
        fig1 = go.Figure()
        
        # Create text annotations for all matrices
        hover_text_D = []
        hover_text_L = []
        hover_text_norm = []
        hover_text_B = []
        
        for i in range(n_row):
            row_D = []
            row_L = []
            row_norm = []
            row_B = []
            for j in range(n_col):
                # D_chunks values
                if (i, j) in have and np.isfinite(D_chunks[i, j]):
                    row_D.append(f"D[{i},{j}]={D_chunks[i,j]:.2f}")
                    row_L.append(f"L[{i},{j}]={L_chunks[i,j]:.1f}")
                    norm_val = D_chunks[i,j] / L_chunks[i,j] if L_chunks[i,j] > 0 else np.inf
                    row_norm.append(f"Norm[{i},{j}]={norm_val:.4f}")
                    bi_back, bj_back = B_chunks[i, j]
                    if bi_back == -1:
                        row_B.append(f"B[{i},{j}]=START")
                    else:
                        row_B.append(f"B[{i},{j}]=({bi_back},{bj_back})")
                else:
                    row_D.append(f"[{i},{j}]=N/A")
                    row_L.append(f"[{i},{j}]=N/A")
                    row_norm.append(f"[{i},{j}]=N/A")
                    row_B.append(f"[{i},{j}]=N/A")
            hover_text_D.append(row_D)
            hover_text_L.append(row_L)
            hover_text_norm.append(row_norm)
            hover_text_B.append(row_B)
        
        # D_chunks heatmap
        D_display = np.where(np.isinf(D_chunks), np.nan, D_chunks)
        fig1.add_trace(go.Heatmap(
            z=D_display,
            text=hover_text_D,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorscale='Viridis',
            name='D_chunks',
            visible=True
        ))
        
        # L_chunks heatmap
        L_display = np.where(np.isinf(L_chunks), np.nan, L_chunks)
        fig1.add_trace(go.Heatmap(
            z=L_display,
            text=hover_text_L,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorscale='Blues',
            name='L_chunks',
            visible=False
        ))
        
        # Normalized cost heatmap
        norm_display = np.where(np.isinf(L_chunks), np.nan, 
                                np.where(L_chunks > 0, D_chunks / L_chunks, np.nan))
        fig1.add_trace(go.Heatmap(
            z=norm_display,
            text=hover_text_norm,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorscale='RdYlGn_r',
            name='Normalized',
            visible=False
        ))
        
        # Backtrace visualization (show as text)
        fig1.add_trace(go.Heatmap(
            z=np.zeros((n_row, n_col)),
            text=hover_text_B,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0, 'white'], [1, 'white']],
            showscale=False,
            name='B_chunks',
            visible=False
        ))
        
        # Add dropdown to switch between matrices
        fig1.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'D_chunks (Cumulative Cost)', 
                     'method': 'update',
                     'args': [{'visible': [True, False, False, False]}]},
                    {'label': 'L_chunks (Cumulative Length)', 
                     'method': 'update',
                     'args': [{'visible': [False, True, False, False]}]},
                    {'label': 'Normalized (D/L)', 
                     'method': 'update',
                     'args': [{'visible': [False, False, True, False]}]},
                    {'label': 'B_chunks (Backtrace)', 
                     'method': 'update',
                     'args': [{'visible': [False, False, False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.01, 'y': 1.15
            }],
            title="DP Matrices (Chunk-Level Grid)",
            xaxis_title="Block column (bj)",
            yaxis_title="Block row (bi)",
            width=800, height=600,
            template="plotly_white"
        )
        fig1.update_yaxes(autorange='reversed')  # Top-left origin
        fig1.show()
        
        # ============ FIGURE 2: Per-Chunk Details Table ============
        chunk_details = []
        for (bi, bj) in have:
            b = by_idx[(bi, bj)]
            C_curr = b['best_cost']
            L_curr = b['path_length']
            is_start = is_valid_start_chunk(bi, bj)
            is_end = is_valid_end_chunk(bi, bj)
            
            # Check if on selected path
            on_path = (bi, bj) in ordered_blocks
            
            # Get predecessor
            if np.isfinite(D_chunks[bi, bj]):
                bi_back, bj_back = B_chunks[bi, bj]
                if bi_back == -1:
                    pred = "START"
                else:
                    pred = f"({bi_back},{bj_back})"
                
                # Calculate discontinuity from predecessor
                if bi_back >= 0:
                    mag = magn_discontinuity((bi_back, bj_back), (bi, bj))
                    penalty = (C_curr / L_curr) * mag if L_curr > 0 else 0
                else:
                    mag = 0
                    penalty = 0
            else:
                pred = "N/A"
                mag = np.nan
                penalty = np.nan
            
            chunk_details.append({
                'Chunk': f"({bi},{bj})",
                'On Path': '✓' if on_path else '',
                'Start?': '✓' if is_start else '',
                'End?': '✓' if is_end else '',
                'C_chunk': f"{C_curr:.2f}",
                'L_chunk': f"{L_curr:.0f}",
                'D_chunks': f"{D_chunks[bi,bj]:.2f}" if np.isfinite(D_chunks[bi,bj]) else "INF",
                'L_chunks': f"{L_chunks[bi,bj]:.0f}" if np.isfinite(L_chunks[bi,bj]) else "INF",
                'Normalized': f"{D_chunks[bi,bj]/L_chunks[bi,bj]:.4f}" if np.isfinite(D_chunks[bi,bj]) and L_chunks[bi,bj] > 0 else "INF",
                'Pred': pred,
                'Discontinuity': f"{mag:.2f}" if np.isfinite(mag) else "N/A",
                'Penalty': f"{penalty:.2f}" if np.isfinite(penalty) else "N/A"
            })
        
        import pandas as pd
        df = pd.DataFrame(chunk_details)
        
        fig2 = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                       fill_color='paleturquoise',
                       align='left',
                       font=dict(size=11)),
            cells=dict(values=[df[col] for col in df.columns],
                      fill_color='lavender',
                      align='left',
                      font=dict(size=10))
        )])
        fig2.update_layout(
            title="Per-Chunk DP Details",
            width=1400, height=600
        )
        fig2.show()
        
        # ============ FIGURE 3: Path Visualization ============
        fig3 = go.Figure()

        # 1) All tile boxes with color coding
        for b in blocks:
            (r0, r1), (c0, c1) = b['rows'], b['cols']
            bi, bj = b['bi'], b['bj']
            
            # Color code: on path vs not on path
            if (bi, bj) in ordered_blocks:
                line_color = "rgba(0,255,0,0.8)"
                line_width = 3
            else:
                line_color = "rgba(140,140,140,0.4)"
                line_width = 1
            
            fig3.add_shape(
                type="rect",
                x0=c0, x1=c1, y0=r0, y1=r1,
                line=dict(color=line_color, width=line_width, dash="solid" if (bi,bj) in ordered_blocks else "dash"),
                fillcolor="rgba(0,255,0,0.05)" if (bi,bj) in ordered_blocks else "rgba(0,0,0,0)",
                layer="below"
            )

        # 2) All local paths (faded for non-selected, bright for selected)
        palette = pc.qualitative.Plotly
        for idx, b in enumerate(blocks):
            wp = b['wp_global']
            bi, bj = b['bi'], b['bj']
            on_path = (bi, bj) in ordered_blocks
            
            fig3.add_trace(go.Scatter(
                x=wp[:, 1], y=wp[:, 0],
                mode="lines",
                line=dict(width=3 if on_path else 1, 
                         color=palette[idx % len(palette)]),
                name=f"blk({bi},{bj})" + (" [SELECTED]" if on_path else ""),
                showlegend=on_path,
                hovertemplate=f"blk({bi},{bj}) C={b['best_cost']:.1f} L={b['path_length']}<extra></extra>",
                opacity=0.8 if on_path else 0.2
            ))

        # 3) Numbered block order with larger markers
        centers_x, centers_y, labels = [], [], []
        for t, (bi, bj) in enumerate(ordered_blocks, start=1):
            b = by_idx[(bi, bj)]
            (r0, r1), (c0, c1) = b['rows'], b['cols']
            centers_x.append((c0 + c1) / 2.0)
            centers_y.append((r0 + r1) / 2.0)
            labels.append(str(t))
        
        fig3.add_trace(go.Scatter(
            x=centers_x, y=centers_y,
            mode="markers+text",
            marker=dict(size=25, symbol="circle", color="yellow", 
                       line=dict(color="black", width=2)),
            text=labels, textposition="middle center",
            textfont=dict(size=14, color="black", family="Arial Black"),
            name=f"Order (norm={normalized_cost:.4f})",
            hovertemplate="Step %{text}<br>(j=%{x:.0f}, i=%{y:.0f})<extra></extra>"
        ))

        # 4) Start/end markers
        start_x, start_y, end_x, end_y = [], [], [], []
        for (bi, bj) in ordered_blocks:
            b = by_idx[(bi, bj)]
            is_, js_ = b['start_global']
            ie, je = b['end_global']
            start_x.append(js_); start_y.append(is_)
            end_x.append(je); end_y.append(ie)

        fig3.add_trace(go.Scatter(
            x=start_x, y=start_y, mode="markers",
            marker=dict(size=12, symbol="triangle-up", color="lime",
                       line=dict(color="darkgreen", width=2)),
            name="Chunk start points"
        ))
        fig3.add_trace(go.Scatter(
            x=end_x, y=end_y, mode="markers",
            marker=dict(size=12, symbol="diamond", color="red",
                       line=dict(color="darkred", width=2)),
            name="Chunk end points"
        ))

        # 5) Stitched path (thick black line)
        if stitched_wp.size:
            fig3.add_trace(go.Scatter(
                x=stitched_wp[:, 1], y=stitched_wp[:, 0],
                mode="lines",
                line=dict(width=5, color="black"),
                name=f"Stitched path (norm={normalized_cost:.4f})",
                opacity=0.7
            ))

        # 6) Optional global DTW reference
        if show_full_path and isinstance(tiled_result.get('full_global'), dict):
            wp_full = tiled_result['full_global'].get('wp', None)
            if wp_full is not None:
                step = max(1, len(wp_full) // 5000)
                fig3.add_trace(go.Scatter(
                    x=wp_full[::step, 1], y=wp_full[::step, 0],
                    mode="lines",
                    line=dict(width=2, color="rgba(255,0,255,0.4)", dash="dot"),
                    name="Reference Global DTW"
                ))

        fig3.update_layout(
            title=f"Selected Path Visualization<br>Normalized Cost={normalized_cost:.4f}, Total Length={int(total_path_length)}, Goal={best_goal}",
            xaxis_title="F2 frame j (global)",
            yaxis_title="F1 frame i (global)",
            template="plotly_white",
            width=1000, height=800,
            legend=dict(orientation="v", x=1.02, y=1)
        )
        fig3.update_xaxes(range=[0, L2], showgrid=True, gridcolor='lightgray')
        fig3.update_yaxes(range=[0, L1], showgrid=True, gridcolor='lightgray', 
                         scaleanchor="x", scaleratio=1)
        fig3.show()

    # ---- Return results ----
    return {
        'ordered_blocks': ordered_blocks,
        'ordered_linear': [bi * n_col + bj for (bi, bj) in ordered_blocks],
        'path_nodes': np.array(ordered_blocks, dtype=int),
        'stitched_wp': stitched_wp,
        'normalized_cost': normalized_cost,
        'total_unnormalized_cost': total_unnorm_cost,
        'total_path_length': int(total_path_length),
        'D_chunks': D_chunks,
        'L_chunks': L_chunks,
        'B_chunks': B_chunks,
        'n_row': n_row,
        'n_col': n_col,
        'goal_chunk': best_goal
    }

    # ---- Return results ----
    return {
        'ordered_blocks': ordered_blocks,
        'ordered_linear': [bi * n_col + bj for (bi, bj) in ordered_blocks],
        'path_nodes': np.array(ordered_blocks, dtype=int),
        'stitched_wp': stitched_wp,
        'normalized_cost': normalized_cost,
        'total_unnormalized_cost': total_unnorm_cost,
        'total_path_length': int(total_path_length),
        'D_chunks': D_chunks,
        'L_chunks': L_chunks,
        'B_chunks': B_chunks,
        'n_row': n_row,
        'n_col': n_col,
        'goal_chunk': best_goal
    }
