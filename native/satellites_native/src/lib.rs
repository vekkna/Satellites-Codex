use pyo3::prelude::*;
use std::sync::OnceLock;

/// Small smoke function to verify module loads.
#[pyfunction]
fn ping() -> &'static str {
    "satellites_native_ok"
}

/// Count valid tank add targets from precomputed cell masks.
///
/// A tank add is legal if:
/// - cell has own tank already, OR
/// - cell is empty and not opponent start and not artefact.
#[pyfunction]
fn count_valid_tank_adds(
    unit_owner: Vec<i8>,
    unit_kind: Vec<u8>,
    me: i8,
    is_opp_start: Vec<bool>,
    is_artefact: Vec<bool>,
) -> PyResult<usize> {
    let n = unit_owner.len();
    if unit_kind.len() != n || is_opp_start.len() != n || is_artefact.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }
    let mut out = 0usize;
    for i in 0..n {
        let owner = unit_owner[i];
        let kind = unit_kind[i];
        if owner == me && kind == 2 {
            out += 1;
            continue;
        }
        if owner == -1 && !is_opp_start[i] && !is_artefact[i] {
            out += 1;
        }
    }
    Ok(out)
}

fn row_offsets() -> &'static [usize; 9] {
    static OFFSETS: [usize; 9] = [0, 8, 17, 27, 38, 50, 61, 71, 80];
    &OFFSETS
}

fn neighbors_by_cell_id() -> &'static Vec<Vec<usize>> {
    static NEIGHBORS: OnceLock<Vec<Vec<usize>>> = OnceLock::new();
    NEIGHBORS.get_or_init(|| {
        let row_widths: [usize; 9] = [8, 9, 10, 11, 12, 11, 10, 9, 8];
        let offsets = row_offsets();
        let num_cells = row_widths.iter().sum::<usize>();
        let mut out = vec![Vec::new(); num_cells];

        for r in 0..9usize {
            for c in 0..row_widths[r] {
                let sid = offsets[r] + c;
                let mut candidates: [(isize, isize); 6] = [
                    (r as isize, c as isize - 1),
                    (r as isize, c as isize + 1),
                    (-1, -1),
                    (-1, -1),
                    (-1, -1),
                    (-1, -1),
                ];
                if r > 0 {
                    if r <= 4 {
                        candidates[2] = (r as isize - 1, c as isize - 1);
                        candidates[3] = (r as isize - 1, c as isize);
                    } else {
                        candidates[2] = (r as isize - 1, c as isize);
                        candidates[3] = (r as isize - 1, c as isize + 1);
                    }
                }
                if r < 8 {
                    if r < 4 {
                        candidates[4] = (r as isize + 1, c as isize);
                        candidates[5] = (r as isize + 1, c as isize + 1);
                    } else {
                        candidates[4] = (r as isize + 1, c as isize - 1);
                        candidates[5] = (r as isize + 1, c as isize);
                    }
                }

                for (nr, nc) in candidates {
                    if nr < 0 || nr >= 9 {
                        continue;
                    }
                    let ur = nr as usize;
                    if nc < 0 || nc >= row_widths[ur] as isize {
                        continue;
                    }
                    let uc = nc as usize;
                    out[sid].push(offsets[ur] + uc);
                }
            }
        }
        out
    })
}

fn generate_legal_action_indices_inner(
    state_code: u8,
    action_type_code: u8,
    turn: i8,
    sat_charges: &[u8],
    unit_owner: &[i8],
    unit_kind: &[u8],
    unit_count: &[u8],
    is_artefact: &[bool],
    is_p0_start: &[bool],
    is_p1_start: &[bool],
    max_move_amount: u8,
) -> PyResult<Vec<usize>> {
    let n = unit_owner.len();
    if unit_kind.len() != n
        || unit_count.len() != n
        || is_artefact.len() != n
        || is_p0_start.len() != n
        || is_p1_start.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All board arrays must have the same length",
        ));
    }
    if max_move_amount == 0 {
        return Ok(Vec::new());
    }

    if state_code == 0 {
        return Ok(Vec::new());
    }
    if state_code == 1 {
        let mut out = Vec::with_capacity(6);
        for (i, &c) in sat_charges.iter().enumerate().take(6) {
            if c > 0 {
                out.push(i);
            }
        }
        return Ok(out);
    }
    if state_code == 2 {
        return Ok(vec![6usize, 7usize]);
    }
    if state_code != 3 {
        return Ok(Vec::new());
    }

    let add_base = 8usize;
    let move_base = add_base + n;
    let max_move_amount_usize = max_move_amount as usize;

    if action_type_code == 1 || action_type_code == 2 {
        let mut owner_total = 0usize;
        for i in 0..n {
            if unit_owner[i] == turn {
                owner_total += unit_count[i] as usize;
            }
        }
        if owner_total >= 20 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(64);
        if action_type_code == 1 {
            let opp_start = if turn == 0 { is_p1_start } else { is_p0_start };
            for cid in 0..n {
                let owner = unit_owner[cid];
                let kind = unit_kind[cid];
                if owner == -1 {
                    if !opp_start[cid] && !is_artefact[cid] {
                        out.push(add_base + cid);
                    }
                } else if owner == turn && kind == 2 {
                    out.push(add_base + cid);
                }
            }
        } else {
            let my_start = if turn == 0 { is_p0_start } else { is_p1_start };
            for cid in 0..n {
                if unit_owner[cid] == turn && unit_kind[cid] == 1 {
                    out.push(add_base + cid);
                }
            }
            for cid in 0..n {
                if my_start[cid] && unit_owner[cid] == -1 {
                    out.push(add_base + cid);
                }
            }
        }
        return Ok(out);
    }

    if action_type_code == 3 || action_type_code == 4 {
        let req_kind = if action_type_code == 3 { 2u8 } else { 1u8 };
        let neighbors = neighbors_by_cell_id();
        if neighbors.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input array length does not match board topology",
            ));
        }

        let (opp_start_a, opp_start_b) = if turn == 0 { (83usize, 84usize) } else { (3usize, 4usize) };
        let mut out: Vec<usize> = Vec::with_capacity(256);
        let mut edge_ordinal = 0usize;
        for sid in 0..n {
            let src_is_valid = unit_owner[sid] == turn && unit_kind[sid] == req_kind && unit_count[sid] > 0;
            for &eid in &neighbors[sid] {
                if src_is_valid && eid != opp_start_a && eid != opp_start_b {
                    let src_count = unit_count[sid];
                    let target_owner = unit_owner[eid];
                    let target_kind = unit_kind[eid];
                    let target_count = unit_count[eid];

                    let mut lo = 0u8;
                    let mut hi = 0u8;
                    if req_kind == 2 && is_artefact[eid] {
                    } else if target_owner == -1 {
                        lo = 1;
                        hi = src_count;
                    } else if target_owner == turn {
                        if target_kind == req_kind {
                            lo = 1;
                            hi = src_count;
                        }
                    } else if req_kind == 2 {
                        if target_kind == 1 {
                            lo = 1;
                            hi = src_count;
                        } else if target_kind == 2 && src_count > target_count {
                            lo = target_count + 1;
                            hi = src_count;
                        }
                    }

                    if lo > 0 {
                        let hi_clamped = hi.min(max_move_amount);
                        if lo <= hi_clamped {
                            for amount in lo..=hi_clamped {
                                out.push(
                                    move_base
                                        + edge_ordinal * max_move_amount_usize
                                        + (amount as usize - 1),
                                );
                            }
                        }
                    }
                }
                edge_ordinal += 1;
            }
        }
        return Ok(out);
    }
    Ok(Vec::new())
}

/// Generate legal move actions as (src_cell_id, dst_cell_id, amount).
///
/// req_kind: 1=bot, 2=tank
#[pyfunction]
fn generate_move_actions(
    unit_owner: Vec<i8>,
    unit_kind: Vec<u8>,
    unit_count: Vec<u8>,
    turn: i8,
    req_kind: u8,
    is_artefact: Vec<bool>,
) -> PyResult<Vec<(usize, usize, u8)>> {
    let n = unit_owner.len();
    if unit_kind.len() != n || unit_count.len() != n || is_artefact.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }
    if req_kind != 1 && req_kind != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "req_kind must be 1 (bot) or 2 (tank)",
        ));
    }

    let neighbors = neighbors_by_cell_id();
    if neighbors.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input array length does not match board topology",
        ));
    }

    // Opponent starts are fixed by turn on the canonical board.
    let (opp_start_a, opp_start_b) = if turn == 0 {
        (83usize, 84usize) // (8,3), (8,4)
    } else {
        (3usize, 4usize) // (0,3), (0,4)
    };

    let mut out: Vec<(usize, usize, u8)> = Vec::with_capacity(1024);
    for sid in 0..n {
        if unit_owner[sid] != turn || unit_kind[sid] != req_kind {
            continue;
        }
        let src_count = unit_count[sid];
        if src_count == 0 {
            continue;
        }

        for &eid in &neighbors[sid] {
            if eid == opp_start_a || eid == opp_start_b {
                continue;
            }

            let target_owner = unit_owner[eid];
            let target_kind = unit_kind[eid];
            let target_count = unit_count[eid];

            // Tank cannot enter artefact hex.
            if req_kind == 2 && is_artefact[eid] {
                continue;
            }

            if target_owner == -1 {
                for amount in 1..=src_count {
                    out.push((sid, eid, amount));
                }
                continue;
            }

            if target_owner == turn {
                // Merge only with same type.
                if target_kind == req_kind {
                    for amount in 1..=src_count {
                        out.push((sid, eid, amount));
                    }
                }
                continue;
            }

            // Enemy target.
            if req_kind == 1 {
                continue;
            }
            // Tank vs enemy bot: any amount legal.
            if target_kind == 1 {
                for amount in 1..=src_count {
                    out.push((sid, eid, amount));
                }
                continue;
            }
            // Tank vs enemy tank: must be strictly larger.
            if target_kind == 2 && src_count > target_count {
                for amount in (target_count + 1)..=src_count {
                    out.push((sid, eid, amount));
                }
            }
        }
    }
    Ok(out)
}

/// Generate legal global action indices directly (matches rl.action_space.GlobalActionSpace).
///
/// State codes:
/// - 0: GAME_OVER
/// - 1: CHOOSE_SATELLITE
/// - 2: CHOOSE_DIRECTION
/// - 3: PERFORM_ACTIONS
///
/// Action type codes (when state=PERFORM_ACTIONS):
/// - 0: none/unknown
/// - 1: add_tank
/// - 2: add_bot
/// - 3: move_tank
/// - 4: move_bot
#[pyfunction]
fn generate_legal_action_indices(
    state_code: u8,
    action_type_code: u8,
    turn: i8,
    sat_charges: Vec<u8>,
    unit_owner: Vec<i8>,
    unit_kind: Vec<u8>,
    unit_count: Vec<u8>,
    is_artefact: Vec<bool>,
    is_p0_start: Vec<bool>,
    is_p1_start: Vec<bool>,
    max_move_amount: u8,
) -> PyResult<Vec<usize>> {
    generate_legal_action_indices_inner(
        state_code,
        action_type_code,
        turn,
        &sat_charges,
        &unit_owner,
        &unit_kind,
        &unit_count,
        &is_artefact,
        &is_p0_start,
        &is_p1_start,
        max_move_amount,
    )
}

/// Encode game features for RL (matches rl.encode.FeatureEncoder).
#[pyfunction]
fn encode_features(
    unit_owner: Vec<i8>,
    unit_kind: Vec<u8>,
    unit_count: Vec<u8>,
    is_artefact: Vec<bool>,
    is_p0_start: Vec<bool>,
    is_p1_start: Vec<bool>,
    turn: u8,
    score0: i16,
    score1: i16,
    state_code: u8,
    active_satellite_idx: i8,
    actions_remaining: i16,
    picked_up_charges: i16,
    turn_count: i16,
    max_turns: i16,
    sat_type_codes: Vec<u8>,
    sat_charges: Vec<u8>,
) -> PyResult<Vec<f32>> {
    let n = unit_owner.len();
    if unit_kind.len() != n
        || unit_count.len() != n
        || is_artefact.len() != n
        || is_p0_start.len() != n
        || is_p1_start.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All board arrays must have the same length",
        ));
    }
    if sat_type_codes.len() != 6 || sat_charges.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sat_type_codes and sat_charges must each have length 6",
        ));
    }

    // cell features + global features
    let feature_dim = n * 7 + (2 + 2 + 4 + 7 + 3 + 30);
    let mut feat = vec![0.0f32; feature_dim];
    let mut p = 0usize;

    for cid in 0..n {
        let owner = unit_owner[cid];
        let kind = unit_kind[cid];
        let cnt = unit_count[cid] as f32 / 20.0;
        if owner == 0 && kind == 1 {
            feat[p] = cnt;
        } else if owner == 0 && kind == 2 {
            feat[p + 1] = cnt;
        } else if owner == 1 && kind == 1 {
            feat[p + 2] = cnt;
        } else if owner == 1 && kind == 2 {
            feat[p + 3] = cnt;
        }
        feat[p + 4] = if is_artefact[cid] { 1.0 } else { 0.0 };
        feat[p + 5] = if is_p0_start[cid] { 1.0 } else { 0.0 };
        feat[p + 6] = if is_p1_start[cid] { 1.0 } else { 0.0 };
        p += 7;
    }

    // Side to move one-hot.
    let turn_idx = (turn.min(1)) as usize;
    feat[p + turn_idx] = 1.0;
    p += 2;

    // Scores.
    feat[p] = score0 as f32 / 9.0;
    feat[p + 1] = score1 as f32 / 9.0;
    p += 2;

    // Phase one-hot.
    let phase_idx = match state_code {
        1 => 0usize, // CHOOSE_SATELLITE
        2 => 1usize, // CHOOSE_DIRECTION
        3 => 2usize, // PERFORM_ACTIONS
        0 => 3usize, // GAME_OVER
        _ => 0usize,
    };
    feat[p + phase_idx] = 1.0;
    p += 4;

    // Active satellite one-hot; 6 means none.
    let aidx = if (0..=5).contains(&active_satellite_idx) {
        active_satellite_idx as usize
    } else {
        6usize
    };
    feat[p + aidx] = 1.0;
    p += 7;

    // Counters.
    feat[p] = actions_remaining as f32 / 3.0;
    feat[p + 1] = picked_up_charges as f32 / 3.0;
    let mt = if max_turns <= 0 { 1 } else { max_turns };
    feat[p + 2] = turn_count as f32 / mt as f32;
    p += 3;

    // Satellites: per slot one-hot type + charge.
    // sat_type_codes: 0 move_tank, 1 move_bot, 2 add_tank, 3 add_bot
    for i in 0..6usize {
        let t = sat_type_codes[i];
        if t <= 3 {
            feat[p + t as usize] = 1.0;
        }
        feat[p + 4] = sat_charges[i] as f32 / 3.0;
        p += 5;
    }

    Ok(feat)
}

#[pyclass]
#[derive(Clone)]
struct NativeSatGame {
    unit_owner: Vec<i8>,
    unit_kind: Vec<u8>,
    unit_count: Vec<u8>,
    is_artefact: Vec<bool>,
    is_p0_start: Vec<bool>,
    is_p1_start: Vec<bool>,
    sat_type_codes: Vec<u8>,
    sat_charges: Vec<u8>,
    turn: u8,
    scores: [i16; 2],
    state_code: u8,
    active_satellite_idx: i8,
    actions_remaining: i16,
    picked_up_charges: i16,
    action_type_code: u8,
    turn_count: i16,
    max_turns: i16,
    winner: i8,
}

#[pymethods]
impl NativeSatGame {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        unit_owner: Vec<i8>,
        unit_kind: Vec<u8>,
        unit_count: Vec<u8>,
        is_artefact: Vec<bool>,
        is_p0_start: Vec<bool>,
        is_p1_start: Vec<bool>,
        sat_type_codes: Vec<u8>,
        sat_charges: Vec<u8>,
        turn: u8,
        score0: i16,
        score1: i16,
        state_code: u8,
        active_satellite_idx: i8,
        actions_remaining: i16,
        picked_up_charges: i16,
        action_type_code: u8,
        turn_count: i16,
        max_turns: i16,
        winner: i8,
    ) -> PyResult<Self> {
        if sat_type_codes.len() != 6 || sat_charges.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sat_type_codes and sat_charges must have length 6",
            ));
        }
        Ok(Self {
            unit_owner,
            unit_kind,
            unit_count,
            is_artefact,
            is_p0_start,
            is_p1_start,
            sat_type_codes,
            sat_charges,
            turn,
            scores: [score0, score1],
            state_code,
            active_satellite_idx,
            actions_remaining,
            picked_up_charges,
            action_type_code,
            turn_count,
            max_turns,
            winner,
        })
    }

    fn clone_native(&self) -> Self {
        self.clone()
    }

    fn is_terminal(&self) -> bool {
        self.state_code == 0
    }

    fn current_player(&self) -> u8 {
        self.turn
    }

    fn winner(&self) -> i8 {
        self.winner
    }

    fn legal_action_indices(&self, max_move_amount: u8) -> PyResult<Vec<usize>> {
        generate_legal_action_indices_inner(
            self.state_code,
            self.action_type_code,
            self.turn as i8,
            &self.sat_charges,
            &self.unit_owner,
            &self.unit_kind,
            &self.unit_count,
            &self.is_artefact,
            &self.is_p0_start,
            &self.is_p1_start,
            max_move_amount,
        )
    }

    fn encode_features(&self) -> PyResult<Vec<f32>> {
        encode_features(
            self.unit_owner.clone(),
            self.unit_kind.clone(),
            self.unit_count.clone(),
            self.is_artefact.clone(),
            self.is_p0_start.clone(),
            self.is_p1_start.clone(),
            self.turn,
            self.scores[0],
            self.scores[1],
            self.state_code,
            self.active_satellite_idx,
            self.actions_remaining,
            self.picked_up_charges,
            self.turn_count,
            self.max_turns,
            self.sat_type_codes.clone(),
            self.sat_charges.clone(),
        )
    }

    fn apply_action_index(&mut self, action_index: usize, max_move_amount: u8) -> PyResult<bool> {
        if max_move_amount == 0 || self.state_code == 0 {
            return Ok(false);
        }
        let n = self.unit_owner.len();
        let add_base = 8usize;
        let move_base = add_base + n;
        let neighbors = neighbors_by_cell_id();
        let num_edges: usize = neighbors.iter().map(|v| v.len()).sum();
        let move_span = num_edges * max_move_amount as usize;

        // choose satellite
        if self.state_code == 1 {
            if action_index >= 6 {
                return Ok(false);
            }
            let idx = action_index;
            if self.sat_charges[idx] == 0 {
                return Ok(false);
            }
            self.active_satellite_idx = idx as i8;
            self.action_type_code = match self.sat_type_codes[idx] {
                0 => 3, // move_tank
                1 => 4, // move_bot
                2 => 1, // add_tank
                3 => 2, // add_bot
                _ => 0,
            };
            self.picked_up_charges = self.sat_charges[idx] as i16;
            self.sat_charges[idx] = 0;
            self.state_code = 2;
            return Ok(true);
        }

        // choose direction
        if self.state_code == 2 {
            if action_index != 6 && action_index != 7 {
                return Ok(false);
            }
            let dir: i32 = if action_index == 7 { 1 } else { -1 };
            let mut idx = self.active_satellite_idx as i32;
            let mut to_distribute = self.picked_up_charges.max(0) as usize;
            while to_distribute > 0 {
                idx = (idx + dir + 6) % 6;
                self.sat_charges[idx as usize] = self.sat_charges[idx as usize].saturating_add(1);
                to_distribute -= 1;
            }
            let legal = generate_legal_action_indices_inner(
                3,
                self.action_type_code,
                self.turn as i8,
                &self.sat_charges,
                &self.unit_owner,
                &self.unit_kind,
                &self.unit_count,
                &self.is_artefact,
                &self.is_p0_start,
                &self.is_p1_start,
                max_move_amount,
            )?;
            if legal.is_empty() {
                self.end_turn();
            } else {
                self.actions_remaining = self.picked_up_charges;
                self.state_code = 3;
            }
            return Ok(true);
        }

        if self.state_code != 3 {
            return Ok(false);
        }

        let mut changed = false;
        if self.action_type_code == 1 || self.action_type_code == 2 {
            if !(add_base <= action_index && action_index < move_base) {
                return Ok(false);
            }
            let cid = action_index - add_base;
            changed = self.apply_add(cid);
        } else if self.action_type_code == 3 || self.action_type_code == 4 {
            if !(move_base <= action_index && action_index < (move_base + move_span)) {
                return Ok(false);
            }
            let rel = action_index - move_base;
            let edge_ordinal = rel / max_move_amount as usize;
            let amount = (rel % max_move_amount as usize + 1) as u8;
            let mut cur = 0usize;
            let mut sid = 0usize;
            let mut eid = 0usize;
            let mut found = false;
            'outer: for s in 0..neighbors.len() {
                for &e in &neighbors[s] {
                    if cur == edge_ordinal {
                        sid = s;
                        eid = e;
                        found = true;
                        break 'outer;
                    }
                    cur += 1;
                }
            }
            if !found {
                return Ok(false);
            }
            changed = self.apply_move(sid, eid, amount);
        }
        if !changed {
            return Ok(false);
        }

        if self.check_win() {
            return Ok(true);
        }
        self.actions_remaining -= 1;
        if self.actions_remaining <= 0 {
            self.end_turn();
        } else {
            let legal = generate_legal_action_indices_inner(
                3,
                self.action_type_code,
                self.turn as i8,
                &self.sat_charges,
                &self.unit_owner,
                &self.unit_kind,
                &self.unit_count,
                &self.is_artefact,
                &self.is_p0_start,
                &self.is_p1_start,
                max_move_amount,
            )?;
            if legal.is_empty() {
                self.end_turn();
            }
        }
        Ok(true)
    }
}

impl NativeSatGame {
    fn apply_add(&mut self, cid: usize) -> bool {
        if cid >= self.unit_owner.len() {
            return false;
        }
        let turn = self.turn as i8;
        let mut owner_total = 0usize;
        for i in 0..self.unit_owner.len() {
            if self.unit_owner[i] == turn {
                owner_total += self.unit_count[i] as usize;
            }
        }
        if owner_total >= 20 {
            return false;
        }

        if self.action_type_code == 1 {
            // add_tank
            let opp_start = if self.turn == 0 { &self.is_p1_start } else { &self.is_p0_start };
            if self.is_artefact[cid] || opp_start[cid] {
                return false;
            }
            let owner = self.unit_owner[cid];
            let kind = self.unit_kind[cid];
            if owner == -1 {
                self.unit_owner[cid] = turn;
                self.unit_kind[cid] = 2;
                self.unit_count[cid] = 1;
                return true;
            }
            if owner == turn && kind == 2 {
                self.unit_count[cid] = self.unit_count[cid].saturating_add(1);
                return true;
            }
            return false;
        }

        if self.action_type_code == 2 {
            // add_bot
            let own_start = if self.turn == 0 { &self.is_p0_start } else { &self.is_p1_start };
            let owner = self.unit_owner[cid];
            let kind = self.unit_kind[cid];
            if owner == turn && kind == 1 {
                self.unit_count[cid] = self.unit_count[cid].saturating_add(1);
                return true;
            }
            if owner == -1 && own_start[cid] {
                self.unit_owner[cid] = turn;
                self.unit_kind[cid] = 1;
                self.unit_count[cid] = 1;
                return true;
            }
        }
        false
    }

    fn apply_move(&mut self, sid: usize, eid: usize, amount: u8) -> bool {
        if sid >= self.unit_owner.len() || eid >= self.unit_owner.len() || amount == 0 {
            return false;
        }
        let turn = self.turn as i8;
        let req_kind = if self.action_type_code == 3 { 2u8 } else { 1u8 };

        if self.unit_owner[sid] != turn || self.unit_kind[sid] != req_kind {
            return false;
        }
        let src_count = self.unit_count[sid];
        if amount > src_count {
            return false;
        }
        let (opp_start_a, opp_start_b) = if self.turn == 0 { (83usize, 84usize) } else { (3usize, 4usize) };
        if eid == opp_start_a || eid == opp_start_b {
            return false;
        }
        if req_kind == 2 && self.is_artefact[eid] {
            return false;
        }

        let target_owner = self.unit_owner[eid];
        let target_kind = self.unit_kind[eid];
        let target_count = self.unit_count[eid];
        if target_owner == turn {
            if target_kind != req_kind {
                return false;
            }
        } else if target_owner != -1 {
            if req_kind == 1 {
                return false;
            }
            if target_kind == 2 && target_count >= amount {
                return false;
            }
        }

        // execute
        self.unit_count[sid] -= amount;
        if self.unit_count[sid] == 0 {
            self.unit_owner[sid] = -1;
            self.unit_kind[sid] = 0;
        }

        let mut did_move_in = true;
        if target_owner == -1 {
            self.unit_owner[eid] = turn;
            self.unit_kind[eid] = req_kind;
            self.unit_count[eid] = amount;
        } else if target_owner == turn {
            self.unit_count[eid] = self.unit_count[eid].saturating_add(amount);
        } else {
            // attack
            self.unit_owner[eid] = -1;
            self.unit_kind[eid] = 0;
            self.unit_count[eid] = 0;
            if req_kind == 2 {
                // tank shoots/holds
                did_move_in = false;
                if self.unit_owner[sid] == -1 {
                    self.unit_owner[sid] = turn;
                    self.unit_kind[sid] = req_kind;
                    self.unit_count[sid] = amount;
                } else {
                    self.unit_count[sid] = self.unit_count[sid].saturating_add(amount);
                }
            } else {
                self.unit_owner[eid] = turn;
                self.unit_kind[eid] = req_kind;
                self.unit_count[eid] = amount;
            }
        }

        if did_move_in && self.is_artefact[eid] {
            self.is_artefact[eid] = false;
            self.scores[self.turn as usize] += amount as i16;
        }
        true
    }

    fn check_win(&mut self) -> bool {
        if self.scores[self.turn as usize] >= 9 {
            self.winner = self.turn as i8;
            self.state_code = 0;
            return true;
        }
        let remaining_artefacts = self.is_artefact.iter().filter(|&&v| v).count();
        if remaining_artefacts == 0 {
            self.winner = if self.scores[0] > self.scores[1] {
                0
            } else if self.scores[1] > self.scores[0] {
                1
            } else {
                self.turn as i8
            };
            self.state_code = 0;
            return true;
        }
        false
    }

    fn end_turn(&mut self) {
        if self.turn_count >= self.max_turns {
            self.state_code = 0;
            self.winner = if self.scores[0] > self.scores[1] {
                0
            } else if self.scores[1] > self.scores[0] {
                1
            } else {
                -1
            };
            return;
        }
        self.turn = 1 - self.turn;
        if self.turn == 0 {
            self.turn_count += 1;
        }
        self.state_code = 1;
        self.active_satellite_idx = -1;
        self.action_type_code = 0;
        self.actions_remaining = 0;
        self.picked_up_charges = 0;
    }
}

#[pymodule]
fn satellites_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(count_valid_tank_adds, m)?)?;
    m.add_function(wrap_pyfunction!(generate_move_actions, m)?)?;
    m.add_function(wrap_pyfunction!(generate_legal_action_indices, m)?)?;
    m.add_function(wrap_pyfunction!(encode_features, m)?)?;
    m.add_class::<NativeSatGame>()?;
    Ok(())
}
