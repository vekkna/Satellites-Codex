use pyo3::prelude::*;

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

#[pymodule]
fn satellites_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(count_valid_tank_adds, m)?)?;
    Ok(())
}

