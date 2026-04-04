use crate::error::GpuSimError;

/// Maps user-facing qubit IDs to bit positions in the state vector.
///
/// The state vector has `2^(max_bit + 1)` amplitudes, where `max_bit` is the
/// highest bit position ever assigned. Releasing a qubit does NOT shrink the
/// state vector; the bit position is simply marked as inactive and recycled
/// when a new qubit is allocated.
pub struct QubitMap {
    /// Maps qubit ID -> bit position. `None` if the ID is not active.
    id_to_bit: Vec<Option<usize>>,
    /// Maps bit position -> qubit ID. `None` if the bit is inactive.
    bit_to_id: Vec<Option<usize>>,
    /// Next qubit ID to assign (if `free_ids` is empty).
    next_id: usize,
    /// Next bit position to use (if `free_bits` is empty).
    next_bit: usize,
    /// Pool of recycled qubit IDs.
    free_ids: Vec<usize>,
    /// Pool of recycled bit positions.
    free_bits: Vec<usize>,
    /// Number of currently active qubits.
    count: usize,
}

impl QubitMap {
    /// Creates an empty qubit map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id_to_bit: Vec::new(),
            bit_to_id: Vec::new(),
            next_id: 0,
            next_bit: 0,
            free_ids: Vec::new(),
            free_bits: Vec::new(),
            count: 0,
        }
    }

    /// Allocates a new qubit and returns its user-facing ID.
    ///
    /// Reuses a previously freed ID and bit position if available.
    /// Otherwise, assigns the next sequential ID and bit position.
    pub fn allocate(&mut self) -> usize {
        let id = self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        });
        let bit = self.free_bits.pop().unwrap_or_else(|| {
            let bit = self.next_bit;
            self.next_bit += 1;
            bit
        });

        // Ensure vectors are large enough
        if id >= self.id_to_bit.len() {
            self.id_to_bit.resize(id + 1, None);
        }
        if bit >= self.bit_to_id.len() {
            self.bit_to_id.resize(bit + 1, None);
        }

        self.id_to_bit[id] = Some(bit);
        self.bit_to_id[bit] = Some(id);
        self.count += 1;
        id
    }

    /// Releases a qubit, returning the bit position that was freed.
    ///
    /// The bit position is added to the recycling pool. The state vector
    /// size does NOT shrink — the bit position simply becomes inactive.
    pub fn release(&mut self, id: usize) -> Result<usize, GpuSimError> {
        let bit = self.bit_position(id)?;
        self.id_to_bit[id] = None;
        self.bit_to_id[bit] = None;
        self.free_ids.push(id);
        self.free_bits.push(bit);
        self.count -= 1;
        Ok(bit)
    }

    /// Returns the bit position for the given qubit ID.
    pub fn bit_position(&self, id: usize) -> Result<usize, GpuSimError> {
        self.id_to_bit
            .get(id)
            .copied()
            .flatten()
            .ok_or(GpuSimError::QubitNotFound(id))
    }

    /// Returns the number of currently active qubits.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.count
    }

    /// Swaps the bit-position mappings of two qubit IDs.
    ///
    /// After this call, `id1` maps to the bit position that `id2` previously
    /// mapped to, and vice versa. This is a CPU-only operation that does not
    /// modify the GPU state vector.
    pub fn swap(&mut self, id1: usize, id2: usize) -> Result<(), GpuSimError> {
        let bit1 = self.bit_position(id1)?;
        let bit2 = self.bit_position(id2)?;
        self.id_to_bit[id1] = Some(bit2);
        self.id_to_bit[id2] = Some(bit1);
        self.bit_to_id[bit1] = Some(id2);
        self.bit_to_id[bit2] = Some(id1);
        Ok(())
    }

    /// Returns the highest bit position ever used, which determines the
    /// minimum state vector size as `2^(max_bit + 1)`.
    ///
    /// Returns 0 if no qubits have ever been allocated.
    #[must_use]
    pub fn max_bit(&self) -> usize {
        self.next_bit.saturating_sub(1)
    }
}

impl Default for QubitMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_sequential() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        let q1 = map.allocate();
        let q2 = map.allocate();

        assert_eq!(q0, 0);
        assert_eq!(q1, 1);
        assert_eq!(q2, 2);
        assert_eq!(map.bit_position(q0).expect("q0 should exist"), 0);
        assert_eq!(map.bit_position(q1).expect("q1 should exist"), 1);
        assert_eq!(map.bit_position(q2).expect("q2 should exist"), 2);
        assert_eq!(map.active_count(), 3);
        assert_eq!(map.max_bit(), 2);
    }

    #[test]
    fn release_and_recycle() {
        let mut map = QubitMap::new();
        let _q0 = map.allocate();
        let q1 = map.allocate();
        let _q2 = map.allocate();

        // Release q1
        let freed_bit = map.release(q1).expect("release should succeed");
        assert_eq!(freed_bit, 1);
        assert_eq!(map.active_count(), 2);

        // Allocate again -- should reuse ID 1 and bit position 1
        let q3 = map.allocate();
        assert_eq!(q3, 1);
        assert_eq!(map.bit_position(q3).expect("q3 should exist"), 1);
    }

    #[test]
    fn release_nonexistent() {
        let mut map = QubitMap::new();
        let result = map.release(42);
        assert!(matches!(result, Err(GpuSimError::QubitNotFound(42))));
    }

    #[test]
    fn release_already_released() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        map.release(q0).expect("first release should succeed");
        let result = map.release(q0);
        assert!(matches!(result, Err(GpuSimError::QubitNotFound(0))));
    }

    #[test]
    fn bit_position_lookup() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        assert_eq!(map.bit_position(q0).expect("should exist"), 0);
    }

    #[test]
    fn bit_position_not_found() {
        let map = QubitMap::new();
        let result = map.bit_position(99);
        assert!(matches!(result, Err(GpuSimError::QubitNotFound(99))));
    }

    #[test]
    fn active_count_tracks_correctly() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        let _q1 = map.allocate();
        let _q2 = map.allocate();
        assert_eq!(map.active_count(), 3);

        map.release(q0).expect("release should succeed");
        assert_eq!(map.active_count(), 2);
    }

    #[test]
    fn swap_exchanges_bit_positions() {
        let mut map = QubitMap::new();
        let q0 = map.allocate(); // bit 0
        let q1 = map.allocate(); // bit 1
        let q2 = map.allocate(); // bit 2

        map.swap(q0, q2).expect("swap should succeed");
        assert_eq!(map.bit_position(q0).expect("q0 should exist"), 2);
        assert_eq!(map.bit_position(q2).expect("q2 should exist"), 0);
        // q1 is unchanged
        assert_eq!(map.bit_position(q1).expect("q1 should exist"), 1);
    }

    #[test]
    fn swap_is_reversible() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        let q1 = map.allocate();

        map.swap(q0, q1).expect("swap should succeed");
        map.swap(q0, q1).expect("reverse swap should succeed");
        assert_eq!(map.bit_position(q0).expect("q0 should exist"), 0);
        assert_eq!(map.bit_position(q1).expect("q1 should exist"), 1);
    }

    #[test]
    fn swap_nonexistent_qubit() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        let result = map.swap(q0, 99);
        assert!(matches!(result, Err(GpuSimError::QubitNotFound(99))));
    }

    #[test]
    fn max_bit_never_shrinks() {
        let mut map = QubitMap::new();
        let q0 = map.allocate();
        let q1 = map.allocate();
        let q2 = map.allocate();
        assert_eq!(map.max_bit(), 2);

        // Release all qubits
        map.release(q0).expect("release should succeed");
        map.release(q1).expect("release should succeed");
        map.release(q2).expect("release should succeed");

        // max_bit should still be 2 (it never shrinks)
        assert_eq!(map.max_bit(), 2);
        assert_eq!(map.active_count(), 0);
    }
}
