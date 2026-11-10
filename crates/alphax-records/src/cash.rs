//! Cash that moved without a fill.
//!
//! The analysis layer rebuilds every balance from records alone. Fills used to be the only thing that moved
//! cash, so order records were enough. A deposit, an interest payment or a
//! fee charged on a schedule changes the balance without a fill, and the
//! amount actually applied is path-dependent — a withdrawal is capped at the
//! free cash the account has on that bar — so it cannot be recomputed from the
//! inputs afterwards. It has to be recorded, like a fill.
//!
//! These records travel inside [`OrderRecords`](crate::OrderRecords) as its
//! `cash` field, so every sink, merge and constructor that already carries a
//! run's fills carries its cash flows too.

use alphax_model::CashFlowKind;

/// Cash flows that were not fills, struct-of-arrays.
///
/// `col` is the column the flow was booked to. Flows the account receives as a
/// whole — deposits — are booked to the first column of the account's group,
/// the column its starting cash is read from; flows that belong to a symbol
/// are booked to that symbol's column. Per-account series sum a group's
/// columns, so either way the money lands in the right account.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CashRecords {
    /// Column the flow was booked to.
    pub col: Vec<i64>,
    /// Row (bar index) the flow happened on.
    pub idx: Vec<i64>,
    /// Signed amount applied: positive into the account, negative out of it.
    pub amount: Vec<f64>,
    /// What the flow was, as [`CashFlowKind`]'s discriminant.
    pub kind: Vec<i8>,
}

impl CashRecords {
    /// An empty record set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            col: Vec::new(),
            idx: Vec::new(),
            amount: Vec::new(),
            kind: Vec::new(),
        }
    }

    /// Number of flows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// Whether no cash moved outside a fill.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// Records one flow.
    pub fn push(&mut self, row: usize, col: usize, amount: f64, kind: CashFlowKind) {
        self.col.push(col as i64);
        self.idx.push(row as i64);
        self.amount.push(amount);
        self.kind.push(kind as i8);
    }

    /// Appends another record set, preserving order.
    pub fn extend(&mut self, other: &Self) {
        self.col.extend_from_slice(&other.col);
        self.idx.extend_from_slice(&other.idx);
        self.amount.extend_from_slice(&other.amount);
        self.kind.extend_from_slice(&other.kind);
    }

    /// Puts the records into column-major order: by column, then by row.
    ///
    /// A shared-cash group books its flows bar by bar across its columns.
    /// Stable, so a deposit and an earning on the same column and bar keep the
    /// order they were applied in.
    pub fn sort_by_column(&mut self) {
        if self
            .col
            .windows(2)
            .zip(self.idx.windows(2))
            .all(|(c, i)| c[0] < c[1] || (c[0] == c[1] && i[0] <= i[1]))
        {
            return;
        }
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.sort_by_key(|&i| (self.col[i], self.idx[i]));
        macro_rules! reorder {
            ($($field:ident),*) => { $(self.$field = indices.iter().map(|&i| self.$field[i]).collect();)* };
        }
        reorder!(col, idx, amount, kind);
    }

    /// Moves every flow `by` rows later: a segment run on arrays that start
    /// `by` rows into the run writes its rows from 0.
    pub fn shift_rows(&mut self, by: i64) {
        if by != 0 {
            for idx in &mut self.idx {
                *idx += by;
            }
        }
    }

    /// Reorders the flows: the `k`th becomes the one at `order[k]`.
    pub fn permute(&mut self, order: &[usize]) {
        macro_rules! reorder {
            ($($field:ident),*) => { $(self.$field = order.iter().map(|&i| self.$field[i]).collect();)* };
        }
        reorder!(col, idx, amount, kind);
    }

    /// Whether the flow at `i` is capital brought in or taken out rather
    /// than earned: a deposit or withdrawal, or an initial position's cost.
    #[must_use]
    pub fn is_deposit(&self, i: usize) -> bool {
        self.kind[i] == CashFlowKind::Deposit as i8
            || self.kind[i] == CashFlowKind::InitialPosition as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorting_groups_by_column_and_keeps_same_bar_order() {
        let mut r = CashRecords::new();
        r.push(0, 1, 5.0, CashFlowKind::Earning);
        r.push(1, 0, 10.0, CashFlowKind::Deposit);
        r.push(1, 0, -1.0, CashFlowKind::Earning);
        r.push(0, 0, 3.0, CashFlowKind::Deposit);
        r.sort_by_column();
        assert_eq!(r.col, [0, 0, 0, 1]);
        assert_eq!(r.idx, [0, 1, 1, 0]);
        assert_eq!(r.kind[1], CashFlowKind::Deposit as i8);
        assert_eq!(r.kind[2], CashFlowKind::Earning as i8);
        assert!(r.is_deposit(0));
        assert!(!r.is_deposit(3));
    }

    #[test]
    fn already_ordered_records_are_left_alone() {
        let mut r = CashRecords::new();
        r.push(0, 0, 1.0, CashFlowKind::Deposit);
        r.push(2, 0, 1.0, CashFlowKind::Deposit);
        r.push(1, 1, 1.0, CashFlowKind::Earning);
        let before = r.clone();
        r.sort_by_column();
        assert_eq!(r, before);
    }
}
