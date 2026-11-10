//! Records as bytes, to keep a run's results and read them back.
//!
//! A little-endian binary: a magic number and a version, then each column
//! as its length and its values. Library crates do not write files, so
//! where the bytes go is the caller's. Reading refuses another version, a
//! cut, columns of different lengths, and bytes left over.

use alphax_core::error::{Error, Result};

use crate::OrderRecords;
use crate::cash::CashRecords;
use crate::logs::{Journal, LogRecords};

const ORDERS: &[u8; 4] = b"PHOR";
const JOURNAL: &[u8; 4] = b"PHJN";
const VERSION: u32 = 1;

/// Writes values of one type; see [`Reader`] for the other way.
pub trait Put {
    /// Appends `self` to `out`.
    fn put(&self, out: &mut Vec<u8>);
}

impl Put for f64 {
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Put for i64 {
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Put for i8 {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(self.to_le_bytes()[0]);
    }
}

impl Put for u8 {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
}

impl Put for bool {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
}

impl Put for usize {
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(*self as u64).to_le_bytes());
    }
}

impl<T: Put> Put for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) {
        self.len().put(out);
        for v in self {
            v.put(out);
        }
    }
}

/// Reads what [`Put`] wrote.
#[derive(Debug)]
pub struct Reader<'b> {
    bytes: &'b [u8],
    at: usize,
}

/// Values a [`Reader`] can read.
pub trait Get: Sized {
    /// Reads one value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] when the bytes are cut short or hold
    /// something that is not a value of this type.
    fn get(r: &mut Reader<'_>) -> Result<Self>;
}

impl<'b> Reader<'b> {
    /// A reader at the start of `bytes`.
    #[must_use]
    pub const fn new(bytes: &'b [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// The next `n` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] when fewer are left.
    pub fn take(&mut self, n: usize) -> Result<&'b [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        let Some(end) = end else {
            return Err(Error::Invariant("the bytes are cut short".into()));
        };
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    /// Reads a `T`.
    ///
    /// # Errors
    ///
    /// As [`Get::get`].
    pub fn get<T: Get>(&mut self) -> Result<T> {
        T::get(self)
    }

    /// Checks the magic number and the version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] for another magic number or version.
    pub fn header(&mut self, magic: &[u8; 4], version: u32, what: &str) -> Result<()> {
        if self.take(4)? != magic {
            return Err(Error::Invariant(format!("not alphax {what}")));
        }
        let found = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if found != version {
            return Err(Error::Invariant(format!(
                "{what} of version {found}; this build reads version {version}"
            )));
        }
        Ok(())
    }

    /// Checks that nothing is left.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] when bytes are left over.
    pub fn end(&self) -> Result<()> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Invariant(format!(
                "{} bytes left over",
                self.bytes.len() - self.at
            )))
        }
    }
}

/// Writes a magic number and a version.
pub fn put_header(out: &mut Vec<u8>, magic: &[u8; 4], version: u32) {
    out.extend_from_slice(magic);
    out.extend_from_slice(&version.to_le_bytes());
}

impl Get for f64 {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Self::from_le_bytes(
            r.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

impl Get for i64 {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Self::from_le_bytes(
            r.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

impl Get for i8 {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Self::from_le_bytes([r.take(1)?[0]]))
    }
}

impl Get for u8 {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        Ok(r.take(1)?[0])
    }
}

impl Get for bool {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        match r.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(Error::Invariant(format!("{b} is not a flag"))),
        }
    }
}

impl Get for usize {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        let v = u64::from_le_bytes(r.take(8)?.try_into().expect("eight bytes"));
        Self::try_from(v).map_err(|_| Error::Invariant(format!("{v} does not fit a usize")))
    }
}

impl<T: Get> Get for Vec<T> {
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        let n: usize = r.get()?;
        // Bounded by what is left, so a corrupt length cannot reserve more
        // than the bytes could hold.
        let mut out = Self::with_capacity(n.min(r.bytes.len() - r.at));
        for _ in 0..n {
            out.push(r.get()?);
        }
        Ok(out)
    }
}

/// Puts each listed field, in order.
macro_rules! put_fields {
    ($src:expr, $out:expr; $($f:ident),+ $(,)?) => { $( $src.$f.put($out); )+ };
}

/// Reads each listed field, in order, into a struct literal.
macro_rules! get_fields {
    ($ty:ident, $r:expr; $($f:ident),+ $(,)?) => { $ty { $( $f: $r.get()?, )+ } };
}

impl OrderRecords {
    fn put(&self, out: &mut Vec<u8>) {
        put_fields!(self, out; col, idx, size, price, fees, side, debt, locked_cash, margin,
            isolated, liquidation, multiplier);
        put_fields!(self.cash, out; col, idx, amount, kind);
        self.allow_negative_price.put(out);
    }

    fn read(r: &mut Reader<'_>) -> Result<Self> {
        let mut out = Self::new();
        out.col = r.get()?;
        out.idx = r.get()?;
        out.size = r.get()?;
        out.price = r.get()?;
        out.fees = r.get()?;
        out.side = r.get()?;
        out.debt = r.get()?;
        out.locked_cash = r.get()?;
        out.margin = r.get()?;
        out.isolated = r.get()?;
        out.liquidation = r.get()?;
        out.multiplier = r.get()?;
        out.cash = get_fields!(CashRecords, r; col, idx, amount, kind);
        out.allow_negative_price = r.get()?;
        let n = out.col.len();
        let fills = [
            out.idx.len(),
            out.size.len(),
            out.price.len(),
            out.fees.len(),
            out.side.len(),
        ];
        let balances = [
            out.debt.len(),
            out.locked_cash.len(),
            out.margin.len(),
            out.isolated.len(),
            out.liquidation.len(),
            out.multiplier.len(),
        ];
        let c = out.cash.col.len();
        let cash = [
            out.cash.idx.len(),
            out.cash.amount.len(),
            out.cash.kind.len(),
        ];
        if fills.iter().any(|&l| l != n)
            || balances.iter().any(|&l| l != 0 && l != n)
            || balances.iter().any(|&l| l != balances[0])
            || cash.iter().any(|&l| l != c)
        {
            return Err(Error::Invariant(
                "record columns of different lengths".into(),
            ));
        }
        Ok(out)
    }

    /// The records as bytes; see the [module](self).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_header(&mut out, ORDERS, VERSION);
        self.put(&mut out);
        out
    }

    /// Records from [`to_bytes`](Self::to_bytes)' output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] for bytes that are not records of this
    /// version or are inconsistent.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.header(ORDERS, VERSION, "records")?;
        let out = Self::read(&mut r)?;
        r.end()?;
        Ok(out)
    }
}

impl Journal {
    /// The fills and logs as bytes; see the [module](self).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_header(&mut out, JOURNAL, VERSION);
        self.orders.put(&mut out);
        let l = &self.logs;
        put_fields!(l, &mut out; group, col, idx, cash, position, debt, locked_cash, free_cash,
            val_price, value, req_size, req_price, req_size_type, req_direction, req_fees,
            req_fixed_fees, req_slippage, req_min_size, req_max_size, req_size_granularity,
            req_leverage, req_leverage_mode, req_account_type, req_initial_margin,
            req_margin_basis, req_reject_prob, req_lock_cash, req_allow_partial,
            req_raise_reject, req_log, new_cash, new_position, new_debt, new_locked_cash,
            new_free_cash, new_val_price, new_value, res_size, res_price, res_fees, res_side,
            res_status, res_status_info, order_id, liquidation, multiplier);
        out
    }

    /// A journal from [`to_bytes`](Self::to_bytes)' output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] for bytes that are not a journal of this
    /// version or are inconsistent.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.header(JOURNAL, VERSION, "journal")?;
        let orders = OrderRecords::read(&mut r)?;
        let logs = get_fields!(LogRecords, r; group, col, idx, cash, position, debt,
            locked_cash, free_cash, val_price, value, req_size, req_price, req_size_type,
            req_direction, req_fees, req_fixed_fees, req_slippage, req_min_size, req_max_size,
            req_size_granularity, req_leverage, req_leverage_mode, req_account_type,
            req_initial_margin, req_margin_basis, req_reject_prob, req_lock_cash,
            req_allow_partial, req_raise_reject, req_log, new_cash, new_position, new_debt,
            new_locked_cash, new_free_cash, new_val_price, new_value, res_size, res_price,
            res_fees, res_side, res_status, res_status_info, order_id, liquidation, multiplier);
        r.end()?;
        let n = logs.col.len();
        if logs.lens().iter().any(|&l| l != n) {
            return Err(Error::Invariant("log columns of different lengths".into()));
        }
        Ok(Self { orders, logs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Balances, RecordSink};
    use alphax_model::{CashFlowKind, OrderSide};

    #[test]
    fn records_round_trip() {
        let mut r = OrderRecords::new();
        r.push_order(3, 1, 2.0, 10.5, 0.1, OrderSide::Buy);
        r.push_balances(Balances {
            debt: 1.0,
            margin: true,
            ..Balances::default()
        });
        r.push_order(7, 0, 1.0, f64::NAN, 0.0, OrderSide::Sell);
        r.push_balances(Balances::default());
        r.push_cash(2, 0, -5.0, CashFlowKind::Earning);
        r.allow_negative_price = true;
        let back = OrderRecords::from_bytes(&r.to_bytes()).unwrap();
        assert_eq!(back.to_bytes(), r.to_bytes());
        assert_eq!(back.idx, [3, 7]);
        assert!(back.allow_negative_price);
        let bytes = r.to_bytes();
        assert!(OrderRecords::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(Journal::from_bytes(&bytes).is_err());
        let j = Journal {
            orders: r,
            logs: LogRecords::new(),
        };
        let back = Journal::from_bytes(&j.to_bytes()).unwrap();
        assert_eq!(back.to_bytes(), j.to_bytes());
    }
}
