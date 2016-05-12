//! Basic structures and types.

use std ;
use std::ops::Add ;
use std::fmt ;

pub use grid::{ Cell, Grid } ;
pub use frame::{ Frame } ;

extern crate rand;
use rand::{IsaacRng};

use std::iter::*;

/// *Swipe* directions.
#[derive(Debug)]
pub enum Dir {
  /// Up.
  Up,
  /// Down.
  Dw,
  /// Left.
  Lf,
  /// Right.
  Rg,
}

/// Result of a swipe.
///
/// (Also used for the result of a swipe of a single row / column).
#[derive(PartialEq,Eq,Debug)]
pub enum Evolution {
  /// Nothing happened.
  Nothing,
  /// Something moved
  Moved,
  /// At least a merge happened. Stores the score.
  Merged(usize),
}
impl Evolution {
  /// Returns `true` iff something happened.
  #[inline(always)]
  pub fn changed(& self) -> bool {
    match * self {
      Evolution::Nothing => false,
      _ => true,
    }
  }
  /// Transforms `Nothing` in a `Moved`, leaves `self` unchanged otherwise.
  #[inline(always)]
  pub fn mv(& mut self) {
    match * self {
      Evolution::Nothing => * self = Evolution::Moved,
      _ => ()
    }
  }
  /// Returns `true` when called on a `Merged`.
  #[inline(always)]
  pub fn is_merge(& self) -> bool {
    match * self {
      Evolution::Merged(_) => true, _ => false
    }
  }

  /// Transforms an evolution in a `Merged`.
  pub fn mg(& mut self, value: usize) {
    debug_assert!( ! self.is_merge() ) ;
    * self = Evolution::Merged(value)
  }  pub fn score(& self) -> usize {
    match * self {
      Evolution::Merged(score) => score, _ => 0
    }
  }
}
impl Add for Evolution {
  type Output = Evolution ;
  fn add(self, rhs: Evolution) -> Evolution {
    match self {
      Evolution::Nothing => rhs,
      Evolution::Moved => match rhs {
        Evolution::Merged(_) => rhs,
        _ => self,
      },
      Evolution::Merged(pre) => match rhs {
        Evolution::Merged(post) => Evolution::Merged(pre + post),
        _ => self,
      },
    }
  }
}

/// Stores the string representation of a seed.
#[derive(Clone)]
pub struct Seed {
  /// The seed as a string.
  pub s: String,
}
impl fmt::Display for Seed {
  fn fmt(& self, fmt: & mut fmt::Formatter) -> fmt::Result {
    write!(fmt, "{}", self.s)
  }
}
impl Seed {
  /// Creates a seed from a string.
  #[inline(always)]
  pub fn of_str(s: & str) -> Self {
    Seed { s: format!("{}",s) }
  }
  /// Generates a random seed.
  pub fn mk() -> Self {
    match rand::StdRng::new() {
      Ok(mut rng) => {
        use rand::Rand ;
        let size = 13 ;
        let mut s = String::with_capacity(size) ;
        for _ in 0..size {
          let val = (u8::rand(& mut rng) % 26) + 97 ;
          s.push_str(
            std::str::from_utf8( & [ val ] ).unwrap()
          )
        } ;
        Seed { s: s }
      },
      Err(e) => panic!("could not create rng: {:?}", e),
    }
  }

  /// Generates the `IsaacRng` corresponding to a seed.
  pub fn rng(& self) -> IsaacRng {
    use rand::SeedableRng ;
    let seed: Vec<u32> = self.s.as_bytes().iter().map(
      |b| b.clone() as u32
    ).collect() ;
    IsaacRng::from_seed(& seed)
  }
}
