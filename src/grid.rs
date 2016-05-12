//! Types representing cells and grids.

use std::ops::Add ;

use rand::IsaacRng ;

use common::{ Dir, Evolution, Seed } ;

// /// Trait the different types of cells must implement.
// trait HasVal {
//   /// The value stored in whatever `Self` is.
//   fn val(& self) -> usize ;
// }

/// A cell, stores a power of two.
///
/// # Addition over cells.
///
/// ```
/// let cell_1 = Cell { pow: 4 } ;
/// let cell_2 = Cell { pow: 4 } ;
/// // Sum of the same cell (structurally) yields the cell of the next power.
/// assert_eq!( cell_1 + cell_2, Some( Cell { pow: 5 } ) ) ;
/// let cell_2 = Cell { pow: 7 } ;
/// // Sum of different cells yields nothing.
/// assert_eq!( cell_1 + cell_2, None ) ;
/// ```
#[derive(Clone)]
pub struct Cell {
  /// The power of two.
  pow: u32,
}
impl Cell {
  /// The power of two stored.
  ///
  /// # Example
  ///
  /// ```
  /// let cell = Cell { pow: 4 } ;
  /// assert_eq!( cell.pow(), 4 )
  /// ```
  #[inline(always)]
  pub fn pow(& self) -> u32 { self.pow }
  /// Value of the cell.
  pub fn val(& self) -> usize { 2usize.pow(self.pow) }
}
// impl HasVal for Cell {
//   fn val(& self) -> usize { 2usize.pow(self.pow) }
// }
impl<'a, 'b> Add<& 'b Cell> for & 'a Cell {
  type Output = Option<Cell> ;
  fn add(self, rhs: & 'b Cell) -> Option<Cell> {
    if self.pow == rhs.pow {
      Some( Cell { pow: self.pow + 1 } )
    } else { None }
  }
}

/// Stores the grid and its info (score, seed and rng).
#[derive(Clone)]
pub struct Grid {
  /// Size of the (square) grid.
  size: usize,
  /// The grid of optional cells.
  grid: Vec< Vec<Option<Cell>> >,
  /// Current score.
  score: usize,
  /// The Seed.
  seed: Seed,
  /// The rng.
  rng: IsaacRng,
}
impl Grid {
  /// Default, empty frame.
  pub fn mk(seed: Seed) -> Self {
    let size = 4 ;
    let rng = seed.rng() ;
    Grid {
      size: size,
      grid: vec![ vec![None ; size] ; size ],
      score: 0,
      seed: seed,
      rng: rng,
    }
  }

  /// Height of the frame in characters.
  #[inline(always)]
  pub fn size(& self) -> usize { self.size }
  /// The score.
  #[inline(always)]
  pub fn score(& self) -> usize { self.score }
  /// The seed.
  #[inline(always)]
  pub fn seed(& self) -> & Seed { & self.seed }
  /// The grid.
  #[inline(always)]
  pub fn grid(& self) -> & Vec< Vec<Option<Cell>> > {
    & self.grid
  }

  /// The indices of the empty tiles of the grid.
  pub fn get_free(& self) -> Vec<(usize, usize)> {
    let mut vec = vec![] ;
    for row in 0..self.size {
      for col in 0..self.size {
        if self.grid[row][col].is_none() {
          vec.push( (row, col) )
        }
      }
    } ;
    vec
  }


  /// Moves a tile in a direction.
  ///
  /// # Parameters
  ///
  /// * `(row, col)`: indices of the *source* tile
  /// * `dir`: direction of the move
  /// * `merge`: if `false` potential merges will not be performed.
  ///   This happens when the source tile meets a tile that was merged during
  ///   this update.
  fn mv(
    & mut self, row: usize, col: usize, dir: Dir, merge: bool
  ) -> Evolution {
    let mut src = (row, col) ;
    let mut evolution = Evolution::Nothing ;
    if self.grid[row][col].is_none() { return evolution } ;
    loop {
      let (row, col) = src ;
      let tgt = match dir {
        Dir::Up => if row > 0 { (row - 1, col) } else { break },
        Dir::Dw => if row < self.size - 1 { (row + 1, col) } else { break },
        Dir::Lf => if col > 0 { (row, col -1) } else { break },
        Dir::Rg => if col < self.size -1 { (row, col + 1) } else { break },
      } ;

      let new = match self.grid[tgt.0][tgt.1] {
        None => match self.grid[src.0][src.1] {
          Some(ref source) => {
            evolution.mv() ;
            source.clone()
          },
          None => panic!(
            "v_move({},{},{:?}) but no cell there", row, col, dir
          ),
        },
        Some(ref target) => match self.grid[src.0][src.1] {
          Some(ref source) => if merge {
            match source + target {
              None => break,
              Some(new) => {
                evolution.mg(new.val()) ;
                new
              },
            }
          } else { break },
          None => panic!(
            "v_move({},{},{:?}) but no cell there", row, col, dir
          ),
        },
      } ;
      self.grid[src.0][src.1] = None ;
      self.grid[tgt.0][tgt.1] = Some(new) ;
      if evolution.is_merge() { break } ;
      src = tgt
    }
    self.score += evolution.score() ;
    evolution
  }

  /// Swipes the grid up.
  pub fn up(& mut self) -> Evolution {
    let mut res = Evolution::Nothing ;
    for col in 0..self.size {
      let mut merge = true ;
      for row in 1..self.size {
        let evol = self.mv(row, col, Dir::Up, merge) ;
        merge = (! merge) || (! evol.is_merge()) ;
        res = res + evol
      }
    } ;
    res
  }
  /// Swipes the grid down.
  pub fn down(& mut self) -> Evolution {
    let mut res = Evolution::Nothing ;
    for col in 0..self.size {
      let mut merge = true ;
      for row in ( 0..(self.size - 1) ).rev() {
        let evol = self.mv(row, col, Dir::Dw, merge) ;
        merge = (! merge) || (! evol.is_merge()) ;
        res = res + evol
      }
    } ;
    res
  }
  /// Swipes the grid left.
  pub fn left(& mut self) -> Evolution {
    let mut res = Evolution::Nothing ;
    for row in 0..self.size {
      let mut merge = true ;
      for col in 1..self.size {
        let evol = self.mv(row, col, Dir::Lf, merge) ;
        merge = (! merge) || (! evol.is_merge()) ;
        res = res + evol
      }
    } ;
    res
  }
  /// Swipes the grid right.
  pub fn right(& mut self) -> Evolution {
    let mut res = Evolution::Nothing ;
    for row in 0..self.size {
      let mut merge = true ;
      for col in ( 0..(self.size - 1) ).rev() {
        let evol = self.mv(row, col, Dir::Rg, merge) ;
        merge = (! merge) || (! evol.is_merge()) ;
        res = res + evol
      }
    } ;
    res
  }

  /// Spaws a cell on an empty tile using the internal rng.
  pub fn spawn(& mut self) -> bool {
    use rand::Rand ;
    let free = self.get_free() ;
    if free.is_empty() { false } else {
      let (row, col) = free[(u8::rand(& mut self.rng) as usize) % free.len()] ;
      let pow = if u8::rand(& mut self.rng) % 100u8 > 80u8 {
        2u32
      } else { 1u32 } ;
      self.grid[row][col] = Some( Cell { pow: pow } ) ;
      true
    }
  }

  pub fn add_tile_to_clone(&self, action: (usize, usize, i32, f32)) -> Grid {
     let mut new_grid = self.clone();
     let (x,y,val,_) = action;
     new_grid.grid[x][y] = Some( Cell {pow: val as u32/2} ) ;
     new_grid
  }
}
