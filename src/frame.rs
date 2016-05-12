//! Types for rendering.

use ansi::{ Style } ;

use cursor ;
use { Seed, Grid, Evolution } ;

pub use expectimax::{ ExpectiMax } ;

/// Stores the styles to paint numbers and game info.
///
/// Five styles to paint numbers, one for score and one for errors.
pub struct Painter {
  s_1: Style,
  s_2: Style,
  s_3: Style,
  s_4: Style,
  s_5: Style,
  score: Style,
  error: Style,
}
impl Painter {
  /// Default painter.
  ///
  /// * style 1: cyan
  /// * style 2: yellow
  /// * style 3: green
  /// * style 4: red
  /// * style 5: purple
  /// * score style: bold
  /// * error style: red bold
  #[inline]
  #[cfg(
    any(
      target_os = "macos",
      target_os = "linux"
    )
  )]
  pub fn default() -> Self {
    Painter {
      s_1: Colour::Cyan.normal(),
      s_2: Colour::Yellow.normal(),
      s_3: Colour::Green.normal(),
      s_4: Colour::Red.normal(),
      s_5: Colour::Purple.normal(),
      score: Style::new().bold(),
      error: Colour::Red.normal().bold(),
    }
  }

  /// Default painter (windows version, no styling).
  #[inline]
  #[cfg(target_os = "windows")]
  pub fn default() -> Self {
    Painter {
      s_1: Style::new(),
      s_2: Style::new(),
      s_3: Style::new(),
      s_4: Style::new(),
      s_5: Style::new(),
      score: Style::new(),
      error: Style::new(),
    }
  }

  /// No painter.
  ///
  /// * style 1: none
  /// * style 2: none
  /// * style 3: none
  /// * style 4: none
  /// * style 5: none
  /// * score style: none
  /// * error style: none
  #[inline]
  pub fn none() -> Self {
    Painter {
      s_1: Style::new(),
      s_2: Style::new(),
      s_3: Style::new(),
      s_4: Style::new(),
      s_5: Style::new(),
      score: Style::new(),
      error: Style::new(),
    }
  }

  /// Score painting style.
  #[inline]
  pub fn score(& self, s: & str) -> String {
    format!("{}", self.score.paint(s))
  }
  /// Paints an integer.
  pub fn paint(& self, val: usize) -> (String, usize) {
    let s = format!("{}", val) ;
    let len = s.len() ;
    let res = match s.clone().chars().next().unwrap() {
      '1' => self.s_1.paint(s),
      '2' => self.s_2.paint(s),
      '3' => self.s_3.paint(s),
      '4' => self.s_4.paint(s),
      '5' => self.s_5.paint(s),
      '6' => self.s_1.paint(s),
      '7' => self.s_2.paint(s),
      '8' => self.s_3.paint(s),
      '9' => self.s_4.paint(s),
      _ => self.s_5.paint(s),
    }.to_string() ;
    (res, len)
  }

  /// Paints an error.
  pub fn error(& self, bla: & str) -> String {
    format!("{}", self.error.paint(bla))
  }

  /// Sets the first style.
  #[inline(always)]
  pub fn set_1(& mut self, s: Style) { self.s_1 = s }
  /// Sets the second style.
  #[inline(always)]
  pub fn set_2(& mut self, s: Style) { self.s_1 = s }
  /// Sets the third style.
  #[inline(always)]
  pub fn set_3(& mut self, s: Style) { self.s_1 = s }
  /// Sets the fourth style.
  #[inline(always)]
  pub fn set_4(& mut self, s: Style) { self.s_1 = s }
  /// Sets the fifth style.
  #[inline(always)]
  pub fn set_5(& mut self, s: Style) { self.s_1 = s }
  /// Sets the score style.
  #[inline(always)]
  pub fn set_score(& mut self, s: Style) { self.score = s }
}


/// Represents a frame for theh renderer.
pub struct Frame {
  /// 2048 grid.
  grid: Grid,
  /// Vertical separation character.
  v_sep: char,
  /// Horizontal separation character.
  h_sep: char,
  /// Width of a tile (without separators).
  tile_wdth: usize,
  /// Height of a tile (without separators).
  tile_hght: usize,
  /// The painter.
  painter: Painter,
}

impl Frame {
  /// Default, empty frame.
  pub fn default(seed: Seed, painter: Painter) -> Self {
    Frame {
      v_sep: '|',
      h_sep: '-',
      tile_wdth: 6,
      tile_hght: 3,
      grid: Grid::mk(seed),
      painter: painter,
    }
  }

  /// Height of the frame in characters.
  #[inline(always)]
  pub fn hght(& self) -> usize {
    self.grid.size() * ( self.tile_hght + 1 ) + 1
  }
  /// Width of the frame in characters.
  #[inline(always)]
  pub fn wdth(& self) -> usize {
    self.grid.size() * ( self.tile_wdth + 1 ) + 1
  }
  /// The grid.
  #[inline(always)]
  pub fn grid(& self) -> & Grid { & self.grid }

  /// Spaws a cell on an empty tile using the internal rng.
  #[inline(always)]
  pub fn spawn(& mut self) -> bool { self.grid.spawn() }

  /// Moves the grid up.
  #[inline(always)]
  pub fn up(& mut self) -> Evolution { self.grid.up() }
  /// Moves the grid down.
  #[inline(always)]
  pub fn down(& mut self) -> Evolution { self.grid.down() }
  /// Moves the grid left.
  #[inline(always)]
  pub fn left(& mut self) -> Evolution { self.grid.left() }
  /// Moves the grid right.
  #[inline(always)]
  pub fn right(& mut self) -> Evolution { self.grid.right() }

  /// Adds a separation line to a list of strings.
  #[inline]
  fn draw_sep(& self, vec: & mut Vec<String>) {
    let wdth = self.wdth() ;
    let mut s = String::with_capacity(wdth) ;
    for _ in 0..wdth {
      s.push(self.h_sep)
    } ;
    vec.push(s)
  }
  /// Adds a row to a list of strings.
  fn draw_row(& self, row: usize, vec: & mut Vec<String>) {
    use std::iter::Extend ;
    let split = self.tile_hght / 2 ;
    let wdth = self.wdth() ;
    let mut lines = vec![ String::with_capacity(wdth) ; self.tile_hght ] ;

    for l in 0..self.tile_hght {
      lines[l].push(self.v_sep) ;
      for col in 0..self.grid.size() {
        match self.grid.grid()[row][col] {
          Some(ref cell) if l == split => {
            let (val, val_w) = self.painter.paint( cell.val() ) ;
            let pre = (self.tile_wdth - val_w) / 2 ;
            let suf = (self.tile_wdth - val_w) - pre ;
            for _ in 0..pre { lines[l].push(' ') } ;
            lines[l].push_str(& val) ;
            for _ in 0..suf { lines[l].push(' ') }
          },
          _ => for _ in 0..self.tile_wdth {
            lines[l].push(' ')
          },
        }
        lines[l].push(self.v_sep) ;
      }
    } ;
    vec.extend(lines) ;
    ()
  }

  /// The lines representing the grid.
  pub fn draw(& self) -> Vec<String> {
    let mut vec = Vec::with_capacity(self.wdth()) ;
    vec.push(
      format!(
        "|===| [{}] {}: {}",
        self.grid.seed(),
        self.painter.score("Score"),
        self.painter.paint(self.grid.score()).0
      )
    ) ;
    self.draw_sep(& mut vec) ;
    for row in 0..self.grid.size() {
      self.draw_row(row, & mut vec) ;
      self.draw_sep(& mut vec)
    } ;
    vec
  }

  /// Prints the result of `draw`.
  pub fn print(& self) {
    for line in self.draw() { println!("{}", line) }
  }

  /// Colors an error.
  pub fn error(& self, bla: & str) -> String {
    self.painter.error(bla)
  }
}

/// Rendering loop.
///
/// The function it takes as parameter is called to decide the next move to
/// make.
pub fn rendering_loop<Evol: Fn(& mut Frame, & mut ExpectiMax) -> Evolution>(
  seed: Seed, painter: Painter, exp: & mut ExpectiMax, update: Evol
) {
  use std::process::exit ;

  // Building first frame.
  let mut frame = Frame::default(seed, painter) ;

  // Spawn two random cells.
  frame.spawn() ;
  frame.spawn() ;

  // Print first frame.
  println!("") ;
  frame.print() ;
  println!("") ;

  // User input / print loop.
  loop {
    let evol = update(& mut frame, exp) ;
    println!("") ;
    if ! evol.changed() {
      // Nothing moved.
      cursor::goto_log() ;
      println!("Nothing moved") ;
      continue
    }
    // Try to spawn, if `! still_going` then we're done.
    let still_going = frame.spawn() ;
    if ! still_going { println!("You lost") ; exit(0) } ;

    // Move to top of frame to print next one.
    cursor::goto_frame_top(& frame) ;
    frame.print() ;
    println!("") ;
  }
}

/*
/// Rendering loop reacting to user input.
pub fn rendering_loop_user(seed: Seed, painter: Painter) {
  rendering_loop(seed, painter, read_user_move)
}*/
