//! Textual version of the game 2048.

extern crate ansi_term as ansi ;
extern crate rand ;

mod common ;
pub mod cursor ;
pub mod clap ;
mod grid ;
pub mod frame ;
pub mod expectimax ;

pub use common::{ Dir, Evolution, Seed } ;
pub use expectimax::{ State, Move, ExpectiMax} ;
pub use grid::{ Cell, Grid } ;
pub use frame::Frame;

/// Entry point.
fn main() {
  use std::process::exit ;

  // Getting seed and painter from command line arguments.
  let (seed, painter,depth,expect) = match clap::parse() {
    Ok( (seed, painter, depth, expect) ) => (seed, painter, depth, expect),
    Err( (e, painter) ) => {
      println!("{}\n> {}", painter.error("Error:"), e) ;
      exit(2)
    },
  } ;
  let mut e = ExpectiMax::new(depth,expect);
  frame::rendering_loop(seed, painter, &mut e, ai_move)
}

fn ai_move(frame : & mut Frame, exp: & mut ExpectiMax) -> Evolution {
    let cur_state = State::new(Move::Start, 0, frame.grid().clone());
    let (s, _) = exp.max_layer(&cur_state);
    match s.move_used(){
        Move::Up => frame.up(),
        Move::Left => frame.left(),
        Move::Right => frame.right(),
        Move::Down => frame.down(),
        _ => {
            println!("I lost T_T") ;
            println!("") ;
            std::process::exit(0)
        }
    }
}
