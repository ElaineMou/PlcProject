//! Moves the cursor in the terminal.

use frame::Frame ;

#[cfg(
  any(
    target_os = "macos",
    target_os = "linux"
  )
)]
pub fn go_up_clear() {
  print!("\x1B[500D\x1B[1A\x1B[K")
}
#[cfg(target_os = "windows")]
pub fn go_up_clear() { () }

#[cfg(
  any(
    target_os = "macos",
    target_os = "linux"
  )
)]
pub fn goto_frame_top(frame: & Frame) {
  print!("\x1B[500D\x1B[K") ;
  for _ in 0..(frame.hght() + 3) {
    print!("\x1B[1A\x1B[K")
  }
}
#[cfg(target_os = "windows")]
pub fn goto_frame_top(_: & Frame) { () }

#[cfg(
  any(
    target_os = "macos",
    target_os = "linux"
  )
)]
pub fn goto_log() {
  print!("\x1B[500D\x1B[1A\x1B[K") ;
  print!("\x1B[1A\x1B[K")
}
#[cfg(target_os = "windows")]
pub fn goto_log() { () }