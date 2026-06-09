//! dual-cursor — one artificial cursor (white) driven by a second physical
//! mouse via evdev; the first mouse controls the real system cursor normally.
//!
//! How it works
//! ============
//! * Two evdev reader threads grab their respective mouse devices and send
//!   movement deltas and button events through an mpsc channel.
//! * Mouse 0 (primary): its events are passed through as relative movements
//!   on a uinput REL device — it drives the real compositor cursor.
//! * Mouse 1 (secondary): controls an artificial cursor shown as a
//!   wlr-layer-shell OVERLAY surface (cur_white.png, pixels pre-scaled to
//!   CURSOR_DISPLAY_SCALE before upload so the compositor renders it at the
//!   correct size without any compositor-side scaling).
//! * Click-teleport: when mouse 1 presses a button, the real cursor is moved
//!   to the artificial cursor's position, the click is injected, and when
//!   mouse 1 releases ALL buttons (or mouse 0 clicks), the real cursor is
//!   warped back to where it was.
//!
//! Cursor image assets
//! ===================
//!   Place cur_white.png in the `assets/` folder at the project root.
//!   It must be an RGBA or RGB PNG file.
//!
//! Prerequisites
//! =============
//!   # Read mouse devices without sudo:
//!   sudo usermod -aG input $USER
//!
//!   # Write to /dev/uinput for click injection (re-login after each):
//!   sudo usermod -aG uinput $USER
//!   # — or temporarily (until reboot) —
//!   sudo chmod a+rw /dev/uinput
//!
//! Running as root (passing Wayland env vars)
//! ==========================================
//!   sudo WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
//!        ./target/release/double_cursor
//!
//! KDE / KWin — enable layer-shell if needed
//! ==========================================
//!   kwriteconfig6 --file kwinrc --group Plugins \
//!     --key kwin_wayland_layershellenabledPlugin true
//!   qdbus6 org.kde.KWin /KWin reconfigure

use std::{
    env,
    io::Write,
    os::unix::io::AsFd,
    path::PathBuf,
    thread,
};

use evdev::{
    uinput::VirtualDevice,
    AbsInfo, AbsoluteAxisCode, AttributeSet, Device, EventSummary,
    EventType, InputEvent, KeyCode, RelativeAxisCode, UinputAbsSetup,
};
use wayland_client::{
    delegate_noop,
    protocol::{
        wl_buffer, wl_compositor, wl_output, wl_pointer, wl_registry,
        wl_seat, wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, QueueHandle,
};
use wayland_protocols::xdg::xdg_output::zv1::client::{
    zxdg_output_manager_v1, zxdg_output_v1,
};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{self, Layer},
    zwlr_layer_surface_v1::{self, Anchor, ZwlrLayerSurfaceV1},
};

// ─────────────────────────────────────────────────────────────────────────────
// Tunables
// ─────────────────────────────────────────────────────────────────────────────

const SPEED: f64 = 1.25;

/// The artificial cursor is rendered at this fraction of the PNG's natural size.
const CURSOR_DISPLAY_SCALE: f64 = 0.775;

// Linux key codes for left/right mouse buttons
const BTN_LEFT:  u16 = 0x110;
const BTN_RIGHT: u16 = 0x111;

// ─────────────────────────────────────────────────────────────────────────────
// Cursor images — embedded at compile time from assets/
// ─────────────────────────────────────────────────────────────────────────────

const CURSOR_WHITE_PNG: &[u8] = include_bytes!("../assets/cur_white.png");

/// Decode a PNG (RGB or RGBA, 8-bit) into raw ARGB8888 bytes as expected by
/// wl_shm (little-endian 0xAARRGGBB → [B, G, R, A] in memory).
fn load_png_as_argb8888(data: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().expect("PNG decode failed");

    let width      = reader.info().width;
    let height     = reader.info().height;
    let color_type = reader.info().color_type;
    let bit_depth  = reader.info().bit_depth;

    assert!(
        matches!(bit_depth, png::BitDepth::Eight),
        "Cursor PNG must be 8-bit; got {bit_depth:?}"
    );

    let mut img_buf = vec![0u8; reader.output_buffer_size()];
    let frame       = reader.next_frame(&mut img_buf).expect("PNG frame read failed");
    let img         = &img_buf[..frame.buffer_size()];

    let pixels = (width * height) as usize;
    let mut argb = vec![0u8; pixels * 4];

    match color_type {
        png::ColorType::Rgba => {
            for i in 0..pixels {
                let s = i * 4;
                argb[s]     = img[s + 2]; // B
                argb[s + 1] = img[s + 1]; // G
                argb[s + 2] = img[s];     // R
                argb[s + 3] = img[s + 3]; // A
            }
        }
        png::ColorType::Rgb => {
            for i in 0..pixels {
                let s = i * 3;
                argb[i * 4]     = img[s + 2]; // B
                argb[i * 4 + 1] = img[s + 1]; // G
                argb[i * 4 + 2] = img[s];     // R
                argb[i * 4 + 3] = 0xFF;        // A — fully opaque
            }
        }
        other => panic!(
            "Unsupported PNG color type {other:?}. Re-export as RGBA or RGB."
        ),
    }

    (width, height, argb)
}

/// Nearest-neighbour downscale of an ARGB8888 image.
/// Returns the new (width, height, pixels).
fn scale_argb8888(src: &[u8], src_w: u32, src_h: u32, scale: f64) -> (u32, u32, Vec<u8>) {
    let dst_w = ((src_w as f64 * scale) as u32).max(1);
    let dst_h = ((src_h as f64 * scale) as u32).max(1);
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = ((dx as f64 / dst_w as f64) * src_w as f64) as u32;
            let sy = ((dy as f64 / dst_h as f64) * src_h as f64) as u32;
            let si = ((sy * src_w + sx) * 4) as usize;
            let di = ((dy * dst_w + dx) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    (dst_w, dst_h, dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Wayland connection — tolerates missing env vars (common under sudo)
// ─────────────────────────────────────────────────────────────────────────────

fn connect_wayland() -> Connection {
    if let Ok(conn) = Connection::connect_to_env() {
        return conn;
    }

    // Fallback: scan /run/user/<uid>/ for wayland-* sockets.
    let uid = libc_getuid();
    let runtime_dir = format!("/run/user/{uid}");
    if let Ok(entries) = std::fs::read_dir(&runtime_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("wayland-") && !name.ends_with(".lock") {
                let display = name.to_string();
                std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
                std::env::set_var("WAYLAND_DISPLAY", &display);
                if let Ok(conn) = Connection::connect_to_env() {
                    eprintln!("[wayland] auto-connected via {runtime_dir}/{display}");
                    return conn;
                }
            }
        }
    }

    eprintln!(
        "ERROR: Could not connect to Wayland compositor.\n\
         \n\
         This usually happens when running with sudo, which strips WAYLAND_DISPLAY\n\
         and XDG_RUNTIME_DIR.  Fix with one of:\n\
         \n\
         Option A (recommended) — add yourself to the input group (no sudo needed):\n\
           sudo usermod -aG input $USER\n\
           # log out and back in, then run: ./target/release/double_cursor\n\
         \n\
         Option B — pass Wayland env vars through sudo:\n\
           sudo WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \\\n\
                ./target/release/double_cursor\n"
    );
    std::process::exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mouse discovery (evdev)
// ─────────────────────────────────────────────────────────────────────────────

fn discover_mice(limit: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for i in 0..=99u32 {
        if found.len() >= limit { break; }
        let p = PathBuf::from(format!("/dev/input/event{i}"));
        if !p.exists() { continue; }
        if let Ok(dev) = Device::open(&p) {
            let has_rel_xy = dev.supported_relative_axes().map_or(false, |a| {
                a.contains(RelativeAxisCode::REL_X) && a.contains(RelativeAxisCode::REL_Y)
            });
            // Keyboards also expose REL_X/REL_Y for scroll wheels, but they
            // have letter keys — use KEY_A as a reliable keyboard marker.
            let is_keyboard = dev.supported_keys().map_or(false, |k| {
                k.contains(evdev::KeyCode::KEY_A)
            });
            if has_rel_xy && !is_keyboard {
                println!(
                    "[discovery] {} – {}",
                    p.display(),
                    dev.name().unwrap_or("unknown")
                );
                found.push(p);
            }
        }
    }
    found
}

// ─────────────────────────────────────────────────────────────────────────────
// uinput virtual mouse for click injection (ABS — used for the artificial
// cursor's click-teleport to the real cursor position)
// ─────────────────────────────────────────────────────────────────────────────

fn create_virtual_abs_mouse(name: &str, screen_w: i32, screen_h: i32) -> Option<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    for k in [
        KeyCode::BTN_LEFT,
        KeyCode::BTN_RIGHT,
        KeyCode::BTN_MIDDLE,
        KeyCode::BTN_SIDE,
        KeyCode::BTN_EXTRA,
        KeyCode::BTN_FORWARD,
        KeyCode::BTN_BACK,
        KeyCode::BTN_TASK,
    ] {
        keys.insert(k);
    }

    let result: std::io::Result<VirtualDevice> = (|| {
        VirtualDevice::builder()?
            .name(name)
            .with_keys(&keys)?
            .with_absolute_axis(&UinputAbsSetup::new(
                AbsoluteAxisCode::ABS_X,
                AbsInfo::new(0, 0, screen_w, 0, 0, 1),
            ))?
            .with_absolute_axis(&UinputAbsSetup::new(
                AbsoluteAxisCode::ABS_Y,
                AbsInfo::new(0, 0, screen_h, 0, 0, 1),
            ))?
            .build()
    })();

    match result {
        Ok(dev) => {
            println!("[uinput] Virtual ABS mouse '{name}' created");
            Some(dev)
        }
        Err(e) => {
            eprintln!(
                "[uinput] Could not create virtual ABS mouse '{name}': {e}\n\
                 Click injection disabled.\n\
                 To enable: sudo usermod -aG uinput $USER  (then re-login)\n\
                 or: sudo chmod a+rw /dev/uinput"
            );
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// uinput REL mouse — used to pass mouse-0 movement through to the compositor
// so the real cursor moves normally.
// ─────────────────────────────────────────────────────────────────────────────

fn create_virtual_rel_mouse(name: &str) -> Option<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    for k in [
        KeyCode::BTN_LEFT,
        KeyCode::BTN_RIGHT,
        KeyCode::BTN_MIDDLE,
        KeyCode::BTN_SIDE,
        KeyCode::BTN_EXTRA,
        KeyCode::BTN_FORWARD,
        KeyCode::BTN_BACK,
        KeyCode::BTN_TASK,
    ] {
        keys.insert(k);
    }
    let mut axes = AttributeSet::<RelativeAxisCode>::new();
    axes.insert(RelativeAxisCode::REL_X);
    axes.insert(RelativeAxisCode::REL_Y);
    axes.insert(RelativeAxisCode::REL_WHEEL);
    axes.insert(RelativeAxisCode::REL_HWHEEL);

    let result: std::io::Result<VirtualDevice> = (|| {
        VirtualDevice::builder()?
            .name(name)
            .with_keys(&keys)?
            .with_relative_axes(&axes)?
            .build()
    })();

    match result {
        Ok(dev) => {
            println!("[uinput] Virtual REL mouse '{name}' created (primary cursor pass-through)");
            Some(dev)
        }
        Err(e) => {
            eprintln!("[uinput] Could not create virtual REL mouse '{name}': {e}");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SHM buffer helper
// ─────────────────────────────────────────────────────────────────────────────

fn make_shm_buffer(
    shm:       &wl_shm::WlShm,
    qh:        &QueueHandle<AppState>,
    width:     u32,
    height:    u32,
    argb_data: &[u8],
) -> (wl_shm_pool::WlShmPool, wl_buffer::WlBuffer) {
    let mut file = tempfile::tempfile().expect("tempfile");
    file.write_all(argb_data).expect("write shm");
    let pool = shm.create_pool(file.as_fd(), argb_data.len() as i32, qh, ());
    let buf  = pool.create_buffer(
        0, width as i32, height as i32, (width * 4) as i32,
        wl_shm::Format::Argb8888, qh, (),
    );
    (pool, buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel message
// ─────────────────────────────────────────────────────────────────────────────

enum Msg {
    /// Mouse movement delta from one physical device
    Move { idx: usize, dx: f64, dy: f64 },
    /// Mouse button press / release from one physical device
    /// `code`  — raw Linux key code (BTN_LEFT = 0x110, etc.)
    /// `value` — 0 = release, 1 = press, 2 = autorepeat
    Button { idx: usize, code: u16, value: i32 },
    /// Scroll wheel tick(s) — dv = vertical delta, dh = horizontal delta
    Scroll { idx: usize, dv: i32, dh: i32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-cursor state (only one artificial cursor now)
// ─────────────────────────────────────────────────────────────────────────────

struct Cursor {
    surface:       wl_surface::WlSurface,
    layer_surface: ZwlrLayerSurfaceV1,
    _pool:         wl_shm_pool::WlShmPool,
    buffer:        wl_buffer::WlBuffer,
    x: i32,
    y: i32,
    /// Display size (half of the PNG's natural size)
    display_size: i32,
    configured: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────

struct AppState {
    running:       bool,
    compositor:    Option<wl_compositor::WlCompositor>,
    shm:           Option<wl_shm::WlShm>,
    layer_shell:   Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    output:        Option<wl_output::WlOutput>,
    xdg_output_manager: Option<zxdg_output_manager_v1::ZxdgOutputManagerV1>,
    xdg_output:    Option<zxdg_output_v1::ZxdgOutputV1>,
    seat:          Option<wl_seat::WlSeat>,
    pointer:       Option<wl_pointer::WlPointer>,
    /// Fullscreen bottom-layer surface used solely to receive pointer-enter
    /// events so we can call set_cursor(NULL) and hide the real cursor.
    pointer_surface: Option<wl_surface::WlSurface>,
    /// The single artificial cursor (driven by mouse 1).
    cursors:       Vec<Cursor>,
    screen_w:      i32,
    screen_h:      i32,
    output_scale:  i32,
    rx:            std::sync::mpsc::Receiver<Msg>,
    /// Natural PNG size (before halving for display)
    cursor_size:   u32,
    cursor_pixels: Vec<u8>,
    /// ABS virtual device — used for click-teleport injection at the
    /// artificial cursor's position.
    abs_vdev:      Option<VirtualDevice>,
    /// REL virtual device — passes mouse-0 movement to the compositor so
    /// the real cursor behaves normally.
    rel_vdev:      Option<VirtualDevice>,
    /// Saved real-cursor position before a click-teleport (screen coords).
    /// None means we are not currently teleported.
    saved_real_pos: Option<(i32, i32)>,
    /// How many mouse-1 buttons are currently held down.
    mouse1_buttons_held: u32,
    /// Last known real cursor position (updated from mouse-0 REL movements).
    real_cursor_x: i32,
    real_cursor_y: i32,
}

impl AppState {
    fn try_create_cursors(&mut self, qh: &QueueHandle<Self>) {
        if !self.cursors.is_empty() { return; }
        if self.screen_w == 0 || self.screen_h == 0 { return; }
        let (comp, shm, shell, output) = match (
            self.compositor.as_ref(), self.shm.as_ref(),
            self.layer_shell.as_ref(), self.output.as_ref(),
        ) {
            (Some(c), Some(s), Some(sh), Some(o)) => (c, s, sh, o),
            _ => return,
        };

        let png_size     = self.cursor_size;
        let display_size = png_size;
        let start_x      = self.screen_w * 3 / 4;
        let start_y      = self.screen_h / 2;

        let surface = comp.create_surface(qh, ());
        let layer_surface = shell.get_layer_surface(
            &surface, Some(output), Layer::Overlay,
            "dual-cursor-1".to_string(), qh, (),
        );
        layer_surface.set_anchor(Anchor::Top | Anchor::Left);
        // Surface size equals the (already scaled) pixel buffer size
        layer_surface.set_size(display_size, display_size);
        layer_surface.set_keyboard_interactivity(
            zwlr_layer_surface_v1::KeyboardInteractivity::None,
        );
        layer_surface.set_exclusive_zone(-1);

        let (pool, buf) = make_shm_buffer(shm, qh, png_size, png_size, &self.cursor_pixels);

        surface.commit();

        self.cursors.push(Cursor {
            surface,
            layer_surface,
            _pool: pool,
            buffer: buf,
            x: start_x,
            y: start_y,
            display_size: display_size as i32,
            configured: false,
        });
    }

    fn move_cursor(&mut self, idx: usize, dx: f64, dy: f64) {
        if idx == 0 {
            // Mouse 0: pass movement through to the REL virtual device so the
            // real compositor cursor moves.
            self.real_cursor_x = (self.real_cursor_x + dx as i32)
                .clamp(0, self.screen_w - 1);
            self.real_cursor_y = (self.real_cursor_y + dy as i32)
                .clamp(0, self.screen_h - 1);

            if let Some(vdev) = &mut self.rel_vdev {
                let events = [
                    InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_X.0, dx as i32),
                    InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_Y.0, dy as i32),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                if let Err(e) = vdev.emit(&events) {
                    eprintln!("[uinput-rel] emit error: {e}");
                }
            }
            return;
        }

        // Mouse 1: move the artificial cursor surface
        if self.cursors.is_empty() { return; }
        let c = &mut self.cursors[0];
        if !c.configured { return; }
        c.x = (c.x + dx as i32).clamp(0, self.screen_w - 1);
        c.y = (c.y + dy as i32).clamp(0, self.screen_h - 1);
        let (cx, cy) = (c.x, c.y);
        c.layer_surface.set_margin(cy, 0, 0, cx);
        c.surface.commit();

        // While a button is held (drag in progress), keep the real cursor
        // tracking the artificial one so the drag target sees movement.
        if self.saved_real_pos.is_some() {
            if let Some(vdev) = &mut self.abs_vdev {
                let events = [
                    InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_X.0, cx),
                    InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_Y.0, cy),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                let _ = vdev.emit(&events);
            }
        }
    }

    fn commit_cursor(&mut self, idx: usize) {
        if idx >= self.cursors.len() { return; }
        let c            = &mut self.cursors[idx];
        let display_size = c.display_size;
        c.layer_surface.set_margin(c.y, 0, 0, c.x);
        c.surface.attach(Some(&c.buffer), 0, 0);
        c.surface.damage(0, 0, display_size, display_size);
        c.surface.commit();
    }

    /// Swap BTN_LEFT ↔ BTN_RIGHT for mouse 1 (left-handed swap).
    fn remap_button_code(idx: usize, code: u16) -> u16 {
        if idx == 1 {
            if code == BTN_LEFT  { return BTN_RIGHT; }
            if code == BTN_RIGHT { return BTN_LEFT;  }
        }
        code
    }

    /// Handle button events from either mouse.
    ///
    /// Mouse 0: inject directly via the REL virtual device (normal click).
    /// Mouse 1: swap L/R, then:
    ///   - on first press  → teleport real cursor to artificial position, inject press
    ///   - on release of all buttons → inject release, restore real cursor
    ///   If mouse 0 presses a button while teleported → restore real cursor first.
    fn handle_button(&mut self, idx: usize, code: u16, value: i32) {
        let mapped_code = Self::remap_button_code(idx, code);

        if idx == 0 {
            // Mouse 0 pressed while we are teleported → restore immediately,
            // then emit the click at the restored position.
            if value == 1 && self.saved_real_pos.is_some() {
                self.restore_real_cursor();
            }
            if let Some(vdev) = &mut self.rel_vdev {
                let events = [
                    InputEvent::new(EventType::KEY.0, mapped_code, value),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                if let Err(e) = vdev.emit(&events) {
                    eprintln!("[uinput-rel] button emit error: {e}");
                }
            }
            return;
        }

        // Mouse 1 ──────────────────────────────────────────────────────────
        if self.cursors.is_empty() { return; }
        let (cx, cy) = (self.cursors[0].x, self.cursors[0].y);

        if value == 1 {
            // Press: teleport (or we're already teleported from a previous press)
            self.mouse1_buttons_held += 1;
            if self.saved_real_pos.is_none() {
                // Save real cursor position and teleport
                self.saved_real_pos = Some((self.real_cursor_x, self.real_cursor_y));
                if let Some(vdev) = &mut self.abs_vdev {
                    let events = [
                        InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_X.0, cx),
                        InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_Y.0, cy),
                        InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                    ];
                    let _ = vdev.emit(&events);
                }
            }
            // Inject button press at the teleported position
            if let Some(vdev) = &mut self.abs_vdev {
                let events = [
                    InputEvent::new(EventType::KEY.0, mapped_code, 1),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                if let Err(e) = vdev.emit(&events) {
                    eprintln!("[uinput-abs] press emit error: {e}");
                }
            }
        } else if value == 0 {
            // Release
            if self.mouse1_buttons_held > 0 {
                self.mouse1_buttons_held -= 1;
            }
            // Inject release first, then restore if all buttons are up
            if let Some(vdev) = &mut self.abs_vdev {
                let events = [
                    InputEvent::new(EventType::KEY.0, mapped_code, 0),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                if let Err(e) = vdev.emit(&events) {
                    eprintln!("[uinput-abs] release emit error: {e}");
                }
            }
            if self.mouse1_buttons_held == 0 {
                self.restore_real_cursor();
            }
        }
    }

    /// Move the real cursor back to where it was before the teleport.
    fn restore_real_cursor(&mut self) {
        if let Some((rx, ry)) = self.saved_real_pos.take() {
            if let Some(vdev) = &mut self.abs_vdev {
                let events = [
                    InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_X.0, rx),
                    InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_Y.0, ry),
                    InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                ];
                let _ = vdev.emit(&events);
            }
        }
        self.mouse1_buttons_held = 0;
    }

    /// Forward scroll wheel events.
    /// Both mice scroll goes to the REL vdevice (the real cursor's scroll
    /// context). Mouse 1 also forwards to the ABS vdevice when teleported so
    /// the target window receives scroll events during a click-hold.
    fn handle_scroll(&mut self, idx: usize, dv: i32, dh: i32) {
        // Always emit on the REL device (mouse 0's real-cursor context).
        if let Some(vdev) = &mut self.rel_vdev {
            let mut events: Vec<InputEvent> = Vec::new();
            if dv != 0 {
                events.push(InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_WHEEL.0, dv));
            }
            if dh != 0 {
                events.push(InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_HWHEEL.0, dh));
            }
            events.push(InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0));
            if let Err(e) = vdev.emit(&events) {
                eprintln!("[uinput-rel] scroll emit error: {e}");
            }
        }
        // If mouse 1 scrolls while teleported, also send on the ABS device so
        // the window under the artificial cursor receives the scroll.
        if idx == 1 && self.saved_real_pos.is_some() {
            if let Some(vdev) = &mut self.abs_vdev {
                let mut events: Vec<InputEvent> = Vec::new();
                if dv != 0 {
                    events.push(InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_WHEEL.0, dv));
                }
                if dh != 0 {
                    events.push(InputEvent::new(EventType::RELATIVE.0, RelativeAxisCode::REL_HWHEEL.0, dh));
                }
                events.push(InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0));
                if let Err(e) = vdev.emit(&events) {
                    eprintln!("[uinput-abs] scroll emit error: {e}");
                }
            }
        }
    }

    /// Once we have compositor, layer_shell, output, and seat, create a
    /// fullscreen Bottom-layer surface that accepts pointer input.  When the
    /// pointer enters it we call set_cursor(NULL) to hide the real cursor.
    fn try_hide_cursor(&mut self, qh: &QueueHandle<Self>) {
        if self.pointer_surface.is_some() { return; }
        let (comp, shell, output, seat) = match (
            self.compositor.as_ref(), self.layer_shell.as_ref(),
            self.output.as_ref(),     self.seat.as_ref(),
        ) {
            (Some(c), Some(sh), Some(o), Some(s)) => (c, sh, o, s),
            _ => return,
        };
        if self.pointer.is_none() {
            self.pointer = Some(seat.get_pointer(qh, ()));
        }
        let surface = comp.create_surface(qh, ());
        let layer_surface = shell.get_layer_surface(
            &surface, Some(output), Layer::Bottom,
            "cursor-hider".to_string(), qh, (),
        );
        layer_surface.set_anchor(Anchor::Top | Anchor::Left | Anchor::Right | Anchor::Bottom);
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
        surface.commit();
        self.pointer_surface = Some(surface);
        let _ = layer_surface;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch implementations
// ─────────────────────────────────────────────────────────────────────────────

impl Dispatch<wl_registry::WlRegistry, ()> for AppState {
    fn event(
        state: &mut Self, registry: &wl_registry::WlRegistry,
        event: wl_registry::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(
                        registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind::<wl_shm::WlShm, _, _>(name, 1, qh, ()));
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(
                        registry.bind::<zwlr_layer_shell_v1::ZwlrLayerShellV1, _, _>(
                            name, version.min(4), qh, ()));
                }
                "wl_output" => {
                    if state.output.is_none() {
                        let output = registry.bind::<wl_output::WlOutput, _, _>(name, 2, qh, ());
                        if let Some(mgr) = &state.xdg_output_manager {
                            state.xdg_output = Some(mgr.get_xdg_output::<_, AppState>(&output, qh, ()));
                        }
                        state.output = Some(output);
                    }
                }
                "zxdg_output_manager_v1" => {
                    let mgr = registry.bind::<zxdg_output_manager_v1::ZxdgOutputManagerV1, _, _>(
                        name, version.min(3), qh, ());
                    if let Some(output) = &state.output {
                        state.xdg_output = Some(mgr.get_xdg_output::<_, AppState>(output, qh, ()));
                    }
                    state.xdg_output_manager = Some(mgr);
                }
                "wl_seat" => {
                    if state.seat.is_none() {
                        state.seat = Some(
                            registry.bind::<wl_seat::WlSeat, _, _>(name, 1, qh, ()));
                    }
                }
                _ => {}
            }
            state.try_create_cursors(qh);
            state.try_hide_cursor(qh);
        }
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for AppState {
    fn event(
        state: &mut Self, ls: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event, _: &(),
        _: &Connection, _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial, .. } => {
                ls.ack_configure(serial);
                if let Some(idx) = state.cursors.iter().position(|c| &c.layer_surface == ls) {
                    state.cursors[idx].configured = true;
                    state.commit_cursor(idx);
                }
            }
            zwlr_layer_surface_v1::Event::Closed => { state.running = false; }
            _ => {}
        }
    }
}

delegate_noop!(AppState: ignore wl_compositor::WlCompositor);
delegate_noop!(AppState: ignore wl_surface::WlSurface);
delegate_noop!(AppState: ignore wl_shm::WlShm);
delegate_noop!(AppState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(AppState: ignore wl_buffer::WlBuffer);
delegate_noop!(AppState: ignore zwlr_layer_shell_v1::ZwlrLayerShellV1);

impl Dispatch<wl_output::WlOutput, ()> for AppState {
    fn event(
        state: &mut Self, _: &wl_output::WlOutput,
        event: wl_output::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_output::Event::Scale { factor } => {
                state.output_scale = factor;
            }
            wl_output::Event::Mode { flags, width, height, .. }
            if flags.into_result().map_or(false, |f| f.contains(wl_output::Mode::Current)) =>
                {
                    if state.screen_w == 0 {
                        let logical_w = width  / state.output_scale.max(1);
                        let logical_h = height / state.output_scale.max(1);
                        state.screen_w = logical_w;
                        state.screen_h = logical_h;
                        println!("[output] Mode fallback (no xdg_output): {logical_w}×{logical_h}px");
                        state.abs_vdev = create_virtual_abs_mouse("dual-cursor-abs", logical_w, logical_h);
                        state.try_create_cursors(qh);
                    }
                }
            _ => {}
        }
    }
}

impl Dispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for AppState {
    fn event(_: &mut Self, _: &zxdg_output_manager_v1::ZxdgOutputManagerV1,
             _: zxdg_output_manager_v1::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<zxdg_output_v1::ZxdgOutputV1, ()> for AppState {
    fn event(
        state: &mut Self, _: &zxdg_output_v1::ZxdgOutputV1,
        event: zxdg_output_v1::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>,
    ) {
        if let zxdg_output_v1::Event::LogicalSize { width, height } = event {
            state.screen_w = width;
            state.screen_h = height;
            println!("[output] Logical size (xdg_output): {width}×{height}px");
            state.abs_vdev = create_virtual_abs_mouse("dual-cursor-abs", width, height);
            state.try_create_cursors(qh);
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for AppState {
    fn event(
        state: &mut Self, seat: &wl_seat::WlSeat,
        event: wl_seat::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities { capabilities } = event {
            let has_pointer = capabilities
                .into_result()
                .map_or(false, |c| c.contains(wl_seat::Capability::Pointer));
            if has_pointer && state.pointer.is_none() {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for AppState {
    fn event(
        _state: &mut Self, pointer: &wl_pointer::WlPointer,
        event: wl_pointer::Event, _: &(), _: &Connection, _: &QueueHandle<Self>,
    ) {
        if let wl_pointer::Event::Enter { serial, .. } = event {
            pointer.set_cursor(serial, None, 0, 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    // 0. Load cursor image (compile-time embedded from assets/)
    let (w0, h0, px0_full) = load_png_as_argb8888(CURSOR_WHITE_PNG);
    let (sw, sh, px0) = scale_argb8888(&px0_full, w0, h0, CURSOR_DISPLAY_SCALE);
    let cursor_size = sw;
    println!("Cursor image size: {w0}×{h0}px (displayed at {sw}×{sh}px)");

    // 1. Resolve device paths
    let cli: Vec<String> = env::args().skip(1).collect();
    let devices: Vec<PathBuf> = if cli.len() >= 2 {
        cli.iter().take(2).map(PathBuf::from).collect()
    } else {
        let found = discover_mice(2);
        if found.len() < 2 {
            eprintln!(
                "Need 2 mouse devices; found {}.\n\
                 Add yourself to the input group (sudo usermod -aG input $USER, then re-login)\n\
                 or specify devices explicitly: double_cursor /dev/input/eventX /dev/input/eventY",
                found.len()
            );
            std::process::exit(1);
        }
        found
    };
    println!("Mouse 0 (real cursor)       → {}", devices[0].display());
    println!("Mouse 1 (artificial cursor) → {}", devices[1].display());

    // 2. Event channel
    let (tx, rx) = std::sync::mpsc::channel::<Msg>();

    // 3. Spawn evdev reader threads — one per physical mouse
    for (idx, path) in devices.iter().enumerate() {
        let path = path.clone();
        let tx   = tx.clone();
        thread::Builder::new()
            .name(format!("mouse-{idx}"))
            .spawn(move || {
                let mut dev = match Device::open(&path) {
                    Ok(d) => d,
                    Err(e) => { eprintln!("[mouse-{idx}] open: {e}"); return; }
                };
                match dev.grab() {
                    Ok(_)  => {},
                    Err(e) => eprintln!(
                        "[mouse-{idx}] grab failed ({e}) — system cursor will also move.\n\
                         Fix: sudo usermod -aG input $USER, then re-login and run WITHOUT sudo."
                    ),
                }
                let (mut pdx, mut pdy) = (0.0f64, 0.0f64);
                let (mut pdv, mut pdh) = (0i32, 0i32);
                loop {
                    match dev.fetch_events() {
                        Err(e) => { eprintln!("[mouse-{idx}] read error: {e}"); break; }
                        Ok(evs) => for ev in evs {
                            match ev.destructure() {
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_X, v) =>
                                    pdx += v as f64 * SPEED,
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_Y, v) =>
                                    pdy += v as f64 * SPEED,
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_WHEEL, v) =>
                                    pdv += v,
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_HWHEEL, v) =>
                                    pdh += v,
                                EventSummary::Synchronization(..) => {
                                    if pdx != 0.0 || pdy != 0.0 {
                                        let _ = tx.send(Msg::Move { idx, dx: pdx, dy: pdy });
                                        pdx = 0.0; pdy = 0.0;
                                    }
                                    if pdv != 0 || pdh != 0 {
                                        let _ = tx.send(Msg::Scroll { idx, dv: pdv, dh: pdh });
                                        pdv = 0; pdh = 0;
                                    }
                                }
                                EventSummary::Key(_, key, value) => {
                                    let code = key.code();
                                    if code >= 0x110 {
                                        let _ = tx.send(Msg::Button { idx, code, value });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            })
            .expect("thread spawn");
    }
    drop(tx);

    // 4. Create REL virtual device for mouse-0 pass-through immediately
    //    (doesn't need screen dimensions).
    let rel_vdev = create_virtual_rel_mouse("dual-cursor-rel");

    // 5. Connect to Wayland
    let conn = connect_wayland();
    let mut event_queue = conn.new_event_queue::<AppState>();
    let qh = event_queue.handle();
    conn.display().get_registry(&qh, ());

    let mut state = AppState {
        running: true,
        compositor: None, shm: None, layer_shell: None, output: None,
        xdg_output_manager: None, xdg_output: None,
        seat: None, pointer: None, pointer_surface: None,
        cursors: Vec::new(),
        screen_w: 0,
        screen_h: 0,
        output_scale: 1,
        rx,
        cursor_size,
        cursor_pixels: px0,
        abs_vdev: None,
        rel_vdev,
        saved_real_pos: None,
        mouse1_buttons_held: 0,
        real_cursor_x: 0,
        real_cursor_y: 0,
    };

    // First roundtrip: bind globals, receive wl_output::Mode, create layer surfaces
    event_queue.roundtrip(&mut state).expect("initial roundtrip");
    // Second roundtrip: receive Configure events
    event_queue.roundtrip(&mut state).expect("configure roundtrip");

    if state.screen_w == 0 || state.screen_h == 0 {
        eprintln!("WARNING: wl_output did not report screen size; defaulting to 3840×2160.");
        state.screen_w = 3840;
        state.screen_h = 2160;
    }
    if state.abs_vdev.is_none() {
        state.abs_vdev = create_virtual_abs_mouse("dual-cursor-abs", state.screen_w, state.screen_h);
    }
    // Start real_cursor_x/y in the middle of the screen as a reasonable default.
    state.real_cursor_x = state.screen_w / 2;
    state.real_cursor_y = state.screen_h / 2;

    if state.layer_shell.is_none() {
        eprintln!(
            "ERROR: compositor does not advertise zwlr_layer_shell_v1.\n\
             Enable it in KWin:\n\
             \n\
             kwriteconfig6 --file kwinrc --group Plugins \\\n\
               --key kwin_wayland_layershellenabledPlugin true\n\
             qdbus6 org.kde.KWin /KWin reconfigure\n\
             \n\
             Then run this program again."
        );
        std::process::exit(1);
    }

    if state.cursors.iter().filter(|c| c.configured).count() < 1 {
        eprintln!("WARNING: artificial cursor has not received a Configure event yet — it may appear after first mouse-1 movement.");
    }

    println!("Running.");
    println!("  Mouse 0 → real cursor (normal behaviour).");
    println!("  Mouse 1 → artificial white cursor (half-size, left/right buttons swapped).");
    println!("  Mouse 1 click → teleports real cursor there, clicks, then returns it.");
    println!("  Ctrl-C to quit.");

    // 6. Main event loop
    while state.running {
        event_queue.flush().ok();

        loop {
            match state.rx.try_recv() {
                Ok(msg) => match msg {
                    Msg::Move   { idx, dx, dy }      => state.move_cursor(idx, dx, dy),
                    Msg::Button { idx, code, value } => state.handle_button(idx, code, value),
                    Msg::Scroll { idx, dv, dh }      => state.handle_scroll(idx, dv, dh),
                },
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("All mouse threads exited.");
                    state.running = false;
                    break;
                }
            }
        }

        event_queue.flush().ok();

        if let Some(guard) = event_queue.prepare_read() {
            use std::os::unix::io::AsRawFd;
            let fd = guard.connection_fd();
            let mut pfd = PollFd { fd: fd.as_raw_fd(), events: 0x0001, revents: 0 };
            unsafe {
                extern "C" { fn poll(fds: *mut PollFd, n: u64, timeout: i32) -> i32; }
                poll(&mut pfd, 1, 4);
            }
            guard.read().ok();
            event_queue.dispatch_pending(&mut state).ok();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal libc helpers (avoids an extra dependency)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
struct PollFd { fd: i32, events: i16, revents: i16 }

fn libc_getuid() -> u32 {
    extern "C" { fn getuid() -> u32; }
    unsafe { getuid() }
}
