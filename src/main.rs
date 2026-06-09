//! dual-cursor — two independent artificial cursors on Wayland, each driven by
//! a separate physical mouse via evdev.
//!
//! How it works
//! ============
//! * Two evdev reader threads grab their respective mouse devices and send
//!   movement deltas and button events through an mpsc channel.
//! * The main thread runs a Wayland event loop.  On startup it creates two
//!   wlr-layer-shell surfaces on the OVERLAY layer (above everything) that are
//!   click-through.  Each surface shows a cursor image (cur_white.png /
//!   cur_black.png embedded at compile time from the `assets/` folder).
//! * Movement: the corresponding surface is repositioned via set_margin.
//! * Clicks: a uinput virtual absolute-position device per cursor is teleported
//!   to the cursor's current position and the button event is re-emitted, so
//!   the compositor delivers the click to the window underneath.
//!
//! Cursor image assets
//! ===================
//!   Place cur_white.png and cur_black.png in the `assets/` folder at the
//!   project root (i.e. next to `src/`).  Both images must be the same size
//!   (e.g. 32×32 px) and should be RGBA or RGB PNG files.
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

const SPEED: f64 = 1.5;

// ─────────────────────────────────────────────────────────────────────────────
// Cursor images — embedded at compile time from assets/
// ─────────────────────────────────────────────────────────────────────────────

const CURSOR_WHITE_PNG: &[u8] = include_bytes!("../assets/cur_white.png");
const CURSOR_BLACK_PNG: &[u8] = include_bytes!("../assets/cur_black.png");

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
// uinput virtual mouse for click injection
// ─────────────────────────────────────────────────────────────────────────────

fn create_virtual_mouse(name: &str, screen_w: i32, screen_h: i32) -> Option<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    // Standard mouse buttons
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
            println!("[uinput] Virtual mouse '{name}' created (click injection enabled)");
            Some(dev)
        }
        Err(e) => {
            eprintln!(
                "[uinput] Could not create virtual mouse '{name}': {e}\n\
                 Click injection disabled for this cursor.\n\
                 To enable it, make /dev/uinput accessible:\n\
                   sudo usermod -aG uinput $USER  (then re-login)\n\
                 or temporarily:\n\
                   sudo chmod a+rw /dev/uinput"
            );
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-cursor state
// ─────────────────────────────────────────────────────────────────────────────

struct Cursor {
    surface:       wl_surface::WlSurface,
    layer_surface: ZwlrLayerSurfaceV1,
    _pool:         wl_shm_pool::WlShmPool,
    buffer:        wl_buffer::WlBuffer,
    x: i32,
    y: i32,
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
    cursors:       Vec<Cursor>,
    screen_w:      i32,
    screen_h:      i32,
    output_scale:  i32,
    rx:            std::sync::mpsc::Receiver<Msg>,
    cursor_size:   u32,
    cursor_pixels: [Vec<u8>; 2],
    vdevices:      [Option<VirtualDevice>; 2],
}

impl AppState {
    fn try_create_cursors(&mut self, qh: &QueueHandle<Self>) {
        if self.cursors.len() == 2 { return; }
        // Wait until we know the real screen size (from wl_output::Mode) so
        // that start positions and uinput ranges are correct.
        if self.screen_w == 0 || self.screen_h == 0 { return; }
        let (comp, shm, shell, output) = match (
            self.compositor.as_ref(), self.shm.as_ref(),
            self.layer_shell.as_ref(), self.output.as_ref(),
        ) {
            (Some(c), Some(s), Some(sh), Some(o)) => (c, s, sh, o),
            _ => return,
        };

        let size    = self.cursor_size;
        let start_x = [self.screen_w / 4, self.screen_w * 3 / 4];
        let start_y = self.screen_h / 2;

        for i in 0..2 {
            let surface = comp.create_surface(qh, ());
            let layer_surface = shell.get_layer_surface(
                &surface, Some(output), Layer::Overlay,
                format!("dual-cursor-{i}"), qh, (),
            );
            layer_surface.set_anchor(Anchor::Top | Anchor::Left);
            layer_surface.set_size(size, size);
            layer_surface.set_keyboard_interactivity(
                zwlr_layer_surface_v1::KeyboardInteractivity::None,
            );
            // -1 = don't reserve any space; just float above everything
            layer_surface.set_exclusive_zone(-1);

            let (pool, buf) = make_shm_buffer(shm, qh, size, size, &self.cursor_pixels[i]);

            // Required by the layer-shell protocol: commit with no buffer attached
            // to signal that setup is complete and ask the compositor to send Configure.
            // The actual buffer is attached in commit_cursor() once Configure arrives.
            surface.commit();

            self.cursors.push(Cursor {
                surface,
                layer_surface,
                _pool: pool,
                buffer: buf,
                x: start_x[i],
                y: start_y,
                configured: false,
            });
        }
    }

    fn move_cursor(&mut self, idx: usize, dx: f64, dy: f64) {
        if idx >= self.cursors.len() { return; }
        let c = &mut self.cursors[idx];
        if !c.configured { return; }
        c.x = (c.x + dx as i32).clamp(0, self.screen_w - 1);
        c.y = (c.y + dy as i32).clamp(0, self.screen_h - 1);
        c.layer_surface.set_margin(c.y, 0, 0, c.x);
        c.surface.commit();
    }

    fn commit_cursor(&mut self, idx: usize) {
        if idx >= self.cursors.len() { return; }
        let c    = &mut self.cursors[idx];
        let size = self.cursor_size as i32;
        c.layer_surface.set_margin(c.y, 0, 0, c.x);
        c.surface.attach(Some(&c.buffer), 0, 0);
        c.surface.damage(0, 0, size, size);
        c.surface.commit();
    }

    /// Inject a button press/release at the visual cursor's current position
    /// via the uinput virtual device for that cursor index.
    fn handle_button(&mut self, idx: usize, code: u16, value: i32) {
        if idx >= self.cursors.len() { return; }
        let (cx, cy) = {
            let c = &self.cursors[idx];
            (c.x, c.y)
        };
        if let Some(vdev) = &mut self.vdevices[idx] {
            // Teleport the virtual device to the cursor's position, then fire
            // the button event.  The SYN_REPORT after ABS* flushes the
            // position; the second SYN_REPORT flushes the button.
            let events = [
                InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_X.0, cx),
                InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_Y.0, cy),
                InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
                InputEvent::new(EventType::KEY.0, code, value),
                InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
            ];
            if let Err(e) = vdev.emit(&events) {
                eprintln!("[uinput-{idx}] emit error: {e}");
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
        // Get the wl_pointer so we can call set_cursor on pointer-enter.
        if self.pointer.is_none() {
            self.pointer = Some(seat.get_pointer(qh, ()));
        }
        // Fullscreen transparent layer surface on the Bottom layer.
        // It covers the whole screen and receives pointer focus, letting us
        // call set_cursor(NULL).  It has no visual content (no buffer attached,
        // which is valid for a surface that only needs input focus).
        let surface = comp.create_surface(qh, ());
        let layer_surface = shell.get_layer_surface(
            &surface, Some(output), Layer::Bottom,
            "cursor-hider".to_string(), qh, (),
        );
        layer_surface.set_anchor(Anchor::Top | Anchor::Left | Anchor::Right | Anchor::Bottom);
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
        surface.commit();
        // Keep the surface alive; store it so it isn't dropped.
        // We don't need the layer_surface after commit so let it drop (the
        // compositor keeps the role alive as long as the wl_surface lives).
        self.pointer_surface = Some(surface);
        // layer_surface intentionally dropped here — the role persists.
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
                        // If the xdg_output_manager arrived first, get the xdg_output now.
                        if let Some(mgr) = &state.xdg_output_manager {
                            state.xdg_output = Some(mgr.get_xdg_output::<_, AppState>(&output, qh, ()));
                        }
                        state.output = Some(output);
                    }
                }
                "zxdg_output_manager_v1" => {
                    let mgr = registry.bind::<zxdg_output_manager_v1::ZxdgOutputManagerV1, _, _>(
                        name, version.min(3), qh, ());
                    // If wl_output arrived first, get the xdg_output now.
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
                // Identify which cursor owns this layer surface by object identity.
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
            // Scale is an integer; 1 for fractional scales like 1.5×.
            // Used only as a fallback divisor if xdg_output isn't available.
            wl_output::Event::Scale { factor } => {
                state.output_scale = factor;
            }
            // Fallback: if zxdg_output_v1 isn't available, derive logical size
            // from physical size ÷ integer scale.  For fractional scales this
            // will be wrong (scale stays 1), but it's the best we can do without
            // xdg_output.  The zxdg_output_v1::LogicalSize handler overwrites
            // screen_w/h with the correct value when xdg_output is present.
            wl_output::Event::Mode { flags, width, height, .. }
            if flags.into_result().map_or(false, |f| f.contains(wl_output::Mode::Current)) =>
                {
                    if state.screen_w == 0 {
                        let logical_w = width  / state.output_scale.max(1);
                        let logical_h = height / state.output_scale.max(1);
                        state.screen_w = logical_w;
                        state.screen_h = logical_h;
                        println!("[output] Mode fallback (no xdg_output): {logical_w}×{logical_h}px");
                        for i in 0..2 {
                            let name = format!("dual-cursor-{i}");
                            state.vdevices[i] = create_virtual_mouse(&name, logical_w, logical_h);
                        }
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
        // logical_size gives the compositor's coordinate space dimensions —
        // exactly what layer-shell margins and uinput ABS values must use.
        if let zxdg_output_v1::Event::LogicalSize { width, height } = event {
            state.screen_w = width;
            state.screen_h = height;
            println!("[output] Logical size (xdg_output): {width}×{height}px");
            for i in 0..2 {
                let name = format!("dual-cursor-{i}");
                state.vdevices[i] = create_virtual_mouse(&name, width, height);
            }
            state.try_create_cursors(qh);
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for AppState {
    fn event(
        state: &mut Self, seat: &wl_seat::WlSeat,
        event: wl_seat::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>,
    ) {
        // When the seat advertises pointer capability, grab the pointer if we
        // haven't already — needed to call set_cursor on enter events.
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
        // Every time the pointer enters any surface, hide the cursor.
        // This covers our fullscreen bottom-layer surface and any other
        // surface the compositor routes the pointer through.
        if let wl_pointer::Event::Enter { serial, .. } = event {
            pointer.set_cursor(serial, None, 0, 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    // 0. Load cursor images (compile-time embedded from assets/)
    let (w0, h0, px0) = load_png_as_argb8888(CURSOR_BLACK_PNG);
    let (w1, h1, px1) = load_png_as_argb8888(CURSOR_WHITE_PNG);
    assert_eq!(
        (w0, h0), (w1, h1),
        "cur_white.png and cur_black.png must be exactly the same dimensions"
    );
    let cursor_size = w0;
    println!("Cursor image size: {cursor_size}×{h0}px");

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
    println!("Cursor 0 (white) → {}", devices[0].display());
    println!("Cursor 1 (black) → {}", devices[1].display());

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
                loop {
                    match dev.fetch_events() {
                        Err(e) => { eprintln!("[mouse-{idx}] read error: {e}"); break; }
                        Ok(evs) => for ev in evs {
                            match ev.destructure() {
                                // ── Movement ──────────────────────────────
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_X, v) =>
                                    pdx += v as f64 * SPEED,
                                EventSummary::RelativeAxis(_, RelativeAxisCode::REL_Y, v) =>
                                    pdy += v as f64 * SPEED,
                                EventSummary::Synchronization(..) => {
                                    if pdx != 0.0 || pdy != 0.0 {
                                        let _ = tx.send(Msg::Move { idx, dx: pdx, dy: pdy });
                                        pdx = 0.0; pdy = 0.0;
                                    }
                                }
                                // ── Buttons ───────────────────────────────
                                // Mouse buttons start at BTN_LEFT = 0x110.
                                // value: 0 = release, 1 = press, 2 = repeat.
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

    // 4. Create uinput virtual mice — deferred; actual screen dimensions are
    //    read from wl_output::Mode during the first roundtrip so the ABS ranges
    //    match the real screen size exactly (required for correct click position).

    // 5. Connect to Wayland (handles missing env vars under sudo)
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
        cursor_pixels: [px0, px1],
        vdevices: [None, None],
    };

    // First roundtrip: bind globals, receive wl_output::Mode (sets screen_w/h), create layer surfaces
    event_queue.roundtrip(&mut state).expect("initial roundtrip");
    // Second roundtrip: receive Configure events from compositor
    event_queue.roundtrip(&mut state).expect("configure roundtrip");

    // Fallback: if compositor didn't send wl_output::Mode, use a safe default.
    if state.screen_w == 0 || state.screen_h == 0 {
        eprintln!("WARNING: wl_output did not report screen size; defaulting to 3840×2160. Click positions may be inaccurate.");
        state.screen_w = 3840;
        state.screen_h = 2160;
    }
    // Ensure vdevices exist (they're created in the Mode handler; this is a safety net).
    for i in 0..2 {
        if state.vdevices[i].is_none() {
            state.vdevices[i] = create_virtual_mouse(&format!("dual-cursor-{i}"), state.screen_w, state.screen_h);
        }
    }

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

    if state.cursors.iter().filter(|c| c.configured).count() < 2 {
        eprintln!("WARNING: not all cursors received a Configure event yet — they may appear after first mouse movement.");
    }

    println!("Running. Move each mouse to control its cursor. Buttons are forwarded. Ctrl-C to quit.");

    // 6. Main event loop
    while state.running {
        event_queue.flush().ok();

        // Drain all pending mouse events (non-blocking)
        loop {
            match state.rx.try_recv() {
                Ok(msg) => match msg {
                    Msg::Move   { idx, dx, dy }    => state.move_cursor(idx, dx, dy),
                    Msg::Button { idx, code, value } => state.handle_button(idx, code, value),
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

        // Poll the Wayland socket for ~4 ms then dispatch any new events
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
