/// Taken from https://github.com/Gekkio/imgui-rs/blob/0440752/imgui-glutin-support/src/lib.rs
/// We do not use the original crate to avoid headaches between incompatible glutin
/// versions during dependency resolution.
use glium::glutin::{
    ElementState, Event, KeyboardInput, ModifiersState, MouseButton, MouseCursor, MouseScrollDelta,
    TouchPhase, VirtualKeyCode, Window, WindowEvent,
};
use imgui::{FrameSize, ImGui, ImGuiKey, ImGuiMouseCursor};

/// Configure imgui key map with glutin `VirtualKeyCode` values
pub fn configure_keys(imgui: &mut ImGui) {
    imgui.set_imgui_key(ImGuiKey::Tab, VirtualKeyCode::Tab as _);
    imgui.set_imgui_key(ImGuiKey::LeftArrow, VirtualKeyCode::Left as _);
    imgui.set_imgui_key(ImGuiKey::RightArrow, VirtualKeyCode::Right as _);
    imgui.set_imgui_key(ImGuiKey::UpArrow, VirtualKeyCode::Up as _);
    imgui.set_imgui_key(ImGuiKey::DownArrow, VirtualKeyCode::Down as _);
    imgui.set_imgui_key(ImGuiKey::PageUp, VirtualKeyCode::PageUp as _);
    imgui.set_imgui_key(ImGuiKey::PageDown, VirtualKeyCode::PageDown as _);
    imgui.set_imgui_key(ImGuiKey::Home, VirtualKeyCode::Home as _);
    imgui.set_imgui_key(ImGuiKey::End, VirtualKeyCode::End as _);
    imgui.set_imgui_key(ImGuiKey::Delete, VirtualKeyCode::Delete as _);
    imgui.set_imgui_key(ImGuiKey::Backspace, VirtualKeyCode::Back as _);
    imgui.set_imgui_key(ImGuiKey::Enter, VirtualKeyCode::Return as _);
    imgui.set_imgui_key(ImGuiKey::Escape, VirtualKeyCode::Escape as _);
    imgui.set_imgui_key(ImGuiKey::A, VirtualKeyCode::A as _);
    imgui.set_imgui_key(ImGuiKey::C, VirtualKeyCode::C as _);
    imgui.set_imgui_key(ImGuiKey::V, VirtualKeyCode::V as _);
    imgui.set_imgui_key(ImGuiKey::X, VirtualKeyCode::X as _);
    imgui.set_imgui_key(ImGuiKey::Y, VirtualKeyCode::Y as _);
    imgui.set_imgui_key(ImGuiKey::Z, VirtualKeyCode::Z as _);
}

/// Update imgui keyboard state
pub fn handle_keyboard_input(imgui: &mut ImGui, event: KeyboardInput) {
    handle_modifiers(imgui, event.modifiers);
    if let Some(key) = event.virtual_keycode {
        let state_bool = event.state == ElementState::Pressed;
        imgui.set_key(key as _, state_bool);
        match key {
            VirtualKeyCode::LShift | VirtualKeyCode::RShift => imgui.set_key_shift(state_bool),
            VirtualKeyCode::LControl | VirtualKeyCode::RControl => imgui.set_key_ctrl(state_bool),
            VirtualKeyCode::LAlt | VirtualKeyCode::RAlt => imgui.set_key_alt(state_bool),
            VirtualKeyCode::LWin | VirtualKeyCode::RWin => imgui.set_key_super(state_bool),
            _ => (),
        }
    }
}

/// Update imgui keyboard modifier state
pub fn handle_modifiers(imgui: &mut ImGui, modifiers: ModifiersState) {
    imgui.set_key_shift(modifiers.shift);
    imgui.set_key_ctrl(modifiers.ctrl);
    imgui.set_key_alt(modifiers.alt);
    imgui.set_key_super(modifiers.logo);
}

/// Update imgui mouse wheel position
pub fn handle_mouse_scroll_delta(
    imgui: &mut ImGui,
    delta: MouseScrollDelta,
    window_hidpi_factor: f64,
    app_hidpi_factor: f64,
) {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => imgui.set_mouse_wheel(y),
        MouseScrollDelta::PixelDelta(pos) => {
            let pos = pos
                .to_physical(window_hidpi_factor)
                .to_logical(app_hidpi_factor);
            imgui.set_mouse_wheel(pos.y as f32)
        }
    }
}

/// Update imgui mouse button state
pub fn handle_mouse_button_state(imgui: &mut ImGui, button: MouseButton, state: ElementState) {
    let mut states = imgui.mouse_down();
    let state_bool = state == ElementState::Pressed;
    match button {
        MouseButton::Left => states[0] = state_bool,
        MouseButton::Right => states[1] = state_bool,
        MouseButton::Middle => states[2] = state_bool,
        MouseButton::Other(idx @ 0...4) => states[idx as usize] = state_bool,
        _ => (),
    }
    imgui.set_mouse_down(states);
}

/// Update imgui state from glutin event
pub fn handle_event(
    imgui: &mut ImGui,
    event: &Event,
    window_hidpi_factor: f64,
    app_hidpi_factor: f64,
) {
    match event {
        &Event::WindowEvent { ref event, .. } => {
            handle_window_event(imgui, event, window_hidpi_factor, app_hidpi_factor)
        }
        _ => (),
    }
}

/// Update imgui state from glutin window event
pub fn handle_window_event(
    imgui: &mut ImGui,
    event: &WindowEvent,
    window_hidpi_factor: f64,
    app_hidpi_factor: f64,
) {
    use self::WindowEvent::*;
    match event {
        &KeyboardInput { input, .. } => handle_keyboard_input(imgui, input),
        &ReceivedCharacter(ch) => imgui.add_input_character(ch),
        &CursorMoved {
            position,
            modifiers,
            ..
        } => {
            let position = position
                .to_physical(window_hidpi_factor)
                .to_logical(app_hidpi_factor);
            imgui.set_mouse_pos(position.x as f32, position.y as f32);
            handle_modifiers(imgui, modifiers);
        }
        &MouseWheel {
            delta,
            modifiers,
            phase: TouchPhase::Moved,
            ..
        } => {
            handle_mouse_scroll_delta(imgui, delta, window_hidpi_factor, app_hidpi_factor);
            handle_modifiers(imgui, modifiers);
        }
        &MouseInput {
            state,
            button,
            modifiers,
            ..
        } => {
            handle_mouse_button_state(imgui, button, state);
            handle_modifiers(imgui, modifiers);
        }
        _ => (),
    }
}

/// Update glutin window mouse cursor state
pub fn update_mouse_cursor(imgui: &ImGui, window: &Window) {
    let mouse_cursor = imgui.mouse_cursor();
    if imgui.mouse_draw_cursor() || mouse_cursor == ImGuiMouseCursor::None {
        // Hide OS cursor
        window.hide_cursor(true);
    } else {
        // Set OS cursor
        window.hide_cursor(false);
        window.set_cursor(match mouse_cursor {
            ImGuiMouseCursor::None => unreachable!("mouse_cursor was None!"),
            ImGuiMouseCursor::Arrow => MouseCursor::Arrow,
            ImGuiMouseCursor::TextInput => MouseCursor::Text,
            ImGuiMouseCursor::ResizeAll => MouseCursor::Move,
            ImGuiMouseCursor::ResizeNS => MouseCursor::NsResize,
            ImGuiMouseCursor::ResizeEW => MouseCursor::EwResize,
            ImGuiMouseCursor::ResizeNESW => MouseCursor::NeswResize,
            ImGuiMouseCursor::ResizeNWSE => MouseCursor::NwseResize,
            ImGuiMouseCursor::Hand => MouseCursor::Hand,
        });
    }
}

/// Get the current frame size for imgui frame rendering.
///
/// Returns `None` if the window no longer exists
pub fn get_frame_size(window: &Window, app_hidpi_factor: f64) -> Option<FrameSize> {
    window.get_inner_size().map(|logical_size| FrameSize {
        logical_size: logical_size
            .to_physical(window.get_hidpi_factor())
            .to_logical(app_hidpi_factor)
            .into(),
        hidpi_factor: app_hidpi_factor,
    })
}
