
pub struct Camera {
    is_running: bool,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            is_running: false,
        }
    }
}