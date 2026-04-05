
pub struct MissionController {
    is_running: bool,
}

impl MissionController {
    pub fn new() -> Self {
        Self {
            is_running: false,
        }
    }
}