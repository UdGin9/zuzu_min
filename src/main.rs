mod aruco;
mod camera;
mod controller;
mod utils;
use crate::aruco::ArucoDetect;
use crate::camera::Camera;
use crate::controller::MissionController;
use log::{info};

struct MainConfig {
    aruco_detector: ArucoDetect,
    camera: Camera,
    mission_cntroller: MissionController,
}

impl MainConfig {
    pub fn new() -> Self {
        Self {
            aruco_detector: ArucoDetect::new(),
            camera: Camera::new(),
            mission_cntroller: MissionController::new(),
        }
    }
    
}

use std::sync::{Arc, RwLock};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("trace")
    ).init();

    info!("Zuzu started");

    let is_running = Arc::new(RwLock::new(true));
    let mut config = MainConfig::new();

    config.aruco_detector.run_in_background(Arc::clone(&is_running));

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}