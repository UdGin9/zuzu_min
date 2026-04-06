mod aruco;
mod camera;
mod utils;
use crate::aruco::ArucoDetect;
use crate::camera::Camera;
use std::sync::{Arc, RwLock};
use std::sync::mpsc;
use opencv::core::Mat;
use log::{info};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("trace")
    ).init();

    info!("Zuzu started");

    let is_running = Arc::new(RwLock::new(true));
    let (tx, rx) = mpsc::channel::<Mat>();

    let mut aruco = ArucoDetect::new();
    let mut camera = Camera::new(tx);

    camera.run_in_background(Arc::clone(&is_running));
    aruco.run_in_background(Arc::clone(&is_running), rx);

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}