use std::sync::{Arc, RwLock};
use std::sync::mpsc;
use std::thread;
use log::{info, error};
use gstreamer as gst;
use gstreamer_app::AppSink;
use gst::prelude::*;
use opencv::core::Mat;
use opencv::prelude::MatTraitConst;

pub struct Camera {
    tx: mpsc::Sender<Mat>,
}

impl Camera {
    pub fn new(tx: mpsc::Sender<Mat>) -> Self {
        Self { tx }
    }

    pub fn run_in_background(&mut self, is_running: Arc<RwLock<bool>>) {
        let tx = self.tx.clone();

        thread::spawn(move || {
            gst::init().unwrap();

            info!("GStreamer initialized");

            let pipeline = gst::parse::launch("
                libcamerasrc !
                video/x-raw,width=1640,height=1232,format=I420,framerate=24/1 !
                videoscale !
                video/x-raw,width=640,height=480 !
                videoconvert !
                video/x-raw,format=BGR !
                appsink name=sink emit-signals=true max-buffers=1 drop=true
            ").unwrap().downcast::<gst::Pipeline>().unwrap();

            info!("Pipeline created");

            let appsink = pipeline
                .by_name("sink")
                .unwrap()
                .downcast::<AppSink>()
                .unwrap();

            appsink.set_callbacks(
                gstreamer_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        let buffer = sample.buffer().unwrap();
                        let map = buffer.map_readable().unwrap();

                        let mat_ref = Mat::from_slice(map.as_slice()).unwrap();
                        let mat = mat_ref.clone_pointee();

                        let mat_2d = mat.reshape(3, 480).unwrap().try_clone().unwrap();


                        if tx.send(mat_2d).is_err() {
                            error!("Failed to send frame to Aruco");
                            return Err(gst::FlowError::Eos);
                        }
                        Ok(gst::FlowSuccess::Ok)
                    })
                    .build()
            );

            pipeline.set_state(gst::State::Playing).unwrap();
            info!("Started camera");

            while *is_running.read().unwrap() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }

            pipeline.set_state(gst::State::Null).unwrap();
            info!("Stopped camera");
        });
    }
}