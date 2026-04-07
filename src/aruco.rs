use crate::utils::{rotation_matrix_euler, euler_from_matrix, mat_to_mat4};
use std::sync::{Arc, RwLock, mpsc};
use std::time::{Instant};
use log::info;
use std::thread;
use opencv::objdetect::{
    ArucoDetector,
    DetectorParameters,
    PredefinedDictionaryType,
    RefineParameters,
    get_predefined_dictionary,
    Board,
};
use opencv::core::{Mat, Point3f, Vector, transpose, gemm};
use opencv::prelude::{ArucoDetectorTraitConst, BoardTraitConst, MatTraitConst, VectorToVec};
use opencv::calib3d::{solve_pnp, SOLVEPNP_ITERATIVE, rodrigues};

const SEND_PERIOD: f64 = 0.066;

fn cam_rot_mtx() -> [[f64; 3]; 3] {
    let m = rotation_matrix_euler(0.0, 0.0, -90.0);
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

struct MarkerData {
    id:   i32,
    pos:  [f32; 3],
    size: f32,
}

impl MarkerData {
    fn get_points(&self) -> Vector<Point3f> {
        let h = self.size / 2.0;
        let [x, y, z] = self.pos;
        let mut pts = Vector::new();
        pts.push(Point3f::new(x - h, y - h, z));
        pts.push(Point3f::new(x + h, y - h, z));
        pts.push(Point3f::new(x + h, y + h, z));
        pts.push(Point3f::new(x - h, y + h, z));
        pts
    }
}

const ARUCO_MARKERS: &[MarkerData] = &[
    MarkerData { id: 0, pos: [0.0, 0.0, 0.0], size: 0.19 },
    MarkerData { id: 1, pos: [0.0, 1.0, 0.0], size: 0.19 },
    MarkerData { id: 2, pos: [0.0, 2.0, 0.0], size: 0.19 },
    MarkerData { id: 3, pos: [1.0, 0.0, 0.0], size: 0.19 },
    MarkerData { id: 4, pos: [1.0, 1.0, 0.0], size: 0.19 },
    MarkerData { id: 5, pos: [1.0, 2.0, 0.0], size: 0.19 },
    MarkerData { id: 6, pos: [2.0, 0.0, 0.0], size: 0.19 },
    MarkerData { id: 7, pos: [2.0, 1.0, 0.0], size: 0.19 },
    MarkerData { id: 8, pos: [2.0, 2.0, 0.0], size: 0.19 },
];

fn build_board(dict: &opencv::objdetect::Dictionary) -> Board {
    let mut all_points = Vector::<Vector<Point3f>>::new();
    let mut ids_vec: Vec<i32> = Vec::new();

    for marker in ARUCO_MARKERS {
        all_points.push(marker.get_points());
        ids_vec.push(marker.id);
    }

    let ids = Mat::from_slice(&ids_vec).unwrap();
    Board::new(&all_points, dict, &ids).unwrap()
}

struct ArucoState {
    last_send: Option<f64>,
    last_time: Option<Instant>,
    last_pos:  Option<[f64; 3]>,
    last_rot:  Option<[f64; 3]>,
    corners:   Vector<Mat>,
    ids:       Mat,
    rejected:  Vector<Mat>,
}

impl ArucoState {
    fn new() -> Self {
        Self {
            last_pos:  None,
            last_rot:  None,
            last_send: None,
            last_time: None,
            corners:   Vector::new(),
            ids:       Mat::default(),
            rejected:  Vector::new(),
        }
    }

    fn reset_pose(&mut self) {
        self.last_pos = None;
        self.last_rot = None;
    }
}

pub struct ArucoDetect {
    state: Arc<RwLock<ArucoState>>,
}

impl ArucoDetect {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ArucoState::new())),
        }
    }

    pub fn run_in_background(&mut self, is_running: Arc<RwLock<bool>>, rx: mpsc::Receiver<Mat>) {
        let state = Arc::clone(&self.state);

        thread::spawn(move || {
            let dict = get_predefined_dictionary(PredefinedDictionaryType::DICT_4X4_100).unwrap();
            let detector_params = DetectorParameters::default().unwrap();
            let refine_params = RefineParameters::new_def().unwrap();
            let detector = ArucoDetector::new(&dict, &detector_params, refine_params).unwrap();
            let board = build_board(&dict);
            let socket = std::net::UdpSocket::bind("0.0.0.0:0").unwrap();


            let cam_matrix = Mat::from_slice_2d(&[
                [375.9934_f64, 0.0, 339.748],
                [0.0, 373.8946, 212.6516],
                [0.0, 0.0, 1.0],
            ]).unwrap().try_clone().unwrap();
            
            let cam_dist_coeffs = Mat::from_slice_2d(&[
                [-0.3567_f64],
                [0.1101],
                [0.0093],
                [-0.0017],
                [0.0104],
            ]).unwrap().try_clone().unwrap();

            let cam_rot_mtx = cam_rot_mtx();

            info!("Started Aruco detector");

            while *is_running.read().unwrap() {
                let frame = rx.recv().ok();
                next_detect(&state, &detector, &board, &cam_matrix, &cam_dist_coeffs, &cam_rot_mtx, frame, &socket);

                //TODO: добавить отправку через mavlink
            }

            info!("Stopped Aruco detector");
        });
    }

    pub fn position(&self) -> Option<[f64; 3]> {
        self.state.read().unwrap().last_pos
    }

    pub fn rotation(&self) -> Option<[f64; 3]> {
        self.state.read().unwrap().last_rot
    }
}

fn next_detect(
    state:           &Arc<RwLock<ArucoState>>,
    detector:        &ArucoDetector,
    board:           &Board,
    cam_matrix:      &Mat,
    cam_dist_coeffs: &Mat,
    cam_rot_mtx:     &[[f64; 3]; 3],
    frame:           Option<Mat>,
    socket:          &std::net::UdpSocket,
) {

    let Some(frame) = frame else {
        let mut st = state.write().unwrap();
        info!("No frame");
        st.reset_pose();
        return;
    };

    let mut corners  = Vector::<Mat>::new();
    let mut ids      = Mat::default();
    let mut rejected = Vector::<Mat>::new();

    detector.detect_markers(&frame, &mut corners, &mut ids, &mut rejected).unwrap();

    let mut recovered = Mat::default();

    detector.refine_detected_markers(
        &frame, board, &mut corners, &mut ids, &mut rejected,
        cam_matrix, cam_dist_coeffs, &mut recovered,
    ).unwrap();
    
    let mut frame_out = frame.clone();

    if ids.empty() {
        let mut st = state.write().unwrap();
        st.reset_pose();
        return;
    }
    // отрисовка маркеров
    else {
        opencv::objdetect::draw_detected_markers(
            &mut frame_out,
            &corners,
            &ids,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        ).unwrap();
    }

    // кодировка в jpeg
    let mut buf = Vector::<u8>::new();
    let params = Vector::from_slice(&[opencv::imgcodecs::IMWRITE_JPEG_QUALITY, 50]);
    opencv::imgcodecs::imencode(".jpg", &frame_out, &mut buf, &params).unwrap();

    let data = buf.to_vec();
    if data.len() < 65000 {
        socket.send_to(&data, "192.168.31.5:5432").ok();
    } else {
        log::warn!("Frame too large for UDP: {} bytes", data.len());
    }

    let mut obj_points = Mat::default();
    let mut img_points = Mat::default();
    board.match_image_points(&corners, &ids, &mut obj_points, &mut img_points).unwrap();

    if obj_points.empty() || img_points.empty() {
        let mut st = state.write().unwrap();
        st.reset_pose();
        return;
    }

    let mut r_vec = Mat::default();
    let mut t_vec = Mat::default();
    let result = solve_pnp(
        &obj_points, &img_points,
        cam_matrix, cam_dist_coeffs,
        &mut r_vec, &mut t_vec,
        false, SOLVEPNP_ITERATIVE,
    ).unwrap();

    if !result {
        let mut st = state.write().unwrap();
        st.reset_pose();
        return;
    }

    let mut rot = Mat::default();
    rodrigues(&r_vec, &mut rot, &mut Mat::default()).unwrap();

    let mut rot_t = Mat::default();
    transpose(&rot, &mut rot_t).unwrap();

    let mut pos = Mat::default();
    gemm(&rot_t, &t_vec, -1.0, &Mat::default(), 0.0, &mut pos, 0).unwrap();

    let cam_rot_mat = Mat::from_slice_2d(cam_rot_mtx).unwrap();
    let mut final_rot = Mat::default();
    gemm(&rot_t, &cam_rot_mat, 1.0, &Mat::default(), 0.0, &mut final_rot, 0).unwrap();

    let now = Instant::now();
    let mut st = state.write().unwrap();
    st.corners  = corners;
    st.ids      = ids;
    st.rejected = rejected;
    st.last_pos = Some([
        *pos.at_2d::<f64>(0, 0).unwrap(),
        *pos.at_2d::<f64>(1, 0).unwrap(),
        *pos.at_2d::<f64>(2, 0).unwrap(),
    ]);
    st.last_rot  = Some(euler_from_matrix(mat_to_mat4(&final_rot)));
    info!("Позиция: {:?}", st.last_pos);
    info!("Rot: {:?}", st.last_rot);
    st.last_time = Some(now);

}