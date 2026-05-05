use math_matrix::Matrix;
use std::fs::File;
use std::io::Write;

#[derive(Clone)]
struct EKF {
    x: Matrix,  // 状态向量
    p: Matrix,  // 误差协方差矩阵
    q: Matrix,  // 过程噪声协方差矩阵
    r: f64,     // 测量噪声协方差
}

impl EKF {
    fn new(init_theta: f64, init_omega: f64) -> Self {
        EKF {
            x: Matrix::new(vec![init_theta, init_omega], (2, 1)).unwrap(),
            p: Matrix::identity_matrix(2),
            q: Matrix::new(vec![0.01, 0.0, 0.0, 0.1], (2, 2)).unwrap(),
            r: 0.01,
        }
    }

    fn kalman_update_scaler(&mut self, y_meas: f64) {
        // 预测测量值
        let z_pred = h(self.x.get(1, 1).unwrap());
        // 测量残差
        let y = y_meas - z_pred;

        // 计算雅可比矩阵 H
        #[allow(non_snake_case)]
        let mut H = Matrix::new(vec![0.0, 0.0], (1, 2)).unwrap();
        build_h(self.x.get(1, 1).unwrap(), &mut H);

        // 新息协方差 S = H * P * H^T + R
        #[allow(non_snake_case)]
        let S = H.clone() * self.p.clone() * H.transpose() + Matrix::new(vec![self.r], (1, 1)).unwrap();

        // 卡尔曼增益 K = P * H^T * S^-1
        #[allow(non_snake_case)]
        let K = self.p.clone() * H.transpose() * S.inverse().unwrap();

        // 更新状态估计 x = x + K * y
        let ky = K.clone() * Matrix::new(vec![y], (1, 1)).unwrap();
        self.x = self.x.clone() + ky;

        // 更新误差协方差 P = (I - K * H) * P
        self.p = (Matrix::identity_matrix(2) - K * H) * self.p.clone();
    }

    // EKF 的预测步骤
    fn predict(&mut self, dt: f64) {
        // 构建状态转移矩阵 F
        #[allow(non_snake_case)]
        let mut F = Matrix::new(vec![0.0; 4], (2, 2)).unwrap();
        build_f(dt, &mut F);

        // 预测状态 x = F * x
        let x_pred = F.clone() * self.x.clone();
        self.x = x_pred;

        // 预测误差协方差 P = F * P * F^T + Q
        self.p = F.clone() * self.p.clone() * F.transpose() + self.q.clone();
    }
}

// 构建状态转移矩阵 F
#[allow(non_snake_case)]
fn build_f(dt: f64, F: &mut Matrix) {
    let _ = F.set(1, 1, 1.0);
    let _ = F.set(1, 2, dt);
    let _ = F.set(2, 1, 0.0);
    let _ = F.set(2, 2, 1.0);
}

// 观测函数 h(x) -> sin(theta)
fn h(theta: f64) -> f64 {
    theta.sin()
}

// 构建雅可比矩阵 H
#[allow(non_snake_case)]
fn build_h(theta: f64, H: &mut Matrix) {
    let _ = H.set(1, 1, theta.cos());   // dh/dtheta = cos(theta)
    let _ = H.set(1, 2, 0.0);           // dh/domega = 0
}

fn main() {
    // 初始化 EKF，跟踪步长为 0.1 秒，跟踪函数 h(x) = sin(x)
    let mut ekf = EKF::new(0.0, 1.0);

    let dt = 0.1; // 时间步长
    let mut t = 0.0; // 初始时间
    let mut true_theta: f64 = 0.0;
    let true_omega = 1.0;

    // 生成 10000 个模拟测量数据
    let mut measurements = Vec::new();
    let mut simu_theta = true_theta;
    for _ in 0..10000 {
        // 模拟系统的真实状态
        simu_theta += true_omega * dt;

        // 模拟测量值，添加高斯噪声
        let noise = rand::random::<f64>() * 0.1 - 0.05; // [-0.05, 0.05]
        let measurement = h(simu_theta) + noise;
        measurements.push(measurement);
    }

    // 使用 EKF 进行状态估计（按生成顺序处理测量值）
    let mut filtered_measurements = Vec::new();
    for measurement in measurements.clone() {
        ekf.predict(dt);

        // 更新真实状态
        true_theta += true_omega * dt;

        ekf.kalman_update_scaler(measurement);

        // 打印当前时间、真实状态、测量值和 EKF 估计值
        println!(
            "时间: {:.2} s, 真实角度: {:.3} rad, 测量值: {:.3}, EKF 估计值: {:.3}",
            t,
            true_theta,
            measurement,
            h(ekf.x.get(1, 1).unwrap())
        );
        filtered_measurements.push(h(ekf.x.get(1, 1).unwrap()));
        t += dt;
    }

    // 写入原始测量值和滤波后的估计值到文件
    let mut file = File::create("ekf_output.txt").expect("无法创建文件");
    writeln!(file, "原始测量值:").unwrap();
    write!(file, "[").unwrap();
    for (index, measurement) in measurements.iter().enumerate() {
        write!(file, "({:.1}, {:.3}), ", (index + 1) as f64 * 0.1, measurement).unwrap();
    }
    writeln!(file, "]").unwrap();
    writeln!(file, "滤波后的估计值:").unwrap();
    write!(file, "[").unwrap();
    for (index, estimate) in filtered_measurements.iter().enumerate() {
        write!(file, "({:.1}, {:.3}), ", (index + 1) as f64 * 0.1, estimate).unwrap();
    }
    writeln!(file, "]").unwrap();
}