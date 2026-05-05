#[derive(Clone, Copy)]
struct LinearKalmanFilter {
    x: f64,     // 状态估计
    p: f64,     // 估计误差协方差
    k: f64,     // 卡尔曼增益
}

impl LinearKalmanFilter {
    fn new(initial_x: f64, initial_p: f64) -> Self {
        LinearKalmanFilter {
            x: initial_x,
            p: initial_p,
            k: 0.0,
        }
    }

    fn update(mut self, measurement: f64, process_noise: f64, measurement_noise: f64) -> Self {
        // 预测
        let x_pred = self.x; // 线性模型假设状态不变
        let p_pred = self.p + process_noise;

        // 更新
        self.k = p_pred / (p_pred + measurement_noise);

        // 融合测量
        self.x = x_pred + self.k * (measurement - x_pred);

        // 更新误差协方差
        self.p = (1.0 - self.k) * p_pred;

        self
    }
}

fn main() {
    let linear_data = vec![
        99.842, 100.123, 99.987, 
        100.456, 99.654, 99.995, 
        100.321, 99.876, 100.234, 
        99.789, 98.996, 99.256, 
        101.024, 99.654, 100.789,
    ];

    let mut kf = LinearKalmanFilter::new(100.0, 1.0);

    // 定义过程噪声（模型不确定性）和测量噪声（传感器误差） 
    let (process_noise, measurement_noise) = (0.01, 0.2);

    let filtered_data: Vec<f64> = linear_data.iter().map(|&measurement| {
        kf = kf.update(measurement, process_noise, measurement_noise);
        println!("测量: {:.3}, 估计: {:.3}, 卡尔曼增益: {:.3}", measurement, kf.x, kf.k);
        kf.x
    }).collect();

    println!("原始数据: {:?}", linear_data);
    print!("[");
    for (index, i) in linear_data.iter().enumerate() {
        print!("({}, {}), ", index + 1, i);
    }
    println!("]");
    println!("滤波后的数据: {:?}", filtered_data);
    print!("[");
    for (index, i) in filtered_data.iter().enumerate() {
        print!("({}, {}), ", index + 1, i);
    }
    println!("]");
}
