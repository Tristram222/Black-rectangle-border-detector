import time

class PositionalPID:
    """
    一个实现了积分限幅和输出限幅的位置式PID控制器类。
    """

    def __init__(self, Kp, Ki, Kd, setpoint=0):
        """
        初始化PID控制器。

        :param Kp: 比例增益 (Proportional gain)
        :param Ki: 积分增益 (Integral gain)
        :param Kd: 微分增益 (Derivative gain)
        :param setpoint: 目标设定值
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        # 初始化PID参数
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

        # 默认限幅值 (可根据需要进行配置)
        self.integral_min = -float('inf')
        self.integral_max = float('inf')
        self.output_min = -float('inf')
        self.output_max = float('inf')

    def set_integral_limits(self, min_val, max_val):
        """
        设置积分项的上下限，以防止积分饱和。

        :param min_val: 积分项的最小值
        :param max_val: 积分项的最大值
        """
        if min_val > max_val:
            raise ValueError("min_val must not be greater than max_val")
        self.integral_min = min_val
        self.integral_max = max_val

    def set_output_limits(self, min_val, max_val):
        """
        设置控制器输出值的上下限。

        :param min_val: 输出值的最小值
        :param max_val: 输出值的最大值
        """
        if min_val > max_val:
            raise ValueError("min_val must not be greater than max_val")
        self.output_min = min_val
        self.output_max = max_val

    def compute(self, current_value):
        """
        计算PID输出值。

        :param current_value: 当前的测量值 (Process Variable)
        :return: PID控制器的输出值
        """
        # --- 时间处理 ---
        current_time = time.time()
        dt = current_time - self.last_time
        # 如果时间间隔为0，则不进行计算，防止除以零
        if dt <= 0:
            return self._clamp_output(self.Kp * self.last_error + self.Ki * self.integral + self.Kd * 0)

        # --- 计算误差 ---
        error = self.setpoint - current_value

        # --- 比例项 (P) ---
        p_term = self.Kp * error

        # --- 积分项 (I) ---
        self.integral += error * dt
        # 应用积分限幅
        self.integral = self._clamp(self.integral, self.integral_min, self.integral_max)
        i_term = self.Ki * self.integral

        # --- 微分项 (D) ---
        # 微分项计算的是误差的变化率
        derivative = (error - self.last_error) / dt
        d_term = self.Kd * derivative

        # --- 计算总输出 ---
        output = p_term + i_term + d_term

        # --- 应用输出限幅 ---
        output = self._clamp(output, self.output_min, self.output_max)

        # --- 更新状态以备下次计算 ---
        self.last_error = error
        self.last_time = current_time

        return output

    def reset(self):
        """
        重置PID控制器的内部状态。
        """
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def set_setpoint(self, setpoint):
        """
        更新目标设定值。
        """
        self.setpoint = setpoint
        self.reset() # 目标值改变时通常建议重置控制器

    def _clamp(self, value, min_val, max_val):
        """
        一个辅助函数，用于将值限制在指定的范围内。
        """
        return max(min_val, min(value, max_val))


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 初始化PID控制器
    # 假设我们要控制一个加热器的温度
    # Kp, Ki, Kd 参数需要根据实际系统进行整定 (Tuning)
    pid = PositionalPID(Kp=1.2, Ki=0.5, Kd=0.05)

    # 2. 设置目标值
    pid.set_setpoint(100.0) # 目标温度为 100°C

    # 3. 设置限幅
    # 假设积分项累加范围为 -200 到 200
    pid.set_integral_limits(-200, 200)
    # 假设加热器功率输出范围为 0% 到 100%
    pid.set_output_limits(0, 100)

    # 4. 模拟运行
    current_temperature = 25.0  # 初始温度
    print("目标温度: {}°C, 初始温度: {}°C".format(pid.setpoint, current_temperature))
    print("-" * 30)

    # 模拟一个简单的系统响应
    # 注意：这是一个非常简化的模型，实际系统会更复杂
    def simulate_system(current_temp, control_output, dt):
        # 模拟加热过程，输出越大，温度上升越快
        heating_effect = control_output * 0.1
        # 模拟自然冷却
        cooling_effect = (current_temp - 20) * 0.02
        # 温度变化
        delta_temp = (heating_effect - cooling_effect) * dt
        return current_temp + delta_temp

    try:
        for i in range(20):
            # (1) 计算控制输出
            control_output = pid.compute(current_temperature)

            # (2) 应用控制输出到系统
            # 在这个模拟中，我们简单地更新温度
            current_temperature = simulate_system(current_temperature, control_output, 1.0)

            # (3) 打印状态
            print(f"时间: {i+1}s | "
                  f"当前温度: {current_temperature:.2f}°C | "
                  f"PID输出: {control_output:.2f}% | "
                  f"误差: {pid.last_error:.2f} | "
                  f"积分项: {pid.integral:.2f}")

            # 模拟1秒的间隔
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("模拟停止。")