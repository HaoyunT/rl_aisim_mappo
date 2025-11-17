"""
带标注的实时可视化监控脚本
在UE4场景中绘制无人机标签、连线和轨迹
"""
import airsim
import time
import numpy as np
import argparse

def visualize_with_markers(update_rate_hz: float = 4.0, no_flush: bool = False):
    """实时可视化，带标注和连线"""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    drone_names = ["Drone0", "Drone1", "Drone2", "Drone3"]
    colors = {
        "Drone0": [1.0, 0.0, 0.0, 1.0],  # 红色 - 目标
        "Drone1": [0.0, 1.0, 0.0, 1.0],  # 绿色 - 追击者1
        "Drone2": [0.0, 0.0, 1.0, 1.0],  # 蓝色 - 追击者2
        "Drone3": [1.0, 1.0, 0.0, 1.0],  # 黄色 - 追击者3
    }
    
    print("开始可视化监控 (按Ctrl+C停止)...")
    print("=" * 60)
    
    # 存储轨迹点（最近100个点）
    trajectories = {name: [] for name in drone_names}
    max_trajectory_points = 100
    # compute durations from requested update rate (Hz)
    if update_rate_hz <= 0:
        update_rate_hz = 4.0
    # make plot duration larger to avoid rapid flicker in UE
    plot_duration = max(0.2, 1.0 / float(update_rate_hz))  # longer duration reduces flicker
    sleep_interval = plot_duration
    # flush markers less frequently: compute flush interval in loops
    flush_every = max(1, int(update_rate_hz))  # flush roughly once per second (or less)
    iter_count = 0

    try:
        while True:
            # 清除之前的标注 occasionally (to avoid flicker) unless user disabled flushing
            if (not no_flush) and (iter_count % flush_every == 0):
                try:
                    client.simFlushPersistentMarkers()
                except Exception:
                    pass

            positions = {}
            
            # 获取所有无人机位置
            for name in drone_names:
                state = client.getMultirotorState(vehicle_name=name)
                pos = state.kinematics_estimated.position
                positions[name] = np.array([pos.x_val, pos.y_val, pos.z_val])
                
                # 更新轨迹
                trajectories[name].append(positions[name].copy())
                if len(trajectories[name]) > max_trajectory_points:
                    trajectories[name].pop(0)
            
            # 绘制每架无人机
            for name in drone_names:
                pos = positions[name]
                color = colors[name]
                
                # 1. 绘制无人机上方的文字标签
                label_pos = airsim.Vector3r(pos[0], pos[1], pos[2] - 3)  # 上方3米
                client.simPlotStrings(
                    strings=[name],
                    positions=[label_pos],
                    scale=2.0,
                    color_rgba=color,
                    duration=plot_duration
                )
                
                # 2. 绘制无人机位置的球体标记
                client.simPlotPoints(
                    points=[airsim.Vector3r(pos[0], pos[1], pos[2])],
                    color_rgba=color,
                    size=20.0,
                    duration=plot_duration,
                    is_persistent=False
                )
                
                # 3. 绘制轨迹线
                if len(trajectories[name]) > 1:
                    trajectory_points = [
                        airsim.Vector3r(p[0], p[1], p[2]) 
                        for p in trajectories[name]
                    ]
                    # 轨迹用半透明颜色
                    trail_color = color.copy()
                    trail_color[3] = 0.3  # 半透明
                    client.simPlotLineStrip(
                        points=trajectory_points,
                        color_rgba=trail_color,
                        thickness=2.0,
                        duration=plot_duration,
                        is_persistent=False
                    )
            
            # 4. 绘制追击者到目标的连线
            target_pos = positions["Drone0"]
            for hunter_name in ["Drone1", "Drone2", "Drone3"]:
                hunter_pos = positions[hunter_name]
                distance = np.linalg.norm(hunter_pos - target_pos)
                
                # 根据距离改变连线颜色 (近=绿，远=红)
                if distance < 10:
                    line_color = [0.0, 1.0, 0.0, 0.5]  # 绿色
                elif distance < 20:
                    line_color = [1.0, 1.0, 0.0, 0.5]  # 黄色
                else:
                    line_color = [1.0, 0.0, 0.0, 0.5]  # 红色
                
                client.simPlotLineList(
                    points=[
                        airsim.Vector3r(hunter_pos[0], hunter_pos[1], hunter_pos[2]),
                        airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
                    ],
                    color_rgba=line_color,
                    thickness=1.5,
                    duration=plot_duration,
                    is_persistent=False
                )
                
                # 在连线中点显示距离
                mid_point = (hunter_pos + target_pos) / 2
                client.simPlotStrings(
                    strings=[f"{distance:.1f}m"],
                    positions=[airsim.Vector3r(mid_point[0], mid_point[1], mid_point[2])],
                    scale=1.0,
                    color_rgba=[1.0, 1.0, 1.0, 1.0],
                    duration=plot_duration
                )
            
            # 5. 绘制追击者之间的三角形
            hunter_positions = [positions[name] for name in ["Drone1", "Drone2", "Drone3"]]
            for i in range(3):
                p1 = hunter_positions[i]
                p2 = hunter_positions[(i + 1) % 3]
                client.simPlotLineList(
                    points=[
                        airsim.Vector3r(p1[0], p1[1], p1[2]),
                        airsim.Vector3r(p2[0], p2[1], p2[2])
                    ],
                    color_rgba=[1.0, 1.0, 1.0, 0.3],  # 白色半透明
                    thickness=1.0,
                    duration=plot_duration,
                    is_persistent=False
                )
            
            # 控制台输出信息
            print(f"\r目标(Drone0): ({positions['Drone0'][0]:.1f}, {positions['Drone0'][1]:.1f}, {positions['Drone0'][2]:.1f})", end="")
            
            iter_count += 1
            time.sleep(sleep_interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        try:
            client.simFlushPersistentMarkers()
        except:
            pass  # 忽略清理时的错误

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hz', type=float, default=4.0, help='visualization update rate in Hz (default 4)')
    parser.add_argument('--no-flush', action='store_true', help='disable simFlushPersistentMarkers to avoid clearing markers each loop (reduces flicker)')
    args = parser.parse_args()
    visualize_with_markers(update_rate_hz=args.hz, no_flush=args.no_flush)
