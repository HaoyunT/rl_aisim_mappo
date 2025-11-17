"""
MAPPO 训练脚本 - 多无人机协同包围任务
使用 4 阶段 Curriculum + 滑动窗口升级/回退机制
"""

import os
import glob
from mappo import MAPPO, MAPPOConfig, MultiAgentEnvWrapper
from drone_surround_env import DroneSurroundEnv

# Central training total timesteps (edit here to change globally)
TOTAL_TIMESTEPS = 2_000_000  # default: 2 million


def clean_old_models():
    """删除旧的训练模型（保留备份）"""
    print("🧹 清理旧模型...")

    best_model = "./models/mappo_best.pt"
    if os.path.exists(best_model):
        backup_name = "./models/mappo_best_backup.pt"
        if os.path.exists(backup_name):
            os.remove(backup_name)
        os.rename(best_model, backup_name)
        print(f"   ✓ 已备份最佳模型 -> {backup_name}")

    checkpoints = glob.glob("./models/mappo_checkpoint_*.pt")
    for ckpt in checkpoints:
        os.remove(ckpt)
        print(f"   ✓ 已删除 {os.path.basename(ckpt)}")

    for filename in ["mappo_interrupted.pt", "mappo_final.pt"]:
        filepath = f"./models/{filename}"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   ✓ 已删除 {filename}")

    print("🧹 清理完成！\n")


def train(total_timesteps: int = TOTAL_TIMESTEPS, load_checkpoint: str = None):
    """
    训练 MAPPO 模型，使用 DroneSurroundEnv(curriculum=True)
    """
    print("=" * 70)
    print(" 初始化训练环境...")
    print("=" * 70)

    base_env = DroneSurroundEnv(curriculum=True)
    env = MultiAgentEnvWrapper(base_env)

    eval_env_raw = DroneSurroundEnv(curriculum=True)
    eval_env = MultiAgentEnvWrapper(eval_env_raw)

    print(f"\n📋 环境配置:")
    print(f"   最大步数: {base_env.max_steps}")
    print(f"   初始半径: {base_env.init_radius_range}")
    print(f"   世界边界: {base_env.world_limit}")
    print(f"   课程学习: {base_env.curriculum_enabled}")
    if base_env.curriculum_enabled:
        stage = base_env.get_curriculum_stage()
        ring_center = 0.5 * (base_env.r_min + base_env.r_max)
        print(f"   当前阶段: Stage {stage}")
        print(f"   目标圆环: [{base_env.r_min}, {base_env.r_max}] 米 (中心={ring_center:.1f} 米)")
        print(f"   目标速度: {base_env.target_speed} m/s")
        print(f"   成功阈值: 连续保持 {base_env.success_threshold} 步")

    cfg = MAPPOConfig()

    print(f"\n📋 MAPPO 配置:")
    print(f"   学习率: actor={cfg.actor_lr}, critic={cfg.critic_lr}")
    print(f"   Clip 范围: {cfg.clip_range}")
    print(f"   熵系数: {cfg.ent_coef}")
    print(f"   批量大小: {cfg.batch_size}, 步数: {cfg.n_steps}, 迭代数: {cfg.n_epochs}")
    print()

    algo = MAPPO(env, cfg)

    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"📂 加载已有模型: {load_checkpoint}")
        algo.load(load_checkpoint)
        print("✓ 模型加载完成，将继续训练\n")
    else:
        print("🆕 从头开始训练\n")

    print("=" * 70)
    print(" 开始训练")
    print("=" * 70)
    print(f"总步数: {total_timesteps:,}")
    print(f"评估频率: 每 10,000 步")
    print(f"评估回合数: 20")
    print(f"检查点频率: 每 10,000 步（已禁用，使用阶段最佳保存）")
    print("=" * 70)
    print()

    try:
        algo.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=20,
            checkpoint_freq=0,  # disable periodic checkpoints; we save per-stage bests
            save_best_path="./models/mappo_best.pt",
            final_model_path="./models/mappo_final.pt",
            show_progress=True,
            stage_save_dir="./models"
        )

        print("\n" + "=" * 70)
        print(" 训练完成！")
        print("=" * 70)
        print(f"✓ 每阶段最佳模型已保存在 ./models/mappo_best_stage_<stage>.pt")
        print(f"✓ 全局最佳（按平均回报）保存在: ./models/mappo_best.pt")
        print(f"✓ 最终模型已保存: ./models/mappo_final.pt")

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
        interrupted_path = "./models/mappo_interrupted.pt"
        algo.save(interrupted_path)
        print(f"✓ 中断模型已保存: {interrupted_path}")
        print("   可以使用此模型继续训练")

    finally:
        print("\n🔒 关闭环境...")
        base_env.close()
        eval_env_raw.close()
        print("✓ 环境已关闭")


def main():
    """主函数 - 提供交互式训练选项"""
    print()
    print("=" * 70)
    print(" MAPPO 多无人机协同包围训练系统")
    print("=" * 70)
    print()
    print("请选择训练模式:")
    print("  [1] 全新训练（删除旧模型，从头开始）")
    print("  [2] 继续训练（从最佳模型继续）")
    print("  [3] 从中断点继续")
    print("  [4] 仅清理模型")
    print("  [5] 直接开始（使用默认设置）")
    print()

    while True:
        choice = input("请输入选项 (1/2/3/4/5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            break
        print("❌ 无效输入，请重新选择")

    print()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    if choice == '1':
        print("📋 模式: 全新训练")
        clean_old_models()
        train(total_timesteps=TOTAL_TIMESTEPS)  # use central constant

    elif choice == '2':
        print("📋 模式: 继续训练（最佳模型）")
        checkpoint_path = "./models/mappo_best.pt"
        if os.path.exists(checkpoint_path):
            print(f"✓ 找到最佳模型: {checkpoint_path}")
            print("⚠️  注意：如果奖励函数或环境有大改动，建议选择 [1] 全新训练")
            print()
            train(total_timesteps=TOTAL_TIMESTEPS, load_checkpoint=checkpoint_path)
        else:
            print("❌ 未找到最佳模型，将从头开始")
            print()
            train(total_timesteps=TOTAL_TIMESTEPS)

    elif choice == '3':
        print("📋 模式: 从中断点继续")
        checkpoint_path = "./models/mappo_interrupted.pt"
        if os.path.exists(checkpoint_path):
            print(f"✓ 找到中断模型: {checkpoint_path}")
            print()
            train(total_timesteps=TOTAL_TIMESTEPS, load_checkpoint=checkpoint_path)
        else:
            print("❌ 未找到中断模型")
            print("可用的模型:")
            for model_file in glob.glob("./models/*.pt"):
                print(f"  - {model_file}")
            return

    elif choice == '4':
        print("📋 模式: 仅清理")
        clean_old_models()
        print("✓ 已完成清理，未开始训练")

    elif choice == '5':
        print("📋 模式: 直接开始（默认设置）")
        train(total_timesteps=TOTAL_TIMESTEPS)


if __name__ == "__main__":
    main()
