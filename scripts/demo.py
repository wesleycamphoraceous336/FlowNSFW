#!/usr/bin/env python3
"""FlowNSFW Demo - Interactive video NSFW detection demonstration.

Usage:
    python scripts/demo.py --ckpt final.pt --source datasets/demo_videos/
    python scripts/demo.py --ckpt final.pt --video path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import torch
import numpy as np
from flow_nsfw import FlowNSFW


def extract_frames_from_video(video_path: Path, max_frames: int = 64):
    """Extract frames from video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def load_frame_sequence(frame_dir: Path):
    """Load frames from directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    frames = []

    for f in sorted(frame_dir.iterdir()):
        if f.suffix.lower() in exts:
            img = cv2.imread(str(f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)

    return frames


def preprocess_frames(frames, target_size=(320, 320)):
    """Convert frames to model input tensor."""
    processed = []
    for frame in frames:
        # Resize
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)

    # Stack and convert to tensor
    frames_np = np.stack(processed)  # (T, H, W, 3)
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, 3, H, W)
    return frames_t


def sliding_window_inference(model, frames_t, clip_len=8, stride=4, device="cuda"):
    """Run sliding window inference."""
    T = frames_t.shape[0]
    if T < clip_len:
        print(f"⚠️  视频太短({T}帧)，需要至少{clip_len}帧")
        return None

    per_frame_conf = [0.0] * T
    windows = []

    for start in range(0, T - clip_len + 1, stride):
        clip = frames_t[start:start + clip_len].unsqueeze(0).to(device)  # (1, T, 3, H, W)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(clip)

        # Get NSFW confidence
        probs = torch.softmax(out["video_cls"], dim=-1)
        nsfw_conf = probs[0, 1].item()
        sfw_conf = probs[0, 0].item()

        verdict = "NSFW" if nsfw_conf > 0.5 else "SFW"
        windows.append({
            "start": start,
            "end": start + clip_len,
            "nsfw_conf": nsfw_conf,
            "sfw_conf": sfw_conf,
            "verdict": verdict,
        })

        # Update per-frame confidence
        if nsfw_conf > 0.5:
            for i in range(start, start + clip_len):
                per_frame_conf[i] = max(per_frame_conf[i], nsfw_conf)

    max_conf = max(per_frame_conf)
    verdict = "NSFW" if max_conf > 0.5 else "SFW"
    nsfw_windows = sum(1 for w in windows if w["verdict"] == "NSFW")

    return {
        "verdict": verdict,
        "max_conf": max_conf,
        "nsfw_windows": nsfw_windows,
        "total_windows": len(windows),
        "n_frames": T,
        "per_frame_conf": per_frame_conf,
        "windows": windows,
    }


def print_result(result, video_name):
    """Pretty print inference result."""
    print("\n" + "="*70)
    print(f"📹 视频: {video_name}")
    print("="*70)

    # Overall verdict
    verdict_icon = "🔞" if result["verdict"] == "NSFW" else "✅"
    print(f"\n{verdict_icon} 判定: {result['verdict']}")
    print(f"   最高置信度: {result['max_conf']:.2%}")
    print(f"   NSFW 窗口: {result['nsfw_windows']}/{result['total_windows']}")
    print(f"   总帧数: {result['n_frames']}")

    # Per-frame confidence bar chart
    print(f"\n📊 逐帧置信度:")
    bar_width = 60
    for i, conf in enumerate(result["per_frame_conf"]):
        bar_len = int(conf * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        marker = "🔞" if conf > 0.5 else "  "
        print(f"   帧{i:3d} [{bar}] {conf:.2%} {marker}")

    # Window details
    print(f"\n🪟 滑动窗口详情:")
    for w in result["windows"]:
        icon = "🔞" if w["verdict"] == "NSFW" else "✅"
        print(f"   {icon} 帧 {w['start']:3d}-{w['end']:3d}: "
              f"{w['verdict']:4s} (conf={w['nsfw_conf']:.2%})")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="FlowNSFW 交互式演示")
    parser.add_argument("--ckpt", required=True, help="模型权重路径")
    parser.add_argument("--source", help="帧序列目录")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--clip-len", type=int, default=8, help="滑动窗口长度")
    parser.add_argument("--stride", type=int, default=4, help="滑动窗口步长")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--temporal-backend", default="attention", help="时序后端")
    args = parser.parse_args()

    if not args.source and not args.video:
        parser.error("必须指定 --source 或 --video")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 FlowNSFW Demo")
    print(f"   设备: {device}")

    # Load model
    print(f"📦 加载模型: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = FlowNSFW(
        dim=128,
        num_heads=4,
        num_temporal_layers=3,
        topk_global=64,
        temporal_backend=args.temporal_backend,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   参数量: {params:.2f}M")
    print(f"   训练步数: {ck.get('step', '?')}")

    # Collect videos
    videos = []
    if args.video:
        videos.append({"name": Path(args.video).name, "type": "video", "path": Path(args.video)})
    if args.source:
        source_path = Path(args.source)
        if source_path.is_dir():
            for subdir in sorted(source_path.iterdir()):
                if subdir.is_dir():
                    videos.append({"name": subdir.name, "type": "frames", "path": subdir})

    if not videos:
        print("❌ 未找到视频或帧序列")
        return 1

    print(f"\n🎬 找到 {len(videos)} 个视频\n")

    # Process each video
    for i, vid in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] 处理: {vid['name']}")

        # Load frames
        if vid["type"] == "video":
            frames = extract_frames_from_video(vid["path"])
        else:
            frames = load_frame_sequence(vid["path"])

        if not frames:
            print(f"   ❌ 无法加载帧")
            continue

        print(f"   加载了 {len(frames)} 帧")

        # Preprocess
        frames_t = preprocess_frames(frames)

        # Inference
        print(f"   推理中...")
        result = sliding_window_inference(
            model, frames_t,
            clip_len=args.clip_len,
            stride=args.stride,
            device=device,
        )

        if result is None:
            continue

        # Print result
        print_result(result, vid["name"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
