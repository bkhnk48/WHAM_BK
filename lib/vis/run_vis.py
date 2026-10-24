import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
import pdb

from lib.vis.renderer import Renderer, get_global_cameras


def render_mp4(render_frame, video, output_file, fps):
    """
    Render từng frame và ghi thành video MP4.

    Args:
        render_frame: Hàm render một frame.
        video: Đường dẫn video đầu vào.
        output_file: Đường dẫn MP4 đầu ra.
        fps: FPS của video đầu ra.
    """
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = imageio.get_writer(
        output_file,
        fps=fps,
        mode="I",
        format="FFMPEG",
        macro_block_size=1,
    )

    bar = Bar("Rendering MP4 ...", fill="#", max=length)

    frame_i = 0

    try:
        while True:
            flag, org_img = cap.read()

            if not flag:
                break

            # Render mesh lên frame hiện tại.
            img = render_frame(frame_i, org_img)

            # Ghi frame vào MP4.
            writer.append_data(img)

            frame_i += 1
            bar.next()

    finally:
        cap.release()
        writer.close()
        bar.finish()

    print(f"MP4 saved to: {output_file}")


def save_mesh_jpgs(render_frame, video, output_dir):
    """
    Render từng frame và lưu thành ảnh JPG.

    Args:
        render_frame: Hàm render một frame.
        video: Đường dẫn video đầu vào.
        output_dir: Thư mục chứa ảnh JPG.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar("Saving rendered JPGs ...", fill="#", max=length)

    frame_i = 0

    try:
        while True:
            flag, org_img = cap.read()

            if not flag:
                break

            # Render mesh lên frame hiện tại.
            img = render_frame(frame_i, org_img)

            # Renderer tạo ảnh RGB; OpenCV ghi ảnh BGR.
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frame_path = osp.join(
                output_dir,
                f"frame_{frame_i:06d}.jpg",
            )

            if not cv2.imwrite(
                frame_path,
                img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Cannot save image: {frame_path}")

            frame_i += 1
            bar.next()

    finally:
        cap.release()
        bar.finish()

    print(f"Rendered JPGs saved to: {output_dir}")


def run_vis_on_demo(
    cfg,
    video,
    results,
    output_pth,
    smpl,
    vis_global=True,
    mp4=False,
):
    """
    Render kết quả WHAM.

    mp4=True:
        Xuất output.mp4.

    mp4=False:
        Xuất từng frame JPG vào thư mục mp4/.
    """

    # Chuyển NumPy sang Tensor trên thiết bị được cấu hình.
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    # Đọc metadata video.
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    os.makedirs(output_pth, exist_ok=True)

    # Focal length heuristic của renderer, không phải camera calibration.
    focal_length = (width ** 2 + height ** 2) ** 0.5

    renderer = Renderer(
        width,
        height,
        focal_length,
        cfg.DEVICE,
        smpl.faces,
    )

    # ---------------------------------------------------------
    # Chuẩn bị dữ liệu để render góc nhìn tọa độ toàn cục.
    # ---------------------------------------------------------
    if vis_global:
        if not results:
            raise ValueError(
                "results is empty; cannot render global coordinates."
            )

        # Chỉ render global view cho người xuất hiện nhiều frame nhất.
        n_frames = {
            k: len(results[k]["frame_ids"])
            for k in results.keys()
        }

        sid = max(n_frames, key=n_frames.get)

        global_output = smpl.get_output(
            body_pose=tt(results[sid]["pose_world"][:, 3:]),
            global_orient=tt(results[sid]["pose_world"][:, :3]),
            betas=tt(results[sid]["betas"]),
            transl=tt(results[sid]["trans_world"]),
        )

        verts_glob = global_output.vertices.cpu()

        # Đặt đáy mesh trên mặt đất.
        verts_glob[..., 1] -= verts_glob[..., 1].min()

        mean_verts = verts_glob.mean(1)

        cx, cz = (
            mean_verts.max(0)[0] + mean_verts.min(0)[0]
        )[[0, 2]] / 2.0

        sx, sz = (
            mean_verts.max(0)[0] - mean_verts.min(0)[0]
        )[[0, 2]]

        scale = max(sx.item(), sz.item()) * 1.5

        renderer.set_ground(scale, cx.item(), cz.item())

        global_R, global_T, global_lights = get_global_cameras(
            verts_glob,
            cfg.DEVICE,
        )

    # Camera mặc định để render lên ảnh video.
    default_R = torch.eye(3)
    default_T = torch.zeros(3)

    # Cache ảnh global gần nhất, giống ý tưởng của code gốc.
    img_glob_cache = None

    # ---------------------------------------------------------
    # Hàm dùng chung: render một frame.
    # ---------------------------------------------------------
    def render_frame(frame_i, org_img):
        nonlocal img_glob_cache

        # OpenCV đọc BGR; renderer trong code gốc dùng RGB.
        img = org_img[..., ::-1].copy()

        # Render mesh lên video gốc.
        renderer.create_camera(default_R, default_T)

        for _, val in results.items():
            frame_indices = np.where(
                val["frame_ids"] == frame_i
            )[0]

            if len(frame_indices) == 0:
                continue

            idx = frame_indices[0]

            verts = torch.from_numpy(
                val["verts"][idx]
            ).to(cfg.DEVICE)

            img = renderer.render_mesh(verts, img)

        # Render global view nếu được yêu cầu.
        if vis_global:
            frame_indices = np.where(
                results[sid]["frame_ids"] == frame_i
            )[0]

            if len(frame_indices) > 0:
                idx = frame_indices[0]

                verts = verts_glob[[idx]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)

                colors = torch.ones(
                    (1, 4),
                    dtype=torch.float32,
                    device=cfg.DEVICE,
                )
                colors[..., :3] *= 0.9

                cameras = renderer.create_camera(
                    global_R[idx],
                    global_T[idx],
                )

                img_glob_cache = renderer.render_with_ground(
                    verts,
                    faces,
                    colors,
                    cameras,
                    global_lights,
                )

            # Nếu frame chưa có global view, giữ cách xử lý gần
            # với code gốc: sử dụng ảnh global gần nhất nếu có.
            if img_glob_cache is None:
                img_glob = np.ones_like(img) * 255
            else:
                img_glob = img_glob_cache

                if img_glob.shape[0] != img.shape[0]:
                    new_width = int(
                        img_glob.shape[1]
                        * img.shape[0]
                        / img_glob.shape[0]
                    )

                    img_glob = cv2.resize(
                        img_glob,
                        (new_width, img.shape[0]),
                    )

            img = np.concatenate(
                (img, img_glob),
                axis=1,
            )

        return img

    # ---------------------------------------------------------
    # Chọn phương thức xuất kết quả.
    # ---------------------------------------------------------
    if mp4:
        output_file = osp.join(output_pth, "output.mp4")

        render_mp4(
            render_frame=render_frame,
            video=video,
            output_file=output_file,
            fps=fps,
        )

    else:
        output_dir = osp.join(output_pth, "mp4")

        save_mesh_jpgs(
            render_frame=render_frame,
            video=video,
            output_dir=output_dir,
        )
