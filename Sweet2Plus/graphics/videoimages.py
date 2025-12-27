import os
import random
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
import ipdb
import random 

# === Crop settings ===
CROP_SETTINGS = {
    'front': {'y_start': 0, 'y_end': 800, 'x_start': 500, 'x_end': 1300},
    'side':  {'y_start': 0, 'y_end': 800, 'x_start': 100, 'x_end': 900},
}

def find_videos(base_dir, pattern):
    all_videos = glob(os.path.join(base_dir, '**', f'*{pattern}*.avi'), recursive=True)
    # Skip videos containing "cropped" (case-insensitive)
    return [v for v in all_videos if 'cropped' not in v.lower()]

def select_random_frame(video_path, crop_vals):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        raise ValueError(f"No frames in {video_path}")
    rand_frame_num = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {rand_frame_num} from {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    y_start, y_end = crop_vals['y_start'], crop_vals['y_end']
    x_start, x_end = crop_vals['x_start'], crop_vals['x_end']
    return frame_rgb[y_start:y_end, x_start:x_end]

def get_valid_frames(video_list, count, crop_vals):
    print(f"\n🔍 Attempting to collect {count} valid frames from {len(video_list)} videos...")
    filtered_list = [v for v in video_list if 'cropped' not in v.lower()]
    if len(filtered_list) < count:
        raise RuntimeError(f"Only {len(filtered_list)} usable videos after filtering, but {count} needed.")
    random.shuffle(filtered_list)

    valid_frames = []
    used_paths = []

    for candidate in filtered_list:
        if len(valid_frames) >= count:
            break
        try:
            frame = select_random_frame(candidate, crop_vals)
            valid_frames.append(frame)
            used_paths.append(candidate)
            print(f"✅ Loaded frame from {candidate}")
        except Exception as e:
            print(f"❌ Failed on {candidate}: {e}")

    if len(valid_frames) < count:
        raise RuntimeError(
            f"❌ Only collected {len(valid_frames)} valid frames out of {count} needed.\n"
            f"Tried all {len(filtered_list)} videos."
        )

    return valid_frames, used_paths

def find_tif_dirs(base_dir, pattern='reg_tif'):
    dirs_found = []
    for root, subdirs, files in os.walk(base_dir):
        if pattern in os.path.basename(root):
            dirs_found.append(root)
    return dirs_found

def get_tif_files_case_insensitive(tif_dir):
    # Search only inside 'reg_tif' subfolder
    if not os.path.isdir(tif_dir):
        return []
    all_files = os.listdir(tif_dir)
    tif_files = sorted([
        f for f in all_files if f.lower().endswith(('.tif', '.tiff'))
    ])
    return [os.path.join(tif_dir, f) for f in tif_files]

def gamma_correction(img, gamma=0.7):
    """
    Apply gamma correction to brighten or darken an image.
    img should be a float image normalized between 0 and 1.
    gamma < 1 brightens, gamma > 1 darkens.
    """
    img = np.clip(img, 0, 1)
    corrected = np.power(img, gamma)
    return corrected

def load_all_tiffs_and_max(tif_dir, gamma=0.7):
    tif_files = get_tif_files_case_insensitive(tif_dir)
    if len(tif_files) == 0:
        raise RuntimeError(f"No TIFF files found in {tif_dir}")
    
    max_proj = None
    for f in tif_files:
        img = tifffile.imread(f)
        if img.ndim == 3:
            img_max = img.max(axis=0)
        else:
            img_max = img
        if max_proj is None:
            max_proj = img_max
        else:
            max_proj = np.maximum(max_proj, img_max)

    # Normalize to 0-1
    max_proj = max_proj.astype(np.float32)
    max_proj -= max_proj.min()
    max_val = max_proj.max()
    if max_val != 0:
        max_proj /= max_val

    # Apply gamma correction to brighten
    max_proj = gamma_correction(max_proj, gamma=gamma)

    return max_proj, tif_dir

def find_f_mlp_files(base_dir):
    return glob(os.path.join(base_dir, '**', 'F_mlp.npy'), recursive=True)

def load_and_prepare_traces(f_mlp_path):
    data = np.load(f_mlp_path)
    if data.shape[0] < 10:
        raise ValueError(f"{f_mlp_path} has fewer than 10 rows.")
    
    # Offset each row for visualization
    traces = data[:10]
    offsets = np.arange(10) * 2  # adjust spacing if needed
    offset_traces = traces + offsets[:, None]
    return offset_traces, f_mlp_path


def main():
    base_dir = r'C:\Users\listo\tmt_experiment_2024_working_file'

    # === Front and Side Videos ===
    front_videos = find_videos(base_dir, pattern='front')
    side_videos = find_videos(base_dir, pattern='side')
    print(f"Found {len(front_videos)} front videos and {len(side_videos)} side videos after filtering.")

    front_frames, front_paths = get_valid_frames(front_videos, 6, CROP_SETTINGS['front'])
    side_frames, side_paths = get_valid_frames(side_videos, 6, CROP_SETTINGS['side'])

    # === TIFF Max Projections ===
    tif_dirs = find_tif_dirs(base_dir)
    max_proj_images, max_proj_titles = [], []

    for d in tif_dirs:
        try:
            max_img, title = load_all_tiffs_and_max(d)
            max_proj_images.append(max_img)
            max_proj_titles.append(title)
            print(f"✅ Processed max projection for {d}")
        except Exception as e:
            print(f"❌ Skipping {d} - {e}")

    if len(max_proj_images) < 6:
        raise RuntimeError(f"Only found {len(max_proj_images)} valid max projection images, need at least 6.")

    selected_indices = random.sample(range(len(max_proj_images)), 6)
    max_proj_images = [max_proj_images[i] for i in selected_indices]
    max_proj_titles = [max_proj_titles[i] for i in selected_indices]

    # === F_mlp Traces ===
    f_mlp_files = find_f_mlp_files(base_dir)
    if len(f_mlp_files) < 6:
        raise RuntimeError(f"Found only {len(f_mlp_files)} F_mlp.npy files, need at least 6.")

    selected_f_mlp = random.sample(f_mlp_files, 6)
    trace_plots = []
    trace_titles = []

    for path in selected_f_mlp:
        try:
            offset_traces, title = load_and_prepare_traces(path)
            trace_plots.append(offset_traces)
            trace_titles.append(os.path.basename(os.path.dirname(path)))
        except Exception as e:
            print(f"❌ Error loading trace from {path}: {e}")

    # === Plot All ===
    all_images = front_frames + side_frames + max_proj_images  # 3×6 images
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))
    axs = axs.flatten()

    # Row 1–3: image data
    for img, ax in zip(all_images, axs[:18]):
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.axis('off')

    # Row 4: traces
    for i, (trace, ax) in enumerate(zip(trace_plots, axs[18:24])):
        offset = 0
        for row_idx, row in enumerate(trace[:10]):
            ax.plot(row + offset, linewidth=1.5, color='black')
            # Update offset for next trace: max value of current trace + 1
            offset += (row.max() - row.min()) + 1
        ax.axis('off')
        if i < len(trace_titles):
            ax.set_title(os.path.basename(trace_titles[i]), fontsize=7)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02, wspace=0.02)
    plt.savefig('combined_mouse_maxproj.jpg', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print("✅ Saved combined figure as combined_mouse_maxproj.png")


if __name__ == "__main__":
    main()
