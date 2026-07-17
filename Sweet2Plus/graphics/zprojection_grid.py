import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2


def sortday(path):
    _, path = path.split("\\D")
    value = path[2:]
    return int(value)


def build_max_projections(sessions):
    all_projs = []
    for j, session in enumerate(sessions):
        print(j)
        vids = glob.glob(os.path.join(session, '**', '**', 'motion_corrected*'))
        for m, vid in tqdm.tqdm(enumerate(vids), total=len(vids)):
            images = glob.glob(os.path.join(vid, '*.tif*'))
            maxproj = None
            for k, image in tqdm.tqdm(enumerate(images), total=len(images)):
                im = Image.open(image)
                if k == 0:
                    maxproj = np.asarray(im)
                    maxproj = maxproj[..., np.newaxis]
                else:
                    ci = np.asarray(im)
                    ci = ci[..., np.newaxis]
                    maxproj = np.concatenate([maxproj, ci], axis=2)
                    maxproj = maxproj.max(axis=2)
                    maxproj = maxproj[..., np.newaxis]

            all_projs.append([maxproj, m, j])
    return all_projs


def plot_grid(all_projs, output_path='GridOfScans.jpg'):
    if not all_projs:
        print("No projections found; skipping grid plot.")
        return

    fig = plt.figure(figsize=(15, 15), dpi=400)
    spot = 1
    prevday = all_projs[0][2]
    for sample in all_projs:
        image, mouse, day = sample
        if day > prevday:
            spot = day * 9 + 1
            prevday = day
        ax1 = plt.subplot(9, 5, spot)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        ax1.set_aspect('equal')
        spot += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.1, hspace=0)
    plt.savefig(output_path)
    plt.close(fig)


def save_individual_projections(all_projs, output_dir=r'C:\tmt_assay\paper_figures\zproj'):
    for sample in all_projs:
        image, mouse, day = sample
        filename = os.path.join(output_dir, f'mouse{mouse}day{day}.jpg')
        image = np.squeeze(image, axis=2).astype('uint8')
        cv2.imwrite(filename, image)


def main():
    sessions = glob.glob(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day*')
    sessions = sorted(sessions, key=sortday)

    all_projs = build_max_projections(sessions)
    plot_grid(all_projs)
    save_individual_projections(all_projs)


if __name__ == '__main__':
    main()
