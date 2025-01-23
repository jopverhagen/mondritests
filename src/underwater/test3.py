import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

from models import create_model
import options as option
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr


def plot_results_with_summary(metrics, example_images, save_path, avg_metrics, dataset_name):
    """
    Maak een plot van de resultaten inclusief metrics, voorbeeldafbeeldingen en gemiddelde resultaten.
    """
    psnr, ssim, lpips_scores, test_times = metrics
    avg_psnr, avg_ssim, avg_lpips, avg_time = avg_metrics
    lq_img, sr_img, gt_img = example_images

    fig, axs = plt.subplots(3, 2, figsize=(14, 15))

    # Plot PSNR
    axs[0, 0].plot(psnr, label="PSNR")
    axs[0, 0].set_title("PSNR per afbeelding")
    axs[0, 0].set_xlabel("Afbeelding index")
    axs[0, 0].set_ylabel("PSNR (dB)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot SSIM
    axs[0, 1].plot(ssim, label="SSIM")
    axs[0, 1].set_title("SSIM per afbeelding")
    axs[0, 1].set_xlabel("Afbeelding index")
    axs[0, 1].set_ylabel("SSIM")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot LPIPS
    axs[1, 0].plot(lpips_scores, label="LPIPS", color="red")
    axs[1, 0].set_title("LPIPS per afbeelding")
    axs[1, 0].set_xlabel("Afbeelding index")
    axs[1, 0].set_ylabel("LPIPS")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Toon voorbeeldafbeeldingen
    axs[1, 1].imshow(np.hstack([lq_img, sr_img, gt_img]))
    axs[1, 1].set_title("Voorbeeld: LQ | MondriAI | Ground Truth")
    axs[1, 1].axis("off")

    # Gemiddelde resultaten toevoegen
    avg_results_text = (
        f"Gemiddelde resultaten voor {dataset_name}:\n"
        f"PSNR: {avg_psnr:.2f} dB\n"
        f"SSIM: {avg_ssim:.4f}\n"
        f"LPIPS: {avg_lpips:.4f}\n"
        f"Gemiddelde testtijd per afbeelding: {avg_time:.4f} seconden"
    )
    axs[2, 0].axis("off")
    axs[2, 0].text(0.1, 0.5, avg_results_text, fontsize=12, verticalalignment='center')

    # Lege as om layout symmetrie te bewaren
    axs[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_examples_with_table(metrics, examples, save_path, title):
    """
    Maak een plot met meerdere voorbeeldafbeeldingen en een tabel met de resultaten.
    """
    psnr, ssim, lpips_scores = metrics
    num_examples = len(examples)
    fig, axs = plt.subplots(num_examples + 1, 1, figsize=(15, 5 * (num_examples + 1)))

    # Tabelgegevens
    headers = ["Afbeelding", "PSNR (dB)", "SSIM", "LPIPS"]
    data = [
        [f"Afbeelding {i+1}", f"{psnr[i]:.2f}", f"{ssim[i]:.4f}", f"{lpips_scores[i]:.4f}"]
        for i in range(len(psnr))
    ]

    # Resultaten tabel
    axs[0].axis("tight")
    axs[0].axis("off")
    table = axs[0].table(
        cellText=data, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Toon voorbeelden
    for i, (lq_img, sr_img, gt_img) in enumerate(examples):
        axs[i + 1].imshow(np.hstack([lq_img, sr_img, gt_img]))
        axs[i + 1].set_title(f"{title} - Voorbeeld {i+1}: Low quality | MondriAI | Ground Truth")
        axs[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_all_examples(examples, save_path, title):
    """
    Toon alle voorbeelden uit de test set met annotaties.
    """
    num_examples = len(examples)
    fig, axs = plt.subplots(num_examples, 1, figsize=(15, 5 * num_examples))

    for i, (lq_img, sr_img, gt_img) in enumerate(examples):
        axs[i].imshow(np.hstack([lq_img, sr_img, gt_img]))
        axs[i].set_title(
            f"{title} - Voorbeeld {i+1}\n Low quality | MondriAI | Ground Truth"
        )
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Argumenten parseren
parser = argparse.ArgumentParser(description="Script voor het testen van modellen.")
parser.add_argument(
    "-opt",
    type=str,
    default="test.yml",
    help="Pad naar het opties YAML-bestand (standaard: 'test.yml')."
)
args = parser.parse_args()

# Opties laden
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# Zorg dat de map test_results bestaat
output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

# Testdatasets en dataloaders aanmaken
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print(f"Aantal testafbeeldingen in [{dataset_opt['name']}]: {len(test_set)}")
    test_loaders.append(test_loader)

# Model laden
model = create_model(opt)
device = model.device

# Mondri instellen
mondri = util.IRMondri(
    max_sigma=opt["mondri"]["max_sigma"],
    T=opt["mondri"]["T"],
    schedule=opt["mondri"]["schedule"],
    eps=opt["mondri"]["eps"],
    device=device
)
mondri.set_model(model.model)
sampling_mode = opt["mondri"]["sampling_mode"]

# LPIPS-instantie maken
lpips_fn = lpips.LPIPS(net="alex").to(device)

# Testen van datasets
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    dataset_dir = os.path.join(output_dir, test_set_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\nTesten van dataset: [{test_set_name}]")

    # Resultaten opslaan
    psnr_list, ssim_list, lpips_list, times = [], [], [], []
    all_examples = []

    for i, test_data in enumerate(test_loader):
        need_GT = "GT" in test_data and test_data["GT"] is not None
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Model input voorbereiden
        LQ, GT = test_data["LQ"], test_data.get("GT")
        noisy_state = mondri.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        start_time = time.time()
        model.test(mondri, mode=sampling_mode, save_states=False)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        # Resultaten ophalen
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())
        LQ_img = util.tensor2img(visuals["Input"].squeeze())
        GT_img = util.tensor2img(visuals["GT"].squeeze()) if need_GT else None

        # Sla op voor voorbeelden
        all_examples.append((LQ_img, output, GT_img))

        if need_GT:
            gt_img_norm = GT_img / 255.0
            sr_img_norm = output / 255.0

            psnr = util.calculate_psnr(sr_img_norm * 255, gt_img_norm * 255)
            ssim = util.calculate_ssim(sr_img_norm * 255, gt_img_norm * 255)
            lpips_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1
            ).item()

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_score)

            print(f"Afbeelding {img_name} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_score:.4f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    avg_time = np.mean(times)

    print(f"\n---- Gemiddelde resultaten voor {test_set_name} ----")
    print(f"PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    print(f"Gemiddelde testtijd per afbeelding: {avg_time:.4f} seconden")

    # Resultaten plotten
    plot_results_with_summary(
        (psnr_list, ssim_list, lpips_list, times),
        all_examples[0],
        os.path.join(dataset_dir, f"{test_set_name}_results_plot_with_summary.png"),
        (avg_psnr, avg_ssim, avg_lpips, avg_time),
        test_set_name
    )
    plot_examples_with_table(
        (psnr_list, ssim_list, lpips_list),
        all_examples[:3],
        os.path.join(dataset_dir, f"{test_set_name}_examples_with_table.png"),
        f"{test_set_name} MondriAI underwater test set 2.1"
    )
    plot_all_examples(
        all_examples,
        os.path.join(dataset_dir, f"{test_set_name}_all_examples.png"),
        f"{test_set_name} MondriAI underwater test set 2.1"
    )

