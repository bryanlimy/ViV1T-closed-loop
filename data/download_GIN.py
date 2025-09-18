import argparse
import os
from multiprocessing import Pool

import wget

SENSORIUM1_REPO = "https://gin.g-node.org/pollytur/Sensorium2023Data/raw/master"
SENSORIUM1 = [
    "dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip",
    "dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce.zip",
    "dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce.zip",
    "dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce.zip",
    "dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce.zip",
]

SENSORIUM2_REPO = "https://gin.g-node.org/pollytur/sensorium_2023_dataset/raw/master"
SENSORIUM2 = [
    "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20.zip",
    "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20.zip",
    "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20.zip",
    "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20.zip",
    "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20.zip",
]


def download(filename: str, output_dir: str):
    print(f"Downloading {filename} to {output_dir}...\n")
    wget.download(filename, out=output_dir, bar=None)


def main(args):
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    urls = [f"{SENSORIUM1_REPO}/{filename}" for filename in SENSORIUM1]
    urls += [f"{SENSORIUM2_REPO}/{filename}" for filename in SENSORIUM2]

    pool = Pool(processes=args.num_workers)

    pool.starmap(
        download,
        [
            (
                url,
                os.path.join(args.data_dir, os.path.basename(url)),
            )
            for url in urls
        ],
    )

    pool.close()

    print(f"Saved {len(urls)} to {args.data_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to store extracted data from DeepLake datasets.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of concurrent downloads",
    )
    main(parser.parse_args())
