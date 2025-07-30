# Standard Libraries
import os
from pathlib import Path
from time import sleep
from typing import Mapping

# Dependencies
import requests
from tqdm import tqdm


def retrieve_urls(url_dict: Mapping[str, str | os.PathLike]):
    """
    Retrieves data from PDS URLs.

    Parameters
    ----------
    url_dict: dict[str, str]
        Keys are the URLs to download and the values are the paths to save
        them to.
    """
    for url, save in url_dict.items():
        save = Path(save)
        download_with_resume_and_retries(url, save)


def download_with_resume_and_retries(
    url: str,
    filename: str | os.PathLike,
    max_retries=5, backoff=5
):
    filename = Path(filename)
    attempt = 0

    while attempt < max_retries:
        try:
            # Determine how much we already downloaded
            downloaded = os.path.getsize(filename) \
                if os.path.exists(filename) else 0

            # Send a request with the Range header to resume download
            headers = {"Range": f"bytes={downloaded}-"} \
                if downloaded > 0 else {}

            with requests.get(
                url, headers=headers, stream=True, timeout=30
            ) as r:
                r.raise_for_status()

                # Total file size (from Content-Range if available)
                total_size = (
                    int(r.headers.get("Content-Range",
                                      "bytes */0").split("/")[-1])
                    if "Content-Range" in r.headers
                    else int(r.headers.get("Content-Length", 0)) + downloaded
                )

                mode = "ab" if downloaded > 0 else "wb"
                with open(filename, mode) as f, tqdm(
                    total=total_size,
                    initial=downloaded,
                    unit="iB",
                    unit_scale=True,
                    desc=filename.__str__(),
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Download successful
            return True

        except requests.RequestException as e:
            attempt += 1
            print(f"[Attempt {attempt}/{max_retries}] Download error: {e}")
            sleep(backoff)

    print("Download failed after multiple retries.")
    return False
