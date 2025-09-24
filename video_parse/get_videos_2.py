import subprocess
from pathlib import Path

def download_video(url: str, save_as: str):
    full_path = f'data/game_videos/{save_as}.mp4'

    print(f"\nDownloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "311/398/298/136/135/397/134/396/133/395/160/394",         # video-only format
        "--merge-output-format", "mp4",
        '-o', full_path,
        url,
    ]
    # cmd = [
    #     "youtube-dl",                                   # swap yt-dlp → youtube-dl
    #     "-f", "398/298/136/135/397/134/396/133/395/160/394",  # hard-coded itags
    #     "--merge-output-format", "mp4",
    #     "-o", full_path,
    #     url,
    # ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[OK] {url}")
    else:
        print(f"[FAIL] {url}")

# def main():
#     for url in URLS:
#         download_video(url)

if __name__ == "__main__":
    download_video('https://youtu.be/iHL58wcuDGM?si=25U-fUnz_z5_QBBW', 'bc_louisville_02052025')
