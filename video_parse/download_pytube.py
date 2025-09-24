from pytube import YouTube
import os

def download_video(url: str, save_as: str):
    full_path = f"data/game_videos/{save_as}.mp4"
    print(f"\nDownloading: {url}")

    yt = YouTube(url)

    # Your preferred itags in order of priority
    preferred_itags = [311, 398, 298, 136, 135, 397, 134, 396, 133, 395, 160, 394]

    stream = None
    for itag in preferred_itags:
        candidate = yt.streams.get_by_itag(itag)
        if candidate:
            stream = candidate
            break

    if not stream:
        print(f"[FAIL] {url} (no preferred itag found)")
        return

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    stream.download(filename=full_path)
    print(f"[OK] saved -> {full_path}")


if __name__ == '__main__':
    download_video('https://youtu.be/jDLvQH0craI', 'clemson_louisville_01072025')