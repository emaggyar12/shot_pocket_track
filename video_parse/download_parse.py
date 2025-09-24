import pandas as pd
from get_videos_2 import download_video
import argparse
from parse_videos_2 import parse_videos
from send2trash import send2trash
from pathlib import Path
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Download and cut clips for a single game from a master table.")
    parser.add_argument("--team_link_table", type=str, help="Team table with links and video names")
    parser.add_argument("--team_shot_table", type=str, help="Table with shots")

    args = parser.parse_args()

    team_link_table = args.team_link_table
    team_shot_table = args.team_shot_table

    df = pd.read_excel(team_link_table)

    youtube_links = df['link'].tolist()
    game_names = df['name'].tolist()
    download = df['download'].tolist()

    for i, link in enumerate(youtube_links):
        if int(download[i]) == 1:
            game_name = game_names[i]

            download_video(link, game_name)
            parse_videos(game_name, team_shot_table)

            video_path = Path(f'data/game_videos/{game_name}.mp4')
            if video_path.exists():
                send2trash(str(video_path))   # safe: goes to Trash/Recycle Bin
                print(f"[TRASHED] {video_path}")
        else:
            print(f'{game_names[i]} scraped, breaking loop')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Nothing passed → show skeleton usage
        print("\nUsage:")
        print("  python video_parse/download_parse.py --team_link_table <team_links.xlsx> --team_shot_table <team_shots.xlsx>\n")
        sys.exit(1)
    main()