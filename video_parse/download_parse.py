import pandas as pd
from get_videos_2 import download_video
import argparse
from parse_videos_2 import parse_videos
from send2trash import send2trash

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

    for i, link in enumerate(youtube_links):
        game_name = game_names[i]

        download_video(link, game_name)
        parse_videos(game_name, team_shot_table)

        video_path = f'data/game_videos/{game_name}.mp4'
        if video_path.exists():
            send2trash(str(video_path))   # safe: goes to Trash/Recycle Bin
            print(f"[TRASHED] {video_path}")

if __name__ == '__main__':
    main()