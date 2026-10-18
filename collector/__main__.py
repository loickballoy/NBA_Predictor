import argparse

from collector.fetch_raw import fetch_all, load_all_awards, load_all_career_stats
from collector.build_dataset import build_windows, export_splits
from collector.clean import make_season_table

def cmd_fetch(args) -> int:
    failures = fetch_all()
    return 1 if failures else 0

def cmd_build(args):
    if args.val_from > args.test_from:
        raise SystemExit("--val-from can't be higher than --test-from")

    career_stats = load_all_career_stats()
    awards = load_all_awards()
    print(f"{len(career_stats)} stat lines, {len(awards)} awards lines")

    seasons = make_season_table(career_stats, awards)
    print(f"{len(seasons)} players kept after clean")
    
    windows = build_windows(seasons)
    export_splits(windows, val_from=args.val_from, test_from=args.test_from)
    return 0   
    

def main():
    parser = argparse.ArgumentParser(prog="collector", description="Nba data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser('fetch', help="Downloads the raw data from nba_api")
    p_fetch.set_defaults(func=cmd_fetch)

    p_build = sub.add_parser("build", help="build train/val/test from the raw data")
    p_build.add_argument("--val-from", type=int, default=2020)
    p_build.add_argument("--test-from", type=int, default=2023)
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())