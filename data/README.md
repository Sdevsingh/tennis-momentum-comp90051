# Data

**The dataset is not committed to this repo** (the project brief says: *do not
submit data*, and `.gitignore` enforces it).

## Source

Jeff Sackmann — Grand Slam point-by-point:
https://github.com/JeffSackmann/tennis_slam_pointbypoint

Coverage: most main-draw singles Grand Slam matches since 2011, scraped from
the official Grand Slam sites. **Column availability varies by tournament and
year** — serve speed, 1st/2nd-serve flags, and rally length are missing for
many matches. `src/data_loading.py` discovers columns at runtime and reports
coverage, so plan features around what is actually present.

> License: Sackmann's tennis data is released under CC BY-NC-SA 4.0. Cite the
> dataset in the report's bibliography.

## How to fetch

```bash
# from the repo root
git clone https://github.com/JeffSackmann/tennis_slam_pointbypoint.git /tmp/sackmann
mkdir -p data/raw
cp /tmp/sackmann/*-points.csv data/raw/
python -m src.data_loading      # builds data/processed/points.parquet
```

(You can start with a subset of tournaments/years for faster iteration — the
brief explicitly allows using a subset given the cross-validation workload.)

## Columns we depend on

Required (present almost everywhere): `PointServer`, `PointWinner`, `SetNo`,
`GameNo`, `PointNumber`, plus score columns.

Optional (enable extra analysis where present): `Rally`, `Speed_KMH`,
`P1Ace`/`P2Ace`, `P1Winner`/`P2Winner`, `P1UnfErr`/`P2UnfErr`. The ace/winner/
unforced-error flags are what the optional "lucky vs earned point" analysis
would use — check coverage before planning around them.
