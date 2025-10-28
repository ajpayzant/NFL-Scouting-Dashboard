# app.py
# NFL Player Scouting Dashboard (Streamlit + nflreadpy + Polars)
# Sections:
# 1) Player picker (Team/Pos/Player)
# 2) Last 5 games table (+ avg row)
# 3) Last 5 vs current opponent table (+ avg row)
# 4) Opponent Allowances tiles (delta % vs league avg)
# 5) Simple projection with bounded confidence band (narrow, realistic)

import streamlit as st
import polars as pl
import nflreadpy as nfl

st.set_page_config(page_title="NFL Player Scouting Dashboard", layout="wide")

# ---------- Utils ----------
def uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def safe_groupby(df: pl.DataFrame, *keys):
    return df.group_by(*keys) if hasattr(df, "group_by") else df.groupby(*keys)

def add_standard_aliases(ps: pl.DataFrame) -> pl.DataFrame:
    alias_map = {
        "pass_att": "attempts",
        "pass_cmp": "completions",
        "pass_yds": "passing_yards",
        "pass_td": "passing_tds",
        "pass_int": "passing_interceptions",
        "rush_att": "carries",
        "rush_yds": "rushing_yards",
        "rush_td": "rushing_tds",
        "tgt": "targets",
        "rec": "receptions",
        "rec_yds": "receiving_yards",
        "rec_td": "receiving_tds",
    }
    add_cols = []
    for short, src in alias_map.items():
        if (src in ps.columns) and (short not in ps.columns):
            add_cols.append(pl.col(src).alias(short))
    return ps.with_columns(add_cols) if add_cols else ps

def last_n_with_avg(df: pl.DataFrame, n: int, sort_cols=("season","week")) -> pl.DataFrame:
    if df.is_empty():
        return df
    df_sorted = df.sort(list(sort_cols), descending=[True, True])
    recent = df_sorted.head(n)
    numeric_cols = [c for c, dt in zip(recent.columns, recent.dtypes) if pl.datatypes.is_numeric(dt)]
    avg_row = (
        recent.select([pl.mean(c).alias(c) if c in numeric_cols else pl.lit("AVG").alias(c) for c in recent.columns])
    )
    # ensure string cols show "AVG" only for first text col, blanks for others
    text_cols = [c for c in recent.columns if c not in numeric_cols]
    if text_cols:
        avg_row = avg_row.with_columns([pl.lit("").alias(c) for c in text_cols[1:]])
    return pl.concat([recent, avg_row], how="vertical_relaxed")

def opponent_allowances(player_stats: pl.DataFrame, season: int, pos: str) -> pl.DataFrame:
    df = player_stats.filter((pl.col("season") == season) & (pl.col("position") == pos))
    if df.is_empty():
        return pl.DataFrame()
    stat_pool = [
        "pass_att","pass_cmp","pass_yds","pass_td","pass_int",
        "rush_att","rush_yds","rush_td",
        "tgt","rec","rec_yds","rec_td",
        "fantasy_points_ppr",
    ]
    use_stats = [c for c in stat_pool if c in df.columns]
    if not use_stats:
        return pl.DataFrame()
    by_opp = (
        df.select(["opponent_team"] + use_stats)
          .pipe(lambda d: safe_groupby(d, "opponent_team")
                .agg([pl.col(c).mean().alias(f"opp_mean_{c}") for c in use_stats]))
    )
    league = pl.DataFrame({f"lg_mean_{c}": [df.select(pl.mean(c)).item()] for c in use_stats})
    allowances = by_opp.join(league, how="cross")
    for c in use_stats:
        allowances = allowances.with_columns(
            pl.when((pl.col(f"lg_mean_{c}").is_not_null()) & (pl.col(f"lg_mean_{c}") != 0))
              .then((pl.col(f"opp_mean_{c}") / pl.col(f"lg_mean_{c}") - 1.0) * 100.0)
              .otherwise(None)
              .alias(f"delta_pct_{c}")
        )
    keep = ["opponent_team"] + [c for c in allowances.columns if c.startswith("delta_pct_")]
    return allowances.select(keep).sort("opponent_team")

def vs_opponent(df_logs: pl.DataFrame, opp: str) -> pl.DataFrame:
    if df_logs.is_empty():
        return df_logs
    return df_logs.filter(pl.col("opponent_team") == opp)

def stat_volatility(df: pl.DataFrame, stat_cols: list[str]) -> dict:
    # rolling-ish proxy: std over the last 5 (already sliced before calling)
    vol = {}
    for c in stat_cols:
        if c in df.columns:
            vals = df[c].to_list()
            vals = [v for v in vals if isinstance(v, (int, float))]  # ignore nulls
            vol[c] = (pl.Series(vals).std() if len(vals) >= 2 else 0.0) or 0.0
    return vol

def bounded_band(mean_val: float, vol: float, lo_floor: float = 0.0, cap_mult: float = 2.0):
    # keep bands realistic: mean ± (cap_mult * vol), lower bounded at 0 (or lo_floor)
    low = max(lo_floor, mean_val - cap_mult * vol)
    high = max(low, mean_val + cap_mult * vol)
    return low, high

# ---------- Data (cache) ----------
@st.cache_data(show_spinner=False)
def load_all():
    try:
        season = nfl.get_current_season()
    except Exception:
        season = 2025
    players = nfl.load_players()
    schedules = nfl.load_schedules(season)
    player_stats = add_standard_aliases(nfl.load_player_stats(season))
    rosters_weekly = nfl.load_rosters_weekly(season)

    # Player index (QB/RB/WR/TE)
    pos_keep = {"QB","RB","WR","TE"}
    player_index = (
        rosters_weekly
        .filter((pl.col("season") == season) & pl.col("position").is_in(list(pos_keep)))
        .select([
            pl.col("gsis_id").alias("player_id"),
            pl.col("full_name").alias("player_name"),
            pl.col("team"),
            pl.col("position"),
        ])
        .unique(subset=["player_id"], keep="last")
        .sort(["team","position","player_name"])
    )
    teams = sorted(player_index["team"].unique().to_list())
    return season, players, schedules, player_stats, player_index, teams

season, players, schedules, player_stats, player_index, teams = load_all()

st.title("NFL Player Scouting Dashboard")

# ---------- Sidebar: selection ----------
with st.sidebar:
    st.subheader("Filters")
    team = st.selectbox("Team", ["All"] + teams)
    pos_options = ["QB","RB","WR","TE"]
    position = st.selectbox("Position", pos_options)

    pi = player_index.filter(pl.col("position") == position)
    if team != "All":
        pi = pi.filter(pl.col("team") == team)

    player_names = pi["player_name"].to_list() if not pi.is_empty() else []
    player_name = st.selectbox("Player", player_names)
    current_opp = None
    # infer next game opponent from schedule if available
    if team != "All":
        sched_team = schedules.filter((pl.col("season") == season) & ((pl.col("home_team") == team) | (pl.col("away_team") == team)))
        # pick nearest week >= current min existing in stats; else fallback week 1
        if not sched_team.is_empty():
            # naive guess: choose the first upcoming week with missing score
            next_row = sched_team.filter((pl.col("home_score").is_null()) & (pl.col("away_score").is_null()))
            if next_row.is_empty():
                next_row = sched_team.head(1)
            if not next_row.is_empty():
                row = next_row.head(1)
                ht, at = row["home_team"][0], row["away_team"][0]
                current_opp = at if ht == team else ht

st.caption(f"Season: {season}")

# ---------- Main panels ----------
if not player_name:
    st.info("Select a position and player to begin.")
    st.stop()

sel_player = player_index.filter(pl.col("player_name") == player_name).head(1)
player_id = sel_player["player_id"][0]
team_sel = sel_player["team"][0]
pos_sel = sel_player["position"][0]

st.markdown(f"### {player_name} ({pos_sel}, {team_sel})")

# Player logs for the current & previous season
def get_player_logs(player_id: str, years: list[int]) -> pl.DataFrame:
    frames = []
    for y in years:
        try:
            df = add_standard_aliases(nfl.load_player_stats(y))
            df = df.filter(pl.col("player_id") == player_id)
            if not df.is_empty():
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pl.DataFrame()
    logs = pl.concat(frames, how="vertical_relaxed")
    base = ["player_id","player_name","season","week","season_type","team","opponent_team","position"]
    stat_candidates = [
        "pass_att","pass_cmp","pass_yds","pass_td","pass_int",
        "rush_att","rush_yds","rush_td",
        "tgt","rec","rec_yds","rec_td",
        "fantasy_points","fantasy_points_ppr",
    ]
    present = [c for c in stat_candidates if c in logs.columns]
    sel = uniq(base + present)
    missing = [c for c in stat_candidates if c not in logs.columns]
    if missing:
        logs = logs.with_columns([pl.lit(None).alias(c) for c in missing])
    return logs.select(sel).sort(["season","week"], descending=[True, True])

logs = get_player_logs(player_id, [season-1, season])

cols_show = [
    "season","week","team","opponent_team",
    "pass_att","pass_cmp","pass_yds","pass_td","pass_int",
    "rush_att","rush_yds","rush_td",
    "tgt","rec","rec_yds","rec_td",
    "fantasy_points_ppr"
]
cols_show = [c for c in cols_show if c in logs.columns]

# 1) Last 5 games (+ avg)
st.subheader("Last 5 Games")
last5 = last_n_with_avg(logs.select(cols_show), 5)
st.dataframe(last5.to_pandas(), use_container_width=True)

# 2) Last 5 vs Opponent (+ avg)
if current_opp:
    st.subheader(f"Last 5 vs {current_opp}")
    logs_vs = vs_opponent(logs.select(cols_show + ["opponent_team"]), current_opp)
    last5_vs = last_n_with_avg(logs_vs.select(cols_show), 5)
    st.dataframe(last5_vs.to_pandas(), use_container_width=True)
else:
    st.subheader("Last 5 vs Opponent")
    st.caption("Opponent not inferred yet from schedule. (Select a team to try to infer.)")

# 3) Opponent Allowances tiles
st.subheader("Opponent Allowances (Δ% vs League Avg)")
oa = opponent_allowances(player_stats, season, pos_sel)
if current_opp and not oa.is_empty():
    row = oa.filter(pl.col("opponent_team") == current_opp)
    if not row.is_empty():
        # choose a handful of stats to display
        tile_stats = [c for c in row.columns if c.startswith("delta_pct_")]
        # prioritize position-relevant
        prefer = ["delta_pct_pass_att","delta_pct_pass_yds","delta_pct_pass_td",
                  "delta_pct_rush_att","delta_pct_rush_yds","delta_pct_rush_td",
                  "delta_pct_tgt","delta_pct_rec","delta_pct_rec_yds","delta_pct_rec_td",
                  "delta_pct_fantasy_points_ppr"]
        order = [c for c in prefer if c in tile_stats] + [c for c in tile_stats if c not in prefer]
        order = order[:8]  # limit tiles
        cols = st.columns(len(order)) if len(order) > 0 else []
        for i, c in enumerate(order):
            with cols[i]:
                val = float(row[c][0]) if row[c][0] is not None else 0.0
                st.metric(c.replace("delta_pct_","").upper(), f"{val:+.1f}%")
    else:
        st.caption("No opponent allowance row found.")
else:
    st.caption("Allowances need an inferred opponent and player stats present.")

# 4) Simple Projection (narrow, realistic band)
st.subheader("Projection (Simple, Opponent-Adjusted)")

# pick core stat family based on position
if pos_sel == "QB":
    core_stats = ["pass_att","pass_yds","pass_td","pass_int","rush_att","rush_yds","fantasy_points_ppr"]
elif pos_sel == "RB":
    core_stats = ["rush_att","rush_yds","rush_td","rec","rec_yds","fantasy_points_ppr"]
elif pos_sel == "WR":
    core_stats = ["tgt","rec","rec_yds","rec_td","fantasy_points_ppr"]
elif pos_sel == "TE":
    core_stats = ["tgt","rec","rec_yds","rec_td","fantasy_points_ppr"]
else:
    core_stats = ["fantasy_points_ppr"]

stats_present = [c for c in core_stats if c in logs.columns]
recent = logs.select(["season","week"] + stats_present).sort(["season","week"], descending=[True, True]).head(5)

if recent.is_empty():
    st.info("No recent logs available for projection.")
else:
    # base means
    means = {c: float(recent[c].mean()) for c in stats_present}

    # volatility (std over last 5)
    vol = stat_volatility(recent, stats_present)

    # opponent adjustment (use fantasy_points_ppr as broad proxy if specific not present)
    adj_map = {}
    if current_opp and not oa.is_empty():
        row = oa.filter(pl.col("opponent_team") == current_opp)
        if not row.is_empty():
            for c in stats_present:
                key = f"delta_pct_{c}"
                if key in row.columns and row[key][0] is not None:
                    adj_map[c] = 1.0 + float(row[key][0]) / 100.0
                else:
                    # fallback to FPPG delta if specific stat not there
                    if "delta_pct_fantasy_points_ppr" in row.columns and row["delta_pct_fantasy_points_ppr"][0] is not None:
                        adj_map[c] = 1.0 + float(row["delta_pct_fantasy_points_ppr"][0]) / 100.0

    # apply opponent adj (light touch: 50% weight to avoid overfitting)
    opp_weight = 0.5
    proj = {}
    for c in stats_present:
        base = means[c]
        mult = adj_map.get(c, 1.0)
        effective_mult = 1.0 + (mult - 1.0) * opp_weight
        proj[c] = max(0.0, base * effective_mult)

    # bounded confidence bands (narrow; cap_mult = 1.2), floor at 0
    cap_mult = 1.2
    rows = []
    for c in stats_present:
        low, high = bounded_band(proj[c], vol.get(c, 0.0), lo_floor=0.0, cap_mult=cap_mult)
        rows.append((c, proj[c], low, high, vol.get(c, 0.0)))

    proj_df = pl.DataFrame(rows, schema=["stat","proj_mean","low","high","volatility"])
    st.dataframe(proj_df.to_pandas(), use_container_width=True)
    st.caption("Band = mean ± 1.2×volatility (floored at 0). 50% opponent-adjustment weight to keep things stable.")
