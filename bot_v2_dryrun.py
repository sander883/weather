#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_v2_dryrun.py — Dry-Run Version of WeatherBet
==================================================
Simulates what the bot WOULD do without persisting anything:
  - No writes to data/state.json
  - No writes to data/markets/*.json
  - No writes to data/calibration.json
  - Single scan cycle then exit

Useful for:
  - Testing API connectivity
  - Verifying forecast sources work
  - Previewing trade decisions before going live
  - Debugging without polluting real state

Usage:
    python bot_v2_dryrun.py              # Scan all cities once
    python bot_v2_dryrun.py --city nyc   # Scan one city only
    python bot_v2_dryrun.py --limit 3    # Scan first N cities only
"""

import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import bot_v2


# =============================================================================
# DRY-RUN OVERRIDES
# =============================================================================

_dry_markets = {}   # in-memory market store: {(city, date): market_dict}
_dry_state = None   # in-memory state
_dry_stats = {
    "would_open": [],
    "would_close": [],
    "would_resolve": [],
    "skipped_filters": [],
    "forecasts_fetched": 0,
    "markets_seen": 0,
}


def dry_save_market(market):
    """In-memory only, no disk writes."""
    _dry_markets[(market["city"], market["date"])] = market


def dry_load_market(city_slug, date_str):
    """Check in-memory first, then fallback to real disk (read-only)."""
    key = (city_slug, date_str)
    if key in _dry_markets:
        return _dry_markets[key]
    return bot_v2._orig_load_market(city_slug, date_str)


def dry_load_all_markets():
    """Merge in-memory with disk markets (read-only from disk)."""
    disk = bot_v2._orig_load_all_markets()
    # Replace any disk markets that were updated in memory
    keys_in_mem = set(_dry_markets.keys())
    merged = [m for m in disk if (m["city"], m["date"]) not in keys_in_mem]
    merged.extend(_dry_markets.values())
    return merged


def dry_save_state(state):
    """In-memory only."""
    global _dry_state
    _dry_state = state


def dry_load_state():
    """Return in-memory copy if set, else start fresh."""
    global _dry_state
    if _dry_state is not None:
        return _dry_state
    _dry_state = {
        "balance":          bot_v2.BALANCE,
        "starting_balance": bot_v2.BALANCE,
        "total_trades":     0,
        "wins":             0,
        "losses":           0,
        "peak_balance":     bot_v2.BALANCE,
    }
    return _dry_state


def dry_run_calibration(markets):
    """Skip calibration writes — just return existing calibration."""
    return bot_v2.load_cal()


# =============================================================================
# PATCHED CORE FUNCTIONS (with tracking)
# =============================================================================

_orig_scan_and_update = bot_v2.scan_and_update


def patched_scan_and_update(cities_filter=None):
    """
    Instrumented version that tracks decisions without persisting.
    Mostly copies logic from bot_v2.scan_and_update but with filters.
    """
    now      = datetime.now(timezone.utc)
    state    = dry_load_state()
    balance  = state["balance"]
    new_pos  = 0
    closed   = 0
    resolved = 0

    locations = bot_v2.LOCATIONS
    if cities_filter:
        locations = {k: v for k, v in locations.items() if k in cities_filter}

    for city_slug, loc in locations.items():
        unit = loc["unit"]
        unit_sym = "F" if unit == "F" else "C"
        print(f"  -> {loc['name']}...", end=" ", flush=True)

        try:
            dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
            snapshots = bot_v2.take_forecast_snapshot(city_slug, dates)
            _dry_stats["forecasts_fetched"] += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"skipped ({e})")
            continue

        for i, date in enumerate(dates):
            dt    = datetime.strptime(date, "%Y-%m-%d")
            event = bot_v2.get_polymarket_event(
                city_slug, bot_v2.MONTHS[dt.month - 1], dt.day, dt.year
            )
            if not event:
                continue

            _dry_stats["markets_seen"] += 1
            end_date = event.get("endDate", "")
            hours    = bot_v2.hours_to_resolution(end_date) if end_date else 0
            horizon  = f"D+{i}"

            mkt = dry_load_market(city_slug, date)
            if mkt is None:
                if hours < bot_v2.MIN_HOURS or hours > bot_v2.MAX_HOURS:
                    _dry_stats["skipped_filters"].append(
                        f"{loc['name']} {date}: hours={hours:.1f}h (need {bot_v2.MIN_HOURS}-{bot_v2.MAX_HOURS})"
                    )
                    continue
                mkt = bot_v2.new_market(city_slug, date, event, hours)

            if mkt["status"] == "resolved":
                continue

            # Parse outcomes
            outcomes = []
            for market in event.get("markets", []):
                question = market.get("question", "")
                mid      = str(market.get("id", ""))
                volume   = float(market.get("volume", 0))
                rng      = bot_v2.parse_temp_range(question)
                if not rng:
                    continue
                try:
                    prices = json.loads(market.get("outcomePrices", "[0.5,0.5]"))
                    bid = float(prices[0])
                    ask = float(prices[1]) if len(prices) > 1 else bid
                except Exception:
                    continue
                outcomes.append({
                    "question":  question,
                    "market_id": mid,
                    "range":     rng,
                    "bid":       round(bid, 4),
                    "ask":       round(ask, 4),
                    "price":     round(bid, 4),
                    "spread":    round(ask - bid, 4),
                    "volume":    round(volume, 0),
                })

            outcomes.sort(key=lambda x: x["range"][0])
            mkt["all_outcomes"] = outcomes

            snap = snapshots.get(date, {})
            mkt["forecast_snapshots"].append({
                "ts":          snap.get("ts"),
                "horizon":     horizon,
                "hours_left":  round(hours, 1),
                "ecmwf":       snap.get("ecmwf"),
                "hrrr":        snap.get("hrrr"),
                "metar":       snap.get("metar"),
                "best":        snap.get("best"),
                "best_source": snap.get("best_source"),
            })

            forecast_temp = snap.get("best")
            best_source   = snap.get("best_source")

            # --- SIMULATE OPEN DECISION ---
            if not mkt.get("position") and forecast_temp is not None and hours >= bot_v2.MIN_HOURS:
                sigma = bot_v2.get_sigma(city_slug, best_source or "ecmwf")
                matched = None
                for o in outcomes:
                    t_low, t_high = o["range"]
                    if bot_v2.in_bucket(forecast_temp, t_low, t_high):
                        matched = o
                        break

                if matched:
                    o = matched
                    t_low, t_high = o["range"]
                    volume = o["volume"]
                    ask    = o.get("ask", o["price"])
                    spread = o.get("spread", 0)

                    reasons = []
                    if volume < bot_v2.MIN_VOLUME:
                        reasons.append(f"volume ${volume:.0f} < ${bot_v2.MIN_VOLUME}")

                    p  = bot_v2.bucket_prob(forecast_temp, t_low, t_high, sigma)
                    ev = bot_v2.calc_ev(p, ask)
                    if ev < bot_v2.MIN_EV:
                        reasons.append(f"EV {ev:+.3f} < {bot_v2.MIN_EV}")

                    if ask >= bot_v2.MAX_PRICE:
                        reasons.append(f"ask ${ask:.3f} >= ${bot_v2.MAX_PRICE}")

                    if spread > bot_v2.MAX_SLIPPAGE:
                        reasons.append(f"spread ${spread:.3f} > ${bot_v2.MAX_SLIPPAGE}")

                    kelly = bot_v2.calc_kelly(p, ask)
                    size  = bot_v2.bet_size(kelly, balance)
                    if size < 0.50:
                        reasons.append(f"size ${size:.2f} < $0.50")

                    if reasons:
                        _dry_stats["skipped_filters"].append(
                            f"{loc['name']} {date} {t_low}-{t_high}{unit_sym}: {', '.join(reasons)}"
                        )
                    else:
                        bucket_label = f"{t_low}-{t_high}{unit_sym}"
                        _dry_stats["would_open"].append({
                            "city": loc["name"],
                            "date": date,
                            "horizon": horizon,
                            "bucket": bucket_label,
                            "ask": ask,
                            "ev": ev,
                            "p": p,
                            "size": size,
                            "kelly": kelly,
                            "forecast": forecast_temp,
                            "source": best_source,
                        })
                        new_pos += 1
                        print(f"\n  [WOULD BUY] {loc['name']} {horizon} {date} | {bucket_label} | "
                              f"${ask:.3f} | EV {ev:+.2f} | ${size:.2f} ({best_source.upper()})", end="")

            dry_save_market(mkt)
            time.sleep(0.1)

        print(" ok")

    return new_pos, closed, resolved


# =============================================================================
# MAIN
# =============================================================================

def print_dry_summary():
    """Print summary of what would have happened."""
    print(f"\n{'='*60}")
    print(f"  DRY-RUN SUMMARY")
    print(f"{'='*60}")
    print(f"  Forecasts fetched:   {_dry_stats['forecasts_fetched']}")
    print(f"  Polymarket events:   {_dry_stats['markets_seen']}")
    print(f"  Would open:          {len(_dry_stats['would_open'])}")
    print(f"  Filtered out:        {len(_dry_stats['skipped_filters'])}")

    if _dry_stats["would_open"]:
        print(f"\n  --- TRADES BOT WOULD OPEN ---")
        total_cost = 0.0
        for t in _dry_stats["would_open"]:
            total_cost += t["size"]
            print(f"    {t['city']:<16} {t['date']} {t['horizon']:<4} | "
                  f"{t['bucket']:<12} | forecast {t['forecast']} ({t['source']}) | "
                  f"p={t['p']:.2f} ask=${t['ask']:.3f} EV={t['ev']:+.2f} | "
                  f"size=${t['size']:.2f}")
        print(f"\n  Total capital allocated: ${total_cost:.2f}")

    if _dry_stats["skipped_filters"]:
        print(f"\n  --- FIRST 10 FILTERED OUT ---")
        for reason in _dry_stats["skipped_filters"][:10]:
            print(f"    {reason}")
        if len(_dry_stats["skipped_filters"]) > 10:
            print(f"    ... and {len(_dry_stats['skipped_filters']) - 10} more")

    print(f"\n  NOTE: No state was written to disk. Run `python bot_v2.py` to go live.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Dry-run mode for WeatherBet")
    parser.add_argument("--city", help="Scan only a specific city slug (e.g. nyc, chicago)")
    parser.add_argument("--limit", type=int, help="Scan only the first N cities")
    args = parser.parse_args()

    # Save originals so dry-run wrappers can delegate for reads
    bot_v2._orig_load_market = bot_v2.load_market
    bot_v2._orig_load_all_markets = bot_v2.load_all_markets

    # Patch bot_v2 to skip all persistence
    bot_v2.save_market      = dry_save_market
    bot_v2.load_market      = dry_load_market
    bot_v2.load_all_markets = dry_load_all_markets
    bot_v2.save_state       = dry_save_state
    bot_v2.load_state       = dry_load_state
    bot_v2.run_calibration  = dry_run_calibration

    # Filter cities
    cities_filter = None
    if args.city:
        if args.city not in bot_v2.LOCATIONS:
            print(f"Unknown city: {args.city}")
            print(f"Available: {', '.join(bot_v2.LOCATIONS.keys())}")
            sys.exit(1)
        cities_filter = [args.city]
    elif args.limit:
        cities_filter = list(bot_v2.LOCATIONS.keys())[:args.limit]

    print(f"\n{'='*60}")
    print(f"  WEATHERBET — DRY RUN MODE")
    print(f"{'='*60}")
    print(f"  Cities:       {len(cities_filter) if cities_filter else len(bot_v2.LOCATIONS)}")
    print(f"  Balance:      ${bot_v2.BALANCE:,.0f} (simulated, no writes)")
    print(f"  Min EV:       {bot_v2.MIN_EV}")
    print(f"  Max price:    ${bot_v2.MAX_PRICE}")
    print(f"  Max bet:      ${bot_v2.MAX_BET}")
    print(f"  Data mode:    IN-MEMORY (nothing persisted)")
    print(f"{'='*60}\n")

    bot_v2._cal = bot_v2.load_cal()

    try:
        patched_scan_and_update(cities_filter=cities_filter)
    except KeyboardInterrupt:
        print("\n  Interrupted.")

    print_dry_summary()


if __name__ == "__main__":
    main()
