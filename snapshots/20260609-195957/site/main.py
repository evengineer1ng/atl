from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import math
import py_compile
import re
import shutil
import sqlite3
import subprocess
import textwrap
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import ccxt
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_ROOT_DIR = DATA_DIR / "archives" / "league_snapshots"
DEFAULT_ARCHIVE_REPO_DIR = DATA_DIR / "archive_repo"
ML_RUN_ARTIFACT_DIR = DATA_DIR / "ml_runs"
DB_PATH = DATA_DIR / "league.sqlite"
INSTANCE_REGISTRY_PATH = DATA_DIR / "instances.json"
QUESTIONS_PATH = DATA_DIR / "research_questions.json"
ML_HYPOTHESES_PATH = DATA_DIR / "ml_hypotheses.json"
ML_BUCKETS_PATH = DATA_DIR / "ml_buckets.json"
ML_FEATURES_PATH = DATA_DIR / "ml_features.json"
ML_MODELS_PATH = DATA_DIR / "ml_models.json"
ML_DRAFT_BOARD_PATH = DATA_DIR / "ml_draft_board.json"
ML_PROMOTIONS_PATH = DATA_DIR / "ml_promotions.json"
BACKTEST_DIR = PROJECT_DIR / "user_data" / "backtest_results"
UNIVERSE_HISTORY_DIR = PROJECT_DIR / "user_data" / "universe_history"
BACKTEST_STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies" / "backtest"
SENTIMENT_DATA_DIR = PROJECT_DIR / "user_data" / "data" / "sentiment"
DEV_STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies" / "development"
DEV_CONFIG_DIR = PROJECT_DIR / "user_data" / "configs" / "development"
DEV_DATABASE_DIR = PROJECT_DIR / "user_data" / "databases" / "development"
DEV_LOG_DIR = PROJECT_DIR / "user_data" / "logs" / "development"
DEV_SCRIPT_DIR = PROJECT_DIR / "scripts" / "development"
DEV_RUNTIME_DIR = PROJECT_DIR / "user_data"
POLL_INTERVAL_SECONDS = 120
DEV_SCHEDULER_INTERVAL_SECONDS = 60
DEV_GENERATION_INTERVAL_SECONDS = 5
SIX_HOUR_SHIFT_CAPACITY = 6
SEASON_DRAFT_SLOT_LIMIT = 2
LOCAL_TIMEZONE = ZoneInfo("America/New_York")

DEV_TIER_LABELS = {
    "draft_room": "Draft Room",
    "bootcamp": "Bootcamp",
    "six_hour": "Six-Hour Candidate",
    "twelve_hour": "Twelve-Hour Prospect",
    "draft_eligible": "Draft Eligible",
    "drafted": "Drafted",
    "archived": "Cut / Archived",
}

DEV_LIFECYCLE_LABELS = {
    "draft_idea": "Draft Idea",
    "generating_strategy": "Generating Strategy",
    "implemented": "Implemented",
    "reviewed": "Reviewed",
    "instance_assembled": "Instance Assembled",
    "assigned_to_shift": "Assigned to Shift",
    "bootcamp": "Bootcamp",
    "six_hour_candidate": "Six-Hour Candidate",
    "twelve_hour_prospect": "Twelve-Hour Prospect",
    "promoted": "Promoted",
    "draft_eligible": "Draft Eligible",
    "drafted": "Drafted",
    "cut_archived": "Cut / Archived",
}

DEV_RUNTIME_STATUSES = {"running", "off-shift", "failed", "paused", "archived"}
PROJECTION_HIDE_RUNTIME_HOURS = 3.0
PROJECTION_STRONG_WARNING_RUNTIME_HOURS = 10.0
PROJECTION_MIN_CLOSED_TRADES = 5
DEV_SHIFT_WINDOWS = {
    "six_hour": [
        {"code": "A", "label": "Shift A", "start_hour": 0, "end_hour": 6},
        {"code": "B", "label": "Shift B", "start_hour": 6, "end_hour": 12},
        {"code": "C", "label": "Shift C", "start_hour": 12, "end_hour": 18},
        {"code": "D", "label": "Shift D", "start_hour": 18, "end_hour": 24},
    ],
    "twelve_hour": [
        {"code": "A", "label": "Shift A", "start_hour": 0, "end_hour": 12},
        {"code": "B", "label": "Shift B", "start_hour": 12, "end_hour": 24},
    ],
}

POST_SHIFT_STYLE_PROFILES = {
    "aggressive": {"ideal_pace": 8.0, "min_pace": 4.0, "max_pace": 16.0, "review_floor_hours": 3.0, "min_closed_trades": 4},
    "balanced": {"ideal_pace": 4.0, "min_pace": 1.5, "max_pace": 8.0, "review_floor_hours": 4.0, "min_closed_trades": 2},
    "patient": {"ideal_pace": 1.0, "min_pace": 0.0, "max_pace": 3.0, "review_floor_hours": 6.0, "min_closed_trades": 0},
}
POST_SHIFT_GRADE_BANDS = (
    (85.0, "A"),
    (70.0, "B"),
    (55.0, "C"),
    (40.0, "D"),
)
# Exit-reason families used by the clinical post-shift diagnostics. Anything not in
# STOP/ROI/TIME is treated as a strategy-authored custom-signal exit (e.g. a divergence
# revert), which is graded on its own realized PnL.
POST_SHIFT_STOP_EXITS = {"stop_loss", "stoploss", "trailing_stop_loss", "liquidation"}
POST_SHIFT_ROI_EXITS = {"roi"}
POST_SHIFT_TIME_EXITS = {"exit_signal", "timeout", "expired"}
POST_SHIFT_SEVERITY_WEIGHT = {"critical": 3, "warning": 2, "info": 1, "good": 0}
DEV_EXCHANGE_PROFILES = (
    {
        "name": "binance",
        "stake_currency": "USDC",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "rotation_bias": 0,
        "ccxt_options": {
            "defaultType": "swap",
            "defaultSettle": "USDC",
        },
    },
    {
        "name": "bitget",
        "stake_currency": "USDT",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "rotation_bias": 0,
        "ccxt_options": {
            "defaultType": "swap",
            "defaultSettle": "USDT",
        },
    },
    {
        "name": "bybit",
        "stake_currency": "USDT",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "rotation_bias": 0,
        "ccxt_options": {
            "defaultType": "swap",
            "defaultSettle": "USDT",
        },
    },
    {
        "name": "okx",
        "stake_currency": "USDT",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "rotation_bias": 0,
        "ccxt_options": {
            "defaultType": "swap",
            "defaultSettle": "USDT",
        },
    },
    {
        "name": "hyperliquid",
        "stake_currency": "USDC",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "rotation_bias": 3,
        "ccxt_options": {
            # Hyperliquid's ccxt driver REQUIRES fetchMarkets.types to restrict
            # loading to swap. Without it, it also fetches spot markets, which
            # crashes (NoneType base) in ccxt 4.5.x. (Binance is the opposite:
            # it rejects fetchMarkets.types entirely — keep it off there.)
            "fetchMarkets": {"types": ["swap"]},
            "defaultType": "swap",
            "defaultSettle": "USDC",
            "hip3TokensByName": {},
            "cachedCurrenciesById": {},
        },
    },
)
SEASON_GRADE_BANDS = (
    (90.0, "A"),
    (78.0, "B"),
    (62.0, "C"),
    (45.0, "D"),
)
SEASON_DECISION_LABELS = {
    "hold": "Hold",
    "update": "Update",
    "tweak": "Tweak",
    "revamp": "Revamp",
    "relegate": "Relegate",
}


app = FastAPI(title="Algo Trading League")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
sync_lock = threading.Lock()


@dataclass
class TeamLiveState:
    team_id: str
    ts: str
    status: str
    status_detail: str
    bot_name: str | None
    strategy_name: str | None
    strategy_version: str | None
    current_record: str | None
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    trade_count: int
    closed_trade_count: int
    win_rate: float
    avg_roi: float
    max_drawdown: float
    current_drawdown: float
    best_pair: str | None
    best_rate: float | None
    last_trade_at: str | None
    bot_start_at: str | None
    open_trade_count: int
    heartbeat_ok: int


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(get_db()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS live_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                status TEXT NOT NULL,
                status_detail TEXT,
                bot_name TEXT,
                strategy_name TEXT,
                strategy_version TEXT,
                current_record TEXT,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                closed_trade_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_roi REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                current_drawdown REAL NOT NULL,
                best_pair TEXT,
                best_rate REAL,
                last_trade_at TEXT,
                bot_start_at TEXT,
                open_trade_count INTEGER NOT NULL,
                heartbeat_ok INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS team_trades (
                team_id TEXT NOT NULL,
                source_trade_id INTEGER NOT NULL,
                pair TEXT NOT NULL,
                exchange_name TEXT,
                is_open INTEGER NOT NULL,
                is_short INTEGER NOT NULL,
                stake_amount REAL,
                amount REAL,
                leverage REAL,
                open_rate REAL,
                close_rate REAL,
                profit_ratio REAL,
                profit_pct REAL,
                profit_abs REAL,
                realized_profit REAL,
                exit_reason TEXT,
                enter_tag TEXT,
                open_date TEXT,
                close_date TEXT,
                trade_duration_minutes REAL,
                max_rate REAL,
                min_rate REAL,
                strategy_name TEXT,
                source_db_path TEXT,
                last_synced_at TEXT NOT NULL,
                PRIMARY KEY (team_id, source_trade_id)
            );

            CREATE TABLE IF NOT EXISTS timeline_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'league',
                title TEXT NOT NULL,
                team_tags TEXT NOT NULL,
                observation TEXT NOT NULL,
                evidence TEXT NOT NULL,
                interpretation TEXT NOT NULL,
                next_action TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_generated_content (
                section TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_research_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                question TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                hypothesis_id TEXT,
                status TEXT NOT NULL,
                source_question TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS maintenance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                maintenance_type TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                owner TEXT NOT NULL,
                interval_minutes INTEGER NOT NULL,
                duration_hours INTEGER NOT NULL,
                auto_reseed INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT NOT NULL,
                next_run_at TEXT NOT NULL,
                completed_at TEXT,
                summary TEXT NOT NULL DEFAULT '',
                latest_focus TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS research_thread_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                update_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT NOT NULL,
                source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_index_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_key TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_dataset_registry (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_path TEXT NOT NULL,
                timeframe TEXT,
                coverage TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_label_registry (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                label_name TEXT NOT NULL,
                target_variable TEXT NOT NULL,
                horizon_candles INTEGER,
                leakage_risk TEXT NOT NULL,
                live_safe INTEGER NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_model_registry (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                hypothesis_id TEXT,
                dataset_id TEXT,
                label_id TEXT,
                algorithm_type TEXT NOT NULL,
                feature_count INTEGER NOT NULL,
                training_date TEXT,
                train_window TEXT,
                validation_window TEXT,
                metrics TEXT NOT NULL,
                artifact_path TEXT,
                artifact_exists INTEGER NOT NULL,
                influenced_live_strategy INTEGER NOT NULL,
                lineage_status TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_feature_set_versions (
                id TEXT PRIMARY KEY,
                feature_set_name TEXT NOT NULL,
                hypothesis_id TEXT,
                feature_names TEXT NOT NULL,
                leakage_notes TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_label_spec_versions (
                id TEXT PRIMARY KEY,
                label_name TEXT NOT NULL,
                hypothesis_id TEXT,
                target_variable TEXT NOT NULL,
                horizon_candles INTEGER,
                leakage_risk TEXT NOT NULL,
                live_safe INTEGER NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_experiment_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_key TEXT NOT NULL,
                thread_id INTEGER,
                title TEXT NOT NULL,
                lead_question TEXT NOT NULL,
                rationale TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_agent TEXT NOT NULL,
                resolution TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id INTEGER,
                run_slug TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                objective TEXT NOT NULL,
                dataset_id TEXT,
                feature_set_version_id TEXT,
                label_spec_version_id TEXT,
                hypothesis_id TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                summary TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                notes TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_bucket_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_run_id INTEGER,
                candidate_name TEXT NOT NULL,
                hypothesis_id TEXT,
                feature_conditions TEXT NOT NULL,
                expected_behavior TEXT NOT NULL,
                evidence_quality TEXT NOT NULL,
                contamination_risk TEXT NOT NULL,
                status TEXT NOT NULL,
                next_action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_validation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_run_id INTEGER,
                report_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                contamination_checks TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_promotion_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_run_id INTEGER,
                candidate_name TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                rationale TEXT NOT NULL,
                blockers TEXT NOT NULL,
                target_surface TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dev_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL DEFAULT 'draft_idea',
                tier TEXT NOT NULL DEFAULT 'draft_room',
                shift_code TEXT NOT NULL DEFAULT '',
                runtime_window TEXT NOT NULL DEFAULT '',
                runtime_status TEXT NOT NULL DEFAULT 'paused',
                status_detail TEXT NOT NULL DEFAULT '',
                override_mode TEXT NOT NULL DEFAULT 'auto',
                hypothesis TEXT NOT NULL DEFAULT '',
                strategy_notes TEXT NOT NULL DEFAULT '',
                long_short_mode TEXT NOT NULL DEFAULT 'both',
                expected_behavior TEXT NOT NULL DEFAULT '',
                risk_profile TEXT NOT NULL DEFAULT '',
                coin_universe TEXT NOT NULL DEFAULT '',
                timeframe TEXT NOT NULL DEFAULT '',
                eligibility_status TEXT NOT NULL DEFAULT 'not_ready',
                data_quality TEXT NOT NULL DEFAULT 'unknown',
                heartbeat_ok INTEGER NOT NULL DEFAULT 0,
                heartbeat_checked_at TEXT,
                equity REAL NOT NULL DEFAULT 0,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                worst_open_trade REAL NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                last_trade_at TEXT,
                notes TEXT NOT NULL DEFAULT '',
                start_command TEXT NOT NULL DEFAULT '',
                stop_command TEXT NOT NULL DEFAULT '',
                api_url TEXT NOT NULL DEFAULT '',
                api_username TEXT NOT NULL DEFAULT '',
                api_password TEXT NOT NULL DEFAULT '',
                db_path TEXT NOT NULL DEFAULT '',
                config_path TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                strategy_path TEXT NOT NULL DEFAULT '',
                last_start_at TEXT,
                last_stop_at TEXT,
                uptime_seconds INTEGER NOT NULL DEFAULT 0,
                total_runtime_minutes INTEGER NOT NULL DEFAULT 0,
                meaningful_runtime_minutes INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT
            );

            CREATE TABLE IF NOT EXISTS dev_runtime_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                runtime_status TEXT NOT NULL,
                heartbeat_ok INTEGER NOT NULL,
                data_quality TEXT NOT NULL,
                equity REAL NOT NULL DEFAULT 0,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                worst_open_trade REAL NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                last_trade_at TEXT,
                status_detail TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS dev_runtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS dev_runtime_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                duration_hours REAL NOT NULL DEFAULT 0,
                stop_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dev_post_shift_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_key TEXT NOT NULL UNIQUE,
                candidate_id INTEGER NOT NULL,
                review_scope TEXT NOT NULL DEFAULT 'scheduled_shift_end',
                tier TEXT NOT NULL DEFAULT '',
                shift_code TEXT NOT NULL DEFAULT '',
                strategy_style TEXT NOT NULL DEFAULT 'balanced',
                session_started_at TEXT,
                session_stopped_at TEXT NOT NULL,
                runtime_hours REAL NOT NULL DEFAULT 0,
                scheduled_hours REAL NOT NULL DEFAULT 0,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                trade_pace_per_24h REAL NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                worst_open_trade REAL NOT NULL DEFAULT 0,
                data_quality TEXT NOT NULL DEFAULT '',
                overall_score REAL NOT NULL DEFAULT 0,
                grade TEXT NOT NULL DEFAULT '',
                decision_bucket TEXT NOT NULL DEFAULT '',
                evidence_confidence TEXT NOT NULL DEFAULT 'low',
                recommendation TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                mutation_brief TEXT NOT NULL DEFAULT '',
                rubric_json TEXT NOT NULL DEFAULT '[]',
                diagnostics_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Permanent per-shift "episode" record (the mini-season evidence). Captured
            -- at shift end while the bot API is still live, BEFORE the runtime DB is wiped.
            CREATE TABLE IF NOT EXISTS dev_shift_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_key TEXT NOT NULL UNIQUE,
                candidate_id INTEGER NOT NULL,
                slug TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL DEFAULT '',
                tier TEXT NOT NULL DEFAULT '',
                shift_code TEXT NOT NULL DEFAULT '',
                strategy_version TEXT NOT NULL DEFAULT '',
                session_started_at TEXT NOT NULL,
                session_stopped_at TEXT,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                forced_exits INTEGER NOT NULL DEFAULT 0,
                forced_realized_pnl REAL NOT NULL DEFAULT 0,
                strategy_closed_trades INTEGER NOT NULL DEFAULT 0,
                strategy_wins INTEGER NOT NULL DEFAULT 0,
                strategy_win_rate REAL NOT NULL DEFAULT 0,
                strategy_avg_roi REAL NOT NULL DEFAULT 0,
                strategy_realized_pnl REAL NOT NULL DEFAULT 0,
                artifact_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            -- Permanent per-TRADE record for dev shifts (the grain the episode aggregate lacks).
            -- Captured at archive time BEFORE the runtime DB is wiped; this is the durable all-time
            -- trade history that survives the per-shift wipe. Read via dev_all_time_trades().
            CREATE TABLE IF NOT EXISTS dev_archived_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_key TEXT NOT NULL,
                candidate_id INTEGER NOT NULL,
                slug TEXT NOT NULL DEFAULT '',
                strategy_version TEXT NOT NULL DEFAULT '',
                session_started_at TEXT NOT NULL DEFAULT '',
                trade_id INTEGER NOT NULL DEFAULT 0,
                pair TEXT NOT NULL DEFAULT '',
                exit_reason TEXT NOT NULL DEFAULT '',
                enter_tag TEXT NOT NULL DEFAULT '',
                is_short INTEGER NOT NULL DEFAULT 0,
                forced INTEGER NOT NULL DEFAULT 0,
                profit_ratio REAL NOT NULL DEFAULT 0,
                profit_abs REAL NOT NULL DEFAULT 0,
                realized_profit REAL NOT NULL DEFAULT 0,
                open_date TEXT,
                close_date TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(episode_key, trade_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dev_archived_trades_slug ON dev_archived_trades (slug);

            CREATE TABLE IF NOT EXISTS league_seasons (
                season_number INTEGER PRIMARY KEY,
                season_label TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'current',
                awards_json TEXT NOT NULL DEFAULT '[]',
                draft_slots INTEGER NOT NULL DEFAULT 2,
                turnover_processed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS league_team_season_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_key TEXT NOT NULL UNIQUE,
                season_number INTEGER NOT NULL,
                season_label TEXT NOT NULL,
                season_started_at TEXT NOT NULL,
                season_ended_at TEXT NOT NULL,
                team_id TEXT NOT NULL,
                team_name TEXT NOT NULL,
                strategy_family TEXT NOT NULL DEFAULT '',
                pair_universe TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                strategy_path TEXT NOT NULL DEFAULT '',
                strategy_file_hash TEXT NOT NULL DEFAULT '',
                runtime_hours REAL NOT NULL DEFAULT 0,
                scheduled_hours REAL NOT NULL DEFAULT 0,
                heartbeat_ratio REAL NOT NULL DEFAULT 0,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                total_pnl REAL NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                worst_open_trade REAL NOT NULL DEFAULT 0,
                champion_exits INTEGER NOT NULL DEFAULT 0,
                overall_score REAL NOT NULL DEFAULT 0,
                grade TEXT NOT NULL DEFAULT '',
                decision_bucket TEXT NOT NULL DEFAULT 'hold',
                recommendation TEXT NOT NULL DEFAULT '',
                fix_suggestion TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                rubric_json TEXT NOT NULL DEFAULT '[]',
                approval_required INTEGER NOT NULL DEFAULT 0,
                approval_status TEXT NOT NULL DEFAULT 'pending',
                approval_notes TEXT NOT NULL DEFAULT '',
                approved_action TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS league_season_draft_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season_number INTEGER NOT NULL,
                season_label TEXT NOT NULL,
                season_started_at TEXT NOT NULL,
                season_ended_at TEXT NOT NULL,
                candidate_id INTEGER NOT NULL,
                candidate_name TEXT NOT NULL,
                latest_review_key TEXT NOT NULL DEFAULT '',
                candidate_tier TEXT NOT NULL DEFAULT '',
                overall_score REAL NOT NULL DEFAULT 0,
                projected_total_pnl REAL NOT NULL DEFAULT 0,
                recommendation TEXT NOT NULL DEFAULT '',
                rationale TEXT NOT NULL DEFAULT '',
                approval_status TEXT NOT NULL DEFAULT 'pending',
                approval_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(season_number, candidate_id)
            );

            -- Quarterly Champion: a formal capital-eligibility review generated once every
            -- three COMPLETED major-league seasons. Archived permanently; never silently
            -- rewritten (regeneration is an explicit admin action). This is NOT a live race.
            CREATE TABLE IF NOT EXISTS league_quarterly_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quarter_number INTEGER NOT NULL UNIQUE,
                quarter_label TEXT NOT NULL DEFAULT '',
                season_start INTEGER NOT NULL DEFAULT 0,
                season_end INTEGER NOT NULL DEFAULT 0,
                season_range_label TEXT NOT NULL DEFAULT '',
                champion_team_id TEXT NOT NULL DEFAULT '',
                champion_team_name TEXT NOT NULL DEFAULT '',
                executive_summary TEXT NOT NULL DEFAULT '',
                report_json TEXT NOT NULL DEFAULT '{}',
                team_count INTEGER NOT NULL DEFAULT 0,
                llm_used INTEGER NOT NULL DEFAULT 0,
                generated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Strategy Trophy Shelves: permanent, append-only career achievements. One
            -- winner per (award_type, season); never overwritten. A strategy's shelf is
            -- a living identity record of what KIND of organism it was, not its rank.
            CREATE TABLE IF NOT EXISTS strategy_awards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                award_key TEXT NOT NULL UNIQUE,
                award_type TEXT NOT NULL,
                award_title TEXT NOT NULL DEFAULT '',
                emoji TEXT NOT NULL DEFAULT '',
                dimension TEXT NOT NULL DEFAULT '',
                season_number INTEGER NOT NULL DEFAULT 0,
                season_label TEXT NOT NULL DEFAULT '',
                recipient_kind TEXT NOT NULL DEFAULT '',
                recipient_id TEXT NOT NULL DEFAULT '',
                recipient_name TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                metric_value REAL NOT NULL DEFAULT 0,
                awarded_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_families (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_strategy_registry (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'main',
                source_team_id TEXT NOT NULL DEFAULT '',
                source_db_path TEXT NOT NULL DEFAULT '',
                family_slug TEXT NOT NULL DEFAULT '',
                classification TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_slug TEXT NOT NULL,
                parent_slug TEXT NOT NULL,
                relationship_type TEXT NOT NULL DEFAULT 'parent',
                mechanism_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(child_slug, parent_slug)
            );

            CREATE TABLE IF NOT EXISTS ml_traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_slug TEXT NOT NULL,
                trait_name TEXT NOT NULL,
                polarity TEXT NOT NULL DEFAULT 'neutral',
                confidence REAL NOT NULL DEFAULT 0.3,
                evidence_source TEXT NOT NULL DEFAULT '',
                evidence_count INTEGER NOT NULL DEFAULT 1,
                first_observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(strategy_slug, trait_name)
            );

            CREATE TABLE IF NOT EXISTS ml_telemetry_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                strategies_count INTEGER NOT NULL DEFAULT 0,
                essay_markdown TEXT NOT NULL DEFAULT '',
                essay_json TEXT NOT NULL DEFAULT '{}',
                top_findings_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'running',
                llm_used INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS ml_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                strategy_slug TEXT NOT NULL,
                category TEXT NOT NULL,
                value REAL NOT NULL DEFAULT 0,
                sample_size INTEGER NOT NULL DEFAULT 0,
                measurable INTEGER NOT NULL DEFAULT 1,
                computed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_telemetry_divergence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                strategy_slug TEXT NOT NULL,
                category TEXT NOT NULL,
                value REAL NOT NULL DEFAULT 0,
                peer_mean REAL NOT NULL DEFAULT 0,
                peer_median REAL NOT NULL DEFAULT 0,
                peer_mad REAL NOT NULL DEFAULT 0,
                percentile REAL NOT NULL DEFAULT 0,
                robust_z REAL NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'high',
                magnitude REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS ml_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                strategy_a TEXT NOT NULL,
                strategy_b TEXT NOT NULL,
                relationship_type TEXT NOT NULL DEFAULT 'complement',
                similarity_score REAL NOT NULL DEFAULT 0,
                complement_score REAL NOT NULL DEFAULT 0,
                evidence_notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_descendant_hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                parent_a TEXT NOT NULL,
                parent_b TEXT NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                complement_score REAL NOT NULL DEFAULT 0,
                inheritance_score REAL NOT NULL DEFAULT 0,
                environmental_score REAL NOT NULL DEFAULT 0,
                novelty_score REAL NOT NULL DEFAULT 0,
                evidence_score REAL NOT NULL DEFAULT 0,
                conviction_score REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'watching',
                first_seen_cycle INTEGER NOT NULL DEFAULT 0,
                last_updated_cycle INTEGER NOT NULL DEFAULT 0,
                supporting_cycle_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ml_evolution_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                cycle_window_start INTEGER NOT NULL DEFAULT 0,
                cycle_window_end INTEGER NOT NULL DEFAULT 0,
                report_markdown TEXT NOT NULL DEFAULT '',
                report_json TEXT NOT NULL DEFAULT '{}',
                descendants_proposed INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS chronicle_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chronicle_date TEXT NOT NULL UNIQUE,
                emoji TEXT NOT NULL DEFAULT '',
                classification TEXT NOT NULL DEFAULT 'quiet',
                title TEXT NOT NULL DEFAULT '',
                blurb TEXT NOT NULL DEFAULT '',
                event_refs TEXT NOT NULL DEFAULT '{}',
                llm_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- ATL External Resource Governance: exchanges + pairlist universes are
            -- shared league resources. exchange_resources is the registry of what may
            -- be leased; exchange_leases records exactly which exchange + frozen pairlist
            -- universe each dev candidate held for each shift.
            CREATE TABLE IF NOT EXISTS exchange_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange_id TEXT NOT NULL,
                market_type TEXT NOT NULL DEFAULT 'futures',
                enabled INTEGER NOT NULL DEFAULT 1,
                max_dev_bots_per_shift INTEGER NOT NULL DEFAULT 3,
                max_total_concurrent_bots INTEGER NOT NULL DEFAULT 6,
                cooldown_minutes INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(exchange_id, market_type)
            );

            CREATE TABLE IF NOT EXISTS exchange_leases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                candidate_slug TEXT NOT NULL DEFAULT '',
                tier TEXT NOT NULL DEFAULT '',
                shift_code TEXT NOT NULL DEFAULT '',
                shift_id TEXT NOT NULL DEFAULT '',
                exchange_id TEXT NOT NULL,
                market_type TEXT NOT NULL DEFAULT 'futures',
                pairlist_manifest_id TEXT NOT NULL DEFAULT '',
                lease_start TEXT,
                lease_end TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- === Backtesting Department ===========================================
            -- Continuously-operating evidence engine. Jobs are (strategy x universe x
            -- window) backtests scheduled by priority bucket; results are evidence only
            -- and never modify live behavior. See backtesting_department_loop().
            CREATE TABLE IF NOT EXISTS backtest_jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'queued',
                lane_id TEXT NOT NULL DEFAULT '',
                priority_bucket INTEGER NOT NULL DEFAULT 5,
                priority_score REAL NOT NULL DEFAULT 0,
                strategy_key TEXT NOT NULL DEFAULT '',
                strategy_name TEXT NOT NULL DEFAULT '',
                strategy_version TEXT NOT NULL DEFAULT '',
                universe_key TEXT NOT NULL DEFAULT '',
                universe_name TEXT NOT NULL DEFAULT '',
                exchange TEXT NOT NULL DEFAULT 'binance',
                timeframe TEXT NOT NULL DEFAULT '5m',
                timerange TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'rebuild',
                reason TEXT NOT NULL DEFAULT '',
                comparison_group_id TEXT NOT NULL DEFAULT '',
                research_thread_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                failure_reason TEXT NOT NULL DEFAULT '',
                result_summary TEXT NOT NULL DEFAULT '',
                run_id TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_backtest_jobs_status ON backtest_jobs (status, priority_bucket, priority_score);

            CREATE TABLE IF NOT EXISTS backtest_results (
                job_id TEXT PRIMARY KEY,
                strategy_key TEXT NOT NULL DEFAULT '',
                universe_key TEXT NOT NULL DEFAULT '',
                total_pnl REAL NOT NULL DEFAULT 0,
                total_pnl_pct REAL NOT NULL DEFAULT 0,
                closed_trades INTEGER NOT NULL DEFAULT 0,
                open_trades INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                best_trade REAL NOT NULL DEFAULT 0,
                worst_trade REAL NOT NULL DEFAULT 0,
                max_drawdown REAL NOT NULL DEFAULT 0,
                profit_factor REAL NOT NULL DEFAULT 0,
                avg_hold_minutes REAL NOT NULL DEFAULT 0,
                exit_tag_distribution TEXT NOT NULL DEFAULT '{}',
                archive TEXT NOT NULL DEFAULT '',
                run_id TEXT NOT NULL DEFAULT '',
                strategy_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_backtest_results_cell ON backtest_results (strategy_key, universe_key);

            CREATE TABLE IF NOT EXISTS backtest_lanes (
                lane_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'idle',
                current_job_id TEXT NOT NULL DEFAULT '',
                started_at TEXT,
                last_job_id TEXT NOT NULL DEFAULT '',
                failure_reason TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO backtest_lanes (lane_id, status, updated_at) VALUES ('A', 'idle', ?)",
            (iso_now(),),
        )
        conn.commit()
        ensure_column(conn, "timeline_posts", "category", "TEXT NOT NULL DEFAULT 'league'")
        # Durable per-trade entry tag (the predicted "second act" for Second Act; the slot
        # tier / hypothesis tag for others). Added so the Entry Tag -> Exit Tag research
        # corpus survives the per-shift runtime-DB wipe; backfilled rows stay ''.
        ensure_column(conn, "dev_archived_trades", "enter_tag", "TEXT NOT NULL DEFAULT ''")
        for column_name, ddl in (
            ("wins", "INTEGER NOT NULL DEFAULT 0"),
            ("losses", "INTEGER NOT NULL DEFAULT 0"),
            ("avg_roi", "REAL NOT NULL DEFAULT 0"),
        ):
            ensure_column(conn, "dev_runtime_snapshots", column_name, ddl)
        for column_name, ddl in (
            ("follow_up_status", "TEXT NOT NULL DEFAULT ''"),
            ("follow_up_message", "TEXT NOT NULL DEFAULT ''"),
            ("follow_up_queued_at", "TEXT"),
            # Clinical per-trade diagnostics (ranked findings) backing the report card.
            ("diagnostics_json", "TEXT NOT NULL DEFAULT '[]'"),
            # Shift-boundary forced exits split out from strategy-chosen exits so the
            # grade reflects what the strategy did, not what the clock did to it.
            ("forced_exits", "INTEGER NOT NULL DEFAULT 0"),
            ("forced_realized_pnl", "REAL NOT NULL DEFAULT 0"),
            ("strategy_realized_pnl", "REAL NOT NULL DEFAULT 0"),
        ):
            ensure_column(conn, "dev_post_shift_reviews", column_name, ddl)
        for column_name, ddl in (
            ("generation_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("generation_model", "TEXT NOT NULL DEFAULT ''"),
            ("generation_prompt", "TEXT NOT NULL DEFAULT ''"),
            ("generation_error", "TEXT NOT NULL DEFAULT ''"),
            ("generation_progress", "TEXT NOT NULL DEFAULT ''"),
            ("generated_at", "TEXT"),
            ("implementation_summary", "TEXT NOT NULL DEFAULT ''"),
            ("generation_assumptions", "TEXT NOT NULL DEFAULT ''"),
            ("generation_warnings", "TEXT NOT NULL DEFAULT ''"),
            ("suggested_timeframe", "TEXT NOT NULL DEFAULT ''"),
            ("suggested_max_open_trades", "INTEGER NOT NULL DEFAULT 0"),
            ("minimal_config_notes", "TEXT NOT NULL DEFAULT ''"),
            ("strategy_class_name", "TEXT NOT NULL DEFAULT ''"),
            ("validation_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("validation_error", "TEXT NOT NULL DEFAULT ''"),
            ("validated_at", "TEXT"),
            ("review_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("reviewed_at", "TEXT"),
            ("assembly_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("assembly_error", "TEXT NOT NULL DEFAULT ''"),
            ("instance_assembled_at", "TEXT"),
            ("container_name", "TEXT NOT NULL DEFAULT ''"),
            ("api_port", "INTEGER"),
            ("working_directory", "TEXT NOT NULL DEFAULT ''"),
            ("open_trades", "INTEGER NOT NULL DEFAULT 0"),
            ("wins", "INTEGER NOT NULL DEFAULT 0"),
            ("losses", "INTEGER NOT NULL DEFAULT 0"),
            ("win_rate", "REAL NOT NULL DEFAULT 0"),
            ("avg_roi", "REAL NOT NULL DEFAULT 0"),
            ("champion_exits", "INTEGER NOT NULL DEFAULT 0"),
            ("generation_trigger", "TEXT NOT NULL DEFAULT 'manual'"),
            ("auto_apply_generated_strategy", "TEXT NOT NULL DEFAULT 'false'"),
            ("protected", "INTEGER NOT NULL DEFAULT 0"),
            # Temporal niche classified by the strategy generator/regenerator LLM.
            ("temporal_niche_start", "TEXT NOT NULL DEFAULT ''"),
            ("temporal_niche_end", "TEXT NOT NULL DEFAULT ''"),
            ("temporal_niche_note", "TEXT NOT NULL DEFAULT ''"),
            ("temporal_niche_status", "TEXT NOT NULL DEFAULT ''"),
            # Mini-season shift lifecycle. last_stop_reason gates the start-of-shift DB
            # wipe (only wipe after a CLEAN scheduled_shift_end, never crash-recovery).
            # last_shift_split_* hand the wind-down's exit_reason-aware split (captured
            # while the bot API is still live) to the post-shift review that runs after stop.
            ("last_stop_reason", "TEXT NOT NULL DEFAULT ''"),
            ("last_shift_reset_at", "TEXT"),
            ("last_shift_split_key", "TEXT NOT NULL DEFAULT ''"),
            ("last_shift_forced_exits", "INTEGER NOT NULL DEFAULT 0"),
            ("last_shift_forced_pnl", "REAL NOT NULL DEFAULT 0"),
            ("last_shift_strategy_trades", "INTEGER NOT NULL DEFAULT 0"),
            ("last_shift_strategy_wins", "INTEGER NOT NULL DEFAULT 0"),
            ("last_shift_strategy_pnl", "REAL NOT NULL DEFAULT 0"),
            ("last_shift_strategy_avg_roi", "REAL NOT NULL DEFAULT 0"),
        ):
            ensure_column(conn, "dev_candidates", column_name, ddl)
        for column_name, ddl in (
            ("draft_slots", "INTEGER NOT NULL DEFAULT 2"),
            ("turnover_processed_at", "TEXT"),
        ):
            ensure_column(conn, "league_seasons", column_name, ddl)
        for column_name, ddl in (
            ("approval_required", "INTEGER NOT NULL DEFAULT 0"),
            ("approval_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("approval_notes", "TEXT NOT NULL DEFAULT ''"),
            ("approved_action", "TEXT NOT NULL DEFAULT ''"),
            ("strategy_path", "TEXT NOT NULL DEFAULT ''"),
            ("strategy_file_hash", "TEXT NOT NULL DEFAULT ''"),
            ("fix_suggestion", "TEXT NOT NULL DEFAULT ''"),
        ):
            ensure_column(conn, "league_team_season_reviews", column_name, ddl)
        for column_name, ddl in (
            ("approval_status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("approval_notes", "TEXT NOT NULL DEFAULT ''"),
        ):
            ensure_column(conn, "league_season_draft_recommendations", column_name, ddl)
        # Temporal Niche (Signal Timing Spectrum) — descriptive, declared placement
        # of each organism. Band start/end slugs + a free-text note for off-spectrum
        # niches. No compatibility scoring is derived from these.
        for column_name, ddl in (
            ("temporal_niche_start", "TEXT NOT NULL DEFAULT ''"),
            ("temporal_niche_end", "TEXT NOT NULL DEFAULT ''"),
            ("temporal_niche_note", "TEXT NOT NULL DEFAULT ''"),
            # Provenance of the niche: '' / 'seed' / 'llm' (auto-maintained) or
            # 'human' (pinned — never overwritten by re-classification).
            ("temporal_niche_source", "TEXT NOT NULL DEFAULT ''"),
        ):
            ensure_column(conn, "ml_strategy_registry", column_name, ddl)
        # Phase 1: structured research-index metadata. All nullable — existing rows keep
        # NULLs and the recency/keyword retrieval is unchanged. New entries are written
        # with these fields (deterministically, no LLM) so Phase 3 scoped retrieval and
        # the belief map can rely on real columns instead of tag-string markers.
        for column_name, ddl in (
            ("entry_type", "TEXT"),        # observation|question|finding|conclusion|manual_note|update|repo
            ("author_type", "TEXT"),       # human | agent | system
            ("thread_id", "INTEGER"),      # research_threads.id when applicable
            ("parent_entry_id", "INTEGER"),
            ("strategy_id", "TEXT"),
            ("family_id", "TEXT"),
            ("topic_tags", "TEXT"),        # curated comma-separated topics/entities (reserved for Phase 3)
            ("confidence", "REAL"),        # 0..1 (reserved)
            ("status", "TEXT"),            # active|superseded|rejected|canonical
            ("created_at", "TEXT"),        # first-seen; updated_at stays last-touched
            ("supersedes_id", "INTEGER"),
        ):
            ensure_column(conn, "research_index_entries", column_name, ddl)
        conn.commit()
    seed_exchange_resources()
    backfill_dev_archived_trades()


def seed_exchange_resources() -> None:
    """Register one futures resource row per dev exchange profile (idempotent).

    Adding a new governable exchange is a data change (insert a row), not a code
    change. Existing rows are left alone so operator edits to caps/enabled persist.
    """
    setting_defaults = {
        "resource_governance_enabled": "true",
        "pairlist_manifest_minutes": "360",
        "pairlist_manifest_base_url": "http://host.docker.internal:8000",
        "coingecko_api_key": "",
        "exchange_max_dev_bots_per_shift": "3",
        "exchange_max_total_concurrent_bots": "6",
        "exchange_cooldown_minutes": "0",
    }
    with closing(get_db()) as conn:
        for key, value in setting_defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, iso_now()),
            )
        conn.commit()
    cap_per_shift = int(get_setting("exchange_max_dev_bots_per_shift", "3") or 3)
    cap_total = int(get_setting("exchange_max_total_concurrent_bots", "6") or 6)
    cooldown = int(get_setting("exchange_cooldown_minutes", "0") or 0)
    with closing(get_db()) as conn:
        for profile in DEV_EXCHANGE_PROFILES:
            exchange_id = str(profile["name"])
            market_type = str(profile.get("trading_mode") or "futures")
            conn.execute(
                """
                INSERT OR IGNORE INTO exchange_resources
                    (exchange_id, market_type, enabled, max_dev_bots_per_shift,
                     max_total_concurrent_bots, cooldown_minutes, notes, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange_id,
                    market_type,
                    cap_per_shift,
                    cap_total,
                    cooldown,
                    f"Seeded from DEV_EXCHANGE_PROFILES ({profile.get('stake_currency')} settle).",
                    iso_now(),
                    iso_now(),
                ),
            )
        conn.commit()


def seed_files() -> None:
    if not INSTANCE_REGISTRY_PATH.exists():
        INSTANCE_REGISTRY_PATH.write_text(
            json.dumps(
                [
                    {
                        "id": "cosmo-wanda-20-pi",
                        "name": "Cosmo & Wanda",
                        "display_name": "Cosmo & Wanda - Top 20",
                        "strategy_family": "Cosmo & Wanda",
                        "pair_universe": "Top 20",
                        "host_machine": "Raspberry Pi",
                        "api_url": "http://10.0.0.159:8080",
                        "api_username": "admin",
                        "api_password": "password",
                        "db_path": "",
                        "config_path": "user_data/config.json",
                        "strategy_path": "user_data/strategies/CosmoWanda.py",
                        "starting_capital": 250,
                        "stake_currency": "USDC",
                        "start_date": "2026-05-19T00:00:00+00:00",
                        "role": "incumbent champion",
                        "data_quality": "Official",
                        "head_start_note": "Approx. 17-day head start over the local challengers.",
                        "notes": "Primary official dataset. If auth differs on the Pi, update credentials here."
                    },
                    {
                        "id": "cosmo-wanda-50-pc",
                        "name": "Cosmo & Wanda",
                        "display_name": "Cosmo & Wanda - Top 50",
                        "strategy_family": "Cosmo & Wanda",
                        "pair_universe": "Top 50",
                        "host_machine": "PC Docker",
                        "api_url": "http://127.0.0.1:8090",
                        "api_username": "admin",
                        "api_password": "password",
                        "db_path": "user_data/tradesv3_cosmowanda50_bitget.sqlite",
                        "config_path": "user_data/config.cosmowanda50.json",
                        "strategy_path": "user_data/strategies/CosmoWanda.py",
                        "starting_capital": 1000,
                        "stake_currency": "USDT",
                        "start_date": "2026-06-05T00:00:00+00:00",
                        "role": "challenger",
                        "data_quality": "Restarted",
                        "head_start_note": "Local clean challenger started on June 5, 2026.",
                        "notes": "Fresh local challenger run."
                    },
                    {
                        "id": "timmy-50-pc",
                        "name": "Timmy",
                        "display_name": "Timmy - Top 50",
                        "strategy_family": "Timmy",
                        "pair_universe": "Top 50",
                        "host_machine": "PC Docker",
                        "api_url": "http://127.0.0.1:8091",
                        "api_username": "admin",
                        "api_password": "password",
                        "db_path": "user_data/tradesv3_timmy50_bybit.sqlite",
                        "config_path": "user_data/config.timmy50.json",
                        "strategy_path": "user_data/strategies/tiimmyturntup.py",
                        "starting_capital": 1000,
                        "stake_currency": "USDT",
                        "start_date": "2026-06-05T00:00:00+00:00",
                        "role": "prospect",
                        "data_quality": "Clean",
                        "head_start_note": "Local clean challenger started on June 5, 2026.",
                        "notes": "Fresh local Timmy Top 50 run."
                    },
                    {
                        "id": "timmy-20-pc",
                        "name": "Timmy",
                        "display_name": "Timmy - Top 20",
                        "strategy_family": "Timmy",
                        "pair_universe": "Top 20",
                        "host_machine": "PC Docker",
                        "api_url": "http://127.0.0.1:8092",
                        "api_username": "admin",
                        "api_password": "password",
                        "db_path": "user_data/tradesv3_timmy20_okx.sqlite",
                        "config_path": "user_data/config.timmy20.local.json",
                        "strategy_path": "user_data/strategies/tiimmyturntup.py",
                        "starting_capital": 1000,
                        "stake_currency": "USDT",
                        "start_date": "2026-06-05T00:00:00+00:00",
                        "role": "prospect",
                        "data_quality": "Clean",
                        "head_start_note": "Local clean challenger started on June 5, 2026.",
                        "notes": "Fresh local Timmy Top 20 run."
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not QUESTIONS_PATH.exists():
        QUESTIONS_PATH.write_text(
            json.dumps(
                [
                    "Does Cosmo/Wanda Top 20 outperform Top 50?",
                    "Does Timmy perform better on Top 20 or Top 50?",
                    "Do champion exits maintain a higher ROI than dynamic ROI exits?",
                    "Does Timmy's low-win/high-win-size profile survive live dry-run conditions?",
                    "Does broader pair coverage improve opportunity or add noise?"
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_HYPOTHESES_PATH.exists():
        ML_HYPOTHESES_PATH.write_text(
            json.dumps(
                [
                    {
                        "id": "dark-matter-antimatter",
                        "name": "Dark Matter / Antimatter",
                        "nickname": "Timmy Origin Bucket",
                        "description": "Compressed price with hidden internal pressure and unstable motion that may precede a breakout.",
                        "market_behavior": "Hovering tape with internal aggression, imbalance, and latent directional energy.",
                        "features_used": ["compression_ratio", "internal_motion_score", "imbalance_delta", "breakout_pressure"],
                        "target_variable": "Forward 12-candle directional move exceeding loser threshold.",
                        "training_period": "2025-01-01 to 2025-10-31",
                        "validation_period": "2025-11-01 to 2026-01-31",
                        "known_risks": ["Leakage from overlapping windows", "Repeated validation reuse", "Regime fragility"],
                        "status": "dry-run candidate",
                        "theme": "Predictive sniper scouting lane",
                        "evidence_quality": "Medium-high",
                        "next_action": "Continue comparing Timmy live dry-run exits against re-run validation with stricter leakage controls."
                    },
                    {
                        "id": "ghost-ladder",
                        "name": "Ghost Ladder",
                        "nickname": "Third Lane Prospect",
                        "description": "Investigates silent stair-step accumulation before expansion without full Dark Matter instability.",
                        "market_behavior": "Low-drama climbs with repeated shallow pullbacks and persistent order-book reclaim behavior.",
                        "features_used": ["ladder_persistence", "micro_pullback_depth", "reclaim_velocity"],
                        "target_variable": "Forward move persistence over 24 candles.",
                        "training_period": "2025-03-01 to 2025-12-31",
                        "validation_period": "2026-01-01 to 2026-02-15",
                        "known_risks": ["Overfitting to a strong trend regime", "Feature redundancy"],
                        "status": "testing",
                        "theme": "Possible third strategy lane",
                        "evidence_quality": "Early",
                        "next_action": "Build clean backtest prototype and compare against Timmy/Cosmo assumptions."
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_BUCKETS_PATH.exists():
        ML_BUCKETS_PATH.write_text(
            json.dumps(
                [
                    {
                        "id": "dark-matter",
                        "bucket_name": "Dark Matter",
                        "hypothesis_id": "dark-matter-antimatter",
                        "feature_conditions": "compression_ratio < 0.18, internal_motion_score > 0.74, imbalance_delta > 0.41",
                        "samples": 418,
                        "win_loss_profile": "High variance, low hit rate, outsized winner profile",
                        "avg_forward_move": 3.9,
                        "best_example": "ARB 5m coil resolving into impulsive trend continuation",
                        "worst_example": "False breakout regime with news whipsaw contamination",
                        "long_vs_short": "More stable on long side; short side spikes harder but degrades faster.",
                        "example_trades": "Timmy validation clusters around ARB, OP, AVAX.",
                        "causal_rating": "Plausible but not proven",
                        "contamination_flag": "Needs repeat validation without reused validation windows",
                        "notes": "This is the bucket that produced the original Timmy-style thesis."
                    },
                    {
                        "id": "antimatter",
                        "bucket_name": "Antimatter",
                        "hypothesis_id": "dark-matter-antimatter",
                        "feature_conditions": "compression_ratio < 0.15, internal_motion_score > 0.82, downside_imbalance > 0.50",
                        "samples": 179,
                        "win_loss_profile": "Lower sample count, more explosive downside resolution",
                        "avg_forward_move": 4.6,
                        "best_example": "Short-side cascade after fake support retention",
                        "worst_example": "Short squeeze reversal on thin book conditions",
                        "long_vs_short": "Predominantly short behavior",
                        "example_trades": "Short basket candidates with fast stop discipline.",
                        "causal_rating": "Suspicious but interesting",
                        "contamination_flag": "Short-side sample scarcity",
                        "notes": "Use extra caution on leakage and survivorship bias."
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_FEATURES_PATH.exists():
        ML_FEATURES_PATH.write_text(
            json.dumps(
                [
                    {
                        "feature_name": "compression_ratio",
                        "formula": "rolling_true_range / rolling_price_range",
                        "timeframe": "5m",
                        "strategy_using_it": "Timmy hypothesis family",
                        "intuition": "Captures apparent price stillness versus latent movement compression.",
                        "risk_of_leakage": "Low if computed only from historical bars",
                        "safe_for_live_use": True
                    },
                    {
                        "feature_name": "internal_motion_score",
                        "formula": "weighted intrabar excursion + wick stress + reclaim churn",
                        "timeframe": "5m",
                        "strategy_using_it": "Dark Matter / Antimatter",
                        "intuition": "Measures hidden instability inside a seemingly quiet candle sequence.",
                        "risk_of_leakage": "Medium if future candle normalization leaks in",
                        "safe_for_live_use": True
                    },
                    {
                        "feature_name": "forward_return_12",
                        "formula": "(close[t+12] - close[t]) / close[t]",
                        "timeframe": "5m",
                        "strategy_using_it": "Training target only",
                        "intuition": "Labels whether the future move justified the setup.",
                        "risk_of_leakage": "High",
                        "safe_for_live_use": False
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_MODELS_PATH.exists():
        ML_MODELS_PATH.write_text(
            json.dumps(
                [
                    {
                        "model_name": "timmy_bucket_ranker_v3",
                        "algorithm_type": "Gradient Boosted Trees",
                        "dataset": "hyperliquid_futures_5m_regime_set_a",
                        "feature_set": ["compression_ratio", "internal_motion_score", "imbalance_delta", "breakout_pressure"],
                        "training_date": "2026-02-03",
                        "metrics": "AUC 0.61, top-bucket avg forward move 3.9%",
                        "saved_artifact_path": "user_data/notebooks/timmy_bucket_ranker_v3.pkl",
                        "notes": "Primary rediscovery run for Dark Matter bucket.",
                        "influenced_live_strategy": True
                    },
                    {
                        "model_name": "ghost_ladder_probe_v1",
                        "algorithm_type": "Lightweight rules + logistic probe",
                        "dataset": "cross-exchange_ladder_samples_b",
                        "feature_set": ["ladder_persistence", "micro_pullback_depth", "reclaim_velocity"],
                        "training_date": "2026-05-27",
                        "metrics": "Precision unstable, promising prospect bucket separation",
                        "saved_artifact_path": "user_data/notebooks/ghost_ladder_probe_v1.ipynb",
                        "notes": "Not live-influential yet.",
                        "influenced_live_strategy": False
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_DRAFT_BOARD_PATH.exists():
        ML_DRAFT_BOARD_PATH.write_text(
            json.dumps(
                [
                    {
                        "prospect_name": "Ghost Ladder",
                        "strategy_family": "Third Lane Candidate",
                        "expected_edge": "Trend continuation from low-drama accumulation",
                        "evidence_quality": "Developing",
                        "risk_level": "Medium",
                        "backtest_strength": "Unproven",
                        "live_readiness": "Low",
                        "notes": "Interesting because it is neither Cosmo reactive breadth nor Timmy predictive compression.",
                        "draft_status": "Scouting"
                    },
                    {
                        "prospect_name": "Antimatter Short Cartographer",
                        "strategy_family": "Timmy branch",
                        "expected_edge": "Explosive short-side breakdown capture",
                        "evidence_quality": "Mixed",
                        "risk_level": "High",
                        "backtest_strength": "Spiky",
                        "live_readiness": "Medium-low",
                        "notes": "Needs contamination review before promotion.",
                        "draft_status": "Under review"
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    if not ML_PROMOTIONS_PATH.exists():
        ML_PROMOTIONS_PATH.write_text(
            json.dumps(
                [
                    {
                        "strategy_name": "Timmy",
                        "pipeline_stage": "League candidate",
                        "raw_observation": "Compressed-price/high-internal-motion regime produced outsized forward moves.",
                        "feature_bucket": "Dark Matter / Antimatter",
                        "backtest_prototype": "Timmy backtest branch",
                        "paper_dry_run": "Local dry-run instances launched June 5, 2026",
                        "league_candidate": "timmy-20-pc, timmy-50-pc",
                        "official_team": "",
                        "relegated_archived": "",
                        "promotion_reason": "Strong enough causal hypothesis to earn live dry-run observation.",
                        "supporting_evidence": "Validation bucket separation plus behaviorally distinct exit philosophy.",
                        "entered_dry_run": "2026-06-05",
                        "removal_reason": ""
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


def list_instances() -> list[dict[str, Any]]:
    return load_json(INSTANCE_REGISTRY_PATH, [])


def local_now() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def iso_local_now() -> str:
    return local_now().isoformat()


def development_shift_definitions(tier: str) -> list[dict[str, Any]]:
    return DEV_SHIFT_WINDOWS.get(tier, [])


def shift_window_label(shift: dict[str, Any]) -> str:
    return f"{shift['start_hour']:02d}:00-{shift['end_hour'] % 24:02d}:00"


def shift_window_bounds(shift: dict[str, Any], now_local: datetime) -> tuple[datetime, datetime]:
    start = now_local.replace(hour=int(shift["start_hour"]), minute=0, second=0, microsecond=0)
    end_hour = int(shift["end_hour"])
    end = now_local.replace(hour=end_hour % 24, minute=0, second=0, microsecond=0)
    if end_hour == 24:
        end += timedelta(days=1)
    return start, end


def shift_window_bounds_for_date(shift: dict[str, Any], day_local: datetime) -> tuple[datetime, datetime]:
    start = day_local.replace(hour=int(shift["start_hour"]), minute=0, second=0, microsecond=0)
    end_hour = int(shift["end_hour"])
    end = day_local.replace(hour=end_hour % 24, minute=0, second=0, microsecond=0)
    if end_hour == 24:
        end += timedelta(days=1)
    return start, end


def active_shift_for_tier(tier: str, now_local: datetime | None = None) -> dict[str, Any] | None:
    now_local = now_local or local_now()
    for shift in development_shift_definitions(tier):
        start, end = shift_window_bounds(shift, now_local)
        if start <= now_local < end:
            enriched = dict(shift)
            enriched["window"] = shift_window_label(shift)
            enriched["started_at"] = start.isoformat()
            enriched["ends_at"] = end.isoformat()
            enriched["seconds_since_start"] = int((now_local - start).total_seconds())
            enriched["seconds_until_end"] = int((end - now_local).total_seconds())
            return enriched
    return None


def next_shift_for_tier(tier: str, now_local: datetime | None = None) -> dict[str, Any] | None:
    now_local = now_local or local_now()
    shifts = development_shift_definitions(tier)
    if not shifts:
        return None
    for shift in shifts:
        start, end = shift_window_bounds(shift, now_local)
        if now_local < start:
            return {
                **shift,
                "window": shift_window_label(shift),
                "starts_at": start.isoformat(),
                "ends_at": end.isoformat(),
            }
    first = shifts[0]
    next_day = now_local + timedelta(days=1)
    start = next_day.replace(hour=int(first["start_hour"]), minute=0, second=0, microsecond=0)
    end = next_day.replace(hour=int(first["end_hour"]) % 24, minute=0, second=0, microsecond=0)
    if int(first["end_hour"]) == 24:
        end += timedelta(days=1)
    return {**first, "window": shift_window_label(first), "starts_at": start.isoformat(), "ends_at": end.isoformat()}


def hours_between(start: datetime | None, end: datetime | None) -> float:
    if not start or not end:
        return 0.0
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return max(0.0, (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600.0)


def scheduled_runtime_hours_for_period(
    tier: str,
    shift_code: str,
    start_at: datetime | None,
    end_at: datetime | None,
) -> float:
    if not start_at or not end_at:
        return 0.0
    if start_at >= end_at:
        return 0.0
    if tier == "official_24h":
        return hours_between(start_at, end_at)
    if tier == "prospect_12h":
        shifts = development_shift_definitions("twelve_hour")
    elif tier == "candidate_6h":
        shifts = development_shift_definitions("six_hour")
    else:
        return 0.0
    shift = next((item for item in shifts if item["code"] == shift_code), None)
    if not shift:
        return 0.0
    local_start = start_at.astimezone(LOCAL_TIMEZONE)
    local_end = end_at.astimezone(LOCAL_TIMEZONE)
    cursor = local_start.replace(hour=0, minute=0, second=0, microsecond=0)
    total_hours = 0.0
    while cursor <= local_end:
        shift_start, shift_end = shift_window_bounds_for_date(shift, cursor)
        overlap_start = max(local_start, shift_start)
        overlap_end = min(local_end, shift_end)
        total_hours += hours_between(overlap_start, overlap_end)
        cursor += timedelta(days=1)
    return round(total_hours, 4)


def candidate_schedule_markers(row: dict[str, Any]) -> tuple[str, str]:
    tier = str(row.get("tier") or "")
    shift_code = str(row.get("shift_code") or "").upper()
    now_local = local_now()
    if not tier or not shift_code:
        return "", ""
    for shift in development_shift_definitions(tier):
        if shift["code"] != shift_code:
            continue
        active_shift = active_shift_for_tier(tier, now_local)
        if active_shift and active_shift["code"] == shift_code:
            return active_shift["started_at"], active_shift["ends_at"]
        next_shift = next_shift_for_tier(tier, now_local)
        if next_shift and next_shift["code"] == shift_code:
            return next_shift["starts_at"], next_shift["ends_at"]
        start = now_local + timedelta(days=1)
        start = start.replace(hour=int(shift["start_hour"]), minute=0, second=0, microsecond=0)
        end = start.replace(hour=int(shift["end_hour"]) % 24)
        if int(shift["end_hour"]) == 24:
            end += timedelta(days=1)
        return start.isoformat(), end.isoformat()
    return "", ""


def candidate_requirement_copy(row: dict[str, Any]) -> str:
    strategy_path = resolve_path(str(row.get("strategy_path") or ""))
    if row.get("generation_status") in {"queued", "generating"}:
        return "Generating strategy file"
    if not strategy_path or not strategy_path.exists():
        return "Needs strategy file"
    if row.get("validation_status") == "failed":
        return "Strategy validation failed"
    if row.get("review_status") != "reviewed":
        return "Needs human review"
    if row.get("assembly_status") != "assembled":
        return "Needs instance assembly"
    if not row.get("shift_code"):
        return "Needs shift assignment"
    if row.get("runtime_status") == "off-shift":
        return "Waiting for scheduled shift"
    return ""


def runtime_sample_flags(total_runtime_hours: float, closed_trades: int) -> list[str]:
    flags: list[str] = []
    if total_runtime_hours < PROJECTION_HIDE_RUNTIME_HOURS:
        flags.extend(["tiny_sample", "low_runtime", "projection_unreliable"])
    elif total_runtime_hours < PROJECTION_STRONG_WARNING_RUNTIME_HOURS:
        flags.extend(["low_runtime", "projection_unreliable"])
    if closed_trades < PROJECTION_MIN_CLOSED_TRADES:
        flags.append("low_trade_count")
        if "projection_unreliable" not in flags:
            flags.append("projection_unreliable")
    return flags


def safe_rate_per_hour(value: float, total_runtime_hours: float) -> float:
    if total_runtime_hours <= 0:
        return 0.0
    return value / total_runtime_hours


def projected_per_24h(value: float, total_runtime_hours: float) -> float | None:
    if total_runtime_hours < PROJECTION_HIDE_RUNTIME_HOURS:
        return None
    return safe_rate_per_hour(value, total_runtime_hours) * 24.0


def freshness_flags_for_candidate(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    now = utc_now()
    updated_at = resolve_optional_datetime(str(row.get("updated_at") or ""))
    generated_at = resolve_optional_datetime(str(row.get("generated_at") or ""))
    if row.get("lifecycle_state") == "draft_idea" and not row.get("strategy_file_exists"):
        created_at = resolve_optional_datetime(str(row.get("created_at") or ""))
        if created_at and hours_between(created_at, now) >= 72:
            flags.append("inactive_draft_idea")
    if row.get("generation_status") == "generated" and row.get("review_status") != "reviewed":
        flags.append("needs_review")
    if row.get("assembly_status") == "assembled" and float(row.get("total_runtime_hours") or 0) <= 0:
        flags.append("needs_runtime")
    if updated_at and hours_between(updated_at, now) >= 48:
        flags.append("stale_candidate")
    if row.get("tier") in {"six_hour", "twelve_hour"} and float(row.get("total_runtime_hours") or 0) >= 12 and row.get("review_status") == "reviewed":
        flags.append("needs_decision")
    if "stale_candidate" in flags and row.get("runtime_status") != "running" and row.get("tier") in {"six_hour", "twelve_hour", "bootcamp"}:
        flags.append("eligible_for_archive")
    if generated_at and row.get("review_status") != "reviewed" and hours_between(generated_at, now) >= 12:
        flags.append("needs_review")
    return list(dict.fromkeys(flags))


def development_candidate_rows(tier: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM dev_candidates"
    params: list[Any] = []
    if tier:
        query += " WHERE tier = ?"
        params.append(tier)
    query += " ORDER BY CASE tier "
    for index, tier_key in enumerate(DEV_TIER_LABELS.keys(), start=1):
        query += f"WHEN '{tier_key}' THEN {index} "
    query += "ELSE 99 END, shift_code ASC, name COLLATE NOCASE ASC"
    with closing(get_db()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [decorate_development_candidate(dict(row)) for row in rows]


def get_development_candidate(candidate_id: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM dev_candidates WHERE id = ?", (candidate_id,)).fetchone()
    return decorate_development_candidate(dict(row)) if row else None


def decorate_development_candidate(row: dict[str, Any]) -> dict[str, Any]:
    row["tier_label"] = DEV_TIER_LABELS.get(row.get("tier", ""), row.get("tier", "Unknown"))
    row["lifecycle_label"] = DEV_LIFECYCLE_LABELS.get(row.get("lifecycle_state", ""), row.get("lifecycle_state", "Unknown"))
    row["runtime_status"] = row.get("runtime_status") or "paused"
    row["runtime_window"] = row.get("runtime_window") or candidate_runtime_window(row)
    row["display_shift"] = row.get("shift_code") or "Unassigned"
    row["heartbeat"] = bool(row.get("heartbeat_ok"))
    row["can_run"] = bool(row.get("start_command") or row.get("api_url"))
    row["strategy_file_exists"] = bool(resolve_path(str(row.get("strategy_path") or "")) and resolve_path(str(row.get("strategy_path") or "")).exists())
    row["workflow_requirement"] = candidate_requirement_copy(row)
    row["can_review"] = row.get("generation_status") == "generated" and row.get("validation_status") == "passed"
    row["can_assemble"] = row.get("review_status") == "reviewed" and row.get("validation_status") == "passed"
    row["can_assign_shift"] = row.get("assembly_status") == "assembled"
    latest_review = latest_development_post_shift_review(int(row["id"]))
    row["latest_post_shift_review"] = latest_review
    row["latest_post_shift_grade"] = latest_review.get("grade", "") if latest_review else ""
    row["latest_post_shift_decision"] = latest_review.get("decision_bucket", "") if latest_review else ""
    row["latest_post_shift_decision_label"] = latest_review.get("decision_label", "") if latest_review else ""
    row["latest_post_shift_score"] = parse_float(latest_review.get("overall_score")) if latest_review else None
    row["latest_post_shift_summary"] = latest_review.get("summary", "") if latest_review else ""
    next_start, next_stop = candidate_schedule_markers(row)
    row["next_scheduled_start"] = next_start
    row["next_scheduled_stop"] = next_stop
    row["current_runtime_state"] = str(row.get("runtime_status") or "paused").replace("-", "_")
    return candidate_runtime_metrics(row)


def candidate_runtime_window(row: dict[str, Any]) -> str:
    shift_code = str(row.get("shift_code") or "").upper()
    for shift in development_shift_definitions(str(row.get("tier", ""))):
        if shift["code"] == shift_code:
            return shift_window_label(shift)
    return ""


def create_development_candidate(payload: dict[str, Any]) -> int:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    created_at = iso_now()
    slug_base = registry_slug(name) or f"candidate-{int(time.time())}"
    slug = slug_base
    with closing(get_db()) as conn:
        suffix = 2
        while conn.execute("SELECT 1 FROM dev_candidates WHERE slug = ?", (slug,)).fetchone():
            slug = f"{slug_base}-{suffix}"
            suffix += 1
        conn.execute(
            """
            INSERT INTO dev_candidates (
                slug, name, lifecycle_state, tier, shift_code, runtime_window, runtime_status,
                status_detail, override_mode, hypothesis, strategy_notes, long_short_mode,
                expected_behavior, risk_profile, coin_universe, timeframe, eligibility_status,
                notes, start_command, stop_command, api_url, api_username, api_password,
                db_path, config_path, log_path, strategy_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slug,
                name,
                str(payload.get("lifecycle_state", "draft_idea")) or "draft_idea",
                str(payload.get("tier", "draft_room")) or "draft_room",
                str(payload.get("shift_code", "")).upper(),
                "",
                "paused",
                "Created in development pipeline.",
                "auto",
                str(payload.get("hypothesis", "")),
                str(payload.get("strategy_notes", "")),
                str(payload.get("long_short_mode", "both")),
                str(payload.get("expected_behavior", "")),
                str(payload.get("risk_profile", "")),
                str(payload.get("coin_universe", "")),
                str(payload.get("timeframe", "")),
                "not_ready",
                str(payload.get("notes", "")),
                str(payload.get("start_command", "")),
                str(payload.get("stop_command", "")),
                str(payload.get("api_url", "")),
                str(payload.get("api_username", "")),
                str(payload.get("api_password", "")),
                str(payload.get("db_path", "")),
                str(payload.get("config_path", "")),
                str(payload.get("log_path", "")),
                str(payload.get("strategy_path", "")),
                created_at,
                created_at,
            ),
        )
        candidate_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    development_runtime_event(candidate_id, "created", "Candidate added to the development league.", name)
    return candidate_id


def update_development_candidate(candidate_id: int, **updates: Any) -> None:
    if not updates:
        return
    updates["updated_at"] = iso_now()
    fields = [f"{key} = ?" for key in updates]
    values = list(updates.values()) + [candidate_id]
    with closing(get_db()) as conn:
        conn.execute(f"UPDATE dev_candidates SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def development_runtime_event(candidate_id: int, event_type: str, title: str, details: str = "") -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO dev_runtime_events (candidate_id, created_at, event_type, title, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (candidate_id, iso_now(), event_type, title, details),
        )
        conn.commit()


def development_runtime_events(candidate_id: int, limit: int = 40) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM dev_runtime_events
            WHERE candidate_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (candidate_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def development_runtime_history(candidate_id: int, limit: int = 60) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM dev_runtime_snapshots
            WHERE candidate_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (candidate_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def development_runtime_sessions(candidate_id: int, limit: int = 60) -> list[dict[str, Any]]:
    try:
        with closing(get_db()) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM dev_runtime_sessions
                WHERE candidate_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (candidate_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []


def normalize_utc(moment: datetime | None) -> datetime | None:
    if not moment:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def latest_completed_development_session(candidate_id: int) -> dict[str, Any] | None:
    try:
        with closing(get_db()) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM dev_runtime_sessions
                WHERE candidate_id = ? AND stopped_at IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (candidate_id,),
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None


def decode_jsonish_payload(raw: str, fallback: Any) -> Any:
    if not raw.strip():
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def hydrate_post_shift_review(row: dict[str, Any]) -> dict[str, Any]:
    rubric = decode_jsonish_payload(str(row.get("rubric_json") or "[]"), [])
    row["rubric"] = rubric if isinstance(rubric, list) else []
    diagnostics = decode_jsonish_payload(str(row.get("diagnostics_json") or "[]"), [])
    row["diagnostics"] = diagnostics if isinstance(diagnostics, list) else []
    row["decision_label"] = str(row.get("decision_bucket") or "hold").replace("_", " ").title()
    row["score_label"] = f"{parse_float(row.get('overall_score')):.1f}/100"
    return row


def development_post_shift_reviews(candidate_id: int, limit: int = 12) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM dev_post_shift_reviews
            WHERE candidate_id = ?
            ORDER BY session_stopped_at DESC, id DESC
            LIMIT ?
            """,
            (candidate_id, limit),
        ).fetchall()
    return [hydrate_post_shift_review(dict(row)) for row in rows]


def latest_development_post_shift_review(candidate_id: int) -> dict[str, Any] | None:
    rows = development_post_shift_reviews(candidate_id, limit=1)
    return rows[0] if rows else None


def get_development_post_shift_review_by_key(review_key: str) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM dev_post_shift_reviews WHERE review_key = ?",
            (review_key,),
        ).fetchone()
    return hydrate_post_shift_review(dict(row)) if row else None


def recent_development_post_shift_reviews(limit: int = 10) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT r.*, c.name
            FROM dev_post_shift_reviews r
            JOIN dev_candidates c ON c.id = r.candidate_id
            ORDER BY r.session_stopped_at DESC, r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [hydrate_post_shift_review(dict(row)) for row in rows]


def infer_candidate_strategy_style(candidate: dict[str, Any]) -> str:
    text = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "name",
            "expected_behavior",
            "strategy_notes",
            "hypothesis",
            "notes",
            "timeframe",
            "suggested_timeframe",
        )
    ).lower()
    aggressive_score = 0
    patient_score = 0
    for term in ("aggressive", "fast", "quick", "rapid", "roadrunner", "churn", "high trade", "momentum", "scalp"):
        if term in text:
            aggressive_score += 1
    for term in ("patient", "slow", "slaking", "scrupulous", "selective", "barely trades", "swing", "position"):
        if term in text:
            patient_score += 1
    timeframe = str(candidate.get("suggested_timeframe") or candidate.get("timeframe") or "").strip().lower()
    if timeframe.endswith("m") and 0 < parse_intish(timeframe[:-1]) <= 15:
        aggressive_score += 2
    if timeframe.endswith("h") and parse_intish(timeframe[:-1]) >= 4:
        patient_score += 2
    if timeframe.endswith("d"):
        patient_score += 2
    if aggressive_score >= patient_score + 2:
        return "aggressive"
    if patient_score >= aggressive_score + 2:
        return "patient"
    return "balanced"


def development_shift_snapshot_metrics(candidate_id: int, started_at: datetime | None, stopped_at: datetime | None) -> dict[str, Any]:
    metrics = {
        "heartbeat_ratio": 0.0,
        "max_drawdown": 0.0,
        "worst_open_trade": 0.0,
        "realized_pnl_delta": 0.0,
    }
    started_utc = normalize_utc(started_at)
    stopped_utc = normalize_utc(stopped_at)
    if not started_utc or not stopped_utc:
        return metrics
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM dev_runtime_snapshots
            WHERE candidate_id = ?
            ORDER BY captured_at ASC, id ASC
            """,
            (candidate_id,),
        ).fetchall()
    selected: list[dict[str, Any]] = []
    for row in rows:
        captured_at = normalize_utc(resolve_optional_datetime(str(row["captured_at"])))
        if captured_at and started_utc <= captured_at <= stopped_utc:
            selected.append(dict(row))
    if not selected:
        return metrics
    metrics["heartbeat_ratio"] = percentage(sum(1 for row in selected if int(row.get("heartbeat_ok") or 0)), len(selected))
    metrics["max_drawdown"] = max(parse_float(row.get("max_drawdown")) for row in selected)
    metrics["worst_open_trade"] = min(parse_float(row.get("worst_open_trade")) for row in selected)
    metrics["realized_pnl_delta"] = round(
        parse_float(selected[-1].get("realized_pnl")) - parse_float(selected[0].get("realized_pnl")),
        4,
    )
    return metrics


def fetch_closed_trades_via_api(candidate: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Pull closed trades from a bot's REST API. Returns None when the API is unreachable
    so callers can fall back to another data source."""
    if not candidate.get("api_url"):
        return None
    instance = {
        "api_url": candidate.get("api_url", ""),
        "api_username": candidate.get("api_username", ""),
        "api_password": candidate.get("api_password", ""),
    }
    records: list[dict[str, Any]] = []
    try:
        with httpx.Client() as client:
            offset = 0
            limit = 250
            while True:
                payload = api_get(client, instance, f"/api/v1/trades?limit={limit}&offset={offset}")
                page = payload.get("trades", []) if isinstance(payload, dict) else []
                if not page:
                    break
                for trade in page:
                    if int(bool(trade.get("is_open"))):
                        continue
                    records.append(
                        {
                            "close_profit": first_present_value(trade.get("close_profit"), trade.get("profit_ratio")),
                            "close_profit_abs": first_present_value(trade.get("close_profit_abs"), trade.get("profit_abs")),
                            "realized_profit": trade.get("realized_profit"),
                            "close_date": trade.get("close_date"),
                        }
                    )
                total = int(payload.get("total_trades") or len(page))
                offset += len(page)
                if offset >= total:
                    break
    except Exception:  # noqa: BLE001
        return None
    return records


def shift_trade_stats_from_records(records: list[dict[str, Any]], started_utc: datetime, stopped_utc: datetime) -> dict[str, Any]:
    stats = {"closed_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_roi": 0.0, "realized_pnl": 0.0}
    filtered: list[dict[str, Any]] = []
    for record in records:
        close_date = normalize_utc(resolve_optional_datetime(str(record.get("close_date") or "")))
        if close_date and started_utc <= close_date <= stopped_utc:
            filtered.append(record)
    if not filtered:
        return stats
    stats["closed_trades"] = len(filtered)
    stats["wins"] = sum(1 for record in filtered if parse_float(record.get("close_profit_abs")) > 0)
    stats["losses"] = max(0, len(filtered) - stats["wins"])
    stats["win_rate"] = percentage(stats["wins"], len(filtered))
    stats["avg_roi"] = percentage(sum(parse_float(record.get("close_profit")) for record in filtered), len(filtered))
    stats["realized_pnl"] = round(
        sum(parse_float(record.get("realized_profit")) or parse_float(record.get("close_profit_abs")) for record in filtered),
        4,
    )
    return stats


def shift_trade_stats_from_snapshots(candidate_id: int, started_utc: datetime, stopped_utc: datetime) -> dict[str, Any] | None:
    """Derive shift trade stats from the API-sourced runtime snapshots captured during the
    shift. Used when the bot has already been stopped (its REST API is down) by the time the
    post-shift review runs."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT captured_at, closed_trades, wins, losses, avg_roi, realized_pnl
            FROM dev_runtime_snapshots
            WHERE candidate_id = ?
            ORDER BY captured_at ASC, id ASC
            """,
            (candidate_id,),
        ).fetchall()
    selected = []
    for row in rows:
        captured_at = normalize_utc(resolve_optional_datetime(str(row["captured_at"])))
        if captured_at and started_utc <= captured_at <= stopped_utc:
            selected.append(row)
    if not selected:
        return None
    first, last = selected[0], selected[-1]
    closed_trades = max(0, int(last["closed_trades"] or 0) - int(first["closed_trades"] or 0))
    wins = max(0, int(last["wins"] or 0) - int(first["wins"] or 0))
    losses = max(0, int(last["losses"] or 0) - int(first["losses"] or 0))
    return {
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": percentage(wins, closed_trades) if closed_trades else 0.0,
        # avg_roi is a per-trade mean that cannot be reconstructed from cumulative counts;
        # fall back to the latest cumulative value captured from the API.
        "avg_roi": parse_float(last["avg_roi"]),
        "realized_pnl": round(parse_float(last["realized_pnl"]) - parse_float(first["realized_pnl"]), 4),
    }


def development_shift_trade_stats(candidate: dict[str, Any], started_at: datetime | None, stopped_at: datetime | None) -> dict[str, Any]:
    stats = {"closed_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_roi": 0.0, "realized_pnl": 0.0}
    started_utc = normalize_utc(started_at)
    stopped_utc = normalize_utc(stopped_at)
    if not started_utc or not stopped_utc:
        return stats
    # Dev trade DBs live on a per-bot Docker volume, not the host, so read trades over the
    # bot's REST API. While the bot is still running this is exact; once it has been stopped
    # (e.g. at scheduled shift end) fall back to the API-sourced runtime snapshots.
    records = fetch_closed_trades_via_api(candidate)
    if records is not None:
        return shift_trade_stats_from_records(records, started_utc, stopped_utc)
    snapshot_stats = shift_trade_stats_from_snapshots(int(candidate["id"]), started_utc, stopped_utc)
    if snapshot_stats is not None:
        return snapshot_stats
    return stats


def post_shift_grade(score: float) -> str:
    for threshold, grade in POST_SHIFT_GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def post_shift_evidence_confidence(
    style: str,
    runtime_hours: float,
    scheduled_hours: float,
    closed_trades: int,
    data_quality: str,
    heartbeat_ratio: float,
) -> str:
    profile = POST_SHIFT_STYLE_PROFILES[style]
    target_hours = scheduled_hours or profile["review_floor_hours"]
    if style == "patient" and runtime_hours >= max(4.0, target_hours * 0.85):
        confidence = "high"
    elif runtime_hours >= max(profile["review_floor_hours"], target_hours * 0.85) and closed_trades >= profile["min_closed_trades"]:
        confidence = "high"
    elif runtime_hours >= max(2.0, target_hours * 0.5):
        confidence = "medium"
    else:
        confidence = "low"
    if data_quality == "missing" or heartbeat_ratio < 25:
        return "low"
    if data_quality == "db_only" and confidence == "high":
        return "medium"
    return confidence


def score_post_shift_reliability(runtime_hours: float, scheduled_hours: float, data_quality: str, heartbeat_ratio: float) -> tuple[float, str]:
    denominator = scheduled_hours if scheduled_hours > 0 else 6.0
    coverage_ratio = min(1.0, runtime_hours / denominator) if denominator > 0 else 0.0
    score = coverage_ratio * 12.0
    if data_quality == "healthy":
        score += 5.0
    elif data_quality == "db_only":
        score += 3.0
    else:
        score += 1.0
    if heartbeat_ratio >= 75.0:
        score += 3.0
    elif heartbeat_ratio >= 40.0:
        score += 2.0
    elif heartbeat_ratio > 0:
        score += 1.0
    runtime_copy = f"{runtime_hours:.1f}/{scheduled_hours:.1f}h" if scheduled_hours > 0 else f"{runtime_hours:.1f}h"
    note = f"Covered {runtime_copy} with {data_quality} evidence and heartbeat coverage at {heartbeat_ratio:.0f}% during the session."
    return round(max(0.0, min(20.0, score)), 1), note


def score_post_shift_activity(style: str, trade_pace_per_24h: float, closed_trades: int, runtime_hours: float) -> tuple[float, str]:
    profile = POST_SHIFT_STYLE_PROFILES[style]
    if runtime_hours < min(2.0, profile["review_floor_hours"]):
        return 12.0, "Shift sample is still short, so tempo fit remains provisional."
    if style == "patient" and closed_trades == 0:
        return 20.0, "Stayed patient without forcing a trade just to create activity."
    if trade_pace_per_24h < profile["min_pace"]:
        gap = profile["min_pace"] - trade_pace_per_24h
        score = max(6.0, 22.0 - gap * 4.0)
        note = f"Trade pace lagged the {style} mandate ({trade_pace_per_24h:.1f}/24h vs target floor {profile['min_pace']:.1f})."
        return round(score, 1), note
    if trade_pace_per_24h > profile["max_pace"]:
        gap = trade_pace_per_24h - profile["max_pace"]
        score = max(4.0, 22.0 - gap * 2.5)
        note = f"Trade pace ran hotter than the {style} mandate ({trade_pace_per_24h:.1f}/24h vs ceiling {profile['max_pace']:.1f})."
        return round(score, 1), note
    distance = abs(trade_pace_per_24h - profile["ideal_pace"])
    score = max(14.0, 25.0 - distance * 2.0)
    note = f"Trade pace fit the {style} mandate ({trade_pace_per_24h:.1f}/24h around ideal {profile['ideal_pace']:.1f})."
    return round(score, 1), note


def score_post_shift_profitability(style: str, realized_pnl: float, win_rate: float, avg_roi: float, closed_trades: int, runtime_hours: float) -> tuple[float, str]:
    if closed_trades <= 0:
        if style == "patient" and runtime_hours >= 4.0:
            return 18.0, "No closed trades yet, which is still acceptable for a patient profile over a clean shift."
        return 9.0 if runtime_hours >= 4.0 else 12.0, "No closed trades yet, so the shift produced limited scoring evidence."
    score = 15.0
    if realized_pnl > 0:
        score += 8.0
    elif realized_pnl < 0:
        score -= 8.0
    if win_rate >= 60.0:
        score += 4.0
    elif win_rate >= 50.0:
        score += 2.0
    elif win_rate < 35.0:
        score -= 4.0
    if avg_roi >= 1.0:
        score += 3.0
    elif avg_roi >= 0.25:
        score += 1.0
    elif avg_roi < -0.5:
        score -= 3.0
    note = f"Closed-trade economics ended at {realized_pnl:+.2f} realized with {win_rate:.1f}% win rate and {avg_roi:.2f}% average ROI."
    return round(max(0.0, min(30.0, score)), 1), note


def score_post_shift_risk(max_drawdown: float, worst_open_trade: float, realized_pnl: float, closed_trades: int, win_rate: float) -> tuple[float, str]:
    score = 25.0
    if max_drawdown >= 12.0:
        score -= 12.0
    elif max_drawdown >= 8.0:
        score -= 8.0
    elif max_drawdown >= 4.0:
        score -= 4.0
    if worst_open_trade <= -8.0:
        score -= 8.0
    elif worst_open_trade <= -5.0:
        score -= 5.0
    elif worst_open_trade <= -3.0:
        score -= 2.0
    if realized_pnl < 0 and closed_trades >= 3 and win_rate < 40.0:
        score -= 3.0
    note = f"Risk posture peaked at {max_drawdown:.1f}% drawdown with worst open excursion of {worst_open_trade:.1f}%."
    return round(max(0.0, min(25.0, score)), 1), note


def shift_diagnostic_records(candidate: dict[str, Any], started_utc: datetime | None, stopped_utc: datetime | None) -> list[dict[str, Any]]:
    """Per-trade records for the just-completed shift, for clinical diagnostics. Prefers the
    durable dev_archived_trades store (works after the bot is stopped); falls back to a live
    API pull for manual reviews while the bot is still up. Records carry exit_reason, pair,
    is_short, profit_ratio/abs and the forced flag."""
    candidate_id = int(candidate.get("id") or 0)
    started = normalize_utc(started_utc)
    stopped = normalize_utc(stopped_utc)
    rows: list[dict[str, Any]] = []
    with closing(get_db()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM dev_archived_trades WHERE candidate_id = ? ORDER BY close_date", (candidate_id,)
        ).fetchall()]
    windowed: list[dict[str, Any]] = []
    for row in rows:
        closed_at = normalize_utc(resolve_optional_datetime(str(row.get("close_date") or "")))
        if started and closed_at and closed_at < started:
            continue
        if stopped and closed_at and closed_at > stopped + timedelta(hours=2):
            continue
        windowed.append(_dev_archived_trade_record(row))
    if windowed:
        return windowed
    # Live fallback (manual review while the bot is still serving its API).
    live = fetch_shift_closed_trades(candidate, started)
    if not live:
        return []
    out: list[dict[str, Any]] = []
    for record in live:
        out.append({
            "pair": str(record.get("pair") or ""),
            "exit_reason": str(record.get("exit_reason") or ""),
            "is_short": 1 if record.get("is_short") else 0,
            "forced": 1 if str(record.get("exit_reason") or "").lower() in DEV_FORCED_EXIT_REASONS else 0,
            "profit_ratio": parse_float(record.get("close_profit")),
            "profit_abs": parse_float(record.get("close_profit_abs")),
            "open_date": record.get("open_date"),
            "close_date": record.get("close_date"),
        })
    return out


def diagnose_shift_trades(records: list[dict[str, Any]], style: str) -> list[dict[str, Any]]:
    """Clinical, per-trade diagnosis of a shift. Returns ranked findings, each
    {code, severity, headline, fix}: headline states the mechanical fact (with numbers),
    fix is a concrete instruction the regenerator can act on. Forced (shift-bell) exits are
    excluded — they are the clock's doing, not the strategy's. Deterministic; no LLM."""
    profile = POST_SHIFT_STYLE_PROFILES.get(style, POST_SHIFT_STYLE_PROFILES["balanced"])
    strat = [r for r in records if not int(r.get("forced") or 0)]
    findings: list[dict[str, Any]] = []
    n = len(strat)
    if n == 0:
        return findings

    rois = [parse_float(r.get("profit_ratio")) * 100.0 for r in strat]
    pnls = [parse_float(r.get("profit_abs")) for r in strat]
    win_rois = [x for x in rois if x > 0]
    loss_rois = [x for x in rois if x <= 0]
    avg_win = stat_mean(win_rois) if win_rois else 0.0
    avg_loss = stat_mean(loss_rois) if loss_rois else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf")
    expectancy = stat_mean(rois)
    win_rate = percentage(len(win_rois), n) * 100.0

    def add(code: str, severity: str, headline: str, fix: str) -> None:
        findings.append({"code": code, "severity": severity, "headline": headline, "fix": fix})

    # 1. Expectancy — the bottom line of the trade distribution.
    if expectancy < 0:
        if win_rate >= 55.0:
            add("negative_expectancy_big_losers", "critical",
                f"Negative expectancy ({expectancy:+.2f}%/trade) despite a {win_rate:.0f}% hit rate — losers are too big.",
                "Cap downside: tighten the stop and/or size losers down. The entries hit often enough; the loss tail is what sinks the edge.")
        else:
            add("negative_expectancy_weak_entries", "critical",
                f"Negative expectancy ({expectancy:+.2f}%/trade) on a {win_rate:.0f}% hit rate — entries are low quality.",
                "Rework entry qualification (add a trend/volatility/volume confirmation) so fewer, higher-conviction setups fire.")
    elif expectancy > 0.15:
        add("positive_expectancy", "good",
            f"Positive expectancy ({expectancy:+.2f}%/trade) across {n} trades.",
            "Preserve the entry+exit core; keep edits parametric and narrow.")

    # 2. Payoff asymmetry — winners vs losers magnitude.
    if win_rois and loss_rois and payoff < 1.0:
        sev = "critical" if payoff < 0.5 else "warning"
        add("payoff_asymmetry", sev,
            f"Payoff skewed {payoff:.2f} (winners avg {avg_win:+.2f}% vs losers {avg_loss:+.2f}%): exits cut winners early and let losers run.",
            "Rebalance exits: raise ROI/trailing targets so winners run further, and shorten the loss stop or add a time-stop so losers close sooner — target a payoff above 1.0.")
    elif win_rois and loss_rois and payoff >= 1.8:
        add("payoff_healthy", "good",
            f"Strong payoff {payoff:.2f} (winners {avg_win:+.2f}% vs losers {avg_loss:+.2f}%).",
            "Protect the exit logic; do not flatten the winner tail in revision.")

    # 3. Exit-reason mix — which mechanism is actually closing trades.
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "pnl": 0.0})
    for r, pnl in zip(strat, pnls):
        key = str(r.get("exit_reason") or "unknown").lower()
        groups[key]["count"] += 1
        groups[key]["pnl"] += pnl
    stop_count = sum(g["count"] for k, g in groups.items() if k in POST_SHIFT_STOP_EXITS)
    stop_pnl = sum(g["pnl"] for k, g in groups.items() if k in POST_SHIFT_STOP_EXITS)
    roi_count = sum(g["count"] for k, g in groups.items() if k in POST_SHIFT_ROI_EXITS)
    roi_pnl = sum(g["pnl"] for k, g in groups.items() if k in POST_SHIFT_ROI_EXITS)
    if n >= 3 and stop_count / n >= 0.4 and stop_pnl < 0:
        sev = "critical" if stop_count / n >= 0.6 else "warning"
        add("stop_dominant", sev,
            f"Stop-loss exits dominate ({stop_count / n:.0%} of closes, {stop_pnl:+.2f} PnL): entries are early or the stop is too tight for the noise.",
            "Either add an entry-timing confirmation so you enter after the adverse swing, or widen the stop and cut size so ordinary volatility stops stopping you out.")
    elif n >= 4 and roi_count / n >= 0.6 and roi_pnl > 0:
        add("roi_clean", "good",
            f"ROI exits carry the book ({roi_count / n:.0%} of closes, {roi_pnl:+.2f}): the take-profit logic is working.",
            "Keep the ROI ladder; if anything, test extending the final rung to capture more of the winner tail.")
    # Custom strategy-authored exits that bleed.
    for key, g in sorted(groups.items(), key=lambda kv: kv[1]["pnl"]):
        if key in POST_SHIFT_STOP_EXITS or key in POST_SHIFT_ROI_EXITS or key == "unknown":
            continue
        if g["count"] >= 2 and g["pnl"] < -0.01 and g["pnl"] <= stop_pnl:
            add(f"custom_exit_bleed:{key}", "warning",
                f"Custom exit '{key}' bled {g['pnl']:+.2f} over {int(g['count'])} trades.",
                f"Re-examine the '{key}' exit rule — it is realizing losses. Gate it behind a confirmation, raise its threshold, or remove it.")
            break

    # 4. Directional leak — long vs short attribution.
    longs = [(r, p) for r, p in zip(strat, pnls) if not int(r.get("is_short") or 0)]
    shorts = [(r, p) for r, p in zip(strat, pnls) if int(r.get("is_short") or 0)]
    if longs and shorts:
        long_pnl = sum(p for _, p in longs)
        short_pnl = sum(p for _, p in shorts)
        for side, side_pnl, side_n, other_pnl in (
            ("short", short_pnl, len(shorts), long_pnl),
            ("long", long_pnl, len(longs), short_pnl),
        ):
            if side_pnl < 0 and other_pnl > 0 and abs(side_pnl) >= 0.5 * abs(other_pnl) and side_n >= 2:
                add(f"directional_leak:{side}", "warning",
                    f"Directional leak: {side} trades lost {side_pnl:+.2f} over {side_n} while the other side made {other_pnl:+.2f}.",
                    f"Re-qualify or disable {side} entries this regime; the edge is one-directional here. Add a regime filter that only takes {('longs' if side == 'short' else 'shorts')}.")
                break

    # 5. Pair concentration / worst symbol.
    pair_groups: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "pnl": 0.0})
    for r, pnl in zip(strat, pnls):
        pg = pair_groups[str(r.get("pair") or "?")]
        pg["count"] += 1
        pg["pnl"] += pnl
    if n >= 4:
        top_pair, top = max(pair_groups.items(), key=lambda kv: kv[1]["count"])
        if top["count"] / n >= 0.5:
            add("pair_concentration", "warning",
                f"Over-concentrated: {top_pair} is {top['count'] / n:.0%} of trades ({top['pnl']:+.2f}).",
                "Widen the pair universe or cap per-pair frequency/exposure so one symbol cannot dominate the book.")
    worst_pair, worst = min(pair_groups.items(), key=lambda kv: kv[1]["pnl"])
    if worst["pnl"] < 0 and worst["count"] >= 2 and abs(worst["pnl"]) >= 0.3 * (abs(sum(pnls)) or 1.0):
        add("worst_pair_drag", "info",
            f"{worst_pair} is a consistent drag ({worst['pnl']:+.2f} over {int(worst['count'])} trades).",
            f"Blacklist or down-weight {worst_pair}; its behavior does not fit this strategy's signal.")

    # 6. Hold-time fit vs the mandate's tempo.
    holds = [_ml_duration_minutes(r) for r in strat]
    holds = [h for h in holds if h > 0]
    if holds:
        med_hold = stat_median(holds)
        if style in {"patient", "balanced"} and med_hold < 20.0:
            add("scalp_drift", "warning",
                f"Holds are scalp-length (median {med_hold:.0f}m) for a {style} mandate.",
                "Lengthen the horizon: raise the minimal_roi time floor or require a higher-timeframe confirmation so it stops scalping noise.")
        elif style == "aggressive" and med_hold > 240.0:
            add("aggressive_too_slow", "info",
                f"Holds run long (median {med_hold:.0f}m) for an aggressive mandate.",
                "Add a faster take-profit or time-stop so capital recycles at the intended tempo.")

    # 7. Loss clustering — regime brake.
    streak = max_streak = 0
    for x in rois:
        if x <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    if max_streak >= 4:
        add("loss_cluster", "warning",
            f"Lost {max_streak} trades in a row — no regime brake.",
            "Add a cooldown or drawdown-aware throttle that pauses new entries after consecutive losers.")

    # 8. Tail risk — worst single excursion.
    if rois:
        worst_roi = min(rois)
        if worst_roi <= -5.0:
            add("tail_risk", "warning",
                f"Tail risk: worst single trade closed {worst_roi:.1f}%.",
                "Enforce a hard per-trade stop near the strategy's risk budget so no single trade takes an outsized loss.")

    findings.sort(key=lambda f: POST_SHIFT_SEVERITY_WEIGHT.get(f["severity"], 0), reverse=True)
    return findings


def recommend_post_shift_action(
    candidate: dict[str, Any],
    score: float,
    evidence_confidence: str,
    reliability_score: float,
    activity_score: float,
    profitability_score: float,
    risk_score: float,
    recent_reviews: list[dict[str, Any]],
) -> tuple[str, str]:
    weak_streak = 0
    for review in recent_reviews:
        if str(review.get("grade") or "") in {"D", "F"} or str(review.get("decision_bucket") or "") in {"overhaul", "archive"}:
            weak_streak += 1
        else:
            break
    tier = candidate_competition_tier(candidate)
    if reliability_score < 12.0 and evidence_confidence == "low":
        return "hold", "Fix runtime reliability and collect a cleaner full shift before changing strategy logic."
    if score >= 85.0 and evidence_confidence in {"medium", "high"}:
        if tier in {"candidate_6h", "bootcamp"}:
            return "promote", "Promote this build up a tier or expand its runtime exposure. Only minor cleanup is justified."
        return "hold", "Carry the build forward intact and confirm the edge over another scheduled shift before rewriting."
    if score >= 70.0:
        return "tweak", "Keep the core thesis and make a narrow parameter adjustment before the next shift."
    if score >= 55.0:
        if min(activity_score, profitability_score, risk_score) < 14.0:
            return "tweak", "Make a focused correction in the weakest rubric area rather than rewriting the whole strategy."
        return "hold", "Collect one more shift before mutating a stable but mid-table build."
    if score >= 40.0:
        if evidence_confidence == "low":
            return "hold", "The sample is still thin. Gather one more clean shift before making a large rewrite."
        return "overhaul", "The shift failed the contract in more than one category and needs a broader redesign."
    if weak_streak >= 1 and evidence_confidence == "high":
        return "archive", "Multiple weak full-shift report cards suggest shelving this branch and preserving only the notes."
    return "overhaul", "This version is not competitive enough. Regenerate around a materially different entry or risk structure."


def build_post_shift_mutation_brief(
    decision_bucket: str,
    style: str,
    trade_pace_per_24h: float,
    closed_trades: int,
    win_rate: float,
    activity_score: float,
    profitability_score: float,
    risk_score: float,
    reliability_score: float,
    findings: list[dict[str, Any]] | None = None,
) -> str:
    profile = POST_SHIFT_STYLE_PROFILES[style]
    suggestions: list[str] = []
    if reliability_score < 12.0:
        suggestions.append("Fix runtime reliability and API coverage before regenerating; the shift evidence is not clean enough yet.")
    # Clinical, trade-derived fixes take priority — these are specific to what actually
    # happened this shift, so the regenerator gets targeted instructions instead of platitudes.
    elif findings:
        actionable = [f for f in findings if f.get("severity") in {"critical", "warning"}]
        for finding in actionable[:3]:
            fix = str(finding.get("fix") or "").strip()
            if fix:
                suggestions.append(fix)
        if not suggestions:  # only good/info findings → strategy is healthy
            suggestions.append("No mechanical fault stood out this shift — keep edits parametric and protect the working core.")
    else:
        if trade_pace_per_24h < profile["min_pace"]:
            if style == "aggressive":
                suggestions.append("Loosen entry confirmation or widen the pair universe so the build can actually express its aggressive brief.")
            elif style == "patient":
                suggestions.append("Keep patience explicit, but relax one gating filter so the strategy is selective instead of dormant.")
            else:
                suggestions.append("Relax one entry filter or slightly widen the universe so the strategy can find enough valid setups.")
        elif trade_pace_per_24h > profile["max_pace"]:
            suggestions.append("Tighten entry gating or shrink the pair universe so the strategy stops overtrading its brief.")
        if profitability_score < 14.0:
            if closed_trades and win_rate >= 50.0:
                suggestions.append("Preserve the entry idea and rework the exit logic so winners pay more cleanly.")
            elif closed_trades:
                suggestions.append("Rework entry qualification first; the current setups are not high quality enough.")
            else:
                suggestions.append("Collect another full shift before changing trade logic because there is not enough closed-trade evidence.")
        if risk_score < 14.0:
            suggestions.append("Lower per-trade risk with tighter stop logic, smaller exposure, or drawdown-aware throttles.")
        if activity_score < 14.0 and profitability_score >= 14.0:
            suggestions.append("Keep the profitable core intact and adjust only the pacing controls.")
    if decision_bucket == "promote":
        suggestions.append("Promote the current version and keep any edits cosmetic before the next tier.")
    elif decision_bucket == "hold":
        suggestions.append("Carry the current version forward unchanged and gather another scheduled shift.")
    elif decision_bucket == "archive":
        suggestions.append("Archive this branch and preserve the hypothesis plus failure notes for future reuse.")
    else:
        suggestions.append("If you regenerate, feed these report-card notes into the next prompt instead of asking for a generic replacement.")
    unique: list[str] = []
    for item in suggestions:
        if item not in unique:
            unique.append(item)
    # Keep 4 when clinical findings are present so the decision-bucket intent survives
    # alongside the three targeted fixes.
    return " ".join(unique[: 4 if findings else 3])


def store_development_post_shift_review(payload: dict[str, Any]) -> dict[str, Any]:
    now = iso_now()
    payload = dict(payload)
    existing = get_development_post_shift_review_by_key(str(payload.get("review_key") or ""))
    created_at = str(existing.get("created_at") or now) if existing else now
    with closing(get_db()) as conn:
        if existing:
            conn.execute(
                """
                UPDATE dev_post_shift_reviews
                SET review_scope = ?, tier = ?, shift_code = ?, strategy_style = ?,
                    session_started_at = ?, session_stopped_at = ?, runtime_hours = ?, scheduled_hours = ?,
                    closed_trades = ?, trade_pace_per_24h = ?, win_rate = ?, avg_roi = ?, realized_pnl = ?,
                    forced_exits = ?, forced_realized_pnl = ?, strategy_realized_pnl = ?,
                    max_drawdown = ?, worst_open_trade = ?, data_quality = ?, overall_score = ?, grade = ?,
                    decision_bucket = ?, evidence_confidence = ?, recommendation = ?, summary = ?,
                    mutation_brief = ?, rubric_json = ?, diagnostics_json = ?, updated_at = ?
                WHERE review_key = ?
                """,
                (
                    payload.get("review_scope", "scheduled_shift_end"),
                    payload.get("tier", ""),
                    payload.get("shift_code", ""),
                    payload.get("strategy_style", "balanced"),
                    payload.get("session_started_at"),
                    payload.get("session_stopped_at"),
                    parse_float(payload.get("runtime_hours")),
                    parse_float(payload.get("scheduled_hours")),
                    int(payload.get("closed_trades") or 0),
                    parse_float(payload.get("trade_pace_per_24h")),
                    parse_float(payload.get("win_rate")),
                    parse_float(payload.get("avg_roi")),
                    parse_float(payload.get("realized_pnl")),
                    int(payload.get("forced_exits") or 0),
                    parse_float(payload.get("forced_realized_pnl")),
                    parse_float(payload.get("strategy_realized_pnl")),
                    parse_float(payload.get("max_drawdown")),
                    parse_float(payload.get("worst_open_trade")),
                    payload.get("data_quality", ""),
                    parse_float(payload.get("overall_score")),
                    payload.get("grade", ""),
                    payload.get("decision_bucket", "hold"),
                    payload.get("evidence_confidence", "low"),
                    payload.get("recommendation", ""),
                    payload.get("summary", ""),
                    payload.get("mutation_brief", ""),
                    payload.get("rubric_json", "[]"),
                    payload.get("diagnostics_json", "[]"),
                    now,
                    payload.get("review_key", ""),
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO dev_post_shift_reviews (
                    review_key, candidate_id, review_scope, tier, shift_code, strategy_style,
                    session_started_at, session_stopped_at, runtime_hours, scheduled_hours,
                    closed_trades, trade_pace_per_24h, win_rate, avg_roi, realized_pnl,
                    forced_exits, forced_realized_pnl, strategy_realized_pnl,
                    max_drawdown, worst_open_trade, data_quality, overall_score, grade,
                    decision_bucket, evidence_confidence, recommendation, summary,
                    mutation_brief, rubric_json, diagnostics_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("review_key", ""),
                    int(payload.get("candidate_id") or 0),
                    payload.get("review_scope", "scheduled_shift_end"),
                    payload.get("tier", ""),
                    payload.get("shift_code", ""),
                    payload.get("strategy_style", "balanced"),
                    payload.get("session_started_at"),
                    payload.get("session_stopped_at"),
                    parse_float(payload.get("runtime_hours")),
                    parse_float(payload.get("scheduled_hours")),
                    int(payload.get("closed_trades") or 0),
                    parse_float(payload.get("trade_pace_per_24h")),
                    parse_float(payload.get("win_rate")),
                    parse_float(payload.get("avg_roi")),
                    parse_float(payload.get("realized_pnl")),
                    int(payload.get("forced_exits") or 0),
                    parse_float(payload.get("forced_realized_pnl")),
                    parse_float(payload.get("strategy_realized_pnl")),
                    parse_float(payload.get("max_drawdown")),
                    parse_float(payload.get("worst_open_trade")),
                    payload.get("data_quality", ""),
                    parse_float(payload.get("overall_score")),
                    payload.get("grade", ""),
                    payload.get("decision_bucket", "hold"),
                    payload.get("evidence_confidence", "low"),
                    payload.get("recommendation", ""),
                    payload.get("summary", ""),
                    payload.get("mutation_brief", ""),
                    payload.get("rubric_json", "[]"),
                    payload.get("diagnostics_json", "[]"),
                    created_at,
                    now,
                ),
            )
        conn.commit()
    return get_development_post_shift_review_by_key(str(payload.get("review_key") or "")) or {}


def update_development_post_shift_follow_up(review_key: str, status: str, message: str = "") -> dict[str, Any] | None:
    if not review_key:
        return None
    now = iso_now()
    existing = get_development_post_shift_review_by_key(review_key)
    queued_at = existing.get("follow_up_queued_at") if existing else None
    if status == "queued":
        queued_at = now
    with closing(get_db()) as conn:
        conn.execute(
            """
            UPDATE dev_post_shift_reviews
            SET follow_up_status = ?, follow_up_message = ?, follow_up_queued_at = ?, updated_at = ?
            WHERE review_key = ?
            """,
            (status, message, queued_at, now, review_key),
        )
        conn.commit()
    return get_development_post_shift_review_by_key(review_key)


def auto_review_regeneration_decisions() -> set[str]:
    raw = get_setting("auto_review_regeneration_decisions", "tweak,overhaul")
    values = {
        token.strip().lower().replace("-", "_")
        for token in re.split(r"[,\s]+", raw)
        if token.strip()
    }
    return values or {"tweak", "overhaul"}


def maybe_queue_post_shift_follow_up(candidate: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any] | None:
    if not review:
        return review
    decision = str(review.get("decision_bucket") or "").strip().lower().replace("-", "_")
    review_key = str(review.get("review_key") or "")
    follow_up_status = str(review.get("follow_up_status") or "")
    if get_setting("auto_review_regeneration_enabled", "true").lower() != "true":
        if follow_up_status != "disabled":
            return update_development_post_shift_follow_up(review_key, "disabled", "Automatic regeneration is disabled.")
        return review
    if decision not in auto_review_regeneration_decisions():
        if follow_up_status != "not_applicable":
            return update_development_post_shift_follow_up(review_key, "not_applicable", f"{decision or 'hold'} does not auto-queue regeneration.")
        return review
    if follow_up_status in {"queued", "already_queued", "skipped"}:
        return review
    candidate_id = int(candidate.get("id") or 0)
    if int(candidate.get("protected") or 0):
        return update_development_post_shift_follow_up(review_key, "skipped", "Protected Prospect — strategy is immutable until you approve a manual regeneration.")
    lifecycle_state = str(candidate.get("lifecycle_state") or "")
    if lifecycle_state in {"cut_archived", "archived"}:
        return update_development_post_shift_follow_up(review_key, "skipped", "Archived candidates do not auto-regenerate.")
    generation_status = str(candidate.get("generation_status") or "")
    if generation_status in {"queued", "generating"}:
        return update_development_post_shift_follow_up(review_key, "already_queued", f"Candidate is already {generation_status}.")
    cooldown_hours = candidate_auto_update_cooldown_hours(candidate)
    generated_at = normalize_utc(resolve_optional_datetime(str(candidate.get("generated_at") or "")))
    if cooldown_hours > 0 and generated_at and hours_between(generated_at, utc_now()) < cooldown_hours:
        timeframe = str(candidate.get("suggested_timeframe") or candidate.get("timeframe") or "longer timeframe")
        return update_development_post_shift_follow_up(
            review_key,
            "skipped",
            f"{timeframe} candidates wait about {int(cooldown_hours)}h between automatic updates.",
        )
    try:
        queue_candidate_strategy_generation(
            candidate_id,
            force=True,
            generation_trigger=f"post_shift_review:{review_key}",
            auto_apply_generated_strategy=True,
        )
        development_runtime_event(
            candidate_id,
            "generation",
            "Auto-regeneration queued from post-shift review.",
            f"{review.get('grade', '')} · {decision.replace('_', ' ')} · {review.get('recommendation', '')}",
        )
        return update_development_post_shift_follow_up(review_key, "queued", f"Queued regeneration from {decision.replace('_', ' ')} review.")
    except Exception as exc:  # noqa: BLE001
        development_runtime_event(candidate_id, "generation", "Auto-regeneration failed.", str(exc))
        return update_development_post_shift_follow_up(review_key, "error", str(exc))


def record_development_post_shift_review(
    candidate: dict[str, Any],
    review_scope: str = "scheduled_shift_end",
    force: bool = False,
) -> dict[str, Any] | None:
    candidate_id = int(candidate["id"])
    session = latest_completed_development_session(candidate_id)
    if not session or not session.get("stopped_at"):
        return None
    review_key = f"{candidate_id}:{session.get('started_at', '')}:{session.get('stopped_at', '')}"
    existing = get_development_post_shift_review_by_key(review_key)
    if existing and not force:
        return existing
    started_at = normalize_utc(resolve_optional_datetime(str(session.get("started_at") or "")))
    stopped_at = normalize_utc(resolve_optional_datetime(str(session.get("stopped_at") or "")))
    if not started_at or not stopped_at:
        return None
    runtime_hours = parse_float(session.get("duration_hours")) or hours_between(started_at, stopped_at)
    style = infer_candidate_strategy_style(candidate)
    tier_name = candidate_competition_tier(candidate)
    scheduled_hours = 6.0 if tier_name == "candidate_6h" else 12.0 if tier_name == "prospect_12h" else runtime_hours
    trade_stats = development_shift_trade_stats(candidate, started_at, stopped_at)
    snapshot_metrics = development_shift_snapshot_metrics(candidate_id, started_at, stopped_at)
    closed_trades = int(trade_stats.get("closed_trades") or 0)
    trade_pace_per_24h = safe_rate_per_hour(closed_trades, runtime_hours) * 24.0 if runtime_hours > 0 else 0.0
    realized_pnl = parse_float(trade_stats.get("realized_pnl"))
    if closed_trades <= 0 and parse_float(snapshot_metrics.get("realized_pnl_delta")):
        realized_pnl = parse_float(snapshot_metrics.get("realized_pnl_delta"))
    win_rate = parse_float(trade_stats.get("win_rate"))
    avg_roi = parse_float(trade_stats.get("avg_roi"))
    heartbeat_ratio = parse_float(snapshot_metrics.get("heartbeat_ratio"))
    max_drawdown = max(parse_float(snapshot_metrics.get("max_drawdown")), parse_float(candidate.get("max_drawdown")))
    worst_open_trade = min(parse_float(snapshot_metrics.get("worst_open_trade")), parse_float(candidate.get("worst_open_trade")))
    data_quality = str(candidate.get("data_quality") or "unknown")
    # Shift-boundary forced exits (captured live at wind-down) are the clock's doing,
    # not the strategy's — grade economics on the strategy-chosen exits and report the
    # forced liquidations separately, so the clock never lands in Dany's report card.
    forced_exits = 0
    forced_realized_pnl = 0.0
    grade_realized_pnl, grade_win_rate, grade_avg_roi, grade_trades = realized_pnl, win_rate, avg_roi, closed_trades
    split_key = str(candidate.get("last_shift_split_key") or "")
    if split_key and split_key == str(session.get("started_at") or ""):
        forced_exits = int(candidate.get("last_shift_forced_exits") or 0)
        forced_realized_pnl = parse_float(candidate.get("last_shift_forced_pnl"))
        strategy_trades = int(candidate.get("last_shift_strategy_trades") or 0)
        strategy_wins = int(candidate.get("last_shift_strategy_wins") or 0)
        grade_realized_pnl = parse_float(candidate.get("last_shift_strategy_pnl"))
        grade_win_rate = percentage(strategy_wins, strategy_trades) if strategy_trades else 0.0
        grade_avg_roi = parse_float(candidate.get("last_shift_strategy_avg_roi"))
        grade_trades = strategy_trades
    reliability_score, reliability_note = score_post_shift_reliability(runtime_hours, scheduled_hours, data_quality, heartbeat_ratio)
    activity_score, activity_note = score_post_shift_activity(style, trade_pace_per_24h, closed_trades, runtime_hours)
    profitability_score, profitability_note = score_post_shift_profitability(style, grade_realized_pnl, grade_win_rate, grade_avg_roi, grade_trades, runtime_hours)
    if forced_exits:
        profitability_note += f" ({forced_exits} shift-expiry liquidation(s) held out of the grade; {forced_realized_pnl:+.2f} forced P&L.)"
    risk_score, risk_note = score_post_shift_risk(max_drawdown, worst_open_trade, realized_pnl, closed_trades, win_rate)
    overall_score = round(reliability_score + activity_score + profitability_score + risk_score, 1)
    grade = post_shift_grade(overall_score)
    evidence_confidence = post_shift_evidence_confidence(style, runtime_hours, scheduled_hours, closed_trades, data_quality, heartbeat_ratio)
    recent_reviews = [
        row
        for row in development_post_shift_reviews(candidate_id, limit=3)
        if str(row.get("review_key") or "") != review_key
    ]
    decision_bucket, recommendation = recommend_post_shift_action(
        candidate,
        overall_score,
        evidence_confidence,
        reliability_score,
        activity_score,
        profitability_score,
        risk_score,
        recent_reviews,
    )
    # Clinical per-trade diagnosis — the targeted, varied evidence that drives the brief.
    diagnostic_records = shift_diagnostic_records(candidate, started_at, stopped_at)
    findings = diagnose_shift_trades(diagnostic_records, style)
    top_faults = [f for f in findings if f.get("severity") in {"critical", "warning"}]
    if top_faults:
        # Lead the recommendation with the single most material mechanical fault.
        recommendation = f"{recommendation} Primary fault: {top_faults[0]['headline']}"
    mutation_brief = build_post_shift_mutation_brief(
        decision_bucket,
        style,
        trade_pace_per_24h,
        closed_trades,
        win_rate,
        activity_score,
        profitability_score,
        risk_score,
        reliability_score,
        findings,
    )
    # Fold the sharpest economics/risk findings into the rubric notes so the report card
    # reads clinically rather than generically.
    econ_fault = next((f for f in findings if f["code"] in {
        "negative_expectancy_big_losers", "negative_expectancy_weak_entries", "payoff_asymmetry",
        "stop_dominant", "directional_leak:long", "directional_leak:short", "pair_concentration",
    } or f["code"].startswith("custom_exit_bleed")), None)
    if econ_fault:
        profitability_note += f" {econ_fault['headline']}"
    risk_fault = next((f for f in findings if f["code"] in {"loss_cluster", "tail_risk"}), None)
    if risk_fault:
        risk_note += f" {risk_fault['headline']}"
    behavior_fault = next((f for f in findings if f["code"] in {"scalp_drift", "aggressive_too_slow"}), None)
    if behavior_fault:
        activity_note += f" {behavior_fault['headline']}"
    rubric = [
        {"label": "Reliability", "score": reliability_score, "max_score": 20, "note": reliability_note},
        {"label": "Behavior Fit", "score": activity_score, "max_score": 25, "note": activity_note},
        {"label": "Economics", "score": profitability_score, "max_score": 30, "note": profitability_note},
        {"label": "Risk", "score": risk_score, "max_score": 25, "note": risk_note},
    ]
    forced_clause = (
        f"{forced_exits} trade(s) were force-closed at the shift bell ({forced_realized_pnl:+.2f} forced P&L), "
        f"graded on {grade_trades} strategy-chosen exit(s) worth {grade_realized_pnl:+.2f}. "
        if forced_exits else ""
    )
    summary = (
        f"{candidate.get('name', 'Candidate')} posted a {grade} shift ({overall_score:.1f}/100). "
        f"It covered {runtime_hours:.1f}/{scheduled_hours:.1f} scheduled hours, closed {closed_trades} trades "
        f"({trade_pace_per_24h:.1f}/24h pace), finished {realized_pnl:+.2f} realized, and hit {max_drawdown:.1f}% max drawdown. "
        f"{forced_clause}"
        f"Recommendation: {decision_bucket.replace('_', ' ')}. Evidence confidence: {evidence_confidence}."
    )
    stored = store_development_post_shift_review(
        {
            "review_key": review_key,
            "candidate_id": candidate_id,
            "review_scope": review_scope or str(session.get("stop_reason") or "scheduled_shift_end"),
            "tier": str(candidate.get("tier") or ""),
            "shift_code": str(candidate.get("shift_code") or ""),
            "strategy_style": style,
            "session_started_at": session.get("started_at"),
            "session_stopped_at": session.get("stopped_at"),
            "runtime_hours": runtime_hours,
            "scheduled_hours": scheduled_hours,
            "closed_trades": closed_trades,
            "trade_pace_per_24h": round(trade_pace_per_24h, 2),
            "win_rate": round(win_rate, 2),
            "avg_roi": round(avg_roi, 2),
            "realized_pnl": round(realized_pnl, 4),
            "forced_exits": forced_exits,
            "forced_realized_pnl": round(forced_realized_pnl, 4),
            "strategy_realized_pnl": round(grade_realized_pnl, 4),
            "max_drawdown": round(max_drawdown, 2),
            "worst_open_trade": round(worst_open_trade, 2),
            "data_quality": data_quality,
            "overall_score": overall_score,
            "grade": grade,
            "decision_bucket": decision_bucket,
            "evidence_confidence": evidence_confidence,
            "recommendation": recommendation,
            "summary": summary,
            "mutation_brief": mutation_brief,
            "rubric_json": json.dumps(rubric, indent=2),
            "diagnostics_json": json.dumps(findings),
        }
    )
    action_copy = "updated" if existing else "created"
    if stored:
        development_runtime_event(
            candidate_id,
            "post_shift_review",
            f"Post-shift review {action_copy}.",
            f"{stored.get('grade', '')} · {stored.get('decision_label', '')} · {stored.get('recommendation', '')}",
        )
        stored = maybe_queue_post_shift_follow_up(candidate, stored) or stored
    return stored or None


def open_development_runtime_session(candidate_id: int, started_at: str | None = None) -> None:
    started_at = started_at or iso_now()
    try:
        with closing(get_db()) as conn:
            existing = conn.execute(
                "SELECT id FROM dev_runtime_sessions WHERE candidate_id = ? AND stopped_at IS NULL ORDER BY id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
            if existing:
                return
            conn.execute(
                """
                INSERT INTO dev_runtime_sessions (candidate_id, started_at, stopped_at, duration_hours, stop_reason, created_at)
                VALUES (?, ?, NULL, 0, '', ?)
                """,
                (candidate_id, started_at, iso_now()),
            )
            conn.commit()
    except sqlite3.OperationalError:
        return


def close_development_runtime_session(candidate_id: int, stop_reason: str, stopped_at: str | None = None) -> None:
    stopped_at = stopped_at or iso_now()
    try:
        with closing(get_db()) as conn:
            row = conn.execute(
                "SELECT id, started_at FROM dev_runtime_sessions WHERE candidate_id = ? AND stopped_at IS NULL ORDER BY id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
            if not row:
                return
            started_at = resolve_optional_datetime(str(row["started_at"]))
            stopped_dt = resolve_optional_datetime(stopped_at)
            duration_hours = 0.0
            if started_at and stopped_dt:
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=UTC)
                if stopped_dt.tzinfo is None:
                    stopped_dt = stopped_dt.replace(tzinfo=UTC)
                duration_hours = max(0.0, (stopped_dt.astimezone(UTC) - started_at.astimezone(UTC)).total_seconds() / 3600.0)
            conn.execute(
                """
                UPDATE dev_runtime_sessions
                SET stopped_at = ?, duration_hours = ?, stop_reason = ?
                WHERE id = ?
                """,
                (stopped_at, duration_hours, stop_reason, int(row["id"])),
            )
            conn.commit()
    except sqlite3.OperationalError:
        return


def sync_development_runtime_session(candidate_id: int, previous_status: str, new_status: str, stop_reason: str = "") -> None:
    if previous_status != "running" and new_status == "running":
        open_development_runtime_session(candidate_id)
    elif previous_status == "running" and new_status != "running":
        close_development_runtime_session(candidate_id, stop_reason or "state_change")


def persist_development_snapshot(candidate: dict[str, Any]) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO dev_runtime_snapshots (
                candidate_id, captured_at, runtime_status, heartbeat_ok, data_quality,
                equity, closed_trades, wins, losses, avg_roi, realized_pnl, unrealized_pnl, worst_open_trade,
                max_drawdown, last_trade_at, status_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(candidate["id"]),
                iso_now(),
                candidate.get("runtime_status", "paused"),
                int(bool(candidate.get("heartbeat_ok"))),
                candidate.get("data_quality", "unknown"),
                parse_float(candidate.get("equity")),
                int(candidate.get("closed_trades") or 0),
                int(candidate.get("wins") or 0),
                int(candidate.get("losses") or 0),
                parse_float(candidate.get("avg_roi")),
                parse_float(candidate.get("realized_pnl")),
                parse_float(candidate.get("unrealized_pnl")),
                parse_float(candidate.get("worst_open_trade")),
                parse_float(candidate.get("max_drawdown")),
                candidate.get("last_trade_at"),
                candidate.get("status_detail", ""),
            ),
        )
        conn.commit()


def resolve_optional_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def runtime_cooldown_passed(raw: str | None, seconds: int) -> bool:
    moment = resolve_optional_datetime(raw)
    if not moment:
        return True
    now = utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (now - moment.astimezone(UTC)).total_seconds() >= seconds


def candidate_start_offset_seconds(candidate: dict[str, Any]) -> int:
    if candidate.get("tier") not in {"six_hour", "twelve_hour"} or not candidate.get("shift_code"):
        return 0
    cohort = [
        row for row in development_candidate_rows(candidate["tier"])
        if row.get("shift_code") == candidate.get("shift_code") and row.get("lifecycle_state") != "cut_archived"
    ]
    for index, row in enumerate(cohort):
        if int(row["id"]) == int(candidate["id"]):
            return min(90, 30 + index * 12)
    return 30


def candidate_target_runtime(candidate: dict[str, Any], now_local: datetime) -> tuple[bool, str, str]:
    lifecycle = candidate.get("lifecycle_state")
    tier = candidate.get("tier")
    override = candidate.get("override_mode", "auto")

    if lifecycle == "cut_archived" or tier == "archived":
        return False, "archived", "Archived candidates never run."
    if candidate.get("assembly_status") != "assembled":
        return False, "paused", "Needs instance assembly."
    if override == "paused":
        return False, "paused", "Manual pause is active."
    if override == "force_stopped":
        return False, "paused", "Manual stop override is active."
    if override == "force_running":
        return True, "running", "Manual start override is active."
    if lifecycle in {"draft_idea", "generating_strategy", "implemented", "reviewed", "instance_assembled", "drafted", "draft_eligible"}:
        return False, "paused", "This lifecycle state is intentionally off-schedule."
    if tier == "bootcamp":
        return False, "paused", "Bootcamp runs manually in v1 and is excluded from shift automation."
    if tier in {"six_hour", "twelve_hour"}:
        if not candidate.get("shift_code"):
            return False, "paused", "Needs shift assignment."
        active_shift = active_shift_for_tier(tier, now_local)
        if not active_shift or active_shift["code"] != candidate.get("shift_code"):
            return False, "off-shift", "Outside the candidate's assigned shift."
        if active_shift["seconds_since_start"] < candidate_start_offset_seconds(candidate):
            return False, "off-shift", "Waiting for staggered shift start."
        return True, "running", f"Assigned {active_shift['label']} ({active_shift['window']})."
    return False, "paused", "No automated runtime tier is assigned."


def development_db_metrics(db_path: Path | None) -> dict[str, Any]:
    metrics = {
        "closed_trades": 0,
        "open_trades": 0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "worst_open_trade": 0.0,
        "last_trade_at": None,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_roi": 0.0,
        "champion_exits": 0,
    }
    if not db_path or not db_path.exists():
        return metrics
    try:
        with sqlite3.connect(db_path) as source:
            source.row_factory = sqlite3.Row
            rows = source.execute(
                """
                SELECT is_open, close_profit, close_profit_abs, realized_profit, open_date, close_date, exit_reason
                FROM trades
                """
            ).fetchall()
    except Exception:
        return metrics
    closed = [row for row in rows if not int(row["is_open"])]
    open_rows = [row for row in rows if int(row["is_open"])]
    metrics["closed_trades"] = len(closed)
    metrics["open_trades"] = len(open_rows)
    metrics["realized_pnl"] = round(sum(parse_float(row["realized_profit"]) or parse_float(row["close_profit_abs"]) for row in closed), 4)
    metrics["unrealized_pnl"] = round(sum(parse_float(row["close_profit_abs"]) for row in open_rows), 4)
    metrics["worst_open_trade"] = min((parse_float(row["close_profit"]) * 100.0 for row in open_rows), default=0.0)
    metrics["wins"] = sum(1 for row in closed if parse_float(row["close_profit_abs"]) > 0)
    metrics["losses"] = sum(1 for row in closed if parse_float(row["close_profit_abs"]) <= 0)
    metrics["win_rate"] = percentage(metrics["wins"], len(closed))
    metrics["avg_roi"] = percentage(sum(parse_float(row["close_profit"]) for row in closed), len(closed))
    metrics["champion_exits"] = sum(
        1 for row in closed if str(row["exit_reason"] or "") in {"champ_dynamic_roi", "champ_dynamic_roi_hit"}
    )
    metrics["last_trade_at"] = max(
        [str(row["close_date"]) for row in closed if row["close_date"]] + [str(row["open_date"]) for row in open_rows if row["open_date"]],
        default=None,
    )
    return metrics


def development_api_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "heartbeat_ok": 0,
        "status_detail": "API not configured.",
        "equity": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "closed_trades": 0,
        "open_trades": 0,
        "worst_open_trade": 0.0,
        "max_drawdown": 0.0,
        "last_trade_at": None,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_roi": 0.0,
        "champion_exits": 0,
    }
    if not candidate.get("api_url"):
        return metrics
    instance = {
        "api_url": candidate.get("api_url", ""),
        "api_username": candidate.get("api_username", ""),
        "api_password": candidate.get("api_password", ""),
    }
    with httpx.Client() as client:
        try:
            profit = api_get(client, instance, "/api/v1/profit")
            status_payload = api_get(client, instance, "/api/v1/status")
            try:
                balance = api_get(client, instance, "/api/v1/balance")
            except Exception:
                balance = None
        except Exception as exc:  # noqa: BLE001
            metrics["status_detail"] = f"API probe failed: {exc}"
            return metrics
    open_trades = status_payload.get("value", []) if isinstance(status_payload, dict) else status_payload if isinstance(status_payload, list) else []
    equity = parse_float((balance or {}).get("total")) if isinstance(balance, dict) else 0.0
    realized = parse_float(profit.get("profit_closed_coin"))
    total = parse_float(profit.get("profit_all_coin"))
    metrics.update(
        {
            "heartbeat_ok": 1,
            "status_detail": "API heartbeat healthy.",
            "equity": equity,
            "realized_pnl": realized,
            "unrealized_pnl": total - realized,
            "closed_trades": int(profit.get("closed_trade_count") or 0),
            "open_trades": len(open_trades),
            "worst_open_trade": min((parse_float(item.get("profit_pct")) for item in open_trades), default=0.0),
            "max_drawdown": parse_float(profit.get("max_drawdown")) * 100.0,
            "last_trade_at": profit.get("latest_trade_date"),
            "wins": int(profit.get("winning_trades") or 0),
            "losses": int(profit.get("losing_trades") or 0),
            "win_rate": parse_float(profit.get("winrate")) * 100.0 if parse_float(profit.get("winrate")) <= 1 else parse_float(profit.get("winrate")),
            "avg_roi": parse_float(profit.get("profit_closed_ratio_mean")) * 100.0 if profit.get("profit_closed_ratio_mean") is not None else 0.0,
        }
    )
    return metrics


def evaluate_candidate_eligibility(candidate: dict[str, Any]) -> str:
    hypothesis_present = bool(str(candidate.get("hypothesis", "")).strip() or str(candidate.get("strategy_notes", "")).strip() or str(candidate.get("notes", "")).strip())
    if candidate.get("lifecycle_state") == "cut_archived":
        return "archived"
    if candidate.get("lifecycle_state") == "drafted":
        return "drafted"
    if not hypothesis_present:
        return "needs_hypothesis"
    if candidate.get("failure_count", 0) >= 3 and not candidate.get("heartbeat_ok"):
        return "blocked"
    if candidate.get("heartbeat_ok") and candidate.get("data_quality") not in {"broken", "missing"}:
        if int(candidate.get("closed_trades") or 0) > 0 or int(candidate.get("meaningful_runtime_minutes") or 0) >= 60:
            return "draft_eligible" if candidate.get("tier") in {"twelve_hour", "draft_eligible"} else "measurable"
        return "warming_up"
    return "not_ready"


def execute_development_command(candidate: dict[str, Any], command: str, action_name: str) -> tuple[bool, str]:
    command = command.strip()
    if not command:
        message = f"No {action_name} command configured."
        development_runtime_event(int(candidate["id"]), "scheduler", message, candidate.get("name", ""))
        return False, message
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"{action_name} command failed: {exc}"
        development_runtime_event(int(candidate["id"]), "scheduler", f"{action_name.title()} failed.", message)
        return False, message
    combined = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    if result.returncode != 0:
        message = combined or f"{action_name} command exited with {result.returncode}."
        development_runtime_event(int(candidate["id"]), "scheduler", f"{action_name.title()} failed.", message)
        return False, message
    message = combined or f"{action_name.title()} command completed."
    development_runtime_event(int(candidate["id"]), "scheduler", f"{action_name.title()} command issued.", message)
    return True, message


def development_container_runtime_state(candidate: dict[str, Any]) -> tuple[bool, str]:
    container_name = str(candidate.get("container_name") or "").strip()
    if not container_name:
        return False, "No container name configured."
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Container inspect failed: {exc}"
    if result.returncode != 0:
        detail = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
        return False, detail or "Container is not running."
    state = result.stdout.strip().lower()
    return state == "true", "Container is running." if state == "true" else "Container is not running."


def inspect_development_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    db_metrics = development_db_metrics(resolve_path(candidate.get("db_path")))
    api_metrics = development_api_metrics(candidate)
    container_running, container_detail = development_container_runtime_state(candidate)
    heartbeat_ok = int(api_metrics["heartbeat_ok"])
    data_quality = "healthy" if heartbeat_ok else "db_only" if resolve_path(candidate.get("db_path")) and resolve_path(candidate.get("db_path")).exists() else "missing"
    equity = api_metrics["equity"] if heartbeat_ok else parse_float(candidate.get("equity"))
    realized_pnl = api_metrics["realized_pnl"] if heartbeat_ok else db_metrics["realized_pnl"]
    unrealized_pnl = api_metrics["unrealized_pnl"] if heartbeat_ok else db_metrics["unrealized_pnl"]
    closed_trades = api_metrics["closed_trades"] if heartbeat_ok else db_metrics["closed_trades"]
    open_trades = api_metrics["open_trades"] if heartbeat_ok else db_metrics["open_trades"]
    worst_open_trade = api_metrics["worst_open_trade"] if heartbeat_ok else db_metrics["worst_open_trade"]
    max_drawdown = api_metrics["max_drawdown"] if heartbeat_ok else parse_float(candidate.get("max_drawdown"))
    last_trade_at = api_metrics["last_trade_at"] if heartbeat_ok else db_metrics["last_trade_at"]
    wins = api_metrics["wins"] if heartbeat_ok else db_metrics["wins"]
    losses = api_metrics["losses"] if heartbeat_ok else db_metrics["losses"]
    win_rate = api_metrics["win_rate"] if heartbeat_ok else db_metrics["win_rate"]
    avg_roi = api_metrics["avg_roi"] if heartbeat_ok else db_metrics["avg_roi"]
    champion_exits = api_metrics["champion_exits"] if heartbeat_ok else db_metrics["champion_exits"]
    status_detail = api_metrics["status_detail"] if candidate.get("api_url") else ("Trade DB found." if data_quality == "db_only" else "Awaiting runtime configuration.")
    if not heartbeat_ok and container_running:
        status_detail = "Container is up. Awaiting API heartbeat."
    elif not heartbeat_ok and container_detail:
        status_detail = container_detail
    return {
        **candidate,
        "heartbeat_ok": heartbeat_ok,
        "container_running": int(container_running),
        "container_status_detail": container_detail,
        "heartbeat_checked_at": iso_now(),
        "data_quality": data_quality,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "closed_trades": closed_trades,
        "open_trades": open_trades,
        "worst_open_trade": worst_open_trade,
        "max_drawdown": max_drawdown,
        "last_trade_at": last_trade_at,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_roi": avg_roi,
        "champion_exits": champion_exits,
        "status_detail": status_detail,
    }


def summarize_runtime_sessions(sessions: list[dict[str, Any]], now: datetime | None = None) -> dict[str, Any]:
    now = now or utc_now()
    total_runtime_hours = 0.0
    completed_lengths: list[float] = []
    last_start = ""
    last_stop = ""
    for session in sessions:
        started_at = resolve_optional_datetime(str(session.get("started_at") or ""))
        stopped_at = resolve_optional_datetime(str(session.get("stopped_at") or ""))
        duration_hours = parse_float(session.get("duration_hours"))
        if started_at and not stopped_at:
            duration_hours = hours_between(started_at, now)
        total_runtime_hours += duration_hours
        if duration_hours > 0:
            completed_lengths.append(duration_hours)
        if not last_start and session.get("started_at"):
            last_start = str(session.get("started_at"))
        if not last_stop and session.get("stopped_at"):
            last_stop = str(session.get("stopped_at"))
    return {
        "total_runtime_hours": round(total_runtime_hours, 4),
        "active_sessions_count": len(sessions),
        "average_session_length": round(sum(completed_lengths) / len(completed_lengths), 4) if completed_lengths else 0.0,
        "last_session_start": last_start,
        "last_session_stop": last_stop,
    }


def apply_runtime_metric_fields(
    row: dict[str, Any],
    total_runtime_hours: float,
    scheduled_runtime_hours: float,
) -> dict[str, Any]:
    raw_total_pnl = parse_float(row.get("realized_pnl")) + parse_float(row.get("unrealized_pnl"))
    raw_realized_pnl = parse_float(row.get("realized_pnl"))
    raw_unrealized_pnl = parse_float(row.get("unrealized_pnl"))
    raw_closed_trades = int(row.get("closed_trades") or 0)
    raw_open_trades = int(row.get("open_trades") or 0)
    raw_win_rate = parse_float(row.get("win_rate"))
    raw_avg_roi = parse_float(row.get("avg_roi"))
    raw_worst_open_trade = parse_float(row.get("worst_open_trade"))
    raw_max_drawdown = parse_float(row.get("max_drawdown"))
    wins = int(row.get("wins") or round(raw_closed_trades * raw_win_rate / 100.0))
    losses = int(row.get("losses") or max(0, raw_closed_trades - wins))
    champion_exits = int(row.get("champion_exits") or 0)
    sample_flags = runtime_sample_flags(total_runtime_hours, raw_closed_trades)
    row.update(
        {
            "total_runtime_hours": round(total_runtime_hours, 4),
            "scheduled_runtime_hours": round(scheduled_runtime_hours, 4),
            "runtime_percentage": round((total_runtime_hours / scheduled_runtime_hours) * 100.0, 2) if scheduled_runtime_hours > 0 else 0.0,
            "raw_total_pnl": round(raw_total_pnl, 4),
            "raw_realized_pnl": round(raw_realized_pnl, 4),
            "raw_unrealized_pnl": round(raw_unrealized_pnl, 4),
            "raw_closed_trades": raw_closed_trades,
            "raw_open_trades": raw_open_trades,
            "raw_win_rate": raw_win_rate,
            "raw_avg_roi": raw_avg_roi,
            "raw_worst_open_trade": raw_worst_open_trade,
            "raw_max_drawdown": raw_max_drawdown,
            "pnl_per_runtime_hour": round(safe_rate_per_hour(raw_total_pnl, total_runtime_hours), 4),
            "realized_pnl_per_runtime_hour": round(safe_rate_per_hour(raw_realized_pnl, total_runtime_hours), 4),
            "unrealized_pnl_per_runtime_hour": round(safe_rate_per_hour(raw_unrealized_pnl, total_runtime_hours), 4),
            "closed_trades_per_runtime_hour": round(safe_rate_per_hour(raw_closed_trades, total_runtime_hours), 4),
            "wins_per_runtime_hour": round(safe_rate_per_hour(wins, total_runtime_hours), 4),
            "losses_per_runtime_hour": round(safe_rate_per_hour(losses, total_runtime_hours), 4),
            "drawdown_per_runtime_hour": round(safe_rate_per_hour(raw_max_drawdown, total_runtime_hours), 4),
            "champion_exits_per_runtime_hour": round(safe_rate_per_hour(champion_exits, total_runtime_hours), 4),
            "projected_total_pnl_per_24h": projected_per_24h(raw_total_pnl, total_runtime_hours),
            "projected_realized_pnl_per_24h": projected_per_24h(raw_realized_pnl, total_runtime_hours),
            "projected_unrealized_pnl_per_24h": projected_per_24h(raw_unrealized_pnl, total_runtime_hours),
            "projected_closed_trades_per_24h": projected_per_24h(raw_closed_trades, total_runtime_hours),
            "projected_wins_per_24h": projected_per_24h(wins, total_runtime_hours),
            "projected_losses_per_24h": projected_per_24h(losses, total_runtime_hours),
            "projected_champion_exits_per_24h": projected_per_24h(champion_exits, total_runtime_hours),
            "projected_drawdown_per_24h": projected_per_24h(raw_max_drawdown, total_runtime_hours),
            "sample_flags": sample_flags,
            "sample_quality": "unreliable" if "projection_unreliable" in sample_flags else "developing" if sample_flags else "healthy",
            "wins": wins,
            "losses": losses,
        }
    )
    return row


def candidate_competition_tier(row: dict[str, Any]) -> str:
    tier = str(row.get("tier") or "")
    if tier == "six_hour":
        return "candidate_6h"
    if tier == "twelve_hour":
        return "prospect_12h"
    if tier == "bootcamp":
        return "bootcamp"
    if tier == "draft_room":
        return "draft_room"
    if tier == "archived" or row.get("lifecycle_state") == "cut_archived":
        return "archived"
    return tier or "development"


def candidate_runtime_metrics(row: dict[str, Any]) -> dict[str, Any]:
    sessions = development_runtime_sessions(int(row["id"]), limit=500)
    session_summary = summarize_runtime_sessions(sessions)
    now = utc_now()
    created_at = resolve_optional_datetime(str(row.get("created_at") or "")) or now
    tier_name = candidate_competition_tier(row)
    scheduled_hours = scheduled_runtime_hours_for_period(tier_name, str(row.get("shift_code") or ""), created_at, now)
    enriched = dict(row)
    enriched.update(session_summary)
    enriched["tier_competition"] = tier_name
    enriched["scheduled_hours_per_day"] = 6 if tier_name == "candidate_6h" else 12 if tier_name == "prospect_12h" else 24 if tier_name == "official_24h" else 0
    enriched["freshness_flags"] = freshness_flags_for_candidate({**enriched, **session_summary})
    return apply_runtime_metric_fields(enriched, session_summary["total_runtime_hours"], scheduled_hours)


def official_runtime_metrics(instance: dict[str, Any], state: sqlite3.Row | None, trade_stats: dict[str, Any]) -> dict[str, Any]:
    with closing(get_db()) as conn:
        snapshots = conn.execute(
            """
            SELECT captured_at, heartbeat_ok, status, realized_pnl, unrealized_pnl
            FROM live_snapshots
            WHERE team_id = ?
            ORDER BY id ASC
            """,
            (instance["id"],),
        ).fetchall()
    sessions: list[dict[str, Any]] = []
    current_start: str | None = None
    prev_at: datetime | None = None
    observed_runtime = 0.0
    first_seen = resolve_optional_datetime(str(snapshots[0]["captured_at"])) if snapshots else utc_now()
    last_seen = resolve_optional_datetime(str(snapshots[-1]["captured_at"])) if snapshots else utc_now()
    for snap in snapshots:
        captured_at = resolve_optional_datetime(str(snap["captured_at"]))
        heartbeat_ok = bool(snap["heartbeat_ok"])
        if heartbeat_ok and prev_at and captured_at:
            delta = hours_between(prev_at, captured_at)
            observed_runtime += min(delta, (POLL_INTERVAL_SECONDS * 2) / 3600.0)
        if heartbeat_ok and not current_start:
            current_start = str(snap["captured_at"])
        elif not heartbeat_ok and current_start:
            stop_at = str(snap["captured_at"])
            sessions.append(
                {
                    "started_at": current_start,
                    "stopped_at": stop_at,
                    "duration_hours": hours_between(resolve_optional_datetime(current_start), resolve_optional_datetime(stop_at)),
                    "stop_reason": "downtime",
                }
            )
            current_start = None
        if captured_at:
            prev_at = captured_at
    if current_start:
        sessions.append(
            {
                "started_at": current_start,
                "stopped_at": None,
                "duration_hours": hours_between(resolve_optional_datetime(current_start), utc_now()),
                "stop_reason": "",
            }
        )
    if observed_runtime <= 0 and sessions:
        observed_runtime = sum(parse_float(item.get("duration_hours")) for item in sessions)
    total_runtime_hours = max(observed_runtime, sum(parse_float(item.get("duration_hours")) for item in sessions))
    scheduled_hours = scheduled_runtime_hours_for_period("official_24h", "", first_seen or utc_now(), last_seen or utc_now())
    row = {
        "strategy_id": instance["id"],
        "team_id": instance["id"],
        "team": instance["display_name"],
        "name": instance["display_name"],
        "strategy_family": instance.get("strategy_family"),
        "pair_universe": instance.get("pair_universe"),
        "coin_universe": instance.get("pair_universe"),
        "long_short_mode": "both",
        "timeframe": instance.get("timeframe", ""),
        "runtime_status": state["status"] if state else "unknown",
        "current_runtime_state": "running" if (state and state["heartbeat_ok"]) else "failed",
        "tier": "official_24h",
        "tier_competition": "official_24h",
        "shift_id": "",
        "shift_code": "",
        "runtime_window": "24/7",
        "scheduled_hours_per_day": 24,
        "equity": parse_float(state["equity"]) if state else parse_float(instance.get("starting_capital")),
        "realized_pnl": parse_float(state["realized_pnl"]) if state else 0.0,
        "unrealized_pnl": parse_float(state["unrealized_pnl"]) if state else 0.0,
        "closed_trades": trade_stats["closed"],
        "open_trades": trade_stats["open"],
        "win_rate": trade_stats["win_rate"],
        "avg_roi": trade_stats["avg_roi"],
        "best_trade": trade_stats["best_trade"],
        "worst_open_trade": trade_stats["worst_open_trade"],
        "max_drawdown": parse_float(state["max_drawdown"]) if state else 0.0,
        "champion_exits": (
            trade_stats["exit_breakdown"].get("champ_dynamic_roi", {}).get("count", 0)
            + trade_stats["exit_breakdown"].get("champ_dynamic_roi_hit", {}).get("count", 0)
        ),
        "wins": trade_stats["wins"],
        "losses": max(0, trade_stats["closed"] - trade_stats["wins"]),
        "active_sessions_count": len(sessions),
        "average_session_length": round(sum(parse_float(item.get("duration_hours")) for item in sessions) / len(sessions), 4) if sessions else 0.0,
        "last_session_start": sessions[-1]["started_at"] if sessions else "",
        "last_session_stop": next((item["stopped_at"] for item in reversed(sessions) if item.get("stopped_at")), ""),
        "data_quality": instance.get("data_quality"),
        "heartbeat": bool(state["heartbeat_ok"]) if state else False,
        "role": instance.get("role"),
        "freshness_flags": [],
        "sample_flags": [],
        "sample_quality": "healthy",
        "source_league": "official",
    }
    return apply_runtime_metric_fields(row, total_runtime_hours, scheduled_hours)

def strategy_code_preview(candidate: dict[str, Any], max_chars: int = 6000) -> str:
    path = resolve_path(candidate.get("strategy_path"))
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def strategy_generation_prompt(candidate: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    class_name = safe_strategy_class_name(str(candidate.get("name", "")))
    latest_review = latest_development_post_shift_review(int(candidate["id"])) if candidate.get("id") else None
    peer_context = development_generation_peer_context(int(candidate.get("id") or 0), limit=12)
    exchange_rotation = development_exchange_rotation_snapshot()
    existing_strategy_path = resolve_path(candidate.get("strategy_path")) or development_strategy_path(candidate)
    existing_strategy_code = ""
    try:
        if existing_strategy_path and existing_strategy_path.exists():
            existing_strategy_code = existing_strategy_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        existing_strategy_code = ""
    is_revision = bool(existing_strategy_code)
    payload = {
        "candidate_name": candidate.get("name", ""),
        "class_name": class_name,
        "long_short_mode": candidate.get("long_short_mode", "both"),
        "risk_profile": candidate.get("risk_profile", ""),
        "timeframe": candidate.get("timeframe", ""),
        "coin_universe": candidate.get("coin_universe", ""),
        "expected_behavior": candidate.get("expected_behavior", ""),
        "hypothesis": candidate.get("hypothesis", ""),
        "strategy_notes": candidate.get("strategy_notes", ""),
        "general_notes": candidate.get("notes", ""),
        "latest_post_shift_review": {
            "grade": latest_review.get("grade", ""),
            "decision_bucket": latest_review.get("decision_bucket", ""),
            "recommendation": latest_review.get("recommendation", ""),
            "summary": latest_review.get("summary", ""),
            "mutation_brief": latest_review.get("mutation_brief", ""),
            # Ranked, trade-derived mechanical findings — the clinical evidence behind the
            # brief. Each is {code, severity, headline, fix}; act on the critical/warning fixes.
            "diagnostics": [
                {"severity": d.get("severity"), "finding": d.get("headline"), "fix": d.get("fix")}
                for d in latest_review.get("diagnostics", [])
                if d.get("severity") in {"critical", "warning"}
            ],
        }
        if latest_review
        else {},
        "peer_candidate_context": peer_context,
        "exchange_rotation_snapshot": exchange_rotation,
        "generation_mode": "revise_existing" if is_revision else "first_generation",
        "current_strategy_code": existing_strategy_code,
        "generation_guidance": {
            "goal": (
                "Revise the existing strategy using the latest review. Preserve the logic that is already working, apply the review's "
                "mutation_brief, and only rewrite the parts that actually need it. Add depth, robustness, and edge cases instead of starting "
                "over or trimming the strategy down."
                if is_revision
                else "Create the first runnable version of this candidate from its hypothesis and notes. It can start fairly simple; later "
                "shifts will deepen it."
            ),
            "additive_note": (
                "This is a revision, not a rewrite. Keep the prior indicators, parameters, and structure that are working unless the review "
                "calls them out. Build on top of current_strategy_code rather than replacing it wholesale, and do not regress to a smaller or "
                "simpler implementation."
                if is_revision
                else "No prior implementation exists yet, so write it from scratch."
            ),
            "exchange_note": "Runtime config is chosen separately, so the strategy should stay portable across supported CCXT/Freqtrade futures exchanges.",
            "timeframe_note": "Longer timeframe ideas should stay selective instead of being forced into high-frequency behavior.",
        },
        # Temporal Niche classification (Signal Timing Spectrum). The generator is the
        # actor that knows where this version of the strategy seeks its edge.
        "signal_timing_spectrum": [slug for slug, _ in ML_SIGNAL_TIMING_SPECTRUM],
        "temporal_niche_guidance": {
            "what": "Classify where on the move's lifecycle THIS version of the strategy believes its entry edge exists, based on its actual entry logic.",
            "how": "Return temporal_niche = {start, end, note, status}. start/end must be slugs from signal_timing_spectrum (a band; use the same slug for both if it acts at one point). status is 'placed' when the entry logic clearly targets a phase, else 'needs_data'.",
            "honesty": "Prefer status='needs_data' (leave start/end empty, explain in note) when the strategy is exploratory/research-oriented or its edge can't be localized to a phase without live data. Do not force a guess.",
            "scope_note": "This is a descriptive label for the genealogy/biology view only. It does NOT affect trading and is NOT a compatibility score.",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Generate one runnable Freqtrade strategy for dry-run only. "
                "Return strict JSON only with keys strategy_code, implementation_summary, assumptions, warnings, suggested_timeframe, suggested_max_open_trades, minimal_config_notes, temporal_niche. "
                "strategy_code must be a complete Python file using the provided class_name exactly. "
                f"suggested_max_open_trades is an integer in [{DEV_MAX_OPEN_TRADES_MIN}, {DEV_MAX_OPEN_TRADES_MAX}] for how many positions this thesis should hold concurrently: "
                "the dry-run wallet is fixed and split evenly across slots, so more slots = smaller, more diversified positions (fast/scalp/breadth theses) and fewer slots = larger, concentrated bets (slow/high-conviction/multi-timeframe theses). Choose the count its edge actually implies. "
                "temporal_niche classifies where this version seeks its edge on the signal_timing_spectrum per temporal_niche_guidance: "
                "{start, end, note, status} with start/end as spectrum slugs and status 'placed' or 'needs_data'; prefer 'needs_data' when unsure rather than guessing. "
                "When current_strategy_code is provided, treat this as a revision: preserve the working logic, build on it additively, and only "
                "rewrite the parts the review calls out. Do not regress to a smaller or simpler strategy than what already exists; deepen it. "
                "Use the supplied dev-league context to avoid cloning nearby strategies and to react to the latest review with a credible update. "
                "Prefer portable futures logic that can run across supported CCXT/Freqtrade exchanges. "
                "Avoid exotic dependencies and do not use markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]
    return messages, json.dumps(payload, indent=2)


def validate_strategy_file(strategy_path: Path, class_name: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not strategy_path.exists():
        return False, ["Strategy file was not written."]
    strategy_path = strategy_path.resolve()
    source = strategy_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source, filename=str(strategy_path))
    except SyntaxError as exc:
        return False, [f"Syntax error: {exc}"]
    try:
        py_compile.compile(str(strategy_path), doraise=True)
    except py_compile.PyCompileError as exc:
        errors.append(f"Compile check failed: {exc.msg}")
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if not class_node:
        errors.append(f"Expected strategy class `{class_name}` was not found.")
        return False, errors
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    for method_name in ("populate_indicators", "populate_entry_trend", "populate_exit_trend"):
        if method_name not in method_names:
            errors.append(f"Missing required method `{method_name}`.")
    class_assignments = {
        target.id
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for attr_name in ("timeframe", "minimal_roi", "stoploss"):
        if attr_name not in class_assignments:
            errors.append(f"Missing required attribute `{attr_name}`.")
    docker_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_DIR}:/workspace",
        "-w",
        "/workspace",
        "--entrypoint",
        "python",
        "freqtradeorg/freqtrade:stable",
        "-c",
        (
            "import importlib.util, pathlib; "
            f"path=pathlib.Path(r'/workspace/{relative_project_path(strategy_path)}'); "
            "spec=importlib.util.spec_from_file_location('candidate_strategy', path); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            f"cls=getattr(module, '{class_name}'); "
            "required=['populate_indicators','populate_entry_trend','populate_exit_trend']; "
            "missing=[name for name in required if not hasattr(cls, name)]; "
            "assert not missing, 'Missing runtime methods: ' + ', '.join(missing); "
            "assert hasattr(cls, 'timeframe'); "
            "assert hasattr(cls, 'minimal_roi'); "
            "assert hasattr(cls, 'stoploss'); "
            "print('ok')"
        ),
    ]
    try:
        probe = subprocess.run(docker_command, cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            combined = "\n".join(part for part in [probe.stdout.strip(), probe.stderr.strip()] if part).strip()
            errors.append(combined or "Container import validation failed.")
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Container import validation failed: {exc}")
    return not errors, errors


# Stablecoins (USD/EUR-pegged) carry no directional signal, but MarketCapPairList
# surfaces them because they rank high by market cap. Blacklist them by default for
# every dev bot — momentum/trend/breakout/exhaustion strategies just churn fees on a
# flat 1.00 peg. A strategy that genuinely needs a stablecoin pair can drop this from
# its own config. freqtrade matches each pattern with re.fullmatch on "BASE/QUOTE:SETTLE".
DEV_STABLECOIN_BLACKLIST: list[str] = [
    "(USDC|USDD|TUSD|BUSD|DAI|FDUSD|USDP|GUSD|USTC|USDE|PYUSD|FRAX|LUSD|SUSD|USD1|USDX|USDJ|USDB|EURT|EURS|AEUR|EURI)/.*",
]

# Concurrency variety for the dev league. The dry-run wallet stays fixed (so total
# capital is constant); only how finely it's split varies. The strategy generator
# proposes `suggested_max_open_trades` to fit each thesis (scalpers many, slow
# trend-followers few); if it doesn't, a deterministic per-candidate spread still
# guarantees the population diversifies instead of all cloning the base default of 2.
DEV_MAX_OPEN_TRADES_MIN = 1
DEV_MAX_OPEN_TRADES_MAX = 12
# Protected prospects are exceptional, human-curated strategies (e.g. The Turners'
# 9-long / 9-short progression book needs 18 slots). They opt out of both the
# id-keyed spread AND the normal 12 cap, honoring their explicit request up to a
# higher sanity ceiling. Regular dev bots stay capped at 12.
DEV_MAX_OPEN_TRADES_PROTECTED_MAX = 30
DEV_MAX_OPEN_TRADES_SPREAD = [1, 2, 3, 4, 5, 6, 8, 10, 12]


def candidate_max_open_trades(candidate: dict[str, Any]) -> int:
    """Resolve a candidate's max_open_trades: the generator's suggestion when present
    and sane, else a deterministic spread keyed off its id. Always clamped to range.

    Protected prospects bypass the random-feeling spread and the 12 cap: their
    explicit request is honored (clamped only to a higher sanity ceiling), so a
    strategy like The Turners is not silently squeezed down to a generic slot count."""
    suggested = 0
    try:
        suggested = int(float(candidate.get("suggested_max_open_trades") or 0))
    except (TypeError, ValueError):
        suggested = 0
    protected = bool(int(candidate.get("protected") or 0))
    if protected:
        # Never subject a protected prospect to the id-keyed spread. With an explicit
        # request, honor it up to the protected ceiling; without one, grant the full
        # normal allowance (12) rather than a random small slot count.
        target = suggested if suggested > 0 else DEV_MAX_OPEN_TRADES_MAX
        return max(DEV_MAX_OPEN_TRADES_MIN, min(DEV_MAX_OPEN_TRADES_PROTECTED_MAX, target))
    if suggested <= 0:
        try:
            key = int(candidate.get("id") or 0)
        except (TypeError, ValueError):
            key = 0
        if key <= 0:
            key = abs(hash(str(candidate.get("slug") or candidate.get("name") or "")))
        suggested = DEV_MAX_OPEN_TRADES_SPREAD[key % len(DEV_MAX_OPEN_TRADES_SPREAD)]
    return max(DEV_MAX_OPEN_TRADES_MIN, min(DEV_MAX_OPEN_TRADES_MAX, suggested))


# ===========================================================================
# ATL External Resource Governance
# ---------------------------------------------------------------------------
# Pairlists and exchanges are shared league resources. No bot independently
# stampedes CoinGecko or blindly grabs an exchange:
#   * ONE scheduled CoinGecko pull builds canonical universes (top100_marketcap,
#     top50_volume, custom_momentum_30, ...).
#   * Those are filtered per exchange (ccxt load_markets: futures/spot, quote,
#     settle, Freqtrade pair format) into exchange-specific manifests.
#   * Manifests are served over HTTP for Freqtrade RemotePairList.
#   * Each dev candidate LEASES an exchange per shift (capacity + cooldown), and
#     its universe is frozen for the whole scored shift so comparisons are fair.
# Major-league 24h teams are not dev_candidates and never reach this layer.
# ===========================================================================

CANONICAL_UNIVERSE_NAMES = [
    "top100_marketcap",
    "top50_marketcap",
    "top20_marketcap",
    "top50_volume",
    "custom_momentum_30",
    # Cartographer universe: co-moving "constellations" with their laggards
    # surfaced. Built by build_block_party_universe(); first genuinely custom
    # (non-marketcap) universe to be deployed. See Block Party candidate.
    "block_party",
    # Scouting universe: symbols that look closest to BECOMING champions — early
    # in the ATL Signal Timing Spectrum (compression -> early expansion -> breakout
    # readiness -> trend confirmation), penalising already-mature/exhausted moves.
    # Built by build_future_champion_universe(). Powers The Turnstile (A/B vs The
    # Turners), which runs identical strategy logic on this gated universe.
    "future_champion",
    # Scouting universe of opposite intent: assets that JUST experienced extraordinary
    # movement (largest gainers/losers, volatility explosions, relative-performance
    # anomalies). Built by build_big_movers_universe(). Powers Second Act, which studies
    # the lifecycle of extreme events. Decides only "what deserves attention".
    "big_movers",
]


def _stable_base_symbols() -> set[str]:
    # Reuse the single source of truth (the blacklist regex) so the universe
    # builder and the per-bot blacklist can never drift apart.
    match = re.search(r"\(([^)]+)\)", DEV_STABLECOIN_BLACKLIST[0]) if DEV_STABLECOIN_BLACKLIST else None
    if not match:
        return set()
    return {token.strip().upper() for token in match.group(1).split("|") if token.strip()}


STABLE_BASE_SYMBOLS = _stable_base_symbols()


def resource_governance_enabled() -> bool:
    return get_setting("resource_governance_enabled", "true").strip().lower() == "true"


def pairlist_manifest_base_url() -> str:
    return get_setting("pairlist_manifest_base_url", "http://host.docker.internal:8000").rstrip("/")


def exchange_profile_by_id(exchange_id: str) -> dict[str, Any] | None:
    for profile in DEV_EXCHANGE_PROFILES:
        if str(profile.get("name")) == exchange_id:
            return dict(profile)
    return None


def exchange_quote_currency(profile: dict[str, Any]) -> str:
    options = profile.get("ccxt_options") or {}
    return str(options.get("defaultSettle") or profile.get("stake_currency") or "USDT").upper()


def manifest_name(universe: str, exchange_id: str, market_type: str, quote: str) -> str:
    return f"{universe}_{exchange_id}_{market_type}_{quote.lower()}"


def get_pairlist_manifest(name: str) -> dict[str, Any] | None:
    payload = get_generated_json(f"pairlist_manifest:{name}", None)
    return payload if isinstance(payload, dict) else None


# --- canonical universe builder (one external API pull) ---------------------

def fetch_coingecko_markets(per_page: int = 250) -> list[dict[str, Any]]:
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "price_change_percentage": "24h",
        # 7d hourly sparkline feeds the Block Party constellation clustering.
        "sparkline": "true",
    }
    headers: dict[str, str] = {}
    key = get_setting("coingecko_api_key", "").strip()
    if key:
        headers["x-cg-demo-api-key"] = key
    with httpx.Client(timeout=30.0) as client:
        resp = client.get("https://api.coingecko.com/api/v3/coins/markets", params=params, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    return payload if isinstance(payload, list) else []


def coingecko_entries(per_page: int = 250, source: str = "coingecko") -> list[dict[str, Any]]:
    """Fetch the CoinGecko market snapshot and map it to the `entries` shape the
    universe builders consume: {base, rank, volume, chg, spark}. Split out from
    build_canonical_universes() so the backtest engine can substitute a historically
    reconstructed snapshot and run the identical builder logic. `source` is carried
    only for logging/provenance."""
    rows = fetch_coingecko_markets(per_page)
    entries: list[dict[str, Any]] = []
    for row in rows:
        base = str(row.get("symbol") or "").upper()
        if not base or base in STABLE_BASE_SYMBOLS:
            continue
        entries.append({
            "base": base,
            "rank": row.get("market_cap_rank") or 9999,
            "volume": row.get("total_volume") or 0,
            "chg": row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h")) or 0.0,
            "spark": (row.get("sparkline_in_7d") or {}).get("price") or [],
        })
    return entries


def compute_universes_from_entries(entries: list[dict[str, Any]], source: str = "coingecko") -> dict[str, Any]:
    """Pure: an `entries` snapshot -> {universe_name: payload} for every canonical
    universe, with NO persistence. The single source of truth shared by the live
    pairlist cycle (build_canonical_universes) and the backtest engine, which feeds
    it a reconstructed historical snapshot to replay a universe as of a past date."""
    by_rank = sorted(entries, key=lambda e: e["rank"])
    universes: dict[str, Any] = {}

    def simple(name: str, bases: list[str]) -> None:
        universes[name] = {"name": name, "bases": bases, "count": len(bases), "built_at": iso_now(), "source": source}

    simple("top100_marketcap", [e["base"] for e in by_rank[:100]])
    simple("top50_marketcap", [e["base"] for e in by_rank[:50]])
    simple("top20_marketcap", [e["base"] for e in by_rank[:20]])
    by_volume = sorted(entries, key=lambda e: e["volume"], reverse=True)
    simple("top50_volume", [e["base"] for e in by_volume[:50]])
    by_momentum = sorted(by_rank[:100], key=lambda e: e["chg"], reverse=True)
    simple("custom_momentum_30", [e["base"] for e in by_momentum[:30]])
    # Cartographer universe (clusters the market, surfaces constellation laggards).
    universes["block_party"] = build_block_party_universe(entries)
    # Scouting universe (ranks coins by how close they look to becoming champions).
    universes["future_champion"] = build_future_champion_universe(entries)
    # Big Movers universe (ranks coins by how extraordinary their recent movement is).
    universes["big_movers"] = build_big_movers_universe(entries)
    return universes


def build_canonical_universes() -> dict[str, Any]:
    universes = compute_universes_from_entries(coingecko_entries(250))
    for name, payload in universes.items():
        replace_generated_json(f"pairlist_universe:{name}", payload)
    return universes


# --- Block Party: constellation cartographer universe -----------------------

def _clean_sparkline(values: list[Any], length: int) -> list[float] | None:
    """Return the last `length` strictly-positive finite floats, or None if the
    series is too short / has gaps. Keeps junk out of the correlation matrix."""
    series = values[-length:] if values else []
    if len(series) < length:
        return None
    out: list[float] = []
    for raw in series:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val <= 0.0:
            return None
        out.append(val)
    return out


def build_block_party_universe(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The cartographer. Cluster the market into co-moving "constellations" from
    7d hourly sparklines, find the constellation(s) currently lighting up, and
    surface their laggard members ("the group is right, the laggard is next") plus
    a couple of leaders for context. Stores pairlist_universe:block_party with
    rich per-constellation metadata for the strategy/telemetry. Degrades to a
    momentum list (never an empty manifest) if clustering can't run."""
    target_size = max(8, int(parse_float(get_setting("block_party_universe_size", "15")) or 15))
    pool_size = max(40, int(parse_float(get_setting("block_party_pool_size", "150")) or 150))
    min_cluster = max(3, int(parse_float(get_setting("block_party_min_constellation", "3")) or 3))

    def _store(bases: list[str], constellations: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "name": "block_party",
            "bases": bases,
            "count": len(bases),
            "built_at": iso_now(),
            "source": "coingecko+constellation",
            "constellations": constellations,
        }
        if extra:
            payload.update(extra)
        replace_generated_json("pairlist_universe:block_party", payload)
        return payload

    momentum_fallback = [e["base"] for e in sorted(entries, key=lambda x: x["chg"], reverse=True)[:target_size]]

    # Candidate pool: highest-marketcap coins that actually carry a clean sparkline.
    spark_len = 168
    pool: list[dict[str, Any]] = []
    series: list[list[float]] = []
    for entry in sorted(entries, key=lambda x: x["rank"]):
        cleaned = _clean_sparkline(entry.get("spark") or [], spark_len)
        if cleaned is None:
            continue
        pool.append(entry)
        series.append(cleaned)
        if len(pool) >= pool_size:
            break

    if len(pool) < min_cluster * 2:
        return _store(momentum_fallback, [], {"degraded": "insufficient_sparkline_pool"})

    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        prices = np.asarray(series, dtype=float)
        rets = np.diff(np.log(prices), axis=1)
        finite = np.isfinite(rets).all(axis=1)
        if not finite.all():
            pool = [p for p, keep in zip(pool, finite) if keep]
            rets = rets[finite]
        if len(pool) < min_cluster * 2:
            return _store(momentum_fallback, [], {"degraded": "insufficient_finite_returns"})
        corr = np.nan_to_num(np.corrcoef(rets), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.clip(1.0 - corr, 0.0, 2.0)
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        threshold = float(parse_float(get_setting("block_party_distance_threshold", "0.55")) or 0.55)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average",
        ).fit_predict(dist)
    except Exception as exc:  # noqa: BLE001 — never let clustering kill the cycle
        log_maintenance("pairlist", "warn", f"block_party clustering failed ({exc}); momentum fallback.")
        return _store(momentum_fallback, [], {"degraded": f"clustering_error:{exc}"})

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(idx)

    constellations: list[dict[str, Any]] = []
    for label, members in grouped.items():
        size = len(members)
        if size < min_cluster:
            continue
        sub = corr[np.ix_(members, members)]
        coherence = float((sub.sum() - size) / (size * (size - 1)))  # mean off-diagonal correlation
        chg = np.asarray([pool[i]["chg"] for i in members], dtype=float)
        group_move = float(np.mean(chg))
        dir_sign = 1.0 if group_move >= 0.0 else -1.0
        member_info: list[dict[str, Any]] = []
        for i in members:
            belong = float((corr[i, members].sum() - 1.0) / (size - 1))
            own_dir_move = dir_sign * float(pool[i]["chg"])
            group_dir_move = dir_sign * group_move
            # Rudolph metric: belongs strongly, has moved least in the group's
            # direction (= most room left to join the move).
            laggard = max(0.0, belong) * max(0.0, group_dir_move - own_dir_move)
            member_info.append({
                "base": pool[i]["base"],
                "rank": pool[i]["rank"],
                "chg": round(float(pool[i]["chg"]), 3),
                "belong": round(belong, 3),
                "laggard_score": round(laggard, 4),
            })
        member_info.sort(key=lambda m: m["laggard_score"], reverse=True)
        leaders = sorted(member_info, key=lambda m: dir_sign * m["chg"], reverse=True)
        constellations.append({
            "label": int(label),
            "size": size,
            "coherence": round(coherence, 4),
            "group_move": round(group_move, 3),
            "direction": "long" if group_move >= 0.0 else "short",
            "lit_score": round(coherence * abs(group_move), 5),
            "members": member_info,
            "leaders": [m["base"] for m in leaders[:2]],
        })

    if not constellations:
        return _store(momentum_fallback, [], {"degraded": "no_constellations"})

    min_coherence = float(parse_float(get_setting("block_party_min_coherence", "0.30")) or 0.30)
    min_move = float(parse_float(get_setting("block_party_min_group_move", "1.5")) or 1.5)
    lit = [c for c in constellations if c["coherence"] >= min_coherence and abs(c["group_move"]) >= min_move]
    lit.sort(key=lambda c: c["lit_score"], reverse=True)
    if not lit:
        # Nothing clearly lit: trade the single most coherent neighborhood rather
        # than noise, so the universe is still a real constellation.
        lit = sorted(constellations, key=lambda c: c["coherence"], reverse=True)[:1]
    top_lit = lit[: max(1, int(parse_float(get_setting("block_party_max_constellations", "2")) or 2))]

    bases: list[str] = []

    def _add(base: str) -> None:
        if base not in bases:
            bases.append(base)

    per_constellation = max(2, target_size // max(1, len(top_lit)))
    for constellation in top_lit:
        for member in constellation["members"][: per_constellation + 1]:  # laggards first (the Rudolphs)
            if len(bases) >= target_size:
                break
            _add(member["base"])
        for leader in constellation["leaders"]:  # a little context for the story
            if len(bases) >= target_size:
                break
            _add(leader)
        if len(bases) >= target_size:
            break
    for base in momentum_fallback:  # pad if a thin constellation left us short
        if len(bases) >= target_size:
            break
        _add(base)

    return _store(bases[:target_size], top_lit, {
        "all_constellations": len(constellations),
        "lit_constellations": len(lit),
    })


# --- The Turnstile: Future Champion scouting universe -----------------------

def _resample_hourly_to_4h(prices: list[float]) -> list[dict[str, float]] | None:
    """Group a 7d hourly close series into 4h OHLC-ish candles built from hourly
    closes (open=first hour, high/low=bucket extremes, close=last hour). Returns
    chronological candles, or None if too short to score. CoinGecko gives us
    price-only sparklines, so 'volume/participation' factors can't be derived
    here — the scout works on price structure (compression/expansion/trend)."""
    series = [float(p) for p in prices if isinstance(p, (int, float)) and math.isfinite(float(p)) and float(p) > 0.0]
    if len(series) < 24:  # need a couple of days of hourly data
        return None
    usable = len(series) - (len(series) % 4)
    candles: list[dict[str, float]] = []
    for i in range(0, usable, 4):
        bucket = series[i:i + 4]
        candles.append({"open": bucket[0], "high": max(bucket), "low": min(bucket), "close": bucket[-1]})
    return candles if len(candles) >= 8 else None


def _future_champion_factors(candles: list[dict[str, float]]) -> dict[str, float] | None:
    """Raw (un-normalised) scouting factors for one symbol from its 4h candles.
    Positive factors look for the EARLY stages of a move (compression -> early
    expansion -> breakout readiness -> trend confirmation potential); the lone
    negative factor (`exhaustion`) flags moves that already largely played out."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    closes = np.asarray([c["close"] for c in candles], dtype=float)
    highs = np.asarray([c["high"] for c in candles], dtype=float)
    lows = np.asarray([c["low"] for c in candles], dtype=float)
    n = closes.size
    if n < 8 or not np.all(closes > 0):
        return None
    eps = 1e-9
    # Liveness gate: a future champion must be a normally-ACTIVE asset that is
    # currently compressed — not an inert peg / tokenised-treasury / dead coin whose
    # flat line is "maximum compression". Reject anything whose weekly range is below
    # a floor (these are also typically not even tradable perps, so they'd vanish at
    # the futures filter and leave a thin manifest). Floor is in fractional terms.
    min_week_range = float(parse_float(get_setting("future_champion_min_week_range", "0.04")) or 0.04)
    week_range = (float(np.max(closes)) - float(np.min(closes))) / (float(np.mean(closes)) + eps)
    if week_range < min_week_range:
        return None
    # Relative true-range per 4h candle = our ATR proxy.
    tr = (highs - lows) / (closes + eps)
    recent = max(3, n // 6)               # ~last day
    recent_atr = float(np.mean(tr[-recent:]))
    base_atr = float(np.mean(tr[:-recent])) if n - recent >= 2 else float(np.mean(tr))
    # 1) Compression: recent range contracted vs its own baseline.
    contraction = (base_atr - recent_atr) / (base_atr + eps)
    # 2) Tightness: narrow recent Bollinger-style band (low close dispersion).
    win = closes[-recent:]
    tightness = -(float(np.std(win)) / (float(np.mean(win)) + eps))
    # 3) Early expansion: ATR turning UP over the last few candles.
    a, b = tr[-3:], tr[-6:-3] if n >= 6 else tr[:max(1, n - 3)]
    expansion = (float(np.mean(a)) - float(np.mean(b))) / (float(np.mean(b)) + eps)
    # 4) Breakout readiness: close pressed against a recent range extreme.
    w = closes[-min(n, 12):]
    lo, hi = float(np.min(w)), float(np.max(w))
    pos = (float(closes[-1]) - lo) / ((hi - lo) + eps)
    breakout = max(pos, 1.0 - pos)
    # 5) Trend confirmation potential: clean, aligned, not-yet-spent directional bias.
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, closes, 1)
    fit = slope * x + intercept
    ss_res = float(np.sum((closes - fit) ** 2))
    ss_tot = float(np.sum((closes - float(np.mean(closes))) ** 2)) + eps
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    slope_recent = float(np.polyfit(np.arange(w.size, dtype=float), w, 1)[0])
    aligned = 1.0 if (slope_recent >= 0) == (float(slope) >= 0) else 0.5
    # Reward trend QUALITY (clean, aligned structure), NOT raw magnitude — a big
    # completed parabola must not look like "trend confirmation potential". Magnitude
    # is handled (and penalised when overdone) by the exhaustion maturity gate below.
    trend = r2 * aligned
    # 6) Exhaustion (negative): overextended from its mean AND a large move already done.
    ma_win = closes[-min(n, 12):]
    overext = abs(float(closes[-1]) - float(np.mean(ma_win))) / (float(np.std(ma_win)) + eps)
    total_move = abs(float(closes[-1]) / (float(closes[0]) + eps) - 1.0)
    exhaustion = 0.6 * overext + 0.4 * min(total_move, 1.0) * 3.0
    out = {
        "contraction": contraction, "tightness": tightness, "expansion": expansion,
        "breakout": breakout, "trend": trend, "exhaustion": exhaustion,
    }
    return {k: (float(v) if math.isfinite(float(v)) else 0.0) for k, v in out.items()}


def build_future_champion_universe(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The scouting department. Score the large liquid universe by how close each
    coin looks to BECOMING a champion (early in the ATL Signal Timing Spectrum),
    not by how strong it already is, and surface the top N. Stores
    pairlist_universe:future_champion with a per-symbol factor breakdown for
    telemetry. Degrades to a momentum list (never an empty manifest) if scoring
    can't run. Used by The Turnstile as a controlled A/B vs The Turners."""
    target_size = max(10, int(parse_float(get_setting("future_champion_universe_size", "50")) or 50))
    # Positive factors span the EARLY part of the ATL Signal Timing Spectrum;
    # `exhaustion` is the single negative factor (subtracted). All rank-normalised
    # to [0,1] across the pool, so weights are directly comparable. Exhaustion is
    # weighted heavily so already-completed/overextended runs are pushed down hard
    # ("find symbols CLOSEST to becoming champions, not the strongest movers").
    weights = {
        "contraction": float(parse_float(get_setting("future_champion_w_contraction", "1.0")) or 1.0),
        "tightness": float(parse_float(get_setting("future_champion_w_tightness", "0.8")) or 0.8),
        "expansion": float(parse_float(get_setting("future_champion_w_expansion", "1.0")) or 1.0),
        "breakout": float(parse_float(get_setting("future_champion_w_breakout", "0.9")) or 0.9),
        "trend": float(parse_float(get_setting("future_champion_w_trend", "0.7")) or 0.7),
    }
    w_exhaustion = float(parse_float(get_setting("future_champion_w_exhaustion", "1.6")) or 1.6)
    momentum_fallback = [e["base"] for e in sorted(entries, key=lambda x: x["chg"], reverse=True)[:target_size]]

    def _store(bases: list[str], scouts: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "name": "future_champion",
            "bases": bases,
            "count": len(bases),
            "built_at": iso_now(),
            "source": "coingecko+scout",
            "scouts": scouts,
        }
        if extra:
            payload.update(extra)
        replace_generated_json("pairlist_universe:future_champion", payload)
        return payload

    scored: list[dict[str, Any]] = []
    for entry in entries:
        candles = _resample_hourly_to_4h(entry.get("spark") or [])
        if candles is None:
            continue
        factors = _future_champion_factors(candles)
        if factors is None:
            continue
        scored.append({"base": entry["base"], "rank": entry["rank"], **factors})

    if len(scored) < max(12, target_size // 2):
        return _store(momentum_fallback, [], {"degraded": "insufficient_scored_pool"})

    try:
        import numpy as np

        def _rank_norm(values: list[float]) -> "np.ndarray":
            arr = np.asarray(values, dtype=float)
            if arr.size <= 1:
                return np.zeros_like(arr)
            order = np.argsort(np.argsort(arr))  # 0 (lowest) .. n-1 (highest)
            return order.astype(float) / float(arr.size - 1)

        norm = {key: _rank_norm([s[key] for s in scored]) for key in weights}
        exhaustion_n = _rank_norm([s["exhaustion"] for s in scored])
        composite = (
            weights["contraction"] * norm["contraction"]
            + weights["tightness"] * norm["tightness"]
            + weights["expansion"] * norm["expansion"]
            + weights["breakout"] * norm["breakout"]
            + weights["trend"] * norm["trend"]
            - w_exhaustion * exhaustion_n
        )
    except Exception as exc:  # noqa: BLE001 — never let scoring kill the cycle
        log_maintenance("pairlist", "warn", f"future_champion scoring failed ({exc}); momentum fallback.")
        return _store(momentum_fallback, [], {"degraded": f"scoring_error:{exc}"})

    for idx, sc in enumerate(scored):
        sc["score"] = round(float(composite[idx]), 5)
    scored.sort(key=lambda s: s["score"], reverse=True)

    bases: list[str] = []
    scouts: list[dict[str, Any]] = []
    for sc in scored[:target_size]:
        bases.append(sc["base"])
        scouts.append({
            "base": sc["base"], "rank": sc["rank"], "score": sc["score"],
            "contraction": round(sc["contraction"], 4), "expansion": round(sc["expansion"], 4),
            "breakout": round(sc["breakout"], 4), "trend": round(sc["trend"], 4),
            "exhaustion": round(sc["exhaustion"], 4),
        })
    for base in momentum_fallback:  # pad if too few cleanly-scored symbols
        if len(bases) >= target_size:
            break
        if base not in bases:
            bases.append(base)
    return _store(bases[:target_size], scouts, {"scored_pool": len(scored)})


# --- Second Act: Big Movers scouting universe -------------------------------

def _big_mover_metrics(spark: list[Any], chg_24h: float) -> dict[str, float] | None:
    """Recent-extremeness metrics for one symbol from its 7d hourly sparkline (resampled
    to 4h) plus the 24h % change. The Big Movers pairlist SEEKS extraordinary movement
    (the inverse of the future_champion scout) and emits NO direction — only magnitude."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    candles = _resample_hourly_to_4h(spark or [])
    if candles is None:
        return None
    closes = np.asarray([c["close"] for c in candles], dtype=float)
    highs = np.asarray([c["high"] for c in candles], dtype=float)
    lows = np.asarray([c["low"] for c in candles], dtype=float)
    n = closes.size
    if n < 8 or not np.all(closes > 0):
        return None
    eps = 1e-9
    tr = (highs - lows) / (closes + eps)
    recent = max(3, n // 6)
    recent_vol = float(np.mean(tr[-recent:]))
    base_vol = float(np.mean(tr[:-recent])) if n - recent >= 2 else float(np.mean(tr))
    # Volatility explosion: recent range vs its own baseline.
    vol_expansion = (recent_vol - base_vol) / (base_vol + eps)
    # Recent realized swing over the last ~day (largest move regardless of sign).
    w = closes[-min(n, 6):]
    recent_swing = (float(np.max(w)) - float(np.min(w))) / (float(np.min(w)) + eps)
    return {
        "abs_chg": abs(float(chg_24h)),
        "signed_chg": float(chg_24h),
        "vol_expansion": vol_expansion,
        "recent_swing": recent_swing,
        "recent_vol": recent_vol,
    }


def build_big_movers_universe(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The Second Act scouting department: surface assets that just experienced
    EXTRAORDINARY movement (largest gainers AND losers, volatility explosions,
    relative-performance anomalies vs the market). It decides only WHAT DESERVES
    ATTENTION — never direction or entry signals. Stores pairlist_universe:big_movers
    with a per-symbol breakdown. Degrades to a |24h change|-ranked list (never empty)."""
    target_size = max(10, int(parse_float(get_setting("big_movers_universe_size", "50")) or 50))
    weights = {
        "abs_chg": float(parse_float(get_setting("big_movers_w_abs_chg", "1.0")) or 1.0),
        "vol_expansion": float(parse_float(get_setting("big_movers_w_vol_expansion", "0.9")) or 0.9),
        "recent_swing": float(parse_float(get_setting("big_movers_w_recent_swing", "0.8")) or 0.8),
        "rel_anomaly": float(parse_float(get_setting("big_movers_w_rel_anomaly", "0.7")) or 0.7),
    }
    abs_fallback = [e["base"] for e in sorted(entries, key=lambda x: abs(float(x.get("chg") or 0.0)), reverse=True)[:target_size]]

    def _store(bases: list[str], movers: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "name": "big_movers",
            "bases": bases,
            "count": len(bases),
            "built_at": iso_now(),
            "source": "coingecko+movers",
            "movers": movers,
        }
        if extra:
            payload.update(extra)
        replace_generated_json("pairlist_universe:big_movers", payload)
        return payload

    scored: list[dict[str, Any]] = []
    for entry in entries:
        metrics = _big_mover_metrics(entry.get("spark") or [], float(entry.get("chg") or 0.0))
        if metrics is None:
            continue
        scored.append({"base": entry["base"], "rank": entry["rank"], **metrics})

    if len(scored) < max(12, target_size // 2):
        return _store(abs_fallback, [], {"degraded": "insufficient_scored_pool"})

    try:
        import numpy as np

        def _rank_norm(values: list[float]) -> "np.ndarray":
            arr = np.asarray(values, dtype=float)
            if arr.size <= 1:
                return np.zeros_like(arr)
            return np.argsort(np.argsort(arr)).astype(float) / float(arr.size - 1)

        # Relative-performance anomaly: how far this coin's 24h move sits from the market
        # median move (in either direction) — a coin diverging hard from the pack.
        median_chg = float(np.median([s["signed_chg"] for s in scored]))
        for sc in scored:
            sc["rel_anomaly"] = abs(sc["signed_chg"] - median_chg)
        norm = {key: _rank_norm([s[key] for s in scored]) for key in weights}
        composite = (
            weights["abs_chg"] * norm["abs_chg"]
            + weights["vol_expansion"] * norm["vol_expansion"]
            + weights["recent_swing"] * norm["recent_swing"]
            + weights["rel_anomaly"] * norm["rel_anomaly"]
        )
    except Exception as exc:  # noqa: BLE001 — never let scoring kill the cycle
        log_maintenance("pairlist", "warn", f"big_movers scoring failed ({exc}); abs-change fallback.")
        return _store(abs_fallback, [], {"degraded": f"scoring_error:{exc}"})

    for idx, sc in enumerate(scored):
        sc["score"] = round(float(composite[idx]), 5)
    scored.sort(key=lambda s: s["score"], reverse=True)

    bases: list[str] = []
    movers: list[dict[str, Any]] = []
    for sc in scored[:target_size]:
        bases.append(sc["base"])
        movers.append({
            "base": sc["base"], "rank": sc["rank"], "score": sc["score"],
            "chg_24h": round(sc["signed_chg"], 3), "vol_expansion": round(sc["vol_expansion"], 4),
            "recent_swing": round(sc["recent_swing"], 4), "rel_anomaly": round(sc.get("rel_anomaly", 0.0), 4),
        })
    for base in abs_fallback:  # pad if too few cleanly-scored symbols
        if len(bases) >= target_size:
            break
        if base not in bases:
            bases.append(base)
    return _store(bases[:target_size], movers, {"scored_pool": len(scored)})


# --- exchange-specific tradable manifests (ccxt filter) ---------------------

def load_exchange_base_index(exchange_id: str, profile: dict[str, Any], market_type: str, quote: str) -> dict[str, str]:
    """base symbol -> Freqtrade pair (ccxt's symbol already matches, e.g. BTC/USDT:USDT).

    Reuses the profile's ccxt options so per-exchange quirks (hyperliquid needs
    fetchMarkets.types=[swap]; binance must NOT set it) are respected here too.
    """
    klass = getattr(ccxt, exchange_id)
    client = klass({"enableRateLimit": True, "options": copy.deepcopy(profile.get("ccxt_options") or {})})
    markets = client.load_markets()
    by_base: dict[str, str] = {}
    for _symbol, market in markets.items():
        if market.get("active") is False:
            continue
        if str(market.get("quote") or "").upper() != quote:
            continue
        if market_type == "futures":
            if not market.get("swap"):
                continue
            if str(market.get("settle") or "").upper() != quote:
                continue
        elif not market.get("spot"):
            continue
        base = str(market.get("base") or "").upper()
        if not base or base in STABLE_BASE_SYMBOLS:
            continue
        by_base.setdefault(base, str(market.get("symbol")))
    return by_base


def build_exchange_manifests() -> list[str]:
    refresh = max(300, int(get_setting("pairlist_manifest_minutes", "360") or "360") * 60)
    universes: dict[str, list[str]] = {}
    for name in CANONICAL_UNIVERSE_NAMES:
        payload = get_generated_json(f"pairlist_universe:{name}", None)
        if isinstance(payload, dict) and payload.get("bases"):
            universes[name] = list(payload["bases"])
    if not universes:
        return []
    with closing(get_db()) as conn:
        resources = [dict(row) for row in conn.execute(
            "SELECT * FROM exchange_resources WHERE enabled = 1"
        ).fetchall()]
    built: list[str] = []
    for resource in resources:
        exchange_id = str(resource["exchange_id"])
        market_type = str(resource["market_type"])
        profile = exchange_profile_by_id(exchange_id)
        if not profile:
            log_maintenance("pairlist", "warn", f"No exchange profile for {exchange_id}; manifest skipped.")
            continue
        quote = exchange_quote_currency(profile)
        try:
            by_base = load_exchange_base_index(exchange_id, profile, market_type, quote)
        except Exception as exc:  # noqa: BLE001 — one bad exchange must not kill the cycle
            log_maintenance("pairlist", "error", f"load_markets failed for {exchange_id}: {exc}")
            continue
        for universe_name, bases in universes.items():
            pairs = [by_base[base] for base in bases if base in by_base]
            name = manifest_name(universe_name, exchange_id, market_type, quote)
            payload = {
                "pairs": pairs,
                "refresh_period": refresh,
                "built_at": iso_now(),
                "source_universe": universe_name,
                "exchange": exchange_id,
                "market_type": market_type,
                "quote": quote,
                "count": len(pairs),
            }
            replace_generated_json(f"pairlist_manifest:{name}", payload)
            built.append(name)
    return built


# === Universe-Centric Backtesting Engine ====================================
# Freqtrade resolves a pairlist ONCE at backtest start, so it cannot replay a
# self-seeding universe. We instead reconstruct each universe's historical
# membership from the local OHLCV data (re-running the SAME builders via
# compute_universes_from_entries) and gate entries by membership-at-time inside
# the backtest (UniverseBacktestMixin.confirm_trade_entry). v1 reconstruction
# uses only locally-downloaded pairs, so the union whitelist is data-complete by
# construction, and approximates market-cap rank with the current (slow-moving)
# CoinGecko ordering held constant across the window.


# Bases that aren't perp-leverage-tradable (e.g. tokenized commodities) and make
# freqtrade futures backtests abort with "got no leverage tiers available". Excluded
# from the candidate pool entirely so they never enter a universe or union whitelist.
BACKTEST_BASE_BLACKLIST = {"XAUT", "PAXG", "XAU", "EUR", "GBP"}


def _ohlcv_data_dir(exchange: str, market_type: str) -> Path:
    return PROJECT_DIR / "user_data" / "data" / exchange / market_type


def _local_ohlcv_pairs(exchange: str, market_type: str, timeframe: str = "1h") -> dict[str, Path]:
    """base symbol -> feather path for locally-downloaded {timeframe} futures candles."""
    directory = _ohlcv_data_dir(exchange, market_type)
    out: dict[str, Path] = {}
    if not directory.exists():
        return out
    suffix = f"-{timeframe}-futures.feather"
    for path in directory.glob(f"*{suffix}"):
        base = path.name[: -len(suffix)].split("_")[0].upper()
        if base and base not in STABLE_BASE_SYMBOLS and base not in BACKTEST_BASE_BLACKLIST:
            out.setdefault(base, path)
    return out


def _local_base_to_pair(exchange: str, market_type: str, timeframe: str = "1h") -> dict[str, str]:
    """base -> ccxt pair (BASE/QUOTE:SETTLE), parsed from the local feather filenames.
    Used instead of load_exchange_base_index() in backtesting so the reconstructed
    universe only ever contains pairs we actually have candle data for."""
    out: dict[str, str] = {}
    suffix = f"-{timeframe}-futures.feather"
    for base, path in _local_ohlcv_pairs(exchange, market_type, timeframe).items():
        stub = path.name[: -len(suffix)]
        parts = stub.split("_")
        if len(parts) >= 3:
            out[base] = f"{parts[0]}/{parts[1]}:{parts[2]}"
        elif len(parts) == 2:
            out[base] = f"{parts[0]}/{parts[1]}"
    return out


def proxy_rank_index() -> dict[str, int]:
    """Current CoinGecko market-cap ordering as a {base: rank} proxy held constant
    across the backtest window (market-cap rank moves slowly). Empty if the live
    universe has never been built — callers then fall back to a volume proxy."""
    payload = get_generated_json("pairlist_universe:top100_marketcap", None)
    bases = payload.get("bases", []) if isinstance(payload, dict) else []
    return {str(base).upper(): index + 1 for index, base in enumerate(bases)}


def reconstruct_entries_asof(
    exchange: str,
    market_type: str,
    as_of: datetime,
    rank_index: dict[str, int],
    timeframe: str = "1h",
    lookback_hours: int = 168,
) -> list[dict[str, Any]]:
    """Rebuild the `entries` snapshot ({base, rank, volume, chg, spark}) AS OF a past
    date from local OHLCV, so compute_universes_from_entries() yields that day's
    historical universe membership. Candidate pool = locally-downloaded pairs only."""
    import pandas as pd

    pairs = _local_ohlcv_pairs(exchange, market_type, timeframe)
    entries: list[dict[str, Any]] = []
    for base, path in pairs.items():
        try:
            df = pd.read_feather(path, columns=["date", "close", "volume"])
        except Exception:  # noqa: BLE001 - one unreadable file must not kill the snapshot
            continue
        df = df[df["date"] <= as_of]
        if len(df) < 25:  # need >= 24h of history plus the current candle
            continue
        window = df.tail(lookback_hours)
        spark = [float(value) for value in window["close"].tolist()]
        last24 = window.tail(25)
        close_now = float(last24["close"].iloc[-1])
        close_prev = float(last24["close"].iloc[0])
        if not (math.isfinite(close_now) and math.isfinite(close_prev) and close_prev > 0):
            continue
        chg = ((close_now / close_prev) - 1.0) * 100.0
        quote_volume = float((last24["close"] * last24["volume"]).tail(24).sum())
        entries.append({
            "base": base,
            "rank": rank_index.get(base, 9999),
            "volume": quote_volume,
            "chg": chg,
            "spark": spark,
        })
    # No market-cap proxy available -> approximate rank by 24h quote volume so the
    # market-cap universes still produce a sensible ordering instead of arbitrary ties.
    if entries and all(entry["rank"] == 9999 for entry in entries):
        for index, entry in enumerate(sorted(entries, key=lambda e: e["volume"], reverse=True), start=1):
            entry["rank"] = index
    return entries


def membership_timeline_path(universe: str, exchange: str, market_type: str, quote: str) -> Path:
    return UNIVERSE_HISTORY_DIR / f"{universe}_{exchange}_{market_type}_{quote.lower()}.json"


def build_membership_timeline(
    universe: str,
    exchange: str,
    market_type: str,
    quote: str,
    start: str | datetime,
    end: str | datetime,
    step_days: int = 1,
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Reconstruct {date -> [pairs]} membership for one universe across [start, end]
    and persist it (plus the union whitelist) to user_data/universe_history/. The
    file is mounted into the freqtrade container and read by UniverseBacktestMixin."""
    start_dt = normalize_utc(resolve_optional_datetime(start) if isinstance(start, str) else start)
    end_dt = normalize_utc(resolve_optional_datetime(end) if isinstance(end, str) else end)
    if not start_dt or not end_dt or start_dt > end_dt:
        raise ValueError("Invalid timerange for membership timeline.")
    base_to_pair = _local_base_to_pair(exchange, market_type, timeframe)
    # Per-day market-cap rank from real historical caps when available, else the static
    # current-ordering proxy. This is what makes Top-N membership time-vary accurately.
    rank_provider = marketcap_rank_provider()
    rank_source = "coingecko-historical" if rank_provider else "proxy-static"
    if rank_provider is None:
        static_index = proxy_rank_index()
        rank_provider = lambda _as_of: static_index  # noqa: E731
    timeline: dict[str, list[str]] = {}
    union: set[str] = set()
    current = start_dt
    while current <= end_dt:
        rank_index = rank_provider(current)
        entries = reconstruct_entries_asof(exchange, market_type, current, rank_index, timeframe)
        universes = compute_universes_from_entries(entries, source="backtest-local")
        bases = universes.get(universe, {}).get("bases", []) if isinstance(universes.get(universe), dict) else []
        pairs = sorted({base_to_pair[base] for base in bases if base in base_to_pair})
        timeline[current.date().isoformat()] = pairs
        union.update(pairs)
        current += timedelta(days=step_days)
    payload = {
        "universe": universe,
        "exchange": exchange,
        "market_type": market_type,
        "quote": quote.upper(),
        "timeframe": timeframe,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "step_days": step_days,
        "built_at": iso_now(),
        "source": "backtest-local-reconstruction",
        "rank_source": rank_source,
        "union_pairs": sorted(union),
        "days": len(timeline),
        "timeline": timeline,
    }
    UNIVERSE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    membership_timeline_path(universe, exchange, market_type, quote).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def universe_pool_bases(limit: int = 100) -> list[str]:
    """The broad candidate-pool bases: the live Top-100 market-cap universe (the widest
    canonical universe). Reconstruction sub-selects from whatever subset has local data."""
    payload = get_generated_json("pairlist_universe:top100_marketcap", {}) or {}
    return list(payload.get("bases", []))[:limit]


def backtest_quote_for(exchange: str) -> str:
    """Quote currency to use for the broad backtest pool. Binance's deepest perp history
    is USDT (the league profile defaults it to USDC); other exchanges follow the profile."""
    if exchange == "binance":
        return "USDT"
    profile = exchange_profile_by_id(exchange) or {}
    return exchange_quote_currency(profile)


def build_download_config(exchange: str, market_type: str, quote: str, pairs: list[str]) -> dict[str, Any]:
    profile = exchange_profile_by_id(exchange) or {}
    options = copy.deepcopy(profile.get("ccxt_options") or {})
    # Broad backtest pool uses the requested quote (USDT on binance has the deepest
    # perp history); align defaultSettle so those markets load. fetchMarkets.types is
    # left exactly as the profile sets it (binance must omit it; hyperliquid needs it).
    options["defaultSettle"] = quote
    return {
        "dry_run": True,
        "stake_currency": quote,
        "stake_amount": "unlimited",
        "trading_mode": "futures" if market_type == "futures" else "spot",
        "margin_mode": "isolated" if market_type == "futures" else "",
        "exchange": {
            "name": exchange,
            "ccxt_config": {"enableRateLimit": True, "options": options},
            "pair_whitelist": pairs,
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
    }


def download_universe_pool(
    exchange: str = "binance",
    market_type: str = "futures",
    quote: str = "USDT",
    timeframes: tuple[str, ...] = ("1h", "5m"),
    days: int = 180,
    max_pairs: int | None = None,
    timeout: int = 5400,
) -> dict[str, Any]:
    """Download OHLCV for the broad candidate pool (Top-100 bases mapped to the
    exchange's tradable perps) so universe reconstruction has a pool wider than any
    single universe and the universes actually separate. Runs `freqtrade download-data`
    in docker with a proper per-exchange config."""
    profile = exchange_profile_by_id(exchange)
    if not profile:
        return {"ok": False, "error": f"No exchange profile for {exchange}."}
    try:
        by_base = load_exchange_base_index(exchange, profile, market_type, quote)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"load_markets failed for {exchange}: {exc}"}
    bases = universe_pool_bases(100)
    pairs = [by_base[base] for base in bases if base in by_base]
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        return {"ok": False, "error": "No Top-100 bases mapped to this exchange."}
    cfg = build_download_config(exchange, market_type, quote, pairs)
    cfg_dir = PROJECT_DIR / "user_data" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"ubt_download_{exchange}_{quote.lower()}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    cfg_rel = relative_project_path(cfg_path)
    user_data_mount = f"{PROJECT_DIR / 'user_data'}:/freqtrade/user_data"
    tf_args = " ".join(timeframes)
    inner = (
        f"freqtrade download-data --config '/freqtrade/{cfg_rel}' "
        f"--timeframes {tf_args} --days {days} --data-format-ohlcv feather"
    )
    script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        docker run --rm -v "{user_data_mount}" --entrypoint sh freqtradeorg/freqtrade:stable -lc "{inner}"
        """
    ).strip()
    ok, output = _run_powershell(script, timeout=timeout)
    return {
        "ok": ok,
        "exchange": exchange,
        "quote": quote,
        "pairs_requested": len(pairs),
        "timeframes": list(timeframes),
        "days": days,
        "pairs_local_after": len(_local_ohlcv_pairs(exchange, market_type, timeframes[0])),
        "log": output[-3000:],
    }


def marketcap_history_path() -> Path:
    return UNIVERSE_HISTORY_DIR / "marketcap_history.json"


def fetch_coingecko_market_caps_history(days: int = 180, bases: list[str] | None = None) -> dict[str, Any]:
    """Pull real per-coin historical market caps (CoinGecko /coins/{id}/market_chart/range)
    for the candidate-pool coins and cache them, so Top-N membership can time-vary
    accurately instead of using the static current-ordering proxy. Rate-limited and
    resilient: sleeps between calls and skips coins that fail."""
    bases = [b.upper() for b in (bases or universe_pool_bases(100))]
    headers: dict[str, str] = {}
    key = get_setting("coingecko_api_key", "").strip()
    if key:
        headers["x-cg-demo-api-key"] = key
    # Resolve symbol -> CoinGecko id from the current markets snapshot.
    id_by_base: dict[str, str] = {}
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1},
            headers=headers,
        )
        resp.raise_for_status()
        for row in resp.json():
            sym = str(row.get("symbol") or "").upper()
            if sym and sym not in id_by_base and row.get("id"):
                id_by_base[sym] = str(row["id"])
    end = utc_now()
    start = end - timedelta(days=days)
    series: dict[str, list[list[Any]]] = {}
    fetched = 0
    failed = 0
    with httpx.Client(timeout=45.0) as client:
        for base in bases:
            coin_id = id_by_base.get(base)
            if not coin_id:
                continue
            try:
                resp = client.get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range",
                    params={"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())},
                    headers=headers,
                )
                if resp.status_code == 429:
                    time.sleep(20)
                    resp = client.get(
                        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range",
                        params={"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())},
                        headers=headers,
                    )
                resp.raise_for_status()
                caps = resp.json().get("market_caps", []) or []
                daily: dict[str, float] = {}
                for ts_ms, cap in caps:
                    day = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).date().isoformat()
                    daily[day] = float(cap)  # last sample of each day wins
                series[base] = sorted([day, cap] for day, cap in daily.items())
                fetched += 1
            except Exception:  # noqa: BLE001
                failed += 1
            time.sleep(2.5 if not key else 0.5)  # be gentle on the free tier
    payload = {
        "series": series,
        "coins": fetched,
        "failed": failed,
        "days": days,
        "built_at": iso_now(),
        "source": "coingecko-market_chart-range",
    }
    UNIVERSE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    marketcap_history_path().write_text(json.dumps(payload), encoding="utf-8")
    return {"ok": True, "coins": fetched, "failed": failed, "days": days}


def marketcap_rank_provider():
    """Return a callable as_of(datetime) -> {base: rank} from cached historical market
    caps, or None if the cache is absent. Rank 1 = largest cap on/before that date."""
    path = marketcap_history_path()
    if not path.exists():
        return None
    try:
        series = json.loads(path.read_text(encoding="utf-8")).get("series", {}) or {}
    except Exception:  # noqa: BLE001
        return None
    if not series:
        return None

    def provider(as_of: datetime) -> dict[str, int]:
        day = as_of.date().isoformat()
        caps: dict[str, float] = {}
        for base, points in series.items():
            base_upper = base.upper()
            if base_upper in STABLE_BASE_SYMBOLS or base_upper in BACKTEST_BASE_BLACKLIST:
                continue  # a stablecoin must not occupy a Top-N rank slot
            cap = None
            for point_day, point_cap in points:
                if point_day <= day:
                    cap = point_cap
                else:
                    break
            if cap is not None:
                caps[base_upper] = cap
        ranked = sorted(caps.items(), key=lambda item: item[1], reverse=True)
        return {base: index + 1 for index, (base, _) in enumerate(ranked)}

    return provider


BASE_BACKTEST_CONFIG_PATH = PROJECT_DIR / "user_data" / "config.backtest-cosmowanda.json"


def fear_greed_history_path() -> Path:
    return SENTIMENT_DATA_DIR / "fear_greed.json"


def fetch_fear_greed_history() -> dict[str, Any]:
    """Download alternative.me's full Fear & Greed archive and cache it as
    {date -> value} so the backtest mixin can read the as-of-date sentiment instead
    of CosmoWanda's live API call. Best-effort; the mixin falls back to neutral 50."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get("https://api.alternative.me/fng/?limit=0&format=json")
        resp.raise_for_status()
        data = resp.json()
    series: dict[str, int] = {}
    for row in data.get("data", []) or []:
        try:
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=UTC).date().isoformat()
            series[day] = int(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    payload = {"series": series, "count": len(series), "built_at": iso_now(), "source": "alternative.me"}
    SENTIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fear_greed_history_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def strategy_class_in_file(path: Path) -> str | None:
    """Name of the IStrategy subclass defined in a strategy file (class != filename
    for e.g. tiimmyturntup.py -> Timmy)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (isinstance(base, ast.Name) and base.id == "IStrategy") or (
                    isinstance(base, ast.Attribute) and base.attr == "IStrategy"
                ):
                    return node.name
    return None


def _backtest_strategy_payload(strategy_path: Path, key: str, display_name: str, timeframe: str) -> dict[str, Any]:
    if not strategy_path or not strategy_path.exists():
        raise ValueError(f"Strategy file not found for '{key}'.")
    klass = strategy_class_in_file(strategy_path) or strategy_path.stem
    # Container path of the strategy file's directory, so the generated wrapper can add
    # it to sys.path and import the real strategy (dev strategies live in a subdir).
    import_dir_container = "/freqtrade/" + relative_project_path(strategy_path.parent).replace("\\", "/")
    return {
        "module": strategy_path.stem,
        "klass": klass,
        "timeframe": timeframe or "5m",
        "strategy_path": str(strategy_path),
        "import_dir_container": import_dir_container,
        "team_id": key,
        "display_name": display_name or key,
    }


def resolve_backtest_strategy(strategy_key: str) -> dict[str, Any]:
    """Resolve a strategy_key to {module, klass, timeframe, strategy_path, import_dir}.
    Handles official team ids/names AND development candidates ('dev:<slug>' or a
    candidate slug/name)."""
    instances = {item["id"]: item for item in list_instances()}
    instance = instances.get(strategy_key)
    if not instance:
        lowered = strategy_key.strip().lower()
        for item in instances.values():
            if lowered in {
                str(item.get("display_name", "")).lower(),
                str(item.get("strategy_family", "")).lower(),
                str(item.get("name", "")).lower(),
            }:
                instance = item
                break
    if instance:
        return _backtest_strategy_payload(
            resolve_path(instance.get("strategy_path")), instance["id"],
            instance.get("display_name", strategy_key), str(instance.get("timeframe") or "5m"),
        )
    # Development candidate fallback.
    slug = strategy_key[4:] if strategy_key.startswith("dev:") else strategy_key
    lowered = slug.strip().lower()
    for cand in development_candidate_rows():
        if lowered in {str(cand.get("slug", "")).lower(), str(cand.get("name", "")).lower()}:
            payload = _backtest_strategy_payload(
                resolve_path(cand.get("strategy_path")), f"dev:{cand.get('slug')}",
                cand.get("name", slug), str(cand.get("timeframe") or "5m"),
            )
            return payload
    raise ValueError(f"Unknown strategy '{strategy_key}'.")


def make_universe_backtest_strategy(strategy_module: str, strategy_class: str, extra_import_dirs: list[str] | None = None) -> str:
    """Generate a thin subclass <Class>_UBT(UniverseBacktestMixin, <Class>) in
    user_data/strategies/backtest/ so the real strategy file is never touched. The
    wrapper extends sys.path to import both the mixin and the real strategy module
    (extra_import_dirs covers dev strategies that live in a different directory)."""
    BACKTEST_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_class = f"{strategy_class}_UBT"
    extra = list(extra_import_dirs or [])
    code = textwrap.dedent(
        f'''\
        # AUTO-GENERATED by the ATL universe backtesting engine. Do not edit by hand.
        import sys
        from pathlib import Path

        _HERE = Path(__file__).resolve().parent
        _DIRS = [str(_HERE), str(_HERE.parent)] + {extra!r}
        for _p in _DIRS:
            if _p and _p not in sys.path:
                sys.path.insert(0, _p)

        from universe_backtest_mixin import UniverseBacktestMixin
        from {strategy_module} import {strategy_class}


        class {wrapper_class}(UniverseBacktestMixin, {strategy_class}):
            pass
        '''
    )
    (BACKTEST_STRATEGY_DIR / f"{wrapper_class}.py").write_text(code, encoding="utf-8")
    return wrapper_class


def parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    start_raw, _, end_raw = timerange.partition("-")
    start = datetime.strptime(start_raw.strip(), "%Y%m%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_raw.strip(), "%Y%m%d").replace(tzinfo=UTC)
    return start, end


def build_universe_backtest_config(
    universe: str,
    exchange: str,
    quote: str,
    timeframe: str,
    union_pairs: list[str],
    timeline_rel: str,
    fgi_rel: str,
    wrapper_class: str,
) -> dict[str, Any]:
    base = json.loads(BASE_BACKTEST_CONFIG_PATH.read_text(encoding="utf-8")) if BASE_BACKTEST_CONFIG_PATH.exists() else {}
    cfg = copy.deepcopy(base)
    profile = exchange_profile_by_id(exchange) or {}
    cfg["bot_name"] = f"ubt-{wrapper_class}-{universe}"
    cfg["timeframe"] = timeframe
    cfg["trading_mode"] = "futures"
    cfg["margin_mode"] = "isolated"
    cfg["stake_currency"] = quote.upper()
    cfg.setdefault("exchange", {})
    cfg["exchange"]["name"] = exchange
    options = copy.deepcopy(profile.get("ccxt_options") or cfg["exchange"].get("ccxt_config", {}).get("options", {}))
    options["defaultSettle"] = quote.upper()  # honor the backtest pool's quote (binance -> USDT)
    cfg["exchange"]["ccxt_config"] = {"enableRateLimit": True, "options": options}
    cfg["exchange"]["pair_whitelist"] = list(union_pairs)
    cfg["exchange"]["pair_blacklist"] = []
    cfg["pairlists"] = [{"method": "StaticPairList"}]
    # Custom keys consumed by UniverseBacktestMixin (Freqtrade passes them through).
    cfg["universe_key"] = universe
    cfg["universe_timeline_path"] = timeline_rel
    cfg["fear_greed_path"] = fgi_rel
    return cfg


def _run_powershell(script: str, timeout: int = 1800) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            # freqtrade prints unicode tables; decode as utf-8 (not the Windows cp1252
            # default, which raises on box-drawing bytes) and never let output be None.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"command failed: {exc}"
    parts = [(result.stdout or "").strip(), (result.stderr or "").strip()]
    combined = "\n".join(part for part in parts if part).strip()
    return result.returncode == 0, combined


def parse_backtest_result_archive(zip_path: Path, wrapper_class: str | None = None) -> dict[str, Any]:
    """Read a freqtrade backtest archive (.zip) and return summary stats for one
    strategy. Matches the repo's standard export format."""
    with ZipFile(zip_path) as archive:
        inner_names = [
            name for name in archive.namelist()
            if name.endswith(".json") and "config" not in name and "market_change" not in name
        ]
        if not inner_names:
            return {}
        data = json.loads(archive.read(inner_names[0]))
    strategies = data.get("strategy", {}) or {}
    if wrapper_class and wrapper_class in strategies:
        cls, stats = wrapper_class, strategies[wrapper_class]
    elif strategies:
        cls, stats = next(iter(strategies.items()))
    else:
        return {}
    trades = stats.get("trades", []) or []
    roi_list = [parse_float(t.get("profit_ratio")) * 100.0 for t in trades]
    exit_dist: dict[str, int] = {}
    for trade in trades:
        tag = str(trade.get("exit_reason") or "unknown")
        exit_dist[tag] = exit_dist.get(tag, 0) + 1
    return {
        "strategy_class": cls,
        "total_trades": int(stats.get("total_trades", len(trades)) or 0),
        "open_trades": 0,  # a completed backtest closes all trades
        "profit_total_abs": round(parse_float(stats.get("profit_total_abs")), 4),
        "profit_total_pct": round(parse_float(stats.get("profit_total")) * 100.0, 4),
        "winrate_pct": round(parse_float(stats.get("winrate")) * 100.0, 2),
        "avg_roi_pct": round(parse_float(stats.get("profit_mean")) * 100.0, 4),
        "best_trade_pct": round(max(roi_list), 4) if roi_list else 0.0,
        "worst_trade_pct": round(min(roi_list), 4) if roi_list else 0.0,
        "max_drawdown_pct": round(parse_float(stats.get("max_drawdown_account")) * 100.0, 4),
        "profit_factor": round(parse_float(stats.get("profit_factor")), 4),
        "avg_hold_minutes": round(parse_float(stats.get("holding_avg_s")) / 60.0, 2),
        "exit_tag_distribution": exit_dist,
        "trades": trades,
        "archive": zip_path.name,
    }


def _newest_backtest_archive(since_ts: float, wrapper_class: str) -> Path | None:
    """The most recent backtest-result-*.zip created since `since_ts` whose meta.json
    declares `wrapper_class` (freqtrade ignores --export-filename and writes its own
    timestamped name, so we detect the archive by class + mtime instead)."""
    matches: list[Path] = []
    for zip_path in BACKTEST_DIR.glob("backtest-result-*.zip"):
        if zip_path.stat().st_mtime + 2 < since_ts:
            continue
        meta_path = zip_path.parent / (zip_path.stem + ".meta.json")
        try:
            if meta_path.exists() and wrapper_class in json.loads(meta_path.read_text(encoding="utf-8")):
                matches.append(zip_path)
        except Exception:  # noqa: BLE001
            continue
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def universe_backtest_run_record_path(run_id: str) -> Path:
    return UNIVERSE_HISTORY_DIR / "runs" / f"{run_id}.result.json"


def list_universe_backtest_runs() -> list[dict[str, Any]]:
    """All persisted universe-backtest run records, newest first."""
    directory = UNIVERSE_HISTORY_DIR / "runs"
    runs: list[dict[str, Any]] = []
    if not directory.exists():
        return runs
    for path in directory.glob("*.result.json"):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    runs.sort(key=lambda record: str(record.get("finished_at") or ""), reverse=True)
    return runs


def run_universe_backtest(
    strategy_key: str,
    universe: str,
    exchange: str = "hyperliquid",
    market_type: str = "futures",
    timeframe: str | None = None,
    timerange: str = "20260526-20260605",
    step_days: int = 1,
) -> dict[str, Any]:
    """Reconstruct the universe's historical membership, build a StaticPairList
    backtest over the union whitelist with the in-strategy time-gate, run freqtrade
    backtesting in docker, and return run metadata + parsed result location."""
    try:
        strat = resolve_backtest_strategy(strategy_key)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "universe": universe, "strategy": strategy_key}
    timeframe = timeframe or strat["timeframe"]
    quote = backtest_quote_for(exchange)
    start_dt, end_dt = parse_timerange(timerange)
    timeline = build_membership_timeline(universe, exchange, market_type, quote, start_dt, end_dt, step_days)
    union_pairs = timeline["union_pairs"]
    if not union_pairs:
        return {
            "ok": False,
            "universe": universe,
            "strategy": strategy_key,
            "error": "No local OHLCV pairs reconstructed for this universe/exchange/window. "
                     "Download candle data for a broader pool, or pick an exchange with local data (e.g. hyperliquid).",
        }
    wrapper_class = make_universe_backtest_strategy(
        strat["module"], strat["klass"], extra_import_dirs=[strat.get("import_dir_container", "")]
    )
    timeline_rel = relative_project_path(membership_timeline_path(universe, exchange, market_type, quote))
    fgi_rel = relative_project_path(fear_greed_history_path()) if fear_greed_history_path().exists() else ""
    cfg = build_universe_backtest_config(universe, exchange, quote, timeframe, union_pairs, timeline_rel, fgi_rel, wrapper_class)

    run_id = f"{registry_slug(strat['team_id'])}__{universe}__{exchange}__{timerange}__{int(time.time())}"
    BACKTEST_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    (UNIVERSE_HISTORY_DIR / "runs").mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = BACKTEST_STRATEGY_DIR / f"{run_id}.config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    cfg_rel = relative_project_path(cfg_path)
    tape_rel = relative_project_path(UNIVERSE_HISTORY_DIR / "runs" / f"{run_id}.tape.jsonl")
    log_rel = f"user_data/logs/ubt_{run_id}.log"

    user_data_mount = f"{PROJECT_DIR / 'user_data'}:/freqtrade/user_data"
    # freqtrade ignores --export-filename and writes its own timestamped archive, so we
    # snapshot the start time and locate the new archive by wrapper class afterwards.
    started_ts = time.time()
    inner = (
        f"freqtrade backtesting "
        f"--config '/freqtrade/{cfg_rel}' "
        f"--strategy {wrapper_class} "
        f"--strategy-path /freqtrade/user_data/strategies/backtest "
        f"--timeframe {timeframe} "
        f"--timerange {timerange} "
        f"--export trades "
        f"--logfile '/freqtrade/{log_rel}'"
    )
    script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        docker run --rm -e UBT_TAPE_PATH='/freqtrade/{tape_rel}' -v "{user_data_mount}" --entrypoint sh freqtradeorg/freqtrade:stable -lc "{inner}"
        """
    ).strip()
    ok, output = _run_powershell(script, timeout=2400)

    record: dict[str, Any] = {
        "ok": ok,
        "run_id": run_id,
        "strategy": strategy_key,
        "team_id": strat["team_id"],
        "display_name": strat["display_name"],
        "universe": universe,
        "universe_name": universe_label(universe),
        "exchange": exchange,
        "market_type": market_type,
        "timeframe": timeframe,
        "timerange": timerange,
        "union_pairs": len(union_pairs),
        "days": timeline["days"],
        "tape_path": tape_rel,
        "timeline_path": timeline_rel,
        "finished_at": iso_now(),
    }
    archive = _newest_backtest_archive(started_ts, wrapper_class) if ok else None
    if archive is not None:
        stats = parse_backtest_result_archive(archive, wrapper_class)
        record.update({
            "archive": archive.name,
            "total_trades": stats.get("total_trades", 0),
            "open_trades": stats.get("open_trades", 0),
            "profit_total_abs": stats.get("profit_total_abs", 0.0),
            "profit_total_pct": stats.get("profit_total_pct", 0.0),
            "winrate_pct": stats.get("winrate_pct", 0.0),
            "avg_roi_pct": stats.get("avg_roi_pct", 0.0),
            "best_trade_pct": stats.get("best_trade_pct", 0.0),
            "worst_trade_pct": stats.get("worst_trade_pct", 0.0),
            "max_drawdown_pct": stats.get("max_drawdown_pct", 0.0),
            "profit_factor": stats.get("profit_factor", 0.0),
            "avg_hold_minutes": stats.get("avg_hold_minutes", 0.0),
            "exit_tag_distribution": stats.get("exit_tag_distribution", {}),
        })
    else:
        record["ok"] = False
        record["error"] = "Backtest produced no archive."
        record["log_tail"] = output[-1500:]
    universe_backtest_run_record_path(run_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["log"] = output[-4000:]
    return record


def run_universe_matrix(strategy_key: str, universes: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Fan one strategy across many universes (the 'try them all' path)."""
    return [run_universe_backtest(strategy_key, universe, **kwargs) for universe in universes]


# === Backtesting Department: continuously-operating evidence engine ==========
# Schedules (strategy x universe x 30d window) backtests by priority bucket, keeps
# evidence fresh, and surfaces results. Runs ONE lane at a time, paced. Results are
# evidence only and NEVER modify live behavior (the loop only writes backtest_* tables).

STANDARD_UNIVERSES = ["top20_marketcap", "top50_marketcap", "top100_marketcap", "big_movers", "future_champion", "block_party"]
DEPARTMENT_BUCKET_LABELS = {1: "Health Checks", 2: "Habitat Mapping", 3: "Evolution Validation", 4: "Research Queue", 5: "Frontier Exploration"}
DEPARTMENT_EXCHANGE = "binance"
DEPARTMENT_MARKET_TYPE = "futures"
DEPARTMENT_WINDOW_DAYS = 30


def department_enabled() -> bool:
    return get_setting("backtest_department_enabled", "true").strip().lower() == "true"


def department_paused() -> bool:
    return get_setting("backtest_department_paused", "false").strip().lower() == "true"


def department_default_timerange(days: int = DEPARTMENT_WINDOW_DAYS) -> str:
    """Most-recent `days`-day window inside the local binance 5m data range."""
    import pandas as pd
    pairs = _local_ohlcv_pairs(DEPARTMENT_EXCHANGE, DEPARTMENT_MARKET_TYPE, "5m")
    probe = pairs.get("BTC") or (next(iter(pairs.values())) if pairs else None)
    end_dt: datetime | None = None
    if probe:
        try:
            df = pd.read_feather(probe, columns=["date"])
            if len(df):
                end_dt = df["date"].iloc[-1].to_pydatetime()
        except Exception:  # noqa: BLE001
            end_dt = None
    if end_dt is None:
        end_dt = utc_now()
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=UTC)
    start_dt = end_dt - timedelta(days=days)
    return f"{start_dt.strftime('%Y%m%d')}-{end_dt.strftime('%Y%m%d')}"


def active_organisms() -> list[dict[str, Any]]:
    """Every active organism (official teams + non-archived dev candidates with a
    strategy file) the department should keep evidence fresh for."""
    out: list[dict[str, Any]] = []
    for inst in list_instances():
        out.append({
            "key": inst["id"],
            "name": inst.get("display_name", inst["id"]),
            "kind": "official",
            "strategy_path": str(resolve_path(inst.get("strategy_path")) or ""),
            "canonical_universe": universe_key(inst.get("pair_universe") or ""),
        })
    for cand in development_candidate_rows():
        if cand.get("tier") == "archived" or cand.get("lifecycle_state") == "cut_archived":
            continue
        strategy_path = resolve_path(cand.get("strategy_path"))
        if not strategy_path or not strategy_path.exists():
            continue  # only assembled/generated dev strategies are backtestable
        out.append({
            "key": f"dev:{cand.get('slug')}",
            "name": cand.get("name", cand.get("slug", "")),
            "kind": "dev",
            "strategy_path": str(strategy_path),
            "canonical_universe": universe_key(cand.get("coin_universe") or ""),
        })
    return out


def latest_backtest_result(strategy_key: str, universe: str) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM backtest_results WHERE strategy_key=? AND universe_key=? ORDER BY created_at DESC LIMIT 1",
            (strategy_key, universe),
        ).fetchone()
    return dict(row) if row else None


def evidence_is_stale(strategy_key: str, universe: str, strategy_path: str, stale_hours: float) -> bool:
    result = latest_backtest_result(strategy_key, universe)
    if not result:
        return True  # never tested
    path = Path(strategy_path) if strategy_path else None
    current_hash = sha256_file(path)[:16] if path and path.exists() else ""
    if current_hash and result.get("strategy_hash") and result["strategy_hash"] != current_hash:
        return True  # strategy code changed since last evidence
    try:
        created = normalize_utc(datetime.fromisoformat(result["created_at"]))
    except Exception:  # noqa: BLE001
        return True
    return (utc_now() - created).total_seconds() >= stale_hours * 3600


def _backtest_job_queued_or_running(strategy_key: str, universe: str) -> bool:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT 1 FROM backtest_jobs WHERE strategy_key=? AND universe_key=? AND status IN ('queued','running') LIMIT 1",
            (strategy_key, universe),
        ).fetchone()
    return bool(row)


def enqueue_backtest_job(
    strategy_key: str,
    strategy_name: str,
    strategy_version: str,
    universe: str,
    bucket: int,
    score: float,
    reason: str,
    timerange: str,
    timeframe: str = "",
    comparison_group_id: str = "",
    mode: str = "rebuild",
    title: str = "",
) -> str | None:
    """Insert a queued job unless an identical (strategy, universe) job is already
    queued/running. Returns the job_id, or None if deduped."""
    if _backtest_job_queued_or_running(strategy_key, universe):
        return None
    universe_name = universe_label(universe)
    job_id = f"{registry_slug(strategy_key)}__{universe}__{time.time_ns()}"
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO backtest_jobs (
                job_id, title, status, priority_bucket, priority_score, strategy_key, strategy_name,
                strategy_version, universe_key, universe_name, exchange, timeframe, timerange, mode,
                reason, comparison_group_id, created_at
            ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, title or f"{strategy_name} × {universe_name}", bucket, score, strategy_key,
                strategy_name, strategy_version, universe, universe_name, DEPARTMENT_EXCHANGE, timeframe,
                timerange, mode, reason, comparison_group_id, iso_now(),
            ),
        )
        conn.commit()
    return job_id


def replenish_department_queue() -> int:
    """Detect stale/missing evidence and enqueue jobs by priority bucket. Bucket 1 =
    each organism on its canonical universe (health); Bucket 2 = each organism across
    the standard universe set (habitat mapping)."""
    if not department_enabled():
        return 0
    timerange = department_default_timerange()
    health_stale_hours = parse_float(get_setting("backtest_health_stale_hours", "24")) or 24.0
    habitat_stale_hours = parse_float(get_setting("backtest_habitat_stale_hours", "168")) or 168.0
    added = 0
    resolved: list[dict[str, Any]] = []
    for org in active_organisms():
        try:
            strat = resolve_backtest_strategy(org["key"])
        except ValueError:
            continue  # not resolvable to a backtestable strategy
        version = sha256_file(Path(org["strategy_path"]))[:16] if org["strategy_path"] and Path(org["strategy_path"]).exists() else ""
        resolved.append({**org, "timeframe": strat["timeframe"], "version": version})
    # Bucket 1: health (canonical universe) — highest priority.
    for org in resolved:
        cu = org["canonical_universe"]
        if cu and cu != "unassigned" and evidence_is_stale(org["key"], cu, org["strategy_path"], health_stale_hours):
            if enqueue_backtest_job(org["key"], org["name"], org["version"], cu, 1, 100.0,
                                    "Health check: canonical universe evidence stale/missing", timerange, org["timeframe"]):
                added += 1
    # Bucket 2: habitat mapping (standard universe set).
    for org in resolved:
        group = f"habitat:{registry_slug(org['key'])}"
        for universe in STANDARD_UNIVERSES:
            if evidence_is_stale(org["key"], universe, org["strategy_path"], habitat_stale_hours):
                if enqueue_backtest_job(org["key"], org["name"], org["version"], universe, 2, 50.0,
                                        "Habitat mapping across standard universes", timerange, org["timeframe"], comparison_group_id=group):
                    added += 1
    return added


def claim_next_backtest_job(lane_id: str = "A") -> dict[str, Any] | None:
    """Atomically claim the highest-priority queued job for a lane."""
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM backtest_jobs WHERE status='queued' ORDER BY priority_bucket ASC, priority_score DESC, created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        job = dict(row)
        conn.execute(
            "UPDATE backtest_jobs SET status='running', lane_id=?, started_at=? WHERE job_id=? AND status='queued'",
            (lane_id, iso_now(), job["job_id"]),
        )
        if conn.total_changes == 0:
            return None  # lost a race
        conn.execute(
            "UPDATE backtest_lanes SET status='running', current_job_id=?, started_at=?, updated_at=? WHERE lane_id=?",
            (job["job_id"], iso_now(), iso_now(), lane_id),
        )
        conn.commit()
    return job


def complete_backtest_job(job: dict[str, Any], record: dict[str, Any]) -> None:
    lane_id = job.get("lane_id") or "A"
    with closing(get_db()) as conn:
        if record.get("ok"):
            summary = (
                f"{record.get('total_trades', 0)} trades, pnl {parse_float(record.get('profit_total_abs')):.2f}, "
                f"win {parse_float(record.get('winrate_pct')):.1f}%"
            )
            conn.execute(
                "UPDATE backtest_jobs SET status='completed', completed_at=?, result_summary=?, run_id=?, failure_reason='' WHERE job_id=?",
                (iso_now(), summary, record.get("run_id", ""), job["job_id"]),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_results (
                    job_id, strategy_key, universe_key, total_pnl, total_pnl_pct, closed_trades, open_trades,
                    win_rate, avg_roi, best_trade, worst_trade, max_drawdown, profit_factor, avg_hold_minutes,
                    exit_tag_distribution, archive, run_id, strategy_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["job_id"], job["strategy_key"], job["universe_key"],
                    parse_float(record.get("profit_total_abs")), parse_float(record.get("profit_total_pct")),
                    int(record.get("total_trades", 0) or 0), int(record.get("open_trades", 0) or 0),
                    parse_float(record.get("winrate_pct")), parse_float(record.get("avg_roi_pct")),
                    parse_float(record.get("best_trade_pct")), parse_float(record.get("worst_trade_pct")),
                    parse_float(record.get("max_drawdown_pct")), parse_float(record.get("profit_factor")),
                    parse_float(record.get("avg_hold_minutes")), json.dumps(record.get("exit_tag_distribution", {})),
                    record.get("archive", ""), record.get("run_id", ""), job.get("strategy_version", ""), iso_now(),
                ),
            )
        else:
            reason = str(record.get("error") or record.get("log_tail") or "backtest failed")[:600]
            conn.execute(
                "UPDATE backtest_jobs SET status='failed', completed_at=?, failure_reason=? WHERE job_id=?",
                (iso_now(), reason, job["job_id"]),
            )
        conn.execute(
            "UPDATE backtest_lanes SET status='idle', current_job_id='', last_job_id=?, started_at=NULL, updated_at=?, failure_reason=? WHERE lane_id=?",
            (job["job_id"], iso_now(), "" if record.get("ok") else str(record.get("error", ""))[:300], lane_id),
        )
        conn.commit()


def run_department_once() -> bool:
    """Claim and run a single job on lane A. Returns True if a job ran."""
    if not department_enabled() or department_paused():
        return False
    with closing(get_db()) as conn:
        lane = conn.execute("SELECT status FROM backtest_lanes WHERE lane_id='A'").fetchone()
    if lane and lane["status"] == "running":
        return False
    job = claim_next_backtest_job("A")
    if not job:
        return False
    try:
        record = run_universe_backtest(
            job["strategy_key"], job["universe_key"],
            exchange=job["exchange"] or DEPARTMENT_EXCHANGE,
            timeframe=(job["timeframe"] or None),
            timerange=job["timerange"],
        )
    except Exception as exc:  # noqa: BLE001 - a bad job must not kill the lane
        record = {"ok": False, "error": str(exc)}
    complete_backtest_job(job, record)
    return True


def _department_gap_elapsed() -> bool:
    last = get_setting("backtest_department_last_run", "")
    gap_minutes = parse_float(get_setting("backtest_department_min_gap_minutes", "2")) or 2.0
    if not last:
        return True
    try:
        last_dt = normalize_utc(datetime.fromisoformat(last))
    except Exception:  # noqa: BLE001
        return True
    return (utc_now() - last_dt).total_seconds() >= gap_minutes * 60


def backtesting_department_loop() -> None:
    """Always-on, single-lane, paced scheduler. Keeps evidence fresh by priority
    bucket. SAFETY: only ever runs backtests and writes the backtest_* tables; it
    never edits configs, promotes, regenerates, or touches live trading."""
    while True:
        try:
            if department_enabled() and not department_paused() and _department_gap_elapsed():
                replenish_department_queue()
                run_department_once()
                set_setting("backtest_department_last_run", iso_now())
        except Exception as exc:  # noqa: BLE001
            log_maintenance("backtesting", "error", f"Department loop failed: {exc}")
        time.sleep(60)


def run_pairlist_manifest_cycle() -> None:
    if not resource_governance_enabled():
        return
    universes = build_canonical_universes()
    manifests = build_exchange_manifests()
    set_setting("pairlist_manifest_last_run", iso_now())
    log_maintenance(
        "pairlist",
        "ok",
        f"Built {len(universes)} canonical universes and {len(manifests)} exchange manifests.",
    )


# --- exchange lease allocator ----------------------------------------------

def sanitize_manifest_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-")


def shift_seconds_for_tier(tier: str) -> int:
    return 43200 if tier == "twelve_hour" else 21600


def shift_id_for_candidate(candidate: dict[str, Any], now_local: datetime | None = None) -> dict[str, Any] | None:
    now_local = now_local or local_now()
    tier = str(candidate.get("tier") or "")
    shift_code = str(candidate.get("shift_code") or "")
    if tier not in {"six_hour", "twelve_hour"} or not shift_code:
        return None
    active = active_shift_for_tier(tier, now_local)
    if active and active["code"] == shift_code:
        day = active["started_at"][:10]
        lease_start, lease_end = active["started_at"], active["ends_at"]
    else:
        shift = next((s for s in development_shift_definitions(tier) if s["code"] == shift_code), None)
        if not shift:
            return None
        start_dt, end_dt = shift_window_bounds(shift, now_local)
        day = now_local.date().isoformat()
        lease_start, lease_end = start_dt.isoformat(), end_dt.isoformat()
    return {
        "shift_id": f"{tier}:{shift_code}:{day}",
        "tier": tier,
        "shift_code": shift_code,
        "lease_start": lease_start,
        "lease_end": lease_end,
    }


def canonical_universe_for_candidate(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("coin_universe") or "").lower()
    # Cartographer candidates (Block Party) lease the constellation universe.
    if ("block" in text and "party" in text) or "constellation" in text:
        return "block_party"
    # Scouting candidates (The Turnstile) lease the Future Champion universe.
    if "future champion" in text or "turnstile" in text or "scout" in text:
        return "future_champion"
    # Event-cartographer candidates (Second Act) lease the Big Movers universe.
    if "second act" in text or "big mover" in text or "movers" in text:
        return "big_movers"
    size = parse_candidate_universe_size(str(candidate.get("coin_universe") or "")) or 20
    if size <= 20:
        return "top20_marketcap"
    if size <= 50:
        return "top50_marketcap"
    return "top100_marketcap"


def get_active_lease_for_candidate(candidate: dict[str, Any], now_local: datetime | None = None) -> dict[str, Any] | None:
    info = shift_id_for_candidate(candidate, now_local)
    if not info:
        return None
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM exchange_leases WHERE candidate_id = ? AND shift_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
            (int(candidate["id"]), info["shift_id"]),
        ).fetchone()
    return dict(row) if row else None


def freeze_lease_manifest(base_manifest: str, shift_id: str, exchange_id: str, quote: str, market_type: str) -> tuple[str, int]:
    """Snapshot the live exchange manifest into a sealed per-shift copy so a
    mid-shift canonical rebuild can never mutate a scored bot's universe."""
    base = get_pairlist_manifest(base_manifest)
    pairs = list(base.get("pairs", [])) if base else []
    frozen_name = f"shift_{sanitize_manifest_token(shift_id)}_{exchange_id}_{quote.lower()}"
    payload = {
        "pairs": pairs,
        "refresh_period": 86400,
        "built_at": iso_now(),
        "frozen_from": base_manifest,
        "shift_id": shift_id,
        "exchange": exchange_id,
        "market_type": market_type,
        "quote": quote,
        "sealed": True,
        "count": len(pairs),
    }
    replace_generated_json(f"pairlist_manifest:{frozen_name}", payload)
    return frozen_name, len(pairs)


def request_exchange_lease(candidate: dict[str, Any], now_local: datetime | None = None) -> dict[str, Any] | None:
    if not resource_governance_enabled():
        return None
    info = shift_id_for_candidate(candidate, now_local)
    if not info:
        return None
    existing = get_active_lease_for_candidate(candidate, now_local)
    if existing:
        return existing
    market_type = "futures"
    with closing(get_db()) as conn:
        resources = [dict(r) for r in conn.execute(
            "SELECT * FROM exchange_resources WHERE enabled = 1 AND market_type = ?", (market_type,)
        ).fetchall()]
        active = [dict(r) for r in conn.execute(
            "SELECT * FROM exchange_leases WHERE status = 'active'"
        ).fetchall()]
        released = [dict(r) for r in conn.execute(
            "SELECT exchange_id, MAX(updated_at) AS rel FROM exchange_leases WHERE status IN ('released','expired') GROUP BY exchange_id"
        ).fetchall()]
    per_shift = Counter(l["exchange_id"] for l in active if l["shift_id"] == info["shift_id"])
    total = Counter(l["exchange_id"] for l in active)
    last_release = {r["exchange_id"]: r["rel"] for r in released}
    universe = canonical_universe_for_candidate(candidate)
    eligible: list[dict[str, Any]] = []
    for resource in resources:
        exchange_id = resource["exchange_id"]
        if per_shift[exchange_id] >= int(resource["max_dev_bots_per_shift"]):
            continue
        if total[exchange_id] >= int(resource["max_total_concurrent_bots"]):
            continue
        cooldown = int(resource["cooldown_minutes"] or 0)
        if cooldown > 0 and last_release.get(exchange_id):
            try:
                if (utc_now() - datetime.fromisoformat(last_release[exchange_id])).total_seconds() < cooldown * 60:
                    continue
            except ValueError:
                pass
        # Only consider exchanges that can actually serve a sealed universe right now.
        # (Skips e.g. an exchange whose ccxt load_markets failed this cycle.)
        candidate_quote = exchange_quote_currency(exchange_profile_by_id(exchange_id) or {})
        candidate_manifest = get_pairlist_manifest(manifest_name(universe, exchange_id, market_type, candidate_quote))
        if not candidate_manifest or not candidate_manifest.get("pairs"):
            continue
        eligible.append(resource)
    if not eligible:
        log_maintenance("exchange_lease", "warn",
                        f"No eligible exchange for {candidate.get('slug')} shift {info['shift_id']} "
                        f"(capacity/cooldown/empty manifests).")
        return None
    rotation = development_exchange_rotation_snapshot().get("rotation_order", [])

    def load_key(resource: dict[str, Any]) -> tuple[int, int, str]:
        exchange_id = resource["exchange_id"]
        return (total[exchange_id], rotation.index(exchange_id) if exchange_id in rotation else 99, exchange_id)

    chosen = min(eligible, key=load_key)
    exchange_id = chosen["exchange_id"]
    profile = exchange_profile_by_id(exchange_id) or {}
    quote = exchange_quote_currency(profile)
    base_manifest = manifest_name(universe, exchange_id, market_type, quote)
    base = get_pairlist_manifest(base_manifest)
    if not base or not base.get("pairs"):
        # Manifests not built yet (or empty) — don't seal an empty universe; the
        # config assembler falls back to MarketCapPairList until the cycle runs.
        log_maintenance("exchange_lease", "warn",
                        f"Manifest {base_manifest} empty; deferring lease for {candidate.get('slug')}.")
        return None
    frozen_id, frozen_count = freeze_lease_manifest(base_manifest, info["shift_id"], exchange_id, quote, market_type)
    now_iso = iso_now()
    with closing(get_db()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO exchange_leases
                (candidate_id, candidate_slug, tier, shift_code, shift_id, exchange_id,
                 market_type, pairlist_manifest_id, lease_start, lease_end, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                int(candidate["id"]), str(candidate.get("slug") or ""), info["tier"], info["shift_code"],
                info["shift_id"], exchange_id, market_type, frozen_id,
                info["lease_start"], info["lease_end"], now_iso, now_iso,
            ),
        )
        conn.commit()
        lease_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM exchange_leases WHERE id = ?", (lease_id,)).fetchone()
    log_maintenance("exchange_lease", "ok",
                    f"{candidate.get('slug')} leased {exchange_id} for {info['shift_id']} "
                    f"({universe}, {frozen_count} pairs).")
    return dict(row) if row else None


def release_exchange_lease(candidate: dict[str, Any], now_local: datetime | None = None) -> None:
    info = shift_id_for_candidate(candidate, now_local)
    with closing(get_db()) as conn:
        if info:
            conn.execute(
                "UPDATE exchange_leases SET status = 'released', updated_at = ? WHERE candidate_id = ? AND shift_id = ? AND status = 'active'",
                (iso_now(), int(candidate["id"]), info["shift_id"]),
            )
        else:
            conn.execute(
                "UPDATE exchange_leases SET status = 'released', updated_at = ? WHERE candidate_id = ? AND status = 'active'",
                (iso_now(), int(candidate["id"])),
            )
        conn.commit()


def expire_stale_leases() -> int:
    now = utc_now()
    stale: list[int] = []
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT id, lease_end FROM exchange_leases WHERE status = 'active'").fetchall()
        for row in rows:
            lease_end = row["lease_end"]
            if not lease_end:
                continue
            try:
                end_dt = datetime.fromisoformat(lease_end)
            except ValueError:
                continue
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=LOCAL_TIMEZONE)
            if end_dt.astimezone(UTC) < now:
                stale.append(int(row["id"]))
        for lease_id in stale:
            conn.execute("UPDATE exchange_leases SET status = 'expired', updated_at = ? WHERE id = ?", (iso_now(), lease_id))
        if stale:
            conn.commit()
    return len(stale)


def lease_pairlist_block(lease: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    number_assets = parse_candidate_universe_size(str(candidate.get("coin_universe") or "")) or 20
    tier = str(candidate.get("tier") or "")
    scored = tier in {"six_hour", "twelve_hour"}
    # Scored shifts: refresh >= shift length so the sealed universe never changes
    # mid-episode. Always-on bots may refresh on the build cadence.
    if scored:
        refresh = shift_seconds_for_tier(tier)
    else:
        refresh = max(900, int(get_setting("pairlist_manifest_minutes", "360") or "360") * 60)
    return {
        "method": "RemotePairList",
        "pairlist_url": f"{pairlist_manifest_base_url()}/pairlists/{lease['pairlist_manifest_id']}.json",
        "number_assets": int(number_assets),
        "refresh_period": int(refresh),
        "keep_pairlist_on_failure": True,
    }


def governance_apply_for_shift_start(candidate: dict[str, Any]) -> None:
    """At a clean shift start, lease an exchange and rewrite the dev bot's config
    to RemotePairList so the bot never picks its own exchange/universe. Idempotent
    and change-guarded to avoid disk churn on retry ticks."""
    if not resource_governance_enabled():
        return
    if str(candidate.get("tier")) not in {"six_hour", "twelve_hour"}:
        return
    lease = request_exchange_lease(candidate)
    if not lease:
        return  # fallback: existing config (MarketCapPairList) keeps the bot running
    config_path = resolve_path(candidate.get("config_path"))
    if not config_path:
        return
    config = deterministic_candidate_config(candidate)
    current = load_json(config_path, {})
    if (
        isinstance(current, dict)
        and current.get("pairlists") == config.get("pairlists")
        and str(current.get("exchange", {}).get("name")) == str(config.get("exchange", {}).get("name"))
    ):
        return
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    development_runtime_event(
        int(candidate["id"]), "governance",
        f"Exchange lease {lease['exchange_id']} ({lease['market_type']}); pairlist {lease['pairlist_manifest_id']}.",
        f"shift {lease['shift_id']}",
    )


# Freqtrade-legal timeframes (case-sensitive: lowercase 'm' = minute, uppercase 'M' = month).
FREQTRADE_TIMEFRAMES = {
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M",
}


def normalize_freqtrade_timeframe(raw: Any) -> str:
    """Extract the first freqtrade-legal timeframe token from a raw value, else "".
    The strategy generator sometimes returns a descriptive phrase (e.g.
    '4h with 1d informative context') in suggested_timeframe; copied verbatim into a
    config it crash-loops the bot. This pulls the first valid token ('4h')."""
    text = str(raw or "").strip()
    if not text:
        return ""
    tokens = re.findall(r"[0-9]+[a-zA-Z]+", text)
    for token in tokens:  # exact match preserves the 1m/1M (minute vs month) distinction
        if token in FREQTRADE_TIMEFRAMES:
            return token
    for token in tokens:  # lenient: accept case variants like '4H' -> '4h'
        lowered = token[:-1] + token[-1].lower()
        if lowered in FREQTRADE_TIMEFRAMES:
            return lowered
    return ""


def strategy_declared_timeframe(candidate: dict[str, Any]) -> str:
    """The `timeframe = '...'` literal declared in the candidate's strategy .py — the
    source of truth when the candidate's timeframe metadata is junk."""
    path = resolve_path(candidate.get("strategy_path"))
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""
    match = re.search(r"^\s*timeframe\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return normalize_freqtrade_timeframe(match.group(1)) if match else ""


def strategy_declared_order_type(candidate: dict[str, Any], side: str) -> str:
    """The order type ('market'/'limit'/'') a strategy declares for a given side in its
    `order_types = {...}` literal. freqtrade rejects a market order whose matching
    *_pricing.price_side is "same" ("Market exit orders require price_side = 'other'"),
    so deterministic_candidate_config uses this to reconcile pricing and avoid the
    crash-loop a market-exit strategy (e.g. Dany) hits on a base config defaulting to
    "same". Returns "" when not declared (freqtrade then defaults to limit)."""
    path = resolve_path(candidate.get("strategy_path"))
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""
    block = re.search(r"order_types\s*=\s*\{(.*?)\}", text, re.DOTALL)
    if not block:
        return ""
    match = re.search(rf"['\"]{side}['\"]\s*:\s*['\"](\w+)['\"]", block.group(1))
    return match.group(1).lower() if match else ""


def resolve_candidate_config_timeframe(candidate: dict[str, Any], existing: str = "") -> str:
    """Pick a freqtrade-legal timeframe for the assembled config, preferring (in order):
    the LLM's suggested_timeframe, the candidate's timeframe field, the strategy-declared
    literal, any existing config value, then a 5m default. Each candidate source is run
    through normalize_freqtrade_timeframe so a descriptive phrase can never reach the config."""
    return (
        normalize_freqtrade_timeframe(candidate.get("suggested_timeframe"))
        or normalize_freqtrade_timeframe(candidate.get("timeframe"))
        or strategy_declared_timeframe(candidate)
        or normalize_freqtrade_timeframe(existing)
        or "5m"
    )


def deterministic_candidate_config(candidate: dict[str, Any]) -> dict[str, Any]:
    base_config = load_json(PROJECT_DIR / "user_data" / "config.json", {})
    config = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    # Under resource governance, the leased exchange (not the bot's own rotation
    # pick) decides the exchange + universe. Read-only here; the lease is created
    # at shift start by governance_apply_for_shift_start.
    governance_lease = get_active_lease_for_candidate(candidate) if resource_governance_enabled() else None
    if governance_lease:
        exchange_profile = exchange_profile_by_id(str(governance_lease.get("exchange_id"))) or pick_candidate_exchange_profile(candidate)
    else:
        exchange_profile = pick_candidate_exchange_profile(candidate)
    config["bot_name"] = development_container_name(candidate)
    config["dry_run"] = True
    config["initial_state"] = "running"
    config["timeframe"] = resolve_candidate_config_timeframe(candidate, str(config.get("timeframe") or ""))
    config["trading_mode"] = str(exchange_profile.get("trading_mode") or config.get("trading_mode") or "futures")
    config["margin_mode"] = str(exchange_profile.get("margin_mode") or config.get("margin_mode") or "isolated")
    config["stake_currency"] = str(exchange_profile.get("stake_currency") or config.get("stake_currency") or "USDT")
    # Vary concurrency per strategy while keeping the wallet fixed; with stake_amount
    # "unlimited", freqtrade auto-splits dry_run_wallet across these slots.
    config["max_open_trades"] = candidate_max_open_trades(candidate)
    config["exchange"] = {
        "name": str(exchange_profile.get("name") or "binance"),
        "ccxt_config": {
            "enableRateLimit": True,
            "options": copy.deepcopy(exchange_profile.get("ccxt_options") or {}),
        },
        "pair_whitelist": [],
        "pair_blacklist": list(DEV_STABLECOIN_BLACKLIST),
    }
    # Reconcile pricing with the strategy's order types: freqtrade refuses to start a
    # market-order strategy whose matching *_pricing.price_side is "same" (the base
    # config default). Without this a market-exit dev bot (e.g. Dany) crash-loops every
    # shift on "Market exit orders require exit_pricing.price_side = 'other'".
    if strategy_declared_order_type(candidate, "exit") == "market":
        config.setdefault("exit_pricing", {})["price_side"] = "other"
    if strategy_declared_order_type(candidate, "entry") == "market":
        config.setdefault("entry_pricing", {})["price_side"] = "other"
    config.setdefault("pairlists", [])
    config.setdefault("api_server", {})
    config["api_server"]["enabled"] = True
    port = int(candidate.get("api_port") or 18080 + int(candidate["id"]))
    config["api_server"]["listen_ip_address"] = "0.0.0.0"
    config["api_server"]["listen_port"] = port
    config["api_server"]["username"] = f"dev_{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}"
    config["api_server"]["password"] = "devpassword"
    config["api_server"]["jwt_secret_key"] = f"dev-jwt-{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}"
    config["api_server"]["ws_token"] = f"dev-ws-{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}"
    universe_size = parse_candidate_universe_size(str(candidate.get("coin_universe") or ""))
    if config["pairlists"]:
        first_pairlist = config["pairlists"][0]
        if isinstance(first_pairlist, dict) and universe_size:
            first_pairlist["number_assets"] = universe_size
            if "max_rank" in first_pairlist:
                first_pairlist["max_rank"] = max(universe_size, int(first_pairlist.get("max_rank") or universe_size))
    elif universe_size:
        config["pairlists"] = [{"method": "MarketCapPairList", "number_assets": universe_size, "max_rank": universe_size, "refresh_period": 86400}]
    # Governance override: when a sealed exchange manifest exists for this lease,
    # serve it via RemotePairList instead of each bot hitting CoinGecko itself.
    # Empty/missing manifest ⇒ keep the MarketCapPairList fallback above.
    if governance_lease:
        manifest = get_pairlist_manifest(str(governance_lease.get("pairlist_manifest_id")))
        if manifest and manifest.get("pairs"):
            config["pairlists"] = [lease_pairlist_block(governance_lease, candidate)]
    return config


def assemble_candidate_instance(candidate_id: int) -> None:
    candidate = get_development_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.get("review_status") != "reviewed" or candidate.get("validation_status") != "passed":
        raise HTTPException(status_code=400, detail="Candidate must be reviewed and validated before assembly.")
    DEV_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    DEV_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEV_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    DEV_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DEV_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_path = resolve_path(candidate.get("strategy_path")) or development_strategy_path(candidate)
    if not strategy_path.exists():
        raise HTTPException(status_code=400, detail="Strategy file is missing.")
    config_path = development_config_path(candidate)
    db_path = development_db_path(candidate)
    log_path = development_log_path(candidate)
    start_script_path, stop_script_path = development_script_paths(candidate)
    container_name = development_container_name(candidate)
    class_name = str(candidate.get("strategy_class_name") or safe_strategy_class_name(str(candidate.get("name", ""))))
    api_port = int(candidate.get("api_port") or 18080 + int(candidate["id"]))
    config_payload = deterministic_candidate_config({**candidate, "api_port": api_port})
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    db_volume_name = development_db_volume_name(candidate)
    container_db_url = f"sqlite:////freqtrade/databases/{db_path.name}"
    start_script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        $existing = docker ps -aq --filter "name=^{container_name}$"
        if ($existing) {{
          docker start "{container_name}" | Out-Null
        }} else {{
          docker volume create "{db_volume_name}" | Out-Null
          docker run --rm -v "{db_volume_name}:/freqtrade/databases" --user root --entrypoint sh freqtradeorg/freqtrade:stable -c "chown 1000:1000 /freqtrade/databases" | Out-Null
          docker run -d --name "{container_name}" --restart unless-stopped -v "{PROJECT_DIR / 'user_data'}:/freqtrade/user_data" -v "{db_volume_name}:/freqtrade/databases" -p "127.0.0.1:{api_port}:{api_port}" --entrypoint sh freqtradeorg/freqtrade:stable -lc "freqtrade trade --logfile '/freqtrade/{relative_project_path(log_path)}' --db-url '{container_db_url}' --config '/freqtrade/{relative_project_path(config_path)}' --strategy-path '/freqtrade/user_data/strategies/development' --strategy '{class_name}'"
        }}
        """
    ).strip()
    stop_script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        $existing = docker ps -aq --filter "name=^{container_name}$"
        if ($existing) {{
          docker stop "{container_name}" | Out-Null
        }}
        Write-Output "Stopped {container_name}"
        """
    ).strip()
    start_script_path.write_text(start_script + "\n", encoding="utf-8")
    stop_script_path.write_text(stop_script + "\n", encoding="utf-8")
    # A candidate that already holds a shift assignment (e.g. re-assembled after an
    # auto-regeneration from a post-shift review) must return to that shift instead of
    # being parked off-schedule. First-time assemblies have no shift_code yet, so they
    # fall through to the default "needs shift assignment" state.
    previously_assigned_to_shift = bool(str(candidate.get("shift_code") or "").strip()) and str(
        candidate.get("tier") or ""
    ) in {"six_hour", "twelve_hour"}
    if previously_assigned_to_shift:
        assembly_lifecycle_state = "assigned_to_shift"
        assembly_runtime_status = "off-shift"
        assembly_status_detail = "Re-assembled; returned to existing shift assignment."
    else:
        assembly_lifecycle_state = "instance_assembled"
        assembly_runtime_status = "paused"
        assembly_status_detail = "Instance assembled. Needs shift assignment."
    update_development_candidate(
        candidate_id,
        config_path=relative_project_path(config_path),
        db_path=relative_project_path(db_path),
        log_path=relative_project_path(log_path),
        strategy_path=relative_project_path(strategy_path),
        start_command=f"& '{start_script_path}'",
        stop_command=f"& '{stop_script_path}'",
        api_url=f"http://127.0.0.1:{api_port}",
        api_username=config_payload["api_server"]["username"],
        api_password=config_payload["api_server"]["password"],
        container_name=container_name,
        api_port=api_port,
        working_directory=relative_project_path(DEV_RUNTIME_DIR),
        assembly_status="assembled",
        assembly_error="",
        instance_assembled_at=iso_now(),
        lifecycle_state=assembly_lifecycle_state,
        runtime_status=assembly_runtime_status,
        status_detail=assembly_status_detail,
    )
    development_runtime_event(candidate_id, "assembly", "Instance assembled.", relative_project_path(config_path))


def queue_candidate_strategy_generation(
    candidate_id: int,
    force: bool = False,
    generation_trigger: str = "manual",
    auto_apply_generated_strategy: bool = False,
    approved_override: bool = False,
) -> None:
    candidate = get_development_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    # Protected Prospects have an immutable strategy file: any non-approved
    # (automated) regeneration is refused. Manual user actions pass approved_override.
    if int(candidate.get("protected") or 0) and not approved_override:
        development_runtime_event(
            candidate_id,
            "protection",
            "Strategy regeneration blocked — Protected Prospect.",
            f"trigger={generation_trigger}; requires manual approval.",
        )
        return
    if candidate.get("generation_status") in {"queued", "generating"} and not force:
        return
    update_development_candidate(
        candidate_id,
        generation_status="queued",
        generation_model="",
        generation_trigger=generation_trigger,
        auto_apply_generated_strategy="true" if auto_apply_generated_strategy else "false",
        generation_prompt="",
        generation_error="",
        generation_progress="Queued for Kimi strategy generation.",
        implementation_summary="",
        generation_assumptions="",
        generation_warnings="",
        suggested_timeframe="",
        suggested_max_open_trades=0,
        minimal_config_notes="",
        validation_status="pending",
        validation_error="",
        validated_at=None,
        review_status="pending",
        reviewed_at=None,
        assembly_status="pending",
        assembly_error="",
        instance_assembled_at=None,
        lifecycle_state="generating_strategy",
        status_detail="Strategy generation queued with Kimi.",
    )
    development_runtime_event(
        candidate_id,
        "generation",
        "Strategy generation queued.",
        f"{pick_strategy_generation_model()} · trigger={generation_trigger} · auto_apply={'yes' if auto_apply_generated_strategy else 'no'}",
    )


def process_generation_queue() -> None:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM dev_candidates
            WHERE generation_status = 'queued'
            ORDER BY updated_at ASC, id ASC
            LIMIT 1
            """
        ).fetchall()
    for row in rows:
        candidate_id = int(row["id"])
        candidate = get_development_candidate(candidate_id)
        if not candidate:
            continue
        auto_apply_generated_strategy = str(candidate.get("auto_apply_generated_strategy") or "false").lower() == "true"
        messages, prompt_used = strategy_generation_prompt(candidate)
        generation_model = pick_strategy_generation_model()
        update_development_candidate(
            candidate_id,
            generation_status="generating",
            generation_model=generation_model,
            generation_prompt=prompt_used,
            generation_progress="Preparing generation request.",
            lifecycle_state="generating_strategy",
            status_detail="Generating strategy file with Kimi.",
        )
        try:
            generation_timeout_seconds = max(420.0, parse_float(get_setting("development_strategy_generation_timeout_seconds", "480")))
            generation_retry_count = max(1, parse_intish(get_setting("development_strategy_generation_retry_count", "4")))
            generation_fallback_model = pick_strategy_generation_fallback_model()

            def attempt_callback(
                candidate_model: str,
                attempt_index: int,
                attempts_for_model: int,
                overall_attempt: int,
                total_attempts: int,
                outcome: str,
            ) -> None:
                if outcome == "success":
                    progress = f"Strategy generation succeeded with {candidate_model} on overall attempt {overall_attempt}/{total_attempts}."
                elif outcome:
                    progress = (
                        f"Attempt {overall_attempt}/{total_attempts} failed with {candidate_model} "
                        f"(model attempt {attempt_index}/{attempts_for_model}): {outcome}"
                    )
                else:
                    progress = (
                        f"Calling {candidate_model} for strategy generation "
                        f"(overall attempt {overall_attempt}/{total_attempts}, model attempt {attempt_index}/{attempts_for_model})."
                    )
                update_development_candidate(
                    candidate_id,
                    generation_progress=progress,
                    generation_error="" if not outcome or outcome == "success" else progress,
                    status_detail=progress,
                )

            payload = parse_json_block(
                ollama_chat(
                    messages,
                    preferred_models=[generation_model],
                    timeout_seconds_override=generation_timeout_seconds,
                    retry_count_override=generation_retry_count,
                    fallback_model_override=generation_fallback_model,
                    include_default_fallback=False,
                    attempt_callback=attempt_callback,
                )
            )
            strategy_code = str(payload.get("strategy_code", "")).strip()
            if not strategy_code:
                raise RuntimeError("The model returned no strategy code.")
            strategy_path = development_strategy_path(candidate)
            class_name = safe_strategy_class_name(str(candidate.get("name", "")))
            DEV_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
            strategy_path.write_text(strategy_code.rstrip() + "\n", encoding="utf-8")
            valid, validation_errors = validate_strategy_file(strategy_path, class_name)
            niche = normalize_temporal_niche(payload.get("temporal_niche"))
            try:
                suggested_trades = int(float(payload.get("suggested_max_open_trades") or 0))
            except (TypeError, ValueError):
                suggested_trades = 0
            suggested_trades = max(0, min(DEV_MAX_OPEN_TRADES_MAX, suggested_trades))  # 0 = let the deterministic spread decide
            update_development_candidate(
                candidate_id,
                generation_status="generated",
                generation_model=generation_model,
                generation_prompt=prompt_used,
                generated_at=iso_now(),
                generation_progress="Strategy file generated.",
                temporal_niche_start=niche["start"],
                temporal_niche_end=niche["end"],
                temporal_niche_note=niche["note"],
                temporal_niche_status=niche["status"],
                implementation_summary=str(payload.get("implementation_summary", "")),
                generation_assumptions=json.dumps(payload.get("assumptions", []), indent=2),
                generation_warnings=json.dumps(payload.get("warnings", []), indent=2),
                suggested_timeframe=str(payload.get("suggested_timeframe", "")),
                suggested_max_open_trades=suggested_trades,
                minimal_config_notes=json.dumps(payload.get("minimal_config_notes", []), indent=2),
                strategy_path=relative_project_path(strategy_path),
                strategy_class_name=class_name,
                validation_status="passed" if valid else "failed",
                validation_error="\n".join(validation_errors),
                validated_at=iso_now(),
                review_status="reviewed" if valid and auto_apply_generated_strategy else "pending",
                reviewed_at=iso_now() if valid and auto_apply_generated_strategy else None,
                auto_apply_generated_strategy="false",
                lifecycle_state="reviewed" if valid and auto_apply_generated_strategy else "implemented",
                status_detail="Strategy generated, auto-reviewed, and ready for reassembly." if valid and auto_apply_generated_strategy else "Strategy generated and validated. Needs human review." if valid else "Strategy generated but validation failed.",
            )
            development_runtime_event(candidate_id, "generation", "Strategy generated.", relative_project_path(strategy_path))
            development_runtime_event(candidate_id, "validation", "Validation passed." if valid else "Validation failed.", "\n".join(validation_errors))
            if valid and auto_apply_generated_strategy:
                assemble_candidate_instance(candidate_id)
                development_runtime_event(candidate_id, "review", "Auto-approved from post-shift review.", str(candidate.get("generation_trigger") or "post_shift_review"))
        except Exception as exc:  # noqa: BLE001
            update_development_candidate(
                candidate_id,
                generation_status="failed",
                generation_error=str(exc),
                generation_progress=f"Generation failed after retries: {exc}",
                auto_apply_generated_strategy="false",
                lifecycle_state="draft_idea",
                status_detail=f"Strategy generation failed: {exc}",
            )
            development_runtime_event(candidate_id, "generation", "Strategy generation failed.", str(exc))


def development_generation_loop() -> None:
    while True:
        try:
            process_generation_queue()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(DEV_GENERATION_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Mini-season shift lifecycle: force-close at the bell, archive the episode,
# wipe the runtime DB at the next clean shift start. DEV LEAGUE ONLY — every
# destructive op is triple-guarded (callable only from the dev pipeline, gated
# on shift tier, and the volume name must be an atl-dev-*-db). Major-league
# 24/7 teams are not dev_candidates and never reach this code.
# ---------------------------------------------------------------------------

DEV_FORCED_EXIT_REASONS = {"force_exit", "forced_exit", "forceexit", "force_sell", "forcesell"}
DEV_SHIFT_ARCHIVE_DIR = DATA_DIR / "archives" / "dev_shifts"


def _assert_dev_db_volume(volume: str) -> str:
    """Hard guard: refuse to operate on anything that is not a dev-league DB volume.
    This is the last line of defense protecting major-league data from a wipe."""
    name = str(volume or "").strip()
    if not (name.startswith("atl-dev-") and name.endswith("-db")):
        raise ValueError(f"Refusing destructive op on non-dev volume {name!r}")
    return name


def api_post(client: httpx.Client, instance: dict[str, Any], path: str, payload: dict[str, Any]) -> Any:
    response = client.post(
        f"{instance['api_url'].rstrip('/')}{path}",
        headers={**build_auth_header(instance), "Content-Type": "application/json"},
        json=payload,
        timeout=8.0,
    )
    response.raise_for_status()
    return response.json()


def _candidate_api_instance(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_url": candidate.get("api_url", ""),
        "api_username": candidate.get("api_username", ""),
        "api_password": candidate.get("api_password", ""),
    }


def strategy_version_token(candidate: dict[str, Any]) -> str:
    """Stable identifier for the strategy version that ran this shift, so an episode
    is unambiguously 'Dany Shift 17, version <token>'."""
    token = str(candidate.get("generated_at") or candidate.get("validated_at") or "").strip()
    if token:
        return token
    strategy_path = resolve_path(candidate.get("strategy_path"))
    if strategy_path and strategy_path.exists():
        try:
            return f"sha:{sha256_file(strategy_path)[:12]}"
        except Exception:  # noqa: BLE001
            return ""
    return ""


def current_open_session_started_at(candidate_id: int) -> str:
    try:
        with closing(get_db()) as conn:
            row = conn.execute(
                "SELECT started_at FROM dev_runtime_sessions WHERE candidate_id = ? AND stopped_at IS NULL ORDER BY id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return str(row["started_at"]) if row and row["started_at"] else ""
    except sqlite3.OperationalError:
        return ""


def fetch_shift_closed_trades(candidate: dict[str, Any], started_utc: datetime | None) -> list[dict[str, Any]] | None:
    """Pull closed trades (with exit_reason) from the bot's REST API for the current
    shift window. Returns None when the API is unreachable. Must run while the bot is live."""
    if not candidate.get("api_url"):
        return None
    instance = _candidate_api_instance(candidate)
    records: list[dict[str, Any]] = []
    try:
        with httpx.Client() as client:
            offset, limit = 0, 250
            while True:
                payload = api_get(client, instance, f"/api/v1/trades?limit={limit}&offset={offset}")
                page = payload.get("trades", []) if isinstance(payload, dict) else []
                if not page:
                    break
                for trade in page:
                    if bool(trade.get("is_open")):
                        continue
                    records.append({
                        "trade_id": trade.get("trade_id"),
                        "pair": trade.get("pair"),
                        "exit_reason": str(trade.get("exit_reason") or ""),
                        "enter_tag": str(trade.get("enter_tag") or ""),
                        "close_profit": first_present_value(trade.get("close_profit"), trade.get("profit_ratio")),
                        "close_profit_abs": first_present_value(trade.get("close_profit_abs"), trade.get("profit_abs")),
                        "realized_profit": trade.get("realized_profit"),
                        "open_date": trade.get("open_date"),
                        "close_date": trade.get("close_date"),
                        "is_short": bool(trade.get("is_short")),
                    })
                total = int(payload.get("total_trades") or len(page))
                offset += len(page)
                if offset >= total:
                    break
    except Exception:  # noqa: BLE001
        return None
    if started_utc is not None:
        windowed: list[dict[str, Any]] = []
        for record in records:
            close_date = normalize_utc(resolve_optional_datetime(str(record.get("close_date") or "")))
            if close_date is None or close_date >= started_utc:
                windowed.append(record)
        records = windowed
    return records


def split_shift_trades(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split closed trades into clock-forced (shift_expiry) vs strategy-chosen exits."""
    forced = [r for r in records if str(r.get("exit_reason") or "").lower() in DEV_FORCED_EXIT_REASONS]
    strategy = [r for r in records if str(r.get("exit_reason") or "").lower() not in DEV_FORCED_EXIT_REASONS]

    def pnl(rows: list[dict[str, Any]]) -> float:
        return round(sum(parse_float(r.get("realized_profit")) or parse_float(r.get("close_profit_abs")) for r in rows), 4)

    def wins(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if parse_float(r.get("close_profit_abs")) > 0)

    s_wins = wins(strategy)
    return {
        "closed_trades": len(records),
        "wins": wins(records),
        "losses": max(0, len(records) - wins(records)),
        "win_rate": percentage(wins(records), len(records)) if records else 0.0,
        "avg_roi": percentage(sum(parse_float(r.get("close_profit")) for r in records), len(records)) if records else 0.0,
        "realized_pnl": pnl(records),
        "forced_exits": len(forced),
        "forced_realized_pnl": pnl(forced),
        "strategy_closed_trades": len(strategy),
        "strategy_wins": s_wins,
        "strategy_win_rate": percentage(s_wins, len(strategy)) if strategy else 0.0,
        "strategy_avg_roi": percentage(sum(parse_float(r.get("close_profit")) for r in strategy), len(strategy)) if strategy else 0.0,
        "strategy_realized_pnl": pnl(strategy),
    }


def reset_candidate_runtime_db(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Wipe + re-own the bot's runtime SQLite volume so the next shift starts clean.
    DEV-ONLY and volume-guarded. Never call for a major-league team."""
    volume = _assert_dev_db_volume(development_db_volume_name(candidate))
    inner = (
        "rm -f /freqtrade/databases/*.sqlite /freqtrade/databases/*.sqlite-wal "
        "/freqtrade/databases/*.sqlite-shm 2>/dev/null; chown 1000:1000 /freqtrade/databases"
    )
    command = (
        f'docker volume create "{volume}" | Out-Null; '
        f'docker run --rm -v "{volume}:/freqtrade/databases" --user root '
        f'--entrypoint sh freqtradeorg/freqtrade:stable -c "{inner}"'
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"DB reset failed: {exc}"
    if result.returncode != 0:
        return False, (result.stderr.strip() or result.stdout.strip() or f"DB reset exited {result.returncode}")
    return True, "Runtime DB wiped for fresh shift."


def archive_shift_episode(candidate: dict[str, Any], split: dict[str, Any], records: list[dict[str, Any]], started_at: str) -> str:
    """Write the permanent per-shift episode (artifact + indexed row). Returns episode_key."""
    candidate_id = int(candidate["id"])
    slug = registry_slug(str(candidate.get("slug") or candidate.get("name") or "candidate"))
    episode_key = f"{candidate_id}:{started_at}"
    version = strategy_version_token(candidate)
    artifact_dir = DEV_SHIFT_ARCHIVE_DIR / slug
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_started = re.sub(r"[^0-9A-Za-z]", "", started_at) or iso_now().replace(":", "")
    artifact_path = artifact_dir / f"{safe_started}.json"
    artifact = {
        "candidate_id": candidate_id, "slug": slug, "name": candidate.get("name", ""),
        "tier": candidate.get("tier", ""), "shift_code": candidate.get("shift_code", ""),
        "strategy_version": version, "session_started_at": started_at, "captured_at": iso_now(),
        "stats": split, "trades": records,
    }
    try:
        artifact_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
        rel_artifact = relative_project_path(artifact_path)
    except Exception:  # noqa: BLE001
        rel_artifact = ""
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO dev_shift_episodes (
                episode_key, candidate_id, slug, name, tier, shift_code, strategy_version,
                session_started_at, session_stopped_at, closed_trades, wins, losses, win_rate, avg_roi, realized_pnl,
                forced_exits, forced_realized_pnl, strategy_closed_trades, strategy_wins, strategy_win_rate,
                strategy_avg_roi, strategy_realized_pnl, artifact_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_key) DO UPDATE SET
                session_stopped_at=excluded.session_stopped_at, closed_trades=excluded.closed_trades,
                wins=excluded.wins, losses=excluded.losses, win_rate=excluded.win_rate, avg_roi=excluded.avg_roi,
                realized_pnl=excluded.realized_pnl, forced_exits=excluded.forced_exits,
                forced_realized_pnl=excluded.forced_realized_pnl, strategy_closed_trades=excluded.strategy_closed_trades,
                strategy_wins=excluded.strategy_wins, strategy_win_rate=excluded.strategy_win_rate,
                strategy_avg_roi=excluded.strategy_avg_roi, strategy_realized_pnl=excluded.strategy_realized_pnl,
                artifact_path=excluded.artifact_path
            """,
            (
                episode_key, candidate_id, slug, str(candidate.get("name", "")), str(candidate.get("tier", "")),
                str(candidate.get("shift_code", "")), version, started_at, None,
                split["closed_trades"], split["wins"], split["losses"], split["win_rate"], split["avg_roi"], split["realized_pnl"],
                split["forced_exits"], split["forced_realized_pnl"], split["strategy_closed_trades"], split["strategy_wins"],
                split["strategy_win_rate"], split["strategy_avg_roi"], split["strategy_realized_pnl"], rel_artifact, iso_now(),
            ),
        )
        _persist_dev_archived_trades(conn, episode_key, candidate_id, slug, version, started_at, records)
        conn.commit()
    return episode_key


def _persist_dev_archived_trades(
    conn: sqlite3.Connection,
    episode_key: str,
    candidate_id: int,
    slug: str,
    strategy_version: str,
    session_started_at: str,
    records: list[dict[str, Any]],
) -> None:
    """Upsert each closed trade of a shift episode into the durable per-trade store.
    Idempotent on (episode_key, trade_id). Field names mirror fetch_shift_closed_trades /
    the episode artifact JSON so backfill and live archiving share one writer."""
    now = iso_now()
    for record in records:
        trade_id = record.get("trade_id")
        if trade_id is None:
            continue
        forced = 1 if str(record.get("exit_reason") or "").lower() in DEV_FORCED_EXIT_REASONS else 0
        profit_abs = parse_float(record.get("close_profit_abs"))
        conn.execute(
            """
            INSERT INTO dev_archived_trades (
                episode_key, candidate_id, slug, strategy_version, session_started_at,
                trade_id, pair, exit_reason, enter_tag, is_short, forced,
                profit_ratio, profit_abs, realized_profit, open_date, close_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_key, trade_id) DO UPDATE SET
                pair=excluded.pair, exit_reason=excluded.exit_reason, enter_tag=excluded.enter_tag,
                is_short=excluded.is_short,
                forced=excluded.forced, profit_ratio=excluded.profit_ratio, profit_abs=excluded.profit_abs,
                realized_profit=excluded.realized_profit, open_date=excluded.open_date,
                close_date=excluded.close_date, strategy_version=excluded.strategy_version
            """,
            (
                episode_key, candidate_id, slug, strategy_version, session_started_at,
                int(trade_id), str(record.get("pair") or ""), str(record.get("exit_reason") or ""),
                str(record.get("enter_tag") or ""),
                1 if record.get("is_short") else 0, forced,
                parse_float(record.get("close_profit")), profit_abs,
                parse_float(record.get("realized_profit")) or profit_abs,
                str(record.get("open_date") or "") or None, str(record.get("close_date") or "") or None, now,
            ),
        )


def backfill_dev_archived_trades() -> None:
    """One-time idempotent backfill: replay existing shift-episode artifacts into the
    per-trade store so history captured before this table existed becomes queryable.
    Skips episodes already represented; best-effort per episode."""
    try:
        with closing(get_db()) as conn:
            episodes = conn.execute(
                "SELECT episode_key, candidate_id, slug, strategy_version, session_started_at, artifact_path "
                "FROM dev_shift_episodes WHERE artifact_path != ''"
            ).fetchall()
            done = {r["episode_key"] for r in conn.execute(
                "SELECT DISTINCT episode_key FROM dev_archived_trades"
            ).fetchall()}
    except Exception:  # noqa: BLE001
        return
    for ep in episodes:
        if ep["episode_key"] in done:
            continue
        artifact = resolve_path(str(ep["artifact_path"] or ""))
        if not artifact or not artifact.exists():
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            records = payload.get("trades") or []
        except Exception:  # noqa: BLE001
            continue
        if not records:
            continue
        try:
            with closing(get_db()) as conn:
                _persist_dev_archived_trades(
                    conn, ep["episode_key"], int(ep["candidate_id"]), str(ep["slug"] or ""),
                    str(ep["strategy_version"] or payload.get("strategy_version") or ""),
                    str(ep["session_started_at"] or payload.get("session_started_at") or ""), records,
                )
                conn.commit()
        except Exception:  # noqa: BLE001
            continue


def wind_down_candidate_shift(candidate: dict[str, Any]) -> None:
    """Shift-bell wind-down (bot still live): force-close all open trades so nothing
    carries into the next version, capture the exit_reason-aware episode, and hand the
    forced/strategy split to the post-shift review. Best-effort; never blocks the stop."""
    candidate_id = int(candidate["id"])
    instance = _candidate_api_instance(candidate)
    started_at = current_open_session_started_at(candidate_id) or str(candidate.get("last_start_at") or iso_now())
    started_utc = normalize_utc(resolve_optional_datetime(started_at))
    try:
        with httpx.Client() as client:
            status = api_get(client, instance, "/api/v1/status")
            open_ids = [t.get("trade_id") for t in status] if isinstance(status, list) else []
            if open_ids:
                try:
                    api_post(client, instance, "/api/v1/forceexit", {"tradeid": "all"})
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("dev-shift", "warning", f"{candidate.get('name','candidate')}: forceexit failed: {exc}")
                # Dry-run fills are immediate; poll briefly to confirm flat.
                for _ in range(5):
                    time.sleep(1.5)
                    try:
                        check = api_get(client, instance, "/api/v1/status")
                    except Exception:  # noqa: BLE001
                        break
                    if not (isinstance(check, list) and check):
                        break
                development_runtime_event(candidate_id, "scheduler",
                                          "Shift bell: force-closed open trades.",
                                          f"{len(open_ids)} position(s) liquidated at shift end.")
    except Exception as exc:  # noqa: BLE001
        log_maintenance("dev-shift", "warning", f"{candidate.get('name','candidate')}: wind-down status/forceexit failed: {exc}")

    records = fetch_shift_closed_trades(candidate, started_utc)
    if records is None:
        records = []
    split = split_shift_trades(records)
    try:
        archive_shift_episode(candidate, split, records, started_at)
    except Exception as exc:  # noqa: BLE001
        log_maintenance("dev-shift", "warning", f"{candidate.get('name','candidate')}: episode archive failed: {exc}")
    # Hand the live-captured split to the post-stop review (its API will be down by then).
    update_development_candidate(
        candidate_id,
        last_shift_split_key=started_at,
        last_shift_forced_exits=split["forced_exits"],
        last_shift_forced_pnl=split["forced_realized_pnl"],
        last_shift_strategy_trades=split["strategy_closed_trades"],
        last_shift_strategy_wins=split["strategy_wins"],
        last_shift_strategy_pnl=split["strategy_realized_pnl"],
        last_shift_strategy_avg_roi=split["strategy_avg_roi"],
    )


def sync_development_pipeline() -> None:
    now_local = local_now()
    if resource_governance_enabled():
        try:
            expire_stale_leases()
        except Exception as exc:  # noqa: BLE001
            log_maintenance("exchange_lease", "error", f"expire_stale_leases failed: {exc}")
    for candidate in development_candidate_rows():
        inspected = inspect_development_candidate(candidate)
        should_run, idle_status, reason = candidate_target_runtime(inspected, now_local)
        previous_runtime_status = str(candidate.get("runtime_status") or "paused")
        status_updates: dict[str, Any] = {
            "heartbeat_ok": inspected["heartbeat_ok"],
            "heartbeat_checked_at": inspected["heartbeat_checked_at"],
            "data_quality": inspected["data_quality"],
            "equity": inspected["equity"],
            "closed_trades": inspected["closed_trades"],
            "open_trades": inspected["open_trades"],
            "realized_pnl": inspected["realized_pnl"],
            "unrealized_pnl": inspected["unrealized_pnl"],
            "worst_open_trade": inspected["worst_open_trade"],
            "max_drawdown": inspected["max_drawdown"],
            "last_trade_at": inspected["last_trade_at"],
            "wins": inspected["wins"],
            "losses": inspected["losses"],
            "win_rate": inspected["win_rate"],
            "avg_roi": inspected["avg_roi"],
            "champion_exits": inspected["champion_exits"],
            "runtime_window": candidate_runtime_window(inspected),
        }
        stop_reason = ""
        if should_run and not inspected["heartbeat_ok"] and runtime_cooldown_passed(inspected.get("last_start_at"), 300):
            # Fresh-shift runtime reset: wipe the DB ONLY after a clean completed shift,
            # never on crash-recovery within a shift (that would erase the in-progress shift).
            # Dev shift tiers only; major-league teams never reach this pipeline.
            if str(inspected.get("tier")) in {"six_hour", "twelve_hour"} and str(inspected.get("last_stop_reason") or "") == "scheduled_shift_end":
                reset_ok, reset_detail = reset_candidate_runtime_db(inspected)
                update_development_candidate(int(candidate["id"]), last_stop_reason="", last_shift_reset_at=iso_now())
                inspected["last_stop_reason"] = ""
                development_runtime_event(int(candidate["id"]), "scheduler",
                                          "New shift: runtime DB reset." if reset_ok else "New shift: runtime DB reset failed.",
                                          reset_detail)
            # Resource governance: lease an exchange for this shift and rewrite the
            # config to RemotePairList before the bot starts. Existing dev bots thus
            # migrate at their next clean shift boundary; no mid-shift disruption.
            try:
                governance_apply_for_shift_start(inspected)
                inspected = inspect_development_candidate(get_development_candidate(int(candidate["id"])) or inspected)
            except Exception as exc:  # noqa: BLE001
                log_maintenance("exchange_lease", "error", f"governance_apply_for_shift_start failed for {inspected.get('slug')}: {exc}")
            ok, detail = execute_development_command(inspected, str(inspected.get("start_command", "")), "start")
            if ok:
                refreshed_after_start = inspect_development_candidate(get_development_candidate(int(candidate["id"])) or candidate)
                status_updates["heartbeat_ok"] = refreshed_after_start["heartbeat_ok"]
                status_updates["heartbeat_checked_at"] = refreshed_after_start["heartbeat_checked_at"]
                status_updates["data_quality"] = refreshed_after_start["data_quality"]
                status_updates["equity"] = refreshed_after_start["equity"]
                status_updates["closed_trades"] = refreshed_after_start["closed_trades"]
                status_updates["open_trades"] = refreshed_after_start["open_trades"]
                status_updates["realized_pnl"] = refreshed_after_start["realized_pnl"]
                status_updates["unrealized_pnl"] = refreshed_after_start["unrealized_pnl"]
                status_updates["worst_open_trade"] = refreshed_after_start["worst_open_trade"]
                status_updates["max_drawdown"] = refreshed_after_start["max_drawdown"]
                status_updates["last_trade_at"] = refreshed_after_start["last_trade_at"]
                status_updates["wins"] = refreshed_after_start["wins"]
                status_updates["losses"] = refreshed_after_start["losses"]
                status_updates["win_rate"] = refreshed_after_start["win_rate"]
                status_updates["avg_roi"] = refreshed_after_start["avg_roi"]
                status_updates["champion_exits"] = refreshed_after_start["champion_exits"]
                container_running = bool(refreshed_after_start.get("container_running"))
                status_updates["runtime_status"] = "running" if refreshed_after_start["heartbeat_ok"] else "paused" if container_running else "failed"
                status_updates["status_detail"] = (
                    refreshed_after_start["status_detail"]
                    if refreshed_after_start["heartbeat_ok"] or container_running
                    else detail
                )
            else:
                status_updates["runtime_status"] = "failed"
                status_updates["status_detail"] = detail
            status_updates["last_start_at"] = iso_now()
            if not ok:
                status_updates["failure_count"] = int(inspected.get("failure_count") or 0) + 1
                status_updates["last_error"] = detail
        elif not should_run and inspected["heartbeat_ok"] and runtime_cooldown_passed(inspected.get("last_stop_at"), 300):
            # Shift bell: force-close + archive the episode WHILE the bot API is live,
            # before we stop the container. Only at a genuine shift end, dev tiers only.
            if idle_status == "off-shift" and str(inspected.get("tier")) in {"six_hour", "twelve_hour"}:
                wind_down_candidate_shift(inspected)
                if resource_governance_enabled():
                    try:
                        release_exchange_lease(inspected)
                    except Exception as exc:  # noqa: BLE001
                        log_maintenance("exchange_lease", "error", f"release_exchange_lease failed for {inspected.get('slug')}: {exc}")
            ok, detail = execute_development_command(inspected, str(inspected.get("stop_command", "")), "stop")
            status_updates["runtime_status"] = idle_status if ok else "failed"
            status_updates["status_detail"] = detail if ok else detail
            status_updates["last_stop_at"] = iso_now()
            stop_reason = "scheduled_shift_end" if idle_status == "off-shift" else "pause"
            if not ok:
                status_updates["failure_count"] = int(inspected.get("failure_count") or 0) + 1
                status_updates["last_error"] = detail
                stop_reason = "crash"
        else:
            if inspected["heartbeat_ok"]:
                status_updates["runtime_status"] = "running"
                status_updates["status_detail"] = inspected["status_detail"]
            elif should_run and bool(inspected.get("container_running")):
                status_updates["runtime_status"] = "paused"
                status_updates["status_detail"] = "Container is up. Awaiting API heartbeat."
            elif should_run and previous_runtime_status == "failed":
                status_updates["runtime_status"] = "failed"
                status_updates["status_detail"] = str(inspected.get("last_error") or inspected.get("status_detail") or reason)
            else:
                status_updates["runtime_status"] = idle_status
                status_updates["status_detail"] = reason
            if previous_runtime_status == "running" and status_updates["runtime_status"] != "running":
                stop_reason = "scheduled_shift_end" if status_updates["runtime_status"] == "off-shift" else "pause"
        if status_updates["runtime_status"] == "running":
            status_updates["uptime_seconds"] = int(inspected.get("uptime_seconds") or 0) + DEV_SCHEDULER_INTERVAL_SECONDS
            status_updates["total_runtime_minutes"] = int(inspected.get("total_runtime_minutes") or 0) + 1
            if inspected["heartbeat_ok"] or int(inspected.get("closed_trades") or 0) > 0:
                status_updates["meaningful_runtime_minutes"] = int(inspected.get("meaningful_runtime_minutes") or 0) + 1
        status_updates["eligibility_status"] = evaluate_candidate_eligibility({**inspected, **status_updates})
        # Remember WHY we stopped so the next start can decide whether to wipe (only a
        # clean scheduled_shift_end earns a fresh DB; pause/crash preserve the shift).
        if stop_reason:
            status_updates["last_stop_reason"] = stop_reason
        sync_development_runtime_session(int(candidate["id"]), previous_runtime_status, str(status_updates["runtime_status"]), stop_reason)
        update_development_candidate(int(candidate["id"]), **status_updates)
        refreshed = get_development_candidate(int(candidate["id"]))
        if refreshed:
            persist_development_snapshot(refreshed)
            if previous_runtime_status == "running" and str(refreshed.get("runtime_status") or "") != "running":
                record_development_post_shift_review(refreshed, review_scope=stop_reason or "state_change")


def development_scheduler_loop() -> None:
    while True:
        try:
            sync_development_pipeline()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(DEV_SCHEDULER_INTERVAL_SECONDS)


def development_league_context() -> dict[str, Any]:
    rows = development_candidate_rows()
    standings = development_standings_context()
    counts_by_tier = {tier: 0 for tier in DEV_TIER_LABELS}
    counts_by_status = {status: 0 for status in DEV_RUNTIME_STATUSES}
    for row in rows:
        counts_by_tier[row["tier"]] = counts_by_tier.get(row["tier"], 0) + 1
        counts_by_status[row["runtime_status"]] = counts_by_status.get(row["runtime_status"], 0) + 1
    six_hour_rows = [row for row in rows if row["tier"] == "six_hour"]
    twelve_hour_rows = [row for row in rows if row["tier"] == "twelve_hour"]
    eligible_rows = [row for row in rows if row["tier"] == "draft_eligible"]
    return {
        "rows": rows,
        "counts_by_tier": counts_by_tier,
        "counts_by_status": counts_by_status,
        "bootcamp_rows": [row for row in rows if row["tier"] == "bootcamp"],
        "draft_rows": [row for row in rows if row["tier"] == "draft_room"],
        "six_hour_rows": six_hour_rows,
        "twelve_hour_rows": twelve_hour_rows,
        "eligible_rows": eligible_rows,
        "current_time_local": iso_local_now(),
        "active_six_shift": active_shift_for_tier("six_hour"),
        "active_twelve_shift": active_shift_for_tier("twelve_hour"),
        "next_six_shift": next_shift_for_tier("six_hour"),
        "next_twelve_shift": next_shift_for_tier("twelve_hour"),
        "recent_events": recent_development_events(),
        "recent_reviews": recent_development_post_shift_reviews(),
        "six_hour_target_remaining": max(0, 24 - len(six_hour_rows)),
        "twelve_hour_target_remaining": max(0, 10 - len(twelve_hour_rows)),
        "eligible_target_remaining": max(0, 2 - len(eligible_rows)),
        "raw_standings": standings["raw_rows"][:8],
        "adjusted_standings": standings["adjusted_rows"][:8],
        "shift_standings": standings["shift_groups"],
        "tier_standings": standings["tier_groups"],
    }


def recent_development_events(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT e.*, c.name
            FROM dev_runtime_events e
            JOIN dev_candidates c ON c.id = e.candidate_id
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def development_board_context(tier: str) -> dict[str, Any]:
    rows = development_candidate_rows(tier)
    return {
        "rows": rows,
        "tier": tier,
        "tier_label": DEV_TIER_LABELS.get(tier, tier),
        "active_shift": active_shift_for_tier(tier),
        "next_shift": next_shift_for_tier(tier),
    }


def development_schedule_context() -> dict[str, Any]:
    rows = development_candidate_rows()
    schedules: list[dict[str, Any]] = []
    now_local = local_now()
    for tier in ("six_hour", "twelve_hour"):
        assignments = defaultdict(list)
        for row in rows:
            if row["tier"] == tier and row.get("shift_code"):
                assignments[row["shift_code"]].append(row)
        shift_rows = []
        active_shift = active_shift_for_tier(tier, now_local)
        for shift in development_shift_definitions(tier):
            shift_rows.append(
                {
                    **shift,
                    "window": shift_window_label(shift),
                    "is_active": bool(active_shift and active_shift["code"] == shift["code"]),
                    "candidates": assignments.get(shift["code"], []),
                }
            )
        schedules.append({"tier": tier, "tier_label": DEV_TIER_LABELS[tier], "shifts": shift_rows})
    return {"schedules": schedules, "current_time_local": iso_local_now()}


def least_loaded_shift_code(tier: str) -> str:
    shifts = development_shift_definitions(tier)
    if not shifts:
        return ""
    rows = development_candidate_rows(tier)
    counts = {shift["code"]: 0 for shift in shifts}
    for row in rows:
        if row.get("shift_code") in counts:
            counts[row["shift_code"]] += 1
    return min(counts.items(), key=lambda item: (item[1], item[0]))[0]


def promote_candidate_state(candidate: dict[str, Any]) -> dict[str, Any]:
    lifecycle = candidate.get("lifecycle_state")
    if lifecycle == "draft_idea":
        return {"lifecycle_state": "bootcamp", "tier": "bootcamp", "shift_code": "", "override_mode": "auto", "runtime_status": "paused"}
    if lifecycle == "assigned_to_shift":
        shift_code = least_loaded_shift_code("twelve_hour")
        return {"lifecycle_state": "twelve_hour_prospect", "tier": "twelve_hour", "shift_code": shift_code, "override_mode": "auto", "runtime_status": "off-shift"}
    if lifecycle == "bootcamp":
        shift_code = least_loaded_shift_code("six_hour")
        return {"lifecycle_state": "six_hour_candidate", "tier": "six_hour", "shift_code": shift_code, "override_mode": "auto", "runtime_status": "off-shift"}
    if lifecycle == "six_hour_candidate":
        shift_code = least_loaded_shift_code("twelve_hour")
        return {"lifecycle_state": "twelve_hour_prospect", "tier": "twelve_hour", "shift_code": shift_code, "override_mode": "auto", "runtime_status": "off-shift"}
    if lifecycle == "twelve_hour_prospect":
        return {"lifecycle_state": "draft_eligible", "tier": "draft_eligible", "shift_code": "", "override_mode": "auto", "runtime_status": "paused"}
    if lifecycle == "draft_eligible":
        return {"lifecycle_state": "drafted", "tier": "drafted", "shift_code": "", "override_mode": "auto", "runtime_status": "paused"}
    return {}


def apply_development_candidate_action(candidate_id: int, action: str, params: dict[str, str] | None = None) -> None:
    params = params or {}
    candidate = get_development_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    previous_runtime_status = str(candidate.get("runtime_status") or "paused")
    if action == "start_now":
        if candidate.get("assembly_status") != "assembled" or candidate.get("validation_status") != "passed":
            raise HTTPException(status_code=400, detail="Candidate must be assembled and validated before it can start.")
        ok, detail = execute_development_command(candidate, str(candidate.get("start_command", "")), "start")
        runtime_status = "failed"
        status_detail = detail
        if ok:
            refreshed_after_start = inspect_development_candidate(get_development_candidate(candidate_id) or candidate)
            if refreshed_after_start["heartbeat_ok"]:
                runtime_status = "running"
                status_detail = refreshed_after_start["status_detail"]
            elif refreshed_after_start.get("container_running"):
                runtime_status = "paused"
                status_detail = "Container is up. Awaiting API heartbeat."
            else:
                runtime_status = "failed"
        sync_development_runtime_session(candidate_id, str(candidate.get("runtime_status") or "paused"), runtime_status, "manual_start")
        update_development_candidate(candidate_id, override_mode="force_running", runtime_status=runtime_status, status_detail=status_detail, last_start_at=iso_now())
        development_runtime_event(candidate_id, "manual", "Candidate started manually.", detail)
    elif action == "stop_now":
        ok, detail = execute_development_command(candidate, str(candidate.get("stop_command", "")), "stop")
        sync_development_runtime_session(candidate_id, str(candidate.get("runtime_status") or "paused"), "paused" if ok else "failed", "manual_stop")
        update_development_candidate(candidate_id, override_mode="force_stopped", runtime_status="paused" if ok else "failed", status_detail=detail, last_stop_at=iso_now())
        development_runtime_event(candidate_id, "manual", "Candidate stopped manually.", detail)
    elif action == "pause_candidate":
        execute_development_command(candidate, str(candidate.get("stop_command", "")), "stop")
        sync_development_runtime_session(candidate_id, str(candidate.get("runtime_status") or "paused"), "paused", "pause")
        update_development_candidate(candidate_id, override_mode="paused", runtime_status="paused", status_detail="Paused manually.")
        development_runtime_event(candidate_id, "manual", "Candidate paused.", "")
    elif action == "resume_auto":
        update_development_candidate(candidate_id, override_mode="auto", status_detail="Returned to automated scheduling.")
        development_runtime_event(candidate_id, "manual", "Candidate returned to auto mode.", "")
    elif action in {"generate_strategy", "regenerate_strategy"}:
        # A manual generate/regenerate from the dev tab is the human approving a
        # strategy change, so it overrides Protected Prospect immutability.
        queue_candidate_strategy_generation(
            candidate_id, force=action == "regenerate_strategy", approved_override=True
        )
    elif action == "protect_candidate":
        update_development_candidate(candidate_id, protected=1, status_detail="Protected Prospect — strategy immutable until you approve a regeneration.")
        development_runtime_event(candidate_id, "protection", "Marked as Protected Prospect.", "Strategy file is now immutable to automated regeneration.")
    elif action == "unprotect_candidate":
        update_development_candidate(candidate_id, protected=0, status_detail="Returned to normal prospect. Automated regeneration re-enabled.")
        development_runtime_event(candidate_id, "protection", "Protection removed.", "Returned to normal prospect.")
    elif action == "run_post_shift_review":
        review = record_development_post_shift_review(candidate, review_scope="manual_review", force=True)
        if not review:
            raise HTTPException(status_code=400, detail="No completed runtime session is available to review yet.")
    elif action == "mark_reviewed":
        if candidate.get("generation_status") != "generated" or candidate.get("validation_status") != "passed":
            raise HTTPException(status_code=400, detail="Candidate must have a validated generated strategy before review.")
        update_development_candidate(
            candidate_id,
            review_status="reviewed",
            reviewed_at=iso_now(),
            lifecycle_state="reviewed",
            status_detail="Reviewed by human. Ready for instance assembly.",
        )
        development_runtime_event(candidate_id, "review", "Strategy marked reviewed.", "")
    elif action == "reject_generated_file":
        update_development_candidate(
            candidate_id,
            review_status="rejected",
            reviewed_at=iso_now(),
            generation_status="failed",
            assembly_status="pending",
            lifecycle_state="draft_idea",
            status_detail="Generated strategy rejected. Regenerate before proceeding.",
        )
        development_runtime_event(candidate_id, "review", "Generated strategy rejected.", "")
    elif action == "assemble_instance":
        assemble_candidate_instance(candidate_id)
    elif action == "assign_shift":
        shift_code = str(params.get("shift_code", "")).upper()
        valid_shift_codes = {shift["code"] for shift in development_shift_definitions("six_hour")}
        if shift_code not in valid_shift_codes:
            raise HTTPException(status_code=400, detail="A valid six-hour shift is required.")
        if candidate.get("assembly_status") != "assembled":
            raise HTTPException(status_code=400, detail="Candidate must be assembled before shift assignment.")
        with closing(get_db()) as conn:
            assigned_count = int(
                conn.execute(
                """
                SELECT COUNT(*)
                FROM dev_candidates
                WHERE tier = 'six_hour' AND shift_code = ? AND id != ? AND lifecycle_state != 'cut_archived'
                """,
                (shift_code, candidate_id),
            ).fetchone()[0]
            )
        if assigned_count >= SIX_HOUR_SHIFT_CAPACITY:
            raise HTTPException(
                status_code=400,
                detail=f"Shift {shift_code} is full ({assigned_count}/{SIX_HOUR_SHIFT_CAPACITY} candidates).",
            )
        updates = {
            "tier": "six_hour",
            "shift_code": shift_code,
            "runtime_window": candidate_runtime_window({"tier": "six_hour", "shift_code": shift_code}),
            "runtime_status": "off-shift",
            "lifecycle_state": "assigned_to_shift",
            "status_detail": "Assigned to six-hour development shift.",
            "override_mode": "auto",
        }
        update_development_candidate(candidate_id, **updates)
        development_runtime_event(candidate_id, "schedule", "Assigned to shift.", shift_code)
    elif action == "archive_candidate":
        execute_development_command(candidate, str(candidate.get("stop_command", "")), "stop")
        sync_development_runtime_session(candidate_id, str(candidate.get("runtime_status") or "paused"), "archived", "archive")
        update_development_candidate(
            candidate_id,
            lifecycle_state="cut_archived",
            tier="archived",
            shift_code="",
            override_mode="paused",
            runtime_status="archived",
            archived_at=iso_now(),
            assembly_status="pending",
            status_detail="Archived manually.",
        )
        development_runtime_event(candidate_id, "manual", "Candidate archived.", "")
    elif action == "promote":
        updates = promote_candidate_state(candidate)
        if updates:
            updates["runtime_window"] = candidate_runtime_window({**candidate, **updates})
            update_development_candidate(candidate_id, **updates)
            development_runtime_event(candidate_id, "manual", "Candidate promoted.", json.dumps(updates))
    elif action == "demote_bootcamp":
        execute_development_command(candidate, str(candidate.get("stop_command", "")), "stop")
        sync_development_runtime_session(candidate_id, str(candidate.get("runtime_status") or "paused"), "paused", "manual_stop")
        update_development_candidate(
            candidate_id,
            lifecycle_state="bootcamp",
            tier="bootcamp",
            shift_code="",
            runtime_window="",
            runtime_status="paused",
            override_mode="auto",
            status_detail="Returned to Bootcamp.",
        )
        development_runtime_event(candidate_id, "manual", "Candidate returned to Bootcamp.", "")
    else:
        raise HTTPException(status_code=400, detail="Unknown development candidate action")
    refreshed = get_development_candidate(candidate_id)
    if refreshed:
        persist_development_snapshot(refreshed)
        if previous_runtime_status == "running" and str(refreshed.get("runtime_status") or "") != "running" and action in {"stop_now", "pause_candidate", "archive_candidate", "demote_bootcamp"}:
            record_development_post_shift_review(refreshed, review_scope=action, force=True)

def get_setting(key: str, default: str = "") -> str:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (key,),
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, value, iso_now()),
        )
        conn.commit()


def replace_generated_content(section: str, content: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ai_generated_content (section, content, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(section) DO UPDATE SET
                content=excluded.content,
                updated_at=excluded.updated_at
            """,
            (section, content, iso_now()),
        )
        conn.commit()


def get_generated_content(section: str, default: str = "") -> str:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT content FROM ai_generated_content WHERE section = ?",
            (section,),
        ).fetchone()
    return row["content"] if row else default


def replace_generated_json(section: str, payload: Any) -> None:
    replace_generated_content(section, json.dumps(payload, indent=2))


def get_generated_json(section: str, default: Any) -> Any:
    raw = get_generated_content(section, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def decode_jsonish_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


def tokenize_search(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


# ===========================================================================
# ML Lab — Strategy Biology Department
# ---------------------------------------------------------------------------
# Deterministic-first research engine. Python measures everything (telemetry,
# divergence, relationships, conviction); the LLM is only used for language.
# ===========================================================================

# Telemetry categories. `good` marks the favorable divergence direction:
#   'high' = larger value is better, 'low' = smaller is better,
#   'neutral' = directionless (used for similarity, not strength/weakness).
# `measurable` flags whether v1 can compute it from trade records alone.
ML_TELEMETRY_CATEGORIES: list[dict[str, Any]] = [
    {"name": "throughput", "good": "neutral", "measurable": True},
    {"name": "win_rate", "good": "high", "measurable": True},
    {"name": "avg_roi", "good": "high", "measurable": True},
    {"name": "avg_hold_time", "good": "neutral", "measurable": True},
    {"name": "hold_time_dispersion", "good": "neutral", "measurable": True},
    {"name": "realized_conversion", "good": "high", "measurable": True},
    {"name": "unrealized_drag", "good": "low", "measurable": True},
    {"name": "long_short_bias", "good": "neutral", "measurable": True},
    {"name": "pair_concentration", "good": "neutral", "measurable": True},
    # champion_exit_rate intentionally NOT a league-wide telemetry category: champion exits
    # are a Cosmo/Wanda-only mechanic, so peer-relative divergence on it unfairly penalizes
    # strategies that structurally cannot produce them. Tracked as family telemetry elsewhere.
    {"name": "exit_efficiency", "good": "high", "measurable": True},
    {"name": "selectivity", "good": "neutral", "measurable": True},
    # Regime-dependent — need OHLC/market context not yet plumbed. Recorded
    # as measurable=0 so the LLM never sees a fabricated number.
    {"name": "volatility_preference", "good": "neutral", "measurable": False},
    {"name": "trend_following_tendency", "good": "neutral", "measurable": False},
    {"name": "reversal_sensitivity", "good": "neutral", "measurable": False},
    {"name": "chop_vulnerability", "good": "low", "measurable": False},
    {"name": "drawdown_pressure", "good": "low", "measurable": False},
    {"name": "exposure_load", "good": "neutral", "measurable": False},
    {"name": "environment_sensitivity", "good": "neutral", "measurable": False},
]
ML_CATEGORY_GOOD = {row["name"]: row["good"] for row in ML_TELEMETRY_CATEGORIES}
ML_MEASURABLE_CATEGORIES = [row["name"] for row in ML_TELEMETRY_CATEGORIES if row["measurable"]]
ML_MIN_SAMPLE = 3  # minimum closed trades before a strategy's value enters peer stats

# Signal Timing Spectrum: a descriptive lens for the "Temporal Niche" dimension —
# where on a move each organism *believes* its edge is and chooses to act. It is a
# declared theory rendered for legibility, NOT a measured fact and NOT a
# compatibility metric (compatibility lives in the descendant engine). The ordering
# is a display axis, not a claim that markets move through fixed phases.
ML_SIGNAL_TIMING_SPECTRUM = [
    ("compression", "Compression"),
    ("early_expansion", "Early Expansion"),
    ("breakout", "Breakout"),
    ("trend_confirmation", "Trend Confirmation"),
    ("trend_maturity", "Trend Maturity"),
    ("exhaustion", "Exhaustion"),
    ("reversal", "Reversal"),
]
ML_SIGNAL_TIMING_INDEX = {slug: i for i, (slug, _) in enumerate(ML_SIGNAL_TIMING_SPECTRUM)}

# Guiding charter for the 15-day Evolution Review. This is the one place that states
# what the review is *for*: speciation, not mutation. Referenced by both the
# descendant generation brief (what gets built) and the review narrative prompt (how
# the committee reasons). Edit the wording here to retune ambition.
ML_EVOLUTION_REVIEW_CHARTER = (
    "ATL EVOLUTION REVIEW CHARTER — read before reasoning.\n"
    "Purpose: speciation, not mutation. The point of this review is NOT to produce a "
    "slightly better version of an existing organism (Timmy but 10% stricter, Cosmo but "
    "more selective). The point is to identify combinations, inheritances, and emergent "
    "architectures that would not exist without the Biology Department.\n"
    "Build big machines, not little machines. Favor additive, multi-stage SYSTEMS that "
    "preserve substantial machinery from each parent as distinct organs (e.g. one parent's "
    "entry/filter, another's confirmation, another's portfolio or exit philosophy) staged "
    "by where each parent hunts on the Signal Timing Spectrum (Wind Tunnel). Do not distill "
    "two parents down into one small idea; the descendant should be MORE intricate than "
    "either parent, not a refinement of one.\n"
    "Be ambitious and combinatorial. Combining more than two organisms is welcome when "
    "there is a real reason — think of the catalogued traits, niches, and genealogy as Lego "
    "bricks to assemble into larger structures. Pursue novel concepts and new architectures, "
    "not minor tweaks.\n"
    "Be imaginative, not reckless. Ambition must be grounded in the taxonomy (traits, Wind "
    "Tunnel niches, genealogy) and in real ecosystem gaps — not random mashups. The best "
    "outcome is a proposal that makes a human say 'I never would have thought to combine "
    "those.'"
)


def upsert_ml_family(slug: str, name: str, description: str = "") -> None:
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ml_families (slug, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at
            """,
            (slug, name, description, now, now),
        )
        conn.commit()


def upsert_ml_strategy(
    slug: str,
    name: str,
    kind: str = "main",
    source_team_id: str = "",
    source_db_path: str = "",
    family_slug: str = "",
    classification: str = "",
    active: int = 1,
) -> None:
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ml_strategy_registry
                (slug, name, kind, source_team_id, source_db_path, family_slug,
                 classification, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                name=excluded.name, kind=excluded.kind,
                source_team_id=excluded.source_team_id,
                source_db_path=excluded.source_db_path,
                family_slug=excluded.family_slug,
                classification=excluded.classification,
                updated_at=excluded.updated_at
            """,
            (slug, name, kind, source_team_id, source_db_path, family_slug,
             classification, int(active), now, now),
        )
        conn.commit()


def ml_lineage_add(child_slug: str, parent_slug: str, relationship_type: str = "parent", mechanism_notes: str = "") -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ml_lineage (child_slug, parent_slug, relationship_type, mechanism_notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (child_slug, parent_slug, relationship_type, mechanism_notes, iso_now()),
        )
        conn.commit()


def ml_lineage_all() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT * FROM ml_lineage ORDER BY child_slug, parent_slug").fetchall()
    return [dict(row) for row in rows]


def ml_lineage_parents(child_slug: str) -> list[str]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT parent_slug FROM ml_lineage WHERE child_slug = ?", (child_slug,)
        ).fetchall()
    return [row["parent_slug"] for row in rows]


def ml_families_all() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT * FROM ml_families ORDER BY name").fetchall()
    return [dict(row) for row in rows]


def ml_registry_all(active_only: bool = True) -> list[dict[str, Any]]:
    query = "SELECT * FROM ml_strategy_registry"
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY family_slug, name"
    with closing(get_db()) as conn:
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def ml_registry_get(slug: str) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM ml_strategy_registry WHERE slug = ?", (slug,)).fetchone()
    return dict(row) if row else None


def upsert_ml_trait(
    strategy_slug: str,
    trait_name: str,
    polarity: str = "neutral",
    confidence: float = 0.3,
    evidence_source: str = "",
    bump_evidence: bool = False,
) -> None:
    now = iso_now()
    confidence = clamp(float(confidence), 0.0, 1.0)
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT id, evidence_count FROM ml_traits WHERE strategy_slug = ? AND trait_name = ?",
            (strategy_slug, trait_name),
        ).fetchone()
        if existing:
            new_count = int(existing["evidence_count"]) + (1 if bump_evidence else 0)
            conn.execute(
                """
                UPDATE ml_traits
                SET polarity = ?, confidence = ?, evidence_source = ?, evidence_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (polarity, confidence, evidence_source, new_count, now, existing["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO ml_traits
                    (strategy_slug, trait_name, polarity, confidence, evidence_source,
                     evidence_count, first_observed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (strategy_slug, trait_name, polarity, confidence, evidence_source, 1, now, now),
            )
        conn.commit()


def ml_traits_for(strategy_slug: str) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM ml_traits WHERE strategy_slug = ? ORDER BY confidence DESC, trait_name",
            (strategy_slug,),
        ).fetchall()
    return [dict(row) for row in rows]


def ml_traits_all() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM ml_traits ORDER BY strategy_slug, confidence DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def seed_temporal_niche_if_empty(slug: str, start: str, end: str = "", note: str = "") -> None:
    """Bootstrap an organism's temporal niche (Signal Timing Spectrum) only when it is
    currently blank, so a manual/LLM declaration is never clobbered. Records source
    'seed' (overwritable later by LLM re-classification). Purely descriptive — no
    compatibility is derived."""
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT temporal_niche_start, temporal_niche_end, temporal_niche_note "
            "FROM ml_strategy_registry WHERE slug = ?",
            (slug,),
        ).fetchone()
        if row is None:
            return
        if (row["temporal_niche_start"] or row["temporal_niche_end"] or row["temporal_niche_note"]):
            return  # already declared — never clobber
        conn.execute(
            "UPDATE ml_strategy_registry SET temporal_niche_start = ?, temporal_niche_end = ?, "
            "temporal_niche_note = ?, temporal_niche_source = 'seed', updated_at = ? WHERE slug = ?",
            (start, end or start, note, iso_now(), slug),
        )
        conn.commit()


def normalize_temporal_niche(raw: Any) -> dict[str, str]:
    """Sanitize an LLM-proposed temporal niche into a canonical, trusted shape:
    {start, end, note, status}. status is 'placed' only when start maps to a real
    Signal Timing Spectrum slug; anything invalid/ambiguous/empty becomes
    'needs_data' (unplaced). The LLM proposes; this guard decides what is allowed
    onto the Wind Tunnel — a hallucinated phase can never reach it."""
    note = ""
    start = end = ""
    status = "needs_data"
    if isinstance(raw, dict):
        note = str(raw.get("note", "") or "").strip()
        declared_status = str(raw.get("status", "") or "").strip().lower()
        start_slug = str(raw.get("start", "") or "").strip().lower()
        end_slug = str(raw.get("end", "") or "").strip().lower()
        if declared_status != "needs_data" and start_slug in ML_SIGNAL_TIMING_INDEX:
            start = start_slug
            end = end_slug if end_slug in ML_SIGNAL_TIMING_INDEX else start_slug
            # Normalize order to start <= end on the spectrum.
            if ML_SIGNAL_TIMING_INDEX[end] < ML_SIGNAL_TIMING_INDEX[start]:
                start, end = end, start
            status = "placed"
    return {"start": start, "end": end, "note": note[:200], "status": status}


def apply_temporal_niche_to_registry(slug: str, niche: dict[str, str], source: str = "llm") -> bool:
    """Write a normalized niche onto a registry row, honoring provenance: a value
    pinned by a human ('human') is locked and never overwritten by auto
    re-classification. 'needs_data' lands as unplaced (blank band) + note. Returns
    True when a write happened."""
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT temporal_niche_source FROM ml_strategy_registry WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            return False
        if (row["temporal_niche_source"] or "") == "human":
            return False  # pinned — never auto-overwrite
        placed = niche.get("status") == "placed"
        conn.execute(
            "UPDATE ml_strategy_registry SET temporal_niche_start = ?, temporal_niche_end = ?, "
            "temporal_niche_note = ?, temporal_niche_source = ?, updated_at = ? WHERE slug = ?",
            (
                niche["start"] if placed else "",
                niche["end"] if placed else "",
                niche.get("note", ""),
                source,
                iso_now(),
                slug,
            ),
        )
        conn.commit()
    return True


def ml_temporal_band(strategy: dict[str, Any]) -> tuple[int, int] | None:
    """Rendering helper only: map a registry row's declared niche to spectrum indices
    [start, end] (normalized so start <= end). Returns None when the organism is not
    placed on the spectrum. This is NOT a compatibility function."""
    start = ML_SIGNAL_TIMING_INDEX.get((strategy.get("temporal_niche_start") or "").strip())
    if start is None:
        return None
    end = ML_SIGNAL_TIMING_INDEX.get((strategy.get("temporal_niche_end") or "").strip(), start)
    return (start, end) if start <= end else (end, start)


def ml_temporal_view(strategy: dict[str, Any]) -> dict[str, Any]:
    """Template-facing shape for an organism's temporal niche. `placed` is True only
    when a band is declared on the spectrum; `note` carries any off-spectrum
    description. Purely descriptive — never a compatibility judgement."""
    band = ml_temporal_band(strategy)
    note = (strategy.get("temporal_niche_note") or "").strip()
    source = (strategy.get("temporal_niche_source") or "").strip()
    if band is None:
        return {"placed": False, "note": note, "source": source}
    start, end = band
    return {
        "placed": True,
        "start": start,
        "end": end,
        "start_label": ML_SIGNAL_TIMING_SPECTRUM[start][1],
        "end_label": ML_SIGNAL_TIMING_SPECTRUM[end][1],
        "note": note,
        "source": source,
    }


def seed_ml_biology() -> None:
    """Idempotently seed the family registry, strategy roster, lineage and the
    starting trait catalog from the league's known organisms."""
    families = [
        ("fairly-odd-family", "Fairly Odd Family",
         "Cosmo/Wanda and Timmy bloodlines, bridged by The Turners' hybrid inheritance."),
        ("roadrunner", "Roadrunner", "Independent momentum-hunter bloodline."),
        ("slacking", "Slacking", "Independent patient mean-reversion bloodline."),
        ("le-phare", "Le Phare", "Independent timeframe-alignment bloodline."),
    ]
    for slug, name, desc in families:
        upsert_ml_family(slug, name, desc)

    # slug, name, kind, source_team_id, source_db_path, family, classification, active
    strategies = [
        ("cosmowanda-top-20", "CosmoWanda Top 20", "main", "cosmo-wanda-20-pi", "",
         "fairly-odd-family", "Reactive expansion engine", 1),
        ("cosmowanda-top-50", "CosmoWanda Top 50", "main", "cosmo-wanda-50-pc", "",
         "fairly-odd-family", "Reactive expansion engine", 1),
        ("timmy-top-20", "Timmy Top 20", "main", "timmy-20-pc", "",
         "fairly-odd-family", "Predictive sniper", 1),
        ("timmy-top-50", "Timmy Top 50", "main", "timmy-50-pc", "",
         "fairly-odd-family", "Predictive sniper", 1),
        # Dev-league organisms. source_team_id holds the dev_candidates slug used
        # to resolve the live sqlite db_path at telemetry time.
        ("the-turners", "The Turners", "dev", "the-turners", "",
         "fairly-odd-family", "Hybrid: reactive expansion + predictive gate", 1),
        ("roadrunner", "Roadrunner", "dev", "roadrunner", "",
         "roadrunner", "Momentum hunter", 1),
        ("slacking", "Slacking", "dev", "slaking", "",
         "slacking", "Patient mean reversion", 1),
        ("le-phare", "Le Phare", "dev", "le-phare", "",
         "le-phare", "Timeframe alignment", 1),
    ]
    for row in strategies:
        upsert_ml_strategy(*row)

    # Genealogy: The Turners is a documented crossing of Cosmo/Wanda and Timmy.
    ml_lineage_add("the-turners", "cosmowanda-top-20", "hybrid",
                   "Inherits reactive expansion engine from Cosmo/Wanda.")
    ml_lineage_add("the-turners", "timmy-top-20", "hybrid",
                   "Inherits predictive confirmation gate from Timmy.")

    # Starting trait catalog (modest confidence; the 3h cycle revises these).
    seed_traits = {
        "timmy-top-20": [
            ("predictive", "strength", 0.6), ("compression-based", "strength", 0.55),
            ("low-gap-detection", "strength", 0.5), ("high-selectivity", "strength", 0.6),
            ("conviction-lock-in", "strength", 0.5), ("low-throughput", "weakness", 0.5),
        ],
        "timmy-top-50": [
            ("predictive", "strength", 0.55), ("high-selectivity", "strength", 0.55),
            ("low-throughput", "weakness", 0.5),
        ],
        "cosmowanda-top-20": [
            ("reactive", "strength", 0.6), ("expansion-based", "strength", 0.55),
            ("high-throughput", "strength", 0.6), ("champion-promotion", "strength", 0.55),
            ("dynamic-roi", "strength", 0.55), ("trend-exploitation", "strength", 0.5),
        ],
        "cosmowanda-top-50": [
            ("reactive", "strength", 0.55), ("high-throughput", "strength", 0.55),
            ("champion-promotion", "strength", 0.5),
        ],
        "the-turners": [
            ("reactive-expansion-engine", "strength", 0.5),
            ("predictive-confirmation-gate", "strength", 0.5),
            ("reduced-throughput", "neutral", 0.45),
            ("higher-selectivity", "strength", 0.45),
            ("hybrid-inheritance", "strength", 0.5),
        ],
        "roadrunner": [
            ("fast-entry-detection", "strength", 0.4),
            ("early-signal-recognition", "strength", 0.4),
            ("reacts-too-quickly", "weakness", 0.35),
        ],
        "slacking": [
            ("patient-trade-selection", "strength", 0.4),
            ("trade-filtering", "strength", 0.4),
            ("can-be-too-slow", "weakness", 0.35),
        ],
        "le-phare": [
            ("timeframe-alignment", "strength", 0.4),
        ],
    }
    for slug, traits in seed_traits.items():
        for trait_name, polarity, confidence in traits:
            # Only seed if absent — never clobber learned confidence on restart.
            if not any(t["trait_name"] == trait_name for t in ml_traits_for(slug)):
                upsert_ml_trait(slug, trait_name, polarity, confidence, "seed")

    # Temporal Niche starting theories (Signal Timing Spectrum). Declared belief
    # about where each organism seeks its edge — descriptive only, editable later,
    # and seeded only-if-empty. Jade and future founders stay unplaced on purpose.
    temporal_niches = {
        "timmy-top-20": ("compression", "early_expansion"),       # edge at the compression->expansion transition
        "timmy-top-50": ("compression", "early_expansion"),
        "cosmowanda-top-20": ("breakout", "trend_confirmation"),  # acts once the move is confirmed
        "cosmowanda-top-50": ("breakout", "trend_confirmation"),
        "the-turners": ("compression", "breakout"),               # Timmy-read filter -> Cosmo execution
        "roadrunner": ("early_expansion", "breakout"),            # chases acceleration
        "slacking": ("exhaustion", "reversal"),                   # contrarian mean reversion
        "le-phare": ("trend_confirmation", "trend_confirmation"), # multi-timeframe alignment
    }
    for slug, (start, end) in temporal_niches.items():
        seed_temporal_niche_if_empty(slug, start, end)

    # The Turners is a high-value prospect: seed it as a Protected Prospect ONCE
    # (so its strategy file is immutable to automated regeneration). After this the
    # user fully controls protection from the Dev League tab, including unprotecting.
    if get_setting("ml_turners_protected_seeded", "") != "true":
        with closing(get_db()) as conn:
            cur = conn.execute("UPDATE dev_candidates SET protected = 1 WHERE slug = ?", ("the-turners",))
            conn.commit()
            if cur.rowcount:
                set_setting("ml_turners_protected_seeded", "true")

    # Enroll any organism that exists operationally (Draft Room / Dev League) but
    # was never registered biologically. Additive only — see reconcile_ml_organisms.
    reconcile_ml_organisms()


def _ml_founder_family_description(hypothesis: Any) -> str:
    """One-line founder-family blurb derived from the candidate's hypothesis."""
    text = (hypothesis or "").strip()
    if not text:
        return "Founder bloodline (auto-enrolled from the Draft Room)."
    first = text.replace("\r", "\n").split("\n", 1)[0].strip()
    if len(first) > 160:
        first = first[:157].rstrip() + "..."
    return f"Founder bloodline. {first}"


def reconcile_ml_organisms() -> None:
    """Deterministically enroll dev-league candidates that exist operationally but
    were never registered as ML organisms (e.g. Draft Room ideas created outside the
    descendant flow, like Jade). Founder by default: a candidate with no recorded
    lineage becomes the founder of its own family.

    Additive only. It never deletes, never overwrites an existing registry row's
    family_slug/classification, and never rewrites an existing family — reconcile
    fills missing organisms, it does not revise biology. Hybrids and descendants
    inherit their parents' family via the descendant flow, not here. Runs inside
    seed_ml_biology() at startup and at the top of every telemetry cycle."""
    registry = ml_registry_all(active_only=False)
    # Registry rows store the originating dev_candidate slug in source_team_id; that
    # is the enrollment key (a registry slug can differ from the candidate slug,
    # e.g. registry "slacking" vs candidate "slaking").
    enrolled_sources = {(r.get("source_team_id") or "").strip() for r in registry}
    enrolled_sources.discard("")
    existing_org_slugs = {(r.get("slug") or "").strip() for r in registry}
    existing_family_slugs = {(f.get("slug") or "").strip() for f in ml_families_all()}

    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT slug, name, hypothesis, temporal_niche_start, temporal_niche_end, "
            "temporal_niche_note, temporal_niche_status FROM dev_candidates WHERE archived_at IS NULL"
        ).fetchall()
    candidates = [dict(r) for r in rows]

    for candidate in candidates:
        slug = (candidate.get("slug") or "").strip()
        # Fill missing organisms only. Skip if already enrolled (matched on
        # source_team_id) or if the slug already names a registry row, so the
        # upsert below can only ever INSERT — never revise existing biology.
        if not slug or slug in enrolled_sources or slug in existing_org_slugs:
            continue
        name = (candidate.get("name") or slug).strip() or slug
        # Founder family: the organism seeds its own bloodline. Only create the
        # family when absent; never rewrite one that already exists.
        if slug not in existing_family_slugs:
            upsert_ml_family(slug, name, _ml_founder_family_description(candidate.get("hypothesis")))
            existing_family_slugs.add(slug)
        upsert_ml_strategy(
            slug, name, kind="dev", source_team_id=slug, source_db_path="",
            family_slug=slug, classification="Founder organism", active=1,
        )
        log_maintenance("ml", "success", f"Reconcile enrolled new organism '{slug}' as founder of family '{slug}'.")

    # Sync the LLM-classified temporal niche from each candidate onto its registry
    # row. The strategy generator/regenerator is the niche's author, so this keeps
    # the Wind Tunnel current on every (re)generation. Provenance-aware: a
    # human-pinned niche is never overwritten (handled in apply_temporal_niche_to_registry).
    source_to_slug = {
        (r.get("source_team_id") or "").strip(): r["slug"]
        for r in ml_registry_all(active_only=False)
        if (r.get("source_team_id") or "").strip()
    }
    for candidate in candidates:
        cand_slug = (candidate.get("slug") or "").strip()
        status = (candidate.get("temporal_niche_status") or "").strip()
        if not cand_slug or not status:
            continue  # candidate hasn't been classified by the generator yet
        registry_slug = source_to_slug.get(cand_slug)
        if not registry_slug:
            continue
        niche = normalize_temporal_niche({
            "start": candidate.get("temporal_niche_start"),
            "end": candidate.get("temporal_niche_end"),
            "note": candidate.get("temporal_niche_note"),
            "status": status,
        })
        apply_temporal_niche_to_registry(registry_slug, niche, source="llm")


def seed_default_settings() -> None:
    season_anchor = local_now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    defaults = {
        "ollama_api_key": "",
        "ollama_model": "gpt-oss:120b",
        "ollama_fallback_model": "",
        "development_strategy_generation_model": "kimi-k2.6:cloud",
        "development_strategy_generation_fallback_model": "",
        "development_strategy_generation_timeout_seconds": "480",
        "development_strategy_generation_retry_count": "4",
        "ollama_timeout_seconds": "180",
        "ollama_retry_count": "2",
        "league_maintenance_minutes": "30",
        "ml_maintenance_minutes": "180",
        "archive_maintenance_minutes": "720",
        "league_maintenance_enabled": "true",
        "ml_maintenance_enabled": "true",
        "archive_maintenance_enabled": "true",
        "archive_push_enabled": "true",
        "archive_repo_url": "https://github.com/escapeware/atl.git",
        "archive_repo_branch": "main",
        "archive_repo_local_path": relative_project_path(DEFAULT_ARCHIVE_REPO_DIR),
        "archive_last_run": "",
        "archive_last_snapshot_path": "",
        "auto_review_regeneration_enabled": "true",
        "auto_review_regeneration_decisions": "tweak,overhaul",
        "league_season_anchor": season_anchor,
        "research_agent_enabled": "true",
        "research_agent_interval_minutes": "30",
        "research_agent_duration_hours": "12",
        "league_maintenance_last_run": "",
        "ml_maintenance_last_run": "",
        "ml_evolution_review_minutes": "21600",
        "ml_evolution_review_enabled": "true",
        "ml_evolution_review_last_run": "",
        "ml_descendant_conviction_threshold": "0.75",
        "ml_descendant_min_evidence_cycles": "8",
        "ml_descendant_novelty_threshold": "0.35",
        "ml_descendant_max_per_review": "3",
        "ml_complement_pair_threshold": "0.55",
        "chronicle_enabled": "true",
        "chronicle_run_time": "23:11",
        "chronicle_last_run": "",
        "chronicle_last_date": "",
    }
    for key, value in defaults.items():
        if get_setting(key, "") == "":
            set_setting(key, value)


def app_settings_snapshot() -> dict[str, str]:
    season_anchor = local_now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    return {
        "ollama_api_key": get_setting("ollama_api_key", ""),
        "ollama_model": get_setting("ollama_model", "gpt-oss:120b"),
        "ollama_fallback_model": get_setting("ollama_fallback_model", ""),
        "development_strategy_generation_model": get_setting("development_strategy_generation_model", "kimi-k2.6:cloud"),
        "development_strategy_generation_fallback_model": get_setting("development_strategy_generation_fallback_model", ""),
        "development_strategy_generation_timeout_seconds": get_setting("development_strategy_generation_timeout_seconds", "480"),
        "development_strategy_generation_retry_count": get_setting("development_strategy_generation_retry_count", "4"),
        "ollama_timeout_seconds": get_setting("ollama_timeout_seconds", "180"),
        "ollama_retry_count": get_setting("ollama_retry_count", "2"),
        "league_maintenance_minutes": get_setting("league_maintenance_minutes", "30"),
        "ml_maintenance_minutes": get_setting("ml_maintenance_minutes", "180"),
        "archive_maintenance_minutes": get_setting("archive_maintenance_minutes", "720"),
        "league_maintenance_enabled": get_setting("league_maintenance_enabled", "true"),
        "ml_maintenance_enabled": get_setting("ml_maintenance_enabled", "true"),
        "archive_maintenance_enabled": get_setting("archive_maintenance_enabled", "true"),
        "archive_push_enabled": get_setting("archive_push_enabled", "true"),
        "archive_repo_url": get_setting("archive_repo_url", "https://github.com/escapeware/atl.git"),
        "archive_repo_branch": get_setting("archive_repo_branch", "main"),
        "archive_repo_local_path": get_setting("archive_repo_local_path", relative_project_path(DEFAULT_ARCHIVE_REPO_DIR)),
        "archive_last_run": get_setting("archive_last_run", ""),
        "archive_last_snapshot_path": get_setting("archive_last_snapshot_path", ""),
        "auto_review_regeneration_enabled": get_setting("auto_review_regeneration_enabled", "true"),
        "auto_review_regeneration_decisions": get_setting("auto_review_regeneration_decisions", "tweak,overhaul"),
        "league_season_anchor": get_setting("league_season_anchor", season_anchor),
        "research_agent_enabled": get_setting("research_agent_enabled", "true"),
        "research_agent_interval_minutes": get_setting("research_agent_interval_minutes", "30"),
        "research_agent_duration_hours": get_setting("research_agent_duration_hours", "12"),
        "league_maintenance_last_run": get_setting("league_maintenance_last_run", ""),
        "ml_maintenance_last_run": get_setting("ml_maintenance_last_run", ""),
        "ml_evolution_review_minutes": get_setting("ml_evolution_review_minutes", "21600"),
        "ml_evolution_review_enabled": get_setting("ml_evolution_review_enabled", "true"),
        "ml_evolution_review_last_run": get_setting("ml_evolution_review_last_run", ""),
        "ml_descendant_conviction_threshold": get_setting("ml_descendant_conviction_threshold", "0.75"),
        "ml_descendant_min_evidence_cycles": get_setting("ml_descendant_min_evidence_cycles", "8"),
        "ml_descendant_novelty_threshold": get_setting("ml_descendant_novelty_threshold", "0.35"),
        "ml_descendant_max_per_review": get_setting("ml_descendant_max_per_review", "3"),
        "ml_complement_pair_threshold": get_setting("ml_complement_pair_threshold", "0.55"),
        "chronicle_enabled": get_setting("chronicle_enabled", "true"),
        "chronicle_run_time": get_setting("chronicle_run_time", "23:11"),
        "chronicle_last_run": get_setting("chronicle_last_run", ""),
        "chronicle_last_date": get_setting("chronicle_last_date", ""),
    }


def ml_hypotheses() -> list[dict[str, Any]]:
    return load_json(ML_HYPOTHESES_PATH, [])


def ml_buckets() -> list[dict[str, Any]]:
    return load_json(ML_BUCKETS_PATH, [])


def ml_features() -> list[dict[str, Any]]:
    return load_json(ML_FEATURES_PATH, [])


def ml_models() -> list[dict[str, Any]]:
    return load_json(ML_MODELS_PATH, [])


def ml_draft_board() -> list[dict[str, Any]]:
    return load_json(ML_DRAFT_BOARD_PATH, [])


def ml_promotions() -> list[dict[str, Any]]:
    return load_json(ML_PROMOTIONS_PATH, [])


def registry_slug(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    slug = "".join(cleaned).strip("-")
    if slug:
        return slug
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def safe_strategy_class_name(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    class_name = "".join(part[:1].upper() + part[1:] for part in parts)
    if not class_name:
        class_name = "DevelopmentStrategy"
    if class_name[0].isdigit():
        class_name = f"Strategy{class_name}"
    return class_name


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def development_strategy_path(candidate: dict[str, Any]) -> Path:
    class_name = safe_strategy_class_name(str(candidate.get("name", "")))
    return DEV_STRATEGY_DIR / f"{class_name}.py"


def development_config_path(candidate: dict[str, Any]) -> Path:
    return DEV_CONFIG_DIR / f"{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}.json"


def development_log_path(candidate: dict[str, Any]) -> Path:
    return DEV_LOG_DIR / f"{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}.log"


def development_db_path(candidate: dict[str, Any]) -> Path:
    return DEV_DATABASE_DIR / f"{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}.sqlite"


def development_container_name(candidate: dict[str, Any]) -> str:
    return f"atl-dev-{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}"


def development_db_volume_name(candidate: dict[str, Any]) -> str:
    return f"{development_container_name(candidate)}-db"


def development_script_paths(candidate: dict[str, Any]) -> tuple[Path, Path]:
    slug = registry_slug(str(candidate.get("slug") or candidate.get("name") or "candidate"))
    return DEV_SCRIPT_DIR / f"start-{slug}.ps1", DEV_SCRIPT_DIR / f"stop-{slug}.ps1"


def pick_strategy_generation_model() -> str:
    preferred = get_setting("development_strategy_generation_model", "kimi-k2.6:cloud").strip()
    return preferred or "kimi-k2.6:cloud"


def pick_strategy_generation_fallback_model() -> str:
    return get_setting("development_strategy_generation_fallback_model", "").strip()


def read_exchange_name_from_config(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("exchange", {}).get("name") or "").strip().lower()


def development_exchange_rotation_snapshot() -> dict[str, Any]:
    profiles = {str(profile["name"]): profile for profile in DEV_EXCHANGE_PROFILES}
    counts = {name: 0 for name in profiles}
    for instance in list_instances():
        exchange_name = read_exchange_name_from_config(resolve_path(instance.get("config_path")))
        if exchange_name in counts:
            counts[exchange_name] += 1
    with closing(get_db()) as conn:
        rows = conn.execute("SELECT config_path, lifecycle_state FROM dev_candidates").fetchall()
    for row in rows:
        lifecycle_state = str(row["lifecycle_state"] or "")
        if lifecycle_state in {"cut_archived", "archived"}:
            continue
        exchange_name = read_exchange_name_from_config(resolve_path(row["config_path"]))
        if exchange_name in counts:
            counts[exchange_name] += 1
    ordered = sorted(
        counts.items(),
        key=lambda item: (item[1] + int(profiles[item[0]].get("rotation_bias") or 0), item[0]),
    )
    return {
        "usage_counts": counts,
        "rotation_order": [name for name, _count in ordered],
    }


def pick_candidate_exchange_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    profiles = {str(profile["name"]): profile for profile in DEV_EXCHANGE_PROFILES}
    existing_exchange = read_exchange_name_from_config(resolve_path(candidate.get("config_path")))
    if existing_exchange in profiles:
        return dict(profiles[existing_exchange])
    snapshot = development_exchange_rotation_snapshot()
    for exchange_name in snapshot.get("rotation_order", []):
        if exchange_name in profiles:
            return dict(profiles[exchange_name])
    return dict(DEV_EXCHANGE_PROFILES[0])


def candidate_timeframe_hours(candidate: dict[str, Any]) -> float:
    timeframe = str(candidate.get("suggested_timeframe") or candidate.get("timeframe") or "").strip().lower()
    if timeframe.endswith("m"):
        minutes = parse_intish(timeframe[:-1])
        return (minutes or 0) / 60.0
    if timeframe.endswith("h"):
        return float(parse_intish(timeframe[:-1]) or 0)
    if timeframe.endswith("d"):
        return float(parse_intish(timeframe[:-1]) or 1) * 24.0
    return 0.0


def candidate_auto_update_cooldown_hours(candidate: dict[str, Any]) -> float:
    timeframe_hours = candidate_timeframe_hours(candidate)
    if timeframe_hours >= 24.0:
        return 72.0
    if timeframe_hours >= 4.0:
        return 24.0
    if timeframe_hours >= 1.0:
        return 12.0
    return 6.0


def development_generation_peer_context(candidate_id: int, limit: int = 12) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, name, tier, timeframe, suggested_timeframe, coin_universe, expected_behavior,
                   generation_status, review_status, config_path
            FROM dev_candidates
            WHERE id != ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (candidate_id, limit),
        ).fetchall()
    context_rows: list[dict[str, Any]] = []
    for row in rows:
        context_rows.append(
            {
                "name": row["name"],
                "tier": row["tier"],
                "timeframe": row["suggested_timeframe"] or row["timeframe"],
                "coin_universe": row["coin_universe"],
                "expected_behavior": row["expected_behavior"],
                "generation_status": row["generation_status"],
                "review_status": row["review_status"],
                "exchange": read_exchange_name_from_config(resolve_path(row["config_path"])),
            }
        )
    return context_rows


def parse_candidate_universe_size(value: str) -> int | None:
    match = re.search(r"top\s*(\d+)", value.lower())
    if match:
        return int(match.group(1))
    return extract_first_integer(value)


def extract_first_integer(value: str) -> int | None:
    digits = ""
    for char in value:
        if char.isdigit():
            digits += char
        elif digits:
            return int(digits)
    if digits:
        return int(digits)
    return None


def infer_dataset_source_path(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "hyperliquid" in lowered:
        return "user_data/data/hyperliquid"
    if "bitget" in lowered:
        return "user_data/data/bitget"
    if "bybit" in lowered:
        return "user_data/data/bybit"
    if "okx" in lowered:
        return "user_data/data/okx"
    if "binance" in lowered:
        return "user_data/data/binance"
    return "user_data/data"


def infer_model_hypothesis_id(
    model: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    buckets: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
) -> str:
    model_features = set(model.get("feature_set", []))
    for hypothesis in hypotheses:
        hypothesis_features = set(hypothesis.get("features_used", []))
        if model_features and hypothesis_features and model_features.issubset(hypothesis_features):
            return hypothesis.get("id", "")

    model_name = model.get("model_name", "").lower().replace("_", " ").replace("-", " ")
    for promotion in promotions:
        strategy_name = promotion.get("strategy_name", "").lower()
        if not strategy_name or strategy_name not in model_name:
            continue
        feature_bucket = promotion.get("feature_bucket", "").lower()
        for bucket in buckets:
            bucket_name = bucket.get("bucket_name", "").lower()
            if bucket_name and bucket_name in feature_bucket:
                return bucket.get("hypothesis_id", "")

    for hypothesis in hypotheses:
        for field_name in ("id", "name", "nickname", "theme"):
            candidate = str(hypothesis.get(field_name, "")).lower().replace("_", " ").replace("-", " ")
            if not candidate:
                continue
            terms = [term for term in candidate.split() if len(term) > 3]
            if terms and all(term in model_name for term in terms[:2]):
                return hypothesis.get("id", "")
    return ""


def list_ml_dataset_registry() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_dataset_registry
            ORDER BY source_kind, dataset_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_ml_label_registry() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_label_registry
            ORDER BY label_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_ml_model_registry() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_model_registry
            ORDER BY training_date DESC, model_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def parse_json_value(raw: Any, default: Any) -> Any:
    if raw in {None, ""}:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return default


def list_ml_feature_set_versions() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_feature_set_versions
            ORDER BY feature_set_name
            """
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["feature_names"] = parse_json_value(item.get("feature_names"), [])
        items.append(item)
    return items


def list_ml_label_spec_versions() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_label_spec_versions
            ORDER BY label_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_ml_experiment_queue(status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        if status:
            rows = conn.execute(
                """
                SELECT *
                FROM ml_experiment_queue
                WHERE status = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM ml_experiment_queue
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def create_ml_queue_item(
    source_type: str,
    source_key: str,
    title: str,
    lead_question: str,
    rationale: str,
    thread_id: int | None = None,
    priority: str = "normal",
    status: str = "queued",
    assigned_agent: str = "ml-agent",
) -> int:
    now = iso_now()
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT id, status FROM ml_experiment_queue WHERE source_key = ? AND status IN ('queued', 'running')",
            (source_key,),
        ).fetchone()
        if existing:
            return int(existing["id"])
        conn.execute(
            """
            INSERT INTO ml_experiment_queue (
                source_type, source_key, thread_id, title, lead_question, rationale,
                priority, status, assigned_agent, resolution, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                source_key,
                thread_id,
                title,
                lead_question,
                rationale,
                priority,
                status,
                assigned_agent,
                "",
                now,
                now,
            ),
        )
        queue_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    upsert_research_index_entry(
        "ml-queue",
        f"ml-queue:{queue_id}",
        title,
        f"Question: {lead_question}\n\nRationale: {rationale}",
        " ".join(tokenize_search(title + " " + lead_question + " " + rationale)),
        entry_type="question",
        author_type="agent",
        thread_id=thread_id,
        status="active",
    )
    return queue_id


def update_ml_queue_item(queue_id: int, **updates: Any) -> None:
    if not updates:
        return
    updates["updated_at"] = iso_now()
    fields = []
    values = []
    for key, value in updates.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(queue_id)
    with closing(get_db()) as conn:
        conn.execute(f"UPDATE ml_experiment_queue SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def list_ml_experiment_runs(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ml_experiment_runs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_ml_experiment_run(payload: dict[str, Any]) -> int:
    now = iso_now()
    run_slug = payload.get("run_slug") or registry_slug(payload.get("title", "experiment-run"))
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT id FROM ml_experiment_runs WHERE run_slug = ?",
            (run_slug,),
        ).fetchone()
        if existing:
            run_id = int(existing["id"])
            conn.execute(
                """
                UPDATE ml_experiment_runs
                SET queue_id = ?, title = ?, status = ?, objective = ?, dataset_id = ?,
                    feature_set_version_id = ?, label_spec_version_id = ?, hypothesis_id = ?,
                    started_at = ?, completed_at = ?, summary = ?, artifact_path = ?, notes = ?
                WHERE id = ?
                """,
                (
                    payload.get("queue_id"),
                    payload.get("title", run_slug),
                    payload.get("status", "planned"),
                    payload.get("objective", ""),
                    payload.get("dataset_id"),
                    payload.get("feature_set_version_id"),
                    payload.get("label_spec_version_id"),
                    payload.get("hypothesis_id"),
                    payload.get("started_at"),
                    payload.get("completed_at"),
                    payload.get("summary", ""),
                    payload.get("artifact_path", ""),
                    payload.get("notes", ""),
                    run_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO ml_experiment_runs (
                    queue_id, run_slug, title, status, objective, dataset_id, feature_set_version_id,
                    label_spec_version_id, hypothesis_id, created_at, started_at, completed_at,
                    summary, artifact_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("queue_id"),
                    run_slug,
                    payload.get("title", run_slug),
                    payload.get("status", "planned"),
                    payload.get("objective", ""),
                    payload.get("dataset_id"),
                    payload.get("feature_set_version_id"),
                    payload.get("label_spec_version_id"),
                    payload.get("hypothesis_id"),
                    now,
                    payload.get("started_at") or now,
                    payload.get("completed_at"),
                    payload.get("summary", ""),
                    payload.get("artifact_path", ""),
                    payload.get("notes", ""),
                ),
            )
            run_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return run_id


def list_ml_bucket_candidates(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT bc.*, mr.run_slug, mr.title AS run_title
            FROM ml_bucket_candidates bc
            LEFT JOIN ml_experiment_runs mr ON mr.id = bc.experiment_run_id
            ORDER BY bc.updated_at DESC, bc.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_ml_bucket_candidate(payload: dict[str, Any]) -> int:
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ml_bucket_candidates (
                experiment_run_id, candidate_name, hypothesis_id, feature_conditions, expected_behavior,
                evidence_quality, contamination_risk, status, next_action, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("experiment_run_id"),
                payload.get("candidate_name", "Unnamed bucket candidate"),
                payload.get("hypothesis_id"),
                payload.get("feature_conditions", ""),
                payload.get("expected_behavior", ""),
                payload.get("evidence_quality", "Early"),
                payload.get("contamination_risk", "Review"),
                payload.get("status", "candidate"),
                payload.get("next_action", "Review in workbench."),
                now,
                now,
            ),
        )
        candidate_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return candidate_id


def list_ml_validation_reports(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT vr.*, mr.run_slug, mr.title AS run_title
            FROM ml_validation_reports vr
            LEFT JOIN ml_experiment_runs mr ON mr.id = vr.experiment_run_id
            ORDER BY vr.created_at DESC, vr.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["metrics_json"] = parse_json_value(item.get("metrics_json"), {})
        items.append(item)
    return items


def add_ml_validation_report(payload: dict[str, Any]) -> int:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ml_validation_reports (
                experiment_run_id, report_type, summary, metrics_json, contamination_checks, recommendation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("experiment_run_id"),
                payload.get("report_type", "validation"),
                payload.get("summary", ""),
                json.dumps(payload.get("metrics_json", {})),
                payload.get("contamination_checks", ""),
                payload.get("recommendation", ""),
                iso_now(),
            ),
        )
        report_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return report_id


def list_ml_promotion_recommendations(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT pr.*, mr.run_slug, mr.title AS run_title
            FROM ml_promotion_recommendations pr
            LEFT JOIN ml_experiment_runs mr ON mr.id = pr.experiment_run_id
            ORDER BY pr.created_at DESC, pr.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_ml_promotion_recommendation(payload: dict[str, Any]) -> int:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO ml_promotion_recommendations (
                experiment_run_id, candidate_name, recommendation, rationale, blockers, target_surface, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("experiment_run_id"),
                payload.get("candidate_name", "Unnamed candidate"),
                payload.get("recommendation", "hold"),
                payload.get("rationale", ""),
                payload.get("blockers", ""),
                payload.get("target_surface", "workbench"),
                iso_now(),
            ),
        )
        recommendation_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    return recommendation_id


def get_ml_experiment_run_id_by_slug(run_slug: str) -> int | None:
    if not run_slug:
        return None
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT id FROM ml_experiment_runs WHERE run_slug = ?",
            (run_slug,),
        ).fetchone()
    return int(row["id"]) if row else None


def get_ml_experiment_run(run_slug: str) -> dict[str, Any] | None:
    if not run_slug:
        return None
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM ml_experiment_runs WHERE run_slug = ?",
            (run_slug,),
        ).fetchone()
    return dict(row) if row else None


def load_ml_run_artifact(artifact_path: str) -> dict[str, Any]:
    resolved = resolve_path(artifact_path)
    if not resolved or not resolved.exists():
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_ml_queue_item(queue_id: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM ml_experiment_queue WHERE id = ?",
            (queue_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_ml_run_outputs(experiment_run_id: int) -> None:
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM ml_bucket_candidates WHERE experiment_run_id = ?", (experiment_run_id,))
        conn.execute("DELETE FROM ml_validation_reports WHERE experiment_run_id = ?", (experiment_run_id,))
        conn.execute("DELETE FROM ml_promotion_recommendations WHERE experiment_run_id = ?", (experiment_run_id,))
        conn.commit()


def infer_queue_hypothesis_id(queue_item: dict[str, Any]) -> str:
    combined = " ".join(
        str(queue_item.get(key, "")) for key in ("title", "lead_question", "rationale", "source_key")
    ).lower()
    best_hypothesis_id = ""
    best_score = 0
    for row in merged_ml_hypotheses():
        score = 0
        for field_name in ("id", "name", "nickname", "description", "target", "theme"):
            field_value = str(row.get(field_name, "")).strip().lower()
            if not field_value:
                continue
            terms = [token for token in re.findall(r"[a-z0-9]+", field_value) if len(token) > 3]
            if field_value and field_value in combined:
                score += 4
            elif terms and any(term in combined for term in terms[:3]):
                score += len(terms[:3])
        if score > best_score:
            best_score = score
            best_hypothesis_id = str(row.get("id", ""))
    return best_hypothesis_id


def choose_workbench_registry_row(rows: list[dict[str, Any]], hypothesis_id: str, fallback_key: str) -> dict[str, Any]:
    if hypothesis_id:
        for row in rows:
            if str(row.get("hypothesis_id", "")) == hypothesis_id:
                return row
    for row in rows:
        if str(row.get("status", "")) in {"available", "active", "candidate", "complete"}:
            return row
    return rows[0] if rows else {fallback_key: ""}


def duration_bucket(minutes: float) -> str:
    if minutes <= 0:
        return "unknown"
    if minutes < 60:
        return "scalp"
    if minutes < 360:
        return "intraday"
    if minutes < 1440:
        return "swing"
    return "position"


def leverage_bucket(leverage: float) -> str:
    if leverage <= 1.1:
        return "spot"
    if leverage <= 3:
        return "low"
    if leverage <= 7:
        return "medium"
    return "high"


def range_bucket(open_rate: float, max_rate: float, min_rate: float) -> str:
    if open_rate <= 0:
        return "unknown"
    realized_range = abs(max_rate - min_rate) / open_rate
    if realized_range < 0.01:
        return "quiet"
    if realized_range < 0.03:
        return "active"
    return "expansion"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def percentage(part: float, whole: float) -> float:
    if not whole:
        return 0.0
    return part / whole


# --- ML biology statistics helpers (pure Python, no numpy) -----------------
def stat_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def stat_median(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return 0.0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def stat_pstdev(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    mu = sum(clean) / len(clean)
    variance = sum((v - mu) ** 2 for v in clean) / len(clean)
    return math.sqrt(variance)


def stat_mad(values: list[float]) -> float:
    """Median absolute deviation (robust spread)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0
    med = stat_median(clean)
    return stat_median([abs(v - med) for v in clean])


def robust_z(value: float, values: list[float]) -> float:
    """Robust z-score using median/MAD, falling back to mean/stdev."""
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    med = stat_median(clean)
    mad = stat_mad(clean)
    if mad > 1e-9:
        # 1.4826 scales MAD to be consistent with stdev for normal data.
        return (value - med) / (1.4826 * mad)
    spread = stat_pstdev(clean)
    if spread > 1e-9:
        return (value - sum(clean) / len(clean)) / spread
    return 0.0


def percentile_rank(value: float, values: list[float]) -> float:
    """Fraction of peers at or below value, in [0, 1]."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0
    below = sum(1 for v in clean if v <= value)
    return below / len(clean)


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    keys = set(vec_a) & set(vec_b)
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(vec_a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(vec_b[k] ** 2 for k in keys))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def hhi(weights: list[float]) -> float:
    """Herfindahl-Hirschman concentration index in [0, 1]."""
    clean = [abs(float(w)) for w in weights if w is not None]
    total = sum(clean)
    if total <= 0:
        return 0.0
    return sum((w / total) ** 2 for w in clean)


def session_bucket(timestamp_ms: int) -> str:
    if timestamp_ms <= 0:
        return "unknown"
    hour = datetime.fromtimestamp(timestamp_ms / 1000.0, UTC).hour
    if hour < 6:
        return "asia"
    if hour < 12:
        return "europe-open"
    if hour < 18:
        return "us-open"
    return "late-session"


def weekday_bucket(timestamp_ms: int) -> str:
    if timestamp_ms <= 0:
        return "unknown"
    return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC).strftime("%a").lower()


def month_bucket(timestamp_ms: int) -> str:
    if timestamp_ms <= 0:
        return "unknown"
    return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC).strftime("%Y-%m")


def backtest_archive_strategy_runs(limit_archives: int = 12) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    zip_paths = sorted(BACKTEST_DIR.glob("backtest-result-*.zip"))
    if limit_archives > 0:
        zip_paths = zip_paths[-limit_archives:]
    for zip_path in zip_paths:
        try:
            with ZipFile(zip_path) as archive:
                json_member = next(
                    (
                        member
                        for member in archive.namelist()
                        if member.endswith(".json") and "_config" not in member and "meta" not in member
                    ),
                    "",
                )
                if not json_member:
                    continue
                with archive.open(json_member) as handle:
                    payload = json.load(handle)
        except Exception:  # noqa: BLE001
            continue
        for strategy_name, strategy_block in (payload.get("strategy") or {}).items():
            trades = strategy_block.get("trades") or []
            if not trades:
                continue
            groups.append(
                {
                    "archive_id": zip_path.stem,
                    "strategy_name": strategy_name,
                    "trade_count": len(trades),
                    "backtest_start_ts": int(strategy_block.get("backtest_start_ts") or 0),
                    "backtest_end_ts": int(strategy_block.get("backtest_end_ts") or 0),
                    "trades": trades,
                }
            )
    return groups


def backtest_training_records(limit_archives: int = 12) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = backtest_archive_strategy_runs(limit_archives=limit_archives)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group in groups:
        group_id = f"{group['archive_id']}:{group['strategy_name']}"
        summaries.append(
            {
                "group_id": group_id,
                "archive_id": group["archive_id"],
                "strategy_name": group["strategy_name"],
                "trade_count": group["trade_count"],
            }
        )
        for trade in group["trades"]:
            open_timestamp = int(trade.get("open_timestamp") or 0)
            profit_ratio = parse_float(trade.get("profit_ratio"))
            records.append(
                {
                    "group_id": group_id,
                    "archive_id": group["archive_id"],
                    "strategy_name": group["strategy_name"],
                    "pair": trade.get("pair") or "unknown",
                    "side": "short" if trade.get("is_short") else "long",
                    "enter_tag": trade.get("enter_tag") or "unknown",
                    "leverage_bucket": leverage_bucket(parse_float(trade.get("leverage"))),
                    "session_bucket": session_bucket(open_timestamp),
                    "weekday_bucket": weekday_bucket(open_timestamp),
                    "month_bucket": month_bucket(open_timestamp),
                    "profit_ratio": profit_ratio,
                    "profit_abs": parse_float(trade.get("profit_abs")),
                    "label": 1 if profit_ratio > 0 else 0,
                    "timestamp": int(trade.get("close_timestamp") or open_timestamp),
                }
            )
    records.sort(key=lambda item: (item["timestamp"], item["archive_id"], item["strategy_name"], item["pair"]))
    return records, summaries


def build_categorical_trainer(records: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    trainer = {
        "feature_names": feature_names,
        "label_counts": {0: 0.0, 1: 0.0},
        "base_profit_ratio": percentage(sum(record["profit_ratio"] for record in records), len(records)),
        "feature_stats": {},
    }
    for record in records:
        trainer["label_counts"][int(record["label"])] += 1
    for feature_name in feature_names:
        buckets: dict[str, dict[str, Any]] = {}
        for record in records:
            value = str(record.get(feature_name, "unknown"))
            bucket = buckets.setdefault(
                value,
                {"count": 0.0, "label_counts": {0: 0.0, 1: 0.0}, "profit_sum": 0.0},
            )
            bucket["count"] += 1
            bucket["label_counts"][int(record["label"])] += 1
            bucket["profit_sum"] += parse_float(record["profit_ratio"])
        for bucket in buckets.values():
            bucket["win_rate"] = percentage(bucket["label_counts"][1], bucket["count"])
            bucket["avg_profit_ratio"] = percentage(bucket["profit_sum"], bucket["count"])
        trainer["feature_stats"][feature_name] = buckets
    trainer["base_win_rate"] = percentage(trainer["label_counts"][1], len(records))
    return trainer


def predict_categorical_record(record: dict[str, Any], trainer: dict[str, Any]) -> dict[str, Any]:
    label_counts = trainer["label_counts"]
    base_win_rate = trainer["base_win_rate"]
    base_profit_ratio = trainer["base_profit_ratio"]
    positive_total = label_counts[1]
    negative_total = label_counts[0]
    record_count = positive_total + negative_total
    positive_prior = (positive_total + 1.0) / (record_count + 2.0)
    negative_prior = (negative_total + 1.0) / (record_count + 2.0)
    log_positive = math.log(positive_prior)
    log_negative = math.log(negative_prior)
    profit_adjustments: list[float] = []
    influential_buckets: list[dict[str, Any]] = []
    for feature_name in trainer["feature_names"]:
        buckets = trainer["feature_stats"].get(feature_name, {})
        vocab_size = max(1, len(buckets))
        value = str(record.get(feature_name, "unknown"))
        bucket = buckets.get(value)
        positive_bucket = bucket["label_counts"][1] if bucket else 0.0
        negative_bucket = bucket["label_counts"][0] if bucket else 0.0
        log_positive += math.log((positive_bucket + 1.0) / (positive_total + vocab_size))
        log_negative += math.log((negative_bucket + 1.0) / (negative_total + vocab_size))
        if bucket and bucket["count"] >= 4:
            profit_delta = bucket["avg_profit_ratio"] - base_profit_ratio
            win_delta = bucket["win_rate"] - base_win_rate
            profit_adjustments.append(profit_delta)
            influential_buckets.append(
                {
                    "feature": feature_name,
                    "bucket": value,
                    "count": int(bucket["count"]),
                    "win_rate_delta": round(win_delta, 4),
                    "profit_ratio_delta": round(profit_delta, 4),
                }
            )
    logit = clamp(log_positive - log_negative, -20.0, 20.0)
    probability = 1.0 / (1.0 + math.exp(-logit))
    predicted_profit_ratio = base_profit_ratio + (
        sum(profit_adjustments) / len(profit_adjustments) if profit_adjustments else 0.0
    )
    influential_buckets.sort(key=lambda item: abs(item["profit_ratio_delta"]) + abs(item["win_rate_delta"]), reverse=True)
    predicted_positive = int(probability >= max(0.55, base_win_rate))
    return {
        "predicted_probability": probability,
        "predicted_profit_ratio": predicted_profit_ratio,
        "predicted_positive": float(predicted_positive),
        "feature_influences": influential_buckets[:4],
    }


def derive_bucket_candidates(
    feature_stats: dict[str, dict[str, dict[str, Any]]],
    base_win_rate: float,
    base_profit_ratio: float,
    hypothesis_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feature_name, buckets in feature_stats.items():
        for bucket_name, bucket in buckets.items():
            count = int(bucket.get("count", 0))
            win_rate = bucket.get("win_rate", 0.0)
            avg_profit_ratio = bucket.get("avg_profit_ratio", 0.0)
            if count < 5 or win_rate <= base_win_rate or avg_profit_ratio <= base_profit_ratio:
                continue
            evidence_quality = "high" if count >= 15 and win_rate >= base_win_rate + 0.08 else "medium"
            candidates.append(
                {
                    "candidate_name": f"{feature_name}:{bucket_name}",
                    "hypothesis_id": hypothesis_id or None,
                    "feature_conditions": f"{feature_name} == {bucket_name}",
                    "expected_behavior": (
                        f"Observed win rate {win_rate:.1%} versus base {base_win_rate:.1%} with average ROI {avg_profit_ratio * 100:.2f}%"
                    ),
                    "evidence_quality": evidence_quality,
                    "contamination_risk": "low",
                    "status": "candidate",
                    "next_action": "Replay this regime in a fresh backtest split and then confirm in dry-run.",
                    "count": count,
                    "win_rate": win_rate,
                    "avg_profit_ratio": avg_profit_ratio,
                }
            )
    candidates.sort(
        key=lambda item: (
            item["win_rate"] - base_win_rate,
            item["avg_profit_ratio"],
            item["count"],
        ),
        reverse=True,
    )
    return candidates[:5]


def run_local_ml_executor(queue_id: int, force: bool = False) -> dict[str, Any]:
    sync_ml_platform_registry()
    queue_item = get_ml_queue_item(queue_id)
    if not queue_item:
        raise ValueError(f"Queue item {queue_id} was not found.")
    if queue_item.get("status") == "blocked" and not force:
        raise ValueError(f"Queue item {queue_id} is blocked.")

    hypothesis_id = infer_queue_hypothesis_id(queue_item)
    dataset_rows = list_ml_dataset_registry()
    dataset = next((row for row in dataset_rows if row.get("id") == "backtest-archives"), None) or choose_workbench_registry_row(dataset_rows, hypothesis_id, "id")
    feature_set = choose_workbench_registry_row(list_ml_feature_set_versions(), hypothesis_id, "id")
    label_spec = choose_workbench_registry_row(list_ml_label_spec_versions(), hypothesis_id, "id")
    run_slug = f"queue-{queue_id}-{registry_slug(queue_item.get('title', 'ml-run'))}"
    started_at = iso_now()
    update_ml_queue_item(queue_id, status="running", resolution="Launching backtest archive trainer.")
    run_id = upsert_ml_experiment_run(
        {
            "queue_id": queue_id,
            "run_slug": run_slug,
            "title": f"Workbench Run: {queue_item.get('title', 'ML Study')}",
            "status": "running",
            "objective": queue_item.get("lead_question", ""),
            "dataset_id": dataset.get("id"),
            "feature_set_version_id": feature_set.get("id"),
            "label_spec_version_id": label_spec.get("id"),
            "hypothesis_id": hypothesis_id or None,
            "started_at": started_at,
            "summary": "",
            "artifact_path": "",
            "notes": queue_item.get("rationale", ""),
        }
    )

    try:
        records, group_summaries = backtest_training_records(limit_archives=12)
        if len(records) < 30:
            raise ValueError("Need at least 30 backtest trades from archive JSON payloads before the executor can train.")
        if len(group_summaries) < 2:
            raise ValueError("Need at least two archive strategy groups in backtest_results for train and validation splits.")

        split_group_index = max(1, int(len(group_summaries) * 0.7))
        if split_group_index >= len(group_summaries):
            split_group_index = len(group_summaries) - 1
        train_group_ids = {row["group_id"] for row in group_summaries[:split_group_index]}
        validation_group_ids = {row["group_id"] for row in group_summaries[split_group_index:]}
        train_records = [row for row in records if row["group_id"] in train_group_ids]
        validation_records = [row for row in records if row["group_id"] in validation_group_ids]
        if len(validation_records) < 10:
            split_index = max(12, int(len(records) * 0.7))
            if split_index >= len(records):
                split_index = len(records) - 6
            train_records = records[:split_index]
            validation_records = records[split_index:]
        if not validation_records:
            raise ValueError("Not enough backtest records to score the archive trainer.")

        feature_names = [
            "strategy_name",
            "pair",
            "side",
            "enter_tag",
            "leverage_bucket",
            "session_bucket",
            "weekday_bucket",
            "month_bucket",
        ]
        trainer = build_categorical_trainer(train_records, feature_names)
        base_win_rate = trainer["base_win_rate"]
        base_profit_ratio = trainer["base_profit_ratio"]
        predictions = [predict_categorical_record(record, trainer) for record in validation_records]

        tp = fp = tn = fn = 0
        predicted_profit_ratios: list[float] = []
        predicted_positive_count = 0
        prediction_confidences: list[float] = []
        notable_influences: list[dict[str, Any]] = []
        for record, prediction in zip(validation_records, predictions):
            predicted_positive = int(prediction["predicted_positive"])
            actual_positive = int(record["label"])
            predicted_positive_count += predicted_positive
            predicted_profit_ratios.append(prediction["predicted_profit_ratio"])
            prediction_confidences.append(prediction["predicted_probability"])
            notable_influences.extend(prediction.get("feature_influences", []))
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and actual_positive:
                fn += 1
            else:
                tn += 1

        metrics = {
            "dataset_contract": "freqtrade-backtest-json-v1",
            "archive_group_count": len(group_summaries),
            "train_count": len(train_records),
            "validation_count": len(validation_records),
            "base_win_rate": round(base_win_rate, 4),
            "base_profit_ratio": round(base_profit_ratio, 4),
            "accuracy": round(percentage(tp + tn, len(validation_records)), 4),
            "precision": round(percentage(tp, tp + fp), 4),
            "recall": round(percentage(tp, tp + fn), 4),
            "predicted_positive_rate": round(percentage(predicted_positive_count, len(validation_records)), 4),
            "mean_predicted_probability": round(percentage(sum(prediction_confidences), len(prediction_confidences)), 4),
            "validation_avg_profit_ratio": round(
                percentage(sum(record["profit_ratio"] for record in validation_records), len(validation_records)), 4
            ),
            "predicted_avg_profit_ratio": round(
                percentage(sum(predicted_profit_ratios), len(predicted_profit_ratios)), 4
            ),
        }
        bucket_candidates = derive_bucket_candidates(trainer["feature_stats"], base_win_rate, base_profit_ratio, hypothesis_id)
        notable_influences.sort(key=lambda item: abs(item["profit_ratio_delta"]) + abs(item["win_rate_delta"]), reverse=True)
        contamination_checks = [
            f"Feature set notes: {feature_set.get('leakage_notes', 'Not recorded')}",
            f"Label leakage risk: {label_spec.get('leakage_risk', 'unknown')}",
            "Contract uses freqtrade backtest archive JSON payloads only, excluding post-trade fields like exit_reason, trade_duration, min_rate, max_rate, and close_rate.",
            "Validation is split by archive strategy group when possible, then falls back to a chronological record split.",
        ]
        recommendation = "promote_to_dry_run" if metrics["precision"] >= 0.55 and bucket_candidates else "hold"
        artifact_payload = {
            "executor": "backtest-archive-categorical-trainer",
            "generated_at": iso_now(),
            "dataset_contract": {
                "id": "freqtrade-backtest-json-v1",
                "source_path": str(BACKTEST_DIR.relative_to(PROJECT_DIR)),
                "description": "Freqtrade backtest zip archives with an inner JSON result payload containing strategy.trades rows.",
                "archive_strategy_groups": group_summaries,
            },
            "queue": queue_item,
            "dataset": dataset,
            "feature_set": {
                "id": feature_set.get("id"),
                "name": feature_set.get("feature_set_name"),
                "features_used": feature_names,
            },
            "label_spec": label_spec,
            "metrics": metrics,
            "top_bucket_candidates": bucket_candidates,
            "contamination_checks": contamination_checks,
            "top_feature_influences": notable_influences[:8],
        }
        ML_RUN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = ML_RUN_ARTIFACT_DIR / f"{run_slug}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
        summary = (
            f"Backtest archive trainer fit {len(train_records)} trades from {len(group_summaries[:split_group_index])} archive groups and validated on {len(validation_records)} trades. "
            f"Precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}, accuracy {metrics['accuracy']:.2f}."
        )
        delete_ml_run_outputs(run_id)
        upsert_ml_experiment_run(
            {
                "queue_id": queue_id,
                "run_slug": run_slug,
                "title": f"Workbench Run: {queue_item.get('title', 'ML Study')}",
                "status": "completed",
                "objective": queue_item.get("lead_question", ""),
                "dataset_id": dataset.get("id"),
                "feature_set_version_id": feature_set.get("id"),
                "label_spec_version_id": label_spec.get("id"),
                "hypothesis_id": hypothesis_id or None,
                "started_at": started_at,
                "completed_at": iso_now(),
                "summary": summary,
                "artifact_path": str(artifact_path.relative_to(PROJECT_DIR)),
                "notes": queue_item.get("rationale", ""),
            }
        )
        for candidate in bucket_candidates:
            add_ml_bucket_candidate({**candidate, "experiment_run_id": run_id})
        add_ml_validation_report(
            {
                "experiment_run_id": run_id,
                "report_type": "backtest-archive-validation",
                "summary": summary,
                "metrics_json": metrics,
                "contamination_checks": " ".join(contamination_checks),
                "recommendation": recommendation,
            }
        )
        add_ml_promotion_recommendation(
            {
                "experiment_run_id": run_id,
                "candidate_name": bucket_candidates[0]["candidate_name"] if bucket_candidates else queue_item.get("title", "ML Lead"),
                "recommendation": recommendation,
                "rationale": summary,
                "blockers": "Collect more trades before promotion." if recommendation == "hold" else "Run dry-run confirmation before live influence.",
                "target_surface": "ml_workbench",
            }
        )
        update_ml_queue_item(queue_id, status="completed", resolution=summary)
        upsert_research_index_entry(
            "ml-run",
            f"ml-run:{run_slug}",
            f"ML Run {queue_item.get('title', 'ML Study')}",
            json.dumps(
                {
                    "summary": summary,
                    "metrics": metrics,
                    "artifact_path": str(artifact_path.relative_to(PROJECT_DIR)),
                },
                indent=2,
            ),
            "ml run backtest archive trainer " + " ".join(tokenize_search(queue_item.get("lead_question", ""))),
            entry_type="finding",
            author_type="agent",
            thread_id=queue_item.get("thread_id"),
            status="active",
        )
        return {
            "queue_id": queue_id,
            "run_id": run_id,
            "run_slug": run_slug,
            "summary": summary,
            "metrics": metrics,
            "artifact_path": str(artifact_path.relative_to(PROJECT_DIR)),
        }
    except Exception as exc:  # noqa: BLE001
        failure_message = str(exc)
        upsert_ml_experiment_run(
            {
                "queue_id": queue_id,
                "run_slug": run_slug,
                "title": f"Workbench Run: {queue_item.get('title', 'ML Study')}",
                "status": "failed",
                "objective": queue_item.get("lead_question", ""),
                "dataset_id": dataset.get("id"),
                "feature_set_version_id": feature_set.get("id"),
                "label_spec_version_id": label_spec.get("id"),
                "hypothesis_id": hypothesis_id or None,
                "started_at": started_at,
                "completed_at": iso_now(),
                "summary": failure_message,
                "artifact_path": "",
                "notes": queue_item.get("rationale", ""),
            }
        )
        update_ml_queue_item(queue_id, status="failed", resolution=failure_message)
        raise


def sync_ml_platform_registry() -> None:
    timestamp = iso_now()
    hypotheses = ml_hypotheses()
    buckets = ml_buckets()
    features = {row.get("feature_name"): row for row in ml_features()}
    models = ml_models()
    promotions = ml_promotions()
    hypothesis_by_id = {row.get("id"): row for row in hypotheses}

    datasets: dict[str, dict[str, Any]] = {
        "historical-ohlcv": {
            "id": "historical-ohlcv",
            "dataset_name": "Historical OHLCV Lake",
            "source_kind": "source-layer",
            "source_path": "user_data/data",
            "timeframe": "mixed",
            "coverage": "Exchange-organized candles used for offline feature generation and replay.",
            "status": "available" if (PROJECT_DIR / "user_data" / "data").exists() else "missing",
            "notes": "Raw market data source for feature tables and walk-forward splits.",
            "updated_at": timestamp,
        },
        "freqtrade-live-trades": {
            "id": "freqtrade-live-trades",
            "dataset_name": "Freqtrade Live Trade Journals",
            "source_kind": "source-layer",
            "source_path": "user_data",
            "timeframe": "5m live",
            "coverage": "SQLite trade journals from live and dry-run bot instances.",
            "status": "available",
            "notes": "Primary live evidence layer for post-trade attribution and promotion review.",
            "updated_at": timestamp,
        },
        "backtest-archives": {
            "id": "backtest-archives",
            "dataset_name": "Backtest Archives",
            "source_kind": "source-layer",
            "source_path": "user_data/backtest_results",
            "timeframe": "5m",
            "coverage": "Packed historical backtest result sets and metadata snapshots.",
            "status": "available" if BACKTEST_DIR.exists() else "missing",
            "notes": "Backtest runs need lineage to config, feature version, and promotion decisions.",
            "updated_at": timestamp,
        },
        "hyperopt-archives": {
            "id": "hyperopt-archives",
            "dataset_name": "Hyperopt Archives",
            "source_kind": "source-layer",
            "source_path": "user_data/hyperopt_results",
            "timeframe": "5m",
            "coverage": "Hyperopt outputs and cached ticker snapshots used for parameter exploration.",
            "status": "available" if (PROJECT_DIR / "user_data" / "hyperopt_results").exists() else "missing",
            "notes": "Useful for provenance, but not yet tied to model artifacts in a reproducible chain.",
            "updated_at": timestamp,
        },
    }

    labels = []
    label_versions = []
    feature_versions = []
    for hypothesis in hypotheses:
        target_variable = str(hypothesis.get("target_variable", "")).strip()
        if not target_variable:
            continue
        label_id = f"{hypothesis.get('id', registry_slug(target_variable))}-label"
        label_row = {
            "id": label_id,
            "hypothesis_id": hypothesis.get("id", ""),
            "label_name": hypothesis.get("name", target_variable),
            "target_variable": target_variable,
            "horizon_candles": extract_first_integer(target_variable),
            "leakage_risk": "high" if "forward" in target_variable.lower() else "review",
            "live_safe": 0,
            "notes": ", ".join(hypothesis.get("known_risks", [])) or "Training-only target awaiting stricter validation.",
            "updated_at": timestamp,
        }
        labels.append(label_row)
        label_versions.append(dict(label_row))
        feature_names = [name for name in hypothesis.get("features_used", []) if name]
        leakage_notes = "; ".join(
            f"{name}: {features[name].get('risk_of_leakage', 'unknown')}"
            for name in feature_names
            if name in features
        ) or "Awaiting explicit leakage review."
        feature_versions.append(
            {
                "id": f"{hypothesis.get('id', registry_slug(hypothesis.get('name', 'feature-set')))}-feature-set-v1",
                "feature_set_name": f"{hypothesis.get('name', 'Feature Set')} v1",
                "hypothesis_id": hypothesis.get("id", ""),
                "feature_names": json.dumps(feature_names),
                "leakage_notes": leakage_notes,
                "notes": hypothesis.get("theme", "Initial feature-set version derived from hypothesis registry."),
                "updated_at": timestamp,
            }
        )

    model_registry_rows = []
    for model in models:
        dataset_name = str(model.get("dataset", "") or model.get("model_name", "dataset"))
        dataset_id = registry_slug(dataset_name)
        feature_set = model.get("feature_set", [])
        feature_timeframes = sorted({features[name].get("timeframe", "") for name in feature_set if name in features})
        dataset_path = infer_dataset_source_path(dataset_name)
        resolved_dataset_path = resolve_path(dataset_path)
        datasets.setdefault(
            dataset_id,
            {
                "id": dataset_id,
                "dataset_name": dataset_name,
                "source_kind": "research-dataset",
                "source_path": dataset_path,
                "timeframe": ", ".join(item for item in feature_timeframes if item) or "unknown",
                "coverage": "Named experiment dataset referenced by the model registry.",
                "status": "available" if resolved_dataset_path and resolved_dataset_path.exists() else "missing",
                "notes": model.get("notes", "Registered from the existing model catalog."),
                "updated_at": timestamp,
            },
        )

        hypothesis_id = infer_model_hypothesis_id(model, hypotheses, buckets, promotions)
        hypothesis = hypothesis_by_id.get(hypothesis_id, {})
        label_id = f"{hypothesis_id}-label" if hypothesis_id else ""
        artifact_path = str(model.get("saved_artifact_path", "")).strip()
        artifact = resolve_path(artifact_path)
        artifact_exists = int(bool(artifact and artifact.exists()))
        lineage_status = "complete" if hypothesis_id and label_id and artifact_exists else "partial"
        model_registry_rows.append(
            {
                "id": registry_slug(str(model.get("model_name", "model"))),
                "model_name": model.get("model_name", "Unknown model"),
                "hypothesis_id": hypothesis_id,
                "dataset_id": dataset_id,
                "label_id": label_id,
                "algorithm_type": model.get("algorithm_type", "Unknown"),
                "feature_count": len(feature_set),
                "training_date": model.get("training_date", ""),
                "train_window": hypothesis.get("training_period", ""),
                "validation_window": hypothesis.get("validation_period", ""),
                "metrics": model.get("metrics", ""),
                "artifact_path": artifact_path,
                "artifact_exists": artifact_exists,
                "influenced_live_strategy": int(bool(model.get("influenced_live_strategy"))),
                "lineage_status": lineage_status,
                "notes": hypothesis.get("next_action") or model.get("notes", ""),
                "updated_at": timestamp,
            }
        )

    with closing(get_db()) as conn:
        conn.execute("DELETE FROM ml_dataset_registry")
        conn.executemany(
            """
            INSERT INTO ml_dataset_registry (
                id, dataset_name, source_kind, source_path, timeframe, coverage, status, notes, updated_at
            ) VALUES (:id, :dataset_name, :source_kind, :source_path, :timeframe, :coverage, :status, :notes, :updated_at)
            """,
            list(datasets.values()),
        )
        conn.execute("DELETE FROM ml_label_registry")
        conn.executemany(
            """
            INSERT INTO ml_label_registry (
                id, hypothesis_id, label_name, target_variable, horizon_candles, leakage_risk, live_safe, notes, updated_at
            ) VALUES (:id, :hypothesis_id, :label_name, :target_variable, :horizon_candles, :leakage_risk, :live_safe, :notes, :updated_at)
            """,
            labels,
        )
        conn.execute("DELETE FROM ml_model_registry")
        conn.executemany(
            """
            INSERT INTO ml_model_registry (
                id, model_name, hypothesis_id, dataset_id, label_id, algorithm_type, feature_count, training_date,
                train_window, validation_window, metrics, artifact_path, artifact_exists, influenced_live_strategy,
                lineage_status, notes, updated_at
            ) VALUES (
                :id, :model_name, :hypothesis_id, :dataset_id, :label_id, :algorithm_type, :feature_count, :training_date,
                :train_window, :validation_window, :metrics, :artifact_path, :artifact_exists, :influenced_live_strategy,
                :lineage_status, :notes, :updated_at
            )
            """,
            model_registry_rows,
        )
        conn.execute("DELETE FROM ml_feature_set_versions")
        conn.executemany(
            """
            INSERT INTO ml_feature_set_versions (
                id, feature_set_name, hypothesis_id, feature_names, leakage_notes, notes, updated_at
            ) VALUES (:id, :feature_set_name, :hypothesis_id, :feature_names, :leakage_notes, :notes, :updated_at)
            """,
            feature_versions,
        )
        conn.execute("DELETE FROM ml_label_spec_versions")
        conn.executemany(
            """
            INSERT INTO ml_label_spec_versions (
                id, label_name, hypothesis_id, target_variable, horizon_candles, leakage_risk, live_safe, notes, updated_at
            ) VALUES (:id, :label_name, :hypothesis_id, :target_variable, :horizon_candles, :leakage_risk, :live_safe, :notes, :updated_at)
            """,
            label_versions,
        )
        conn.commit()


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def read_config_summary(instance: dict[str, Any]) -> dict[str, Any]:
    config_path = resolve_path(instance.get("config_path"))
    if not config_path or not config_path.exists():
        return {}
    raw = load_json(config_path, {})
    return {
        "exchange": raw.get("exchange", {}).get("name"),
        "stake_currency": raw.get("stake_currency"),
        "dry_run_wallet": raw.get("dry_run_wallet"),
        "max_open_trades": raw.get("max_open_trades"),
        "timeframe": raw.get("timeframe"),
        "pairlist_method": (raw.get("pairlists") or [{}])[0].get("method"),
        "pair_count": len(raw.get("exchange", {}).get("pair_whitelist", [])),
        "pair_whitelist_preview": (raw.get("exchange", {}).get("pair_whitelist") or [])[:8],
    }


def read_strategy_excerpt(instance: dict[str, Any], max_chars: int = 7000) -> str:
    strategy_path = resolve_path(instance.get("strategy_path"))
    if not strategy_path or not strategy_path.exists():
        return ""
    return strategy_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def read_textish_file(path: Path, max_chars: int = 3000) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".ipynb":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            parts: list[str] = []
            for cell in payload.get("cells", [])[:8]:
                source = cell.get("source", [])
                if isinstance(source, list):
                    parts.append("".join(source))
                elif isinstance(source, str):
                    parts.append(source)
            return "\n".join(parts)[:max_chars]
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:  # noqa: BLE001
        return ""


def gather_file_record(path: Path, label: str | None = None, max_chars: int = 2200) -> dict[str, Any]:
    return {
        "label": label or path.name,
        "path": str(path),
        "exists": path.exists(),
        "excerpt": read_textish_file(path, max_chars=max_chars) if path.exists() else "",
    }


def search_workspace_files(keywords: list[str], limit: int = 6) -> list[dict[str, Any]]:
    allowed_suffixes = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".ipynb", ".log"}
    skip_parts = {"__pycache__", ".git", ".venv", "node_modules"}
    normalized = [token.lower() for token in keywords if token and len(token) > 2]
    if not normalized:
        return []
    results: list[tuple[int, Path]] = []
    for path in PROJECT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        text_target = str(path.relative_to(PROJECT_DIR)).lower()
        score = 0
        for token in normalized:
            if token in text_target:
                score += 4
        if score == 0:
            try:
                content = read_textish_file(path, max_chars=2500).lower()
            except Exception:  # noqa: BLE001
                content = ""
            for token in normalized:
                if token in content:
                    score += 1
        if score:
            results.append((score, path))
    results.sort(key=lambda item: (-item[0], str(item[1])))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, path in results:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(gather_file_record(path))
        if len(unique) >= limit:
            break
    return unique


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_auth_header(instance: dict[str, Any]) -> dict[str, str]:
    raw = f"{instance.get('api_username', '')}:{instance.get('api_password', '')}"
    token = base64.b64encode(raw.encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def parse_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_intish(value: Any) -> int:
    if value is None:
        return 0
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return 0
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
            embedded = extract_first_integer(value)
            return embedded or 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def first_present_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def api_get(client: httpx.Client, instance: dict[str, Any], path: str) -> Any:
    response = client.get(
        f"{instance['api_url'].rstrip('/')}{path}",
        headers=build_auth_header(instance),
        timeout=8.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_live_state(instance: dict[str, Any]) -> TeamLiveState:
    captured_at = iso_now()
    with httpx.Client() as client:
        try:
            profit = api_get(client, instance, "/api/v1/profit")
            status = api_get(client, instance, "/api/v1/status")
            config = api_get(client, instance, "/api/v1/show_config")
            try:
                balance = api_get(client, instance, "/api/v1/balance")
            except Exception:  # noqa: BLE001
                balance = None
        except httpx.HTTPStatusError as exc:
            detail = f"HTTP {exc.response.status_code}"
            return TeamLiveState(
                team_id=instance["id"],
                ts=captured_at,
                status="auth_error" if exc.response.status_code == 401 else "api_error",
                status_detail=detail,
                bot_name=None,
                strategy_name=None,
                strategy_version=None,
                current_record=None,
                equity=float(instance.get("starting_capital", 0)),
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_pnl=0.0,
                trade_count=0,
                closed_trade_count=0,
                win_rate=0.0,
                avg_roi=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                best_pair=None,
                best_rate=None,
                last_trade_at=None,
                bot_start_at=None,
                open_trade_count=0,
                heartbeat_ok=0,
            )
        except Exception as exc:  # noqa: BLE001
            return TeamLiveState(
                team_id=instance["id"],
                ts=captured_at,
                status="offline",
                status_detail=str(exc),
                bot_name=None,
                strategy_name=None,
                strategy_version=None,
                current_record=None,
                equity=float(instance.get("starting_capital", 0)),
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_pnl=0.0,
                trade_count=0,
                closed_trade_count=0,
                win_rate=0.0,
                avg_roi=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                best_pair=None,
                best_rate=None,
                last_trade_at=None,
                bot_start_at=None,
                open_trade_count=0,
                heartbeat_ok=0,
            )

    starting_capital = parse_float(instance.get("starting_capital"))
    realized = parse_float(profit.get("profit_closed_coin"))
    total = parse_float(profit.get("profit_all_coin"))
    unrealized = total - realized
    equity = starting_capital + total
    if isinstance(balance, dict):
        equity = (
            parse_float(balance.get("total_bot"))
            or parse_float(balance.get("value_bot"))
            or parse_float(balance.get("total"))
            or parse_float(balance.get("value"))
            or equity
        )
    closed = int(profit.get("closed_trade_count") or 0)
    wins = int(profit.get("winning_trades") or 0)
    current_record = f"{wins}-{max(closed - wins, 0)}"
    if isinstance(status, list):
        open_trade_count = len(status)
    elif isinstance(status, dict) and isinstance(status.get("value"), list):
        open_trade_count = len(status.get("value", []))
    else:
        open_trade_count = 0
    return TeamLiveState(
        team_id=instance["id"],
        ts=captured_at,
        status="online",
        status_detail="healthy",
        bot_name=config.get("bot_name"),
        strategy_name=config.get("strategy"),
        strategy_version=config.get("strategy_version"),
        current_record=current_record,
        equity=equity,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_pnl=total,
        trade_count=int(profit.get("trade_count") or 0),
        closed_trade_count=closed,
        win_rate=parse_float(profit.get("winrate")) * 100.0,
        avg_roi=parse_float(profit.get("profit_closed_ratio_mean")) * 100.0,
        max_drawdown=parse_float(profit.get("max_drawdown")) * 100.0,
        current_drawdown=parse_float(profit.get("current_drawdown")) * 100.0,
        best_pair=profit.get("best_pair"),
        best_rate=parse_float(profit.get("best_rate")),
        last_trade_at=profit.get("latest_trade_date"),
        bot_start_at=profit.get("bot_start_date"),
        open_trade_count=open_trade_count,
        heartbeat_ok=1,
    )


def persist_live_snapshot(state: TeamLiveState) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO live_snapshots (
                team_id, captured_at, status, status_detail, bot_name, strategy_name,
                strategy_version, current_record, equity, realized_pnl, unrealized_pnl,
                total_pnl, trade_count, closed_trade_count, win_rate, avg_roi,
                max_drawdown, current_drawdown, best_pair, best_rate, last_trade_at,
                bot_start_at, open_trade_count, heartbeat_ok
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.team_id,
                state.ts,
                state.status,
                state.status_detail,
                state.bot_name,
                state.strategy_name,
                state.strategy_version,
                state.current_record,
                state.equity,
                state.realized_pnl,
                state.unrealized_pnl,
                state.total_pnl,
                state.trade_count,
                state.closed_trade_count,
                state.win_rate,
                state.avg_roi,
                state.max_drawdown,
                state.current_drawdown,
                state.best_pair,
                state.best_rate,
                state.last_trade_at,
                state.bot_start_at,
                state.open_trade_count,
                state.heartbeat_ok,
            ),
        )
        conn.commit()


def sync_trade_db(instance: dict[str, Any]) -> None:
    db_path = resolve_path(instance.get("db_path"))
    if not db_path or not db_path.exists():
        return
    with sqlite3.connect(db_path) as source:
        source.row_factory = sqlite3.Row
        trades = source.execute(
            """
            SELECT id, pair, exchange, is_open, is_short, stake_amount, amount,
                   leverage, open_rate, close_rate, close_profit, close_profit_abs,
                   realized_profit, exit_reason, enter_tag, open_date, close_date,
                   max_rate, min_rate, strategy
            FROM trades
            """
        ).fetchall()

    synced_at = iso_now()
    with closing(get_db()) as conn:
        for trade in trades:
            open_date = trade["open_date"]
            close_date = trade["close_date"]
            duration = 0.0
            if open_date and close_date:
                try:
                    opened = datetime.fromisoformat(str(open_date))
                    closed = datetime.fromisoformat(str(close_date))
                    duration = (closed - opened).total_seconds() / 60.0
                except ValueError:
                    duration = 0.0
            profit_ratio = parse_float(trade["close_profit"])
            conn.execute(
                """
                INSERT INTO team_trades (
                    team_id, source_trade_id, pair, exchange_name, is_open, is_short,
                    stake_amount, amount, leverage, open_rate, close_rate, profit_ratio,
                    profit_pct, profit_abs, realized_profit, exit_reason, enter_tag,
                    open_date, close_date, trade_duration_minutes, max_rate, min_rate,
                    strategy_name, source_db_path, last_synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, source_trade_id) DO UPDATE SET
                    pair=excluded.pair,
                    exchange_name=excluded.exchange_name,
                    is_open=excluded.is_open,
                    is_short=excluded.is_short,
                    stake_amount=excluded.stake_amount,
                    amount=excluded.amount,
                    leverage=excluded.leverage,
                    open_rate=excluded.open_rate,
                    close_rate=excluded.close_rate,
                    profit_ratio=excluded.profit_ratio,
                    profit_pct=excluded.profit_pct,
                    profit_abs=excluded.profit_abs,
                    realized_profit=excluded.realized_profit,
                    exit_reason=excluded.exit_reason,
                    enter_tag=excluded.enter_tag,
                    open_date=excluded.open_date,
                    close_date=excluded.close_date,
                    trade_duration_minutes=excluded.trade_duration_minutes,
                    max_rate=excluded.max_rate,
                    min_rate=excluded.min_rate,
                    strategy_name=excluded.strategy_name,
                    source_db_path=excluded.source_db_path,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    instance["id"],
                    trade["id"],
                    trade["pair"],
                    trade["exchange"],
                    int(trade["is_open"]),
                    int(trade["is_short"]),
                    parse_float(trade["stake_amount"]),
                    parse_float(trade["amount"]),
                    parse_float(trade["leverage"]),
                    parse_float(trade["open_rate"]),
                    parse_float(trade["close_rate"]),
                    profit_ratio,
                    profit_ratio * 100.0,
                    parse_float(trade["close_profit_abs"]),
                    parse_float(trade["realized_profit"]),
                    trade["exit_reason"],
                    trade["enter_tag"],
                    open_date,
                    close_date,
                    duration,
                    parse_float(trade["max_rate"]),
                    parse_float(trade["min_rate"]),
                    trade["strategy"],
                    str(db_path),
                    synced_at,
                ),
            )
        conn.commit()


def upsert_trade_rows(team_id: str, source_db_path: str, trades: list[dict[str, Any]]) -> None:
    synced_at = iso_now()
    with closing(get_db()) as conn:
        for trade in trades:
            is_open = int(bool(trade.get("is_open")))
            open_date = trade.get("open_date")
            close_date = trade.get("close_date")
            duration = parse_float(trade.get("trade_duration"))
            if not duration and open_date and close_date:
                try:
                    opened = datetime.fromisoformat(str(open_date))
                    closed = datetime.fromisoformat(str(close_date))
                    duration = (closed - opened).total_seconds() / 60.0
                except ValueError:
                    duration = 0.0
            ratio_candidates = (trade.get("profit_ratio"), trade.get("close_profit")) if is_open else (
                trade.get("close_profit"),
                trade.get("profit_ratio"),
            )
            pct_candidates = (trade.get("profit_pct"), trade.get("close_profit_pct")) if is_open else (
                trade.get("close_profit_pct"),
                trade.get("profit_pct"),
            )
            abs_candidates = (trade.get("profit_abs"), trade.get("close_profit_abs")) if is_open else (
                trade.get("close_profit_abs"),
                trade.get("profit_abs"),
            )
            profit_ratio = parse_float(first_present_value(*ratio_candidates))
            profit_pct = parse_float(first_present_value(*pct_candidates))
            if not profit_pct and profit_ratio:
                profit_pct = profit_ratio * 100.0
            profit_abs = parse_float(first_present_value(*abs_candidates))
            conn.execute(
                """
                INSERT INTO team_trades (
                    team_id, source_trade_id, pair, exchange_name, is_open, is_short,
                    stake_amount, amount, leverage, open_rate, close_rate, profit_ratio,
                    profit_pct, profit_abs, realized_profit, exit_reason, enter_tag,
                    open_date, close_date, trade_duration_minutes, max_rate, min_rate,
                    strategy_name, source_db_path, last_synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, source_trade_id) DO UPDATE SET
                    pair=excluded.pair,
                    exchange_name=excluded.exchange_name,
                    is_open=excluded.is_open,
                    is_short=excluded.is_short,
                    stake_amount=excluded.stake_amount,
                    amount=excluded.amount,
                    leverage=excluded.leverage,
                    open_rate=excluded.open_rate,
                    close_rate=excluded.close_rate,
                    profit_ratio=excluded.profit_ratio,
                    profit_pct=excluded.profit_pct,
                    profit_abs=excluded.profit_abs,
                    realized_profit=excluded.realized_profit,
                    exit_reason=excluded.exit_reason,
                    enter_tag=excluded.enter_tag,
                    open_date=excluded.open_date,
                    close_date=excluded.close_date,
                    trade_duration_minutes=excluded.trade_duration_minutes,
                    max_rate=excluded.max_rate,
                    min_rate=excluded.min_rate,
                    strategy_name=excluded.strategy_name,
                    source_db_path=excluded.source_db_path,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    team_id,
                    int(trade.get("trade_id") or trade.get("id") or 0),
                    trade.get("pair", ""),
                    trade.get("exchange"),
                    is_open,
                    int(bool(trade.get("is_short"))),
                    parse_float(trade.get("stake_amount")),
                    parse_float(trade.get("amount")),
                    parse_float(trade.get("leverage")),
                    parse_float(trade.get("open_rate")),
                    parse_float(trade.get("close_rate")),
                    profit_ratio,
                    profit_pct,
                    profit_abs,
                    parse_float(trade.get("realized_profit")),
                    trade.get("exit_reason"),
                    trade.get("enter_tag"),
                    open_date,
                    close_date,
                    duration,
                    parse_float(trade.get("max_rate")),
                    parse_float(trade.get("min_rate")),
                    trade.get("strategy"),
                    source_db_path,
                    synced_at,
                ),
            )
        conn.commit()


def sync_trade_api(instance: dict[str, Any]) -> None:
    source_name = f"api:{instance['api_url']}"
    closed_trades: list[dict[str, Any]] = []
    with httpx.Client() as client:
        offset = 0
        limit = 250
        while True:
            payload = api_get(
                client,
                instance,
                f"/api/v1/trades?limit={limit}&offset={offset}",
            )
            page = payload.get("trades", []) if isinstance(payload, dict) else []
            if not page:
                break
            closed_trades.extend(page)
            total = int(payload.get("total_trades") or len(page))
            offset += len(page)
            if offset >= total:
                break
        status_payload = api_get(client, instance, "/api/v1/status")
        open_trades = (
            status_payload.get("value", [])
            if isinstance(status_payload, dict)
            else status_payload
            if isinstance(status_payload, list)
            else []
        )
    upsert_trade_rows(instance["id"], source_name, closed_trades + open_trades)


def sync_trades_for_instance(instance: dict[str, Any]) -> None:
    try:
        sync_trade_api(instance)
    except Exception:
        if instance.get("db_path"):
            sync_trade_db(instance)


def run_sync() -> dict[str, Any]:
    with sync_lock:
        instances = list_instances()
        summary: dict[str, Any] = {"synced_at": iso_now(), "teams": []}
        for instance in instances:
            state = fetch_live_state(instance)
            persist_live_snapshot(state)
            sync_trades_for_instance(instance)
            summary["teams"].append(
                {
                    "team_id": instance["id"],
                    "status": state.status,
                    "total_pnl": state.total_pnl,
                    "trade_count": state.trade_count,
                }
            )
        return summary


def sync_loop() -> None:
    while True:
        try:
            run_sync()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(POLL_INTERVAL_SECONDS)


def latest_snapshot_map() -> dict[str, sqlite3.Row]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT ls.*
            FROM live_snapshots ls
            JOIN (
                SELECT team_id, MAX(id) AS max_id
                FROM live_snapshots
                GROUP BY team_id
            ) latest
            ON latest.max_id = ls.id
            """
        ).fetchall()
        return {row["team_id"]: row for row in rows}


def snapshot_history(team_id: str, limit: int = 50) -> list[sqlite3.Row]:
    with closing(get_db()) as conn:
        return conn.execute(
            """
            SELECT *
            FROM live_snapshots
            WHERE team_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (team_id, limit),
        ).fetchall()


def team_trade_rows(team_id: str, include_open: bool = True) -> list[sqlite3.Row]:
    with closing(get_db()) as conn:
        query = """
            SELECT *
            FROM team_trades
            WHERE team_id = ?
        """
        params: list[Any] = [team_id]
        if not include_open:
            query += " AND is_open = 0"
        query += " ORDER BY COALESCE(close_date, open_date) DESC"
        return conn.execute(query, params).fetchall()


def trade_aggregates(team_id: str) -> dict[str, Any]:
    trades = team_trade_rows(team_id)
    closed = [row for row in trades if not row["is_open"]]
    open_trades = [row for row in trades if row["is_open"]]
    wins = [row for row in closed if parse_float(row["profit_abs"]) > 0]
    avg_roi = sum(parse_float(row["profit_pct"]) for row in closed) / len(closed) if closed else 0.0
    best_trade = max((parse_float(row["profit_pct"]) for row in closed), default=0.0)
    worst_open = min((parse_float(row["profit_pct"]) for row in open_trades), default=0.0)
    exit_breakdown: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "avg_roi": 0.0, "total_profit": 0.0, "best": -999.0, "worst": 999.0, "avg_hold_minutes": 0.0})
    for row in closed:
        key = row["exit_reason"] or "unknown"
        bucket = exit_breakdown[key]
        bucket["count"] += 1
        bucket["total_profit"] += parse_float(row["profit_abs"])
        bucket["avg_roi"] += parse_float(row["profit_pct"])
        bucket["avg_hold_minutes"] += parse_float(row["trade_duration_minutes"])
        bucket["best"] = max(bucket["best"], parse_float(row["profit_pct"]))
        bucket["worst"] = min(bucket["worst"], parse_float(row["profit_pct"]))
    for bucket in exit_breakdown.values():
        if bucket["count"]:
            bucket["avg_roi"] /= bucket["count"]
            bucket["avg_hold_minutes"] /= bucket["count"]
    return {
        "total": len(trades),
        "closed": len(closed),
        "open": len(open_trades),
        "wins": len(wins),
        "win_rate": (len(wins) / len(closed) * 100.0) if closed else 0.0,
        "avg_roi": avg_roi,
        "best_trade": best_trade,
        "worst_open_trade": worst_open,
        "exit_breakdown": dict(sorted(exit_breakdown.items(), key=lambda item: item[1]["count"], reverse=True)),
    }


def compute_power_rankings() -> list[dict[str, Any]]:
    instances = list_instances()
    latest = latest_snapshot_map()
    overrides = power_ranking_overrides()
    rankings: list[dict[str, Any]] = []
    for instance in instances:
        state = latest.get(instance["id"])
        trades = trade_aggregates(instance["id"])
        override = overrides.get(instance["id"], {})
        data_quality = override.get("data_quality") or instance.get("data_quality", "Unknown")
        trust_score = 40
        if data_quality.lower() == "official":
            trust_score += 30
        elif data_quality.lower() == "clean":
            trust_score += 20
        elif data_quality.lower() == "restarted":
            trust_score += 10
        if state and state["heartbeat_ok"]:
            trust_score += 10
        maturity = min(trades["closed"] * 2, 20)
        recent_form = max(min((state["total_pnl"] if state else 0.0) * 5, 20), -20)
        risk_profile = max(0.0, 20.0 - abs((state["max_drawdown"] if state else 0.0)))
        if override.get("trust_score") not in {None, ""}:
            trust_score = parse_float(override.get("trust_score"))
        score = trust_score + maturity + recent_form + risk_profile
        rankings.append(
            {
                "team_id": instance["id"],
                "team": instance["display_name"],
                "trust_score": round(trust_score, 1),
                "trust_note": override.get("trust_note", ""),
                "data_quality": data_quality,
                "data_quality_note": override.get("data_quality_note", ""),
                "strategy_maturity": round(maturity, 1),
                "recent_form": round(recent_form, 1),
                "risk_profile": round(risk_profile, 1),
                "interesting_discoveries": override.get("interesting_discoveries") or instance.get("notes", ""),
                "score": round(score, 1),
            }
        )
    rankings.sort(key=lambda item: item["score"], reverse=True)
    return rankings


def version_registry() -> list[dict[str, Any]]:
    teams_by_strategy: dict[str, list[str]] = defaultdict(list)
    instances = list_instances()
    records: dict[str, dict[str, Any]] = {}
    for instance in instances:
        strategy_path = resolve_path(instance.get("strategy_path"))
        if not strategy_path or not strategy_path.exists():
            continue
        key = str(strategy_path)
        teams_by_strategy[key].append(instance["display_name"])
        if key in records:
            continue
        records[key] = {
            "filename": strategy_path.name,
            "strategy_class": strategy_path.stem,
            "git_hash": None,
            "file_hash": sha256_file(strategy_path)[:16],
            "last_modified": datetime.fromtimestamp(strategy_path.stat().st_mtime, UTC).isoformat(),
            "notes": "Derived from local strategy file checksum.",
            "path": str(strategy_path),
        }
    result = []
    for key, record in records.items():
        item = dict(record)
        item["teams"] = teams_by_strategy[key]
        result.append(item)
    result.sort(key=lambda row: row["filename"].lower())
    return extend_version_registry_with_development(result)


def add_months(value: datetime, months: int) -> datetime:
    month_index = (value.month - 1) + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month, day=1)


def league_season_anchor() -> datetime:
    raw = get_setting("league_season_anchor", "")
    anchor = resolve_optional_datetime(raw)
    if anchor:
        return anchor.astimezone(LOCAL_TIMEZONE).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return local_now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def league_season_for_number(season_number: int) -> dict[str, Any]:
    anchor = league_season_anchor()
    start = add_months(anchor, max(0, season_number - 1))
    end = add_months(start, 1)
    now_local = local_now()
    status = "completed" if end <= now_local else "current" if start <= now_local < end else "upcoming"
    return {
        "season_number": season_number,
        "season_label": f"Season {season_number}",
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "status": status,
        "is_current": status == "current",
    }


def current_league_season() -> dict[str, Any]:
    anchor = league_season_anchor()
    now_local = local_now()
    month_delta = (now_local.year - anchor.year) * 12 + (now_local.month - anchor.month)
    if now_local < anchor:
        month_delta = 0
    return league_season_for_number(month_delta + 1)


def hydrate_league_season(row: dict[str, Any]) -> dict[str, Any]:
    row["awards"] = decode_jsonish_payload(str(row.get("awards_json") or "[]"), [])
    return row


def hydrate_league_team_season_review(row: dict[str, Any]) -> dict[str, Any]:
    row["rubric"] = decode_jsonish_payload(str(row.get("rubric_json") or "[]"), [])
    decision = str(row.get("decision_bucket") or "hold")
    row["decision_label"] = SEASON_DECISION_LABELS.get(decision, decision.replace("_", " ").title())
    row["approval_required_bool"] = bool(int(row.get("approval_required") or 0))
    return row


def list_league_seasons(limit: int = 6) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM league_seasons
            ORDER BY season_number DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [hydrate_league_season(dict(row)) for row in rows]


def get_league_season(season_number: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM league_seasons WHERE season_number = ?", (season_number,)).fetchone()
    return hydrate_league_season(dict(row)) if row else None


def upsert_league_season(payload: dict[str, Any]) -> dict[str, Any]:
    now = iso_now()
    season_number = int(payload.get("season_number") or 0)
    existing = get_league_season(season_number)
    created_at = str(existing.get("created_at") or now) if existing else now
    with closing(get_db()) as conn:
        if existing:
            conn.execute(
                """
                UPDATE league_seasons
                SET season_label = ?, started_at = ?, ended_at = ?, status = ?, awards_json = ?, draft_slots = ?, turnover_processed_at = ?, updated_at = ?
                WHERE season_number = ?
                """,
                (
                    payload.get("season_label", f"Season {season_number}"),
                    payload.get("started_at", ""),
                    payload.get("ended_at", ""),
                    payload.get("status", "current"),
                    payload.get("awards_json", "[]"),
                    int(payload.get("draft_slots") or SEASON_DRAFT_SLOT_LIMIT),
                    payload.get("turnover_processed_at"),
                    now,
                    season_number,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO league_seasons (
                    season_number, season_label, started_at, ended_at, status, awards_json, draft_slots, turnover_processed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season_number,
                    payload.get("season_label", f"Season {season_number}"),
                    payload.get("started_at", ""),
                    payload.get("ended_at", ""),
                    payload.get("status", "current"),
                    payload.get("awards_json", "[]"),
                    int(payload.get("draft_slots") or SEASON_DRAFT_SLOT_LIMIT),
                    payload.get("turnover_processed_at"),
                    created_at,
                    now,
                ),
            )
        conn.commit()
    return get_league_season(season_number) or {}


def list_league_team_season_reviews(
    season_number: int | None = None,
    team_id: str | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if season_number is not None:
        clauses.append("season_number = ?")
        params.append(season_number)
    if team_id:
        clauses.append("team_id = ?")
        params.append(team_id)
    query = "SELECT * FROM league_team_season_reviews"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY season_number DESC, team_name ASC LIMIT ?"
    params.append(limit)
    with closing(get_db()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [hydrate_league_team_season_review(dict(row)) for row in rows]


def latest_league_team_season_review(team_id: str) -> dict[str, Any] | None:
    rows = list_league_team_season_reviews(team_id=team_id, limit=1)
    return rows[0] if rows else None


def get_league_team_season_review(review_key: str) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM league_team_season_reviews WHERE review_key = ?", (review_key,)).fetchone()
    return hydrate_league_team_season_review(dict(row)) if row else None


def upsert_league_team_season_review(payload: dict[str, Any]) -> dict[str, Any]:
    now = iso_now()
    review_key = str(payload.get("review_key") or "")
    existing = get_league_team_season_review(review_key)
    created_at = str(existing.get("created_at") or now) if existing else now
    values = (
        int(payload.get("season_number") or 0),
        payload.get("season_label", ""),
        payload.get("season_started_at", ""),
        payload.get("season_ended_at", ""),
        payload.get("team_id", ""),
        payload.get("team_name", ""),
        payload.get("strategy_family", ""),
        payload.get("pair_universe", ""),
        payload.get("role", ""),
        payload.get("strategy_path", ""),
        payload.get("strategy_file_hash", ""),
        parse_float(payload.get("runtime_hours")),
        parse_float(payload.get("scheduled_hours")),
        parse_float(payload.get("heartbeat_ratio")),
        int(payload.get("closed_trades") or 0),
        parse_float(payload.get("win_rate")),
        parse_float(payload.get("avg_roi")),
        parse_float(payload.get("realized_pnl")),
        parse_float(payload.get("total_pnl")),
        parse_float(payload.get("max_drawdown")),
        parse_float(payload.get("worst_open_trade")),
        int(payload.get("champion_exits") or 0),
        parse_float(payload.get("overall_score")),
        payload.get("grade", ""),
        payload.get("decision_bucket", "hold"),
        payload.get("recommendation", ""),
        payload.get("fix_suggestion", ""),
        payload.get("summary", ""),
        payload.get("rubric_json", "[]"),
        int(payload.get("approval_required") or 0),
        payload.get("approval_status", "pending"),
        payload.get("approval_notes", ""),
        payload.get("approved_action", ""),
    )
    with closing(get_db()) as conn:
        if existing:
            conn.execute(
                """
                UPDATE league_team_season_reviews
                SET season_number = ?, season_label = ?, season_started_at = ?, season_ended_at = ?,
                    team_id = ?, team_name = ?, strategy_family = ?, pair_universe = ?, role = ?,
                    strategy_path = ?, strategy_file_hash = ?, runtime_hours = ?, scheduled_hours = ?, heartbeat_ratio = ?,
                    closed_trades = ?, win_rate = ?, avg_roi = ?, realized_pnl = ?, total_pnl = ?, max_drawdown = ?,
                    worst_open_trade = ?, champion_exits = ?, overall_score = ?, grade = ?, decision_bucket = ?,
                    recommendation = ?, fix_suggestion = ?, summary = ?, rubric_json = ?, approval_required = ?,
                    approval_status = ?, approval_notes = ?, approved_action = ?, updated_at = ?
                WHERE review_key = ?
                """,
                (*values, now, review_key),
            )
        else:
            conn.execute(
                """
                INSERT INTO league_team_season_reviews (
                    review_key, season_number, season_label, season_started_at, season_ended_at, team_id, team_name,
                    strategy_family, pair_universe, role, strategy_path, strategy_file_hash, runtime_hours,
                    scheduled_hours, heartbeat_ratio, closed_trades, win_rate, avg_roi, realized_pnl, total_pnl,
                    max_drawdown, worst_open_trade, champion_exits, overall_score, grade, decision_bucket,
                    recommendation, fix_suggestion, summary, rubric_json, approval_required, approval_status,
                    approval_notes, approved_action, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (review_key, *values, created_at, now),
            )
        conn.commit()
    return get_league_team_season_review(review_key) or {}


def list_league_season_draft_recommendations(season_number: int | None = None, limit: int = 12) -> list[dict[str, Any]]:
    query = "SELECT * FROM league_season_draft_recommendations"
    params: list[Any] = []
    if season_number is not None:
        query += " WHERE season_number = ?"
        params.append(season_number)
    query += " ORDER BY season_number DESC, overall_score DESC, projected_total_pnl DESC LIMIT ?"
    params.append(limit)
    with closing(get_db()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_league_season_draft_recommendation(draft_id: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM league_season_draft_recommendations WHERE id = ?", (draft_id,)).fetchone()
    return dict(row) if row else None


def upsert_league_season_draft_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    now = iso_now()
    season_number = int(payload.get("season_number") or 0)
    candidate_id = int(payload.get("candidate_id") or 0)
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT id, created_at FROM league_season_draft_recommendations WHERE season_number = ? AND candidate_id = ?",
            (season_number, candidate_id),
        ).fetchone()
        created_at = str(existing["created_at"] or now) if existing else now
        values = (
            payload.get("season_label", f"Season {season_number}"),
            payload.get("season_started_at", ""),
            payload.get("season_ended_at", ""),
            payload.get("candidate_name", ""),
            payload.get("latest_review_key", ""),
            payload.get("candidate_tier", ""),
            parse_float(payload.get("overall_score")),
            parse_float(payload.get("projected_total_pnl")),
            payload.get("recommendation", ""),
            payload.get("rationale", ""),
            payload.get("approval_status", "pending"),
            payload.get("approval_notes", ""),
        )
        if existing:
            conn.execute(
                """
                UPDATE league_season_draft_recommendations
                SET season_label = ?, season_started_at = ?, season_ended_at = ?, candidate_name = ?, latest_review_key = ?,
                    candidate_tier = ?, overall_score = ?, projected_total_pnl = ?, recommendation = ?, rationale = ?,
                    approval_status = ?, approval_notes = ?, updated_at = ?
                WHERE season_number = ? AND candidate_id = ?
                """,
                (*values, now, season_number, candidate_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO league_season_draft_recommendations (
                    season_number, season_label, season_started_at, season_ended_at, candidate_id, candidate_name,
                    latest_review_key, candidate_tier, overall_score, projected_total_pnl, recommendation, rationale,
                    approval_status, approval_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season_number,
                    *values[:3],
                    candidate_id,
                    *values[3:],
                    created_at,
                    now,
                ),
            )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM league_season_draft_recommendations WHERE season_number = ? AND candidate_id = ?",
            (season_number, candidate_id),
        ).fetchone()
    return dict(row) if row else {}


def season_trade_rows(team_id: str, season_started_at: datetime, season_ended_at: datetime) -> list[sqlite3.Row]:
    season_started_utc = normalize_utc(season_started_at)
    season_ended_utc = normalize_utc(season_ended_at)
    rows = team_trade_rows(team_id)
    filtered = []
    for row in rows:
        timestamp = normalize_utc(resolve_optional_datetime(str(row["close_date"] or row["open_date"] or "")))
        if timestamp and season_started_utc and season_ended_utc and season_started_utc <= timestamp < season_ended_utc:
            filtered.append(row)
    return filtered


def season_snapshot_rows(team_id: str, season_started_at: datetime, season_ended_at: datetime) -> list[dict[str, Any]]:
    season_started_utc = normalize_utc(season_started_at)
    season_ended_utc = normalize_utc(season_ended_at)
    rows = snapshot_history(team_id, limit=1000)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        row_dict.setdefault("max_drawdown", 0)
        row_dict.setdefault("worst_open_trade", 0)
        row_dict.setdefault("unrealized_pnl", 0)
        row_dict.setdefault("heartbeat_ok", 0)
        captured_at = normalize_utc(resolve_optional_datetime(str(row_dict.get("captured_at") or "")))
        if captured_at and season_started_utc and season_ended_utc and season_started_utc <= captured_at < season_ended_utc:
            filtered.append(row_dict)
    filtered.reverse()
    return filtered


def build_season_runtime_summary(snapshots: list[dict[str, Any]], season_started_at: datetime, season_ended_at: datetime) -> tuple[float, float]:
    observed_runtime = 0.0
    heartbeat_count = 0
    prev_at: datetime | None = None
    for snap in snapshots:
        captured_at = resolve_optional_datetime(str(snap["captured_at"] or ""))
        if not captured_at:
            continue
        if bool(snap["heartbeat_ok"]):
            heartbeat_count += 1
            if prev_at:
                observed_runtime += min(hours_between(prev_at, captured_at), (POLL_INTERVAL_SECONDS * 2) / 3600.0)
        prev_at = captured_at
    scheduled_hours = hours_between(season_started_at, season_ended_at)
    heartbeat_ratio = (heartbeat_count / len(snapshots)) if snapshots else 0.0
    return observed_runtime, heartbeat_ratio if scheduled_hours > 0 else 0.0


def season_review_grade(score: float) -> str:
    for threshold, label in POST_SHIFT_GRADE_BANDS:
        if score >= threshold:
            return label
    return "F"


def build_official_team_fix_suggestion(decision_bucket: str, weakest_labels: list[str], row: dict[str, Any]) -> str:
    family = row.get("strategy_family") or row.get("team_name") or "team"
    weakness_copy = ", ".join(weakest_labels) if weakest_labels else "risk control"
    if decision_bucket == "hold":
        return f"Hold the {family} core. Preserve the current version and keep auditing {weakness_copy} for next month."
    if decision_bucket == "tweak":
        return f"Tighten {weakness_copy} without changing the core thesis. Keep the live file lineage and test a small parameter branch first."
    if decision_bucket == "update":
        return f"Prepare a new approved branch for {family} focused on {weakness_copy}, then compare the new version against the incumbent before season rollover."
    if decision_bucket == "revamp":
        return f"Treat the next approved change as a deeper redesign. Rework {weakness_copy} and validate the new version as a challenger before it touches the incumbent slot."
    return f"Queue a relegation review and replacement search. {family} needs owner approval before it leaves the top tier, and the likely weak spots are {weakness_copy}."


def preview_official_team_season_review(team: dict[str, Any], season: dict[str, Any]) -> dict[str, Any]:
    season_started_at = resolve_optional_datetime(str(season.get("started_at") or "")) or utc_now()
    season_ended_at = resolve_optional_datetime(str(season.get("ended_at") or "")) or utc_now()
    snapshots = season_snapshot_rows(str(team.get("id") or ""), season_started_at, season_ended_at)
    trades = season_trade_rows(str(team.get("id") or ""), season_started_at, season_ended_at)
    runtime_hours, heartbeat_ratio = build_season_runtime_summary(snapshots, season_started_at, season_ended_at)
    closed_trades = [row for row in trades if not bool(row["is_open"])]
    wins = [row for row in closed_trades if parse_float(row["profit_abs"]) > 0]
    win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0
    avg_roi = sum(parse_float(row["profit_pct"]) for row in closed_trades) / len(closed_trades) if closed_trades else 0.0
    realized_pnl = sum(parse_float(row["profit_abs"]) for row in closed_trades)
    latest_snapshot = snapshots[-1] if snapshots else latest_snapshot_map().get(str(team.get("id") or ""))
    total_pnl = realized_pnl + (parse_float(latest_snapshot["unrealized_pnl"]) if latest_snapshot else 0.0)
    max_drawdown = max((parse_float(row["max_drawdown"]) for row in snapshots), default=0.0)
    worst_open_trade = min((parse_float(row["worst_open_trade"]) for row in snapshots), default=0.0)
    champion_exits = sum(
        1
        for row in closed_trades
        if str(row["exit_reason"] or "") in {"champ_dynamic_roi", "champ_dynamic_roi_hit"}
    )
    scheduled_hours = hours_between(season_started_at, season_ended_at)
    starting_capital = max(1.0, parse_float(team.get("starting_capital")))
    pnl_pct = percentage(total_pnl, starting_capital)
    profitability_score = clamp(12.0 + pnl_pct * 2.4 + max(avg_roi, 0.0) * 0.9, 0.0, 30.0)
    consistency_score = clamp((win_rate / 100.0) * 12.0 + min(len(closed_trades), 16) * 0.5, 0.0, 20.0)
    risk_score = clamp(20.0 - max_drawdown * 1.1 - max(0.0, abs(min(worst_open_trade, 0.0)) - 2.0) * 0.35, 0.0, 20.0)
    reliability_score = clamp((runtime_hours / scheduled_hours if scheduled_hours else 0.0) * 10.0 + heartbeat_ratio * 5.0, 0.0, 15.0)
    # Champion exits are a Cosmo/Wanda-only mechanic — excluded from this universal grade so
    # strategies without it aren't penalized. Efficiency = ROI quality + trade activity only.
    efficiency_score = clamp(max(avg_roi, 0.0) * 3.0 + min(len(closed_trades), 12) * 0.6, 0.0, 15.0)
    rubric = [
        {"label": "Performance", "score": round(profitability_score, 1), "max_score": 30, "note": f"{total_pnl:+.2f} total P&L ({pnl_pct:+.2f}% of starting capital)."},
        {"label": "Consistency", "score": round(consistency_score, 1), "max_score": 20, "note": f"{len(closed_trades)} closed trades with a {win_rate:.1f}% win rate."},
        {"label": "Risk", "score": round(risk_score, 1), "max_score": 20, "note": f"{max_drawdown:.1f}% max drawdown and {worst_open_trade:.1f}% worst open trade."},
        {"label": "Reliability", "score": round(reliability_score, 1), "max_score": 15, "note": f"{runtime_hours:.1f}/{scheduled_hours:.1f} observed runtime hours, heartbeat ratio {heartbeat_ratio * 100:.1f}%."},
        {"label": "Efficiency", "score": round(efficiency_score, 1), "max_score": 15, "note": f"{avg_roi:.2f}% average ROI across {len(closed_trades)} closed trades."},
    ]
    overall_score = round(sum(parse_float(item.get("score")) for item in rubric), 1)
    grade = season_review_grade(overall_score)
    if overall_score >= 82 and total_pnl >= 0 and max_drawdown <= 14:
        decision_bucket = "hold"
    elif overall_score >= 68 and total_pnl >= -starting_capital * 0.04:
        decision_bucket = "tweak"
    elif overall_score >= 55:
        decision_bucket = "update"
    elif overall_score >= 40:
        decision_bucket = "revamp"
    else:
        decision_bucket = "relegate"
    weakest = sorted(rubric, key=lambda item: parse_float(item.get("score")) / max(1.0, parse_float(item.get("max_score"))))[:2]
    weakest_labels = [str(item.get("label") or "") for item in weakest if item.get("label")]
    fix_suggestion = build_official_team_fix_suggestion(decision_bucket, weakest_labels, {"strategy_family": team.get("strategy_family"), "team_name": team.get("display_name")})
    recommendation = (
        "No change required. Keep the incumbent live version as-is." if decision_bucket == "hold"
        else "Awaiting owner approval before any top-team mutation or relegation."
    )
    strategy_path = resolve_path(team.get("strategy_path"))
    strategy_file_hash = sha256_file(strategy_path)[:16] if strategy_path and strategy_path.exists() else ""
    summary = (
        f"{team.get('display_name', team.get('name', 'Team'))} graded {grade} for {season.get('season_label', '')}: "
        f"{total_pnl:+.2f} total P&L, {len(closed_trades)} closed trades, {win_rate:.1f}% win rate, "
        f"and {max_drawdown:.1f}% max drawdown. Recommendation: {SEASON_DECISION_LABELS.get(decision_bucket, decision_bucket)}."
    )
    return hydrate_league_team_season_review(
        {
            "review_key": f"season:{season.get('season_number')}:{team.get('id')}",
            "season_number": season.get("season_number"),
            "season_label": season.get("season_label", ""),
            "season_started_at": season.get("started_at", ""),
            "season_ended_at": season.get("ended_at", ""),
            "team_id": team.get("id", ""),
            "team_name": team.get("display_name", team.get("name", "")),
            "strategy_family": team.get("strategy_family", ""),
            "pair_universe": team.get("pair_universe", ""),
            "role": team.get("role", ""),
            "strategy_path": team.get("strategy_path", ""),
            "strategy_file_hash": strategy_file_hash,
            "runtime_hours": round(runtime_hours, 2),
            "scheduled_hours": round(scheduled_hours, 2),
            "heartbeat_ratio": round(heartbeat_ratio, 4),
            "closed_trades": len(closed_trades),
            "win_rate": round(win_rate, 2),
            "avg_roi": round(avg_roi, 3),
            "realized_pnl": round(realized_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "max_drawdown": round(max_drawdown, 2),
            "worst_open_trade": round(worst_open_trade, 2),
            "champion_exits": champion_exits,
            "overall_score": overall_score,
            "grade": grade,
            "decision_bucket": decision_bucket,
            "recommendation": recommendation,
            "fix_suggestion": fix_suggestion,
            "summary": summary,
            "rubric_json": json.dumps(rubric, indent=2),
            "approval_required": 0 if decision_bucket == "hold" else 1,
            "approval_status": "not_required" if decision_bucket == "hold" else "pending",
            "approval_notes": "",
            "approved_action": "",
        }
    )


def build_season_awards(preview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not preview_rows:
        return []
    champion = max(preview_rows, key=lambda row: parse_float(row.get("total_pnl")))
    efficient = max(preview_rows, key=lambda row: parse_float(row.get("avg_roi")) if int(row.get("closed_trades") or 0) >= 2 else -999.0)
    iron_bot = max(preview_rows, key=lambda row: parse_float(row.get("runtime_hours")))
    risk_manager = max(preview_rows, key=lambda row: -parse_float(row.get("max_drawdown")) if parse_float(row.get("total_pnl")) >= 0 else -999.0)
    return [
        {"title": "Season Champion", "winner": champion.get("team_name"), "reason": f"Finished with {parse_float(champion.get('total_pnl')):+.2f} total P&L."},
        {"title": "Efficiency Award", "winner": efficient.get("team_name"), "reason": f"Posted {parse_float(efficient.get('avg_roi')):.2f}% average ROI on closed trades."},
        {"title": "Iron Bot", "winner": iron_bot.get("team_name"), "reason": f"Logged {parse_float(iron_bot.get('runtime_hours')):.1f} observed runtime hours."},
        {"title": "Risk Manager", "winner": risk_manager.get("team_name"), "reason": f"Managed season drawdown at {parse_float(risk_manager.get('max_drawdown')):.1f}%."},
    ]


def preview_season_draft_recommendations(season: dict[str, Any], limit: int = SEASON_DRAFT_SLOT_LIMIT) -> list[dict[str, Any]]:
    rows = [row for row in development_candidate_rows() if row.get("tier_competition") == "prospect_12h" and row.get("tier") != "archived"]
    rows.sort(
        key=lambda row: (
            parse_float(row.get("latest_post_shift_score")),
            parse_float(row.get("projected_total_pnl_per_24h")),
            parse_float(row.get("raw_total_pnl")),
        ),
        reverse=True,
    )
    preview_rows = []
    for row in rows[:limit]:
        latest_review = row.get("latest_post_shift_review") or {}
        recommendation = "Draft consideration"
        rationale = (
            f"{row.get('name', 'Candidate')} is one of the strongest 12-hour prospects this season with "
            f"{parse_float(row.get('projected_total_pnl_per_24h')):+.2f} projected 24h pace and "
            f"{latest_review.get('grade', 'n/a')} latest review."
        )
        preview_rows.append(
            {
                "season_number": season.get("season_number"),
                "season_label": season.get("season_label", ""),
                "season_started_at": season.get("started_at", ""),
                "season_ended_at": season.get("ended_at", ""),
                "candidate_id": int(row.get("id") or 0),
                "candidate_name": row.get("name", ""),
                "latest_review_key": latest_review.get("review_key", ""),
                "candidate_tier": row.get("tier_competition") or row.get("tier", ""),
                "overall_score": parse_float(row.get("latest_post_shift_score")),
                "projected_total_pnl": parse_float(row.get("projected_total_pnl_per_24h")),
                "recommendation": recommendation,
                "rationale": rationale,
                "approval_status": "pending",
                "approval_notes": "",
            }
        )
    return preview_rows


def seasons_pending_turnover(force: bool = False) -> list[dict[str, Any]]:
    current = current_league_season()
    if force:
        return [current]
    seasons = []
    for season_number in range(1, int(current.get("season_number") or 1)):
        season = league_season_for_number(season_number)
        existing = get_league_season(season_number)
        if existing and str(existing.get("status") or "") == "processed":
            continue
        if resolve_optional_datetime(str(season.get("ended_at") or "")) and resolve_optional_datetime(str(season.get("ended_at") or "")) <= local_now():
            seasons.append(season)
    return seasons


def run_season_turnover(force: bool = False) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for season in seasons_pending_turnover(force=force):
        preview_rows = [preview_official_team_season_review(team, season) for team in list_instances()]
        awards = build_season_awards(preview_rows)
        status = "preview" if force and season.get("status") == "current" else "processed"
        upsert_league_season(
            {
                "season_number": season.get("season_number"),
                "season_label": season.get("season_label", ""),
                "started_at": season.get("started_at", ""),
                "ended_at": season.get("ended_at", ""),
                "status": status,
                "awards_json": json.dumps(awards, indent=2),
                "draft_slots": SEASON_DRAFT_SLOT_LIMIT,
                "turnover_processed_at": iso_now(),
            }
        )
        for row in preview_rows:
            upsert_league_team_season_review(row)
        for row in preview_season_draft_recommendations(season):
            upsert_league_season_draft_recommendation(row)
        # Permanent trophy-shelf awards — only for a genuinely completed season, never a
        # mid-season force-preview (awards must reflect a closed body of evidence).
        if status == "processed":
            try:
                evaluate_strategy_awards({**season, "status": status})
            except Exception as exc:  # noqa: BLE001
                log_maintenance("awards", "error", f"Award evaluation failed for {season.get('season_label','')}: {exc}")
        processed.append({**season, "awards": awards, "review_count": len(preview_rows)})
    if processed:
        log_maintenance("season", "success", f"Processed season turnover for {', '.join(item['season_label'] for item in processed)}.")
    # A completed season may close a quarter — generate the Quarterly Champion report.
    try:
        maybe_generate_quarterly_reports()
    except Exception as exc:  # noqa: BLE001
        log_maintenance("quarterly", "error", f"Quarterly review check failed: {exc}")
    return processed


# ---------------------------------------------------------------------------
# Quarterly Champion — a formal capital-eligibility review every three COMPLETED
# major-league seasons. It aggregates the official per-season evidence ATL already
# produces (league_team_season_reviews) using peer-relative percentile scoring and
# recency weighting. It is NOT a live race: a report exists only after a quarter
# closes, and is then a permanent archived artifact. Winning ≠ deploying real money.
# ---------------------------------------------------------------------------

QUARTER_SEASON_SPAN = 3
# Recency weights by position within the quarter (season 1 / 2 / 3): lately matters more.
QUARTER_RECENCY_WEIGHTS = [0.20, 0.30, 0.50]
# Official metrics aggregated per season — all already stored on league_team_season_reviews.
# direction +1 = higher is better, -1 = lower is better.
QUARTER_METRICS: list[tuple[str, int, str]] = [
    ("overall_score", 1, "Season score"),
    ("realized_pnl", 1, "Realized P&L"),
    ("total_pnl", 1, "Total P&L"),
    ("win_rate", 1, "Win rate"),
    ("closed_trades", 1, "Closed trades"),
    ("avg_roi", 1, "Average ROI"),
    ("max_drawdown", -1, "Max drawdown"),
    ("worst_open_trade", 1, "Worst open trade"),  # negative %; closer to 0 is better
]


def quarter_for_season(season_number: int) -> int:
    return (max(1, season_number) + QUARTER_SEASON_SPAN - 1) // QUARTER_SEASON_SPAN


def quarter_season_range(quarter_number: int) -> tuple[int, int]:
    end = quarter_number * QUARTER_SEASON_SPAN
    return end - QUARTER_SEASON_SPAN + 1, end


def official_team_directory() -> dict[str, str]:
    """id -> display name for the OFFICIAL major-league teams only (the scope of this review)."""
    return {
        str(t.get("id")): str(t.get("display_name") or t.get("name") or t.get("id"))
        for t in list_instances() if t.get("id")
    }


def hydrate_quarterly_report(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["report"] = decode_jsonish_payload(str(row.get("report_json") or "{}"), {})
    return row


def get_quarterly_report(quarter_number: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        r = conn.execute("SELECT * FROM league_quarterly_reports WHERE quarter_number = ?", (quarter_number,)).fetchone()
    return hydrate_quarterly_report(dict(r)) if r else None


def list_quarterly_reports(limit: int = 20) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM league_quarterly_reports ORDER BY quarter_number DESC LIMIT ?", (limit,)
        ).fetchall()
    return [hydrate_quarterly_report(dict(r)) for r in rows]


def upsert_quarterly_report(payload: dict[str, Any]) -> dict[str, Any]:
    now = iso_now()
    quarter_number = int(payload.get("quarter_number") or 0)
    existing = get_quarterly_report(quarter_number)
    created_at = str(existing.get("created_at") or now) if existing else now
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO league_quarterly_reports (
                quarter_number, quarter_label, season_start, season_end, season_range_label,
                champion_team_id, champion_team_name, executive_summary, report_json,
                team_count, llm_used, generated_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(quarter_number) DO UPDATE SET
                quarter_label=excluded.quarter_label, season_start=excluded.season_start,
                season_end=excluded.season_end, season_range_label=excluded.season_range_label,
                champion_team_id=excluded.champion_team_id, champion_team_name=excluded.champion_team_name,
                executive_summary=excluded.executive_summary, report_json=excluded.report_json,
                team_count=excluded.team_count, llm_used=excluded.llm_used,
                generated_at=excluded.generated_at, updated_at=excluded.updated_at
            """,
            (
                quarter_number, str(payload.get("quarter_label", "")), int(payload.get("season_start") or 0),
                int(payload.get("season_end") or 0), str(payload.get("season_range_label", "")),
                str(payload.get("champion_team_id", "")), str(payload.get("champion_team_name", "")),
                str(payload.get("executive_summary", "")), str(payload.get("report_json", "{}")),
                int(payload.get("team_count") or 0), int(payload.get("llm_used") or 0),
                str(payload.get("generated_at") or now), created_at, now,
            ),
        )
        conn.commit()
    return get_quarterly_report(quarter_number) or {}


def _quarter_metric_percentiles(values: dict[str, float], direction: int) -> dict[str, float]:
    """Peer-relative percentile (spec): rank 1 of n = 100, rank n of n = (1/n)*100.
    Ties share the average rank."""
    n = len(values)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(values)): 100.0}
    ordered = sorted(values.items(), key=lambda kv: kv[1] * direction, reverse=True)  # best first
    percentiles: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = sum(range(i + 1, j + 2)) / (j - i + 1)  # 1-based ranks; ties averaged
        pct = round((n - avg_rank + 1) / n * 100.0, 2)
        for k in range(i, j + 1):
            percentiles[ordered[k][0]] = pct
        i = j + 1
    return percentiles


def _quarter_season_scores(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For one season's official reviews: each team's mean metric-percentile + breakdown."""
    teams = {str(r.get("team_id")): r for r in reviews if r.get("team_id")}
    out: dict[str, dict[str, Any]] = {
        tid: {"metrics": {}, "percentiles": {}, "team_name": str(r.get("team_name") or tid)}
        for tid, r in teams.items()
    }
    for key, direction, _label in QUARTER_METRICS:
        values = {tid: parse_float(r.get(key)) for tid, r in teams.items()}
        pcts = _quarter_metric_percentiles(values, direction)
        for tid in teams:
            out[tid]["metrics"][key] = round(values[tid], 4)
            out[tid]["percentiles"][key] = pcts.get(tid, 0.0)
    for tid in teams:
        pcts = out[tid]["percentiles"]
        out[tid]["season_score"] = round(sum(pcts.values()) / len(pcts), 2) if pcts else 0.0
    return out


def _quarterly_fallback_language(quarter_label: str, season_range_label: str, teams: list[dict[str, Any]]) -> tuple[dict[str, str], str]:
    blurbs: dict[str, str] = {}
    for t in teams:
        place = "leads the quarter and is now eligible for commissioner review" if t["rank"] == 1 else f"ranks #{t['rank']} of {len(teams)}"
        cov = "" if t["full_quarter"] else f" (partial evidence: {t['seasons_covered']}/{QUARTER_SEASON_SPAN} seasons)"
        blurbs[t["team_id"]] = f"{t['team_name']} {place} with a {t['quarterly_score']:.1f} peer-relative quarterly score{cov}."
    if teams:
        champ = teams[0]
        exec_summary = (
            f"{quarter_label} ({season_range_label}) is closed. {champ['team_name']} is the Quarterly Champion with a "
            f"{champ['quarterly_score']:.1f} peer-relative score across the three completed seasons (recency-weighted "
            f"20/30/50). It is now eligible for commissioner capital-review — eligibility, not deployment."
        )
    else:
        exec_summary = f"{quarter_label} ({season_range_label}) produced no eligible major-league evidence."
    return blurbs, exec_summary


def generate_quarterly_report(quarter_number: int, force: bool = False) -> dict[str, Any] | None:
    """Build + archive the Quarterly Champion report for a CLOSED quarter. Returns None
    (no-op) when the quarter's three seasons aren't all processed with evidence yet.
    Never silently rewrites an existing report unless force=True (explicit admin action)."""
    existing = get_quarterly_report(quarter_number)
    if existing and not force:
        return existing
    s_start, s_end = quarter_season_range(quarter_number)
    official = official_team_directory()
    if not official:
        return None
    per_season: list[dict[str, Any]] = []
    for pos, season_number in enumerate(range(s_start, s_end + 1)):
        season = league_season_for_number(season_number)
        reviews = [
            r for r in list_league_team_season_reviews(season_number=season_number, limit=200)
            if str(r.get("team_id")) in official
        ]
        if not reviews:
            return None  # quarter not fully closed / evidence not yet generated
        per_season.append({
            "season_number": season_number, "season_label": season.get("season_label", f"Season {season_number}"),
            "weight": QUARTER_RECENCY_WEIGHTS[pos], "scores": _quarter_season_scores(reviews),
        })

    # Recency-weighted aggregate per team (renormalized over the seasons it appears in).
    team_rows: dict[str, dict[str, Any]] = {}
    for entry in per_season:
        for tid, sc in entry["scores"].items():
            row = team_rows.setdefault(tid, {"team_id": tid, "team_name": sc["team_name"], "contributions": [], "_w": 0.0, "_wx": 0.0})
            row["team_name"] = sc["team_name"] or row["team_name"]
            row["contributions"].append({
                "season_number": entry["season_number"], "season_label": entry["season_label"],
                "weight": entry["weight"], "season_score": sc["season_score"],
                "percentiles": sc["percentiles"], "metrics": sc["metrics"],
            })
            row["_wx"] += entry["weight"] * sc["season_score"]
            row["_w"] += entry["weight"]
    teams: list[dict[str, Any]] = []
    span = s_end - s_start + 1
    for tid, row in team_rows.items():
        teams.append({
            "team_id": tid, "team_name": row["team_name"], "contributions": row["contributions"],
            "quarterly_score": round(row["_wx"] / row["_w"], 2) if row["_w"] else 0.0,
            "seasons_covered": len(row["contributions"]), "full_quarter": len(row["contributions"]) == span,
        })
    teams.sort(key=lambda t: (t["quarterly_score"], t["team_name"]), reverse=True)
    for idx, t in enumerate(teams, start=1):
        t["rank"] = idx
    champion = teams[0] if teams else None
    season_range_label = f"{per_season[0]['season_label']} – {per_season[-1]['season_label']}"
    quarter_label = f"Quarter {quarter_number}"

    blurbs, exec_summary = _quarterly_fallback_language(quarter_label, season_range_label, teams)
    llm_used = 0
    if get_setting("ollama_api_key", "") and teams:
        try:
            payload = {
                "quarter": quarter_label, "season_range": season_range_label,
                "recency_weights": {f"season_{i+1}": w for i, w in enumerate(QUARTER_RECENCY_WEIGHTS)},
                "ranking": [
                    {"rank": t["rank"], "team": t["team_name"], "quarterly_score": t["quarterly_score"],
                     "season_scores": [{"season": c["season_label"], "score": c["season_score"]} for c in t["contributions"]],
                     "full_quarter": t["full_quarter"]}
                    for t in teams
                ],
                "note": "These ranks and scores are precomputed and authoritative. Do not invent or recompute any number.",
            }
            content = ollama_chat([
                {"role": "system", "content": (
                    "You are the ATL commissioner's analyst writing a Quarterly Champion report — a capital-eligibility "
                    "review, like a parole hearing, not a hype piece. Winning the quarter does NOT deploy real money; it "
                    "only makes the top team eligible for later commissioner review. Return strict JSON with keys "
                    "executive_summary (one tight paragraph grounded only in the supplied ranking) and blurbs (object "
                    "mapping each team name to one concise, evidence-based sentence). Do not invent numbers or write biographies."
                )},
                {"role": "user", "content": json.dumps(payload)},
            ])
            parsed = parse_json_block(content)
            if parsed.get("executive_summary"):
                exec_summary = str(parsed["executive_summary"]).strip()
            raw_blurbs = parsed.get("blurbs") or {}
            if isinstance(raw_blurbs, dict):
                for t in teams:
                    val = raw_blurbs.get(t["team_name"])
                    if val:
                        blurbs[t["team_id"]] = str(val).strip()
            llm_used = 1
        except Exception as exc:  # noqa: BLE001
            log_maintenance("quarterly", "warning", f"Quarterly LLM call failed, using deterministic text: {exc}")
    for t in teams:
        t["blurb"] = blurbs.get(t["team_id"], "")

    report = {
        "quarter_number": quarter_number, "quarter_label": quarter_label,
        "season_start": s_start, "season_end": s_end, "season_range_label": season_range_label,
        "recency_weights": QUARTER_RECENCY_WEIGHTS,
        "metrics": [{"key": k, "label": l, "direction": d} for k, d, l in QUARTER_METRICS],
        "generated_at": iso_now(), "teams": teams,
        "champion": {"team_id": champion["team_id"], "team_name": champion["team_name"], "quarterly_score": champion["quarterly_score"]} if champion else {},
        "executive_summary": exec_summary,
    }
    stored = upsert_quarterly_report({
        "quarter_number": quarter_number, "quarter_label": quarter_label,
        "season_start": s_start, "season_end": s_end, "season_range_label": season_range_label,
        "champion_team_id": champion["team_id"] if champion else "",
        "champion_team_name": champion["team_name"] if champion else "",
        "executive_summary": exec_summary, "report_json": json.dumps(report, indent=2),
        "team_count": len(teams), "llm_used": llm_used, "generated_at": report["generated_at"],
    })
    log_maintenance("quarterly", "success",
                    f"Generated {quarter_label} ({season_range_label}); champion: {champion['team_name'] if champion else 'n/a'}, llm_used={llm_used}.")
    return stored


def maybe_generate_quarterly_reports() -> None:
    """Generate any quarterly report whose final season has fully ended and isn't archived yet."""
    current = current_league_season()
    current_num = int(current.get("season_number") or 1)
    for q in range(1, quarter_for_season(current_num) + 1):
        _, s_end = quarter_season_range(q)
        if league_season_for_number(s_end).get("status") != "completed":
            continue  # quarter not closed yet
        if get_quarterly_report(q):
            continue
        generate_quarterly_report(q)


def next_quarterly_review_timing() -> dict[str, Any]:
    """When the next (not-yet-archived) quarter closes — for the explainer, NOT a live race."""
    reports = list_quarterly_reports(limit=1)
    next_q = (int(reports[0]["quarter_number"]) + 1) if reports else 1
    s_start, s_end = quarter_season_range(next_q)
    season = league_season_for_number(s_end)
    return {
        "quarter_number": next_q, "quarter_label": f"Quarter {next_q}",
        "season_start": s_start, "season_end": s_end,
        "season_range_label": f"Season {s_start} – Season {s_end}",
        "closes_at": season.get("ended_at"), "closed": season.get("status") == "completed",
    }


def quarterly_office_context() -> dict[str, Any]:
    """Page context — ONLY archived reports + the explainer. No interim/projected standings."""
    reports = list_quarterly_reports(limit=40)
    return {
        "reports": reports,
        "has_reports": bool(reports),
        "next_review": next_quarterly_review_timing(),
        "season_span": QUARTER_SEASON_SPAN,
        "recency_weights": QUARTER_RECENCY_WEIGHTS,
        "metric_labels": [l for _k, _d, l in QUARTER_METRICS],
    }


# ---------------------------------------------------------------------------
# Strategy Trophy Shelves — permanent, append-only career achievements. Each award
# has a DETERMINISTIC rule from metrics ATL already tracks (no vibes), recognizes a
# genuinely different dimension of excellence (not "2nd-best renamed"), and uses an
# emoji set deliberately disjoint from Chronicle's day-classification emojis.
# Granted once per (award_type, season); never overwritten.
# ---------------------------------------------------------------------------

STRATEGY_AWARD_CATALOG: dict[str, dict[str, str]] = {
    # Major-league season awards (from league_team_season_reviews).
    "champion":      {"title": "Champion",      "emoji": "🏅", "dimension": "winning",             "scope": "major"},
    "surgeon":       {"title": "Surgeon",       "emoji": "🎯", "dimension": "per-trade precision",  "scope": "major"},
    "iron_bot":      {"title": "Iron Bot",      "emoji": "⚙️", "dimension": "durability / uptime",  "scope": "major"},
    "gravity_well":  {"title": "Gravity Well",  "emoji": "🪐", "dimension": "throughput",           "scope": "major"},
    "steady_hand":   {"title": "Steady Hand",   "emoji": "🧊", "dimension": "capital preservation", "scope": "major"},
    "dragon":        {"title": "Dragon",        "emoji": "🐉", "dimension": "dominance",            "scope": "major"},
    "survivor":      {"title": "Survivor",      "emoji": "🪨", "dimension": "resilience",           "scope": "major"},
    "wildcard":      {"title": "Wildcard",      "emoji": "🃏", "dimension": "specialist",           "scope": "major"},
    # Dev & research wing (from dev_shift_episodes, ml_lineage, ml_traits).
    "mad_scientist": {"title": "Mad Scientist", "emoji": "⚗️", "dimension": "experimentation",      "scope": "dev"},
    "patriarch":     {"title": "Patriarch",     "emoji": "🧬", "dimension": "influence / legacy",   "scope": "dev"},
    "professor":     {"title": "Professor",     "emoji": "📚", "dimension": "knowledge",            "scope": "dev"},
    "marathoner":    {"title": "Marathoner",    "emoji": "🐢", "dimension": "dev longevity",        "scope": "dev"},
    # Universal.
    "rookie":        {"title": "Rookie",        "emoji": "🍼", "dimension": "debut",                "scope": "both"},
}


def grant_strategy_award(award_type: str, season_number: int, season_label: str, recipient_kind: str,
                         recipient_id: str, recipient_name: str, reason: str, metric_value: float = 0.0) -> None:
    """Permanently record an award. INSERT OR IGNORE on (award_type:season) keeps the first
    winner forever — re-evaluation never overwrites history."""
    meta = STRATEGY_AWARD_CATALOG.get(award_type)
    if not meta or not recipient_id:
        return
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO strategy_awards (
                award_key, award_type, award_title, emoji, dimension, season_number, season_label,
                recipient_kind, recipient_id, recipient_name, reason, metric_value, awarded_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"{award_type}:{season_number}", award_type, meta["title"], meta["emoji"], meta["dimension"],
             int(season_number), season_label, recipient_kind, str(recipient_id), recipient_name,
             reason, float(metric_value or 0.0), now, now),
        )
        conn.commit()


def strategy_awards_for(recipient_kind: str, recipient_id: str, limit: int = 200) -> list[dict[str, Any]]:
    if not recipient_id:
        return []
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM strategy_awards WHERE recipient_kind = ? AND recipient_id = ? "
            "ORDER BY season_number ASC, id ASC LIMIT ?",
            (recipient_kind, str(recipient_id), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def team_has_prior_season_review(team_id: str, season_number: int) -> bool:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT 1 FROM league_team_season_reviews WHERE team_id = ? AND season_number < ? LIMIT 1",
            (team_id, int(season_number)),
        ).fetchone()
    return bool(row)


def evaluate_major_league_season_awards(season: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    season_number = int(season.get("season_number") or 0)
    season_label = str(season.get("season_label") or f"Season {season_number}")
    official = official_team_directory()
    rows = [r for r in reviews if str(r.get("team_id")) in official]
    if not rows:
        return

    def grant(award_type: str, row: dict[str, Any], reason: str, value: float) -> None:
        grant_strategy_award(award_type, season_number, season_label, "team",
                             str(row.get("team_id")), str(row.get("team_name") or row.get("team_id")), reason, value)

    champ = max(rows, key=lambda r: parse_float(r.get("total_pnl")))
    grant("champion", champ, f"Finished #1 with {parse_float(champ.get('total_pnl')):+.2f} total P&L.", parse_float(champ.get("total_pnl")))

    surg_pool = [r for r in rows if int(r.get("closed_trades") or 0) >= 3]
    if surg_pool:
        surg = max(surg_pool, key=lambda r: parse_float(r.get("avg_roi")))
        grant("surgeon", surg, f"{parse_float(surg.get('avg_roi')):.2f}% average ROI across {int(surg.get('closed_trades') or 0)} trades.", parse_float(surg.get("avg_roi")))

    iron = max(rows, key=lambda r: (parse_float(r.get("runtime_hours")), parse_float(r.get("heartbeat_ratio"))))
    if parse_float(iron.get("runtime_hours")) > 0:
        grant("iron_bot", iron, f"{parse_float(iron.get('runtime_hours')):.1f} runtime hours at {parse_float(iron.get('heartbeat_ratio')):.0f}% heartbeat.", parse_float(iron.get("runtime_hours")))

    gwell = max(rows, key=lambda r: int(r.get("closed_trades") or 0))
    if int(gwell.get("closed_trades") or 0) > 0:
        grant("gravity_well", gwell, f"Pulled in {int(gwell.get('closed_trades') or 0)} closed trades — the league's highest throughput.", int(gwell.get("closed_trades") or 0))

    steady_pool = [r for r in rows if parse_float(r.get("total_pnl")) >= 0]
    if steady_pool:
        steady = min(steady_pool, key=lambda r: parse_float(r.get("max_drawdown")))
        grant("steady_hand", steady, f"Held max drawdown to {parse_float(steady.get('max_drawdown')):.1f}% while finishing profitable.", parse_float(steady.get("max_drawdown")))

    by_pnl = sorted(rows, key=lambda r: parse_float(r.get("total_pnl")), reverse=True)
    if len(by_pnl) >= 2:
        top, second = parse_float(by_pnl[0].get("total_pnl")), parse_float(by_pnl[1].get("total_pnl"))
        if top > 0 and (top - second) >= 0.25 * (abs(second) + 1.0):
            grant("dragon", by_pnl[0], f"Dominated the season — beat the runner-up by {top - second:+.2f} P&L.", top - second)

    surv_pool = [r for r in rows if parse_float(r.get("total_pnl")) >= 0 and parse_float(r.get("max_drawdown")) >= 10.0]
    if surv_pool:
        surv = max(surv_pool, key=lambda r: parse_float(r.get("max_drawdown")))
        grant("survivor", surv, f"Clawed back to {parse_float(surv.get('total_pnl')):+.2f} after a {parse_float(surv.get('max_drawdown')):.1f}% drawdown.", parse_float(surv.get("max_drawdown")))

    # Wildcard — top-quartile in >=1 metric AND bottom-quartile in >=1 (the biggest specialist).
    metric_keys = [("total_pnl", 1), ("avg_roi", 1), ("win_rate", 1), ("closed_trades", 1), ("runtime_hours", 1), ("max_drawdown", -1)]
    pcts_by_team: dict[str, list[float]] = {str(r.get("team_id")): [] for r in rows}
    for key, direction in metric_keys:
        values = {str(r.get("team_id")): parse_float(r.get(key)) for r in rows}
        for tid, v in _quarter_metric_percentiles(values, direction).items():
            pcts_by_team[tid].append(v)
    wild_row, wild_disp = None, -1.0
    for r in rows:
        ps = pcts_by_team.get(str(r.get("team_id"))) or []
        if ps and max(ps) >= 75 and min(ps) <= 25 and (max(ps) - min(ps)) > wild_disp:
            wild_disp, wild_row = max(ps) - min(ps), r
    if wild_row is not None:
        grant("wildcard", wild_row, "A true specialist — elite in one dimension, deliberately weak in another.", wild_disp)

    # Rookie — first competing season, finishing top-half by total P&L.
    ranked = sorted(rows, key=lambda r: parse_float(r.get("total_pnl")), reverse=True)
    top_half = ranked[: max(1, (len(ranked) + 1) // 2)]
    for r in top_half:
        if not team_has_prior_season_review(str(r.get("team_id")), season_number):
            grant("rookie", r, f"A standout debut — finished top-half at {parse_float(r.get('total_pnl')):+.2f} P&L.", parse_float(r.get("total_pnl")))
            break


def evaluate_dev_season_awards(season: dict[str, Any]) -> None:
    season_number = int(season.get("season_number") or 0)
    season_label = str(season.get("season_label") or f"Season {season_number}")
    start = normalize_utc(resolve_optional_datetime(str(season.get("started_at") or "")))
    end = normalize_utc(resolve_optional_datetime(str(season.get("ended_at") or "")))
    if not start or not end:
        return
    name_by_slug = {str(c.get("slug")): str(c.get("name") or c.get("slug")) for c in development_candidate_rows() if c.get("slug")}
    dev_slugs = set(name_by_slug)
    if not dev_slugs:
        return

    def in_window(ts: Any) -> bool:
        d = normalize_utc(resolve_optional_datetime(str(ts or "")))
        return d is not None and start <= d <= end

    def grant(award_type: str, slug: str, reason: str, value: float) -> None:
        grant_strategy_award(award_type, season_number, season_label, "dev", slug, name_by_slug.get(slug, slug), reason, value)

    versions: dict[str, set] = {}
    shifts: dict[str, int] = {}
    first_episode: dict[str, datetime] = {}
    with closing(get_db()) as conn:
        for r in conn.execute("SELECT slug, strategy_version, session_started_at FROM dev_shift_episodes ORDER BY session_started_at ASC"):
            slug = str(r["slug"])
            if slug not in dev_slugs:
                continue
            d = normalize_utc(resolve_optional_datetime(str(r["session_started_at"] or "")))
            if d and slug not in first_episode:
                first_episode[slug] = d
            if in_window(r["session_started_at"]):
                versions.setdefault(slug, set()).add(str(r["strategy_version"] or ""))
                shifts[slug] = shifts.get(slug, 0) + 1

    if versions:
        slug, vers = max(versions.items(), key=lambda kv: len(kv[1]))
        if len(vers) >= 2:
            grant("mad_scientist", slug, f"Ran {len(vers)} distinct strategy versions this season — relentless experimentation.", len(vers))
    if shifts:
        slug, n = max(shifts.items(), key=lambda kv: kv[1])
        if n >= 2:
            grant("marathoner", slug, f"Completed {n} shifts this season without being archived.", n)

    descendants: dict[str, int] = {}
    traits: dict[str, int] = {}
    with closing(get_db()) as conn:
        for r in conn.execute("SELECT parent_slug, created_at FROM ml_lineage"):
            if str(r["parent_slug"]) in dev_slugs and in_window(r["created_at"]):
                descendants[str(r["parent_slug"])] = descendants.get(str(r["parent_slug"]), 0) + 1
        for r in conn.execute("SELECT strategy_slug, first_observed_at FROM ml_traits"):
            if str(r["strategy_slug"]) in dev_slugs and in_window(r["first_observed_at"]):
                traits[str(r["strategy_slug"])] = traits.get(str(r["strategy_slug"]), 0) + 1
    if descendants:
        slug, n = max(descendants.items(), key=lambda kv: kv[1])
        grant("patriarch", slug, f"Spawned {n} descendant{'s' if n != 1 else ''} into the ecosystem this season.", n)
    if traits:
        slug, n = max(traits.items(), key=lambda kv: kv[1])
        if n >= 2:
            grant("professor", slug, f"Contributed {n} newly-characterized traits to the league's knowledge.", n)

    # Dev Rookie — debuted this season (first-ever episode in window) with the most shifts.
    # rookie:<season> is single-winner; a major-league rookie (rarer) takes precedence if present.
    debutants = {s: shifts.get(s, 0) for s, d in first_episode.items() if start <= d <= end and s in dev_slugs}
    if debutants:
        slug, n = max(debutants.items(), key=lambda kv: kv[1])
        if n >= 1:
            grant("rookie", slug, f"Debuted this season and completed {n} shift{'s' if n != 1 else ''}.", n)


def evaluate_strategy_awards(season: dict[str, Any]) -> None:
    """Grant all season awards for a fully-processed season (major first, then dev)."""
    season_number = int(season.get("season_number") or 0)
    reviews = list_league_team_season_reviews(season_number=season_number, limit=200)
    try:
        evaluate_major_league_season_awards(season, reviews)
    except Exception as exc:  # noqa: BLE001
        log_maintenance("awards", "warning", f"Major-league award eval failed for season {season_number}: {exc}")
    try:
        evaluate_dev_season_awards(season)
    except Exception as exc:  # noqa: BLE001
        log_maintenance("awards", "warning", f"Dev award eval failed for season {season_number}: {exc}")


def update_league_team_season_review_approval(review_key: str, approval_status: str, approval_notes: str = "", approved_action: str = "") -> dict[str, Any] | None:
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            UPDATE league_team_season_reviews
            SET approval_status = ?, approval_notes = ?, approved_action = ?, updated_at = ?
            WHERE review_key = ?
            """,
            (approval_status, approval_notes, approved_action, now, review_key),
        )
        conn.commit()
    return get_league_team_season_review(review_key)


def apply_league_season_draft_action(draft_id: int, approval_status: str, approval_notes: str = "") -> dict[str, Any] | None:
    draft_row = get_league_season_draft_recommendation(draft_id)
    if not draft_row:
        return None
    if approval_status == "approved":
        approved_count = sum(1 for row in list_league_season_draft_recommendations(int(draft_row["season_number"]), limit=20) if row.get("approval_status") == "approved")
        if approved_count >= SEASON_DRAFT_SLOT_LIMIT and draft_row.get("approval_status") != "approved":
            raise HTTPException(status_code=400, detail=f"Only {SEASON_DRAFT_SLOT_LIMIT} candidates can be drafted each season.")
        candidate_id = int(draft_row.get("candidate_id") or 0)
        candidate = get_development_candidate(candidate_id)
        if candidate:
            update_development_candidate(
                candidate_id,
                lifecycle_state="drafted",
                tier="drafted",
                shift_code="",
                override_mode="paused",
                runtime_status="paused",
                status_detail=f"Drafted into {draft_row.get('season_label', 'the major league cycle')}.",
            )
            development_runtime_event(candidate_id, "promotion", "Drafted into the big-league season cycle.", draft_row.get("season_label", ""))
    with closing(get_db()) as conn:
        conn.execute(
            """
            UPDATE league_season_draft_recommendations
            SET approval_status = ?, approval_notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (approval_status, approval_notes, iso_now(), draft_id),
        )
        conn.commit()
    return get_league_season_draft_recommendation(draft_id)


def extend_version_registry_with_development(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {str(record.get("path") or ""): dict(record) for record in records}
    for candidate in development_candidate_rows():
        strategy_path = resolve_path(candidate.get("strategy_path"))
        if not strategy_path or not strategy_path.exists():
            continue
        key = str(strategy_path)
        if key in by_path:
            teams = by_path[key].setdefault("teams", [])
            candidate_label = f"DEV: {candidate.get('name', 'Candidate')}"
            if candidate_label not in teams:
                teams.append(candidate_label)
            continue
        by_path[key] = {
            "filename": strategy_path.name,
            "strategy_class": strategy_path.stem,
            "git_hash": None,
            "file_hash": sha256_file(strategy_path)[:16],
            "last_modified": datetime.fromtimestamp(strategy_path.stat().st_mtime, UTC).isoformat(),
            "notes": "Development strategy tracked through generated file checksum and archive snapshots.",
            "path": str(strategy_path),
            "teams": [f"DEV: {candidate.get('name', 'Candidate')}"] ,
        }
    result = list(by_path.values())
    result.sort(key=lambda row: str(row.get("filename") or "").lower())
    return result


def season_office_context() -> dict[str, Any]:
    current_season = current_league_season()
    current_preview_rows = [preview_official_team_season_review(team, current_season) for team in list_instances()]
    current_preview_rows.sort(key=lambda row: parse_float(row.get("overall_score")), reverse=True)
    stored_seasons = list_league_seasons(limit=6)
    stored_reviews = list_league_team_season_reviews(limit=24)
    stored_drafts = list_league_season_draft_recommendations(limit=16)
    pending_reviews = [row for row in stored_reviews if row.get("approval_status") == "pending"]
    pending_drafts = [row for row in stored_drafts if row.get("approval_status") == "pending"]
    return {
        "current_season": current_season,
        "current_preview_rows": current_preview_rows,
        "current_awards_preview": build_season_awards(current_preview_rows),
        "stored_seasons": stored_seasons,
        "stored_reviews": stored_reviews,
        "stored_drafts": stored_drafts,
        "pending_reviews": pending_reviews,
        "pending_drafts": pending_drafts,
        "draft_preview_rows": preview_season_draft_recommendations(current_season),
    }


def parse_backtest_archive() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not BACKTEST_DIR.exists():
        return results
    for meta_path in sorted(BACKTEST_DIR.glob("*.meta.json"), reverse=True):
        meta = load_json(meta_path, {})
        for strategy_name, payload in meta.items():
            base_name = meta_path.name.replace(".meta.json", "")
            zip_path = meta_path.with_suffix("").with_suffix(".zip")
            summary = {
                "strategy": strategy_name,
                "version_hash": payload.get("run_id"),
                "timeframe": payload.get("timeframe"),
                "date_range": f"{payload.get('backtest_start_ts')} -> {payload.get('backtest_end_ts')}",
                "notes": "Metadata discovered from Freqtrade backtest archive.",
                "source": meta_path.name,
                "pair_list": "",
                "results_summary": "",
            }
            if zip_path.exists():
                try:
                    with ZipFile(zip_path) as archive:
                        for member in archive.namelist():
                            if member.endswith(".json") and "meta" not in member:
                                with archive.open(member) as handle:
                                    inner = json.load(handle)
                                strategy_block = inner.get("strategy", {}).get(strategy_name, {})
                                summary["pair_list"] = ", ".join((strategy_block.get("pairlist") or [])[:8])
                                summary["results_summary"] = (
                                    f"Trades: {strategy_block.get('total_trades', 'n/a')}, "
                                    f"Profit total: {strategy_block.get('profit_total_abs', 'n/a')}, "
                                    f"Winrate: {strategy_block.get('winrate', 'n/a')}"
                                )
                                break
                except Exception:  # noqa: BLE001
                    pass
            results.append(summary)
    return results[:20]


def list_posts(category: str | None = None) -> list[sqlite3.Row]:
    with closing(get_db()) as conn:
        if category:
            return conn.execute(
                "SELECT * FROM timeline_posts WHERE category = ? ORDER BY created_at DESC, id DESC",
                (category,),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM timeline_posts ORDER BY created_at DESC, id DESC"
        ).fetchall()


def seed_initial_post() -> None:
    with closing(get_db()) as conn:
        count = conn.execute("SELECT COUNT(*) FROM timeline_posts").fetchone()[0]
        if count:
            return
        conn.execute(
            """
            INSERT INTO timeline_posts (
                created_at, category, title, team_tags, observation, evidence, interpretation, next_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iso_now(),
                "league",
                "League Tracker Initialized",
                "cosmo-wanda-20-pi,cosmo-wanda-50-pc,timmy-20-pc,timmy-50-pc",
                "Set up the local-first league dashboard with one official Pi incumbent and three local challengers.",
                "Configured API endpoints at 10.0.0.159:8080 and 127.0.0.1:8090/8091/8092 with local trade database bindings.",
                "The evidence stack is now structured enough to compare clean challenger runs against the incumbent without blending legacy containers.",
                "Verify Pi credentials if needed, run sync, and start posting observations as new exits and trade patterns appear.",
            ),
        )
        conn.commit()


def seed_initial_ml_post() -> None:
    with closing(get_db()) as conn:
        count = conn.execute("SELECT COUNT(*) FROM timeline_posts WHERE category = 'ml'").fetchone()[0]
        if count:
            return
        conn.execute(
            """
            INSERT INTO timeline_posts (
                created_at, category, title, team_tags, observation, evidence, interpretation, next_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iso_now(),
                "ml",
                "Dark Matter Bucket Rediscovered",
                "dark-matter-antimatter,timmy",
                "Found a compressed-price, high-internal-motion regime consistent with the original Timmy thesis.",
                "Validation bucket retained outsized forward moves and behavior distinct from broad reactive expansion setups.",
                "This remains scouting evidence, not scoreboard proof. Timmy dry-run behavior decides whether the idea earns promotion.",
                "Re-run with stricter leakage checks and compare against the live Timmy dry-run cohort.",
            ),
        )
        conn.commit()


def scout_outputs_for_row(row: dict[str, Any]) -> dict[str, str]:
    pace = parse_float(row.get("projected_total_pnl_per_24h") or 0)
    runtime_hours = float(row.get("total_runtime_hours") or 0)
    scout_grade = "C"
    if pace > 10 and runtime_hours >= 10:
        scout_grade = "A"
    elif pace > 3 and runtime_hours >= 6:
        scout_grade = "B"
    elif pace < 0 and runtime_hours >= 6:
        scout_grade = "D"
    promotion_recommendation = "needs more runtime"
    if row.get("tier_competition") == "candidate_6h" and scout_grade in {"A", "B"} and runtime_hours >= 10:
        promotion_recommendation = "promote consideration"
    elif row.get("tier_competition") == "prospect_12h" and scout_grade == "A" and runtime_hours >= 24:
        promotion_recommendation = "draft consideration"
    elif pace < 0 and runtime_hours >= 12:
        promotion_recommendation = "cut consideration"
    runtime_recommendation = "keep current runtime"
    if runtime_hours < PROJECTION_STRONG_WARNING_RUNTIME_HOURS:
        runtime_recommendation = "needs more runtime"
    elif scout_grade == "A" and row.get("tier_competition") == "candidate_6h":
        runtime_recommendation = "consider 12h runtime"
    risk_warning = "projection_unreliable" if "projection_unreliable" in row.get("sample_flags", []) else ""
    return {
        "scout_grade": scout_grade,
        "promotion_recommendation": promotion_recommendation,
        "runtime_recommendation": runtime_recommendation,
        "risk_warning": risk_warning,
        "sample_size_warning": ", ".join(row.get("sample_flags", [])),
        "watchlist_flag": "watchlist" if scout_grade in {"A", "B"} else "",
        "needs_more_runtime_flag": "needs more runtime" if runtime_recommendation == "needs more runtime" else "",
        "cut_candidate_flag": "cut candidate" if promotion_recommendation == "cut consideration" else "",
    }


def ml_latest_cycle() -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM ml_telemetry_cycles WHERE status = 'complete' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["findings"] = json.loads(data.get("top_findings_json") or "[]")
    except json.JSONDecodeError:
        data["findings"] = []
    return data


def ml_cycle_relationships(cycle_id: int, limit: int = 12) -> list[dict[str, Any]]:
    names = {s["slug"]: s["name"] for s in ml_registry_all(active_only=False)}
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM ml_relationships WHERE cycle_id = ? ORDER BY complement_score DESC, similarity_score DESC LIMIT ?",
            (cycle_id, limit),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["name_a"] = names.get(row["strategy_a"], row["strategy_a"])
        item["name_b"] = names.get(row["strategy_b"], row["strategy_b"])
        out.append(item)
    return out


def ml_latest_review() -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM ml_evolution_reviews WHERE status = 'complete' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def ml_biology_context() -> dict[str, Any]:
    """Context for the revamped ML Lab home — the Strategy Biology Department."""
    registry = ml_registry_all(active_only=False)
    traits_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trait in ml_traits_all():
        traits_by_strategy[trait["strategy_slug"]].append(trait)
    parents_by_child: dict[str, list[str]] = defaultdict(list)
    for edge in ml_lineage_all():
        parents_by_child[edge["child_slug"]].append(edge["parent_slug"])

    families = []
    for fam in ml_families_all():
        members = [s for s in registry if s["family_slug"] == fam["slug"]]
        for member in members:
            member["traits"] = traits_by_strategy.get(member["slug"], [])
            member["parents"] = parents_by_child.get(member["slug"], [])
            member["temporal"] = ml_temporal_view(member)
        families.append({**fam, "members": members})

    # Global Wind Tunnel: every active organism on one Signal Timing Spectrum.
    # Placed organisms sorted by band (start, then end); unplaced ones appended.
    timeline_entries = [
        {"slug": s["slug"], "name": s["name"], "family_slug": s["family_slug"],
         "temporal": ml_temporal_view(s)}
        for s in registry if s.get("active", 1)
    ]
    timeline_entries.sort(
        key=lambda e: (0, e["temporal"]["start"], e["temporal"]["end"])
        if e["temporal"]["placed"] else (1, 0, 0)
    )

    latest_cycle = ml_latest_cycle()
    relationships = ml_cycle_relationships(latest_cycle["id"]) if latest_cycle else []
    conviction_threshold = parse_float(get_setting("ml_descendant_conviction_threshold", "0.75"))

    return {
        "families": families,
        "strategies": registry,
        "spectrum": [{"slug": slug, "label": label} for slug, label in ML_SIGNAL_TIMING_SPECTRUM],
        "organisms_timeline": timeline_entries,
        "latest_cycle": latest_cycle,
        "relationships": relationships,
        "hypotheses": ml_descendant_hypotheses_all(order_by_conviction=True),
        "conviction_threshold": conviction_threshold,
        "latest_review": ml_latest_review(),
        "strategy_count": len(registry),
        "family_count": len(families),
        "trait_count": len(ml_traits_all()),
    }


def ml_lab_context(
    scope_filter: str = "all",
    tier_filter: str = "",
    shift_filter: str = "",
    sample_quality_filter: str = "",
    long_short_filter: str = "",
    timeframe_filter: str = "",
    universe_filter: str = "",
    runtime_bucket: str = "",
) -> dict[str, Any]:
    hypotheses = merged_ml_hypotheses()
    buckets = ml_buckets()
    models = ml_models()
    draft = merged_ml_draft_board()
    universe = all_strategy_universe(include_archived=True)
    filtered_universe = filter_strategy_universe(
        universe,
        scope_filter=scope_filter,
        tier_filter=tier_filter,
        shift_filter=shift_filter,
        long_short_filter=long_short_filter,
        timeframe_filter=timeframe_filter,
        universe_filter=universe_filter,
        runtime_bucket=runtime_bucket,
        sample_quality_filter=sample_quality_filter,
    )
    for row in filtered_universe:
        row.update(scout_outputs_for_row(row))
    active = [item for item in hypotheses if item.get("status") not in {"rejected", "archived"}]
    failed = [item for item in hypotheses if item.get("status") in {"rejected", "archived"}]
    warnings = [
        "Lookahead bias",
        "Data leakage",
        "Overfitting",
        "Reused validation sets",
        "Survivorship bias",
        "Too many repeated tests on the same data",
        "No live confirmation yet",
    ]
    pipeline_stages = [
        "Raw observation",
        "Feature bucket",
        "Backtest prototype",
        "Paper/dry run",
        "League candidate",
        "Official team",
        "Relegated/archived",
    ]
    return {
        "hypotheses": hypotheses,
        "active_hypotheses": active,
        "buckets": buckets,
        "models": models,
        "draft_board": draft,
        "failed_ideas": failed,
        "warnings": warnings,
        "pipeline_stages": pipeline_stages,
        "generated_findings": list_ml_findings(),
        "generated_questions": list_ai_research_questions("ml"),
        "strategy_universe": filtered_universe,
        "top_runtime_adjusted": ranked_rows(filtered_universe, "projected_total_pnl_per_24h", "adjusted_rank")[:12],
        "underperformers": [row for row in filtered_universe if parse_float(row.get("projected_total_pnl_per_24h") or 0) < 0][:12],
        "needs_runtime_rows": [row for row in filtered_universe if row.get("needs_more_runtime_flag")][:12],
        "scope_filter": scope_filter,
        "tier_filter": tier_filter,
        "shift_filter": shift_filter,
        "sample_quality_filter": sample_quality_filter,
        "long_short_filter": long_short_filter,
        "timeframe_filter": timeframe_filter,
        "universe_filter": universe_filter,
        "runtime_bucket": runtime_bucket,
    }


def ml_workbench_context(
    queue_status: str = "all",
    selected_queue_id: int | None = None,
    selected_run_slug: str = "",
    compare_a: str = "",
    compare_b: str = "",
) -> dict[str, Any]:
    datasets = list_ml_dataset_registry()
    labels = list_ml_label_registry()
    models = list_ml_model_registry()
    feature_sets = list_ml_feature_set_versions()
    label_versions = list_ml_label_spec_versions()
    all_queue = list_ml_experiment_queue(limit=50)
    queue = [row for row in all_queue if queue_status in {"", "all"} or row.get("status") == queue_status]
    runs = list_ml_experiment_runs(limit=12)
    bucket_candidates = list_ml_bucket_candidates(limit=12)
    validation_reports = list_ml_validation_reports(limit=12)
    promotion_recommendations = list_ml_promotion_recommendations(limit=12)
    selected_queue = None
    if selected_queue_id:
        selected_queue = next((row for row in all_queue if parse_intish(row.get("id")) == selected_queue_id), None)
        if not selected_queue:
            selected_queue = get_ml_queue_item(selected_queue_id)
    selected_run = None
    if selected_run_slug:
        selected_run = next((row for row in runs if str(row.get("run_slug", "")) == selected_run_slug), None)
        if not selected_run:
            selected_run = get_ml_experiment_run(selected_run_slug)
    selected_run_artifact = load_ml_run_artifact(str(selected_run.get("artifact_path", ""))) if selected_run else {}
    queue_status_counts = {
        "all": len(all_queue),
        "queued": sum(1 for row in all_queue if row.get("status") == "queued"),
        "running": sum(1 for row in all_queue if row.get("status") == "running"),
        "completed": sum(1 for row in all_queue if row.get("status") == "completed"),
        "failed": sum(1 for row in all_queue if row.get("status") == "failed"),
        "blocked": sum(1 for row in all_queue if row.get("status") == "blocked"),
    }
    complete_runs = sum(1 for row in models if row.get("lineage_status") == "complete")
    live_runs = sum(1 for row in models if row.get("influenced_live_strategy"))
    artifact_coverage = round((sum(row.get("artifact_exists", 0) for row in models) / len(models)) * 100) if models else 0
    universe = all_strategy_universe(include_archived=True)
    universe_lookup = {str(row.get("strategy_id") or row.get("team_id") or row.get("id")): row for row in universe}
    selected_compare_a = universe_lookup.get(compare_a)
    selected_compare_b = universe_lookup.get(compare_b)
    shift_average = None
    if selected_compare_a and selected_compare_a.get("shift_code"):
        selected_compare_a_id = str(selected_compare_a.get("strategy_id") or selected_compare_a.get("team_id") or selected_compare_a.get("id"))
        shift_peer_rows = [
            row for row in universe
            if row.get("shift_code") == selected_compare_a.get("shift_code")
            and str(row.get("strategy_id") or row.get("team_id") or row.get("id")) != selected_compare_a_id
        ]
        if shift_peer_rows:
            shift_average = {
                "projected_total_pnl_per_24h": round(sum(parse_float(row.get("projected_total_pnl_per_24h") or 0) for row in shift_peer_rows if row.get("projected_total_pnl_per_24h") is not None) / max(1, len([row for row in shift_peer_rows if row.get("projected_total_pnl_per_24h") is not None])), 4),
                "raw_total_pnl": round(sum(parse_float(row.get("raw_total_pnl")) for row in shift_peer_rows) / len(shift_peer_rows), 4),
            }
    return {
        "dataset_rows": datasets,
        "label_rows": labels,
        "model_rows": models,
        "feature_set_rows": feature_sets,
        "label_spec_rows": label_versions,
        "queue_rows": queue,
        "experiment_rows": runs,
        "bucket_candidate_rows": bucket_candidates,
        "validation_rows": validation_reports,
        "promotion_rows": promotion_recommendations,
        "queue_status": queue_status,
        "queue_status_counts": queue_status_counts,
        "selected_queue_row": selected_queue,
        "selected_queue_runs": [row for row in runs if parse_intish(row.get("queue_id")) == parse_intish(selected_queue_id or 0)],
        "selected_run_slug": selected_run_slug,
        "selected_run_row": selected_run,
        "selected_run_artifact": selected_run_artifact,
        "dataset_count": len(datasets),
        "label_count": len(labels),
        "model_count": len(models),
        "feature_set_count": len(feature_sets),
        "label_spec_count": len(label_versions),
        "queued_lead_count": sum(1 for row in queue if row.get("status") == "queued"),
        "experiment_run_count": len(runs),
        "bucket_candidate_count": len(bucket_candidates),
        "complete_runs": complete_runs,
        "live_runs": live_runs,
        "artifact_coverage": artifact_coverage,
        "strategy_universe": ranked_rows(universe, "projected_total_pnl_per_24h", "adjusted_rank"),
        "compare_a": compare_a,
        "compare_b": compare_b,
        "selected_compare_a": selected_compare_a,
        "selected_compare_b": selected_compare_b,
        "shift_average": shift_average,
    }


def list_ai_research_questions(category: str) -> list[sqlite3.Row]:
    with closing(get_db()) as conn:
        return conn.execute(
            """
            SELECT *
            FROM ai_research_questions
            WHERE category = ?
            ORDER BY id DESC
            """,
            (category,),
        ).fetchall()


def replace_ai_research_questions(category: str, items: list[dict[str, str]]) -> None:
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM ai_research_questions WHERE category = ?", (category,))
        for item in items:
            conn.execute(
                """
                INSERT INTO ai_research_questions (category, question, rationale, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    category,
                    item.get("question", ""),
                    item.get("rationale", ""),
                    item.get("status", "active"),
                    iso_now(),
                ),
            )
        conn.commit()


def upsert_research_index_entry(
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    tags: str,
    *,
    entry_type: str | None = None,
    author_type: str | None = None,
    thread_id: int | None = None,
    parent_entry_id: int | None = None,
    strategy_id: str | None = None,
    family_id: str | None = None,
    topic_tags: str | None = None,
    confidence: float | None = None,
    status: str | None = None,
    supersedes_id: int | None = None,
) -> None:
    # Phase 1: the trailing keyword-only args are structured metadata. They default to
    # None so every existing caller keeps working unchanged (old rows / un-annotated
    # writers simply leave them NULL). `created_at` is set once on insert and never
    # overwritten; `updated_at` remains last-touched.
    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO research_index_entries (
                source_type, source_key, title, content, tags, updated_at,
                entry_type, author_type, thread_id, parent_entry_id, strategy_id,
                family_id, topic_tags, confidence, status, created_at, supersedes_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                source_type=excluded.source_type,
                title=excluded.title,
                content=excluded.content,
                tags=excluded.tags,
                updated_at=excluded.updated_at,
                entry_type=excluded.entry_type,
                author_type=excluded.author_type,
                thread_id=excluded.thread_id,
                parent_entry_id=excluded.parent_entry_id,
                strategy_id=excluded.strategy_id,
                family_id=excluded.family_id,
                topic_tags=excluded.topic_tags,
                confidence=excluded.confidence,
                status=excluded.status,
                supersedes_id=excluded.supersedes_id
                -- created_at is intentionally NOT updated here: it stays first-seen.
            -- QF2: only re-stamp / rewrite when something actually changed. Otherwise a
            -- no-op upsert (e.g. the repo refresh re-touching 60+ unchanged files) would
            -- bump every row to "now" and crowd genuine research out of recency-ordered
            -- retrieval. `IS NOT` is null-safe. Metadata fields are included so a genuine
            -- metadata-only change still lands.
            WHERE excluded.content IS NOT research_index_entries.content
               OR excluded.title IS NOT research_index_entries.title
               OR excluded.tags IS NOT research_index_entries.tags
               OR excluded.source_type IS NOT research_index_entries.source_type
               OR excluded.entry_type IS NOT research_index_entries.entry_type
               OR excluded.author_type IS NOT research_index_entries.author_type
               OR excluded.status IS NOT research_index_entries.status
               OR excluded.confidence IS NOT research_index_entries.confidence
            """,
            (
                source_type, source_key, title, content, tags, now,
                entry_type, author_type, thread_id, parent_entry_id, strategy_id,
                family_id, topic_tags, confidence, status, now, supersedes_id,
            ),
        )
        conn.commit()


def recent_research_index_entries(limit: int = 8) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT source_type, source_key, title, content, tags, updated_at
            FROM research_index_entries
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


# QF1 retrieval scan budget. The keyword scorer below loads this many of the most
# recent entries and ranks them in Python. The previous hard cap of 400 silently
# dropped everything older than the newest 400 rows from retrieval, so recall
# collapsed as the archive grew (40% coverage at 1k entries, 0.4% at 100k). 8k
# comfortably covers the near-term archive; Phase 2's FTS5 index removes the ceiling
# entirely and lets this constant go away.
RESEARCH_INDEX_SCAN_BUDGET = 8000


def search_research_index(
    query: str,
    limit: int = 8,
    *,
    exclude_repo: bool = False,
    with_scores: bool = False,
) -> list[dict[str, Any]]:
    tokens = tokenize_search(query)
    if not tokens:
        return recent_research_index_entries(limit)
    where = "WHERE source_type != 'repo'" if exclude_repo else ""
    with closing(get_db()) as conn:
        rows = conn.execute(
            f"""
            SELECT source_type, source_key, title, content, tags, updated_at
            FROM research_index_entries
            {where}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (RESEARCH_INDEX_SCAN_BUDGET,),
        ).fetchall()
    scored: list[tuple[int, dict[str, Any]]] = []
    for raw_row in rows:
        row = dict(raw_row)
        title_l = str(row.get("title", "")).lower()
        tags_l = str(row.get("tags", "")).lower()
        haystack = " ".join([title_l, str(row.get("content", "")).lower(), tags_l])
        score = 0
        reasons: list[str] = []
        for token in tokens:
            if token in title_l:
                score += 4
                reasons.append(f"title:{token}")
            if token in tags_l:
                score += 2
                reasons.append(f"tag:{token}")
            if token in haystack:
                score += 1
                reasons.append(f"body:{token}")
        if score:
            if with_scores:
                # QF6: expose why this entry was retrieved so the /research page (and
                # later the debug view) can show the agent's evidence trail.
                row["_score"] = score
                row["_match_reasons"] = reasons
            scored.append((score, row))
    # Highest score first, newest first as the tiebreaker (the old code broke ties
    # toward the *oldest* entry).
    scored.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
    return [row for _, row in scored[:limit]]


def question_similarity(a: str, b: str) -> float:
    """Jaccard token overlap between two research questions, 0..1. Used to detect
    near-duplicate reseeds (QF5) so the agent can't loop on the same question."""
    tokens_a, tokens_b = set(tokenize_search(a)), set(tokenize_search(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def refresh_repo_research_index() -> None:
    index_roots = [
        BASE_DIR,
        PROJECT_DIR / "user_data" / "strategies",
        PROJECT_DIR / "user_data" / "notebooks",
    ]
    allowed_suffixes = {".py", ".json", ".md", ".txt", ".html", ".css", ".ipynb"}
    skip_parts = {"__pycache__", ".git", ".venv", "node_modules", "backtest_results", "hyperopt_results", "logs", "plot", "freqaimodels", "data"}

    repo_files: list[Path] = []
    readme_path = PROJECT_DIR / "README.md"
    if readme_path.exists():
        repo_files.append(readme_path)
    for config_path in sorted((PROJECT_DIR / "user_data").glob("config*.json")):
        repo_files.append(config_path)
    for root in index_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            relative = path.relative_to(PROJECT_DIR)
            if any(part in skip_parts for part in relative.parts[:-1]):
                continue
            repo_files.append(path)

    seen: set[str] = set()
    for path in repo_files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        relative = str(path.relative_to(PROJECT_DIR)).replace("\\", "/")
        content = read_textish_file(path, max_chars=3500)
        if not content.strip():
            continue
        tags = " ".join(tokenize_search(relative))
        upsert_research_index_entry(
            "repo", f"repo:{relative}", relative, content, tags,
            entry_type="repo", author_type="system", status="active",
        )


def list_research_threads(limit: int = 12) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM research_threads
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_research_thread(thread_id: int) -> dict[str, Any] | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT * FROM research_threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def list_research_thread_updates(thread_id: int, limit: int = 24) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM research_thread_updates
            WHERE thread_id = ?
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (thread_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def update_research_thread_state(thread_id: int, **updates: Any) -> None:
    if not updates:
        return
    fields = []
    values = []
    for key, value in updates.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(thread_id)
    with closing(get_db()) as conn:
        conn.execute(
            f"UPDATE research_threads SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()


def add_research_thread_update(
    thread_id: int,
    update_type: str,
    title: str,
    content: str,
    source: str,
    citations: list[str] | None = None,
) -> int:
    citations = citations or []
    created_at = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO research_thread_updates (thread_id, created_at, update_type, title, content, citations, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (thread_id, created_at, update_type, title, content, json.dumps(citations), source),
        )
        update_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        thread = conn.execute("SELECT question FROM research_threads WHERE id = ?", (thread_id,)).fetchone()
        conn.commit()
    question = thread["question"] if thread else f"Thread {thread_id}"
    # QF3: carry author + entry-type signal into the index. The thread_updates table
    # knows a manual_note (source="user") from agent chatter, but that signal was lost
    # at index time — so on retrieval a human note was indistinguishable from the
    # agent's own ramblings. Folding matchable markers into `tags` lets the existing
    # keyword search surface human notes and conclusions, and lets the model see them
    # as evidence/steering rather than fact. (Phase 1 promotes these to real columns.)
    author_type = "human" if source == "user" else "agent"
    markers = f"authortype_{author_type} entrytype_{update_type}"
    tags = " ".join(tokenize_search(question + " " + title + " " + " ".join(citations))) + " " + markers
    # Phase 1: deterministic entry_type from the update_type — a summary is the thread's
    # conclusion, a manual_note is human evidence, the rest are running updates.
    entry_type = {
        "summary": "conclusion",
        "manual_note": "manual_note",
        "question": "question",
        "agent_update": "update",
    }.get(update_type, "update")
    upsert_research_index_entry(
        "research-update",
        f"research-update:{update_id}",
        f"{question} :: {title}",
        content,
        tags,
        entry_type=entry_type,
        author_type=author_type,
        thread_id=thread_id,
        status="active",
    )
    return update_id


def create_research_thread(
    question: str,
    owner: str = "user",
    scope: str = "research",
    auto_reseed: bool = True,
    interval_minutes: int | None = None,
    duration_hours: int | None = None,
) -> int:
    interval = interval_minutes or int(get_setting("research_agent_interval_minutes", "30") or "30")
    duration = duration_hours or int(get_setting("research_agent_duration_hours", "12") or "12")
    now = utc_now()
    started_at = now.isoformat()
    next_run_at = (now + timedelta(minutes=interval)).isoformat()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO research_threads (
                question, scope, status, owner, interval_minutes, duration_hours, auto_reseed,
                created_at, started_at, next_run_at, completed_at, summary, latest_focus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                scope,
                "active",
                owner,
                interval,
                duration,
                int(bool(auto_reseed)),
                started_at,
                started_at,
                next_run_at,
                None,
                "",
                question,
            ),
        )
        thread_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
    add_research_thread_update(thread_id, "question", "Question launched", question, owner)
    return thread_id


def seed_research_threads() -> None:
    with closing(get_db()) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM research_threads").fetchone()[0])
    if count:
        return
    questions = load_json(QUESTIONS_PATH, [])
    opening_question = questions[0] if questions else "Which live league behaviors most deserve systematic research next?"
    create_research_thread(opening_question, owner="agent", auto_reseed=True)


def due_research_threads() -> list[dict[str, Any]]:
    now = iso_now()
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM research_threads
            WHERE status = 'active' AND next_run_at <= ?
            ORDER BY next_run_at ASC, id ASC
            """,
            (now,),
        ).fetchall()
    return [dict(row) for row in rows]


def queue_research_thread_for_ml(thread_id: int) -> int | None:
    thread = get_research_thread(thread_id)
    if not thread:
        return None
    updates = list_research_thread_updates(thread_id, limit=12)
    latest_update = updates[-1] if updates else {}
    lead_question = str(thread.get("latest_focus") or thread.get("question") or "").strip()
    rationale = str(thread.get("summary") or latest_update.get("content") or thread.get("question") or "").strip()
    title = f"Research Lead: {lead_question[:96]}" if lead_question else f"Research Thread {thread_id}"
    source_key = f"research-thread:{thread_id}:{registry_slug(lead_question or title)}"
    return create_ml_queue_item(
        "research-thread",
        source_key,
        title,
        lead_question or title,
        rationale or "Research thread nominated for ML study.",
        thread_id=thread_id,
        priority="high" if thread.get("owner") == "agent" else "normal",
    )


def list_ml_findings() -> list[sqlite3.Row]:
    with closing(get_db()) as conn:
        return conn.execute(
            "SELECT * FROM ml_findings ORDER BY updated_at DESC, id DESC"
        ).fetchall()


def replace_ml_findings(items: list[dict[str, str]], source_question: str) -> None:
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM ml_findings")
        for item in items:
            conn.execute(
                """
                INSERT INTO ml_findings (title, content, hypothesis_id, status, source_question, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("title", ""),
                    item.get("content", ""),
                    item.get("hypothesis_id", ""),
                    item.get("status", "active"),
                    source_question,
                    iso_now(),
                ),
            )
        conn.commit()


def log_maintenance(maintenance_type: str, status: str, message: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO maintenance_runs (maintenance_type, status, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (maintenance_type, status, message, iso_now()),
        )
        conn.commit()


def parse_json_block(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def scrub_sensitive_data(value: Any, key_name: str = "") -> Any:
    secret_key = bool(re.search(r"(password|passphrase|secret|token|api[_-]?key|private[_-]?key)", key_name, re.IGNORECASE))
    if isinstance(value, dict):
        return {
            key: ("[redacted]" if re.search(r"(password|passphrase|secret|token|api[_-]?key|private[_-]?key)", str(key), re.IGNORECASE) else scrub_sensitive_data(item, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_sensitive_data(item, key_name) for item in value]
    if secret_key and value not in {None, "", [], {}}:
        return "[redacted]"
    return value


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def export_sqlite_snapshot(db_path: Path, destination_dir: Path) -> list[dict[str, Any]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    table_summaries: list[dict[str, Any]] = []
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for row in tables:
            table_name = str(row["name"])
            records = [scrub_sensitive_data(dict(item)) for item in conn.execute(f'SELECT * FROM "{table_name}"').fetchall()]
            write_json_file(destination_dir / f"{table_name}.json", records)
            table_summaries.append({"table": table_name, "row_count": len(records)})
    return table_summaries


def archive_repo_local_path() -> Path:
    return resolve_path(get_setting("archive_repo_local_path", relative_project_path(DEFAULT_ARCHIVE_REPO_DIR))) or DEFAULT_ARCHIVE_REPO_DIR


def archive_snapshot_contexts() -> dict[str, Any]:
    return {
        "dashboard": scrub_sensitive_data(dashboard_context()),
        "development": scrub_sensitive_data(development_league_context()),
        "development_schedule": scrub_sensitive_data(development_schedule_context()),
        "ml_lab": scrub_sensitive_data(ml_lab_context()),
        "settings": scrub_sensitive_data(app_settings_snapshot()),
    }


def create_archive_snapshot() -> Path:
    timestamp = utc_now()
    snapshot_slug = timestamp.strftime("%Y%m%d-%H%M%S")
    snapshot_dir = ARCHIVE_ROOT_DIR / snapshot_slug
    if snapshot_dir.exists():
        snapshot_slug = f"{snapshot_slug}-{int(time.time())}"
        snapshot_dir = ARCHIVE_ROOT_DIR / snapshot_slug
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    contexts = archive_snapshot_contexts()
    table_summaries = export_sqlite_snapshot(DB_PATH, snapshot_dir / "database")
    for path in sorted(DATA_DIR.glob("*.json")):
        write_json_file(snapshot_dir / "data" / path.name, scrub_sensitive_data(load_json(path, {} if path.name.endswith('.json') else [])))
    for path in sorted(DEV_CONFIG_DIR.glob("*.json")):
        write_json_file(snapshot_dir / "development_configs" / path.name, scrub_sensitive_data(load_json(path, {})))
    if DEV_STRATEGY_DIR.exists():
        shutil.copytree(DEV_STRATEGY_DIR, snapshot_dir / "development_strategies", dirs_exist_ok=True)
    site_snapshot_dir = snapshot_dir / "site"
    site_snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_DIR / "main.py", site_snapshot_dir / "main.py")
    shutil.copytree(BASE_DIR / "templates", site_snapshot_dir / "templates", dirs_exist_ok=True)
    shutil.copytree(BASE_DIR / "static", site_snapshot_dir / "static", dirs_exist_ok=True)
    for name, payload in contexts.items():
        write_json_file(snapshot_dir / "contexts" / f"{name}.json", payload)
    write_json_file(
        snapshot_dir / "manifest.json",
        {
            "created_at": timestamp.isoformat(),
            "snapshot_slug": snapshot_slug,
            "project_root": relative_project_path(PROJECT_DIR),
            "database_tables": table_summaries,
            "data_files": sorted(path.name for path in DATA_DIR.glob("*.json")),
            "development_config_files": sorted(path.name for path in DEV_CONFIG_DIR.glob("*.json")),
        },
    )
    return snapshot_dir


def run_git_command(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=180,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or "Git command failed."
        raise RuntimeError(detail)
    return (completed.stdout or "").strip()


def ensure_archive_repo_checkout(repo_url: str, branch: str, repo_path: Path) -> None:
    repo_url = repo_url.strip()
    branch = branch.strip() or "main"
    if not repo_url:
        raise RuntimeError("Archive repo URL is empty.")
    if repo_path.exists() and not (repo_path / ".git").exists():
        if any(repo_path.iterdir()):
            raise RuntimeError(f"Archive repo path is not a Git checkout: {relative_project_path(repo_path)}")
        repo_path.rmdir()
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        run_git_command(["clone", repo_url, str(repo_path)])
    run_git_command(["fetch", "origin"], cwd=repo_path)
    remote_branch = run_git_command(["ls-remote", "--heads", "origin", branch], cwd=repo_path)
    local_branch = run_git_command(["branch", "--list", branch], cwd=repo_path)
    if remote_branch:
        if local_branch:
            run_git_command(["checkout", branch], cwd=repo_path)
        else:
            run_git_command(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo_path)
    else:
        branch_list = run_git_command(["branch", "--list"], cwd=repo_path)
        if local_branch:
            run_git_command(["checkout", branch], cwd=repo_path)
        elif not branch_list.strip():
            run_git_command(["checkout", "--orphan", branch], cwd=repo_path)
        else:
            run_git_command(["checkout", "-B", branch], cwd=repo_path)
    if remote_branch:
        run_git_command(["pull", "--ff-only", "origin", branch], cwd=repo_path)


def sync_archive_repo(snapshot_dir: Path) -> str:
    repo_path = archive_repo_local_path()
    repo_url = get_setting("archive_repo_url", "")
    branch = get_setting("archive_repo_branch", "main")
    ensure_archive_repo_checkout(repo_url, branch, repo_path)
    destination = repo_path / "snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ARCHIVE_ROOT_DIR, destination, dirs_exist_ok=True)
    run_git_command(["add", "snapshots"], cwd=repo_path)
    status = run_git_command(["status", "--short", "--", "snapshots"], cwd=repo_path)
    if not status:
        return f"No new archive changes to push after {snapshot_dir.name}."
    run_git_command(["commit", "-m", f"Archive snapshot {snapshot_dir.name}"], cwd=repo_path)
    if get_setting("archive_push_enabled", "true").lower() == "true":
        run_git_command(["push", "origin", branch], cwd=repo_path)
        return f"Snapshot {snapshot_dir.name} pushed to {branch}."
    return f"Snapshot {snapshot_dir.name} committed locally without push."


def run_archive_maintenance() -> None:
    snapshot_dir = create_archive_snapshot()
    set_setting("archive_last_run", iso_now())
    set_setting("archive_last_snapshot_path", relative_project_path(snapshot_dir))
    try:
        message = sync_archive_repo(snapshot_dir)
        log_maintenance("archive", "success", message)
    except Exception as exc:  # noqa: BLE001
        log_maintenance("archive", "warning", f"Created local snapshot {relative_project_path(snapshot_dir)} but repo sync failed: {exc}")


def ollama_chat(
    messages: list[dict[str, str]],
    preferred_models: list[str] | None = None,
    timeout_seconds_override: float | None = None,
    retry_count_override: int | None = None,
    fallback_model_override: str | None = None,
    include_default_fallback: bool = True,
    attempt_callback: Any | None = None,
) -> str:
    api_key = get_setting("ollama_api_key", "")
    model = get_setting("ollama_model", "gpt-oss:120b")
    fallback_model = get_setting("ollama_fallback_model", "").strip()
    timeout_seconds = max(30.0, timeout_seconds_override if timeout_seconds_override is not None else parse_float(get_setting("ollama_timeout_seconds", "120")))
    retry_count = max(0, retry_count_override if retry_count_override is not None else int(parse_float(get_setting("ollama_retry_count", "2"))))
    if not api_key:
        raise RuntimeError("Ollama API key is not configured.")
    candidate_models: list[str] = []
    for candidate_model in preferred_models or []:
        if candidate_model and candidate_model not in candidate_models:
            candidate_models.append(candidate_model)
    if model not in candidate_models:
        candidate_models.append(model)
    override_fallback = (fallback_model_override or "").strip()
    if override_fallback and override_fallback not in candidate_models:
        candidate_models.append(override_fallback)
    if include_default_fallback and fallback_model and fallback_model != model and fallback_model not in candidate_models:
        candidate_models.append(fallback_model)
    timeout = httpx.Timeout(timeout_seconds, connect=min(20.0, timeout_seconds))
    failures: list[str] = []
    total_attempts = max(1, len(candidate_models) * (retry_count + 1))
    attempt_number = 0
    for candidate_model in candidate_models:
        for attempt in range(retry_count + 1):
            attempt_number += 1
            if attempt_callback:
                attempt_callback(candidate_model, attempt + 1, retry_count + 1, attempt_number, total_attempts, "")
            try:
                response = httpx.post(
                    "https://ollama.com/api/chat",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": candidate_model, "messages": messages, "stream": False},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if candidate_model != model:
                    log_maintenance("ai", "fallback", f"Ollama call succeeded with fallback model {candidate_model}.")
                if attempt_callback:
                    attempt_callback(candidate_model, attempt + 1, retry_count + 1, attempt_number, total_attempts, "success")
                return payload.get("message", {}).get("content", "")
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                failures.append(f"{candidate_model} attempt {attempt + 1}: {exc}")
                if attempt_callback:
                    attempt_callback(candidate_model, attempt + 1, retry_count + 1, attempt_number, total_attempts, str(exc))
    raise RuntimeError("Ollama chat failed after retries: " + " | ".join(failures[-4:]))


def run_league_maintenance() -> None:
    standings = standings_rows()
    manual_questions = load_json(QUESTIONS_PATH, [])
    prompt = {
        "manual_questions": manual_questions,
        "standings": standings,
        "power_rankings": compute_power_rankings(),
        "note": "You are maintaining a local algo trading league site. Do not invent live performance. Generate only lightweight site-maintenance outputs."
    }
    content = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "Return strict JSON with keys league_overview, research_questions, and power_ranking_overrides. "
                    "league_overview must be a concise site overview paragraph. "
                    "research_questions must be an array of up to 5 objects with question, rationale, status. "
                    "power_ranking_overrides must be an array of up to 4 objects with team_id, trust_score, trust_note, data_quality, data_quality_note, interesting_discoveries. "
                    "These are scouting or monitoring questions only, not timeline posts. "
                    "Replace previous AI-managed league outputs instead of appending to them."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ]
    )
    payload = parse_json_block(content)
    replace_generated_content("league_overview", payload.get("league_overview", ""))
    replace_ai_research_questions("league", payload.get("research_questions", []))
    replace_generated_json("power_ranking_overrides", payload.get("power_ranking_overrides", []))
    set_setting("league_maintenance_last_run", iso_now())
    log_maintenance("league", "success", "League AI-managed overview, questions, and ranking overlays replaced.")


# ===========================================================================
# ML Lab — deterministic measurement core (Phase 2)
# ===========================================================================


def _ml_parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _ml_duration_minutes(record: dict[str, Any]) -> float:
    raw = record.get("trade_duration_minutes")
    if raw not in (None, ""):
        return parse_float(raw)
    opened = _ml_parse_dt(record.get("open_date"))
    closed = _ml_parse_dt(record.get("close_date"))
    if opened and closed:
        return max(0.0, (closed - opened).total_seconds() / 60.0)
    return 0.0


def _ml_read_sqlite_trades(db_path: Path) -> list[dict[str, Any]]:
    """Read a freqtrade tradesv3 sqlite, selecting only columns that exist."""
    if not db_path or not db_path.exists():
        return []
    wanted = [
        "is_open", "is_short", "leverage", "pair", "open_date", "close_date",
        "close_profit", "close_profit_abs", "realized_profit", "exit_reason", "enter_tag",
    ]
    try:
        with sqlite3.connect(db_path) as source:
            source.row_factory = sqlite3.Row
            available = {row[1] for row in source.execute("PRAGMA table_info(trades)").fetchall()}
            cols = [c for c in wanted if c in available]
            if "is_open" not in cols:
                return []
            rows = source.execute(f"SELECT {', '.join(cols)} FROM trades").fetchall()
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        records.append(
            {
                "is_open": int(data.get("is_open") or 0),
                "is_short": int(data.get("is_short") or 0),
                "leverage": parse_float(data.get("leverage")) or 1.0,
                "pair": str(data.get("pair") or ""),
                "open_date": data.get("open_date"),
                "close_date": data.get("close_date"),
                "profit_ratio": parse_float(data.get("close_profit")),
                "profit_abs": parse_float(data.get("close_profit_abs")),
                "realized_profit": parse_float(data.get("realized_profit")) or parse_float(data.get("close_profit_abs")),
                "exit_reason": str(data.get("exit_reason") or ""),
                "enter_tag": str(data.get("enter_tag") or ""),
            }
        )
    return records


def _ml_dev_db_path(dev_slug: str) -> Path | None:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT db_path FROM dev_candidates WHERE slug = ? ORDER BY id DESC LIMIT 1",
            (dev_slug,),
        ).fetchone()
    if not row or not row["db_path"]:
        return None
    return resolve_path(row["db_path"])


def _dev_archived_trade_record(row: dict[str, Any]) -> dict[str, Any]:
    """Map a dev_archived_trades row into the normalized trade shape that
    _ml_read_sqlite_trades / team_trades consumers expect (plus dev-only extras)."""
    return {
        "is_open": 0,  # archived trades are, by definition, closed
        "is_short": int(row.get("is_short") or 0),
        "leverage": 1.0,
        "pair": str(row.get("pair") or ""),
        "open_date": row.get("open_date"),
        "close_date": row.get("close_date"),
        "profit_ratio": parse_float(row.get("profit_ratio")),
        "profit_abs": parse_float(row.get("profit_abs")),
        "realized_profit": parse_float(row.get("realized_profit")) or parse_float(row.get("profit_abs")),
        "exit_reason": str(row.get("exit_reason") or ""),
        "enter_tag": str(row.get("enter_tag") or ""),
        # dev-only extras used by the profile timeline; ignored by ML telemetry.
        "forced": int(row.get("forced") or 0),
        "strategy_version": str(row.get("strategy_version") or ""),
        "session_started_at": str(row.get("session_started_at") or ""),
    }


def dev_all_time_trades(slug: str, include_live: bool = True) -> list[dict[str, Any]]:
    """Durable all-time trade history for a dev strategy: every archived shift across
    all strategy versions, optionally unioned with the current (not-yet-archived) live
    shift. Survives the per-shift runtime-DB wipe. Records match the _ml_read_sqlite_trades
    shape so ML telemetry and the profile timeline can consume the same source.

    Live/archive dedup is by (pair, close_date): a shift that was archived at the bell but
    whose runtime DB has not yet been wiped will not be double-counted."""
    slug = registry_slug(str(slug or "")) or str(slug or "")
    if not slug:
        return []
    with closing(get_db()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM dev_archived_trades WHERE slug = ? ORDER BY close_date", (slug,)
        ).fetchall()]
        cand = conn.execute(
            "SELECT * FROM dev_candidates WHERE slug = ? ORDER BY id DESC LIMIT 1", (slug,)
        ).fetchone()
    records = [_dev_archived_trade_record(r) for r in rows]
    if not include_live or not cand:
        return records
    live_db = _ml_dev_db_path(slug)
    if not live_db:
        return records
    archived_sig = {(r.get("pair"), str(r.get("close_date") or "")) for r in rows if r.get("close_date")}
    version = strategy_version_token(dict(cand))
    session_started = current_open_session_started_at(int(cand["id"])) or ""
    for trade in _ml_read_sqlite_trades(live_db):
        close_date = str(trade.get("close_date") or "")
        if close_date and (trade.get("pair"), close_date) in archived_sig:
            continue  # already captured by the bell archive — avoid double counting
        rec = dict(trade)
        rec["forced"] = 1 if str(rec.get("exit_reason") or "").lower() in DEV_FORCED_EXIT_REASONS else 0
        rec["strategy_version"] = version
        rec["session_started_at"] = session_started
        records.append(rec)
    return records


def _matrix_hypothesis(tag: str, kind: str) -> str:
    """Reduce an entry/exit tag to its underlying hypothesis so they can be compared.
    Entry tags are `<hypothesis>_long|short`; exit tags are `<hypothesis>_exit`. The
    hypothesis itself may contain underscores (mean_reversion, base_building, regime_change),
    so strip only the known role suffix rather than splitting on '_'."""
    t = str(tag or "").strip().lower()
    if not t:
        return ""
    if kind == "entry":
        for suffix in ("_long", "_short"):
            if t.endswith(suffix):
                return t[: -len(suffix)]
        return t
    if t.endswith("_exit"):
        return t[: -len("_exit")]
    return t


def compute_entry_exit_matrix(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Entry Tag -> Exit Tag confusion matrix for a dev strategy's all-time trades —
    Second Act's primary research artifact. Excludes open and forced (shift-bell) exits so
    only strategy-chosen, resolved classifications are counted. Each cell carries
    count / pnl / win_rate; the `accuracy` diagonal credits a trade whose observed-outcome
    exit hypothesis matches its entry prediction (continuation_* -> continuation_exit).
    Generic to any dev strategy; only an event-classifier like Second Act fills it richly."""
    cells: dict[tuple[str, str], dict[str, float]] = {}
    entry_totals: dict[str, int] = {}
    exit_totals: dict[str, int] = {}
    matched = 0
    classifiable = 0
    total = 0
    for tr in trades:
        if int(tr.get("is_open") or 0):
            continue
        exit_reason = str(tr.get("exit_reason") or "").strip()
        if exit_reason.lower() in DEV_FORCED_EXIT_REASONS:
            continue
        enter_tag = str(tr.get("enter_tag") or "").strip() or "(untagged)"
        exit_key = exit_reason or "(none)"
        pnl = parse_float(tr.get("realized_profit")) or parse_float(tr.get("profit_abs"))
        win = 1 if (parse_float(tr.get("profit_abs")) or 0.0) > 0 else 0
        cell = cells.setdefault((enter_tag, exit_key), {"count": 0, "pnl": 0.0, "wins": 0})
        cell["count"] += 1
        cell["pnl"] += pnl
        cell["wins"] += win
        entry_totals[enter_tag] = entry_totals.get(enter_tag, 0) + 1
        exit_totals[exit_key] = exit_totals.get(exit_key, 0) + 1
        total += 1
        if enter_tag != "(untagged)":
            classifiable += 1
            eh = _matrix_hypothesis(enter_tag, "entry")
            if eh and eh == _matrix_hypothesis(exit_key, "exit"):
                matched += 1
    row_order = sorted(entry_totals, key=lambda k: entry_totals[k], reverse=True)
    col_order = sorted(exit_totals, key=lambda k: exit_totals[k], reverse=True)
    rows: list[dict[str, Any]] = []
    for et in row_order:
        cols: list[dict[str, Any]] = []
        for xt in col_order:
            c = cells.get((et, xt))
            cols.append({
                "exit_tag": xt,
                "count": int(c["count"]) if c else 0,
                "pnl": round(c["pnl"], 2) if c else 0.0,
                "win_rate": round(100.0 * c["wins"] / c["count"], 1) if c else 0.0,
            })
        rows.append({"entry_tag": et, "total": entry_totals[et], "cells": cols})
    return {
        "rows": rows,
        "exit_tags": [{"exit_tag": xt, "total": exit_totals[xt]} for xt in col_order],
        "total_classified": total,
        "classifiable": classifiable,
        "matched": matched,
        "accuracy": round(100.0 * matched / classifiable, 1) if classifiable else 0.0,
    }


_DEV_TIMELINE_VERSION_PALETTE = ("#6366f1", "#0ea5e9", "#f59e0b", "#ec4899", "#10b981", "#a855f7", "#ef4444")


def dev_strategy_timeline(slug: str, max_days: int = 120) -> dict[str, Any]:
    """All-time per-day PnL/trade-count timeline for a dev strategy, plus strategy-version
    bands and headline summary stats. Sourced from dev_all_time_trades so it survives the
    per-shift wipe. PnL is strategy-chosen exits; bell force-exits are tracked separately."""
    trades = dev_all_time_trades(slug)
    closed = [t for t in trades if not int(t.get("is_open") or 0)]

    daily: dict[Any, dict[str, Any]] = {}
    version_span: dict[str, list[Any]] = {}
    for trade in closed:
        dt = _ml_parse_dt(trade.get("close_date"))
        if not dt:
            continue
        day = dt.date()
        bucket = daily.setdefault(day, {"strategy_pnl": 0.0, "forced_pnl": 0.0, "count": 0, "versions": Counter()})
        pnl = parse_float(trade.get("profit_abs"))
        if int(trade.get("forced") or 0):
            bucket["forced_pnl"] += pnl
        else:
            bucket["strategy_pnl"] += pnl
        bucket["count"] += 1
        version = str(trade.get("strategy_version") or "")
        bucket["versions"][version] += 1
        if version:
            span = version_span.setdefault(version, [day, day])
            span[0] = min(span[0], day)
            span[1] = max(span[1], day)

    strat_pnl = sum(parse_float(t.get("profit_abs")) for t in closed if not int(t.get("forced") or 0))
    forced_pnl = sum(parse_float(t.get("profit_abs")) for t in closed if int(t.get("forced") or 0))
    forced_count = sum(1 for t in closed if int(t.get("forced") or 0))
    wins = sum(1 for t in closed if parse_float(t.get("profit_abs")) > 0)
    summary = {
        "closed_trades": len(closed),
        "wins": wins,
        "win_rate": round(percentage(wins, len(closed)) * 100.0, 1),
        "realized_pnl": round(strat_pnl + forced_pnl, 4),
        "strategy_pnl": round(strat_pnl, 4),
        "forced_pnl": round(forced_pnl, 4),
        "forced_exits": forced_count,
        "version_count": len(version_span),
        "active_days": len(daily),
    }

    ordered_versions = sorted(version_span.items(), key=lambda kv: kv[1][0])
    version_label = {ver: f"v{i + 1}" for i, (ver, _span) in enumerate(ordered_versions)}
    version_color = {ver: _DEV_TIMELINE_VERSION_PALETTE[i % len(_DEV_TIMELINE_VERSION_PALETTE)]
                     for i, (ver, _span) in enumerate(ordered_versions)}
    bands = [
        {"label": version_label[ver], "version": ver, "color": version_color[ver],
         "start": span[0].isoformat(), "end": span[1].isoformat(),
         "days": (span[1] - span[0]).days + 1}
        for ver, span in ordered_versions
    ]

    if not daily:
        return {"has_data": False, "summary": summary, "version_bands": bands, "svg": ""}

    first, last = min(daily), max(daily)
    all_days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    if len(all_days) > max_days:
        all_days = all_days[-max_days:]
    svg = _dev_timeline_svg(all_days, daily, ordered_versions, version_label, version_color)
    return {"has_data": True, "summary": summary, "version_bands": bands, "svg": svg}


def _dev_timeline_svg(all_days, daily, ordered_versions, version_label, version_color) -> str:
    """Render a self-contained inline SVG: per-day strategy-PnL bars (green/red) over
    semi-transparent trade-count ghost bars, under a strategy-version ribbon. No JS."""
    n = len(all_days)
    slot_w = max(9, min(26, int(900 / max(1, n))))
    left_pad, right_pad, top_pad, bottom_pad = 48, 14, 34, 36
    plot_h = 160
    width = left_pad + n * slot_w + right_pad
    height = top_pad + plot_h + bottom_pad
    plot_top, plot_bottom = top_pad, top_pad + plot_h
    zero_y = plot_top + plot_h / 2
    half = plot_h / 2 - 6

    day_index = {day: i for i, day in enumerate(all_days)}
    max_abs = max((abs(b["strategy_pnl"]) for b in daily.values()), default=0.0) or 1.0
    max_count = max((b["count"] for b in daily.values()), default=0) or 1

    def cx(i: int) -> float:
        return left_pad + i * slot_w + slot_w / 2

    parts: list[str] = [
        # Explicit px size + height:auto so a narrow (few-day) chart keeps its natural aspect
        # ratio instead of stretching to the container width and ballooning vertically.
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'style="max-width:100%;height:auto;max-height:240px;font-family:inherit;display:block">',
        '<style>.tl-lbl{fill:#64748b;font-size:10px}.tl-axis{stroke:#cbd5e1;stroke-width:1}</style>',
    ]

    # Version ribbon across the top.
    for ver, span in ordered_versions:
        si = day_index.get(span[0])
        ei = day_index.get(span[1])
        if si is None and ei is None:
            continue
        si = 0 if si is None else si
        ei = (n - 1) if ei is None else ei
        x0 = left_pad + si * slot_w
        x1 = left_pad + (ei + 1) * slot_w
        color = version_color[ver]
        parts.append(
            f'<rect x="{x0:.1f}" y="6" width="{max(2.0, x1 - x0):.1f}" height="16" rx="3" '
            f'fill="{color}" opacity="0.28"><title>{version_label[ver]} — {ver}</title></rect>'
        )
        if x1 - x0 > 26:
            parts.append(
                f'<text x="{(x0 + x1) / 2:.1f}" y="18" text-anchor="middle" '
                f'style="fill:{color};font-size:10px;font-weight:600">{version_label[ver]}</text>'
            )

    # Zero line + y-axis labels.
    parts.append(f'<line x1="{left_pad}" y1="{zero_y:.1f}" x2="{width - right_pad}" y2="{zero_y:.1f}" class="tl-axis"/>')
    parts.append(f'<text x="{left_pad - 6}" y="{plot_top + 4:.1f}" text-anchor="end" class="tl-lbl">{max_abs:+.1f}</text>')
    parts.append(f'<text x="{left_pad - 6}" y="{zero_y + 3:.1f}" text-anchor="end" class="tl-lbl">0</text>')
    parts.append(f'<text x="{left_pad - 6}" y="{plot_bottom:.1f}" text-anchor="end" class="tl-lbl">{-max_abs:+.1f}</text>')

    bar_w = max(3.0, slot_w * 0.6)
    label_step = max(1, math.ceil(n / 12))
    for i, day in enumerate(all_days):
        bucket = daily.get(day)
        x = cx(i)
        # Ghost trade-count bar (behind), scaled to its own axis, drawn from the baseline up.
        if bucket and bucket["count"]:
            gh = (bucket["count"] / max_count) * plot_h
            parts.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{plot_bottom - gh:.1f}" width="{bar_w:.1f}" '
                f'height="{gh:.1f}" fill="#94a3b8" opacity="0.16"/>'
            )
        # Strategy PnL bar diverging from zero.
        if bucket:
            pnl = bucket["strategy_pnl"]
            bh = (abs(pnl) / max_abs) * half
            if pnl >= 0:
                y = zero_y - bh
            else:
                y = zero_y
            color = "#22c55e" if pnl >= 0 else "#ef4444"
            forced_note = f', {bucket["forced_pnl"]:+.2f} forced' if bucket["forced_pnl"] else ""
            parts.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(0.6, bh):.1f}" '
                f'fill="{color}" rx="1"><title>{day.isoformat()}: {pnl:+.2f} strat{forced_note} · '
                f'{bucket["count"]} trade(s)</title></rect>'
            )
            if bucket["forced_pnl"]:  # mark days that carried a bell force-exit
                parts.append(f'<circle cx="{x:.1f}" cy="{plot_bottom + 5:.1f}" r="2" fill="#f59e0b"/>')
        if i % label_step == 0:
            parts.append(
                f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle" class="tl-lbl">'
                f'{day.strftime("%m/%d")}</text>'
            )

    parts.append('</svg>')
    return "".join(parts)


def ml_collect_trades(strategy: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized trade records for a registry strategy from whichever
    source it maps to (main league team_trades, or a dev candidate sqlite)."""
    kind = str(strategy.get("kind") or "main")
    if kind == "main":
        team_id = str(strategy.get("source_team_id") or "")
        if not team_id:
            return []
        with closing(get_db()) as conn:
            rows = conn.execute(
                """
                SELECT is_open, is_short, leverage, pair, open_date, close_date,
                       profit_ratio, profit_abs, realized_profit, exit_reason, enter_tag,
                       trade_duration_minutes
                FROM team_trades WHERE team_id = ?
                """,
                (team_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    # dev kind: source_team_id holds the dev_candidates slug. Read the durable all-time
    # history (archived shifts across all versions + current live shift) rather than the
    # ephemeral runtime DB, which is wiped every shift and otherwise reads as "0 trades".
    return dev_all_time_trades(str(strategy.get("source_team_id") or strategy.get("slug")))


def ml_compute_telemetry(strategy: dict[str, Any], trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute the telemetry vector for one strategy. Returns
    {category: {value, sample_size, measurable}}. Regime-dependent categories
    are emitted with measurable=0 and value=0 (never fabricated)."""
    closed = [t for t in trades if not int(t.get("is_open") or 0)]
    open_rows = [t for t in trades if int(t.get("is_open") or 0)]
    sample = len(closed)

    durations = [_ml_duration_minutes(t) for t in closed]
    roi_list = [parse_float(t.get("profit_ratio")) for t in closed]
    winners = [r for r in roi_list if r > 0]
    realized_pnl = sum(parse_float(t.get("realized_profit")) or parse_float(t.get("profit_abs")) for t in closed)
    unrealized_pnl = sum(parse_float(t.get("profit_abs")) for t in open_rows)

    # Span (hours) for throughput, from earliest open to latest close.
    dts = [d for d in (_ml_parse_dt(t.get("open_date")) for t in closed) if d]
    dts += [d for d in (_ml_parse_dt(t.get("close_date")) for t in closed) if d]
    span_hours = 0.0
    if len(dts) >= 2:
        span_hours = max(0.0, (max(dts) - min(dts)).total_seconds() / 3600.0)
    throughput = (sample / (span_hours / 24.0)) if span_hours >= 1.0 else float(sample)

    pair_counts: dict[str, int] = defaultdict(int)
    for t in closed:
        pair_counts[str(t.get("pair") or "?")] += 1

    wins = len(winners)
    shorts = sum(1 for t in closed if int(t.get("is_short") or 0))

    measured = {
        "throughput": throughput,
        "win_rate": percentage(wins, sample) * 100.0,
        "avg_roi": stat_mean(roi_list) * 100.0,
        "avg_hold_time": stat_mean(durations),
        "hold_time_dispersion": stat_pstdev(durations),
        "realized_conversion": percentage(realized_pnl, abs(realized_pnl) + abs(unrealized_pnl)) * 100.0,
        "unrealized_drag": unrealized_pnl,
        "long_short_bias": percentage(shorts, sample) * 100.0,
        "pair_concentration": hhi([float(c) for c in pair_counts.values()]) * 100.0,
        "exit_efficiency": stat_mean(winners) * 100.0,
        "selectivity": 100.0 / (throughput + 1.0),
    }
    result: dict[str, dict[str, Any]] = {}
    for category in ML_TELEMETRY_CATEGORIES:
        name = category["name"]
        if category["measurable"]:
            result[name] = {"value": round(float(measured.get(name, 0.0)), 6), "sample_size": sample, "measurable": 1}
        else:
            result[name] = {"value": 0.0, "sample_size": sample, "measurable": 0}
    return result


def ml_persist_telemetry(cycle_id: int, strategy_slug: str, telemetry: dict[str, dict[str, Any]]) -> None:
    now = iso_now()
    with closing(get_db()) as conn:
        for category, payload in telemetry.items():
            conn.execute(
                """
                INSERT INTO ml_telemetry (cycle_id, strategy_slug, category, value, sample_size, measurable, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cycle_id, strategy_slug, category, float(payload["value"]),
                 int(payload["sample_size"]), int(payload["measurable"]), now),
            )
        conn.commit()


def ml_compute_and_persist_divergence(cycle_id: int) -> None:
    """For each measurable category, compute peer-relative divergence (robust z,
    percentile) across all strategies with enough samples."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT strategy_slug, category, value, sample_size
            FROM ml_telemetry
            WHERE cycle_id = ? AND measurable = 1 AND sample_size >= ?
            """,
            (cycle_id, ML_MIN_SAMPLE),
        ).fetchall()
    by_category: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append((row["strategy_slug"], float(row["value"])))

    now = iso_now()
    with closing(get_db()) as conn:
        for category, entries in by_category.items():
            if len(entries) < 2:
                continue
            values = [v for _, v in entries]
            mean = stat_mean(values)
            median = stat_median(values)
            mad = stat_mad(values)
            for slug, value in entries:
                z = robust_z(value, values)
                conn.execute(
                    """
                    INSERT INTO ml_telemetry_divergence
                        (cycle_id, strategy_slug, category, value, peer_mean, peer_median,
                         peer_mad, percentile, robust_z, direction, magnitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cycle_id, slug, category, value, mean, median, mad,
                     percentile_rank(value, values), z,
                     "high" if value >= median else "low", abs(z)),
                )
        conn.commit()


def ml_cycle_z_vectors(cycle_id: int) -> dict[str, dict[str, float]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT strategy_slug, category, robust_z FROM ml_telemetry_divergence WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchall()
    vectors: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        vectors[row["strategy_slug"]][row["category"]] = float(row["robust_z"])
    return vectors


def _ml_favorable(category: str, z: float) -> str:
    good = ML_CATEGORY_GOOD.get(category, "neutral")
    if good == "neutral":
        return "neutral"
    if good == "high":
        return "favorable" if z > 0 else "unfavorable"
    return "favorable" if z < 0 else "unfavorable"


def ml_select_fearsome_five(cycle_id: int, limit: int = 5, per_strategy_cap: int = 2) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM ml_telemetry_divergence
            WHERE cycle_id = ?
            ORDER BY magnitude DESC
            """,
            (cycle_id,),
        ).fetchall()
    names = {s["slug"]: s["name"] for s in ml_registry_all(active_only=False)}
    findings: list[dict[str, Any]] = []
    per_strategy: dict[str, int] = defaultdict(int)
    for row in rows:
        slug = row["strategy_slug"]
        if per_strategy[slug] >= per_strategy_cap:
            continue
        if abs(float(row["robust_z"])) < 0.5:  # not actually divergent
            continue
        per_strategy[slug] += 1
        findings.append(
            {
                "strategy_slug": slug,
                "strategy_name": names.get(slug, slug),
                "category": row["category"],
                "value": round(float(row["value"]), 4),
                "peer_mean": round(float(row["peer_mean"]), 4),
                "peer_median": round(float(row["peer_median"]), 4),
                "robust_z": round(float(row["robust_z"]), 3),
                "direction": row["direction"],
                "magnitude": round(float(row["magnitude"]), 3),
                "favorable": _ml_favorable(row["category"], float(row["robust_z"])),
            }
        )
        if len(findings) >= limit:
            break
    return findings


def ml_compute_relationships(cycle_id: int) -> list[dict[str, Any]]:
    vectors = ml_cycle_z_vectors(cycle_id)
    slugs = sorted(vectors.keys())
    registry = {s["slug"]: s for s in ml_registry_all(active_only=False)}
    lineage_pairs = {(row["child_slug"], row["parent_slug"]) for row in ml_lineage_all()}
    directional = [c for c in ML_MEASURABLE_CATEGORIES if ML_CATEGORY_GOOD.get(c) in {"high", "low"}]
    complement_threshold = parse_float(get_setting("ml_complement_pair_threshold", "0.55"))

    now = iso_now()
    results: list[dict[str, Any]] = []
    with closing(get_db()) as conn:
        for i in range(len(slugs)):
            for j in range(i + 1, len(slugs)):
                a, b = slugs[i], slugs[j]
                za, zb = vectors[a], vectors[b]
                similarity = cosine_similarity(za, zb)
                # Complement: strength/weakness coverage + non-overlapping strength.
                raw = 0.0
                active_cats = 0
                for cat in directional:
                    fa = _ml_favorable(cat, za.get(cat, 0.0))
                    fb = _ml_favorable(cat, zb.get(cat, 0.0))
                    strong_a = abs(za.get(cat, 0.0)) >= 1.0
                    strong_b = abs(zb.get(cat, 0.0)) >= 1.0
                    if not (strong_a or strong_b):
                        continue
                    active_cats += 1
                    if strong_a and strong_b and fa != fb and "neutral" not in (fa, fb):
                        raw += 1.0  # one covers the other's weakness
                    elif (strong_a and fa == "favorable" and not strong_b) or (strong_b and fb == "favorable" and not strong_a):
                        raw += 0.5  # non-overlapping strength
                complement = clamp(raw / active_cats, 0.0, 1.0) if active_cats else 0.0

                fam_a = registry.get(a, {}).get("family_slug")
                fam_b = registry.get(b, {}).get("family_slug")
                if (a, b) in lineage_pairs or (b, a) in lineage_pairs:
                    rel_type = "parent"
                elif fam_a and fam_a == fam_b and similarity > 0.5:
                    rel_type = "sibling"
                elif complement >= complement_threshold and similarity < 0.3:
                    rel_type = "complement"
                elif similarity > 0.6:
                    rel_type = "rival"
                elif similarity < -0.3:
                    rel_type = "opposite"
                elif fam_a and fam_a == fam_b:
                    rel_type = "cousin"
                else:
                    rel_type = "complement" if complement >= 0.3 else "rival"

                conn.execute(
                    """
                    INSERT INTO ml_relationships
                        (cycle_id, strategy_a, strategy_b, relationship_type,
                         similarity_score, complement_score, evidence_notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cycle_id, a, b, rel_type, round(similarity, 4), round(complement, 4), "", now),
                )
                results.append({"a": a, "b": b, "type": rel_type, "similarity": similarity, "complement": complement})
        conn.commit()
    return results


def ml_update_descendant_hypotheses(cycle_id: int) -> None:
    """Runs every cycle so conviction accrues continuously. Upserts a hypothesis
    per high-complement pair and recomputes its sub-scores over a rolling window."""
    complement_threshold = parse_float(get_setting("ml_complement_pair_threshold", "0.55"))
    min_evidence = max(1, int(parse_float(get_setting("ml_descendant_min_evidence_cycles", "8"))))
    conv_threshold = parse_float(get_setting("ml_descendant_conviction_threshold", "0.75"))
    novelty_threshold = parse_float(get_setting("ml_descendant_novelty_threshold", "0.35"))
    window = 120  # ~15 days at one cycle / 3h

    vectors = ml_cycle_z_vectors(cycle_id)
    registry = {s["slug"]: s for s in ml_registry_all(active_only=False)}

    # Candidate pairs from this cycle's relationships above the complement bar.
    with closing(get_db()) as conn:
        cycle_rels = conn.execute(
            "SELECT strategy_a, strategy_b, complement_score, similarity_score FROM ml_relationships WHERE cycle_id = ? AND complement_score >= ?",
            (cycle_id, complement_threshold),
        ).fetchall()
        window_floor = max(0, cycle_id - window)

    now = iso_now()
    for rel in cycle_rels:
        a, b = sorted([rel["strategy_a"], rel["strategy_b"]])
        pair_slug = f"{a}__x__{b}"
        # Complement history across the rolling window.
        with closing(get_db()) as conn:
            hist = conn.execute(
                """
                SELECT cycle_id, complement_score FROM ml_relationships
                WHERE ((strategy_a = ? AND strategy_b = ?) OR (strategy_a = ? AND strategy_b = ?))
                  AND cycle_id >= ?
                """,
                (a, b, b, a, window_floor),
            ).fetchall()
        comp_values = [float(r["complement_score"]) for r in hist]
        complement_avg = stat_mean(comp_values) if comp_values else float(rel["complement_score"])
        evidence_cycles = sum(1 for v in comp_values if v >= complement_threshold)
        support_ids = sorted({int(r["cycle_id"]) for r in hist if float(r["complement_score"]) >= complement_threshold})

        # Inheritance: do both parents have identifiable strengths?
        def strong_strengths(slug: str) -> int:
            return sum(1 for t in ml_traits_for(slug) if t["polarity"] == "strength" and float(t["confidence"]) >= 0.5)
        inheritance = (min(1.0, strong_strengths(a) / 2.0) + min(1.0, strong_strengths(b) / 2.0)) / 2.0

        # Environmental coverage: how differently the parents behave on env-ish axes.
        env_cats = ["long_short_bias", "avg_hold_time", "throughput", "pair_concentration"]
        za, zb = vectors.get(a, {}), vectors.get(b, {})
        diffs = [abs(za.get(c, 0.0) - zb.get(c, 0.0)) for c in env_cats]
        environmental = clamp(stat_mean(diffs) / 3.0, 0.0, 1.0)

        # Novelty: a hypothetical child (mean of parent z-vectors) vs every existing strategy.
        child_vec = {c: (za.get(c, 0.0) + zb.get(c, 0.0)) / 2.0 for c in set(za) | set(zb)}
        max_sim = 0.0
        for slug, vec in vectors.items():
            if slug in (a, b):
                continue
            max_sim = max(max_sim, cosine_similarity(child_vec, vec))
        novelty = clamp(1.0 - max_sim, 0.0, 1.0)

        evidence = clamp(evidence_cycles / float(min_evidence), 0.0, 1.0)
        conviction = (
            0.30 * complement_avg + 0.20 * inheritance + 0.15 * environmental
            + 0.15 * novelty + 0.20 * evidence
        )
        if conviction >= conv_threshold and evidence_cycles >= min_evidence and inheritance > 0 and novelty >= novelty_threshold:
            status = "proposed"
        elif conviction >= 0.5:
            status = "strengthening"
        else:
            status = "watching"

        name_a = registry.get(a, {}).get("name", a)
        name_b = registry.get(b, {}).get("name", b)
        rationale = (
            f"{name_a} × {name_b}: complement {complement_avg:.2f}, inheritance {inheritance:.2f}, "
            f"environmental {environmental:.2f}, novelty {novelty:.2f}, evidence {evidence_cycles}/{min_evidence} cycles."
        )

        with closing(get_db()) as conn:
            existing = conn.execute(
                "SELECT id, first_seen_cycle, status FROM ml_descendant_hypotheses WHERE slug = ?", (pair_slug,)
            ).fetchone()
            if existing:
                # Never downgrade out of 'promoted'.
                final_status = "promoted" if existing["status"] == "promoted" else status
                conn.execute(
                    """
                    UPDATE ml_descendant_hypotheses SET
                        parent_a = ?, parent_b = ?, rationale = ?, complement_score = ?,
                        inheritance_score = ?, environmental_score = ?, novelty_score = ?,
                        evidence_score = ?, conviction_score = ?, status = ?,
                        last_updated_cycle = ?, supporting_cycle_ids_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (a, b, rationale, round(complement_avg, 4), round(inheritance, 4),
                     round(environmental, 4), round(novelty, 4), round(evidence, 4),
                     round(conviction, 4), final_status, cycle_id, json.dumps(support_ids), now, existing["id"]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO ml_descendant_hypotheses
                        (slug, parent_a, parent_b, rationale, complement_score, inheritance_score,
                         environmental_score, novelty_score, evidence_score, conviction_score,
                         status, first_seen_cycle, last_updated_cycle, supporting_cycle_ids_json,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pair_slug, a, b, rationale, round(complement_avg, 4), round(inheritance, 4),
                     round(environmental, 4), round(novelty, 4), round(evidence, 4), round(conviction, 4),
                     status, cycle_id, cycle_id, json.dumps(support_ids), now, now),
                )
            conn.commit()


def ml_descendant_hypotheses_all(order_by_conviction: bool = True) -> list[dict[str, Any]]:
    query = "SELECT * FROM ml_descendant_hypotheses"
    query += " ORDER BY conviction_score DESC" if order_by_conviction else " ORDER BY updated_at DESC"
    with closing(get_db()) as conn:
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def _ml_previous_cycle_open_questions(limit: int = 6) -> list[str]:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT essay_json FROM ml_telemetry_cycles WHERE status = 'complete' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return []
    try:
        data = json.loads(row["essay_json"] or "{}")
    except json.JSONDecodeError:
        return []
    return [str(q) for q in (data.get("open_questions") or [])][:limit]


def _ml_fallback_essay(findings: list[dict[str, Any]], hypotheses: list[dict[str, Any]]) -> str:
    lines = ["# ML Telemetry Essay (deterministic)", ""]
    if not findings:
        lines.append("No strategy diverged meaningfully from its peers this cycle. The field is behaving homogeneously, or sample sizes are still too thin to measure. Continuing to watch.")
    else:
        lines.append("The five most divergent telemetry signals this cycle:")
        lines.append("")
        for i, f in enumerate(findings, 1):
            verdict = {"favorable": "looks favorable", "unfavorable": "looks like a weakness", "neutral": "is directionally ambiguous"}.get(f.get("favorable"), "")
            lines.append(
                f"{i}. **{f['strategy_name']} — {f['category']}**: value {f['value']} vs peer median "
                f"{f['peer_median']} (robust z {f['robust_z']}, {f['direction']}). This {verdict}."
            )
    if hypotheses:
        lines += ["", "## Descendant watchlist", ""]
        for h in hypotheses[:5]:
            lines.append(f"- `{h['slug']}` — conviction {h['conviction_score']:.2f} ({h['status']}): {h['rationale']}")
    return "\n".join(lines)


def run_ml_telemetry_cycle() -> None:
    """3-hour evidence cycle. Python measures telemetry + divergence + relationships
    + descendant conviction deterministically; the LLM only writes interpretive prose.
    Degrades gracefully (templated essay) when no LLM is configured."""
    seed_ml_biology()
    started = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO ml_telemetry_cycles (started_at, status) VALUES (?, 'running')", (started,)
        )
        cycle_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()

    # --- Deterministic measurement ---
    strategies_with_data = 0
    for strategy in ml_registry_all(active_only=True):
        trades = ml_collect_trades(strategy)
        telemetry = ml_compute_telemetry(strategy, trades)
        ml_persist_telemetry(cycle_id, strategy["slug"], telemetry)
        if telemetry.get("win_rate", {}).get("sample_size", 0) >= ML_MIN_SAMPLE:
            strategies_with_data += 1
    ml_compute_and_persist_divergence(cycle_id)
    findings = ml_select_fearsome_five(cycle_id)
    ml_compute_relationships(cycle_id)
    ml_update_descendant_hypotheses(cycle_id)
    hypotheses = ml_descendant_hypotheses_all()

    # --- Language layer (LLM optional) ---
    essay_markdown = _ml_fallback_essay(findings, hypotheses)
    essay_json: dict[str, Any] = {
        "findings": findings,
        "open_questions": [],
        "hypothesis_effects": [],
        "per_finding_confidence": {},
        "hypotheses_top": [
            {"slug": h["slug"], "conviction": h["conviction_score"], "status": h["status"]}
            for h in hypotheses[:5]
        ],
    }
    llm_used = 0
    if get_setting("ollama_api_key", "") and findings:
        try:
            trait_context = {}
            for f in findings:
                trait_context[f["strategy_slug"]] = [
                    {"trait": t["trait_name"], "polarity": t["polarity"], "confidence": t["confidence"]}
                    for t in ml_traits_for(f["strategy_slug"])
                ]
            user_payload = {
                "fearsome_five": findings,
                "strategy_traits": trait_context,
                "descendant_watchlist": essay_json["hypotheses_top"],
                "previous_open_questions": _ml_previous_cycle_open_questions(),
                "note": "These numbers are precomputed and authoritative. Do not invent or recompute any number.",
            }
            content = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the ATL strategy biologist. You are handed five pre-computed telemetry "
                            "divergences. Do NOT invent or recompute any numbers. Return strict JSON with keys "
                            "essay_markdown, per_finding_confidence, trait_updates, open_questions, hypothesis_effects. "
                            "essay_markdown: a readable markdown essay interpreting each of the five findings — what it "
                            "means, whether it is favorable/unfavorable/ambiguous, the trait it may represent, the code "
                            "mechanism that may explain it, relationship implications, and what to watch next. "
                            "per_finding_confidence: object mapping the finding category to a 0-1 confidence. "
                            "trait_updates: array (max 6) of {strategy_slug, trait_name, polarity, confidence} where "
                            "strategy_slug is one of the supplied slugs and confidence is 0-1. "
                            "open_questions: array (max 5) of short strings. "
                            "hypothesis_effects: array of {slug, effect} where effect is strengthen|weaken|unchanged."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload)},
                ]
            )
            payload = parse_json_block(content)
            if payload.get("essay_markdown"):
                essay_markdown = str(payload["essay_markdown"])
            essay_json["per_finding_confidence"] = payload.get("per_finding_confidence", {}) or {}
            essay_json["open_questions"] = [str(q) for q in (payload.get("open_questions") or [])][:5]
            essay_json["hypothesis_effects"] = payload.get("hypothesis_effects", []) or []
            # LLM proposes trait updates; Python clamps and records evidence.
            for upd in (payload.get("trait_updates") or [])[:6]:
                if not isinstance(upd, dict):
                    continue
                slug = str(upd.get("strategy_slug") or "")
                trait_name = str(upd.get("trait_name") or "").strip()
                if not slug or not trait_name or not ml_registry_get(slug):
                    continue
                upsert_ml_trait(
                    slug, trait_name,
                    polarity=str(upd.get("polarity") or "neutral"),
                    confidence=parse_float(upd.get("confidence")) or 0.3,
                    evidence_source=f"telemetry-cycle-{cycle_id}",
                    bump_evidence=True,
                )
            llm_used = 1
        except Exception as exc:  # noqa: BLE001
            log_maintenance("ml", "warning", f"Telemetry essay LLM call failed, using deterministic essay: {exc}")

    # --- Persist + index ---
    with closing(get_db()) as conn:
        conn.execute(
            """
            UPDATE ml_telemetry_cycles SET
                completed_at = ?, strategies_count = ?, essay_markdown = ?,
                essay_json = ?, top_findings_json = ?, status = 'complete', llm_used = ?
            WHERE id = ?
            """,
            (iso_now(), strategies_with_data, essay_markdown, json.dumps(essay_json),
             json.dumps(findings), llm_used, cycle_id),
        )
        conn.commit()
    upsert_research_index_entry(
        "ml_essay",
        f"ml_essay:cycle-{cycle_id}",
        f"ML Telemetry Essay — cycle {cycle_id} ({started})",
        essay_markdown,
        "ml telemetry essay fearsome five divergence descendant",
        entry_type="finding",
        author_type="agent",
        status="active",
    )
    set_setting("ml_maintenance_last_run", iso_now())
    log_maintenance("ml", "success", f"Telemetry cycle {cycle_id}: {len(findings)} findings, {len(hypotheses)} descendant hypotheses, llm_used={llm_used}.")


def run_ml_maintenance() -> None:
    """Backward-compatible entry point; the 3-hour ML cycle is now the telemetry essay."""
    run_ml_telemetry_cycle()


# ===========================================================================
# ML Lab — 15-day Evolution Review (Phase 4)
# ===========================================================================
def _ml_dev_slug_for_candidate(candidate_id: int) -> str:
    with closing(get_db()) as conn:
        row = conn.execute("SELECT slug FROM dev_candidates WHERE id = ?", (candidate_id,)).fetchone()
    return row["slug"] if row else ""


def ml_create_descendant(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Turn an approved descendant hypothesis into a real Draft Room prospect:
    create the dev_candidate, record lineage, register the organism, and queue
    strategy generation + validation (stopping at the human-review gate)."""
    a, b = hypothesis["parent_a"], hypothesis["parent_b"]
    reg = {s["slug"]: s for s in ml_registry_all(active_only=False)}
    name_a = reg.get(a, {}).get("name", a)
    name_b = reg.get(b, {}).get("name", b)
    family_slug = reg.get(a, {}).get("family_slug") or reg.get(b, {}).get("family_slug") or ""

    # Deterministic descendant name from parent stems.
    stem_a = re.sub(r"[^A-Za-z]", "", name_a.split()[0])[:5].title()
    stem_b = re.sub(r"[^A-Za-z]", "", name_b.split()[0])[:5].title()
    name = f"{stem_a}{stem_b} Hybrid"

    # Trait targets: parent strengths to inherit, parent weaknesses to cover.
    strengths_a = [t["trait_name"] for t in ml_traits_for(a) if t["polarity"] == "strength"][:3]
    strengths_b = [t["trait_name"] for t in ml_traits_for(b) if t["polarity"] == "strength"][:3]
    weaknesses = [t["trait_name"] for t in (ml_traits_for(a) + ml_traits_for(b)) if t["polarity"] == "weakness"][:3]

    # Where each parent hunts on the Signal Timing Spectrum (Wind Tunnel) — used to
    # stage the parents as distinct organs rather than distilling them into one idea.
    def _niche_phrase(slug: str, fallback: str) -> str:
        view = ml_temporal_view(reg.get(slug, {}))
        if view.get("placed"):
            span = view["start_label"]
            if view["end"] > view["start"]:
                span += f"→{view['end_label']}"
            return span
        return view.get("note") or fallback
    niche_a = _niche_phrase(a, "niche undeclared")
    niche_b = _niche_phrase(b, "niche undeclared")

    # Speciation brief (see ML_EVOLUTION_REVIEW_CHARTER): build a multi-stage machine
    # that preserves both parents' machinery as distinct organs, not a distilled tweak.
    hypothesis_text = (
        f"ML Evolution Review descendant — a deliberate CROSS of {name_a} × {name_b} "
        f"(conviction {hypothesis['conviction_score']:.2f}). Build a multi-stage system, not a "
        f"refinement of either parent. Treat {name_a}'s mechanism (hunts {niche_a}) and {name_b}'s "
        f"mechanism (hunts {niche_b}) as distinct, complementary organs staged by where each acts "
        f"in a move's lifecycle — e.g. one parent's logic gates/filters entries, the other's "
        f"executes or manages them. Preserve substantial machinery from BOTH parents; the result "
        f"should be more intricate than either, not smaller. "
        f"Inherit from {name_a}: {', '.join(strengths_a) or 'core behavior'}. "
        f"Inherit from {name_b}: {', '.join(strengths_b) or 'core behavior'}. "
        f"Cover weaknesses: {', '.join(weaknesses) or 'none catalogued'}."
    )
    expected_behavior = (
        f"A staged machine combining {name_a} ({niche_a}) and {name_b} ({niche_b}) as separate "
        f"organs so each parent's strength covers the other's weakness, spanning more of the move "
        f"lifecycle than either parent alone. {hypothesis['rationale']}"
    )
    strategy_notes = (
        f"{ML_EVOLUTION_REVIEW_CHARTER}\n\n"
        f"This prospect is a cross of {name_a} and {name_b}. Keep both parents' mechanisms as "
        f"identifiable subsystems (do not collapse to a single indicator/idea); stage them by their "
        f"Wind Tunnel niches ({name_a}: {niche_a}; {name_b}: {niche_b}) and add the connective "
        f"logic that lets them operate as one larger system."
    )

    payload = {
        "name": name,
        "lifecycle_state": "draft_idea",
        "tier": "draft_room",
        "hypothesis": hypothesis_text,
        "strategy_notes": strategy_notes,
        "long_short_mode": "both",
        "expected_behavior": expected_behavior,
        "risk_profile": "Hybrid prospect — a multi-stage machine inheriting both parents' mechanisms as distinct organs; monitor for over-trading and stage interaction.",
        "coin_universe": "Top 20",
        "timeframe": "",
        "notes": f"Auto-proposed by ML Evolution Review from hypothesis {hypothesis['slug']}.",
    }
    candidate_id = create_development_candidate(payload)
    update_development_candidate(candidate_id, generation_trigger="ml_evolution_review")

    # Family/genealogy is tracked in the ML registry + lineage (the dev_candidates
    # table has no family column).
    dev_slug = _ml_dev_slug_for_candidate(candidate_id) or registry_slug(name)
    upsert_ml_strategy(
        dev_slug, name, kind="dev", source_team_id=dev_slug, source_db_path="",
        family_slug=family_slug, classification="ML-proposed hybrid", active=1,
    )
    ml_lineage_add(dev_slug, a, "hybrid", f"ML hybrid inheriting from {name_a}.")
    ml_lineage_add(dev_slug, b, "hybrid", f"ML hybrid inheriting from {name_b}.")

    # Generate + validate the strategy file, then stop at the human-review gate.
    try:
        queue_candidate_strategy_generation(
            candidate_id, generation_trigger="ml_evolution_review", auto_apply_generated_strategy=False
        )
    except Exception as exc:  # noqa: BLE001
        log_maintenance("ml", "warning", f"Descendant {dev_slug} created but generation queue failed: {exc}")

    with closing(get_db()) as conn:
        conn.execute(
            "UPDATE ml_descendant_hypotheses SET status = 'promoted', updated_at = ? WHERE slug = ?",
            (iso_now(), hypothesis["slug"]),
        )
        conn.commit()

    return {"candidate_id": candidate_id, "dev_slug": dev_slug, "name": name,
            "parents": [name_a, name_b], "conviction": hypothesis["conviction_score"]}


def _ml_ecosystem_coverage() -> dict[str, Any]:
    """Deterministic population-level taxonomy for the Evolution Review: where
    organisms sit on the Signal Timing Spectrum (Wind Tunnel) and where the gaps are,
    the trait landscape, and which families already have descendants. Lets the review
    ask ecosystem questions ('nothing operates in Trend Maturity', 'no bridge between
    Exhaustion and Early Expansion') instead of only optimization questions."""
    registry = ml_registry_all(active_only=False)
    # Niche occupancy per spectrum phase (a band occupies every phase it spans).
    occupancy: dict[str, list[str]] = {slug: [] for slug, _ in ML_SIGNAL_TIMING_SPECTRUM}
    placed = 0
    for s in registry:
        band = ml_temporal_band(s)
        if band is None:
            continue
        placed += 1
        for idx in range(band[0], band[1] + 1):
            occupancy[ML_SIGNAL_TIMING_SPECTRUM[idx][0]].append(s["name"])
    phase_coverage = [
        {"phase": label, "organisms": occupancy[slug], "count": len(occupancy[slug])}
        for slug, label in ML_SIGNAL_TIMING_SPECTRUM
    ]
    gaps = [label for slug, label in ML_SIGNAL_TIMING_SPECTRUM if not occupancy[slug]]
    clusters = [
        {"phase": label, "count": len(occupancy[slug])}
        for slug, label in ML_SIGNAL_TIMING_SPECTRUM if len(occupancy[slug]) >= 3
    ]

    # Trait landscape across the population.
    strengths: Counter[str] = Counter()
    weaknesses: Counter[str] = Counter()
    for t in ml_traits_all():
        if t["polarity"] == "strength":
            strengths[t["trait_name"]] += 1
        elif t["polarity"] == "weakness":
            weaknesses[t["trait_name"]] += 1

    # Genealogy: which families have descendants (lineage children) vs none.
    fam_of = {s["slug"]: s.get("family_slug", "") for s in registry}
    families_with_descendants = sorted(
        {fam_of.get(edge["child_slug"], "") for edge in ml_lineage_all() if fam_of.get(edge["child_slug"])}
    )
    all_families = sorted({f["slug"] for f in ml_families_all()})
    families_without_descendants = [f for f in all_families if f not in families_with_descendants]

    return {
        "organisms_total": len(registry),
        "organisms_placed_on_spectrum": placed,
        "phase_coverage": phase_coverage,
        "spectrum_gaps": gaps,
        "spectrum_clusters": clusters,
        "common_strengths": [{"trait": k, "organisms": v} for k, v in strengths.most_common(8)],
        "common_weaknesses": [{"trait": k, "organisms": v} for k, v in weaknesses.most_common(8)],
        "families_with_descendants": families_with_descendants,
        "families_without_descendants": families_without_descendants,
    }


def _ml_review_window_aggregates(window_start: int, window_end: int) -> dict[str, Any]:
    complement_threshold = parse_float(get_setting("ml_complement_pair_threshold", "0.55"))
    names = {s["slug"]: s["name"] for s in ml_registry_all(active_only=False)}
    with closing(get_db()) as conn:
        div_rows = conn.execute(
            """
            SELECT strategy_slug, category, COUNT(*) AS hits
            FROM ml_telemetry_divergence
            WHERE cycle_id BETWEEN ? AND ? AND magnitude >= 1.0
            GROUP BY strategy_slug, category ORDER BY hits DESC LIMIT 10
            """,
            (window_start, window_end),
        ).fetchall()
        strat_rows = conn.execute(
            """
            SELECT strategy_slug, COUNT(*) AS hits
            FROM ml_telemetry_divergence
            WHERE cycle_id BETWEEN ? AND ? AND magnitude >= 1.0
            GROUP BY strategy_slug ORDER BY hits DESC LIMIT 6
            """,
            (window_start, window_end),
        ).fetchall()
        pair_rows = conn.execute(
            """
            SELECT strategy_a, strategy_b, COUNT(*) AS hits
            FROM ml_relationships
            WHERE cycle_id BETWEEN ? AND ? AND complement_score >= ?
            GROUP BY strategy_a, strategy_b ORDER BY hits DESC LIMIT 6
            """,
            (window_start, window_end, complement_threshold),
        ).fetchall()
    return {
        "recurring_divergences": [
            {"strategy": names.get(r["strategy_slug"], r["strategy_slug"]), "category": r["category"], "cycles": r["hits"]}
            for r in div_rows
        ],
        "persistent_strategies": [
            {"strategy": names.get(r["strategy_slug"], r["strategy_slug"]), "cycles": r["hits"]} for r in strat_rows
        ],
        "recurring_pairs": [
            {"pair": f"{names.get(r['strategy_a'], r['strategy_a'])} × {names.get(r['strategy_b'], r['strategy_b'])}", "cycles": r["hits"]}
            for r in pair_rows
        ],
    }


def run_ml_evolution_review() -> None:
    """15-day review. Allocates scarce development capital: 0-3 descendants, each
    gated by hard conviction thresholds. Zero is a valid, explained outcome."""
    started = iso_now()
    conv_threshold = parse_float(get_setting("ml_descendant_conviction_threshold", "0.75"))
    min_evidence = max(1, int(parse_float(get_setting("ml_descendant_min_evidence_cycles", "8"))))
    novelty_threshold = parse_float(get_setting("ml_descendant_novelty_threshold", "0.35"))
    max_per = max(0, int(parse_float(get_setting("ml_descendant_max_per_review", "3"))))

    with closing(get_db()) as conn:
        last = conn.execute(
            "SELECT cycle_window_end FROM ml_evolution_reviews ORDER BY id DESC LIMIT 1"
        ).fetchone()
        window_start = (int(last["cycle_window_end"]) + 1) if last else 0
        latest = conn.execute(
            "SELECT MAX(id) AS m FROM ml_telemetry_cycles WHERE status = 'complete'"
        ).fetchone()
        window_end = int(latest["m"] or 0)
        conn.execute(
            "INSERT INTO ml_evolution_reviews (started_at, cycle_window_start, cycle_window_end, status) VALUES (?, ?, ?, 'running')",
            (started, window_start, window_end),
        )
        review_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()

    aggregates = _ml_review_window_aggregates(window_start, window_end)
    coverage = _ml_ecosystem_coverage()
    ranked = ml_descendant_hypotheses_all(order_by_conviction=True)

    selected: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    for h in ranked:
        if h["status"] == "promoted":
            continue
        evidence_cycles = len(json.loads(h["supporting_cycle_ids_json"] or "[]"))
        gaps = []
        if h["conviction_score"] < conv_threshold:
            gaps.append(f"conviction {h['conviction_score']:.2f} < {conv_threshold:.2f}")
        if evidence_cycles < min_evidence:
            gaps.append(f"evidence {evidence_cycles}/{min_evidence} cycles")
        if h["inheritance_score"] <= 0:
            gaps.append("no identifiable inheritable strengths")
        if h["novelty_score"] < novelty_threshold:
            gaps.append(f"novelty {h['novelty_score']:.2f} < {novelty_threshold:.2f}")
        if gaps:
            if len(rejection_reasons) < 5:
                rejection_reasons.append(f"{h['slug']}: " + "; ".join(gaps))
            continue
        if len(selected) < max_per:
            selected.append(h)

    created: list[dict[str, Any]] = []
    for h in selected:
        try:
            created.append(ml_create_descendant(h))
        except Exception as exc:  # noqa: BLE001
            log_maintenance("ml", "error", f"Descendant creation failed for {h['slug']}: {exc}")

    # --- Build report (deterministic, optionally narrated by the LLM) ---
    report_lines = [f"# ML Evolution Review (cycles {window_start}-{window_end})", ""]
    if created:
        report_lines.append(f"**{len(created)} descendant(s) proposed** and parked at the Draft Room human-review gate:")
        report_lines.append("")
        for c in created:
            report_lines.append(f"- **{c['name']}** ← {c['parents'][0]} × {c['parents'][1]} (conviction {c['conviction']:.2f}, candidate #{c['candidate_id']})")
    else:
        report_lines.append("**0 descendants proposed.** No hypothesis cleared the conviction thresholds this cycle.")
        if rejection_reasons:
            report_lines.append("")
            report_lines.append("Closest misses:")
            for reason in rejection_reasons:
                report_lines.append(f"- {reason}")
    report_lines += ["", "## Recurring divergences", ""]
    for d in aggregates["recurring_divergences"][:6]:
        report_lines.append(f"- {d['strategy']} — {d['category']} ({d['cycles']} cycles)")
    report_lines += ["", "## Persistently divergent strategies", ""]
    for s in aggregates["persistent_strategies"]:
        report_lines.append(f"- {s['strategy']} ({s['cycles']} divergent readings)")
    report_lines += ["", "## Recurring complement pairs", ""]
    for p in aggregates["recurring_pairs"]:
        report_lines.append(f"- {p['pair']} ({p['cycles']} cycles)")
    report_lines += ["", "## Ecosystem coverage", ""]
    report_lines.append(
        f"{coverage['organisms_placed_on_spectrum']}/{coverage['organisms_total']} organisms placed on the Signal Timing Spectrum."
    )
    for pc in coverage["phase_coverage"]:
        who = ", ".join(pc["organisms"]) if pc["organisms"] else "—"
        report_lines.append(f"- {pc['phase']}: {pc['count']} ({who})")
    if coverage["spectrum_gaps"]:
        report_lines.append("")
        report_lines.append(f"**Uncovered phases (gaps):** {', '.join(coverage['spectrum_gaps'])}")
    if coverage["families_without_descendants"]:
        report_lines.append(f"**Families with no descendants yet:** {', '.join(coverage['families_without_descendants'])}")
    report_markdown = "\n".join(report_lines)

    if get_setting("ollama_api_key", ""):
        try:
            user_payload = {
                "charter": ML_EVOLUTION_REVIEW_CHARTER,
                "window": {"start": window_start, "end": window_end},
                "aggregates": aggregates,
                "ecosystem_coverage": coverage,
                "proposed_descendants": created,
                "closest_misses": rejection_reasons,
                "note": "proposed_descendants and all numbers are precomputed and authoritative. Do not invent numbers or add to proposed_descendants. Ambitious combination ideas belong only in the advisory 'Ambition & ecosystem' section.",
            }
            content = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the ATL strategy biology committee writing a 15-day Evolution Review — the most "
                            "imaginative organ of the lab. Internalize the provided charter: speciation over mutation, "
                            "big machines not little machines, ambitious combination over refinement. "
                            "You are given precomputed aggregates, ecosystem_coverage, and the descendants that already "
                            "passed hard thresholds. Two firm rules: (1) the proposed_descendants list and every number "
                            "are authoritative — never invent numbers or add descendants to that list. (2) Your imagination "
                            "goes into a clearly labeled advisory section, not into fabricated results. "
                            "Return strict JSON {report_markdown, summary}. report_markdown must contain: a narrative of the "
                            "cycle window (what recurred, who stayed divergent, which pairings matured, why the proposed "
                            "descendants or their absence make sense); and an '## Ambition & ecosystem' section that reads "
                            "ecosystem_coverage to name niche gaps and over-clustered phases, and proposes ambitious, "
                            "charter-aligned future directions — including multi-organ or three-plus-parent architectures — "
                            "as advice for the owner and the next review (these are recommendations, not creations). "
                            "summary: one-sentence headline."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload)},
                ]
            )
            payload = parse_json_block(content)
            if payload.get("report_markdown"):
                report_markdown = str(payload["report_markdown"])
            summary = str(payload.get("summary") or "")
        except Exception as exc:  # noqa: BLE001
            log_maintenance("ml", "warning", f"Evolution review LLM narrative failed, using deterministic report: {exc}")
            summary = f"{len(created)} descendant(s) proposed from cycles {window_start}-{window_end}."
    else:
        summary = f"{len(created)} descendant(s) proposed from cycles {window_start}-{window_end}."

    report_json = {"aggregates": aggregates, "ecosystem_coverage": coverage, "proposed_descendants": created, "closest_misses": rejection_reasons}
    with closing(get_db()) as conn:
        conn.execute(
            """
            UPDATE ml_evolution_reviews SET
                completed_at = ?, report_markdown = ?, report_json = ?,
                descendants_proposed = ?, summary = ?, status = 'complete'
            WHERE id = ?
            """,
            (iso_now(), report_markdown, json.dumps(report_json), len(created), summary, review_id),
        )
        conn.commit()
    upsert_research_index_entry(
        "ml_evolution_review",
        f"ml_evolution_review:{review_id}",
        f"ML Evolution Review {review_id} ({started})",
        report_markdown,
        "ml evolution review descendant proposal genealogy",
        entry_type="finding",
        author_type="agent",
        status="active",
    )
    set_setting("ml_evolution_review_last_run", iso_now())
    log_maintenance("ml", "success", f"Evolution review {review_id}: {len(created)} descendant(s) proposed from cycles {window_start}-{window_end}.")


# ---------------------------------------------------------------------------
# The Historian — once-a-day narrative chronicle of the league's evolution.
# Deterministic-first: Python detects the day's events and picks ONE primary
# emoji; the LLM only writes the blurb (and degrades to templated prose).
# ---------------------------------------------------------------------------

# Priority order, highest first. The day's primary emoji is the first class
# that has any events; "quiet" is the honest fallback when nothing clears a bar.
CHRONICLE_CLASSES: list[tuple[str, str, str]] = [
    ("crisis", "🔥", "Crisis Day"),
    ("coronation", "👑", "Coronation Day"),
    ("extinction", "☠️", "Extinction Day"),
    ("birth", "🐣", "Birth Day"),
    ("breakthrough", "🏆", "Breakthrough Day"),
    ("experiment", "🧪", "Experiment Day"),
    ("rivalry", "⚔️", "Rivalry Day"),
    ("growth", "🌱", "Growth Day"),
    ("quiet", "😴", "Quiet Day"),
]
CHRONICLE_EMOJI = {key: emoji for key, emoji, _ in CHRONICLE_CLASSES}
CHRONICLE_LABEL = {key: label for key, _, label in CHRONICLE_CLASSES}
CHRONICLE_RIVALRY_Z = 3.5            # robust-z magnitude that counts as a divergence flare
CHRONICLE_GROWTH_MIN_ENTRIES = 3     # new non-repo knowledge entries that count as growth


def _chronicle_local_date(ts: Any) -> Any:
    """Local calendar date of an ISO timestamp (naive treated as UTC). None if unparseable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(LOCAL_TIMEZONE).date()


def _chronicle_collect_events(day: Any) -> dict[str, list[dict[str, str]]]:
    """Deterministically gather the day's events, keyed by classification.
    `day` is a local `date`. Small playground tables — recent-row scans + an
    exact local-date filter in Python (robust to mixed UTC/offset timestamps)."""
    events: dict[str, list[dict[str, str]]] = {key: [] for key, _, _ in CHRONICLE_CLASSES}

    def on_day(ts: Any) -> bool:
        return _chronicle_local_date(ts) == day

    with closing(get_db()) as conn:
        # 🐣 birth — new descendant lineage + new dev candidates
        for row in conn.execute("SELECT child_slug, parent_slug, created_at FROM ml_lineage ORDER BY id DESC LIMIT 1000"):
            if on_day(row["created_at"]):
                events["birth"].append({"kind": "descendant", "ref": row["child_slug"],
                                        "detail": f"{row['child_slug']} was born from {row['parent_slug']}"})
        for row in conn.execute("SELECT slug, name, created_at FROM dev_candidates ORDER BY id DESC LIMIT 1000"):
            if on_day(row["created_at"]):
                events["birth"].append({"kind": "candidate", "ref": row["slug"],
                                        "detail": f"{row['name']} entered the Development League"})
        # ☠️ extinction — archived candidates
        for row in conn.execute("SELECT slug, name, archived_at FROM dev_candidates WHERE archived_at IS NOT NULL ORDER BY id DESC LIMIT 1000"):
            if on_day(row["archived_at"]):
                events["extinction"].append({"kind": "archived", "ref": row["slug"],
                                             "detail": f"{row['name']} was archived"})
        # 🧪 experiment — evolution reviews + newly opened research threads
        for row in conn.execute("SELECT id, started_at, descendants_proposed FROM ml_evolution_reviews ORDER BY id DESC LIMIT 200"):
            if on_day(row["started_at"]):
                events["experiment"].append({"kind": "evolution_review", "ref": str(row["id"]),
                                             "detail": f"Evolution review #{row['id']} ran ({row['descendants_proposed']} descendants proposed)"})
        for row in conn.execute("SELECT id, question, created_at FROM research_threads ORDER BY id DESC LIMIT 500"):
            if on_day(row["created_at"]):
                events["experiment"].append({"kind": "research_open", "ref": str(row["id"]),
                                             "detail": f"New research question opened: {row['question']}"})
        # 🏆 breakthrough — research threads that reached a conclusion
        for row in conn.execute("SELECT id, question, completed_at FROM research_threads WHERE completed_at IS NOT NULL ORDER BY id DESC LIMIT 500"):
            if on_day(row["completed_at"]):
                events["breakthrough"].append({"kind": "research_conclusion", "ref": str(row["id"]),
                                               "detail": f"Research concluded: {row['question']}"})
        # ⚔️ rivalry — sharp telemetry divergences from cycles that ran today
        cycle_ids = [int(row["id"]) for row in conn.execute("SELECT id, started_at FROM ml_telemetry_cycles ORDER BY id DESC LIMIT 50")
                     if on_day(row["started_at"])]
        if cycle_ids:
            placeholders = ",".join("?" * len(cycle_ids))
            for row in conn.execute(
                f"SELECT strategy_slug, category, robust_z, magnitude FROM ml_telemetry_divergence "
                f"WHERE cycle_id IN ({placeholders}) AND magnitude >= ? ORDER BY magnitude DESC LIMIT 12",
                (*cycle_ids, CHRONICLE_RIVALRY_Z),
            ):
                events["rivalry"].append({"kind": "divergence", "ref": row["strategy_slug"],
                                          "detail": f"{row['strategy_slug']} diverged sharply on {row['category']} (z={float(row['robust_z']):.1f})"})
        # 🔥 crisis — maintenance errors/warnings
        for row in conn.execute("SELECT maintenance_type, status, message, created_at FROM maintenance_runs WHERE status IN ('error', 'warning') ORDER BY id DESC LIMIT 1000"):
            if on_day(row["created_at"]):
                events["crisis"].append({"kind": f"maintenance_{row['status']}", "ref": row["maintenance_type"],
                                         "detail": f"{row['maintenance_type']}: {str(row['message'])[:160]}"})
        # 👑 coronation — promotion events in the dev runtime log
        for row in conn.execute("SELECT candidate_id, event_type, title, created_at FROM dev_runtime_events ORDER BY id DESC LIMIT 1000"):
            if on_day(row["created_at"]) and ("promot" in str(row["title"]).lower() or str(row["event_type"]).lower() == "promotion"):
                events["coronation"].append({"kind": "promotion", "ref": str(row["candidate_id"]),
                                             "detail": str(row["title"])})
        # 🌱 growth — genuine (non-repo) knowledge entries added/refreshed today
        knowledge_today = sum(1 for row in conn.execute(
            "SELECT updated_at FROM research_index_entries WHERE source_type NOT IN ('repo') ORDER BY id DESC LIMIT 2000")
            if on_day(row["updated_at"]))
        knowledge_total = int(conn.execute("SELECT COUNT(*) FROM research_index_entries WHERE source_type NOT IN ('repo')").fetchone()[0])
    if knowledge_today >= CHRONICLE_GROWTH_MIN_ENTRIES:
        events["growth"].append({"kind": "index_growth", "ref": str(knowledge_total),
                                 "detail": f"{knowledge_today} knowledge entries added or refreshed (index now {knowledge_total})"})
    return events


def _chronicle_classify(events: dict[str, list[dict[str, str]]]) -> str:
    for key, _, _ in CHRONICLE_CLASSES:
        if key == "quiet":
            return "quiet"
        if events.get(key):
            return key
    return "quiet"


def _chronicle_fallback_blurb(day: Any, classification: str, events: dict[str, list[dict[str, str]]]) -> tuple[str, str]:
    """Deterministic title + blurb used when no LLM is configured (or it fails)."""
    label = CHRONICLE_LABEL.get(classification, "Quiet Day")
    try:
        pretty = day.strftime("%B %d, %Y").replace(" 0", " ")  # Windows-safe; strips leading zero
    except (AttributeError, ValueError):
        pretty = str(day)
    primary = events.get(classification, [])
    if classification == "quiet":
        blurb = "A quiet day in the league. The bots kept their shifts, the indexers kept indexing, and the record turned a page without incident."
        return f"{pretty} — A Quiet Watch", blurb
    lead = primary[0]["detail"] if primary else label
    extra_count = sum(len(v) for k, v in events.items() if v) - 1
    pieces = [lead.rstrip(".") + "."]
    if extra_count > 0:
        pieces.append(f"{extra_count} other notable event{'s' if extra_count != 1 else ''} were logged across the league.")
    return f"{pretty} — {label}", " ".join(pieces)


def write_chronicle_day(day: Any) -> dict[str, Any]:
    """Classify `day` (a local date) deterministically, then write its chapter.
    Consults the historical record (prior chronicle + the index) before the LLM
    narrates, so the blurb is grounded in evidence rather than assumption."""
    events = _chronicle_collect_events(day)
    classification = _chronicle_classify(events)
    emoji = CHRONICLE_EMOJI.get(classification, "😴")
    event_list = [e for evs in events.values() for e in evs]

    # Consult the record first: prior chapters + index hits on the day's keywords.
    focus_terms = " ".join(sorted({e["ref"] for e in event_list} | {classification}))
    index_hits = search_research_index(focus_terms, limit=6) if focus_terms.strip() else []
    prior_days = recent_chronicle_days(7)

    title, blurb = _chronicle_fallback_blurb(day, classification, events)
    llm_used = 0
    if get_setting("ollama_api_key", "") and event_list:
        try:
            user_payload = {
                "date": str(day),
                "decided_classification": classification,
                "decided_emoji": emoji,
                "todays_events": event_list,
                "recent_chronicle": [{"date": d["chronicle_date"], "emoji": d["emoji"], "title": d["title"]} for d in prior_days],
                "related_record": [{"title": h.get("title", ""), "content": str(h.get("content", ""))[:600]} for h in index_hits],
                "note": "The classification and emoji are already decided deterministically — do not change them. Narrate ONLY from the supplied events and record; invent nothing.",
            }
            content = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the ATL Historian. Your job is to record what happened — not to predict "
                            "or opine. You are handed today's deterministically-detected events, the recent "
                            "chronicle, and related indexed records. Place today in the context of the record. "
                            "Be conservative and factual; never invent events, numbers, or names not present in "
                            "the input. Return strict JSON with keys title and blurb. title: a short evocative "
                            "chapter title (<= 80 chars). blurb: one tight paragraph (2-4 sentences) telling the "
                            "story of the day, grounded only in the supplied events and record."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload)},
                ]
            )
            payload = parse_json_block(content)
            if payload.get("title"):
                title = str(payload["title"]).strip()[:160]
            if payload.get("blurb"):
                blurb = str(payload["blurb"]).strip()
            llm_used = 1
        except Exception as exc:  # noqa: BLE001
            log_maintenance("chronicle", "warning", f"Chronicle LLM call failed, using deterministic blurb: {exc}")

    now = iso_now()
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO chronicle_days (chronicle_date, emoji, classification, title, blurb, event_refs, llm_used, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chronicle_date) DO UPDATE SET
                emoji=excluded.emoji, classification=excluded.classification, title=excluded.title,
                blurb=excluded.blurb, event_refs=excluded.event_refs, llm_used=excluded.llm_used,
                updated_at=excluded.updated_at
            """,
            (str(day), emoji, classification, title, blurb, json.dumps(events), llm_used, now, now),
        )
        conn.commit()

    # Index today's chapter so tomorrow's Historian can consult it.
    upsert_research_index_entry(
        "chronicle",
        f"chronicle:{day}",
        f"{emoji} {title}",
        blurb,
        f"chronicle history {classification} " + " ".join(sorted({e['ref'] for e in event_list})),
        entry_type="observation",
        author_type="system",
        status="active",
    )
    return {
        "chronicle_date": str(day), "emoji": emoji, "classification": classification,
        "title": title, "blurb": blurb, "event_count": len(event_list), "llm_used": llm_used,
    }


def recent_chronicle_days(limit: int = 7) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT chronicle_date, emoji, classification, title, blurb, event_refs, llm_used, updated_at "
            "FROM chronicle_days ORDER BY chronicle_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def all_chronicle_days() -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT chronicle_date, emoji, classification, title, blurb, event_refs, llm_used, updated_at "
            "FROM chronicle_days ORDER BY chronicle_date ASC"
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["events"] = [e for evs in json.loads(item.get("event_refs") or "{}").values() for e in evs]
        except (json.JSONDecodeError, AttributeError):
            item["events"] = []
        result.append(item)
    return result


def chronicle_due() -> bool:
    """Fire once per local calendar day, on/after the configured wall-clock time."""
    if get_setting("chronicle_enabled", "true").lower() != "true":
        return False
    now_local = local_now()
    raw = get_setting("chronicle_run_time", "23:11").strip()
    try:
        hh, mm = (int(part) for part in raw.split(":", 1))
    except (ValueError, TypeError):
        hh, mm = 23, 11
    if (now_local.hour, now_local.minute) < (hh, mm):
        return False
    return get_setting("chronicle_last_date", "") != now_local.date().isoformat()


def run_chronicle_cycle(target_day: Any = None) -> dict[str, Any]:
    """Write the chronicle for `target_day` (default: today, local) and mark it done."""
    day = target_day or local_now().date()
    record = write_chronicle_day(day)
    set_setting("chronicle_last_run", iso_now())
    set_setting("chronicle_last_date", day.isoformat() if hasattr(day, "isoformat") else str(day))
    log_maintenance("chronicle", "success",
                    f"Chronicle {record['chronicle_date']}: {record['emoji']} {record['classification']} "
                    f"({record['event_count']} events, llm_used={record['llm_used']}).")
    return record


def run_research_maintenance() -> None:
    processed = 0
    for thread in due_research_threads()[:3]:
        thread_id = int(thread["id"])
        started_at = datetime.fromisoformat(str(thread["started_at"]))
        expires_at = started_at + timedelta(hours=int(thread.get("duration_hours") or 12))
        # QF4: a 12h thread at a 30-min cadence produces ~24 updates plus any manual
        # notes; the old limit of 12 made a thread forget its own early reasoning
        # halfway through its window. 48 covers a full window with headroom.
        updates = list_research_thread_updates(thread_id, limit=48)
        question = str(thread.get("question", ""))
        focus = str(thread.get("latest_focus") or question)
        index_hits = search_research_index(focus, limit=6)
        relevant_files = search_workspace_files(tokenize_search(question), limit=6)
        citations = [hit.get("title", "") for hit in index_hits] + [item.get("path", "") for item in relevant_files]
        if utc_now() >= expires_at:
            summary_prompt = {
                "question": question,
                "thread": thread,
                "updates": updates,
                "index_hits": index_hits,
                "relevant_files": relevant_files,
                "standings": standings_rows(),
                "development_league": development_research_brief(),
                "note": "Summarize what this local research thread learned using only supplied local context. Propose one next question."
            }
            content = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a local research summarizer for an algo trading playground. "
                            "Use only the supplied indexed local research, repo excerpts, and live-site context. "
                            "Return strict JSON with keys title, summary, next_question, keywords."
                        ),
                    },
                    {"role": "user", "content": json.dumps(summary_prompt)},
                ]
            )
            payload = parse_json_block(content)
            summary = payload.get("summary", "")
            add_research_thread_update(thread_id, "summary", payload.get("title", "12 Hour Summary"), summary, "agent", citations[:8])
            update_research_thread_state(
                thread_id,
                status="completed",
                completed_at=iso_now(),
                summary=summary,
                latest_focus=payload.get("next_question", focus),
                next_run_at=(utc_now() + timedelta(days=3650)).isoformat(),
            )
            next_question = str(payload.get("next_question", "")).strip()
            if next_question and int(thread.get("auto_reseed") or 0):
                # QF5: don't let auto-reseed loop on a question we've effectively just
                # asked. Compare against recent thread questions; if a near-duplicate
                # exists, suppress the reseed and log it so the repeat is detectable.
                recent_questions = [str(row.get("question", "")) for row in list_research_threads(limit=24)]
                duplicate_of = next(
                    (q for q in recent_questions if question_similarity(next_question, q) >= 0.8),
                    "",
                )
                if duplicate_of:
                    log_maintenance(
                        "research",
                        "skipped",
                        f"Suppressed duplicate reseed (≈ prior question): {next_question[:120]}",
                    )
                else:
                    create_research_thread(
                        next_question,
                        owner="agent",
                        scope=str(thread.get("scope") or "research"),
                        auto_reseed=True,
                        interval_minutes=int(thread.get("interval_minutes") or 30),
                        duration_hours=int(thread.get("duration_hours") or 12),
                    )
        else:
            pull_prompt = {
                "question": question,
                "current_focus": focus,
                "thread": thread,
                "updates": updates,
                "index_hits": index_hits,
                "relevant_files": relevant_files,
                "standings": standings_rows(),
                "power_rankings": compute_power_rankings(),
                "development_league": development_research_brief(),
                "ml_hypotheses": merged_ml_hypotheses(),
                "ml_draft_board": merged_ml_draft_board(),
                "note": "Pull the thread forward using only local indexed research, repo excerpts, and live site evidence. Do not browse the web or pretend to know external facts."
            }
            content = ollama_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a repo-aware research agent inside a local algo trading playground. "
                            "Search only the supplied local research index, repo excerpts, and live-site context. "
                            "Return strict JSON with keys title, focus_question, content, keywords."
                        ),
                    },
                    {"role": "user", "content": json.dumps(pull_prompt)},
                ]
            )
            payload = parse_json_block(content)
            focus_question = payload.get("focus_question", focus)
            body = payload.get("content", "")
            if focus_question:
                body = f"Focus question: {focus_question}\n\n{body}"
            add_research_thread_update(thread_id, "agent_update", payload.get("title", "Thread Pull"), body, "agent", citations[:8])
            update_research_thread_state(
                thread_id,
                latest_focus=focus_question,
                next_run_at=(utc_now() + timedelta(minutes=int(thread.get("interval_minutes") or 30))).isoformat(),
            )
        processed += 1
    if processed:
        log_maintenance("research", "success", f"Processed {processed} research thread updates.")


def maintenance_due(setting_key: str, minutes_setting_key: str) -> bool:
    enabled_key = {
        "league_maintenance_last_run": "league_maintenance_enabled",
        "ml_maintenance_last_run": "ml_maintenance_enabled",
        "ml_evolution_review_last_run": "ml_evolution_review_enabled",
        "archive_last_run": "archive_maintenance_enabled",
        "pairlist_manifest_last_run": "resource_governance_enabled",
    }.get(setting_key, "league_maintenance_enabled")
    if get_setting(enabled_key, "true").lower() != "true":
        return False
    last_run = get_setting(setting_key, "")
    interval_minutes = int(get_setting(minutes_setting_key, "30") or "30")
    if not last_run:
        return True
    try:
        last_dt = datetime.fromisoformat(last_run)
    except ValueError:
        return True
    return (utc_now() - last_dt).total_seconds() >= interval_minutes * 60


def maintenance_loop() -> None:
    while True:
        try:
            run_season_turnover()
        except Exception as exc:  # noqa: BLE001
            log_maintenance("background", "error", f"Season turnover failed: {exc}")
        if maintenance_due("archive_last_run", "archive_maintenance_minutes"):
            try:
                run_archive_maintenance()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"Archive maintenance failed: {exc}")
        # The ML biology lab is deterministic-first and runs with or without an LLM key.
        if maintenance_due("ml_maintenance_last_run", "ml_maintenance_minutes"):
            try:
                run_ml_telemetry_cycle()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"ML telemetry cycle failed: {exc}")
        if maintenance_due("ml_evolution_review_last_run", "ml_evolution_review_minutes"):
            try:
                run_ml_evolution_review()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"ML evolution review failed: {exc}")
        # The Historian is deterministic-first too — it writes today's chapter with
        # or without an LLM, once per day on/after the configured wall-clock time.
        if chronicle_due():
            try:
                run_chronicle_cycle()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"Chronicle cycle failed: {exc}")
        # ATL External Resource Governance: refresh canonical universes + exchange
        # manifests on a controlled cadence (one central API pull, not per-bot).
        if maintenance_due("pairlist_manifest_last_run", "pairlist_manifest_minutes"):
            try:
                run_pairlist_manifest_cycle()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"Pairlist manifest cycle failed: {exc}")
        if get_setting("ollama_api_key", ""):
            if maintenance_due("league_maintenance_last_run", "league_maintenance_minutes"):
                try:
                    run_league_maintenance()
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("background", "error", f"League maintenance failed: {exc}")
            if get_setting("research_agent_enabled", "true").lower() == "true":
                try:
                    run_research_maintenance()
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("background", "error", f"Research maintenance failed: {exc}")
        time.sleep(60)


# ===========================================================================
# Operations Schedule / League Clock — a read-only visibility layer over every
# recurring job this process runs. It introspects the SAME settings, gates, and
# tables the loops above use, so it reports real cadence and real last/next runs
# (never design copy). Adding a new scheduled job? Add a row to schedule_jobs().
# ===========================================================================


def _schedule_parse_dt(value: str) -> datetime | None:
    """Parse an ISO timestamp (naive treated as UTC) into an aware datetime."""
    moment = resolve_optional_datetime(str(value or ""))
    if moment is None:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _schedule_fmt(value: Any) -> str:
    """Render a stored UTC timestamp in local league time, or an em dash."""
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        moment = _schedule_parse_dt(str(value or ""))
    if moment is None:
        text = str(value or "").strip()
        return text or "—"
    return moment.astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")


def _schedule_last_success(maintenance_type: str) -> str:
    with closing(get_db()) as conn:
        row = conn.execute(
            "SELECT created_at FROM maintenance_runs WHERE maintenance_type = ? "
            "AND status = 'success' ORDER BY id DESC LIMIT 1",
            (maintenance_type,),
        ).fetchone()
    return str(row["created_at"]) if row else ""


def _schedule_next_interval(last_run: str, minutes: int, enabled: bool) -> str:
    """Next fire time for an interval-gated maintenance job (maintenance_due logic)."""
    if not enabled:
        return "disabled"
    last_dt = _schedule_parse_dt(last_run)
    if last_dt is None:
        return "due now"
    nxt = last_dt + timedelta(minutes=minutes)
    return "due now" if nxt <= utc_now() else _schedule_fmt(nxt)


def schedule_jobs() -> list[dict[str, Any]]:
    """The live recurring-job inventory. Every field is read from real settings,
    gate logic, and tables — not from prose. Used by both the page and the export."""
    llm_on = bool(get_setting("ollama_api_key", ""))

    # --- Interval settings (minutes) the maintenance loop actually reads ---
    league_min = int(parse_float(get_setting("league_maintenance_minutes", "30")) or 30)
    ml_min = int(parse_float(get_setting("ml_maintenance_minutes", "180")) or 180)
    evo_min = int(parse_float(get_setting("ml_evolution_review_minutes", "21600")) or 21600)
    archive_min = int(parse_float(get_setting("archive_maintenance_minutes", "720")) or 720)
    research_interval = int(parse_float(get_setting("research_agent_interval_minutes", "30")) or 30)
    research_duration = int(parse_float(get_setting("research_agent_duration_hours", "12")) or 12)
    chronicle_time = get_setting("chronicle_run_time", "23:11")

    league_enabled = get_setting("league_maintenance_enabled", "true").lower() == "true"
    ml_enabled = get_setting("ml_maintenance_enabled", "true").lower() == "true"
    evo_enabled = get_setting("ml_evolution_review_enabled", "true").lower() == "true"
    archive_enabled = get_setting("archive_maintenance_enabled", "true").lower() == "true"
    research_enabled = get_setting("research_agent_enabled", "true").lower() == "true"
    chronicle_enabled = get_setting("chronicle_enabled", "true").lower() == "true"

    # --- Research-thread state (drives the research agent next-run) ---
    with closing(get_db()) as conn:
        research_next = conn.execute(
            "SELECT MIN(next_run_at) AS n FROM research_threads WHERE status = 'active'"
        ).fetchone()["n"]
        research_last = conn.execute(
            "SELECT MAX(created_at) AS m FROM research_thread_updates"
        ).fetchone()["m"]
        active_threads = int(conn.execute(
            "SELECT COUNT(*) FROM research_threads WHERE status = 'active'"
        ).fetchone()[0])

    # --- Season / quarter cadence (event-driven, not interval) ---
    current_season = current_league_season()
    season_end = str(current_season.get("ended_at") or "")
    quarterly_reports = list_quarterly_reports(limit=1)
    quarterly_last = str(quarterly_reports[0].get("created_at") or "") if quarterly_reports else ""
    next_quarter = next_quarterly_review_timing()

    # --- Chronicle next fire (once/day at wall-clock time) ---
    chronicle_last_date = get_setting("chronicle_last_date", "")
    if not chronicle_enabled:
        chronicle_next = "disabled"
    elif chronicle_due():
        chronicle_next = "due now"
    else:
        today_done = chronicle_last_date == local_now().date().isoformat()
        base_day = local_now().date() + (timedelta(days=1) if today_done else timedelta(0))
        chronicle_next = f"{base_day.isoformat()} {chronicle_time} ({str(LOCAL_TIMEZONE)})"

    yes, no, opt = "Yes", "No", "Optional"

    return [
        # ---------------- Intraday background threads ----------------
        {
            "name": "Live Sync / Trade Ingestion", "category": "Intraday",
            "cadence": f"Every {POLL_INTERVAL_SECONDS}s ({POLL_INTERVAL_SECONDS // 60} min)",
            "trigger": "Daemon thread (sync_loop), continuous",
            "last_run": "continuous", "next_run": "continuous",
            "source": "sync_loop → run_sync (main.py:7195)",
            "writes_to": "live_snapshots, team_trades",
            "uses_llm": no, "status": "Active",
            "notes": "Polls each major-league bot's REST API: live state, heartbeat, and trade ingestion. Also runs once on startup and behind the manual Sync button.",
        },
        {
            "name": "Dev Shift Scheduler", "category": "Intraday",
            "cadence": f"Every {DEV_SCHEDULER_INTERVAL_SECONDS}s",
            "trigger": "Daemon thread (development_scheduler_loop), continuous",
            "last_run": "continuous", "next_run": "continuous",
            "source": "development_scheduler_loop → sync_development_pipeline (main.py:4111)",
            "writes_to": "dev_candidates, dev_runtime_events, dev_shift_episodes",
            "uses_llm": no, "status": "Active",
            "notes": "Auto starts/stops candidate containers per shift window, checks heartbeats, force-closes + archives the episode at the shift bell, and wipes the runtime DB at a clean next-shift start.",
        },
        {
            "name": "Dev Strategy Generation Queue", "category": "Intraday",
            "cadence": f"Every {DEV_GENERATION_INTERVAL_SECONDS}s (polls queue)",
            "trigger": "Any dev_candidate with generation_status = 'queued'",
            "last_run": "continuous", "next_run": "continuous",
            "source": "development_generation_loop → process_generation_queue (main.py:3714)",
            "writes_to": "strategy .py files, dev_candidates",
            "uses_llm": f"{yes} ({get_setting('development_strategy_generation_model', 'kimi-k2.6:cloud')})",
            "status": "Active",
            "notes": "Generates one queued strategy file at a time. Idle (no LLM call) unless something is queued — e.g. a new Draft Room idea or an evolution-review descendant.",
        },
        {
            "name": "Backtesting Department", "category": "Intraday",
            "cadence": f"Every 60s; one lane, paced (min-gap {int(parse_float(get_setting('backtest_department_min_gap_minutes', '2')) or 2)} min)",
            "trigger": "department enabled, not paused, lane idle, gap elapsed",
            "last_run": _schedule_fmt(get_setting("backtest_department_last_run", "")),
            "next_run": "continuous" if (department_enabled() and not department_paused()) else ("paused" if department_paused() else "disabled"),
            "source": "backtesting_department_loop → run_department_once (main.py)",
            "writes_to": "backtest_jobs, backtest_results, backtest_lanes",
            "uses_llm": no, "status": "Active" if department_enabled() else "Disabled",
            "notes": "Continuously keeps backtest evidence fresh by priority bucket (health → habitat). Runs one freqtrade backtest at a time. Evidence only — never modifies live behavior.",
        },
        # ---------------- Interval-gated maintenance loop ----------------
        {
            "name": "League Maintenance (front-page AI)", "category": "Intraday",
            "cadence": f"Every {league_min} min",
            "trigger": "maintenance_due + requires Ollama key",
            "last_run": _schedule_fmt(get_setting("league_maintenance_last_run", "")),
            "next_run": _schedule_next_interval(get_setting("league_maintenance_last_run", ""), league_min, league_enabled and llm_on),
            "source": "run_league_maintenance (main.py:9572)",
            "writes_to": "generated_content: league_overview, ai_research_questions, power_ranking_overrides",
            "uses_llm": f"{yes} (required)",
            "status": "Active" if (league_enabled and llm_on) else ("Idle — no Ollama key" if league_enabled else "Disabled"),
            "notes": "Regenerates the dashboard overview blurb, AI scouting questions, and power-ranking trust/quality overlays. Replaces (never appends) prior AI output.",
        },
        {
            "name": "Research Agent", "category": "Intraday",
            "cadence": f"Per thread every {research_interval} min, then concludes after {research_duration}h",
            "trigger": "research_threads.next_run_at <= now + requires Ollama key & toggle",
            "last_run": _schedule_fmt(research_last),
            "next_run": (_schedule_fmt(research_next) if (research_enabled and llm_on) else ("disabled" if not research_enabled else "Idle — no Ollama key")),
            "source": "run_research_maintenance (main.py:10886)",
            "writes_to": "research_threads, research_thread_updates, research_index_entries",
            "uses_llm": f"{yes} (required)",
            "status": ("Active" if (research_enabled and llm_on) else ("Idle — no Ollama key" if research_enabled else "Disabled")) + f" · {active_threads} active thread(s)",
            "notes": f"Pulls the active question forward every {research_interval} min for ~{research_duration}h, then writes a summary, proposes a next question, and auto-reseeds a fresh thread. Up to 3 threads per tick.",
        },
        {
            "name": "ML Biology / Telemetry Cycle", "category": "Multi-hour",
            "cadence": f"Every {ml_min} min ({ml_min / 60:.0f}h)",
            "trigger": "maintenance_due (runs with or without LLM)",
            "last_run": _schedule_fmt(get_setting("ml_maintenance_last_run", "")),
            "next_run": _schedule_next_interval(get_setting("ml_maintenance_last_run", ""), ml_min, ml_enabled),
            "source": "run_ml_telemetry_cycle (main.py:10103)",
            "writes_to": "ml_telemetry_cycles, ml_traits, ml relationships/divergence, research_index_entries",
            "uses_llm": opt,
            "status": "Active" if ml_enabled else "Disabled",
            "notes": "Deterministic-first: Python measures telemetry, divergence, relationships, descendant conviction and the Fearsome Five. LLM only writes the interpretive essay (templated fallback when no key).",
        },
        {
            "name": "Archive Maintenance / Backup", "category": "Multi-hour",
            "cadence": f"Every {archive_min} min ({archive_min / 60:.0f}h)",
            "trigger": "maintenance_due",
            "last_run": _schedule_fmt(get_setting("archive_last_run", "")),
            "next_run": _schedule_next_interval(get_setting("archive_last_run", ""), archive_min, archive_enabled),
            "source": "run_archive_maintenance (main.py:9504)",
            "writes_to": "data/archives snapshot + git push to archive repo",
            "uses_llm": no,
            "status": "Active" if archive_enabled else "Disabled",
            "notes": "Exports a scrubbed sqlite/json snapshot and pushes it to the configured archive git repo (push best-effort).",
        },
        # ---------------- Multi-day ----------------
        {
            "name": "ML Evolution Review", "category": "Multi-day",
            "cadence": f"Every {evo_min} min ({evo_min / 1440:.0f} days)",
            "trigger": "maintenance_due",
            "last_run": _schedule_fmt(get_setting("ml_evolution_review_last_run", "")),
            "next_run": _schedule_next_interval(get_setting("ml_evolution_review_last_run", ""), evo_min, evo_enabled),
            "source": "run_ml_evolution_review (main.py:10450)",
            "writes_to": "ml_evolution_reviews, dev_candidates (descendants), ml_lineage",
            "uses_llm": opt,
            "status": "Active" if evo_enabled else "Disabled",
            "notes": "Allocates scarce dev capital: 0–3 descendants, each gated by hard conviction/evidence/novelty thresholds. Approved crosses become dev_candidates parked at the Draft Room human-review gate. Zero is a valid outcome. LLM only narrates.",
        },
        {
            "name": "Chronicle (Historian)", "category": "Daily",
            "cadence": f"Once per day at {chronicle_time} local",
            "trigger": "chronicle_due (wall-clock, once per calendar day)",
            "last_run": _schedule_fmt(get_setting("chronicle_last_run", "")),
            "next_run": chronicle_next,
            "source": "run_chronicle_cycle (main.py:10874)",
            "writes_to": "chronicle, research_index_entries",
            "uses_llm": opt,
            "status": "Active" if chronicle_enabled else "Disabled",
            "notes": "Deterministic-first: Python classifies the day (crisis/coronation/birth/…) and picks one emoji; the LLM only writes the blurb (templated fallback otherwise).",
        },
        # ---------------- Monthly / seasonal ----------------
        {
            "name": "Season Turnover", "category": "Seasonal",
            "cadence": "Monthly (when a season's end date passes)",
            "trigger": "seasons_pending_turnover — season ended & not yet processed; checked every 60s",
            "last_run": _schedule_fmt(_schedule_last_success("season")),
            "next_run": _schedule_fmt(season_end) + " (current season close)",
            "source": "run_season_turnover (main.py:7910)",
            "writes_to": "league_seasons, league_team_season_reviews, season_draft_recommendations, strategy_awards (trophy shelves)",
            "uses_llm": no,
            "status": "Active",
            "notes": "Processes each ended season: previews/locks reviews, builds season awards, draft-night recommendations from the 12-hour pool, and grants permanent trophy-shelf awards. Then runs the quarterly check. (No standalone draft-night job — intake happens via approval-gated season draft recs.)",
        },
        # ---------------- Quarterly / long-term ----------------
        {
            "name": "Quarterly Champion Report", "category": "Quarterly",
            "cadence": "Every 3 completed major-league seasons",
            "trigger": "maybe_generate_quarterly_reports — fires from season turnover when a quarter's 3rd season closes",
            "last_run": _schedule_fmt(quarterly_last),
            "next_run": (_schedule_fmt(next_quarter.get("closes_at")) + f" (Quarter {next_quarter.get('quarter_number')})") if next_quarter.get("closes_at") else f"Quarter {next_quarter.get('quarter_number')} (date pending)",
            "source": "maybe_generate_quarterly_reports → generate_quarterly_report (main.py:8221)",
            "writes_to": "league_quarterly_reports",
            "uses_llm": opt,
            "status": "Active",
            "notes": "Capital-eligibility review using peer-relative percentile scoring and 20/30/50 recency weights. NOT a live race — a report only exists once a quarter closes, then is a permanent archived artifact. Capital League is not built; winning ≠ deploying real money.",
        },
    ]


SCHEDULE_CATEGORY_ORDER = ["Intraday", "Multi-hour", "Daily", "Multi-day", "Seasonal", "Quarterly"]


def operations_schedule_context() -> dict[str, Any]:
    jobs = schedule_jobs()
    by_category: dict[str, list[dict[str, Any]]] = {cat: [] for cat in SCHEDULE_CATEGORY_ORDER}
    for job in jobs:
        by_category.setdefault(job["category"], []).append(job)
    grouped = [{"category": cat, "jobs": by_category[cat]} for cat in SCHEDULE_CATEGORY_ORDER if by_category.get(cat)]
    return {
        "jobs": jobs,
        "grouped": grouped,
        "generated_at": _schedule_fmt(utc_now()),
        "llm_configured": bool(get_setting("ollama_api_key", "")),
        "export_path": "docs/ATL_OPERATIONAL_CADENCE.md",
    }


def build_operations_cadence_markdown() -> str:
    """Live markdown export of the operational cadence — the same data the page shows,
    plus the three-level summary and a Commissioner Expectations section."""
    ctx = operations_schedule_context()
    jobs = ctx["jobs"]
    lines: list[str] = []
    lines.append("# ATL Operational Cadence")
    lines.append("")
    lines.append(f"_Generated {ctx['generated_at']} · Ollama key configured: {'yes' if ctx['llm_configured'] else 'no'}._")
    lines.append("")
    lines.append("Recurring operational map of the Algo Trading League: what the system does automatically "
                 "if left running for hours, days, weeks, or seasons. Last/next-run values are read live from "
                 "settings and tables. **Implemented behavior is separated from intended design** — everything "
                 "in the table below is code that actually runs in this process.")
    lines.append("")

    # --- Full table ---
    lines.append("## Full job table")
    lines.append("")
    header = ["Name", "Category", "Cadence", "Trigger", "Last run", "Next run", "Source", "Writes to", "LLM", "Status", "Notes"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for j in jobs:
        cells = [
            j["name"], j["category"], j["cadence"], j["trigger"], str(j["last_run"]), str(j["next_run"]),
            j["source"], j["writes_to"], j["uses_llm"], j["status"], j["notes"],
        ]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines.append("")

    # --- Three-level summary ---
    lines.append("## Daily / intraday")
    lines.append("")
    lines.append(f"- **Live sync / trade ingestion** — every {POLL_INTERVAL_SECONDS // 60} min: bot REST polling, heartbeat, trade ingestion (no LLM).")
    lines.append(f"- **Dev shift scheduler** — every {DEV_SCHEDULER_INTERVAL_SECONDS}s: auto start/stop of candidate containers per shift window, episode archive at the bell (no LLM).")
    lines.append(f"- **Dev strategy generation** — every {DEV_GENERATION_INTERVAL_SECONDS}s when the queue is non-empty: generates strategy files (Kimi, LLM).")
    lines.append("- **League maintenance** — every 30 min *(requires Ollama key)*: front-page overview, AI questions, ranking overlays.")
    lines.append("- **Research agent** — every 30 min per active thread for ~12h, then concludes & reseeds *(requires Ollama key)*.")
    lines.append("- **ML biology / telemetry** — every 3 hours: deterministic telemetry + Fearsome Five essay (LLM optional).")
    lines.append("- **Chronicle** — once per day at the configured time: day classification + blurb (LLM optional).")
    lines.append("")
    lines.append("## Multi-day")
    lines.append("")
    lines.append("- **Archive maintenance / backup** — every 12 hours: scrubbed snapshot + git push (no LLM).")
    lines.append("- **ML evolution review** — every 15 days: 0–3 threshold-gated descendants parked at the Draft Room gate (LLM narrates only).")
    lines.append("- **Candidate assessment windows** — every 6h (six-hour) / 12h (twelve-hour) shift: post-shift review + optional auto-regeneration on tweak/overhaul decisions.")
    lines.append("")
    lines.append("## Monthly / seasonal")
    lines.append("")
    lines.append("- **Season turnover** — at each monthly season close: locked reviews, season awards, draft-night recommendations, and permanent trophy-shelf awards.")
    lines.append("- **Trophy shelves** — granted at season turnover (processed seasons only), append-only.")
    lines.append("- *Draft night:* intake is the approval-gated season draft recommendations from the 12-hour pool; there is **no separate draft-party job**.")
    lines.append("")
    lines.append("## Quarterly / long-term")
    lines.append("")
    lines.append("- **Quarterly Champion report** — every 3 completed major-league seasons, generated when the quarter's 3rd season closes.")
    lines.append("- Peer-relative percentile scoring + 20/30/50 recency weighting; capital-eligibility language only.")
    lines.append("- **No live projected quarterly leader** — a report exists only after the quarter closes. Capital League is not built.")
    lines.append("")

    # --- Commissioner expectations ---
    lines.append("## Commissioner Expectations")
    lines.append("")
    lines.append("_If ATL is left running, here is what updates on its own (assumes an Ollama key is configured; jobs marked optional/required still run their deterministic core without one, except League Maintenance and the Research Agent, which need the key)._")
    lines.append("")
    lines.append("- **Every ~2 minutes:** live standings, equity, P&L, heartbeats, and trade history refresh from the bots.")
    lines.append("- **Every 30 minutes:** the front-page AI overview, scouting questions, and power-ranking overlays refresh; the research agent advances its active question.")
    lines.append("- **Every 3 hours:** the ML biology lab publishes a new telemetry cycle (Fearsome Five, traits, divergences, descendant conviction).")
    lines.append("- **Every 12 hours:** an archive snapshot is taken and pushed. Research questions conclude ~12h after they start and a new one is seeded.")
    lines.append("- **Once per day:** the Chronicle writes that day's chapter.")
    lines.append("- **Every 15 days:** the ML evolution review may park 0–3 new descendant candidates at the Draft Room gate (never auto-deployed).")
    lines.append("- **Every season (monthly):** the ended season is locked with reviews, awards, trophy shelves, and draft recommendations.")
    lines.append("- **Every quarter (3 seasons):** a Quarterly Champion capital-eligibility report is archived. No live projected leader before then.")
    lines.append("")
    lines.append("### Continuous side-effects to expect")
    lines.append("")
    lines.append("- Dev candidate containers start and stop on their own at shift boundaries; runtime DBs are wiped at clean shift starts (dev league only, triple-guarded).")
    lines.append("- Queued strategy generations (new ideas or evolution descendants) are picked up within seconds.")
    lines.append("")
    return "\n".join(lines)


def power_ranking_overrides() -> dict[str, dict[str, Any]]:
    rows = get_generated_json("power_ranking_overrides", [])
    return {row.get("team_id", ""): row for row in rows if row.get("team_id")}


def merged_ml_hypotheses() -> list[dict[str, Any]]:
    updates = {item.get("id"): item for item in get_generated_json("ml_hypothesis_updates", []) if item.get("id")}
    rows = []
    for row in ml_hypotheses():
        merged = dict(row)
        update = updates.get(row["id"], {})
        if update.get("status"):
            merged["status"] = update["status"]
        if update.get("evidence_quality"):
            merged["evidence_quality"] = update["evidence_quality"]
        if update.get("next_action"):
            merged["next_action"] = update["next_action"]
        rows.append(merged)
    for item in get_generated_json("ml_hypothesis_candidates", []):
        candidate = dict(item)
        if not candidate.get("id"):
            candidate["id"] = f"generated-{registry_slug(candidate.get('name', 'hypothesis'))}"
        candidate.setdefault("nickname", "AI Candidate")
        candidate.setdefault("description", candidate.get("rationale", "AI-generated hypothesis candidate."))
        candidate.setdefault("target_variable", candidate.get("target", "Pending target"))
        candidate.setdefault("training_period", "Pending")
        candidate.setdefault("validation_period", "Pending")
        candidate.setdefault("status", "candidate")
        candidate.setdefault("theme", "AI-generated scouting lead")
        candidate.setdefault("evidence_quality", "Early")
        candidate.setdefault("known_risks", ["Needs human review"]) 
        candidate.setdefault("features_used", [])
        candidate.setdefault("market_behavior", candidate.get("description", "Pending characterization."))
        candidate.setdefault("next_action", candidate.get("next_action", "Review and decide whether to promote into the registry."))
        rows.append(candidate)
    return rows


def merged_ml_draft_board() -> list[dict[str, Any]]:
    updates = get_generated_json("ml_draft_board_updates", [])
    update_map = {row.get("prospect_name", row.get("id", "")): row for row in updates if row.get("prospect_name") or row.get("id")}
    rows = []
    for row in ml_draft_board():
        merged = dict(row)
        update = update_map.get(row.get("prospect_name", ""), {})
        for field in ("expected_edge", "evidence_quality", "risk_level", "backtest_strength", "live_readiness", "draft_status", "notes"):
            if update.get(field):
                merged[field] = update[field]
        rows.append(merged)
    for item in get_generated_json("ml_draft_board_candidates", []):
        candidate = dict(item)
        candidate.setdefault("prospect_name", candidate.get("name", "AI Prospect"))
        candidate.setdefault("strategy_family", "AI Candidate")
        candidate.setdefault("expected_edge", candidate.get("rationale", "Pending articulation."))
        candidate.setdefault("evidence_quality", "Early")
        candidate.setdefault("risk_level", "Medium")
        candidate.setdefault("backtest_strength", "Unproven")
        candidate.setdefault("live_readiness", "Low")
        candidate.setdefault("draft_status", "AI Watchlist")
        candidate.setdefault("notes", candidate.get("next_action", "Review in the lab before promotion."))
        rows.append(candidate)
    return rows


def research_playground_context(search_query: str = "") -> dict[str, Any]:
    threads = list_research_threads()
    for thread in threads:
        updates = list_research_thread_updates(int(thread["id"]), limit=12)
        for update in updates:
            try:
                update["citations"] = json.loads(update.get("citations", "[]") or "[]")
            except json.JSONDecodeError:
                update["citations"] = []
        thread["updates"] = updates
    with closing(get_db()) as conn:
        indexed_count = int(conn.execute("SELECT COUNT(*) FROM research_index_entries").fetchone()[0])
    # QF6: when the user runs a query, surface the score + matched tokens behind each
    # hit so the archive is inspectable ("why was this retrieved?").
    index_hits = (
        search_research_index(search_query, limit=10, with_scores=True)
        if search_query
        else recent_research_index_entries(10)
    )
    development_brief = development_research_brief()
    return {
        "questions": load_json(QUESTIONS_PATH, []),
        "generated_questions": [dict(row) for row in list_ai_research_questions("league")],
        "threads": threads,
        "active_thread_count": sum(1 for row in threads if row.get("status") == "active"),
        "indexed_count": indexed_count,
        "search_query": search_query,
        "index_hits": index_hits,
        "development_brief": development_brief,
    }


def base_page_context(scope: str, label: str, entity_id: str = "") -> dict[str, Any]:
    scope_payload = build_chat_scope_payload(scope)
    relevant_files = scope_payload.get("relevant_files", []) if isinstance(scope_payload, dict) else []
    return {
        "chat_context": {"scope": scope, "label": label, "entity_id": entity_id},
        "chat_settings": {
            "configured": bool(get_setting("ollama_api_key", "")),
            "model": get_setting("ollama_model", "gpt-oss:120b"),
        },
        "chat_file_context": relevant_files[:6],
    }


def page_context_bundle(scope: str, label: str, entity_id: str = "", **context: Any) -> dict[str, Any]:
    bundle = base_page_context(scope, label, entity_id)
    bundle.update(context)
    return bundle


def build_chat_scope_payload(scope: str) -> dict[str, Any]:
    if scope == "general":
        return {
            "scope": scope,
            "standings": standings_rows(),
            "manual_research_questions": load_json(QUESTIONS_PATH, []),
            "generated_research_questions": [dict(row) for row in list_ai_research_questions("league")],
            "ml_hypotheses": merged_ml_hypotheses(),
        }
    if scope == "dashboard":
        return dashboard_context()
    if scope == "standings":
        return {"standings": standings_rows()}
    if scope == "power-rankings":
        return {"power_rankings": compute_power_rankings()}
    if scope == "research":
        return research_playground_context()
    if scope == "trade-explorer":
        return {"trades": trade_explorer_rows()[:150]}
    if scope == "exit-tags":
        return {"exit_tags": exit_tag_report_rows()}
    if scope == "ml":
        return ml_lab_context()
    if scope == "ml-workbench":
        return ml_workbench_context()
    if scope == "dev":
        return development_league_context()
    if scope in {"dev-draft", "dev-bootcamp", "dev-six", "dev-twelve", "dev-eligible"}:
        tier = {
            "dev-draft": "draft_room",
            "dev-bootcamp": "bootcamp",
            "dev-six": "six_hour",
            "dev-twelve": "twelve_hour",
            "dev-eligible": "draft_eligible",
        }[scope]
        return development_board_context(tier)
    if scope == "dev-schedule":
        return development_schedule_context()
    if scope.startswith("dev-candidate:"):
        candidate_id = int(scope.split(":", 1)[1] or 0)
        candidate = get_development_candidate(candidate_id)
        if not candidate:
            return {"error": "Unknown candidate"}
        relevant_files = []
        for label, path_value in (
            ("strategy_file", candidate.get("strategy_path")),
            ("config_file", candidate.get("config_path")),
            ("log_file", candidate.get("log_path")),
            ("trade_db", candidate.get("db_path")),
        ):
            path = resolve_path(path_value)
            if path:
                relevant_files.append(gather_file_record(path, label))
        return {
            "candidate": candidate,
            "events": development_runtime_events(candidate_id, limit=20),
            "history": development_runtime_history(candidate_id, limit=20),
            "relevant_files": relevant_files[:6],
        }
    if scope == "ml-hypotheses":
        hypotheses = merged_ml_hypotheses()
        return {
            "hypotheses": hypotheses,
            "relevant_files": search_workspace_files(
                ["ghost ladder", "dark matter", "antimatter", "timmy", "lowgapbucket15m"],
                limit=8,
            ),
        }
    if scope.startswith("team:"):
        team_id = scope.split(":", 1)[1]
        instances = {item["id"]: item for item in list_instances()}
        team = instances.get(team_id)
        if not team:
            return {"error": "Unknown team"}
        strategy_path = resolve_path(team.get("strategy_path"))
        config_path = resolve_path(team.get("config_path"))
        direct_files = []
        if strategy_path:
            direct_files.append(gather_file_record(strategy_path, "strategy_file"))
        if config_path:
            direct_files.append(gather_file_record(config_path, "config_file"))
        return {
            "team": team,
            "latest": dict(latest_snapshot_map().get(team_id) or {}),
            "trade_stats": trade_aggregates(team_id),
            "recent_trades": [dict(row) for row in team_trade_rows(team_id)[:20]],
            "config_summary": read_config_summary(team),
            "strategy_excerpt": read_strategy_excerpt(team),
            "relevant_files": direct_files
            + search_workspace_files(
                [
                    team.get("display_name", ""),
                    team.get("strategy_family", ""),
                    Path(team.get("strategy_path", "")).stem,
                ],
                limit=6,
            ),
        }
    if scope.startswith("hypothesis:"):
        hypothesis_id = scope.split(":", 1)[1]
        rows = {row["id"]: row for row in merged_ml_hypotheses()}
        hypothesis = rows.get(hypothesis_id, {})
        related_models = [
            row for row in ml_models()
            if hypothesis.get("name", "").lower().split("/")[0].strip() in json.dumps(row).lower()
            or hypothesis_id.replace("-", "_") in json.dumps(row).lower()
        ]
        direct_files = []
        for model in related_models:
            artifact_path = resolve_path(model.get("saved_artifact_path"))
            if artifact_path:
                direct_files.append(gather_file_record(artifact_path, model.get("model_name", "artifact")))
        if hypothesis.get("name") == "Ghost Ladder":
            direct_files.extend(
                search_workspace_files(
                    [
                        "ghost ladder",
                        "ghost_ladder",
                        "ladder_persistence",
                        "reclaim_velocity",
                        "lowgapbucket15m",
                    ],
                    limit=8,
                )
            )
        else:
            direct_files.extend(
                search_workspace_files(
                    [
                        hypothesis.get("name", ""),
                        hypothesis.get("nickname", ""),
                        *hypothesis.get("features_used", []),
                    ],
                    limit=8,
                )
            )
        deduped = []
        seen: set[str] = set()
        for item in direct_files:
            key = item.get("path", "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return {
            "hypothesis": hypothesis,
            "related_buckets": [row for row in ml_buckets() if row.get("hypothesis_id") == hypothesis_id],
            "related_models": related_models,
            "findings": [dict(row) for row in list_ml_findings() if row["hypothesis_id"] == hypothesis_id],
            "relevant_files": deduped,
        }
    return {"scope": scope}


def standings_rows() -> list[dict[str, Any]]:
    latest = latest_snapshot_map()
    rows = []
    for instance in list_instances():
        state = latest.get(instance["id"])
        stats = trade_aggregates(instance["id"])
        row = official_runtime_metrics(instance, state, stats)
        row["total_pnl"] = row["raw_total_pnl"]
        row["uptime"] = state["status"] if state else "unknown"
        row["last_trade"] = state["last_trade_at"] if state else None
        rows.append(row)
    rows.sort(key=lambda item: item["raw_total_pnl"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def development_universe_rows(include_archived: bool = True) -> list[dict[str, Any]]:
    rows = development_candidate_rows()
    if not include_archived:
        rows = [row for row in rows if row.get("tier") != "archived"]
    for row in rows:
        row["source_league"] = "development"
        row["shift_id"] = row.get("shift_code", "")
    return rows


def all_strategy_universe(include_archived: bool = True) -> list[dict[str, Any]]:
    official = standings_rows()
    development = development_universe_rows(include_archived=include_archived)
    return official + development


# --- Universe / Pairlist Championship (presentation + aggregation layer) ------
#
# A "universe" is the tradable market habitat (pairlist) a strategy runs on. The
# same strategy on Top 20 vs Top 50 produces materially different outcomes, so we
# track the habitat as a first-class competitive entity alongside the strategy.
# This layer aggregates the existing per-team metrics by universe; it changes no
# strategy behavior and generates no new telemetry.

# Canonical universe key -> display label + type. Keys mirror the generated
# pairlist manifests (see CANONICAL_UNIVERSE_NAMES) plus the official market-cap
# universes carried on instances.json as `pair_universe`.
UNIVERSE_DISPLAY_MAP: dict[str, dict[str, str]] = {
    "top20_marketcap": {"name": "Top 20", "type": "market-cap"},
    "top50_marketcap": {"name": "Top 50", "type": "market-cap"},
    "top100_marketcap": {"name": "Top 100", "type": "market-cap"},
    "top50_volume": {"name": "Top 50 Volume", "type": "market-cap"},
    "custom_momentum_30": {"name": "Momentum 30", "type": "custom"},
    "block_party": {"name": "Neighborhoods", "type": "remote"},
    "future_champion": {"name": "Future Champions", "type": "remote"},
    "big_movers": {"name": "Big Movers", "type": "remote"},
}

# Normalized (lowercase, alnum-only) label fragments that fold onto a canonical key.
_UNIVERSE_ALIASES: dict[str, str] = {
    "top20": "top20_marketcap",
    "top20marketcap": "top20_marketcap",
    "top50": "top50_marketcap",
    "top50marketcap": "top50_marketcap",
    "top100": "top100_marketcap",
    "top100marketcap": "top100_marketcap",
    "top50volume": "top50_volume",
    "custommomentum30": "custom_momentum_30",
    "momentum30": "custom_momentum_30",
    "blockparty": "block_party",
    "neighborhoods": "block_party",
    "futurechampion": "future_champion",
    "futurechampions": "future_champion",
    "bigmovers": "big_movers",
}

# Ordered substring fragments that fold a verbose free-text universe label onto a
# canonical remote universe. Only the named research universes are matched this way
# (their names are strong, unambiguous signals); market-cap labels stay exact so a
# "custom wind-tunnel" universe that merely mentions "Top 20" is not mis-folded.
_UNIVERSE_SUBSTRING_FRAGMENTS: list[tuple[str, str]] = [
    ("bigmovers", "big_movers"),
    ("futurechampion", "future_champion"),
    ("blockparty", "block_party"),
    ("neighborhood", "block_party"),
]

# Universes with fewer than this many closed trades are flagged experimental.
UNIVERSE_EXPERIMENTAL_TRADE_FLOOR = 5


def _normalize_universe_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def universe_key(label: Any) -> str:
    """Collapse any raw universe value (pair_universe, coin_universe, leased
    manifest name, or a parsed display-name tail) onto a stable slug key."""
    token = _normalize_universe_token(label)
    if not token:
        return "unassigned"
    if token in _UNIVERSE_ALIASES:
        return _UNIVERSE_ALIASES[token]
    for key in UNIVERSE_DISPLAY_MAP:
        if _normalize_universe_token(key) == token:
            return key
    for fragment, key in _UNIVERSE_SUBSTRING_FRAGMENTS:
        if fragment in token:
            return key
    return token


def universe_label(key: str, fallback: str = "") -> str:
    if key in UNIVERSE_DISPLAY_MAP:
        return UNIVERSE_DISPLAY_MAP[key]["name"]
    if fallback:
        return fallback
    if key == "unassigned":
        return "Unassigned"
    return key.replace("_", " ").replace("-", " ").title()


def resolve_row_universe(row: dict[str, Any]) -> tuple[str, str]:
    """Return (key, display_label) for the universe a strategy row belongs to.
    Prefers an explicit metadata field over name parsing, per the data model."""
    raw = str(row.get("pair_universe") or row.get("coin_universe") or "").strip()
    # Development governance assigns a leased pairlist manifest per shift.
    if not raw and row.get("source_league") == "development":
        try:
            lease = get_active_lease_for_candidate(row)
            if lease and lease.get("pairlist_manifest_id"):
                raw = str(lease["pairlist_manifest_id"]).strip()
        except Exception:  # noqa: BLE001 - lease lookup is best-effort
            pass
    # Fall back to parsing a "Strategy — Universe" / "Strategy - Universe" name.
    if not raw:
        for field in ("display_name", "team", "name"):
            text = str(row.get(field) or "")
            for sep in (" — ", " – ", " - "):
                if sep in text:
                    raw = text.split(sep, 1)[1].strip()
                    break
            if raw:
                break
    if not raw:
        return ("unassigned", "Unassigned")
    key = universe_key(raw)
    return (key, universe_label(key, fallback=raw))


def universe_catalog(rows: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Every known universe (registered keys ∪ keys actually used by teams) with
    type/status/current-symbol metadata. Status is 'active' when at least one team
    has closed trades on it, otherwise 'experimental'."""
    if rows is None:
        rows = all_strategy_universe(include_archived=True)
    usage: dict[str, dict[str, Any]] = {}
    for row in rows:
        key, label = resolve_row_universe(row)
        if key == "unassigned":
            continue
        entry = usage.setdefault(key, {"teams": 0, "closed_trades": 0, "label": label})
        entry["teams"] += 1
        entry["closed_trades"] += universe_row_metrics(row)["closed_trades"]
    catalog: dict[str, dict[str, Any]] = {}
    for key in set(UNIVERSE_DISPLAY_MAP) | set(usage):
        meta = UNIVERSE_DISPLAY_MAP.get(key, {})
        used = usage.get(key, {})
        manifest = get_pairlist_manifest(key)
        symbol_count = len(manifest.get("pairs", []) or []) if isinstance(manifest, dict) else None
        has_evidence = bool(used.get("teams")) and bool(used.get("closed_trades"))
        catalog[key] = {
            "key": key,
            "name": meta.get("name") or used.get("label") or universe_label(key),
            "type": meta.get("type", "custom"),
            "status": "active" if has_evidence else "experimental",
            "current_symbol_count": symbol_count,
            "team_count": used.get("teams", 0),
        }
    return catalog


def dev_all_time_stats(slug: str) -> dict[str, Any]:
    """All-time trade metrics for a development strategy, computed from the durable
    dev_all_time_trades() union (archived shifts + current live shift). The dev
    runtime DB is wiped between shifts, so the live candidate row reports 0 between
    shifts — this reads the persistent archive instead so the universe championship
    reflects a strategy's whole career, not just the current container."""
    records = dev_all_time_trades(str(slug or ""))
    closed = [record for record in records if not int(record.get("is_open") or 0)]
    open_trades = [record for record in records if int(record.get("is_open") or 0)]
    wins = [record for record in closed if parse_float(record.get("profit_abs")) > 0]

    def roi_pct(record: dict[str, Any]) -> float:
        return parse_float(record.get("profit_ratio")) * 100.0

    realized = sum(parse_float(record.get("realized_profit")) or parse_float(record.get("profit_abs")) for record in closed)
    unrealized = sum(parse_float(record.get("profit_abs")) for record in open_trades)
    avg_roi = sum(roi_pct(record) for record in closed) / len(closed) if closed else 0.0
    return {
        "closed_trades": len(closed),
        "open_trades": len(open_trades),
        "realized_pnl": round(realized, 4),
        "unrealized_pnl": round(unrealized, 4),
        "total_pnl": round(realized + unrealized, 4),
        "win_rate": (len(wins) / len(closed) * 100.0) if closed else 0.0,
        "avg_roi": avg_roi,
        "best_trade": max((roi_pct(record) for record in closed), default=0.0),
        "worst_open_trade": min((roi_pct(record) for record in open_trades), default=0.0),
    }


def universe_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Normalized trade metrics for any strategy row, memoized on the row. Official
    teams use their (non-wiped) live raw_* fields; development candidates use durable
    all-time stats so per-shift DB resets don't zero out their universe's totals."""
    cached = row.get("_universe_metrics")
    if cached is not None:
        return cached
    if row.get("source_league") == "development":
        metrics = dev_all_time_stats(str(row.get("slug") or ""))
    else:
        metrics = {
            "closed_trades": int(row.get("raw_closed_trades") or 0),
            "open_trades": int(row.get("raw_open_trades") or 0),
            "realized_pnl": parse_float(row.get("raw_realized_pnl")),
            "unrealized_pnl": parse_float(row.get("raw_unrealized_pnl")),
            "total_pnl": parse_float(row.get("raw_total_pnl")),
            "win_rate": parse_float(row.get("raw_win_rate")),
            "avg_roi": parse_float(row.get("raw_avg_roi")),
            "best_trade": parse_float(row.get("best_trade")),
            "worst_open_trade": parse_float(row.get("raw_worst_open_trade")),
        }
    row["_universe_metrics"] = metrics
    return metrics


def _empty_universe_bucket(key: str, name: str) -> dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "closed_trades": 0,
        "open_trades": 0,
        "best_trade": 0.0,
        "worst_open_trade": 0.0,
        "active_teams": 0,
        "historical_teams": 0,
        "_win_weighted": 0.0,
        "_roi_weighted": 0.0,
        "_weight": 0,
        "_families": set(),
    }


def _accumulate_universe_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    metrics = universe_row_metrics(row)
    closed = metrics["closed_trades"]
    bucket["total_pnl"] += metrics["total_pnl"]
    bucket["realized_pnl"] += metrics["realized_pnl"]
    bucket["unrealized_pnl"] += metrics["unrealized_pnl"]
    bucket["closed_trades"] += closed
    bucket["open_trades"] += metrics["open_trades"]
    bucket["best_trade"] = max(bucket["best_trade"], metrics["best_trade"])
    bucket["worst_open_trade"] = min(bucket["worst_open_trade"], metrics["worst_open_trade"])
    bucket["historical_teams"] += 1
    if row.get("heartbeat"):
        bucket["active_teams"] += 1
    bucket["_win_weighted"] += metrics["win_rate"] * closed
    bucket["_roi_weighted"] += metrics["avg_roi"] * closed
    bucket["_weight"] += closed
    family = row.get("strategy_family") or row.get("name") or row.get("team")
    if family:
        bucket["_families"].add(str(family))


def _finalize_universe_bucket(bucket: dict[str, Any], catalog_entry: dict[str, Any]) -> dict[str, Any]:
    weight = bucket["_weight"] or 0
    return {
        "key": bucket["key"],
        "name": bucket["name"] or catalog_entry.get("name") or universe_label(bucket["key"]),
        "type": catalog_entry.get("type", "custom"),
        "status": catalog_entry.get("status", "experimental"),
        "total_pnl": round(bucket["total_pnl"], 2),
        "realized_pnl": round(bucket["realized_pnl"], 2),
        "unrealized_pnl": round(bucket["unrealized_pnl"], 2),
        "closed_trades": bucket["closed_trades"],
        "open_trades": bucket["open_trades"],
        "win_rate": round(bucket["_win_weighted"] / weight, 1) if weight else 0.0,
        "avg_roi": round(bucket["_roi_weighted"] / weight, 2) if weight else 0.0,
        "best_trade": round(bucket["best_trade"], 2),
        "worst_open_trade": round(bucket["worst_open_trade"], 2),
        "active_teams": bucket["active_teams"],
        "historical_teams": bucket["historical_teams"],
        "strategy_count": len(bucket["_families"]),
        "current_symbol_count": catalog_entry.get("current_symbol_count"),
    }


def universe_standings_rows() -> list[dict[str, Any]]:
    """One aggregated row per universe, summing the per-team metrics of every team
    (official + development) assigned to it. Registered-but-unused universes appear
    as experimental zero-rows so the championship is visible from day one."""
    rows = all_strategy_universe(include_archived=True)
    catalog = universe_catalog(rows)
    buckets: dict[str, dict[str, Any]] = {
        key: _empty_universe_bucket(key, entry["name"]) for key, entry in catalog.items()
    }
    for row in rows:
        key, label = resolve_row_universe(row)
        if key == "unassigned":
            continue
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = _empty_universe_bucket(key, label)
        _accumulate_universe_bucket(bucket, row)
    finalized = [_finalize_universe_bucket(bucket, catalog.get(key, {})) for key, bucket in buckets.items()]
    finalized.sort(key=lambda item: item["total_pnl"], reverse=True)
    for index, item in enumerate(finalized, start=1):
        item["rank"] = index
    return finalized


def _normalize_universe_trade(trade: Any, team_name: str, team_link: str) -> dict[str, Any]:
    """Map an official (team_trades row) or development (dev_all_time_trades dict)
    trade into a common shape for the universe profile trade tables."""
    get = trade.__getitem__ if isinstance(trade, sqlite3.Row) else trade.get
    def field(name: str, default: Any = None) -> Any:
        try:
            value = get(name)
        except (KeyError, IndexError):
            return default
        return default if value is None else value
    profit_pct = field("profit_pct")
    if profit_pct is None:
        profit_pct = parse_float(field("profit_ratio")) * 100.0
    return {
        "team": team_name,
        "team_link": team_link,
        "pair": field("pair", ""),
        "is_open": bool(field("is_open", 0)),
        "profit_pct": parse_float(profit_pct),
        "profit_abs": parse_float(field("profit_abs")),
        "exit_tag": field("exit_reason", "") or "",
        "enter_tag": field("enter_tag", "") or "",
        "open_date": field("open_date", ""),
        "close_date": field("close_date", ""),
    }


def universe_detail(key: str) -> dict[str, Any]:
    """Full profile payload for one universe: aggregate summary, member teams,
    merged exit-tag distribution, and recent open/closed trades across members."""
    rows = all_strategy_universe(include_archived=True)
    summary = next((row for row in universe_standings_rows() if row["key"] == key), None)
    catalog_entry = universe_catalog(rows).get(key, {})
    members: list[dict[str, Any]] = []
    exit_breakdown: dict[str, dict[str, Any]] = {}
    closed_trades: list[dict[str, Any]] = []
    open_trades: list[dict[str, Any]] = []
    first_seen: str = ""
    last_seen: str = ""
    for row in rows:
        rkey, _ = resolve_row_universe(row)
        if rkey != key:
            continue
        if row.get("source_league") == "development":
            team_name = str(row.get("name") or "")
            team_link = f"/development/candidates/{row.get('id')}"
            raw_trades: list[Any] = dev_all_time_trades(str(row.get("slug") or ""))
            seen = str(row.get("created_at") or "")
        else:
            team_id = str(row.get("team_id") or "")
            team_name = str(row.get("team") or "")
            team_link = f"/teams/{team_id}"
            raw_trades = list(team_trade_rows(team_id))
            seen = str(row.get("start_date") or "")
        if seen:
            first_seen = min(first_seen, seen) if first_seen else seen
            last_seen = max(last_seen, seen) if last_seen else seen
        metrics = universe_row_metrics(row)
        members.append({
            "name": team_name,
            "link": team_link,
            "league": row.get("source_league"),
            "family": row.get("strategy_family") or row.get("name") or team_name,
            "total_pnl": metrics["total_pnl"],
            "closed_trades": metrics["closed_trades"],
            "win_rate": metrics["win_rate"],
            "avg_roi": metrics["avg_roi"],
            "heartbeat": bool(row.get("heartbeat")),
            "first_seen": seen,
        })
        for raw in raw_trades:
            norm = _normalize_universe_trade(raw, team_name, team_link)
            if norm["is_open"]:
                open_trades.append(norm)
            else:
                closed_trades.append(norm)
                tag = norm["exit_tag"] or "unknown"
                bucket = exit_breakdown.setdefault(tag, {"count": 0, "total_profit": 0.0, "_roi_sum": 0.0})
                bucket["count"] += 1
                bucket["total_profit"] += norm["profit_abs"]
                bucket["_roi_sum"] += norm["profit_pct"]
    for bucket in exit_breakdown.values():
        bucket["avg_roi"] = round(bucket["_roi_sum"] / bucket["count"], 2) if bucket["count"] else 0.0
        bucket["total_profit"] = round(bucket["total_profit"], 2)
        bucket.pop("_roi_sum", None)
    exit_rows = [
        {"exit_tag": tag, **data}
        for tag, data in sorted(exit_breakdown.items(), key=lambda item: item[1]["count"], reverse=True)
    ]
    closed_trades.sort(key=lambda item: str(item.get("close_date") or ""), reverse=True)
    open_trades.sort(key=lambda item: str(item.get("open_date") or ""), reverse=True)
    members.sort(key=lambda item: item["total_pnl"], reverse=True)
    manifest = get_pairlist_manifest(key)
    current_symbols = list(manifest.get("pairs", []) or []) if isinstance(manifest, dict) else []
    return {
        "key": key,
        "summary": summary,
        "catalog": catalog_entry,
        "members": members,
        "active_members": [m for m in members if m["heartbeat"]],
        "first_team": min(members, key=lambda m: str(m.get("first_seen") or "~"), default=None),
        "best_team": members[0] if members else None,
        "exit_rows": exit_rows,
        "closed_trades": closed_trades[:25],
        "open_trades": open_trades[:25],
        "current_symbols": current_symbols,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


def compute_universe_power_rankings() -> list[dict[str, Any]]:
    """Front-office confidence score per universe. Transparent additive model over
    net PnL, realized quality, win rate, ROI, sample size, open-trade drag, and a
    breadth bonus that rewards universes proven by more than one strategy family."""
    rankings: list[dict[str, Any]] = []
    for row in universe_standings_rows():
        closed = row["closed_trades"]
        maturity = min(closed * 1.0, 20.0)
        recent_form = max(min(row["total_pnl"] * 5.0, 20.0), -20.0)
        total = row["total_pnl"]
        realized = row["realized_pnl"]
        if total > 0:
            realized_quality = max(0.0, min(1.0, realized / total)) * 10.0
        elif realized > 0:
            realized_quality = 5.0
        else:
            realized_quality = 0.0
        win_component = (row["win_rate"] / 100.0) * 10.0
        roi_component = max(min(row["avg_roi"], 5.0), -5.0)
        open_drag = -min(abs(row["worst_open_trade"]) * 0.5, 10.0)
        breadth_bonus = min(row["strategy_count"], 3) * 4.0
        sample_base = 20.0 if closed >= 20 else 10.0 if closed >= UNIVERSE_EXPERIMENTAL_TRADE_FLOOR else 0.0
        score = sample_base + maturity + recent_form + realized_quality + win_component + roi_component + open_drag + breadth_bonus
        experimental = closed < UNIVERSE_EXPERIMENTAL_TRADE_FLOOR
        rankings.append({
            "key": row["key"],
            "name": row["name"],
            "type": row["type"],
            "status": "experimental" if experimental else row["status"],
            "score": round(score, 1),
            "net_pnl": row["total_pnl"],
            "realized_quality": round(realized_quality, 1),
            "win_rate": row["win_rate"],
            "avg_roi": row["avg_roi"],
            "team_count": row["historical_teams"],
            "active_teams": row["active_teams"],
            "strategy_count": row["strategy_count"],
            "breadth_bonus": round(breadth_bonus, 1),
            "open_trade_drag": round(open_drag, 1),
            "maturity": round(maturity, 1),
            "recent_form": round(recent_form, 1),
            "sample_base": round(sample_base, 1),
            "closed_trades": closed,
            "experimental": experimental,
        })
    rankings.sort(key=lambda item: item["score"], reverse=True)
    for index, item in enumerate(rankings, start=1):
        item["rank"] = index
    return rankings


def universe_season_rows(preview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate official team season-review previews by universe for the Seasons
    Universe View. Lightweight first pass: official teams only (the seasons tab is
    official-focused)."""
    buckets: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        key = universe_key(row.get("pair_universe"))
        if key == "unassigned":
            continue
        bucket = buckets.setdefault(key, {
            "key": key,
            "name": universe_label(key, fallback=str(row.get("pair_universe") or "")),
            "season_pnl": 0.0,
            "closed_trades": 0,
            "teams": 0,
            "_win_weighted": 0.0,
            "_roi_weighted": 0.0,
            "_weight": 0,
            "best_team": None,
            "best_team_pnl": None,
        })
        closed = int(row.get("closed_trades") or 0)
        pnl = parse_float(row.get("total_pnl"))
        bucket["season_pnl"] += pnl
        bucket["closed_trades"] += closed
        bucket["teams"] += 1
        bucket["_win_weighted"] += parse_float(row.get("win_rate")) * closed
        bucket["_roi_weighted"] += parse_float(row.get("avg_roi")) * closed
        bucket["_weight"] += closed
        if bucket["best_team_pnl"] is None or pnl > bucket["best_team_pnl"]:
            bucket["best_team_pnl"] = pnl
            bucket["best_team"] = row.get("team_name")
    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        weight = bucket["_weight"] or 0
        out.append({
            "key": bucket["key"],
            "name": bucket["name"],
            "season_pnl": round(bucket["season_pnl"], 2),
            "closed_trades": bucket["closed_trades"],
            "teams": bucket["teams"],
            "win_rate": round(bucket["_win_weighted"] / weight, 1) if weight else 0.0,
            "avg_roi": round(bucket["_roi_weighted"] / weight, 2) if weight else 0.0,
            "best_team": bucket["best_team"],
            "best_team_pnl": round(bucket["best_team_pnl"], 2) if bucket["best_team_pnl"] is not None else None,
        })
    out.sort(key=lambda item: item["season_pnl"], reverse=True)
    for index, item in enumerate(out, start=1):
        item["rank"] = index
    return out


# Expose universe key normalization to templates so strategy-view tables can link
# a raw `pair_universe` string (e.g. "Top 20") to its canonical profile route.
templates.env.globals["universe_key"] = universe_key


def filter_strategy_universe(
    rows: list[dict[str, Any]],
    scope_filter: str = "all",
    tier_filter: str = "",
    shift_filter: str = "",
    long_short_filter: str = "",
    timeframe_filter: str = "",
    universe_filter: str = "",
    runtime_bucket: str = "",
    sample_quality_filter: str = "",
) -> list[dict[str, Any]]:
    filtered = rows
    if scope_filter == "official":
        filtered = [row for row in filtered if row.get("source_league") == "official"]
    elif scope_filter == "development":
        filtered = [row for row in filtered if row.get("source_league") == "development"]
    elif scope_filter in {"bootcamp", "draft_room", "archived"}:
        filtered = [row for row in filtered if row.get("tier_competition") == scope_filter or row.get("tier") == scope_filter]
    if tier_filter:
        filtered = [row for row in filtered if row.get("tier_competition") == tier_filter or row.get("tier") == tier_filter]
    if shift_filter:
        filtered = [row for row in filtered if str(row.get("shift_code") or "") == shift_filter]
    if long_short_filter:
        filtered = [row for row in filtered if str(row.get("long_short_mode") or "") == long_short_filter]
    if timeframe_filter:
        filtered = [row for row in filtered if str(row.get("timeframe") or "") == timeframe_filter]
    if universe_filter:
        filtered = [row for row in filtered if universe_filter.lower() in json.dumps(row).lower()]
    if runtime_bucket:
        if runtime_bucket == "low":
            filtered = [row for row in filtered if float(row.get("total_runtime_hours") or 0) < PROJECTION_STRONG_WARNING_RUNTIME_HOURS]
        elif runtime_bucket == "medium":
            filtered = [row for row in filtered if PROJECTION_STRONG_WARNING_RUNTIME_HOURS <= float(row.get("total_runtime_hours") or 0) < 24]
        elif runtime_bucket == "high":
            filtered = [row for row in filtered if float(row.get("total_runtime_hours") or 0) >= 24]
    if sample_quality_filter:
        filtered = [row for row in filtered if str(row.get("sample_quality") or "") == sample_quality_filter]
    return filtered


def ranked_rows(rows: list[dict[str, Any]], ranking_key: str, rank_field: str) -> list[dict[str, Any]]:
    enriched = [dict(row) for row in rows]
    enriched.sort(key=lambda item: parse_float(item.get(ranking_key)) if item.get(ranking_key) is not None else -999999, reverse=True)
    for index, row in enumerate(enriched, start=1):
        row[rank_field] = index
    return enriched


def development_standings_context() -> dict[str, Any]:
    rows = development_universe_rows(include_archived=False)
    raw_rows = ranked_rows(rows, "raw_total_pnl", "raw_rank")
    adjusted_rows = ranked_rows(rows, "projected_total_pnl_per_24h", "adjusted_rank")
    shift_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tier_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("shift_code"):
            shift_groups[f"{row.get('tier_competition')}:{row['shift_code']}"].append(row)
        tier_groups[str(row.get("tier_competition") or row.get("tier"))].append(row)
    shift_rows = []
    for key, value in sorted(shift_groups.items()):
        tier_name, shift_code = key.split(":", 1)
        shift_rows.append(
            {
                "tier": tier_name,
                "shift_code": shift_code,
                "rows": ranked_rows(value, "projected_total_pnl_per_24h", "adjusted_rank"),
            }
        )
    tier_rows = [{"tier": key, "rows": ranked_rows(value, "projected_total_pnl_per_24h", "adjusted_rank")} for key, value in sorted(tier_groups.items())]
    return {
        "raw_rows": raw_rows,
        "adjusted_rows": adjusted_rows,
        "shift_groups": shift_rows,
        "tier_groups": tier_rows,
    }


def development_research_brief(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = rows or development_universe_rows(include_archived=True)
    active = [row for row in rows if row.get("tier_competition") in {"candidate_6h", "prospect_12h", "bootcamp"} and row.get("tier") != "archived"]
    momentum = ranked_rows(active, "projected_total_pnl_per_24h", "adjusted_rank")[:5]
    under_runtime = [row for row in active if "needs_runtime" in row.get("freshness_flags", []) or float(row.get("total_runtime_hours") or 0) < PROJECTION_STRONG_WARNING_RUNTIME_HOURS]
    stale = [row for row in rows if "stale_candidate" in row.get("freshness_flags", [])]
    cut_watch = [row for row in rows if "eligible_for_archive" in row.get("freshness_flags", [])]
    return {
        "daily_recap_title": "Daily Development League Recap",
        "momentum_rows": momentum,
        "under_runtime_rows": under_runtime[:8],
        "stale_rows": stale[:8],
        "cut_watch_rows": cut_watch[:8],
        "report_titles": [
            "Daily Development League Recap",
            "Six-Hour Shift Report",
            "Prospect Watch",
            "Bootcamp Health Report",
            "Draft Board Movement",
            "Cut List / Relegation Watch",
            "End-of-Phase Draft Report",
        ],
    }


def trade_explorer_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    team_lookup = {item["id"]: item["display_name"] for item in list_instances()}
    with closing(get_db()) as conn:
        trades = conn.execute(
            """
            SELECT *
            FROM team_trades
            ORDER BY COALESCE(close_date, open_date) DESC
            """
        ).fetchall()
    for row in trades:
        rows.append(
            {
                "team": team_lookup.get(row["team_id"], row["team_id"]),
                "team_id": row["team_id"],
                "pair": row["pair"],
                "side": "Short" if row["is_short"] else "Long",
                "entry_time": row["open_date"],
                "exit_time": row["close_date"],
                "duration": round(parse_float(row["trade_duration_minutes"]), 1),
                "profit_pct": parse_float(row["profit_pct"]),
                "profit_abs": parse_float(row["profit_abs"]),
                "exit_tag": row["exit_reason"] or "",
                "slot_tier": row["enter_tag"] or "",
                "notes": row["strategy_name"] or "",
            }
        )
    return rows


def exit_tag_report_rows() -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "avg_roi": 0.0, "total_profit": 0.0, "best": -999.0, "worst": 999.0, "avg_hold_time": 0.0})
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT exit_reason, profit_pct, profit_abs, trade_duration_minutes
            FROM team_trades
            WHERE is_open = 0
            """
        ).fetchall()
    for row in rows:
        key = row["exit_reason"] or "unknown"
        bucket = buckets[key]
        bucket["count"] += 1
        bucket["avg_roi"] += parse_float(row["profit_pct"])
        bucket["total_profit"] += parse_float(row["profit_abs"])
        bucket["best"] = max(bucket["best"], parse_float(row["profit_pct"]))
        bucket["worst"] = min(bucket["worst"], parse_float(row["profit_pct"]))
        bucket["avg_hold_time"] += parse_float(row["trade_duration_minutes"])
    report = []
    for tag, bucket in buckets.items():
        if bucket["count"]:
            bucket["avg_roi"] /= bucket["count"]
            bucket["avg_hold_time"] /= bucket["count"]
        report.append({"exit_tag": tag, **bucket})
    report.sort(key=lambda item: item["count"], reverse=True)
    return report


def dashboard_context() -> dict[str, Any]:
    standings = standings_rows()
    latest = latest_snapshot_map()
    total_equity = sum(item["equity"] for item in standings)
    total_pnl = sum(item["total_pnl"] for item in standings)
    active_teams = sum(1 for item in standings if item["heartbeat"])
    league_trades = sum(item["closed_trades"] for item in standings)
    champion = standings[0] if standings else None
    return {
        "standings": standings,
        "total_equity": total_equity,
        "total_pnl": total_pnl,
        "active_teams": active_teams,
        "league_trades": league_trades,
        "champion": champion,
        "latest_snapshots": latest,
        "questions": load_json(QUESTIONS_PATH, []),
        "generated_overview": get_generated_content("league_overview", ""),
        "generated_questions": list_ai_research_questions("league"),
        "recent_posts": list_posts("league")[:4],
    }


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    seed_files()
    seed_ml_biology()
    seed_default_settings()
    seed_initial_post()
    seed_initial_ml_post()
    seed_research_threads()
    refresh_repo_research_index()
    sync_ml_platform_registry()
    run_sync()
    sync_development_pipeline()
    thread = threading.Thread(target=sync_loop, daemon=True)
    thread.start()
    dev_thread = threading.Thread(target=development_scheduler_loop, daemon=True)
    dev_thread.start()
    generation_thread = threading.Thread(target=development_generation_loop, daemon=True)
    generation_thread.start()
    maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
    maintenance_thread.start()
    backtesting_thread = threading.Thread(target=backtesting_department_loop, daemon=True)
    backtesting_thread.start()


@app.get("/pairlists/{name}.json")
def serve_pairlist_manifest(name: str) -> JSONResponse:
    # Freqtrade RemotePairList endpoint. Returns {"pairs": [...], "refresh_period": N}.
    manifest = get_pairlist_manifest(name)
    if not manifest:
        raise HTTPException(status_code=404, detail="Unknown pairlist manifest")
    return JSONResponse({
        "pairs": manifest.get("pairs", []),
        "refresh_period": int(manifest.get("refresh_period") or 1800),
    })


@app.get("/api/resource-governance/status")
def resource_governance_status() -> JSONResponse:
    with closing(get_db()) as conn:
        resources = [dict(r) for r in conn.execute(
            "SELECT * FROM exchange_resources ORDER BY exchange_id, market_type"
        ).fetchall()]
        leases = [dict(r) for r in conn.execute(
            "SELECT * FROM exchange_leases WHERE status = 'active' ORDER BY shift_id, exchange_id"
        ).fetchall()]
    per_shift_load: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for lease in leases:
        per_shift_load[lease["shift_id"]][lease["exchange_id"]] += 1
    universes = {
        name: (get_generated_json(f"pairlist_universe:{name}", {}) or {}).get("built_at")
        for name in CANONICAL_UNIVERSE_NAMES
    }
    return JSONResponse({
        "enabled": resource_governance_enabled(),
        "base_url": pairlist_manifest_base_url(),
        "manifest_last_run": get_setting("pairlist_manifest_last_run", ""),
        "universes_built_at": universes,
        "exchange_resources": resources,
        "active_leases": leases,
        "per_shift_load": {shift: dict(load) for shift, load in per_shift_load.items()},
    })


@app.get("/", response_class=HTMLResponse)
def home(request: Request, view: str = "strategy") -> HTMLResponse:
    view = "universe" if view == "universe" else "strategy"
    context = dashboard_context()
    if view == "universe":
        context["universe_rows"] = universe_standings_rows()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        page_context_bundle(
            "dashboard",
            "League Dashboard",
            page_title="League Dashboard",
            view=view,
            **context,
        ),
    )


@app.get("/standings", response_class=HTMLResponse)
def standings_page(request: Request, view: str = "strategy") -> HTMLResponse:
    view = "universe" if view == "universe" else "strategy"
    current_season = current_league_season()
    preview_map = {
        row.get("team_id", ""): row
        for row in [preview_official_team_season_review(instance, current_season) for instance in list_instances()]
    }
    return templates.TemplateResponse(
        request,
        "standings.html",
        page_context_bundle(
            "standings",
            "Standings",
            page_title="Standings",
            view=view,
            rows=standings_rows(),
            universe_rows=universe_standings_rows() if view == "universe" else [],
            current_season=current_season,
            season_preview_map=preview_map,
        ),
    )


@app.get("/teams/{team_id}", response_class=HTMLResponse)
def team_page(request: Request, team_id: str) -> HTMLResponse:
    instances = {item["id"]: item for item in list_instances()}
    team = instances.get(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    latest = latest_snapshot_map().get(team_id)
    trades = team_trade_rows(team_id)
    trade_stats = trade_aggregates(team_id)
    config_summary = read_config_summary(team)
    history = snapshot_history(team_id)
    open_trades = [row for row in trades if row["is_open"]][:10]
    closed_trades = [row for row in trades if not row["is_open"]][:20]
    current_season = current_league_season()
    season_preview_review = preview_official_team_season_review(team, current_season)
    latest_season_review = latest_league_team_season_review(team_id)
    return templates.TemplateResponse(
        request,
        "team.html",
        {
            "page_title": team["display_name"],
            "team": team,
            "latest": latest,
            "trade_stats": trade_stats,
            "config_summary": config_summary,
            "history": history,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
            "current_season": current_season,
            "season_preview_review": season_preview_review,
            "latest_season_review": latest_season_review,
            "trophy_shelf": strategy_awards_for("team", team_id),
            **base_page_context(f"team:{team_id}", team["display_name"], team_id),
        },
    )


@app.get("/seasons", response_class=HTMLResponse)
def seasons_page(request: Request, view: str = "strategy") -> HTMLResponse:
    view = "universe" if view == "universe" else "strategy"
    context = season_office_context()
    if view == "universe":
        context["universe_season_rows"] = universe_season_rows(context.get("current_preview_rows", []))
    return templates.TemplateResponse(
        request,
        "seasons.html",
        page_context_bundle("seasons", "Season Office", page_title="Season Office", view=view, **context),
    )


@app.post("/api/seasons/turnover")
async def run_seasons_turnover_action(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    force = payload.get("force", [""])[0].lower() in {"1", "true", "yes", "on"}
    run_season_turnover(force=force)
    return RedirectResponse(url="/seasons", status_code=303)


@app.get("/quarterly", response_class=HTMLResponse)
def quarterly_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "quarterly.html",
        page_context_bundle("quarterly", "Quarterly Champion", page_title="Quarterly Champion", **quarterly_office_context()),
    )


@app.get("/schedule", response_class=HTMLResponse)
def operations_schedule_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "operations_schedule.html",
        page_context_bundle("schedule", "League Clock", page_title="League Clock", **operations_schedule_context()),
    )


@app.get("/schedule/export.md", response_class=PlainTextResponse)
def operations_schedule_export() -> PlainTextResponse:
    return PlainTextResponse(build_operations_cadence_markdown(), media_type="text/markdown; charset=utf-8")


@app.post("/api/quarterly/generate")
async def generate_quarterly_report_action(request: Request) -> JSONResponse:
    """Admin/explicit generation. Without quarter -> fill any closed-but-missing reports.
    With quarter + force=1 -> regenerate that archived report (the only sanctioned rewrite)."""
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    raw_quarter = payload.get("quarter", [""])[0].strip()
    force = payload.get("force", [""])[0].lower() in {"1", "true", "yes", "on"}
    try:
        if raw_quarter:
            report = generate_quarterly_report(int(raw_quarter), force=force)
            if not report:
                raise HTTPException(status_code=400, detail="That quarter has not closed yet (its three seasons aren't all processed).")
            return JSONResponse({"status": "ok", "quarter": report.get("quarter_number"), "champion": report.get("champion_team_name")})
        maybe_generate_quarterly_reports()
        return JSONResponse({"status": "ok"})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/seasons/reviews")
async def apply_season_review_action(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    review_key = payload.get("review_key", [""])[0]
    action = payload.get("action", [""])[0]
    notes = payload.get("notes", [""])[0]
    apply_league_team_season_review_action(review_key, action, notes)
    return RedirectResponse(url="/seasons", status_code=303)


@app.post("/api/seasons/drafts")
async def apply_season_draft_action(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    draft_id = parse_intish(payload.get("draft_id", [""])[0])
    approval_status = payload.get("action", [""])[0]
    notes = payload.get("notes", [""])[0]
    if not draft_id:
        raise HTTPException(status_code=400, detail="Draft recommendation is required")
    apply_league_season_draft_action(draft_id, approval_status, notes)
    return RedirectResponse(url="/seasons", status_code=303)


@app.get("/development", response_class=HTMLResponse)
def development_home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_dashboard.html",
        page_context_bundle("dev", "Development League", page_title="Development League", **development_league_context()),
    )


@app.get("/development/draft-room", response_class=HTMLResponse)
def development_draft_room_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_board.html",
        page_context_bundle("dev-draft", "Draft Room", page_title="Draft Room", allow_create=True, **development_board_context("draft_room")),
    )


@app.get("/development/bootcamp", response_class=HTMLResponse)
def development_bootcamp_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_board.html",
        page_context_bundle("dev-bootcamp", "Bootcamp", page_title="Bootcamp", allow_create=False, **development_board_context("bootcamp")),
    )


@app.get("/development/six-hour", response_class=HTMLResponse)
def development_six_hour_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_board.html",
        page_context_bundle("dev-six", "Six-Hour Candidates", page_title="Six-Hour Candidates", allow_create=False, **development_board_context("six_hour")),
    )


@app.get("/development/twelve-hour", response_class=HTMLResponse)
def development_twelve_hour_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_board.html",
        page_context_bundle("dev-twelve", "Twelve-Hour Prospects", page_title="Twelve-Hour Prospects", allow_create=False, **development_board_context("twelve_hour")),
    )


@app.get("/development/draft-eligible", response_class=HTMLResponse)
def development_draft_eligible_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_board.html",
        page_context_bundle("dev-eligible", "Draft Eligible", page_title="Draft Eligible", allow_create=False, **development_board_context("draft_eligible")),
    )


@app.get("/development/schedule", response_class=HTMLResponse)
def development_schedule_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "development_schedule.html",
        page_context_bundle("dev-schedule", "Shift Schedule", page_title="Shift Schedule", **development_schedule_context()),
    )


@app.get("/development/candidates/{candidate_id}", response_class=HTMLResponse)
def development_candidate_detail_page(request: Request, candidate_id: int) -> HTMLResponse:
    candidate = get_development_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    strategy_preview = strategy_code_preview(candidate)
    post_shift_reviews = development_post_shift_reviews(candidate_id)
    slug = str(candidate.get("slug") or "")
    timeline = dev_strategy_timeline(slug)
    event_cartography = compute_entry_exit_matrix(dev_all_time_trades(slug))
    return templates.TemplateResponse(
        request,
        "development_candidate.html",
        {
            "page_title": candidate["name"],
            "candidate": candidate,
            "timeline": timeline,
            "event_cartography": event_cartography,
            "events": development_runtime_events(candidate_id),
            "history": development_runtime_history(candidate_id),
            "sessions": development_runtime_sessions(candidate_id),
            "latest_review": post_shift_reviews[0] if post_shift_reviews else None,
            "post_shift_reviews": post_shift_reviews,
            "strategy_preview": strategy_preview,
            "assumptions": decode_jsonish_list(str(candidate.get("generation_assumptions") or "")),
            "warnings": decode_jsonish_list(str(candidate.get("generation_warnings") or "")),
            "config_notes": decode_jsonish_list(str(candidate.get("minimal_config_notes") or "")),
            "auto_refresh_seconds": 7 if candidate.get("generation_status") in {"queued", "generating"} else 0,
            "shift_options": development_shift_definitions("six_hour"),
            "trophy_shelf": strategy_awards_for("dev", str(candidate.get("slug") or "")),
            **base_page_context(f"dev-candidate:{candidate_id}", candidate["name"], str(candidate_id)),
        },
    )


@app.post("/development/candidates")
async def create_development_candidate_route(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    candidate_id = create_development_candidate(
        {
            "name": pick("name"),
            "hypothesis": pick("hypothesis"),
            "strategy_notes": pick("strategy_notes"),
            "long_short_mode": pick("long_short_mode") or "both",
            "expected_behavior": pick("expected_behavior"),
            "risk_profile": pick("risk_profile"),
            "coin_universe": pick("coin_universe"),
            "timeframe": pick("timeframe"),
            "notes": pick("notes"),
            "db_path": pick("db_path"),
            "config_path": pick("config_path"),
            "log_path": pick("log_path"),
            "strategy_path": pick("strategy_path"),
            "api_url": pick("api_url"),
            "api_username": pick("api_username"),
            "api_password": pick("api_password"),
            "start_command": pick("start_command"),
            "stop_command": pick("stop_command"),
        }
    )
    return RedirectResponse(url=f"/development/candidates/{candidate_id}", status_code=303)


@app.post("/development/candidates/{candidate_id}/action")
async def development_candidate_action_route(candidate_id: int, request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    apply_development_candidate_action(candidate_id, pick("action") or "resume_auto", {key: pick(key) for key in payload})
    redirect_to = pick("redirect_to") or f"/development/candidates/{candidate_id}"
    return RedirectResponse(url=redirect_to, status_code=303)


@app.get("/power-rankings", response_class=HTMLResponse)
def power_rankings_page(request: Request, view: str = "strategy") -> HTMLResponse:
    view = "universe" if view == "universe" else "strategy"
    return templates.TemplateResponse(
        request,
        "power_rankings.html",
        page_context_bundle(
            "power-rankings",
            "Power Rankings",
            page_title="Power Rankings",
            view=view,
            rows=compute_power_rankings(),
            universe_rows=compute_universe_power_rankings() if view == "universe" else [],
        ),
    )


@app.get("/universes/{universe_key}", response_class=HTMLResponse)
def universe_page(request: Request, universe_key: str) -> HTMLResponse:
    detail = universe_detail(universe_key)
    if not detail.get("members") and universe_key not in UNIVERSE_DISPLAY_MAP:
        raise HTTPException(status_code=404, detail="Universe not found")
    name = detail.get("catalog", {}).get("name") or universe_label(universe_key)
    return templates.TemplateResponse(
        request,
        "universe.html",
        {
            "page_title": name,
            "detail": detail,
            **base_page_context("standings", name, universe_key),
        },
    )


@app.get("/timeline", response_class=HTMLResponse)
def timeline_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "timeline.html",
        {
            "page_title": "Timeline",
            "posts": list_posts("league"),
            "teams": list_instances(),
            **base_page_context("dashboard", "League Timeline"),
        },
    )


@app.get("/chronicle", response_class=HTMLResponse)
def chronicle_page(request: Request) -> HTMLResponse:
    days = all_chronicle_days()
    return templates.TemplateResponse(
        request,
        "chronicle.html",
        {
            "page_title": "Chronicle",
            "days": days,
            "day_count": len(days),
            "last_run": get_setting("chronicle_last_run", ""),
            "run_time": get_setting("chronicle_run_time", "23:11"),
            **base_page_context("chronicle", "ATL Chronicle"),
        },
    )


@app.post("/timeline")
async def create_timeline_post(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    def pick(key: str) -> str:
        return payload.get(key, [""])[0]
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO timeline_posts (
                created_at, category, title, team_tags, observation, evidence, interpretation, next_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iso_now(),
                "league",
                pick("title"),
                pick("team_tags"),
                pick("observation"),
                pick("evidence"),
                pick("interpretation"),
                pick("next_action"),
            ),
        )
        conn.commit()
    return RedirectResponse(url="/timeline", status_code=303)


@app.get("/trade-explorer", response_class=HTMLResponse)
def trade_explorer_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "trade_explorer.html",
        page_context_bundle("trade-explorer", "Trade Explorer", page_title="Trade Explorer", rows=trade_explorer_rows()),
    )


@app.get("/exit-tags", response_class=HTMLResponse)
def exit_tags_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "exit_tags.html",
        page_context_bundle("exit-tags", "Exit Tag Report", page_title="Exit Tag Report", rows=exit_tag_report_rows()),
    )


def universe_backtest_lab_context() -> dict[str, Any]:
    runs = list_universe_backtest_runs()
    # Group runs by strategy + window + exchange so the same strategy across universes
    # shows up as one side-by-side comparison block.
    groups: dict[str, dict[str, Any]] = {}
    for run in runs:
        key = f"{run.get('team_id')}|{run.get('timerange')}|{run.get('exchange')}"
        group = groups.setdefault(key, {
            "display_name": run.get("display_name"),
            "timerange": run.get("timerange"),
            "exchange": run.get("exchange"),
            "rows": [],
        })
        group["rows"].append(run)
    comparison_groups = sorted(groups.values(), key=lambda item: str(item.get("timerange")), reverse=True)
    for group in comparison_groups:
        group["rows"].sort(key=lambda run: parse_float(run.get("profit_total_abs")), reverse=True)
    pool_status = [
        {
            "exchange": exchange,
            "pairs_1h": len(_local_ohlcv_pairs(exchange, "futures", "1h")),
            "pairs_5m": len(_local_ohlcv_pairs(exchange, "futures", "5m")),
        }
        for exchange in ("binance", "hyperliquid", "bybit")
    ]
    mc_path = marketcap_history_path()
    marketcap_coins = 0
    if mc_path.exists():
        try:
            marketcap_coins = len(json.loads(mc_path.read_text(encoding="utf-8")).get("series", {}))
        except Exception:  # noqa: BLE001
            marketcap_coins = 0
    return {
        "strategies": [{"id": item["id"], "display_name": item["display_name"]} for item in list_instances()],
        "universes": [{"key": key, "name": meta["name"], "type": meta["type"]} for key, meta in UNIVERSE_DISPLAY_MAP.items()],
        "comparison_groups": comparison_groups,
        "fgi_ready": fear_greed_history_path().exists(),
        "pool_status": pool_status,
        "marketcap_ready": mc_path.exists(),
        "marketcap_coins": marketcap_coins,
        "default_timerange": "20260527-20260605",
        "default_exchange": "binance",
        "default_timeframe": "5m",
    }


def backtesting_department_context() -> dict[str, Any]:
    organisms = active_organisms()
    health_stale_hours = parse_float(get_setting("backtest_health_stale_hours", "24")) or 24.0
    habitat_stale_hours = parse_float(get_setting("backtest_habitat_stale_hours", "168")) or 168.0
    with closing(get_db()) as conn:
        lanes = [dict(row) for row in conn.execute("SELECT * FROM backtest_lanes ORDER BY lane_id").fetchall()]
        queued = [dict(row) for row in conn.execute(
            "SELECT * FROM backtest_jobs WHERE status='queued' ORDER BY priority_bucket ASC, priority_score DESC, created_at ASC LIMIT 40"
        ).fetchall()]
        active_jobs = {
            (row["strategy_key"], row["universe_key"]): row["status"]
            for row in conn.execute("SELECT strategy_key, universe_key, status FROM backtest_jobs WHERE status IN ('queued','running')").fetchall()
        }
        recent = [dict(row) for row in conn.execute(
            """
            SELECT j.job_id, j.status, j.strategy_name, j.universe_name, j.completed_at, j.failure_reason,
                   j.run_id, j.priority_bucket, r.total_pnl, r.win_rate, r.avg_roi, r.closed_trades, r.max_drawdown
            FROM backtest_jobs j LEFT JOIN backtest_results r ON r.job_id = j.job_id
            WHERE j.status IN ('completed','failed') ORDER BY j.completed_at DESC LIMIT 25
            """
        ).fetchall()]
        n_queued = conn.execute("SELECT COUNT(*) FROM backtest_jobs WHERE status='queued'").fetchone()[0]
        n_failed = conn.execute("SELECT COUNT(*) FROM backtest_jobs WHERE status='failed'").fetchone()[0]
        result_rows = conn.execute("SELECT * FROM backtest_results ORDER BY created_at DESC").fetchall()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        cell = (row["strategy_key"], row["universe_key"])
        if cell not in latest:
            latest[cell] = dict(row)
    org_hash = {
        org["key"]: (sha256_file(Path(org["strategy_path"]))[:16] if org["strategy_path"] and Path(org["strategy_path"]).exists() else "")
        for org in organisms
    }

    def cell_state(strategy_key: str, universe: str, stale_hours: float) -> str:
        if (strategy_key, universe) in active_jobs:
            return active_jobs[(strategy_key, universe)]  # 'queued' or 'running'
        res = latest.get((strategy_key, universe))
        if not res:
            return "missing"
        if org_hash.get(strategy_key) and res.get("strategy_hash") and res["strategy_hash"] != org_hash[strategy_key]:
            return "stale"
        try:
            created = normalize_utc(datetime.fromisoformat(res["created_at"]))
        except Exception:  # noqa: BLE001
            return "stale"
        return "stale" if (utc_now() - created).total_seconds() >= stale_hours * 3600 else "complete"

    coverage = {bucket: {"complete": 0, "stale": 0, "missing": 0, "queued": 0, "running": 0} for bucket in (1, 2)}
    for org in organisms:
        cu = org["canonical_universe"]
        if cu and cu != "unassigned":
            coverage[1][cell_state(org["key"], cu, health_stale_hours)] += 1
        for universe in STANDARD_UNIVERSES:
            coverage[2][cell_state(org["key"], universe, habitat_stale_hours)] += 1

    matrix_cols = [{"key": u, "name": universe_label(u)} for u in STANDARD_UNIVERSES]
    matrix_rows = []
    for org in organisms:
        cells = []
        for universe in STANDARD_UNIVERSES:
            res = latest.get((org["key"], universe))
            cells.append({
                "state": cell_state(org["key"], universe, habitat_stale_hours),
                "pnl": parse_float(res.get("total_pnl")) if res else None,
                "run_id": res.get("run_id") if res else "",
            })
        matrix_rows.append({"name": org["name"], "key": org["key"], "cells": cells})

    stale_count = sum(coverage[b][s] for b in (1, 2) for s in ("stale", "missing"))
    last_completed = next((r for r in recent if r["status"] == "completed"), None)
    last_failure = next((r for r in recent if r["status"] == "failed"), None)
    for job in queued:
        job["bucket_label"] = DEPARTMENT_BUCKET_LABELS.get(job["priority_bucket"], "")
    for lane in lanes:
        lane["current_title"] = ""
        if lane.get("current_job_id"):
            with closing(get_db()) as conn:
                jrow = conn.execute("SELECT title FROM backtest_jobs WHERE job_id=?", (lane["current_job_id"],)).fetchone()
            lane["current_title"] = jrow["title"] if jrow else ""
    return {
        "dept_enabled": department_enabled(),
        "dept_paused": department_paused(),
        "dept_lanes": lanes,
        "dept_queue": queued,
        "dept_queue_count": n_queued,
        "dept_failed_count": n_failed,
        "dept_recent": recent,
        "dept_coverage": coverage,
        "dept_bucket_labels": DEPARTMENT_BUCKET_LABELS,
        "dept_stale_count": stale_count,
        "dept_last_completed": last_completed,
        "dept_last_failure": last_failure,
        "dept_matrix_cols": matrix_cols,
        "dept_matrix_rows": matrix_rows,
        "dept_default_timerange": department_default_timerange(),
        "dept_organisms": [{"key": o["key"], "name": o["name"]} for o in organisms],
    }


@app.get("/backtests", response_class=HTMLResponse)
def backtests_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "backtests.html",
        page_context_bundle(
            "dashboard",
            "Backtesting Department",
            page_title="Backtesting Department",
            archive_rows=parse_backtest_archive(),
            **universe_backtest_lab_context(),
            **backtesting_department_context(),
        ),
    )


@app.post("/api/backtests/dept/toggle")
async def toggle_department(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    action = payload.get("action", ["pause"])[0].strip()
    set_setting("backtest_department_paused", "true" if action == "pause" else "false")
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/dept/run-next")
async def department_run_next(request: Request) -> RedirectResponse:
    threading.Thread(target=run_department_once, daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/dept/cancel")
async def department_cancel_job(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    job_id = payload.get("job_id", [""])[0].strip()
    with closing(get_db()) as conn:
        # Only queued jobs can be cancelled cleanly; a running docker job is left to finish.
        conn.execute("UPDATE backtest_jobs SET status='cancelled', completed_at=? WHERE job_id=? AND status='queued'", (iso_now(), job_id))
        conn.commit()
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/dept/retry")
async def department_retry_job(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    job_id = payload.get("job_id", [""])[0].strip()
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM backtest_jobs WHERE job_id=?", (job_id,)).fetchone()
    if row:
        job = dict(row)
        enqueue_backtest_job(
            job["strategy_key"], job["strategy_name"], job["strategy_version"], job["universe_key"],
            int(job["priority_bucket"]), parse_float(job["priority_score"]), f"Retry of {job_id}",
            job["timerange"], job["timeframe"], job["comparison_group_id"], job["mode"], job["title"],
        )
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/dept/queue")
async def department_queue_custom(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    strategy_key = payload.get("strategy_key", [""])[0].strip()
    universe = payload.get("universe", [""])[0].strip()
    timerange = payload.get("timerange", [""])[0].strip() or department_default_timerange()
    if not strategy_key or not universe:
        raise HTTPException(status_code=400, detail="Strategy and universe are required.")
    try:
        strat = resolve_backtest_strategy(strategy_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    version = sha256_file(Path(strat["strategy_path"]))[:16] if Path(strat["strategy_path"]).exists() else ""
    enqueue_backtest_job(
        strategy_key, strat["display_name"], version, universe_key(universe), 4, 60.0,
        "Manual research-queue request", timerange, strat["timeframe"], title=f"Manual: {strat['display_name']} × {universe_label(universe_key(universe))}",
    )
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/universe/run")
async def launch_universe_backtest(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    strategy_key = payload.get("strategy_key", [""])[0].strip()
    universes = [value for value in payload.get("universes", []) if value.strip()]
    timerange = payload.get("timerange", ["20260527-20260605"])[0].strip() or "20260527-20260605"
    exchange = payload.get("exchange", ["hyperliquid"])[0].strip() or "hyperliquid"
    timeframe = payload.get("timeframe", [""])[0].strip()
    if not strategy_key or not universes:
        raise HTTPException(status_code=400, detail="Pick a strategy and at least one universe.")

    def worker() -> None:
        run_universe_matrix(
            strategy_key,
            universes,
            exchange=exchange,
            timeframe=timeframe or None,
            timerange=timerange,
        )

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/universe/fear-greed")
async def refresh_fear_greed_history(request: Request) -> RedirectResponse:
    try:
        fetch_fear_greed_history()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Fear & Greed download failed: {exc}") from exc
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/universe/download-pool")
async def launch_pool_download(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    exchange = payload.get("exchange", ["binance"])[0].strip() or "binance"
    days = parse_intish(payload.get("days", ["180"])[0]) or 180
    timeframes = tuple(tf for tf in payload.get("timeframes", ["1h", "5m"]) if tf.strip()) or ("1h", "5m")

    def worker() -> None:
        download_universe_pool(exchange=exchange, quote=backtest_quote_for(exchange), timeframes=timeframes, days=days)

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)


@app.post("/api/backtests/universe/marketcap-history")
async def launch_marketcap_history(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    days = parse_intish(payload.get("days", ["180"])[0]) or 180

    def worker() -> None:
        try:
            fetch_coingecko_market_caps_history(days=days)
        except Exception:  # noqa: BLE001 - background best-effort
            pass

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse(url="/backtests", status_code=303)


@app.get("/backtests/universe/{run_id}", response_class=HTMLResponse)
def universe_backtest_run_page(request: Request, run_id: str) -> HTMLResponse:
    record_path = universe_backtest_run_record_path(run_id)
    if not record_path.exists():
        raise HTTPException(status_code=404, detail="Backtest run not found")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    tape: list[dict[str, Any]] = []
    tape_file = PROJECT_DIR / str(record.get("tape_path") or "")
    if tape_file.exists():
        for line in tape_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    tape.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    return templates.TemplateResponse(
        request,
        "universe_backtest_run.html",
        page_context_bundle(
            "dashboard",
            record.get("display_name", "Backtest Run"),
            page_title=f"{record.get('universe_name', record.get('universe'))} backtest",
            record=record,
            tape=tape,
        ),
    )


@app.get("/versions", response_class=HTMLResponse)
def versions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "versions.html",
        page_context_bundle("dashboard", "Version Registry", page_title="Version Registry", rows=version_registry()),
    )


@app.get("/research", response_class=HTMLResponse)
def research_page(request: Request) -> HTMLResponse:
    search_query = request.query_params.get("q", "").strip()
    return templates.TemplateResponse(
        request,
        "research.html",
        page_context_bundle("research", "Research Questions", page_title="Research Questions", **research_playground_context(search_query)),
    )


@app.post("/research/threads")
async def create_research_thread_route(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    question = pick("question").strip()
    if question:
        create_research_thread(
            question,
            owner="user",
            scope="research",
            auto_reseed=bool(pick("auto_reseed")),
        )
    return RedirectResponse(url="/research", status_code=303)


@app.post("/research/notes")
async def create_research_note_route(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    thread_id = int(pick("thread_id") or 0)
    note = pick("note").strip()
    title = pick("title").strip() or "Manual Note"
    if thread_id and note:
        add_research_thread_update(thread_id, "manual_note", title, note, "user")
    return RedirectResponse(url="/research", status_code=303)


@app.post("/ml/workbench/queue/thread")
async def queue_research_thread_for_ml_route(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    thread_id = int(pick("thread_id") or 0)
    if thread_id:
        queue_research_thread_for_ml(thread_id)
    redirect_to = pick("redirect_to") or "/research"
    return RedirectResponse(url=redirect_to, status_code=303)


@app.post("/ml/workbench/queue")
async def create_ml_workbench_queue_route(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    title = pick("title").strip() or "Manual ML Lead"
    lead_question = pick("lead_question").strip()
    rationale = pick("rationale").strip()
    if lead_question:
        queue_id = create_ml_queue_item(
            "manual",
            f"manual:{registry_slug(title)}:{registry_slug(lead_question)}",
            title,
            lead_question,
            rationale or lead_question,
            priority=pick("priority") or "normal",
        )
        if pick("execute_now"):
            run_local_ml_executor(queue_id, force=True)
    return RedirectResponse(url="/ml/workbench", status_code=303)


@app.post("/ml/workbench/queue/{queue_id}/action")
async def ml_workbench_queue_action_route(queue_id: int, request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))

    def pick(key: str) -> str:
        return payload.get(key, [""])[0]

    action = pick("action") or "inspect"
    priority = pick("priority") or "normal"
    if action == "execute":
        run_local_ml_executor(queue_id, force=True)
    elif action == "retry":
        update_ml_queue_item(queue_id, status="queued", resolution="Queued for retry.")
    elif action == "block":
        update_ml_queue_item(queue_id, status="blocked", resolution=pick("reason") or "Blocked from the workbench UI.")
    elif action == "reprioritize":
        update_ml_queue_item(queue_id, priority=priority, resolution=f"Priority set to {priority}.")
    redirect_status = urllib.parse.quote(pick("status") or "all")
    return RedirectResponse(url=f"/ml/workbench?status={redirect_status}&queue_id={queue_id}", status_code=303)


@app.get("/ml", response_class=HTMLResponse)
def ml_home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_home.html",
        page_context_bundle(
            "ml",
            "ML Lab — Strategy Biology Department",
            page_title="ML Lab — Strategy Biology Department",
            **ml_biology_context(),
        ),
    )


@app.get("/ml/families", response_class=HTMLResponse)
def ml_families_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_families.html",
        page_context_bundle("ml", "Family Registry", page_title="Family Registry", **ml_biology_context()),
    )


@app.get("/ml/evolution", response_class=HTMLResponse)
def ml_evolution_page(request: Request) -> HTMLResponse:
    with closing(get_db()) as conn:
        reviews = [dict(row) for row in conn.execute(
            "SELECT * FROM ml_evolution_reviews ORDER BY id DESC LIMIT 20"
        ).fetchall()]
    return templates.TemplateResponse(
        request,
        "ml_evolution.html",
        page_context_bundle(
            "ml", "Evolution Reviews", page_title="Evolution Reviews",
            reviews=reviews,
            hypotheses=ml_descendant_hypotheses_all(order_by_conviction=True),
            conviction_threshold=parse_float(get_setting("ml_descendant_conviction_threshold", "0.75")),
        ),
    )


@app.get("/ml/telemetry/{cycle_id}", response_class=HTMLResponse)
def ml_telemetry_cycle_page(request: Request, cycle_id: int) -> HTMLResponse:
    with closing(get_db()) as conn:
        cycle = conn.execute("SELECT * FROM ml_telemetry_cycles WHERE id = ?", (cycle_id,)).fetchone()
        if not cycle:
            raise HTTPException(status_code=404, detail="Telemetry cycle not found")
        cycle = dict(cycle)
        divergence = [dict(row) for row in conn.execute(
            "SELECT * FROM ml_telemetry_divergence WHERE cycle_id = ? ORDER BY magnitude DESC", (cycle_id,)
        ).fetchall()]
    try:
        cycle["findings"] = json.loads(cycle.get("top_findings_json") or "[]")
    except json.JSONDecodeError:
        cycle["findings"] = []
    names = {s["slug"]: s["name"] for s in ml_registry_all(active_only=False)}
    for row in divergence:
        row["strategy_name"] = names.get(row["strategy_slug"], row["strategy_slug"])
    return templates.TemplateResponse(
        request,
        "ml_telemetry_cycle.html",
        page_context_bundle("ml", f"Telemetry Cycle {cycle_id}", page_title=f"Telemetry Cycle {cycle_id}",
                            cycle=cycle, divergence=divergence),
    )


@app.get("/ml/workbench", response_class=HTMLResponse)
def ml_workbench_page(request: Request) -> HTMLResponse:
    sync_ml_platform_registry()
    queue_status = request.query_params.get("status", "all").strip().lower() or "all"
    selected_queue_id = int(request.query_params.get("queue_id", "0") or 0) or None
    selected_run_slug = request.query_params.get("run_slug", "").strip()
    compare_a = request.query_params.get("compare_a", "").strip()
    compare_b = request.query_params.get("compare_b", "").strip()
    return templates.TemplateResponse(
        request,
        "ml_workbench.html",
        page_context_bundle(
            "ml-workbench",
            "ML Workbench",
            page_title="ML Workbench",
            **ml_workbench_context(queue_status, selected_queue_id, selected_run_slug, compare_a, compare_b),
        ),
    )


@app.get("/ml/hypotheses", response_class=HTMLResponse)
def ml_hypotheses_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_hypotheses.html",
        page_context_bundle("ml-hypotheses", "Hypothesis Registry", page_title="Hypothesis Registry", rows=merged_ml_hypotheses()),
    )


@app.get("/ml/hypotheses/{hypothesis_id}", response_class=HTMLResponse)
def ml_hypothesis_detail_page(request: Request, hypothesis_id: str) -> HTMLResponse:
    rows = {row["id"]: row for row in merged_ml_hypotheses()}
    hypothesis = rows.get(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    related_buckets = [row for row in ml_buckets() if row.get("hypothesis_id") == hypothesis_id]
    return templates.TemplateResponse(
        request,
        "ml_hypothesis_detail.html",
        {
            "page_title": hypothesis["name"],
            "hypothesis": hypothesis,
            "related_buckets": related_buckets,
            "findings": [row for row in list_ml_findings() if row["hypothesis_id"] == hypothesis_id],
            **base_page_context(f"hypothesis:{hypothesis_id}", hypothesis["name"], hypothesis_id),
        },
    )


@app.get("/ml/buckets", response_class=HTMLResponse)
def ml_buckets_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_buckets.html",
        page_context_bundle("ml", "Bucket Explorer", page_title="Bucket Explorer", rows=ml_buckets()),
    )


@app.get("/ml/features", response_class=HTMLResponse)
def ml_features_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_features.html",
        page_context_bundle("ml", "Feature Library", page_title="Feature Library", rows=ml_features()),
    )


@app.get("/ml/models", response_class=HTMLResponse)
def ml_models_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_models.html",
        page_context_bundle("ml", "Model Registry", page_title="Model Registry", rows=ml_models()),
    )


@app.get("/ml/pipeline", response_class=HTMLResponse)
def ml_pipeline_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_pipeline.html",
        {
            "page_title": "Candidate Strategy Pipeline",
            "rows": ml_promotions(),
            "pipeline_stages": ml_lab_context()["pipeline_stages"],
            **base_page_context("ml", "Candidate Strategy Pipeline"),
        },
    )


@app.get("/ml/draft-board", response_class=HTMLResponse)
def ml_draft_board_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_draft_board.html",
        page_context_bundle("ml", "Draft Board", page_title="Draft Board", rows=merged_ml_draft_board()),
    )


@app.get("/ml/promotions", response_class=HTMLResponse)
def ml_promotions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_promotions.html",
        page_context_bundle("ml", "Promotion / Relegation", page_title="Promotion / Relegation", rows=ml_promotions()),
    )


@app.get("/ml/timeline", response_class=HTMLResponse)
def ml_timeline_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_timeline.html",
        {
            "page_title": "ML Timeline Posts",
            "posts": list_posts("ml"),
            "hypotheses": merged_ml_hypotheses(),
            **base_page_context("ml", "ML Timeline"),
        },
    )


@app.post("/ml/timeline")
async def create_ml_timeline_post(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    def pick(key: str) -> str:
        return payload.get(key, [""])[0]
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO timeline_posts (
                created_at, category, title, team_tags, observation, evidence, interpretation, next_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iso_now(),
                "ml",
                pick("title"),
                pick("team_tags"),
                pick("observation"),
                pick("evidence"),
                pick("interpretation"),
                pick("next_action"),
            ),
        )
        conn.commit()
    return RedirectResponse(url="/ml/timeline", status_code=303)


@app.get("/ml/contamination", response_class=HTMLResponse)
def ml_contamination_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ml_contamination.html",
        page_context_bundle("ml", "Contamination / Leakage Warnings", page_title="Contamination / Leakage Warnings", **ml_lab_context()),
    )


@app.get("/settings/ai", response_class=HTMLResponse)
def ai_settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "ai_settings.html",
        page_context_bundle("general", "AI Settings", page_title="AI Settings", settings=app_settings_snapshot()),
    )


@app.post("/settings/ai")
async def save_ai_settings(request: Request) -> RedirectResponse:
    payload = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    def pick(key: str) -> str:
        return payload.get(key, [""])[0]
    set_setting("ollama_api_key", pick("ollama_api_key"))
    set_setting("ollama_model", pick("ollama_model") or "gpt-oss:120b")
    set_setting("ollama_fallback_model", pick("ollama_fallback_model"))
    set_setting("ollama_timeout_seconds", pick("ollama_timeout_seconds") or "180")
    set_setting("ollama_retry_count", pick("ollama_retry_count") or "2")
    set_setting("development_strategy_generation_model", pick("development_strategy_generation_model") or "kimi-k2.6:cloud")
    set_setting("development_strategy_generation_fallback_model", pick("development_strategy_generation_fallback_model"))
    set_setting("development_strategy_generation_timeout_seconds", pick("development_strategy_generation_timeout_seconds") or "480")
    set_setting("development_strategy_generation_retry_count", pick("development_strategy_generation_retry_count") or "4")
    set_setting("league_maintenance_minutes", pick("league_maintenance_minutes") or "30")
    set_setting("ml_maintenance_minutes", pick("ml_maintenance_minutes") or "180")
    set_setting("archive_maintenance_minutes", pick("archive_maintenance_minutes") or "720")
    set_setting("league_maintenance_enabled", "true" if pick("league_maintenance_enabled") else "false")
    set_setting("ml_maintenance_enabled", "true" if pick("ml_maintenance_enabled") else "false")
    set_setting("archive_maintenance_enabled", "true" if pick("archive_maintenance_enabled") else "false")
    set_setting("archive_push_enabled", "true" if pick("archive_push_enabled") else "false")
    set_setting("archive_repo_url", pick("archive_repo_url"))
    set_setting("archive_repo_branch", pick("archive_repo_branch") or "main")
    set_setting("archive_repo_local_path", pick("archive_repo_local_path") or relative_project_path(DEFAULT_ARCHIVE_REPO_DIR))
    set_setting("auto_review_regeneration_enabled", "true" if pick("auto_review_regeneration_enabled") else "false")
    set_setting("auto_review_regeneration_decisions", pick("auto_review_regeneration_decisions") or "tweak,overhaul")
    set_setting("research_agent_enabled", "true" if pick("research_agent_enabled") else "false")
    set_setting("research_agent_interval_minutes", pick("research_agent_interval_minutes") or "30")
    set_setting("research_agent_duration_hours", pick("research_agent_duration_hours") or "12")
    return RedirectResponse(url="/settings/ai", status_code=303)


@app.get("/api/league")
def api_league() -> JSONResponse:
    context = dashboard_context()
    return JSONResponse(
        {
            "standings": context["standings"],
            "total_equity": context["total_equity"],
            "total_pnl": context["total_pnl"],
            "active_teams": context["active_teams"],
            "league_trades": context["league_trades"],
            "champion": context["champion"],
            "questions": context["questions"],
        }
    )


@app.get("/api/teams/{team_id}")
def api_team(team_id: str) -> JSONResponse:
    instances = {item["id"]: item for item in list_instances()}
    if team_id not in instances:
        raise HTTPException(status_code=404, detail="Team not found")
    return JSONResponse(
        {
            "team": instances[team_id],
            "latest": dict(latest_snapshot_map().get(team_id) or {}),
            "trade_stats": trade_aggregates(team_id),
            "trades": [dict(row) for row in team_trade_rows(team_id)],
        }
    )


@app.get("/api/trades")
def api_trades() -> JSONResponse:
    return JSONResponse(trade_explorer_rows())


@app.get("/api/development")
def api_development() -> JSONResponse:
    return JSONResponse(
        {
            "dashboard": development_league_context(),
            "schedule": development_schedule_context(),
        }
    )


@app.get("/api/ml")
def api_ml() -> JSONResponse:
    return JSONResponse(
        {
            "hypotheses": merged_ml_hypotheses(),
            "buckets": ml_buckets(),
            "features": ml_features(),
            "models": ml_models(),
            "draft_board": merged_ml_draft_board(),
            "promotions": ml_promotions(),
        }
    )


@app.get("/api/research")
def api_research() -> JSONResponse:
    return JSONResponse(research_playground_context())


@app.get("/api/ml/workbench")
def api_ml_workbench(request: Request) -> JSONResponse:
    sync_ml_platform_registry()
    queue_status = request.query_params.get("status", "all").strip().lower() or "all"
    selected_queue_id = int(request.query_params.get("queue_id", "0") or 0) or None
    selected_run_slug = request.query_params.get("run_slug", "").strip()
    compare_a = request.query_params.get("compare_a", "").strip()
    compare_b = request.query_params.get("compare_b", "").strip()
    return JSONResponse(ml_workbench_context(queue_status, selected_queue_id, selected_run_slug, compare_a, compare_b))


@app.post("/api/ml/queue")
async def api_ml_queue(request: Request) -> JSONResponse:
    payload = json.loads((await request.body()).decode("utf-8") or "{}")
    lead_question = str(payload.get("lead_question", "")).strip()
    if not lead_question:
        raise HTTPException(status_code=400, detail="lead_question is required")
    queue_id = create_ml_queue_item(
        str(payload.get("source_type", "manual")),
        str(payload.get("source_key", f"api:{registry_slug(lead_question)}")),
        str(payload.get("title", "Manual ML Lead")),
        lead_question,
        str(payload.get("rationale", lead_question)),
        thread_id=int(payload.get("thread_id") or 0) or None,
        priority=str(payload.get("priority", "normal")),
    )
    return JSONResponse({"status": "ok", "queue_id": queue_id})


@app.post("/api/ml/queue/{queue_id}/action")
async def api_ml_queue_action(queue_id: int, request: Request) -> JSONResponse:
    payload = json.loads((await request.body()).decode("utf-8") or "{}")
    action = str(payload.get("action", "inspect"))
    if action == "execute":
        result = run_local_ml_executor(queue_id, force=bool(payload.get("force", True)))
        return JSONResponse({"status": "ok", "result": result})
    if action == "retry":
        update_ml_queue_item(queue_id, status="queued", resolution="Queued for retry.")
    elif action == "block":
        update_ml_queue_item(queue_id, status="blocked", resolution=str(payload.get("reason", "Blocked via API.")))
    elif action == "reprioritize":
        update_ml_queue_item(queue_id, priority=str(payload.get("priority", "normal")), resolution="Priority updated via API.")
    else:
        raise HTTPException(status_code=400, detail="Unknown action")
    return JSONResponse({"status": "ok", "queue_id": queue_id, "action": action})


@app.get("/api/settings/ai")
def api_ai_settings() -> JSONResponse:
    settings = app_settings_snapshot()
    settings["ollama_api_key"] = "***configured***" if settings["ollama_api_key"] else ""
    return JSONResponse(settings)


@app.post("/api/maintenance/league")
def api_league_maintenance() -> JSONResponse:
    try:
        run_league_maintenance()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


@app.post("/api/maintenance/pairlists")
def api_pairlist_cycle() -> JSONResponse:
    """Force a CoinGecko pull + rebuild of canonical universes and exchange
    manifests (including the Block Party constellation universe). Normally runs on
    the maintenance loop every pairlist_manifest_minutes; this is the on-demand
    trigger used when standing up a new custom universe."""
    try:
        run_pairlist_manifest_cycle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    universe = get_generated_json("pairlist_universe:block_party", {}) or {}
    return JSONResponse({
        "status": "ok",
        "block_party_count": universe.get("count", 0),
        "block_party_built_at": universe.get("built_at"),
    })


@app.post("/api/maintenance/ml")
def api_ml_maintenance() -> JSONResponse:
    try:
        run_ml_maintenance()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


@app.post("/api/maintenance/ml-telemetry")
def api_ml_telemetry_cycle() -> JSONResponse:
    try:
        run_ml_telemetry_cycle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


@app.post("/api/maintenance/ml-evolution")
def api_ml_evolution_review() -> JSONResponse:
    try:
        run_ml_evolution_review()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


@app.post("/api/maintenance/research")
def api_research_maintenance() -> JSONResponse:
    try:
        run_research_maintenance()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


@app.post("/api/maintenance/chronicle")
def api_chronicle_cycle() -> JSONResponse:
    """Manually write today's chapter (the 'Write today's chapter' button)."""
    try:
        record = run_chronicle_cycle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok", "chronicle": record})


@app.post("/api/chat")
async def api_chat(request: Request) -> JSONResponse:
    payload = json.loads((await request.body()).decode("utf-8") or "{}")
    message = payload.get("message", "").strip()
    scope = payload.get("scope", "general")
    current_page = payload.get("current_page", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")
    context_payload = build_chat_scope_payload(scope)
    relevant_files = context_payload.get("relevant_files", []) if isinstance(context_payload, dict) else []
    try:
        reply = ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the assistant for a local algo trading research dashboard. "
                        "Answer using the provided page context only. "
                        "If relevant_files are present, you should inspect them first and treat them as primary evidence for provenance, implementation, notebook, script, config, and experiment questions. "
                        "Explicitly mention file paths when they answer the question. "
                        "If a referenced artifact path is listed but does not exist, say that clearly. "
                        "Do not claim ML results are live scoreboard proof. "
                        "If the scope is a team page, explain strategy behavior using the strategy excerpt and data supplied."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "selected_scope": scope,
                            "current_page": current_page,
                            "question": message,
                            "relevant_file_inventory": [
                                {
                                    "label": item.get("label", ""),
                                    "path": item.get("path", ""),
                                    "exists": item.get("exists", False),
                                }
                                for item in relevant_files
                            ],
                            "context": context_payload,
                        }
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"reply": reply})


@app.post("/api/sync")
def api_sync() -> JSONResponse:
    return JSONResponse(run_sync())


@app.post("/sync")
def sync_and_redirect() -> RedirectResponse:
    run_sync()
    return RedirectResponse(url="/", status_code=303)
