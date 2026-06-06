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
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        ensure_column(conn, "timeline_posts", "category", "TEXT NOT NULL DEFAULT 'league'")
        for column_name, ddl in (
            ("follow_up_status", "TEXT NOT NULL DEFAULT ''"),
            ("follow_up_message", "TEXT NOT NULL DEFAULT ''"),
            ("follow_up_queued_at", "TEXT"),
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
        ):
            ensure_column(conn, "dev_candidates", column_name, ddl)
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


def development_shift_trade_stats(db_path: Path | None, started_at: datetime | None, stopped_at: datetime | None) -> dict[str, Any]:
    stats = {
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_roi": 0.0,
        "realized_pnl": 0.0,
    }
    started_utc = normalize_utc(started_at)
    stopped_utc = normalize_utc(stopped_at)
    if not db_path or not db_path.exists() or not started_utc or not stopped_utc:
        return stats
    try:
        with sqlite3.connect(db_path) as source:
            source.row_factory = sqlite3.Row
            rows = source.execute(
                """
                SELECT close_profit, close_profit_abs, realized_profit, close_date
                FROM trades
                WHERE is_open = 0
                """
            ).fetchall()
    except Exception:
        return stats
    filtered: list[sqlite3.Row] = []
    for row in rows:
        close_date = normalize_utc(resolve_optional_datetime(str(row["close_date"] or "")))
        if close_date and started_utc <= close_date <= stopped_utc:
            filtered.append(row)
    if not filtered:
        return stats
    stats["closed_trades"] = len(filtered)
    stats["wins"] = sum(1 for row in filtered if parse_float(row["close_profit_abs"]) > 0)
    stats["losses"] = max(0, len(filtered) - stats["wins"])
    stats["win_rate"] = percentage(stats["wins"], len(filtered))
    stats["avg_roi"] = percentage(sum(parse_float(row["close_profit"]) for row in filtered), len(filtered))
    stats["realized_pnl"] = round(
        sum(parse_float(row["realized_profit"]) or parse_float(row["close_profit_abs"]) for row in filtered),
        4,
    )
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
) -> str:
    profile = POST_SHIFT_STYLE_PROFILES[style]
    suggestions: list[str] = []
    if reliability_score < 12.0:
        suggestions.append("Fix runtime reliability and API coverage before regenerating; the shift evidence is not clean enough yet.")
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
    return " ".join(unique[:3])


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
                    max_drawdown = ?, worst_open_trade = ?, data_quality = ?, overall_score = ?, grade = ?,
                    decision_bucket = ?, evidence_confidence = ?, recommendation = ?, summary = ?,
                    mutation_brief = ?, rubric_json = ?, updated_at = ?
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
                    max_drawdown, worst_open_trade, data_quality, overall_score, grade,
                    decision_bucket, evidence_confidence, recommendation, summary,
                    mutation_brief, rubric_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    lifecycle_state = str(candidate.get("lifecycle_state") or "")
    if lifecycle_state in {"cut_archived", "archived"}:
        return update_development_post_shift_follow_up(review_key, "skipped", "Archived candidates do not auto-regenerate.")
    generation_status = str(candidate.get("generation_status") or "")
    if generation_status in {"queued", "generating"}:
        return update_development_post_shift_follow_up(review_key, "already_queued", f"Candidate is already {generation_status}.")
    try:
        queue_candidate_strategy_generation(candidate_id, force=True)
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
    db_path = resolve_path(candidate.get("db_path"))
    trade_stats = development_shift_trade_stats(db_path, started_at, stopped_at)
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
    reliability_score, reliability_note = score_post_shift_reliability(runtime_hours, scheduled_hours, data_quality, heartbeat_ratio)
    activity_score, activity_note = score_post_shift_activity(style, trade_pace_per_24h, closed_trades, runtime_hours)
    profitability_score, profitability_note = score_post_shift_profitability(style, realized_pnl, win_rate, avg_roi, closed_trades, runtime_hours)
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
    )
    rubric = [
        {"label": "Reliability", "score": reliability_score, "max_score": 20, "note": reliability_note},
        {"label": "Behavior Fit", "score": activity_score, "max_score": 25, "note": activity_note},
        {"label": "Economics", "score": profitability_score, "max_score": 30, "note": profitability_note},
        {"label": "Risk", "score": risk_score, "max_score": 25, "note": risk_note},
    ]
    summary = (
        f"{candidate.get('name', 'Candidate')} posted a {grade} shift ({overall_score:.1f}/100). "
        f"It covered {runtime_hours:.1f}/{scheduled_hours:.1f} scheduled hours, closed {closed_trades} trades "
        f"({trade_pace_per_24h:.1f}/24h pace), finished {realized_pnl:+.2f} realized, and hit {max_drawdown:.1f}% max drawdown. "
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
                equity, closed_trades, realized_pnl, unrealized_pnl, worst_open_trade,
                max_drawdown, last_trade_at, status_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(candidate["id"]),
                iso_now(),
                candidate.get("runtime_status", "paused"),
                int(bool(candidate.get("heartbeat_ok"))),
                candidate.get("data_quality", "unknown"),
                parse_float(candidate.get("equity")),
                int(candidate.get("closed_trades") or 0),
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
        }
        if latest_review
        else {},
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Generate one conservative runnable Freqtrade strategy for dry-run only. "
                "Return strict JSON only with keys strategy_code, implementation_summary, assumptions, warnings, suggested_timeframe, minimal_config_notes. "
                "strategy_code must be a complete Python file using the provided class_name exactly. "
                "Keep assumptions minimal, avoid exotic dependencies, no markdown fences, and keep arrays short."
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


def deterministic_candidate_config(candidate: dict[str, Any]) -> dict[str, Any]:
    base_config = load_json(PROJECT_DIR / "user_data" / "config.json", {})
    config = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    config["bot_name"] = development_container_name(candidate)
    config["dry_run"] = True
    config["initial_state"] = "running"
    config["timeframe"] = str(candidate.get("suggested_timeframe") or candidate.get("timeframe") or config.get("timeframe") or "5m")
    config.setdefault("exchange", {})
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
    db_runtime_name = f"{registry_slug(str(candidate.get('slug') or candidate.get('name') or 'candidate'))}.sqlite"
    class_name = str(candidate.get("strategy_class_name") or safe_strategy_class_name(str(candidate.get("name", ""))))
    api_port = int(candidate.get("api_port") or 18080 + int(candidate["id"]))
    config_payload = deterministic_candidate_config({**candidate, "api_port": api_port})
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    start_script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        $existing = docker ps -aq --filter "name=^{container_name}$"
        if ($existing) {{
          docker start "{container_name}" | Out-Null
        }} else {{
          docker run -d --name "{container_name}" --restart unless-stopped -v "{PROJECT_DIR / 'user_data'}:/freqtrade/user_data" -p "127.0.0.1:{api_port}:{api_port}" --entrypoint sh freqtradeorg/freqtrade:stable -lc "mkdir -p /freqtrade/runtime && freqtrade trade --logfile '/freqtrade/{relative_project_path(log_path)}' --db-url 'sqlite:////freqtrade/runtime/{db_runtime_name}' --config '/freqtrade/{relative_project_path(config_path)}' --strategy-path '/freqtrade/user_data/strategies/development' --strategy '{class_name}'"
        }}
        """
    ).strip()
    stop_script = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        $existing = docker ps -aq --filter "name=^{container_name}$"
        if ($existing) {{
          docker stop "{container_name}" | Out-Null
          try {{
            docker cp "{container_name}:/freqtrade/runtime/{db_runtime_name}" "{db_path}" 2>$null | Out-Null
          }} catch {{
          }}
        }}
        Write-Output "Stopped {container_name}"
        """
    ).strip()
    start_script_path.write_text(start_script + "\n", encoding="utf-8")
    stop_script_path.write_text(stop_script + "\n", encoding="utf-8")
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
        lifecycle_state="instance_assembled",
        runtime_status="paused",
        status_detail="Instance assembled. Needs shift assignment.",
    )
    development_runtime_event(candidate_id, "assembly", "Instance assembled.", relative_project_path(config_path))


def queue_candidate_strategy_generation(candidate_id: int, force: bool = False) -> None:
    candidate = get_development_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.get("generation_status") in {"queued", "generating"} and not force:
        return
    update_development_candidate(
        candidate_id,
        generation_status="queued",
        generation_model="",
        generation_prompt="",
        generation_error="",
        generation_progress="Queued for Kimi strategy generation.",
        implementation_summary="",
        generation_assumptions="",
        generation_warnings="",
        suggested_timeframe="",
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
    development_runtime_event(candidate_id, "generation", "Strategy generation queued.", pick_strategy_generation_model())


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
            generation_timeout_seconds = max(60.0, parse_float(get_setting("development_strategy_generation_timeout_seconds", "240")))
            generation_retry_count = max(0, parse_intish(get_setting("development_strategy_generation_retry_count", "3")))
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
            update_development_candidate(
                candidate_id,
                generation_status="generated",
                generation_model=generation_model,
                generation_prompt=prompt_used,
                generated_at=iso_now(),
                generation_progress="Strategy file generated.",
                implementation_summary=str(payload.get("implementation_summary", "")),
                generation_assumptions=json.dumps(payload.get("assumptions", []), indent=2),
                generation_warnings=json.dumps(payload.get("warnings", []), indent=2),
                suggested_timeframe=str(payload.get("suggested_timeframe", "")),
                minimal_config_notes=json.dumps(payload.get("minimal_config_notes", []), indent=2),
                strategy_path=relative_project_path(strategy_path),
                strategy_class_name=class_name,
                validation_status="passed" if valid else "failed",
                validation_error="\n".join(validation_errors),
                validated_at=iso_now(),
                lifecycle_state="implemented",
                status_detail="Strategy generated and validated. Needs human review." if valid else "Strategy generated but validation failed.",
            )
            development_runtime_event(candidate_id, "generation", "Strategy generated.", relative_project_path(strategy_path))
            development_runtime_event(candidate_id, "validation", "Validation passed." if valid else "Validation failed.", "\n".join(validation_errors))
        except Exception as exc:  # noqa: BLE001
            update_development_candidate(
                candidate_id,
                generation_status="failed",
                generation_error=str(exc),
                generation_progress=f"Generation failed after retries: {exc}",
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


def sync_development_pipeline() -> None:
    now_local = local_now()
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
        queue_candidate_strategy_generation(candidate_id, force=action == "regenerate_strategy")
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


def seed_default_settings() -> None:
    defaults = {
        "ollama_api_key": "",
        "ollama_model": "gpt-oss:120b",
        "ollama_fallback_model": "",
        "development_strategy_generation_model": "kimi-k2.6:cloud",
        "development_strategy_generation_fallback_model": "",
        "development_strategy_generation_timeout_seconds": "240",
        "development_strategy_generation_retry_count": "3",
        "ollama_timeout_seconds": "120",
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
        "research_agent_enabled": "true",
        "research_agent_interval_minutes": "30",
        "research_agent_duration_hours": "12",
        "league_maintenance_last_run": "",
        "ml_maintenance_last_run": "",
    }
    for key, value in defaults.items():
        if get_setting(key, "") == "":
            set_setting(key, value)


def app_settings_snapshot() -> dict[str, str]:
    return {
        "ollama_api_key": get_setting("ollama_api_key", ""),
        "ollama_model": get_setting("ollama_model", "gpt-oss:120b"),
        "ollama_fallback_model": get_setting("ollama_fallback_model", ""),
        "development_strategy_generation_model": get_setting("development_strategy_generation_model", "kimi-k2.6:cloud"),
        "development_strategy_generation_fallback_model": get_setting("development_strategy_generation_fallback_model", ""),
        "development_strategy_generation_timeout_seconds": get_setting("development_strategy_generation_timeout_seconds", "240"),
        "development_strategy_generation_retry_count": get_setting("development_strategy_generation_retry_count", "3"),
        "ollama_timeout_seconds": get_setting("ollama_timeout_seconds", "120"),
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
        "research_agent_enabled": get_setting("research_agent_enabled", "true"),
        "research_agent_interval_minutes": get_setting("research_agent_interval_minutes", "30"),
        "research_agent_duration_hours": get_setting("research_agent_duration_hours", "12"),
        "league_maintenance_last_run": get_setting("league_maintenance_last_run", ""),
        "ml_maintenance_last_run": get_setting("ml_maintenance_last_run", ""),
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
    return result


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


def upsert_research_index_entry(source_type: str, source_key: str, title: str, content: str, tags: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO research_index_entries (source_type, source_key, title, content, tags, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                source_type=excluded.source_type,
                title=excluded.title,
                content=excluded.content,
                tags=excluded.tags,
                updated_at=excluded.updated_at
            """,
            (source_type, source_key, title, content, tags, iso_now()),
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


def search_research_index(query: str, limit: int = 8) -> list[dict[str, Any]]:
    tokens = tokenize_search(query)
    if not tokens:
        return recent_research_index_entries(limit)
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT source_type, source_key, title, content, tags, updated_at
            FROM research_index_entries
            ORDER BY updated_at DESC, id DESC
            LIMIT 400
            """
        ).fetchall()
    scored: list[tuple[int, dict[str, Any]]] = []
    for raw_row in rows:
        row = dict(raw_row)
        haystack = " ".join(
            [
                str(row.get("title", "")).lower(),
                str(row.get("content", "")).lower(),
                str(row.get("tags", "")).lower(),
            ]
        )
        score = 0
        for token in tokens:
            if token in str(row.get("title", "")).lower():
                score += 4
            if token in str(row.get("tags", "")).lower():
                score += 2
            if token in haystack:
                score += 1
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1].get("updated_at", "")), reverse=False)
    return [row for _, row in scored[:limit]]


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
        upsert_research_index_entry("repo", f"repo:{relative}", relative, content, tags)


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
    tags = " ".join(tokenize_search(question + " " + title + " " + " ".join(citations)))
    upsert_research_index_entry(
        "research-update",
        f"research-update:{update_id}",
        f"{question} :: {title}",
        content,
        tags,
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


def run_ml_maintenance() -> None:
    executed_runs: list[dict[str, Any]] = []
    for row in list_ml_experiment_queue("queued", limit=3):
        try:
            executed_runs.append(run_local_ml_executor(int(row["id"])))
        except Exception as exc:  # noqa: BLE001
            log_maintenance("ml-executor", "error", f"Queue {row['id']} failed: {exc}")
    queued_leads = list_ml_experiment_queue("queued", limit=6)
    prompt = {
        "hypotheses": merged_ml_hypotheses(),
        "buckets": ml_buckets(),
        "models": ml_models(),
        "draft_board": merged_ml_draft_board(),
        "standings": standings_rows(),
        "executor_results": executed_runs,
        "workbench": {
            "datasets": list_ml_dataset_registry(),
            "feature_set_versions": list_ml_feature_set_versions(),
            "label_spec_versions": list_ml_label_spec_versions(),
            "queued_leads": queued_leads,
            "experiment_runs": list_ml_experiment_runs(8),
            "bucket_candidates": list_ml_bucket_candidates(8),
            "validation_reports": list_ml_validation_reports(8),
            "promotion_recommendations": list_ml_promotion_recommendations(8),
        },
        "note": "You are the ML scouting department. Do not report ML as scoreboard truth. Ask one useful ML question and answer it using only the supplied context."
    }
    content = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "Return strict JSON with keys question, question_rationale, findings, hypothesis_updates, hypothesis_candidates, draft_board_updates, draft_board_candidates, queue_updates, experiment_runs, bucket_candidates, validation_reports, promotion_recommendations. "
                    "findings must be an array of up to 4 objects with title, content, hypothesis_id, status. "
                    "hypothesis_updates must be an array of up to 4 objects with id, status, evidence_quality, next_action. "
                    "hypothesis_candidates must be an array of up to 3 objects with id, name, nickname, description, target, rationale, next_action, status. "
                    "draft_board_updates must be an array of up to 4 objects with prospect_name, evidence_quality, risk_level, backtest_strength, live_readiness, draft_status, notes. "
                    "draft_board_candidates must be an array of up to 3 objects with prospect_name, strategy_family, expected_edge, evidence_quality, risk_level, backtest_strength, live_readiness, draft_status, notes. "
                    "queue_updates must be an array of up to 6 objects with id, status, resolution. "
                    "experiment_runs must be an array of up to 3 objects with queue_id, run_slug, title, status, objective, dataset_id, feature_set_version_id, label_spec_version_id, hypothesis_id, summary, artifact_path, notes. "
                    "bucket_candidates must be an array of up to 6 objects with run_slug, candidate_name, hypothesis_id, feature_conditions, expected_behavior, evidence_quality, contamination_risk, status, next_action. "
                    "validation_reports must be an array of up to 6 objects with run_slug, report_type, summary, metrics_json, contamination_checks, recommendation. "
                    "promotion_recommendations must be an array of up to 4 objects with run_slug, candidate_name, recommendation, rationale, blockers, target_surface. "
                    "Keep updates grounded in supplied context and clearly scouting-oriented. "
                    "When queued leads are present, the local executor may already have produced real runs and validation reports. Review those outputs instead of inventing duplicate artifacts. "
                    "Replace previous AI-managed ML outputs instead of appending to them."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ]
    )
    payload = parse_json_block(content)
    replace_ai_research_questions(
        "ml",
        [
            {
                "question": payload.get("question", ""),
                "rationale": payload.get("question_rationale", ""),
                "status": "active",
            }
        ],
    )
    replace_ml_findings(payload.get("findings", []), payload.get("question", ""))
    replace_generated_content(
        "ml_hypothesis_updates",
        json.dumps(payload.get("hypothesis_updates", [])),
    )
    replace_generated_json("ml_hypothesis_candidates", payload.get("hypothesis_candidates", []))
    replace_generated_json("ml_draft_board_updates", payload.get("draft_board_updates", []))
    replace_generated_json("ml_draft_board_candidates", payload.get("draft_board_candidates", []))

    run_ids: dict[str, int] = {}
    for row in payload.get("experiment_runs", []):
        if not isinstance(row, dict):
            continue
        run_id = upsert_ml_experiment_run(row)
        run_slug = row.get("run_slug") or registry_slug(row.get("title", "experiment-run"))
        run_ids[run_slug] = run_id
        queue_id = parse_intish(row.get("queue_id"))
        if queue_id:
            queue_status = "completed" if str(row.get("status", "")).lower() in {"completed", "validated", "rejected", "archived"} else "running"
            update_ml_queue_item(queue_id, status=queue_status, resolution=row.get("summary", ""))

    for row in payload.get("queue_updates", []):
        if not isinstance(row, dict):
            continue
        queue_id = parse_intish(row.get("id"))
        if queue_id:
            update_ml_queue_item(queue_id, status=row.get("status", "queued"), resolution=row.get("resolution", ""))

    for row in payload.get("bucket_candidates", []):
        if not isinstance(row, dict):
            continue
        candidate = dict(row)
        run_slug = candidate.pop("run_slug", "")
        if not candidate.get("experiment_run_id") and run_slug:
            candidate["experiment_run_id"] = run_ids.get(run_slug) or get_ml_experiment_run_id_by_slug(run_slug)
        add_ml_bucket_candidate(candidate)

    for row in payload.get("validation_reports", []):
        if not isinstance(row, dict):
            continue
        report = dict(row)
        run_slug = report.pop("run_slug", "")
        if not report.get("experiment_run_id") and run_slug:
            report["experiment_run_id"] = run_ids.get(run_slug) or get_ml_experiment_run_id_by_slug(run_slug)
        add_ml_validation_report(report)

    for row in payload.get("promotion_recommendations", []):
        if not isinstance(row, dict):
            continue
        recommendation = dict(row)
        run_slug = recommendation.pop("run_slug", "")
        if not recommendation.get("experiment_run_id") and run_slug:
            recommendation["experiment_run_id"] = run_ids.get(run_slug) or get_ml_experiment_run_id_by_slug(run_slug)
        add_ml_promotion_recommendation(recommendation)

    set_setting("ml_maintenance_last_run", iso_now())
    log_maintenance("ml", "success", "ML AI-managed findings, queue execution contracts, hypothesis overlays, and draft board overlays replaced.")


def run_research_maintenance() -> None:
    processed = 0
    for thread in due_research_threads()[:3]:
        thread_id = int(thread["id"])
        started_at = datetime.fromisoformat(str(thread["started_at"]))
        expires_at = started_at + timedelta(hours=int(thread.get("duration_hours") or 12))
        updates = list_research_thread_updates(thread_id, limit=12)
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
        "archive_last_run": "archive_maintenance_enabled",
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
        if maintenance_due("archive_last_run", "archive_maintenance_minutes"):
            try:
                run_archive_maintenance()
            except Exception as exc:  # noqa: BLE001
                log_maintenance("background", "error", f"Archive maintenance failed: {exc}")
        if get_setting("ollama_api_key", ""):
            if maintenance_due("league_maintenance_last_run", "league_maintenance_minutes"):
                try:
                    run_league_maintenance()
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("background", "error", f"League maintenance failed: {exc}")
            if maintenance_due("ml_maintenance_last_run", "ml_maintenance_minutes"):
                try:
                    run_ml_maintenance()
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("background", "error", f"ML maintenance failed: {exc}")
            if get_setting("research_agent_enabled", "true").lower() == "true":
                try:
                    run_research_maintenance()
                except Exception as exc:  # noqa: BLE001
                    log_maintenance("background", "error", f"Research maintenance failed: {exc}")
        time.sleep(60)


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
    index_hits = search_research_index(search_query, limit=10) if search_query else recent_research_index_entries(10)
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


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        page_context_bundle(
            "dashboard",
            "League Dashboard",
            page_title="League Dashboard",
            **dashboard_context(),
        ),
    )


@app.get("/standings", response_class=HTMLResponse)
def standings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "standings.html",
        page_context_bundle("standings", "Standings", page_title="Standings", rows=standings_rows()),
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
            **base_page_context(f"team:{team_id}", team["display_name"], team_id),
        },
    )


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
    return templates.TemplateResponse(
        request,
        "development_candidate.html",
        {
            "page_title": candidate["name"],
            "candidate": candidate,
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
def power_rankings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "power_rankings.html",
        page_context_bundle("power-rankings", "Power Rankings", page_title="Power Rankings", rows=compute_power_rankings()),
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


@app.get("/backtests", response_class=HTMLResponse)
def backtests_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "backtests.html",
        page_context_bundle("dashboard", "Backtest Archive", page_title="Backtest Archive", rows=parse_backtest_archive()),
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
    scope_filter = request.query_params.get("scope", "all").strip().lower() or "all"
    tier_filter = request.query_params.get("tier", "").strip()
    shift_filter = request.query_params.get("shift", "").strip().upper()
    sample_quality_filter = request.query_params.get("sample_quality", "").strip().lower()
    long_short_filter = request.query_params.get("long_short", "").strip().lower()
    timeframe_filter = request.query_params.get("timeframe", "").strip()
    universe_filter = request.query_params.get("q", "").strip()
    runtime_bucket = request.query_params.get("runtime_bucket", "").strip().lower()
    return templates.TemplateResponse(
        request,
        "ml_home.html",
        page_context_bundle(
            "ml",
            "ML Lab Home",
            page_title="ML Lab Home",
            **ml_lab_context(
                scope_filter,
                tier_filter,
                shift_filter,
                sample_quality_filter,
                long_short_filter,
                timeframe_filter,
                universe_filter,
                runtime_bucket,
            ),
        ),
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
    set_setting("ollama_timeout_seconds", pick("ollama_timeout_seconds") or "120")
    set_setting("ollama_retry_count", pick("ollama_retry_count") or "2")
    set_setting("development_strategy_generation_model", pick("development_strategy_generation_model") or "kimi-k2.6:cloud")
    set_setting("development_strategy_generation_fallback_model", pick("development_strategy_generation_fallback_model"))
    set_setting("development_strategy_generation_timeout_seconds", pick("development_strategy_generation_timeout_seconds") or "240")
    set_setting("development_strategy_generation_retry_count", pick("development_strategy_generation_retry_count") or "3")
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


@app.post("/api/maintenance/ml")
def api_ml_maintenance() -> JSONResponse:
    try:
        run_ml_maintenance()
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
