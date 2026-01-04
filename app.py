# app.py



import streamlit as st
import pandas as pd
import sqlite3
import json
import io
import os
from dataclasses import dataclass
from datetime import datetime

# ✅ Matplotlib (Altair 회피)
import matplotlib.pyplot as plt

# PDF (ReportLab)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Optional PDF merge (if installed)
try:
    from PyPDF2 import PdfReader, PdfWriter
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False


# =============================
# App Meta
# =============================
APP_VERSION = "0.6.2"  # ✅ 업데이트 버전(Altair 회피 패치)
DEVELOPER_NAME = "한동석"
COPYRIGHT_TEXT = f"© {datetime.now().year} {DEVELOPER_NAME}. All rights reserved."
DB_PATH = "mvp_finance.db"

st.set_page_config(page_title="원가·손익·BEP 경영분석", layout="wide")


# =============================
# DB
# =============================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS periods (
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        payload TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (year, month)
    )
    """)
    return conn


def save_period(year: int, month: int, payload: dict):
    conn = db_conn()
    conn.execute(
        "INSERT OR REPLACE INTO periods(year, month, payload, updated_at) VALUES (?,?,?,?)",
        (year, month, json.dumps(payload, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()


def load_period(year: int, month: int) -> dict:
    conn = db_conn()
    cur = conn.execute("SELECT payload FROM periods WHERE year=? AND month=?", (year, month))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def list_periods():
    conn = db_conn()
    cur = conn.execute("SELECT year, month, updated_at FROM periods ORDER BY year DESC, month DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def load_periods_range(year: int, months: list[int]) -> list[dict]:
    if not months:
        return []
    conn = db_conn()
    placeholders = ",".join(["?"] * len(months))
    query = f"SELECT month, payload FROM periods WHERE year=? AND month IN ({placeholders}) ORDER BY month"
    cur = conn.execute(query, [year, *months])
    rows = cur.fetchall()
    conn.close()

    out = []
    for m, payload_json in rows:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
        payload["_year"] = int(year)
        payload["_month"] = int(m)
        out.append(payload)
    return out


def delete_period(year: int, month: int):
    conn = db_conn()
    conn.execute("DELETE FROM periods WHERE year=? AND month=?", (year, month))
    conn.commit()
    conn.close()


# =============================
# Tax (KR) - annualized estimate
# =============================
def corp_tax_national_kr(tax_base_krw: float) -> float:
    tb = max(0.0, float(tax_base_krw))
    if tb <= 200_000_000:
        return tb * 0.09
    elif tb <= 20_000_000_000:
        return tb * 0.19 - 20_000_000
    elif tb <= 300_000_000_000:
        return tb * 0.21 - 420_000_000
    else:
        return tb * 0.24 - 9_420_000_000


def corp_tax_total_kr(tax_base_krw: float, include_local: bool = True) -> dict:
    national = corp_tax_national_kr(tax_base_krw)
    local = national * 0.10 if include_local else 0.0
    total = national + local
    return {
        "tax_base": max(0.0, float(tax_base_krw)),
        "national_cit": max(0.0, national),
        "local_income_tax": max(0.0, local),
        "total_tax": max(0.0, total),
        "assumption": "지방소득세 = 법인세(국세) × 10% (경영관리 추정)"
    }


def estimate_monthly_tax_from_pretax_annualized(monthly_pretax: float, include_local: bool = True) -> dict:
    pretax_m = float(monthly_pretax)
    if pretax_m <= 0:
        return {
            "annualized_tax_base": 0.0,
            "monthly_tax": 0.0,
            "detail": corp_tax_total_kr(0.0, include_local=include_local),
            "note": "당월 이익이 0 이하 → 월 법인세 0(추정)"
        }
    annual_base = pretax_m * 12.0
    detail = corp_tax_total_kr(annual_base, include_local=include_local)
    monthly_tax = detail["total_tax"] / 12.0
    return {
        "annualized_tax_base": annual_base,
        "monthly_tax": monthly_tax,
        "detail": detail,
        "note": "연환산(월×12) 기준 월추정 법인세"
    }


# =============================
# Helpers
# =============================
def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def grade_cm_ratio(cm_ratio: float) -> str:
    if cm_ratio >= 0.30:
        return "안정"
    if cm_ratio >= 0.20:
        return "주의"
    if cm_ratio >= 0.10:
        return "위험"
    return "심각"


def to_money(x: float) -> str:
    return f"{x:,.0f}"


def ratio_str(x: float) -> str:
    return f"{x * 100:,.1f}%"


def register_korean_font_if_possible() -> str:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\Malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_name = "KOR_FONT"
                pdfmetrics.registerFont(TTFont(font_name, path))
                return font_name
            except Exception:
                pass
    return "Helvetica"


# ✅ 직접재료비는 '당기사용원재료'만 원가 포함
def get_dm_used_only(cost_items: dict) -> float:
    try:
        return float(cost_items.get("직접재료비", {}).get("당기사용원재료", 0.0) or 0.0)
    except Exception:
        return 0.0


def months_for_period(selected_month: int, kind: str) -> list[int]:
    m = int(selected_month)
    if kind == "월별(선택월)":
        return [m]
    if kind == "분기누적(QTD)":
        q = (m - 1) // 3 + 1
        start = (q - 1) * 3 + 1
        return list(range(start, m + 1))
    if kind == "연도누적(YTD)":
        return list(range(1, m + 1))
    return list(range(1, 13))


# ✅ Altair 회피: Matplotlib 라인차트
def plot_lines_matplotlib(df: pd.DataFrame, title: str, height_px: int = 320):
    """
    df: index = x축(월), columns = series
    """
    if df is None or df.empty:
        st.info("그래프를 표시할 데이터가 없습니다.")
        return

    # 높이 픽셀을 인치로 대략 변환(96dpi 가정)
    fig_h = max(2.5, height_px / 120)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    for col in df.columns:
        ax.plot(df.index, df[col].values, marker="o", label=str(col))
    ax.set_title(title)
    ax.set_xlabel("월")
    ax.set_ylabel("금액")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# =============================
# Default templates
# =============================
DEFAULT_COST_ITEMS = {
    "직접재료비": [
        "기초원재료",
        "당기원재료입고",
        "당기사용원재료",
        "기말원재료",
    ],
    "직접노무비": [
        "생산직 급여",
        "생산직 상여/수당",
        "생산직 4대보험(회사부담)",
        "생산직 퇴직급여충당금",
        "기타 직접노무"
    ],
    "제조간접비": [
        "부재료/잉크/접착제/철선 등(소모성)",
        "포장재/부포재/랩/밴드 등(소모성)",
        "외주가공비",
        "공정 소모품/부품/보수자재",
        "공무/현장관리 급여",
        "공무/현장관리 4대보험",
        "공무/현장관리 퇴직급여충당금",
        "전기료",
        "가스/연료비",
        "수도/환경비",
        "수선비/정비비",
        "공구기구비/소모품",
        "안전/방역/청소비",
        "공장 임차료",
        "보험료(공장/설비)",
        "차량관리비(공장)",
        "운반비(공장 내/외)",
        "설비 감가상각비",
        "공장 건물 감가상각비",
        "기타 제조간접비"
    ]
}

DEFAULT_SGA_ITEMS = [
    ("사무직 급여", "고정"),
    ("상여/수당/포상", "고정"),
    ("4대보험 회사부담", "고정"),
    ("퇴직급여충당금", "고정"),
    ("복리후생비", "고정"),
    ("사무용 소모품", "고정"),
    ("사무실 임차료/렌탈비", "고정"),
    ("통신비", "고정"),
    ("광고선전비", "변동"),
    ("판매수수료", "변동"),
    ("물류/배송비", "변동"),
    ("차량 렌트료(영업)", "고정"),
    ("차량 관리비(영업)", "고정"),
    ("지급수수료", "변동"),
    ("세금과공과", "고정"),
    ("대손상각비", "변동"),
    ("접대비", "고정"),
    ("출장비", "변동"),
    ("도서/자료 구입비", "고정"),
    ("특허/상표 유지비", "고정"),
    ("보험료(사무/영업)", "고정"),
    ("소프트웨어 사용료", "고정"),
    ("클라우드 서버 비용", "고정"),
    ("홈페이지 유지비", "고정"),
    ("사무실 비품 감가상각비", "고정"),
    ("사무실 건물 감가상각비", "고정"),
    ("영업용 차량 감가상각비", "고정"),
]

DEFAULT_NONOP_INCOME = ["이자수익", "환차익", "자산처분이익", "잡이익(파지매출 등)"]
DEFAULT_NONOP_EXPENSE = ["이자비용", "환차손", "자산처분손실", "잡손실"]


def normalize_sga_type(typ: str) -> str:
    return "변동" if typ == "변동" else "고정"


def build_default_cost_items():
    cost = {}
    for sec, items in DEFAULT_COST_ITEMS.items():
        cost[sec] = {it: 0.0 for it in items}
    return cost


def build_default_sga_rows():
    return [{"item": name, "amount": 0.0, "type": typ} for name, typ in DEFAULT_SGA_ITEMS]


def build_default_nonop_rows(items):
    return [{"item": it, "amount": 0.0} for it in items]


# =============================
# Core compute
# =============================
@dataclass
class PeriodInput:
    sales: float
    beg_fg: float
    end_fg: float
    cost_items: dict
    sga_rows: list
    nonop_income_rows: list
    nonop_expense_rows: list
    tax_mode: str
    tax_manual: float
    include_local_tax: bool


def sum_cost_section(cost_items: dict, section: str) -> float:
    sec = cost_items.get(section, {})
    return float(sum(float(v or 0) for v in sec.values()))


def compute_all(p: PeriodInput) -> dict:
    dm = get_dm_used_only(p.cost_items)
    dl = sum_cost_section(p.cost_items, "직접노무비")
    moh = sum_cost_section(p.cost_items, "제조간접비")
    cogm = dm + dl + moh

    cogs = p.beg_fg + cogm - p.end_fg

    sga_fixed = sum(float(r["amount"]) for r in p.sga_rows if r["type"] == "고정")
    sga_variable = sum(float(r["amount"]) for r in p.sga_rows if r["type"] == "변동")
    sga_total = sga_fixed + sga_variable

    nonop_income = sum(float(r["amount"]) for r in p.nonop_income_rows)
    nonop_expense = sum(float(r["amount"]) for r in p.nonop_expense_rows)

    gross_profit = p.sales - cogs
    op_profit = gross_profit - sga_total
    pretax = op_profit + nonop_income - nonop_expense

    tax_info = None
    if p.tax_mode == "AUTO":
        tax_info = estimate_monthly_tax_from_pretax_annualized(pretax, include_local=p.include_local_tax)
        tax = tax_info["monthly_tax"]
    elif p.tax_mode == "MANUAL":
        tax = p.tax_manual
    else:
        tax = 0.0

    net_income = pretax - tax

    variable_total = cogs + sga_variable
    cm = p.sales - variable_total
    cm_ratio = safe_div(cm, p.sales)
    bep_sales = safe_div(sga_fixed, cm_ratio) if cm_ratio > 0 else 0.0

    return {
        "dm": dm, "dl": dl, "moh": moh,
        "cogm": cogm,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "sga_fixed": sga_fixed,
        "sga_variable": sga_variable,
        "sga_total": sga_total,
        "op_profit": op_profit,
        "nonop_income": nonop_income,
        "nonop_expense": nonop_expense,
        "pretax": pretax,
        "tax": tax,
        "tax_info": tax_info,
        "net_income": net_income,
        "variable_total": variable_total,
        "cm": cm,
        "cm_ratio": cm_ratio,
        "grade": grade_cm_ratio(cm_ratio),
        "bep_sales": bep_sales,
        "gap_sales_vs_bep": p.sales - bep_sales,
        "op_margin": safe_div(op_profit, p.sales),
        "gross_margin": safe_div(gross_profit, p.sales),
    }


def payload_to_period_input(payload: dict) -> PeriodInput:
    raw_cost = payload.get("cost_items", {}) or build_default_cost_items()
    raw_sga = payload.get("sga_rows", []) or build_default_sga_rows()

    sga = []
    for r in raw_sga:
        sga.append({
            "item": r.get("item", ""),
            "amount": float(r.get("amount", 0.0)),
            "type": normalize_sga_type(r.get("type", "고정"))
        })

    return PeriodInput(
        sales=float(payload.get("sales", 0.0)),
        beg_fg=float(payload.get("beg_fg", 0.0)),
        end_fg=float(payload.get("end_fg", 0.0)),
        cost_items=raw_cost,
        sga_rows=sga,
        nonop_income_rows=[{"item": r.get("item", ""), "amount": float(r.get("amount", 0.0))}
                           for r in payload.get("nonop_income_rows", build_default_nonop_rows(DEFAULT_NONOP_INCOME))],
        nonop_expense_rows=[{"item": r.get("item", ""), "amount": float(r.get("amount", 0.0))}
                            for r in payload.get("nonop_expense_rows", build_default_nonop_rows(DEFAULT_NONOP_EXPENSE))],
        tax_mode=str(payload.get("tax_mode", "AUTO")),
        tax_manual=float(payload.get("tax_manual", 0.0)),
        include_local_tax=bool(payload.get("include_local_tax", True)),
    )


def aggregate_periods(payloads: list[dict]) -> dict:
    total = {
        "sales": 0.0,
        "cogs": 0.0,
        "gross_profit": 0.0,
        "sga_total": 0.0,
        "sga_fixed": 0.0,
        "sga_variable": 0.0,
        "op_profit": 0.0,
        "nonop_income": 0.0,
        "nonop_expense": 0.0,
        "pretax": 0.0,
        "tax": 0.0,
        "net_income": 0.0,
        "variable_total": 0.0,
        "months_included": [],
    }

    for payload in payloads:
        p = payload_to_period_input(payload)
        out = compute_all(p)

        total["sales"] += p.sales
        total["cogs"] += out["cogs"]
        total["gross_profit"] += out["gross_profit"]
        total["sga_total"] += out["sga_total"]
        total["sga_fixed"] += out["sga_fixed"]
        total["sga_variable"] += out["sga_variable"]
        total["op_profit"] += out["op_profit"]
        total["nonop_income"] += out["nonop_income"]
        total["nonop_expense"] += out["nonop_expense"]
        total["pretax"] += out["pretax"]
        total["tax"] += out["tax"]
        total["net_income"] += out["net_income"]
        total["variable_total"] += out["variable_total"]
        total["months_included"].append(int(payload.get("_month", 0)))

    cm = total["sales"] - total["variable_total"]
    cm_ratio = safe_div(cm, total["sales"])
    grade = grade_cm_ratio(cm_ratio)
    bep_sales = safe_div(total["sga_fixed"], cm_ratio) if cm_ratio > 0 else 0.0

    total.update({
        "cm": cm,
        "cm_ratio": cm_ratio,
        "grade": grade,
        "bep_sales": bep_sales,
        "gap_sales_vs_bep": total["sales"] - bep_sales,
        "gross_profit_rate": safe_div(total["gross_profit"], total["sales"]),
        "op_margin": safe_div(total["op_profit"], total["sales"]),
    })
    return total


# =============================
# DataFrames
# =============================
def df_cost_statement(cost_items: dict, cogm: float) -> pd.DataFrame:
    rows = []

    dm_sec = cost_items.get("직접재료비", {})

    beg = float(dm_sec.get("기초원재료", 0.0) or 0.0)
    inbound_default = dm_sec.get("당기원재료입고", None)
    if inbound_default is None:
        inbound_default = dm_sec.get("당기원재료매입", 0.0)
    inbound = float(inbound_default or 0.0)

    used = float(dm_sec.get("당기사용원재료", 0.0) or 0.0)
    end = float(dm_sec.get("기말원재료", 0.0) or 0.0)

    rows.append(["직접재료비(참고)", "기초원재료", beg, ""])
    rows.append(["직접재료비(참고)", "당기원재료입고", inbound, ""])
    rows.append(["직접재료비", "당기사용원재료(원가반영)", used, ratio_str(safe_div(used, cogm)) if cogm else ""])
    rows.append(["직접재료비(참고)", "기말원재료", end, ""])
    rows.append(["직접재료비 소계(원가반영)", "직접재료비 소계", used, ratio_str(safe_div(used, cogm)) if cogm else ""])
    rows.append(["", "", 0.0, ""])

    dl_sum = sum_cost_section(cost_items, "직접노무비")
    for item, amt in cost_items.get("직접노무비", {}).items():
        amt = float(amt or 0)
        rows.append(["직접노무비", item, amt, ratio_str(safe_div(amt, cogm)) if cogm else ""])
    rows.append(["직접노무비 소계", "직접노무비 소계", dl_sum, ratio_str(safe_div(dl_sum, cogm)) if cogm else ""])
    rows.append(["", "", 0.0, ""])

    moh_sum = sum_cost_section(cost_items, "제조간접비")
    for item, amt in cost_items.get("제조간접비", {}).items():
        amt = float(amt or 0)
        rows.append(["제조간접비", item, amt, ratio_str(safe_div(amt, cogm)) if cogm else ""])
    rows.append(["제조간접비 소계", "제조간접비 소계", moh_sum, ratio_str(safe_div(moh_sum, cogm)) if cogm else ""])

    return pd.DataFrame(rows, columns=["구분", "항목", "금액", "당기제조원가 대비 비율"])


def df_pl_statement(p: PeriodInput, out: dict) -> pd.DataFrame:
    lines = [
        ("매출액", p.sales),
        ("매출원가", out["cogs"]),
        ("매출총이익", out["gross_profit"]),
        ("판관비(고정)", out["sga_fixed"]),
        ("판관비(변동)", out["sga_variable"]),
        ("판관비 합계", out["sga_total"]),
        ("영업이익", out["op_profit"]),
        ("영업외수익", out["nonop_income"]),
        ("영업외비용", out["nonop_expense"]),
        ("법인세차감전이익", out["pretax"]),
        ("법인세비용", out["tax"]),
        ("당기순이익", out["net_income"]),
    ]
    df = pd.DataFrame(lines, columns=["항목", "금액"])
    df["매출액 대비 비율"] = df["금액"].apply(lambda v: ratio_str(safe_div(float(v), p.sales)) if p.sales else "")
    return df


def df_bep(out: dict, fixed_base: float, sales_base: float) -> pd.DataFrame:
    return pd.DataFrame([
        ["변동비 합계(매출원가+판관비 변동)", out["variable_total"]],
        ["공헌이익", out["cm"]],
        ["공헌이익률", out["cm_ratio"]],
        ["고정비(판관비 고정)", fixed_base],
        ["BEP 매출액", out["bep_sales"]],
        ["BEP 대비 매출", sales_base - out["bep_sales"]],
        ["공헌이익률 등급", out["grade"]],
    ], columns=["항목", "값"])


def build_agg_pl_df(agg: dict) -> pd.DataFrame:
    sales_base = float(agg.get("sales", 0.0))
    df = pd.DataFrame([
        ["매출액", agg.get("sales", 0.0)],
        ["매출원가(합산)", agg.get("cogs", 0.0)],
        ["매출총이익", agg.get("gross_profit", 0.0)],
        ["판관비(고정)", agg.get("sga_fixed", 0.0)],
        ["판관비(변동)", agg.get("sga_variable", 0.0)],
        ["판관비 합계", agg.get("sga_total", 0.0)],
        ["영업이익", agg.get("op_profit", 0.0)],
        ["영업외수익", agg.get("nonop_income", 0.0)],
        ["영업외비용", agg.get("nonop_expense", 0.0)],
        ["법인세차감전이익", agg.get("pretax", 0.0)],
        ["법인세비용(월합산)", agg.get("tax", 0.0)],
        ["당기순이익", agg.get("net_income", 0.0)],
    ], columns=["항목", "금액"])
    df["매출액 대비 비율"] = df["금액"].apply(lambda v: ratio_str(safe_div(float(v), sales_base)) if sales_base else "")
    return df


# =============================
# Report builders
# =============================
def build_report_html(title: str, period_label: str, sales_base: float, out: dict,
                      df_pl: pd.DataFrame, df_cost: pd.DataFrame | None,
                      tax_mode: str, tax_info: dict | None) -> str:
    tax_note = ""
    if tax_info:
        d = tax_info["detail"]
        tax_note = f"""
        <p><b>법인세(추정) 기준</b><br/>
        연환산 과세표준: {tax_info["annualized_tax_base"]:,.0f}원<br/>
        국세 법인세: {d["national_cit"]:,.0f}원 / 지방소득세(가정): {d["local_income_tax"]:,.0f}원<br/>
        ({d["assumption"]})
        </p>
        """
    else:
        if tax_mode == "OFF":
            tax_note = "<p><b>법인세:</b> 미반영(0원)</p>"
        elif tax_mode == "MANUAL":
            tax_note = "<p><b>법인세:</b> 수동 입력값 반영</p>"

    cost_block = ""
    if df_cost is not None:
        cost_block = f"""
        <h3>제조원가명세서(당기제조원가 대비 비율)</h3>
        <p><small>※ 직접재료비는 '당기사용원재료'만 원가에 반영됩니다. 기초/입고/기말은 참고 표시입니다.</small></p>
        {df_cost.to_html(index=False)}
        """

    html = f"""
    <h2>{title}</h2>
    <p><b>기간:</b> {period_label}</p>

    <h3>핵심 KPI</h3>
    <ul>
      <li>매출액: {sales_base:,.0f}</li>
      <li>영업이익: {out['op_profit']:,.0f} (영업이익률 {out.get('op_margin', 0)*100:,.1f}%)</li>
      <li>공헌이익률: {out['cm_ratio']*100:,.1f}% → <b>{out['grade']}</b></li>
      <li>BEP 매출액: {out['bep_sales']:,.0f}</li>
      <li>BEP 대비 매출: {(sales_base - out['bep_sales']):,.0f}</li>
    </ul>

    {tax_note}

    <h3>손익계산서 요약</h3>
    {df_pl.to_html(index=False)}

    {cost_block}

    <hr/>
    <p><small>
    ※ 본 리포트는 경영관리 목적의 자동 산출 결과입니다.
    특히 법인세는 '연환산 추정' 기반으로 실제 신고세액과 차이가 있을 수 있습니다.
    </small></p>
    """
    return html


def make_excel_report(df_pl: pd.DataFrame, df_cost: pd.DataFrame | None,
                      df_bep_tbl: pd.DataFrame, extra_sheets: dict[str, pd.DataFrame] | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_pl.to_excel(writer, index=False, sheet_name="손익요약")
        df_bep_tbl.to_excel(writer, index=False, sheet_name="BEP")
        if df_cost is not None:
            df_cost.to_excel(writer, index=False, sheet_name="제조원가")
        if extra_sheets:
            for name, df in extra_sheets.items():
                df.to_excel(writer, index=False, sheet_name=name[:31])
    buf.seek(0)
    return buf.getvalue()


def make_pdf_report(title: str, period_label: str, sales_base: float, out: dict,
                    df_pl: pd.DataFrame, tax_mode: str, tax_info: dict | None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    font = register_korean_font_if_possible()

    c.setFont(font, 14)
    c.drawString(18 * mm, height - 18 * mm, title)

    c.setFont(font, 11)
    y = height - 28 * mm
    line = 6.5 * mm

    c.drawString(18 * mm, y, f"기간: {period_label}")
    y -= line

    c.drawString(18 * mm, y, f"매출액: {to_money(sales_base)}")
    y -= line
    c.drawString(18 * mm, y, f"영업이익: {to_money(out['op_profit'])}   |   공헌이익률: {out['cm_ratio']*100:,.1f}% ({out['grade']})")
    y -= line
    c.drawString(18 * mm, y, f"BEP 매출액: {to_money(out['bep_sales'])}   |   BEP 대비 매출: {to_money(sales_base - out['bep_sales'])}")
    y -= (line * 1.2)

    if tax_info:
        d = tax_info["detail"]
        c.drawString(18 * mm, y, f"법인세(추정): {tax_info['note']}")
        y -= line
        c.drawString(18 * mm, y, f"- 연환산 과세표준 {to_money(tax_info['annualized_tax_base'])} | 국세 {to_money(d['national_cit'])} + 지방(10%가정) {to_money(d['local_income_tax'])}")
        y -= (line * 1.2)
    else:
        if tax_mode == "OFF":
            c.drawString(18 * mm, y, "법인세: 미반영(0원)")
            y -= (line * 1.2)
        elif tax_mode == "MANUAL":
            c.drawString(18 * mm, y, "법인세: 수동 입력값 반영")
            y -= (line * 1.2)

    c.setFont(font, 12)
    c.drawString(18 * mm, y, "손익계산서 요약")
    y -= (line * 1.2)

    c.setFont(font, 10)
    for _, row in df_pl.iterrows():
        item = str(row["항목"])
        amt = float(row["금액"])
        ratio = str(row.get("매출액 대비 비율", ""))

        c.drawString(20 * mm, y, item)
        c.drawRightString(width - 50 * mm, y, f"{amt:,.0f}")
        c.drawRightString(width - 18 * mm, y, ratio)

        y -= line
        if y < 18 * mm:
            c.showPage()
            c.setFont(font, 10)
            y = height - 18 * mm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def merge_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
    if not HAS_PYPDF2:
        raise RuntimeError("PyPDF2 not available")
    writer = PdfWriter()
    for b in pdf_bytes_list:
        r = PdfReader(io.BytesIO(b))
        for page in r.pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


def build_compare_report_html(title: str, label_a: str, label_b: str,
                              agg_a: dict, agg_b: dict, comp_df: pd.DataFrame) -> str:
    def kpi_block(lbl: str, agg: dict) -> str:
        sales = float(agg.get("sales", 0.0))
        op = float(agg.get("op_profit", 0.0))
        net = float(agg.get("net_income", 0.0))
        cmr = float(agg.get("cm_ratio", 0.0))
        bep = float(agg.get("bep_sales", 0.0))
        gap = sales - bep
        grade = grade_cm_ratio(cmr)
        opm = safe_div(op, sales)

        return f"""
        <h3>{lbl}</h3>
        <ul>
          <li>매출액: {sales:,.0f}</li>
          <li>영업이익: {op:,.0f} (영업이익률 {opm*100:,.1f}%)</li>
          <li>당기순이익: {net:,.0f}</li>
          <li>공헌이익률: {cmr*100:,.1f}% → <b>{grade}</b></li>
          <li>BEP 매출액: {bep:,.0f}</li>
          <li>BEP 대비 매출: {gap:,.0f}</li>
        </ul>
        """

    html = f"""
    <h2>{title}</h2>
    <p><b>비교:</b> A = {label_a} / B = {label_b}</p>

    <h2>A/B 핵심 KPI</h2>
    <div style="display:flex; gap:24px;">
      <div style="flex:1; border:1px solid #ddd; padding:12px; border-radius:8px;">
        {kpi_block("A", agg_a)}
      </div>
      <div style="flex:1; border:1px solid #ddd; padding:12px; border-radius:8px;">
        {kpi_block("B", agg_b)}
      </div>
    </div>

    <h2>비교표 (A vs B)</h2>
    {comp_df.to_html(index=False)}

    <hr/>
    <p><small>
    ※ 누적/분기/연도 비교는 '저장된 월'만 합산한 경영관리 목적 결과입니다.
    </small></p>
    """
    return html


# =============================
# Trend / Compare
# =============================
def build_year_month_table_for_trend(year: int) -> pd.DataFrame:
    payloads = load_periods_range(year, list(range(1, 13)))
    if not payloads:
        return pd.DataFrame(columns=[
            "월", "매출액", "영업이익", "당기순이익",
            "누적매출", "누적영업이익", "누적당기순이익",
            "공헌이익률", "BEP매출액"
        ])

    rows = []
    for pp in payloads:
        m = int(pp.get("_month", 0))
        p = payload_to_period_input(pp)
        out = compute_all(p)
        rows.append({
            "월": m,
            "매출액": p.sales,
            "영업이익": out["op_profit"],
            "당기순이익": out["net_income"],
            "공헌이익률": out["cm_ratio"],
            "BEP매출액": out["bep_sales"],
        })

    df = pd.DataFrame(rows).sort_values("월")
    df["누적매출"] = df["매출액"].cumsum()
    df["누적영업이익"] = df["영업이익"].cumsum()
    df["누적당기순이익"] = df["당기순이익"].cumsum()
    return df


def build_period_aggregate(year: int, anchor_month: int, kind: str) -> tuple[dict, str, list[int]]:
    months = months_for_period(anchor_month, kind)
    payloads = load_periods_range(year, months)

    if kind == "월별(선택월)":
        label = f"{year}-{anchor_month:02d}"
    elif kind == "분기누적(QTD)":
        q = (anchor_month - 1) // 3 + 1
        label = f"{year} Q{q} (저장분)"
    elif kind == "연도누적(YTD)":
        label = f"{year} YTD (저장분)"
    else:
        label = f"{year} 연도전체(저장분)"

    if not payloads:
        return {}, label, []

    agg = aggregate_periods(payloads)
    included = sorted([int(x) for x in agg.get("months_included", []) if x])
    return agg, label, included


def compare_two(agg_a: dict, agg_b: dict) -> pd.DataFrame:
    keys = [
        ("매출액", "sales", "원"),
        ("매출원가", "cogs", "원"),
        ("매출총이익", "gross_profit", "원"),
        ("판관비(고정)", "sga_fixed", "원"),
        ("판관비(변동)", "sga_variable", "원"),
        ("판관비 합계", "sga_total", "원"),
        ("영업이익", "op_profit", "원"),
        ("법인세차감전이익", "pretax", "원"),
        ("법인세비용(월합산)", "tax", "원"),
        ("당기순이익", "net_income", "원"),
        ("공헌이익률", "cm_ratio", "%"),
        ("BEP 매출액", "bep_sales", "원"),
    ]

    rows = []
    for label, k, unit in keys:
        a = float(agg_a.get(k, 0.0) or 0.0)
        b = float(agg_b.get(k, 0.0) or 0.0)
        delta = a - b
        if unit == "%":
            rows.append({
                "항목": label,
                "A": f"{a*100:,.2f}%",
                "B": f"{b*100:,.2f}%",
                "A-B": f"{delta*100:,.2f}%p",
            })
        else:
            rows.append({
                "항목": label,
                "A": f"{a:,.0f}",
                "B": f"{b:,.0f}",
                "A-B": f"{delta:,.0f}",
            })
    return pd.DataFrame(rows)


# =============================
# UI
# =============================
st.title("제조원가·손익·BEP 경영분석 앱")

with st.sidebar:
    st.header("연도/월 선택")
    now = datetime.now()
    year = st.number_input("연도", value=now.year, min_value=2000, max_value=2100, step=1)
    month = st.number_input("월", value=now.month, min_value=1, max_value=12, step=1)

    saved = list_periods()
    if saved:
        st.caption("저장된 월")
        st.dataframe(pd.DataFrame(saved, columns=["year", "month", "updated_at"]),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("데이터 삭제")
    confirm = st.checkbox("정말 삭제합니다(되돌릴 수 없음).")
    if st.button("선택한 연/월 완전 삭제", use_container_width=True, disabled=not confirm):
        delete_period(int(year), int(month))
        st.success("삭제 완료")
        st.rerun()

defaults = load_period(int(year), int(month))


def dget(key, fallback):
    return defaults.get(key, fallback)


sales_default = float(dget("sales", 0.0))
beg_fg_default = float(dget("beg_fg", 0.0))
end_fg_default = float(dget("end_fg", 0.0))

cost_items_default = dget("cost_items", None) or build_default_cost_items()
sga_rows_default = dget("sga_rows", None) or build_default_sga_rows()
nonop_income_default = dget("nonop_income_rows", None) or build_default_nonop_rows(DEFAULT_NONOP_INCOME)
nonop_expense_default = dget("nonop_expense_rows", None) or build_default_nonop_rows(DEFAULT_NONOP_EXPENSE)
tax_mode_default = dget("tax_mode", "AUTO")
tax_manual_default = float(dget("tax_manual", 0.0))
include_local_default = bool(dget("include_local_tax", True))
auto_dm_default = bool(dget("auto_dm_used", True))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["입력", "결과", "누적 조회", "리포트 다운로드", "비교분석·추이그래프"])


# -----------------------------
# TAB1: INPUT
# -----------------------------
with tab1:
    st.subheader("1) 입력")
    st.caption("※ 브라우저 자동 번역이 켜져 있으면 항목명이 왜곡될 수 있습니다. '이 사이트 번역 안 함' 권장.")

    left, right = st.columns(2)

    with left:
        st.markdown("### 매출 / 완제품 재고")
        sales = st.number_input("매출액", value=sales_default, step=1000.0, format="%.0f")
        beg_fg = st.number_input("기초 완제품 재고", value=beg_fg_default, step=1000.0, format="%.0f")
        end_fg = st.number_input("기말 완제품 재고", value=end_fg_default, step=1000.0, format="%.0f")

        st.markdown("### 제조원가명세서")
        st.info(
            "당기제조원가 = 직접재료비(당기사용원재료만) + 직접노무비 + 제조간접비\n\n"
            "직접재료비는 원재료만, 부재료/외주가공비/포장재/소모품은 제조간접비\n\n"
            "옵션 ON 시: 당기사용원재료 = (기초원재료 + 당기원재료입고 - 기말원재료)"
        )

        auto_dm_used = st.toggle("당기사용원재료 자동계산(기초+입고-기말, 음수면 0)", value=auto_dm_default)

        cost_items = {}
        for sec in ["직접재료비", "직접노무비", "제조간접비"]:
            with st.expander(sec, expanded=True if sec != "제조간접비" else False):
                cost_items[sec] = {}

                if sec == "직접재료비":
                    dm_sec_default = cost_items_default.get(sec, {})

                    beg = st.number_input(
                        "기초원재료", value=float(dm_sec_default.get("기초원재료", 0.0)),
                        step=1000.0, format="%.0f", help="참고용", key="dm_beg"
                    )

                    inbound_default = dm_sec_default.get("당기원재료입고", None)
                    if inbound_default is None:
                        inbound_default = dm_sec_default.get("당기원재료매입", 0.0)

                    inbound = st.number_input(
                        "당기원재료입고", value=float(inbound_default or 0.0),
                        step=1000.0, format="%.0f", help="입고 기준(실물/수불 기준)", key="dm_inbound"
                    )

                    end = st.number_input(
                        "기말원재료", value=float(dm_sec_default.get("기말원재료", 0.0)),
                        step=1000.0, format="%.0f", help="참고용", key="dm_end"
                    )

                    raw_used = float(beg) + float(inbound) - float(end)
                    auto_used = max(0.0, raw_used)

                    used_default = float(dm_sec_default.get("당기사용원재료", 0.0))
                    used_val = auto_used if auto_dm_used else st.number_input(
                        "당기사용원재료(원가반영)", value=used_default,
                        step=1000.0, format="%.0f",
                        help="원가에 반영되는 직접재료비(원재료 사용액)", disabled=auto_dm_used,
                        key="dm_used_manual"
                    )

                    if auto_dm_used:
                        if raw_used < 0:
                            st.warning(f"자동계산(기초+입고-기말) = {to_money(raw_used)} → 음수이므로 0으로 보정되었습니다.")
                        st.caption(f"자동계산된 당기사용원재료(원가반영): {to_money(used_val)}")

                    cost_items[sec]["기초원재료"] = float(beg)
                    cost_items[sec]["당기원재료입고"] = float(inbound)
                    cost_items[sec]["기말원재료"] = float(end)
                    cost_items[sec]["당기사용원재료"] = float(used_val)

                else:
                    for item in list(cost_items_default.get(sec, {}).keys()):
                        val = float(cost_items_default.get(sec, {}).get(item, 0.0))
                        amt = st.number_input(item, value=val, step=1000.0, format="%.0f", key=f"{sec}_{item}")
                        cost_items[sec][item] = float(amt)

                    st.caption("필요 시 항목 추가(선택)")
                    new_name = st.text_input(f"{sec} 추가 항목명", value="", key=f"new_{sec}")
                    if new_name.strip():
                        if new_name not in cost_items[sec]:
                            amt = st.number_input(
                                f"[추가] {new_name}", value=0.0, step=1000.0, format="%.0f", key=f"{sec}_add_{new_name}"
                            )
                            cost_items[sec][new_name] = float(amt)

    with right:
        st.markdown("### 판매관리비 (고정비/변동비)")
    

        sga_rows = []
        for i, r in enumerate(sga_rows_default):
            c1, c2, c3 = st.columns([3, 1.5, 2.5])
            with c1:
                item = st.text_input("항목", value=r.get("item", ""), key=f"sga_item_{i}")
            with c2:
                typ0 = normalize_sga_type(r.get("type", "고정"))
                typ = st.selectbox("구분", ["고정", "변동"], index=["고정", "변동"].index(typ0), key=f"sga_type_{i}")
            with c3:
                amt = st.number_input("금액", value=float(r.get("amount", 0.0)), step=1000.0, format="%.0f", key=f"sga_amt_{i}")
            sga_rows.append({"item": item, "type": typ, "amount": float(amt)})

        st.divider()
        st.caption("판관비 항목 추가(선택)")
        add_name = st.text_input("새 판관비 항목명", value="", key="sga_new_name")
        add_type = st.selectbox("새 항목 구분", ["고정", "변동"], key="sga_new_type")
        add_amt = st.number_input("새 항목 금액", value=0.0, step=1000.0, format="%.0f", key="sga_new_amt")

        if st.button("판관비 항목 추가", use_container_width=True):
            if add_name.strip():
                sga_rows.append({"item": add_name.strip(), "type": add_type, "amount": float(add_amt)})
                st.success("추가됨! (저장 버튼을 누르면 다음에도 유지됩니다)")
            else:
                st.warning("항목명을 입력해 주세요.")

        st.markdown("### 영업외 수익/비용")
        with st.expander("영업외수익", expanded=False):
            nonop_income_rows = []
            for i, r in enumerate(nonop_income_default):
                c1, c2 = st.columns([3, 2])
                with c1:
                    item = st.text_input("항목", value=r.get("item", ""), key=f"noi_item_{i}")
                with c2:
                    amt = st.number_input("금액", value=float(r.get("amount", 0.0)), step=1000.0, format="%.0f", key=f"noi_amt_{i}")
                nonop_income_rows.append({"item": item, "amount": float(amt)})

        with st.expander("영업외비용", expanded=False):
            nonop_expense_rows = []
            for i, r in enumerate(nonop_expense_default):
                c1, c2 = st.columns([3, 2])
                with c1:
                    item = st.text_input("항목", value=r.get("item", ""), key=f"noe_item_{i}")
                with c2:
                    amt = st.number_input("금액", value=float(r.get("amount", 0.0)), step=1000.0, format="%.0f", key=f"noe_amt_{i}")
                nonop_expense_rows.append({"item": item, "amount": float(amt)})

        st.markdown("### 법인세 설정")
        tax_mode = st.selectbox("법인세 반영 방식", ["AUTO", "MANUAL", "OFF"],
                                index=["AUTO", "MANUAL", "OFF"].index(tax_mode_default))
        include_local_tax = st.checkbox("지방소득세(국세의 10%) 포함(추정)", value=include_local_default)
        tax_manual = 0.0
        if tax_mode == "MANUAL":
            tax_manual = st.number_input("법인세비용(수동 입력)", value=tax_manual_default, step=1000.0, format="%.0f")

    payload = {
        "sales": float(sales),
        "beg_fg": float(beg_fg),
        "end_fg": float(end_fg),
        "cost_items": cost_items,
        "sga_rows": sga_rows,
        "nonop_income_rows": nonop_income_rows,
        "nonop_expense_rows": nonop_expense_rows,
        "tax_mode": tax_mode,
        "tax_manual": float(tax_manual) if tax_mode == "MANUAL" else float(tax_manual_default),
        "include_local_tax": bool(include_local_tax),
        "auto_dm_used": bool(auto_dm_used),
    }

    s1, s2 = st.columns(2)
    with s1:
        if st.button("저장", use_container_width=True):
            save_period(int(year), int(month), payload)
            st.success("저장 완료")
    with s2:
        if st.button("초기화(이번 달)", use_container_width=True):
            save_period(int(year), int(month), {})
            st.warning("초기화 완료")


# -----------------------------
# Current compute
# -----------------------------
p = PeriodInput(
    sales=float(payload["sales"]),
    beg_fg=float(payload["beg_fg"]),
    end_fg=float(payload["end_fg"]),
    cost_items=payload["cost_items"],
    sga_rows=[{"item": r["item"], "type": normalize_sga_type(r["type"]), "amount": float(r["amount"])} for r in payload["sga_rows"]],
    nonop_income_rows=[{"item": r["item"], "amount": float(r["amount"])} for r in payload["nonop_income_rows"]],
    nonop_expense_rows=[{"item": r["item"], "amount": float(r["amount"])} for r in payload["nonop_expense_rows"]],
    tax_mode=payload["tax_mode"],
    tax_manual=float(payload.get("tax_manual", 0.0)),
    include_local_tax=bool(payload.get("include_local_tax", True)),
)
out = compute_all(p)
df_cost = df_cost_statement(p.cost_items, out["cogm"])
df_pl = df_pl_statement(p, out)
df_bep_tbl = df_bep(out, out["sga_fixed"], p.sales)


# -----------------------------
# TAB2: RESULT
# -----------------------------
with tab2:
    st.subheader("2) 결과 (선택월)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("당기제조원가", to_money(out["cogm"]))
    c2.metric("매출원가", to_money(out["cogs"]))
    c3.metric("영업이익", to_money(out["op_profit"]))
    c4.metric("당기순이익", to_money(out["net_income"]))

    st.divider()

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("공헌이익률", f"{out['cm_ratio']*100:,.1f}%")
    c6.metric("공헌이익률 판정", out["grade"])
    c7.metric("BEP 매출액", to_money(out["bep_sales"]))
    c8.metric("BEP 대비 매출", to_money(out["gap_sales_vs_bep"]))

    st.caption("※ 직접재료비는 '당기사용원재료'만 원가 반영 (기초/입고/기말은 참고).")

    if out["tax_info"]:
        d = out["tax_info"]["detail"]
        st.caption(
            f"법인세(추정): {out['tax_info']['note']} | "
            f"연환산 과세표준 {to_money(out['tax_info']['annualized_tax_base'])}원 | "
            f"국세 {to_money(d['national_cit'])} + 지방(10%가정) {to_money(d['local_income_tax'])}"
        )
    else:
        st.caption("법인세: " + ("미반영(OFF)" if p.tax_mode == "OFF" else "수동 입력(MANUAL)"))

    left, right = st.columns(2)
    with left:
        st.markdown("### 손익계산서 요약")
        st.dataframe(df_pl, use_container_width=True, hide_index=True, height=520)

        st.markdown("### BEP / 공헌이익")
        st.dataframe(df_bep_tbl, use_container_width=True, hide_index=True, height=290)

    with right:
        st.markdown("### 제조원가명세서 상세")
        st.dataframe(df_cost, use_container_width=True, hide_index=True, height=820)


# -----------------------------
# TAB3: CUMULATIVE
# -----------------------------
with tab3:
    st.subheader("3) 누적 조회 (저장된 월만 합산)")

    view_type = st.selectbox("조회 기준", ["월별(선택월)", "분기누적(QTD)", "연도누적(YTD)", "연도전체(저장된 월 합산)"])
    selected_year = int(year)
    selected_month = int(month)

    months = months_for_period(selected_month, view_type)
    if view_type == "월별(선택월)":
        period_label = f"{selected_year}-{selected_month:02d}"
    elif view_type == "분기누적(QTD)":
        q = (selected_month - 1) // 3 + 1
        period_label = f"{selected_year} Q{q} (저장분)"
    elif view_type == "연도누적(YTD)":
        period_label = f"{selected_year} YTD (저장분)"
    else:
        period_label = f"{selected_year} 연도전체(저장분)"

    payloads = load_periods_range(selected_year, months)

    if not payloads:
        st.warning("선택한 기간에 저장된 데이터가 없습니다. (입력 탭에서 저장 후 다시 확인)")
    else:
        expected = set(months)
        included = set(int(pp.get("_month", 0)) for pp in payloads)
        missing = sorted(list(expected - included))
        if missing:
            st.info(f"저장되지 않은 월은 누적에서 제외됩니다: {', '.join(map(str, missing))}")

        agg = aggregate_periods(payloads)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("누적 매출액", to_money(agg["sales"]))
        a2.metric("누적 영업이익", to_money(agg["op_profit"]))
        a3.metric("누적 당기순이익", to_money(agg["net_income"]))
        a4.metric("누적 공헌이익률", f"{agg['cm_ratio']*100:,.1f}% ({agg['grade']})")

        st.caption(f"포함 월: {', '.join([str(m) for m in sorted([x for x in agg['months_included'] if x])])}")

        df_agg_pl = build_agg_pl_df(agg)
        st.markdown("### 누적 손익계산서 요약")
        st.dataframe(df_agg_pl, use_container_width=True, hide_index=True, height=420)

        st.markdown("### 누적 BEP / 공헌이익")
        df_agg_bep = pd.DataFrame([
            ["변동비 합계(매출원가+판관비 변동)", agg["variable_total"]],
            ["공헌이익", agg["cm"]],
            ["공헌이익률", agg["cm_ratio"]],
            ["고정비(판관비 고정)", agg["sga_fixed"]],
            ["BEP 매출액", agg["bep_sales"]],
            ["BEP 대비 매출", agg["gap_sales_vs_bep"]],
            ["공헌이익률 등급", agg["grade"]],
        ], columns=["항목", "값"])
        st.dataframe(df_agg_bep, use_container_width=True, hide_index=True, height=300)


# -----------------------------
# TAB4: DOWNLOAD
# -----------------------------
with tab4:
    st.subheader("4) 리포트 다운로드 (월별/분기누적/YTD/연도전체 원클릭)")

    rpt_type = st.selectbox("리포트 기준 선택", ["월별(선택월)", "분기누적(QTD)", "연도누적(YTD)", "연도전체(저장된 월 합산)"])
    selected_year = int(year)
    selected_month = int(month)

    months = months_for_period(selected_month, rpt_type)

    if rpt_type == "월별(선택월)":
        period_label = f"{selected_year}-{selected_month:02d}"
        title = f"{selected_year}년 {selected_month:02d}월 경영분석 리포트(월별)"
        mode = "MONTH"
    elif rpt_type == "분기누적(QTD)":
        q = (selected_month - 1) // 3 + 1
        period_label = f"{selected_year} Q{q} (저장분)"
        title = f"{selected_year}년 Q{q} 누적 경영분석 리포트(QTD)"
        mode = "QTD"
    elif rpt_type == "연도누적(YTD)":
        period_label = f"{selected_year} YTD (저장분)"
        title = f"{selected_year}년 누적 경영분석 리포트(YTD)"
        mode = "YTD"
    else:
        period_label = f"{selected_year} 연도전체(저장분)"
        title = f"{selected_year}년 연도전체 경영분석 리포트"
        mode = "YEAR"

    payloads = load_periods_range(selected_year, months)

    if not payloads:
        st.warning("선택한 기간에 저장된 데이터가 없습니다. (입력 탭에서 저장 후 다시 시도)")
    else:
        expected = set(months)
        included = set(int(pp.get("_month", 0)) for pp in payloads)
        missing = sorted(list(expected - included))
        if missing:
            st.info(f"저장되지 않은 월은 리포트 누적에서 제외됩니다: {', '.join(map(str, missing))}")

        if mode == "MONTH":
            sales_base = p.sales
            rpt_out = out
            rpt_df_pl = df_pl
            rpt_df_cost = df_cost
            rpt_tax_mode = p.tax_mode
            rpt_tax_info = out["tax_info"]
            rpt_bep = df_bep_tbl
            extra = {
                "판관비_상세": pd.DataFrame(p.sga_rows),
                "영업외수익": pd.DataFrame(p.nonop_income_rows),
                "영업외비용": pd.DataFrame(p.nonop_expense_rows),
            }
        else:
            agg = aggregate_periods(payloads)
            sales_base = agg["sales"]

            rpt_df_pl = build_agg_pl_df(agg)
            rpt_bep = pd.DataFrame([
                ["변동비 합계(매출원가+판관비 변동)", agg["variable_total"]],
                ["공헌이익", agg["cm"]],
                ["공헌이익률", agg["cm_ratio"]],
                ["고정비(판관비 고정)", agg["sga_fixed"]],
                ["BEP 매출액", agg["bep_sales"]],
                ["BEP 대비 매출", agg["gap_sales_vs_bep"]],
                ["공헌이익률 등급", agg["grade"]],
            ], columns=["항목", "값"])

            rpt_df_cost = None
            rpt_tax_mode = "AGG"
            rpt_tax_info = None
            rpt_out = {
                "op_profit": agg["op_profit"],
                "cm_ratio": agg["cm_ratio"],
                "grade": agg["grade"],
                "bep_sales": agg["bep_sales"],
                "op_margin": safe_div(agg["op_profit"], sales_base),
            }
            extra = {"포함월": pd.DataFrame({"월": [int(pp.get("_month", 0)) for pp in payloads]})}

        report_html = build_report_html(
            title=title,
            period_label=period_label,
            sales_base=sales_base,
            out=rpt_out,
            df_pl=rpt_df_pl,
            df_cost=rpt_df_cost,
            tax_mode=rpt_tax_mode if mode != "MONTH" else p.tax_mode,
            tax_info=rpt_tax_info
        )

        pdf_bytes = make_pdf_report(
            title=title,
            period_label=period_label,
            sales_base=sales_base,
            out=rpt_out,
            df_pl=rpt_df_pl,
            tax_mode=rpt_tax_mode if mode != "MONTH" else p.tax_mode,
            tax_info=rpt_tax_info
        )

        excel_bytes = make_excel_report(
            df_pl=rpt_df_pl,
            df_cost=rpt_df_cost,
            df_bep_tbl=rpt_bep,
            extra_sheets=extra
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button("리포트 PDF 다운로드", pdf_bytes,
                               file_name=f"report_{selected_year}_{selected_month:02d}_{mode}.pdf",
                               mime="application/pdf", use_container_width=True)
        with b2:
            st.download_button("리포트 HTML 다운로드", report_html,
                               file_name=f"report_{selected_year}_{selected_month:02d}_{mode}.html",
                               mime="text/html", use_container_width=True)
        with b3:
            st.download_button("리포트 Excel 다운로드", excel_bytes,
                               file_name=f"report_{selected_year}_{selected_month:02d}_{mode}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        st.divider()
        st.markdown("### 리포트 미리보기(HTML)")
        st.components.v1.html(report_html, height=650, scrolling=True)


# -----------------------------
# TAB5: COMPARE + TREND + DOWNLOAD(A/B)
# -----------------------------
with tab5:
    st.subheader("5) 비교분석 · 월별 누적 그래프")
    st.markdown("## A) 월별/분기/연도 비교분석")
    st.caption("두 기간을 선택하면 A와 B를 나란히 비교하고 차이를 보여줍니다. (저장된 월만 집계)")

    def build_period_aggregate(year_: int, anchor_month_: int, kind_: str):
        months_ = months_for_period(anchor_month_, kind_)
        payloads_ = load_periods_range(year_, months_)

        if kind_ == "월별(선택월)":
            label_ = f"{year_}-{anchor_month_:02d}"
        elif kind_ == "분기누적(QTD)":
            q_ = (anchor_month_ - 1) // 3 + 1
            label_ = f"{year_} Q{q_} (저장분)"
        elif kind_ == "연도누적(YTD)":
            label_ = f"{year_} YTD (저장분)"
        else:
            label_ = f"{year_} 연도전체(저장분)"

        if not payloads_:
            return {}, label_, []
        agg_ = aggregate_periods(payloads_)
        included_ = sorted([int(x) for x in agg_.get("months_included", []) if x])
        return agg_, label_, included_

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 비교기간 A")
        a_year = st.number_input("A 연도", min_value=2000, max_value=2100, value=int(year), step=1, key="a_year")
        a_month = st.number_input("A 기준월(앵커)", min_value=1, max_value=12, value=int(month), step=1, key="a_month")
        a_kind = st.selectbox("A 기준", ["월별(선택월)", "분기누적(QTD)", "연도누적(YTD)", "연도전체(저장된 월 합산)"], key="a_kind")
        agg_a, label_a, included_a = build_period_aggregate(int(a_year), int(a_month), a_kind)

    with colB:
        st.markdown("### 비교기간 B")
        b_year = st.number_input("B 연도", min_value=2000, max_value=2100, value=int(year), step=1, key="b_year")
        b_month = st.number_input("B 기준월(앵커)", min_value=1, max_value=12, value=max(1, int(month) - 1), step=1, key="b_month")
        b_kind = st.selectbox("B 기준", ["월별(선택월)", "분기누적(QTD)", "연도누적(YTD)", "연도전체(저장된 월 합산)"], key="b_kind")
        agg_b, label_b, included_b = build_period_aggregate(int(b_year), int(b_month), b_kind)

    if not agg_a or not agg_b:
        st.warning("A 또는 B 기간에 저장된 데이터가 없습니다. (입력 탭에서 월별 저장 후 비교해 주세요.)")
    else:
        st.success(f"A: {label_a} | 포함월: {included_a}  /  B: {label_b} | 포함월: {included_b}")

        comp_df = compare_two(agg_a, agg_b)
        st.markdown("### 비교표 (A vs B)")
        st.dataframe(comp_df, use_container_width=True, hide_index=True, height=520)

        st.divider()
        st.markdown("### A기간 vs B기간 리포트 다운로드 (PDF/HTML/Excel)")

        df_a_pl = build_agg_pl_df(agg_a)
        df_b_pl = build_agg_pl_df(agg_b)

        html_ab = build_compare_report_html(
            title="A기간 vs B기간 비교 리포트",
            label_a=label_a,
            label_b=label_b,
            agg_a=agg_a,
            agg_b=agg_b,
            comp_df=comp_df
        )

        excel_bytes_ab = make_excel_report(
            df_pl=df_a_pl,
            df_cost=None,
            df_bep_tbl=pd.DataFrame([
                ["(A기간) 공헌이익률", agg_a["cm_ratio"]],
                ["(A기간) BEP매출액", agg_a["bep_sales"]],
                ["(B기간) 공헌이익률", agg_b["cm_ratio"]],
                ["(B기간) BEP매출액", agg_b["bep_sales"]],
            ], columns=["항목", "값"]),
            extra_sheets={
                "B_손익요약": df_b_pl,
                "A_vs_B_비교표": comp_df,
            }
        )

        out_a_pdf = {
            "op_profit": agg_a["op_profit"],
            "cm_ratio": agg_a["cm_ratio"],
            "grade": grade_cm_ratio(agg_a["cm_ratio"]),
            "bep_sales": agg_a["bep_sales"],
            "op_margin": safe_div(agg_a["op_profit"], agg_a["sales"]),
        }
        out_b_pdf = {
            "op_profit": agg_b["op_profit"],
            "cm_ratio": agg_b["cm_ratio"],
            "grade": grade_cm_ratio(agg_b["cm_ratio"]),
            "bep_sales": agg_b["bep_sales"],
            "op_margin": safe_div(agg_b["op_profit"], agg_b["sales"]),
        }

        pdf_a = make_pdf_report(
            title=f"[A] {label_a} 경영분석 리포트",
            period_label=label_a,
            sales_base=agg_a["sales"],
            out=out_a_pdf,
            df_pl=df_a_pl,
            tax_mode="AGG",
            tax_info=None
        )
        pdf_b = make_pdf_report(
            title=f"[B] {label_b} 경영분석 리포트",
            period_label=label_b,
            sales_base=agg_b["sales"],
            out=out_b_pdf,
            df_pl=df_b_pl,
            tax_mode="AGG",
            tax_info=None
        )

        pdf_ab_merged = None
        if HAS_PYPDF2:
            try:
                pdf_ab_merged = merge_pdfs([pdf_a, pdf_b])
            except Exception:
                pdf_ab_merged = None

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("비교리포트 HTML 다운로드", html_ab, file_name="compare_A_vs_B.html",
                               mime="text/html", use_container_width=True)
        with d2:
            st.download_button("비교리포트 Excel 다운로드", excel_bytes_ab, file_name="compare_A_vs_B.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with d3:
            if pdf_ab_merged is not None:
                st.download_button("비교리포트 PDF 다운로드(1파일)", pdf_ab_merged, file_name="compare_A_vs_B.pdf",
                                   mime="application/pdf", use_container_width=True)
            else:
                st.download_button("A리포트 PDF 다운로드", pdf_a, file_name="report_A.pdf",
                                   mime="application/pdf", use_container_width=True)
                st.download_button("B리포트 PDF 다운로드", pdf_b, file_name="report_B.pdf",
                                   mime="application/pdf", use_container_width=True)

        st.markdown("#### 비교리포트 미리보기(HTML)")
        st.components.v1.html(html_ab, height=520, scrolling=True)

    st.divider()
    st.markdown("## B) 월별 누적 그래프(매출액/영업이익/당기순이익)")
    trend_year = st.number_input("그래프 연도 선택", min_value=2000, max_value=2100, value=int(year), step=1, key="trend_year")
    df_trend = build_year_month_table_for_trend(int(trend_year))

    if df_trend.empty:
        st.warning("선택한 연도에 저장된 데이터가 없습니다.")
    else:
        st.caption("저장된 월만 표시됩니다. (월별 입력 후 저장하면 자동으로 누적선이 연결됩니다.)")

        st.markdown("### 월별 값(단월)")
        df_monthly_chart = df_trend.set_index("월")[["매출액", "영업이익", "당기순이익"]]
        plot_lines_matplotlib(df_monthly_chart, title="월별: 매출액/영업이익/당기순이익", height_px=320)

        st.markdown("### 누적 값(연도 누적)")
        df_cum_chart = df_trend.set_index("월")[["누적매출", "누적영업이익", "누적당기순이익"]]
        plot_lines_matplotlib(df_cum_chart, title="누적: 매출/영업이익/순이익", height_px=320)

        st.markdown("### 월별 상세표")
        show_cols = ["월", "매출액", "영업이익", "당기순이익", "누적매출", "누적영업이익", "누적당기순이익", "공헌이익률", "BEP매출액"]
        df_show = df_trend[show_cols].copy()
        df_show["공헌이익률"] = df_show["공헌이익률"].apply(lambda x: f"{x*100:,.1f}%")
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=420)


# =============================
# Footer
# =============================
st.divider()
st.caption(f"버전: v{APP_VERSION} | 개발자: {DEVELOPER_NAME} | {COPYRIGHT_TEXT}")

