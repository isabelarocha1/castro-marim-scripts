import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- Ajuste global de fontes ---
plt.rcParams.update({
    'font.size': 16,          # tamanho geral do texto
    'axes.titlesize': 18,     # título dos gráficos
    'axes.labelsize': 16,     # labels dos eixos
    'xtick.labelsize': 16,    # rótulos do eixo X
    'ytick.labelsize': 16,    # rótulos do eixo Y
    'legend.fontsize': 16     # legenda
})
# --- Paleta de cores personalizada ---
PALETTE_CFU = {
    "MSA": "#FF0000",   # vermelho
    "MA": "#F4A460",   # marrom claro (sandy brown)
    "VRBL": "#4B0082"  # roxo escuro (indigo)
}
"""
CFU_counts.py — versão corrigida para Excel (.xlsx)

→ Lê o arquivo **cfu_table.xlsx** (na mesma pasta).
→ Espera as colunas: Date, Samples, Medium, Colonies, Dilution factor,
  CFU/mL (estimate), Status, LOQ low, LOQ high
→ Gera automaticamente:
   1) Barras por amostragem (MA+MSA+VRBL) e (MA+MSA apenas)
   2) Linhas A–D (x = A,B,C,D; 2 linhas = Saltwork 1 e 2) para **cada meio e cada amostragem**
→ Salva PNGs em ./figures_cfu/

Observações
- WRED entra nos gráficos de barras, mas é ignorado nos gráficos de linhas A–D.
- Converte formatos como "5.60 x 10³", "3,11x10^4", "1.00 10¹" e vírgula decimal.
- Datas como "24/mar" → "2025-03-24" (ajuste o ano em DEFAULT_YEAR se precisar).
"""

# =========================
# CONFIG
# =========================
XLSX_PATH = "cfu_table.xlsx"  # nome do seu arquivo Excel
OUTDIR = Path("figures_cfu")
DEFAULT_YEAR = 2025

OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
MONTH_MAP = {
    # pt-br/pt-pt
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out":10, "nov":11, "dez":12,
    # en (caso apareça)
    "jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6,
    "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12
}

SUPERSCRIPT_MAP = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9","⁻":"-"}


def parse_pt_date(s: str, year: int = DEFAULT_YEAR) -> str:
    """Converte '24/mar' ou '16/abr' em 'YYYY-MM-DD'. Se não bater, retorna string original."""
    if pd.isna(s):
        return ""
    t = str(s).strip().lower().replace(" ", "/")
    if "/" not in t:
        return str(s)
    try:
        d, m = t.split("/")[:2]
        d = int(re.sub(r"\D", "", d))
        mnum = MONTH_MAP.get(m[:3], None)
        if not mnum:
            return str(s)
        return f"{year:04d}-{mnum:02d}-{d:02d}"
    except Exception:
        return str(s)


def normalize_cfu(val) -> float:
    """Converte strings como '5.60 x 10³', '3,11x10^4', '1.00 10¹' em float.
    Retorna np.nan se não conversível."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    # troca vírgula decimal
    s = s.replace(",", ".")
    # mapeia expoentes em sobrescrito
    s = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
    # normaliza alguns padrões comuns
    s = s.replace("×", "x").replace(" 10^", " x 10^").replace(" 10 ", " x 10^")
    s = re.sub(r"\s+", " ", s)
    # 1) base x 10^exp
    m = re.search(r"([\d.]+)\s*x\s*10\s*\^\s*([\-\d]+)", s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    # 2) base x 10exp (sem ^)
    m2 = re.search(r"([\d.]+)\s*x\s*10\s*([\-\d]+)", s)
    if m2:
        base = float(m2.group(1))
        exp = int(m2.group(2))
        return base * (10 ** exp)
    # 3) número simples
    try:
        return float(s)
    except Exception:
        return np.nan

# =========================
# LOAD DATA
# =========================
if not Path(XLSX_PATH).exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {Path(XLSX_PATH).resolve()}")

df = pd.read_excel(XLSX_PATH)

# padroniza nomes das colunas
expected = [
    "Date","Samples","Medium","Colonies","Dilution factor",
    "CFU/mL (estimate)","Status","LOQ low","LOQ high"
]
colmap = {c: c.strip() for c in df.columns}
df.rename(columns=colmap, inplace=True)
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Colunas faltantes: {missing}. Esperado: {expected}")

# limpeza básica
# corrige erros de digitação
df["Medium"] = df["Medium"].astype(str).str.upper().str.replace(",", "", regex=False).str.strip()
df["Samples"] = df["Samples"].astype(str).str.strip()

# converte CFU
df["CFU_num"] = df["CFU/mL (estimate)"].apply(normalize_cfu)

# extrai salina e ponto (A,B,C,D,WRED)
df["saltwork"] = df["Samples"].str.extract(r"(\d)")
df["pond"] = df["Samples"].str.extract(r"([A-Z]+)")

# normaliza data (para nomes de arquivo)
df["sampling"] = df["Date"].apply(parse_pt_date)

# ordenações
pond_order = ["A","B","C","D"]
medium_order = ["MA","MSA","VRBL"]

# =========================
# 1) BARRAS: por amostragem
# =========================
for samp, sub in df.groupby("sampling"):
    sub = sub.copy()
    # rótulo Sample = saltwork+pond (ex.: '1A', '2C', 'WRED')
    sub["SampleLabel"] = sub.apply(
        lambda r: (str(r["saltwork"]) + str(r["pond"])) if pd.notna(r["saltwork"]) and pd.notna(r["pond"]) else str(r["Samples"]),
        axis=1,
    )

    # a) MA + MSA + VRBL
    piv = sub.pivot_table(index="SampleLabel", columns="Medium", values="CFU_num", aggfunc="mean")
    cols_present = [m for m in medium_order if m in piv.columns]
    piv = piv.reindex(columns=cols_present).sort_index()
    if not piv.empty:
        ax = piv.plot(kind="bar", figsize=(10,6), color=[PALETTE_CFU.get(c, "gray") for c in piv.columns])
        ax.set_title(f"CFU/mL by Sample and Medium — {samp}")
        ax.set_ylabel("CFU/mL (log10)")
        ax.set_xlabel("Sample (Saltwork+Pond)")
        ax.set_yscale("log")
        ax.legend(title="Medium")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"cfu_bar_all_{samp}.png", dpi=300)
        plt.close()

    # b) apenas MA + MSA
    sub_mm = sub[sub["Medium"].isin(["MA","MSA"])]
    piv2 = sub_mm.pivot_table(index="SampleLabel", columns="Medium", values="CFU_num", aggfunc="mean")
    if not piv2.empty:
        piv2 = piv2.reindex(columns=[m for m in ["MA","MSA"] if m in piv2.columns]).sort_index()
        ax = piv2.plot(kind="bar", figsize=(10,6))
        ax.set_title(f"CFU/mL (MA & MSA only) — {samp}")
        ax.set_ylabel("CFU/mL (log10)")
        ax.set_xlabel("Sample (Saltwork+Pond)")
        ax.set_yscale("log")
        ax.legend(title="Medium")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"cfu_bar_MA_MSA_{samp}.png", dpi=300)
        plt.close()

# =========================
# 2) LINHAS A–D: por amostragem e por meio (duas linhas = saltwork 1 e 2)
# =========================
for samp, grp in df.groupby("sampling"):
    for med in medium_order:
        sub = grp[(grp["Medium"].str.upper() == med) & (grp["pond"].isin(pond_order))]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="pond", columns="saltwork", values="CFU_num", aggfunc="mean")
        pivot = pivot.reindex(index=pond_order)

        plt.figure(figsize=(8,5))
        for sw in sorted([c for c in pivot.columns if pd.notna(c)]):
            plt.plot(pivot.index.astype(str), pivot[sw], marker="o", label=f"Saltwork {sw}")
        plt.yscale("log")
        plt.title(f"CFU/mL along ponds A–D — {med} — {samp}")
        plt.xlabel("Pond")
        plt.ylabel("CFU/mL (log10)")
        plt.legend()
        plt.tight_layout()
        fname = f"cfu_line_{med}_{samp}.png".replace(" ", "_")
        plt.savefig(OUTDIR / fname, dpi=300)
        plt.close()

print(f"Figuras salvas em: {OUTDIR.resolve()}")
