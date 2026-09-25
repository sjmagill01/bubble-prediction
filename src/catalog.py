"""
catalog.py — Unified catalog of 27 bubbles + 42 near-bubbles + 3 targets.

All US. All post-1993 (EDGAR era) unless noted. Each entry carries its own
ticker list so the project is self-contained.
"""


def _bubble(id, name, period, peak_date, category, severity, tickers):
    return {
        "id": id, "name": name, "period": period, "peak_date": peak_date,
        "category": category, "severity": severity, "tickers": tickers,
        "market": "US", "is_bubble": True, "is_target": peak_date is None,
    }


def _near(id, name, period, peak_date, category, max_drawdown, outcome, tickers):
    return {
        "id": id, "name": name, "period": period, "peak_date": peak_date,
        "category": category, "max_drawdown": max_drawdown, "outcome": outcome,
        "tickers": tickers, "market": "US", "is_bubble": False, "is_target": False,
    }


# ── Training Bubbles (27 with known peaks) ───────────────────────────

BUBBLES = [
    # Pre-2000
    _bubble("internet1", "Internet Stocks I", (1994, 1996, 1998), "1996-06",
            "mania", 35, ["NSCP", "YHOO", "SPYG", "NETC", "PSIX"]),
    _bubble("dotcom", "Dot-Com Bubble", (1997, 2000, 2002), "2000-03",
            "capex", 78, ["AMZN", "YHOO", "EBAY", "CMGI", "TGLO", "ETYS", "ICGE", "DCLK", "INKT"]),
    _bubble("telecom", "Telecom / Fiber", (1997, 2000, 2002), "2000-03",
            "capex", 85, ["WCOM", "GBLX", "Q", "LVLT", "JDSU", "NT", "LU", "GLW"]),
    _bubble("daytrading", "Day-Trading Brokerages", (1998, 2000, 2001), "2000-03",
            "mania", 80, ["QCOM", "JDSU", "CMGI"]),

    # 2000s
    _bubble("homebuilders", "Homebuilders", (2002, 2005, 2009), "2005-07",
            "capex", 80, ["TOL", "DHI", "LEN", "KBH", "BZH", "HOV"]),
    _bubble("subprime", "Subprime Lenders", (2003, 2007, 2009), "2007-02",
            "financial", 95, ["CFC", "NEW", "IMB", "WM"]),
    _bubble("banks_gfc", "Banks / GFC", (2003, 2007, 2009), "2007-06",
            "financial", 85, ["BSC", "LEH", "MER", "C", "AIG", "GS", "MS"]),
    _bubble("monolines", "Monoline Insurers", (2003, 2007, 2009), "2007-06",
            "financial", 98, ["MBI", "ABK", "AGO", "RDN", "PMI"]),
    _bubble("uranium", "Uranium Bubble", (2005, 2007, 2009), "2007-06",
            "commodity", 80, ["CCJ", "USU", "UEC", "URZ", "URRE"]),
    _bubble("solar1", "Solar I", (2006, 2008, 2009), "2008-01",
            "capex", 90, ["FSLR", "SPWR", "CSIQ"]),
    _bubble("shipping", "Dry Bulk Shipping", (2006, 2008, 2009), "2008-05",
            "commodity", 95, ["DRYS", "DSX", "GNK", "EGLE"]),
    _bubble("commodity_super", "Commodity Supercycle", (2005, 2008, 2009), "2008-06",
            "commodity", 65, ["FCX", "AA", "X", "CLF", "NUE"]),
    # 2010s
    _bubble("printing3d", "3D Printing", (2012, 2014, 2015), "2014-01",
            "mania", 90, ["DDD", "SSYS", "XONE", "VJET"]),
    _bubble("socialmedia", "Social Media IPOs", (2012, 2014, 2016), "2014-03",
            "mania", 50, ["FB", "TWTR", "YELP", "GRPN", "ZNGA"]),
    _bubble("shale", "Shale Oil & Gas", (2010, 2014, 2016), "2014-06",
            "capex", 75, ["CHK", "SD", "LINE", "CLR", "PXD", "EOG", "HK", "PVA"]),
    _bubble("biotech3", "Biotech III", (2012, 2015, 2016), "2015-07",
            "mania", 40, ["GILD", "VRX", "CELG", "REGN", "ALXN"]),
    _bubble("fang", "FANG Momentum", (2016, 2018, 2019), "2018-09",
            "mania", 25, ["FB", "AMZN", "NFLX", "GOOGL"]),
    _bubble("cannabis", "Cannabis", (2017, 2018, 2020), "2018-10",
            "mania", 85, ["TLRY", "CGC", "ACB", "CRON"]),
    _bubble("sharedeconomy", "Shared Economy IPOs", (2019, 2019, 2020), "2019-09",
            "mania", 60, ["UBER", "LYFT", "PTON"]),

    # 2020s
    _bubble("beyondmeat", "Beyond Meat / Alt Protein", (2019, 2021, 2022), "2021-01",
            "mania", 90, ["BYND", "OTLY", "TTCF"]),
    _bubble("memestocks", "Meme Stocks", (2020, 2021, 2022), "2021-01",
            "mania", 80, ["GME", "AMC", "BBBY", "BB", "NOK"]),
    _bubble("stayhome", "Stay-at-Home Tech", (2020, 2021, 2022), "2021-02",
            "mania", 75, ["ZM", "PTON", "TDOC", "DOCU", "CHWY"]),
    _bubble("spacs", "SPACs", (2020, 2021, 2022), "2021-02",
            "financial", 70, ["DKNG", "SPCE", "OPEN", "CLOV", "WISH", "LAZR", "JOBY", "MAPS"]),
    _bubble("ev", "Electric Vehicles", (2020, 2021, 2022), "2021-02",
            "capex", 85, ["TSLA", "RIVN", "LCID", "RIDE", "NKLA", "QS", "FSKR", "PLUG", "FCEL"]),
    _bubble("genomics2", "Genomics / ARK Bio", (2020, 2021, 2022), "2021-02",
            "mania", 75, ["CRSP", "BEAM", "NTLA", "EXAS", "PACB"]),
    _bubble("crypto", "Crypto-Adjacent Equities", (2020, 2021, 2023), "2021-11",
            "mania", 80, ["COIN", "MARA", "MSTR", "RIOT"]),
    _bubble("metaverse", "Metaverse / VR", (2020, 2021, 2023), "2021-11",
            "mania", 70, ["META", "RBLX", "U", "MTTR"]),
]


# ── Near-Bubbles (42 negative controls) ──────────────────────────────

NEAR_BUBBLES = [
    # Pre-2000
    _near("nb_hmo_1990s", "HMO / Managed Care", (1994, 1997, 2000), "1997-12",
          "mania", 30, "mild_correction",
          ["UNH", "AET", "CI", "HUM", "WLP", "HCA", "THC", "HNT"]),
    _near("nb_tobacco", "Tobacco Settlement Rally", (1996, 1998, 2000), "1998-04",
          "mania", 35, "mild_correction",
          ["MO", "PM", "RAI", "LO", "UST", "RJR"]),

    # 2000s
    _near("nb_defense_gwot", "Defense Post-9/11", (2002, 2007, 2009), "2007-10",
          "capex", 30, "mild_correction",
          ["LMT", "RTN", "GD", "NOC", "BA", "LLL", "L3", "HRS", "ITT", "COL"]),

    # 2010s
    _near("nb_gold_2009", "Gold Miners 2009-2011", (2009, 2011, 2015), "2011-09",
          "commodity", 38, "gradual_deflation",
          ["NEM", "ABX", "GDX", "AEM", "KGC", "GG", "AU", "GOLD", "FNV", "RGLD"]),
    _near("nb_railroads", "Railroad Renaissance", (2012, 2014, 2016), "2014-11",
          "capex", 25, "mild_correction",
          ["UNP", "CSX", "NSC", "CP", "KSU", "GWR"]),
    _near("nb_banks_post_gfc", "Bank Recovery 2009-2014", (2009, 2014, 2016), "2014-12",
          "financial", 20, "grew_into_valuations",
          ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "BK", "STT", "SCHW", "CME"]),
    _near("nb_biotech_2012", "Biotech Rally 2012-2015", (2012, 2015, 2016), "2015-07",
          "mania", 30, "mild_correction",
          ["GILD", "BIIB", "REGN", "CELG", "AMGN", "VRTX", "ALXN", "INCY", "BMRN", "SGEN"]),
    _near("nb_cybersec", "Cybersecurity 2014-2015", (2014, 2015, 2016), "2015-07",
          "capex", 30, "mild_correction",
          ["PANW", "FTNT", "CYBR", "FEYE", "SPLK", "RPD", "IMPV", "FIRE"]),
    _near("nb_housing_recovery", "Homebuilder Recovery", (2012, 2018, 2019), "2018-01",
          "capex", 30, "mild_correction",
          ["DHI", "LEN", "PHM", "TOL", "NVR", "MDC", "KBH", "MHO", "TMHC", "MTH"]),
    _near("nb_banks_trump", "Bank Rally Under Trump", (2016, 2018, 2019), "2018-01",
          "financial", 20, "mild_correction",
          ["JPM", "BAC", "GS", "MS", "C", "WFC", "USB", "PNC", "SCHW", "BK"]),
    _near("nb_semis_2016", "Semiconductor Rally 2016-2018", (2016, 2018, 2019), "2018-03",
          "capex", 25, "mild_correction",
          ["NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "TXN", "AVGO", "QCOM"]),
    _near("nb_fang_2016", "FANG 2016-2018", (2016, 2018, 2019), "2018-09",
          "mania", 25, "grew_into_valuations",
          ["FB", "AMZN", "NFLX", "GOOGL", "AAPL", "MSFT"]),
    _near("nb_energy_recovery", "Energy Recovery 2016-2018", (2016, 2018, 2019), "2018-10",
          "commodity", 25, "mild_correction",
          ["PXD", "EOG", "CLR", "FANG", "DVN", "COP", "OXY", "MRO", "APA", "HES"]),
    _near("nb_cloud_saas", "Cloud / SaaS 2017-2019", (2017, 2019, 2020), "2019-07",
          "capex", 30, "grew_into_valuations",
          ["CRM", "NOW", "WDAY", "TEAM", "ZS", "DDOG", "CRWD", "OKTA", "TWLO", "MDB", "NET", "SNOW", "BILL"]),
    _near("nb_streaming", "Streaming Wars", (2019, 2021, 2022), "2021-03",
          "capex", 35, "mild_correction",
          ["DIS", "NFLX", "CMCSA", "PARA", "WBD", "ROKU", "FUBO", "LUMN"]),

    # NEW 2020s near-bubbles
    _near("nb_semis_2020", "Semiconductor Chip Shortage Rally", (2020, 2021, 2022), "2021-11",
          "capex", 35, "grew_into_valuations",
          ["NVDA", "AMD", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "MCHP"]),
    _near("nb_cybersec_2021", "Cybersecurity Post-SolarWinds", (2020, 2021, 2022), "2021-11",
          "capex", 35, "grew_into_valuations",
          ["CRWD", "ZS", "PANW", "FTNT", "S", "NET", "OKTA", "QLYS"]),
    _near("nb_enterprise_saas", "Enterprise SaaS 2020-2021", (2020, 2021, 2022), "2021-11",
          "capex", 35, "mild_correction",
          ["NOW", "WDAY", "SNOW", "DDOG", "MDB", "TEAM", "HUBS", "ZM"]),
    _near("nb_payments", "Payments Rally", (2019, 2021, 2022), "2021-07",
          "financial", 30, "grew_into_valuations",
          ["V", "MA", "PYPL", "SQ", "FIS", "FISV", "GPN"]),
    _near("nb_infrastructure", "Infrastructure / Reshoring", (2020, 2022, 2023), "2022-01",
          "capex", 25, "grew_into_valuations",
          ["CAT", "DE", "VMC", "MLM", "NUE", "STLD", "URI", "PWR", "EME", "ACM"]),
    _near("nb_industrial_reits", "Industrial / Data Center REITs", (2020, 2022, 2023), "2022-01",
          "financial", 30, "mild_correction",
          ["PLD", "AMT", "EQIX", "DLR", "PSA", "SPG", "O", "WELL"]),
    _near("nb_shipping_2021", "Container Shipping Supercycle", (2020, 2022, 2023), "2022-03",
          "commodity", 35, "mild_correction",
          ["ZIM", "MATX", "DAC", "GSL", "CMRE", "GOGL", "SBLK", "GNK"]),
    _near("nb_ag_commodities", "Ag / Fertilizer Rally", (2021, 2022, 2023), "2022-04",
          "commodity", 30, "mild_correction",
          ["ADM", "BG", "CTVA", "MOS", "NTR", "CF", "DE", "AGCO"]),
    _near("nb_energy_2021", "Energy Rally 2021-2022", (2020, 2022, 2023), "2022-06",
          "commodity", 25, "mild_correction",
          ["XOM", "CVX", "OXY", "MPC", "VLO", "PSX", "DVN", "EOG", "PXD", "FANG"]),
    _near("nb_defense_ukraine", "Defense Post-Ukraine", (2022, 2022, 2023), "2022-10",
          "capex", 20, "grew_into_valuations",
          ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TXT"]),
    _near("nb_lithium", "Lithium / Battery Metals", (2021, 2022, 2023), "2022-11",
          "commodity", 38, "gradual_deflation",
          ["ALB", "SQM", "LTHM", "LAC", "PLL", "MP"]),
    _near("nb_managed_care", "Managed Care Rally", (2020, 2022, 2023), "2022-12",
          "mania", 20, "grew_into_valuations",
          ["UNH", "ELV", "CI", "HUM", "CNC", "MOH", "CVS", "HCA"]),
    _near("nb_insurance_hard", "Insurance Hard Market", (2020, 2023, 2024), "2023-06",
          "financial", 20, "grew_into_valuations",
          ["ALL", "TRV", "PGR", "CB", "AIG", "MET", "PRU", "AFL"]),

    # NEW: Hype/mania near-bubbles across decades (low leverage, high vol, survived)

    # 1990s
    _near("nb_restaurants_1990s", "Restaurant / Coffee Boom", (1995, 1999, 2001), "1999-12",
          "mania", 30, "grew_into_valuations",
          ["MCD", "SBUX", "YUM", "DRI", "CAKE"]),
    _near("nb_pharma_generics", "Generic Drug Boom", (1997, 2001, 2003), "2001-06",
          "mania", 30, "mild_correction",
          ["TEVA", "MYL", "BARR", "PRGO", "WAT"]),
    _near("nb_consumer_staples_1990s", "Consumer Staples Rally", (1996, 2000, 2002), "2000-01",
          "mania", 25, "mild_correction",
          ["PG", "KO", "CL", "CLX", "KMB", "SJM"]),

    # 2000s
    _near("nb_medical_devices", "Medical Devices / Robotics", (2004, 2008, 2009), "2008-01",
          "mania", 38, "mild_correction",  # GFC-driven DD, not sector-specific
          ["ISRG", "SYK", "MDT", "ZBH", "EW"]),
    # nb_online_retail_2005 dropped — only 3 tickers, AMZN -60% in GFC
    _near("nb_discount_retail_2007", "Off-Price Retail Boom", (2004, 2007, 2009), "2007-10",
          "mania", 20, "grew_into_valuations",
          ["TJX", "ROST", "DG", "DLTR"]),

    # 2010-2015
    _near("nb_cloud_early", "Enterprise Cloud Wave 1", (2012, 2016, 2017), "2016-03",
          "mania", 25, "grew_into_valuations",
          ["CRM", "WDAY", "NOW", "ADBE", "VMW"]),
    _near("nb_mobile_payments_2015", "Cashless Society Narrative", (2013, 2016, 2017), "2016-09",
          "mania", 25, "grew_into_valuations",
          ["V", "MA", "FIS", "FISV", "PAYX"]),
    _near("nb_fast_casual", "Fast Casual Revolution", (2012, 2015, 2017), "2015-10",
          "mania", 35, "mild_correction",
          ["CMG", "SHAK", "PNRA", "WING"]),
    _near("nb_athleisure", "Athleisure Trend", (2013, 2016, 2017), "2016-06",
          "mania", 30, "mild_correction",
          ["NKE", "DECK", "COLM", "SKX", "FL"]),

    # 2016-2019
    _near("nb_saas_ipo_2017", "SaaS IPO Wave", (2016, 2018, 2019), "2018-09",
          "mania", 30, "mild_correction",
          ["TWLO", "OKTA", "AYX", "PLAN", "COUP"]),
    # nb_fintech_2019 dropped — COVID crash -60%, overlaps with nb_payments
    _near("nb_ecommerce_2010", "E-Commerce Maturation", (2010, 2013, 2014), "2013-12",
          "mania", 25, "grew_into_valuations",
          ["AMZN", "W", "OSTK", "ETSY"]),

    # 2020s
    _near("nb_reopening_travel", "Reopening Travel Rally", (2020, 2022, 2023), "2022-03",
          "mania", 25, "grew_into_valuations",
          ["BKNG", "ABNB", "EXPE", "MAR", "HLT", "RCL"]),
    _near("nb_home_improvement", "COVID Home Renovation Boom", (2020, 2022, 2023), "2022-01",
          "mania", 30, "grew_into_valuations",
          ["HD", "LOW", "TSCO", "WSM", "POOL"]),
    _near("nb_gaming_2020", "COVID Gaming Surge", (2019, 2021, 2022), "2021-11",
          "mania", 30, "mild_correction",
          ["ATVI", "EA", "TTWO", "RBLX"]),
]


# ── Targets (ongoing, never trained) ─────────────────────────────────

TARGETS = [
    _bubble("AI", "AI Semiconductors", (2023, None, None), None,
            "mania", None, ["NVDA", "AMD", "AVGO", "SMCI", "MSFT", "AMZN", "GOOGL", "META", "MRVL", "ARM"]),
    _bubble("quantum", "Quantum Computing", (2024, None, None), None,
            "mania", None, ["IONQ", "RGTI", "QBTS"]),
    _bubble("nuclear2", "Nuclear Renaissance II", (2023, None, None), None,
            "capex", None, ["CEG", "VST", "SMR", "OKLO"]),
]


# ── Accessor functions ───────────────────────────────────────────────

def get_all():
    """All 72 episodes (27 bubbles + 42 near-bubbles + 3 targets)."""
    return BUBBLES + NEAR_BUBBLES + TARGETS


def get_training_episodes():
    """All training episodes with known peaks (bubbles + near-bubbles, excluding targets)."""
    return [e for e in get_all() if not e["is_target"]]


def get_bubbles():
    """27 training bubbles (known peaks)."""
    return [e for e in BUBBLES if e["peak_date"] is not None]


def get_near_bubbles():
    """42 near-bubble controls."""
    return list(NEAR_BUBBLES)


def get_targets():
    """3 ongoing targets."""
    return list(TARGETS)


def get_walk_forward_train(cutoff="2015-01"):
    """Episodes with peak before cutoff."""
    return [e for e in get_training_episodes() if e["peak_date"] and e["peak_date"] < cutoff]


def get_walk_forward_test(cutoff="2015-01"):
    """Episodes with peak on or after cutoff."""
    return [e for e in get_training_episodes() if e["peak_date"] and e["peak_date"] >= cutoff]


if __name__ == "__main__":
    all_ep = get_all()
    bubbles = get_bubbles()
    nears = get_near_bubbles()
    targets = get_targets()
    wf_train = get_walk_forward_train()
    wf_test = get_walk_forward_test()

    print(f"Total episodes: {len(all_ep)}")
    print(f"  Training bubbles: {len(bubbles)}")
    print(f"  Near-bubbles: {len(nears)}")
    print(f"  Targets: {len(targets)}")
    print(f"  Walk-forward train (peak < 2015): {len(wf_train)}")
    print(f"  Walk-forward test (peak >= 2015): {len(wf_test)}")

    total_tickers = sum(len(e["tickers"]) for e in all_ep)
    print(f"\nTotal tickers: {total_tickers}")
