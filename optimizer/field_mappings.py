"""
Field Mappings — Defines which fields to keep from WealthSpectrum data
for each use case. This is the core of data optimization.

Each mapping has:
  - keep_fields: fields to include in the optimized output
  - rename: optional field renaming for shorter tokens
  - round_decimals: decimal places for numeric fields
"""

from typing import Dict, List, Any

# ===================================================================
# Holdings Field Mapping
# ===================================================================

HOLDINGS_FIELDS = {
    "keep": [
        "symbolName", "categoryName", "astclsName", "description",
        "totalCost", "mktVal", "gainPerc",
        "xirrYield", "assetsPerc", "units", "nav", "income", "level",
    ],
    "rename": {
        "symbolName": "name",
        "categoryName": "type",
        "astclsName": "asset",
        "description": "sector",
        "totalCost": "invested",
        "mktVal": "current",
        "gainPerc": "gain_pct",
        "xirrYield": "xirr",
        "assetsPerc": "weight",
    },
    "round_decimals": 2,
}

# Holdings with debt-specific fields
HOLDINGS_DEBT_FIELDS = {
    "keep": [
        "symbolName", "categoryName", "astclsName", "description",
        "totalCost", "mktVal", "gainPerc",
        "xirrYield", "assetsPerc", "units", "nav", "income", "level",
        "creditRating", "maturityDate", "couponRate", "ytm",
    ],
    "rename": {
        "symbolName": "name",
        "categoryName": "type",
        "totalCost": "invested",
        "mktVal": "current",
        "gainPerc": "gain_pct",
        "xirrYield": "xirr",
        "assetsPerc": "weight",
    },
    "round_decimals": 2,
}


# ===================================================================
# Performance Field Mapping
# ===================================================================

PERFORMANCE_FIELDS = {
    "keep": [
        "astClsName", "astCls",
        "totalCost", "endMktVal", "beginMktVal",
        "gain", "xirr", "bmXirr", "xirrAbs", "bmXirrAbs",
        "benchMarkName", "netFlow", "unrealGain", "realGain", "income",
        "fromDate", "toDate",
    ],
    "rename": {
        "astClsName": "name",
        "astCls": "level",
        "totalCost": "invested",
        "endMktVal": "current",
        "beginMktVal": "begin_val",
        "xirrAbs": "return_pct",
        "bmXirr": "bm_xirr",
        "bmXirrAbs": "bm_return_pct",
        "benchMarkName": "benchmark",
    },
    "round_decimals": 2,
}


# ===================================================================
# Allocation Field Mapping
# ===================================================================

ALLOCATION_FIELDS = {
    "keep": [
        "assetClassName", "currentAllocation", "targetAllocation",
        "marketValue", "deviation",
        # WealthSpectrum alternate field names
        "astclsName", "endMktVal", "endPerc", "beginPerc",
    ],
    "rename": {
        "assetClassName": "asset_class",
        "astclsName": "asset_class",
        "currentAllocation": "current_pct",
        "endPerc": "current_pct",
        "beginPerc": "begin_pct",
        "targetAllocation": "target_pct",
        "marketValue": "value",
        "endMktVal": "value",
    },
    "round_decimals": 2,
}


# ===================================================================
# Transaction Field Mapping
# ===================================================================

TRANSACTION_FIELDS = {
    "keep": [
        "securityName", "transactionType", "transactionDate",
        "quantity", "price", "amount", "assetClass",
    ],
    "rename": {
        "securityName": "name",
        "transactionType": "type",
        "transactionDate": "date",
    },
    "round_decimals": 2,
}


# ===================================================================
# Capital Gain Field Mapping
# ===================================================================

CAPITAL_GAIN_FIELDS = {
    "keep": [
        "securityName", "assetClass", "quantity", "investedValue",
        "marketValue", "stcg", "ltcg", "holdingPeriod",
    ],
    "rename": {
        "securityName": "name",
        "investedValue": "cost",
        "marketValue": "current",
    },
    "round_decimals": 2,
}


# ===================================================================
# IPS Review Field Mapping
# ===================================================================

IPS_FIELDS = {
    "keep": [
        "assetClassName", "minAllocation", "maxAllocation",
        "targetAllocation", "currentAllocation", "status",
    ],
    "rename": {
        "assetClassName": "asset_class",
        "minAllocation": "min_pct",
        "maxAllocation": "max_pct",
        "targetAllocation": "target_pct",
        "currentAllocation": "current_pct",
    },
    "round_decimals": 2,
}


# ===================================================================
# Question-Aware Tag Selection
# Maps keywords in user question to relevant data tags.
# ===================================================================

QUESTION_TAG_MAP = {
    # Performance questions
    "perform": ["Performance"],
    "return": ["Performance", "Return Summary"],
    "xirr": ["Performance"],
    "irr": ["Performance"],
    "best": ["Performance", "Holdings"],
    "worst": ["Performance", "Holdings"],
    "top": ["Performance", "Holdings"],
    "bottom": ["Performance", "Holdings"],
    "underperform": ["Performance", "Holdings"],
    "outperform": ["Performance", "Holdings"],

    # Holdings questions
    "holding": ["Holdings"],
    "stock": ["Holdings"],
    "fund": ["Holdings"],
    "mutual fund": ["Holdings"],
    "bond": ["Holdings"],
    "security": ["Holdings"],

    # Allocation questions
    "allocat": ["Holdings", "Diversification"],
    "rebalanc": ["Holdings", "Diversification", "IPS Review"],
    "diversif": ["Diversification"],
    "concentrat": ["Holdings", "Diversification"],
    "sector": ["Holdings", "Diversification"],
    "overweight": ["Holdings", "Diversification", "IPS Review"],
    "underweight": ["Holdings", "Diversification", "IPS Review"],

    # Tax questions
    "tax": ["Capital Gain Impact"],
    "capital gain": ["Capital Gain Impact"],
    "stcg": ["Capital Gain Impact"],
    "ltcg": ["Capital Gain Impact"],

    # Cash flow
    "cashflow": ["Cash Flow Projection"],
    "cash flow": ["Cash Flow Projection"],
    "maturity": ["Cash Flow Projection", "Holdings"],
    "coupon": ["Cash Flow Projection", "Holdings"],
    "sip": ["Cash Flow Projection"],
    "swp": ["Cash Flow Projection"],

    # IPS / Compliance
    "ips": ["IPS Review"],
    "compliance": ["IPS Review"],
    "policy": ["IPS Review"],
    "deviation": ["IPS Review", "Diversification"],

    # Look Through
    "look through": ["Look Through"],
    "underlying": ["Look Through"],
    "overlap": ["Look Through"],

    # Transactions
    "transaction": ["Transaction"],
    "bought": ["Transaction"],
    "sold": ["Transaction"],
    "buy": ["Transaction"],
    "sell": ["Transaction"],
    "dividend": ["Transaction"],

    # Risk
    "risk": ["Holdings", "Diversification", "IPS Review"],
    "volatil": ["Holdings", "Performance"],

    # Debt specific
    "debt": ["Holdings"],
    "fixed income": ["Holdings"],
    "yield": ["Holdings"],
    "ytm": ["Holdings"],
    "credit": ["Holdings"],
    "rating": ["Holdings"],
    "duration": ["Holdings"],
}


def get_relevant_tags(question: str) -> List[str]:
    """
    Analyze the user's question and return relevant data tags.
    If no specific keywords found, return common defaults.
    """
    question_lower = question.lower()
    matched_tags = set()

    for keyword, tags in QUESTION_TAG_MAP.items():
        if keyword in question_lower:
            matched_tags.update(tags)

    # Default: if nothing matched, include basic overview data
    if not matched_tags:
        matched_tags = {"Holdings", "Performance", "Diversification"}

    return list(matched_tags)
