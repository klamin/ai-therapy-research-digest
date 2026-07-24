import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import requests
import yaml


CONFIG_PATH = "config.yaml"
SEEN_PATH = "seen_articles.json"
OPENALEX_URL = "https://api.openalex.org/works"


CATEGORY_LABELS = {
    "core_relevance": "Přímo relevantní pro BP",
    "field_orientation": "Širší orientace v poli",
    "quantitative_signal": "Kvantitativní / metodologický signál",
    "psychodynamic_overlap": "Psychoanalytický / psychodynamický přesah",
    "serendipity": "Řízený šum",
}


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seen() -> List[Dict[str, Any]]:
    if not os.path.exists(SEEN_PATH):
        return []
    with open(SEEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_seen(seen: List[Dict[str, Any]]) -> None:
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, ensure_ascii=False, indent=2)


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9á-ž]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def article_key(article: Dict[str, Any]) -> str:
    doi = article.get("doi")
    if doi:
        return doi.lower().replace("https://doi.org/", "").strip()
    return normalize_text(article.get("title"))


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    words_by_position = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words_by_position[pos] = word
    return " ".join(words_by_position[i] for i in sorted(words_by_position))


def match_terms(text: str, terms: List[str]) -> List[str]:
    normalized = normalize_text(text)
    matches = []
    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term and normalized_term in normalized:
            matches.append(term)
    return matches


def first_sentences(text: str, max_sentences: int = 2, max_chars: int = 550) -> str:
    if not text:
        return "OpenAlex u tohoto záznamu nemá abstrakt, takže posouzení stojí hlavně na názvu a metadatech."

    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(pieces[:max_sentences]).strip()

    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"

    return summary


def fetch_openalex(query: Dict[str, Any], config: Dict[str, Any], from_date: str) -> List[Dict[str, Any]]:
    filters = [
        f"from_publication_date:{from_date}",
        "is_retracted:false",
    ]

    params = {
        "search": query["text"],
        "filter": ",".join(filters),
        "sort": "relevance_score:desc",
        "per-page": int(config.get("max_results_per_query", 20)),
    }

    response = requests.get(OPENALEX_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    articles = []

    for item in data.get("results", []):
        title = item.get("title") or ""
        if not title:
            continue

        authors = []
        for author_item in item.get("authorships", [])[:8]:
            author = author_item.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)

        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        articles.append({
            "id": item.get("id"),
            "doi": item.get("doi"),
            "title": title,
            "authors": authors,
            "year": item.get("publication_year"),
            "publication_date": item.get("publication_date"),
            "type": item.get("type"),
            "source_name": source.get("display_name"),
            "url": item.get("doi") or primary_location.get("landing_page_url") or item.get("id"),
            "abstract": abstract,
            "source_lanes": [query["lane"]],
            "matched_queries": [query["text"]],
        })

    return articles


def deduplicate(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}

    for article in articles:
        key = article_key(article)
        if not key:
            continue

        if key not in by_key:
            by_key[key] = article
            continue

        existing = by_key[key]
        existing["source_lanes"] = sorted(
            set(existing.get("source_lanes", [])) | set(article.get("source_lanes", []))
        )
        existing["matched_queries"] = sorted(
            set(existing.get("matched_queries", [])) | set(article.get("matched_queries", []))
        )

        if not existing.get("abstract") and article.get("abstract"):
            existing["abstract"] = article["abstract"]
        if not existing.get("doi") and article.get("doi"):
            existing["doi"] = article["doi"]
        if not existing.get("url") and article.get("url"):
            existing["url"] = article["url"]

    return list(by_key.values())


def score_article(article: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join([
        article.get("title") or "",
        article.get("abstract") or "",
        article.get("source_name") or "",
        article.get("type") or "",
    ])

    hard_matches = match_terms(text, config.get("hard_exclude_terms", []))
    if hard_matches:
        return {
            "excluded": True,
            "score": -100,
            "primary_category": "excluded",
            "matched_terms": {"hard_exclude": hard_matches},
            "why": f"Vyřazeno kvůli tvrdému vyloučení: {', '.join(hard_matches)}.",
        }

    category_scores: Dict[str, int] = {}
    matched_terms: Dict[str, List[str]] = {}
    source_lanes: Set[str] = set(article.get("source_lanes", []))

    for category, settings in config.get("categories", {}).items():
        terms = settings.get("terms", [])
        matches = match_terms(text, terms)
        matched_terms[category] = matches

        score = 0
        if category in source_lanes:
            score += int(settings.get("lane_bonus", 0))
        score += len(matches) * int(settings.get("term_weight", 1))
        category_scores[category] = score

    soft_matches = match_terms(text, config.get("soft_downrank_terms", []))
    soft_penalty = len(soft_matches) * int(config.get("soft_downrank_penalty", 2))

    total_score = sum(category_scores.values()) - soft_penalty

    if category_scores:
        primary_category = max(category_scores, key=lambda c: category_scores[c])
    else:
        primary_category = "field_orientation"

    if category_scores.get(primary_category, 0) == 0:
        primary_category = "field_orientation"

    all_matches = []
    for matches in matched_terms.values():
        all_matches.extend(matches)

    if all_matches:
        why = "Zachyceno hlavně díky výrazům: " + ", ".join(sorted(set(all_matches))[:10]) + "."
    else:
        why = "Zařazeno hlavně kvůli širšímu vyhledávacímu dotazu; relevanci je potřeba zkontrolovat ručně."

    if soft_matches:
        why += " Mírně sníženo kvůli: " + ", ".join(soft_matches[:5]) + "."

    return {
        "excluded": False,
        "score": total_score,
        "primary_category": primary_category,
        "category_scores": category_scores,
        "matched_terms": matched_terms,
        "soft_downrank": soft_matches,
        "why": why,
    }


def select_digest_articles(scored: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    digest_size = int(config.get("digest_size", 7))
    slot_targets = config.get("slot_targets", {})

    candidates = [a for a in scored if not a["score"]["excluded"]]
    candidates.sort(key=lambda a: a["score"]["score"], reverse=True)

    selected = []
    selected_keys = set()

    for category, target in slot_targets.items():
        bucket = [a for a in candidates if a["score"]["primary_category"] == category]
        for article in bucket[: int(target)]:
            key = article_key(article)
            if key not in selected_keys:
                selected.append(article)
                selected_keys.add(key)

    for article in candidates:
        if len(selected) >= digest_size:
            break
        key = article_key(article)
        if key not in selected_keys:
            selected.append(article)
            selected_keys.add(key)

    return selected[:digest_size]


def build_issue_body(selected: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    now = datetime.now(ZoneInfo("Europe/Prague"))

    lines = [
        f"# Research digest — {now.strftime('%Y-%m-%d')}",
        "",
        f"**Téma:** {config.get('topic_name', 'AI a psychoterapie')}",
        "",
        "Poznámka: tahle verze nepoužívá žádný OpenAI API klíč. Výběr je založený na vyhledávání v OpenAlexu, deduplikaci a pravidlovém skórování podle názvu, abstraktu a metadat. Ber ho jako kurátorovaný filtr, ne jako definitivní odborné posouzení.",
        "",
    ]

    if not selected:
        lines.extend([
            "Nenašel jsem žádné nové položky po odstranění už poslaných článků a tvrdě vyloučených výsledků.",
            "",
        ])
        return "\n".join(lines)

    for category, label in CATEGORY_LABELS.items():
        items = [a for a in selected if a["score"]["primary_category"] == category]
        if not items:
            continue

        lines.append(f"## {label}")
        lines.append("")

        for article in items:
            score = article["score"]
            authors = ", ".join(article.get("authors", [])[:6]) or "neuvedeno"
            if len(article.get("authors", [])) > 6:
                authors += " et al."

            lines.extend([
                f"### {article.get('title')}",
                "",
                f"**Autoři:** {authors}",
                f"**Rok / datum:** {article.get('year') or 'neuvedeno'} / {article.get('publication_date') or 'neuvedeno'}",
                f"**Zdroj:** {article.get('source_name') or 'neuvedeno'}",
                f"**Typ v OpenAlex:** {article.get('type') or 'neuvedeno'}",
                f"**Odkaz / DOI:** {article.get('url') or 'neuvedeno'}",
                f"**Pravidlové skóre:** {score.get('score')}",
                "",
                f"**Stručně:** {first_sentences(article.get('abstract', ''))}",
                "",
                f"**Proč se objevuje v digestu:** {score.get('why')}",
                "",
            ])

    lines.extend([
        "## Přidané do seen_articles.json",
        "",
    ])

    for article in selected:
        lines.append(f"- {article.get('title')} ({article.get('year')}) — `{article_key(article)}`")

    return "\n".join(lines)


def create_github_issue(title: str, body: str) -> None:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]

    response = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "title": title,
            "body": body,
        },
        timeout=30,
    )

    response.raise_for_status()


def main() -> None:
    config = load_config()
    seen = load_seen()
    seen_keys = {item.get("key") for item in seen if item.get("key")}

    from_date = (
        datetime.now(ZoneInfo("Europe/Prague")).date()
        - timedelta(days=int(config.get("days_back", 180)))
    ).isoformat()

    fetched = []
    for query in config.get("queries", []):
        fetched.extend(fetch_openalex(query, config, from_date))

    unique_articles = deduplicate(fetched)
    new_articles = [a for a in unique_articles if article_key(a) not in seen_keys]

    scored = []
    for article in new_articles:
        article["score"] = score_article(article, config)
        scored.append(article)

    selected = select_digest_articles(scored, config)

    today = datetime.now(ZoneInfo("Europe/Prague")).strftime("%Y-%m-%d")
    issue_title = f"Research digest — {today}"
    issue_body = build_issue_body(selected, config)

    if os.environ.get("DRY_RUN") == "1":
        print(issue_title)
        print()
        print(issue_body)
        return

    create_github_issue(issue_title, issue_body)

    for article in selected:
        seen.append({
            "key": article_key(article),
            "title": article.get("title"),
            "year": article.get("year"),
            "doi": article.get("doi"),
            "url": article.get("url"),
            "sent_date": today,
            "category": article["score"].get("primary_category"),
            "score": article["score"].get("score"),
        })

    save_seen(seen)


if __name__ == "__main__":
    main()