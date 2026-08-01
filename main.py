import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import requests
import yaml

import time


CONFIG_PATH = "config.yaml"
SEEN_PATH = "seen_articles.json"
KNOWN_PATH = "known_articles.json"
OPENALEX_URL = "https://api.openalex.org/works"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
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


def normalize_doi(value: Optional[str]) -> str:
    if not value:
        return ""
    return (
        value.lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("doi:", "")
        .strip()
    )


def article_key(article: Dict[str, Any]) -> str:
    doi = normalize_doi(article.get("doi") or article.get("key"))
    if doi:
        return doi
    return normalize_text(article.get("title"))


def known_key(item: Dict[str, Any]) -> str:
    doi = normalize_doi(item.get("doi") or item.get("key"))
    if doi:
        return doi
    return normalize_text(item.get("title"))


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


def format_abstract(abstract: str, max_chars: int = 2000) -> str:
    if not abstract:
        return "Abstract not available in OpenAlex."

    abstract = re.sub(r"\s+", " ", abstract).strip()

    # Remove simple XML/HTML-like tags that sometimes appear in preprint abstracts.
    abstract = re.sub(r"</?title>", "", abstract, flags=re.IGNORECASE)
    abstract = re.sub(r"<[^>]+>", "", abstract)

    # Make structured abstract labels readable, but do not over-format the whole text.
    abstract = re.sub(
        r"\b(Background|Objective|Objectives|Aim|Aims|Purpose|Methods|Method|Results|Findings|Conclusion|Conclusions|Discussion|Unlabelled):\s*",
        r"**\1:** ",
        abstract,
        flags=re.IGNORECASE,
    )

    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rsplit(" ", 1)[0] + "…"

    return abstract


def method_signal(article: Dict[str, Any]) -> str:
    text = " ".join([
        article.get("title") or "",
        article.get("abstract") or "",
        article.get("type") or "",
    ])
    normalized = normalize_text(text)
    signals = []

    if any(term in normalized for term in ["qualitative", "interview", "interviews", "thematic analysis"]):
        signals.append("qualitative")
    if any(term in normalized for term in ["survey", "questionnaire", "cross sectional", "prevalence", "predictors", "attitudes"]):
        signals.append("survey / quantitative")
    if any(term in normalized for term in ["randomized", "randomised", "experiment", "trial", "rct"]):
        signals.append("experimental / trial")
    if any(term in normalized for term in ["systematic review", "scoping review", "meta analysis", "review"]):
        signals.append("review")
    if article.get("type") == "preprint":
        signals.append("preprint")
    if article.get("type") == "conference-paper":
        signals.append("conference paper")
    if article.get("type") == "article" and not signals:
        signals.append("journal article")

    return ", ".join(dict.fromkeys(signals)) if signals else "not clear from metadata"


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

    for attempt in range(5):
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 10 * (attempt + 1)
            print(f"OpenAlex rate limit hit. Waiting {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        data = response.json()

        time.sleep(float(config.get("openalex_pause_seconds", 1.5)))

        return data.get("results", [])

    print(f"OpenAlex kept rate-limiting this query, skipping: {query}")
    return []

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
            "keywords": hard_matches,
        }

    topic_matches = match_terms(text, config.get("required_topic_terms", []))
    if not topic_matches:
        return {
            "excluded": True,
            "score": -90,
            "primary_category": "excluded",
            "matched_terms": {"missing_topic_signal": []},
            "keywords": [],
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
        primary_category = "broader_field"

    if category_scores.get(primary_category, 0) == 0:
        primary_category = "broader_field"

    keywords = []
    for matches in matched_terms.values():
        keywords.extend(matches)

    keywords.extend(topic_matches)

    return {
        "excluded": False,
        "score": total_score,
        "primary_category": primary_category,
        "category_scores": category_scores,
        "matched_terms": matched_terms,
        "soft_downrank": soft_matches,
        "keywords": sorted(set(keywords))[:14],
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


def infer_digest_themes(selected: List[Dict[str, Any]]) -> List[str]:
    all_keywords = set()
    for article in selected:
        all_keywords.update(k.lower() for k in article["score"].get("keywords", []))

    themes = []

    if any(k in all_keywords for k in ["psychotherapist", "psychotherapists", "therapist", "therapists", "mental health professionals"]):
        themes.append("therapists’ and clinicians’ encounters with generative AI")
    if any(k in all_keywords for k in ["therapeutic relationship", "therapeutic alliance", "trust", "distrust"]):
        themes.append("trust, distrust, and the therapeutic relationship")
    if any(k in all_keywords for k in ["survey", "questionnaire", "cross-sectional", "attitudes", "acceptance", "prevalence"]):
        themes.append("survey evidence on adoption, attitudes, and use")
    if any(k in all_keywords for k in ["ai chatbot", "chatbot", "mental health", "digital mental health"]):
        themes.append("AI chatbots and digital mental-health support")
    if any(k in all_keywords for k in ["psychoanalysis", "psychoanalytic", "psychodynamic", "transference", "countertransference"]):
        themes.append("psychoanalytic or psychodynamic conceptual links")
    if any(k in all_keywords for k in ["self-disclosure", "intimacy", "emotional support", "ai companion", "anthropomorphism"]):
        themes.append("self-disclosure, intimacy, and human–AI relations")

    return themes[:3]


def category_label(config: Dict[str, Any], category: str) -> str:
    return config.get("categories", {}).get(category, {}).get("label", category)


def build_issue_body(selected: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    now = datetime.now(ZoneInfo("Europe/Prague"))
    themes = infer_digest_themes(selected)

    lines = [
        f"# Research digest — {now.strftime('%Y-%m-%d')}",
        "",
    ]

    if selected:
        if themes:
            lines.append(
                "This digest brings together new or newly surfaced work on "
                + "; ".join(themes)
                + "."
            )
        else:
            lines.append(
                "This digest brings together new or newly surfaced work related to AI, psychotherapy, mental health, and adjacent human–AI questions."
            )
        lines.append("")
    else:
        lines.extend([
            "No new items were selected after removing already seen or known articles.",
            "",
        ])
        return "\n".join(lines)

    ordered_categories = [
        "direct_relevance",
        "broader_field",
        "psychoanalytic_perspective",
        "exploratory_adjacent",
    ]

    for category in ordered_categories:
        items = [a for a in selected if a["score"]["primary_category"] == category]
        if not items:
            continue

        lines.append(f"## {category_label(config, category)}")
        lines.append("")

        for article in items:
            authors = ", ".join(article.get("authors", [])[:6]) or "not listed"
            if len(article.get("authors", [])) > 6:
                authors += " et al."

            keywords = article["score"].get("keywords", [])
            keyword_text = ", ".join(keywords) if keywords else "none"

            lines.extend([
                f"### *{article.get('title')}*",
                "",
                f"**Authors:** {authors}",
                f"**Year / date:** {article.get('year') or 'not listed'} / {article.get('publication_date') or 'not listed'}",
                f"**Source:** {article.get('source_name') or 'not listed'}",
                f"**Type:** {article.get('type') or 'not listed'}",
                f"**Method / format signal:** {method_signal(article)}",
                f"**Link / DOI:** {article.get('url') or 'not listed'}",
                "",
                f"**Abstract excerpt:** {format_abstract(article.get('abstract', ''), int(config.get('max_abstract_chars', 2000)))}",
                "",
                f"**Keywords matched:** {keyword_text}",
                "",
            ])

    lines.extend([
        "## Added to seen_articles.json",
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
    seen = load_json_list(SEEN_PATH)
    known = load_json_list(KNOWN_PATH)

    blocked_keys = {
        known_key(item)
        for item in seen + known
        if known_key(item)
    }

    from_date = (
        datetime.now(ZoneInfo("Europe/Prague")).date()
        - timedelta(days=int(config.get("days_back", 180)))
    ).isoformat()

    fetched = []
    for query in config.get("queries", []):
        fetched.extend(fetch_openalex(query, config, from_date))

    unique_articles = deduplicate(fetched)
    new_articles = [a for a in unique_articles if article_key(a) not in blocked_keys]

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
        })

    save_seen(seen)


if __name__ == "__main__":
    main()