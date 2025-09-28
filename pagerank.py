import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages = list(corpus.keys())
    n = len(pages)
    distribution = {p: (1 - damping_factor) / n for p in pages}

    links = corpus.get(page, set())
    # If no outgoing links, treat it as linking to all pages
    if not links:
        for p in pages:
            distribution[p] += damping_factor / n
    else:
        k = len(links)
        for p in links:
            distribution[p] += damping_factor / k

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {p: 0 for p in corpus}
    pages = list(corpus.keys())

    # First sample: choose a page at random
    current = random.choice(pages)
    counts[current] += 1

    # Sample based on transition model
    for _ in range(1, n):
        dist = transition_model(corpus, current, damping_factor)
        choices = list(dist.keys())
        weights = [dist[p] for p in choices]
        current = random.choices(choices, weights=weights, k=1)[0]
        counts[current] += 1

    # Convert counts to probabilities
    pagerank = {p: counts[p] / n for p in corpus}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    n = len(pages)

    # Initialize all ranks equally
    ranks = {p: 1 / n for p in pages}

    # Identify dangling pages (no links)
    dangling = {p for p, links in corpus.items() if len(links) == 0}

    while True:
        new_ranks = {}
        for p in pages:
            # Base rank from random jump
            rank = (1 - damping_factor) / n

            # Contribution from dangling pages
            if dangling:
                rank += damping_factor * sum(ranks[d] / n for d in dangling)

            # Contribution from pages linking to p
            for q, links in corpus.items():
                if len(links) > 0 and p in links:
                    rank += damping_factor * (ranks[q] / len(links))

            new_ranks[p] = rank

        # Check if all changes are <= 0.001
        if all(abs(new_ranks[p] - ranks[p]) <= 0.001 for p in pages):
            ranks = new_ranks
            break

        ranks = new_ranks

    # Normalize (just in case floating point math drifts)
    total = sum(ranks.values())
    ranks = {p: r / total for p, r in ranks.items()}

    return ranks


if __name__ == "__main__":
    main()
