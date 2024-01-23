from pathlib import Path

import requests
from joblib import Memory


class Wikipedia:
    """Used for ReAct Foundation approach"""

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.current_page_content = None

    def search(self, query):
        """Search Wikipedia for a query and store the first page's content or top-5 similar entities."""
        params = params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 10,  # Limit to top 5 results
        }
        response = requests.get(self.base_url, params=params)
        search_results = response.json().get("query", {}).get("search", [])

        if search_results and search_results[0]["title"].lower() == query.lower():
            page_title = search_results[0]["title"]
            self.current_page_content = self._get_page_content(page_title)
            first_five_sentences = self.current_page_content.split(".")[:5]
            return ".".join(first_five_sentences)  # Return first 5 sentences
        else:
            similar_titles = [result["title"] for result in search_results][:5]
            return f"Could not find [{query}]. Similar: {similar_titles}"

    def _get_page_content(self, title):
        """Helper method to fetch a page's content."""
        params = {"action": "query", "titles": title, "prop": "extracts", "explaintext": True, "format": "json"}
        response = requests.get(self.base_url, params=params)
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        return page.get("extract", "")

    def lookup(self, string):
        """Lookup sentences in the stored page content containing the string."""
        results = []
        if self.current_page_content:
            sentences = self.current_page_content.split(".")
            for sentence in sentences:
                if string in sentence:
                    results.append(sentence.strip())

        formatted_results = []
        for i, result in enumerate(results[:5]):
            formatted_results.append(f"(Result {i + 1} / {len(results)}) {result}")

        return ". ".join(formatted_results)


class WikipediaAPI:
    def __init__(self, cache_dir, cache_reset=False):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        # self.current_page_content = None
        self.query = None
        self.top_pages = None
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir, "memory").resolve()
        memory = Memory(cache_dir / "Wiki", verbose=0)
        if cache_reset:
            memory.clear(warn=False)
        self.request_wiki = memory.cache(request_wikipedia)

    @property
    def typescript(self):
        """Return the typescript for the search"""
        signature = "\n".join(
            [
                "namespace functions {",
                "",
                "// Search Wikipedia for a query and store the first page's content or top-5 similar entities.",
                "type Wikipedia = (_: {",
                "// Query to search for.",
                "query: string,",
                "}) => any;",
                "",
                "} // namespace functions",
            ]
        )
        return signature

    @property
    def typescript_select_article(self):
        """Return the typescript for the select top article"""
        signature = "\n".join(
            [
                "namespace functions {",
                "",
                "// Select the most relevant article from the given list by responding with the index of the article.",
                "type TopArticle = (_: {",
                "// Index of top article to select.",
                "index: number,",
                "}) => any;",
                "",
                "} // namespace functions",
            ]
        )
        return signature

    def search(self, query):
        self.query = query
        """Search Wikipedia for a query and store the first page's content or top-5 similar entities."""
        params = params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 10,  # Limit to top 5 results
        }
        response = self.request_wiki(self.base_url, params)  # requests.get(self.base_url, params=params)
        search_results = response.json().get("query", {}).get("search", [])  # type: ignore

        if search_results:  # and search_results[0]["title"].lower() == query.lower():
            page_title = search_results[0]["title"]
            page_content = self._get_page_content(page_title)
            return page_content
            # self.current_page_content = self._get_page_content(page_title)
            # first_five_sentences = self.current_page_content.split(".")[:5]
            # return ".".join(first_five_sentences)  # Return first 5 sentences
        return None

        # similar_titles = [result["title"] for result in search_results][:5]
        closest_page_title = search_results[0]["title"]
        page_content = self._get_page_content(closest_page_title)
        # print(f"Choosing {closest_page_title} instead of {query}")
        return page_content
        # return f"Could not find [{query}]. Similar: {similar_titles}"

    def search_top_pages(self, query):
        self.query = query
        """Search Wikipedia for a query and store the first page's content or top-5 similar entities."""
        params = params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 10,  # Limit to top 5 results
        }

        response = self.request_wiki(self.base_url, params)
        search_results = response.json().get("query", {}).get("search", [])  # type: ignore
        self.top_pages = [result["title"] for result in search_results][:10]
        return self.top_pages

    def get_page_content_by_index(self, index: str):
        """Get the page content by index."""
        try:
            index_as_int = int(index)
        except ValueError:
            index_as_int = 0

        return self._get_page_content(self.top_pages[index_as_int])  # type: ignore

    def _get_page_content(self, title):
        """Helper method to fetch a page's content."""
        params = {"action": "query", "titles": title, "prop": "extracts", "explaintext": True, "format": "json"}
        response = self.request_wiki(self.base_url, params)  # requests.get(self.base_url, params=params)
        pages = response.json().get("query", {}).get("pages", {})  # type: ignore
        page = next(iter(pages.values()))
        return page.get("extract", "")


def request_wikipedia(url, params):
    return requests.get(url, params=params)
