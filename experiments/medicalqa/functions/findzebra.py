import requests


class FindZebraAPI:
    """A simple wrapper for the FindZebra API."""

    def __init__(self):
        self.api_key = "0a385c73-e5e0-4a2e-8054-7b6f913ce041"  # os.environ.get("FINDZEBRA_API_KEY")
        self.url = "https://www.findzebra.com/api/v1/query"

    def search(
        self,
        query: str,
        response_format: str = "json",
        rows: int = 1,
        start: int = 0,
        fields: list[str] = None,  # type: ignore
    ) -> str:
        """Search the FindZebra API."""
        # Prepare the query parameters
        if fields is None:
            fields = ["title", "display_content"]
        params = {"api_key": self.api_key, "q": query, "response_format": response_format, "rows": rows, "start": start}

        if fields is not None:
            params["fl"] = ",".join(fields)

        # Make the GET request
        response = requests.get(self.url, params=params, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:  # noqa: PLR2004
            response_json = response.json()
            if response_json["response"]["numFound"] == 0:
                return ""
            document = response_json["response"]["docs"][0]["display_content"]
            # document should not be too long
            return document[:4000]
        print("Error:", response.status_code)
        return ""

    @property
    def typescript(self) -> str:
        """Generate the typescript for the API."""
        lines = [
            "// Supported function definitions that must be used.",
            "namespace functions {",
            "",
            "",
            "// Search in a medical database given a query.",
            "type SearchMedicalDatabase = (_: {",
            "// Query to search for in the medical database.",
            "query: string,",
            "}) => { [key: string]: string };",
            "",
            "// namespace functions",
        ]
        return "\n".join(lines)
