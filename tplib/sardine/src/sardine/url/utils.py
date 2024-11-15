from typing import Union
from urllib.parse import ParseResult, parse_qsl

URLType = Union[str, ParseResult]

def query(input):
    """
    @brief Parses a query string or `ParseResult` input into a list of key-value pairs.

    @param input The input query string or `ParseResult` object.
    @return A list of key-value pairs parsed from the query string.
    """
    # If the input is a ParseResult, extract the query attribute
    if isinstance(input, ParseResult):
        input = input.query

    # Parse the query string into a list of (key, value) pairs
    return parse_qsl(input)
