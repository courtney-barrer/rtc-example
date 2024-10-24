from urllib.parse import parse_qsl, ParseResult

from baldr import _sardine as sa

def from_url(type, url):
    if hasattr(type, '__from_url__'):
        return type.__from_url__(url)
    else:
        return sa.from_url(type, url)

def url_of(value):
    if hasattr(value, '__url_of__'):
        return value.__url_of__()
    else:
        return sa.url_of(value)

def query(input):
    if isinstance(input, ParseResult):
        input = input.query

    return parse_qsl(input)
