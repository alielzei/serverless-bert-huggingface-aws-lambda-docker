from __future__ import absolute_import
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect
from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib

def is_fp_closed(obj):
    """
    Checks whether a given file-like object is closed.

    :param obj:
        The file-like object to check.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.response.is_fp_closed', 'is_fp_closed(obj)', {'obj': obj}, 1)

def assert_header_parsing(headers):
    """
    Asserts whether all headers have been successfully parsed.
    Extracts encountered errors from the result of parsing headers.

    Only works on Python 3.

    :param http.client.HTTPMessage headers: Headers to verify.

    :raises urllib3.exceptions.HeaderParsingError:
        If parsing errors are found.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.util.response.assert_header_parsing', 'assert_header_parsing(headers)', {'httplib': httplib, 'StartBoundaryNotFoundDefect': StartBoundaryNotFoundDefect, 'MultipartInvariantViolationDefect': MultipartInvariantViolationDefect, 'HeaderParsingError': HeaderParsingError, 'headers': headers}, 0)

def is_response_to_head(response):
    """
    Checks whether the request of a response has been a HEAD-request.
    Handles the quirks of AppEngine.

    :param http.client.HTTPResponse response:
        Response to check if the originating request
        used 'HEAD' as a method.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.response.is_response_to_head', 'is_response_to_head(response)', {'response': response}, 1)

