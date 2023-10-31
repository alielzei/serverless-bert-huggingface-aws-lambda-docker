"""The match_hostname() function from Python 3.3.3, essential when using SSL."""

import re
import sys
try:
    import ipaddress
except ImportError:
    ipaddress = None
__version__ = '3.5.0.1'


class CertificateError(ValueError):
    pass


def _dnsname_match(dn, hostname, max_wildcards=1):
    """Matching according to RFC 6125, section 6.4.3

    http://tools.ietf.org/html/rfc6125#section-6.4.3
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.packages.ssl_match_hostname._implementation._dnsname_match', '_dnsname_match(dn, hostname, max_wildcards=1)', {'CertificateError': CertificateError, 're': re, 'dn': dn, 'hostname': hostname, 'max_wildcards': max_wildcards}, 1)

def _to_unicode(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.packages.ssl_match_hostname._implementation._to_unicode', '_to_unicode(obj)', {'sys': sys, 'unicode': unicode, 'obj': obj}, 1)

def _ipaddress_match(ipname, host_ip):
    """Exact matching of IP addresses.

    RFC 6125 explicitly doesn't define an algorithm for this
    (section 1.7.2 - "Out of Scope").
    """
    ip = ipaddress.ip_address(_to_unicode(ipname).rstrip())
    return ip == host_ip

def match_hostname(cert, hostname):
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed, but IP addresses are not accepted for *hostname*.

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.packages.ssl_match_hostname._implementation.match_hostname', 'match_hostname(cert, hostname)', {'ipaddress': ipaddress, '_to_unicode': _to_unicode, '_dnsname_match': _dnsname_match, '_ipaddress_match': _ipaddress_match, 'CertificateError': CertificateError, 'cert': cert, 'hostname': hostname}, 1)

