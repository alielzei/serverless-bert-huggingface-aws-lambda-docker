from .ssl_ import create_urllib3_context, resolve_cert_reqs, resolve_ssl_version

def connection_requires_http_tunnel(proxy_url=None, proxy_config=None, destination_scheme=None):
    """
    Returns True if the connection requires an HTTP CONNECT through the proxy.

    :param URL proxy_url:
        URL of the proxy.
    :param ProxyConfig proxy_config:
        Proxy configuration from poolmanager.py
    :param str destination_scheme:
        The scheme of the destination. (i.e https, http, etc)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.proxy.connection_requires_http_tunnel', 'connection_requires_http_tunnel(proxy_url=None, proxy_config=None, destination_scheme=None)', {'proxy_url': proxy_url, 'proxy_config': proxy_config, 'destination_scheme': destination_scheme}, 1)

def create_proxy_ssl_context(ssl_version, cert_reqs, ca_certs=None, ca_cert_dir=None, ca_cert_data=None):
    """
    Generates a default proxy ssl context if one hasn't been provided by the
    user.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.proxy.create_proxy_ssl_context', 'create_proxy_ssl_context(ssl_version, cert_reqs, ca_certs=None, ca_cert_dir=None, ca_cert_data=None)', {'create_urllib3_context': create_urllib3_context, 'resolve_ssl_version': resolve_ssl_version, 'resolve_cert_reqs': resolve_cert_reqs, 'ssl_version': ssl_version, 'cert_reqs': cert_reqs, 'ca_certs': ca_certs, 'ca_cert_dir': ca_cert_dir, 'ca_cert_data': ca_cert_data}, 1)

