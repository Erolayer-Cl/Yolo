"""Servidor HTTP del mapa de rescate (página en web/mapa.html).

    http://<host>:8081/              mapa + lista de personas con sus coordenadas
    http://<host>:8081/api/state     JSON: home, recorrido y personas (Trail.web_state)
    http://<host>:8081/api/live      JSON: posición del dron ahora
    http://<host>:8081/personas.csv  descarga, mismo formato que el del vuelo

Solo stdlib: no hace falta instalar nada, y la página no carga nada de internet
(el mapa de calles es opcional). Los datos los arma trail_node en su hilo y los
deja acá ya serializados; el servidor solo los entrega, así que no comparte
estructuras que se estén modificando.

Las respuestas llevan ETag: si nada cambió, el navegador recibe un 304 vacío y
no se manda de nuevo el recorrido entero cada segundo.
"""
import os
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'mapa.html')

CONTENT_TYPES = {
    '/api/state': 'application/json',
    '/api/live': 'application/json',
    '/personas.csv': 'text/csv; charset=utf-8',
}


def _etag(body: bytes) -> str:
    return f'"{zlib.crc32(body):08x}-{len(body)}"'


class MapServer:

    def __init__(self, port: int, bind: str = '0.0.0.0'):
        with open(PAGE_PATH, 'rb') as f:
            page = f.read()
        self._docs = {'/': (page, _etag(page))}
        for path in CONTENT_TYPES:
            self.set(path, b'{}' if path.startswith('/api/') else b'')
        self.httpd = ThreadingHTTPServer((bind, port), self._handler())
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name='web_map',
                                        daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def set(self, path: str, body: bytes):
        """Reemplaza lo que se entrega en `path` (una asignación: atómica)."""
        self._docs[path] = (body, _etag(body))

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        docs = self._docs

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                path = urlparse(self.path).path
                if path == '/index.html':
                    path = '/'
                doc = docs.get(path)
                if doc is None:
                    self._reply(404, 'text/plain; charset=utf-8', b'no existe\n')
                    return
                body, etag = doc
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(304)
                    self.send_header('ETag', etag)
                    self.end_headers()
                    return
                ctype = CONTENT_TYPES.get(path, 'text/html; charset=utf-8')
                extra = ({'Content-Disposition': 'attachment; filename="personas.csv"'}
                         if path == '/personas.csv' else {})
                self._reply(200, ctype, body, etag, extra)

            def _reply(self, code, ctype, body, etag=None, extra=None):
                try:
                    self.send_response(code)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Content-Length', str(len(body)))
                    # no-cache = revalidar siempre con el ETag (no-store lo impediría).
                    self.send_header('Cache-Control', 'no-cache')
                    if etag:
                        self.send_header('ETag', etag)
                    for k, v in (extra or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass                    # el navegador cerró la pestaña

        return Handler
