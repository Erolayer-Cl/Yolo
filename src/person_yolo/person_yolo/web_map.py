"""Servidor HTTP del mapa de rescate (página en web/mapa.html).

    http://<host>:8081/              mapa + lista de personas con sus coordenadas
    http://<host>:8081/api/state     JSON: home, recorrido y personas (Trail.web_state)
    http://<host>:8081/api/live      JSON: posición del dron ahora
    http://<host>:8081/personas.csv  descarga, mismo formato que el del vuelo
    POST /api/mark                   marca a mano en la posición del dron ahora
    POST /api/mark/delete {"id": -1} borra una marca manual

Solo stdlib: no hace falta instalar nada, y la página no carga nada de internet
(el mapa de calles es opcional). Los datos los arma trail_node en su hilo y los
deja acá ya serializados; el servidor solo los entrega, así que no comparte
estructuras que se estén modificando.

Las respuestas llevan ETag: si nada cambió, el navegador recibe un 304 vacío y
no se manda de nuevo el recorrido entero cada segundo.

Los POST los resuelve `on_post(path, datos) -> (código, dict)`, que pasa
trail_node: corre en el hilo del servidor, así que solo debe leer datos ya
armados y encolar el cambio (lo aplica trail_node en su hilo).
"""
import json
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

    def __init__(self, port: int, bind: str = '0.0.0.0', on_post=None):
        with open(PAGE_PATH, 'rb') as f:
            page = f.read()
        self._docs = {'/': (page, _etag(page))}
        self._on_post = on_post
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
        on_post = self._on_post

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                path = urlparse(self.path).path
                if on_post is None or not path.startswith('/api/'):
                    self._json(404, {'error': 'no existe'})
                    return
                # CSRF: otra página abierta en el mismo navegador (de internet) no
                # puede marcar ni borrar. Un <form> no puede mandar application/json,
                # y un fetch con ese tipo desde otro origen exige un preflight CORS que
                # este servidor no responde. Si viene Origin, tiene que ser este host.
                ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip()
                if ctype.lower() != 'application/json':
                    self._json(415, {'error': 'se esperaba Content-Type: application/json'})
                    return
                origin = self.headers.get('Origin')
                if origin is not None and urlparse(origin).netloc != self.headers.get('Host'):
                    self._json(403, {'error': 'origen no permitido'})
                    return
                try:
                    n = int(self.headers.get('Content-Length') or 0)
                    if n < 0:                   # rfile.read(-1) leería hasta que se cierre
                        raise ValueError('Content-Length inválido')
                    if n > 4096:
                        raise ValueError('demasiado grande')
                    raw = self.rfile.read(n) if n else b''
                    data = json.loads(raw) if raw.strip() else {}
                    if not isinstance(data, dict):
                        raise ValueError('se esperaba un objeto JSON')
                except ValueError as exc:
                    self._json(400, {'error': str(exc)})
                    return
                try:
                    code, out = on_post(path, data)
                except Exception as exc:        # que un error no tire el servidor
                    code, out = 500, {'error': str(exc)}
                self._json(code, out)

            def _json(self, code, obj):
                self._reply(code, 'application/json', json.dumps(obj).encode())

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
