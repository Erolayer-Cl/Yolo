"""`main()` común a todos los nodos, con un cierre por Ctrl-C que no se corta.

Ctrl-C en tmux le llega a cada nodo DOS veces casi juntas: una por el grupo
de procesos de la terminal y otra reenviada por `ros2 launch`. Con el handler
de rclpy, la segunda queda pendiente y explota como KeyboardInterrupt en medio
del cierre -- p.ej. a mitad del último guardado del recorrido en trail_node.

Acá se desactiva el handler de rclpy. El propio solo levanta una bandera e
ignora las señales siguientes; el bucle usa spin_once con timeout para volver
siempre a Python a mirarla (un spin() bloqueante sin mensajes entrantes no
vuelve nunca) y no se lanza ninguna excepción en medio de un callback, p.ej.
a mitad de una inferencia de torch.

Un error en un callback (p.ej. la GPU sin memoria porque Unreal la está usando)
se registra y el nodo sigue: perder un frame es mejor que dejar de detectar
durante el resto del vuelo.
"""
import signal
import traceback

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def run(node_cls, args=None, on_exit=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    node = None
    try:
        node = node_cls()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while not _stop and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.2)
            except ExternalShutdownException:
                raise
            except Exception:
                node.get_logger().error(
                    f'Error en un callback, el nodo sigue:\n{traceback.format_exc()}',
                    throttle_duration_sec=5.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        _on_signal(None, None)
        if node is not None:
            try:
                if on_exit is not None:
                    on_exit(node)
            finally:
                node.destroy_node()
        rclpy.try_shutdown()
