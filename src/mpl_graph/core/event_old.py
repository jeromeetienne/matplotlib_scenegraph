class Event:
    def __init__(self):
        # TODO could i use template to make it stronger typed?
        self._listeners = []

    def subscribe(self, callback):
        """
        Subscribe a callback to the event.
        The callback should accept the same arguments as those passed during dispatch.
        """
        self._listeners.append(callback)

    def dispatch(self, *args, **kwargs):
        """
        Dispatch the event to all subscribed listeners.
        Additional arguments are passed to the listeners.
        """
        for listener in self._listeners:
            listener(*args, **kwargs)
