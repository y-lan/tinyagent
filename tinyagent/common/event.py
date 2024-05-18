import collections


class SimpleEventManager:
    def __init__(self):
        self.listeners = collections.defaultdict(list)

    def subscribe(self, event_name, listener):
        self.listeners[event_name].append(listener)

    def publish(self, event_name, **kwargs):
        for listener in self.listeners[event_name]:
            listener(data=kwargs)
