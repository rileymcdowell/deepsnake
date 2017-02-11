from collections import deque

class SnakeDeque(deque):
    def __init__(self, *args, **kwargs):
        super(SnakeDeque, self).__init__(*args, **kwargs)

    def appendleft_maybepop(self, value):
        popped = None
        if len(self) == self.maxlen:
            popped = self.pop()
        self.appendleft(value)
        return popped 

