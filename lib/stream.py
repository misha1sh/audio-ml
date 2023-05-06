from collections import deque

class Stream:
    def __init__(self, generator):
        try:
            self.generator = iter(generator)
        except TypeError:
            self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    @staticmethod
    def repeat(element, n):
        def generator():
          for i in range(n):
              yield element
        return Stream(generator())

    def chain(self, another_stream):
        def generator():
            for i in self:
                yield i
            for i in another_stream:
                yield i
        return Stream(generator())

    def slide_window(self, window_size):
        res = deque()
        for i in self:
          res.append(i)
          if len(res) == window_size:
            yield Stream(res)
            res.popleft()

    def skip(self, count):
        def generator():
            n = count
            for i in self.generator:
                n -= 1
                if n == 0: break
            for i in self.generator:
                yield i
        return Stream(generator())

    def get(self, count):
        res = []
        for i in self:
            res.append(i)
            if len(res) == count:
                return res
        return res

    def limit(self, count):
        def generator():
            n = count
            for i in self.generator:
                yield i
                n -= 1
                if n == 0: break
        return Stream(generator())

    def map(self, func):
        def generator():
            for i in self.generator:
                yield func(i)
        return Stream(generator())

    def starmap(self, func):
        def generator():
            for i in self.generator:
                for j in func(i):
                    yield j
        return Stream(generator())

    def group(self, n):
        def generator():
            grouped = []
            for i in self.generator:
                grouped.append(i)
                if len(grouped) >= n:
                    yield grouped
                    grouped = []
            if len(grouped) != 0:
                yield grouped

        return Stream(generator())