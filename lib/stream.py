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