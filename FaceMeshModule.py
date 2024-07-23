class FPS():
    def __init__(self):
        self.start = time.time()
        self.frames = 0
        self.fps = 0
        
    def update(self):
        self.frames += 1
        if time.time() - self.start >= 1:
            self.fps = self.frames
            self.frames = 0
            self.start = time.time()
        return self.fps