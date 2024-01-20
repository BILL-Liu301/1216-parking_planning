from threading import Thread
import matlab.engine as engine


class CallOptim(Thread):
    def __init__(self, init_data):
        Thread.__init__(self)
        self.init_data = init_data
        self.eng = engine.start_matlab()
        self.eng.cd("../matlabs/parking_4steps/")
        self.future = None

    def start_matlab(self):
        self.future = self.eng.call_optim(self.init_data[0], self.init_data[1], self.init_data[2], background=True)

    def done(self):
        return self.future.done()

    def cancel(self):
        return self.future.cancel()

    def result(self):
        return self.future.result()

    def quit(self):
        self.eng.quit()

    def run(self):
        self.start_matlab()
