# Imports

from pages import *

from threading import Thread
# from queue import Queue
from multiprocessing import Queue

from time import sleep


# Classes

class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        # Variables
        self.frames = {}
        self.queue = None
        self.stepcount = 0
        self.substepcount = 0

        # Initialise frames
        container = Frame(self)
        container.pack(side='top', fill='both', expand=True)

        pages = [Home, Vuelosophy, PupilLabs]

        for f in pages:
            frame = f(container, self)
            self.frames[f] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame(Home)
        self.create_queue()

    def show_frame(self, frame):
        frame = self.frames[frame]
        frame.tkraise()

    def create_queue(self):
        self.queue = Queue()
        t = Thread(target=self.handle_queue)
        t.start()

    def handle_queue(self):
        while True:
            message = self.queue.get()

            if "START_PROCESSING_VUE" in message:
                self.stepcount = 0
                self.substepcount = 0

            if "START_PROCESSING_PUP" in message:
                print("Oegaboega")
                self.stepcount = 0
                self.substepcount = 0

            elif message == "END_PROCESSING_VUE":
                self.frames[Vuelosophy].play_video()
                self.frames[Vuelosophy].plot()

            elif "END_PROCESSING_PUP" in message:
                self.frames[PupilLabs].play_video()
                self.frames[PupilLabs].plot()

            elif "FILENAME:" in message:
                self.filename = message.lstrip("FILENAME:")

            elif "FRAMES:" in message:
                self.frame_count = message.lstrip("FRAMES: ")

            elif "STEP_VUE" in message:
                if "SUB" in message:
                    self.substepcount += 1
                    self.frames[Vuelosophy].message.config(text="Substep: Processing frames %i/%s" % (self.substepcount, self.frame_count))
                else:
                    step = message.lstrip("STEP_VUE:")
                    self.stepcount += 1
                    self.frames[Vuelosophy].message.config(text="Step %i/5: %s" % (self.stepcount, step))

            elif "STEP_PUP" in message:
                if "SUB" in message:
                    self.substepcount += 1
                    self.frames[PupilLabs].message.config(text="Substep: Calculating fixations %i/%s" % (self.substepcount, self.frame_count))
                else:
                    step = message.lstrip("STEP_PUP:")
                    self.stepcount += 1
                    self.frames[PupilLabs].message.config(text="Step %i/4: %s" % (self.stepcount, step))
# App
if __name__ == "__main__":
    root = App()
    root.title("Howest ExpertGaze")
    root.geometry("1600x900")
    root.mainloop()
