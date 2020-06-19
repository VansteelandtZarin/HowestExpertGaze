# Imports

from tkinter import ttk
from tkinter import *
from tkinter.font import *
from tkinter import filedialog

import cv2
import PIL.Image, PIL.ImageTk

import os
from time import sleep
from datetime import datetime
import numpy as np
import pandas as pd
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing import app as processing
from processing_pupil_labs import pupil_labs


class Navigation(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # --------------------- Inherit controller ---------------------
        self.controller = controller

        # --------------------- Colors ---------------------
        self.background = '#f5f7f7'
        self.text = '#4b4b4b'
        self.light_text = 'white'
        self.teal = '#008080'
        self.light_teal = '#00a0a0'

        # --------------------- Fonts ---------------------
        self.font_sm = Font(family='Roboto', size=12)
        self.font_sm_bold = Font(family='Roboto', size=12, weight='bold')
        self.font_md = Font(family='Roboto', size=20, weight='bold')

        # --------------------- Layout ---------------------
        self.frame = Frame(self, bg=self.background)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.navbar = Frame(self.frame, bg=self.teal)
        self.navbar.place(relx=0, rely=0, relwidth=1, height=100)

        # --------------------- Nav items ---------------------

        # --------------------- Whitespace ---------------------
        self.whitespace = Frame(self.navbar, bg=self.teal, width=150)
        self.whitespace.pack(side='left')

        self.border_left = Frame(self.navbar, bg=self.background, width=1, height=100)
        self.border_left.pack(side='left')

        # --------------------- Home ---------------------
        self.home = Button(self.navbar, text="Home", font=self.font_sm_bold, bg=self.teal, height=100, padx=20,
        activebackground=self.light_teal, fg=self.light_text, activeforeground=self.light_text, bd=0,
        command=lambda: self.controller.show_frame(Home))
        self.home.pack(side='left')

        self.border_center_1 = Frame(self.navbar, bg=self.background, width=1, height=100)
        self.border_center_1.pack(side='left')


        # --------------------- Vuelosophy processing ---------------------
        self.vuelosophy = Button(self.navbar, text="Vuelosophy", font=self.font_sm_bold, bg=self.teal,
        height=100, padx=20, activebackground=self.light_teal, fg=self.light_text, activeforeground=self.light_text,
        bd=0, command=lambda: self.controller.show_frame(Vuelosophy))
        self.vuelosophy.pack(side='left')

        self.border_center_2 = Frame(self.navbar, bg=self.background, width=1, height=100)
        self.border_center_2.pack(side='left')

        # --------------------- Pupil Labs processing ---------------------
        self.pupil_labs = Button(self.navbar, text="Pupil labs", font=self.font_sm_bold, bg=self.teal,
        padx=20, activebackground=self.light_teal, fg=self.light_text, activeforeground=self.light_text, bd=0
        , height=100, command=lambda: self.controller.show_frame(PupilLabs))
        self.pupil_labs.pack(side='left')

        self.border_right = Frame(self.navbar, bg=self.background, width=1, height=100)
        self.border_right.pack(side='left')


class Home(Navigation):
    def __init__(self, parent, controller):
        Navigation.__init__(self, parent, controller)

        # Variables
        self.controller = controller

        # Navigation bar
        self.home.config(bg=self.light_teal)
        self.vuelosophy.config(bg=self.teal)
        self.pupil_labs.config(bg=self.teal)

        # Layout
        self.title = Label(self.frame, text="Expertgaze", bg=self.background, fg=self.text, font=self.font_md,
                           anchor='w')
        self.title.place(relx=0.1, rely=0.2, relwidth=0.8)

        self.intro = Text(self.frame, bg=self.background, fg=self.text, bd=0, font=self.font_sm)
        self.intro.insert(INSERT, "This app is used for:\n\n\t- Object recognition\n\t- Small insights\n\n"
                                  "This is available for the Vuelosophy eyetracker and the Pupil Labs eyetracker\n\n\n\n"
                                  "---------------- Vuelosophy ----------------\n\nTo start processing, you only need a .h264 file that can be "
                                  "extracted from the usb\nin the Raspberry Pi from the eyetracker using IOS (no Windows support yet)\n\n\n\n"
                                  "---------------- Pupil Labs ----------------\n\nTo start processing, you need 3 files:\n\n\t- world.mp4 (not the exported one)"
                                  "\n\t- fixations.csv\n\t- gaze_positions.csv\n\nThese files must be in the same directory\n\n\n\n"
                                  "Note: In order to get the 3 files needed to process the Pupil Labs eyetracker video, you first need to "
                                  "process the files using the Pupil Labs Player software")
        self.intro.place(relx=0.1, rely=0.325, relwidth=0.8, relheight=0.6)


class Vuelosophy(Navigation):
    def __init__(self, parent, controller):
        Navigation.__init__(self, parent, controller)

        # Variables
        self.controller = controller
        self.filepath = ""
        self.filename = ""
        self.fig = None
        self.gnt = None
        self.play = True

        # Navigation bar
        self.home.config(bg=self.teal)
        self.vuelosophy.config(bg=self.light_teal)
        self.pupil_labs.config(bg=self.teal)

        # Layout
        self.title = Label(self.frame, text='Select a file (.h264)', bg=self.background, fg=self.text,
        font=self.font_md, anchor="w")
        self.title.place(relx=0.1, rely=0.2, relwidth=0.8)

        self.message = Label(self.frame, text='No file selected', bg=self.background, fg=self.text, font=self.font_sm,
        anchor="w")
        self.message.place(relx=0.1, rely=0.3, relwidth=0.8)

        self.select = Button(self.frame, text='Select a file', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
        activebackground=self.light_teal, activeforeground='white', command=lambda: self.select_file())
        self.select.place(relx=0.1, rely=0.4, relwidth=0.2, height=60)

        self.next_step = Button(self.frame, text='Proceed', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
        activebackground=self.light_teal, activeforeground='white', command=lambda: self.next())

        self.error = Label(self.frame, text='', bg=self.background, fg="red", font=self.font_sm, anchor="w")

    def select_file(self):
        # Select a file
        self.filepath = filedialog.askopenfilename(initialdir="/", title="Select file")

        if self.filepath == "":
            self.error.place_forget()
            self.message.config(text="No file selected")
            self.next_step.place_forget()
        else:
            # Show the selected path
            self.message.config(text=self.filepath)

            # Create the filename
            split_path = self.filepath.split("/")

            if ".h264" in split_path[len(split_path) - 1]:
                self.filename = split_path[len(split_path) - 1].rstrip(".h264")
                self.next_step.place(relx=0.35, rely=0.4, relwidth=0.2, height=60)
            else:
                self.error.config(text="Wrong file type")
                self.error.place(relx=0.1, rely=0.5, relwidth=0.8)
                self.next_step.place_forget()

    def next(self):
        # Forget current layout
        self.title.config(text="Processing")
        self.select.place_forget()
        self.next_step.place_forget()
        self.error.place_forget()

        try:

            t = Thread(target=lambda: processing(self.filepath, self.filename, self.controller.queue))
            t.start()

        except Exception as ex:
            print(ex)

    def play_video(self):

        video_path = './Vuelosophy_IO/output_MP4/%s.mp4' % self.filename

        if self.filename != "":

            # Create a Video object
            self.video = Video(video_path)

            self.title.place_forget()
            self.message.place_forget()

            print(self.video.width)
            print(self.video.height)

            self.canvas = Canvas(self.frame, width=self.video.width, height=self.video.height)
            self.canvas.place(relx='0.05', rely='0.2')

            self.progress = ttk.Progressbar(self.frame, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress.place(relx='0.05', rely='0.75', width=self.video.width)
            self.progress["value"] = 0
            self.progress["maximum"] = self.video.total_nr_frames

            self.toolbar = Frame(self.frame, bg=self.background)
            self.toolbar.place(relx='0.05', rely='0.8', width=self.video.width)

            self.begin = Button(self.toolbar, text='Begin', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("BEGIN"), width=10)
            self.begin.pack(expand='true', side='left')

            self.back = Button(self.toolbar, text='Back', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("BACK"), width=10)
            self.back.pack(expand='true', side='left')

            self.pauze_play = Button(self.toolbar, text='Pauze', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.toggle_play(), width=10)
            self.pauze_play.pack(expand='true', side='left')

            self.forward = Button(self.toolbar, text='Forward', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("FORWARD"), width=10)
            self.forward.pack(expand='true', side='left')

            self.end = Button(self.toolbar, text='End', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("END"), width=10)
            self.end.pack(expand='true', side='left')

            self.delay = 0.01

            self.video_thread = Thread(target=lambda: self.update())
            self.video_thread.start()

        else:
            raise ValueError("Unable to open video source", video_path)

    def update(self):
        while True:
            if self.play:
                # Get a frame from the video source
                ret, frame = self.video.get_frame()
                if ret:
                    self.video_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.video_frame, anchor="nw")
                    self.plot()
                    self.progress['value'] = self.video.video.get(cv2.CAP_PROP_POS_FRAMES)
                    self.progress.update()

                    sleep(self.delay)
                else:
                    self.play = False

    def toggle_play(self):
        if self.play:
            self.play = False
            self.pauze_play.config(text="Play")
        else:
            self.play = True
            self.pauze_play.config(text="Pauze")

    def go_to(self, action=""):
        self.play = False
        self.pauze_play.config(text="Play")

        if action == "BEGIN":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif action == "BACK":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.video.get(cv2.CAP_PROP_POS_FRAMES) - (self.video.fps * 2))
        elif action == "FORWARD":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.video.get(cv2.CAP_PROP_POS_FRAMES) + (self.video.fps * 2))
        elif action == "END":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.total_nr_frames - 2.0)
        else:
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Cant get a new frame
        ret, frame = self.video.get_frame()

        if ret:
            self.video_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.video_frame, anchor="nw")
        self.progress['value'] = self.video.video.get(cv2.CAP_PROP_POS_FRAMES)
        self.progress.update()

    def plot(self):

        df_body = pd.read_pickle("./Vuelosophy_IO/output_PKL/body_final_%s.pkl" % self.filename)


        # Get the usefull bit of the body
        # df_body = df_body.dropna()[0:int(self.video.video.get(cv2.CAP_PROP_POS_FRAMES))]
        df_body = df_body[df_body["frame_number"] <= int(self.video.video.get(cv2.CAP_PROP_POS_FRAMES))]

        # Get the unique classes
        viewed_classes = df_body['object'].unique()

        graph_data = {}

        # Sort data by class
        for classname in viewed_classes:
            graph_data[classname] = df_body[df_body['object'] == classname]

        if self.fig is not None:
            plt.close(self.fig)

        self.fig, self.gnt = plt.subplots()

        self.gnt.set_yticks(np.arange(10, len(viewed_classes) * 10 + 1, 10))
        self.gnt.set_xlabel('Timespan')

        self.gnt.set_xlim([0, self.video.total_nr_frames])

        self.gnt.set_yticklabels(viewed_classes)
        self.gnt.grid(True)

        colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange']
        for key, val in graph_data.items():
            output = [(row['frame_number'], 1) for index, row in val.iterrows()]
            nr = list(viewed_classes).index(key)
            self.gnt.broken_barh(output, (nr * 10 + 5, 10), facecolors=colors[nr])

        self.graph = FigureCanvasTkAgg(self.fig, self.frame)
        self.graph.get_tk_widget().place(relx=0.5, rely=0.2, relwidth=0.4)


class PupilLabs(Navigation):
    def __init__(self, parent, controller):
        Navigation.__init__(self, parent, controller)

        # Variables
        self.controller = controller
        self.fig = None
        self.gnt = None
        self.play = True

        # Navigation bar
        self.home.config(bg=self.teal)
        self.vuelosophy.config(bg=self.teal)
        self.pupil_labs.config(bg=self.light_teal)

        # Layout
        self.title = Label(self.frame, text='Select a folder', bg=self.background, fg=self.text,
                           font=self.font_md, anchor="w")
        self.title.place(relx=0.1, rely=0.2, relwidth=0.8)

        self.message = Label(self.frame, text='No folder selected', bg=self.background, fg=self.text, font=self.font_sm,
                             anchor="w")
        self.message.place(relx=0.1, rely=0.3, relwidth=0.8)

        self.select = Button(self.frame, text='Select a folder', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
                             activebackground=self.light_teal, activeforeground='white',
                             command=lambda: self.select_folder())
        self.select.place(relx=0.1, rely=0.4, relwidth=0.2, height=60)

        self.next_step = Button(self.frame, text='Proceed', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
                                activebackground=self.light_teal, activeforeground='white', command=lambda: self.next())

        self.error = Label(self.frame, text='', bg=self.background, fg="red", font=self.font_sm, anchor="w")

    def select_folder(self):
        # Select a file
        self.folderpath = filedialog.askdirectory(initialdir="/", title="Select file")

        if self.folderpath == "":
            self.error.place_forget()
            self.message.config(text="No file selected")
            self.next_step.place_forget()
        else:
            # Show the selected path
            self.message.config(text=self.folderpath)

            if "world.mp4" in os.listdir(self.folderpath):
                self.next_step.place(relx=0.35, rely=0.4, relwidth=0.2, height=60)
            else:
                self.error.config(text="world.mp4 does not exist")
                self.error.place(relx=0.1, rely=0.5, relwidth=0.8)
                self.next_step.place_forget()

            if "fixations.csv" in os.listdir(self.folderpath):
                self.next_step.place(relx=0.35, rely=0.4, relwidth=0.2, height=60)
            else:
                self.error.config(text="fixations.csv does not exist")
                self.error.place(relx=0.1, rely=0.5, relwidth=0.8)
                self.next_step.place_forget()

            if "gaze_positions.csv" in os.listdir(self.folderpath):
                self.next_step.place(relx=0.35, rely=0.4, relwidth=0.2, height=60)
            else:
                self.error.config(text="gaze_positions.csv does not exist")
                self.error.place(relx=0.1, rely=0.5, relwidth=0.8)
                self.next_step.place_forget()

    def next(self):
        # Forget current layout
        self.title.config(text="Processing")
        self.select.place_forget()
        self.next_step.place_forget()
        self.error.place_forget()

        try:
            t = Thread(target=lambda: pupil_labs(self.folderpath, self.controller.queue))
            t.start()

        except Exception as ex:
            print(ex)

    def play_video(self):

        video_path = './Pupil_Labs_IO/output_MP4/world.mp4'

        if self.folderpath != "":

            # Create a Video object
            self.video = Video(video_path)

            self.title.place_forget()
            self.message.place_forget()

            self.body = pd.read_pickle("./Pupil_Labs_IO/output_PKL/body.pkl")

            self.canvas = Canvas(self.frame, width=640, height=480)
            self.canvas.place(relx='0.05', rely='0.2')

            self.progress = ttk.Progressbar(self.frame, orient=HORIZONTAL, length=100, mode='determinate')
            self.progress.place(relx='0.05', rely='0.75', width=640)
            self.progress["value"] = 0
            self.progress["maximum"] = self.video.total_nr_frames

            self.toolbar = Frame(self.frame, bg=self.background)
            self.toolbar.place(relx='0.05', rely='0.8', width=640)

            self.begin = Button(self.toolbar, text='Begin', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("BEGIN"), width=10)
            self.begin.pack(expand='true', side='left')

            self.back = Button(self.toolbar, text='Back', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("BACK"), width=10)
            self.back.pack(expand='true', side='left')

            self.pauze_play = Button(self.toolbar, text='Pauze', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.toggle_play(), width=10)
            self.pauze_play.pack(expand='true', side='left')

            self.forward = Button(self.toolbar, text='Forward', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("FORWARD"), width=10)
            self.forward.pack(expand='true', side='left')

            self.end = Button(self.toolbar, text='End', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
            activebackground=self.light_teal, activeforeground='white', command=lambda: self.go_to("END"), width=10)
            self.end.pack(expand='true', side='left')

            self.delay = 0.01

            self.video_thread = Thread(target=lambda: self.update())
            self.video_thread.start()

        else:
            raise ValueError("Unable to open video source", video_path)

    def update(self):
        while True:
            if self.play:
                # Get a frame from the video source
                ret, frame = self.video.get_frame()
                if ret:

                    frame = cv2.resize(frame, (640, 480))

                    self.video_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.video_frame, anchor="nw")
                    self.plot()
                    self.progress['value'] = self.video.video.get(cv2.CAP_PROP_POS_FRAMES)
                    self.progress.update()

                    sleep(self.delay)
                else:
                    self.play = False

    def toggle_play(self):
        if self.play:
            self.play = False
            self.pauze_play.config(text="Play")
        else:
            self.play = True
            self.pauze_play.config(text="Pauze")

    def go_to(self, action=""):
        self.play = False
        self.pauze_play.config(text="Play")

        if action == "BEGIN":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif action == "BACK":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.video.get(cv2.CAP_PROP_POS_FRAMES) - (self.video.fps * 2))
        elif action == "FORWARD":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.video.get(cv2.CAP_PROP_POS_FRAMES) + (self.video.fps * 2))
        elif action == "END":
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, self.video.total_nr_frames - 3.0)
        else:
            self.video.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Cant get a new frame
        ret, frame = self.video.get_frame()

        if ret:
            self.video_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.video_frame, anchor="nw")
        self.progress['value'] = self.video.video.get(cv2.CAP_PROP_POS_FRAMES)
        self.progress.update()

    def plot(self):

        # Get the usefull bit of the body
        usefull = self.body[self.body["frame_number"] <= int(self.video.video.get(cv2.CAP_PROP_POS_FRAMES))]

        print(self.video.video.get(cv2.CAP_PROP_POS_FRAMES))

        # Get the unique classes
        viewed_classes = usefull['object'].unique()
        viewed_classes = viewed_classes.tolist()

        viewed_classes.remove(None)

        graph_data = {}

        # Sort data by class
        for classname in viewed_classes:
            graph_data[classname] = usefull[usefull['object'] == classname]

        if self.fig is not None:
            plt.close(self.fig)

        self.fig, self.gnt = plt.subplots()

        self.gnt.set_yticks(np.arange(10, len(viewed_classes) * 10 + 1, 10))
        self.gnt.set_xlabel('Timespan')

        self.gnt.set_xlim([0, self.video.total_nr_frames])

        self.gnt.set_yticklabels(viewed_classes)
        self.gnt.grid(True)

        colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange']
        for key, val in graph_data.items():
            output = [(row['frame_number'], 1) for index, row in val.iterrows()]
            nr = list(viewed_classes).index(key)
            self.gnt.broken_barh(output, (nr * 10 + 5, 10), facecolors=colors[nr])

        self.graph = FigureCanvasTkAgg(self.fig, self.frame)
        self.graph.get_tk_widget().place(relx=0.5, rely=0.2, relwidth=0.4)


class Video:
    def __init__(self, video_path):

        # Open the video source
        self.video = cv2.VideoCapture(video_path)

        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_path)

        # Get video source width and height
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_nr_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        ret = None

        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                # Return a boolean success flag and the current frame converted to RGB
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            else:
                return (ret, None)
        else:
            return (ret, None)
