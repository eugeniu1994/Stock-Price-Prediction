
import matplotlib
from PIL import ImageTk, Image
import threading
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
from pandas import read_csv
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import Label
from configparser import ConfigParser
import center_tk_window
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import time
from datetime import datetime
import queue
import tkinter.font as font
from tkinter import scrolledtext
import logging
from os import listdir
from tkinter import simpledialog
import re
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras
from tensorflow import Session, Graph
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg


LARGE_FONT = ("Verdana", 12)
style.use("ggplot")

version = "Predict.exe"

import win32gui
import sys

hwnd = win32gui.FindWindow(None, version)
if hwnd:
    win32gui.ShowWindow(hwnd, 5)
    win32gui.SetForegroundWindow(hwnd)
    sys.exit(0)

config = ConfigParser()
global LoadedModel,graph, thread_session, canvas, f
LoadedModel = thread_session = graph = None
f = Figure(figsize=(5, 5), dpi=100)
a = f.add_subplot(111)
global ax
fisier1 = ''
scaler = MinMaxScaler(feature_range=(0, 1))

logger = logging.getLogger(__name__)
global StopTran

def load_model_mine():
    thread_graph = Graph()
    with thread_graph.as_default():
        thread_session = Session()
        with thread_session.as_default():
            model = keras.models.load_model('models/' + config.get('section_data', 'model_name'))
            graph = tf.get_default_graph()

    return model, graph, thread_session

class Model_Predictor():
    def __init__(self, prices, model,graph, thread_session):
        self.model = model
        self.graph = graph
        self.thread_session = thread_session

        self.loop_back = config.getint('section_data', 'loop_back')
        self.num_output = config.getint('section_data', 'num_output')
        self.prediction_len = config.getint('section_data', 'int_val')
        self.prices = scaler.fit_transform(prices)  # already scaled

    def create_dataset(self, ds, loop_back=1):
        X, Y = [], []
        for i in range(len(ds) - loop_back): #-1
            a = ds[i:(i + loop_back)]
            X.append(a)
            Y.append(ds[i + loop_back])

        return np.array(X), np.array(Y)

    def get_test_data(self):
        testPredict = []
        try:
            ratio = config.getfloat('section_data', 'training_size')
            train_size = int(len(self.prices) * ratio)
            test_size = len(self.prices) - train_size
            test = self.prices[train_size:len(self.prices)]

            X_test, Y_test = self.create_dataset(test, config.getint('section_data', 'loop_back'))
            #testPredict = self.model.predict(X_test)
            with self.graph.as_default():
                with self.thread_session.as_default():
                    testPredict = self.model.predict(X_test)

        except Exception as err:
            e = str(err)
            tk.messagebox.showerror("Error", "Please choose the history length according to your chosen model " + str(
                err)) if e.find("to have shape") > 0 else tk.messagebox.showerror("Error","Unexpected error " + str(err))
            pass

        return testPredict, train_size, test_size

    def get_test_data_future(self):
        prediction = []
        try:
            ratio = config.getfloat('section_data', 'training_size')
            train_size = int(len(self.prices) * ratio)
            test = self.prices[train_size:len(self.prices)]

            prediction = self.predict_dataset(test, config.getint('section_data', 'int_val'))
        except Exception as err:
            e = str(err)
            tk.messagebox.showerror("Error", "Please choose the history length according to your chosen model " + str(
                err)) if e.find("to have shape") > 0 else tk.messagebox.showerror("Error","Unexpected error " + str(err))
            pass
        return prediction, train_size

    def predict_next_future_point(self, Multiple=True):
        batch = self.prices[-self.loop_back:].reshape((1, self.loop_back, self.num_output))
        if Multiple:
            rv = self.predict_batch(batch, config.getint('section_data', 'int_val'))
            if rv == []:
                return [],[]
            difference = abs(rv[0] - self.prices[-1]) if rv[0] >= self.prices[-1] else abs(self.prices[-1] - rv[0])
            Adjusted = (rv - difference) if rv[0] >= self.prices[-1] else (rv + difference)
            return rv, Adjusted
        else:
            rv = self.predict_batch(batch, 1)
            if rv == []:
                return [],[]
            difference = abs(rv[0] - self.prices[-1]) if rv[0] >= self.prices[-1] else abs(self.prices[-1] - rv[0])
            Adjusted = (rv - difference) if rv[0] >= self.prices[-1] else (rv + difference)
            return rv, Adjusted

    def predict_batch(self, batch, episodes):
        rv = []
        try:
            for i in range(episodes):
                #rv.append(self.model.predict(batch)[0])
                with self.graph.as_default():
                    with self.thread_session.as_default():
                        rv.append(self.model.predict(batch)[0])
                batch = np.append(batch[:, 1:, :], [[rv[i]]], axis=1)
        except Exception as err:
            e = str(err)
            try:
                if e.find("to have shape") >= 0:
                    model_name = config.get('section_data', 'model_name')[:-3]
                    _ = model_name.index('_')+1
                    loop_back = int(model_name[_:])
                    self.loop_back = loop_back
                    config.set('section_data', 'loop_back', str(loop_back))
                    with open('logs/Configuration.ini', 'w') as configfile:
                        config.write(configfile)

                    for i in range(episodes):
                        #rv.append(self.model.predict(batch)[0])
                        with self.graph.as_default():
                            with self.thread_session.as_default():
                                rv.append(self.model.predict(batch)[0])
                        batch = np.append(batch[:, 1:, :], [[rv[i]]], axis=1)
                else:
                    tk.messagebox.showerror("Error", "Unexpected error " + str(err))
            except Exception as err2:
                pass

        return rv

    def predict_dataset(self, ds, episodes):
        rv = []
        try:
            for i in range(0, len(ds) - self.loop_back + 1, episodes):
                window = ds[i:(i + self.loop_back)].reshape((1, self.loop_back, self.num_output))
                rv.append(self.predict_batch(window, episodes))
        except Exception as err:
            e = str(err)
            if e.find("to have shape") < 0:
                tk.messagebox.showerror("Error", "Unexpected error "+str(err))
            pass

        return np.array(rv)

    def getArchitecture(self, win_len, win_feature, Compiled = True):
        model = Sequential()
        model.add(LSTM(100, input_shape=(win_len, win_feature)))
        model.add(Dropout(0.4))
        model.add(Dense(config.getint('section_data', 'num_output')))
        if Compiled:
            model.compile(loss='mse', optimizer='adam')
        return model

    def getArchitecture_Default(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(config.getint('section_data', 'loop_back'), config.getint('section_data', 'num_output'))))
        model.add(Dropout(0.4))
        model.add(Dense(config.getint('section_data', 'num_output')))
        return model

    def TrainModel(self, Compiled=True):
        try:
            train_size = int(len(self.prices) * config.getfloat('section_data', 'training_size'))
            train, test = self.prices[0:train_size], self.prices[train_size:len(self.prices)]

            X_train, Y_train = self.create_dataset(train, config.getint('section_data', 'loop_back'))
            X_test, Y_test = self.create_dataset(test, config.getint('section_data', 'loop_back'))
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
            Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))

            batch_size = 32
            num_epochs = config.getint('section_data', 'num_epochs')
            patience_epochs = int(num_epochs / 3)
            if Compiled:
                model = self.getArchitecture(X_train.shape[1], X_train.shape[2], Compiled=Compiled)  # window history, num feature (50,1)
            else:
                model = self.getArchitecture_Default()

            return batch_size, num_epochs, patience_epochs, model, X_train, Y_train, X_test, Y_test
        except:
            pass

def animate(fisier1):
    pltData = []
    try:
        f.clf()
        a = f.add_subplot(111)
        data_column = config.get('section_data', 'data_column')
        points_from_file = config.getint('section_data', 'points_from_file')
        dataList = read_csv(fisier1, delimiter=',', usecols=[data_column])
        pltData = dataList[-points_from_file:len(dataList)] if len(dataList) > points_from_file else dataList
        pltData.index = range(0, len(pltData))
        a.set_ylabel('Prices')
        a.set_xlabel('Data points')
        a.plot(pltData, c='tab:blue', label='Real data')
        legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data') ]
        a.legend(handles=legend_elements, facecolor='white', framealpha=1)
    except  Exception as err:
        tk.messagebox.showerror("Error", "Unexpected error " + str(err))
    return pltData

def toggle(widget, predict=True, waiting=True, cancel=False):
    if widget.visible:
        widget.pack_forget()
    else:
        if predict and waiting:
            widget.pack(pady=100)
        elif predict == False and waiting:
            widget.pack(pady=10)
        elif cancel:
            widget.pack(side=tk.BOTTOM, padx=50, pady=0, expand=True, fill=tk.BOTH)
        else:
            widget.pack(side=tk.TOP, padx=50, pady=10, expand=True, fill=tk.BOTH)
    widget.visible = not widget.visible

class SeaofBTCapp():
    def __init__(self, root):
        self.root = root
        self.root.iconbitmap(self, default="icons/my.ico")
        self.root.wm_title("Stocks predictor")
        self.root.protocol("WM_DELETE_WINDOW", self.quitMethod)
        self.container = tk.Frame(root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=19)
        self.container.grid_rowconfigure(1, weight=1)
        self.frames = {}

        for F in (StartPage, PageThree):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
        self.model_predictor = None

        self.footer = tk.Frame(self.container, height=10)
        self.footer.grid(row=1, sticky='news')
        tk.Label(self.footer, text="Powered by Vezeteu Eugeniu", bg='white', font="Helvetica 7 italic").pack(
            side=tk.RIGHT, expand=True, fill='both')

        center_tk_window.center_on_screen(self.root)
        # center_tk_window.center(root_window, window_to_center)
        # center_tk_window.center_on_parent(root_window, window_to_center)
        self.waitingFrame = None
        self.clock = None
        self.startAgentFlag = False

    def TestModel(self):
        f.clf()
        ax = f.subplots(ncols=3, nrows=2)
        try:
            ratio = config.getfloat('section_data', 'training_size')
            train_size = int(len(self.model_predictor.prices) * ratio)
            test = self.model_predictor.prices[train_size:len(self.model_predictor.prices)]

            X_test, Y_test = self.model_predictor.create_dataset(test, config.getint('section_data', 'loop_back'))
            predictWhole = True
            if predictWhole:
                step = self.model_predictor.loop_back+6
                X_test, Y_test = X_test[-step:, ], scaler.inverse_transform(Y_test[-step:, ])
                with self.model_predictor.graph.as_default():
                    with self.model_predictor.thread_session.as_default():
                        rez = self.model_predictor.model.predict(X_test)
                i = 0
                for axs in ax.flat:
                    batch = X_test[i+self.model_predictor.loop_back].reshape((1, self.model_predictor.loop_back, self.model_predictor.num_output))
                    self.show_plot_whole([scaler.inverse_transform(X_test[i+self.model_predictor.loop_back]), Y_test[i+self.model_predictor.loop_back],
                                    scaler.inverse_transform(self.model_predictor.predict_batch(batch, 1))], 0, axs, scaler.inverse_transform(rez[i:(i + self.model_predictor.loop_back)]))
                    i = i + 1
            else:
                X_test, Y_test = X_test[-6:,], scaler.inverse_transform(Y_test[-6:,])
                i=0
                for axs in ax.flat:
                    batch = X_test[i].reshape((1, self.model_predictor.loop_back, self.model_predictor.num_output))
                    self.show_plot([scaler.inverse_transform(X_test[i]), Y_test[i], scaler.inverse_transform(self.model_predictor.predict_batch(batch,1))], 0, axs)
                    i=i+1

            canvas.draw()
        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error "+str(err))
            raise

    def show_plot_whole(self, plot_data, delta, axes, modelPred):
        labels = ['True history', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        for i, x in enumerate(plot_data):
            if i:
                axes.plot(future, plot_data[i], marker[i], markersize=7,
                          label=labels[i])
            else:
                axes.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i], c='tab:blue')
                axes.plot(time_steps, modelPred, marker[i], label='Predicted history', c='tab:green')
        axes.legend(facecolor='white', framealpha=1, fontsize='x-small')

    def show_plot(self, plot_data, delta, axes):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        for i, x in enumerate(plot_data):
            if i:
                axes.plot(future, plot_data[i], marker[i], markersize=7,
                         label=labels[i])
            else:
                axes.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i],c = 'tab:blue')
        axes.legend(facecolor='white', framealpha=1,fontsize = 'x-small')

    def create_time_steps(self, length):
        return list(range(-length, 0))

    def TrainModel(self):
        try:
            MsgBox = tk.messagebox.askquestion('Train your model', 'Are you sure you want to train a new model ?  \n (For data security please double check the configurations)',
                                               icon='warning')
            if MsgBox == 'yes':
                btnPredict['state'] = 'disabled'
                btnTrain['state'] = 'disabled'
                btnUpdate['state'] = 'disabled'
                btnBack['state'] = 'disabled'
                btnBack['state'] = 'disabled'
                btnTest['state'] = 'disabled'
                f.clf()
                f.set_visible(False)
                canvas.draw()

                toggle(WaitingWidget, False, True)
                toggle(btnCancel, False, False, True)
                toggle(trainTerminal, False, False)
                self.root.update()
                StopTran.set(1)
                answer = simpledialog.askstring("Model name", "Please specify your model name", parent=self.root)
                if answer is None:
                    answer = 'NewModel_' + str(config.getint('section_data', 'loop_back'))
                else:
                    answer = re.sub(r"[^a-zA-Z]", "", answer)
                    answer = answer + '_' + str(config.getint('section_data', 'loop_back'))

                modelNewName = 'models/' + answer + '.h5'

                self.clock = Clock(self.model_predictor, modelNewName)
                self.clock.Flag = True
                self.clock.start()

        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error "+str(err))
            raise

    def CancelTraining(self):
        try:
            MsgBox = tk.messagebox.askquestion('Stop training',
                                               'Are you sure you want to stop training your model ?  \n (All trained model data will be lost)',
                                               icon='warning')
            if MsgBox == 'yes':
                StopTran.set(0)
                btnPredict['state'] = 'normal'
                btnTrain['state'] = 'normal'
                btnUpdate['state'] = 'normal'
                btnBack['state'] = 'normal'
                btnTest['state'] = 'normal'
                toggle(WaitingWidget, False, True)
                toggle(btnCancel, False, False, True)
                toggle(trainTerminal, False, False)
                self.clock.stop()

                f.set_visible(True)
                f.clf()
                a = f.add_subplot(111)
                a.plot(scaler.inverse_transform(self.model_predictor.prices), c='tab:blue', label='Real data')
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=1, label='Real data')]
                a.set_ylabel('Prices')
                a.set_xlabel('Data points')
                a.legend(handles=legend_elements, facecolor='white', framealpha=1)
                canvas.draw()
                self.clock.Flag = False

        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error "+str(err))
            raise

    def ComputePredictions(self, var1, var2, out):
        try:
            if var1 == 1:  # predict from real points
                self_predictions, train_size, test_size = self.model_predictor.get_test_data()
                out.put((self_predictions, train_size))
            if var2 == 1:
                future_predict, train_size = self.model_predictor.get_test_data_future()
                out.put((future_predict, train_size))
                next_future, Adjusted = self.model_predictor.predict_next_future_point(Multiple=True)
                out.put((next_future, Adjusted))
            else:
                next_future, Adjusted = self.model_predictor.predict_next_future_point(Multiple=False)
                out.put((next_future, Adjusted))
        except Exception as e:
            if var1 == 1:  # predict from real points
                out.put(([], 0))
            if var2 == 1:
                out.put(([], 0))
                out.put(([], 0))
            else:
                out.put(([], 0))
            pass

    def PredictStock(self):
        try:
            if self.model_predictor is not None:
                ratio = config.getfloat('section_data', 'training_size')
                train_size = int(len(self.model_predictor.prices) * ratio)
                test_size = len(self.model_predictor.prices) - train_size - 1
                if test_size <= self.model_predictor.loop_back:
                    msg = 'There is no enough points for testing, testing data is smaller than history window, we suggest you to change trains size parameter or to upload a bigger file'
                    tk.messagebox.showerror("Warning", msg)
                else:
                    toggle(WaitingWidget)
                    self.root.update()
                    self.root.wm_attributes("-disabled", True)
                    self.generate()
        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error "+str(err))
            raise

    def generate(self):
        try:
            self.my_queue = queue.Queue()
            self.update_Thread = threading.Thread(target=self.ComputePredictions,
                                                  args=(var1_selfData.get(), var2_multiple_points.get(), self.my_queue))
            self.update_Thread.setDaemon(True)
            self.update_Thread.start()
            self.wait_generate()
        except Exception as err:
            self.update_Thread.join()
            pass

    def wait_generate(self):
        if self.update_Thread.isAlive():
            self.root.after(500, self.wait_generate)
        else:
            self.update_Thread.join()
            f.clf()
            a = f.add_subplot(111)
            a.plot(scaler.inverse_transform(self.model_predictor.prices), c='tab:blue', label='Real data')
            legend_elements = [Line2D([0], [0], color='tab:blue', lw=1, label='Real data')]

            try:
                if var1_selfData.get() == 1:  # predict from real points
                    self_predictions, train_size = self.my_queue.get()
                    shift = 0
                    a.plot(range(train_size + self.model_predictor.loop_back,
                                 train_size + len(self_predictions)+ self.model_predictor.loop_back),
                           scaler.inverse_transform(self_predictions), c='tab:purple')

                    legend_elements.append(
                        Line2D([0], [0], color='tab:purple', label='Self test', markerfacecolor='tab:purple'))

                if var2_multiple_points.get() == 1:
                    future_predict, train_size = self.my_queue.get()
                    for i in range(future_predict.shape[0]):
                        idx = train_size + self.model_predictor.loop_back - 1 + (
                                i * config.getint('section_data', 'int_val'))
                        range_ = range(idx, idx + config.getint('section_data', 'int_val'))
                        # print(range_)
                        a.plot(range_, scaler.inverse_transform(future_predict[i]), c='limegreen')
                    legend_elements.append(
                        Line2D([0], [0], color='limegreen', label='Test predicts', markerfacecolor='r'))

                    next_future, Adjusted = self.my_queue.get()
                    idx = len(self.model_predictor.prices) - 1
                    range_ = range(idx, idx + len(next_future))
                    a.plot(range_, scaler.inverse_transform(next_future), c='m', lw=2)
                    legend_elements.append(Line2D([0], [0], color='m', lw=2, label='Future', markerfacecolor='m'))
                    a.plot(range_, scaler.inverse_transform(Adjusted), c='r', lw=2)
                    legend_elements.append(
                        Line2D([0], [0], color='r', lw=2, label='Adjusted Future', markerfacecolor='r'))
                    a.axvline(x=idx, color='k', linestyle='--')
                else:
                    next_future, Adjusted = self.my_queue.get()
                    idx = len(self.model_predictor.prices)
                    a.plot(idx, scaler.inverse_transform(next_future), 'rx', lw=5, markersize=10, label='Future')
                    legend_elements.append(Line2D([0], [0], color='r', lw=1, label='Future', markerfacecolor='rx'))
            except Exception as err:
                tk.messagebox.showerror("Error","Unexpected error " + str(err))
                pass
            a.set_ylabel('Prices')
            a.set_xlabel('Data points')
            a.legend(handles=legend_elements, facecolor='white', framealpha=1)
            canvas.draw()
            self.root.wm_attributes("-disabled", False)
            toggle(WaitingWidget)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def callback_(self):
        fisier1 = askopenfilename(filetypes=[("CSV", "*.csv")])
        if fisier1 is not None and fisier1 is not "":
            self.show_frame(PageThree)
            prices = animate(fisier1)
            # canvas.update_idletasks
            canvas.draw()
            self.prices = prices
            self.model_predictor = Model_Predictor(prices, LoadedModel, graph, thread_session)

    def updateConfig(self, predictionLen_var, historyLen_var, train_size_var, epochs_var, modelSelected):
        try:
            int_val = predictionLen_var.get()
            config.set('section_data', 'int_val', int_val)

            train_size = train_size_var.get()
            config.set('section_data', 'training_size', train_size)

            loop_back = historyLen_var.get()
            if loop_back <=0:
                loop_back=1
            elif loop_back >120:
                loop_back = 120
            config.set('section_data', 'loop_back', str(loop_back))

            selected_model = modelSelected.get()
            config.set('section_data', 'model_name', str(selected_model))

            epochs = epochs_var.get()
            if epochs <=0:
                epochs=1
            config.set('section_data', 'num_epochs', str(epochs))

            with open('logs/Configuration.ini', 'w') as configfile:
                config.write(configfile)

            config.read('logs/Configuration.ini')
            self.model_predictor.loop_back = config.getint('section_data', 'loop_back')
            self.model_predictor.num_output = config.getint('section_data', 'num_output')
            self.model_predictor.prediction_len = config.getint('section_data', 'int_val')

            filepath = 'models/' + config.get('section_data', 'model_name')
            #self.model_predictor.model = load_model(filepath)
            self.model_predictor.model, self.model_predictor.graph, self.model_predictor.thread_session = load_model_mine()
            #self.model_predictor.model.summary()
            PageThree.reloadConfig(PageThree)
            tk.messagebox.showinfo("Showinfo", "Data saved successfully")
        except Exception as err:
            tk.messagebox.showerror("Error", "Cannot update configuration,  " + str(err))
            pass

    def quitMethod(self):
        if self.clock:
            self.clock.stop()
        self.root.withdraw()
        show_last_img = config.getboolean('section_data','show_last_img')
        if show_last_img:
            app = AppLicense(tk.Toplevel(self.root),'icons/graph.gif')
            self.root.update()
            self.root.after(4000, self.cmdClose)
        else:
            self.cmdClose()

    def cmdClose(self):
        self.root.quit()
        self.root.destroy()

    def NN_PredFrame(self, data_to_predict):
        rv, status, msg = None, False, ''
        try:
            NN = self.model_predictor.predict_dataset(scaler.fit_transform(data_to_predict), 1).reshape(-1,1)
            rv = scaler.inverse_transform(NN)
            status = True
        except Exception as err:
            status = False
            msg = str(err)
        return rv, status, msg

    def TestYourself(self, SMA_var,EMA_var,NN_var):
        if self.startAgentFlag==False:
            money = 0.0
            if config.getboolean('section_data', 'limit_money'):
                try:
                    answer = simpledialog.askfloat("Init", "Please specify initial cash", parent=self.root)
                    if answer is not None:
                        money = float(answer)
                except:
                    money = 0.0
                    pass
            else:
                 money=1.0

            if money>0.0:
                self.clock2 = Clock2(self.root, SMA_var,EMA_var,NN_var, self.prices, self.model_predictor,money)
                self.root.update()
                self.clock2.start()
                self.startAgentFlag = True
                btnPredict_1['text'] = 'Stop  agent  '
                btnPredict_1.configure(image=btnPredict_1.image2)
                btnBack_1['state'] = 'disabled'
                btnTest_['state'] = 'disabled'
                tabControl.tab(0, state="disabled")
                tabControl.tab(1, state="disabled")
                tabControl.tab(2, state="disabled")
        else:
            self.clock2.stop()
            self.clock2.C.place_forget()
            self.root.update()
            self.clock2 = None
            self.startAgentFlag = False
            btnPredict_1['text'] = 'Start agent  '
            btnPredict_1.configure(image = btnPredict_1.image)
            btnBack_1['state'] = 'normal'
            btnTest_['state'] = 'normal'
            tabControl.tab(0, state="normal")
            tabControl.tab(1, state="normal")
            tabControl.tab(2, state="normal")

    def ComputePredictions2(self, SMA_var,EMA_var,NN_var,df,dataset, out):
        try:
            window = config.getint('section_data', 'loop_back')
            if SMA_var == 1:
                SMA = df.iloc[:, 0].rolling(window=window).mean()
                out.put(SMA)
            if EMA_var == 1:
                EMA = df.iloc[:, 0].ewm(span=window, adjust=False).mean()
                out.put(EMA)
            if NN_var == 1:
                NN, status, msg = self.NN_PredFrame(dataset.reshape(-1, 1))
                out.put((NN, status))
        except Exception as e:
            if SMA_var == 1:
                out.put([])
            if EMA_var == 1:
                out.put([])
            if NN_var == 1:
                out.put(([], False))
            pass

    def generatePrediction(self,SMA_var,EMA_var,NN_var,df,dataset):
        try:
            self.my_queue = queue.Queue()
            self.update_Thread = threading.Thread(target=self.ComputePredictions2,
                                                  args=(SMA_var,EMA_var,NN_var,df,dataset, self.my_queue))
            self.update_Thread.setDaemon(True)
            self.update_Thread.start()
            self.wait_generate_prediction()
        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error in generatePrediction " + str(err))
            self.update_Thread.join()
            btnPredict_1['state'] = 'normal'
            btnBack_1['state'] = 'normal'
            btnTest_['state'] = 'normal'
            tabControl.tab(0, state="normal")
            tabControl.tab(1, state="normal")
            tabControl.tab(2, state="normal")
            toggle(WaitingWidget)
            pass

    def StartAgent(self, SMA_var,EMA_var,NN_var):
        btnPredict_1['state'] = 'disabled'
        btnBack_1['state'] = 'disabled'
        btnTest_['state'] = 'disabled'
        tabControl.tab(0, state="disabled")
        tabControl.tab(1, state="disabled")
        tabControl.tab(2, state="disabled")
        toggle(WaitingWidget)

        df = self.prices
        dataset = np.array(self.prices)
        self.generatePrediction(SMA_var,EMA_var,NN_var,df,dataset)

    def wait_generate_prediction(self):
        if self.update_Thread.isAlive():
            self.root.after(1000, self.wait_generate_prediction)
        else:
            self.update_Thread.join()
            f.clf()
            a = f.add_subplot(111)
            a.plot(scaler.inverse_transform(self.model_predictor.prices), c='tab:blue', label='Real data')
            legend_elements = [Line2D([0], [0], color='tab:blue', lw=1, label='Real data')]
            SMA_var, EMA_var, NN_var = var1_SMA.get(), var1_EMA.get(), var1_NN.get()

            try:
                if SMA_var == 1:
                    SMA = self.my_queue.get()
                    a.plot((SMA), c='orange')
                    legend_elements.append(
                        Line2D([0], [0], color='orange', label='SMA predict', markerfacecolor='orange'))
                if EMA_var == 1:
                    EMA = self.my_queue.get()
                    a.plot(EMA, c='red')
                    legend_elements.append(
                        Line2D([0], [0], color='tab:cyan', label='EMA predict', markerfacecolor='tab:cyan'))
                if NN_var == 1:
                    NN, status = self.my_queue.get()
                    if status:
                        if NN.shape[0] > 1:
                            a.plot(range(self.model_predictor.loop_back - 1, self.model_predictor.loop_back + len(NN) - 1),
                                NN,'tab:purple', label='NN predict')
                            legend_elements.append(
                                Line2D([0], [0], color='tab:purple', label='NN predict', markerfacecolor='tab:purple'))

                a.set_ylabel('Prices')
                a.set_xlabel('Data points')
                a.legend(handles=legend_elements, facecolor='white', framealpha=1)

                canvas.draw()
            except Exception as err:
                tk.messagebox.showerror("Error", "Unexpected error in wait_generate_prediction() " + str(err))
                pass
            a.set_ylabel('Prices')
            a.set_xlabel('Data points')
            a.legend(handles=legend_elements, facecolor='white', framealpha=1)
            canvas.draw()
            btnPredict_1['state'] = 'normal'
            btnBack_1['state'] = 'normal'
            btnTest_['state'] = 'normal'
            tabControl.tab(0, state="normal")
            tabControl.tab(1, state="normal")
            tabControl.tab(2, state="normal")
            toggle(WaitingWidget)

    def PredictAnalytic(self, SMA_,EMA_):
        f.clf()
        a = f.add_subplot(111)
        a.plot(scaler.inverse_transform(self.model_predictor.prices), c='tab:blue', label='Real data')
        legend_elements = [Line2D([0], [0], color='tab:blue', lw=1, label='Real data')]

        window = config.getint('section_data', 'loop_back')
        df = pd.DataFrame(self.prices)
        if SMA_==1:
            SMA = df.iloc[:, 0].rolling(window=window).mean()
            a.plot((SMA), c='orange')
            legend_elements.append(
                Line2D([0], [0], color='orange', label='SMA predict', markerfacecolor='orange'))

        if EMA_==1:
            EMA = df.iloc[:,0].ewm(span=window,adjust=False).mean()
            a.plot((EMA), c='red')
            legend_elements.append(
                Line2D([0], [0], color='red', label='EMA predict', markerfacecolor='red'))

        a.set_ylabel('Prices')
        a.set_xlabel('Data points')
        a.legend(handles=legend_elements, facecolor='white', framealpha=1)

        canvas.draw()

    def buy_stock_SA(self, window,  real, signal, init_money_=20000, buy_max=1, sell_max=1):

        init_money = init_money_
        starting_money = init_money_
        sells, buys = [],[]
        num_stocks = 0

        def buy(i, money, stocks):
            shares = float(money // real[i])
            if shares >= 1:
                if shares > buy_max:
                    units_to_buy = buy_max
                else:
                    units_to_buy = shares
                money -= units_to_buy*real[i]
                stocks += units_to_buy
                buys.append(i)
            return money, stocks

        for index in range(real.shape[0] - window):
            if signal[index] == 1:
                init_money, num_stocks = buy(index, init_money, num_stocks)
            elif signal[index] == -1:
                if num_stocks > 0:
                    if num_stocks > sell_max:
                        units_to_sell = sell_max
                    else:
                        units_to_sell = num_stocks

                    num_stocks -= units_to_sell
                    total_sell = float(units_to_sell*real[index])
                    init_money += total_sell
                    sells.append(index)

        gains = init_money - starting_money
        return buys, sells, gains, starting_money, num_stocks

    def buy_Stock_MA(self, window, real, signal, init_money_ = 20000, buy_max=1, sell_max=1):
        reversed = config.getboolean('section_data', 'reversed')
        init_money = init_money_
        starting_money = init_money_
        sells, buys = [], []
        num_stocks = 0

        def buy(index, init_money, num_stocks):
            shares = float(init_money // real[index])
            if shares > 0:
                if shares > buy_max:
                    buy_units = buy_max
                else:
                    buy_units = shares
                init_money -= float(buy_units * real[index])
                num_stocks += buy_units
                buys.append(index)
            return init_money, num_stocks

        for index in range(real.shape[0] - window):
            if reversed:
                if signal[index] == -1:
                    init_money, current_inventory = buy(index, init_money, num_stocks)
                    # buys.append(index)
                elif signal[index] == 1:
                    if current_inventory > 0:
                        if current_inventory > sell_max:
                            sell_units = sell_max
                        else:
                            sell_units = current_inventory
                        current_inventory -= sell_units
                        total_sell = float(sell_units * real[index])
                        init_money += total_sell

                        sells.append(index)
            else:
                if signal[index] == 1:
                    init_money, current_inventory = buy(index, init_money, num_stocks)
                    #buys.append(index)
                elif signal[index] == -1:
                    if current_inventory > 0:
                        if current_inventory > sell_max:
                            sell_units = sell_max
                        else:
                            sell_units = current_inventory
                        current_inventory -= sell_units
                        total_sell = float(sell_units * real[index])
                        init_money += total_sell

                        sells.append(index)

        gains = init_money - starting_money
        return buys, sells, gains, starting_money, num_stocks

    def buy_stock_SignalRolling(self, real, delay=500, initial_state=1, initial_money=20000,max_buy=1,max_sell=1,):
        reversed = config.getboolean('section_data', 'reversed')
        starting_money = initial_money
        delay_change_decision = delay
        current_decision = 0
        state = initial_state
        current_val = float(real[0])
        sells,buys = [],[]
        num_stocks = 0

        def buy(i, initial_money, num_stocks):
            shares = float(initial_money // real[i])
            if shares > 0:
                if shares > max_buy:
                    buy_units = max_buy
                else:
                    buy_units = shares
                initial_money -= float(buy_units * real[i])
                num_stocks += buy_units

            return initial_money, num_stocks

        if state == 1:
            initial_money, num_stocks = buy(0, initial_money, num_stocks)
            buys.append(0)

        for i in range(1, real.shape[0], 1):
            if reversed:
                if float(real[i]) > current_val and state == 0:
                    if current_decision < delay_change_decision:
                        current_decision += 1
                    else:
                        state = 1
                        initial_money, num_stocks = buy(i, initial_money, num_stocks)
                        current_decision = 0
                        buys.append(i)

                if float(real[i]) < current_val and state == 1:
                    if current_decision < delay_change_decision:
                        current_decision += 1
                    else:
                        state = 0
                        if num_stocks > 0:
                            if num_stocks > max_sell:
                                sell_units = max_sell
                            else:
                                sell_units = num_stocks

                            num_stocks -= sell_units
                            total_sell = float(sell_units * real[i])
                            initial_money += total_sell
                            sells.append(i)

                        current_decision = 0
            else:
                if float(real[i]) < current_val and state == 0:
                    if current_decision < delay_change_decision:
                        current_decision += 1
                    else:
                        state = 1
                        initial_money, num_stocks = buy(i, initial_money, num_stocks)
                        current_decision = 0
                        buys.append(i)

                if float(real[i]) > current_val and state == 1:
                    if current_decision < delay_change_decision:
                        current_decision += 1
                    else:
                        state = 0
                        if num_stocks > 0:
                            if num_stocks > max_sell:
                                sell_units = max_sell
                            else:
                                sell_units = num_stocks

                            num_stocks -= sell_units
                            total_sell = float(sell_units * real[i])
                            initial_money += total_sell
                            sells.append(i)

                        current_decision = 0

            current_val = float(real[i])
        total_gains = initial_money - starting_money
        return buys, sells, total_gains, starting_money, num_stocks

    def runAutonomousAgent(self, agent,var1_DisplayDetails):
        money = 0.0
        try:
            answer = simpledialog.askfloat("Init", "Please specify initial cash", parent=self.root)
            if answer is not None:
                money = float(answer)
        except:
            pass

        buy_max = config.getint('section_agents', 'buy_max')
        sell_max = config.getint('section_agents', 'sell_max')
        delay = config.getint('section_agents', 'delay')

        window = config.getint('section_data', 'loop_back')
        displayDetails = True if var1_DisplayDetails == 1 else False
        f.clf()
        a = f.add_subplot(111)
        a.plot(scaler.inverse_transform(self.model_predictor.prices), c='tab:blue', label='Real data')
        legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data')]
        if money > 0:
            if agent ==  "Simple average":
                reversed = config.getboolean('section_data', 'reversed')
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data'),
                                   Line2D([0], [0], marker='^', color='green', label='Buy',
                                          markerfacecolor='green', markersize=7),
                                   Line2D([0], [0], marker='v', color='r', label='Sell',
                                          markerfacecolor='r', markersize=7)
                                   ]
                signals = pd.DataFrame(index=self.prices.index)
                signals['signal'] = 0.0
                signals['trend'] = self.prices
                signals['RollingMax'] = (signals.trend.shift(1).rolling(window).max())
                signals['RollingMin'] = (signals.trend.shift(1).rolling(window).min())
                signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = 1 if reversed else -1
                signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = -1 if reversed else 1

                if displayDetails:
                    a.plot((signals['RollingMax']), c='orange')
                    legend_elements.append(Line2D([0], [0], color='orange', label='RollingMax', markerfacecolor='orange'))
                    a.plot((signals['RollingMin']), c='tab:cyan')
                    legend_elements.append(Line2D([0], [0], color='tab:cyan', label='RollingMin', markerfacecolor='tab:cyan'))

                buys, sells, gains, starting_money, num_stocks = self.buy_stock_SA(window, np.array(self.prices), signals['signal'],init_money_ = money, buy_max = buy_max, sell_max=sell_max)

                a.plot(self.prices, '^', markersize=7, color='green', label='buy', markevery=buys)
                a.plot(self.prices, 'v', markersize=7, color='r', label='sell', markevery=sells)
                f.suptitle('Init money: {}, Final money: {}, Total gains: {} Stocks: {}'.format(np.round(starting_money,2), np.round(starting_money+gains,2), np.round(gains), num_stocks), fontsize=10)

            elif agent == "Moving average":
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data'),
                                   Line2D([0], [0], marker='^', color='green', label='Buy',
                                          markerfacecolor='green', markersize=7),
                                   Line2D([0], [0], marker='v', color='r', label='Sell',
                                          markerfacecolor='r', markersize=7)
                                   ]
                short_window = window
                long_window = short_window*3
                signals = pd.DataFrame(index=self.prices.index)
                signals['signal'] = 0.0
                signals['short_ma'] = self.prices.rolling(window=short_window, min_periods=1, center=False).mean()
                signals['long_ma'] = self.prices.rolling(window=long_window, min_periods=1, center=False).mean()
                signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:]
                                                            > signals['long_ma'][short_window:], 1.0, 0.0)
                signals['positions'] = signals['signal'].diff()

                if displayDetails:
                    a.plot((signals['short_ma']), c='orange')
                    legend_elements.append(
                        Line2D([0], [0], color='orange', label='short MA', markerfacecolor='orange'))
                    a.plot((signals['long_ma']), c='tab:cyan')
                    legend_elements.append(
                        Line2D([0], [0], color='tab:cyan', label='long MA', markerfacecolor='tab:cyan'))

                buys, sells, gains, starting_money, num_stocks = self.buy_Stock_MA(window, np.array(self.prices),
                                                                                   signals['signal'], init_money_=money, buy_max = buy_max, sell_max=sell_max)

                a.plot(self.prices, '^', markersize=7, color='green', label='buy', markevery=buys)
                a.plot(self.prices, 'v', markersize=7, color='r', label='sell', markevery=sells)
                f.suptitle(
                    'Init money: {}, Final money: {}, Total gains: {} Stocks: {}'.format(np.round(starting_money, 2),
                                                                                         np.round(
                                                                                             starting_money + gains, 2),
                                                                                         np.round(gains), num_stocks),
                    fontsize=10)

            elif agent == "Signal rolling":
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data'),
                                   Line2D([0], [0], marker='^', color='green', label='Buy',
                                          markerfacecolor='green', markersize=7),
                                   Line2D([0], [0], marker='v', color='r', label='Sell',
                                          markerfacecolor='r', markersize=7)
                                   ]
                buys, sells, gains, starting_money, num_stocks = self.buy_stock_SignalRolling(np.array(self.prices),initial_state=1,
                                                                                              delay=delay, initial_money=money, max_buy = buy_max, max_sell = sell_max)
                a.plot(self.prices, '^', markersize=7, color='green', label='buy', markevery=buys)
                a.plot(self.prices, 'v', markersize=7, color='r', label='sell', markevery=sells)
                f.suptitle(
                    'Init money: {}, Final money: {}, Total gains: {} Stocks: {}'.format(np.round(starting_money, 2),
                                                                                         np.round(
                                                                                             starting_money + gains, 2),
                                                                                         np.round(gains), num_stocks),
                    fontsize=10)

            elif agent == "Policy gradient":
                print(agent)

            elif agent == "Q-learning":
                print(agent)
            elif agent == "Evolution strategy":
                print(agent)
            elif agent == "Recurrent Q-learning":
                print(agent)
            elif agent == "Evolve neural nets":
                print(agent)
        else:
            tk.messagebox.showerror("Error", "Not enough money {}".format(money))

        a.set_ylabel('Prices')
        a.set_xlabel('Data points')
        a.legend(handles=legend_elements, facecolor='white', framealpha=1)

        canvas.draw()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.config(background="white")

        self.photo = tk.PhotoImage(file='icons/hand.png')
        self.label = tk.Label(self, image=self.photo)
        self.label.image = self.photo
        self.label.place(x=0, y=0, relwidth=1, relheight=1)
        self.label.pack()

        buttonImage = Image.open('icons/upload.png')
        self.buttonPhoto = ImageTk.PhotoImage(buttonImage)
        btnCsv = ttk.Button(self, command=lambda: controller.callback_(), image=self.buttonPhoto, cursor="hand2")
        btnCsv.image = self.buttonPhoto
        btnCsv.pack(pady=10)
        button1_ttp = CreateToolTip(btnCsv, 'Choose .csv file for training or testing')

class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500
        self.wraplength = 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

class Gif(Label):
    def __init__(self, master, filename):
        evanGif = Image.open(filename)
        gifSeq = []
        try:
            while 1:
                gifSeq.append(evanGif.copy())
                evanGif.seek(len(gifSeq))  # skip to next frame
        except EOFError:
            pass  # we're done
        try:
            if evanGif.info['duration'] == 0:
                self.delay = 100
            else:
                self.delay = evanGif.info['duration']
        except KeyError:
            self.delay = 100
        gifFirst = gifSeq[0].convert('RGBA')
        self.gifFrames = [ImageTk.PhotoImage(gifFirst)]

        Label.__init__(self, master, image=self.gifFrames[0])
        temp = gifSeq[0]
        for image in gifSeq[1:]:
            temp.paste(image)
            frame = temp.convert('RGBA')
            self.gifFrames.append(ImageTk.PhotoImage(frame))

        self.gifIdx = 0
        self.cancel = self.after(self.delay, self.play)

    def play(self):
        self.config(image=self.gifFrames[self.gifIdx])
        self.gifIdx += 1
        if self.gifIdx == len(self.gifFrames):
            self.gifIdx = 0
        self.cancel = self.after(self.delay, self.play)

class App:
    def __init__(self, master):
        self.master = master
        self.loadingFrame = tk.Frame(self.master, background='light gray')
        self.loadingFrame.pack()
        self.anim = Gif(self.loadingFrame, 'icons/loading_1.gif')
        self.anim.pack()

        self.text = tk.StringVar()
        self.text.set("Welcome ...")
        tk.Label(self.master, textvariable=self.text, font="Helvetica 8 italic").pack(side=tk.LEFT)

        self.p = threading.Thread(target=self.hang)
        self.p.daemon = True
        self.p.start()

    def command(self):
        self.p.join()
        self.master.withdraw()
        toplevel = tk.Toplevel(self.master)
        self.app = SeaofBTCapp(toplevel)

    def hang(self):
        global LoadedModel, graph, thread_session
        try:
            self.text.set("Loading configurations ... ")
            config.read('logs/Configuration.ini')
            self.text.set("Loading model ... ")
            filepath = 'models/' + config.get('section_data', 'model_name')
            #LoadedModel = load_model(filepath)
            LoadedModel, graph, thread_session = load_model_mine()
            #LoadedModel.summary()
            time.sleep(1)
            self.text.set("")
            self.master.after(100, self.command)
        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error, cannot load init configuration, "+str(err))
            application.destroy()
            raise

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EarlyStoppingAtMinLoss, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if StopTran.get() == 0:
            self.model.stop_training = True
            logger.log(logging.ERROR, 'Forced training stop !!!')
        else:
            logger.log(logging.CRITICAL,
                       'Epoch: {}/{},  loss: {}, val_loss: {}'.format(epoch, config.getint('section_data', 'num_epochs'), round(logs['loss'], 4),
                                                                      round(logs['val_loss'], 4)))

    def on_train_end(self, logs=None):
        logger.log(logging.ERROR,'training finished')

def getArchitecture_Default():
    model = Sequential()
    model.add(LSTM(100, input_shape=(config.getint('section_data', 'loop_back'), config.getint('section_data', 'num_output'))))
    model.add(Dropout(0.4))
    model.add(Dense(config.getint('section_data', 'num_output')))
    model.compile(loss='mse', optimizer='adam')
    return model

class Clock(threading.Thread):
    def __init__(self, modelClass, modelNewName):
        super().__init__()
        self._stop_event = threading.Event()
        self.modelClass = modelClass
        self.Flag = True
        self.modelNewName = modelNewName

    def run(self):
        if not self._stop_event.is_set():
            logger.debug('Clock started')
            logger.log(logging.ERROR, '*************_Start train new model_*************')
            try:
                save_op_callback = EarlyStoppingAtMinLoss()
                batch_size, num_epochs, patience_epochs, _model, X_train, Y_train, X_test, Y_test = self.modelClass.TrainModel(
                        Compiled=True)
                #_model, graph, thread_session = load_model_mine()
                thread_graph = Graph()
                with thread_graph.as_default():
                    thread_session = Session()
                    with thread_session.as_default():
                        model = getArchitecture_Default()
                        model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size,
                                  validation_data=(X_test, Y_test),
                                  callbacks=[
                                      EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience_epochs,
                                                    verbose=2),
                                      save_op_callback
                                  ],
                                  verbose=2,
                                  shuffle=False)

                f.set_visible(True)
                if StopTran.get() == 0:  # force stopped
                    tk.messagebox.showinfo("Showinfo", "Model training stopped")
                else:  # finished
                    with graph.as_default():
                        with thread_session.as_default():
                            model.save(self.modelNewName)
                    tk.messagebox.showinfo("Showinfo",
                                           "Training finished, \n Model " + str(self.modelNewName) + ' was saved')
                    btnPredict['state'] = 'normal'
                    btnTrain['state'] = 'normal'
                    btnUpdate['state'] = 'normal'
                    btnBack['state'] = 'normal'
                    btnTest['state'] = 'normal'
                    toggle(WaitingWidget, False, True)
                    toggle(btnCancel, False, False, True)
                    toggle(trainTerminal, False, False)
                    self._stop_event.set()
                    PageThree.reloadConfig(PageThree)
            except Exception as err:
                tk.messagebox.showinfo("Showinfo", "Wrong configuration, please change the train size, error in Clock function, "+str(err))
                f.set_visible(True)
                StopTran.set(0)
                btnPredict['state'] = 'normal'
                btnTrain['state'] = 'normal'
                btnUpdate['state'] = 'normal'
                btnBack['state'] = 'normal'
                btnTest['state'] = 'normal'
                toggle(WaitingWidget, False, True)
                toggle(btnCancel, False, False, True)
                toggle(trainTerminal, False, False)
                self._stop_event.set()
        else:
            StopTran.set(0)

    def stop(self):
        f.set_visible(True)
        StopTran.set(0)
        self._stop_event.set()

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class ConsoleUi:
    def __init__(self, frame):
        self.frame = frame
        self.scrolled_text = scrolledtext.ScrolledText(frame, state='disabled')
        self.scrolled_text.pack(expand=True, fill=tk.BOTH)
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('ERROR', foreground='red', underline=1)
        self.scrolled_text.tag_config('CRITICAL', foreground='black', )

        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)

        self.frame.after(500, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.frame.after(500, self.poll_log_queue)

class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.config(background="white")
        self.controller = controller
        global tabControl
        tabControl = ttk.Notebook(self)
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)
        tab5 = ttk.Frame(tabControl)
        allAgents =ttk.Frame(tabControl)

        tabControl.add(tab2, text='Configurations ')
        tabControl.add(tab1, text='Neural Network')
        tabControl.add(tab4, text='Analytic methods')
        tabControl.add(tab5, text='Autonomous agent')
        tabControl.add(allAgents, text='All agents')

        tabControl.pack(side=tk.TOP, fill="both")

        photo1 = tk.PhotoImage(file="icons/goback.png")
        btnBackAgent = tk.Button(allAgents, width=45, height=45, image=photo1, command=lambda: controller.show_frame(StartPage),
                              cursor='hand2', bg='white', fg='black')
        btnBackAgent.grid(row=0, column=0, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        #btnBack_1.pack(side=tk.LEFT, padx=2, pady=2)
        btnBackAgent.image = photo1
        btnBackAgentTool = CreateToolTip(btnBackAgent, 'Go back')

        labelTop = tk.Label(allAgents,text="Choose your favourite agent")
        labelTop.grid(column=2, row=0)
        fontExample = ("Courier", 12, "bold")
        comboExample = ttk.Combobox(allAgents,
                                    values=[
                                        "Simple average",
                                        "Moving average",
                                        "Signal rolling",
                                        "Policy gradient",
                                        "Q-learning",
                                        "Evolution strategy",
                                        "Recurrent Q-learning",
                                        "Evolve neural nets"
                                    ],
                                    font=fontExample)

        allAgents.option_add('*TCombobox*Listbox.font', fontExample)
        comboExample.grid(column=2, row=1)

        var1_DisplayDetails = tk.IntVar()
        var1_DisplayDetails.set(0)
        DisplayDetails = tk.Checkbutton(allAgents, text="Display details", variable=var1_DisplayDetails)
        DisplayDetailsTool = CreateToolTip(DisplayDetails, 'Used to display extra details of agent')
        DisplayDetails.grid(row=0, column=8, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)

        photo1 = tk.PhotoImage(file="icons/predict.png")
        btnStarAllAgents = tk.Button(allAgents, compound=tk.RIGHT, width=220, height=30, image=photo1, cursor='hand2',
                                 command=lambda: controller.runAutonomousAgent(comboExample.get(),var1_DisplayDetails.get()),
                                 text="Start agent  ", bg='white', fg='black')
        btnStarAllAgents.grid(row=0, column=5, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        btnStarAllAgents.image = photo1
        btnStarAllAgents.image2 = tk.PhotoImage(file="icons/Stop.png")
        myFont = font.Font(family='Helvetica', weight="bold", size=14)
        btnStarAllAgents['font'] = myFont
        btnStarAllAgentsTool = CreateToolTip(btnStarAllAgents, 'Start autonomous agent')

        global btnBack_1,btnPredict_1,btnTest_
        photo1 = tk.PhotoImage(file="icons/goback.png")
        btnBack_1 = tk.Button(tab5, width=45, height=45, image=photo1, command=lambda: controller.show_frame(StartPage),
                             cursor='hand2', bg='white', fg='black')
        #btnBack_1.grid(row=0, column=0, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        btnBack_1.pack(side=tk.LEFT, padx=2, pady=2)

        btnBack_1.image = photo1
        btnBackTool = CreateToolTip(btnBack_1, 'Go back')

        global var1_SMA, var1_EMA, var1_NN
        var1_SMA = tk.IntVar()
        var1_SMA.set(1)
        selfSMA = tk.Checkbutton(tab5, text="SMA", variable=var1_SMA)
        selfselfSMA = CreateToolTip(selfSMA, 'Use Simple Moving Average')
        selfSMA.pack(side=tk.LEFT)

        var1_EMA = tk.IntVar()
        selfEMA = tk.Checkbutton(tab5, text="EMA", variable=var1_EMA)
        selfselfEMA = CreateToolTip(selfEMA, 'Use Exponential Moving Average')
        selfEMA.pack(side=tk.LEFT)

        var1_NN = tk.IntVar()
        selfNN = tk.Checkbutton(tab5, text="NN", variable=var1_NN)
        selfselfNN = CreateToolTip(selfNN, 'Use Neural Network model')
        selfNN.pack(side=tk.LEFT)

        photo1 = tk.PhotoImage(file="icons/predict.png")
        btnPredict_1 = tk.Button(tab5, compound=tk.RIGHT, width=120, height=30, image=photo1, cursor='hand2',
                                command=lambda: controller.TestYourself(var1_SMA.get(), var1_EMA.get(), var1_NN.get()),
                                text="Start agent  ", bg='white', fg='black')
        #btnPredict_1.grid(row=0, column=5, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        btnPredict_1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2, pady=2)
        btnPredict_1.image = photo1
        btnPredict_1.image2 = tk.PhotoImage(file="icons/Stop.png")
        myFont = font.Font(family='Helvetica', weight="bold", size=14)
        btnPredict_1['font'] = myFont
        btnPredictTool = CreateToolTip(btnPredict_1, 'Start autonomous agent')

        #photo1 = tk.PhotoImage(file="icons/train.png")
        img = Image.open("icons/train.png")
        img = img.resize((32, 32), Image.ANTIALIAS)
        photo1 = ImageTk.PhotoImage(img)
        btnTest_ = tk.Button(tab5, compound=tk.RIGHT, width=120, height=30, image=photo1, cursor='hand2',
                             command=lambda: controller.StartAgent(var1_SMA.get(), var1_EMA.get(), var1_NN.get()),

                             text="Test yourself  ", bg='white', fg='black')
        btnTest_.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.BOTH)
        btnTest_.image = photo1
        btnTest_['font'] = myFont
        btnTestTool = CreateToolTip(btnTest_,
                                    'Allows you to learn how to use stocks price indicators and Neural Network predictions')

        photo1 = tk.PhotoImage(file="icons/goback.png")
        btnBack_ = tk.Button(tab4, width=45, height=45, image=photo1, command=lambda: controller.show_frame(StartPage),
                            cursor='hand2', bg='white', fg='black')
        #btnBack_.grid(row=0, column=0, columnspan=2, rowspan=2,sticky=tk.W + tk.E + tk.N + tk.S, padx=2,pady=2)
        btnBack_.pack(side=tk.LEFT, padx=2,pady=2)
        btnBack_.image = photo1
        btnBackTool = CreateToolTip(btnBack_, 'Go back')

        panel1_ = ttk.PanedWindow(tab4, orient=tk.HORIZONTAL)
        panel1_.pack(side=tk.LEFT, padx=2, pady=0)

        console_frame_ = ttk.Labelframe(panel1_)
        panel1_.add(console_frame_)

        var1_selfData_ = tk.IntVar()
        selfPredCkb_ = tk.Checkbutton(tab4, text="SMA", variable=var1_selfData_)
        selfPredCkb_update = CreateToolTip(selfPredCkb_, 'Simple Moving Average')
        #selfPredCkb_.grid(row=0, column=3)
        selfPredCkb_.pack(side = tk.LEFT)

        var2_multiple_points_ = tk.IntVar()
        multipleCkb_ = tk.Checkbutton(tab4, text="EMA", variable=var2_multiple_points_)
        multipleCkb_update = CreateToolTip(multipleCkb_, 'Exponential Moving Average')
        #multipleCkb_.grid(row=1, column=3)
        multipleCkb_.pack(side = tk.LEFT)

        photo1 = tk.PhotoImage(file="icons/predict.png")
        btnPredict_ = tk.Button(tab4, compound=tk.RIGHT, width=120, height=30, image=photo1, cursor='hand2',
                               command=lambda: controller.PredictAnalytic(var1_selfData_.get(),var2_multiple_points_.get()),
                               text="Predict  ", bg='white', fg='black')
        #btnPredict_.grid(row=0, column=5, columnspan=2, rowspan=2,sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        btnPredict_.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2, pady=2)
        btnPredict_.image = photo1
        myFont = font.Font(family='Helvetica', weight="bold", size=14)
        btnPredict_['font'] = myFont
        btnPredictTool = CreateToolTip(btnPredict_, 'Model prediction')

        global btnPredict, btnTrain, btnUpdate, btnBack, btnTest

        photo1 = tk.PhotoImage(file="icons/goback.png")
        btnBack_1_ = tk.Button(tab2, width=45, height=45, image=photo1, command=lambda: controller.show_frame(StartPage),
                              cursor='hand2', bg='white', fg='black')
        btnBack_1_.grid(row=0, column=0, columnspan=2, rowspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=2, pady=2)
        btnBack_1_.image = photo1
        btnBackTool = CreateToolTip(btnBack_1_, 'Go back')

        lblTrainSize = tk.Label(tab2, text="Train size: ", font='Helvetica 10 bold')
        lblTrainSize.grid(row=0, column=2)
        epochTool = CreateToolTip(lblTrainSize, 'Split data between train and test')

        train_size_options = ["0.50", "0.60", "0.70", "0.80", "0.90"]
        train_size_var = tk.StringVar(tab2)
        train_size_var.set(str(config.get('section_data', 'training_size')))
        train_sizeOM = tk.OptionMenu(tab2, train_size_var, *train_size_options)
        train_sizeOM.grid(row=0, column=3)
        epochTool = CreateToolTip(train_sizeOM, 'Split data between train and test')

        lblEpochs = tk.Label(tab2, text="Epochs: ", font='Helvetica 10 bold')
        lblEpochs.grid(row=1, column=2)
        epochTool = CreateToolTip(lblEpochs, 'Number of epochs, used for training the model')

        epochs_var = tk.IntVar()
        epochs_var.set(config.getint('section_data', 'num_epochs'))
        epochsSpin = tk.Spinbox(tab2, from_=1, to=120, width=5, textvariable=epochs_var)
        epochsSpin.grid(row=1, column=3)
        epochTool = CreateToolTip(epochsSpin, 'Number of epochs, used for training the model')

        lblPredictionLen = tk.Label(tab2, text="Prediction length: ", font='Helvetica 10 bold')
        lblPredictionLen.grid(row=0, column=4)
        predLenTool = CreateToolTip(lblPredictionLen, 'Number of predicted future points')

        prediction_options = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "25", "30"]
        predictionLen_var = tk.StringVar(tab2)
        predictionLen_var.set(str(config.getint('section_data', 'int_val')))
        predictionOM = tk.OptionMenu(tab2, predictionLen_var, *prediction_options)
        predictionOM.grid(row=0, column=5)
        predLenTool = CreateToolTip(predictionOM, 'Number of predicted future points')

        lblHistoryLen = tk.Label(tab2, text="History length: ", font='Helvetica 10 bold')
        lblHistoryLen.grid(row=1, column=4)
        historyTool = CreateToolTip(lblHistoryLen,
                                    'Number of past points, used by model to predict future points (depends on your model)')

        historyLen_var = tk.IntVar()
        historyLen_var.set(config.getint('section_data', 'loop_back'))
        historySpin = tk.Spinbox(tab2, from_=1, to=150, width=5, textvariable=historyLen_var)
        historySpin.grid(row=1, column=5)
        historyTool = CreateToolTip(historySpin,
                                    'Number of past points, used by model to predict future points (depends on your model)')

        panel1 = ttk.PanedWindow(tab2, orient=tk.VERTICAL)
        panel1.grid(row=0, column=10, padx=5, rowspan=2)
        console_frame = ttk.Labelframe(panel1, text="Models")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        panel1.add(console_frame, weight=1)

        global models_options,modelOM,chosedModel_var
        models_options = listdir('./models')
        chosedModel_var = tk.StringVar(tab2)
        chosedModel_var.set(config.get('section_data', 'model_name'))
        modelOM = tk.OptionMenu(console_frame, chosedModel_var, *models_options)
        modelOM.pack(expand=True, fill=tk.BOTH)
        modelOMTool = CreateToolTip(modelOM, 'Choose the model for testing and prediction')

        photo1 = tk.PhotoImage(file="icons/refresh.png")
        btnUpdate = tk.Button(tab2, compound=tk.RIGHT, width=250, height=25, image=photo1, cursor='exchange',
                              command=lambda: controller.updateConfig(predictionLen_var, historyLen_var, train_size_var,
                                                                      epochs_var, chosedModel_var),
                              text="Update configurations  ", bg='white', fg='black')
        btnUpdate.grid(row=0, column=12, columnspan=2, rowspan=2,sticky=tk.W + tk.E + tk.N + tk.S, padx=5)
        btnUpdate.image = photo1
        myFont = font.Font(family='Helvetica', weight="bold", size=14)
        btnUpdate['font'] = myFont
        btnUpdateTool = CreateToolTip(btnUpdate, 'Update configuration')

        global var1_selfData, var2_multiple_points

        var1_selfData = tk.IntVar()
        var1_selfData.set(1)
        selfPredCkb = tk.Checkbutton(tab2, text="Self      predict", variable=var1_selfData)
        selfPredCkb_update = CreateToolTip(selfPredCkb, 'Display model prediction on test data')
        selfPredCkb.grid(row=0, column=6)

        var2_multiple_points = tk.IntVar()
        multipleCkb = tk.Checkbutton(tab2, text="Future predict", variable=var2_multiple_points)
        multipleCkb_update = CreateToolTip(multipleCkb, 'Display future model predicts')
        multipleCkb.grid(row=1, column=6)

        photo1 = tk.PhotoImage(file="icons/goback.png")
        btnBack = tk.Button(tab1, width=45, height=45, image=photo1, command=lambda: controller.show_frame(StartPage),cursor='hand2',
                            text="Predict  ", bg='white', fg='black')
        btnBack.pack(side=tk.LEFT, padx=2, pady=3)
        btnBack.image = photo1
        btnBackTool = CreateToolTip(btnBack, 'Go back')
        # -------------------------------------------------------------------
        photo1 = tk.PhotoImage(file="icons/predict.png")
        btnPredict = tk.Button(tab1, compound=tk.RIGHT, width=120, height=30, image=photo1,cursor='hand2',
                               command=lambda: controller.PredictStock(),
                               text="Predict  ", bg='white', fg='black')
        btnPredict.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.BOTH)
        btnPredict.image = photo1
        myFont = font.Font(family='Helvetica', weight="bold", size=14)
        btnPredict['font'] = myFont
        btnPredictTool = CreateToolTip(btnPredict, 'Model prediction')

        photo1 = tk.PhotoImage(file="icons/sand.png")
        btnTrain = tk.Button(tab1, compound=tk.RIGHT, width=120, height=30, image=photo1,cursor='hand2',
                             command=lambda: controller.TrainModel(),
                             text="Train  ", bg='white', fg='black')
        btnTrain.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.BOTH)
        btnTrain.image = photo1
        btnTrain['font'] = myFont
        btnTrain['font'] = myFont
        btnPredictTool = CreateToolTip(btnTrain, 'Train the model')

        img = Image.open("icons/train.png")
        img = img.resize((32, 32), Image.ANTIALIAS)
        photo1 = ImageTk.PhotoImage(img)
        btnTest = tk.Button(tab1, compound=tk.RIGHT, width=120, height=30, image=photo1, cursor='hand2',
                             command=lambda: controller.TestModel(),
                             text="Test model  ", bg='white', fg='black')
        btnTest.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.BOTH)
        btnTest.image = photo1
        btnTest['font'] = myFont
        btnTestTool = CreateToolTip(btnTest, 'Test model on current data')

        self.addCanvas()

    def addCanvas(self):
        global canvas, WaitingWidget, trainTerminal, btnCancel
        canvas = FigureCanvasTkAgg(f, self)
        canvas.get_tk_widget().pack(expand=True, fill="both")

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        WaitingWidget = Gif(canvas._tkcanvas, 'icons/loading_1.gif')
        WaitingWidget.visible = True
        WaitingWidget.pack(pady=10)
        toggle(WaitingWidget)

        photo1 = tk.PhotoImage(file="icons/Stop.png")
        btnCancel = tk.Button(canvas._tkcanvas, compound=tk.RIGHT, image=photo1,cursor='pirate',
                              command=lambda: self.controller.CancelTraining(),
                              text="Cancel  ", bg='white', fg='black')
        btnCancel.pack(side=tk.BOTTOM, padx=50, pady=0, expand=True, fill=tk.BOTH)
        btnCancel.image = photo1
        btnCancel['font'] = font.Font(family='Helvetica', weight="bold", size=14)
        btnCancelTool = CreateToolTip(btnCancel, 'Stop training, all training data will be lost')
        btnCancel.visible = True
        toggle(btnCancel, False, False)

        trainTerminal = ttk.PanedWindow(canvas._tkcanvas, orient=tk.VERTICAL)
        trainTerminal.visible = True
        trainTerminal.pack(side=tk.TOP, padx=50, pady=0, expand=True, fill=tk.BOTH)
        console_frame = ttk.Labelframe(trainTerminal, text="Training details console")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        trainTerminal.add(console_frame, weight=1)
        self.console = ConsoleUi(console_frame)
        toggle(trainTerminal, False, False)

    def reloadConfig(self):
        models_options = listdir('./models')
        modelOM['menu'].delete(0, 'end')
        for choice in models_options:
            modelOM['menu'].add_command(label=choice, command=tk._setit(chosedModel_var, choice))

class Clock2(threading.Thread):
    def __init__(self, master,SMA_var,EMA_var,NN_var,prices,model_predictor,cash):
        super().__init__()
        self.master = master
        self.setDaemon(True)
        self._stop_event = threading.Event()
        self.fig = Figure()
        self.axe = self.fig.add_subplot(111)

        self.graph = FigureCanvasAgg(self.fig)
        self.C = tk.Canvas(master, bd=0, width=-10, height=-10,
                           highlightthickness=0, bg='white')

        self.C.place(x=1, y=85, relwidth=1, relheight=0.9, width=-10, height=-10)
        self.C.bind("<Configure>", self.on_resize)
        self.SMA_var = SMA_var
        self.EMA_var = EMA_var
        self.NN_var = NN_var
        self.prices = prices
        self.model_predictor = model_predictor
        self.cash = cash

    def on_resize(self,event):
        if not self._stop_event.is_set():
            if event.height >25 and event.width >25:
                value_height = float(event.height - 25) * 0.0104166667
                value_width = float(event.width - 25) *0.0104166667
                self.fig.set_figheight(value_height)
                self.fig.set_figwidth(value_width)

    def run(self):
        try:
            stocks = 0
            self.fig.clf()
            self.axe = self.fig.add_subplot(111)
            reversed = config.getboolean('section_data', 'reversed')
            threshold_of_percentage = config.getfloat('section_data', 'threshold_of_percentage')
            use_percentage_intersection = config.getboolean('section_data', 'use_percentage_intersection')
            limit_money = config.getboolean('section_data', 'limit_money')
            step = config.getint('section_data', 'int_val')
            pause = config.getfloat('section_data', 'tick_for_autonomous')
            precision = 4
            window = config.getint('section_data', 'loop_back')
            legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, label='Real data'),
                               Line2D([0], [0], marker='o', color='g', label='Buy',
                                      markerfacecolor='g', markersize=7),
                               Line2D([0], [0], marker='o', color='r', label='Sell',
                                      markerfacecolor='r', markersize=7)
                               ]
            if self.SMA_var == 1:
                legend_elements.append(
                Line2D([0], [0], color='orange', label='SMA predict', markerfacecolor='orange'))
            if self.EMA_var==1:
                legend_elements.append(
                Line2D([0], [0], color='tab:cyan', label='EMA predict', markerfacecolor='tab:cyan'))
            if self.NN_var==1:
                legend_elements.append(
                Line2D([0], [0], color='tab:purple', label='NN predict', markerfacecolor='tab:purple'))

            self.axe.set_ylabel('Prices')
            self.axe.set_xlabel('Data points')
            self.axe.legend(handles=legend_elements, facecolor='white', framealpha=1)
            dataset = np.array(self.prices)[config.getint('section_data', 'points_for_AUS'):, ]
            df_ge = pd.DataFrame(dataset)
            rl, ln, sm, em = None, None, None, None

            percentageG = []
            NN_Out = []
            for it in range(window,len(dataset)+step, step):
                if self._stop_event.is_set():
                    break
                y = dataset[0:it,:]
                if rl:
                    self.axe.lines.remove(rl)
                rl, = self.axe.plot(y, 'tab:blue', label="Real data")
                real = np.round(y[-1], precision)

                pred, k = 0, 0
                if self.SMA_var==1:
                    SMA = np.array(df_ge.iloc[:it, 0].rolling(window=window).mean())
                    if sm:
                        self.axe.lines.remove(sm)
                    sm, = self.axe.plot(SMA, 'orange', label="SMA")
                    pred_SMA = np.round(SMA[-1], precision)
                    pred += pred_SMA
                    k += 1
                if self.EMA_var==1:
                    EMA = np.array(df_ge.iloc[:it, 0].ewm(span=window, adjust=False).mean())
                    if em:
                        self.axe.lines.remove(em)
                    em, = self.axe.plot(EMA, 'tab:cyan', label='EMA')
                    pred_EMA = np.round(EMA[-1], precision)
                    pred += pred_EMA
                    k += 1
                if self.NN_var==1:
                    if it > window+step:
                        batch = dataset[it-window-step+1:it,:]
                        NN, status, msg = self.NN_PredFrame(np.array(batch).reshape(-1,1))
                        if status:
                            if len(NN_Out) > 1:
                                NN_Out = np.concatenate((NN_Out, NN)) if step is not 1 else np.append(NN_Out, NN)
                                if ln:
                                    self.axe.lines.remove(ln)
                                ln, = self.axe.plot(
                                    range(self.model_predictor.loop_back + step, self.model_predictor.loop_back + NN_Out.shape[0] +step),
                                    NN_Out, 'tab:purple', label='NN')
                                pred_NN = np.round(NN[-1], precision)
                                pred += pred_NN
                                k += 1
                            else:
                                NN_Out = NN if step is not 1 else np.append(NN_Out, NN)
                if k>0:
                    pred = np.round((float(pred)/float(k)),precision)

                    if limit_money:
                        if reversed :
                            if real >= pred and stocks > 0:  # price grows => buy
                                self.axe.scatter(it - 1, y[-1], c='green', label='Buy')
                                stocks += 1
                                self.cash -= real
                            elif real < pred and self.cash >= real:  # price decresases => sell
                                self.axe.scatter(it - 1, y[-1], c='red', label='Sell')
                                stocks -= 1
                                self.cash += real
                        else:
                            if real >= pred and stocks > 0:  # price grows => buy
                                self.axe.scatter(it - 1, y[-1], c='red', label='Sell')
                                stocks -= 1
                                self.cash += real
                            elif real < pred and self.cash >= real:  # price decresases => sell
                                self.axe.scatter(it - 1, y[-1], c='green', label='Buy')
                                stocks += 1
                                self.cash -= real
                    else:
                        if reversed:
                            if real >= pred:  # price grows => buy
                                self.axe.scatter(it - 1, y[-1], c='green', label='Buy')
                            elif real < pred:  # price decresases => sell
                                self.axe.scatter(it - 1, y[-1], c='red', label='Sell')
                        else:
                            if real >= pred: #price grows => buy
                                self.axe.scatter(it-1, y[-1], c='red', label='Sell')
                            elif real < pred: #price decresases => sell
                                self.axe.scatter(it-1, y[-1], c='green', label='Buy')

                self.graph.draw()
                plt.pause(pause)
                s, (width, height) = self.graph.print_to_buffer()
                X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
                img = ImageTk.PhotoImage(image=Image.fromarray(X))
                self.C.create_image(0, 0, anchor="nw", image=img)

            if limit_money:
                tk.messagebox.showinfo("Agent result", "Cash: {},  Stocks: {}".format(self.cash,stocks))
            else:
                tk.messagebox.showinfo("Agent result", "Agent testing done")

        except Exception as err:
            tk.messagebox.showerror("Error", "Unexpected error in thread 2 " + str(err))
            self.stop()
            self.C.place_forget()
            btnPredict_1['text'] = 'Start agent  '
            btnPredict_1.configure(image=btnPredict_1.image)
            btnBack_1['state'] = 'normal'
            btnTest_['state'] = 'normal'
            tabControl.tab(0, state="normal")
            tabControl.tab(1, state="normal")
            tabControl.tab(2, state="normal")

    def stop(self):
        self._stop_event.set()

    def NN_PredFrame(self, data_to_predict):
        rv, status, msg = None, False, ''
        try:
            NN = self.model_predictor.predict_dataset(scaler.fit_transform(data_to_predict), 1).reshape(-1,1)
            rv = scaler.inverse_transform(NN)
            status = True
        except Exception as err:
            status = False
            msg = str(err)
        return rv, status, msg

class AppWithThread(threading.Thread):
    def __init__(self, root, txtmessage=''):
        root = application
        threading.Thread.__init__(self)
        self.master = tk.Toplevel(application)  # application# tk.Toplevel(root)
        self.parentRoot = root
        self.txtmessage = txtmessage if txtmessage else "Data processing ..."
        self.setDaemon(True)
        self.start()

    def close(self):
        # self.parentRoot.wm_attributes("-disabled", False)
        self.join()
        self.master.destroy()

    def run(self):
        # self.parentRoot.wm_attributes("-disabled", True)
        self.loadingFrame = tk.Frame(self.master, background='light gray')
        self.loadingFrame.pack()
        self.anim = Gif(self.loadingFrame, 'icons/loading_1.gif')
        self.anim.pack()
        self.text = tk.StringVar()
        self.text.set(self.txtmessage)
        tk.Label(self.master, textvariable=self.text, font="Helvetica 8 italic").pack(side=tk.LEFT)
        # center_tk_window.center_on_parent(self.parentRoot, self.master)
        self.master.wm_attributes("-topmost", True)
        self.master.overrideredirect(True)

class AppLicense:
    def __init__(self, master, img, Is_close=True):
        self.img = img
        self.master = master
        self.master["bg"] = "white"
        if not Is_close:
            self.text = tk.StringVar()
            self.text.set("We're so sorry, Your licence expired ...")
            self.lbl = tk.Label(self.master, textvariable=self.text, font="Helvetica 12 italic")
            self.lbl["bg"] = "white"
            self.lbl.pack(side=tk.TOP)

            photo1 = tk.PhotoImage(file="icons/close.png")
            btnClose = tk.Button(self.master, compound=tk.RIGHT, width=60, height=25, image=photo1, cursor='hand2',
                            command = self.quitM,
                            text="Close  ", bg='white', fg='black')
            btnClose.pack(side=tk.BOTTOM, padx=2, pady=2, expand=True, fill=tk.BOTH)
            btnClose.image = photo1
            btnClose['font'] = font.Font(family='Helvetica', weight="bold", size=12)
            btnClosetTool = CreateToolTip(btnClose, 'Close the window')

        self.p = threading.Thread(target=self.hang)
        self.p.daemon = True
        self.p.start()

    def hang(self):
        self.loadingFrame = tk.Frame(self.master, background='white')
        self.loadingFrame.pack()
        self.loadingFrame["bg"] = "white"
        self.anim = Gif(self.loadingFrame, self.img)
        self.anim["bg"] = "white"
        self.anim.pack()
        center_tk_window.center_on_screen(self.master)
        self.master.wm_attributes("-topmost", True)
        self.master.overrideredirect(True)

    def quitM(self):
        self.master.quit()
        self.master.destroy()

TooLate = False
present = datetime.now()
finalDate = datetime(2025, 4, 19)
if present >= finalDate:
    TooLate = True

global application
application = tk.Tk()
application.eval('tk::PlaceWindow . center')
application.overrideredirect(True)
application.iconbitmap(application, default="icons/my.ico")
application.attributes('-topmost', True)
application.wm_title("Stocks predictor")

if TooLate:
    application.configure(background='white')
    app = AppLicense(application,'icons/sadness.png', Is_close=False)
else:
    app = App(application)

StopTran = tk.IntVar()
StopTran.set(1)
application.mainloop()
