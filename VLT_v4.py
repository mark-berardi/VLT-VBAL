#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on July 28, 2026, at 18:15
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_ready
import time
import csv
import os
import numpy as np
from datetime import date
import sounddevice as sd


# Run 'Before Experiment' code from code_Noise
import soundfile as sf

# Run 'Before Experiment' code from code_10
TCTOpacity_SpeechIntro = 0
# Run 'Before Experiment' code from code_VLT_init
pagelen = len(data.importConditions('PsychoPy/Pics_all.xlsx'))
pages = list(range(0,pagelen))

# Run 'Before Experiment' code from STIM_code
import pandas as pd

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'VLT_v4'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 960]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\markberardi\\OneDrive - University of Iowa\\Documents\\GitHub\\VLT-VBAL\\VLT_v4.py',
        savePickle=True, saveWideText=False,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_ready') is None:
        # initialise key_resp_ready
        key_resp_ready = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_ready',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_Intro') is None:
        # initialise key_resp_Intro
        key_resp_Intro = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_Intro',
        )
    if deviceManager.getDevice('key_resp_StimIntro1') is None:
        # initialise key_resp_StimIntro1
        key_resp_StimIntro1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_StimIntro1',
        )
    if deviceManager.getDevice('key_resp_Practice') is None:
        # initialise key_resp_Practice
        key_resp_Practice = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_Practice',
        )
    if deviceManager.getDevice('key_resp_StimIntro2') is None:
        # initialise key_resp_StimIntro2
        key_resp_StimIntro2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_StimIntro2',
        )
    if deviceManager.getDevice('VER_key_resp_2') is None:
        # initialise VER_key_resp_2
        VER_key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='VER_key_resp_2',
        )
    if deviceManager.getDevice('VER_key_resp_3') is None:
        # initialise VER_key_resp_3
        VER_key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='VER_key_resp_3',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('VER_key_resp') is None:
        # initialise VER_key_resp
        VER_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='VER_key_resp',
        )
    if deviceManager.getDevice('key_resp_Begin') is None:
        # initialise key_resp_Begin
        key_resp_Begin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_Begin',
        )
    # create speaker 'sound_SpeechIntro'
    deviceManager.addDevice(
        deviceName='sound_SpeechIntro',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample=True,
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('STIM_key_resp') is None:
        # initialise STIM_key_resp
        STIM_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='STIM_key_resp',
        )
    if deviceManager.getDevice('key_resp_End') is None:
        # initialise key_resp_End
        key_resp_End = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_End',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Ready" ---
    # Set experiment start values for variable component Vol
    Vol = 0.62
    VolContainer = []
    # Run 'Begin Experiment' code from code_ready
    fs = 44100
    times = np.linspace(0, 0.1, fs, endpoint=True)
    beep = 0.7*np.sin(2 * np.pi * 440 * times)
    
    today = date.today()
    filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expName, expInfo['participant'],today.strftime('%Y-%m-%d'))
    wavfilename = _thisDir + os.sep + u'data/%s_%s_%s' % (expName, expInfo['participant'],today.strftime('%Y-%m-%d'))+ os.sep + u'data/%s_%s' % (expName, expInfo['participant'])
    
    if expInfo['participant'] == '999':
        CT_opac = 1
        record_TF = False
    else:
        record_TF = True
    
    timeStart = time.time();
    #readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeStart))
    
    # Construct the file path
    file_path = filename + "_segtimes.csv"
    ver_path = filename + "_ver.csv"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        with open(filename+'_segtimes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'T1', 'T2'])
    
    # Check if the file exists
    if not os.path.exists(ver_path):
        with open(filename+'_Ratings.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'Rating'])
    
    syncDev = 'Speakers (Cable Creation), Windows DirectSound'
    
    text_ready = visual.TextStim(win=win, name='text_ready',
        text='Ready for microphone calibration.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_ready = keyboard.Keyboard(deviceName='key_resp_ready')
    mouse_TEMP = event.Mouse(win=win)
    x, y = [None, None]
    mouse_TEMP.mouseClock = core.Clock()
    GO = visual.ShapeStim(
        win=win, name='GO',units='norm', 
        size=(0.12, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[-0.5294, 0.4039, -0.1137],
        opacity=None, depth=-5.0, interpolate=True)
    
    # --- Initialize components for Routine "Ready2" ---
    text_11 = visual.TextStim(win=win, name='text_11',
        text='Ready to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "Intro" ---
    # Set experiment start values for variable component CTC_Opac
    CTC_Opac = -1
    CTC_OpacContainer = []
    text_Intro = visual.TextStim(win=win, name='text_Intro',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_Intro = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Intro.mouseClock = core.Clock()
    key_resp_Intro = keyboard.Keyboard(deviceName='key_resp_Intro')
    text_CTC_Intro = visual.TextStim(win=win, name='text_CTC_Intro',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "StimIntro1" ---
    mouse_StimIntro1 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_StimIntro1.mouseClock = core.Clock()
    key_resp_StimIntro1 = keyboard.Keyboard(deviceName='key_resp_StimIntro1')
    text_CTC_StimIntro1 = visual.TextStim(win=win, name='text_CTC_StimIntro1',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-3.0);
    StimIntro1_image = visual.ImageStim(
        win=win,
        name='StimIntro1_image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "Practice" ---
    text_Practice = visual.TextStim(win=win, name='text_Practice',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_Practice = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Practice.mouseClock = core.Clock()
    key_resp_Practice = keyboard.Keyboard(deviceName='key_resp_Practice')
    text_CTC_Practice = visual.TextStim(win=win, name='text_CTC_Practice',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "StimIntro2" ---
    mouse_StimIntro2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_StimIntro2.mouseClock = core.Clock()
    key_resp_StimIntro2 = keyboard.Keyboard(deviceName='key_resp_StimIntro2')
    StimIntro2_image = visual.ImageStim(
        win=win,
        name='StimIntro2_image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "Intro" ---
    # Set experiment start values for variable component CTC_Opac
    CTC_Opac = -1
    CTC_OpacContainer = []
    text_Intro = visual.TextStim(win=win, name='text_Intro',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_Intro = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Intro.mouseClock = core.Clock()
    key_resp_Intro = keyboard.Keyboard(deviceName='key_resp_Intro')
    text_CTC_Intro = visual.TextStim(win=win, name='text_CTC_Intro',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "BorgRating_Intro" ---
    VER_key_resp_2 = keyboard.Keyboard(deviceName='VER_key_resp_2')
    image_BorgScale_Intro = visual.ImageStim(
        win=win,
        name='image_BorgScale_Intro', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-2.0)
    text_BorgScale_Intro = visual.TextStim(win=win, name='text_BorgScale_Intro',
        text='You will use the scale on the left to rate your vocal effort. Please read the handout on the Borg CR100 scale. Then click anywhere to continue.',
        font='Arial',
        units='norm', pos=(0.5, 0), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_BorgRating_Intro = event.Mouse(win=win)
    x, y = [None, None]
    mouse_BorgRating_Intro.mouseClock = core.Clock()
    text_CTC_BorgRating_Intro = visual.TextStim(win=win, name='text_CTC_BorgRating_Intro',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0.3, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "BorgRating_Intro_2" ---
    VER_key_resp_3 = keyboard.Keyboard(deviceName='VER_key_resp_3')
    image_BorgScale_Intro_2 = visual.ImageStim(
        win=win,
        name='image_BorgScale_Intro_2', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-2.0)
    text_BorgScale_Intro_2 = visual.TextStim(win=win, name='text_BorgScale_Intro_2',
        text='For this experiment you will say "My vocal effort level is..." and then speak the value that corresponds with your vocal effort level. \n\nType this into the computer using the onscreen numbers. You may use decimals and backspace. When you are finished with your rating click the "ENTER" button.\n\nNow you will practice speaking in noise and then rating your vocal effort level.',
        font='Arial',
        units='norm', pos=(0.5, 0), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_BorgRating_Intro_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_BorgRating_Intro_2.mouseClock = core.Clock()
    text_CTC_BorgRating_Intro_2 = visual.TextStim(win=win, name='text_CTC_BorgRating_Intro_2',
        text='~~Click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0.3, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "NoiseTest" ---
    # Run 'Begin Experiment' code from code_Noise
    noise_filename = './Noise/noise_2.wav'
    noisedata, noisefs = sf.read(noise_filename, dtype='float32')  
    
    text_NoiseTest = visual.TextStim(win=win, name='text_NoiseTest',
        text='Speak the names of the months of the year:\n\nJanuary, February, March, April, May, June, July, August, September, October, November, December',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    mouse_NoiseTest = event.Mouse(win=win)
    x, y = [None, None]
    mouse_NoiseTest.mouseClock = core.Clock()
    text_CTC_NoiseTest = visual.TextStim(win=win, name='text_CTC_NoiseTest',
        text='~~When finished, click anywhere to continue~~',
        font='Arial',
        units='norm', pos=(0, -0.7), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "BorgRating" ---
    # Set experiment start values for variable component inputText
    inputText = '#'
    inputTextContainer = []
    VER_key_resp = keyboard.Keyboard(deviceName='VER_key_resp')
    # Run 'Begin Experiment' code from code_Borg
    BorgIter = -1
    
    NumResponse = ''
    
    VER_num = 0
    image_BorgScale = visual.ImageStim(
        win=win,
        name='image_BorgScale', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-3.0)
    text_BorgScale = visual.TextStim(win=win, name='text_BorgScale',
        text='Please rate your vocal effort level by saying\n"My vocal effort level is..."\nand then speak the value that corresponds with your vocal effort level.\nType this value.',
        font='Arial',
        units='norm', pos=(0.5, 0.6), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    VER_num_text = visual.TextStim(win=win, name='VER_num_text',
        text='',
        font='Arial',
        units='norm', pos=(0.5, .25), draggable=False, height=0.125, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_Borg = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Borg.mouseClock = core.Clock()
    button_Borg = visual.ButtonStim(win, 
        text='ENTER', font='Arvo',
        pos=(0.5, -.8),units='norm',
        letterHeight=0.05,
        size=(0.3, .15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=1.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_Borg',
        depth=-7
    )
    button_Borg.buttonClock = core.Clock()
    polygon_1 = visual.Rect(
        win=win, name='polygon_1',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    polygon_3 = visual.Rect(
        win=win, name='polygon_3',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    polygon_4 = visual.Rect(
        win=win, name='polygon_4',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    polygon_5 = visual.Rect(
        win=win, name='polygon_5',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    polygon_6 = visual.Rect(
        win=win, name='polygon_6',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    polygon_7 = visual.Rect(
        win=win, name='polygon_7',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    polygon_8 = visual.Rect(
        win=win, name='polygon_8',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-15.0, interpolate=True)
    polygon_9 = visual.Rect(
        win=win, name='polygon_9',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-16.0, interpolate=True)
    polygon_Back = visual.Rect(
        win=win, name='polygon_Back',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-17.0, interpolate=True)
    polygon_0 = visual.Rect(
        win=win, name='polygon_0',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-18.0, interpolate=True)
    polygon_Dot = visual.Rect(
        win=win, name='polygon_Dot',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-19.0, interpolate=True)
    text_1 = visual.TextStim(win=win, name='text_1',
        text='1',
        font='Arial',
        units='norm', pos=(0.4, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-20.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='2',
        font='Arial',
        units='norm', pos=(0.5, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-21.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='3',
        font='Arial',
        units='norm', pos=(0.6, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-22.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='4',
        font='Arial',
        units='norm', pos=(0.4, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-23.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='5',
        font='Arial',
        units='norm', pos=(0.5, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-24.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text='6',
        font='Arial',
        units='norm', pos=(0.6, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-25.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='7',
        font='Arial',
        units='norm', pos=(0.4, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-26.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='8',
        font='Arial',
        units='norm', pos=(0.5, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-27.0);
    text_9 = visual.TextStim(win=win, name='text_9',
        text='9',
        font='Arial',
        units='norm', pos=(0.6, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-28.0);
    text_Back = visual.TextStim(win=win, name='text_Back',
        text='<—',
        font='Arial',
        units='norm', pos=(0.4, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-29.0);
    text_0 = visual.TextStim(win=win, name='text_0',
        text='0',
        font='Arial',
        units='norm', pos=(0.5, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-30.0);
    text_Dot = visual.TextStim(win=win, name='text_Dot',
        text='.',
        font='Arial',
        units='norm', pos=(0.6, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-31.0);
    
    # --- Initialize components for Routine "Begin" ---
    text_Begin = visual.TextStim(win=win, name='text_Begin',
        text='Any questions before you begin?',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_Begin = keyboard.Keyboard(deviceName='key_resp_Begin')
    
    # --- Initialize components for Routine "SpeechIntro" ---
    # Run 'Begin Experiment' code from code_10
    sound_SpeechIntro = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='sound_SpeechIntro',sampleRate = 44100)
    text_SpeechIntro = visual.TextStim(win=win, name='text_SpeechIntro',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_SpeechIntro = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SpeechIntro.mouseClock = core.Clock()
    sound_SpeechIntro = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_SpeechIntro',    name='sound_SpeechIntro'
    )
    sound_SpeechIntro.setVolume(Vol)
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "SpeechTask" ---
    text_SpeechTask_mid = visual.TextStim(win=win, name='text_SpeechTask_mid',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=1.3, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_SpeechTask = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SpeechTask.mouseClock = core.Clock()
    text_SpeechTask_high = visual.TextStim(win=win, name='text_SpeechTask_high',
        text='',
        font='Arial',
        units='norm', pos=(-.33,.5), draggable=False, height=0.08, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_SpeechTask_low = visual.TextStim(win=win, name='text_SpeechTask_low',
        text='',
        font='Arial',
        units='norm', pos=(0.33, -.5), draggable=False, height=0.08, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    # Set experiment start values for variable component SpeechTextLow
    SpeechTextLow = ''
    SpeechTextLowContainer = []
    # Set experiment start values for variable component SpeechTextHigh
    SpeechTextHigh = ''
    SpeechTextHighContainer = []
    # Set experiment start values for variable component prepost
    prepost = 'pre'
    prepostContainer = []
    
    # --- Initialize components for Routine "BorgRating" ---
    # Set experiment start values for variable component inputText
    inputText = '#'
    inputTextContainer = []
    VER_key_resp = keyboard.Keyboard(deviceName='VER_key_resp')
    # Run 'Begin Experiment' code from code_Borg
    BorgIter = -1
    
    NumResponse = ''
    
    VER_num = 0
    image_BorgScale = visual.ImageStim(
        win=win,
        name='image_BorgScale', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-3.0)
    text_BorgScale = visual.TextStim(win=win, name='text_BorgScale',
        text='Please rate your vocal effort level by saying\n"My vocal effort level is..."\nand then speak the value that corresponds with your vocal effort level.\nType this value.',
        font='Arial',
        units='norm', pos=(0.5, 0.6), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    VER_num_text = visual.TextStim(win=win, name='VER_num_text',
        text='',
        font='Arial',
        units='norm', pos=(0.5, .25), draggable=False, height=0.125, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_Borg = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Borg.mouseClock = core.Clock()
    button_Borg = visual.ButtonStim(win, 
        text='ENTER', font='Arvo',
        pos=(0.5, -.8),units='norm',
        letterHeight=0.05,
        size=(0.3, .15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=1.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_Borg',
        depth=-7
    )
    button_Borg.buttonClock = core.Clock()
    polygon_1 = visual.Rect(
        win=win, name='polygon_1',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    polygon_3 = visual.Rect(
        win=win, name='polygon_3',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    polygon_4 = visual.Rect(
        win=win, name='polygon_4',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    polygon_5 = visual.Rect(
        win=win, name='polygon_5',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    polygon_6 = visual.Rect(
        win=win, name='polygon_6',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    polygon_7 = visual.Rect(
        win=win, name='polygon_7',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    polygon_8 = visual.Rect(
        win=win, name='polygon_8',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-15.0, interpolate=True)
    polygon_9 = visual.Rect(
        win=win, name='polygon_9',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-16.0, interpolate=True)
    polygon_Back = visual.Rect(
        win=win, name='polygon_Back',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-17.0, interpolate=True)
    polygon_0 = visual.Rect(
        win=win, name='polygon_0',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-18.0, interpolate=True)
    polygon_Dot = visual.Rect(
        win=win, name='polygon_Dot',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-19.0, interpolate=True)
    text_1 = visual.TextStim(win=win, name='text_1',
        text='1',
        font='Arial',
        units='norm', pos=(0.4, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-20.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='2',
        font='Arial',
        units='norm', pos=(0.5, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-21.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='3',
        font='Arial',
        units='norm', pos=(0.6, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-22.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='4',
        font='Arial',
        units='norm', pos=(0.4, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-23.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='5',
        font='Arial',
        units='norm', pos=(0.5, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-24.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text='6',
        font='Arial',
        units='norm', pos=(0.6, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-25.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='7',
        font='Arial',
        units='norm', pos=(0.4, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-26.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='8',
        font='Arial',
        units='norm', pos=(0.5, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-27.0);
    text_9 = visual.TextStim(win=win, name='text_9',
        text='9',
        font='Arial',
        units='norm', pos=(0.6, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-28.0);
    text_Back = visual.TextStim(win=win, name='text_Back',
        text='<—',
        font='Arial',
        units='norm', pos=(0.4, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-29.0);
    text_0 = visual.TextStim(win=win, name='text_0',
        text='0',
        font='Arial',
        units='norm', pos=(0.5, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-30.0);
    text_Dot = visual.TextStim(win=win, name='text_Dot',
        text='.',
        font='Arial',
        units='norm', pos=(0.6, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-31.0);
    
    # --- Initialize components for Routine "VLT_init" ---
    # Set experiment start values for variable component pagenum
    pagenum = 0
    pagenumContainer = []
    # Run 'Begin Experiment' code from code_VLT_init
    
    
    
    # --- Initialize components for Routine "STIMULI" ---
    # Set experiment start values for variable component STIM_text
    STIM_text = ""
    STIM_textContainer = []
    # Run 'Begin Experiment' code from STIM_code
    noise_high_filename = './Noise/Noise_High.wav'
    noise_low_filename = './Noise/Noise_Low.wav'
    NHdata, NHfs = sf.read(noise_high_filename, dtype='float32') 
    NLdata, NLfs = sf.read(noise_low_filename, dtype='float32') 
    
    previousLevel = None
    STIM_key_resp = keyboard.Keyboard(deviceName='STIM_key_resp')
    STIM_image = visual.ImageStim(
        win=win,
        name='STIM_image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(2,2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    STIM_mouse = event.Mouse(win=win)
    x, y = [None, None]
    STIM_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "BorgRating" ---
    # Set experiment start values for variable component inputText
    inputText = '#'
    inputTextContainer = []
    VER_key_resp = keyboard.Keyboard(deviceName='VER_key_resp')
    # Run 'Begin Experiment' code from code_Borg
    BorgIter = -1
    
    NumResponse = ''
    
    VER_num = 0
    image_BorgScale = visual.ImageStim(
        win=win,
        name='image_BorgScale', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-3.0)
    text_BorgScale = visual.TextStim(win=win, name='text_BorgScale',
        text='Please rate your vocal effort level by saying\n"My vocal effort level is..."\nand then speak the value that corresponds with your vocal effort level.\nType this value.',
        font='Arial',
        units='norm', pos=(0.5, 0.6), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    VER_num_text = visual.TextStim(win=win, name='VER_num_text',
        text='',
        font='Arial',
        units='norm', pos=(0.5, .25), draggable=False, height=0.125, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_Borg = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Borg.mouseClock = core.Clock()
    button_Borg = visual.ButtonStim(win, 
        text='ENTER', font='Arvo',
        pos=(0.5, -.8),units='norm',
        letterHeight=0.05,
        size=(0.3, .15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=1.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_Borg',
        depth=-7
    )
    button_Borg.buttonClock = core.Clock()
    polygon_1 = visual.Rect(
        win=win, name='polygon_1',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    polygon_3 = visual.Rect(
        win=win, name='polygon_3',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    polygon_4 = visual.Rect(
        win=win, name='polygon_4',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    polygon_5 = visual.Rect(
        win=win, name='polygon_5',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    polygon_6 = visual.Rect(
        win=win, name='polygon_6',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    polygon_7 = visual.Rect(
        win=win, name='polygon_7',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    polygon_8 = visual.Rect(
        win=win, name='polygon_8',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-15.0, interpolate=True)
    polygon_9 = visual.Rect(
        win=win, name='polygon_9',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-16.0, interpolate=True)
    polygon_Back = visual.Rect(
        win=win, name='polygon_Back',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-17.0, interpolate=True)
    polygon_0 = visual.Rect(
        win=win, name='polygon_0',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-18.0, interpolate=True)
    polygon_Dot = visual.Rect(
        win=win, name='polygon_Dot',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-19.0, interpolate=True)
    text_1 = visual.TextStim(win=win, name='text_1',
        text='1',
        font='Arial',
        units='norm', pos=(0.4, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-20.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='2',
        font='Arial',
        units='norm', pos=(0.5, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-21.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='3',
        font='Arial',
        units='norm', pos=(0.6, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-22.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='4',
        font='Arial',
        units='norm', pos=(0.4, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-23.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='5',
        font='Arial',
        units='norm', pos=(0.5, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-24.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text='6',
        font='Arial',
        units='norm', pos=(0.6, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-25.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='7',
        font='Arial',
        units='norm', pos=(0.4, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-26.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='8',
        font='Arial',
        units='norm', pos=(0.5, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-27.0);
    text_9 = visual.TextStim(win=win, name='text_9',
        text='9',
        font='Arial',
        units='norm', pos=(0.6, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-28.0);
    text_Back = visual.TextStim(win=win, name='text_Back',
        text='<—',
        font='Arial',
        units='norm', pos=(0.4, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-29.0);
    text_0 = visual.TextStim(win=win, name='text_0',
        text='0',
        font='Arial',
        units='norm', pos=(0.5, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-30.0);
    text_Dot = visual.TextStim(win=win, name='text_Dot',
        text='.',
        font='Arial',
        units='norm', pos=(0.6, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-31.0);
    
    # --- Initialize components for Routine "TLX" ---
    # Set experiment start values for variable component TLX_Labelstr
    TLX_Labelstr = 'Very\nLow','','','','','','','','','','Very\nHigh'
    TLX_LabelstrContainer = []
    # Run 'Begin Experiment' code from code_TLX
    TLX_num = 0
    
    text_TLXTitle = visual.TextStim(win=win, name='text_TLXTitle',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.4), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_TLXDescription = visual.TextStim(win=win, name='text_TLXDescription',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    slider_TLX = visual.Slider(win=win, name='slider_TLX',
        startValue=6, size=(1.5, 0.1), pos=(0, -.3), units='norm',
        labels=TLX_Labelstr, ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.05,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    button_TLX = visual.ButtonStim(win, 
        text='ENTER', font='Arvo',
        pos=(0, -.7),units='norm',
        letterHeight=0.05,
        size=(0.4, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_TLX',
        depth=-5
    )
    button_TLX.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "BorgRating" ---
    # Set experiment start values for variable component inputText
    inputText = '#'
    inputTextContainer = []
    VER_key_resp = keyboard.Keyboard(deviceName='VER_key_resp')
    # Run 'Begin Experiment' code from code_Borg
    BorgIter = -1
    
    NumResponse = ''
    
    VER_num = 0
    image_BorgScale = visual.ImageStim(
        win=win,
        name='image_BorgScale', units='norm', 
        image='PsychoPy/borg_cr100.png', mask=None, anchor='center',
        ori=0, pos=(-0.5, 0), draggable=False, size=(0.8,1.9),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=512, interpolate=True, depth=-3.0)
    text_BorgScale = visual.TextStim(win=win, name='text_BorgScale',
        text='Please rate your vocal effort level by saying\n"My vocal effort level is..."\nand then speak the value that corresponds with your vocal effort level.\nType this value.',
        font='Arial',
        units='norm', pos=(0.5, 0.6), draggable=False, height=0.06, wrapWidth=0.8, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    VER_num_text = visual.TextStim(win=win, name='VER_num_text',
        text='',
        font='Arial',
        units='norm', pos=(0.5, .25), draggable=False, height=0.125, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    mouse_Borg = event.Mouse(win=win)
    x, y = [None, None]
    mouse_Borg.mouseClock = core.Clock()
    button_Borg = visual.ButtonStim(win, 
        text='ENTER', font='Arvo',
        pos=(0.5, -.8),units='norm',
        letterHeight=0.05,
        size=(0.3, .15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=1.0,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_Borg',
        depth=-7
    )
    button_Borg.buttonClock = core.Clock()
    polygon_1 = visual.Rect(
        win=win, name='polygon_1',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    polygon_3 = visual.Rect(
        win=win, name='polygon_3',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.1), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    polygon_4 = visual.Rect(
        win=win, name='polygon_4',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    polygon_5 = visual.Rect(
        win=win, name='polygon_5',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    polygon_6 = visual.Rect(
        win=win, name='polygon_6',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.2), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    polygon_7 = visual.Rect(
        win=win, name='polygon_7',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    polygon_8 = visual.Rect(
        win=win, name='polygon_8',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-15.0, interpolate=True)
    polygon_9 = visual.Rect(
        win=win, name='polygon_9',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.3), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-16.0, interpolate=True)
    polygon_Back = visual.Rect(
        win=win, name='polygon_Back',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.4, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-17.0, interpolate=True)
    polygon_0 = visual.Rect(
        win=win, name='polygon_0',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.5, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-18.0, interpolate=True)
    polygon_Dot = visual.Rect(
        win=win, name='polygon_Dot',units='norm', 
        width=(0.1, 0.1)[0], height=(0.1, 0.1)[1],
        ori=0.0, pos=(0.6, -0.4), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-19.0, interpolate=True)
    text_1 = visual.TextStim(win=win, name='text_1',
        text='1',
        font='Arial',
        units='norm', pos=(0.4, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-20.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='2',
        font='Arial',
        units='norm', pos=(0.5, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-21.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='3',
        font='Arial',
        units='norm', pos=(0.6, -0.1), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-22.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='4',
        font='Arial',
        units='norm', pos=(0.4, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-23.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='5',
        font='Arial',
        units='norm', pos=(0.5, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-24.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text='6',
        font='Arial',
        units='norm', pos=(0.6, -0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-25.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='7',
        font='Arial',
        units='norm', pos=(0.4, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-26.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='8',
        font='Arial',
        units='norm', pos=(0.5, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-27.0);
    text_9 = visual.TextStim(win=win, name='text_9',
        text='9',
        font='Arial',
        units='norm', pos=(0.6, -0.3), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-28.0);
    text_Back = visual.TextStim(win=win, name='text_Back',
        text='<—',
        font='Arial',
        units='norm', pos=(0.4, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-29.0);
    text_0 = visual.TextStim(win=win, name='text_0',
        text='0',
        font='Arial',
        units='norm', pos=(0.5, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-30.0);
    text_Dot = visual.TextStim(win=win, name='text_Dot',
        text='.',
        font='Arial',
        units='norm', pos=(0.6, -0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-31.0);
    
    # --- Initialize components for Routine "End" ---
    text_End = visual.TextStim(win=win, name='text_End',
        text='Thank you for your participation.',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_End = keyboard.Keyboard(deviceName='key_resp_End')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Ready" ---
    # create an object to store info about Routine Ready
    Ready = data.Routine(
        name='Ready',
        components=[text_ready, key_resp_ready, mouse_TEMP, GO],
    )
    Ready.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_ready
    key_resp_ready.keys = []
    key_resp_ready.rt = []
    _key_resp_ready_allKeys = []
    # setup some python lists for storing info about the mouse_TEMP
    mouse_TEMP.x = []
    mouse_TEMP.y = []
    mouse_TEMP.leftButton = []
    mouse_TEMP.midButton = []
    mouse_TEMP.rightButton = []
    mouse_TEMP.time = []
    mouse_TEMP.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Ready
    Ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Ready.tStart = globalClock.getTime(format='float')
    Ready.status = STARTED
    thisExp.addData('Ready.started', Ready.tStart)
    Ready.maxDuration = None
    # keep track of which components have finished
    ReadyComponents = Ready.components
    for thisComponent in Ready.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Ready" ---
    Ready.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ready* updates
        
        # if text_ready is starting this frame...
        if text_ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ready.frameNStart = frameN  # exact frame index
            text_ready.tStart = t  # local t and not account for scr refresh
            text_ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ready.started')
            # update status
            text_ready.status = STARTED
            text_ready.setAutoDraw(True)
        
        # if text_ready is active this frame...
        if text_ready.status == STARTED:
            # update params
            pass
        
        # *key_resp_ready* updates
        waitOnFlip = False
        
        # if key_resp_ready is starting this frame...
        if key_resp_ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ready.frameNStart = frameN  # exact frame index
            key_resp_ready.tStart = t  # local t and not account for scr refresh
            key_resp_ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ready.started')
            # update status
            key_resp_ready.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ready.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_ready.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ready.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ready_allKeys.extend(theseKeys)
            if len(_key_resp_ready_allKeys):
                key_resp_ready.keys = _key_resp_ready_allKeys[-1].name  # just the last key pressed
                key_resp_ready.rt = _key_resp_ready_allKeys[-1].rt
                key_resp_ready.duration = _key_resp_ready_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # *mouse_TEMP* updates
        
        # if mouse_TEMP is starting this frame...
        if mouse_TEMP.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_TEMP.frameNStart = frameN  # exact frame index
            mouse_TEMP.tStart = t  # local t and not account for scr refresh
            mouse_TEMP.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_TEMP, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_TEMP.started', t)
            # update status
            mouse_TEMP.status = STARTED
            mouse_TEMP.mouseClock.reset()
            prevButtonState = mouse_TEMP.getPressed()  # if button is down already this ISN'T a new click
        if mouse_TEMP.status == STARTED:  # only update if started and not finished!
            buttons = mouse_TEMP.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(GO, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_TEMP):
                            gotValidClick = True
                            mouse_TEMP.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_TEMP.clicked_name.append(None)
                    x, y = mouse_TEMP.getPos()
                    mouse_TEMP.x.append(x)
                    mouse_TEMP.y.append(y)
                    buttons = mouse_TEMP.getPressed()
                    mouse_TEMP.leftButton.append(buttons[0])
                    mouse_TEMP.midButton.append(buttons[1])
                    mouse_TEMP.rightButton.append(buttons[2])
                    mouse_TEMP.time.append(mouse_TEMP.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *GO* updates
        
        # if GO is starting this frame...
        if GO.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            GO.frameNStart = frameN  # exact frame index
            GO.tStart = t  # local t and not account for scr refresh
            GO.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(GO, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'GO.started')
            # update status
            GO.status = STARTED
            GO.setAutoDraw(True)
        
        # if GO is active this frame...
        if GO.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Ready,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Ready.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Ready.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Ready" ---
    for thisComponent in Ready.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Ready
    Ready.tStop = globalClock.getTime(format='float')
    Ready.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Ready.stopped', Ready.tStop)
    
    # Run 'End Routine' code from code_ready
    with open(filename+'_segtimes.csv', mode='r') as file:
        reader = csv.reader(file)
        segData = [row for row in reader]
    timeStart = time.time();
    
    #sd.play(beep, samplerate=fs, device=syncDev)
    new_row = ['Beep', timeStart, 0]
    segData.append(new_row)
    
    # Save the modified data back to the CSV file
    with open(filename+'_segtimes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(segData)
    
    time.sleep(1.0)
    # check responses
    if key_resp_ready.keys in ['', [], None]:  # No response was made
        key_resp_ready.keys = None
    thisExp.addData('key_resp_ready.keys',key_resp_ready.keys)
    if key_resp_ready.keys != None:  # we had a response
        thisExp.addData('key_resp_ready.rt', key_resp_ready.rt)
        thisExp.addData('key_resp_ready.duration', key_resp_ready.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_TEMP.x', mouse_TEMP.x)
    thisExp.addData('mouse_TEMP.y', mouse_TEMP.y)
    thisExp.addData('mouse_TEMP.leftButton', mouse_TEMP.leftButton)
    thisExp.addData('mouse_TEMP.midButton', mouse_TEMP.midButton)
    thisExp.addData('mouse_TEMP.rightButton', mouse_TEMP.rightButton)
    thisExp.addData('mouse_TEMP.time', mouse_TEMP.time)
    thisExp.addData('mouse_TEMP.clicked_name', mouse_TEMP.clicked_name)
    thisExp.nextEntry()
    # the Routine "Ready" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Ready2" ---
    # create an object to store info about Routine Ready2
    Ready2 = data.Routine(
        name='Ready2',
        components=[text_11, key_resp_3],
    )
    Ready2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for Ready2
    Ready2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Ready2.tStart = globalClock.getTime(format='float')
    Ready2.status = STARTED
    thisExp.addData('Ready2.started', Ready2.tStart)
    Ready2.maxDuration = None
    # keep track of which components have finished
    Ready2Components = Ready2.components
    for thisComponent in Ready2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Ready2" ---
    Ready2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_11* updates
        
        # if text_11 is starting this frame...
        if text_11.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_11.frameNStart = frameN  # exact frame index
            text_11.tStart = t  # local t and not account for scr refresh
            text_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_11.started')
            # update status
            text_11.status = STARTED
            text_11.setAutoDraw(True)
        
        # if text_11 is active this frame...
        if text_11.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Ready2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Ready2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Ready2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Ready2" ---
    for thisComponent in Ready2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Ready2
    Ready2.tStop = globalClock.getTime(format='float')
    Ready2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Ready2.stopped', Ready2.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "Ready2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_Intro1 = data.TrialHandler2(
        name='trials_Intro1',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/Intro1_pic.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_Intro1)  # add the loop to the experiment
    thisTrials_Intro1 = trials_Intro1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_Intro1.rgb)
    if thisTrials_Intro1 != None:
        for paramName in thisTrials_Intro1:
            globals()[paramName] = thisTrials_Intro1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_Intro1 in trials_Intro1:
        trials_Intro1.status = STARTED
        if hasattr(thisTrials_Intro1, 'status'):
            thisTrials_Intro1.status = STARTED
        currentLoop = trials_Intro1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_Intro1.rgb)
        if thisTrials_Intro1 != None:
            for paramName in thisTrials_Intro1:
                globals()[paramName] = thisTrials_Intro1[paramName]
        
        # --- Prepare to start Routine "Intro" ---
        # create an object to store info about Routine Intro
        Intro = data.Routine(
            name='Intro',
            components=[text_Intro, mouse_Intro, key_resp_Intro, text_CTC_Intro],
        )
        Intro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_Intro
        trial_filename = './Sounds/'+IntroWav
        trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
        
        sd.play(Vol*trialdata, samplerate=trialfs)
        
        text_Intro.setText(IntroText)
        # setup some python lists for storing info about the mouse_Intro
        mouse_Intro.x = []
        mouse_Intro.y = []
        mouse_Intro.leftButton = []
        mouse_Intro.midButton = []
        mouse_Intro.rightButton = []
        mouse_Intro.time = []
        gotValidClick = False  # until a click is received
        # create starting attributes for key_resp_Intro
        key_resp_Intro.keys = []
        key_resp_Intro.rt = []
        _key_resp_Intro_allKeys = []
        # store start times for Intro
        Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Intro.tStart = globalClock.getTime(format='float')
        Intro.status = STARTED
        thisExp.addData('Intro.started', Intro.tStart)
        Intro.maxDuration = None
        # keep track of which components have finished
        IntroComponents = Intro.components
        for thisComponent in Intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Intro" ---
        Intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_Intro1, 'status') and thisTrials_Intro1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_Intro
            playing = sd.get_stream().active
            
            if playing:
                CTC_Opac = -1
            elif not playing:
                CTC_Opac = 1
            
            # *text_Intro* updates
            
            # if text_Intro is starting this frame...
            if text_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Intro.frameNStart = frameN  # exact frame index
                text_Intro.tStart = t  # local t and not account for scr refresh
                text_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_Intro.started')
                # update status
                text_Intro.status = STARTED
                text_Intro.setAutoDraw(True)
            
            # if text_Intro is active this frame...
            if text_Intro.status == STARTED:
                # update params
                pass
            # *mouse_Intro* updates
            
            # if mouse_Intro is starting this frame...
            if mouse_Intro.status == NOT_STARTED and t >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                mouse_Intro.frameNStart = frameN  # exact frame index
                mouse_Intro.tStart = t  # local t and not account for scr refresh
                mouse_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_Intro.started', t)
                # update status
                mouse_Intro.status = STARTED
                mouse_Intro.mouseClock.reset()
                prevButtonState = mouse_Intro.getPressed()  # if button is down already this ISN'T a new click
            if mouse_Intro.status == STARTED:  # only update if started and not finished!
                buttons = mouse_Intro.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_Intro.getPos()
                        mouse_Intro.x.append(x)
                        mouse_Intro.y.append(y)
                        buttons = mouse_Intro.getPressed()
                        mouse_Intro.leftButton.append(buttons[0])
                        mouse_Intro.midButton.append(buttons[1])
                        mouse_Intro.rightButton.append(buttons[2])
                        mouse_Intro.time.append(mouse_Intro.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *key_resp_Intro* updates
            waitOnFlip = False
            
            # if key_resp_Intro is starting this frame...
            if key_resp_Intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_Intro.frameNStart = frameN  # exact frame index
                key_resp_Intro.tStart = t  # local t and not account for scr refresh
                key_resp_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_Intro.started')
                # update status
                key_resp_Intro.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_Intro.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_Intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_Intro.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_Intro.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_Intro_allKeys.extend(theseKeys)
                if len(_key_resp_Intro_allKeys):
                    key_resp_Intro.keys = _key_resp_Intro_allKeys[-1].name  # just the last key pressed
                    key_resp_Intro.rt = _key_resp_Intro_allKeys[-1].rt
                    key_resp_Intro.duration = _key_resp_Intro_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_CTC_Intro* updates
            
            # if text_CTC_Intro is starting this frame...
            if text_CTC_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_CTC_Intro.frameNStart = frameN  # exact frame index
                text_CTC_Intro.tStart = t  # local t and not account for scr refresh
                text_CTC_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_CTC_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_CTC_Intro.started')
                # update status
                text_CTC_Intro.status = STARTED
                text_CTC_Intro.setAutoDraw(True)
            
            # if text_CTC_Intro is active this frame...
            if text_CTC_Intro.status == STARTED:
                # update params
                text_CTC_Intro.setOpacity(CTC_Opac, log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Intro,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Intro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Intro" ---
        for thisComponent in Intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Intro
        Intro.tStop = globalClock.getTime(format='float')
        Intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Intro.stopped', Intro.tStop)
        
        # Run 'End Routine' code from code_Intro
        sd.stop()
        # store data for trials_Intro1 (TrialHandler)
        trials_Intro1.addData('mouse_Intro.x', mouse_Intro.x)
        trials_Intro1.addData('mouse_Intro.y', mouse_Intro.y)
        trials_Intro1.addData('mouse_Intro.leftButton', mouse_Intro.leftButton)
        trials_Intro1.addData('mouse_Intro.midButton', mouse_Intro.midButton)
        trials_Intro1.addData('mouse_Intro.rightButton', mouse_Intro.rightButton)
        trials_Intro1.addData('mouse_Intro.time', mouse_Intro.time)
        # check responses
        if key_resp_Intro.keys in ['', [], None]:  # No response was made
            key_resp_Intro.keys = None
        trials_Intro1.addData('key_resp_Intro.keys',key_resp_Intro.keys)
        if key_resp_Intro.keys != None:  # we had a response
            trials_Intro1.addData('key_resp_Intro.rt', key_resp_Intro.rt)
            trials_Intro1.addData('key_resp_Intro.duration', key_resp_Intro.duration)
        # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_Intro1 as finished
        if hasattr(thisTrials_Intro1, 'status'):
            thisTrials_Intro1.status = FINISHED
        # if awaiting a pause, pause now
        if trials_Intro1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_Intro1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_Intro1'
    trials_Intro1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials_stimintro1 = data.TrialHandler2(
        name='trials_stimintro1',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/Pics_Intro1.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_stimintro1)  # add the loop to the experiment
    thisTrials_stimintro1 = trials_stimintro1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_stimintro1.rgb)
    if thisTrials_stimintro1 != None:
        for paramName in thisTrials_stimintro1:
            globals()[paramName] = thisTrials_stimintro1[paramName]
    
    for thisTrials_stimintro1 in trials_stimintro1:
        trials_stimintro1.status = STARTED
        if hasattr(thisTrials_stimintro1, 'status'):
            thisTrials_stimintro1.status = STARTED
        currentLoop = trials_stimintro1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_stimintro1.rgb)
        if thisTrials_stimintro1 != None:
            for paramName in thisTrials_stimintro1:
                globals()[paramName] = thisTrials_stimintro1[paramName]
        
        # --- Prepare to start Routine "StimIntro1" ---
        # create an object to store info about Routine StimIntro1
        StimIntro1 = data.Routine(
            name='StimIntro1',
            components=[mouse_StimIntro1, key_resp_StimIntro1, text_CTC_StimIntro1, StimIntro1_image],
        )
        StimIntro1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_StimIntro1
        trialdata, trialfs = sf.read(sound_intro_1, dtype='float32')  
        sd.play(Vol*trialdata, samplerate=trialfs)
        # setup some python lists for storing info about the mouse_StimIntro1
        mouse_StimIntro1.x = []
        mouse_StimIntro1.y = []
        mouse_StimIntro1.leftButton = []
        mouse_StimIntro1.midButton = []
        mouse_StimIntro1.rightButton = []
        mouse_StimIntro1.time = []
        gotValidClick = False  # until a click is received
        # create starting attributes for key_resp_StimIntro1
        key_resp_StimIntro1.keys = []
        key_resp_StimIntro1.rt = []
        _key_resp_StimIntro1_allKeys = []
        StimIntro1_image.setImage(picture_image_intro1)
        # store start times for StimIntro1
        StimIntro1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        StimIntro1.tStart = globalClock.getTime(format='float')
        StimIntro1.status = STARTED
        thisExp.addData('StimIntro1.started', StimIntro1.tStart)
        StimIntro1.maxDuration = None
        # keep track of which components have finished
        StimIntro1Components = StimIntro1.components
        for thisComponent in StimIntro1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "StimIntro1" ---
        StimIntro1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_stimintro1, 'status') and thisTrials_stimintro1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_StimIntro1
            playing = sd.get_stream().active
            
            if playing:
                #CTC_Opac = -1
                pass
            elif not playing:
                #CTC_Opac = 1
                time.sleep(0.25)
                continueRoutine = False
            # *mouse_StimIntro1* updates
            
            # if mouse_StimIntro1 is starting this frame...
            if mouse_StimIntro1.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                mouse_StimIntro1.frameNStart = frameN  # exact frame index
                mouse_StimIntro1.tStart = t  # local t and not account for scr refresh
                mouse_StimIntro1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_StimIntro1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_StimIntro1.started', t)
                # update status
                mouse_StimIntro1.status = STARTED
                mouse_StimIntro1.mouseClock.reset()
                prevButtonState = mouse_StimIntro1.getPressed()  # if button is down already this ISN'T a new click
            if mouse_StimIntro1.status == STARTED:  # only update if started and not finished!
                buttons = mouse_StimIntro1.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_StimIntro1.getPos()
                        mouse_StimIntro1.x.append(x)
                        mouse_StimIntro1.y.append(y)
                        buttons = mouse_StimIntro1.getPressed()
                        mouse_StimIntro1.leftButton.append(buttons[0])
                        mouse_StimIntro1.midButton.append(buttons[1])
                        mouse_StimIntro1.rightButton.append(buttons[2])
                        mouse_StimIntro1.time.append(mouse_StimIntro1.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *key_resp_StimIntro1* updates
            waitOnFlip = False
            
            # if key_resp_StimIntro1 is starting this frame...
            if key_resp_StimIntro1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_StimIntro1.frameNStart = frameN  # exact frame index
                key_resp_StimIntro1.tStart = t  # local t and not account for scr refresh
                key_resp_StimIntro1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_StimIntro1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_StimIntro1.started')
                # update status
                key_resp_StimIntro1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_StimIntro1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_StimIntro1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_StimIntro1.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_StimIntro1.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_StimIntro1_allKeys.extend(theseKeys)
                if len(_key_resp_StimIntro1_allKeys):
                    key_resp_StimIntro1.keys = _key_resp_StimIntro1_allKeys[-1].name  # just the last key pressed
                    key_resp_StimIntro1.rt = _key_resp_StimIntro1_allKeys[-1].rt
                    key_resp_StimIntro1.duration = _key_resp_StimIntro1_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_CTC_StimIntro1* updates
            
            # if text_CTC_StimIntro1 is starting this frame...
            if text_CTC_StimIntro1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_CTC_StimIntro1.frameNStart = frameN  # exact frame index
                text_CTC_StimIntro1.tStart = t  # local t and not account for scr refresh
                text_CTC_StimIntro1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_CTC_StimIntro1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_CTC_StimIntro1.started')
                # update status
                text_CTC_StimIntro1.status = STARTED
                text_CTC_StimIntro1.setAutoDraw(True)
            
            # if text_CTC_StimIntro1 is active this frame...
            if text_CTC_StimIntro1.status == STARTED:
                # update params
                text_CTC_StimIntro1.setOpacity(CTC_Opac, log=False)
            
            # *StimIntro1_image* updates
            
            # if StimIntro1_image is starting this frame...
            if StimIntro1_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                StimIntro1_image.frameNStart = frameN  # exact frame index
                StimIntro1_image.tStart = t  # local t and not account for scr refresh
                StimIntro1_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(StimIntro1_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'StimIntro1_image.started')
                # update status
                StimIntro1_image.status = STARTED
                StimIntro1_image.setAutoDraw(True)
            
            # if StimIntro1_image is active this frame...
            if StimIntro1_image.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=StimIntro1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                StimIntro1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in StimIntro1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "StimIntro1" ---
        for thisComponent in StimIntro1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for StimIntro1
        StimIntro1.tStop = globalClock.getTime(format='float')
        StimIntro1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('StimIntro1.stopped', StimIntro1.tStop)
        # Run 'End Routine' code from code_StimIntro1
        sd.stop()
        # store data for trials_stimintro1 (TrialHandler)
        trials_stimintro1.addData('mouse_StimIntro1.x', mouse_StimIntro1.x)
        trials_stimintro1.addData('mouse_StimIntro1.y', mouse_StimIntro1.y)
        trials_stimintro1.addData('mouse_StimIntro1.leftButton', mouse_StimIntro1.leftButton)
        trials_stimintro1.addData('mouse_StimIntro1.midButton', mouse_StimIntro1.midButton)
        trials_stimintro1.addData('mouse_StimIntro1.rightButton', mouse_StimIntro1.rightButton)
        trials_stimintro1.addData('mouse_StimIntro1.time', mouse_StimIntro1.time)
        # check responses
        if key_resp_StimIntro1.keys in ['', [], None]:  # No response was made
            key_resp_StimIntro1.keys = None
        trials_stimintro1.addData('key_resp_StimIntro1.keys',key_resp_StimIntro1.keys)
        if key_resp_StimIntro1.keys != None:  # we had a response
            trials_stimintro1.addData('key_resp_StimIntro1.rt', key_resp_StimIntro1.rt)
            trials_stimintro1.addData('key_resp_StimIntro1.duration', key_resp_StimIntro1.duration)
        # the Routine "StimIntro1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_stimintro1 as finished
        if hasattr(thisTrials_stimintro1, 'status'):
            thisTrials_stimintro1.status = FINISHED
        # if awaiting a pause, pause now
        if trials_stimintro1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_stimintro1.status = STARTED
    # completed 1.0 repeats of 'trials_stimintro1'
    trials_stimintro1.status = FINISHED
    
    
    # --- Prepare to start Routine "Practice" ---
    # create an object to store info about Routine Practice
    Practice = data.Routine(
        name='Practice',
        components=[text_Practice, mouse_Practice, key_resp_Practice, text_CTC_Practice],
    )
    Practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_Practice
    trial_filename = './Sounds/practice.wav'
    trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
    
    sd.play(Vol*trialdata, samplerate=trialfs)
    
    text_Practice.setText('Now you practice. Click anywhere when you need to advance the screen.')
    # setup some python lists for storing info about the mouse_Practice
    mouse_Practice.x = []
    mouse_Practice.y = []
    mouse_Practice.leftButton = []
    mouse_Practice.midButton = []
    mouse_Practice.rightButton = []
    mouse_Practice.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for key_resp_Practice
    key_resp_Practice.keys = []
    key_resp_Practice.rt = []
    _key_resp_Practice_allKeys = []
    # store start times for Practice
    Practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Practice.tStart = globalClock.getTime(format='float')
    Practice.status = STARTED
    thisExp.addData('Practice.started', Practice.tStart)
    Practice.maxDuration = None
    # keep track of which components have finished
    PracticeComponents = Practice.components
    for thisComponent in Practice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Practice" ---
    Practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_Practice
        playing = sd.get_stream().active
        
        if playing:
            CTC_Opac = -1
        elif not playing:
            CTC_Opac = 1
        
        # *text_Practice* updates
        
        # if text_Practice is starting this frame...
        if text_Practice.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Practice.frameNStart = frameN  # exact frame index
            text_Practice.tStart = t  # local t and not account for scr refresh
            text_Practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_Practice.started')
            # update status
            text_Practice.status = STARTED
            text_Practice.setAutoDraw(True)
        
        # if text_Practice is active this frame...
        if text_Practice.status == STARTED:
            # update params
            pass
        # *mouse_Practice* updates
        
        # if mouse_Practice is starting this frame...
        if mouse_Practice.status == NOT_STARTED and t >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            mouse_Practice.frameNStart = frameN  # exact frame index
            mouse_Practice.tStart = t  # local t and not account for scr refresh
            mouse_Practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_Practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_Practice.started', t)
            # update status
            mouse_Practice.status = STARTED
            mouse_Practice.mouseClock.reset()
            prevButtonState = mouse_Practice.getPressed()  # if button is down already this ISN'T a new click
        if mouse_Practice.status == STARTED:  # only update if started and not finished!
            buttons = mouse_Practice.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_Practice.getPos()
                    mouse_Practice.x.append(x)
                    mouse_Practice.y.append(y)
                    buttons = mouse_Practice.getPressed()
                    mouse_Practice.leftButton.append(buttons[0])
                    mouse_Practice.midButton.append(buttons[1])
                    mouse_Practice.rightButton.append(buttons[2])
                    mouse_Practice.time.append(mouse_Practice.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *key_resp_Practice* updates
        waitOnFlip = False
        
        # if key_resp_Practice is starting this frame...
        if key_resp_Practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_Practice.frameNStart = frameN  # exact frame index
            key_resp_Practice.tStart = t  # local t and not account for scr refresh
            key_resp_Practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_Practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_Practice.started')
            # update status
            key_resp_Practice.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_Practice.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_Practice.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_Practice.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_Practice.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_Practice_allKeys.extend(theseKeys)
            if len(_key_resp_Practice_allKeys):
                key_resp_Practice.keys = _key_resp_Practice_allKeys[-1].name  # just the last key pressed
                key_resp_Practice.rt = _key_resp_Practice_allKeys[-1].rt
                key_resp_Practice.duration = _key_resp_Practice_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_CTC_Practice* updates
        
        # if text_CTC_Practice is starting this frame...
        if text_CTC_Practice.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_CTC_Practice.frameNStart = frameN  # exact frame index
            text_CTC_Practice.tStart = t  # local t and not account for scr refresh
            text_CTC_Practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_CTC_Practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_CTC_Practice.started')
            # update status
            text_CTC_Practice.status = STARTED
            text_CTC_Practice.setAutoDraw(True)
        
        # if text_CTC_Practice is active this frame...
        if text_CTC_Practice.status == STARTED:
            # update params
            text_CTC_Practice.setOpacity(CTC_Opac, log=False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Practice,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Practice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Practice" ---
    for thisComponent in Practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Practice
    Practice.tStop = globalClock.getTime(format='float')
    Practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Practice.stopped', Practice.tStop)
    # Run 'End Routine' code from code_Practice
    sd.stop()
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_Practice.x', mouse_Practice.x)
    thisExp.addData('mouse_Practice.y', mouse_Practice.y)
    thisExp.addData('mouse_Practice.leftButton', mouse_Practice.leftButton)
    thisExp.addData('mouse_Practice.midButton', mouse_Practice.midButton)
    thisExp.addData('mouse_Practice.rightButton', mouse_Practice.rightButton)
    thisExp.addData('mouse_Practice.time', mouse_Practice.time)
    # check responses
    if key_resp_Practice.keys in ['', [], None]:  # No response was made
        key_resp_Practice.keys = None
    thisExp.addData('key_resp_Practice.keys',key_resp_Practice.keys)
    if key_resp_Practice.keys != None:  # we had a response
        thisExp.addData('key_resp_Practice.rt', key_resp_Practice.rt)
        thisExp.addData('key_resp_Practice.duration', key_resp_Practice.duration)
    thisExp.nextEntry()
    # the Routine "Practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_stimintro2 = data.TrialHandler2(
        name='trials_stimintro2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/Pics_Intro2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_stimintro2)  # add the loop to the experiment
    thisTrials_stimintro2 = trials_stimintro2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_stimintro2.rgb)
    if thisTrials_stimintro2 != None:
        for paramName in thisTrials_stimintro2:
            globals()[paramName] = thisTrials_stimintro2[paramName]
    
    for thisTrials_stimintro2 in trials_stimintro2:
        trials_stimintro2.status = STARTED
        if hasattr(thisTrials_stimintro2, 'status'):
            thisTrials_stimintro2.status = STARTED
        currentLoop = trials_stimintro2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_stimintro2.rgb)
        if thisTrials_stimintro2 != None:
            for paramName in thisTrials_stimintro2:
                globals()[paramName] = thisTrials_stimintro2[paramName]
        
        # --- Prepare to start Routine "StimIntro2" ---
        # create an object to store info about Routine StimIntro2
        StimIntro2 = data.Routine(
            name='StimIntro2',
            components=[mouse_StimIntro2, key_resp_StimIntro2, StimIntro2_image],
        )
        StimIntro2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the mouse_StimIntro2
        mouse_StimIntro2.x = []
        mouse_StimIntro2.y = []
        mouse_StimIntro2.leftButton = []
        mouse_StimIntro2.midButton = []
        mouse_StimIntro2.rightButton = []
        mouse_StimIntro2.time = []
        gotValidClick = False  # until a click is received
        # create starting attributes for key_resp_StimIntro2
        key_resp_StimIntro2.keys = []
        key_resp_StimIntro2.rt = []
        _key_resp_StimIntro2_allKeys = []
        StimIntro2_image.setImage(picture_image_intro2)
        # store start times for StimIntro2
        StimIntro2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        StimIntro2.tStart = globalClock.getTime(format='float')
        StimIntro2.status = STARTED
        thisExp.addData('StimIntro2.started', StimIntro2.tStart)
        StimIntro2.maxDuration = None
        # keep track of which components have finished
        StimIntro2Components = StimIntro2.components
        for thisComponent in StimIntro2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "StimIntro2" ---
        StimIntro2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_stimintro2, 'status') and thisTrials_stimintro2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *mouse_StimIntro2* updates
            
            # if mouse_StimIntro2 is starting this frame...
            if mouse_StimIntro2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                mouse_StimIntro2.frameNStart = frameN  # exact frame index
                mouse_StimIntro2.tStart = t  # local t and not account for scr refresh
                mouse_StimIntro2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_StimIntro2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_StimIntro2.started', t)
                # update status
                mouse_StimIntro2.status = STARTED
                mouse_StimIntro2.mouseClock.reset()
                prevButtonState = mouse_StimIntro2.getPressed()  # if button is down already this ISN'T a new click
            if mouse_StimIntro2.status == STARTED:  # only update if started and not finished!
                buttons = mouse_StimIntro2.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_StimIntro2.getPos()
                        mouse_StimIntro2.x.append(x)
                        mouse_StimIntro2.y.append(y)
                        buttons = mouse_StimIntro2.getPressed()
                        mouse_StimIntro2.leftButton.append(buttons[0])
                        mouse_StimIntro2.midButton.append(buttons[1])
                        mouse_StimIntro2.rightButton.append(buttons[2])
                        mouse_StimIntro2.time.append(mouse_StimIntro2.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *key_resp_StimIntro2* updates
            waitOnFlip = False
            
            # if key_resp_StimIntro2 is starting this frame...
            if key_resp_StimIntro2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_StimIntro2.frameNStart = frameN  # exact frame index
                key_resp_StimIntro2.tStart = t  # local t and not account for scr refresh
                key_resp_StimIntro2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_StimIntro2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_StimIntro2.started')
                # update status
                key_resp_StimIntro2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_StimIntro2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_StimIntro2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_StimIntro2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_StimIntro2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_StimIntro2_allKeys.extend(theseKeys)
                if len(_key_resp_StimIntro2_allKeys):
                    key_resp_StimIntro2.keys = _key_resp_StimIntro2_allKeys[-1].name  # just the last key pressed
                    key_resp_StimIntro2.rt = _key_resp_StimIntro2_allKeys[-1].rt
                    key_resp_StimIntro2.duration = _key_resp_StimIntro2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *StimIntro2_image* updates
            
            # if StimIntro2_image is starting this frame...
            if StimIntro2_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                StimIntro2_image.frameNStart = frameN  # exact frame index
                StimIntro2_image.tStart = t  # local t and not account for scr refresh
                StimIntro2_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(StimIntro2_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'StimIntro2_image.started')
                # update status
                StimIntro2_image.status = STARTED
                StimIntro2_image.setAutoDraw(True)
            
            # if StimIntro2_image is active this frame...
            if StimIntro2_image.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=StimIntro2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                StimIntro2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in StimIntro2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "StimIntro2" ---
        for thisComponent in StimIntro2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for StimIntro2
        StimIntro2.tStop = globalClock.getTime(format='float')
        StimIntro2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('StimIntro2.stopped', StimIntro2.tStop)
        # store data for trials_stimintro2 (TrialHandler)
        trials_stimintro2.addData('mouse_StimIntro2.x', mouse_StimIntro2.x)
        trials_stimintro2.addData('mouse_StimIntro2.y', mouse_StimIntro2.y)
        trials_stimintro2.addData('mouse_StimIntro2.leftButton', mouse_StimIntro2.leftButton)
        trials_stimintro2.addData('mouse_StimIntro2.midButton', mouse_StimIntro2.midButton)
        trials_stimintro2.addData('mouse_StimIntro2.rightButton', mouse_StimIntro2.rightButton)
        trials_stimintro2.addData('mouse_StimIntro2.time', mouse_StimIntro2.time)
        # check responses
        if key_resp_StimIntro2.keys in ['', [], None]:  # No response was made
            key_resp_StimIntro2.keys = None
        trials_stimintro2.addData('key_resp_StimIntro2.keys',key_resp_StimIntro2.keys)
        if key_resp_StimIntro2.keys != None:  # we had a response
            trials_stimintro2.addData('key_resp_StimIntro2.rt', key_resp_StimIntro2.rt)
            trials_stimintro2.addData('key_resp_StimIntro2.duration', key_resp_StimIntro2.duration)
        # the Routine "StimIntro2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_stimintro2 as finished
        if hasattr(thisTrials_stimintro2, 'status'):
            thisTrials_stimintro2.status = FINISHED
        # if awaiting a pause, pause now
        if trials_stimintro2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_stimintro2.status = STARTED
    # completed 1.0 repeats of 'trials_stimintro2'
    trials_stimintro2.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    trials_Intro2 = data.TrialHandler2(
        name='trials_Intro2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/Intro2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_Intro2)  # add the loop to the experiment
    thisTrials_Intro2 = trials_Intro2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_Intro2.rgb)
    if thisTrials_Intro2 != None:
        for paramName in thisTrials_Intro2:
            globals()[paramName] = thisTrials_Intro2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_Intro2 in trials_Intro2:
        trials_Intro2.status = STARTED
        if hasattr(thisTrials_Intro2, 'status'):
            thisTrials_Intro2.status = STARTED
        currentLoop = trials_Intro2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_Intro2.rgb)
        if thisTrials_Intro2 != None:
            for paramName in thisTrials_Intro2:
                globals()[paramName] = thisTrials_Intro2[paramName]
        
        # --- Prepare to start Routine "Intro" ---
        # create an object to store info about Routine Intro
        Intro = data.Routine(
            name='Intro',
            components=[text_Intro, mouse_Intro, key_resp_Intro, text_CTC_Intro],
        )
        Intro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_Intro
        trial_filename = './Sounds/'+IntroWav
        trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
        
        sd.play(Vol*trialdata, samplerate=trialfs)
        
        text_Intro.setText(IntroText)
        # setup some python lists for storing info about the mouse_Intro
        mouse_Intro.x = []
        mouse_Intro.y = []
        mouse_Intro.leftButton = []
        mouse_Intro.midButton = []
        mouse_Intro.rightButton = []
        mouse_Intro.time = []
        gotValidClick = False  # until a click is received
        # create starting attributes for key_resp_Intro
        key_resp_Intro.keys = []
        key_resp_Intro.rt = []
        _key_resp_Intro_allKeys = []
        # store start times for Intro
        Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Intro.tStart = globalClock.getTime(format='float')
        Intro.status = STARTED
        thisExp.addData('Intro.started', Intro.tStart)
        Intro.maxDuration = None
        # keep track of which components have finished
        IntroComponents = Intro.components
        for thisComponent in Intro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Intro" ---
        Intro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_Intro2, 'status') and thisTrials_Intro2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_Intro
            playing = sd.get_stream().active
            
            if playing:
                CTC_Opac = -1
            elif not playing:
                CTC_Opac = 1
            
            # *text_Intro* updates
            
            # if text_Intro is starting this frame...
            if text_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Intro.frameNStart = frameN  # exact frame index
                text_Intro.tStart = t  # local t and not account for scr refresh
                text_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_Intro.started')
                # update status
                text_Intro.status = STARTED
                text_Intro.setAutoDraw(True)
            
            # if text_Intro is active this frame...
            if text_Intro.status == STARTED:
                # update params
                pass
            # *mouse_Intro* updates
            
            # if mouse_Intro is starting this frame...
            if mouse_Intro.status == NOT_STARTED and t >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                mouse_Intro.frameNStart = frameN  # exact frame index
                mouse_Intro.tStart = t  # local t and not account for scr refresh
                mouse_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_Intro.started', t)
                # update status
                mouse_Intro.status = STARTED
                mouse_Intro.mouseClock.reset()
                prevButtonState = mouse_Intro.getPressed()  # if button is down already this ISN'T a new click
            if mouse_Intro.status == STARTED:  # only update if started and not finished!
                buttons = mouse_Intro.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_Intro.getPos()
                        mouse_Intro.x.append(x)
                        mouse_Intro.y.append(y)
                        buttons = mouse_Intro.getPressed()
                        mouse_Intro.leftButton.append(buttons[0])
                        mouse_Intro.midButton.append(buttons[1])
                        mouse_Intro.rightButton.append(buttons[2])
                        mouse_Intro.time.append(mouse_Intro.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *key_resp_Intro* updates
            waitOnFlip = False
            
            # if key_resp_Intro is starting this frame...
            if key_resp_Intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_Intro.frameNStart = frameN  # exact frame index
                key_resp_Intro.tStart = t  # local t and not account for scr refresh
                key_resp_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_Intro.started')
                # update status
                key_resp_Intro.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_Intro.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_Intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_Intro.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_Intro.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_Intro_allKeys.extend(theseKeys)
                if len(_key_resp_Intro_allKeys):
                    key_resp_Intro.keys = _key_resp_Intro_allKeys[-1].name  # just the last key pressed
                    key_resp_Intro.rt = _key_resp_Intro_allKeys[-1].rt
                    key_resp_Intro.duration = _key_resp_Intro_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_CTC_Intro* updates
            
            # if text_CTC_Intro is starting this frame...
            if text_CTC_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_CTC_Intro.frameNStart = frameN  # exact frame index
                text_CTC_Intro.tStart = t  # local t and not account for scr refresh
                text_CTC_Intro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_CTC_Intro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_CTC_Intro.started')
                # update status
                text_CTC_Intro.status = STARTED
                text_CTC_Intro.setAutoDraw(True)
            
            # if text_CTC_Intro is active this frame...
            if text_CTC_Intro.status == STARTED:
                # update params
                text_CTC_Intro.setOpacity(CTC_Opac, log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Intro,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Intro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Intro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Intro" ---
        for thisComponent in Intro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Intro
        Intro.tStop = globalClock.getTime(format='float')
        Intro.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Intro.stopped', Intro.tStop)
        
        # Run 'End Routine' code from code_Intro
        sd.stop()
        # store data for trials_Intro2 (TrialHandler)
        trials_Intro2.addData('mouse_Intro.x', mouse_Intro.x)
        trials_Intro2.addData('mouse_Intro.y', mouse_Intro.y)
        trials_Intro2.addData('mouse_Intro.leftButton', mouse_Intro.leftButton)
        trials_Intro2.addData('mouse_Intro.midButton', mouse_Intro.midButton)
        trials_Intro2.addData('mouse_Intro.rightButton', mouse_Intro.rightButton)
        trials_Intro2.addData('mouse_Intro.time', mouse_Intro.time)
        # check responses
        if key_resp_Intro.keys in ['', [], None]:  # No response was made
            key_resp_Intro.keys = None
        trials_Intro2.addData('key_resp_Intro.keys',key_resp_Intro.keys)
        if key_resp_Intro.keys != None:  # we had a response
            trials_Intro2.addData('key_resp_Intro.rt', key_resp_Intro.rt)
            trials_Intro2.addData('key_resp_Intro.duration', key_resp_Intro.duration)
        # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_Intro2 as finished
        if hasattr(thisTrials_Intro2, 'status'):
            thisTrials_Intro2.status = FINISHED
        # if awaiting a pause, pause now
        if trials_Intro2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_Intro2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_Intro2'
    trials_Intro2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "BorgRating_Intro" ---
    # create an object to store info about Routine BorgRating_Intro
    BorgRating_Intro = data.Routine(
        name='BorgRating_Intro',
        components=[VER_key_resp_2, image_BorgScale_Intro, text_BorgScale_Intro, mouse_BorgRating_Intro, text_CTC_BorgRating_Intro],
    )
    BorgRating_Intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_BorgRating_Intro
    trial_filename = './Sounds/BorgIntroA.wav'
    trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
    
    sd.play(trialdata, samplerate=trialfs)
    
    # create starting attributes for VER_key_resp_2
    VER_key_resp_2.keys = []
    VER_key_resp_2.rt = []
    _VER_key_resp_2_allKeys = []
    # setup some python lists for storing info about the mouse_BorgRating_Intro
    mouse_BorgRating_Intro.x = []
    mouse_BorgRating_Intro.y = []
    mouse_BorgRating_Intro.leftButton = []
    mouse_BorgRating_Intro.midButton = []
    mouse_BorgRating_Intro.rightButton = []
    mouse_BorgRating_Intro.time = []
    gotValidClick = False  # until a click is received
    # store start times for BorgRating_Intro
    BorgRating_Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BorgRating_Intro.tStart = globalClock.getTime(format='float')
    BorgRating_Intro.status = STARTED
    thisExp.addData('BorgRating_Intro.started', BorgRating_Intro.tStart)
    BorgRating_Intro.maxDuration = None
    # keep track of which components have finished
    BorgRating_IntroComponents = BorgRating_Intro.components
    for thisComponent in BorgRating_Intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BorgRating_Intro" ---
    BorgRating_Intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_BorgRating_Intro
        playing = sd.get_stream().active
        
        if playing:
            CTC_Opac = -1
        elif not playing:
            CTC_Opac = 1
        
        # *VER_key_resp_2* updates
        waitOnFlip = False
        
        # if VER_key_resp_2 is starting this frame...
        if VER_key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            VER_key_resp_2.frameNStart = frameN  # exact frame index
            VER_key_resp_2.tStart = t  # local t and not account for scr refresh
            VER_key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_key_resp_2.started')
            # update status
            VER_key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(VER_key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(VER_key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if VER_key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = VER_key_resp_2.getKeys(keyList=['return','y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _VER_key_resp_2_allKeys.extend(theseKeys)
            if len(_VER_key_resp_2_allKeys):
                VER_key_resp_2.keys = _VER_key_resp_2_allKeys[-1].name  # just the last key pressed
                VER_key_resp_2.rt = _VER_key_resp_2_allKeys[-1].rt
                VER_key_resp_2.duration = _VER_key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_BorgScale_Intro* updates
        
        # if image_BorgScale_Intro is starting this frame...
        if image_BorgScale_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            image_BorgScale_Intro.frameNStart = frameN  # exact frame index
            image_BorgScale_Intro.tStart = t  # local t and not account for scr refresh
            image_BorgScale_Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_BorgScale_Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_BorgScale_Intro.started')
            # update status
            image_BorgScale_Intro.status = STARTED
            image_BorgScale_Intro.setAutoDraw(True)
        
        # if image_BorgScale_Intro is active this frame...
        if image_BorgScale_Intro.status == STARTED:
            # update params
            pass
        
        # *text_BorgScale_Intro* updates
        
        # if text_BorgScale_Intro is starting this frame...
        if text_BorgScale_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_BorgScale_Intro.frameNStart = frameN  # exact frame index
            text_BorgScale_Intro.tStart = t  # local t and not account for scr refresh
            text_BorgScale_Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_BorgScale_Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_BorgScale_Intro.started')
            # update status
            text_BorgScale_Intro.status = STARTED
            text_BorgScale_Intro.setAutoDraw(True)
        
        # if text_BorgScale_Intro is active this frame...
        if text_BorgScale_Intro.status == STARTED:
            # update params
            pass
        # *mouse_BorgRating_Intro* updates
        
        # if mouse_BorgRating_Intro is starting this frame...
        if mouse_BorgRating_Intro.status == NOT_STARTED and t >= 1-frameTolerance:
            # keep track of start time/frame for later
            mouse_BorgRating_Intro.frameNStart = frameN  # exact frame index
            mouse_BorgRating_Intro.tStart = t  # local t and not account for scr refresh
            mouse_BorgRating_Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_BorgRating_Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_BorgRating_Intro.started', t)
            # update status
            mouse_BorgRating_Intro.status = STARTED
            mouse_BorgRating_Intro.mouseClock.reset()
            prevButtonState = mouse_BorgRating_Intro.getPressed()  # if button is down already this ISN'T a new click
        if mouse_BorgRating_Intro.status == STARTED:  # only update if started and not finished!
            buttons = mouse_BorgRating_Intro.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_BorgRating_Intro.getPos()
                    mouse_BorgRating_Intro.x.append(x)
                    mouse_BorgRating_Intro.y.append(y)
                    buttons = mouse_BorgRating_Intro.getPressed()
                    mouse_BorgRating_Intro.leftButton.append(buttons[0])
                    mouse_BorgRating_Intro.midButton.append(buttons[1])
                    mouse_BorgRating_Intro.rightButton.append(buttons[2])
                    mouse_BorgRating_Intro.time.append(mouse_BorgRating_Intro.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *text_CTC_BorgRating_Intro* updates
        
        # if text_CTC_BorgRating_Intro is starting this frame...
        if text_CTC_BorgRating_Intro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_CTC_BorgRating_Intro.frameNStart = frameN  # exact frame index
            text_CTC_BorgRating_Intro.tStart = t  # local t and not account for scr refresh
            text_CTC_BorgRating_Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_CTC_BorgRating_Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_CTC_BorgRating_Intro.started')
            # update status
            text_CTC_BorgRating_Intro.status = STARTED
            text_CTC_BorgRating_Intro.setAutoDraw(True)
        
        # if text_CTC_BorgRating_Intro is active this frame...
        if text_CTC_BorgRating_Intro.status == STARTED:
            # update params
            text_CTC_BorgRating_Intro.setOpacity(CTC_Opac, log=False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BorgRating_Intro,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BorgRating_Intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BorgRating_Intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BorgRating_Intro" ---
    for thisComponent in BorgRating_Intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BorgRating_Intro
    BorgRating_Intro.tStop = globalClock.getTime(format='float')
    BorgRating_Intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BorgRating_Intro.stopped', BorgRating_Intro.tStop)
    # Run 'End Routine' code from code_BorgRating_Intro
    sd.stop()
    # check responses
    if VER_key_resp_2.keys in ['', [], None]:  # No response was made
        VER_key_resp_2.keys = None
    thisExp.addData('VER_key_resp_2.keys',VER_key_resp_2.keys)
    if VER_key_resp_2.keys != None:  # we had a response
        thisExp.addData('VER_key_resp_2.rt', VER_key_resp_2.rt)
        thisExp.addData('VER_key_resp_2.duration', VER_key_resp_2.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_BorgRating_Intro.x', mouse_BorgRating_Intro.x)
    thisExp.addData('mouse_BorgRating_Intro.y', mouse_BorgRating_Intro.y)
    thisExp.addData('mouse_BorgRating_Intro.leftButton', mouse_BorgRating_Intro.leftButton)
    thisExp.addData('mouse_BorgRating_Intro.midButton', mouse_BorgRating_Intro.midButton)
    thisExp.addData('mouse_BorgRating_Intro.rightButton', mouse_BorgRating_Intro.rightButton)
    thisExp.addData('mouse_BorgRating_Intro.time', mouse_BorgRating_Intro.time)
    thisExp.nextEntry()
    # the Routine "BorgRating_Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "BorgRating_Intro_2" ---
    # create an object to store info about Routine BorgRating_Intro_2
    BorgRating_Intro_2 = data.Routine(
        name='BorgRating_Intro_2',
        components=[VER_key_resp_3, image_BorgScale_Intro_2, text_BorgScale_Intro_2, mouse_BorgRating_Intro_2, text_CTC_BorgRating_Intro_2],
    )
    BorgRating_Intro_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_BorgRating_Intro_2
    trial_filename = './Sounds/BorgIntroB.wav'
    trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
    
    sd.play(trialdata, samplerate=trialfs)
    
    # create starting attributes for VER_key_resp_3
    VER_key_resp_3.keys = []
    VER_key_resp_3.rt = []
    _VER_key_resp_3_allKeys = []
    # setup some python lists for storing info about the mouse_BorgRating_Intro_2
    mouse_BorgRating_Intro_2.x = []
    mouse_BorgRating_Intro_2.y = []
    mouse_BorgRating_Intro_2.leftButton = []
    mouse_BorgRating_Intro_2.midButton = []
    mouse_BorgRating_Intro_2.rightButton = []
    mouse_BorgRating_Intro_2.time = []
    gotValidClick = False  # until a click is received
    # store start times for BorgRating_Intro_2
    BorgRating_Intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BorgRating_Intro_2.tStart = globalClock.getTime(format='float')
    BorgRating_Intro_2.status = STARTED
    thisExp.addData('BorgRating_Intro_2.started', BorgRating_Intro_2.tStart)
    BorgRating_Intro_2.maxDuration = None
    # keep track of which components have finished
    BorgRating_Intro_2Components = BorgRating_Intro_2.components
    for thisComponent in BorgRating_Intro_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BorgRating_Intro_2" ---
    BorgRating_Intro_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_BorgRating_Intro_2
        playing = sd.get_stream().active
        
        if playing:
            CTC_Opac = -1
        elif not playing:
            CTC_Opac = 1
        
        # *VER_key_resp_3* updates
        waitOnFlip = False
        
        # if VER_key_resp_3 is starting this frame...
        if VER_key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            VER_key_resp_3.frameNStart = frameN  # exact frame index
            VER_key_resp_3.tStart = t  # local t and not account for scr refresh
            VER_key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_key_resp_3.started')
            # update status
            VER_key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(VER_key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(VER_key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if VER_key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = VER_key_resp_3.getKeys(keyList=['return','y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _VER_key_resp_3_allKeys.extend(theseKeys)
            if len(_VER_key_resp_3_allKeys):
                VER_key_resp_3.keys = _VER_key_resp_3_allKeys[-1].name  # just the last key pressed
                VER_key_resp_3.rt = _VER_key_resp_3_allKeys[-1].rt
                VER_key_resp_3.duration = _VER_key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image_BorgScale_Intro_2* updates
        
        # if image_BorgScale_Intro_2 is starting this frame...
        if image_BorgScale_Intro_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            image_BorgScale_Intro_2.frameNStart = frameN  # exact frame index
            image_BorgScale_Intro_2.tStart = t  # local t and not account for scr refresh
            image_BorgScale_Intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_BorgScale_Intro_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_BorgScale_Intro_2.started')
            # update status
            image_BorgScale_Intro_2.status = STARTED
            image_BorgScale_Intro_2.setAutoDraw(True)
        
        # if image_BorgScale_Intro_2 is active this frame...
        if image_BorgScale_Intro_2.status == STARTED:
            # update params
            pass
        
        # *text_BorgScale_Intro_2* updates
        
        # if text_BorgScale_Intro_2 is starting this frame...
        if text_BorgScale_Intro_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_BorgScale_Intro_2.frameNStart = frameN  # exact frame index
            text_BorgScale_Intro_2.tStart = t  # local t and not account for scr refresh
            text_BorgScale_Intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_BorgScale_Intro_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_BorgScale_Intro_2.started')
            # update status
            text_BorgScale_Intro_2.status = STARTED
            text_BorgScale_Intro_2.setAutoDraw(True)
        
        # if text_BorgScale_Intro_2 is active this frame...
        if text_BorgScale_Intro_2.status == STARTED:
            # update params
            pass
        # *mouse_BorgRating_Intro_2* updates
        
        # if mouse_BorgRating_Intro_2 is starting this frame...
        if mouse_BorgRating_Intro_2.status == NOT_STARTED and t >= 1-frameTolerance:
            # keep track of start time/frame for later
            mouse_BorgRating_Intro_2.frameNStart = frameN  # exact frame index
            mouse_BorgRating_Intro_2.tStart = t  # local t and not account for scr refresh
            mouse_BorgRating_Intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_BorgRating_Intro_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_BorgRating_Intro_2.started', t)
            # update status
            mouse_BorgRating_Intro_2.status = STARTED
            mouse_BorgRating_Intro_2.mouseClock.reset()
            prevButtonState = mouse_BorgRating_Intro_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_BorgRating_Intro_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_BorgRating_Intro_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_BorgRating_Intro_2.getPos()
                    mouse_BorgRating_Intro_2.x.append(x)
                    mouse_BorgRating_Intro_2.y.append(y)
                    buttons = mouse_BorgRating_Intro_2.getPressed()
                    mouse_BorgRating_Intro_2.leftButton.append(buttons[0])
                    mouse_BorgRating_Intro_2.midButton.append(buttons[1])
                    mouse_BorgRating_Intro_2.rightButton.append(buttons[2])
                    mouse_BorgRating_Intro_2.time.append(mouse_BorgRating_Intro_2.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *text_CTC_BorgRating_Intro_2* updates
        
        # if text_CTC_BorgRating_Intro_2 is starting this frame...
        if text_CTC_BorgRating_Intro_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_CTC_BorgRating_Intro_2.frameNStart = frameN  # exact frame index
            text_CTC_BorgRating_Intro_2.tStart = t  # local t and not account for scr refresh
            text_CTC_BorgRating_Intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_CTC_BorgRating_Intro_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_CTC_BorgRating_Intro_2.started')
            # update status
            text_CTC_BorgRating_Intro_2.status = STARTED
            text_CTC_BorgRating_Intro_2.setAutoDraw(True)
        
        # if text_CTC_BorgRating_Intro_2 is active this frame...
        if text_CTC_BorgRating_Intro_2.status == STARTED:
            # update params
            text_CTC_BorgRating_Intro_2.setOpacity(CTC_Opac, log=False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BorgRating_Intro_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BorgRating_Intro_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BorgRating_Intro_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BorgRating_Intro_2" ---
    for thisComponent in BorgRating_Intro_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BorgRating_Intro_2
    BorgRating_Intro_2.tStop = globalClock.getTime(format='float')
    BorgRating_Intro_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BorgRating_Intro_2.stopped', BorgRating_Intro_2.tStop)
    # Run 'End Routine' code from code_BorgRating_Intro_2
    sd.stop()
    # check responses
    if VER_key_resp_3.keys in ['', [], None]:  # No response was made
        VER_key_resp_3.keys = None
    thisExp.addData('VER_key_resp_3.keys',VER_key_resp_3.keys)
    if VER_key_resp_3.keys != None:  # we had a response
        thisExp.addData('VER_key_resp_3.rt', VER_key_resp_3.rt)
        thisExp.addData('VER_key_resp_3.duration', VER_key_resp_3.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_BorgRating_Intro_2.x', mouse_BorgRating_Intro_2.x)
    thisExp.addData('mouse_BorgRating_Intro_2.y', mouse_BorgRating_Intro_2.y)
    thisExp.addData('mouse_BorgRating_Intro_2.leftButton', mouse_BorgRating_Intro_2.leftButton)
    thisExp.addData('mouse_BorgRating_Intro_2.midButton', mouse_BorgRating_Intro_2.midButton)
    thisExp.addData('mouse_BorgRating_Intro_2.rightButton', mouse_BorgRating_Intro_2.rightButton)
    thisExp.addData('mouse_BorgRating_Intro_2.time', mouse_BorgRating_Intro_2.time)
    thisExp.nextEntry()
    # the Routine "BorgRating_Intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "NoiseTest" ---
    # create an object to store info about Routine NoiseTest
    NoiseTest = data.Routine(
        name='NoiseTest',
        components=[text_NoiseTest, key_resp, mouse_NoiseTest, text_CTC_NoiseTest],
    )
    NoiseTest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_Noise
    sd.play(noisedata, samplerate=noisefs)
    
    CTC_Opac = 1
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # setup some python lists for storing info about the mouse_NoiseTest
    mouse_NoiseTest.x = []
    mouse_NoiseTest.y = []
    mouse_NoiseTest.leftButton = []
    mouse_NoiseTest.midButton = []
    mouse_NoiseTest.rightButton = []
    mouse_NoiseTest.time = []
    gotValidClick = False  # until a click is received
    # store start times for NoiseTest
    NoiseTest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    NoiseTest.tStart = globalClock.getTime(format='float')
    NoiseTest.status = STARTED
    thisExp.addData('NoiseTest.started', NoiseTest.tStart)
    NoiseTest.maxDuration = None
    # keep track of which components have finished
    NoiseTestComponents = NoiseTest.components
    for thisComponent in NoiseTest.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "NoiseTest" ---
    NoiseTest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_NoiseTest* updates
        
        # if text_NoiseTest is starting this frame...
        if text_NoiseTest.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_NoiseTest.frameNStart = frameN  # exact frame index
            text_NoiseTest.tStart = t  # local t and not account for scr refresh
            text_NoiseTest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_NoiseTest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_NoiseTest.started')
            # update status
            text_NoiseTest.status = STARTED
            text_NoiseTest.setAutoDraw(True)
        
        # if text_NoiseTest is active this frame...
        if text_NoiseTest.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # *mouse_NoiseTest* updates
        
        # if mouse_NoiseTest is starting this frame...
        if mouse_NoiseTest.status == NOT_STARTED and t >= 1-frameTolerance:
            # keep track of start time/frame for later
            mouse_NoiseTest.frameNStart = frameN  # exact frame index
            mouse_NoiseTest.tStart = t  # local t and not account for scr refresh
            mouse_NoiseTest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_NoiseTest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_NoiseTest.started', t)
            # update status
            mouse_NoiseTest.status = STARTED
            mouse_NoiseTest.mouseClock.reset()
            prevButtonState = mouse_NoiseTest.getPressed()  # if button is down already this ISN'T a new click
        if mouse_NoiseTest.status == STARTED:  # only update if started and not finished!
            buttons = mouse_NoiseTest.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_NoiseTest.getPos()
                    mouse_NoiseTest.x.append(x)
                    mouse_NoiseTest.y.append(y)
                    buttons = mouse_NoiseTest.getPressed()
                    mouse_NoiseTest.leftButton.append(buttons[0])
                    mouse_NoiseTest.midButton.append(buttons[1])
                    mouse_NoiseTest.rightButton.append(buttons[2])
                    mouse_NoiseTest.time.append(mouse_NoiseTest.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *text_CTC_NoiseTest* updates
        
        # if text_CTC_NoiseTest is starting this frame...
        if text_CTC_NoiseTest.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_CTC_NoiseTest.frameNStart = frameN  # exact frame index
            text_CTC_NoiseTest.tStart = t  # local t and not account for scr refresh
            text_CTC_NoiseTest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_CTC_NoiseTest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_CTC_NoiseTest.started')
            # update status
            text_CTC_NoiseTest.status = STARTED
            text_CTC_NoiseTest.setAutoDraw(True)
        
        # if text_CTC_NoiseTest is active this frame...
        if text_CTC_NoiseTest.status == STARTED:
            # update params
            text_CTC_NoiseTest.setOpacity(CTC_Opac, log=False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=NoiseTest,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            NoiseTest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in NoiseTest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "NoiseTest" ---
    for thisComponent in NoiseTest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for NoiseTest
    NoiseTest.tStop = globalClock.getTime(format='float')
    NoiseTest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('NoiseTest.stopped', NoiseTest.tStop)
    # Run 'End Routine' code from code_Noise
    sd.stop()
    CTC_Opac = -1
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_NoiseTest.x', mouse_NoiseTest.x)
    thisExp.addData('mouse_NoiseTest.y', mouse_NoiseTest.y)
    thisExp.addData('mouse_NoiseTest.leftButton', mouse_NoiseTest.leftButton)
    thisExp.addData('mouse_NoiseTest.midButton', mouse_NoiseTest.midButton)
    thisExp.addData('mouse_NoiseTest.rightButton', mouse_NoiseTest.rightButton)
    thisExp.addData('mouse_NoiseTest.time', mouse_NoiseTest.time)
    thisExp.nextEntry()
    # the Routine "NoiseTest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "BorgRating" ---
    # create an object to store info about Routine BorgRating
    BorgRating = data.Routine(
        name='BorgRating',
        components=[VER_key_resp, image_BorgScale, text_BorgScale, VER_num_text, mouse_Borg, button_Borg, polygon_1, polygon_2, polygon_3, polygon_4, polygon_5, polygon_6, polygon_7, polygon_8, polygon_9, polygon_Back, polygon_0, polygon_Dot, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_Back, text_0, text_Dot],
    )
    BorgRating.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    inputText = '#'  # Set Routine start values for inputText
    # create starting attributes for VER_key_resp
    VER_key_resp.keys = []
    VER_key_resp.rt = []
    _VER_key_resp_allKeys = []
    # Run 'Begin Routine' code from code_Borg
    #sd.stop()
    
    theseKeys=""
    VER_num_text.alignHoriz ='left'
    first_press = True
    
    # setup some python lists for storing info about the mouse_Borg
    mouse_Borg.x = []
    mouse_Borg.y = []
    mouse_Borg.leftButton = []
    mouse_Borg.midButton = []
    mouse_Borg.rightButton = []
    mouse_Borg.time = []
    gotValidClick = False  # until a click is received
    # reset button_Borg to account for continued clicks & clear times on/off
    button_Borg.reset()
    # store start times for BorgRating
    BorgRating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BorgRating.tStart = globalClock.getTime(format='float')
    BorgRating.status = STARTED
    thisExp.addData('BorgRating.started', BorgRating.tStart)
    BorgRating.maxDuration = None
    # keep track of which components have finished
    BorgRatingComponents = BorgRating.components
    for thisComponent in BorgRating.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BorgRating" ---
    BorgRating.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *VER_key_resp* updates
        waitOnFlip = False
        
        # if VER_key_resp is starting this frame...
        if VER_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            VER_key_resp.frameNStart = frameN  # exact frame index
            VER_key_resp.tStart = t  # local t and not account for scr refresh
            VER_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_key_resp.started')
            # update status
            VER_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(VER_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(VER_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if VER_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = VER_key_resp.getKeys(keyList=['return','y','n','left','right','space','1','2','3','4','5','6','7','8','9','0','period','return','backspace','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9','num_0','num_decimal','num_subtract','period','comma'], ignoreKeys=["escape"], waitRelease=False)
            _VER_key_resp_allKeys.extend(theseKeys)
            if len(_VER_key_resp_allKeys):
                VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                VER_key_resp.duration = _VER_key_resp_allKeys[-1].duration
        # Run 'Each Frame' code from code_Borg
        if mouse_Borg.isPressedIn(polygon_1):
            NumResponse = '1'
        if mouse_Borg.isPressedIn(polygon_2):
            NumResponse = '2'
        if mouse_Borg.isPressedIn(polygon_3):
            NumResponse = '3'
        if mouse_Borg.isPressedIn(polygon_4):
            NumResponse = '4'
        if mouse_Borg.isPressedIn(polygon_5):
            NumResponse = '5'
        if mouse_Borg.isPressedIn(polygon_6):
            NumResponse = '6'
        if mouse_Borg.isPressedIn(polygon_7):
            NumResponse = '7'
        if mouse_Borg.isPressedIn(polygon_8):
            NumResponse = '8'
        if mouse_Borg.isPressedIn(polygon_9):
            NumResponse = '9'
        if mouse_Borg.isPressedIn(polygon_0):
            NumResponse = '0'
        if mouse_Borg.isPressedIn(polygon_Dot):
            NumResponse = ''
            if len(inputText) == 0:
                inputText += '0.'
            else:
                inputText += '.'
            time.sleep(.1)
        
        if NumResponse != '':
            if inputText == '#':
                inputText = NumResponse
            else:
                inputText += NumResponse
            NumResponse = ''
            time.sleep(.1)
            
        if mouse_Borg.isPressedIn(polygon_Back):
            inputText = inputText[:-1]  # lose the final character
            time.sleep(.1)
        
        if len(_VER_key_resp_allKeys) and keyReady:
            keyReady = False
            VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
            VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
            if first_press:
                inputText = ''
                first_press = False
        
            elif VER_key_resp.keys in ['right','space']:
                continueRoutine = False
                
            elif VER_key_resp.keys in ['num_1', '1']:
                inputText += '1'
        
            elif VER_key_resp.keys in ['num_2', '2']:
                inputText += '2'
        
            elif VER_key_resp.keys in ['num_3', '3']:
                inputText += '3'
        
            elif VER_key_resp.keys in ['num_4', '4']:
                inputText += '4'
        
            elif VER_key_resp.keys in ['num_5', '5']:
                inputText += '5'
        
            elif VER_key_resp.keys in ['num_6', '6']:
                inputText += '6'
        
            elif VER_key_resp.keys in ['num_7', '7']:
                inputText += '7'
        
            elif VER_key_resp.keys in ['num_8', '8']:
                inputText += '8'
        
            elif VER_key_resp.keys in ['num_9', '9']:
                inputText += '9'
        
            elif VER_key_resp.keys in ['num_0', '0']:
                inputText += '0'
        
            elif VER_key_resp.keys in ['period', 'comma','num_decimal']:
                if len(inputText) == 0:
                    inputText += '0,'
                else:
                    inputText += '.'
        
            elif VER_key_resp.keys in ['backspace','num_subtract']:
                inputText = inputText[:-1]  # lose the final character
        
            #VER_key_resp = []
            _VER_key_resp_allKeys = []
            theseKeys = []
        elif len(_VER_key_resp_allKeys) == 0:
            keyReady = True
        
        # *image_BorgScale* updates
        
        # if image_BorgScale is starting this frame...
        if image_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            image_BorgScale.frameNStart = frameN  # exact frame index
            image_BorgScale.tStart = t  # local t and not account for scr refresh
            image_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_BorgScale, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_BorgScale.started')
            # update status
            image_BorgScale.status = STARTED
            image_BorgScale.setAutoDraw(True)
        
        # if image_BorgScale is active this frame...
        if image_BorgScale.status == STARTED:
            # update params
            pass
        
        # *text_BorgScale* updates
        
        # if text_BorgScale is starting this frame...
        if text_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_BorgScale.frameNStart = frameN  # exact frame index
            text_BorgScale.tStart = t  # local t and not account for scr refresh
            text_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_BorgScale, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_BorgScale.started')
            # update status
            text_BorgScale.status = STARTED
            text_BorgScale.setAutoDraw(True)
        
        # if text_BorgScale is active this frame...
        if text_BorgScale.status == STARTED:
            # update params
            pass
        
        # *VER_num_text* updates
        
        # if VER_num_text is starting this frame...
        if VER_num_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            VER_num_text.frameNStart = frameN  # exact frame index
            VER_num_text.tStart = t  # local t and not account for scr refresh
            VER_num_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_num_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_num_text.started')
            # update status
            VER_num_text.status = STARTED
            VER_num_text.setAutoDraw(True)
        
        # if VER_num_text is active this frame...
        if VER_num_text.status == STARTED:
            # update params
            VER_num_text.setText(inputText, log=False)
        # *mouse_Borg* updates
        if mouse_Borg.status == STARTED:  # only update if started and not finished!
            buttons = mouse_Borg.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_Borg.getPos()
                    mouse_Borg.x.append(x)
                    mouse_Borg.y.append(y)
                    buttons = mouse_Borg.getPressed()
                    mouse_Borg.leftButton.append(buttons[0])
                    mouse_Borg.midButton.append(buttons[1])
                    mouse_Borg.rightButton.append(buttons[2])
                    mouse_Borg.time.append(mouse_Borg.mouseClock.getTime())
        # *button_Borg* updates
        
        # if button_Borg is starting this frame...
        if button_Borg.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            button_Borg.frameNStart = frameN  # exact frame index
            button_Borg.tStart = t  # local t and not account for scr refresh
            button_Borg.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_Borg, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_Borg.started')
            # update status
            button_Borg.status = STARTED
            win.callOnFlip(button_Borg.buttonClock.reset)
            button_Borg.setAutoDraw(True)
        
        # if button_Borg is active this frame...
        if button_Borg.status == STARTED:
            # update params
            pass
            # check whether button_Borg has been pressed
            if button_Borg.isClicked:
                if not button_Borg.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_Borg.timesOn.append(button_Borg.buttonClock.getTime())
                    button_Borg.timesOff.append(button_Borg.buttonClock.getTime())
                elif len(button_Borg.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_Borg.timesOff[-1] = button_Borg.buttonClock.getTime()
                if not button_Borg.wasClicked:
                    # run callback code when button_Borg is clicked
                    if inputText != '' and inputText != '#':
                        continueRoutine = False
                        break
        # take note of whether button_Borg was clicked, so that next frame we know if clicks are new
        button_Borg.wasClicked = button_Borg.isClicked and button_Borg.status == STARTED
        
        # *polygon_1* updates
        
        # if polygon_1 is starting this frame...
        if polygon_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_1.frameNStart = frameN  # exact frame index
            polygon_1.tStart = t  # local t and not account for scr refresh
            polygon_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_1.started')
            # update status
            polygon_1.status = STARTED
            polygon_1.setAutoDraw(True)
        
        # if polygon_1 is active this frame...
        if polygon_1.status == STARTED:
            # update params
            pass
        
        # *polygon_2* updates
        
        # if polygon_2 is starting this frame...
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_2.started')
            # update status
            polygon_2.status = STARTED
            polygon_2.setAutoDraw(True)
        
        # if polygon_2 is active this frame...
        if polygon_2.status == STARTED:
            # update params
            pass
        
        # *polygon_3* updates
        
        # if polygon_3 is starting this frame...
        if polygon_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_3.frameNStart = frameN  # exact frame index
            polygon_3.tStart = t  # local t and not account for scr refresh
            polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_3.started')
            # update status
            polygon_3.status = STARTED
            polygon_3.setAutoDraw(True)
        
        # if polygon_3 is active this frame...
        if polygon_3.status == STARTED:
            # update params
            pass
        
        # *polygon_4* updates
        
        # if polygon_4 is starting this frame...
        if polygon_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_4.frameNStart = frameN  # exact frame index
            polygon_4.tStart = t  # local t and not account for scr refresh
            polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_4.started')
            # update status
            polygon_4.status = STARTED
            polygon_4.setAutoDraw(True)
        
        # if polygon_4 is active this frame...
        if polygon_4.status == STARTED:
            # update params
            pass
        
        # *polygon_5* updates
        
        # if polygon_5 is starting this frame...
        if polygon_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_5.frameNStart = frameN  # exact frame index
            polygon_5.tStart = t  # local t and not account for scr refresh
            polygon_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_5.started')
            # update status
            polygon_5.status = STARTED
            polygon_5.setAutoDraw(True)
        
        # if polygon_5 is active this frame...
        if polygon_5.status == STARTED:
            # update params
            pass
        
        # *polygon_6* updates
        
        # if polygon_6 is starting this frame...
        if polygon_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_6.frameNStart = frameN  # exact frame index
            polygon_6.tStart = t  # local t and not account for scr refresh
            polygon_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_6.started')
            # update status
            polygon_6.status = STARTED
            polygon_6.setAutoDraw(True)
        
        # if polygon_6 is active this frame...
        if polygon_6.status == STARTED:
            # update params
            pass
        
        # *polygon_7* updates
        
        # if polygon_7 is starting this frame...
        if polygon_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_7.frameNStart = frameN  # exact frame index
            polygon_7.tStart = t  # local t and not account for scr refresh
            polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_7.started')
            # update status
            polygon_7.status = STARTED
            polygon_7.setAutoDraw(True)
        
        # if polygon_7 is active this frame...
        if polygon_7.status == STARTED:
            # update params
            pass
        
        # *polygon_8* updates
        
        # if polygon_8 is starting this frame...
        if polygon_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_8.frameNStart = frameN  # exact frame index
            polygon_8.tStart = t  # local t and not account for scr refresh
            polygon_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_8.started')
            # update status
            polygon_8.status = STARTED
            polygon_8.setAutoDraw(True)
        
        # if polygon_8 is active this frame...
        if polygon_8.status == STARTED:
            # update params
            pass
        
        # *polygon_9* updates
        
        # if polygon_9 is starting this frame...
        if polygon_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_9.frameNStart = frameN  # exact frame index
            polygon_9.tStart = t  # local t and not account for scr refresh
            polygon_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_9.started')
            # update status
            polygon_9.status = STARTED
            polygon_9.setAutoDraw(True)
        
        # if polygon_9 is active this frame...
        if polygon_9.status == STARTED:
            # update params
            pass
        
        # *polygon_Back* updates
        
        # if polygon_Back is starting this frame...
        if polygon_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_Back.frameNStart = frameN  # exact frame index
            polygon_Back.tStart = t  # local t and not account for scr refresh
            polygon_Back.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_Back, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_Back.started')
            # update status
            polygon_Back.status = STARTED
            polygon_Back.setAutoDraw(True)
        
        # if polygon_Back is active this frame...
        if polygon_Back.status == STARTED:
            # update params
            pass
        
        # *polygon_0* updates
        
        # if polygon_0 is starting this frame...
        if polygon_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_0.frameNStart = frameN  # exact frame index
            polygon_0.tStart = t  # local t and not account for scr refresh
            polygon_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_0.started')
            # update status
            polygon_0.status = STARTED
            polygon_0.setAutoDraw(True)
        
        # if polygon_0 is active this frame...
        if polygon_0.status == STARTED:
            # update params
            pass
        
        # *polygon_Dot* updates
        
        # if polygon_Dot is starting this frame...
        if polygon_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_Dot.frameNStart = frameN  # exact frame index
            polygon_Dot.tStart = t  # local t and not account for scr refresh
            polygon_Dot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_Dot, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_Dot.started')
            # update status
            polygon_Dot.status = STARTED
            polygon_Dot.setAutoDraw(True)
        
        # if polygon_Dot is active this frame...
        if polygon_Dot.status == STARTED:
            # update params
            pass
        
        # *text_1* updates
        
        # if text_1 is starting this frame...
        if text_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_1.frameNStart = frameN  # exact frame index
            text_1.tStart = t  # local t and not account for scr refresh
            text_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_1.status = STARTED
            text_1.setAutoDraw(True)
        
        # if text_1 is active this frame...
        if text_1.status == STARTED:
            # update params
            pass
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # *text_8* updates
        
        # if text_8 is starting this frame...
        if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_8.frameNStart = frameN  # exact frame index
            text_8.tStart = t  # local t and not account for scr refresh
            text_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_8.status = STARTED
            text_8.setAutoDraw(True)
        
        # if text_8 is active this frame...
        if text_8.status == STARTED:
            # update params
            pass
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *text_Back* updates
        
        # if text_Back is starting this frame...
        if text_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Back.frameNStart = frameN  # exact frame index
            text_Back.tStart = t  # local t and not account for scr refresh
            text_Back.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Back, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_Back.status = STARTED
            text_Back.setAutoDraw(True)
        
        # if text_Back is active this frame...
        if text_Back.status == STARTED:
            # update params
            pass
        
        # *text_0* updates
        
        # if text_0 is starting this frame...
        if text_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_0.frameNStart = frameN  # exact frame index
            text_0.tStart = t  # local t and not account for scr refresh
            text_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_0, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_0.status = STARTED
            text_0.setAutoDraw(True)
        
        # if text_0 is active this frame...
        if text_0.status == STARTED:
            # update params
            pass
        
        # *text_Dot* updates
        
        # if text_Dot is starting this frame...
        if text_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Dot.frameNStart = frameN  # exact frame index
            text_Dot.tStart = t  # local t and not account for scr refresh
            text_Dot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Dot, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_Dot.status = STARTED
            text_Dot.setAutoDraw(True)
        
        # if text_Dot is active this frame...
        if text_Dot.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BorgRating,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BorgRating.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BorgRating.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BorgRating" ---
    for thisComponent in BorgRating.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BorgRating
    BorgRating.tStop = globalClock.getTime(format='float')
    BorgRating.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BorgRating.stopped', BorgRating.tStop)
    thisExp.addData('inputText.routineEndVal', inputText)  # Save end Routine value
    # check responses
    if VER_key_resp.keys in ['', [], None]:  # No response was made
        VER_key_resp.keys = None
    thisExp.addData('VER_key_resp.keys',VER_key_resp.keys)
    if VER_key_resp.keys != None:  # we had a response
        thisExp.addData('VER_key_resp.rt', VER_key_resp.rt)
        thisExp.addData('VER_key_resp.duration', VER_key_resp.duration)
    # Run 'End Routine' code from code_Borg
    #sd.stop()
    
    # Load the CSV file
    with open(filename+'_Ratings.csv', mode='r') as file:
        reader = csv.reader(file)
        segData = [row for row in reader]
    
    VER_num += 1
    
    new_row = ['VER_'+str(VER_num), inputText]
    segData.append(new_row)
    
    # Save the modified data back to the CSV file
    with open(filename+'_Ratings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(segData)
    
    
    # let's store the final text string into the results finle...
    thisExp.addData('VER', inputText)
    inputText="#"
    thisExp.addData('Task','VER')
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_Borg.x', mouse_Borg.x)
    thisExp.addData('mouse_Borg.y', mouse_Borg.y)
    thisExp.addData('mouse_Borg.leftButton', mouse_Borg.leftButton)
    thisExp.addData('mouse_Borg.midButton', mouse_Borg.midButton)
    thisExp.addData('mouse_Borg.rightButton', mouse_Borg.rightButton)
    thisExp.addData('mouse_Borg.time', mouse_Borg.time)
    thisExp.addData('button_Borg.numClicks', button_Borg.numClicks)
    if button_Borg.numClicks:
       thisExp.addData('button_Borg.timesOn', button_Borg.timesOn)
       thisExp.addData('button_Borg.timesOff', button_Borg.timesOff)
    else:
       thisExp.addData('button_Borg.timesOn', "")
       thisExp.addData('button_Borg.timesOff', "")
    thisExp.nextEntry()
    # the Routine "BorgRating" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Begin" ---
    # create an object to store info about Routine Begin
    Begin = data.Routine(
        name='Begin',
        components=[text_Begin, key_resp_Begin],
    )
    Begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_Begin
    trial_filename = './Sounds/Begin.wav'
    trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
    
    sd.play(trialdata, samplerate=trialfs)
    
    # create starting attributes for key_resp_Begin
    key_resp_Begin.keys = []
    key_resp_Begin.rt = []
    _key_resp_Begin_allKeys = []
    # store start times for Begin
    Begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Begin.tStart = globalClock.getTime(format='float')
    Begin.status = STARTED
    thisExp.addData('Begin.started', Begin.tStart)
    Begin.maxDuration = None
    # keep track of which components have finished
    BeginComponents = Begin.components
    for thisComponent in Begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Begin" ---
    Begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_Begin
        playing = sd.get_stream().active
        
        if playing:
            CTC_Opac = -1
        elif not playing:
            CTC_Opac = 1
        
        # *text_Begin* updates
        
        # if text_Begin is starting this frame...
        if text_Begin.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Begin.frameNStart = frameN  # exact frame index
            text_Begin.tStart = t  # local t and not account for scr refresh
            text_Begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Begin, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_Begin.started')
            # update status
            text_Begin.status = STARTED
            text_Begin.setAutoDraw(True)
        
        # if text_Begin is active this frame...
        if text_Begin.status == STARTED:
            # update params
            pass
        
        # *key_resp_Begin* updates
        waitOnFlip = False
        
        # if key_resp_Begin is starting this frame...
        if key_resp_Begin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_Begin.frameNStart = frameN  # exact frame index
            key_resp_Begin.tStart = t  # local t and not account for scr refresh
            key_resp_Begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_Begin, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_Begin.started')
            # update status
            key_resp_Begin.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_Begin.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_Begin.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_Begin.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_Begin.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_Begin_allKeys.extend(theseKeys)
            if len(_key_resp_Begin_allKeys):
                key_resp_Begin.keys = _key_resp_Begin_allKeys[-1].name  # just the last key pressed
                key_resp_Begin.rt = _key_resp_Begin_allKeys[-1].rt
                key_resp_Begin.duration = _key_resp_Begin_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Begin,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Begin" ---
    for thisComponent in Begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Begin
    Begin.tStop = globalClock.getTime(format='float')
    Begin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Begin.stopped', Begin.tStop)
    # Run 'End Routine' code from code_Begin
    sd.stop()
    # check responses
    if key_resp_Begin.keys in ['', [], None]:  # No response was made
        key_resp_Begin.keys = None
    thisExp.addData('key_resp_Begin.keys',key_resp_Begin.keys)
    if key_resp_Begin.keys != None:  # we had a response
        thisExp.addData('key_resp_Begin.rt', key_resp_Begin.rt)
        thisExp.addData('key_resp_Begin.duration', key_resp_Begin.duration)
    thisExp.nextEntry()
    # the Routine "Begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pre_loop = data.TrialHandler2(
        name='pre_loop',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('prepost.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(pre_loop)  # add the loop to the experiment
    thisPre_loop = pre_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPre_loop.rgb)
    if thisPre_loop != None:
        for paramName in thisPre_loop:
            globals()[paramName] = thisPre_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPre_loop in pre_loop:
        pre_loop.status = STARTED
        if hasattr(thisPre_loop, 'status'):
            thisPre_loop.status = STARTED
        currentLoop = pre_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPre_loop.rgb)
        if thisPre_loop != None:
            for paramName in thisPre_loop:
                globals()[paramName] = thisPre_loop[paramName]
        
        # --- Prepare to start Routine "SpeechIntro" ---
        # create an object to store info about Routine SpeechIntro
        SpeechIntro = data.Routine(
            name='SpeechIntro',
            components=[text_SpeechIntro, mouse_SpeechIntro, sound_SpeechIntro, key_resp_10],
        )
        SpeechIntro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_10
        if isIntro == 0:
            continueRoutine = False
        text_SpeechIntro.setText(IntroText)
        # setup some python lists for storing info about the mouse_SpeechIntro
        gotValidClick = False  # until a click is received
        sound_SpeechIntro.setSound("./sounds/"+IntroSound, hamming=True)
        sound_SpeechIntro.setVolume(Vol, log=False)
        sound_SpeechIntro.seek(0)
        # create starting attributes for key_resp_10
        key_resp_10.keys = []
        key_resp_10.rt = []
        _key_resp_10_allKeys = []
        # store start times for SpeechIntro
        SpeechIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SpeechIntro.tStart = globalClock.getTime(format='float')
        SpeechIntro.status = STARTED
        thisExp.addData('SpeechIntro.started', SpeechIntro.tStart)
        SpeechIntro.maxDuration = None
        # keep track of which components have finished
        SpeechIntroComponents = SpeechIntro.components
        for thisComponent in SpeechIntro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SpeechIntro" ---
        SpeechIntro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPre_loop, 'status') and thisPre_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_10
            if sound_SpeechIntro.status == FINISHED:
                TCTOpacity_SpeechIntro = 1
            
            # *text_SpeechIntro* updates
            
            # if text_SpeechIntro is starting this frame...
            if text_SpeechIntro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_SpeechIntro.frameNStart = frameN  # exact frame index
                text_SpeechIntro.tStart = t  # local t and not account for scr refresh
                text_SpeechIntro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SpeechIntro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_SpeechIntro.started')
                # update status
                text_SpeechIntro.status = STARTED
                text_SpeechIntro.setAutoDraw(True)
            
            # if text_SpeechIntro is active this frame...
            if text_SpeechIntro.status == STARTED:
                # update params
                pass
            # *mouse_SpeechIntro* updates
            
            # if mouse_SpeechIntro is starting this frame...
            if mouse_SpeechIntro.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                mouse_SpeechIntro.frameNStart = frameN  # exact frame index
                mouse_SpeechIntro.tStart = t  # local t and not account for scr refresh
                mouse_SpeechIntro.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_SpeechIntro, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_SpeechIntro.started', t)
                # update status
                mouse_SpeechIntro.status = STARTED
                mouse_SpeechIntro.mouseClock.reset()
                prevButtonState = mouse_SpeechIntro.getPressed()  # if button is down already this ISN'T a new click
            if mouse_SpeechIntro.status == STARTED:  # only update if started and not finished!
                buttons = mouse_SpeechIntro.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        continueRoutine = False  # end routine on response            
            # *sound_SpeechIntro* updates
            
            # if sound_SpeechIntro is starting this frame...
            if sound_SpeechIntro.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                sound_SpeechIntro.frameNStart = frameN  # exact frame index
                sound_SpeechIntro.tStart = t  # local t and not account for scr refresh
                sound_SpeechIntro.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_SpeechIntro.started', tThisFlipGlobal)
                # update status
                sound_SpeechIntro.status = STARTED
                sound_SpeechIntro.play(when=win)  # sync with win flip
            
            # if sound_SpeechIntro is stopping this frame...
            if sound_SpeechIntro.status == STARTED:
                if bool(False) or sound_SpeechIntro.isFinished:
                    # keep track of stop time/frame for later
                    sound_SpeechIntro.tStop = t  # not accounting for scr refresh
                    sound_SpeechIntro.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_SpeechIntro.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_SpeechIntro.stopped')
                    # update status
                    sound_SpeechIntro.status = FINISHED
                    sound_SpeechIntro.stop()
            
            # *key_resp_10* updates
            waitOnFlip = False
            
            # if key_resp_10 is starting this frame...
            if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_10.frameNStart = frameN  # exact frame index
                key_resp_10.tStart = t  # local t and not account for scr refresh
                key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_10.started')
                # update status
                key_resp_10.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_10.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_10.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_10_allKeys.extend(theseKeys)
                if len(_key_resp_10_allKeys):
                    key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                    key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                    key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=SpeechIntro,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                SpeechIntro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SpeechIntro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SpeechIntro" ---
        for thisComponent in SpeechIntro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SpeechIntro
        SpeechIntro.tStop = globalClock.getTime(format='float')
        SpeechIntro.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SpeechIntro.stopped', SpeechIntro.tStop)
        # store data for pre_loop (TrialHandler)
        x, y = mouse_SpeechIntro.getPos()
        buttons = mouse_SpeechIntro.getPressed()
        pre_loop.addData('mouse_SpeechIntro.x', x)
        pre_loop.addData('mouse_SpeechIntro.y', y)
        pre_loop.addData('mouse_SpeechIntro.leftButton', buttons[0])
        pre_loop.addData('mouse_SpeechIntro.midButton', buttons[1])
        pre_loop.addData('mouse_SpeechIntro.rightButton', buttons[2])
        sound_SpeechIntro.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if key_resp_10.keys in ['', [], None]:  # No response was made
            key_resp_10.keys = None
        pre_loop.addData('key_resp_10.keys',key_resp_10.keys)
        if key_resp_10.keys != None:  # we had a response
            pre_loop.addData('key_resp_10.rt', key_resp_10.rt)
            pre_loop.addData('key_resp_10.duration', key_resp_10.duration)
        # the Routine "SpeechIntro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        speech_trials_pre = data.TrialHandler2(
            name='speech_trials_pre',
            nReps=10.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(speech_trials_pre)  # add the loop to the experiment
        thisSpeech_trials_pre = speech_trials_pre.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSpeech_trials_pre.rgb)
        if thisSpeech_trials_pre != None:
            for paramName in thisSpeech_trials_pre:
                globals()[paramName] = thisSpeech_trials_pre[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisSpeech_trials_pre in speech_trials_pre:
            speech_trials_pre.status = STARTED
            if hasattr(thisSpeech_trials_pre, 'status'):
                thisSpeech_trials_pre.status = STARTED
            currentLoop = speech_trials_pre
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisSpeech_trials_pre.rgb)
            if thisSpeech_trials_pre != None:
                for paramName in thisSpeech_trials_pre:
                    globals()[paramName] = thisSpeech_trials_pre[paramName]
            
            # --- Prepare to start Routine "SpeechTask" ---
            # create an object to store info about Routine SpeechTask
            SpeechTask = data.Routine(
                name='SpeechTask',
                components=[text_SpeechTask_mid, mouse_SpeechTask, text_SpeechTask_high, text_SpeechTask_low, key_resp_9],
            )
            SpeechTask.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_9
            if Task == "Reading":
                text_SpeechTask_mid.alignText='left'
                text_SpeechTask_mid.color = (-1,-1,-1)
                SpeechTextHigh = ''
                SpeechTextLow = ''
            
            elif Task == "Vowel" or "RFF":
                SpeechTextHigh = SpeechText
                SpeechTextLow = SpeechText
                text_SpeechTask_mid.alignText='center'
                if RFF_rep == 1:
                    text_SpeechTask_high.color = (-1,-1,-1)
                    text_SpeechTask_mid.color = (.8,.8,.8)
                    text_SpeechTask_low.color = (.8,.8,.8)
            #        text_ClickToContinue_4.text = ""
                elif RFF_rep == 2:
                    text_SpeechTask_high.color = (.8,.8,.8)
                    text_SpeechTask_mid.color = (-1,-1,-1)
                    text_SpeechTask_low.color = (.8,.8,.8)
            #        text_ClickToContinue_4.text = ""
                elif RFF_rep == 3:
                    text_SpeechTask_high.color = (.8,.8,.8)
                    text_SpeechTask_mid.color = (.8,.8,.8)
                    text_SpeechTask_low.color = (-1,-1,-1)
            #        text_ClickToContinue_4.text = ""
            
            isStop = 0
            #mic.start()
            t1 = time.time()-timeStart
            # setup some python lists for storing info about the mouse_SpeechTask
            gotValidClick = False  # until a click is received
            # create starting attributes for key_resp_9
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            # store start times for SpeechTask
            SpeechTask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            SpeechTask.tStart = globalClock.getTime(format='float')
            SpeechTask.status = STARTED
            thisExp.addData('SpeechTask.started', SpeechTask.tStart)
            SpeechTask.maxDuration = None
            # keep track of which components have finished
            SpeechTaskComponents = SpeechTask.components
            for thisComponent in SpeechTask.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "SpeechTask" ---
            SpeechTask.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisSpeech_trials_pre, 'status') and thisSpeech_trials_pre.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_9
                #mic.poll()
                
                if key_resp_6.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_6.getKeys(keyList=['y','n','left','right','space','backspace'], waitRelease=False)
                    if 'backspace' in theseKeys or 'left' in theseKeys:
                        if RFF_rep == 1:
                            text_SpeechTask_high.color = (-1*text_SpeechTask_high.color[0],-1,-1)
                            text_SpeechTask_mid.color = (.8,.8,.8)
                            text_SpeechTask_low.color = (.8,.8,.8)
                    #        text_ClickToContinue_4.text = ""
                        elif RFF_rep == 2:
                            text_SpeechTask_high.color = (.8,.8,.8)
                            text_SpeechTask_mid.color = (-1*text_SpeechTask_mid.color[0],-1,-1)
                            text_SpeechTask_low.color = (.8,.8,.8)
                    #        text_ClickToContinue_4.text = ""
                        elif RFF_rep == 3:
                            text_SpeechTask_high.color = (.8,.8,.8)
                            text_SpeechTask_mid.color = (.8,.8,.8)
                            text_SpeechTask_low.color = (-1*text_SpeechTask_low.color[0],-1,-1)
                        else:
                            text_SpeechTask_mid.color = (-1*text_SpeechTask_mid.color[0],-1,-1)
                    else:
                        if prepost == 'pre':
                            speech_trials_pre.finished = True
                        if prepost == 'post':
                            speech_trials_post.finished = True
                        _key_resp_6_allKeys.extend(theseKeys)
                        if len(_key_resp_6_allKeys):
                            key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                            key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                            # a response ends the routine
                #            if isRFF < 2:
                #                continueRoutine = False
                            if isStop == 1:
                                time.sleep(1)
                                continueRoutine = False
                            else:
                                time.sleep(5)
                                text_SpeechTask_mid.alignText='center'
                                text_SpeechTask_mid.color = (-1,-1,-1)
                                SpeechTextHigh = ''
                                SpeechTextLow = ''
                                SpeechText = 'STOP'
                                isStop = 1
                
                # *text_SpeechTask_mid* updates
                
                # if text_SpeechTask_mid is starting this frame...
                if text_SpeechTask_mid.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_SpeechTask_mid.frameNStart = frameN  # exact frame index
                    text_SpeechTask_mid.tStart = t  # local t and not account for scr refresh
                    text_SpeechTask_mid.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_SpeechTask_mid, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_SpeechTask_mid.started')
                    # update status
                    text_SpeechTask_mid.status = STARTED
                    text_SpeechTask_mid.setAutoDraw(True)
                
                # if text_SpeechTask_mid is active this frame...
                if text_SpeechTask_mid.status == STARTED:
                    # update params
                    text_SpeechTask_mid.setText(SpeechText, log=False)
                # *mouse_SpeechTask* updates
                
                # if mouse_SpeechTask is starting this frame...
                if mouse_SpeechTask.status == NOT_STARTED and t >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_SpeechTask.frameNStart = frameN  # exact frame index
                    mouse_SpeechTask.tStart = t  # local t and not account for scr refresh
                    mouse_SpeechTask.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_SpeechTask, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse_SpeechTask.started', t)
                    # update status
                    mouse_SpeechTask.status = STARTED
                    mouse_SpeechTask.mouseClock.reset()
                    prevButtonState = mouse_SpeechTask.getPressed()  # if button is down already this ISN'T a new click
                if mouse_SpeechTask.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_SpeechTask.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            continueRoutine = False  # end routine on response                
                # *text_SpeechTask_high* updates
                
                # if text_SpeechTask_high is starting this frame...
                if text_SpeechTask_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_SpeechTask_high.frameNStart = frameN  # exact frame index
                    text_SpeechTask_high.tStart = t  # local t and not account for scr refresh
                    text_SpeechTask_high.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_SpeechTask_high, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_SpeechTask_high.started')
                    # update status
                    text_SpeechTask_high.status = STARTED
                    text_SpeechTask_high.setAutoDraw(True)
                
                # if text_SpeechTask_high is active this frame...
                if text_SpeechTask_high.status == STARTED:
                    # update params
                    text_SpeechTask_high.setText(SpeechTextHigh, log=False)
                
                # *text_SpeechTask_low* updates
                
                # if text_SpeechTask_low is starting this frame...
                if text_SpeechTask_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    text_SpeechTask_low.frameNStart = frameN  # exact frame index
                    text_SpeechTask_low.tStart = t  # local t and not account for scr refresh
                    text_SpeechTask_low.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_SpeechTask_low, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_SpeechTask_low.started')
                    # update status
                    text_SpeechTask_low.status = STARTED
                    text_SpeechTask_low.setAutoDraw(True)
                
                # if text_SpeechTask_low is active this frame...
                if text_SpeechTask_low.status == STARTED:
                    # update params
                    text_SpeechTask_low.setText(SpeechTextLow, log=False)
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['y','n','left','right','space','backspace'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=SpeechTask,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    SpeechTask.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in SpeechTask.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "SpeechTask" ---
            for thisComponent in SpeechTask.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for SpeechTask
            SpeechTask.tStop = globalClock.getTime(format='float')
            SpeechTask.tStopRefresh = tThisFlipGlobal
            thisExp.addData('SpeechTask.stopped', SpeechTask.tStop)
            # Run 'End Routine' code from code_9
            t2 = time.time()-timeStart
            
            #mic.stop()
            #audioClip = mic.getRecording()
            
            if Task == "Vowel" or "RFF":
                trialName = WavName+'_'+str(RFF_rep)+'_'+prepost  # save the recorded audio as a 'wav' file
            else:
                trialName = WavName+'_'+prepost  # save the recorded audio as a 'wav' file
            
            # Load the CSV file
            with open(filename+'_segtimes.csv', mode='r') as file:
                reader = csv.reader(file)
                segData = [row for row in reader]
            
            new_row = [trialName, t1, t2]
            segData.append(new_row)
            
            # Save the modified data back to the CSV file
            with open(filename+'_segtimes.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(segData)
            
            theseKeys = key_resp_6.getKeys(keyList=['y','n','left','right','space','backspace'], waitRelease=False)
            if 'backspace' in theseKeys or 'left' in theseKeys:
                RepeatSS = True
                print('REPEAT')
            else:
                if prepost == 'pre':
                    speech_trials_pre.finished = True
                if prepost == 'post':
                    speech_trials_post.finished = True
            # store data for speech_trials_pre (TrialHandler)
            x, y = mouse_SpeechTask.getPos()
            buttons = mouse_SpeechTask.getPressed()
            speech_trials_pre.addData('mouse_SpeechTask.x', x)
            speech_trials_pre.addData('mouse_SpeechTask.y', y)
            speech_trials_pre.addData('mouse_SpeechTask.leftButton', buttons[0])
            speech_trials_pre.addData('mouse_SpeechTask.midButton', buttons[1])
            speech_trials_pre.addData('mouse_SpeechTask.rightButton', buttons[2])
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            speech_trials_pre.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                speech_trials_pre.addData('key_resp_9.rt', key_resp_9.rt)
                speech_trials_pre.addData('key_resp_9.duration', key_resp_9.duration)
            
            
            
            # the Routine "SpeechTask" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisSpeech_trials_pre as finished
            if hasattr(thisSpeech_trials_pre, 'status'):
                thisSpeech_trials_pre.status = FINISHED
            # if awaiting a pause, pause now
            if speech_trials_pre.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                speech_trials_pre.status = STARTED
            thisExp.nextEntry()
            
        # completed 10.0 repeats of 'speech_trials_pre'
        speech_trials_pre.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "BorgRating" ---
        # create an object to store info about Routine BorgRating
        BorgRating = data.Routine(
            name='BorgRating',
            components=[VER_key_resp, image_BorgScale, text_BorgScale, VER_num_text, mouse_Borg, button_Borg, polygon_1, polygon_2, polygon_3, polygon_4, polygon_5, polygon_6, polygon_7, polygon_8, polygon_9, polygon_Back, polygon_0, polygon_Dot, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_Back, text_0, text_Dot],
        )
        BorgRating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        inputText = '#'  # Set Routine start values for inputText
        # create starting attributes for VER_key_resp
        VER_key_resp.keys = []
        VER_key_resp.rt = []
        _VER_key_resp_allKeys = []
        # Run 'Begin Routine' code from code_Borg
        #sd.stop()
        
        theseKeys=""
        VER_num_text.alignHoriz ='left'
        first_press = True
        
        # setup some python lists for storing info about the mouse_Borg
        mouse_Borg.x = []
        mouse_Borg.y = []
        mouse_Borg.leftButton = []
        mouse_Borg.midButton = []
        mouse_Borg.rightButton = []
        mouse_Borg.time = []
        gotValidClick = False  # until a click is received
        # reset button_Borg to account for continued clicks & clear times on/off
        button_Borg.reset()
        # store start times for BorgRating
        BorgRating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        BorgRating.tStart = globalClock.getTime(format='float')
        BorgRating.status = STARTED
        thisExp.addData('BorgRating.started', BorgRating.tStart)
        BorgRating.maxDuration = None
        # keep track of which components have finished
        BorgRatingComponents = BorgRating.components
        for thisComponent in BorgRating.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BorgRating" ---
        BorgRating.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPre_loop, 'status') and thisPre_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *VER_key_resp* updates
            waitOnFlip = False
            
            # if VER_key_resp is starting this frame...
            if VER_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                VER_key_resp.frameNStart = frameN  # exact frame index
                VER_key_resp.tStart = t  # local t and not account for scr refresh
                VER_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(VER_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'VER_key_resp.started')
                # update status
                VER_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(VER_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(VER_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if VER_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = VER_key_resp.getKeys(keyList=['return','y','n','left','right','space','1','2','3','4','5','6','7','8','9','0','period','return','backspace','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9','num_0','num_decimal','num_subtract','period','comma'], ignoreKeys=["escape"], waitRelease=False)
                _VER_key_resp_allKeys.extend(theseKeys)
                if len(_VER_key_resp_allKeys):
                    VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                    VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                    VER_key_resp.duration = _VER_key_resp_allKeys[-1].duration
            # Run 'Each Frame' code from code_Borg
            if mouse_Borg.isPressedIn(polygon_1):
                NumResponse = '1'
            if mouse_Borg.isPressedIn(polygon_2):
                NumResponse = '2'
            if mouse_Borg.isPressedIn(polygon_3):
                NumResponse = '3'
            if mouse_Borg.isPressedIn(polygon_4):
                NumResponse = '4'
            if mouse_Borg.isPressedIn(polygon_5):
                NumResponse = '5'
            if mouse_Borg.isPressedIn(polygon_6):
                NumResponse = '6'
            if mouse_Borg.isPressedIn(polygon_7):
                NumResponse = '7'
            if mouse_Borg.isPressedIn(polygon_8):
                NumResponse = '8'
            if mouse_Borg.isPressedIn(polygon_9):
                NumResponse = '9'
            if mouse_Borg.isPressedIn(polygon_0):
                NumResponse = '0'
            if mouse_Borg.isPressedIn(polygon_Dot):
                NumResponse = ''
                if len(inputText) == 0:
                    inputText += '0.'
                else:
                    inputText += '.'
                time.sleep(.1)
            
            if NumResponse != '':
                if inputText == '#':
                    inputText = NumResponse
                else:
                    inputText += NumResponse
                NumResponse = ''
                time.sleep(.1)
                
            if mouse_Borg.isPressedIn(polygon_Back):
                inputText = inputText[:-1]  # lose the final character
                time.sleep(.1)
            
            if len(_VER_key_resp_allKeys) and keyReady:
                keyReady = False
                VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                if first_press:
                    inputText = ''
                    first_press = False
            
                elif VER_key_resp.keys in ['right','space']:
                    continueRoutine = False
                    
                elif VER_key_resp.keys in ['num_1', '1']:
                    inputText += '1'
            
                elif VER_key_resp.keys in ['num_2', '2']:
                    inputText += '2'
            
                elif VER_key_resp.keys in ['num_3', '3']:
                    inputText += '3'
            
                elif VER_key_resp.keys in ['num_4', '4']:
                    inputText += '4'
            
                elif VER_key_resp.keys in ['num_5', '5']:
                    inputText += '5'
            
                elif VER_key_resp.keys in ['num_6', '6']:
                    inputText += '6'
            
                elif VER_key_resp.keys in ['num_7', '7']:
                    inputText += '7'
            
                elif VER_key_resp.keys in ['num_8', '8']:
                    inputText += '8'
            
                elif VER_key_resp.keys in ['num_9', '9']:
                    inputText += '9'
            
                elif VER_key_resp.keys in ['num_0', '0']:
                    inputText += '0'
            
                elif VER_key_resp.keys in ['period', 'comma','num_decimal']:
                    if len(inputText) == 0:
                        inputText += '0,'
                    else:
                        inputText += '.'
            
                elif VER_key_resp.keys in ['backspace','num_subtract']:
                    inputText = inputText[:-1]  # lose the final character
            
                #VER_key_resp = []
                _VER_key_resp_allKeys = []
                theseKeys = []
            elif len(_VER_key_resp_allKeys) == 0:
                keyReady = True
            
            # *image_BorgScale* updates
            
            # if image_BorgScale is starting this frame...
            if image_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                image_BorgScale.frameNStart = frameN  # exact frame index
                image_BorgScale.tStart = t  # local t and not account for scr refresh
                image_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_BorgScale, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_BorgScale.started')
                # update status
                image_BorgScale.status = STARTED
                image_BorgScale.setAutoDraw(True)
            
            # if image_BorgScale is active this frame...
            if image_BorgScale.status == STARTED:
                # update params
                pass
            
            # *text_BorgScale* updates
            
            # if text_BorgScale is starting this frame...
            if text_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_BorgScale.frameNStart = frameN  # exact frame index
                text_BorgScale.tStart = t  # local t and not account for scr refresh
                text_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_BorgScale, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_BorgScale.started')
                # update status
                text_BorgScale.status = STARTED
                text_BorgScale.setAutoDraw(True)
            
            # if text_BorgScale is active this frame...
            if text_BorgScale.status == STARTED:
                # update params
                pass
            
            # *VER_num_text* updates
            
            # if VER_num_text is starting this frame...
            if VER_num_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                VER_num_text.frameNStart = frameN  # exact frame index
                VER_num_text.tStart = t  # local t and not account for scr refresh
                VER_num_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(VER_num_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'VER_num_text.started')
                # update status
                VER_num_text.status = STARTED
                VER_num_text.setAutoDraw(True)
            
            # if VER_num_text is active this frame...
            if VER_num_text.status == STARTED:
                # update params
                VER_num_text.setText(inputText, log=False)
            # *mouse_Borg* updates
            if mouse_Borg.status == STARTED:  # only update if started and not finished!
                buttons = mouse_Borg.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_Borg.getPos()
                        mouse_Borg.x.append(x)
                        mouse_Borg.y.append(y)
                        buttons = mouse_Borg.getPressed()
                        mouse_Borg.leftButton.append(buttons[0])
                        mouse_Borg.midButton.append(buttons[1])
                        mouse_Borg.rightButton.append(buttons[2])
                        mouse_Borg.time.append(mouse_Borg.mouseClock.getTime())
            # *button_Borg* updates
            
            # if button_Borg is starting this frame...
            if button_Borg.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                button_Borg.frameNStart = frameN  # exact frame index
                button_Borg.tStart = t  # local t and not account for scr refresh
                button_Borg.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_Borg, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_Borg.started')
                # update status
                button_Borg.status = STARTED
                win.callOnFlip(button_Borg.buttonClock.reset)
                button_Borg.setAutoDraw(True)
            
            # if button_Borg is active this frame...
            if button_Borg.status == STARTED:
                # update params
                pass
                # check whether button_Borg has been pressed
                if button_Borg.isClicked:
                    if not button_Borg.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_Borg.timesOn.append(button_Borg.buttonClock.getTime())
                        button_Borg.timesOff.append(button_Borg.buttonClock.getTime())
                    elif len(button_Borg.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_Borg.timesOff[-1] = button_Borg.buttonClock.getTime()
                    if not button_Borg.wasClicked:
                        # run callback code when button_Borg is clicked
                        if inputText != '' and inputText != '#':
                            continueRoutine = False
                            break
            # take note of whether button_Borg was clicked, so that next frame we know if clicks are new
            button_Borg.wasClicked = button_Borg.isClicked and button_Borg.status == STARTED
            
            # *polygon_1* updates
            
            # if polygon_1 is starting this frame...
            if polygon_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_1.frameNStart = frameN  # exact frame index
                polygon_1.tStart = t  # local t and not account for scr refresh
                polygon_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_1.started')
                # update status
                polygon_1.status = STARTED
                polygon_1.setAutoDraw(True)
            
            # if polygon_1 is active this frame...
            if polygon_1.status == STARTED:
                # update params
                pass
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # *polygon_3* updates
            
            # if polygon_3 is starting this frame...
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_3.started')
                # update status
                polygon_3.status = STARTED
                polygon_3.setAutoDraw(True)
            
            # if polygon_3 is active this frame...
            if polygon_3.status == STARTED:
                # update params
                pass
            
            # *polygon_4* updates
            
            # if polygon_4 is starting this frame...
            if polygon_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_4.frameNStart = frameN  # exact frame index
                polygon_4.tStart = t  # local t and not account for scr refresh
                polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_4.started')
                # update status
                polygon_4.status = STARTED
                polygon_4.setAutoDraw(True)
            
            # if polygon_4 is active this frame...
            if polygon_4.status == STARTED:
                # update params
                pass
            
            # *polygon_5* updates
            
            # if polygon_5 is starting this frame...
            if polygon_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_5.frameNStart = frameN  # exact frame index
                polygon_5.tStart = t  # local t and not account for scr refresh
                polygon_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_5.started')
                # update status
                polygon_5.status = STARTED
                polygon_5.setAutoDraw(True)
            
            # if polygon_5 is active this frame...
            if polygon_5.status == STARTED:
                # update params
                pass
            
            # *polygon_6* updates
            
            # if polygon_6 is starting this frame...
            if polygon_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_6.frameNStart = frameN  # exact frame index
                polygon_6.tStart = t  # local t and not account for scr refresh
                polygon_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_6.started')
                # update status
                polygon_6.status = STARTED
                polygon_6.setAutoDraw(True)
            
            # if polygon_6 is active this frame...
            if polygon_6.status == STARTED:
                # update params
                pass
            
            # *polygon_7* updates
            
            # if polygon_7 is starting this frame...
            if polygon_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_7.frameNStart = frameN  # exact frame index
                polygon_7.tStart = t  # local t and not account for scr refresh
                polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_7.started')
                # update status
                polygon_7.status = STARTED
                polygon_7.setAutoDraw(True)
            
            # if polygon_7 is active this frame...
            if polygon_7.status == STARTED:
                # update params
                pass
            
            # *polygon_8* updates
            
            # if polygon_8 is starting this frame...
            if polygon_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_8.frameNStart = frameN  # exact frame index
                polygon_8.tStart = t  # local t and not account for scr refresh
                polygon_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_8.started')
                # update status
                polygon_8.status = STARTED
                polygon_8.setAutoDraw(True)
            
            # if polygon_8 is active this frame...
            if polygon_8.status == STARTED:
                # update params
                pass
            
            # *polygon_9* updates
            
            # if polygon_9 is starting this frame...
            if polygon_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_9.frameNStart = frameN  # exact frame index
                polygon_9.tStart = t  # local t and not account for scr refresh
                polygon_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_9.started')
                # update status
                polygon_9.status = STARTED
                polygon_9.setAutoDraw(True)
            
            # if polygon_9 is active this frame...
            if polygon_9.status == STARTED:
                # update params
                pass
            
            # *polygon_Back* updates
            
            # if polygon_Back is starting this frame...
            if polygon_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_Back.frameNStart = frameN  # exact frame index
                polygon_Back.tStart = t  # local t and not account for scr refresh
                polygon_Back.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_Back, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_Back.started')
                # update status
                polygon_Back.status = STARTED
                polygon_Back.setAutoDraw(True)
            
            # if polygon_Back is active this frame...
            if polygon_Back.status == STARTED:
                # update params
                pass
            
            # *polygon_0* updates
            
            # if polygon_0 is starting this frame...
            if polygon_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_0.frameNStart = frameN  # exact frame index
                polygon_0.tStart = t  # local t and not account for scr refresh
                polygon_0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_0.started')
                # update status
                polygon_0.status = STARTED
                polygon_0.setAutoDraw(True)
            
            # if polygon_0 is active this frame...
            if polygon_0.status == STARTED:
                # update params
                pass
            
            # *polygon_Dot* updates
            
            # if polygon_Dot is starting this frame...
            if polygon_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_Dot.frameNStart = frameN  # exact frame index
                polygon_Dot.tStart = t  # local t and not account for scr refresh
                polygon_Dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_Dot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_Dot.started')
                # update status
                polygon_Dot.status = STARTED
                polygon_Dot.setAutoDraw(True)
            
            # if polygon_Dot is active this frame...
            if polygon_Dot.status == STARTED:
                # update params
                pass
            
            # *text_1* updates
            
            # if text_1 is starting this frame...
            if text_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_1.frameNStart = frameN  # exact frame index
                text_1.tStart = t  # local t and not account for scr refresh
                text_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_1, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_1.status = STARTED
                text_1.setAutoDraw(True)
            
            # if text_1 is active this frame...
            if text_1.status == STARTED:
                # update params
                pass
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
                # update params
                pass
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *text_Back* updates
            
            # if text_Back is starting this frame...
            if text_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Back.frameNStart = frameN  # exact frame index
                text_Back.tStart = t  # local t and not account for scr refresh
                text_Back.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Back, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_Back.status = STARTED
                text_Back.setAutoDraw(True)
            
            # if text_Back is active this frame...
            if text_Back.status == STARTED:
                # update params
                pass
            
            # *text_0* updates
            
            # if text_0 is starting this frame...
            if text_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_0.frameNStart = frameN  # exact frame index
                text_0.tStart = t  # local t and not account for scr refresh
                text_0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_0, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_0.status = STARTED
                text_0.setAutoDraw(True)
            
            # if text_0 is active this frame...
            if text_0.status == STARTED:
                # update params
                pass
            
            # *text_Dot* updates
            
            # if text_Dot is starting this frame...
            if text_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Dot.frameNStart = frameN  # exact frame index
                text_Dot.tStart = t  # local t and not account for scr refresh
                text_Dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Dot, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_Dot.status = STARTED
                text_Dot.setAutoDraw(True)
            
            # if text_Dot is active this frame...
            if text_Dot.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=BorgRating,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                BorgRating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BorgRating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BorgRating" ---
        for thisComponent in BorgRating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for BorgRating
        BorgRating.tStop = globalClock.getTime(format='float')
        BorgRating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('BorgRating.stopped', BorgRating.tStop)
        thisExp.addData('inputText.routineEndVal', inputText)  # Save end Routine value
        # check responses
        if VER_key_resp.keys in ['', [], None]:  # No response was made
            VER_key_resp.keys = None
        pre_loop.addData('VER_key_resp.keys',VER_key_resp.keys)
        if VER_key_resp.keys != None:  # we had a response
            pre_loop.addData('VER_key_resp.rt', VER_key_resp.rt)
            pre_loop.addData('VER_key_resp.duration', VER_key_resp.duration)
        # Run 'End Routine' code from code_Borg
        #sd.stop()
        
        # Load the CSV file
        with open(filename+'_Ratings.csv', mode='r') as file:
            reader = csv.reader(file)
            segData = [row for row in reader]
        
        VER_num += 1
        
        new_row = ['VER_'+str(VER_num), inputText]
        segData.append(new_row)
        
        # Save the modified data back to the CSV file
        with open(filename+'_Ratings.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(segData)
        
        
        # let's store the final text string into the results finle...
        thisExp.addData('VER', inputText)
        inputText="#"
        thisExp.addData('Task','VER')
        # store data for pre_loop (TrialHandler)
        pre_loop.addData('mouse_Borg.x', mouse_Borg.x)
        pre_loop.addData('mouse_Borg.y', mouse_Borg.y)
        pre_loop.addData('mouse_Borg.leftButton', mouse_Borg.leftButton)
        pre_loop.addData('mouse_Borg.midButton', mouse_Borg.midButton)
        pre_loop.addData('mouse_Borg.rightButton', mouse_Borg.rightButton)
        pre_loop.addData('mouse_Borg.time', mouse_Borg.time)
        pre_loop.addData('button_Borg.numClicks', button_Borg.numClicks)
        if button_Borg.numClicks:
           pre_loop.addData('button_Borg.timesOn', button_Borg.timesOn)
           pre_loop.addData('button_Borg.timesOff', button_Borg.timesOff)
        else:
           pre_loop.addData('button_Borg.timesOn', "")
           pre_loop.addData('button_Borg.timesOff', "")
        # the Routine "BorgRating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPre_loop as finished
        if hasattr(thisPre_loop, 'status'):
            thisPre_loop.status = FINISHED
        # if awaiting a pause, pause now
        if pre_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pre_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'pre_loop'
    pre_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials_noise = data.TrialHandler2(
        name='trials_noise',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/VLT_params.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_noise)  # add the loop to the experiment
    thisTrials_noise = trials_noise.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_noise.rgb)
    if thisTrials_noise != None:
        for paramName in thisTrials_noise:
            globals()[paramName] = thisTrials_noise[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_noise in trials_noise:
        trials_noise.status = STARTED
        if hasattr(thisTrials_noise, 'status'):
            thisTrials_noise.status = STARTED
        currentLoop = trials_noise
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_noise.rgb)
        if thisTrials_noise != None:
            for paramName in thisTrials_noise:
                globals()[paramName] = thisTrials_noise[paramName]
        
        # --- Prepare to start Routine "VLT_init" ---
        # create an object to store info about Routine VLT_init
        VLT_init = data.Routine(
            name='VLT_init',
            components=[],
        )
        VLT_init.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_VLT_init
        pages = slice(pagenum, pagelen)
        # store start times for VLT_init
        VLT_init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        VLT_init.tStart = globalClock.getTime(format='float')
        VLT_init.status = STARTED
        thisExp.addData('VLT_init.started', VLT_init.tStart)
        VLT_init.maxDuration = None
        # keep track of which components have finished
        VLT_initComponents = VLT_init.components
        for thisComponent in VLT_init.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "VLT_init" ---
        VLT_init.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_noise, 'status') and thisTrials_noise.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=VLT_init,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                VLT_init.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in VLT_init.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "VLT_init" ---
        for thisComponent in VLT_init.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for VLT_init
        VLT_init.tStop = globalClock.getTime(format='float')
        VLT_init.tStopRefresh = tThisFlipGlobal
        thisExp.addData('VLT_init.stopped', VLT_init.tStop)
        
        # Run 'End Routine' code from code_VLT_init
        prepost = 'post'
        # the Routine "VLT_init" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        VLT_stim = data.TrialHandler2(
            name='VLT_stim',
            nReps=99.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            'PsychoPy/Pics_all.xlsx', 
            selection=pages
        )
        , 
            seed=None, 
        )
        thisExp.addLoop(VLT_stim)  # add the loop to the experiment
        thisVLT_stim = VLT_stim.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisVLT_stim.rgb)
        if thisVLT_stim != None:
            for paramName in thisVLT_stim:
                globals()[paramName] = thisVLT_stim[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisVLT_stim in VLT_stim:
            VLT_stim.status = STARTED
            if hasattr(thisVLT_stim, 'status'):
                thisVLT_stim.status = STARTED
            currentLoop = VLT_stim
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisVLT_stim.rgb)
            if thisVLT_stim != None:
                for paramName in thisVLT_stim:
                    globals()[paramName] = thisVLT_stim[paramName]
            
            # --- Prepare to start Routine "STIMULI" ---
            # create an object to store info about Routine STIMULI
            STIMULI = data.Routine(
                name='STIMULI',
                components=[STIM_key_resp, STIM_image, STIM_mouse],
            )
            STIMULI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from STIM_code
            if Level != previousLevel:
                sd.stop()
                if Level == 'Low':
                    sd.play(NLdata, samplerate=NLfs, loop=True)
                elif Level == 'High':
                    sd.play(NHdata, samplerate=NHfs, loop=True)
                previousLevel = Level
            # create starting attributes for STIM_key_resp
            STIM_key_resp.keys = []
            STIM_key_resp.rt = []
            _STIM_key_resp_allKeys = []
            STIM_image.setImage(picture_image)
            # setup some python lists for storing info about the STIM_mouse
            STIM_mouse.x = []
            STIM_mouse.y = []
            STIM_mouse.leftButton = []
            STIM_mouse.midButton = []
            STIM_mouse.rightButton = []
            STIM_mouse.time = []
            gotValidClick = False  # until a click is received
            # store start times for STIMULI
            STIMULI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            STIMULI.tStart = globalClock.getTime(format='float')
            STIMULI.status = STARTED
            thisExp.addData('STIMULI.started', STIMULI.tStart)
            STIMULI.maxDuration = None
            # keep track of which components have finished
            STIMULIComponents = STIMULI.components
            for thisComponent in STIMULI.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "STIMULI" ---
            STIMULI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisVLT_stim, 'status') and thisVLT_stim.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from STIM_code
                if t > Duration:
                    VLT_stim.finished = True
                    continueRoutine = False
                    pagenum -= 1
                
                # *STIM_key_resp* updates
                waitOnFlip = False
                
                # if STIM_key_resp is starting this frame...
                if STIM_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    STIM_key_resp.frameNStart = frameN  # exact frame index
                    STIM_key_resp.tStart = t  # local t and not account for scr refresh
                    STIM_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(STIM_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'STIM_key_resp.started')
                    # update status
                    STIM_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(STIM_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(STIM_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if STIM_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = STIM_key_resp.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                    _STIM_key_resp_allKeys.extend(theseKeys)
                    if len(_STIM_key_resp_allKeys):
                        STIM_key_resp.keys = _STIM_key_resp_allKeys[-1].name  # just the last key pressed
                        STIM_key_resp.rt = _STIM_key_resp_allKeys[-1].rt
                        STIM_key_resp.duration = _STIM_key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *STIM_image* updates
                
                # if STIM_image is starting this frame...
                if STIM_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    STIM_image.frameNStart = frameN  # exact frame index
                    STIM_image.tStart = t  # local t and not account for scr refresh
                    STIM_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(STIM_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'STIM_image.started')
                    # update status
                    STIM_image.status = STARTED
                    STIM_image.setAutoDraw(True)
                
                # if STIM_image is active this frame...
                if STIM_image.status == STARTED:
                    # update params
                    pass
                # *STIM_mouse* updates
                
                # if STIM_mouse is starting this frame...
                if STIM_mouse.status == NOT_STARTED and t >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    STIM_mouse.frameNStart = frameN  # exact frame index
                    STIM_mouse.tStart = t  # local t and not account for scr refresh
                    STIM_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(STIM_mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('STIM_mouse.started', t)
                    # update status
                    STIM_mouse.status = STARTED
                    STIM_mouse.mouseClock.reset()
                    prevButtonState = STIM_mouse.getPressed()  # if button is down already this ISN'T a new click
                if STIM_mouse.status == STARTED:  # only update if started and not finished!
                    buttons = STIM_mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = STIM_mouse.getPos()
                            STIM_mouse.x.append(x)
                            STIM_mouse.y.append(y)
                            buttons = STIM_mouse.getPressed()
                            STIM_mouse.leftButton.append(buttons[0])
                            STIM_mouse.midButton.append(buttons[1])
                            STIM_mouse.rightButton.append(buttons[2])
                            STIM_mouse.time.append(STIM_mouse.mouseClock.getTime())
                            
                            continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=STIMULI,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    STIMULI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in STIMULI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "STIMULI" ---
            for thisComponent in STIMULI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for STIMULI
            STIMULI.tStop = globalClock.getTime(format='float')
            STIMULI.tStopRefresh = tThisFlipGlobal
            thisExp.addData('STIMULI.stopped', STIMULI.tStop)
            
            # Run 'End Routine' code from STIM_code
            pagenum += 1
            
            if pagenum > pagelen:
                pagenum = 0
            # check responses
            if STIM_key_resp.keys in ['', [], None]:  # No response was made
                STIM_key_resp.keys = None
            VLT_stim.addData('STIM_key_resp.keys',STIM_key_resp.keys)
            if STIM_key_resp.keys != None:  # we had a response
                VLT_stim.addData('STIM_key_resp.rt', STIM_key_resp.rt)
                VLT_stim.addData('STIM_key_resp.duration', STIM_key_resp.duration)
            # store data for VLT_stim (TrialHandler)
            VLT_stim.addData('STIM_mouse.x', STIM_mouse.x)
            VLT_stim.addData('STIM_mouse.y', STIM_mouse.y)
            VLT_stim.addData('STIM_mouse.leftButton', STIM_mouse.leftButton)
            VLT_stim.addData('STIM_mouse.midButton', STIM_mouse.midButton)
            VLT_stim.addData('STIM_mouse.rightButton', STIM_mouse.rightButton)
            VLT_stim.addData('STIM_mouse.time', STIM_mouse.time)
            # the Routine "STIMULI" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisVLT_stim as finished
            if hasattr(thisVLT_stim, 'status'):
                thisVLT_stim.status = FINISHED
            # if awaiting a pause, pause now
            if VLT_stim.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                VLT_stim.status = STARTED
            thisExp.nextEntry()
            
        # completed 99.0 repeats of 'VLT_stim'
        VLT_stim.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "BorgRating" ---
        # create an object to store info about Routine BorgRating
        BorgRating = data.Routine(
            name='BorgRating',
            components=[VER_key_resp, image_BorgScale, text_BorgScale, VER_num_text, mouse_Borg, button_Borg, polygon_1, polygon_2, polygon_3, polygon_4, polygon_5, polygon_6, polygon_7, polygon_8, polygon_9, polygon_Back, polygon_0, polygon_Dot, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_Back, text_0, text_Dot],
        )
        BorgRating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        inputText = '#'  # Set Routine start values for inputText
        # create starting attributes for VER_key_resp
        VER_key_resp.keys = []
        VER_key_resp.rt = []
        _VER_key_resp_allKeys = []
        # Run 'Begin Routine' code from code_Borg
        #sd.stop()
        
        theseKeys=""
        VER_num_text.alignHoriz ='left'
        first_press = True
        
        # setup some python lists for storing info about the mouse_Borg
        mouse_Borg.x = []
        mouse_Borg.y = []
        mouse_Borg.leftButton = []
        mouse_Borg.midButton = []
        mouse_Borg.rightButton = []
        mouse_Borg.time = []
        gotValidClick = False  # until a click is received
        # reset button_Borg to account for continued clicks & clear times on/off
        button_Borg.reset()
        # store start times for BorgRating
        BorgRating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        BorgRating.tStart = globalClock.getTime(format='float')
        BorgRating.status = STARTED
        thisExp.addData('BorgRating.started', BorgRating.tStart)
        BorgRating.maxDuration = None
        # keep track of which components have finished
        BorgRatingComponents = BorgRating.components
        for thisComponent in BorgRating.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BorgRating" ---
        BorgRating.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_noise, 'status') and thisTrials_noise.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *VER_key_resp* updates
            waitOnFlip = False
            
            # if VER_key_resp is starting this frame...
            if VER_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                VER_key_resp.frameNStart = frameN  # exact frame index
                VER_key_resp.tStart = t  # local t and not account for scr refresh
                VER_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(VER_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'VER_key_resp.started')
                # update status
                VER_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(VER_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(VER_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if VER_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = VER_key_resp.getKeys(keyList=['return','y','n','left','right','space','1','2','3','4','5','6','7','8','9','0','period','return','backspace','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9','num_0','num_decimal','num_subtract','period','comma'], ignoreKeys=["escape"], waitRelease=False)
                _VER_key_resp_allKeys.extend(theseKeys)
                if len(_VER_key_resp_allKeys):
                    VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                    VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                    VER_key_resp.duration = _VER_key_resp_allKeys[-1].duration
            # Run 'Each Frame' code from code_Borg
            if mouse_Borg.isPressedIn(polygon_1):
                NumResponse = '1'
            if mouse_Borg.isPressedIn(polygon_2):
                NumResponse = '2'
            if mouse_Borg.isPressedIn(polygon_3):
                NumResponse = '3'
            if mouse_Borg.isPressedIn(polygon_4):
                NumResponse = '4'
            if mouse_Borg.isPressedIn(polygon_5):
                NumResponse = '5'
            if mouse_Borg.isPressedIn(polygon_6):
                NumResponse = '6'
            if mouse_Borg.isPressedIn(polygon_7):
                NumResponse = '7'
            if mouse_Borg.isPressedIn(polygon_8):
                NumResponse = '8'
            if mouse_Borg.isPressedIn(polygon_9):
                NumResponse = '9'
            if mouse_Borg.isPressedIn(polygon_0):
                NumResponse = '0'
            if mouse_Borg.isPressedIn(polygon_Dot):
                NumResponse = ''
                if len(inputText) == 0:
                    inputText += '0.'
                else:
                    inputText += '.'
                time.sleep(.1)
            
            if NumResponse != '':
                if inputText == '#':
                    inputText = NumResponse
                else:
                    inputText += NumResponse
                NumResponse = ''
                time.sleep(.1)
                
            if mouse_Borg.isPressedIn(polygon_Back):
                inputText = inputText[:-1]  # lose the final character
                time.sleep(.1)
            
            if len(_VER_key_resp_allKeys) and keyReady:
                keyReady = False
                VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                if first_press:
                    inputText = ''
                    first_press = False
            
                elif VER_key_resp.keys in ['right','space']:
                    continueRoutine = False
                    
                elif VER_key_resp.keys in ['num_1', '1']:
                    inputText += '1'
            
                elif VER_key_resp.keys in ['num_2', '2']:
                    inputText += '2'
            
                elif VER_key_resp.keys in ['num_3', '3']:
                    inputText += '3'
            
                elif VER_key_resp.keys in ['num_4', '4']:
                    inputText += '4'
            
                elif VER_key_resp.keys in ['num_5', '5']:
                    inputText += '5'
            
                elif VER_key_resp.keys in ['num_6', '6']:
                    inputText += '6'
            
                elif VER_key_resp.keys in ['num_7', '7']:
                    inputText += '7'
            
                elif VER_key_resp.keys in ['num_8', '8']:
                    inputText += '8'
            
                elif VER_key_resp.keys in ['num_9', '9']:
                    inputText += '9'
            
                elif VER_key_resp.keys in ['num_0', '0']:
                    inputText += '0'
            
                elif VER_key_resp.keys in ['period', 'comma','num_decimal']:
                    if len(inputText) == 0:
                        inputText += '0,'
                    else:
                        inputText += '.'
            
                elif VER_key_resp.keys in ['backspace','num_subtract']:
                    inputText = inputText[:-1]  # lose the final character
            
                #VER_key_resp = []
                _VER_key_resp_allKeys = []
                theseKeys = []
            elif len(_VER_key_resp_allKeys) == 0:
                keyReady = True
            
            # *image_BorgScale* updates
            
            # if image_BorgScale is starting this frame...
            if image_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                image_BorgScale.frameNStart = frameN  # exact frame index
                image_BorgScale.tStart = t  # local t and not account for scr refresh
                image_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_BorgScale, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_BorgScale.started')
                # update status
                image_BorgScale.status = STARTED
                image_BorgScale.setAutoDraw(True)
            
            # if image_BorgScale is active this frame...
            if image_BorgScale.status == STARTED:
                # update params
                pass
            
            # *text_BorgScale* updates
            
            # if text_BorgScale is starting this frame...
            if text_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_BorgScale.frameNStart = frameN  # exact frame index
                text_BorgScale.tStart = t  # local t and not account for scr refresh
                text_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_BorgScale, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_BorgScale.started')
                # update status
                text_BorgScale.status = STARTED
                text_BorgScale.setAutoDraw(True)
            
            # if text_BorgScale is active this frame...
            if text_BorgScale.status == STARTED:
                # update params
                pass
            
            # *VER_num_text* updates
            
            # if VER_num_text is starting this frame...
            if VER_num_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                VER_num_text.frameNStart = frameN  # exact frame index
                VER_num_text.tStart = t  # local t and not account for scr refresh
                VER_num_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(VER_num_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'VER_num_text.started')
                # update status
                VER_num_text.status = STARTED
                VER_num_text.setAutoDraw(True)
            
            # if VER_num_text is active this frame...
            if VER_num_text.status == STARTED:
                # update params
                VER_num_text.setText(inputText, log=False)
            # *mouse_Borg* updates
            if mouse_Borg.status == STARTED:  # only update if started and not finished!
                buttons = mouse_Borg.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_Borg.getPos()
                        mouse_Borg.x.append(x)
                        mouse_Borg.y.append(y)
                        buttons = mouse_Borg.getPressed()
                        mouse_Borg.leftButton.append(buttons[0])
                        mouse_Borg.midButton.append(buttons[1])
                        mouse_Borg.rightButton.append(buttons[2])
                        mouse_Borg.time.append(mouse_Borg.mouseClock.getTime())
            # *button_Borg* updates
            
            # if button_Borg is starting this frame...
            if button_Borg.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                button_Borg.frameNStart = frameN  # exact frame index
                button_Borg.tStart = t  # local t and not account for scr refresh
                button_Borg.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_Borg, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_Borg.started')
                # update status
                button_Borg.status = STARTED
                win.callOnFlip(button_Borg.buttonClock.reset)
                button_Borg.setAutoDraw(True)
            
            # if button_Borg is active this frame...
            if button_Borg.status == STARTED:
                # update params
                pass
                # check whether button_Borg has been pressed
                if button_Borg.isClicked:
                    if not button_Borg.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_Borg.timesOn.append(button_Borg.buttonClock.getTime())
                        button_Borg.timesOff.append(button_Borg.buttonClock.getTime())
                    elif len(button_Borg.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_Borg.timesOff[-1] = button_Borg.buttonClock.getTime()
                    if not button_Borg.wasClicked:
                        # run callback code when button_Borg is clicked
                        if inputText != '' and inputText != '#':
                            continueRoutine = False
                            break
            # take note of whether button_Borg was clicked, so that next frame we know if clicks are new
            button_Borg.wasClicked = button_Borg.isClicked and button_Borg.status == STARTED
            
            # *polygon_1* updates
            
            # if polygon_1 is starting this frame...
            if polygon_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_1.frameNStart = frameN  # exact frame index
                polygon_1.tStart = t  # local t and not account for scr refresh
                polygon_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_1.started')
                # update status
                polygon_1.status = STARTED
                polygon_1.setAutoDraw(True)
            
            # if polygon_1 is active this frame...
            if polygon_1.status == STARTED:
                # update params
                pass
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # *polygon_3* updates
            
            # if polygon_3 is starting this frame...
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_3.started')
                # update status
                polygon_3.status = STARTED
                polygon_3.setAutoDraw(True)
            
            # if polygon_3 is active this frame...
            if polygon_3.status == STARTED:
                # update params
                pass
            
            # *polygon_4* updates
            
            # if polygon_4 is starting this frame...
            if polygon_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_4.frameNStart = frameN  # exact frame index
                polygon_4.tStart = t  # local t and not account for scr refresh
                polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_4.started')
                # update status
                polygon_4.status = STARTED
                polygon_4.setAutoDraw(True)
            
            # if polygon_4 is active this frame...
            if polygon_4.status == STARTED:
                # update params
                pass
            
            # *polygon_5* updates
            
            # if polygon_5 is starting this frame...
            if polygon_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_5.frameNStart = frameN  # exact frame index
                polygon_5.tStart = t  # local t and not account for scr refresh
                polygon_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_5.started')
                # update status
                polygon_5.status = STARTED
                polygon_5.setAutoDraw(True)
            
            # if polygon_5 is active this frame...
            if polygon_5.status == STARTED:
                # update params
                pass
            
            # *polygon_6* updates
            
            # if polygon_6 is starting this frame...
            if polygon_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_6.frameNStart = frameN  # exact frame index
                polygon_6.tStart = t  # local t and not account for scr refresh
                polygon_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_6.started')
                # update status
                polygon_6.status = STARTED
                polygon_6.setAutoDraw(True)
            
            # if polygon_6 is active this frame...
            if polygon_6.status == STARTED:
                # update params
                pass
            
            # *polygon_7* updates
            
            # if polygon_7 is starting this frame...
            if polygon_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_7.frameNStart = frameN  # exact frame index
                polygon_7.tStart = t  # local t and not account for scr refresh
                polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_7.started')
                # update status
                polygon_7.status = STARTED
                polygon_7.setAutoDraw(True)
            
            # if polygon_7 is active this frame...
            if polygon_7.status == STARTED:
                # update params
                pass
            
            # *polygon_8* updates
            
            # if polygon_8 is starting this frame...
            if polygon_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_8.frameNStart = frameN  # exact frame index
                polygon_8.tStart = t  # local t and not account for scr refresh
                polygon_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_8.started')
                # update status
                polygon_8.status = STARTED
                polygon_8.setAutoDraw(True)
            
            # if polygon_8 is active this frame...
            if polygon_8.status == STARTED:
                # update params
                pass
            
            # *polygon_9* updates
            
            # if polygon_9 is starting this frame...
            if polygon_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_9.frameNStart = frameN  # exact frame index
                polygon_9.tStart = t  # local t and not account for scr refresh
                polygon_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_9.started')
                # update status
                polygon_9.status = STARTED
                polygon_9.setAutoDraw(True)
            
            # if polygon_9 is active this frame...
            if polygon_9.status == STARTED:
                # update params
                pass
            
            # *polygon_Back* updates
            
            # if polygon_Back is starting this frame...
            if polygon_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_Back.frameNStart = frameN  # exact frame index
                polygon_Back.tStart = t  # local t and not account for scr refresh
                polygon_Back.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_Back, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_Back.started')
                # update status
                polygon_Back.status = STARTED
                polygon_Back.setAutoDraw(True)
            
            # if polygon_Back is active this frame...
            if polygon_Back.status == STARTED:
                # update params
                pass
            
            # *polygon_0* updates
            
            # if polygon_0 is starting this frame...
            if polygon_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_0.frameNStart = frameN  # exact frame index
                polygon_0.tStart = t  # local t and not account for scr refresh
                polygon_0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_0.started')
                # update status
                polygon_0.status = STARTED
                polygon_0.setAutoDraw(True)
            
            # if polygon_0 is active this frame...
            if polygon_0.status == STARTED:
                # update params
                pass
            
            # *polygon_Dot* updates
            
            # if polygon_Dot is starting this frame...
            if polygon_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                polygon_Dot.frameNStart = frameN  # exact frame index
                polygon_Dot.tStart = t  # local t and not account for scr refresh
                polygon_Dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_Dot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_Dot.started')
                # update status
                polygon_Dot.status = STARTED
                polygon_Dot.setAutoDraw(True)
            
            # if polygon_Dot is active this frame...
            if polygon_Dot.status == STARTED:
                # update params
                pass
            
            # *text_1* updates
            
            # if text_1 is starting this frame...
            if text_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_1.frameNStart = frameN  # exact frame index
                text_1.tStart = t  # local t and not account for scr refresh
                text_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_1, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_1.status = STARTED
                text_1.setAutoDraw(True)
            
            # if text_1 is active this frame...
            if text_1.status == STARTED:
                # update params
                pass
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
                # update params
                pass
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *text_Back* updates
            
            # if text_Back is starting this frame...
            if text_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Back.frameNStart = frameN  # exact frame index
                text_Back.tStart = t  # local t and not account for scr refresh
                text_Back.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Back, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_Back.status = STARTED
                text_Back.setAutoDraw(True)
            
            # if text_Back is active this frame...
            if text_Back.status == STARTED:
                # update params
                pass
            
            # *text_0* updates
            
            # if text_0 is starting this frame...
            if text_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_0.frameNStart = frameN  # exact frame index
                text_0.tStart = t  # local t and not account for scr refresh
                text_0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_0, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_0.status = STARTED
                text_0.setAutoDraw(True)
            
            # if text_0 is active this frame...
            if text_0.status == STARTED:
                # update params
                pass
            
            # *text_Dot* updates
            
            # if text_Dot is starting this frame...
            if text_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_Dot.frameNStart = frameN  # exact frame index
                text_Dot.tStart = t  # local t and not account for scr refresh
                text_Dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Dot, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_Dot.status = STARTED
                text_Dot.setAutoDraw(True)
            
            # if text_Dot is active this frame...
            if text_Dot.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=BorgRating,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                BorgRating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BorgRating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BorgRating" ---
        for thisComponent in BorgRating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for BorgRating
        BorgRating.tStop = globalClock.getTime(format='float')
        BorgRating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('BorgRating.stopped', BorgRating.tStop)
        thisExp.addData('inputText.routineEndVal', inputText)  # Save end Routine value
        # check responses
        if VER_key_resp.keys in ['', [], None]:  # No response was made
            VER_key_resp.keys = None
        trials_noise.addData('VER_key_resp.keys',VER_key_resp.keys)
        if VER_key_resp.keys != None:  # we had a response
            trials_noise.addData('VER_key_resp.rt', VER_key_resp.rt)
            trials_noise.addData('VER_key_resp.duration', VER_key_resp.duration)
        # Run 'End Routine' code from code_Borg
        #sd.stop()
        
        # Load the CSV file
        with open(filename+'_Ratings.csv', mode='r') as file:
            reader = csv.reader(file)
            segData = [row for row in reader]
        
        VER_num += 1
        
        new_row = ['VER_'+str(VER_num), inputText]
        segData.append(new_row)
        
        # Save the modified data back to the CSV file
        with open(filename+'_Ratings.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(segData)
        
        
        # let's store the final text string into the results finle...
        thisExp.addData('VER', inputText)
        inputText="#"
        thisExp.addData('Task','VER')
        # store data for trials_noise (TrialHandler)
        trials_noise.addData('mouse_Borg.x', mouse_Borg.x)
        trials_noise.addData('mouse_Borg.y', mouse_Borg.y)
        trials_noise.addData('mouse_Borg.leftButton', mouse_Borg.leftButton)
        trials_noise.addData('mouse_Borg.midButton', mouse_Borg.midButton)
        trials_noise.addData('mouse_Borg.rightButton', mouse_Borg.rightButton)
        trials_noise.addData('mouse_Borg.time', mouse_Borg.time)
        trials_noise.addData('button_Borg.numClicks', button_Borg.numClicks)
        if button_Borg.numClicks:
           trials_noise.addData('button_Borg.timesOn', button_Borg.timesOn)
           trials_noise.addData('button_Borg.timesOff', button_Borg.timesOff)
        else:
           trials_noise.addData('button_Borg.timesOn', "")
           trials_noise.addData('button_Borg.timesOff', "")
        # the Routine "BorgRating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_noise as finished
        if hasattr(thisTrials_noise, 'status'):
            thisTrials_noise.status = FINISHED
        # if awaiting a pause, pause now
        if trials_noise.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_noise.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_noise'
    trials_noise.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials_TLX = data.TrialHandler2(
        name='trials_TLX',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('PsychoPy/TLX.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_TLX)  # add the loop to the experiment
    thisTrials_TLX = trials_TLX.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_TLX.rgb)
    if thisTrials_TLX != None:
        for paramName in thisTrials_TLX:
            globals()[paramName] = thisTrials_TLX[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_TLX in trials_TLX:
        trials_TLX.status = STARTED
        if hasattr(thisTrials_TLX, 'status'):
            thisTrials_TLX.status = STARTED
        currentLoop = trials_TLX
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_TLX.rgb)
        if thisTrials_TLX != None:
            for paramName in thisTrials_TLX:
                globals()[paramName] = thisTrials_TLX[paramName]
        
        # --- Prepare to start Routine "TLX" ---
        # create an object to store info about Routine TLX
        TLX = data.Routine(
            name='TLX',
            components=[text_TLXTitle, text_TLXDescription, slider_TLX, button_TLX],
        )
        TLX.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_TLX
        sd.stop()
        
        if TLX_Label_Num == 1:
            TLX_Labelstr = 'Very\nLow','','','','','','','','','','Very\nHigh'
        elif TLX_Label_Num == 2:
            TLX_Labelstr = 'Very\nGood','','','','','','','','','','Very\nPoor'
        text_TLXTitle.setText(TLX_Title)
        text_TLXDescription.setText(TLX_Description)
        slider_TLX.reset()
        # reset button_TLX to account for continued clicks & clear times on/off
        button_TLX.reset()
        # store start times for TLX
        TLX.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TLX.tStart = globalClock.getTime(format='float')
        TLX.status = STARTED
        thisExp.addData('TLX.started', TLX.tStart)
        TLX.maxDuration = None
        # keep track of which components have finished
        TLXComponents = TLX.components
        for thisComponent in TLX.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TLX" ---
        TLX.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_TLX, 'status') and thisTrials_TLX.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_TLXTitle* updates
            
            # if text_TLXTitle is starting this frame...
            if text_TLXTitle.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_TLXTitle.frameNStart = frameN  # exact frame index
                text_TLXTitle.tStart = t  # local t and not account for scr refresh
                text_TLXTitle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_TLXTitle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_TLXTitle.started')
                # update status
                text_TLXTitle.status = STARTED
                text_TLXTitle.setAutoDraw(True)
            
            # if text_TLXTitle is active this frame...
            if text_TLXTitle.status == STARTED:
                # update params
                pass
            
            # *text_TLXDescription* updates
            
            # if text_TLXDescription is starting this frame...
            if text_TLXDescription.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_TLXDescription.frameNStart = frameN  # exact frame index
                text_TLXDescription.tStart = t  # local t and not account for scr refresh
                text_TLXDescription.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_TLXDescription, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_TLXDescription.started')
                # update status
                text_TLXDescription.status = STARTED
                text_TLXDescription.setAutoDraw(True)
            
            # if text_TLXDescription is active this frame...
            if text_TLXDescription.status == STARTED:
                # update params
                pass
            
            # *slider_TLX* updates
            
            # if slider_TLX is starting this frame...
            if slider_TLX.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                slider_TLX.frameNStart = frameN  # exact frame index
                slider_TLX.tStart = t  # local t and not account for scr refresh
                slider_TLX.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_TLX, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_TLX.started')
                # update status
                slider_TLX.status = STARTED
                slider_TLX.setAutoDraw(True)
            
            # if slider_TLX is active this frame...
            if slider_TLX.status == STARTED:
                # update params
                pass
            # *button_TLX* updates
            
            # if button_TLX is starting this frame...
            if button_TLX.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                button_TLX.frameNStart = frameN  # exact frame index
                button_TLX.tStart = t  # local t and not account for scr refresh
                button_TLX.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_TLX, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_TLX.started')
                # update status
                button_TLX.status = STARTED
                win.callOnFlip(button_TLX.buttonClock.reset)
                button_TLX.setAutoDraw(True)
            
            # if button_TLX is active this frame...
            if button_TLX.status == STARTED:
                # update params
                button_TLX.setOpacity(None, log=False)
                # check whether button_TLX has been pressed
                if button_TLX.isClicked:
                    if not button_TLX.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_TLX.timesOn.append(button_TLX.buttonClock.getTime())
                        button_TLX.timesOff.append(button_TLX.buttonClock.getTime())
                    elif len(button_TLX.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_TLX.timesOff[-1] = button_TLX.buttonClock.getTime()
                    if not button_TLX.wasClicked:
                        # end routine when button_TLX is clicked
                        continueRoutine = False
                    if not button_TLX.wasClicked:
                        # run callback code when button_TLX is clicked
                        pass
            # take note of whether button_TLX was clicked, so that next frame we know if clicks are new
            button_TLX.wasClicked = button_TLX.isClicked and button_TLX.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=TLX,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                TLX.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TLX.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TLX" ---
        for thisComponent in TLX.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TLX
        TLX.tStop = globalClock.getTime(format='float')
        TLX.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TLX.stopped', TLX.tStop)
        
        # Run 'End Routine' code from code_TLX
        # Load the CSV file
        with open(filename+'_Ratings.csv', mode='r') as file:
            reader = csv.reader(file)
            segData = [row for row in reader]
        TLX_num += 1
        new_row = ['TLC_'+TLX_Title+'_'+str(TLX_num), slider_TLX.getRating()]
        segData.append(new_row)
        
        # Save the modified data back to the CSV file
        with open(filename+'_Ratings.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(segData)
        
        trials_TLX.addData('slider_TLX.response', slider_TLX.getRating())
        trials_TLX.addData('slider_TLX.rt', slider_TLX.getRT())
        trials_TLX.addData('button_TLX.numClicks', button_TLX.numClicks)
        if button_TLX.numClicks:
           trials_TLX.addData('button_TLX.timesOn', button_TLX.timesOn)
           trials_TLX.addData('button_TLX.timesOff', button_TLX.timesOff)
        else:
           trials_TLX.addData('button_TLX.timesOn', "")
           trials_TLX.addData('button_TLX.timesOff', "")
        # the Routine "TLX" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_TLX as finished
        if hasattr(thisTrials_TLX, 'status'):
            thisTrials_TLX.status = FINISHED
        # if awaiting a pause, pause now
        if trials_TLX.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_TLX.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_TLX'
    trials_TLX.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "BorgRating" ---
    # create an object to store info about Routine BorgRating
    BorgRating = data.Routine(
        name='BorgRating',
        components=[VER_key_resp, image_BorgScale, text_BorgScale, VER_num_text, mouse_Borg, button_Borg, polygon_1, polygon_2, polygon_3, polygon_4, polygon_5, polygon_6, polygon_7, polygon_8, polygon_9, polygon_Back, polygon_0, polygon_Dot, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9, text_Back, text_0, text_Dot],
    )
    BorgRating.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    inputText = '#'  # Set Routine start values for inputText
    # create starting attributes for VER_key_resp
    VER_key_resp.keys = []
    VER_key_resp.rt = []
    _VER_key_resp_allKeys = []
    # Run 'Begin Routine' code from code_Borg
    #sd.stop()
    
    theseKeys=""
    VER_num_text.alignHoriz ='left'
    first_press = True
    
    # setup some python lists for storing info about the mouse_Borg
    mouse_Borg.x = []
    mouse_Borg.y = []
    mouse_Borg.leftButton = []
    mouse_Borg.midButton = []
    mouse_Borg.rightButton = []
    mouse_Borg.time = []
    gotValidClick = False  # until a click is received
    # reset button_Borg to account for continued clicks & clear times on/off
    button_Borg.reset()
    # store start times for BorgRating
    BorgRating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BorgRating.tStart = globalClock.getTime(format='float')
    BorgRating.status = STARTED
    thisExp.addData('BorgRating.started', BorgRating.tStart)
    BorgRating.maxDuration = None
    # keep track of which components have finished
    BorgRatingComponents = BorgRating.components
    for thisComponent in BorgRating.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BorgRating" ---
    BorgRating.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *VER_key_resp* updates
        waitOnFlip = False
        
        # if VER_key_resp is starting this frame...
        if VER_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            VER_key_resp.frameNStart = frameN  # exact frame index
            VER_key_resp.tStart = t  # local t and not account for scr refresh
            VER_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_key_resp.started')
            # update status
            VER_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(VER_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(VER_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if VER_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = VER_key_resp.getKeys(keyList=['return','y','n','left','right','space','1','2','3','4','5','6','7','8','9','0','period','return','backspace','num_1','num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9','num_0','num_decimal','num_subtract','period','comma'], ignoreKeys=["escape"], waitRelease=False)
            _VER_key_resp_allKeys.extend(theseKeys)
            if len(_VER_key_resp_allKeys):
                VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
                VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
                VER_key_resp.duration = _VER_key_resp_allKeys[-1].duration
        # Run 'Each Frame' code from code_Borg
        if mouse_Borg.isPressedIn(polygon_1):
            NumResponse = '1'
        if mouse_Borg.isPressedIn(polygon_2):
            NumResponse = '2'
        if mouse_Borg.isPressedIn(polygon_3):
            NumResponse = '3'
        if mouse_Borg.isPressedIn(polygon_4):
            NumResponse = '4'
        if mouse_Borg.isPressedIn(polygon_5):
            NumResponse = '5'
        if mouse_Borg.isPressedIn(polygon_6):
            NumResponse = '6'
        if mouse_Borg.isPressedIn(polygon_7):
            NumResponse = '7'
        if mouse_Borg.isPressedIn(polygon_8):
            NumResponse = '8'
        if mouse_Borg.isPressedIn(polygon_9):
            NumResponse = '9'
        if mouse_Borg.isPressedIn(polygon_0):
            NumResponse = '0'
        if mouse_Borg.isPressedIn(polygon_Dot):
            NumResponse = ''
            if len(inputText) == 0:
                inputText += '0.'
            else:
                inputText += '.'
            time.sleep(.1)
        
        if NumResponse != '':
            if inputText == '#':
                inputText = NumResponse
            else:
                inputText += NumResponse
            NumResponse = ''
            time.sleep(.1)
            
        if mouse_Borg.isPressedIn(polygon_Back):
            inputText = inputText[:-1]  # lose the final character
            time.sleep(.1)
        
        if len(_VER_key_resp_allKeys) and keyReady:
            keyReady = False
            VER_key_resp.keys = _VER_key_resp_allKeys[-1].name  # just the last key pressed
            VER_key_resp.rt = _VER_key_resp_allKeys[-1].rt
            if first_press:
                inputText = ''
                first_press = False
        
            elif VER_key_resp.keys in ['right','space']:
                continueRoutine = False
                
            elif VER_key_resp.keys in ['num_1', '1']:
                inputText += '1'
        
            elif VER_key_resp.keys in ['num_2', '2']:
                inputText += '2'
        
            elif VER_key_resp.keys in ['num_3', '3']:
                inputText += '3'
        
            elif VER_key_resp.keys in ['num_4', '4']:
                inputText += '4'
        
            elif VER_key_resp.keys in ['num_5', '5']:
                inputText += '5'
        
            elif VER_key_resp.keys in ['num_6', '6']:
                inputText += '6'
        
            elif VER_key_resp.keys in ['num_7', '7']:
                inputText += '7'
        
            elif VER_key_resp.keys in ['num_8', '8']:
                inputText += '8'
        
            elif VER_key_resp.keys in ['num_9', '9']:
                inputText += '9'
        
            elif VER_key_resp.keys in ['num_0', '0']:
                inputText += '0'
        
            elif VER_key_resp.keys in ['period', 'comma','num_decimal']:
                if len(inputText) == 0:
                    inputText += '0,'
                else:
                    inputText += '.'
        
            elif VER_key_resp.keys in ['backspace','num_subtract']:
                inputText = inputText[:-1]  # lose the final character
        
            #VER_key_resp = []
            _VER_key_resp_allKeys = []
            theseKeys = []
        elif len(_VER_key_resp_allKeys) == 0:
            keyReady = True
        
        # *image_BorgScale* updates
        
        # if image_BorgScale is starting this frame...
        if image_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            image_BorgScale.frameNStart = frameN  # exact frame index
            image_BorgScale.tStart = t  # local t and not account for scr refresh
            image_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_BorgScale, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_BorgScale.started')
            # update status
            image_BorgScale.status = STARTED
            image_BorgScale.setAutoDraw(True)
        
        # if image_BorgScale is active this frame...
        if image_BorgScale.status == STARTED:
            # update params
            pass
        
        # *text_BorgScale* updates
        
        # if text_BorgScale is starting this frame...
        if text_BorgScale.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_BorgScale.frameNStart = frameN  # exact frame index
            text_BorgScale.tStart = t  # local t and not account for scr refresh
            text_BorgScale.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_BorgScale, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_BorgScale.started')
            # update status
            text_BorgScale.status = STARTED
            text_BorgScale.setAutoDraw(True)
        
        # if text_BorgScale is active this frame...
        if text_BorgScale.status == STARTED:
            # update params
            pass
        
        # *VER_num_text* updates
        
        # if VER_num_text is starting this frame...
        if VER_num_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            VER_num_text.frameNStart = frameN  # exact frame index
            VER_num_text.tStart = t  # local t and not account for scr refresh
            VER_num_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(VER_num_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'VER_num_text.started')
            # update status
            VER_num_text.status = STARTED
            VER_num_text.setAutoDraw(True)
        
        # if VER_num_text is active this frame...
        if VER_num_text.status == STARTED:
            # update params
            VER_num_text.setText(inputText, log=False)
        # *mouse_Borg* updates
        if mouse_Borg.status == STARTED:  # only update if started and not finished!
            buttons = mouse_Borg.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_Borg.getPos()
                    mouse_Borg.x.append(x)
                    mouse_Borg.y.append(y)
                    buttons = mouse_Borg.getPressed()
                    mouse_Borg.leftButton.append(buttons[0])
                    mouse_Borg.midButton.append(buttons[1])
                    mouse_Borg.rightButton.append(buttons[2])
                    mouse_Borg.time.append(mouse_Borg.mouseClock.getTime())
        # *button_Borg* updates
        
        # if button_Borg is starting this frame...
        if button_Borg.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            button_Borg.frameNStart = frameN  # exact frame index
            button_Borg.tStart = t  # local t and not account for scr refresh
            button_Borg.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_Borg, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_Borg.started')
            # update status
            button_Borg.status = STARTED
            win.callOnFlip(button_Borg.buttonClock.reset)
            button_Borg.setAutoDraw(True)
        
        # if button_Borg is active this frame...
        if button_Borg.status == STARTED:
            # update params
            pass
            # check whether button_Borg has been pressed
            if button_Borg.isClicked:
                if not button_Borg.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_Borg.timesOn.append(button_Borg.buttonClock.getTime())
                    button_Borg.timesOff.append(button_Borg.buttonClock.getTime())
                elif len(button_Borg.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_Borg.timesOff[-1] = button_Borg.buttonClock.getTime()
                if not button_Borg.wasClicked:
                    # run callback code when button_Borg is clicked
                    if inputText != '' and inputText != '#':
                        continueRoutine = False
                        break
        # take note of whether button_Borg was clicked, so that next frame we know if clicks are new
        button_Borg.wasClicked = button_Borg.isClicked and button_Borg.status == STARTED
        
        # *polygon_1* updates
        
        # if polygon_1 is starting this frame...
        if polygon_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_1.frameNStart = frameN  # exact frame index
            polygon_1.tStart = t  # local t and not account for scr refresh
            polygon_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_1.started')
            # update status
            polygon_1.status = STARTED
            polygon_1.setAutoDraw(True)
        
        # if polygon_1 is active this frame...
        if polygon_1.status == STARTED:
            # update params
            pass
        
        # *polygon_2* updates
        
        # if polygon_2 is starting this frame...
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_2.started')
            # update status
            polygon_2.status = STARTED
            polygon_2.setAutoDraw(True)
        
        # if polygon_2 is active this frame...
        if polygon_2.status == STARTED:
            # update params
            pass
        
        # *polygon_3* updates
        
        # if polygon_3 is starting this frame...
        if polygon_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_3.frameNStart = frameN  # exact frame index
            polygon_3.tStart = t  # local t and not account for scr refresh
            polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_3.started')
            # update status
            polygon_3.status = STARTED
            polygon_3.setAutoDraw(True)
        
        # if polygon_3 is active this frame...
        if polygon_3.status == STARTED:
            # update params
            pass
        
        # *polygon_4* updates
        
        # if polygon_4 is starting this frame...
        if polygon_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_4.frameNStart = frameN  # exact frame index
            polygon_4.tStart = t  # local t and not account for scr refresh
            polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_4.started')
            # update status
            polygon_4.status = STARTED
            polygon_4.setAutoDraw(True)
        
        # if polygon_4 is active this frame...
        if polygon_4.status == STARTED:
            # update params
            pass
        
        # *polygon_5* updates
        
        # if polygon_5 is starting this frame...
        if polygon_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_5.frameNStart = frameN  # exact frame index
            polygon_5.tStart = t  # local t and not account for scr refresh
            polygon_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_5.started')
            # update status
            polygon_5.status = STARTED
            polygon_5.setAutoDraw(True)
        
        # if polygon_5 is active this frame...
        if polygon_5.status == STARTED:
            # update params
            pass
        
        # *polygon_6* updates
        
        # if polygon_6 is starting this frame...
        if polygon_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_6.frameNStart = frameN  # exact frame index
            polygon_6.tStart = t  # local t and not account for scr refresh
            polygon_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_6.started')
            # update status
            polygon_6.status = STARTED
            polygon_6.setAutoDraw(True)
        
        # if polygon_6 is active this frame...
        if polygon_6.status == STARTED:
            # update params
            pass
        
        # *polygon_7* updates
        
        # if polygon_7 is starting this frame...
        if polygon_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_7.frameNStart = frameN  # exact frame index
            polygon_7.tStart = t  # local t and not account for scr refresh
            polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_7.started')
            # update status
            polygon_7.status = STARTED
            polygon_7.setAutoDraw(True)
        
        # if polygon_7 is active this frame...
        if polygon_7.status == STARTED:
            # update params
            pass
        
        # *polygon_8* updates
        
        # if polygon_8 is starting this frame...
        if polygon_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_8.frameNStart = frameN  # exact frame index
            polygon_8.tStart = t  # local t and not account for scr refresh
            polygon_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_8.started')
            # update status
            polygon_8.status = STARTED
            polygon_8.setAutoDraw(True)
        
        # if polygon_8 is active this frame...
        if polygon_8.status == STARTED:
            # update params
            pass
        
        # *polygon_9* updates
        
        # if polygon_9 is starting this frame...
        if polygon_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_9.frameNStart = frameN  # exact frame index
            polygon_9.tStart = t  # local t and not account for scr refresh
            polygon_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_9.started')
            # update status
            polygon_9.status = STARTED
            polygon_9.setAutoDraw(True)
        
        # if polygon_9 is active this frame...
        if polygon_9.status == STARTED:
            # update params
            pass
        
        # *polygon_Back* updates
        
        # if polygon_Back is starting this frame...
        if polygon_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_Back.frameNStart = frameN  # exact frame index
            polygon_Back.tStart = t  # local t and not account for scr refresh
            polygon_Back.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_Back, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_Back.started')
            # update status
            polygon_Back.status = STARTED
            polygon_Back.setAutoDraw(True)
        
        # if polygon_Back is active this frame...
        if polygon_Back.status == STARTED:
            # update params
            pass
        
        # *polygon_0* updates
        
        # if polygon_0 is starting this frame...
        if polygon_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_0.frameNStart = frameN  # exact frame index
            polygon_0.tStart = t  # local t and not account for scr refresh
            polygon_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_0.started')
            # update status
            polygon_0.status = STARTED
            polygon_0.setAutoDraw(True)
        
        # if polygon_0 is active this frame...
        if polygon_0.status == STARTED:
            # update params
            pass
        
        # *polygon_Dot* updates
        
        # if polygon_Dot is starting this frame...
        if polygon_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            polygon_Dot.frameNStart = frameN  # exact frame index
            polygon_Dot.tStart = t  # local t and not account for scr refresh
            polygon_Dot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_Dot, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_Dot.started')
            # update status
            polygon_Dot.status = STARTED
            polygon_Dot.setAutoDraw(True)
        
        # if polygon_Dot is active this frame...
        if polygon_Dot.status == STARTED:
            # update params
            pass
        
        # *text_1* updates
        
        # if text_1 is starting this frame...
        if text_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_1.frameNStart = frameN  # exact frame index
            text_1.tStart = t  # local t and not account for scr refresh
            text_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_1.status = STARTED
            text_1.setAutoDraw(True)
        
        # if text_1 is active this frame...
        if text_1.status == STARTED:
            # update params
            pass
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # *text_8* updates
        
        # if text_8 is starting this frame...
        if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_8.frameNStart = frameN  # exact frame index
            text_8.tStart = t  # local t and not account for scr refresh
            text_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_8.status = STARTED
            text_8.setAutoDraw(True)
        
        # if text_8 is active this frame...
        if text_8.status == STARTED:
            # update params
            pass
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *text_Back* updates
        
        # if text_Back is starting this frame...
        if text_Back.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Back.frameNStart = frameN  # exact frame index
            text_Back.tStart = t  # local t and not account for scr refresh
            text_Back.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Back, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_Back.status = STARTED
            text_Back.setAutoDraw(True)
        
        # if text_Back is active this frame...
        if text_Back.status == STARTED:
            # update params
            pass
        
        # *text_0* updates
        
        # if text_0 is starting this frame...
        if text_0.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_0.frameNStart = frameN  # exact frame index
            text_0.tStart = t  # local t and not account for scr refresh
            text_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_0, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_0.status = STARTED
            text_0.setAutoDraw(True)
        
        # if text_0 is active this frame...
        if text_0.status == STARTED:
            # update params
            pass
        
        # *text_Dot* updates
        
        # if text_Dot is starting this frame...
        if text_Dot.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_Dot.frameNStart = frameN  # exact frame index
            text_Dot.tStart = t  # local t and not account for scr refresh
            text_Dot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Dot, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_Dot.status = STARTED
            text_Dot.setAutoDraw(True)
        
        # if text_Dot is active this frame...
        if text_Dot.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BorgRating,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BorgRating.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BorgRating.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BorgRating" ---
    for thisComponent in BorgRating.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BorgRating
    BorgRating.tStop = globalClock.getTime(format='float')
    BorgRating.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BorgRating.stopped', BorgRating.tStop)
    thisExp.addData('inputText.routineEndVal', inputText)  # Save end Routine value
    # check responses
    if VER_key_resp.keys in ['', [], None]:  # No response was made
        VER_key_resp.keys = None
    thisExp.addData('VER_key_resp.keys',VER_key_resp.keys)
    if VER_key_resp.keys != None:  # we had a response
        thisExp.addData('VER_key_resp.rt', VER_key_resp.rt)
        thisExp.addData('VER_key_resp.duration', VER_key_resp.duration)
    # Run 'End Routine' code from code_Borg
    #sd.stop()
    
    # Load the CSV file
    with open(filename+'_Ratings.csv', mode='r') as file:
        reader = csv.reader(file)
        segData = [row for row in reader]
    
    VER_num += 1
    
    new_row = ['VER_'+str(VER_num), inputText]
    segData.append(new_row)
    
    # Save the modified data back to the CSV file
    with open(filename+'_Ratings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(segData)
    
    
    # let's store the final text string into the results finle...
    thisExp.addData('VER', inputText)
    inputText="#"
    thisExp.addData('Task','VER')
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_Borg.x', mouse_Borg.x)
    thisExp.addData('mouse_Borg.y', mouse_Borg.y)
    thisExp.addData('mouse_Borg.leftButton', mouse_Borg.leftButton)
    thisExp.addData('mouse_Borg.midButton', mouse_Borg.midButton)
    thisExp.addData('mouse_Borg.rightButton', mouse_Borg.rightButton)
    thisExp.addData('mouse_Borg.time', mouse_Borg.time)
    thisExp.addData('button_Borg.numClicks', button_Borg.numClicks)
    if button_Borg.numClicks:
       thisExp.addData('button_Borg.timesOn', button_Borg.timesOn)
       thisExp.addData('button_Borg.timesOff', button_Borg.timesOff)
    else:
       thisExp.addData('button_Borg.timesOn', "")
       thisExp.addData('button_Borg.timesOff', "")
    thisExp.nextEntry()
    # the Routine "BorgRating" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[text_End, key_resp_End],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_End
    trial_filename = './Sounds/End.wav'
    trialdata, trialfs = sf.read(trial_filename, dtype='float32')  
    
    sd.play(trialdata, samplerate=trialfs)
    
    # create starting attributes for key_resp_End
    key_resp_End.keys = []
    key_resp_End.rt = []
    _key_resp_End_allKeys = []
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_End* updates
        
        # if text_End is starting this frame...
        if text_End.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            text_End.frameNStart = frameN  # exact frame index
            text_End.tStart = t  # local t and not account for scr refresh
            text_End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_End.started')
            # update status
            text_End.status = STARTED
            text_End.setAutoDraw(True)
        
        # if text_End is active this frame...
        if text_End.status == STARTED:
            # update params
            pass
        
        # *key_resp_End* updates
        waitOnFlip = False
        
        # if key_resp_End is starting this frame...
        if key_resp_End.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_End.frameNStart = frameN  # exact frame index
            key_resp_End.tStart = t  # local t and not account for scr refresh
            key_resp_End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_End.started')
            # update status
            key_resp_End.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_End.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_End.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_End.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_End.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_End_allKeys.extend(theseKeys)
            if len(_key_resp_End_allKeys):
                key_resp_End.keys = _key_resp_End_allKeys[-1].name  # just the last key pressed
                key_resp_End.rt = _key_resp_End_allKeys[-1].rt
                key_resp_End.duration = _key_resp_End_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            End.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    # Run 'End Routine' code from code_End
    sd.stop()
    # check responses
    if key_resp_End.keys in ['', [], None]:  # No response was made
        key_resp_End.keys = None
    thisExp.addData('key_resp_End.keys',key_resp_End.keys)
    if key_resp_End.keys != None:  # we had a response
        thisExp.addData('key_resp_End.rt', key_resp_End.rt)
        thisExp.addData('key_resp_End.duration', key_resp_End.duration)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
