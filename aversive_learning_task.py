# -*- coding: utf-8 -*-
from psychopy import core, visual, event, gui, monitors, data
import pandas as pd
import os
import yaml
import numpy as np
import random
from psychopy.iohub.constants import EventConstants
from psychopy.hardware import keyboard

def rgb_convert(rgb):
    return tuple(i * 2 - 1 for i in rgb)

class AversiveLearningTask(object):

    def __init__(self, config=None):

        # Load config
        # this sets self.config (the config attribute of our experiment class) to a dictionary containing all the values
        # in our config file. The keys of this dictionary correspond to the section headings in the config file, and
        # each value is another dictionary with keys that refer to the subheadings in the config file. This means you can
        # reference things in the dictionary by e.g. self.config['heading']['subheading']

        with open(config) as f:
            self.config = yaml.load(f)

        # ------------------------------------#
        # Subject/task information and saving #
        # ------------------------------------#

        # Enter subject ID and other information
        dialogue = gui.Dlg()
        dialogue.addText("Subject info")
        dialogue.addField('Subject ID')
        dialogue.show()

        # check that values are OK and assign them to variables
        if dialogue.OK:
            self.subject_id = dialogue.data[0]
        else:
            core.quit()

        # Recode blank subject ID to zero - useful for testing
        if self.subject_id == '':
            self.subject_id = '0'

        # Folder for saving data
        self.save_folder = self.config['directories']['saved_data']
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        self.save_path = '{0}/Subject{1}_data.csv'.format(self.save_folder, self.subject_id, data.getDateStr())

        # Data to be saved
        self.data = dict(trial_number=[],  # Trial number
                         response=[],  # Subject's response
                         block=[],
                         first_stage_set=[],
                         x_flipped=[]) 

        # ------------- #
        # Task settings #
        # ------------- #

        self.outcome_info = pd.read_csv(self.config['task_settings']['outcome_info_file'])
        self.outcome_info['block'] = np.floor(np.arange(len(self.outcome_info)) / float(self.config['task_settings']['trials_per_block'])).astype(int)
        self.n_blocks = len(self.outcome_info['block'].unique())
        self.trials_per_block = np.unique(self.outcome_info['block'], return_counts=True)[1]
        self.current_trial = 0

        self.outcome_info_practice = pd.read_csv(self.config['task_settings']['practice_outcome_info_file'])

        # -----------------------#
        # Monitor & window setup #
        # -----------------------#

        monitor = monitors.Monitor('monitor', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=self.config['task_settings']['fullscreen'],
                                 allowGUI=False,
                                 color='#c2c2c2',
                                 units='deg', colorSpace='rgb')
        self.win.mouseVisible = False  # make the mouse invisible
        self.frame_rate = 60

        # Inputs
        self.kb = keyboard.Keyboard()

        self.left_key = self.config['response_keys']['left_key']
        self.right_key = self.config['response_keys']['right_key']

        # Clock
        self.clock = core.Clock()

        # --------#
        # Stimuli #
        # --------#

        self.instruction_text = visual.TextStim(win=self.win, height=self.config['stimuli']['text_size'],
                                                color=self.config['stimuli']['text_colour'], wrapWidth=30)
        self.instruction_text.fontFiles = [self.config['fonts']['font_path']]  # Arial is horrible
        self.instruction_text.font = self.config['fonts']['font_name']

        img_size = self.config['stimuli']['image_size']

        # First stage
        first_stage_images_A = [self.config['stimuli']['stimulus_{0}_image'.format(i+1)] for i in range(2)]
        first_stage_images_B = [self.config['stimuli']['stimulus_{0}_image'.format(i+1)] for i in range(2, 4)]
        random.shuffle(first_stage_images_A)
        random.shuffle(first_stage_images_B)
        self.first_stage_stimuli = {'set_A': {},
                                    'set_B': {}}
        self.first_stage_stimuli['set_A']['stim_1'] = StateImage(0, 0, img_size, first_stage_images_A[0], self)
        self.first_stage_stimuli['set_A']['stim_2'] = StateImage(0, 0, img_size, first_stage_images_A[1], self)
        self.first_stage_stimuli['set_B']['stim_1'] = StateImage(0, 0, img_size, first_stage_images_B[0], self)
        self.first_stage_stimuli['set_B']['stim_2'] = StateImage(0, 0, img_size, first_stage_images_B[1], self)

        # Second stage
        second_stage_images = [self.config['stimuli']['stimulus_{0}_image'.format(i)] for i in range(5, 7)]
        random.shuffle(second_stage_images)
        self.second_stage_stimuli = {}
        self.second_stage_stimuli['stim_1'] = StateImage(self.config['stimuli']['left_stim_xpos'], 0, img_size, second_stage_images[0], self)
        self.second_stage_stimuli['stim_2'] = StateImage(self.config['stimuli']['left_stim_xpos'], 0, img_size, second_stage_images[1], self)

        self.outcome_stimuli = {}
        self.outcome_stimuli['shock'] = StateImage(self.config['stimuli']['right_stim_xpos'], 0, img_size, self.config['stimuli']['shock_image'], self)
        self.outcome_stimuli['safe'] = StateImage(self.config['stimuli']['right_stim_xpos'], 0, img_size, self.config['stimuli']['safe_image'], self)


    def run(self):

        """
        Runs the experiment
        """

        training_passed = False

        # Practice confidence ratings
        self.main_instructions(self.load_instructions(self.config['instructions']['start_instructions']))

        # Repeat training until criterion reached
        while not training_passed:
            self.main_instructions(self.load_instructions(self.config['instructions']['transition_training_instructions']))
            self.run_block(self.config['task_settings']['n_training_trials'], block=0, practice=True, show_outcome=False)
            self.main_instructions(self.load_instructions(self.config['instructions']['transition_test_instructions']))
            training_passed = self.run_transition_training(self.config['task_settings']['n_test_trials'])
            self.current_trial = 0 

        # Practice trials 
        self.main_instructions(self.load_instructions(self.config['instructions']['practice_instructions']))
        self.run_block(len(self.outcome_info_practice), block=0, practice=True)
        self.current_trial = 0 

        block_number = 0  # count blocks

        self.main_instructions(self.load_instructions(self.config['instructions']['task_instructions']))

        # Loop through blocks
        for block in range(block_number, self.n_blocks):
            print("Block {0} / {1}\n" \
            "-----------------".format(block + 1, self.n_blocks))
            self.run_block(self.trials_per_block[block], block=block, practice=False)
            block_number += 1

            # Break screen
            self.main_instructions(self.load_instructions(self.config['instructions']['break_instructions']))

        # End screen
        self.main_instructions(self.load_instructions(self.config['instructions']['end_instructions']))

    def run_block(self, n_trials, block, practice, show_outcome=True):

        for i in range(n_trials):  # TRIAL LOOP - everything in here is repeated each trial

            print("Trial {0} / {1}".format(i + 1, n_trials))

            x_flipped = bool(np.random.randint(0,2))
            trial = AversiveLearningTrial(self, self.current_trial, practice=practice, x_flipped=x_flipped)
            trial.run(show_outcome=show_outcome)
            if not practice:
                trial.save_data(self.save_path, block=block)
            self.current_trial += 1

    def run_transition_training(self, n_trials):

        n_correct = 0

        for i in range(n_trials):  # TRIAL LOOP - everything in here is repeated each trial

            print("Trial {0} / {1}".format(i + 1, n_trials))

            x_flipped = bool(np.random.randint(0,2))
            trial = AversiveLearningTransitionTest(self, self.current_trial, practice=False, x_flipped=x_flipped)
            correct = trial.run()
            if correct:
                n_correct += 1
            self.current_trial += 1

        if n_correct / n_trials > self.config['task_settings']['transition_training_threshold']:
            return True
        else:
            return False



    def load_instructions(self, text_file):

        """
        Loads a text file containing instructions and splits it into a list of strings
        Args:
            text_file: A text file containing instructions, with each page preceded by an asterisk
        Returns: List of instruction strings
        """

        with open(text_file, 'r') as f:
            instructions = f.read()

        return instructions.split('*')

    def instructions(self, text, max_wait=2):

        """
        Shows instruction text
        Args:
            text: Text to display
            max_wait: The maximum amount of time to wait for a response before moving on
        Returns:
        """

        # set text
        self.instruction_text.text = text
        # draw
        self.instruction_text.draw()
        if max_wait > 0:
            self.win.flip()
        # waitkeys
        event.waitKeys(maxWait=max_wait, keyList=['space', '1', ' '])

    def main_instructions(self, text_file, continue_keys=['a', 'space'], return_keys='l'):

        """
        Displays instruction text files for main instructions, training instructions, task instructions
        Args:
            text_file: A list of strings
            continue_keys: Keys to continue task
            return_key: Used to return to something else
        Returns:
            True if continuing, False if returning
        """

        if continue_keys is not None:
            continue_keys = ['escape', 'esc'] + list(continue_keys)

        if return_keys is not None:
            return_keys = list(return_keys)
        else:
            return_keys = []

        if not isinstance(text_file, list):
            raise TypeError("Text input is not a list")
        print(text_file)
        for i in text_file:
            self.instruction_text.text = i
            self.instruction_text.draw()
            self.win.flip()
            core.wait(1)
            key = event.waitKeys(keyList=continue_keys + return_keys)
            if key[0] in ['escape', 'esc']:
                core.quit()
            # elif key[0] in continue_keys:
            #     return True
            elif key[0] in return_keys:
                print("Returning")
                return False



class StateImage():

    def __init__(self, xpos, ypos, size, image, task, **kwargs):
        self.xpos = xpos
        self.ypos = ypos
        self.size = size
        self.task = task
        self.highlighted = False

        self.image_stim = visual.ImageStim(self.task.win, image=image, pos=[xpos, ypos], size=size, **kwargs)

        rect_vertices = [[xpos-(size/2), ypos-(size/2)],
                         [xpos-(size/2), ypos+(size/2)],
                         [xpos+(size/2), ypos+(size/2)],
                         [xpos+(size/2), ypos-(size/2)]]
        self.rect_stim = visual.ShapeStim(self.task.win, vertices=rect_vertices, lineColor='red', lineWidth=0)

    def get_rect_vertices(self):
        xpos = self.image_stim.pos[0]
        ypos = self.image_stim.pos[1]
        size = self.image_stim.size[0]
        rect_vertices = [[xpos-(size/2), ypos-(size/2)],
                    [xpos-(size/2), ypos+(size/2)],
                    [xpos+(size/2), ypos+(size/2)],
                    [xpos+(size/2), ypos-(size/2)]]
        return rect_vertices

    def draw(self):

        self.image_stim.draw()
        self.rect_stim.draw()
        # if self.highlighted:
        #     self.rect_stim.draw()

    def set_xpos(self, xpos):
        self.image_stim.pos = [xpos, self.ypos]
        

    def highlight(self):
        self.rect_stim.setVertices(self.get_rect_vertices())
        self.rect_stim.lineWidth = 5
        self.highlighted = True
    
    def highlight_off(self):
        self.rect_stim.lineWidth = 0
        self.highlighted = False


class AversiveLearningTrial(object):

    def __init__(self, task, trial_number, x_flipped=False, practice=False):

        """
        Averive learning trial class
        Parameters
        ----------
        task: Instance of the task class
        trial_number: Trial number
        stimulation: Indicates whether stimulation is given - can be used to turn off stimulation
        max_voltage: Voltage for stimulation
        """

        self.task = task
        self.trial_number = trial_number
        self.response = None
        self.rt = None
        self.x_flipped = x_flipped
        self.outcome = 'safe'
        self.n_shocks_given = 0
        self.second_stage_stimulus = None

        if practice:
            self.outcome_info = self.task.outcome_info_practice
        else:
            self.outcome_info = self.task.outcome_info

        self.first_stage_set = ['A', 'B'][self.outcome_info['first_stage_set'][trial_number]]

        self.x_positions = [self.task.config['stimuli']['left_stim_xpos'], 
                            self.task.config['stimuli']['right_stim_xpos']]
        if self.x_flipped:
            self.x_positions = self.x_positions[::-1]

        # Set up stimuli
        self.first_stage_stimuli = {}
        for i in range(2):
            this_stim = self.task.first_stage_stimuli['set_{0}'.format(self.first_stage_set)]['stim_{0}'.format(i+1)]
            this_stim.set_xpos(self.x_positions[i])
            if self.x_positions[i] < 0:
                self.first_stage_stimuli['left'] = this_stim
            else:
                self.first_stage_stimuli['right'] = this_stim      

    def show_first_stage_stimuli(self):

        if self.response is not None and self.rt is None:
            self.response = None

        # Draw stimuli
        self.first_stage_stimuli['left'].draw()
        self.first_stage_stimuli['right'].draw()

        # Get keybaord events
        keys = self.task.kb.getKeys([self.task.left_key, self.task.right_key])

        # Get response if key pressed
        if len(keys) and self.response is None:
            # Deal with keyboard presses
            for key_event in keys:

                # If the subject pressed a valid key
                key_pressed = (key_event.name, key_event.rt)
                self.rt = key_pressed[1]

                # Get response
                if key_pressed[0] == self.task.left_key:
                    self.first_stage_stimuli['left'].highlight()
                    if self.x_flipped:
                        self.response = 'B'
                    else:
                        self.response = 'A'
                elif key_pressed[0] == self.task.right_key:
                    self.first_stage_stimuli['right'].highlight()
                    if self.x_flipped:
                        self.response = 'A'
                    else:
                        self.response = 'B'

                # Set up second stage based on response
                if self.response == 'A':
                    if self.outcome_info['1_shock'][self.trial_number] == 1:
                        self.outcome = 'shock'
                elif self.response == 'B':
                    if self.outcome_info['2_shock'][self.trial_number] == 1:
                        self.outcome = 'shock'

    def show_second_state_stimuli(self):

        # Random response if none made
        if self.response is None:
            possible_responses = ['A', 'B']
            random.shuffle(possible_responses)
            self.response = possible_responses[0]
            print(r"RESPONSE", self.response)

        # Selecting A at the first stage = stim 1 at second stage, 
        if self.response == 'A':
            self.task.second_stage_stimuli['stim_1'].draw()
            self.second_stage_stimulus = 'A'
        elif self.response == 'B':
            self.task.second_stage_stimuli['stim_2'].draw()
            self.second_stage_stimulus = 'B'

    def show_outcome(self):

        self.task.outcome_stimuli[self.outcome].draw()
        """
        CODE FOR SHOCK ADMINISTRATION - NEEDS TO BE COMPLETED <<<
        """
        if self.n_shocks_given < self.task.config['stimulation']['n_shocks']:
            self.send_shock()
            self.n_shocks_given += 1

    def send_shock(self):

        """
        CODE FOR SHOCK ADMINISTRATION - NEEDS TO BE COMPLETED <<<
        """

        print('SHOCK')

    def save_data(self, save_path, block=0):

        self.task.data['trial_number'].append(self.trial_number)
        self.task.data['block'].append(block)
        self.task.data['response'].append(self.response)
        self.task.data['first_stage_set'].append(self.first_stage_set)
        self.task.data['x_flipped'].append(self.x_flipped)


        df = pd.DataFrame(self.task.data)

        df.to_csv(save_path, index=False)

    def run(self, show_outcome=True):

        """
        Runs the trial

        """

        # Reset the clock
        self.task.clock.reset()
        self.task.kb.clock.reset()

        continue_trial = True

        # RUN THE TRIAL
        while continue_trial:

            t = self.task.clock.getTime()  # get the time

            if t < self.task.config['durations']['first_stage_time']:
                self.show_first_stage_stimuli()

            if self.task.config['durations']['first_stage_time'] < t < self.task.config['durations']['first_stage_time'] + \
                                                                          self.task.config['durations']['second_stage_time'] + \
                                                                              self.task.config['durations']['outcome_time']:
                self.show_second_state_stimuli()

            if show_outcome and self.task.config['durations']['first_stage_time'] + \
                 self.task.config['durations']['second_stage_time'] < t < self.task.config['durations']['first_stage_time'] + \
                                                                          self.task.config['durations']['second_stage_time'] + \
                                                                              self.task.config['durations']['outcome_time']:
                self.show_outcome()

            # flip to draw everything
            self.task.win.flip()

            # End trial
            if t > self.task.config['durations']['first_stage_time'] + self.task.config['durations']['second_stage_time'] + self.task.config['durations']['outcome_time']:
                continue_trial = False

            # If the trial has ended
            if not continue_trial:
                self.first_stage_stimuli['left'].highlight_off()
                self.first_stage_stimuli['right'].highlight_off()
                print("Trial done")

            # quit if subject pressed scape
            if event.getKeys(["escape", "esc"]):
                core.quit()

class AversiveLearningTransitionTest(AversiveLearningTrial):


    def __init__(self, task, trial_number, x_flipped, practice):
        super().__init__(task, trial_number, x_flipped=x_flipped, practice=practice)

        self.correct = None
        self.correct_answer = None

    def show_success(self):
        
        if self.correct is None:
            if self.response == self.second_stage_stimulus:
                self.correct = True
            else:
                self.correct = False

        # set text
        if self.correct:
            self.task.instruction_text.text = 'CORRECT!'
        else:
            self.task.instruction_text.text = 'INCORRECT'
        # draw
        self.task.instruction_text.draw()


    def run(self):

        """
        Runs the trial

        """

        # Reset the clock
        self.task.clock.reset()
        self.task.kb.clock.reset()

        continue_trial = True

        # RUN THE TRIAL
        while continue_trial:

            t = self.task.clock.getTime()  # get the time

            if t < self.task.config['durations']['transition_training_probe_time']:
                self.show_second_state_stimuli()

            if self.task.config['durations']['transition_training_probe_time'] < t < self.task.config['durations']['transition_training_probe_time'] + \
                                                                                     self.task.config['durations']['transition_training_options_time']: 
                self.show_first_stage_stimuli()

            if self.task.config['durations']['transition_training_probe_time'] + \
               self.task.config['durations']['transition_training_options_time'] < t < self.task.config['durations']['transition_training_probe_time'] + \
                                                                                       self.task.config['durations']['transition_training_options_time'] + \
                                                                                       self.task.config['durations']['transition_training_success_time']:
                self.show_success()

            # flip to draw everything
            self.task.win.flip()

            # End trial
            if t > self.task.config['durations']['transition_training_probe_time'] + \
                   self.task.config['durations']['transition_training_options_time'] + \
                   self.task.config['durations']['transition_training_success_time']:
                continue_trial = False

            # If the trial has ended
            if not continue_trial:
                self.first_stage_stimuli['left'].highlight_off()
                self.first_stage_stimuli['right'].highlight_off()
                print("Trial done")

            # quit if subject pressed scape
            if event.getKeys(["escape", "esc"]):
                core.quit()

        return self.correct

def trigger(port, data):
    port.setData(data)


## RUN THE EXPERIMENT

# Create experiment class
task = AversiveLearningTask('aversive_learning_task_settings.yaml')

# Run the experiment
task.run()
