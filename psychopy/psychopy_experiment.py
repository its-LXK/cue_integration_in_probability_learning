# Full PsychoPy experiment script using predefined stimuli DataFrame (stimuli.csv)

#from tkinter import EXTENDED
from tabnanny import check
from tkinter import font
import pandas as pd
import numpy as np
from typing import List

from psychopy import visual, core, event
import csv
from dataclasses import dataclass
from pathlib import Path


# ----------------- SET PARTICIPANT ID ----------------------

ID: int = -1  # Set participant ID here

# ----------------- Load Predefined Stimuli -----------------

#STIMULI_CSV_PATH = "experimental_setup/stimuli.csv"
EXTENDED_STIMULI_CSV_PATH = "hyperparameter_and_stimuli/2025-06-15_13-07_all_balanced_stimuli.csv"
TEST_GROUP_1_CSV_PATH = "hyperparameter_and_stimuli/2025-06-17_18-17_group1_balanced_stimuli.csv"
TEST_GROUP_2_CSV_PATH = "hyperparameter_and_stimuli/2025-06-17_18-17_group2_balanced_stimuli.csv"
VARIABLES_CSV_PATH = "hyperparameter_and_stimuli/hyperparameters.csv"
#RESULTS_FOLDER_PATH = "experiment_data/"
RESULTS_FOLDER_PATH = "after_experimental_runs/"

# DEFINE TRAIN DATAFRAME
# Load extended stimuli DataFrame containing all possible permutations and shuffle it
all_stimuli_df = pd.read_csv(EXTENDED_STIMULI_CSV_PATH)
all_stimuli_df = all_stimuli_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
# Remove duplicates based on all columns except 'permutation_test' for the training phase
subset = [col for col in all_stimuli_df.columns if col != "permutation_test"]
train_stimuli_df = all_stimuli_df.copy().drop_duplicates(subset=subset)



# LOAD VARIABLES DATAFRAME
variables_df = pd.read_csv(VARIABLES_CSV_PATH)

# DEFINE TEST DATAFRAME BASED ON ID
if ID % 2 == 1:
    test_group_csv_path = TEST_GROUP_1_CSV_PATH
    GROUP_ID = 'group_1'
else:
    test_group_csv_path = TEST_GROUP_2_CSV_PATH
    GROUP_ID = 'group_2'
    
PARTICIPANT_ID = f'{ID}_{GROUP_ID}'

test_stimuli_df = pd.read_csv(test_group_csv_path)
test_stimuli_df = test_stimuli_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Select random trials from all trials for demo phase 3, each of which has a different variable list in ‘permutation_test’
demo_stimuli_df = test_stimuli_df.drop_duplicates(subset=['permutation_test'])  # ensure unique permutations

# Print loaded DataFrames for debugging
print("train_stimuli_df:", Path(EXTENDED_STIMULI_CSV_PATH).name, len(train_stimuli_df))
print("test_stimuli_df:", test_stimuli_df['permutation_test'].unique(), len(test_stimuli_df))
print("PARTICIPANT_ID:", PARTICIPANT_ID)

# Prepare demo trials
demo_train_trials = demo_stimuli_df
demo_test_trials = demo_stimuli_df

# ----------------- Experiment Settings -----------------

N_TRAIN_TRIALS = len(train_stimuli_df)  # total: 3 * this
N_TEST_BLOCKS = len(test_stimuli_df) # total: 3 * this (but there are two groups)

PAUSE_BETWEEN_TRIALS = 0.01
MAX_RESPONSE_TIME = 10.0 # 10 Sekunden
MAX_FEEDBACK_IMG_TIME = 2.0 # 2 Sekunden

SHOW_PROGESS_AFTER_N_TRIALS = 100  # Show progress every 100 trials

# ----------------- Layout Constants -----------------

FEEDBACK_CENTER_X = 0.55
FEEDBACK_BOX_POS = (FEEDBACK_CENTER_X, -0.1)
FEEDBACK_BOX_WIDTH = 0.8
FEEDBACK_BOX_HEIGHT = 1.6

SLIDER_Y = -0.3
SLIDER_HEIGHT = 0.9
SLIDER_SIZE = (0.05, SLIDER_HEIGHT)

LABEL_SPACING = 0.1
LABEL_TOP_Y = SLIDER_Y + SLIDER_HEIGHT / 2 + LABEL_SPACING
LABEL_BOTTOM_Y = SLIDER_Y - SLIDER_HEIGHT / 2 - LABEL_SPACING
VAR_LABEL_Y = SLIDER_Y + SLIDER_HEIGHT / 2 + 3 * LABEL_SPACING
LABEL_E_TEST = "Wahrscheinlichkeit der Einhornsichtung"

IMG_SIZE = 0.7
IMG_E1_PATH = "images/einhorn.png"
IMG_E0_PATH = "images/kein_einhorn.png"
FEEDBACK_IMG_Y = -0.25

TEXT_SIZE = 0.06

# ----------------- Data Structure -----------------

@dataclass
class Variable:
    variable: str
    name: str
    labels: list

def parse_labels(label_str: str) -> List[str]:
    # Remove square brackets if present and split by comma and also remove "'"
    return [l.strip().replace("'", "") for l in label_str.strip("[]").split(",")]

variables = {
    row["Variable"]: Variable(
        variable=row["Variable"],
        name=row["name"],
        labels=parse_labels(row["labels"]) if isinstance(row["labels"], str) else row["labels"]
    )
    for _, row in variables_df.iterrows()
}

# ----------------- Utility Functions -----------------

def create_slider(win, xpos, var: Variable, value, readOnly=True, fillColor="blue"):
    slider = visual.Slider(
        win, ticks=(-5, 5), labels=["", ""], granularity=0.001,
        style='rating', size=SLIDER_SIZE, pos=(xpos, SLIDER_Y),
        color="black", fillColor=fillColor, borderColor="black",
        readOnly=readOnly,
        font="Arial")
    slider.markerPos = value
    var_label = visual.TextStim(win, text=var.name, pos=(xpos, VAR_LABEL_Y), color="black", height=0.05, font="Arial")
    label_top = visual.TextStim(win, text=var.labels[1], pos=(xpos, LABEL_TOP_Y), height=0.04, color="black", font="Arial")
    label_bottom = visual.TextStim(win, text=var.labels[0], pos=(xpos, LABEL_BOTTOM_Y), height=0.04, color="black", font="Arial")
    return var_label, label_top, label_bottom, slider

def draw_slider_elements(elements):
    for item in elements:
        for obj in item:
            obj.draw()

def get_feedback_image_path(E):
    return IMG_E1_PATH if E == 1 else IMG_E0_PATH

def check_escape():
    keys = event.getKeys(keyList=['escape', 'space'], modifiers=True)
    if keys and any(k[0] == 'escape' for k in keys): 
            core.quit()
            
def show_progress_screen(win, current, total, phase_name="Phase"):
    """Displays a progress screen indicating completed trials/blocks."""
    progress_text = f"{phase_name} Fortschritt:\n\n{current} von {total} Trials abgeschlossen.\n\nDrücken Sie die Leertaste, wenn Sie bereit sind fortzufahren."
    visual.TextStim(win, 
                    text=progress_text, 
                    color="black", 
                    height=TEXT_SIZE-0.01, 
                    wrapWidth=1.5,
                    font="Arial").draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    check_escape()
    
def instructions_screen(win, instruction_text: str, text_size: float = TEXT_SIZE) -> None:
    """Displays an instruction screen for a given phase."""
    instruction = visual.TextStim(win, 
                                  text=instruction_text, 
                                  color="black", 
                                  height=text_size, 
                                  wrapWidth=1.75,
                                  font="Arial")
    instruction.draw()
    win.flip()
    check_escape()
    event.waitKeys(keyList=['space'])

                
# ----------------- Pause Between Trials Logic -----------------

def run_pause_screen(win, box_only=True):
    check_escape()
    feedback_box = visual.Rect(win, 
                               width=FEEDBACK_BOX_WIDTH, 
                               height=FEEDBACK_BOX_HEIGHT,
                               pos=FEEDBACK_BOX_POS, 
                               lineColor="black", 
                               fillColor=None)
    win.flip()
    if box_only:
        feedback_box.draw()
        win.flip()
    core.wait(PAUSE_BETWEEN_TRIALS)

def run_subtrial_pause(win, slider_elements):
    """Draw all cue sliders on the left but hide the feedback slider."""
    check_escape()
    feedback_box = visual.Rect(
        win,
        width=FEEDBACK_BOX_WIDTH,
        height=FEEDBACK_BOX_HEIGHT,
        pos=FEEDBACK_BOX_POS,
        lineColor="black",
        fillColor=None
    )

    draw_slider_elements(slider_elements)  # show all observed variable sliders
    feedback_box.draw()
    win.flip()
    core.wait(PAUSE_BETWEEN_TRIALS)


# ----------------- Training Trial Logic -----------------  

def run_training_trial(win, stim_row, results_file, block_idx, trial_idx):
    check_escape()
    print(stim_row)
    
    cue = stim_row['permutation_train'].strip(" []'")   
    cue_value = stim_row[cue]
    trial_id = stim_row['stimulus_id']
    E = stim_row['E']
    
    xpos = -0.5
    slider_elements = create_slider(win, 
                                    xpos=xpos, 
                                    var=variables[cue], 
                                    value=cue_value)
    
    feedback_box = visual.Rect(win, 
                               width=FEEDBACK_BOX_WIDTH, 
                               height=FEEDBACK_BOX_HEIGHT, 
                               pos=FEEDBACK_BOX_POS,
                               lineColor="black", 
                               fillColor=None)
    
    feedback_var_label = visual.TextStim(win, 
                                         text="Einhornsichtung?", 
                                         pos=(FEEDBACK_CENTER_X, VAR_LABEL_Y), 
                                         height=0.05, 
                                         color="black",
                                         font="Arial")

    est_slider = visual.Slider(win, 
                               ticks=(0, 100), 
                               labels=["", ""], 
                               granularity=1,
                               style='rating', 
                               size=SLIDER_SIZE, 
                               pos=(FEEDBACK_CENTER_X, SLIDER_Y),
                               color="black", 
                               fillColor="red", 
                               borderColor="black",
                               readOnly=False,
                               font="Arial")
    
    est_label_top = visual.TextStim(win, 
                                    text=variables['E'].labels[1], 
                                    pos=(FEEDBACK_CENTER_X, LABEL_TOP_Y), 
                                    height=0.04, 
                                    color="black",
                                    font="Arial")
    
    est_label_bottom = visual.TextStim(win, 
                                       text=variables['E'].labels[0], 
                                       pos=(FEEDBACK_CENTER_X, LABEL_BOTTOM_Y), 
                                       height=0.04, 
                                       color="black",
                                       font="Arial")

    est_slider.reset()
    clock = core.Clock()
    rt = None
    check_escape()         ### DAS IST DAS COOLE CHECK ESCAPE

    while clock.getTime() < MAX_RESPONSE_TIME:
        draw_slider_elements([slider_elements])
        feedback_box.draw()
        est_slider.draw()
        est_label_top.draw()
        est_label_bottom.draw()
        feedback_var_label.draw()
        win.flip()
        #check_escape()
        if est_slider.rating is not None and rt is None:
            rt = clock.getTime()
        keys = event.getKeys(keyList=['escape', 'space'], modifiers=True)
        if keys and any(k[0] == 'space' for k in keys) and est_slider.rating is not None:
            check_escape()
            break
        

    feedback_img = visual.ImageStim(win, 
                                    image=get_feedback_image_path(E),
                                    size=(IMG_SIZE, IMG_SIZE * 1.5),
                                    pos=(FEEDBACK_CENTER_X, FEEDBACK_IMG_Y))
    
    draw_slider_elements([slider_elements])
    feedback_box.draw()
    feedback_img.draw()
    feedback_var_label.draw()
    check_escape()
    win.flip()
    
    fb_clock = core.Clock()
    while fb_clock.getTime() < MAX_FEEDBACK_IMG_TIME:
        keys = event.getKeys(keyList=['escape', 'space'], modifiers=True)
        if keys and any(k[0] == 'space' for k in keys):
            check_escape()
            break
    check_escape()
    run_pause_screen(win)
    
    with open(results_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["train", block_idx, trial_id, trial_idx, E, [cue], round(cue_value, 2) if cue_value is not None else -100, 
                         round(est_slider.rating, 2) if est_slider.rating is not None else -1, 
                         round(rt, 3) if rt is not None else -1])



# ----------------- Test Block Logic -----------------

def run_test_block(win, stim_row, results_file, block_idx):
    cues = [cue.strip(" []'") for cue in stim_row['permutation_test'].split(",")]
    trial_idx = stim_row['stimulus_id']
    E = stim_row['E']

    shown_cues = []

    for i, cue in enumerate(cues):
        check_escape()
        shown_cues.append(cue)
        shown_cues_values = [round(float(stim_row[c]), 3) for c in shown_cues]
        slider_elements = [
            create_slider(win, -0.8 + idx * 0.4, variables[c], stim_row[c])
            for idx, c in enumerate(shown_cues)
        ]

        # Right side visuals
        feedback_box = visual.Rect(win, width=FEEDBACK_BOX_WIDTH, height=FEEDBACK_BOX_HEIGHT,
                                   pos=FEEDBACK_BOX_POS, lineColor="black", fillColor=None)
        feedback_var_label = visual.TextStim(win, text=LABEL_E_TEST,
                                             pos=(FEEDBACK_CENTER_X, VAR_LABEL_Y),
                                             height=0.05, 
                                             color="black",
                                             font="Arial")
        feedback_img = visual.ImageStim(win, image="images/einhorn_schaetzen.png",
                                        size=(IMG_SIZE, IMG_SIZE * 1.5),
                                        pos=(FEEDBACK_CENTER_X, FEEDBACK_IMG_Y),
                                        opacity=0.1)
        est_slider = visual.Slider(win, ticks=(0, 100), labels=["", ""], granularity=1,
                                   style='rating', size=SLIDER_SIZE, pos=(FEEDBACK_CENTER_X, SLIDER_Y),
                                   color="black", fillColor="red", borderColor="black",
                                   readOnly=False, font="Arial")
        est_label_top = visual.TextStim(win, text=variables['E'].labels[1],
                                        pos=(FEEDBACK_CENTER_X, LABEL_TOP_Y),
                                        height=0.04, 
                                        color="black",
                                        font="Arial")
        est_label_bottom = visual.TextStim(win, text=variables['E'].labels[0],
                                           pos=(FEEDBACK_CENTER_X, LABEL_BOTTOM_Y),
                                           height=0.04, 
                                           color="black",
                                           font="Arial")

        # Start response loop
        est_slider.reset()
        rt = None
        clock = core.Clock()
        check_escape()          ### DAS IST DAS COOLE CHECK ESCAPE
        while clock.getTime() < MAX_RESPONSE_TIME:
            draw_slider_elements(slider_elements)
            feedback_box.draw()
            feedback_img.draw()
            est_slider.draw()
            est_label_top.draw()
            est_label_bottom.draw()
            feedback_var_label.draw()
            win.flip()

            if est_slider.rating is not None and rt is None:
                rt = clock.getTime()

            keys = event.getKeys(keyList=['escape', 'space'], modifiers=True)
            if keys and any(k[0] == 'space' for k in keys) and est_slider.rating is not None:
                check_escape()
                break

        # Save response
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "test", block_idx, trial_idx, i+1, E, shown_cues.copy(), shown_cues_values,
                round(est_slider.rating, 2) if est_slider.rating is not None else -1,
                round(rt, 3) if rt is not None else -1
            ])

        # Short pause after estimate before next cue
        run_subtrial_pause(win, slider_elements)
    
    
    
# ----------------- Start of Actual Experiment -----------------

win = visual.Window(fullscr=True, 
                    color="white", 
                    units="norm")

current_datetime = core.getTime().strftime("%Y%m%d_%H%M%S")

results_file = RESULTS_FOLDER_PATH + f"{current_datetime}_{PARTICIPANT_ID}_experiment_results.csv"
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["phase", "block", "trial_id", "trial", "E", "cue", "cue_value", "estimate", "rt"])

# WELCOME-SCREEN INSTRUCTION
welcome_instruction_text = """
    Willkommen zum Einhorn-Experiment.\n\n
    Gleich sehen Sie eine kurze Demo-Phase, bestehend aus einer Trainings- und einer Testphase. Darin wird Ihnen anhand von Beispielen erklärt, wie das Experiment abläuft.\n\n
    Drücken Sie die Leertaste, um zu starten.
    """
instructions_screen(win=win, instruction_text=welcome_instruction_text)

# ----------------- DEMO PHASE -----------------

# TRAINING DEMO INSTRUCTION
demo_train_instruction_text = """
    DEMO TRAINING\n\n
    Sie sehen nun den Ablauf der Trainingsphase.\n\n
    Links wird eine von drei möglichen Variablen angezeigt: Regenbogen, Temperatur oder Helligkeit. Rechts schätzen Sie mit einem Schieberegler die Wahrscheinlichkeit, dass gerade ein Einhorn in der Gegend ist.\n
    Klicken Sie dafür innerhalb von 10 Sekunden mit der Maus auf den Slider und bestätigen Sie Ihre Einschätzung mit der Leertaste. Andernfalls wird der Trial als unvollständig markiert und ein neuer Trial beginnt. Anschließend sehen Sie ein Feedbackbild, das anzeigt, ob ein Einhorn da ist. Das Feedbackbild kann durch erneutes Drücken der Leertaste vorzeitig weggeklickt werden.\n\n
    Drücken Sie die Leertaste, um zu starten.
    """
instructions_screen(win=win, instruction_text=demo_train_instruction_text)

# DEMO TRAINING BLOCK
for block_idx, (_, stim_row) in enumerate(demo_train_trials.iterrows()):
    run_training_trial(win, stim_row, results_file, block_idx=-1, trial_idx=block_idx)
    check_escape()
    
# TEST DEMO INSTRUCTION
demo_test_instruction_text = """
    DEMO TEST\n\n
    Jetzt sehen Sie, wie die Testphase abläuft.\n\n
    Jeder Testdurchgang besteht aus drei Teilen: 
    Sie sehen nacheinander bis zu drei Variablen, die zum selben Stimulus gehören. Nach jeder Variable geben Sie ebenfalls innerhalb von 10 Sekunden nach jeder gezeigten Variable eine Schätzung darüber ab, für wie wahrscheinlich Sie es halten, dass ein Einhorns präsent ist.\n
    In der Testphase gibt es kein Feedback. Achtung: Die Reihenfolge der gezeigten Variablen kann variieren!\n\n
    Drücken Sie die Leertaste, um zu starten.
    """
instructions_screen(win=win, instruction_text=demo_test_instruction_text)

# DEMO TEST BLOCK
for block_idx, (_, stim_row) in enumerate(demo_test_trials.iterrows()):
    run_test_block(win, stim_row, results_file, block_idx=-1)
    check_escape()

# ----------------- TRAINING PHASE -----------------

# TRAINING INSTRUCTION
train_instruction_text = """
    TRAININGSPHASE\n\n
    Jetzt beginnt das eigentliche Experiment.\n\n
    In der Trainingsphase wird Ihnen pro Durchgang genau eine Variable gezeigt. Schätzen Sie auf dieser Grundlage, wie wahrscheinlich die Präsenz eines Einhorns ist.\n\n
    Drücken Sie die Leertaste, um zu starten. 
    """
instructions_screen(win=win, instruction_text=train_instruction_text)

train_stimuli_df = train_stimuli_df.iloc[:N_TRAIN_TRIALS]
for block_idx, (trial_idx, stim_row) in enumerate(train_stimuli_df.iterrows()):
    check_escape()
    # Show progress after every SHOW_PROGESS_AFTER_N_TRIALS Trials
    if block_idx > 0 and block_idx % SHOW_PROGESS_AFTER_N_TRIALS == 0:
        show_progress_screen(win, block_idx, N_TRAIN_TRIALS, phase_name="Trainingsphase")

    run_training_trial(win=win, stim_row=stim_row, results_file=results_file, block_idx=block_idx, trial_idx=trial_idx)

# ----------------- TESTING PHASE -----------------

# TESTING INSTRUCTION
test_instruction_text = """
    TESTPHASE\n\n
    Die Trainingsphase ist abgeschlossen – nun folgt die Testphase.\n\n
    In jedem Durchgang sehen Sie nacheinander eine, dann zwei und schließlich alle drei Variablen. Nach jeder Anzeige geben Sie eine neue Einschätzung darüber ab, wie wahrscheinlich die Präsenz eines Einhorns ist.\n\n
    Drücken Sie die Leertaste, um zu starten.
    """
instructions_screen(win=win, instruction_text=test_instruction_text)

# TESTING
test_stimuli_df = test_stimuli_df.iloc[:N_TEST_BLOCKS]  # Limit to N_TEST_BLOCKS
for block_idx, (_, stim_row) in enumerate(test_stimuli_df.iterrows()):
    print(len(test_stimuli_df), block_idx)
    check_escape()
    if block_idx > 0 and block_idx % SHOW_PROGESS_AFTER_N_TRIALS == 0:
        show_progress_screen(win=win, current=block_idx, total=N_TEST_BLOCKS, phase_name="Testphase")

    run_test_block(win=win, stim_row=stim_row, results_file=results_file, block_idx=block_idx)

# END
visual.TextStim(win, 
                text="Experiment beendet. Vielen Dank für Ihre Teilnahme!", 
                color="black",
                font="Arial"
                ).draw()
win.flip()
core.wait(3)
check_escape()
win.close()
core.quit()

# ----------------- End of Experiment -----------------