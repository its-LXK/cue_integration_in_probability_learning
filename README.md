# 🦄 Cue Integration in Probability Learning

This project examines how people integrate probabilities from multiple cues to predict the presence of a unicorn. It combines experimental data collection with simulations and psychophysical experiments in PsychoPy.

---

## 📁 Project Structure

```
cue_integration_in_probability_learning/
├── psychopy/                        # PsychoPy experiment and stimuli
│   ├── psychopy_experiment.py
│   ├── hyperparameter_and_stimuli/
│   └── images/
├── psychopy_variables_setup/        # Generation of stimuli and hyperparameters
│   ├── create_stimuli.ipynb
│   └── hyperparameters.csv
├── notebooks/                       # helper functions and space for analysis notebooks
│   └── helper_functions.py
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

1. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

2. **Generate or load stimuli**
    - New stimuli can be generated using the `psychopy_variables_setup/create_stimuli.ipynb` notebook.
    - Pre-generated stimuli are available in `psychopy/hyperparameter_and_stimuli/`.

3. **Start the experiment**
    Prerequisites: installed PsychoPy software (Peirce et al., 2019) 
    
    The repository can be opened in PsychoPy, where the ```psychopy_experiment.py``` script can be started directly.
    Peirce, J. W., Gray, J. R., Simpson, S., MacAskill, M. R., Höchenberger, R., Sogo, H., Kastman, E., Lindeløv, J. (2019). PsychoPy2: experiments in behavior made easy. Behavior Research Methods. 10.3758/s13428-018-01193-y
---

## 🧪 Contents

- **PsychoPy Experiment:**  
  An interactive experiment for probability estimation with multiple cues.

- **Stimulus Generation:**  
  Scripts and notebooks for creating balanced sets of stimuli and hyperparameters.

- **Analysis:**  
  Helper functions are provided in ```helper_functions.py``` for evaluating experimental data and can be imported into analysis notebooks.

---

## 📊 Example: Stimulus CSV

| stimulus_id | E | R    | T    | H    | permutation_train | permutation_test     |
|-------------|---|------|------|------|-------------------|----------------------|
| 43237       | 1 | -1.05| 1.26 | 0.93 | R                 | ['R', 'T', 'H']      |
| 43285       | 0 | -2.59| 0.89 | 0.61 | R                 | ['R', 'T', 'H']      |
| ...         |...| ...  | ...  | ...  | ...               | ...                  |

---

## 📝 Notes

- The key parameters and stimuli are version controlled as CSV files.
- The stimulus permutations allow systematic variation of cue order in the test.

---

## 👩‍🔬 Authors

- Alexandra Kraft

---

## 📄 License

The use, further development, or citation of this code is only permitted provided that the original author is explicitly named. Any use, distribution, or publication must include a clear notice of authorship. See [LICENSE](LICENSE)