# 7PhrasalNMT

A neural machine translation (NMT) project focused on phrasal parsing in English-Hindi translation. This repository implements a character-level Transformer model for text translation, with custom preprocessing and training scripts.

## Features
- Character-level Transformer architecture for NMT
- Custom dataset preprocessing
- Training and evaluation scripts
- Utilities for data handling and model management

## Repository Structure
- `char_phrase_dataset.py` — Dataset preparation and loading for character-level phrasal translation
- `char-transformer-text-translation.ipynb` — Jupyter notebook for experiments and demonstrations
- `preprocessing.py` — Data preprocessing utilities
- `trainer.py` — Model training and evaluation logic
- `transformer.py` — Transformer model implementation
- `utils.py` — Helper functions and utilities
- `readme.md` — Project documentation

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd 7PhrasalNMT
   ```
2. **Install dependencies:**
   Ensure you have Python 3.8+ and install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` if not present, based on your environment)*
3. **Prepare your data:**
   - Place your parallel English-Hindi dataset in the appropriate format as expected by `char_phrase_dataset.py` and `preprocessing.py`.
4. **Train the model:**
   ```bash
   python trainer.py
   ```
5. **Experiment in Notebook:**
   - Open `char-transformer-text-translation.ipynb` in Jupyter for interactive experiments.

## Usage
- Modify `trainer.py` and `transformer.py` to adjust model parameters or training settings.
- Use `preprocessing.py` for custom data cleaning and preparation. By default, phrase extraction is performed on English sentences.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Acknowledgements
- Inspired by research in neural machine translation and phrasal parsing.
- Special thanks to contributors and the open-source community.
